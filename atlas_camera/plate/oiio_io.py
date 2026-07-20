"""OpenImageIO-backed float image read/write with OCIO colour conversion.

Heavy deps are imported lazily so the core package stays dependency-free — the
same contract as ``inference/depth_estimator.py`` and ``raw/undistort.py``.
Install with:  pip install -e .[oiio]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# File-extension -> the colourspace such a file conventionally holds. EXR/DPX/HDR
# are scene-referred float formats; 8-bit delivery formats are display-referred.
# Matches the convention ComfyUI-OCIO's reader uses, so migrating a workflow does
# not silently change how a plate is interpreted.
_SCENE_LINEAR_EXT = {".exr", ".hdr", ".dpx", ".cin"}
_DISPLAY_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".tga"}

DEFAULT_SCENE_COLORSPACE = "ACEScg"
DEFAULT_DISPLAY_COLORSPACE = "sRGB - Display"
#: What ComfyUI itself expects on an IMAGE tensor: display-referred 0-1 sRGB.
COMFY_WORKING_COLORSPACE = "sRGB - Display"


@dataclass
class PlateRead:
    """One decoded plate plus the provenance needed to hand it to a DCC."""

    pixels: Any                      # (H, W, 3) float32, in `output_colorspace`
    alpha: Any | None = None         # (H, W) float32 when the file carries alpha
    width: int = 0
    height: int = 0
    nchannels: int = 0
    file_format: str = ""
    file_bit_depth: str = ""         # the ON-DISK format: half / float / uint16 ...
    input_colorspace: str = ""       # what the file was taken to be
    output_colorspace: str = ""      # what `pixels` is now in ("" = raw passthrough)
    channel_names: tuple = ()
    metadata: dict = field(default_factory=dict)

    @property
    def is_float(self) -> bool:
        return self.file_bit_depth in ("half", "float", "double")

    def summary(self) -> str:
        cs = (f"{self.input_colorspace} -> {self.output_colorspace}"
              if self.output_colorspace else f"{self.input_colorspace} (raw)")
        return (f"{self.width}x{self.height}x{self.nchannels} {self.file_format} "
                f"{self.file_bit_depth} · {cs}")


def _require_oiio():
    try:
        import OpenImageIO as oiio
    except ImportError as exc:  # pragma: no cover - exercised by install, not tests
        raise RuntimeError(
            "Colour-managed plate I/O requires OpenImageIO. Install with:\n"
            "    pip install -e .[oiio]\n"
            "OpenImageIO ships wheels for Windows, Linux and macOS (including "
            "Apple Silicon) and carries a built-in ACES OCIO config, so nothing "
            "else needs installing."
        ) from exc
    return oiio


def oiio_available() -> bool:
    """Cheap, network-free probe. Never imports the heavy module twice."""
    import importlib.util

    return importlib.util.find_spec("OpenImageIO") is not None


def oiio_diagnostics() -> str:
    """One line describing the float-I/O backend, for node reports.

    Deliberately also reports the OPENCV situation when OIIO is absent: the
    historic failure mode is a user being told to set
    ``OPENCV_IO_ENABLE_OPENEXR=1`` on a wheel whose codec was never compiled in,
    which can never work. Naming the real cause saves a long detour.
    """
    if oiio_available():
        oiio = _require_oiio()
        try:
            n = ColorConfigCache.get().getNumColorSpaces()
        except Exception:  # noqa: BLE001
            n = 0
        return f"OpenImageIO {oiio.__version__} ({n} OCIO colorspaces)"

    detail = "OpenImageIO not installed (pip install -e .[oiio])"
    try:
        import cv2  # noqa: PLC0415

        exr = "OpenEXR:" in cv2.getBuildInformation() and \
              "OpenEXR:                     NO" not in cv2.getBuildInformation()
        detail += f"; opencv {cv2.__version__} EXR codec built in: {exr}"
        if not exr:
            detail += (" — this wheel has no EXR codec compiled in, so "
                       "OPENCV_IO_ENABLE_OPENEXR cannot enable it")
    except Exception:  # noqa: BLE001
        pass
    return detail


class ColorConfigCache:
    """One ColorConfig for the process. Building it parses the OCIO config, so
    doing it per image would be wasteful in a node that runs per frame."""

    _cfg = None

    @classmethod
    def get(cls):
        if cls._cfg is None:
            oiio = _require_oiio()
            cls._cfg = oiio.ColorConfig()
        return cls._cfg


def list_colorspaces() -> list[str]:
    """Every colourspace the active OCIO config knows, for a node's combo box.

    Honours ``$OCIO`` when the user has a studio config; otherwise OIIO's
    built-in ACES config supplies ACEScg/ACEScct/ACES2065-1 and the display
    spaces with nothing to install.
    """
    if not oiio_available():
        return []
    cfg = ColorConfigCache.get()
    return [cfg.getColorSpaceNameByIndex(i) for i in range(cfg.getNumColorSpaces())]


def auto_colorspace_for_path(path: str) -> str:
    """Infer what a file of this type conventionally holds."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in _SCENE_LINEAR_EXT:
        return DEFAULT_SCENE_COLORSPACE
    if ext in _DISPLAY_EXT:
        return DEFAULT_DISPLAY_COLORSPACE
    return DEFAULT_DISPLAY_COLORSPACE


def read_plate(path: str, *, input_colorspace: str = "auto",
               output_colorspace: str | None = COMFY_WORKING_COLORSPACE,
               raw_data: bool = False) -> PlateRead:
    """Read any OIIO-supported image as float32, optionally colour-converted.

    ``input_colorspace='auto'`` infers from the extension, unless the file
    self-describes via an ``oiio:ColorSpace`` attribute, which always wins —
    a plate that states what it is should be believed over a guess.

    ``raw_data=True`` is Nuke's "raw data": skip conversion entirely and hand
    back the file's own values. Use it whenever the numbers are DATA rather than
    colour (depth, normals, mattes, UV passes) — converting those corrupts them.
    """
    oiio = _require_oiio()
    from OpenImageIO import ImageBuf, ImageBufAlgo

    buf = ImageBuf(str(path))
    if buf.has_error:
        raise RuntimeError(f"Could not read {path}: {buf.geterror()}")
    buf.read()
    if buf.has_error:
        raise RuntimeError(f"Could not decode {path}: {buf.geterror()}")

    spec = buf.spec()
    declared = spec.getattribute("oiio:ColorSpace")
    resolved_in = (declared if declared else
                   (auto_colorspace_for_path(path) if input_colorspace == "auto"
                    else input_colorspace))

    resolved_out = "" if raw_data else (output_colorspace or "")
    if resolved_out and resolved_out != resolved_in:
        converted = ImageBufAlgo.colorconvert(buf, resolved_in, resolved_out)
        if converted.has_error:
            raise RuntimeError(
                f"Colour conversion {resolved_in!r} -> {resolved_out!r} failed: "
                f"{converted.geterror()}. Available spaces: {list_colorspaces()[:8]}...")
        buf = converted

    px = buf.get_pixels(oiio.FLOAT)
    alpha = None
    names = tuple(spec.channelnames)
    if spec.alpha_channel >= 0 and px.shape[2] > spec.alpha_channel:
        alpha = px[..., spec.alpha_channel].copy()
    rgb = px[..., :3] if px.shape[2] >= 3 else px[..., :1].repeat(3, axis=2)

    meta = {}
    for p in spec.extra_attribs:
        try:
            meta[p.name] = p.value if isinstance(p.value, (str, int, float)) else str(p.value)
        except Exception:  # noqa: BLE001
            continue

    return PlateRead(
        pixels=rgb, alpha=alpha,
        width=spec.width, height=spec.height, nchannels=spec.nchannels,
        file_format=os.path.splitext(str(path))[1].lstrip(".").lower(),
        file_bit_depth=str(spec.format),
        input_colorspace=resolved_in, output_colorspace=resolved_out,
        channel_names=names, metadata=meta,
    )


def write_exr(path: str, pixels: Any, *, bit_depth: str = "half",
              compression: str = "zip", source_colorspace: str | None = None,
              output_colorspace: str | None = None,
              extra_attribs: dict | None = None) -> str:
    """Write an EXR, converting colour if asked, and TAG what it contains.

    The tag matters: a plate that records its own colourspace via
    ``oiio:ColorSpace`` can be read back correctly with no out-of-band
    knowledge, which is exactly what a DCC handoff needs. opencv's writer
    cannot do this at all.

    ``bit_depth`` is 'half' (16-bit float, the VFX default — half the size and
    ample for imagery) or 'float' (32-bit, for data passes needing full range).
    """
    oiio = _require_oiio()
    from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec

    import numpy as np

    arr = np.ascontiguousarray(np.asarray(pixels, dtype="float32"))
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape

    buf = ImageBuf(ImageSpec(w, h, c, "float"))
    buf.set_pixels(oiio.ROI(), arr)
    if buf.has_error:
        raise RuntimeError(f"Could not stage pixels for {path}: {buf.geterror()}")

    tagged = source_colorspace
    if source_colorspace and output_colorspace and source_colorspace != output_colorspace:
        conv = ImageBufAlgo.colorconvert(buf, source_colorspace, output_colorspace)
        if conv.has_error:
            raise RuntimeError(
                f"Colour conversion {source_colorspace!r} -> {output_colorspace!r} "
                f"failed: {conv.geterror()}")
        buf = conv
        tagged = output_colorspace

    spec = ImageSpec(w, h, c, bit_depth)
    spec.attribute("compression", compression)
    if tagged:
        spec.attribute("oiio:ColorSpace", tagged)
    spec.attribute("Software", "Atlas Camera")
    for key, value in (extra_attribs or {}).items():
        try:
            spec.attribute(str(key), value if isinstance(value, (int, float)) else str(value))
        except Exception:  # noqa: BLE001
            continue

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"No OpenImageIO writer for {path} ({oiio.geterror()})")
    if not out.open(str(path), spec):
        raise RuntimeError(f"Could not open {path} for writing: {out.geterror()}")
    ok = out.write_image(buf.get_pixels(oiio.FLOAT))
    err = out.geterror()
    out.close()
    if not ok:
        raise RuntimeError(f"Could not write {path}: {err}")
    return str(path)
