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

#: Atlas's own colourspace tag. OIIO drops the standard ``oiio:ColorSpace``
#: attribute on write whenever the active OCIO config cannot supply a
#: ``colorInteropID`` for the space — true of older studio configs such as
#: fn-nuke_cg v1.0.0 / OCIO v2.1 — so a plate written under one carried NO
#: colourspace at all. This attribute is written unconditionally and read as a
#: fallback, so a plate always self-describes regardless of config.
ATLAS_COLORSPACE_ATTR = "atlas:ColorSpace"


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


# Canonical Atlas colourspace name -> the OCIO ROLE that denotes the same space,
# where a standard role exists. Roles are config-INDEPENDENT: `scene_linear`
# resolves to whatever a config calls its rendering space, and `aces_interchange`
# is REQUIRED by the ACES spec to be ACES2065-1 in any ACES config. Resolving
# through a role is what lets Atlas honour a user's $OCIO studio config instead of
# only OIIO's built-in ACES config. NOTE: "Linear Rec.709 (sRGB)" and the display
# space deliberately have NO entry here — no standard role denotes them, so they
# fall through to the name-alias table below rather than being forced onto a role.
_SPACE_ROLES = {
    "ACEScg": "scene_linear",
    "ACES2065-1": "aces_interchange",
}

# Canonical name -> ordered candidate names to try when neither the exact name nor
# a role resolves in the active config. Covers the spaces likely to differ across
# the configs Atlas meets: OIIO's built-in cg/studio config, the older ACES 1.x
# reference config, and common studio naming.
_SPACE_ALIASES = {
    "ACEScg": ("ACEScg", "ACES - ACEScg", "lin_ap1"),
    "ACES2065-1": ("ACES2065-1", "ACES - ACES2065-1", "lin_ap0"),
    "Linear Rec.709 (sRGB)": (
        "Linear Rec.709 (sRGB)",        # OCIO v2 built-in cg / studio config
        "Utility - Linear - Rec.709",   # ACES 1.0.3 reference config
        "Linear Rec.709",
        "lin_rec709",
    ),
    "sRGB - Display": ("sRGB - Display", "Output - sRGB", "sRGB", "srgb_display"),
}


def _config_resolve(cfg, name: str) -> str:
    """`ColorConfig.resolve(name)` where available, else the name unchanged.

    OIIO gained `resolve` (alias -> canonical name) after the versions Atlas
    also supports, and it returns its input untouched for an unknown name, so
    callers must still check the result against the config's own space list.
    """
    resolver = getattr(cfg, "resolve", None)
    if resolver is None:
        return name
    try:
        return resolver(name) or name
    except Exception:                  # noqa: BLE001 - binding differences degrade
        return name


def _display_names(cfg) -> set:
    """The config's DISPLAY names, which are not colourspaces to OIIO's count."""
    try:
        names = cfg.getDisplayNames()
    except Exception:                  # noqa: BLE001 - older bindings
        try:
            names = [cfg.getDisplayNameByIndex(i)
                     for i in range(cfg.getNumDisplays())]
        except Exception:              # noqa: BLE001
            return set()
    return {str(n) for n in (names or ())}


def _config_name(cfg) -> str:
    """`configname` is a METHOD on the OIIO binding, not a property — reading it
    as an attribute yields a bound-method repr, which then lands in a manifest
    as the config's "name"."""
    attr = getattr(cfg, "configname", None)
    if attr is None:
        return ""
    try:
        return str(attr() if callable(attr) else attr)
    except Exception:                  # noqa: BLE001 - binding differences
        return ""


def config_identity() -> dict:
    """Identify the ACTIVE OCIO config: name, file path, sha256, space count.

    A colourspace NAME is not a contract — a plate tagged `ACEScg` under two
    different configs is two different plates. The paint bridges record this
    dict in their manifests so a score produced against one config is never
    silently compared with a score produced against another, and so a claim
    that two applications shared a config can be checked rather than trusted.
    """
    if not oiio_available():
        return {"available": False, "config_name": "", "config_path": "",
                "config_sha256": "", "n_colorspaces": 0, "n_displays": 0}
    cfg = ColorConfigCache.get()
    path = os.environ.get("OCIO", "")
    digest = ""
    if path and os.path.isfile(path):
        import hashlib
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "available": True,
        "config_name": _config_name(cfg),
        # Empty path means OIIO's built-in config (ocio://default), which has
        # no file to hash — an empty sha256 is the honest answer, not a bug.
        "config_path": path,
        "config_sha256": digest,
        "n_colorspaces": len(list_colorspaces()),
        "n_displays": len(_display_names(cfg)),
    }


def resolve_colorspace(name: str) -> str:
    """Map an Atlas colourspace request to a name that EXISTS in the ACTIVE OCIO
    config, so a conversion works whether the user is on OIIO's built-in ACES
    config or their own $OCIO studio config.

    A hardcoded ``"ACEScg"`` only works because it happens to be spelled that way
    in the built-in config; a studio config may call the same space something else
    (ACES 1.0.3 names linear Rec.709 ``Utility - Linear - Rec.709``). Resolving
    first through OCIO roles, then through known cross-config aliases, removes that
    brittleness while staying colourimetrically exact — a role or alias names the
    SAME space, it never approximates it.

    First hit wins:
      1. ``name`` is already a colourspace in the active config.
      2. ``name`` is (or maps to) an OCIO role that resolves in the config.
      3. a known cross-config alias for ``name`` exists in the config.

    Anything unresolved raises with the config's actual spaces listed, so a
    mis-set $OCIO fails loudly at the call site instead of silently mis-tagging a
    plate. If OIIO is unavailable the name is returned untouched — the caller's own
    ``_require_oiio`` raises the actionable install error first.
    """
    if not name:
        return name
    known = set(list_colorspaces())
    if not known:                      # no OIIO / no config — let the caller raise
        return name
    if name in known:                  # 1. already valid in this config
        return name

    cfg = ColorConfigCache.get()

    # 1b. The config's OWN alias table. `list_colorspaces` enumerates canonical
    # names only, so a config that carries a name as an ALIAS looks absent here
    # even though OCIO resolves it perfectly. Found live: a plate tagged
    # `lin_rec709_scene` (the built-in config's canonical name) read fine under
    # the built-in config and raised under fn-nuke_cg, which carries that same
    # space as `Linear Rec.709 (sRGB)` with `lin_rec709_scene` among its
    # aliases. `resolve()` returns the input unchanged for a name the config
    # does not know, so the `in known` test below is what makes it safe.
    resolved_alias = _config_resolve(cfg, name)
    if resolved_alias and resolved_alias != name and resolved_alias in known:
        return resolved_alias

    for role in (name, _SPACE_ROLES.get(name, "")):   # 2. role resolution
        if not role:
            continue
        try:
            resolved = cfg.getColorSpaceNameByRole(role)
        except Exception:              # noqa: BLE001 - binding differences degrade to aliases
            resolved = ""
        if resolved and resolved in known:
            return resolved

    for cand in _SPACE_ALIASES.get(name, ()):         # 3. cross-config aliases
        if cand in known:
            return cand

    # 4. A DISPLAY colourspace. OCIO v2 configs may declare display-referred
    # spaces in their own block, and OIIO's `getNumColorSpaces()` enumeration
    # does not include them — so `sRGB - Display` reads as "missing" from a
    # config that in fact converts to it correctly (0.18 ACEScg -> 0.46135
    # under fn-nuke_cg, matching the value recorded in plate/__init__.py).
    # That matters because `sRGB - Display` is COMFY_WORKING_COLORSPACE, the
    # default output of every read_plate: without this step, pointing $OCIO at
    # such a config makes essentially every plate read raise.
    if name in _display_names(cfg):
        return name

    raise RuntimeError(
        f"Colourspace {name!r} is not in the active OCIO config, and no role or "
        f"known alias maps to it. The active config offers: {sorted(known)[:12]}"
        f"{' ...' if len(known) > 12 else ''}. Point $OCIO at an ACES-compatible "
        f"config, or install [oiio] to use the built-in ACES config."
    )


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
    # Decode to FLOAT, not the file's native type. ImageBufAlgo.colorconvert
    # works in the buffer's own precision, so a half EXR — which is most of
    # them, since AtlasLoadRAW writes half sidecars and write_exr defaults to
    # half — had its colour conversion performed in HALF. Measured on a real
    # plate: a Rec.709 -> ACES2065-1 -> Rec.709 round trip came back with
    # max abs error 0.00292969 at native precision and 0.00000107 forced to
    # float. That is a 2700x difference, and it silently set the noise floor
    # for every colour-managed handoff: a Photoshop round trip measured 2.9e-3
    # end to end, of which Photoshop itself contributed 1.2e-7.
    buf.read(0, 0, True, oiio.FLOAT)
    if buf.has_error:
        raise RuntimeError(f"Could not decode {path}: {buf.geterror()}")

    spec = buf.spec()
    # Prefer the standard tag when it survived; fall back to Atlas's own, which
    # is written unconditionally because OIIO drops the standard one under a
    # config with no colorInteropID for the space. A file written by another
    # application therefore still wins on the standard tag.
    declared = (spec.getattribute("oiio:ColorSpace")
                or spec.getattribute(ATLAS_COLORSPACE_ATTR))
    resolved_in = (declared if declared else
                   (auto_colorspace_for_path(path) if input_colorspace == "auto"
                    else input_colorspace))

    resolved_out = "" if raw_data else (output_colorspace or "")
    # Resolve both endpoints to what the ACTIVE OCIO config actually calls these
    # spaces (role- or alias-based) before comparing or converting: this honours a
    # user's $OCIO studio config, and comparing the RESOLVED names skips a needless
    # convert when input and output are the same space spelled two different ways.
    conv_in = resolve_colorspace(resolved_in) if resolved_out else resolved_in
    conv_out = resolve_colorspace(resolved_out) if resolved_out else ""
    if resolved_out and conv_out != conv_in:
        # OCIO transforms RGB, never alpha. Files with alpha are treated as
        # associated/premultiplied: OIIO must divide RGB by alpha, transform
        # the straight colour, then re-premultiply while leaving alpha itself
        # unchanged. Keep this explicit instead of depending on binding defaults.
        converted = ImageBufAlgo.colorconvert(
            buf, conv_in, conv_out, unpremult=True)
        if converted.has_error:
            raise RuntimeError(
                f"Colour conversion {conv_in!r} -> {conv_out!r} failed: "
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
        # The ON-DISK format, from the NATIVE spec. `spec().format` is the
        # buffer's format, which is float since the read is forced to float for
        # conversion precision — reporting that would tell a caller a half
        # plate is float, and AtlasLoadPlate derives `is_proxy` from it.
        file_bit_depth=str(buf.nativespec().format),
        input_colorspace=resolved_in, output_colorspace=resolved_out,
        channel_names=names, metadata=meta,
    )


def write_exr(path: str, pixels: Any, *, bit_depth: str = "half",
              compression: str = "auto", source_colorspace: str | None = None,
              output_colorspace: str | None = None,
              extra_attribs: dict | None = None) -> str:
    """Write an EXR, converting colour if asked, and TAG what it contains.

    The tag matters: a plate that records its own colourspace via
    ``oiio:ColorSpace`` can be read back correctly with no out-of-band
    knowledge, which is exactly what a DCC handoff needs. opencv's writer
    cannot do this at all.

    ``bit_depth`` is 'half' (16-bit float, the VFX default — half the size and
    ample for imagery) or 'float' (32-bit, for data passes needing full range).

    ``compression`` defaults to **auto**, which reads the intent already
    encoded in ``bit_depth``:

      half  -> **dwab**, the lossy DCT codec modern VFX ships plates with:
               roughly 2-4x smaller than zip and visually lossless on
               continuous-tone imagery. That is what 'half' means here.
      float -> **zip**, lossless. 'float' is this module's marker for a DATA
               pass, and a DCT codec on channels that are numbers rather than
               colour — an ST map, depth, a matte read as a mask, normals —
               is the same class of error as storing them in half float, which
               ``raw/redistort`` rejected for a measured 2.5 px.

    Pass an explicit codec ('dwab', 'dwaa', 'zip', 'piz', 'none') to override.
    A float-precision COLOUR plate is the one case auto gets conservative
    about; say ``compression="dwab"`` if the size matters more than exactness.
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
    if source_colorspace and output_colorspace:
        # Resolve both endpoints to the active OCIO config's real names (role- or
        # alias-based) so a studio $OCIO config works and equal spaces spelled two
        # ways don't trigger a needless convert. Tag the file with the resolved
        # OUTPUT name — what the pixels actually are in this config — so read_plate
        # (which resolves tags the same way) reads it back correctly anywhere.
        conv_src = resolve_colorspace(source_colorspace)
        conv_out = resolve_colorspace(output_colorspace)
        if conv_src != conv_out:
            # Same associated-alpha contract as read_plate: transform straight RGB
            # between an unpremultiply/re-premultiply pair; alpha remains data.
            conv = ImageBufAlgo.colorconvert(
                buf, conv_src, conv_out, unpremult=True)
            if conv.has_error:
                raise RuntimeError(
                    f"Colour conversion {conv_src!r} -> {conv_out!r} "
                    f"failed: {conv.geterror()}")
            buf = conv
            tagged = conv_out

    spec = ImageSpec(w, h, c, bit_depth)
    if compression == "auto":
        # bit_depth already carries the imagery/data distinction — see above.
        compression = "dwab" if str(bit_depth).lower() == "half" else "zip"
    spec.attribute("compression", compression)
    if tagged:
        spec.attribute("oiio:ColorSpace", tagged)
        # OIIO only PERSISTS oiio:ColorSpace when the active config can supply a
        # `colorInteropID` for the space. OIIO's built-in config does; an older
        # studio config (fn-nuke_cg v1.0.0 / OCIO v2.1) does not, and OIIO then
        # silently drops the tag — writing an EXR with no colourspace at all.
        # That is silent data loss with a nasty tail: read_plate on 'auto' falls
        # back to guessing from the extension, so a RAW sidecar (Rec.709-linear)
        # comes back read as ACEScg and the colour is quietly wrong, which is
        # exactly the failure docs/USER_GUIDE.md warns about.
        #
        # So Atlas writes its OWN tag as well. It is under our control, it
        # survives any config, and read_plate prefers oiio:ColorSpace when
        # present so a file written by anything else still wins.
        spec.attribute(ATLAS_COLORSPACE_ATTR, tagged)
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
