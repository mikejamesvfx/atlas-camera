"""rawpy demosaic -> (scene-linear, display-sRGB) float arrays.

ONE demosaic per file: the linear array is the master, the display array is
derived from it — this guarantees pixel-identical geometry between the solve
tensor and the EXR sidecar, and halves decode time on 36–100MP files.

The linear master is SCENE-REFERRED with diffuse white near 1.0, because
``decode_raw`` applies a ``headroom`` multiply (default 6.0) matching
rawtoaces' ``--headroom`` convention. Sensor clip therefore lands ~6x above
1.0, which is what an ACES scene-linear pipeline expects. It still carries
**sRGB/Rec.709 primaries** — it is NOT ACEScg, and retagging it as such would
be a lie; conversion belongs to ``AtlasExportPlateEXR``.
"""

from __future__ import annotations


def _require_rawpy():
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError(
            "Camera RAW decoding requires rawpy (bundles libraw: NEF/CR2/CR3/"
            "RAF/ARW incl. X-Trans). Install with: pip install -e .[raw]") from exc
    return rawpy


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Camera RAW decoding requires numpy. "
            "Install with: pip install -e .[raw]") from exc
    return np


def srgb_encode(linear):
    """Pure-numpy sRGB OETF (matches the viewport shader's atlasLinearToSRGB)."""
    np = _require_numpy()
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308,
                    linear * 12.92,
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055).astype(np.float32)


def display_from_linear(linear_rgb, *, percentile: float = 99.5):
    """Deterministic display render: map the given luminance percentile to 1.0
    (replaces rawpy's non-deterministic auto-bright), then sRGB-encode.

    The solver consumes geometry, not radiometry, so a display-referred tensor
    is the right input for VP detection and GeoCalib (trained on
    display-referred images).
    """
    np = _require_numpy()
    peak = float(np.percentile(linear_rgb, percentile))
    scale = 1.0 / peak if peak > 1e-8 else 1.0
    return srgb_encode(linear_rgb * scale)


def decode_raw(path: str, *, half_size: bool = False, white_balance: str = "camera",
               exposure_ev: float = 0.0, headroom: float = 6.0):
    """Decode a RAW file. Returns ``(linear_rgb, display_srgb)`` float32 HxWx3.

    ``linear_rgb`` is scene-linear with **sRGB/Rec.709 primaries** (rawpy's
    sRGB output colorspace with gamma (1,1)) — NOT ACEScg; tag it honestly
    downstream. ``display_srgb`` is display-encoded for solve/preview.

    ``headroom`` multiplies the LINEAR array only. Dividing by 65535 alone puts
    sensor clip at 1.0 and mid-grey around 0.04, where ACES scene-linear wants
    ~0.18 — correct under an un-tone-mapped sRGB view, ~2.6 stops crushed under
    an ACES view transform. rawtoaces solves this with the same multiply, and
    its ``--headroom`` default of 6.0 is the default here
    (``AcademySoftwareFoundation/rawtoaces``,
    ``src/rawtoaces_util/image_converter.cpp:2443``, default at ``:845-854``).
    Measured on a D810 frame: ``p75 0.150 x 6 = 0.90``, so diffuse white lands
    near 1.0. ``headroom=1.0`` reproduces the pre-2026-08 output exactly.

    It is deliberately NOT applied to the display path, and does not need to
    be: ``display_from_linear`` normalises by a luminance percentile, so a
    constant factor cancels by construction. The solve tensor is therefore
    unchanged at any headroom, which is what makes this scale safe to change
    without re-verifying the solver. (Unchanged, not bit-identical — the
    cancellation is exact in real arithmetic but this is float32, so it lands
    within an ULP or two. That is ~1e-7, many orders below anything VP
    detection or GeoCalib can resolve.)
    """
    rawpy = _require_rawpy()
    np = _require_numpy()
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=(white_balance == "camera"),
            use_auto_wb=(white_balance == "auto"),
            no_auto_bright=True,
            gamma=(1, 1),
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            half_size=bool(half_size),
        )
    scale = float(headroom) * float(2.0 ** exposure_ev)
    linear = (rgb16.astype(np.float32) / 65535.0) * scale
    return linear, display_from_linear(linear)
