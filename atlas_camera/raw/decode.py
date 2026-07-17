"""rawpy demosaic -> (scene-linear, display-sRGB) float arrays.

ONE demosaic per file: the linear array is the master, the display array is
derived from it — this guarantees pixel-identical geometry between the solve
tensor and the EXR sidecar, and halves decode time on 36–100MP files.
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
               exposure_ev: float = 0.0):
    """Decode a RAW file. Returns ``(linear_rgb, display_srgb)`` float32 HxWx3.

    ``linear_rgb`` is scene-linear with **sRGB/Rec.709 primaries** (rawpy's
    sRGB output colorspace with gamma (1,1)) — NOT ACEScg; tag it honestly
    downstream. ``display_srgb`` is display-encoded for solve/preview.
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
    linear = (rgb16.astype(np.float32) / 65535.0) * float(2.0 ** exposure_ev)
    return linear, display_from_linear(linear)
