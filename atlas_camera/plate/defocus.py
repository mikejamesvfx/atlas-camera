"""Depth-driven defocus for plates (pure numpy).

The lens-blur gap flagged in the community scans, done the Atlas way: the
circle of confusion comes from the SHARED metric depth map and a focus
distance in METRES, not from a hand-painted blur mask. Thin-lens shape:

    coc(z) = strength_px * |1 - S / z|

which is the real thin-lens falloff up to the aperture constant: background
blur saturates toward ``strength_px`` at infinity, foreground blur grows past
it and is clamped at ``4 * strength_px``.

Rendering is layered gather: the CoC field is quantized into ``levels``
blur bands, each band is blurred with the iterated masked box blur (borrowed
from plate/deband's kernel — Gaussian-ish, O(n)), and the bands are lerped
per-pixel. Simple and fast; the classic gather artifact (sharp foreground
edges bleeding over a blurred background) is accepted and stated in the node
report — this is a preview/finishing defocus, not a render-grade scatter.

Float-safe: no clamping of HDR values; NaN depth (holes/sky) inherits the
FAR blur level rather than punching sharp holes into a blurred sky.
"""

from __future__ import annotations


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of the package
        raise ImportError("plate defocus needs numpy — pip install -e .[vision]") from exc
    return np


_FOREGROUND_COC_CAP = 4.0  # foreground CoC grows unbounded as z -> 0; cap it


def coc_field(depth_m, focus_distance_m: float, strength_px: float):
    """Per-pixel circle of confusion (px). NaN depth -> the far-limit blur."""
    np = _require_numpy()
    z = np.asarray(depth_m, dtype=np.float64)
    s = max(float(focus_distance_m), 1e-6)
    k = max(float(strength_px), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        coc = k * np.abs(1.0 - s / np.maximum(z, 1e-6))
    coc = np.where(np.isfinite(z), coc, k)          # holes/sky = far blur
    return np.clip(coc, 0.0, _FOREGROUND_COC_CAP * k)


def defocus_plate(
    image,
    depth_m,
    *,
    focus_distance_m: float,
    strength_px: float = 8.0,
    levels: int = 6,
):
    """Depth-defocus a HxWxC float plate. Returns (float32 image, coc float32)."""
    np = _require_numpy()
    from atlas_camera.plate.deband import _box_blur_2d

    img = np.asarray(image, dtype=np.float64)
    chan = img if img.ndim == 3 else img[..., None]
    coc = coc_field(depth_m, focus_distance_m, strength_px)
    if coc.shape != chan.shape[:2]:
        raise ValueError(f"depth {coc.shape} vs image {chan.shape[:2]} shape mismatch")
    if float(strength_px) <= 0.0 or float(coc.max()) < 0.25:
        out = img.astype(np.float32)
        return out, coc.astype(np.float32)

    n_levels = int(max(2, min(levels, 12)))
    max_coc = float(coc.max())
    radii = np.linspace(0.0, max_coc, n_levels)

    # Band positions in level units; each pixel sits between two bands.
    t = np.clip(coc / (max_coc or 1.0), 0.0, 1.0) * (n_levels - 1)

    # Accumulate band-by-band with a hat weight instead of materializing the
    # whole (L, H, W, C) stack and fancy-indexing it. The hat weights
    # max(0, 1 - |t - level|) sum to exactly 1 per pixel, so this is the same
    # two-band lerp written as a scatter -- numerically identical, but two
    # full-size buffers stay live instead of L of them. Measured on the
    # original gather: 1.5 GB peak at 1920x1080 with levels=12, and the stack
    # alone would be 2.39 GB at 4K -- and Atlas plates are 4K.
    out = np.zeros_like(chan)
    for level, r in enumerate(radii):
        w = np.clip(1.0 - np.abs(t - level), 0.0, 1.0)
        if not w.any():
            continue                       # no pixel reads this band
        rb = max(0, int(round(r / 3.0)))
        if rb == 0:
            band = chan
        else:
            band = np.stack(
                [_box_blur_2d(_box_blur_2d(_box_blur_2d(chan[..., c], rb), rb), rb)
                 for c in range(chan.shape[-1])], axis=-1)
        out += band * w[..., None]

    if img.ndim == 2:
        out = out[..., 0]
    return out.astype(np.float32), coc.astype(np.float32)
