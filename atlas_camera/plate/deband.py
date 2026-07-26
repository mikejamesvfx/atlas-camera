"""Model-free debanding for 8-bit-born plates — the numpy side.

AI-generated and JPEG plates arrive 8-bit: smooth skies and gradients carry
quantization plateaus that survive the float conversion and read as contour
rings once projected onto a sky dome or graded in ACEScg. This is the classic
VFX prep fix, done deterministically (no model, no GPU):

  * gradient-gated: only pixels whose local luma gradient sits BELOW the
    banding threshold (plateaus) are touched — real edges gate to zero and
    pass through bit-exact;
  * masked normalized blur: the smoothing average is weighted by the same
    gate, so values never bleed across a gated edge;
  * range-clamped: the correction per pixel is clipped to a few banding steps,
    so even a mis-gated pixel cannot move visibly;
  * optional triangular-noise grain to break any residual contour.

Float-safe by construction: no quantization, no clamping of HDR values — a
pixel the gate does not select is returned exactly as it came in, and >1.0
values ride through the arithmetic untouched.

Lives in ``plate/`` (host-agnostic, guarded numpy) beside ``ops.py`` because it
is a thing you do to a PLATE before projection — not a generic image op, and
deliberately NOT part of the P0 plate-ref trust path: the ComfyUI node consumes
and returns IMAGE tensors only, so a debanded tensor can never masquerade as a
registered final plate.
"""

from __future__ import annotations


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of the package
        raise ImportError("plate deband needs numpy — pip install -e .[vision]") from exc
    return np


# Correction is clipped to this many banding steps: plateaus differ by ~1 LSB,
# so a few steps is all a legitimate fix ever needs — anything larger would be
# cross-edge bleed.
_DELTA_CLAMP_STEPS = 4.0


def _box_blur_2d(a, radius: int):
    """Separable box blur with edge-replicated borders (cumsum, O(n))."""
    np = _require_numpy()
    r = int(radius)
    if r <= 0:
        return a.copy()

    def blur_axis(x, axis):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r + 1, r)
        xp = np.pad(x, pad, mode="edge")
        c = np.cumsum(xp, axis=axis, dtype=np.float64)
        upper = np.take(c, range(2 * r + 1, xp.shape[axis]), axis=axis)
        lower = np.take(c, range(0, xp.shape[axis] - 2 * r - 1), axis=axis)
        return (upper - lower) / (2 * r + 1)

    return blur_axis(blur_axis(a.astype(np.float64), 0), 1)


def deband_plate(
    image,
    *,
    strength: float = 0.5,
    band_threshold_lsb: float = 2.0,
    radius_px: int = 24,
    preserve_detail: float = 0.5,
    grain: float = 0.0,
    seed: int = 0,
):
    """Deband a HxWxC (or HxW) float plate. Returns float32, same shape.

    ``band_threshold_lsb`` is the largest luma step, in 8-bit LSB units, still
    treated as banding; ``radius_px`` the smoothing reach; ``preserve_detail``
    sharpens the edge gate (1.0 = most conservative); ``grain`` adds triangular
    dither of that amplitude (in [0,1] units) on gated pixels only.
    ``strength=0`` is an exact identity.
    """
    np = _require_numpy()
    img = np.asarray(image, dtype=np.float64)
    s = float(np.clip(strength, 0.0, 1.0))
    if s == 0.0 or img.size == 0:
        return img.astype(np.float32)

    chan = img if img.ndim == 3 else img[..., None]
    luma = chan.mean(axis=-1)

    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = 0.5 * np.abs(luma[:, 2:] - luma[:, :-2])
    gy[1:-1, :] = 0.5 * np.abs(luma[2:, :] - luma[:-2, :])
    gmag = np.maximum(gx, gy)

    thr = max(float(band_threshold_lsb), 1e-3) / 255.0
    # The gate must be ~1 ON the banding step columns themselves (their
    # gradient is a fraction of thr — they are precisely the pixels that need
    # the correction; a gate proportional to 1-g/thr under-corrects exactly
    # there and leaves a residual staircase, found by the banding-energy
    # test). So: full pass below a rolloff knee, zero at/above thr.
    # preserve_detail moves the knee down (1.0 = most conservative).
    pd = float(np.clip(preserve_detail, 0.0, 1.0))
    lo = thr * (0.25 + 0.5 * (1.0 - pd))
    gate = np.clip((thr - gmag) / max(thr - lo, 1e-9), 0.0, 1.0)

    # Iterated masked box blur ~ Gaussian; the gate weights the average so a
    # hard edge (gate 0) contributes nothing to its neighbours' smoothing.
    r = max(1, int(radius_px) // 3)
    num = chan * gate[..., None]
    den = gate
    for _ in range(3):
        num = np.stack([_box_blur_2d(num[..., c], r) for c in range(num.shape[-1])], axis=-1)
        den = _box_blur_2d(den, r)
    smoothed = num / np.maximum(den, 1e-6)[..., None]

    delta = np.clip(smoothed - chan,
                    -_DELTA_CLAMP_STEPS * thr, _DELTA_CLAMP_STEPS * thr)
    out = chan + s * gate[..., None] * delta

    g = float(np.clip(grain, 0.0, 1.0))
    if g > 0.0:
        rng = np.random.default_rng(int(seed))
        tri = rng.random(luma.shape) - rng.random(luma.shape)  # triangular dither
        out = out + (g * s) * (gate * tri)[..., None]

    if img.ndim == 2:
        out = out[..., 0]
    return out.astype(np.float32)
