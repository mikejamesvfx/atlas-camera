"""Mask geometry shared by every external-paint bridge.

Moved verbatim out of the original Affinity confine tool (2026-08-21, since
superseded by ``tools/paint_confine_plate.py``) so the behaviour these
functions had when the Affinity numbers were measured is the behaviour the
tests now pin. Any later tuning shows up as a test diff rather than as silent
drift in results that are already published.

One deliberate exception, made immediately after the move because the new
parity test caught it: the SciPy-free dilation fallback was a separable BOX
built from ``np.roll``, which both produced a square instead of a disc and
WRAPPED around the frame edges. See ``_dilate``.

Each takes ``np`` as its first argument: the callers import numpy lazily so the
core package keeps its zero-required-dependency guarantee.
"""
from __future__ import annotations


def _disc(np, radius: int):
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _hband(np, mask, half_width: int):
    """Dilate horizontally by `half_width`, WITHOUT wrapping at the edges.

    A running window over a cumulative sum, so the cost does not grow with the
    radius: window [x-k, x+k] is non-empty iff its summed count is positive.
    """
    if half_width <= 0:
        return mask
    k = int(half_width) * 2 + 1
    pad = int(half_width)
    # Zero padding, not edge padding: a pixel outside the frame is absent, not
    # a copy of the border. Edge padding here would smear the frame's own edge
    # inward and quietly enlarge the authorised region along every border.
    padded = np.pad(mask.astype(np.int32), ((0, 0), (pad, pad)), mode="constant")
    zeros = np.zeros((padded.shape[0], 1), dtype=np.int32)
    csum = np.cumsum(np.concatenate([zeros, padded], axis=1), axis=1)
    n = mask.shape[1]
    return (csum[:, k:k + n] - csum[:, 0:n]) > 0


def _dilate(np, mask, radius: int):
    """Binary dilation by a DISC, via SciPy when present, else an exact fallback.

    The fallback must agree with SciPy pixel for pixel, because the result is
    the authorised region: if it did not, containment would silently mean
    something different on a machine without SciPy installed.

    It did not, until 2026-08-21. The original fallback was a separable BOX
    dilation built from ``np.roll``, which was wrong twice over — a square
    instead of a disc (56 corner pixels too many at r=6 on a single point), and
    ``roll`` WRAPS, so a mask touching the left edge grew onto the right edge of
    the frame. Both were caught the moment the two paths were compared.

    The exact version decomposes the disc by row: for each vertical offset
    ``dy``, the disc spans ``dx = floor(sqrt(r^2 - dy^2))`` horizontally, so
    OR-ing a horizontally-dilated, vertically-shifted copy per ``dy`` builds
    the true disc. Slicing rather than rolling keeps it non-wrapping.
    """
    if radius <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation           # optional
        return binary_dilation(mask, structure=_disc(np, radius))
    except ImportError:
        pass

    r = int(radius)
    height = mask.shape[0]
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-r, r + 1):
        dx = int(np.floor(np.sqrt(max(0, r * r - dy * dy))))
        src0, src1 = max(0, -dy), height - max(0, dy)
        if src0 >= src1:
            continue
        dst0, dst1 = max(0, dy), height - max(0, -dy)
        out[dst0:dst1] |= _hband(np, mask[src0:src1], dx)
    return out


def _drop(np, mask, distance: int):
    """Extend the mask downward by `distance` px per column (gravity-directed).

    Rows are y-down (image origin top-left), so 'below the object' is a
    positive row shift. Cumulative OR over the shifts fills the whole column
    run rather than only its far end.
    """
    if distance <= 0:
        return mask
    out = mask.copy()
    step = 1
    remaining = int(distance)
    # Doubling shifts: OR-ing a run of length n with itself shifted by n
    # yields length 2n, so a `distance`-long run costs log2(distance) passes.
    while step <= remaining:
        shifted = np.zeros_like(out)
        shifted[step:, :] = out[:-step, :]
        out |= shifted
        remaining -= step
        step *= 2
    if remaining > 0:
        shifted = np.zeros_like(out)
        shifted[remaining:, :] = out[:-remaining, :]
        out |= shifted
    return out


def _feather(np, mask_f, radius: int):
    """Box-blur the 0/1 mask `radius` px into a ramp (two passes ~ Gaussian)."""
    if radius <= 0:
        return mask_f
    k = int(radius) * 2 + 1
    pad = int(radius)
    out = mask_f
    for _ in range(2):
        for axis in (0, 1):
            widths = [(pad, pad) if a == axis else (0, 0) for a in range(out.ndim)]
            padded = np.pad(out, widths, mode="edge")
            # Prepend a zero plane so window [i, i+k) is csum0[i+k] - csum0[i]
            # with no negative index at i = 0.
            zeros = np.zeros_like(np.take(padded, [0], axis=axis))
            csum = np.cumsum(np.concatenate([zeros, padded], axis=axis), axis=axis)
            n = out.shape[axis]
            lo = np.take(csum, np.arange(0, n), axis=axis)
            hi = np.take(csum, np.arange(k, k + n), axis=axis)
            out = (hi - lo) / float(k)
    return np.clip(out, 0.0, 1.0)



# Public names. The underscore-prefixed originals are kept exactly as they were
# in tools/affinity_confine_plate.py so the move stays reviewable as a move;
# these are what new code should import.
disc = _disc
dilate = _dilate
drop = _drop
feather = _feather

__all__ = ["disc", "dilate", "drop", "feather",
           "_disc", "_dilate", "_drop", "_feather"]
