"""Binary mask and scalar-field neighbourhood ops, written once.

WHY THIS MODULE EXISTS. Six independent binary dilations lived in the node
layer — three of them inside a single method of `AtlasAddPatchView`, one in
`AtlasOcclusionMask`, one in `nodes_depth`, one in `AtlasDisocclusionGuide` —
and they did not agree. The `np.roll` implementation WRAPPED at the frame
border, so a mask touching the left edge grew onto the right edge; the defect
was recorded in a comment beside the code instead of in a test.

Clamped borders are the only sensible semantic for an image mask: there is no
geometry off the left edge that continues on the right. `wrap=True` exists so
an equirectangular caller can ask for the other behaviour deliberately.

Host-agnostic: numpy only.
"""
from __future__ import annotations

from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.mask_ops requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFFSETS_8 = _OFFSETS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _shift(mask: Any, dr: int, dc: int, *, wrap: bool) -> Any:
    np = _require_numpy()
    if wrap:
        return np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
    out = np.zeros_like(mask)
    rows, cols = mask.shape[:2]
    dst_r = slice(max(dr, 0), rows + min(dr, 0))
    dst_c = slice(max(dc, 0), cols + min(dc, 0))
    src_r = slice(max(-dr, 0), rows + min(-dr, 0))
    src_c = slice(max(-dc, 0), cols + min(-dc, 0))
    out[dst_r, dst_c] = mask[src_r, src_c]
    return out


def dilate(mask: Any, iterations: int = 1, *, connectivity: int = 4,
           wrap: bool = False) -> Any:
    """Grow `mask` by `iterations` steps. Borders CLAMP unless `wrap` is set.

    Returns a new array; the input is never mutated.
    """
    np = _require_numpy()
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    out = np.asarray(mask, dtype=bool).copy()
    offsets = _OFFSETS_4 if connectivity == 4 else _OFFSETS_8
    for _ in range(max(0, int(iterations))):
        grown = out
        for dr, dc in offsets:
            grown = grown | _shift(out, dr, dc, wrap=wrap)
        out = grown
    return out


def box_blur(field: Any, radius: int, *, wrap: bool = False) -> Any:
    """Separable box blur over a float field, normalized per pixel.

    Normalizing by the count of contributing samples rather than by the window
    area is what keeps edge pixels unbiased — a border pixel sees fewer
    neighbours, and dividing by the full window would darken the frame edge.
    """
    np = _require_numpy()
    src = np.asarray(field, dtype=np.float64)
    r = max(0, int(radius))
    if r == 0:
        return src.copy()
    total = np.zeros_like(src)
    count = np.zeros_like(src)
    ones = np.ones_like(src)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            total += _shift(src, dr, dc, wrap=wrap)
            count += _shift(ones, dr, dc, wrap=wrap)
    return total / np.where(count > 0, count, 1.0)
