"""Extend a depth map past the frame, to match an outpainted plate.

THE GAP THIS FILLS
``AtlasCleanPlateLayer.frame_outpaint_px`` already widens a layer past the frame
edges, and its own tooltip says this closes "the frame-edge reveal that Safe Zone
measurements show is the binding constraint on wide scenes". But the ring it adds
is edge-replicated smear — "the ring is INVENTED pixels" — and depth cannot
follow it: ``AtlasMogeNormals`` explicitly refuses to run when
``frame_outpaint_px != 0`` because the normal map falls out of registration with
the widened plate.

So Atlas can widen the picture but not the geometry. A camera push that needs
those extra pixels gets colour with no surface under it.

Given a plate that has ALREADY been outpainted (by SDXL, by any generative node,
by hand), this re-runs depth on the widened image and stitches the result to the
original depth map.

WHY RE-ANCHORING IS THE WHOLE JOB
A monocular model run on the widened image returns a DIFFERENT scale than the
same model run on the original — different framing, different content, different
implied camera. Pasting the ring straight on puts a step at the frame boundary
that reads as a wall of geometry. The widened depth is therefore affine-fitted
onto the original across the region they share before anything is blended.

The interior always keeps the ORIGINAL depth. It was estimated from real pixels;
the widened pass saw invented ones and has no claim on the part we already knew.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Overlap samples below this and the affine fit is not trustworthy.
MIN_ANCHOR_SAMPLES = 512


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("depth outpainting requires numpy") from exc
    return np


@dataclass
class OutpaintedDepth:
    """Widened depth, the mask of invented pixels, and how the fit went."""

    depth: Any
    ring_mask: Any                 # True where depth is INVENTED, not measured
    scale: float = 1.0
    shift: float = 0.0
    anchor_samples: int = 0
    anchor_residual: float = 0.0   # median |anchored - original| on the overlap
    metadata: dict = field(default_factory=dict)


def ring_mask_for(width: int, height: int, pad: tuple, np) -> Any:
    """True outside the original frame, False inside it."""
    left, top, right, bottom = pad
    m = np.ones((height + top + bottom, width + left + right), dtype=bool)
    m[top:top + height, left:left + width] = False
    return m


def outpaint_depth(original_depth, widened_depth, *, pad, feather_px: int = 0,
                   anchor: bool = True) -> OutpaintedDepth:
    """Stitch ``widened_depth`` around ``original_depth``.

    ``pad`` is ``(left, top, right, bottom)`` in pixels — the amount the plate
    grew on each side, so ``widened_depth`` must be
    ``(H+top+bottom, W+left+right)``.

    ``feather_px`` blends the two across a band INSIDE the original frame. Zero
    keeps the original exactly and takes the new depth only outside it; a small
    value hides residual mismatch at the boundary at the cost of overwriting a
    few rows of real measurement with a mixture.
    """
    np = _require_numpy()
    od = np.asarray(original_depth, dtype=np.float64)
    wd = np.asarray(widened_depth, dtype=np.float64)
    if od.ndim != 2 or wd.ndim != 2:
        raise ValueError("both depth maps must be 2-D")

    left, top, right, bottom = (int(v) for v in pad)
    if min(left, top, right, bottom) < 0:
        raise ValueError(f"padding cannot be negative: {pad}")
    h, w = od.shape
    want = (h + top + bottom, w + left + right)
    if wd.shape != want:
        raise ValueError(
            f"widened depth is {wd.shape} but padding {pad} around a {od.shape} "
            f"plate implies {want} — the outpainted image and the padding disagree")

    inner = wd[top:top + h, left:left + w]

    # ---- anchor the widened pass onto the original -----------------------
    scale, shift, n, resid = 1.0, 0.0, 0, 0.0
    if anchor:
        a_ok = (np.isfinite(inner) & np.isfinite(od) & (inner > 0) & (od > 0))
        n = int(a_ok.sum())
        if n >= MIN_ANCHOR_SAMPLES:
            x, y = inner[a_ok], od[a_ok]
            if float(x.std()) > 1e-9:
                fit = np.polyfit(x, y, 1)
                if np.all(np.isfinite(fit)) and fit[0] > 0:
                    scale, shift = float(fit[0]), float(fit[1])
            else:
                shift = float(np.median(y) - np.median(x))
            resid = float(np.median(np.abs((x * scale + shift) - y)))

    adjusted = wd * scale + shift
    adjusted[np.isfinite(adjusted) & (adjusted <= 0)] = np.nan

    # ---- compose: original inside, adjusted outside ----------------------
    out = adjusted.copy()
    out[top:top + h, left:left + w] = od

    ring = ring_mask_for(w, h, (left, top, right, bottom), np)

    if feather_px > 0:
        # Ramp from the original toward the adjusted depth over a band just
        # INSIDE the frame edge, so any residual mismatch is spread rather than
        # landing on one pixel line. Only edges that actually grew are feathered
        # — blending an edge with no ring beyond it would corrupt real data for
        # nothing.
        f = int(feather_px)
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.full((h, w), np.inf, dtype=np.float64)
        if left > 0:
            dist = np.minimum(dist, xx)
        if right > 0:
            dist = np.minimum(dist, (w - 1) - xx)
        if top > 0:
            dist = np.minimum(dist, yy)
        if bottom > 0:
            dist = np.minimum(dist, (h - 1) - yy)
        t = np.clip(dist / max(1, f), 0.0, 1.0)          # 0 at edge, 1 inside
        blend = 0.5 - 0.5 * np.cos(np.pi * t)            # smoothstep
        inner_adj = adjusted[top:top + h, left:left + w]
        both = np.isfinite(inner_adj) & np.isfinite(od)
        mixed = np.where(both, blend * od + (1.0 - blend) * inner_adj, od)
        out[top:top + h, left:left + w] = mixed

    return OutpaintedDepth(
        depth=out.astype(np.float32),
        ring_mask=ring,
        scale=scale, shift=shift, anchor_samples=n, anchor_residual=resid,
        metadata={
            "pad": [left, top, right, bottom],
            "anchored": bool(anchor and n >= MIN_ANCHOR_SAMPLES),
            "feather_px": int(feather_px),
            "ring_fraction": float(ring.mean()),
        },
    )
