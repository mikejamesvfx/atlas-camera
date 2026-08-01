"""Tile layout, feather blending, and affine anchoring for large images.

Pure array arithmetic, host-agnostic: no torch, no transformers, no ComfyUI.
Lifted out of ``inference/depth_estimator.py`` (2026-08-01) because none of it
is vendor-specific — it takes numpy arrays in and gives numpy arrays back, and
belongs anywhere a plate is too large to process in one pass.

Its first consumer is native-resolution tiled depth inference. A monocular model
resamples its input to a fixed token budget, so a 36 MP plate is effectively
inferred at a fraction of its resolution. Running the model over source-scale
crops raises effective resolution with no training and no new model.

THE PART THAT IS NOT OBVIOUS: the tiles cannot simply be pasted together.
Monocular depth is scale- and shift-ambiguous per input, so a tile of sky and a
tile of pavement come back on different scales even from a "metric" model.
Pasting them puts a step at every seam, and feathering only turns a hard step
into a soft one. Every tile is therefore fitted onto ONE global low-resolution
pass first (:func:`fit_affine_to_reference`), so they share a frame of reference
before :func:`assemble_tiles` blends them.

Numpy is passed in by callers that already hold it (the historical signature,
kept so existing call sites are untouched) or resolved lazily via
``_require_numpy`` when omitted — either way it is never imported at module
scope, which keeps ``core`` importable with zero dependencies.
"""

from __future__ import annotations

from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Image tiling requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


def tile_boxes(width: int, height: int, tile_side: int, overlap: float) -> list:
    """Cover ``width x height`` in tiles of ~``tile_side``, overlapping by ``overlap``.

    Returns ``[(x0, y0, x1, y1), ...]`` in source pixels. Tiles are distributed
    EVENLY rather than laid left-to-right with a ragged remainder: a thin final
    strip would get its own inferred scale from almost no context, which is the
    worst possible input to a monocular model and shows up as a bright or dark
    band down one edge.
    """
    step = max(1, int(round(tile_side * (1.0 - float(overlap)))))

    def starts(total: int) -> list:
        if total <= tile_side:
            return [0]
        n = int(-(-(total - tile_side) // step)) + 1        # ceil division
        # Spread the n tiles evenly across the axis so every tile is full-size.
        return [int(round(i * (total - tile_side) / (n - 1))) for i in range(n)]

    return [(x, y, min(x + tile_side, width), min(y + tile_side, height))
            for y in starts(height) for x in starts(width)]


def _feather_weights(h: int, w: int, box, width: int, height: int, ramp: int,
                     np: Any = None):
    """Cosine ramp toward tile edges, except where the tile meets the frame edge.

    Feathering an edge that has no neighbour to blend with would fade the
    outermost pixels toward zero and leave a dark rim around the whole plate.
    """
    if np is None:
        np = _require_numpy()
    x0, y0, x1, y1 = box
    wx = np.ones(w, dtype=np.float64)
    wy = np.ones(h, dtype=np.float64)
    if ramp > 0:
        t = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, min(ramp, w // 2 or 1)))
        if x0 > 0:
            wx[:len(t)] = np.minimum(wx[:len(t)], t)
        if x1 < width:
            wx[-len(t):] = np.minimum(wx[-len(t):], t[::-1])
        t = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, min(ramp, h // 2 or 1)))
        if y0 > 0:
            wy[:len(t)] = np.minimum(wy[:len(t)], t)
        if y1 < height:
            wy[-len(t):] = np.minimum(wy[-len(t):], t[::-1])
    return np.outer(wy, wx)


def fit_affine_to_reference(tile_depth, reference, np: Any = None,
                            min_samples: int = 64):
    """Least-squares ``(a, b)`` minimising ``|a*tile + b - reference|``.

    THE reason tiling needs more than a paste. Monocular depth is scale- and
    shift-ambiguous per input, so a tile of sky and a tile of pavement come back
    on different scales even from a "metric" model. Stitching them directly puts
    a visible step at every seam that no amount of feathering hides — the blend
    just turns a hard step into a soft one.

    Anchoring every tile to one global low-resolution pass puts them all in a
    single frame of reference first; the feather then only has to hide model
    noise, which it can.

    Returns ``(1.0, 0.0)`` when there is too little overlap to fit — better a
    tile that is merely unadjusted than one warped by a fit on ten pixels.
    """
    if np is None:
        np = _require_numpy()
    t = np.asarray(tile_depth, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(r) & (t > 0) & (r > 0)
    if int(ok.sum()) < min_samples:
        return 1.0, 0.0
    t, r = t[ok], r[ok]
    # Guard a degenerate tile (flat depth): the normal equations are singular
    # and would produce an enormous `a`.
    if float(t.std()) < 1e-6:
        return 1.0, float(np.median(r) - np.median(t))
    a, b = np.polyfit(t, r, 1)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return 1.0, 0.0
    return float(a), float(b)


def assemble_tiles(tiles: list, width: int, height: int, ramp: int,
                   np: Any = None):
    """Feather-blend ``[(box, depth_tile), ...]`` into one HxW map.

    Weights accumulate per pixel and the sum divides at the end, so overlapping
    regions are a true weighted mean rather than whichever tile happened to be
    written last.
    """
    if np is None:
        np = _require_numpy()
    acc = np.zeros((height, width), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)
    for box, tile in tiles:
        x0, y0, x1, y1 = box
        th, tw = tile.shape[:2]
        w = _feather_weights(th, tw, box, width, height, ramp, np)
        finite = np.isfinite(tile)
        acc[y0:y1, x0:x1] += np.where(finite, tile, 0.0) * w * finite
        wsum[y0:y1, x0:x1] += w * finite
    out = np.full((height, width), np.nan, dtype=np.float32)
    hit = wsum > 1e-9
    out[hit] = (acc[hit] / wsum[hit]).astype(np.float32)
    return out
