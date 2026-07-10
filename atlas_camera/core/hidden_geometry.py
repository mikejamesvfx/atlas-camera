"""Hidden-surface selection from layered ray intersections (EXPERIMENTAL).

Pure-numpy consumers of a layered depth stack (H, W, L) — per pixel, the
ordered front-to-back depths of every surface the camera ray intersects, as
predicted by a layered-ray-intersection model (LaRI, World-Tracing-style).
Layer 0 is the visible surface; for a solid occluder layer 1 is usually its
own BACK face, and the background continuation lives in later layers — so
selection is per-pixel ("first layer that clears the occluder"), never a
fixed layer index. Verified empirically in the 2026-07-09 spike; see
docs/dev/hidden_geometry_training_free_research.md.

Everything here treats hidden geometry as a HYPOTHESIS with confidence,
never as fact (the report's discipline): outputs carry registration and
coverage statistics, and callers must surface provenance to the artist.
"""

from __future__ import annotations

from typing import Any


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Hidden-geometry selection requires numpy. Install with: pip install -e .[vision]"
        ) from exc


def register_layers_to_depth(
    layered_z: Any,
    visible_depth: Any,
    *,
    valid_min: float = 1e-6,
) -> tuple[float, float, Any]:
    """Robust scale registering a layered stack's layer-0 to a trusted depth map.

    ``layered_z`` (H, W, L) is in the layered model's own (normalized) units;
    ``visible_depth`` (H, W) is the pipeline's depth for the same pixels (e.g.
    the shared ATLAS_DEPTH_MAP's raw metric depth). Because layer 0 is the
    visible surface, a median per-pixel ratio registers the WHOLE stack into
    the pipeline's depth space — layers share the camera rays by construction.

    Returns ``(scale, rel_mad, valid_mask)`` where ``rel_mad`` (median absolute
    deviation of the ratio, relative to the scale) is the registration-quality
    signal: ~0.1 on architectural scenes, worse when layer-0 disagrees
    structurally with the trusted depth.
    """
    np = _require_numpy()
    layered_z = np.asarray(layered_z, dtype=np.float64)
    visible_depth = np.asarray(visible_depth, dtype=np.float64)
    z0 = layered_z[..., 0]
    valid = (z0 > valid_min) & (visible_depth > valid_min) & np.isfinite(visible_depth)
    if int(valid.sum()) < 100:
        return 1.0, float("inf"), valid
    ratio = visible_depth[valid] / z0[valid]
    scale = float(np.median(ratio))
    rel_mad = float(np.median(np.abs(ratio - scale)) / max(scale, 1e-12))
    return scale, rel_mad, valid


def fill_hidden_gaps(
    hidden: Any,
    hidden_valid: Any,
    region: Any,
    *,
    downsample: int = 8,
    iterations: int = 300,
) -> tuple[Any, Any]:
    """Diffuse sparse hidden-depth predictions across ``region`` -> one
    coherent surface.

    Per-pixel layer selection produces FRAGMENTED predictions (dense foliage:
    adjacent pixels pick different layers or none), and fragmented depth
    shreds the downstream relief mesh via its world-edge check regardless of
    ``depth_edge_rel`` (measured — see the 2026-07-09 calibration). This
    treats the valid predictions as Dirichlet samples of a single hidden
    surface and Jacobi-diffuses them across the rest of ``region`` on a
    ``downsample``d grid (dual-field trick: diffuse value*weight and weight,
    divide) — the same "invent smooth depth from trusted samples" move as
    ``build_relief_mesh``'s fill_occluded.

    Returns ``(hidden_out, valid_out)``: ``hidden_out`` has diffused values on
    reached region pixels (original predictions kept verbatim), ``valid_out``
    = predictions ∪ reached fill. Unreachable pixels stay invalid.
    """
    np = _require_numpy()
    hidden = np.asarray(hidden, dtype=np.float64)
    hidden_valid = np.asarray(hidden_valid, dtype=bool)
    region = np.asarray(region, dtype=bool)
    H, W = hidden.shape
    ds = max(1, int(downsample))
    hs, ws = (H + ds - 1) // ds, (W + ds - 1) // ds
    padH, padW = hs * ds - H, ws * ds - W
    hp = np.pad(hidden, ((0, padH), (0, padW)))
    vp = np.pad(hidden_valid, ((0, padH), (0, padW)))
    rp = np.pad(region, ((0, padH), (0, padW)))
    # Block pooling: seed value = mean of valid predictions per block.
    vb = vp.reshape(hs, ds, ws, ds)
    hb = (hp * vp).reshape(hs, ds, ws, ds)
    cnt = vb.sum(axis=(1, 3)).astype(np.float64)
    seed = np.divide(hb.sum(axis=(1, 3)), cnt, out=np.zeros((hs, ws)),
                     where=cnt > 0)
    seeded = cnt > 0
    reg_s = rp.reshape(hs, ds, ws, ds).any(axis=(1, 3))

    # Dual-field Jacobi: diffuse (value*weight) and (weight) together, keep
    # seeds pinned; weight > eps marks cells the diffusion actually reached.
    val = np.where(seeded, seed, 0.0)
    wgt = seeded.astype(np.float64)
    inside = reg_s & ~seeded
    for _ in range(int(iterations)):
        nv = (np.roll(val, 1, 0) + np.roll(val, -1, 0)
              + np.roll(val, 1, 1) + np.roll(val, -1, 1)) * 0.25
        nw = (np.roll(wgt, 1, 0) + np.roll(wgt, -1, 0)
              + np.roll(wgt, 1, 1) + np.roll(wgt, -1, 1)) * 0.25
        val = np.where(inside, nv, val)
        wgt = np.where(inside, nw, wgt)
    reached = seeded | (inside & (wgt > 1e-6))
    filled_s = np.divide(val, np.maximum(wgt, 1e-12))
    filled_s = np.where(reached, filled_s, 0.0)

    # Nearest-upsample back (blocks << the downstream mesh grid cell size;
    # the node's median smoothing softens block edges further).
    up = np.repeat(np.repeat(filled_s, ds, 0), ds, 1)[:H, :W]
    reached_up = np.repeat(np.repeat(reached, ds, 0), ds, 1)[:H, :W]
    fill_px = region & reached_up & ~hidden_valid
    hidden_out = np.where(fill_px, up, hidden)
    valid_out = hidden_valid | fill_px
    return hidden_out, valid_out


def select_hidden_surface(
    layered_z: Any,
    visible_depth: Any,
    *,
    clear_rel: float = 0.15,
    min_clear: float | None = None,
    valid_min: float = 1e-6,
) -> tuple[Any, Any, dict[str, Any]]:
    """Per-pixel first-clearing-layer selection -> hidden-surface depth map.

    For each pixel, walk layers 1..L-1 (registered into ``visible_depth``'s
    units via layer 0) and pick the FIRST one at least ``margin`` behind the
    visible surface, where ``margin = max(clear_rel * visible, min_clear)`` —
    scene-adaptive, because a fixed relative margin is too strict on shallow
    scenes and a fixed absolute margin too loose on deep ones (spike finding).
    ``min_clear`` defaults to 2% of the median visible depth.

    Returns ``(hidden_depth, hidden_valid, stats)``: ``hidden_depth`` (H, W) in
    ``visible_depth``'s units (0 where invalid), ``hidden_valid`` (H, W) bool,
    and a stats dict (registration scale/rel_mad, coverage, median separation)
    for the confidence report.
    """
    np = _require_numpy()
    layered_z = np.asarray(layered_z, dtype=np.float64)
    visible_depth = np.asarray(visible_depth, dtype=np.float64)
    if layered_z.ndim != 3 or layered_z.shape[-1] < 2:
        raise ValueError("layered_z must be (H, W, L>=2)")

    scale, rel_mad, valid0 = register_layers_to_depth(
        layered_z, visible_depth, valid_min=valid_min
    )
    stats: dict[str, Any] = {"scale": scale, "registration_rel_mad": rel_mad}
    if not np.isfinite(rel_mad):
        stats.update(coverage=0.0, median_separation=None,
                     warning="registration failed: too few valid layer-0 pixels")
        return np.zeros_like(visible_depth), np.zeros(visible_depth.shape, bool), stats

    zm = layered_z * scale                      # whole stack -> pipeline units
    behind = zm[..., 1:]                        # candidate hidden layers
    if min_clear is None:
        min_clear = 0.02 * float(np.median(visible_depth[valid0]))
    margin = np.maximum(clear_rel * visible_depth, float(min_clear))

    clears = (behind > (visible_depth + margin)[..., None]) & (behind > valid_min)
    has_clear = clears.any(axis=-1)
    first = np.argmax(clears, axis=-1)          # 0 when none; masked by has_clear
    hidden = np.take_along_axis(behind, first[..., None], axis=-1)[..., 0]
    hidden_valid = has_clear & valid0
    hidden = np.where(hidden_valid, hidden, 0.0)

    sep = hidden[hidden_valid] - visible_depth[hidden_valid]
    stats.update(
        coverage=float(hidden_valid.mean()),
        n_hidden_pixels=int(hidden_valid.sum()),
        median_separation=float(np.median(sep)) if sep.size else None,
        min_clear=float(min_clear),
        clear_rel=float(clear_rel),
        layer_used_histogram=(
            np.bincount(first[hidden_valid] + 1, minlength=layered_z.shape[-1])
            .tolist()
        ),
    )
    return hidden, hidden_valid, stats
