"""Per-band projection geometry — the `relief` / `card` / `ground` maths.

Extracted verbatim from ``AtlasCleanPlateLayer`` (2026-08-01) so the
camera/geometry arithmetic behind a band layer lives in host-agnostic core
and is unit-testable without constructing a ComfyUI node. Pure numpy — no
torch, no ComfyUI, no DCC conversions.

The doctrine these functions implement (docs/DESIGN_RULES.md, "Per-band
geometry types"): the flat modes substitute the depth FIELD fed to
``relief_mesh.build_relief_mesh``; **band membership still comes from the REAL
depth (which pixels belong to this layer); geometry only changes WHERE those
pixels sit.** Out-of-region pixels become NaN, which is
invalid-but-regrowable exactly like band clipping (matte skirts still grow),
while real exclusions stay the hard skirt forbid.

``card`` is one fronto-parallel plane at the band's median RAW depth (the
projection_backdrop / sky-dome constant-forward-Z convention). ``ground`` is
the exact analytic Y=0 plane along each pixel ray, returned in raw units
(``metric / scale``) so ``build_relief_mesh``'s internal rescale-about-camera
lands vertices on Y=0 on the nose.

Convention (critical, identical to depth_geometry.py / relief_mesh.py):
always the full 4x4 ``extrinsics.camera_view_matrix`` (row-major, world->cam),
never the 3x3 rotation.
"""

from __future__ import annotations

from typing import Any

from atlas_camera.core.depth_geometry import _analytic_ground_forward_depth

#: Percentile of the band's REAL metric depth used to bound analytic ground
#: depth when the band has no far edge (see :func:`ground_cap_metres`).
GROUND_CAP_PERCENTILE = 99.0

#: Multiplier applied to that percentile to form the open-band ground cap.
GROUND_CAP_FACTOR = 4.0

#: Boundary cells a matte-carrying layer always overhangs by, before any
#: edge-extend or exclusion-choke allowance.
BASE_OVERHANG_CELLS = 2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Band geometry requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


def band_membership(valid, metric, near, far, *, fill_mask=None, exclude_mask=None):
    """Return the boolean mask of pixels belonging to the ``[near, far]`` band.

    Membership always comes from the REAL metric depth, never from the
    substituted flat field. Both band edges are inclusive; ``far`` may be
    ``float("inf")`` for an open band. ``fill_mask`` (the ``fill_occluded``
    footprint of nearer-than-band occluders) is UNIONED in — flat depth covers
    that footprint for free, so it is included in the region instead of being
    diffusion-filled — minus ``exclude_mask`` when one is supplied, because
    excluded (sky) regions are never filled.
    """
    np = _require_numpy()
    # NOTE: the membership expression's operator precedence is preserved
    # exactly as it ran inside AtlasCleanPlateLayer — do not "simplify".
    band_region = valid & (metric >= near)
    if far != float("inf"):
        band_region &= metric <= far
    if fill_mask is not None:
        band_region = band_region | (
            fill_mask if exclude_mask is None else (fill_mask & ~exclude_mask))
    return band_region


def card_plane_depth(depth, band_region) -> float:
    """Median RAW depth of the in-band pixels — the card plane's constant.

    One fronto-parallel plane at the band's median depth: the classic DMP
    card, matching the projection_backdrop / sky dome constant-forward-Z
    convention. An empty band falls back to ``1.0`` (the region is all-NaN
    downstream anyway, so the value never reaches a vertex).
    """
    np = _require_numpy()
    return float(np.median(depth[band_region])) if band_region.any() else 1.0


def ground_plane_depth_field(extr, fx, fy, cx, cy, height, width):
    """Per-pixel forward depth of the ray n (Y=0 ground plane), in METRES.

    NaN where the ray never hits ground (at/above the horizon). Raises when no
    ray hits Y=0 at all — i.e. the solved camera sits on or below the ground
    plane, which no amount of band tuning can rescue.
    """
    np = _require_numpy()
    geo_metric = _analytic_ground_forward_depth(extr, fx, fy, cx, cy, height, width)
    if not np.isfinite(geo_metric).any():
        raise ValueError(
            "band_geometry='ground' needs a camera above the ground plane "
            "(solved camera height <= 0, or no ray ever hits Y=0).")
    return geo_metric


def ground_cap_metres(metric, band_region, far) -> float:
    """Upper bound on analytic ground depth admitted into the band.

    Non-ground pixels in the band (a wall base, an occluder's side) have
    analytic ground depths FAR beyond the band — near-horizontal rays run out
    toward the horizon. Cap at the band's far edge, or at
    ``GROUND_CAP_FACTOR`` x the band's real ``GROUND_CAP_PERCENTILE`` depth
    when the band is open-ended, so only plausible ground-plane membership
    survives; the rest become holes/skirt.
    """
    np = _require_numpy()
    if far != float("inf"):
        return float(far)
    if band_region.any():
        return GROUND_CAP_FACTOR * float(
            np.percentile(metric[band_region], GROUND_CAP_PERCENTILE))
    return float("inf")


def flat_band_depth_field(geometry, *, depth, metric, valid, near, far, scale,
                          extr, fx, fy, cx, cy, height, width,
                          fill_mask=None, exclude_mask=None):
    """Depth field to feed ``build_relief_mesh`` for this band's geometry type.

    ``geometry == "relief"`` returns ``depth`` unchanged (the caller keeps the
    normal band-clipped relief path). ``card`` and ``ground`` return a field
    that is the flat geometry inside the band region and NaN everywhere else,
    in the SAME raw units as ``depth`` (the builder's rescale-about-camera is
    what turns them back into metres).

    ``scale`` is the solve's metric scale (metres per raw depth unit); it is
    floored at 1e-9 to keep a degenerate solve from dividing by zero.
    """
    np = _require_numpy()
    if geometry == "relief":
        return depth

    band_region = band_membership(valid, metric, near, far,
                                  fill_mask=fill_mask, exclude_mask=exclude_mask)
    if geometry == "card":
        const_raw = card_plane_depth(depth, band_region)
        geo_depth = np.full(depth.shape, const_raw, dtype=np.float64)
    else:  # ground
        geo_metric = ground_plane_depth_field(extr, fx, fy, cx, cy, height, width)
        band_region &= np.isfinite(geo_metric)
        ground_cap = ground_cap_metres(metric, band_region, far)
        with np.errstate(invalid="ignore"):
            band_region &= ~(geo_metric > ground_cap)
        geo_depth = geo_metric / max(float(scale), 1e-9)
    return np.where(band_region, geo_depth, np.nan)


def boundary_overhang_cells(*, embed_matte, edge_extend_px, relief_grid,
                            height, width, choke_cells) -> int:
    """Grid cells the mesh's boundary skirt overhangs by.

    A matte-carrying layer always overhangs ``BASE_OVERHANG_CELLS``; an
    edge-extend needs enough extra rings to receive the smear (the extension
    in pixels over the grid's cell size, rounded up); and the choked ring must
    be fully regrown by the skirt BEFORE extending, so the choke is added on
    top. Without a matte there is no edge to extend past and the overhang is 0.
    """
    np = _require_numpy()
    if not embed_matte:
        return 0
    overhang_cells = BASE_OVERHANG_CELLS
    if edge_extend_px and int(edge_extend_px) > 0:
        cell_px = max(1, int(round(max(height, width) / max(int(relief_grid), 2))))
        overhang_cells = BASE_OVERHANG_CELLS + int(np.ceil(int(edge_extend_px) / cell_px))
    # The skirt must regrow the choked ring fully before extending.
    return overhang_cells + int(choke_cells)
