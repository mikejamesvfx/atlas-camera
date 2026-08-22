"""Contracts for splat-fused occlusion-hole repair — the torch-free half.

WHAT A HOLE IS HERE, AND WHY IT IS NOT ``ReliefMesh.hole_mask``. An occlusion
hole is the region behind an occluder that no camera photographed. In the
PRIMARY view it has zero extent by construction: it projects to exactly the
pixels the occluder covers. ``ReliefMesh.hole_mask`` in that frame is the
depth-cliff band plus sky — a real thing, but not this thing. The hole only
acquires area once the camera MOVES, which is why every mask this module
consumes is expressed in a moved camera's raster, produced by
``dynamic.occlusion_fill.render_disocclusion_sequence`` and clustered by
``survey_hole_rois``.

That module lives in ``dynamic/``, which may import ``core`` and never the
reverse — so nothing here imports it. Masks, z-buffers and cameras arrive as
plain arrays; the driver does the rendering. It also keeps every function in
this file testable on synthetic data with no scene, no GPU and no ComfyUI.

THE SEED IS VOLUMETRIC, NOT A POINT CLOUD. The usual "back-project the depth
map" initialisation cannot work inside a hole: there is no depth there to
back-project. What IS measured is the hole's RIM — the near lip of the
occluder and the far surface that continues behind it. So the seed fills the
frustum slab between those two measured depths, and parallax across the
registered fill views is what resolves it. A seed outside that interval is not
a worse guess, it is an unmeasured one.

OWNERSHIP. The mesh owns every pixel with measured depth; splats own only the
hole interior plus a fixed overlap band used for blending. Stated once here so
the seam cannot be decided differently in two places (brief failure mode 2).

Host-agnostic: numpy only, no torch, no ComfyUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: ``metadata`` tag for anything this module seeds. Splat output is the
#: ``generated`` trust tier; verdicts still come only from ``core.scene_health``.
SPLAT_SOURCE = "hole_splat_fusion"

#: Pixels of overlap between the splat region and the surrounding mesh. Wide
#: enough to blend, narrow enough that the mesh keeps everything it measured.
DEFAULT_OVERLAP_PX = 8

#: Depth samples per seeded ray between the near and far rim.
DEFAULT_SLAB_LAYERS = 6


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise RuntimeError(
            "atlas_camera.core.hole_splat requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def _dilate(mask: Any, radius: int) -> Any:
    """Square-structuring-element dilation, no scipy."""
    np = _require_numpy()
    out = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return out.copy()
    for _ in range(int(radius)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


@dataclass(frozen=True, slots=True)
class HoleOwnership:
    """Who renders which pixel, decided once."""

    splat_mask: Any
    mesh_mask: Any
    overlap_mask: Any
    report: dict = field(default_factory=dict)


def hole_ownership(
    hole_mask: Any,
    coverage: Any,
    *,
    overlap_px: int = DEFAULT_OVERLAP_PX,
    sky_mask: Any = None,
) -> HoleOwnership:
    """Split a moved view into mesh-owned, splat-owned and blend pixels.

    ``hole_mask`` is the disocclusion mask in this view (True where nothing
    covered), ``coverage`` the rasteriser's own coverage (True where the mesh
    rendered). ``sky_mask`` is subtracted from the splat region and never added
    back: sky holes are not occlusion holes, and the existing survey path
    already refuses them for the same reason.

    The mesh keeps every covered pixel. Splats get the hole minus sky. The
    overlap band is grown from the hole INTO covered territory, so blending has
    somewhere to happen without the splats claiming measured pixels.
    """

    np = _require_numpy()
    hole = np.asarray(hole_mask, dtype=bool)
    covered = np.asarray(coverage, dtype=bool)
    if hole.shape != covered.shape:
        raise ValueError(
            f"hole_mask {hole.shape} and coverage {covered.shape} must match")
    if hole.ndim != 2:
        raise ValueError("masks must be 2D (H, W)")

    dropped_sky = 0
    if sky_mask is not None:
        sky = np.asarray(sky_mask, dtype=bool)
        if sky.shape != hole.shape:
            raise ValueError("sky_mask must match hole_mask")
        dropped_sky = int((hole & sky).sum())
        hole = hole & ~sky

    splat = hole & ~covered
    overlap = _dilate(splat, int(overlap_px)) & covered
    return HoleOwnership(
        splat_mask=splat,
        mesh_mask=covered,
        overlap_mask=overlap,
        report={
            "splat_px": int(splat.sum()),
            "mesh_px": int(covered.sum()),
            "overlap_px": int(overlap.sum()),
            "sky_px_dropped": dropped_sky,
            "overlap_radius_px": int(overlap_px),
        },
    )


@dataclass(frozen=True, slots=True)
class RimDepth:
    """The measured depth interval a hole is bounded by."""

    near_m: float
    far_m: float
    n_ring_px: int
    report: dict = field(default_factory=dict)


def rim_depth_interval(
    zbuffer: Any,
    splat_mask: Any,
    *,
    ring_px: int = DEFAULT_OVERLAP_PX,
    near_percentile: float = 5.0,
    far_percentile: float = 95.0,
) -> RimDepth:
    """Depth bounds for a hole, taken from the measured ring around it.

    ``zbuffer`` is forward distance in metres with ``inf`` where uncovered —
    exactly what ``move_budget.rasterize_coverage`` returns. The ring is the
    dilation of the hole intersected with the finite part of the buffer, so
    every sample is a real rendered surface.

    Percentiles rather than min/max: one stray pixel of far backdrop bleeding
    into the ring would otherwise stretch the slab to the horizon and put the
    entire seed budget where nothing is.
    """

    np = _require_numpy()
    z = np.asarray(zbuffer, dtype=np.float64)
    splat = np.asarray(splat_mask, dtype=bool)
    if z.shape != splat.shape:
        raise ValueError("zbuffer and splat_mask must have the same shape")

    ring = _dilate(splat, int(ring_px)) & ~splat & np.isfinite(z) & (z > 0)
    samples = z[ring]
    if samples.size < 8:
        raise ValueError(
            f"hole rim has only {samples.size} measured pixels; cannot bound "
            "the seed volume — widen ring_px or pick a hole with a real rim")

    near = float(np.percentile(samples, float(near_percentile)))
    far = float(np.percentile(samples, float(far_percentile)))
    if not (near > 0 and far > near):
        raise ValueError(
            f"degenerate rim interval near={near:.4f} far={far:.4f}; the ring "
            "found no depth range to seed between")
    return RimDepth(
        near_m=near,
        far_m=far,
        n_ring_px=int(samples.size),
        report={
            "ring_radius_px": int(ring_px),
            "near_percentile": float(near_percentile),
            "far_percentile": float(far_percentile),
            "ring_depth_median_m": float(np.median(samples)),
        },
    )


@dataclass(frozen=True, slots=True)
class HoleSeed:
    """A volumetric gaussian seed for one hole, in Atlas world metres."""

    points_world: Any
    colors: Any
    near_m: float
    far_m: float
    scale_m: float
    report: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(len(self.points_world))


def seed_hole_volume(
    splat_mask: Any,
    rim: RimDepth,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    layers: int = DEFAULT_SLAB_LAYERS,
    pixel_stride: int = 2,
    colors: Any = None,
    seed: int = 0,
) -> HoleSeed:
    """Fill the hole's frustum slab with gaussians between the rim depths.

    Rays are the same pinhole unprojection the rest of core uses — the full 4x4
    ``camera_view_matrix``, +X right, +Y up, -Z forward, ``z = -depth``, image
    origin top-left (``depth_geometry.back_project_normals``). Never rebuild
    this from the 3x3 rotation.

    Depth placement is stratified: each ray gets one sample per layer inside its
    own depth stratum, jittered by a seeded generator. Stratified rather than
    uniform-random so a small seed budget still spans the slab, and seeded so a
    re-run reproduces the experiment exactly.
    """

    np = _require_numpy()
    splat = np.asarray(splat_mask, dtype=bool)
    if splat.ndim != 2:
        raise ValueError("splat_mask must be 2D (H, W)")
    if int(layers) < 1:
        raise ValueError("layers must be >= 1")
    if int(pixel_stride) < 1:
        raise ValueError("pixel_stride must be >= 1")
    for name, value in (("fx", fx), ("fy", fy)):
        if not float(value) > 0:
            raise ValueError(f"{name} must be positive")

    view = np.asarray(view_matrix, dtype=np.float64)
    if view.shape != (4, 4):
        raise ValueError("view_matrix must be the full 4x4 camera_view_matrix")
    cam_to_world = np.linalg.inv(view)
    rot_cw = cam_to_world[:3, :3]
    cam_pos = cam_to_world[:3, 3]

    rows, cols = np.nonzero(splat)
    if not len(rows):
        raise ValueError("splat_mask is empty; nothing to seed")
    rows = rows[:: int(pixel_stride)]
    cols = cols[:: int(pixel_stride)]

    rng = np.random.default_rng(int(seed))
    n_rays = len(rows)
    n_layers = int(layers)

    # Stratified depth: layer k occupies [near + k*step, near + (k+1)*step).
    step = (float(rim.far_m) - float(rim.near_m)) / n_layers
    strata = np.arange(n_layers, dtype=np.float64)[None, :]
    jitter = rng.random((n_rays, n_layers))
    depths = float(rim.near_m) + (strata + jitter) * step

    u = cols.astype(np.float64)[:, None]
    v = rows.astype(np.float64)[:, None]
    x = (u - float(cx)) / float(fx) * depths
    y = -(v - float(cy)) / float(fy) * depths
    z = -depths
    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pts_world = pts_cam @ rot_cw.T + cam_pos

    if colors is None:
        rgb = np.full((len(pts_world), 3), 0.5, dtype=np.float32)
    else:
        source = np.asarray(colors, dtype=np.float32)
        if source.ndim != 3 or source.shape[:2] != splat.shape:
            raise ValueError("colors must be (H, W, C) matching splat_mask")
        picked = source[rows, cols][:, :3]
        rgb = np.repeat(picked, n_layers, axis=0).astype(np.float32)

    # A seeded point must be metrically inside the measured interval, or the
    # slab bound this module exists to enforce means nothing.
    assert float(depths.min()) >= float(rim.near_m) - 1e-9
    assert float(depths.max()) <= float(rim.far_m) + 1e-9

    return HoleSeed(
        points_world=pts_world.astype(np.float64),
        colors=rgb,
        near_m=float(rim.near_m),
        far_m=float(rim.far_m),
        scale_m=float(step),
        report={
            "n_rays": int(n_rays),
            "layers": n_layers,
            "pixel_stride": int(pixel_stride),
            "seed": int(seed),
            "slab_depth_m": float(rim.far_m - rim.near_m),
            "source": SPLAT_SOURCE,
        },
    )


def split_visible_occluded(
    error: Any,
    splat_mask: Any,
    mesh_mask: Any,
) -> dict:
    """Report an error image split by who owned the pixel.

    A whole-frame average lets a fill score well by reproducing what it was
    already shown — the same trap ``research/volfill/roundtrip_eval.py`` splits
    VISIBLE from OCCLUDED to avoid. OCCLUDED is the number the experiment turns
    on; VISIBLE is the regression guard that must not move.
    """

    np = _require_numpy()
    err = np.asarray(error, dtype=np.float64)
    splat = np.asarray(splat_mask, dtype=bool)
    mesh = np.asarray(mesh_mask, dtype=bool)
    if err.shape[:2] != splat.shape or splat.shape != mesh.shape:
        raise ValueError("error, splat_mask and mesh_mask must share (H, W)")
    if err.ndim == 3:
        err = err.mean(axis=-1)

    def _stats(mask: Any) -> dict:
        values = err[mask]
        if not values.size:
            return {"n_px": 0, "mean": None, "p95": None}
        return {
            "n_px": int(values.size),
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95.0)),
        }

    return {
        "occluded": _stats(splat),
        "visible": _stats(mesh & ~splat),
        "whole_frame": _stats(np.ones_like(splat, dtype=bool)),
    }
