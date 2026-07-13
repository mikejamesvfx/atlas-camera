"""Shared depth back-projection / plane-fitting primitives.

Factored out of :mod:`atlas_camera.core.proxy_geometry` (steps 1-3 and 8 of
``derive_projection_proxies``) so every geometry-derivation strategy —
the default vertical-wall fitter (``proxy_geometry.py``), the any-orientation
RANSAC/Hough plane extractor (``plane_extraction.py``), and the Manhattan
room-cuboid fitter (``room_layout.py``) — agrees bit-for-bit on world points,
per-pixel normals, and metric ground scale for a given depth map + camera.

``proxy_geometry.py`` itself is intentionally left untouched (its own copies
of this logic are tested and shipped); this module is purely additive.

Convention (critical, identical to proxy_geometry.py): always use the full
4×4 ``extrinsics.camera_view_matrix`` (row-major, world→cam, column-vector
points, translation in column 3) and its inverse — never the 3×3
``camera_rotation_matrix``. Numpy-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Depth geometry helpers require numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class BackProjection:
    """World-space back-projection of a depth map, with per-pixel normals."""

    pts_world: Any        # (H,W,3) float64
    normals: Any           # (H,W,3) float64, zero where invalid
    valid_normal: Any      # (H,W) bool
    valid_depth: Any       # (H,W) bool
    R_cw: Any              # (3,3) cam->world rotation
    cam_pos: Any            # (3,) world-space camera position
    vv: Any                 # (H,W) float64 pixel-row grid


def back_project_normals(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_edge_rel: float = 0.05,
) -> BackProjection:
    """Back-project a forward-z depth map to world space and compute normals.

    Verbatim port of ``proxy_geometry.derive_projection_proxies`` steps 1-2.
    """
    np = _require_numpy()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape

    vm = np.asarray(view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    R_cw = cam_to_world[:3, :3]
    cam_pos = cam_to_world[:3, 3]

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    x = (uu - cx) / fx * depth
    y = -(vv - cy) / fy * depth
    z = -depth
    pts_cam = np.stack([x, y, z], axis=-1)
    pts_world = pts_cam @ R_cw.T + cam_pos

    valid_depth = np.isfinite(depth) & (depth > 1e-4)

    du = pts_world[:, 2:, :] - pts_world[:, :-2, :]
    dv = pts_world[2:, :, :] - pts_world[:-2, :, :]
    du = du[1:-1, :, :]
    dv = dv[:, 1:-1, :]
    normals_inner = np.cross(du, dv)
    nrm = np.linalg.norm(normals_inner, axis=-1, keepdims=True)
    normals_inner = normals_inner / np.maximum(nrm, 1e-12)

    normals = np.zeros((height, width, 3), dtype=np.float64)
    normals[1:-1, 1:-1] = normals_inner

    ddx = np.abs(depth[:, 2:] - depth[:, :-2])
    ddy = np.abs(depth[2:, :] - depth[:-2, :])
    edge = np.zeros((height, width), dtype=bool)
    edge[:, 1:-1] |= ddx > depth_edge_rel * 2.0 * np.maximum(depth[:, 1:-1], 1e-6)
    edge[1:-1, :] |= ddy > depth_edge_rel * 2.0 * np.maximum(depth[1:-1, :], 1e-6)

    inner = np.zeros((height, width), dtype=bool)
    inner[1:-1, 1:-1] = True
    valid_normal = inner & valid_depth & ~edge

    return BackProjection(
        pts_world=pts_world, normals=normals, valid_normal=valid_normal,
        valid_depth=valid_depth, R_cw=R_cw, cam_pos=cam_pos, vv=vv,
    )


def primary_camera_validity_mask(
    pts_world: Any,
    valid_depth: Any,
    normals: Any,
    valid_normal: Any,
    *,
    primary_view_matrix: Any,
    primary_fx: float,
    primary_fy: float,
    primary_cx: float,
    primary_cy: float,
    primary_width: int,
    primary_height: int,
    angle_threshold_deg: float = 90.0,
    primary_depth_map: Any = None,
    depth_bias_rel: float = 0.05,
) -> Any:
    """Test a field of world points against a SECOND ("primary") camera's
    projection validity — behind-camera, outside-frame, too-grazing, or
    (optionally) hidden behind nearer geometry.

    ``pts_world``/``valid_depth``/``normals``/``valid_normal`` are typically a
    :class:`BackProjection` from a *different* (target/patch) camera's own
    depth map — the primary camera here is only used as the thing being
    projected INTO, not the camera the points were derived from. Mirrors the
    facing-ratio mask in ``atlas_blockout.js``'s projection shader
    (``facing = abs(dot(normal, toCam))``) and this repo's universal "-Z
    forward" view-matrix convention (row-major, world->cam,
    ``cam_to_world = inv(view_matrix)``).

    ``primary_depth_map`` (2D array, default ``None``) enables the true
    MPTK-style depth-shadow test: shadow mapping with the primary camera as
    the light, and its own monocular depth estimate as the shadow map — no
    rasterizer or render pass needed, since a depth map from the primary's
    viewpoint IS a depth buffer from the primary's viewpoint. Each in-front,
    in-frame point's primary-camera depth is compared against the map sampled
    at its projected pixel (nearest-neighbour; coordinates rescaled if the
    map's resolution differs from ``primary_width``/``primary_height``):
    farther than the stored depth by more than ``depth_bias_rel`` (relative
    bias against depth-precision false positives) means the point was HIDDEN
    behind nearer geometry from the primary — invalid, a patch should fill
    it. Points sampling invalid map depth (NaN/<=0, e.g. sky) also count as
    invalid (the primary has no data there). CRITICAL: the map's depth values
    must be in the SAME metric world scale as ``pts_world`` — ground-pin both
    sides with ``relief_mesh.estimate_ground_scale`` before calling (see
    ``AtlasOcclusionMask``'s depth_shadow mode).

    Returns an ``(H, W)`` bool array — ``True`` where the primary camera's
    projection is INVALID at that point (should be filled by another source).
    """
    np = _require_numpy()
    vm = np.asarray(primary_view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    primary_cam_pos = cam_to_world[:3, 3]

    pts_world = np.asarray(pts_world, dtype=np.float64)
    R = vm[:3, :3]
    t = vm[:3, 3]
    cam_pts = pts_world @ R.T + t  # world -> primary camera space

    cam_z = cam_pts[..., 2]
    behind = cam_z >= 0.0  # "-Z forward" convention

    depth = np.where(behind, np.nan, -cam_z)
    px = primary_cx + primary_fx * cam_pts[..., 0] / depth
    py = primary_cy - primary_fy * cam_pts[..., 1] / depth
    out_of_frame = ~np.isfinite(px) | ~np.isfinite(py) | \
        (px < 0) | (px >= primary_width) | (py < 0) | (py >= primary_height)

    to_primary = primary_cam_pos - pts_world
    to_primary = to_primary / np.maximum(
        np.linalg.norm(to_primary, axis=-1, keepdims=True), 1e-12
    )
    facing = np.abs(np.sum(np.asarray(normals, dtype=np.float64) * to_primary, axis=-1))
    # facing is always >= 0, so an explicit -1.0 at the 90-degree ceiling
    # guarantees "never facing-excludes" exactly, immune to cos(90 deg) not
    # being exactly 0 in floating point.
    threshold_cos = -1.0 if angle_threshold_deg >= 90.0 else math.cos(math.radians(angle_threshold_deg))
    grazing = facing < threshold_cos

    shadowed = np.zeros(behind.shape, dtype=bool)
    if primary_depth_map is not None:
        dm = np.asarray(primary_depth_map, dtype=np.float64)
        can_sample = ~behind & ~out_of_frame
        # px/py are NaN for behind-camera points — substitute 0 before the
        # int cast (those points are already excluded via can_sample).
        sx = np.where(can_sample, px, 0.0) * (dm.shape[1] / float(primary_width))
        sy = np.where(can_sample, py, 0.0) * (dm.shape[0] / float(primary_height))
        sx = np.clip(np.round(sx), 0, dm.shape[1] - 1).astype(np.int64)
        sy = np.clip(np.round(sy), 0, dm.shape[0] - 1).astype(np.int64)
        sampled = dm[sy, sx]
        sample_invalid = ~np.isfinite(sampled) | (sampled <= 1e-4)
        point_depth = -cam_z
        shadowed = can_sample & (
            sample_invalid | (point_depth > sampled * (1.0 + depth_bias_rel)))

    return behind | out_of_frame | grazing | shadowed \
        | ~np.asarray(valid_depth, dtype=bool) | ~np.asarray(valid_normal, dtype=bool)


@dataclass(slots=True)
class GroundFit:
    """Result of fitting the ground plane and reconciling metric scale."""

    scale: float
    y0: float | None
    ground_inlier: Any      # (H,W) bool, in the RESCALED world
    pts_world_scaled: Any   # (H,W,3) float64, points rescaled about the camera
    tol: float
    inliers: int = 0


def fit_ground_and_scale(
    bp: BackProjection,
    *,
    horizon_y: float,
    ground_normal_min: float = 0.90,
) -> GroundFit:
    """Fit the ground plane and rescale the world about the camera so the
    fitted ground lands exactly on Y=0 (pins depth-map scale to the solve's
    adopted metric camera height). Verbatim port of step 3.

    Returns ``scale=1.0`` and an empty inlier mask when ground support is
    insufficient (<300 candidates), matching the existing graceful fallback.
    """
    np = _require_numpy()
    n_y = bp.normals[..., 1]
    world_y = bp.pts_world[..., 1]
    below = bp.vv > horizon_y
    ground_cand = bp.valid_normal & below & (np.abs(n_y) > ground_normal_min)

    scale = 1.0
    pts_world = bp.pts_world
    ground_inlier = np.zeros(bp.valid_normal.shape, dtype=bool)
    y0 = None
    tol = 0.15

    if int(ground_cand.sum()) >= 300:
        ys = world_y[ground_cand]
        lo, hi = np.percentile(ys, [1, 99])
        span = float(hi - lo)
        if span < 1e-3:
            y0 = float(np.median(ys))
        else:
            hist, edges = np.histogram(ys, bins=48, range=(lo, hi))
            peak = int(np.argmax(hist))
            y0 = 0.5 * (edges[peak] + edges[peak + 1])
        tol = max(0.15, 0.03 * max(span, 1e-3))
        refine = np.abs(ys - y0) < tol
        if int(refine.sum()) >= 50:
            y0 = float(np.median(ys[refine]))
        denom = bp.cam_pos[1] - y0
        if bp.cam_pos[1] > 1e-6 and denom > 1e-6:
            scale = float(bp.cam_pos[1] / denom)
        pts_world = bp.cam_pos + scale * (bp.pts_world - bp.cam_pos)
        world_y = pts_world[..., 1]
        ground_inlier = ground_cand & (np.abs(world_y) < tol * scale)

    return GroundFit(
        scale=scale, y0=y0, ground_inlier=ground_inlier,
        pts_world_scaled=pts_world, tol=tol, inliers=int(ground_inlier.sum()),
    )


def detect_sky_mask(
    depth: Any,
    *,
    horizon_y: float,
    far_clip_percentile: float = 97.0,
    roughness_window: int = 5,
    roughness_factor: float = 8.0,
    disarm_on_incoherent: bool = True,
    disarm_min_coverage: float = 0.08,
    disarm_top_anchored: float = 0.35,
    disarm_runs_per_col: float = 4.0,
) -> Any:
    """Heuristic sky mask: pixels above the horizon whose depth is unreliable.

    Monocular depth (Depth Anything and similar) has no strong cue on
    featureless sky/cloud regions and often hallucinates noisy,
    spatially-incoherent depth there instead of a smooth "far" value.
    Triangulating that noise produces jagged, distorted geometry (see
    ``relief_mesh.build_relief_mesh``) and can even get mistaken for real
    wall height by the primitive fitters. A pixel qualifies as sky when it is
    **above the solved horizon line** AND *either*:

    - its depth is at/beyond ``far_clip_percentile`` of all depth in the
      image (plausibly "very far"), OR
    - its local **roughness** — mean squared discrete Laplacian
      (``depth[i,j] - average of its 4 neighbours``) over a
      ``roughness_window`` x ``roughness_window`` neighborhood
      (``roughness_window`` must be odd) — exceeds ``roughness_factor`` x the
      median roughness found *below* the horizon.

    Roughness, not raw variance, is deliberate: a real sloped surface (a roof,
    a ramp) has a *constant* local gradient, so its Laplacian is ~0 even
    though its raw variance in a window can be large — plain variance would
    misclassify real sloped architecture as sky. Genuine per-pixel model
    noise has no such coherent gradient, so its Laplacian stays large. This
    is self-calibrated per image (relative to the below-horizon baseline)
    rather than a fixed absolute threshold, so it also catches noisy sky
    before the far-clip percentile alone would.

    Returns a boolean array (``True`` = sky), same shape as ``depth``.
    Numpy-only — no training data or new dependency, reuses the horizon line
    the camera solve already produced.

    Self-disarms on interiors: if the candidate mask has the fragmented,
    non-top-anchored shape signature of a false positive (the roughness term
    firing on a detailed ceiling / far wall rather than real sky), it returns
    an empty mask instead of shredding real geometry. Controlled by
    ``disarm_on_incoherent`` and its thresholds; pass ``disarm_on_incoherent=
    False`` for the raw heuristic. See the inline comment for the measured
    interior-vs-sky separation.
    """
    np = _require_numpy()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-4)

    vv = np.arange(height, dtype=np.float64)[:, None] * np.ones((1, width))
    above = vv < horizon_y

    if not valid.any() or not above.any():
        return np.zeros((height, width), dtype=bool)

    far_thresh = float(np.percentile(depth[valid], far_clip_percentile))
    far = depth >= far_thresh

    lap = np.zeros_like(depth)
    lap[1:-1, 1:-1] = depth[1:-1, 1:-1] - 0.25 * (
        depth[:-2, 1:-1] + depth[2:, 1:-1] + depth[1:-1, :-2] + depth[1:-1, 2:]
    )
    sq = lap ** 2

    pad = roughness_window // 2
    padded = np.pad(sq, pad, mode="edge")
    # Box-filter mean via an integral image — O(HW) with no window-sized
    # temporary. sliding_window_view here materialized W*W scalars per pixel
    # through the mean reduction (~25x the map at the default window), the
    # peak-memory hotspot on 4K+ plates. Same edge-padded semantics; only
    # float summation order differs, and the consumer is an 8x-median
    # threshold, insensitive to that.
    ii = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    w = roughness_window
    roughness = (ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]) / float(w * w)

    below = ~above & valid
    baseline_roughness = float(np.median(roughness[below])) if below.any() else 0.0
    noisy = roughness > max(roughness_factor * baseline_roughness, 1e-9)

    sky = above & valid & (far | noisy)

    # Self-disarm on interiors (mirrors AtlasScopeMask's self-disarming
    # fallbacks). The roughness term catches genuine noisy sky, but on an
    # INTERIOR it also fires on detailed ceilings / far greebled walls that are
    # above the (arbitrary, sky-free) horizon — punching large scattered holes
    # in real geometry (measured live: a sci-fi hangar lost 39% of its back
    # wall to this, halved by turning the heuristic off). When the candidate
    # mask has the fragmented, un-anchored shape of a false positive we exclude
    # nothing rather than shredding the mesh. Wire a real sky segmentation
    # (AtlasSkyDomeLayer / an exclude_mask) for genuine indoor-with-window cases.
    if disarm_on_incoherent and _sky_mask_incoherent(
            np, sky, min_coverage=disarm_min_coverage,
            top_anchored_max=disarm_top_anchored,
            runs_per_col_min=disarm_runs_per_col):
        return np.zeros((height, width), dtype=bool)

    return sky


def _sky_mask_incoherent(np, sky, *, min_coverage, top_anchored_max, runs_per_col_min):
    """True when a candidate sky mask has the interior-misfire signature:
    meaningful coverage, yet scattered and NOT anchored to the top of frame.

    Real outdoor sky is one region anchored to the top and vertically
    contiguous; an interior false positive (roughness firing on a detailed
    ceiling / far wall) is scattered fragments with almost nothing top-anchored.
    Two cheap numpy-only shape signals separate them with a wide margin
    (measured on a real hangar: top_anchored 0.02 / 22 runs-per-column vs real
    sky 1.0 / 1.0). Requiring BOTH signals keeps the disarm conservative — a
    coherent-but-large sky (a landscape, a top-occluded sky) still has ~1
    run/column and stays flagged.
    """
    total = float(sky.sum())
    if total == 0.0 or total / sky.size < min_coverage:
        return False
    top_anchored = float(np.cumprod(sky, axis=0).sum()) / total
    runs = float((sky[1:] & ~sky[:-1]).sum() + int(sky[0].sum()))
    cols_with_sky = int(sky.any(axis=0).sum())
    runs_per_col = runs / max(cols_with_sky, 1)
    return top_anchored < top_anchored_max and runs_per_col > runs_per_col_min


def plane_transform(u: Any, v: Any, n: Any, c: Any) -> tuple:
    """Row-major 4×4 with columns = local axes (u, v, n) and translation c.

    Locals are the THREE.PlaneGeometry frame: local X=u, Y=v, Z=n (normal).
    Port of ``proxy_geometry._plane_transform``.
    """
    return (
        (float(u[0]), float(v[0]), float(n[0]), float(c[0])),
        (float(u[1]), float(v[1]), float(n[1]), float(c[1])),
        (float(u[2]), float(v[2]), float(n[2]), float(c[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def arbitrary_plane_axes(np: Any, n: Any) -> tuple[Any, Any, Any]:
    """Right-handed (u, v, n) in-plane basis for a plane of ANY orientation.

    Generalizes ``proxy_geometry._wall_axes`` (which only handles horizontal
    normals via ``u = up × n``) to arbitrary normals — needed for sloped
    surfaces (roofs, ramps) where ``n`` may be nearly parallel to world-up and
    ``cross(up, n)`` degenerates. Falls back to a second reference axis in
    that case. For a horizontal normal this reduces to the same result as
    ``_wall_axes``.
    """
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n) or 1.0)
    up = np.array([0.0, 1.0, 0.0])
    ref = up if abs(float(np.dot(n, up))) < 0.98 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-6:
        ref = np.array([0.0, 0.0, 1.0])
        u = np.cross(ref, n)
        u_norm = np.linalg.norm(u)
    u = u / (u_norm or 1.0)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) or 1.0)
    return u, v, n


def build_backdrop_primitive(
    *,
    bp: BackProjection,
    scaled_depth: Any,
    valid_depth: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    scale: float,
    backdrop_depth_percentile: float = 96.0,
    backdrop_margin: float = 1.35,
) -> Any:
    """Always-emitted far cyclorama, exactly sized to the recovered frustum.

    Port of ``proxy_geometry.derive_projection_proxies`` step 8: intersects
    the four frustum-corner rays with the backdrop plane and encloses the
    hits (+margin) — covers any pitch/roll, unlike a flat horizontal-frustum
    margin. Returns an ``AtlasProxyPrimitive``.
    """
    np = _require_numpy()
    from atlas_camera.core.schema import AtlasProxyPrimitive

    R_cw, cam_pos = bp.R_cw, bp.cam_pos
    backdrop_d_raw = float(np.percentile(
        scaled_depth[valid_depth], backdrop_depth_percentile
    )) if valid_depth.any() else 60.0

    D = 1.02 * backdrop_d_raw
    fwd = R_cw @ np.array([0.0, 0.0, -1.0])
    fwd_h = np.array([fwd[0], 0.0, fwd[2]])
    fl = np.linalg.norm(fwd_h)
    fwd_h = fwd_h / fl if fl > 1e-6 else np.array([0.0, 0.0, -1.0])
    n_b = -fwd_h
    u_b, v_b, _ = arbitrary_plane_axes(np, n_b)
    c0 = cam_pos + fwd_h * D

    us: list[float] = []
    ys_: list[float] = []
    for (u_px, v_px) in ((0.0, 0.0), (float(width), 0.0),
                         (0.0, float(height)), (float(width), float(height))):
        d_cam = np.array([(u_px - cx) / fx, -(v_px - cy) / fy, -1.0])
        d_w = R_cw @ d_cam
        denom = float(np.dot(n_b, d_w))
        if abs(denom) < 1e-6:
            continue
        t = float(np.dot(n_b, c0 - cam_pos)) / denom
        if t <= 0:
            continue
        p = cam_pos + min(t, 4.0 * D) * d_w
        us.append(float(np.dot(p - c0, u_b)))
        ys_.append(float(p[1]))
    if not us:
        us, ys_ = [-D, D], [0.0, D]
    mfrac = backdrop_margin - 1.0
    u_lo, u_hi = min(us), max(us)
    y_lo, y_hi = min(ys_), max(ys_)
    u_pad = 0.5 * (u_hi - u_lo) * mfrac + 1.0
    y_pad = 0.5 * (y_hi - y_lo) * mfrac + 1.0
    u_lo -= u_pad
    u_hi += u_pad
    y_lo = min(y_lo - y_pad, -1.0)
    y_hi += y_pad
    bw = u_hi - u_lo
    bh = y_hi - y_lo
    c_b = c0 + u_b * (0.5 * (u_lo + u_hi))
    c_b[1] = 0.5 * (y_lo + y_hi)

    from atlas_camera.core.proxy_geometry import PROXY_ROLE

    return AtlasProxyPrimitive(
        name="projection_backdrop",
        primitive_type="plane",
        transform_matrix=plane_transform(u_b, v_b, n_b, c_b),
        dimensions=(float(bw), float(bh), 0.0),
        material="atlas_projection_proxy",
        metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                  "distance_m": float(D), "depth_scale_applied": scale},
    )
