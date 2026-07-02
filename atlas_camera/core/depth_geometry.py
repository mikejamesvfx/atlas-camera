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
