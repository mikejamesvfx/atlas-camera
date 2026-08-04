"""Hand-authored polygon planes — fit a 3D plane to an artist-clicked outline.

Depth-derived planes (``proxy_geometry._cluster_walls_by_azimuth``,
``plane_extraction.extract_planes_ransac``) emit axis-extent RECTANGLES around
each cluster. That rectangle covers image area the real surface never occupied,
and since the viewport projects by WORLD POSITION (see
``atlas_blockout.js:makeProjectionMaterial`` — geometry UVs are not used for
projection), the overshooting region receives paint from unrelated parts of the
plate. That is the smear artists see on derived walls.

This module takes the outline directly: given a polygon in pixel coordinates,
it fits the plane that outline lies on and returns a mesh clipped to exactly
those corners. Two fitters, tiered:

* ``depth_ransac`` — plane RANSAC over the back-projected depth INSIDE the
  polygon only. Preferred whenever the region's depth is trustworthy.
* ``rectangle_homography`` — depth-free. Treats a quad as a real-world
  rectangle and recovers its orientation from the vanishing points of its two
  edge pairs plus the known intrinsics (single-view metrology). Carries the
  cases where monocular depth is unusable — glass, water, flat sky-lit facades.

Which one ran is always reported; the fallback is never silent.

World points are scaled through ``depth_geometry.fit_ground_and_scale`` exactly
as every derive path does, so a hand-authored plane lands in the same metric
world as derived geometry rather than its own.

Numpy-only, host-agnostic: no ComfyUI, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from atlas_camera.core.depth_geometry import (
    arbitrary_plane_axes,
    back_project_normals,
    fit_ground_and_scale,
)

Point2D = tuple[float, float]

MIN_DEPTH_SAMPLES = 30
RANSAC_ITERS = 200
RANSAC_MAX_SAMPLES = 20000
RANSAC_SEED = 0


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Polygon plane fitting requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class PolygonPlaneFit:
    """One fitted polygon. ``ok=False`` carries the reason and nothing else."""

    ok: bool
    reason: str = ""
    method: str = ""
    vertices: list[float] = field(default_factory=list)   # flat world xyz
    faces: list[int] = field(default_factory=list)        # flat triangle indices
    uvs: list[float] = field(default_factory=list)        # flat planar uv, 0..1
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance_m: float = 0.0                               # plane offset: n . P
    confidence: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Polygon utilities (pure 2D, pixel space)
# ---------------------------------------------------------------------------

def points_from_normalized(points: Sequence[Sequence[float]],
                           width: int, height: int) -> list[Point2D]:
    """Map 0..1 plate-relative click points to pixel coordinates.

    Clicks are stored normalized so re-solving at a different plate resolution
    does not invalidate an artist's outlines.
    """
    out: list[Point2D] = []
    for p in points:
        u, v = float(p[0]), float(p[1])
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError(f"normalized point out of range: ({u}, {v})")
        out.append((u * float(width), v * float(height)))
    return out


def _signed_area(points: Sequence[Point2D]) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _segments_cross(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    def orient(p, q, r):
        val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else -1

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 != o2 and o3 != o4


def _is_simple(points: Sequence[Point2D]) -> bool:
    """True when no two non-adjacent edges cross."""
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            c, d = points[j], points[(j + 1) % n]
            if _segments_cross(a, b, c, d):
                return False
    return True


def _point_in_triangle(p, a, b, c) -> bool:
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def triangulate_polygon(points: Sequence[Point2D]) -> list[tuple[int, int, int]]:
    """Ear-clip a simple polygon into triangles, returning index triples.

    Ear clipping rather than a fan: a fan silently emits inverted triangles on
    a concave outline, and concave outlines (rooflines, L-shaped facades) are a
    primary use case. Indices refer to the INPUT ordering, whatever its winding.
    """
    n = len(points)
    if n < 3:
        raise ValueError("a polygon needs at least 3 points")
    if not _is_simple(points):
        raise ValueError("polygon is self-intersecting")

    order = list(range(n))
    if _signed_area(points) < 0:
        order.reverse()

    faces: list[tuple[int, int, int]] = []
    remaining = list(order)
    guard = 0
    while len(remaining) > 3 and guard < n * n:
        guard += 1
        clipped = False
        for k in range(len(remaining)):
            i_prev = remaining[k - 1]
            i_cur = remaining[k]
            i_next = remaining[(k + 1) % len(remaining)]
            a, b, c = points[i_prev], points[i_cur], points[i_next]
            if _signed_area([a, b, c]) <= 1e-12:
                continue  # reflex or degenerate corner: not an ear
            if any(_point_in_triangle(points[idx], a, b, c)
                   for idx in remaining if idx not in (i_prev, i_cur, i_next)):
                continue
            faces.append((i_prev, i_cur, i_next))
            remaining.pop(k)
            clipped = True
            break
        if not clipped:
            raise ValueError("polygon could not be triangulated")
    faces.append((remaining[0], remaining[1], remaining[2]))
    return faces


def _polygon_mask(np, points: Sequence[Point2D], height: int, width: int):
    """Even-odd rasterization of the polygon onto an (H, W) bool grid."""
    ys, xs = np.mgrid[0:height, 0:width]
    px = xs + 0.5
    py = ys + 0.5
    inside = np.zeros((height, width), dtype=bool)
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        if y0 == y1:
            continue
        straddles = ((y0 > py) != (y1 > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (x1 - x0) * (py - y0) / (y1 - y0) + x0
        inside ^= straddles & (px < x_cross)
    return inside


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _camera_frame(np, view_matrix):
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    return cam_to_world[:3, :3], cam_to_world[:3, 3]


def _pixel_rays(np, points, *, r_cw, fx, fy, cx, cy):
    """Unit world-space ray directions through each pixel."""
    dirs = []
    for x, y in points:
        d_cam = np.array([(x - cx) / fx, -(y - cy) / fy, -1.0])
        d = r_cw @ d_cam
        dirs.append(d / (np.linalg.norm(d) or 1.0))
    return np.asarray(dirs)


def _intersect_rays_with_plane(np, dirs, origin, normal, offset):
    """Ray/plane intersection for every corner; None if any lands behind."""
    denom = dirs @ normal
    if np.any(np.abs(denom) < 1e-9):
        return None
    t = (offset - float(np.dot(normal, origin))) / denom
    if np.any(t <= 1e-6) or not np.all(np.isfinite(t)):
        return None
    return origin[None, :] + t[:, None] * dirs


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------

def _ransac_plane(np, pts):
    """(normal, offset, inlier_fraction) for the dominant plane in `pts`."""
    rng = np.random.default_rng(RANSAC_SEED)
    if len(pts) > RANSAC_MAX_SAMPLES:
        pts = pts[rng.choice(len(pts), RANSAC_MAX_SAMPLES, replace=False)]

    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    tol = max(0.01, 0.01 * extent)

    best_inliers = None
    best_count = 0
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n = n / norm
        inliers = np.abs((pts - a) @ n) < tol
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < 3:
        return None

    # Least-squares refine on the consensus set (SVD of the centred points).
    sel = pts[best_inliers]
    centroid = sel.mean(axis=0)
    _u, _s, vh = np.linalg.svd(sel - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) or 1.0)
    offset = float(np.dot(normal, centroid))
    refined = np.abs((pts @ normal) - offset) < tol
    return normal, offset, float(refined.sum()) / float(len(pts))


def _vanishing_direction(np, p0, p1, p2, p3, *, fx, fy, cx, cy):
    """Camera-space direction of the world lines imaged as p0p1 and p3p2."""
    def homog(p):
        return np.array([p[0], p[1], 1.0])

    l1 = np.cross(homog(p0), homog(p1))
    l2 = np.cross(homog(p3), homog(p2))
    vp = np.cross(l1, l2)
    if abs(vp[2]) > 1e-9 * max(abs(vp[0]), abs(vp[1]), 1.0):
        x, y = vp[0] / vp[2], vp[1] / vp[2]
        d = np.array([(x - cx) / fx, -(y - cy) / fy, -1.0])
    else:
        # Vanishing point at infinity — the edges are parallel in the image, so
        # the world direction is parallel to the image plane.
        d = np.array([vp[0] / fx, -vp[1] / fy, 0.0])
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    return d / norm


def _rectangle_normal(np, points, *, r_cw, fx, fy, cx, cy):
    """World normal of a quad assumed to be a real-world rectangle."""
    p0, p1, p2, p3 = points
    d1 = _vanishing_direction(np, p0, p1, p2, p3, fx=fx, fy=fy, cx=cx, cy=cy)
    d2 = _vanishing_direction(np, p1, p2, p3, p0, fx=fx, fy=fy, cx=cx, cy=cy)
    if d1 is None or d2 is None:
        return None
    n_cam = np.cross(d1, d2)
    norm = float(np.linalg.norm(n_cam))
    if norm < 1e-6:
        return None
    n_world = r_cw @ (n_cam / norm)
    n_world = n_world / (np.linalg.norm(n_world) or 1.0)
    # A true rectangle's two edge directions are orthogonal in 3D; how far off
    # they are is an honest confidence for the assumption this path makes.
    confidence = float(max(0.0, 1.0 - abs(float(np.dot(d1, d2)))))
    return n_world, confidence


def _rectangle_anchor(np, points, *, mask, pts_world, valid, r_cw, cam_pos,
                      fx, fy, cx, cy):
    """A world point the rectangle passes through: measured depth, else ground."""
    usable = mask & valid
    if bool(usable.any()):
        inside = pts_world[usable]
        return inside[len(inside) // 2] if len(inside) == 1 else np.median(inside, axis=0)

    # No depth at all inside the outline: fall back to where the outline's
    # lowest edge meets the analytic Y=0 ground plane.
    lowest = max(points, key=lambda p: p[1])
    second = sorted(points, key=lambda p: p[1])[-2]
    mid = ((lowest[0] + second[0]) * 0.5, (lowest[1] + second[1]) * 0.5)
    d = _pixel_rays(np, [mid], r_cw=r_cw, fx=fx, fy=fy, cx=cx, cy=cy)[0]
    if d[1] >= -1e-6 or cam_pos[1] <= 1e-6:
        return None
    t = -cam_pos[1] / d[1]
    return cam_pos + t * d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_polygon_plane(
    points_px: Sequence[Sequence[float]],
    *,
    depth: Any,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mode: str = "auto",
    min_inlier_fraction: float = 0.35,
) -> PolygonPlaneFit:
    """Fit the plane an artist-clicked outline lies on.

    ``mode`` is ``auto`` (RANSAC, falling back to the rectangle solve),
    ``depth_ransac``, or ``rectangle``. The returned mesh is clipped to exactly
    the clicked corners: each corner is the intersection of its own camera ray
    with the fitted plane, so it reprojects back onto the pixel that was
    clicked.
    """
    np = _require_numpy()
    points = [(float(p[0]), float(p[1])) for p in points_px]

    if len(points) < 3:
        return PolygonPlaneFit(ok=False, reason="too_few_points")
    # Self-intersection is checked first: a symmetric bowtie also has zero
    # signed area, and "self_intersecting" is the diagnosis that helps.
    if not _is_simple(points):
        return PolygonPlaneFit(ok=False, reason="self_intersecting")
    if abs(_signed_area(points)) < 1e-6:
        return PolygonPlaneFit(ok=False, reason="zero_area")

    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be a 2-D (H, W) array, got shape {depth.shape}")
    height, width = depth.shape

    r_cw, cam_pos = _camera_frame(np, view_matrix)
    bp = back_project_normals(depth, view_matrix=view_matrix, fx=fx, fy=fy,
                              cx=cx, cy=cy)
    gf = fit_ground_and_scale(bp, horizon_y=height * 0.45)
    pts_world = gf.pts_world_scaled
    mask = _polygon_mask(np, points, height, width)

    stats: dict[str, Any] = {
        "point_count": len(points),
        "depth_scale_applied": float(gf.scale),
        "polygon_pixels": int(mask.sum()),
    }

    fallback_note = ""
    solution = None

    if mode in ("auto", "depth_ransac"):
        usable = mask & bp.valid_depth
        samples = int(usable.sum())
        stats["depth_samples"] = samples
        if samples >= MIN_DEPTH_SAMPLES:
            fitted = _ransac_plane(np, pts_world[usable])
            if fitted is not None:
                normal, offset, fraction = fitted
                stats["inlier_fraction"] = round(fraction, 4)
                if fraction >= min_inlier_fraction or mode == "depth_ransac":
                    solution = ("depth_ransac", normal, offset, fraction, "")
                else:
                    fallback_note = (f"fallback: inliers {fraction:.2f} "
                                     f"< {min_inlier_fraction:.2f}")
            else:
                fallback_note = "fallback: no consensus plane in region"
        else:
            fallback_note = f"fallback: only {samples} usable depth samples in region"
        if mode == "depth_ransac" and solution is None:
            return PolygonPlaneFit(ok=False, reason=fallback_note or "no_fit",
                                   stats=stats)

    if solution is None and mode in ("auto", "rectangle"):
        if len(points) != 4:
            if mode == "rectangle":
                return PolygonPlaneFit(
                    ok=False, reason="rectangle solve requires_quad (4 points)",
                    stats=stats)
        else:
            rect = _rectangle_normal(np, points, r_cw=r_cw, fx=fx, fy=fy, cx=cx, cy=cy)
            anchor = _rectangle_anchor(
                np, points, mask=mask, pts_world=pts_world, valid=bp.valid_depth,
                r_cw=r_cw, cam_pos=cam_pos, fx=fx, fy=fy, cx=cx, cy=cy)
            if rect is not None and anchor is not None:
                normal, confidence = rect
                solution = ("rectangle_homography", normal,
                            float(np.dot(normal, anchor)), confidence,
                            fallback_note or "rectangle solve")

    if solution is None:
        return PolygonPlaneFit(ok=False, reason="no_fit", stats=stats)

    method, normal, offset, confidence, note = solution

    # Face the camera. Yaw/normal sign is a free convention for a single plane,
    # and a camera-facing normal is what the projection material expects.
    if float(np.dot(normal, cam_pos)) - offset < 0.0:
        normal, offset = -normal, -offset

    dirs = _pixel_rays(np, points, r_cw=r_cw, fx=fx, fy=fy, cx=cx, cy=cy)
    corners = _intersect_rays_with_plane(np, dirs, cam_pos, normal, offset)
    if corners is None:
        return PolygonPlaneFit(ok=False, reason="corner_behind_camera", stats=stats)

    try:
        faces = triangulate_polygon(points)
    except ValueError as exc:
        return PolygonPlaneFit(ok=False, reason=str(exc), stats=stats)

    # Wind faces consistently with the reported normal so exported geometry is
    # front-facing (the viewport itself renders DoubleSide).
    i, j, k = faces[0]
    if float(np.dot(np.cross(corners[j] - corners[i], corners[k] - corners[i]),
                    normal)) < 0.0:
        faces = [(a, c, b) for a, b, c in faces]

    u_ax, v_ax, _n = arbitrary_plane_axes(np, normal)
    local = corners - corners.mean(axis=0)
    uu = local @ u_ax
    vv = local @ v_ax
    span_u = float(uu.max() - uu.min()) or 1.0
    span_v = float(vv.max() - vv.min()) or 1.0
    uvs = np.stack([(uu - uu.min()) / span_u, (vv - vv.min()) / span_v], axis=-1)

    stats["width_m"] = round(span_u, 4)
    stats["height_m"] = round(span_v, 4)

    return PolygonPlaneFit(
        ok=True,
        reason=note,
        method=method,
        vertices=np.round(corners.reshape(-1), 4).tolist(),
        faces=[int(v) for tri in faces for v in tri],
        uvs=np.round(uvs.reshape(-1), 4).tolist(),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        distance_m=float(offset),
        confidence=float(confidence),
        stats=stats,
    )
