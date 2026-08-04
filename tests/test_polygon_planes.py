"""Tests for hand-authored polygon plane fitting (core.polygon_planes).

Self-contained analytic depth (same convention as test_derive_geometry_nodes.py:
level camera at (0, h, 0), identity rotation, forward-z depth) so these run with
numpy alone — no [neural] extra, no model download, no ComfyUI.
"""

import numpy as np
import pytest

from atlas_camera.core.polygon_planes import (
    fit_polygon_plane,
    points_from_normalized,
    triangulate_polygon,
)

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 200.0
CAM_HEIGHT = 1.6


def _view_matrix(h=CAM_HEIGHT):
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


CAMERA = dict(view_matrix=_view_matrix(), fx=FX, fy=FY, cx=CX, cy=CY)


def _ray_dirs():
    """Per-pixel (a, b) so a world point at forward depth d is (a*d, h + b*d, -d)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return (uu - CX) / FX, -(vv - CY) / FY


def _plane_depth(normal, p0, h=CAM_HEIGHT):
    """Forward-z depth of an infinite plane, +inf where the plane is behind/parallel."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a, b = _ray_dirs()
    denom = n[0] * a + n[1] * b - n[2]
    rhs = float(np.dot(n, np.asarray(p0, dtype=float))) - n[1] * h
    with np.errstate(divide="ignore", invalid="ignore"):
        d = rhs / denom
    d[~np.isfinite(d)] = np.inf
    d[d <= 1e-6] = np.inf
    return d


def _ground_depth(h=CAM_HEIGHT):
    return _plane_depth((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), h)


def _scene(normal, p0, h=CAM_HEIGHT):
    """Ground plane + one tilted plane, nearest surface wins. Returns (depth, mask)."""
    ground = _ground_depth(h)
    plane = _plane_depth(normal, p0, h)
    depth = np.minimum(ground, plane)
    visible = np.isfinite(plane) & (plane <= ground)
    depth = np.where(np.isfinite(depth), depth, SKY)
    return depth.astype(np.float64), visible


def _inset_polygon(mask, inset=12):
    """Axis-aligned quad well inside `mask` — avoids the depth-discontinuity seam."""
    rows, cols = np.where(mask)
    y0, y1 = rows.min() + inset, rows.max() - inset
    x0, x1 = cols.min() + inset, cols.max() - inset
    assert y1 > y0 and x1 > x0, "mask too small to inset"
    return [(float(x0), float(y0)), (float(x1), float(y0)),
            (float(x1), float(y1)), (float(x0), float(y1))]


def _angle_deg(a, b):
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


# --- depth RANSAC path ------------------------------------------------------

def test_ransac_recovers_a_known_tilted_plane():
    normal = np.array([0.0, 0.45, 0.89])
    normal /= np.linalg.norm(normal)
    p0 = np.array([0.0, 2.0, -7.0])
    depth, visible = _scene(normal, p0)

    fit = fit_polygon_plane(_inset_polygon(visible), depth=depth, **CAMERA)

    assert fit.ok, fit.reason
    assert fit.method == "depth_ransac"
    assert _angle_deg(fit.normal, normal) < 1.0
    expected_offset = abs(float(np.dot(normal, p0)))
    assert abs(abs(fit.distance_m) - expected_offset) < 0.01 * expected_offset


def test_fitted_corners_reproject_onto_the_clicked_pixels():
    """The whole point of the feature: the mesh covers exactly what was clicked."""
    normal = np.array([0.0, 0.45, 0.89])
    p0 = np.array([0.0, 2.0, -7.0])
    depth, visible = _scene(normal, p0)
    polygon = _inset_polygon(visible)

    fit = fit_polygon_plane(polygon, depth=depth, **CAMERA)

    verts = np.asarray(fit.vertices, dtype=float).reshape(-1, 3)
    cam = np.array([0.0, CAM_HEIGHT, 0.0])
    for (px, py), world in zip(polygon, verts):
        local = world - cam
        u = CX + local[0] / (-local[2]) * FX
        v = CY - local[1] / (-local[2]) * FY
        assert abs(u - px) < 0.5
        assert abs(v - py) < 0.5


def test_vertical_wall_recovers_a_horizontal_normal():
    normal = np.array([0.0, 0.0, 1.0])
    p0 = np.array([0.0, 0.0, -8.0])
    depth, visible = _scene(normal, p0)

    fit = fit_polygon_plane(_inset_polygon(visible), depth=depth, **CAMERA)

    assert fit.ok, fit.reason
    assert _angle_deg(fit.normal, normal) < 1.0
    assert abs(abs(fit.distance_m) - 8.0) < 0.08


def test_normal_points_back_toward_the_camera():
    normal = np.array([0.0, 0.0, 1.0])
    depth, visible = _scene(normal, (0.0, 0.0, -8.0))

    fit = fit_polygon_plane(_inset_polygon(visible), depth=depth, **CAMERA)

    cam = np.array([0.0, CAM_HEIGHT, 0.0])
    centre = np.asarray(fit.vertices, dtype=float).reshape(-1, 3).mean(axis=0)
    assert float(np.dot(np.asarray(fit.normal), cam - centre)) > 0.0


# --- rectangle-homography fallback -----------------------------------------

def test_rectangle_path_recovers_orientation_without_usable_depth():
    normal = np.array([0.0, 0.0, 1.0])
    depth, visible = _scene(normal, (0.0, 0.0, -8.0))
    polygon = _inset_polygon(visible)
    # Depth inside the polygon is destroyed; only the ground remains usable, so
    # RANSAC has nothing to fit and the rectangle solve has to carry it.
    ruined = depth.copy()
    ys = slice(int(min(p[1] for p in polygon)) - 4, int(max(p[1] for p in polygon)) + 5)
    xs = slice(int(min(p[0] for p in polygon)) - 4, int(max(p[0] for p in polygon)) + 5)
    ruined[ys, xs] = np.nan

    fit = fit_polygon_plane(polygon, depth=ruined, **CAMERA, mode="rectangle")

    assert fit.ok, fit.reason
    assert fit.method == "rectangle_homography"
    assert _angle_deg(fit.normal, normal) < 1.0


def test_auto_falls_back_to_rectangle_and_says_why():
    normal = np.array([0.0, 0.0, 1.0])
    depth, visible = _scene(normal, (0.0, 0.0, -8.0))
    polygon = _inset_polygon(visible)
    ruined = depth.copy()
    ruined[:] = np.where(np.isfinite(ruined), ruined, ruined)
    ys = slice(int(min(p[1] for p in polygon)) - 4, int(max(p[1] for p in polygon)) + 5)
    xs = slice(int(min(p[0] for p in polygon)) - 4, int(max(p[0] for p in polygon)) + 5)
    ruined[ys, xs] = np.nan

    fit = fit_polygon_plane(polygon, depth=ruined, **CAMERA, mode="auto")

    assert fit.ok, fit.reason
    assert fit.method == "rectangle_homography"
    assert fit.reason, "a fallback must explain itself"


def test_rectangle_path_refuses_non_quads():
    depth, visible = _scene((0.0, 0.0, 1.0), (0.0, 0.0, -8.0))
    quad = _inset_polygon(visible)
    pentagon = quad + [(quad[0][0] - 10.0, (quad[0][1] + quad[3][1]) / 2.0)]

    fit = fit_polygon_plane(pentagon, depth=depth * np.nan, **CAMERA, mode="rectangle")

    assert not fit.ok
    assert "quad" in fit.reason


# --- validation -------------------------------------------------------------

def test_self_intersecting_polygon_is_rejected():
    bowtie = [(10.0, 10.0), (100.0, 100.0), (100.0, 10.0), (10.0, 100.0)]
    depth, _ = _scene((0.0, 0.0, 1.0), (0.0, 0.0, -8.0))

    fit = fit_polygon_plane(bowtie, depth=depth, **CAMERA)

    assert not fit.ok
    assert fit.reason == "self_intersecting"


def test_degenerate_polygons_are_rejected():
    depth, _ = _scene((0.0, 0.0, 1.0), (0.0, 0.0, -8.0))

    too_few = fit_polygon_plane([(1.0, 1.0), (2.0, 2.0)], depth=depth, **CAMERA)
    assert not too_few.ok and too_few.reason == "too_few_points"

    zero_area = fit_polygon_plane(
        [(10.0, 10.0), (60.0, 10.0), (110.0, 10.0)], depth=depth, **CAMERA)
    assert not zero_area.ok and zero_area.reason == "zero_area"


def test_no_usable_depth_and_no_rectangle_reports_no_fit():
    depth = np.full((H, W), np.nan)
    triangle = [(40.0, 40.0), (120.0, 50.0), (80.0, 130.0)]

    fit = fit_polygon_plane(triangle, depth=depth, **CAMERA, mode="auto")

    assert not fit.ok
    assert fit.reason == "no_fit"


# --- triangulation ----------------------------------------------------------

def _tri_area(a, b, c):
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def _polygon_area(points):
    total = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def test_ear_clipping_handles_a_concave_outline():
    """A naive fan inverts triangles on a concave outline — rooflines are concave."""
    ell = [(0.0, 0.0), (60.0, 0.0), (60.0, 20.0), (20.0, 20.0), (20.0, 60.0), (0.0, 60.0)]

    faces = triangulate_polygon(ell)

    assert len(faces) == len(ell) - 2
    areas = [_tri_area(ell[i], ell[j], ell[k]) for i, j, k in faces]
    assert all(a > 0 for a in areas), "every triangle must keep the same winding"
    assert abs(sum(areas) - abs(_polygon_area(ell))) < 1e-6


def test_triangulation_is_winding_independent():
    ccw = [(0.0, 0.0), (60.0, 0.0), (60.0, 20.0), (20.0, 20.0), (20.0, 60.0), (0.0, 60.0)]
    cw = list(reversed(ccw))

    assert len(triangulate_polygon(cw)) == len(cw) - 2


def test_self_intersecting_outline_refuses_to_triangulate():
    with pytest.raises(ValueError):
        triangulate_polygon([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)])


# --- normalized point mapping ----------------------------------------------

def test_normalized_points_scale_to_any_plate_resolution():
    normalized = [(0.25, 0.5), (0.75, 0.5), (0.75, 0.9)]

    small = points_from_normalized(normalized, 200, 100)
    large = points_from_normalized(normalized, 2000, 1000)

    assert small == [(50.0, 50.0), (150.0, 50.0), (150.0, 90.0)]
    assert large == [(500.0, 500.0), (1500.0, 500.0), (1500.0, 900.0)]


def test_normalized_points_reject_out_of_range_values():
    with pytest.raises(ValueError):
        points_from_normalized([(0.5, 1.4)], 100, 100)
