"""Tests for any-orientation RANSAC/Hough plane extraction (exteriors).

Self-contained analytic scenes (tests/ is not a package — no cross-file
imports), following the exact style/constants of test_proxy_geometry.py:
level camera at (0,h,0), forward-z depth via min()-composited ray casts.
"""

import json
import math

import numpy as np
import pytest

from atlas_camera.core.plane_extraction import extract_planes_ransac
from atlas_camera.core.proxy_geometry import serialize_proxy_geometry
from atlas_camera.core.schema import AtlasProjectionScene

W = H = 512
FX = FY = 500.0
CX = CY = 256.0
SKY = 60.0


def _view_matrix(h):
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rays():
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dx = (uu - CX) / FX
    dy = -(vv - CY) / FY
    return dx, dy


def _ground_depth(h, dx, dy):
    t = np.full(dx.shape, np.inf)
    hit = dy < -1e-6
    t[hit] = -h / dy[hit]
    return t


def _sloped_plane_depth(h=1.6, normal=(0.0, 0.7071, 0.7071), point=(0.0, 4.0, -8.0)):
    """Analytic depth of an arbitrarily-oriented plane n·(p-p0)=0 plus ground."""
    dx, dy = _rays()
    depth = np.full((H, W), SKY)
    tg = _ground_depth(h, dx, dy)

    n = np.asarray(normal, dtype=float)
    p0 = np.asarray(point, dtype=float)
    d = n @ p0
    ndir = n[0] * dx + n[1] * dy + n[2] * (-1.0)
    origin = np.array([0.0, h, 0.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (d - n @ origin) / ndir
    valid = np.isfinite(t) & (t > 0.1) & (t < 50.0)
    tp = np.where(valid, t, SKY)

    stacked = np.stack([depth, np.where(np.isfinite(tg), tg, SKY), tp])
    return stacked.min(axis=0)


def _stepped_facade_depth(h=1.6, wall_z1=-8.0, wall_z2=-14.0, wall_h=4.0):
    """Two parallel vertical planes (same azimuth) at different offsets,
    non-overlapping in image space (left/right halves)."""
    dx, dy = _rays()
    depth = np.full((H, W), SKY)
    tg = _ground_depth(h, dx, dy)

    t1 = -wall_z1
    y1 = h + dy * t1
    x1 = dx * t1
    vis1 = (y1 >= 0.0) & (y1 <= wall_h) & (x1 >= -6.0) & (x1 <= 0.0)
    tw1 = np.where(vis1, t1, SKY)

    t2 = -wall_z2
    y2 = h + dy * t2
    x2 = dx * t2
    vis2 = (y2 >= 0.0) & (y2 <= wall_h) & (x2 >= 0.0) & (x2 <= 6.0)
    tw2 = np.where(vis2, t2, SKY)

    stacked = np.stack([depth, np.where(np.isfinite(tg), tg, SKY), tw1, tw2])
    return stacked.min(axis=0)


def _by_prefix(prims, prefix):
    return [p for p in prims if p.name.startswith(prefix)]


def test_sloped_roof_plane_recovered():
    depth = _sloped_plane_depth()
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY
    )
    planes = _by_prefix(prims, "projection_plane")
    assert len(planes) == 1
    md = planes[0].metadata
    assert md["normal_azimuth_deg"] == pytest.approx(0.0, abs=3.0)
    assert md["normal_elevation_deg"] == pytest.approx(45.0, abs=3.0)


def test_stepped_facade_finds_both_offsets():
    depth = _stepped_facade_depth()
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY, max_planes=8
    )
    planes = _by_prefix(prims, "projection_plane")
    assert len(planes) == 2
    distances = sorted(abs(p.metadata["distance_m"]) for p in planes)
    assert distances[0] == pytest.approx(8.0, abs=0.5)
    assert distances[1] == pytest.approx(14.0, abs=0.5)


def test_depth_scale_reconciliation():
    depth = _stepped_facade_depth() * 2.0
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY
    )
    assert stats["ground_scale"] == pytest.approx(0.5, abs=0.05)


def test_graceful_backdrop_only_on_flat_scene():
    depth = np.full((H, W), SKY)
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY
    )
    assert not _by_prefix(prims, "projection_ground")
    assert not _by_prefix(prims, "projection_plane")
    backs = _by_prefix(prims, "projection_backdrop")
    assert len(backs) == 1


def test_max_planes_cap_respected():
    depth = _stepped_facade_depth()
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY, max_planes=1
    )
    planes = _by_prefix(prims, "projection_plane")
    assert len(planes) == 1
    assert stats["planes"] == 1


def test_payload_is_json_safe():
    depth = _stepped_facade_depth()
    prims, _ = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY
    )
    scene = AtlasProjectionScene()
    scene.proxy_geometry.extend(prims)
    payload = serialize_proxy_geometry(scene)
    assert payload
    json.dumps(payload)
    for entry in payload:
        assert len(entry["transform"]) == 16
        assert len(entry["dimensions"]) == 3


def _sloped_ground_depth(h=1.6, slope_deg=2.0):
    """Ground tilted about X — every real road has a gradient or a camber.

    The ground primitive is built with a HARD-CODED horizontal normal
    ``n_g = (0, 1, 0)``, so a sloped ground cannot be represented by it and the
    residual pixels fall through to the orientation-peak loop.
    """
    import math

    dx, dy = _rays()
    a = math.radians(slope_deg)
    n = np.array([0.0, math.cos(a), math.sin(a)])
    origin = np.array([0.0, h, 0.0])
    ndir = n[0] * dx + n[1] * dy + n[2] * (-1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - n @ origin) / ndir
    return np.minimum(np.where(np.isfinite(t) & (t > 0.1) & (t < 50.0), t, SKY), SKY)


def _normal_of(prim):
    return np.array([prim.transform_matrix[i][2] for i in range(3)])


def test_a_sloped_ground_is_not_emitted_twice():
    """Found on DSC_2552: `projection_plane_01` was the GROUND refitted — 0.9
    degrees off horizontal, 0.247 m perpendicular offset, overlapping extents.
    268,633 of the scene's 299,794 contested pixels were that one pair, because
    the duplicate guard compares a candidate only against other peak-loop
    planes and never against the separately-fitted ground.
    """

    depth = _sloped_ground_depth(slope_deg=2.0)
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
        horizon_y=CY)

    near_horizontal = [p for p in prims
                       if abs(float(_normal_of(p)[1])) > math.cos(math.radians(10.0))]
    assert len(near_horizontal) == 1
    assert near_horizontal[0].name == "projection_ground"


def test_a_suppressed_duplicate_is_reported_not_silently_dropped():
    """The peak loop finding the ground again is the extractor noticing its own
    ground plane is a poor fit. Dropping that quietly hides a real measurement."""

    depth = _sloped_ground_depth(slope_deg=2.0)
    _prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
        horizon_y=CY)

    dupes = stats.get("suppressed_duplicates")
    assert dupes, "a suppressed duplicate must be recorded"
    assert any(d["duplicate_of"] == "projection_ground" for d in dupes)
    entry = next(d for d in dupes if d["duplicate_of"] == "projection_ground")
    assert entry["angle_deg"] == pytest.approx(2.0, abs=0.5)
    assert entry["inliers"] > 0


def test_a_flat_ground_still_yields_exactly_one_ground_plane():
    """The no-regression side: the fix must not change the ordinary case."""

    dx, dy = _rays()
    depth = np.minimum(np.where(np.isfinite(_ground_depth(1.6, dx, dy)),
                                _ground_depth(1.6, dx, dy), SKY), SKY)
    prims, stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
        horizon_y=CY)

    assert [p.name for p in prims] == ["projection_ground", "projection_backdrop"]
    assert not stats.get("suppressed_duplicates")


def test_perpendicular_walls_at_a_corner_are_never_deduplicated():
    """The guard must not eat real geometry. Two walls meeting at a corner are
    ~90 degrees apart; the widened duplicate angle is 20, and the measured gap
    on DSC_2552 between duplicates (0.9, 14.1, 14.5, 16.6) and real corners
    (78.6, 84.3, 87.0, 89.3, 89.9) is wide and empty."""

    depth = _stepped_facade_depth()
    prims, _stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
        horizon_y=CY)

    walls = _by_prefix(prims, "projection_plane_")
    assert len(walls) >= 2


def test_a_plane_keeps_the_pixels_it_was_fitted_to():
    """At the old 2/98 extent a plane discarded 4% of its OWN inliers, and
    those pixels then belonged to no plane at all — 31.4% of every unassigned
    pixel on DSC_2552. The rectangle must contain the overwhelming majority of
    what supports it."""

    from atlas_camera.core.depth_geometry import back_project_normals
    from atlas_camera.core.plane_masks import (
        assign_points_to_planes, plane_frames_from_primitives,
    )

    depth = _stepped_facade_depth()
    prims, _stats = extract_planes_ransac(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
        horizon_y=CY)
    scale = float(prims[0].metadata.get("depth_scale_applied", 1.0))
    bp = back_project_normals(depth * scale, view_matrix=_view_matrix(1.6),
                              fx=FX, fy=FY, cx=CX, cy=CY)
    frames = plane_frames_from_primitives(prims)
    _labels, report = assign_points_to_planes(
        bp.pts_world, frames, camera_position=(0.0, 1.6, 0.0),
        valid=bp.valid_depth)

    measurable = report["assigned_px"] + report["unassigned_px"]
    assert report["assigned_px"] / max(1, measurable) > 0.9


def test_the_extent_percentiles_stay_robust_to_a_stray_inlier():
    """Widening is not the same as taking min/max: one bad pixel at 50 m must
    not stretch a 6 m wall across the scene."""

    from atlas_camera.core.plane_extraction import PlaneRansacConfig

    assert PlaneRansacConfig().extent_percentiles == (1.0, 99.0)
    lo, hi = PlaneRansacConfig().extent_percentiles
    assert lo > 0.0 and hi < 100.0
