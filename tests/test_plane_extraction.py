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
