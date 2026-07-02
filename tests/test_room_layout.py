"""Tests for gravity-aligned Manhattan room-cuboid fitting (interiors).

Self-contained analytic scenes (tests/ is not a package — no cross-file
imports), following test_proxy_geometry.py's style: level camera at
(0,h,0), forward-z depth via analytic ray-plane intersection, composited
with min(). A corridor-proportioned room (narrow in X, long in Z) is used
so side walls are actually visible to a ~54deg-FOV pinhole camera — a
symmetric box room around the camera would hide its side walls entirely
behind the (always nearer) back wall, which is a property of a narrow-FOV
camera, not of the extractor.
"""

import math

import numpy as np
import pytest

from atlas_camera.core.room_layout import extract_room_cuboid

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


def _room_depth(h=1.6, x_neg=-1.5, x_pos=1.5, z_neg=-10.0, ceil_y=3.0,
               rot_deg=0.0, omit=()):
    """Analytic depth of a (possibly Y-rotated) corridor room: floor, optional
    ceiling, two side walls, and a back wall, composited via min(). Surfaces in
    ``omit`` are left out (open doorway / out-of-frame side)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dx = (uu - CX) / FX
    dy = -(vv - CY) / FY
    origin = np.array([0.0, h, 0.0])
    dirs = np.stack([dx, dy, -np.ones_like(dx)], axis=-1)

    th = math.radians(rot_deg)
    ex = np.array([math.cos(th), 0.0, -math.sin(th)])  # room local +X in world
    ez = np.array([math.sin(th), 0.0, math.cos(th)])   # room local +Z in world

    def plane_t(n, p0):
        n = np.asarray(n, dtype=float)
        d = n @ p0
        ndir = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - n @ origin) / ndir
        return t

    surfaces = {
        "floor": plane_t((0, 1, 0), (0, 0, 0)),
        "ceil": plane_t((0, 1, 0), (0, ceil_y, 0)) if ceil_y is not None else None,
        "xpos": plane_t(ex, x_pos * ex),
        "xneg": plane_t(-ex, x_neg * ex),
        "zneg": plane_t(-ez, z_neg * ez),
    }
    hy = ceil_y if ceil_y is not None else 100.0

    depths = [np.full((H, W), SKY)]
    for name, t in surfaces.items():
        if t is None or name in omit:
            continue
        valid = np.isfinite(t) & (t > 0.1)
        pt = origin + t[..., None] * dirs
        local_x, local_z, local_y = pt @ ex, pt @ ez, pt[..., 1]
        valid &= (local_x >= x_neg - 1e-6) & (local_x <= x_pos + 1e-6)
        valid &= (local_z >= z_neg - 1e-6) & (local_z <= 10.0 + 1e-6)
        if name in ("xpos", "xneg", "zneg"):
            valid &= (local_y >= 0) & (local_y <= hy)
        depths.append(np.where(valid, t, SKY))
    return np.stack(depths).min(axis=0)


def _by_prefix(prims, prefix):
    return [p for p in prims if p.name.startswith(prefix)]


def test_axis_aligned_room_floor_ceiling_and_walls():
    depth = _room_depth()
    prims, stats = extract_room_cuboid(depth, view_matrix=_view_matrix(1.6),
                                       fx=FX, fy=FY, cx=CX, cy=CY)
    assert _by_prefix(prims, "projection_floor")
    assert stats["ceiling"] is True
    assert stats["ceiling_height_m"] == pytest.approx(3.0, abs=0.3)
    assert stats["manhattan_azimuth_deg"] == pytest.approx(0.0, abs=5.0)
    # Back wall (z=-10, seen nearly head-on) is recovered precisely; the two
    # side walls are seen at a steep oblique angle (only their far slivers are
    # visible to a ~54deg FOV camera), so their offset estimate has more slop.
    walls = _by_prefix(prims, "projection_wall")
    assert len(walls) == stats["walls_found"] >= 2
    dists = {abs(w.metadata["distance_m"]) for w in walls}
    assert any(d == pytest.approx(10.0, abs=0.5) for d in dists)  # back wall


def test_rotated_room_recovers_azimuth():
    depth = _room_depth(rot_deg=30.0)
    prims, stats = extract_room_cuboid(depth, view_matrix=_view_matrix(1.6),
                                       fx=FX, fy=FY, cx=CX, cy=CY)
    # Folded modulo 90 deg, with 5deg-bin resolution.
    assert stats["manhattan_azimuth_deg"] == pytest.approx(30.0, abs=6.0)
    assert stats["walls_found"] >= 1


def test_partial_room_no_failure_when_a_wall_is_missing():
    depth = _room_depth(omit=("xpos",))
    prims, stats = extract_room_cuboid(depth, view_matrix=_view_matrix(1.6),
                                       fx=FX, fy=FY, cx=CX, cy=CY)
    assert stats["walls_found"] <= 3
    names = {p.name for p in _by_prefix(prims, "projection_wall")}
    assert "projection_wall_B_pos" not in names  # the omitted side is simply absent
    assert _by_prefix(prims, "projection_floor")  # everything else still returned


def test_ceiling_skipped_when_absent():
    depth = _room_depth(ceil_y=None)
    prims, stats = extract_room_cuboid(depth, view_matrix=_view_matrix(1.6),
                                       fx=FX, fy=FY, cx=CX, cy=CY)
    assert stats["ceiling"] is False
    assert not _by_prefix(prims, "projection_ceiling")


def test_depth_scale_reconciliation():
    depth = _room_depth() * 2.0
    prims, stats = extract_room_cuboid(depth, view_matrix=_view_matrix(1.6),
                                       fx=FX, fy=FY, cx=CX, cy=CY)
    assert stats["ground_scale"] == pytest.approx(0.5, abs=0.05)


def test_payload_is_json_safe():
    import json

    from atlas_camera.core.proxy_geometry import serialize_proxy_geometry
    from atlas_camera.core.schema import AtlasProjectionScene

    depth = _room_depth()
    prims, _ = extract_room_cuboid(depth, view_matrix=_view_matrix(1.6),
                                   fx=FX, fy=FY, cx=CX, cy=CY)
    scene = AtlasProjectionScene()
    scene.proxy_geometry.extend(prims)
    payload = serialize_proxy_geometry(scene)
    assert payload
    json.dumps(payload)
    for entry in payload:
        assert len(entry["transform"]) == 16
        assert len(entry["dimensions"]) == 3
