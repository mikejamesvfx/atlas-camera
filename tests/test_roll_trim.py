"""AtlasRollTrim — roll a solve about the view axis, everything else invariant."""
import math

import pytest

from atlas_camera.comfy.nodes import AtlasRollTrim
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasHorizon,
    AtlasSolve,
)


def _solve(eye=(1.0, 2.0, 5.0), target=(0.0, 1.0, 0.0), horizon=True):
    view, world, rot3 = look_at_view_matrix(eye, target)
    intr = build_intrinsics(image_width=1920, image_height=1080,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=tuple(float(v) for v in eye),
        camera_rotation_matrix=rot3,
        camera_world_matrix=world,
        camera_view_matrix=view,
    ))
    solve = AtlasSolve(camera=cam, image_width=1920, image_height=1080)
    if horizon:
        solve.horizon_line = AtlasHorizon(
            line_coefficients=(0.0, 1.0, -540.0),
            endpoints_px=((0.0, 540.0), (1920.0, 540.0)),
            confidence=0.9,
        )
    return solve


def _mat_vec3(m, v):
    return [sum(m[r][k] * v[k] for k in range(3)) for r in range(3)]


def _forward_world(extr):
    # camera looks along -Z in camera space; cam->world rotation columns are
    # the camera axes, so forward = -column 2 of camera_rotation_matrix.
    r = extr.camera_rotation_matrix
    return [-r[0][2], -r[1][2], -r[2][2]]


def _up_world(extr):
    r = extr.camera_rotation_matrix
    return [r[0][1], r[1][1], r[2][1]]


def test_noop_at_zero():
    solve = _solve()
    out, report = AtlasRollTrim().trim(solve, roll_deg=0.0)
    assert out is not solve  # deep copy either way
    assert out.camera.extrinsics.camera_view_matrix == solve.camera.extrinsics.camera_view_matrix
    assert "no-op" in report


def test_position_and_forward_invariant_up_rotates_by_delta():
    solve = _solve()
    e0 = solve.camera.extrinsics
    f0, u0 = _forward_world(e0), _up_world(e0)
    out, _ = AtlasRollTrim().trim(solve, roll_deg=7.5)
    e1 = out.camera.extrinsics
    for a, b in zip(e0.camera_position, e1.camera_position):
        assert abs(a - b) < 1e-9
    f1, u1 = _forward_world(e1), _up_world(e1)
    for a, b in zip(f0, f1):
        assert abs(a - b) < 1e-9  # view direction untouched
    dot = sum(a * b for a, b in zip(u0, u1))
    assert abs(math.degrees(math.acos(max(-1, min(1, dot)))) - 7.5) < 1e-6


def test_view_world_matrices_stay_rigid_inverses():
    out, _ = AtlasRollTrim().trim(_solve(), roll_deg=-11.0)
    e = out.camera.extrinsics
    v, w = e.camera_view_matrix, e.camera_world_matrix
    for r in range(4):
        for c in range(4):
            ident = sum(v[r][k] * w[k][c] for k in range(4))
            assert abs(ident - (1.0 if r == c else 0.0)) < 1e-9
    # rotation3 (cam->world) is the transpose of the view rotation block
    for r in range(3):
        for c in range(3):
            assert abs(e.camera_rotation_matrix[r][c] - v[c][r]) < 1e-12


def test_input_solve_never_mutated():
    solve = _solve()
    before = solve.camera.extrinsics.camera_view_matrix
    AtlasRollTrim().trim(solve, roll_deg=5.0)
    assert solve.camera.extrinsics.camera_view_matrix == before


def test_metadata_accumulates_across_chained_trims():
    out1, _ = AtlasRollTrim().trim(_solve(), roll_deg=2.0)
    out2, _ = AtlasRollTrim().trim(out1, roll_deg=-0.5)
    assert abs(out2.debug_metadata["roll_trim_deg"] - 1.5) < 1e-9


def test_positive_roll_rotates_scene_ccw_on_screen():
    """Sign convention pinned: +roll tips the horizon's LEFT end down and
    RIGHT end up (image y grows downward) — the projected scene turns
    counter-clockwise on screen. The tooltip wording must match this."""
    solve = _solve(eye=(0.0, 1.6, 0.0), target=(0.0, 1.6, -10.0))  # level camera
    out, _ = AtlasRollTrim().trim(solve, roll_deg=4.0)
    (x0, y0), (x1, y1) = out.horizon_line.endpoints_px
    assert x0 == 0.0 and x1 == 1920.0
    assert y0 > 540.0 and y1 < 540.0  # left end down, right end up
    tilt = out.debug_metadata["camera_estimation"]["horizon_angle"]
    assert abs(tilt + 4.0) < 0.05  # image-space tilt = -roll for a level camera


def test_horizon_double_trim_round_trips():
    solve = _solve(eye=(0.0, 1.6, 0.0), target=(0.0, 1.6, -10.0))
    out, _ = AtlasRollTrim().trim(solve, roll_deg=6.0)
    back, _ = AtlasRollTrim().trim(out, roll_deg=-6.0)
    (x0, y0), (x1, y1) = back.horizon_line.endpoints_px
    assert abs(y0 - 540.0) < 1e-6 and abs(y1 - 540.0) < 1e-6


def test_report_warns_when_geometry_already_derived():
    from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive
    solve = _solve()
    solve.projection_scene = AtlasProjectionScene(
        proxy_geometry=[AtlasProxyPrimitive(name="wall", primitive_type="plane")])
    _, report = AtlasRollTrim().trim(solve, roll_deg=3.0)
    assert "BEFORE the depth/derive" in report
