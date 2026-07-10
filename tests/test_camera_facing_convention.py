"""The recovered camera must face world -Z (DCC default forward).

Yaw is unobservable from a single image, so the facing is a free convention —
and it must match Maya/Nuke default cameras (looking down -Z), or every DCC
import needs a manual -180 deg Y rotation (found by a real Maya lineup against
a hand-built blockout, 2026-07-10). All assertions measure the way the
pipeline itself does: ``cam_to_world = inv(camera_view_matrix)``, forward =
its rotation block @ [0,0,-1].
"""

import math

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.solver import (  # noqa: E402
    CameraFromVanishingPoints,
    _face_camera_toward_negative_z,
    solve_from_learned_prior,
)
from atlas_camera.inference.learned_prior import CameraPrior  # noqa: E402


def _prior(pitch_deg=0.0, roll_deg=0.0, focal_px=700.0, size=(1024, 1024)):
    p = math.radians(pitch_deg)
    up = (0.0, math.cos(p), -math.sin(p))
    return CameraPrior(
        focal_px=focal_px,
        fov_h_deg=2 * math.degrees(math.atan(size[0] / 2 / focal_px)),
        fov_v_deg=2 * math.degrees(math.atan(size[1] / 2 / focal_px)),
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        up_cam=up,
        principal_point_px=(size[0] / 2, size[1] / 2),
        image_width=size[0],
        image_height=size[1],
    )


def _forward_world(solve):
    vm = np.array(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
    return np.linalg.inv(vm)[:3, :3] @ np.array([0.0, 0.0, -1.0])


@pytest.mark.parametrize("pitch_deg", [-30.0, -8.0, 0.0, 13.4, 45.0])
def test_learned_solve_faces_negative_z_and_keeps_pitch(pitch_deg):
    solve = solve_from_learned_prior(_prior(pitch_deg=pitch_deg), camera_height=1.6)
    fwd = _forward_world(solve)
    assert fwd[2] < 0, f"camera faces +Z (fwd_world={fwd})"
    # The canonicalization must never disturb gravity: forward's vertical
    # component still matches the pitch sign (down-pitch looks down).
    if abs(pitch_deg) > 1e-6:
        assert math.copysign(1.0, fwd[1]) == math.copysign(1.0, pitch_deg)
    assert fwd[1] == pytest.approx(math.sin(math.radians(pitch_deg)), abs=1e-6)


def test_canonicalizer_world_side_semantics():
    # rotation here is the cam_to_world block (pipeline convention).
    plus_z_facing = np.diag([-1.0, 1.0, -1.0])   # fwd_world = (0,0,+1)
    fixed = _face_camera_toward_negative_z(plus_z_facing, np)
    assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(fixed) == pytest.approx(1.0)
    assert (fixed @ np.array([0.0, 0.0, -1.0]))[2] < 0
    # already -Z-facing input is returned untouched
    ident = np.eye(3)
    assert _face_camera_toward_negative_z(ident, np) is ident


def test_vp_rotation_faces_negative_z():
    w, h, f = 1920.0, 1080.0, 1400.0
    principal = np.array([w / 2, h / 2])
    yaw = math.radians(30.0)
    d_a = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    d_b = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])

    def vp_of(d):
        dz = d[2] if abs(d[2]) > 1e-9 else 1e-9
        return np.array([w / 2 + f * d[0] / dz, h / 2 + f * d[1] / dz])

    R = np.asarray(CameraFromVanishingPoints.estimate_rotation(
        vp_of(d_a), vp_of(d_b), f, principal))
    # estimate_rotation feeds the same extrinsics builder as the learned
    # path, so measure on the same (cam_to_world) side.
    fwd_world = R @ np.array([0.0, 0.0, -1.0])
    assert fwd_world[2] < 0, f"VP path faces +Z (fwd_world={fwd_world})"
