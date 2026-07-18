"""AtlasPitchTrim 🎚 — pitch trim / gravity mirror (the D810 haze repair)."""

import math

import pytest

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, AtlasPitchTrim
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import (
    AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera,
)


def _solve(pitch_deg=39.0, height=45.0):
    """Camera at (0,h,0) pitched UP by pitch_deg (the flipped-gravity shape)."""
    d = 10.0
    target = (0.0, height + d * math.sin(math.radians(pitch_deg)),
              -d * math.cos(math.radians(pitch_deg)))
    view, world, rot3 = look_at_view_matrix((0.0, height, 0.0), target)
    extr = AtlasExtrinsics(camera_position=(0.0, height, 0.0),
                           camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=800, image_height=600, fx_px=700.0,
                           fy_px=700.0, cx_px=400.0, cy_px=300.0)
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _forward_y(solve):
    return -float(solve.camera.extrinsics.camera_world_matrix[1][2])


def test_registered():
    assert NODE_CLASS_MAPPINGS["AtlasPitchTrim"] is AtlasPitchTrim


def test_mirror_flips_forward_y_exactly():
    s = _solve(pitch_deg=39.0)
    up_y = _forward_y(s)
    assert up_y == pytest.approx(math.sin(math.radians(39.0)), abs=1e-6)
    out, report = AtlasPitchTrim().trim(s, mirror_gravity=True)
    assert _forward_y(out) == pytest.approx(-up_y, abs=1e-9)
    assert "MIRRORED" in report
    # Heading (horizontal forward direction) preserved.
    wm_in = s.camera.extrinsics.camera_world_matrix
    wm_out = out.camera.extrinsics.camera_world_matrix
    assert -wm_out[0][2] == pytest.approx(-wm_in[0][2], abs=1e-9)
    assert -wm_out[2][2] == pytest.approx(-wm_in[2][2], abs=1e-9)


def test_position_and_right_axis_invariant():
    s = _solve()
    out, _ = AtlasPitchTrim().trim(s, mirror_gravity=True, pitch_deg=5.0)
    assert out.camera.extrinsics.camera_position == pytest.approx((0.0, 45.0, 0.0))
    # Roll untouched: the camera's RIGHT axis (world matrix column 0) is the
    # rotation axis, so it must be bit-identical.
    for r in range(3):
        assert out.camera.extrinsics.camera_world_matrix[r][0] == \
            pytest.approx(s.camera.extrinsics.camera_world_matrix[r][0], abs=1e-9)


def test_positive_pitch_deg_tilts_down():
    s = _solve(pitch_deg=0.0)
    out, _ = AtlasPitchTrim().trim(s, pitch_deg=10.0)
    assert _forward_y(out) == pytest.approx(-math.sin(math.radians(10.0)), abs=1e-6)


def test_noop_and_stamp():
    s = _solve()
    out, report = AtlasPitchTrim().trim(s)
    assert "no-op" in report
    out2, _ = AtlasPitchTrim().trim(s, pitch_deg=4.0)
    out3, _ = AtlasPitchTrim().trim(out2, pitch_deg=3.0)
    assert out3.debug_metadata["pitch_trim_deg"] == pytest.approx(7.0)


def test_mirror_clears_camera_looks_up_flag():
    from atlas_camera.core.scene_health import evaluate_scene_health
    s = _solve(pitch_deg=39.0)
    s.debug_metadata["scale_source"] = "manual_override"
    assert "camera_looks_up" in {f.code for f in evaluate_scene_health(s).flags}
    out, _ = AtlasPitchTrim().trim(s, mirror_gravity=True)
    assert "camera_looks_up" not in {f.code for f in evaluate_scene_health(out).flags}


def test_gravity_override_sets_absolute_angles():
    from atlas_camera.comfy.nodes import AtlasGravityOverride
    s = _solve(pitch_deg=39.0)  # solved looking UP 39 (the flipped shape)
    out, report = AtlasGravityOverride().override(s, pitch_deg=32.0, roll_deg=0.9)
    # forward.y = -sin(32 down)
    assert _forward_y(out) == pytest.approx(-math.sin(math.radians(32.0)), abs=1e-6)
    # position untouched, heading preserved (horizontal forward unchanged dir)
    assert out.camera.extrinsics.camera_position == pytest.approx((0.0, 45.0, 0.0))
    wm_in = s.camera.extrinsics.camera_world_matrix
    wm_out = out.camera.extrinsics.camera_world_matrix
    import math as m
    az_in = m.atan2(-wm_in[0][2], -wm_in[2][2])
    az_out = m.atan2(-wm_out[0][2], -wm_out[2][2])
    assert az_out == pytest.approx(az_in, abs=1e-6)
    assert out.debug_metadata["gravity_override"]["pitch_deg"] == 32.0
    assert "32.0" in report or "-32.0" in report


def test_gravity_override_level_and_roll():
    from atlas_camera.comfy.nodes import AtlasGravityOverride
    s = _solve(pitch_deg=10.0)
    out, _ = AtlasGravityOverride().override(s, pitch_deg=0.0, roll_deg=0.0)
    assert _forward_y(out) == pytest.approx(0.0, abs=1e-9)
    # Roll: up vector stays world-up for zero roll.
    assert out.camera.extrinsics.camera_world_matrix[1][1] == pytest.approx(1.0, abs=1e-9)
    out2, _ = AtlasGravityOverride().override(s, pitch_deg=0.0, roll_deg=30.0)
    assert out2.camera.extrinsics.camera_world_matrix[1][1] == pytest.approx(
        math.cos(math.radians(30.0)), abs=1e-6)


def test_gravity_override_clears_camera_looks_up():
    from atlas_camera.comfy.nodes import AtlasGravityOverride
    from atlas_camera.core.scene_health import evaluate_scene_health
    s = _solve(pitch_deg=39.0)
    s.debug_metadata["scale_source"] = "manual_override"
    out, _ = AtlasGravityOverride().override(s, pitch_deg=32.0)
    assert "camera_looks_up" not in {f.code for f in evaluate_scene_health(out).flags}
