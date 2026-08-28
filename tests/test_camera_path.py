"""Tests for camera_path.sample_camera_path — the keyframed camera-move sampler
behind AtlasBlockoutViewport's Camera Path mode and AtlasExportCameraPathUSD.

Focused here: exact pass-through at each keyframe's frame_index (regardless
of easing), degenerate 0/1-keyframe cases, and that easing actually bends the
interpolated path away from the middle-t point compared to linear.
"""

import pytest

from atlas_camera.core.camera_path import (
    build_preset_camera_path,
    sample_camera_path,
    sample_camera_path_fov_deg,
)
from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath
from atlas_camera.core.camera_path import AtlasExtrinsics


def _kf(frame_index, position, target, easing="linear", fov_deg=None):
    return AtlasCameraKeyframe(
        frame_index=frame_index, position=position, target=target, easing=easing, fov_deg=fov_deg
    )


def test_zero_keyframes_returns_empty():
    path = AtlasCameraPath(keyframes=[], fps=24.0, frame_count=10)
    assert sample_camera_path(path) == []


def test_zero_frame_count_returns_empty():
    path = AtlasCameraPath(keyframes=[_kf(0, (0, 0, 0), (0, 0, -1))], fps=24.0, frame_count=0)
    assert sample_camera_path(path) == []


def test_single_keyframe_repeats_static_pose():
    kf = _kf(0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    path = AtlasCameraPath(keyframes=[kf], fps=24.0, frame_count=5)
    frames = sample_camera_path(path)
    assert len(frames) == 5
    for extr in frames:
        assert extr.camera_position == pytest.approx((1.0, 2.0, 3.0))


def test_two_keyframes_exact_pass_through_at_endpoints():
    kf0 = _kf(0, (0.0, 1.0, 5.0), (0.0, 1.0, 0.0))
    kf1 = _kf(10, (5.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
    frames = sample_camera_path(path)

    assert len(frames) == 11
    assert frames[0].camera_position == pytest.approx(kf0.position)
    assert frames[10].camera_position == pytest.approx(kf1.position)


def test_middle_keyframe_exact_pass_through():
    kf0 = _kf(0, (0.0, 1.0, 5.0), (0.0, 1.0, 0.0))
    kf1 = _kf(5, (3.0, 2.0, 5.0), (1.0, 1.0, 0.0))
    kf2 = _kf(10, (5.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    path = AtlasCameraPath(keyframes=[kf0, kf1, kf2], fps=24.0, frame_count=11)
    frames = sample_camera_path(path)

    assert frames[0].camera_position == pytest.approx(kf0.position)
    assert frames[5].camera_position == pytest.approx(kf1.position, abs=1e-9)
    assert frames[10].camera_position == pytest.approx(kf2.position)


def test_pass_through_holds_regardless_of_easing():
    for easing in ("linear", "ease_in", "ease_out", "ease_in_out"):
        kf0 = _kf(0, (0.0, 1.0, 0.0), (0.0, 1.0, -1.0), easing=easing)
        kf1 = _kf(8, (4.0, 1.0, 0.0), (0.0, 1.0, -1.0), easing=easing)
        path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=9)
        frames = sample_camera_path(path)
        assert frames[0].camera_position == pytest.approx(kf0.position), easing
        assert frames[8].camera_position == pytest.approx(kf1.position), easing


def test_easing_shifts_midpoint_relative_to_linear():
    kf0 = _kf(0, (0.0, 1.0, 0.0), (0.0, 1.0, -1.0))
    kf1 = _kf(10, (10.0, 1.0, 0.0), (0.0, 1.0, -1.0))

    linear_path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
    linear_mid_x = sample_camera_path(linear_path)[5].camera_position[0]

    ease_in_kf0 = _kf(0, (0.0, 1.0, 0.0), (0.0, 1.0, -1.0), easing="ease_in")
    ease_in_path = AtlasCameraPath(keyframes=[ease_in_kf0, kf1], fps=24.0, frame_count=11)
    ease_in_mid_x = sample_camera_path(ease_in_path)[5].camera_position[0]

    # ease_in (t^2) lags behind linear at the midpoint (t=0.5 -> eased 0.25).
    assert ease_in_mid_x < linear_mid_x
    assert linear_mid_x == pytest.approx(5.0, abs=0.5)


def test_frames_outside_keyframe_range_clamp_to_ends():
    kf0 = _kf(2, (0.0, 1.0, 0.0), (0.0, 1.0, -1.0))
    kf1 = _kf(8, (4.0, 1.0, 0.0), (0.0, 1.0, -1.0))
    path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
    frames = sample_camera_path(path)

    # Frames before the first keyframe and after the last hold at the endpoints.
    assert frames[0].camera_position == pytest.approx(kf0.position)
    assert frames[10].camera_position == pytest.approx(kf1.position)


# ---------------------------------------------------------------------------
# sample_camera_path_fov_deg — the 🌀 Vertigo lens channel
# ---------------------------------------------------------------------------

def test_fov_channel_none_when_no_keyframe_has_fov():
    kf0 = _kf(0, (0.0, 1.0, 5.0), (0.0, 1.0, 0.0))
    kf1 = _kf(10, (5.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
    assert sample_camera_path_fov_deg(path) is None


def test_fov_channel_none_for_empty_path():
    assert sample_camera_path_fov_deg(AtlasCameraPath(keyframes=[], fps=24.0, frame_count=5)) is None
    kf = _kf(0, (0.0, 1.0, 0.0), (0.0, 1.0, -1.0), fov_deg=50.0)
    assert sample_camera_path_fov_deg(AtlasCameraPath(keyframes=[kf], fps=24.0, frame_count=0)) is None


def test_fov_channel_single_keyframe_repeats():
    kf = _kf(0, (0.0, 1.0, 0.0), (0.0, 1.0, -1.0), fov_deg=42.5)
    path = AtlasCameraPath(keyframes=[kf], fps=24.0, frame_count=4)
    assert sample_camera_path_fov_deg(path) == pytest.approx([42.5] * 4)


def test_fov_channel_exact_pass_through_at_keyframes():
    # The vertigo shape: two keyframes, fov widening while the camera pushes in.
    for easing in ("linear", "ease_in", "ease_out", "ease_in_out"):
        kf0 = _kf(0, (0.0, 1.6, 8.0), (0.0, 1.6, 0.0), easing=easing, fov_deg=40.0)
        kf1 = _kf(10, (0.0, 1.6, 6.4), (0.0, 1.6, 0.0), fov_deg=48.0)
        path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
        fovs = sample_camera_path_fov_deg(path)
        assert fovs is not None and len(fovs) == 11
        assert fovs[0] == pytest.approx(40.0), easing
        assert fovs[10] == pytest.approx(48.0), easing
        # Monotone-ish ramp between: interior samples stay inside the range.
        assert all(39.9 <= v <= 48.1 for v in fovs), easing


def test_fov_channel_fills_missing_keyframes_forward():
    # Middle keyframe has no fov -> inherits the previous defined one (40),
    # so frame 5 sits exactly at 40 and the ramp to 55 happens in segment 2.
    kf0 = _kf(0, (0.0, 1.0, 8.0), (0.0, 1.0, 0.0), fov_deg=40.0)
    kf1 = _kf(5, (0.0, 1.0, 7.0), (0.0, 1.0, 0.0))
    kf2 = _kf(10, (0.0, 1.0, 6.0), (0.0, 1.0, 0.0), fov_deg=55.0)
    path = AtlasCameraPath(keyframes=[kf0, kf1, kf2], fps=24.0, frame_count=11)
    fovs = sample_camera_path_fov_deg(path)
    assert fovs[0] == pytest.approx(40.0)
    assert fovs[5] == pytest.approx(40.0, abs=1e-9)
    assert fovs[10] == pytest.approx(55.0)


def test_fov_channel_backfills_leading_missing_keyframes():
    kf0 = _kf(0, (0.0, 1.0, 8.0), (0.0, 1.0, 0.0))  # no fov -> takes 44 from kf1
    kf1 = _kf(10, (0.0, 1.0, 6.0), (0.0, 1.0, 0.0), fov_deg=44.0)
    path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
    fovs = sample_camera_path_fov_deg(path)
    assert fovs == pytest.approx([44.0] * 11)


def test_fov_channel_clamps_outside_keyframe_range():
    kf0 = _kf(2, (0.0, 1.0, 8.0), (0.0, 1.0, 0.0), fov_deg=40.0)
    kf1 = _kf(8, (0.0, 1.0, 6.0), (0.0, 1.0, 0.0), fov_deg=50.0)
    path = AtlasCameraPath(keyframes=[kf0, kf1], fps=24.0, frame_count=11)
    fovs = sample_camera_path_fov_deg(path)
    assert fovs[0] == pytest.approx(40.0)
    assert fovs[1] == pytest.approx(40.0)
    assert fovs[9] == pytest.approx(50.0)
    assert fovs[10] == pytest.approx(50.0)


def test_camera_path_round_trip_to_dict():
    kf = _kf(0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), easing="ease_out")
    path = AtlasCameraPath(
        keyframes=[kf], fps=30.0, frame_count=7, lens_scale=0.72,
        baked_frame_indices=[6])
    data = path.to_dict()
    restored = AtlasCameraPath.from_dict(data)

    assert restored.fps == 30.0
    assert restored.frame_count == 7
    assert restored.lens_scale == pytest.approx(0.72)
    assert restored.baked_frame_indices == [6]
    assert len(restored.keyframes) == 1
    assert restored.keyframes[0].position == pytest.approx((1.0, 2.0, 3.0))
    assert restored.keyframes[0].easing == "ease_out"

def test_arc_preset_shape_orbits_while_dollying():
    """The ⤴ Arc move's 3-keyframe grammar (0/60/99: orbit halfway + dolly
    halfway at the midpoint) samples to a path whose azimuth advances AND
    whose radius to the pivot shrinks — i.e. the spline genuinely curves
    through the combined move instead of cutting a straight chord."""
    import math

    target = (0.0, 1.0, 0.0)

    def arc_kf(frame, angle_deg, dolly_frac, easing):
        a = math.radians(angle_deg)
        base = (0.0, 1.6, 8.0)
        off = (base[0] - target[0], base[1] - target[1], base[2] - target[2])
        rot = (off[0] * math.cos(a) + off[2] * math.sin(a), off[1],
               -off[0] * math.sin(a) + off[2] * math.cos(a))
        k = 1.0 - dolly_frac
        pos = (target[0] + rot[0] * k, target[1] + rot[1] * k, target[2] + rot[2] * k)
        return _kf(frame, pos, target, easing=easing)

    path = AtlasCameraPath(keyframes=[
        arc_kf(0, 0.0, 0.0, "ease_in_out"),
        arc_kf(60, 7.5, 0.075, "linear"),
        arc_kf(99, 15.0, 0.15, "linear"),
    ], fps=24.0, frame_count=100)
    frames = sample_camera_path(path)
    assert len(frames) == 100

    def polar(extr):
        x = extr.camera_position[0] - target[0]
        z = extr.camera_position[2] - target[2]
        return math.atan2(x, z), math.hypot(x, z)

    az0, r0 = polar(frames[0])
    az_mid, r_mid = polar(frames[60])
    az_end, r_end = polar(frames[99])
    assert az0 == pytest.approx(0.0, abs=1e-9) and r0 == pytest.approx(8.0)
    assert 0 < az_mid < az_end          # azimuth advances through the move
    assert r_end < r_mid < r0           # while the camera pushes in
    assert az_end == pytest.approx(math.radians(15.0), abs=1e-6)
    assert r_end == pytest.approx(8.0 * 0.85, rel=1e-6)

def test_camera_path_fov_survives_dict_round_trip():
    kf = _kf(0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), fov_deg=47.25)
    path = AtlasCameraPath(keyframes=[kf], fps=24.0, frame_count=3)
    restored = AtlasCameraPath.from_dict(path.to_dict())
    assert restored.keyframes[0].fov_deg == pytest.approx(47.25)


# --- 🎬 Cinematic rig-noise (shake) ------------------------------------------


def _shake_path(enabled=True, intensity=1.0, seed=7, frame_count=12):
    return AtlasCameraPath(
        keyframes=[
            _kf(0, (0.0, 1.6, 8.0), (0.0, 1.0, 0.0)),
            _kf(frame_count - 1, (2.0, 1.6, 6.0), (0.0, 1.0, 0.0)),
        ],
        fps=24.0, frame_count=frame_count,
        shake_enabled=enabled, shake_intensity=intensity, shake_seed=seed,
    )


def test_shake_offsets_deterministic_and_seed_sensitive():
    from atlas_camera.core.camera_path import shake_offsets

    a = shake_offsets(7.25, 24.0, 1.0, 7)
    b = shake_offsets(7.25, 24.0, 1.0, 7)
    c = shake_offsets(7.25, 24.0, 1.0, 8)
    assert a == b
    assert a != c
    assert any(v != 0.0 for v in a)


def test_shake_offsets_zero_intensity_is_exact_zero():
    from atlas_camera.core.camera_path import shake_offsets

    assert shake_offsets(42.0, 24.0, 0.0, 7) == (0.0,) * 6
    assert shake_offsets(42.0, 24.0, -1.0, 7) == (0.0,) * 6


def test_shake_offsets_linear_in_intensity():
    from atlas_camera.core.camera_path import shake_offsets

    one = shake_offsets(13.5, 24.0, 1.0, 3)
    two = shake_offsets(13.5, 24.0, 2.0, 3)
    for v1, v2 in zip(one, two):
        assert v2 == pytest.approx(2.0 * v1, rel=1e-12)


def test_apply_shake_to_pose_zero_offsets_is_identity():
    from atlas_camera.core.camera_path import apply_shake_to_pose

    pose = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert apply_shake_to_pose(*pose, (0.0,) * 6) == pose


def test_apply_shake_translation_moves_position_and_target_together():
    from atlas_camera.core.camera_path import apply_shake_to_pose

    pos, tgt, up = (0.0, 1.6, 8.0), (0.0, 1.6, 0.0), (0.0, 1.0, 0.0)
    pos2, tgt2, up2 = apply_shake_to_pose(pos, tgt, up, (0.01, -0.02, 0.005, 0, 0, 0))
    trans_p = tuple(b - a for a, b in zip(pos, pos2))
    trans_t = tuple(b - a for a, b in zip(tgt, tgt2))
    for a, b in zip(trans_p, trans_t):
        assert a == pytest.approx(b, abs=1e-12)
    assert up2 == up  # no roll requested


def test_sample_camera_path_is_clean_by_default_even_when_enabled():
    clean = sample_camera_path(_shake_path(enabled=False))
    with_fields = sample_camera_path(_shake_path(enabled=True))
    for a, b in zip(clean, with_fields):
        assert a.camera_position == b.camera_position


def test_sample_camera_path_apply_shake_differs_and_is_bounded():
    clean = sample_camera_path(_shake_path())
    shaken = sample_camera_path(_shake_path(), apply_shake=True)
    moved = 0
    for a, b in zip(clean, shaken):
        d = sum((x - y) ** 2 for x, y in zip(a.camera_position, b.camera_position)) ** 0.5
        assert d < 0.05 * 8.5  # well under ~5% of the ~8m subject distance
        if d > 0:
            moved += 1
    assert moved == len(clean)


def test_sample_camera_path_apply_shake_disabled_or_zero_matches_clean():
    for path in (_shake_path(enabled=False), _shake_path(intensity=0.0)):
        clean = sample_camera_path(path)
        opted_in = sample_camera_path(path, apply_shake=True)
        for a, b in zip(clean, opted_in):
            assert a.camera_position == b.camera_position
            assert a.camera_view_matrix == b.camera_view_matrix


def test_single_keyframe_with_shake_becomes_per_frame_motion():
    path = AtlasCameraPath(
        keyframes=[_kf(0, (0.0, 1.6, 8.0), (0.0, 1.0, 0.0))],
        fps=24.0, frame_count=8,
        shake_enabled=True, shake_intensity=1.0, shake_seed=5,
    )
    frames = sample_camera_path(path, apply_shake=True)
    positions = {f.camera_position for f in frames}
    assert len(positions) > 1  # a locked-off tripod with rig noise moves


def test_shake_fields_survive_dict_round_trip_and_default_off():
    path = _shake_path(enabled=True, intensity=1.35, seed=99)
    restored = AtlasCameraPath.from_dict(path.to_dict())
    assert restored.shake_enabled is True
    assert restored.shake_intensity == pytest.approx(1.35)
    assert restored.shake_seed == 99

    legacy = AtlasCameraPath.from_dict({"keyframes": [], "fps": 24.0, "frame_count": 0})
    assert legacy.shake_enabled is False
    assert legacy.shake_intensity == pytest.approx(1.0)
    assert legacy.shake_seed == 1


# -- handedness: which way "left" actually goes ------------------------------


def _level_camera():
    """Eye 10 m back from a pivot, looking down -Z, no roll.

    In that frame the camera's right is world +X, so "left" is unambiguous and
    a sign error cannot hide behind a rotation nobody can read.
    """
    import numpy as np

    return AtlasExtrinsics(
        camera_position=(0.0, 1.6, 10.0),
        camera_rotation_matrix=np.eye(3).tolist(),
    ), (0.0, 1.6, 0.0)


def test_pan_matches_the_js_handedness():
    """A baked pan_left used to travel RIGHT.

    The sin terms in the pan branch were transposed, so this module rotated the
    opposite way to `atlas_blockout.js` and a bake disagreed with the preview it
    came from. The frontend-mirror test could not see it: it compares the move
    NAMES, not the geometry they produce.

    Measured against the JS algebra on the camera below: pan_left's target
    lands at x = -2.59 and pan_right's at +2.59.
    """
    extrinsics, pivot = _level_camera()

    left = build_preset_camera_path(extrinsics, "pan_left", pivot=pivot)[0]
    right = build_preset_camera_path(extrinsics, "pan_right", pivot=pivot)[0]

    assert left.keyframes[-1].target[0] < 0, "pan_left turned toward camera-right"
    assert right.keyframes[-1].target[0] > 0, "pan_right turned toward camera-left"
    assert left.keyframes[-1].target[0] == pytest.approx(-2.588, abs=0.01)
    assert right.keyframes[-1].target[0] == pytest.approx(2.588, abs=0.01)


def test_orbit_and_pan_turn_the_same_way():
    """Orbit was always right; the point is that pan now agrees with it.

    Orbit goes through `pose_at`, pan hand-rolls its rotation. Two spellings of
    the same turn is exactly how they drifted apart, so this pins them together.
    """
    extrinsics, pivot = _level_camera()

    orbit = build_preset_camera_path(extrinsics, "orbit_left", pivot=pivot)[0]
    pan = build_preset_camera_path(extrinsics, "pan_left", pivot=pivot)[0]

    # Orbit moves the EYE left; pan moves the TARGET left. Both negative in x.
    assert orbit.keyframes[-1].position[0] < 0
    assert pan.keyframes[-1].target[0] < 0


def test_the_compound_moves_turn_the_same_way_as_the_simple_ones():
    extrinsics, pivot = _level_camera()

    for simple, compound in (("pan_left", "dolly_pan_left"),
                             ("pan_right", "dolly_pan_right")):
        a = build_preset_camera_path(extrinsics, simple, pivot=pivot)[0]
        b = build_preset_camera_path(extrinsics, compound, pivot=pivot)[0]
        assert (a.keyframes[-1].target[0] > 0) == (b.keyframes[-1].target[0] > 0), (
            f"{compound} turns the opposite way to {simple}"
        )


def test_crane_up_rises_and_holds_the_pivot():
    extrinsics, pivot = _level_camera()
    path = build_preset_camera_path(extrinsics, "crane_up", pivot=pivot)[0]
    start, end = path.keyframes[0], path.keyframes[-1]

    assert end.position[1] > start.position[1], "crane_up did not rise"
    assert end.position[0] == pytest.approx(start.position[0])
    assert end.position[2] == pytest.approx(start.position[2]), "crane drifted sideways"
    # 25% of a 10 m throw, and the pivot is held so the tilt is geometric.
    assert end.position[1] - start.position[1] == pytest.approx(2.5, abs=0.01)
    assert tuple(end.target) == pytest.approx(tuple(pivot))
