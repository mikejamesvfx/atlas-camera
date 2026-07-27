"""Tests for camera_path.sample_camera_path — the keyframed camera-move sampler
behind AtlasBlockoutViewport's Camera Path mode and AtlasExportCameraPathUSD.

Focused here: exact pass-through at each keyframe's frame_index (regardless
of easing), degenerate 0/1-keyframe cases, and that easing actually bends the
interpolated path away from the middle-t point compared to linear.
"""

import pytest

from atlas_camera.core.camera_path import sample_camera_path
from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath


def _kf(frame_index, position, target, easing="linear"):
    return AtlasCameraKeyframe(frame_index=frame_index, position=position, target=target, easing=easing)


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


def test_camera_path_round_trip_to_dict():
    kf = _kf(0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), easing="ease_out")
    path = AtlasCameraPath(
        keyframes=[kf], fps=30.0, frame_count=7, lens_scale=0.72)
    data = path.to_dict()
    restored = AtlasCameraPath.from_dict(data)

    assert restored.fps == 30.0
    assert restored.frame_count == 7
    assert restored.lens_scale == pytest.approx(0.72)
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
