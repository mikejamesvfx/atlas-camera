"""Face-based metric scale — recovering camera height from a known-size face.

Every accuracy test here works by CONSTRUCTION: place a virtual camera and an
upright subject at known metric positions, project the marked feature to pixels
with the same convention the solver uses, then check the recovered camera height
comes back to the number we started from. That makes a sign error or a
transposed rotation a hard failure rather than a plausible-looking number.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.face_scale import (  # noqa: E402
    DEFAULT_STATURE_M,
    FACE_METRICS,
    camera_height_from_face,
    face_metric_choices,
)

FX = FY = 1200.0
CX, CY = 960.0, 540.0


def project(point_world, rotation=None):
    """World point (camera at origin) -> pixel, inverting solver._ray_world.

    ``_ray_world`` builds ``ray_cam = [(u-cx)/fx, -(v-cy)/fy, -1]`` then rotates
    by cam_to_world. Inverting: rotate world->cam, divide by -Z, undo the axis
    signs. The y flip is the image-origin-top-left convention.
    """
    p = np.asarray(point_world, dtype=np.float64)
    if rotation is not None:
        p = np.asarray(rotation, dtype=np.float64) @ p
    x, y, z = p
    assert z < 0, "point must be in front of the camera (-Z)"
    u = CX + FX * (x / -z)
    v = CY - FY * (y / -z)
    return (float(u), float(v))


def standing_face(camera_height, stature=DEFAULT_STATURE_M, distance=3.0,
                  metric="head_chin_to_crown", lateral=0.0):
    """Endpoints of a marked feature on an upright subject, in world metres.

    Camera sits at the origin, so the ground plane is at ``Y = -camera_height``.
    """
    spec = FACE_METRICS[metric]
    ground_y = -camera_height
    anchor_y = ground_y + spec["anchor_ratio"] * stature
    size = spec["size_m"]

    if spec["axis"] == "vertical":
        bottom = (lateral, anchor_y, -distance)          # chin
        top = (lateral, anchor_y + size, -distance)      # crown
        return bottom, top
    half = size / 2.0
    return ((lateral - half, anchor_y, -distance),
            (lateral + half, anchor_y, -distance))


# ------------------------------------------------------------------ round trip


@pytest.mark.parametrize("metric", list(FACE_METRICS))
def test_every_metric_recovers_the_camera_height_it_was_built_from(metric):
    truth = 1.62
    a, b = standing_face(truth, metric=metric)
    res = camera_height_from_face(
        project(a), project(b), metric=metric,
        rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY,
        stature_m=DEFAULT_STATURE_M)

    assert res.camera_height == pytest.approx(truth, rel=1e-6), res.reason


@pytest.mark.parametrize("truth", [1.2, 1.6, 1.75, 3.0])
def test_recovery_across_camera_heights(truth):
    a, b = standing_face(truth)
    res = camera_height_from_face(
        project(a), project(b), rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height == pytest.approx(truth, rel=1e-6)


@pytest.mark.parametrize("distance", [1.5, 3.0, 8.0, 20.0])
def test_recovery_is_independent_of_subject_distance(distance):
    """A correct solve reads the same height whether the subject is near or far."""
    truth = 1.6
    a, b = standing_face(truth, distance=distance)
    res = camera_height_from_face(
        project(a), project(b), rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height == pytest.approx(truth, rel=1e-5)
    assert res.distance_m == pytest.approx(distance, rel=0.02)


def test_point_order_does_not_matter():
    a, b = standing_face(1.6)
    fwd = camera_height_from_face(project(a), project(b), rotation=np.eye(3),
                                  fx=FX, fy=FY, cx=CX, cy=CY)
    rev = camera_height_from_face(project(b), project(a), rotation=np.eye(3),
                                  fx=FX, fy=FY, cx=CX, cy=CY)
    assert fwd.camera_height == pytest.approx(rev.camera_height)


def test_off_axis_subject_still_recovers():
    """Face away from the principal point — exercises cx/cy handling."""
    truth = 1.6
    a, b = standing_face(truth, lateral=1.4)
    res = camera_height_from_face(project(a), project(b), rotation=np.eye(3),
                                  fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height == pytest.approx(truth, rel=1e-5)


def test_pitched_camera_recovers_through_the_rotation():
    """THE rotation test: a tilted camera must not bias the height.

    ``rotation`` is world->cam. A camera pitched down by 12 degrees sees the
    subject lower in frame; if the transpose were wrong the recovered height
    would move with the tilt instead of staying put.
    """
    truth, pitch = 1.6, math.radians(-12.0)
    c, s = math.cos(pitch), math.sin(pitch)
    world_to_cam = np.array([[1.0, 0.0, 0.0],
                             [0.0, c, -s],
                             [0.0, s, c]])

    a, b = standing_face(truth)
    res = camera_height_from_face(
        project(a, world_to_cam), project(b, world_to_cam),
        rotation=world_to_cam, fx=FX, fy=FY, cx=CX, cy=CY)

    assert res.camera_height == pytest.approx(truth, rel=1e-5)


def test_a_transposed_rotation_would_be_caught():
    """Guard the guard: feeding cam->world instead must NOT quietly agree."""
    truth, pitch = 1.6, math.radians(-20.0)
    c, s = math.cos(pitch), math.sin(pitch)
    world_to_cam = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    a, b = standing_face(truth)
    wrong = camera_height_from_face(
        project(a, world_to_cam), project(b, world_to_cam),
        rotation=world_to_cam.T, fx=FX, fy=FY, cx=CX, cy=CY)

    assert wrong.camera_height is None or abs(wrong.camera_height - truth) > 0.05


# -------------------------------------------------------------- stature effect


def test_taller_assumed_stature_raises_the_recovered_camera():
    """Stature enters the anchor height directly — a real, reportable sensitivity."""
    a, b = standing_face(1.6)
    pa, pb = project(a), project(b)
    short = camera_height_from_face(pa, pb, rotation=np.eye(3), fx=FX, fy=FY,
                                    cx=CX, cy=CY, stature_m=1.55)
    tall = camera_height_from_face(pa, pb, rotation=np.eye(3), fx=FX, fy=FY,
                                   cx=CX, cy=CY, stature_m=1.90)
    assert tall.camera_height > short.camera_height


def test_size_override_replaces_the_population_constant():
    a, b = standing_face(1.6)
    pa, pb = project(a), project(b)
    default = camera_height_from_face(pa, pb, rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    bigger = camera_height_from_face(pa, pb, rotation=np.eye(3), fx=FX, fy=FY,
                                     cx=CX, cy=CY, size_override_m=0.30)
    assert bigger.real_size_m == pytest.approx(0.30)
    assert bigger.distance_m > default.distance_m  # bigger head, same pixels -> farther


def test_uncertainty_band_is_reported_and_scales_with_spread():
    res = camera_height_from_face(*[project(p) for p in standing_face(1.6)],
                                  rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height_sd is not None and res.camera_height_sd >= 0.0
    # Band must be a small fraction of the height, not a token zero.
    assert res.camera_height_sd < res.camera_height


# -------------------------------------------------------------------- failures


def test_unknown_metric_is_named():
    res = camera_height_from_face((0, 0), (0, 10), metric="nose_width",
                                  rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height is None
    assert "unknown face metric" in res.reason


def test_coincident_points_rejected():
    res = camera_height_from_face((500.0, 300.0), (500.0, 300.0),
                                  rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height is None
    assert "extent" in res.reason or "coincident" in res.reason


def test_no_focal_length_is_refused():
    res = camera_height_from_face((0, 0), (0, 50), rotation=np.eye(3),
                                  fx=0.0, fy=0.0, cx=CX, cy=CY)
    assert res.camera_height is None
    assert "focal" in res.reason


def test_non_positive_stature_refused():
    a, b = standing_face(1.6)
    res = camera_height_from_face(project(a), project(b), rotation=np.eye(3),
                                  fx=FX, fy=FY, cx=CX, cy=CY, stature_m=0.0)
    assert res.camera_height is None
    assert "stature" in res.reason


def test_subject_above_the_camera_reports_why_rather_than_a_wrong_number():
    """A face far above the camera implies a negative height — must not pass silently.

    This is the "person on a balcony" case: they are not standing on the ground
    plane the solve is using, and the assumption behind this whole module fails.
    """
    a = (0.0, 3.0, -3.0)      # chin 3 m ABOVE the camera
    b = (0.0, 3.235, -3.0)
    res = camera_height_from_face(project(a), project(b), rotation=np.eye(3),
                                  fx=FX, fy=FY, cx=CX, cy=CY)
    assert res.camera_height is None
    assert "not standing on the solve's ground plane" in res.reason


# --------------------------------------------------------------- table hygiene


def test_metric_choices_are_stable_combo_values():
    """These serialize into saved workflows — append-only, order preserved."""
    choices = face_metric_choices()
    assert choices[:4] == ["head_chin_to_crown", "face_chin_to_hairline",
                           "head_width", "interpupillary"]


def test_every_metric_declares_a_complete_spec():
    for name, spec in FACE_METRICS.items():
        assert spec["axis"] in ("vertical", "horizontal"), name
        assert 0 < spec["size_m"] < 1.0, name
        assert 0 < spec["sd_m"] < spec["size_m"], name
        assert 0.5 < spec["anchor_ratio"] < 1.0, name
        assert spec["label"] and spec["note"], name


def test_result_serializes_json_safely():
    import json

    res = camera_height_from_face(*[project(p) for p in standing_face(1.6)],
                                  rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY)
    payload = json.loads(json.dumps(res.to_dict()))
    assert payload["camera_height_m"] == pytest.approx(1.6, rel=1e-6)
    assert payload["metric"] == "head_chin_to_crown"
