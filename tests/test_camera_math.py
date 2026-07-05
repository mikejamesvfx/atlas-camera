import math

import pytest

from atlas_camera.core import camera_math
from atlas_camera.core.schema import AtlasExtrinsics


def _clean_primary(eye, pivot):
    """An AtlasExtrinsics that already looks at ``pivot`` from ``eye`` with Y-up."""
    view, world, rot3 = camera_math.look_at_view_matrix(eye, pivot)
    return AtlasExtrinsics(
        camera_position=eye,
        camera_rotation_matrix=rot3,
        camera_world_matrix=world,
        camera_view_matrix=view,
    )


def _mat_vec(m, v):
    return tuple(
        sum(m[r][c] * v[c] for c in range(4))
        for r in range(4)
    )


def test_mm_inches_round_trip():
    for mm in (0.0, 1.0, 24.0, 36.0):
        assert camera_math.inches_to_mm(camera_math.mm_to_inches(mm)) == pytest.approx(mm)


def test_fov_focal_round_trip():
    for fov in (18.0, 45.0, 90.0, 120.0):
        focal = camera_math.fov_to_focal_length(fov, 36.0)
        assert camera_math.focal_length_to_fov(focal, 36.0) == pytest.approx(fov)


def test_pixel_offset_normalized_film_offset_round_trip():
    for px_offset in (-200.0, -1.5, 0.0, 52.25, 480.0):
        normalized = camera_math.pixel_offset_to_normalized_film_offset(
            px_offset,
            aperture_mm=36.0,
            image_size_px=1920,
        )
        assert camera_math.normalized_film_offset_to_pixel_offset(
            normalized,
            aperture_mm=36.0,
            image_size_px=1920,
        ) == pytest.approx(px_offset)


def test_estimate_focal_with_fallback_flags_assumption():
    focal, sensor_width, used_fallback, penalty = camera_math.estimate_focal_with_fallback(
        fov_degrees=60.0,
        sensor_width_mm=None,
    )

    assert used_fallback is True
    assert sensor_width == camera_math.FALLBACK_SENSOR_WIDTH_MM
    assert penalty > 0.0
    assert focal == pytest.approx(
        camera_math.fov_to_focal_length(60.0, camera_math.FALLBACK_SENSOR_WIDTH_MM)
    )


# ---------------------------------------------------------------------------
# Orbit / look-at (patch camera construction)
# ---------------------------------------------------------------------------

def test_look_at_view_matrix_convention():
    # Camera at (0,0,10) looking toward the origin, Y-up. The target must land
    # in front of the camera: view @ target has -Z (depth = -cam.z > 0).
    eye = (0.0, 0.0, 10.0)
    target = (0.0, 0.0, 0.0)
    view, world, _rot = camera_math.look_at_view_matrix(eye, target)
    cam_target = _mat_vec(view, (*target, 1.0))
    assert cam_target[2] < 0.0                      # in front (camera looks along -Z)
    assert cam_target[0] == pytest.approx(0.0, abs=1e-9)
    assert cam_target[1] == pytest.approx(0.0, abs=1e-9)
    # view is the inverse of world: view @ world == identity (translation row too).
    prod = tuple(
        tuple(sum(view[r][k] * world[k][c] for k in range(4)) for c in range(4))
        for r in range(4)
    )
    for r in range(4):
        for c in range(4):
            assert prod[r][c] == pytest.approx(1.0 if r == c else 0.0, abs=1e-9)


def test_orbit_zero_reproduces_primary_that_looks_at_pivot():
    pivot = (0.0, 0.0, 10.0)
    eye = (0.0, 2.0, 0.0)
    primary = _clean_primary(eye, pivot)
    orbited = camera_math.orbit_camera(primary, pivot, d_azimuth_deg=0.0, d_elevation_deg=0.0)
    for a, b in zip(orbited.camera_position, eye):
        assert a == pytest.approx(b, abs=1e-9)
    for r in range(4):
        for c in range(4):
            assert orbited.camera_view_matrix[r][c] == pytest.approx(
                primary.camera_view_matrix[r][c], abs=1e-9
            )


def test_orbit_azimuth_preserves_radius_and_height():
    pivot = (0.0, 0.0, 10.0)
    eye = (0.0, 2.0, 0.0)
    primary = _clean_primary(eye, pivot)
    r0 = math.dist(eye, pivot)
    for az in (30.0, 45.0, 90.0, -45.0):
        orbited = camera_math.orbit_camera(primary, pivot, d_azimuth_deg=az, d_elevation_deg=0.0)
        pos = orbited.camera_position
        assert math.dist(pos, pivot) == pytest.approx(r0, abs=1e-6)   # radius preserved
        assert pos[1] == pytest.approx(eye[1], abs=1e-6)              # pure-azimuth keeps height
        # still aims at the pivot: pivot is in front of the orbited camera.
        cam_piv = _mat_vec(orbited.camera_view_matrix, (*pivot, 1.0))
        assert cam_piv[2] < 0.0


def test_orbit_elevation_raises_camera():
    pivot = (0.0, 0.0, 10.0)
    eye = (0.0, 2.0, 0.0)
    primary = _clean_primary(eye, pivot)
    up = camera_math.orbit_camera(primary, pivot, d_azimuth_deg=0.0, d_elevation_deg=35.0)
    assert up.camera_position[1] > eye[1]                             # raised
    assert math.dist(up.camera_position, pivot) == pytest.approx(math.dist(eye, pivot), abs=1e-6)


def test_orbit_distance_scale_changes_radius():
    pivot = (0.0, 0.0, 10.0)
    eye = (0.0, 2.0, 0.0)
    primary = _clean_primary(eye, pivot)
    r0 = math.dist(eye, pivot)
    closer = camera_math.orbit_camera(
        primary, pivot, d_azimuth_deg=0.0, d_elevation_deg=0.0, distance_scale=0.6,
    )
    assert math.dist(closer.camera_position, pivot) == pytest.approx(0.6 * r0, abs=1e-6)


def test_ground_lookat_pivot_hits_ground_for_downward_camera():
    pivot = (0.0, 0.0, 10.0)                 # on the ground plane
    eye = (0.0, 3.0, 0.0)
    primary = _clean_primary(eye, pivot)     # looks down at the ground point
    hit = camera_math.ground_lookat_pivot(primary)
    assert hit[1] == pytest.approx(0.0, abs=1e-9)
    assert hit[0] == pytest.approx(pivot[0], abs=1e-6)
    assert hit[2] == pytest.approx(pivot[2], abs=1e-6)
