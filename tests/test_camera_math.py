import pytest

from atlas_camera.core import camera_math


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
