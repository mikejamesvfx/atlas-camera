from atlas_camera.core.intrinsics import build_intrinsics, derive_sensor_height_mm


def test_intrinsics_compute_pixel_focal_and_principal_point():
    intrinsics = build_intrinsics(
        image_width=1920,
        image_height=1080,
        focal_length_mm=50.0,
        sensor_width_mm=36.0,
    )

    assert round(intrinsics.sensor_height_mm, 4) == 20.25
    assert round(intrinsics.fx_px, 4) == 2666.6667
    assert round(intrinsics.fy_px, 4) == 2666.6667
    assert intrinsics.cx_px == 960
    assert intrinsics.cy_px == 540


def test_sensor_height_derives_from_aspect_ratio():
    assert derive_sensor_height_mm(36.0, 1920, 1080) == 20.25

