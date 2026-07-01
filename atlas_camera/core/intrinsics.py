"""Camera intrinsics helpers."""

from __future__ import annotations

from atlas_camera.core.camera_math import focal_length_to_fov, fov_to_focal_length
from atlas_camera.core.schema import AtlasIntrinsics


def derive_sensor_height_mm(
    sensor_width_mm: float,
    image_width_px: int,
    image_height_px: int,
) -> float:
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("Image dimensions must be positive.")
    return sensor_width_mm * (image_height_px / image_width_px)


def focal_length_mm_to_pixels(
    focal_length_mm: float,
    sensor_size_mm: float,
    image_size_px: int,
) -> float:
    if focal_length_mm <= 0:
        raise ValueError("Focal length must be positive.")
    if sensor_size_mm <= 0:
        raise ValueError("Sensor size must be positive.")
    if image_size_px <= 0:
        raise ValueError("Image size must be positive.")
    return focal_length_mm * image_size_px / sensor_size_mm


def build_intrinsics(
    *,
    image_width: int,
    image_height: int,
    focal_length_mm: float | None = None,
    sensor_width_mm: float = 36.0,
    sensor_height_mm: float | None = None,
    principal_point_px: tuple[float, float] | None = None,
    fx_px: float | None = None,
    fy_px: float | None = None,
) -> AtlasIntrinsics:
    """Build normalized pinhole intrinsics from lens or pixel hints."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if sensor_height_mm is None:
        sensor_height_mm = derive_sensor_height_mm(sensor_width_mm, image_width, image_height)

    if principal_point_px is None:
        principal_point_px = (image_width / 2.0, image_height / 2.0)

    if focal_length_mm is not None:
        fx_px = fx_px or focal_length_mm_to_pixels(
            focal_length_mm,
            sensor_width_mm,
            image_width,
        )
        fy_px = fy_px or focal_length_mm_to_pixels(
            focal_length_mm,
            sensor_height_mm,
            image_height,
        )

    return AtlasIntrinsics(
        image_width=image_width,
        image_height=image_height,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        principal_point_px=principal_point_px,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=principal_point_px[0],
        cy_px=principal_point_px[1],
    )
