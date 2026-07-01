"""Core camera math and physical-unit conversions.

Exporters call these helpers instead of carrying local conversion formulas.
Atlas core remains right-handed Y-up; these functions only handle camera units.
"""

from __future__ import annotations

import math

MM_PER_INCH = 25.4
FALLBACK_SENSOR_WIDTH_MM = 36.0
FALLBACK_SENSOR_HEIGHT_MM = 20.25
FOCAL_FALLBACK_CONFIDENCE_PENALTY = 0.15


def mm_to_inches(mm: float) -> float:
    return float(mm) / MM_PER_INCH


def inches_to_mm(inches: float) -> float:
    return float(inches) * MM_PER_INCH


def derive_sensor_height_mm(intrinsics) -> float:
    """Return sensor height in mm, falling back to aspect-ratio computation."""
    return float(
        intrinsics.sensor_height_mm
        or (intrinsics.sensor_width_mm * intrinsics.image_height / intrinsics.image_width)
    )


def focal_length_to_fov(focal_length_mm: float, sensor_size_mm: float) -> float:
    if focal_length_mm <= 0:
        raise ValueError("Focal length must be positive.")
    if sensor_size_mm <= 0:
        raise ValueError("Sensor size must be positive.")
    return math.degrees(2.0 * math.atan(sensor_size_mm / (2.0 * focal_length_mm)))


def fov_to_focal_length(fov_degrees: float, sensor_size_mm: float) -> float:
    if sensor_size_mm <= 0:
        raise ValueError("Sensor size must be positive.")
    if fov_degrees <= 0 or fov_degrees >= 180:
        raise ValueError("Field of view must be between 0 and 180 degrees.")
    return (sensor_size_mm / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)


def pixel_offset_to_normalized_film_offset(
    pixel_offset: float,
    *,
    aperture_mm: float,
    image_size_px: int,
) -> float:
    """Convert a principal-point pixel offset to a film-aperture fraction."""

    if aperture_mm <= 0:
        raise ValueError("Aperture must be positive.")
    if image_size_px <= 0:
        raise ValueError("Image size must be positive.")
    return float(pixel_offset) / float(image_size_px)


def normalized_film_offset_to_pixel_offset(
    normalized_offset: float,
    *,
    aperture_mm: float,
    image_size_px: int,
) -> float:
    if aperture_mm <= 0:
        raise ValueError("Aperture must be positive.")
    if image_size_px <= 0:
        raise ValueError("Image size must be positive.")
    return float(normalized_offset) * float(image_size_px)


def estimate_focal_with_fallback(
    *,
    fov_degrees: float,
    sensor_width_mm: float | None,
) -> tuple[float, float, bool, float]:
    """Return focal, sensor width, fallback flag, and confidence penalty."""

    used_fallback = sensor_width_mm is None
    width = sensor_width_mm if sensor_width_mm is not None else FALLBACK_SENSOR_WIDTH_MM
    penalty = FOCAL_FALLBACK_CONFIDENCE_PENALTY if used_fallback else 0.0
    return fov_to_focal_length(fov_degrees, width), width, used_fallback, penalty
