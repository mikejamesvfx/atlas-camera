"""
camera_math — every physical-unit conversion Atlas needs, in one place.

See DECISIONS.md §2 and §3.

Rule: exporters NEVER perform their own unit math. They call these
functions and pass through DCC-specific field names only. If a conversion
is wrong, it is wrong here, once, and the golden-camera round-trip test
(tests/test_exports.py) catches it.
"""

from __future__ import annotations

import math

MM_PER_INCH = 25.4

# Last-resort sensor assumption when focal length cannot be recovered
# directly and only FOV is observable. See DECISIONS.md §3 — this is
# NEVER used silently; callers must lower confidence and write a note.
FALLBACK_SENSOR_WIDTH_MM = 36.0
FALLBACK_SENSOR_HEIGHT_MM = 20.25  # 16:9 crop of full-frame


def mm_to_inches(mm: float) -> float:
    """mm -> inches. Maya's horizontalFilmAperture/verticalFilmAperture
    are in inches; sensor dimensions recovered by Atlas are in mm."""
    return mm / MM_PER_INCH


def inches_to_mm(inches: float) -> float:
    return inches * MM_PER_INCH


def px_to_normalized_offset(px_offset: float, aperture_mm: float,
                             image_dim_px: int, focal_mm: float) -> float:
    """Convert a principal-point pixel offset into a normalized fraction
    of film aperture — the convention Maya's horizontalFilmOffset /
    verticalFilmOffset expect.

    This is NOT a simple px/image_dim ratio: it must go through the
    physical aperture, because the aperture (not the pixel grid) is what
    Maya's offset is normalized against.
    """
    if image_dim_px == 0:
        raise ValueError("image_dim_px must be nonzero")
    # Pixel offset -> physical mm offset on the sensor, then -> fraction
    # of physical aperture (Maya works in inches; convert at the end).
    mm_per_px = aperture_mm / image_dim_px
    physical_offset_mm = px_offset * mm_per_px
    aperture_inches = mm_to_inches(aperture_mm)
    offset_inches = mm_to_inches(physical_offset_mm)
    if aperture_inches == 0:
        raise ValueError("aperture_mm must be nonzero")
    return offset_inches / aperture_inches


def normalized_offset_to_px(norm_offset: float, aperture_mm: float,
                             image_dim_px: int) -> float:
    """Inverse of px_to_normalized_offset, for reading Maya data back in
    (used by the golden-camera round-trip test)."""
    aperture_inches = mm_to_inches(aperture_mm)
    offset_inches = norm_offset * aperture_inches
    physical_offset_mm = inches_to_mm(offset_inches)
    mm_per_px = aperture_mm / image_dim_px
    if mm_per_px == 0:
        raise ValueError("aperture_mm/image_dim_px must be nonzero")
    return physical_offset_mm / mm_per_px


def fov_to_focal_length(fov_deg: float, sensor_dim_mm: float) -> float:
    """Horizontal (or vertical) FOV in degrees + sensor dimension in mm
    -> focal length in mm. Standard thin-lens pinhole relation:

        focal = (sensor_dim / 2) / tan(fov / 2)
    """
    fov_rad = math.radians(fov_deg)
    return (sensor_dim_mm / 2.0) / math.tan(fov_rad / 2.0)


def focal_length_to_fov(focal_mm: float, sensor_dim_mm: float) -> float:
    """Inverse of fov_to_focal_length, in degrees."""
    if focal_mm == 0:
        raise ValueError("focal_mm must be nonzero")
    return math.degrees(2.0 * math.atan(sensor_dim_mm / (2.0 * focal_mm)))


def estimate_focal_with_fallback(
    fov_deg: float,
    sensor_width_mm: float | None,
) -> tuple[float, bool, float]:
    """Estimate focal length from FOV, falling back to an assumed sensor
    width only when none was recovered.

    Returns (focal_length_mm, used_fallback, confidence_penalty).

    Per DECISIONS.md §3: this NEVER returns a silent invention. The
    caller (LatentCamera construction) is responsible for using
    `used_fallback` to set `inferred=True`, append a note, and apply
    `confidence_penalty` to the focal metric — this function only
    computes the number and flags that it had to guess the sensor.
    """
    used_fallback = sensor_width_mm is None
    width = sensor_width_mm if sensor_width_mm is not None else FALLBACK_SENSOR_WIDTH_MM
    focal = fov_to_focal_length(fov_deg, width)
    penalty = 0.15 if used_fallback else 0.0
    return focal, used_fallback, penalty
