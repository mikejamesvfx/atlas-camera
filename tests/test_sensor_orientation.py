"""Sensor dimensions must be oriented to the FRAME, not the camera body.

Every consumer computes ``fx = focal_mm / sensor_width_mm * image_width_px``, so
"sensor width" has to mean the sensor extent along the IMAGE's width. The camera
registry stores a body's physical dimensions, which are orientation-free — an
X-H2 is 23.5 x 15.6 mm however it is held. Feeding those straight through put a
portrait plate's fx 34% out, silently (found 2026-08-15).
"""
from __future__ import annotations

import pytest

from atlas_camera.raw.metadata import RawMetadata, resolve_sensor_size

APSC_LONG, APSC_SHORT = 23.5, 15.6


def _meta():
    return RawMetadata(camera_make="FUJIFILM", camera_model="X-H2")


def _has_body():
    from atlas_camera.reference_data.camera_registry import find_camera_body
    return find_camera_body("FUJIFILM", "X-H2") is not None


pytestmark = pytest.mark.skipif(
    not _has_body(), reason="X-H2 not in the camera registry on this checkout")


def test_landscape_frame_keeps_the_long_edge_as_width():
    r = resolve_sensor_size(_meta(), 7752, 5178)
    assert r.source == "camera_db"
    assert r.sensor_width_mm == pytest.approx(APSC_LONG)
    assert r.sensor_height_mm == pytest.approx(APSC_SHORT)


def test_portrait_frame_transposes_to_the_short_edge():
    r = resolve_sensor_size(_meta(), 5178, 7752)
    assert r.source == "camera_db"
    assert r.sensor_width_mm == pytest.approx(APSC_SHORT)
    assert r.sensor_height_mm == pytest.approx(APSC_LONG)
    assert any("transposed" in w for w in r.warnings), (
        "a silent transposition is exactly the failure mode this prevents")


def test_portrait_focal_reproduces_the_measured_rig_intrinsics():
    """The sh001 rig solved metrically to fx = 6207 px off a surveyed 14.6 m
    baseline. The oriented sensor must reproduce that from EXIF alone; the
    unoriented one gives 4120 px."""
    r = resolve_sensor_size(_meta(), 5178, 7752)
    fx = 18.7 / r.sensor_width_mm * 5178
    assert fx == pytest.approx(6207, rel=0.01)
    assert 18.7 / APSC_LONG * 5178 == pytest.approx(4120, rel=0.01)  # the bug


def test_square_frame_is_left_alone():
    r = resolve_sensor_size(_meta(), 4000, 4000)
    assert r.sensor_width_mm == pytest.approx(APSC_LONG)
    assert not any("transposed" in w for w in r.warnings)


def test_unknown_dimensions_do_not_crash_the_orientation_check():
    r = resolve_sensor_size(_meta(), 0, 0)
    assert r.sensor_width_mm == pytest.approx(APSC_LONG)
