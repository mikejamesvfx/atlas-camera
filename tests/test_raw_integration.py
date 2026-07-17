"""Opt-in end-to-end RAW decode test.

Runs only when ATLAS_TEST_RAW_FILE points at a local camera RAW file
(no multi-MB fixtures are committed). Optional extras:
  ATLAS_TEST_RAW_EXPECT_FOCAL — assert the EXIF focal (mm, float).
"""

import os

import pytest

RAW_FILE = os.environ.get("ATLAS_TEST_RAW_FILE")

pytestmark = pytest.mark.skipif(
    not RAW_FILE or not os.path.isfile(RAW_FILE or ""),
    reason="set ATLAS_TEST_RAW_FILE to a local NEF/CR2/CR3/RAF/ARW to run")


def test_import_raw_end_to_end():
    pytest.importorskip("rawpy")
    np = pytest.importorskip("numpy")
    from atlas_camera.raw import import_raw

    result = import_raw(RAW_FILE, half_size=True)
    assert result.linear_rgb.shape == result.display_srgb.shape
    assert result.linear_rgb.dtype == np.float32
    assert result.height, result.width == result.linear_rgb.shape[:2]
    assert result.display_srgb.min() >= 0.0 and result.display_srgb.max() <= 1.0
    # Undistort either applied or degraded to a documented status.
    assert result.undistort_status in (
        "applied", "disabled", "lensfunpy_missing", "no_profile_camera",
        "no_profile_lens", "no_lens_metadata")
    assert result.sensor_source in (
        "camera_db", "exif_focal_plane", "exif_35mm_ratio", "assumed_default")

    expect_focal = os.environ.get("ATLAS_TEST_RAW_EXPECT_FOCAL")
    if expect_focal:
        assert result.focal_length_mm == pytest.approx(float(expect_focal), rel=0.01)

    hint = result.intrinsics_hint()
    if result.focal_length_mm:
        assert hint["focal_length_mm"] == result.focal_length_mm
