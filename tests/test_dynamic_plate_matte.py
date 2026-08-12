"""Matte -> ROI utilities for Dynamic Plates."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.camera_crop import RegionROI
from atlas_camera.core.dynamic_plate import (
    crop_image_region,
    feather_matte,
    matte_bbox,
    validate_matte_dimensions,
)


def _blob_matte(h=200, w=300, y0=50, y1=120, x0=30, x1=250):
    m = np.zeros((h, w), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def test_matte_bbox_exact():
    m = _blob_matte()
    roi = matte_bbox(m)
    assert roi == RegionROI(x=30, y=50, width=220, height=70)


def test_matte_bbox_uint8_threshold():
    m = (_blob_matte() * 255).astype(np.uint8)
    roi = matte_bbox(m, threshold=0.5)
    assert roi == RegionROI(x=30, y=50, width=220, height=70)


def test_matte_bbox_empty_returns_none():
    assert matte_bbox(np.zeros((10, 10), dtype=np.float32)) is None


def test_matte_bbox_soft_values_below_threshold_ignored():
    m = np.full((20, 20), 0.2, dtype=np.float32)
    m[5:8, 5:9] = 0.9
    roi = matte_bbox(m, threshold=0.5)
    assert roi == RegionROI(x=5, y=5, width=4, height=3)


def test_validate_matte_dimensions():
    validate_matte_dimensions((1080, 1920), 1920, 1080)
    with pytest.raises(ValueError):
        validate_matte_dimensions((1080, 1920), 1920, 1081)
    # HxWxC mattes allowed
    validate_matte_dimensions((1080, 1920, 3), 1920, 1080)


def test_feather_matte_preserves_range_and_widens():
    m = _blob_matte()
    f = feather_matte(m, 6.0)
    assert f.dtype == np.float32
    assert float(f.min()) >= 0.0 and float(f.max()) <= 1.0
    # support widens beyond the hard edge
    assert f[45, 100] > 0.0        # above the original blob top (y0=50)
    assert m[45, 100] == 0.0
    # interior stays saturated
    assert f[85, 140] == pytest.approx(1.0, abs=1e-3)


def test_feather_zero_radius_is_noop():
    m = _blob_matte()
    f = feather_matte(m, 0.0)
    assert np.array_equal(f, m.astype(np.float32))


def test_crop_image_region_shapes():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = RegionROI(x=20, y=10, width=60, height=40)
    out = crop_image_region(img, roi)
    assert out.shape == (40, 60, 3)
    gray = np.zeros((100, 200), dtype=np.float32)
    assert crop_image_region(gray, roi).shape == (40, 60)


def test_overscan_expand_clips_at_borders():
    m = np.zeros((100, 100), dtype=np.float32)
    m[80:100, 0:100] = 1.0     # water band touching bottom/left/right
    roi = matte_bbox(m)
    e = roi.expanded(pad_frac=0.10, image_width=100, image_height=100)
    assert e.x == 0 and e.x + e.width == 100
    assert e.y < 80 and e.y + e.height == 100
