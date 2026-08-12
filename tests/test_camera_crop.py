"""Analytic tests for crop-adjusted camera intrinsics (Dynamic Plates).

Release-blocking per the Dynamic Plates v0.1 spec: a visually plausible but
geometrically misregistered crop is not acceptable, so every relationship here
is checked exactly, not approximately.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.camera_crop import (
    CropTransform,
    RegionROI,
    crop_intrinsics,
    scale_intrinsics,
)
from atlas_camera.core.intrinsics import build_intrinsics


def _intrinsics(width=7680, height=4320, focal_mm=32.0):
    return build_intrinsics(
        image_width=width, image_height=height, focal_length_mm=focal_mm)


# ---------------------------------------------------------------- RegionROI

def test_roi_round_trip():
    roi = RegionROI(x=0, y=2730, width=7680, height=1590)
    again = RegionROI.from_dict(roi.to_dict())
    assert again == roi


def test_roi_from_dict_none():
    assert RegionROI.from_dict(None) is None


def test_roi_degenerate_raises():
    with pytest.raises(ValueError):
        RegionROI(x=0, y=0, width=0, height=10)
    with pytest.raises(ValueError):
        RegionROI(x=0, y=0, width=10, height=-1)


def test_roi_clamped_inside_image():
    roi = RegionROI(x=-50, y=4000, width=8000, height=1000)
    c = roi.clamped(7680, 4320)
    assert c.x == 0 and c.y == 4000
    assert c.x + c.width <= 7680
    assert c.y + c.height <= 4320


def test_roi_expanded_pad_px_and_clamp():
    roi = RegionROI(x=100, y=100, width=200, height=100)
    e = roi.expanded(pad_px=50, image_width=1000, image_height=500)
    assert (e.x, e.y, e.width, e.height) == (50, 50, 300, 200)
    # clamped at the borders
    edge = RegionROI(x=0, y=0, width=100, height=100)
    e2 = edge.expanded(pad_px=50, image_width=1000, image_height=500)
    assert (e2.x, e2.y) == (0, 0)
    assert (e2.width, e2.height) == (150, 150)


def test_roi_expanded_pad_frac():
    roi = RegionROI(x=400, y=400, width=200, height=100)
    # pad_frac of max(w, h) = 0.1 * 200 = 20 px each side
    e = roi.expanded(pad_frac=0.1, image_width=1000, image_height=1000)
    assert (e.x, e.y, e.width, e.height) == (380, 380, 240, 140)


# ------------------------------------------------------------ crop_intrinsics

def test_full_frame_crop_is_identity():
    intr = _intrinsics()
    roi = RegionROI(x=0, y=0, width=intr.image_width, height=intr.image_height)
    out = crop_intrinsics(intr, roi)
    assert out.image_width == intr.image_width
    assert out.image_height == intr.image_height
    assert out.fx_px == pytest.approx(intr.fx_px)
    assert out.fy_px == pytest.approx(intr.fy_px)
    assert out.cx_px == pytest.approx(intr.cx_px)
    assert out.cy_px == pytest.approx(intr.cy_px)


def test_offset_crop_shifts_principal_point_exactly():
    intr = _intrinsics()
    roi = RegionROI(x=0, y=2730, width=7680, height=1590)
    out = crop_intrinsics(intr, roi)
    assert out.image_width == 7680 and out.image_height == 1590
    assert out.fx_px == pytest.approx(intr.fx_px)
    assert out.fy_px == pytest.approx(intr.fy_px)
    assert out.cx_px == pytest.approx(intr.cx_px - 0)
    assert out.cy_px == pytest.approx(intr.cy_px - 2730)
    assert out.principal_point_px == pytest.approx((out.cx_px, out.cy_px))


def test_crop_does_not_mutate_source():
    intr = _intrinsics()
    before = (intr.cx_px, intr.cy_px, intr.image_width)
    crop_intrinsics(intr, RegionROI(x=10, y=20, width=100, height=50))
    assert (intr.cx_px, intr.cy_px, intr.image_width) == before


def test_crop_resolves_centre_fallback():
    # Intrinsics with no explicit cx/cy: crop must resolve the ladder first.
    intr = build_intrinsics(image_width=2000, image_height=1000,
                            focal_length_mm=35.0)
    intr.cx_px = None
    intr.cy_px = None
    intr.principal_point_px = None
    out = crop_intrinsics(intr, RegionROI(x=500, y=250, width=1000, height=500))
    assert out.cx_px == pytest.approx(1000.0 - 500)
    assert out.cy_px == pytest.approx(500.0 - 250)


def test_crop_roi_outside_image_raises():
    intr = _intrinsics(width=100, height=100)
    with pytest.raises(ValueError):
        crop_intrinsics(intr, RegionROI(x=50, y=50, width=100, height=10))


# ----------------------------------------------------------- scale_intrinsics

def test_scale_intrinsics_scales_focal_and_principal():
    intr = _intrinsics(width=7680, height=1590)
    out = scale_intrinsics(intr, 1920, 398)
    sx = 1920 / 7680
    sy = 398 / 1590
    assert out.image_width == 1920 and out.image_height == 398
    assert out.fx_px == pytest.approx(intr.fx_px * sx)
    assert out.fy_px == pytest.approx(intr.fy_px * sy)
    assert out.cx_px == pytest.approx(intr.cx_px * sx)
    assert out.cy_px == pytest.approx(intr.cy_px * sy)


def test_scale_intrinsics_identity():
    intr = _intrinsics(width=640, height=480)
    out = scale_intrinsics(intr, 640, 480)
    assert out.fx_px == pytest.approx(intr.fx_px)
    assert out.cx_px == pytest.approx(intr.cx_px)


# -------------------------------------------------------------- CropTransform

def test_crop_transform_round_trip_pixel():
    ct = CropTransform(source_width=7680, source_height=4320,
                       roi=RegionROI(x=128, y=2730, width=7424, height=1590),
                       output_width=1856, output_height=398)
    for px, py in [(128.0, 2730.0), (4000.5, 3500.25), (7551.0, 4319.0)]:
        cx, cy = ct.full_to_crop(px, py)
        back = ct.crop_to_full(cx, cy)
        assert back[0] == pytest.approx(px, abs=1e-9)
        assert back[1] == pytest.approx(py, abs=1e-9)


def test_crop_transform_unscaled_is_pure_offset():
    ct = CropTransform(source_width=1000, source_height=800,
                       roi=RegionROI(x=100, y=200, width=300, height=400),
                       output_width=300, output_height=400)
    assert ct.full_to_crop(100.0, 200.0) == pytest.approx((0.0, 0.0))
    assert ct.full_to_crop(399.0, 599.0) == pytest.approx((299.0, 399.0))


def test_crop_transform_serialization_round_trip():
    ct = CropTransform(source_width=1000, source_height=800,
                       roi=RegionROI(x=100, y=200, width=300, height=400),
                       output_width=150, output_height=200)
    again = CropTransform.from_dict(ct.to_dict())
    assert again == ct
    assert CropTransform.from_dict(None) is None


def test_crop_then_scale_matches_transform():
    """Intrinsics pipeline and pixel-mapping pipeline must agree.

    A world point projecting to full-image pixel p must project to
    ct.full_to_crop(p) under the crop+scaled intrinsics. Equivalent check
    without geometry: the affine maps must be identical.
    """
    intr = _intrinsics(width=2000, height=1000)
    roi = RegionROI(x=300, y=400, width=800, height=500)
    ct = CropTransform(source_width=2000, source_height=1000, roi=roi,
                       output_width=400, output_height=250)
    cropped = scale_intrinsics(crop_intrinsics(intr, roi), 400, 250)
    # pixel that coincides with the principal point maps to cropped principal point
    u, v = ct.full_to_crop(intr.cx_px, intr.cy_px)
    assert u == pytest.approx(cropped.cx_px)
    assert v == pytest.approx(cropped.cy_px)
    # one focal length to the right in x maps to one (scaled) focal length
    u2, _ = ct.full_to_crop(intr.cx_px + intr.fx_px, intr.cy_px)
    assert u2 - u == pytest.approx(cropped.fx_px)
