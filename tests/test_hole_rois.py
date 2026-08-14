"""Hole clustering, grid snapping and composite-back for hole-crop fill.

Release-blocking alongside `test_camera_crop.py`: a hole-derived ROI feeds a
crop CAMERA, so a ROI that is off by a pixel is a misregistered generation, not
a cosmetic framing difference. Registration itself is pinned in
`test_dynamic_plate_receiver.py`; what is pinned here is that hole clusters
become ROIs that are inside the plate, on the model grid, and that nothing is
dropped silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.camera_crop import (
    RegionROI,
    composite_crops,
    crop_intrinsics,
    hole_rois,
)
from atlas_camera.core.intrinsics import build_intrinsics


def _mask(shape, boxes):
    m = np.zeros(shape, dtype=bool)
    for x, y, w, h in boxes:
        m[y:y + h, x:x + w] = True
    return m


# --------------------------------------------------------------- clustering

def test_two_holes_become_two_rois():
    mask = _mask((512, 512), [(40, 40, 64, 64), (300, 320, 80, 48)])
    result = hole_rois([mask], pad_frac=0.0, min_area_px=1, snap=1,
                       max_rois=4)
    assert result.component_count == 2
    assert len(result.rois) == 2
    assert not result.dropped
    # largest first (ranked by hole area, not bbox area)
    assert result.rois[0].width == 64 and result.rois[0].height == 64
    assert (result.rois[1].x, result.rois[1].y) == (300, 320)


def test_diagonally_touching_holes_stay_separate():
    """4-connectivity: a corner touch is two holes, not one straddling ROI."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    mask[20:30, 20:30] = True
    result = hole_rois(mask, pad_frac=0.0, min_area_px=1, snap=1)
    assert result.component_count == 2


def test_c_shaped_hole_is_one_component():
    """Row-run union-find must join runs through a bend, not per-row."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:40, 10:16] = True     # left arm
    mask[34:40, 10:50] = True     # bottom bar
    mask[10:40, 44:50] = True     # right arm
    result = hole_rois(mask, pad_frac=0.0, min_area_px=1, snap=1)
    assert result.component_count == 1
    roi = result.rois[0]
    assert (roi.x, roi.y, roi.width, roi.height) == (10, 10, 40, 30)


def test_masks_are_unioned_across_time():
    """One crop must serve the WHOLE move — the union, not frame 0."""
    a = _mask((128, 128), [(10, 10, 20, 20)])
    b = _mask((128, 128), [(10, 10, 60, 20)])
    result = hole_rois([a, b], pad_frac=0.0, min_area_px=1, snap=1)
    assert len(result.rois) == 1
    assert result.rois[0].width == 60


def test_empty_mask_yields_no_rois():
    result = hole_rois(np.zeros((32, 32), dtype=bool))
    assert result.rois == []
    assert result.hole_area_px == 0
    assert result.coverage_frac == 0.0


def test_mismatched_masks_raise():
    with pytest.raises(ValueError):
        hole_rois([np.zeros((8, 8), dtype=bool), np.zeros((8, 9), dtype=bool)])


def test_uint8_and_float_masks_agree():
    box = [(4, 4, 8, 8)]
    ref = hole_rois(_mask((32, 32), box), pad_frac=0.0, min_area_px=1, snap=1)
    u8 = hole_rois((_mask((32, 32), box) * 255).astype(np.uint8),
                   pad_frac=0.0, min_area_px=1, snap=1)
    f32 = hole_rois(_mask((32, 32), box).astype(np.float32),
                    pad_frac=0.0, min_area_px=1, snap=1)
    assert ref.rois == u8.rois == f32.rois


# ------------------------------------------------------------ budget + drop

def test_max_rois_drops_are_reported_not_silent():
    boxes = [(10, 10, 40, 40), (200, 10, 30, 30), (10, 200, 20, 20),
             (200, 200, 16, 16)]
    result = hole_rois(_mask((256, 256), boxes), pad_frac=0.0, min_area_px=1,
                       snap=1, max_rois=2)
    assert len(result.rois) == 2
    assert len(result.dropped) == 2
    assert result.component_count == 4
    assert all("max_rois" in d["reason"] for d in result.dropped)
    assert result.dropped_area_px == 20 * 20 + 16 * 16


def test_min_area_drops_are_reported():
    result = hole_rois(_mask((128, 128), [(10, 10, 40, 40), (100, 100, 2, 2)]),
                       pad_frac=0.0, min_area_px=100, snap=1)
    assert len(result.rois) == 1
    assert len(result.dropped) == 1
    assert "min_area_px" in result.dropped[0]["reason"]


def test_coverage_frac_is_the_compute_saved_number():
    result = hole_rois(_mask((100, 100), [(10, 10, 20, 20)]), pad_frac=0.0,
                       min_area_px=1, snap=1)
    assert result.coverage_frac == pytest.approx(400 / 10000)
    assert result.to_dict()["coverage_frac"] == pytest.approx(0.04)


# -------------------------------------------------------------------- snap

def test_rois_snap_to_the_model_grid():
    result = hole_rois(_mask((512, 512), [(101, 77, 33, 45)]), pad_frac=0.1,
                       min_area_px=1, snap=64)
    roi = result.rois[0]
    assert roi.width % 64 == 0 and roi.height % 64 == 0
    assert roi.x >= 0 and roi.y >= 0
    assert roi.x + roi.width <= 512 and roi.y + roi.height <= 512


def test_snap_slides_an_edge_roi_inside_instead_of_shrinking_it():
    """A hole at the frame edge must keep its grid extents (clamping would
    break the /64 raster the snap exists to hit)."""
    roi = RegionROI(x=470, y=0, width=40, height=40).snapped(
        64, image_width=512, image_height=512)
    assert (roi.width, roi.height) == (64, 64)
    assert roi.x + roi.width == 512
    assert roi.y == 0


def test_snapped_roi_still_contains_the_hole():
    mask = _mask((256, 256), [(3, 200, 30, 50)])
    result = hole_rois(mask, pad_frac=0.0, min_area_px=1, snap=64)
    roi = result.rois[0]
    inside = mask[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
    assert int(inside.sum()) == int(mask.sum())


def test_snap_larger_than_image_falls_back_to_the_image():
    roi = RegionROI(x=0, y=0, width=10, height=10).snapped(
        64, image_width=40, image_height=40)
    assert (roi.width, roi.height) == (40, 40)


def test_snapped_roi_is_a_legal_crop_camera():
    """The whole point: a hole ROI must be acceptable to crop_intrinsics."""
    intr = build_intrinsics(image_width=7360, image_height=4912,
                            focal_length_mm=35.0)
    mask = np.zeros((4912, 7360), dtype=bool)
    mask[4700:4900, 7100:7350] = True
    result = hole_rois(mask, pad_frac=0.2, min_area_px=1, snap=64)
    roi = result.rois[0]
    cropped = crop_intrinsics(intr, roi)
    assert cropped.image_width == roi.width
    assert cropped.fx_px == pytest.approx(intr.fx_px)  # a crop keeps the lens
    assert cropped.cx_px == pytest.approx(intr.cx_px - roi.x)


# -------------------------------------------------------------- composite

def test_composite_pastes_into_the_roi_only():
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    roi = RegionROI(x=16, y=8, width=16, height=32)
    patch = np.full((32, 16, 3), 200, dtype=np.uint8)
    out = composite_crops(base, [patch], [roi])
    assert out[8:40, 16:32].min() == 200
    assert out[0:8].max() == 0
    assert base.max() == 0  # input untouched


def test_composite_respects_the_mask():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    roi = RegionROI(x=0, y=0, width=32, height=32)
    patch = np.full((32, 32, 3), 255, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True
    out = composite_crops(base, [patch], [roi], masks=[mask])
    assert out[8:16, 8:16].min() == 255
    assert out[0:8].max() == 0


def test_composite_feather_ramps_outside_the_mask():
    """The hole is covered FULLY and the blend happens outside it.

    An inward ramp leaves alpha near zero on the hole's own boundary, where
    the base still holds the inpaint sentinel — that is what painted a green
    rim around every fill on the first live composite.
    """
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    roi = RegionROI(x=0, y=0, width=32, height=32)
    patch = np.full((32, 32, 3), 255, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    out = composite_crops(base, [patch], [roi], masks=[mask], feather_px=3)
    assert out[8:24, 8:24].min() == 255      # every hole pixel fully replaced
    outside = int(out[7, 16, 0])
    assert 0 < outside < 255                 # ramp lives outside the hole
    assert int(out[4, 16, 0]) == 0           # and dies out within feather_px


def test_composite_rejects_a_size_mismatch():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    roi = RegionROI(x=0, y=0, width=16, height=16)
    with pytest.raises(ValueError, match="resize before compositing"):
        composite_crops(base, [np.zeros((8, 8, 3), np.uint8)], [roi])


def test_composite_rejects_an_out_of_frame_roi():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    roi = RegionROI(x=24, y=24, width=16, height=16)
    with pytest.raises(ValueError, match="does not lie within"):
        composite_crops(base, [np.zeros((16, 16, 3), np.uint8)], [roi])


def test_colour_match_undoes_a_global_shift():
    """A diffusion round trip returns the whole crop re-toned; the unmasked
    pixels are a paired sample that pins the shift exactly."""
    from atlas_camera.core.camera_crop import match_reference_colour

    rng = np.random.default_rng(0)
    ref = rng.integers(20, 200, size=(32, 32, 3)).astype(np.uint8)
    shifted = np.clip(ref * 0.8 + 25, 0, 255).astype(np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True                     # the "hole" is excluded
    fixed = match_reference_colour(shifted, ref, mask)
    keep = ~mask
    assert np.abs(fixed[keep].astype(int) - ref[keep].astype(int)).mean() < 2.0
    assert fixed.dtype == np.uint8


def test_colour_match_declines_on_too_few_samples():
    from atlas_camera.core.camera_crop import match_reference_colour

    crop = np.full((8, 8, 3), 100, dtype=np.uint8)
    ref = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)          # nothing unmasked to fit on
    out = match_reference_colour(crop, ref, mask)
    assert np.array_equal(out, crop)


def test_colour_match_rejects_a_raster_mismatch():
    from atlas_camera.core.camera_crop import match_reference_colour

    with pytest.raises(ValueError, match="matching rasters"):
        match_reference_colour(np.zeros((8, 8, 3), np.uint8),
                               np.zeros((4, 4, 3), np.uint8),
                               np.zeros((8, 8), bool))


# ------------------------------------------------- artist-drawn world ROIs

def _view_at(z=0.0):
    """Camera at the origin looking down -Z (Atlas convention)."""
    import numpy as _np
    view = _np.eye(4)
    view[2, 3] = z
    return view


def test_world_region_projects_to_its_screen_bbox():
    from atlas_camera.core.camera_crop import rois_from_world_regions

    # a 2m square 10m in front of the camera, centred on the axis
    region = {"label": "tear", "points_world": [
        (-1.0, -1.0, -10.0), (1.0, -1.0, -10.0),
        (1.0, 1.0, -10.0), (-1.0, 1.0, -10.0)]}
    out = rois_from_world_regions(
        [region], _view_at(), fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
        image_width=1920, image_height=1080, pad_frac=0.0, snap=1)
    roi = out.rois[0]
    # 2m at 10m with f=1000px spans 200px, centred on the principal point
    assert roi.width == 200 and roi.height == 200
    assert roi.x == 860 and roi.y == 440


def test_world_region_tracks_the_camera_through_the_move():
    """The marker is world-anchored: one selection frames the same surface
    from every view, which is why it is not a screen rectangle."""
    import numpy as _np

    from atlas_camera.core.camera_crop import rois_from_world_regions

    region = {"label": "tear", "points_world": [
        (-1.0, -1.0, -10.0), (1.0, -1.0, -10.0),
        (1.0, 1.0, -10.0), (-1.0, 1.0, -10.0)]}
    moved = _np.eye(4)
    moved[0, 3] = -2.0          # camera steps 2m to the right
    kw = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
              image_width=1920, image_height=1080, pad_frac=0.0, snap=1)
    a = rois_from_world_regions([region], _view_at(), **kw).rois[0]
    b = rois_from_world_regions([region], moved, **kw).rois[0]
    assert b.x == a.x - 200     # same surface, shifted by the parallax
    assert b.width == a.width


def test_world_region_behind_the_camera_is_reported():
    from atlas_camera.core.camera_crop import rois_from_world_regions

    behind = {"label": "back", "points_world": [
        (-1.0, -1.0, 10.0), (1.0, -1.0, 10.0), (1.0, 1.0, 10.0)]}
    out = rois_from_world_regions(
        [behind], _view_at(), fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
        image_width=1920, image_height=1080)
    assert out.rois == []
    assert "behind the camera" in out.dropped[0]["reason"]


def test_world_region_needs_three_corners():
    from atlas_camera.core.camera_crop import rois_from_world_regions

    out = rois_from_world_regions(
        [{"label": "thin", "points_world": [(0, 0, -5), (1, 0, -5)]}],
        _view_at(), fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
        image_width=1920, image_height=1080)
    assert out.rois == []
    assert "at least 3 world corners" in out.dropped[0]["reason"]


def test_world_region_roi_snaps_to_the_model_grid():
    from atlas_camera.core.camera_crop import rois_from_world_regions

    region = {"label": "tear", "points_world": [
        (-0.7, -0.3, -9.0), (0.9, -0.3, -9.0), (0.9, 0.5, -9.0)]}
    roi = rois_from_world_regions(
        [region], _view_at(), fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
        image_width=1920, image_height=1080, pad_frac=0.1, snap=64).rois[0]
    assert roi.width % 64 == 0 and roi.height % 64 == 0
    assert roi.x + roi.width <= 1920 and roi.y + roi.height <= 1080


def _green_excess(px):
    px = px.astype(float)
    return float((px[:, 1] - (px[:, 0] + px[:, 2]) / 2).mean())


def test_cast_neutralizer_removes_the_fill_tint():
    from atlas_camera.core.camera_crop import neutralize_fill_cast

    img = np.full((96, 96, 3), 120, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[32:64, 32:64] = True
    img[mask] = np.array([110, 145, 110], dtype=np.uint8)   # green-cast fill
    before = _green_excess(img[mask])
    fixed = neutralize_fill_cast(img, mask, band_px=8)
    after = _green_excess(fixed[mask])
    assert before > 20 and abs(after) < 2.0


def test_cast_neutralizer_keeps_the_fill_brightness():
    """The fill may legitimately be darker than its ring (road vs car) — only
    the colour balance is transferred, never the luminance."""
    from atlas_camera.core.camera_crop import neutralize_fill_cast

    img = np.full((96, 96, 3), 200, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[32:64, 32:64] = True
    img[mask] = 60
    fixed = neutralize_fill_cast(img, mask, band_px=8)
    assert abs(float(fixed[mask].mean()) - 60.0) < 1.0


def test_cast_neutralizer_reads_the_ring_from_the_reference():
    """The ring must come from the PLATE. Sampling it from the generated crop
    measures the model's cast against itself and cancels nothing."""
    from atlas_camera.core.camera_crop import neutralize_fill_cast

    mask = np.zeros((96, 96), dtype=bool)
    mask[32:64, 32:64] = True
    plate = np.full((96, 96, 3), 120, dtype=np.uint8)
    generated = np.array(plate)
    generated[...] = np.array([110, 145, 110], dtype=np.uint8)  # cast is global
    naive = neutralize_fill_cast(generated, mask, band_px=8)
    assert _green_excess(naive[mask]) > 20          # nothing to measure against
    fixed = neutralize_fill_cast(generated, mask, reference=plate, band_px=8)
    assert abs(_green_excess(fixed[mask])) < 2.0


def test_cast_neutralizer_rejects_a_reference_mismatch():
    from atlas_camera.core.camera_crop import neutralize_fill_cast

    with pytest.raises(ValueError, match="reference raster"):
        neutralize_fill_cast(np.zeros((16, 16, 3), np.uint8),
                             np.zeros((16, 16), bool),
                             reference=np.zeros((8, 8, 3), np.uint8))


def test_cast_neutralizer_declines_without_a_ring():
    from atlas_camera.core.camera_crop import neutralize_fill_cast

    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)
    assert np.array_equal(neutralize_fill_cast(img, mask), img)


def test_membrane_blend_erases_a_constant_offset_seam():
    """A fill that is the plate shifted by a constant must come back equal to
    the plate: the rim mismatch is constant, its harmonic extension is that
    constant everywhere."""
    from atlas_camera.core.camera_crop import membrane_blend

    # Photographic-plate stand-in: smooth ramp + mild texture. (Not iid noise
    # at full amplitude — boundary sampling under iid noise is irreducibly
    # ~sigma, which no real plate exhibits.)
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:48, 0:64]
    lum = (80 + yy * 1.2 + xx * 0.8 +
           rng.integers(-5, 6, size=(48, 64))).astype(np.int16)
    plate = np.stack([lum + 3, lum, lum - 3], axis=-1)
    plate = np.clip(plate, 0, 255).astype(np.uint8)
    mask = np.zeros((48, 64), dtype=bool)
    mask[12:36, 16:48] = True
    fill = plate.copy()
    fill[mask] = np.clip(plate[mask].astype(int) - 30, 0, 255).astype(np.uint8)
    out = membrane_blend(fill, plate, mask)
    assert np.abs(out[mask].astype(int) - plate[mask].astype(int)).mean() < 2.5
    assert np.array_equal(out[~mask], fill[~mask])   # only the hole corrected


def test_membrane_blend_preserves_fill_texture():
    """The correction is harmonic (smooth): the fill's own high-frequency
    content must survive — only the offset field changes."""
    from atlas_camera.core.camera_crop import membrane_blend

    rng = np.random.default_rng(4)
    plate = np.full((48, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=bool)
    mask[12:36, 16:48] = True
    fill = plate.copy()
    texture = rng.integers(80, 160, size=(int(mask.sum()), 3))
    fill[mask] = texture.astype(np.uint8)
    out = membrane_blend(fill, plate, mask)
    # interior second differences (texture) preserved within rounding
    inner = np.zeros_like(mask)
    inner[14:34, 18:46] = True
    d_fill = np.diff(fill[14:34, 18:46, 0].astype(int), axis=1)
    d_out = np.diff(out[14:34, 18:46, 0].astype(int), axis=1)
    assert np.abs(d_fill - d_out).mean() < 2.0


def test_membrane_blend_handles_frame_edge_holes_and_empty_masks():
    from atlas_camera.core.camera_crop import membrane_blend

    plate = np.full((32, 32, 3), 100, dtype=np.uint8)
    fill = plate.copy()
    edge = np.zeros((32, 32), dtype=bool)
    edge[0:8, 0:32] = True                     # touches three frame edges
    fill[edge] = 40
    out = membrane_blend(fill, plate, edge)
    assert out.shape == plate.shape
    assert np.abs(out[edge].astype(int) - 100).mean() < 2.0
    empty = membrane_blend(fill, plate, np.zeros((32, 32), bool))
    assert np.array_equal(empty, fill)


def test_membrane_blend_rejects_mismatched_rasters():
    from atlas_camera.core.camera_crop import membrane_blend

    with pytest.raises(ValueError, match="matching HxWx3"):
        membrane_blend(np.zeros((8, 8, 3), np.uint8),
                       np.zeros((4, 4, 3), np.uint8),
                       np.zeros((8, 8), bool))


def test_composite_float_plate_keeps_dtype():
    base = np.zeros((16, 16, 3), dtype=np.float32)
    roi = RegionROI(x=0, y=0, width=16, height=16)
    out = composite_crops(base, [np.full((16, 16, 3), 0.5, np.float32)], [roi])
    assert out.dtype == np.float32
    assert out.max() == pytest.approx(0.5)
