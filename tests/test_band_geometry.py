"""Tests for the host-agnostic band geometry maths
(`atlas_camera/core/band_geometry.py`), extracted out of
`AtlasCleanPlateLayer` so the card/ground/membership/overhang arithmetic is
unit-testable without constructing a ComfyUI node.

Analytic camera fixture, same construction as tests/test_relief_mesh.py: level
camera at (0, h, 0), identity rotation, ground plane at Y=0. With the
unnormalized camera ray ((u-cx)/fx, -(v-cy)/fy, -1) the ray parameter IS
forward depth, so the expected ground field is exactly ``h * fy / (v - cy)``
for rows below the principal point and NaN at/above it.

Numpy only — no torch, no ComfyUI.
"""

import numpy as np
import pytest

from atlas_camera.core.band_geometry import (
    GROUND_CAP_FACTOR,
    GROUND_CAP_PERCENTILE,
    band_membership,
    boundary_overhang_cells,
    card_plane_depth,
    flat_band_depth_field,
    ground_cap_metres,
    ground_plane_depth_field,
)

W = H = 64
FX = FY = 100.0
CX = CY = 32.0
CAM_HEIGHT = 2.0
INF = float("inf")


class _Extrinsics:
    """Minimal stand-in for AtlasExtrinsics (only camera_view_matrix is read)."""

    def __init__(self, view_matrix):
        self.camera_view_matrix = view_matrix


def _view_matrix(h=CAM_HEIGHT, pitch_deg=0.0):
    """world->cam for a camera at (0, h, 0) with an optional pitch about X."""
    t = np.radians(pitch_deg)
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(t), -np.sin(t)],
        [0.0, np.sin(t), np.cos(t)],
    ])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[1, 3] = h
    return np.linalg.inv(c2w).tolist()


def _extr(h=CAM_HEIGHT, pitch_deg=0.0):
    return _Extrinsics(_view_matrix(h, pitch_deg))


# --------------------------------------------------------------------------
# band membership set algebra
# --------------------------------------------------------------------------


def test_band_membership_is_closed_near_and_closed_far():
    metric = np.array([[1.0, 5.0, 10.0, 20.0]])
    valid = np.ones_like(metric, dtype=bool)
    region = band_membership(valid, metric, 5.0, 10.0)
    # near is >=, far is <= — both edges inclusive.
    assert region.tolist() == [[False, True, True, False]]


def test_band_membership_open_far_edge_keeps_everything_beyond_near():
    metric = np.array([[1.0, 5.0, 1e6]])
    valid = np.ones_like(metric, dtype=bool)
    region = band_membership(valid, metric, 5.0, INF)
    assert region.tolist() == [[False, True, True]]


def test_band_membership_excludes_invalid_pixels_even_inside_the_band():
    metric = np.array([[5.0, 6.0, 7.0]])
    valid = np.array([[True, False, True]])
    region = band_membership(valid, metric, 0.0, 10.0)
    assert region.tolist() == [[True, False, True]]


def test_band_membership_unions_the_occluder_fill_footprint():
    metric = np.array([[1.0, 6.0, 20.0]])
    valid = np.ones_like(metric, dtype=bool)
    fill = np.array([[True, False, False]])  # nearer-than-band occluder
    region = band_membership(valid, metric, 5.0, 10.0, fill_mask=fill)
    assert region.tolist() == [[True, True, False]]


def test_band_membership_fill_is_masked_by_an_explicit_exclusion():
    metric = np.array([[1.0, 1.0, 6.0]])
    valid = np.ones_like(metric, dtype=bool)
    fill = np.array([[True, True, False]])
    exclude = np.array([[True, False, False]])  # sky: never filled
    region = band_membership(valid, metric, 5.0, 10.0,
                             fill_mask=fill, exclude_mask=exclude)
    assert region.tolist() == [[False, True, True]]


# --------------------------------------------------------------------------
# card plane
# --------------------------------------------------------------------------


def test_card_plane_depth_is_the_median_of_the_in_band_raw_depth():
    depth = np.array([[1.0, 10.0, 20.0, 30.0, 1000.0]])
    region = np.array([[False, True, True, True, False]])
    assert card_plane_depth(depth, region) == pytest.approx(20.0)


def test_card_plane_depth_ignores_out_of_band_outliers():
    depth = np.array([[7.0, 7.0, 7.0, 5000.0]])
    region = np.array([[True, True, True, False]])
    assert card_plane_depth(depth, region) == pytest.approx(7.0)


def test_card_plane_depth_falls_back_to_one_on_an_empty_band():
    depth = np.array([[3.0, 4.0]])
    region = np.zeros((1, 2), dtype=bool)
    assert card_plane_depth(depth, region) == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason=(
    "PRE-EXISTING BUG (found during the 2026-08-01 extraction, behaviour "
    "deliberately preserved): with fill_occluded on, the occluder footprint is "
    "unioned into the band region BEFORE the card's median is taken, so the "
    "occluder's much-nearer RAW depths join the population and drag the card "
    "plane forward — here a [50, 100] m band whose real members sit at 60/70/80 m "
    "collapses onto the 5 m occluder. The median should come from the real "
    "in-band members only; the fill union exists to give the card its EXTENT, "
    "not to vote on WHERE the plane sits."))
def test_card_plane_median_should_ignore_the_occluder_fill_footprint():
    metric = np.array([[5.0, 5.0, 5.0, 5.0, 60.0, 70.0, 80.0]])
    valid = np.ones_like(metric, dtype=bool)
    fill = metric < 50.0
    region = band_membership(valid, metric, 50.0, 100.0, fill_mask=fill)
    assert card_plane_depth(metric, region) == pytest.approx(70.0)


def test_card_plane_median_currently_is_dragged_onto_the_occluder():
    """Characterisation of the bug above — pins today's actual behaviour so a
    future fix has to change this test deliberately."""
    metric = np.array([[5.0, 5.0, 5.0, 5.0, 60.0, 70.0, 80.0]])
    valid = np.ones_like(metric, dtype=bool)
    fill = metric < 50.0
    region = band_membership(valid, metric, 50.0, 100.0, fill_mask=fill)
    assert card_plane_depth(metric, region) == pytest.approx(5.0)


def test_flat_band_depth_field_card_is_constant_inside_and_nan_outside():
    depth = np.array([[2.0, 4.0, 6.0, 900.0]])
    metric = depth.copy()
    valid = np.ones_like(depth, dtype=bool)
    field = flat_band_depth_field(
        "card", depth=depth, metric=metric, valid=valid, near=2.0, far=6.0,
        scale=1.0, extr=_extr(), fx=FX, fy=FY, cx=CX, cy=CY, height=1, width=4)
    assert np.isnan(field[0, 3])
    inside = field[0, :3]
    assert np.allclose(inside, 4.0)          # median of {2, 4, 6}
    assert float(np.nanstd(field)) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# analytic Y=0 ground plane
# --------------------------------------------------------------------------


def test_ground_plane_depth_field_matches_the_closed_form_ray_plane_solution():
    field = ground_plane_depth_field(_extr(), FX, FY, CX, CY, H, W)
    for row in (33, 40, 48, 63):
        expected = CAM_HEIGHT * FY / (row - CY)
        assert field[row, 0] == pytest.approx(expected)
        assert field[row, W - 1] == pytest.approx(expected)


def test_ground_plane_depth_grows_toward_the_horizon_row():
    field = ground_plane_depth_field(_extr(), FX, FY, CX, CY, H, W)
    column = field[33:, 0]
    assert np.all(np.isfinite(column))
    # Rows nearer the horizon (smaller v) are FARTHER ground.
    assert np.all(np.diff(column) < 0.0)
    assert column[0] == pytest.approx(CAM_HEIGHT * FY / 1.0)


def test_ground_plane_depth_is_nan_at_and_above_the_horizon():
    field = ground_plane_depth_field(_extr(), FX, FY, CX, CY, H, W)
    assert np.all(np.isnan(field[: int(CY) + 1, :]))


def test_ground_plane_depth_scales_linearly_with_camera_height():
    low = ground_plane_depth_field(_extr(1.0), FX, FY, CX, CY, H, W)
    high = ground_plane_depth_field(_extr(4.0), FX, FY, CX, CY, H, W)
    finite = np.isfinite(low)
    assert finite.any()
    assert np.allclose(high[finite] / low[finite], 4.0)


def test_ground_plane_depth_rejects_a_camera_on_or_below_the_ground():
    with pytest.raises(ValueError, match="camera above the ground plane"):
        ground_plane_depth_field(_extr(0.0), FX, FY, CX, CY, H, W)


# --------------------------------------------------------------------------
# the ground cap heuristic
# --------------------------------------------------------------------------


def test_ground_cap_is_the_bands_far_edge_when_the_band_is_bounded():
    metric = np.array([[1.0, 2.0, 3.0]])
    region = np.ones_like(metric, dtype=bool)
    assert ground_cap_metres(metric, region, 42.0) == pytest.approx(42.0)


def test_ground_cap_on_an_open_band_is_the_documented_percentile_times_factor():
    metric = np.arange(1.0, 101.0).reshape(1, 100)
    region = np.ones_like(metric, dtype=bool)
    expected = GROUND_CAP_FACTOR * float(np.percentile(metric, GROUND_CAP_PERCENTILE))
    assert GROUND_CAP_PERCENTILE == pytest.approx(99.0)
    assert GROUND_CAP_FACTOR == pytest.approx(4.0)
    assert ground_cap_metres(metric, region, INF) == pytest.approx(expected)
    assert ground_cap_metres(metric, region, INF) == pytest.approx(4.0 * 99.01)


def test_ground_cap_on_an_open_band_with_no_members_is_unbounded():
    metric = np.array([[1.0, 2.0]])
    region = np.zeros((1, 2), dtype=bool)
    assert ground_cap_metres(metric, region, INF) == INF


def test_ground_cap_actually_clamps_runaway_near_horizon_ground_depths():
    """A wall-base pixel just under the horizon has an analytic ground depth
    that runs out toward infinity; the cap must drop it from the band."""
    valid = np.ones((H, W), dtype=bool)
    # Real metric depth: a flat 10 m band everywhere (so far edge = 12 m).
    metric = np.full((H, W), 10.0)
    field = flat_band_depth_field(
        "ground", depth=metric.copy(), metric=metric, valid=valid,
        near=0.0, far=12.0, scale=1.0, extr=_extr(), fx=FX, fy=FY,
        cx=CX, cy=CY, height=H, width=W)
    analytic = ground_plane_depth_field(_extr(), FX, FY, CX, CY, H, W)
    kept = np.isfinite(field)
    assert kept.any()
    # Everything kept is on-plane AND under the 12 m cap.
    assert np.all(analytic[kept] <= 12.0)
    assert float(np.nanmax(field)) <= 12.0
    # The rows just below the horizon (analytic depth 200 m, 100 m, ...) are gone.
    assert not kept[33].any()
    assert not kept[40].any()
    # Rows far enough down (analytic depth <= 12 m => v - cy >= 200/12) survive.
    assert kept[49].all()


def test_flat_band_ground_field_is_metric_over_scale():
    valid = np.ones((H, W), dtype=bool)
    metric = np.full((H, W), 10.0)
    scale = 4.0
    field = flat_band_depth_field(
        "ground", depth=metric.copy(), metric=metric, valid=valid,
        near=0.0, far=12.0, scale=scale, extr=_extr(), fx=FX, fy=FY,
        cx=CX, cy=CY, height=H, width=W)
    analytic = ground_plane_depth_field(_extr(), FX, FY, CX, CY, H, W)
    kept = np.isfinite(field)
    assert kept.any()
    assert np.allclose(field[kept], analytic[kept] / scale)


def test_flat_band_ground_respects_band_membership_from_the_real_depth():
    """Pixels outside the REAL depth band drop out even though the analytic
    ground plane is defined there."""
    valid = np.ones((H, W), dtype=bool)
    metric = np.full((H, W), 10.0)
    metric[:, : W // 2] = 100.0          # left half is out of the [0, 12] band
    field = flat_band_depth_field(
        "ground", depth=metric.copy(), metric=metric, valid=valid,
        near=0.0, far=12.0, scale=1.0, extr=_extr(), fx=FX, fy=FY,
        cx=CX, cy=CY, height=H, width=W)
    assert np.all(np.isnan(field[:, : W // 2]))
    assert np.isfinite(field[:, W // 2:]).any()


def test_flat_band_relief_geometry_is_a_pass_through():
    depth = np.array([[1.0, 2.0, 3.0]])
    metric = depth.copy()
    valid = np.ones_like(depth, dtype=bool)
    field = flat_band_depth_field(
        "relief", depth=depth, metric=metric, valid=valid, near=0.0, far=INF,
        scale=1.0, extr=_extr(), fx=FX, fy=FY, cx=CX, cy=CY, height=1, width=3)
    assert field is depth


# --------------------------------------------------------------------------
# overhang-cell arithmetic
# --------------------------------------------------------------------------


def test_no_overhang_without_an_embedded_matte():
    assert boundary_overhang_cells(embed_matte=False, edge_extend_px=64,
                                   relief_grid=384, height=1024, width=1024,
                                   choke_cells=2) == 0


def test_matte_alone_gets_the_two_cell_base_overhang():
    assert boundary_overhang_cells(embed_matte=True, edge_extend_px=0,
                                   relief_grid=384, height=1024, width=1024,
                                   choke_cells=0) == 2


def test_edge_extend_adds_ceil_of_extension_over_cell_size():
    # cell_px = round(1024 / 384) = 3  ->  2 + ceil(64 / 3) = 2 + 22 = 24
    assert boundary_overhang_cells(embed_matte=True, edge_extend_px=64,
                                   relief_grid=384, height=1024, width=1024,
                                   choke_cells=0) == 24


def test_edge_extend_uses_the_long_edge_for_the_cell_size():
    # cell_px = round(2048 / 256) = 8  ->  2 + ceil(64 / 8) = 2 + 8 = 10
    assert boundary_overhang_cells(embed_matte=True, edge_extend_px=64,
                                   relief_grid=256, height=512, width=2048,
                                   choke_cells=0) == 10


def test_choke_is_added_on_top_so_the_skirt_regrows_before_extending():
    assert boundary_overhang_cells(embed_matte=True, edge_extend_px=64,
                                   relief_grid=256, height=512, width=2048,
                                   choke_cells=3) == 13
    assert boundary_overhang_cells(embed_matte=True, edge_extend_px=0,
                                   relief_grid=256, height=512, width=2048,
                                   choke_cells=3) == 5


def test_cell_size_never_collapses_to_zero_on_a_huge_grid():
    # round(64 / 4096) == 0 -> clamped to 1 px, so the extension is 1 cell/px.
    assert boundary_overhang_cells(embed_matte=True, edge_extend_px=5,
                                   relief_grid=4096, height=64, width=64,
                                   choke_cells=0) == 7
