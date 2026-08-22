"""Contracts for the torch-free half of splat-fused hole repair.

Everything here runs on synthetic arrays: no scene, no GPU, no ComfyUI. The
load-bearing test is the reprojection round-trip — the camera convention is the
thing most likely to be silently wrong, and a seed placed through a mirrored or
transposed matrix still looks like a plausible cloud.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.hole_splat import (
    DEFAULT_OVERLAP_PX,
    SPLAT_SOURCE,
    HoleOwnership,
    RimDepth,
    hole_ownership,
    rim_depth_interval,
    seed_hole_volume,
    split_visible_occluded,
)

H, W = 48, 64
FX = FY = 80.0
CX, CY = W / 2.0, H / 2.0

# Camera at the origin looking down -Z, Y up: the identity view matrix in
# Atlas's convention (depth_geometry.back_project_normals).
VIEW = np.eye(4, dtype=np.float64)


def _hole(y0=16, y1=32, x0=20, x1=40):
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _coverage_around(hole):
    """Everything except the hole is covered — the ordinary case."""
    return ~hole


def _zbuffer(hole, *, near=4.0, far=11.0):
    """Near lip on the left of the hole, far surface on the right."""
    z = np.full((H, W), np.inf, dtype=np.float64)
    covered = ~hole
    xs = np.arange(W)[None, :].repeat(H, axis=0)
    ramp = near + (far - near) * (xs / max(W - 1, 1))
    z[covered] = ramp[covered]
    return z


# ---------------------------------------------------------------- ownership


def test_the_mesh_keeps_every_pixel_it_measured():
    hole = _hole()
    coverage = _coverage_around(hole)
    own = hole_ownership(hole, coverage)

    assert isinstance(own, HoleOwnership)
    # The rule the seam depends on: splats never claim a measured pixel.
    assert not (own.splat_mask & own.mesh_mask).any()
    assert own.mesh_mask.sum() == coverage.sum()
    assert own.splat_mask.sum() == hole.sum()


def test_the_overlap_band_grows_into_covered_ground_only():
    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole), overlap_px=4)

    assert own.overlap_mask.any()
    # Blending happens on the mesh's side; the band is never part of the hole.
    assert not (own.overlap_mask & own.splat_mask).any()
    # Subset, not coverage of the whole frame: overlap ⊆ mesh.
    assert not (own.overlap_mask & ~own.mesh_mask).any()
    assert own.report["overlap_radius_px"] == 4


def test_a_wider_band_is_a_superset_of_a_narrower_one():
    hole = _hole()
    narrow = hole_ownership(hole, _coverage_around(hole), overlap_px=2)
    wide = hole_ownership(hole, _coverage_around(hole), overlap_px=6)

    assert wide.overlap_mask.sum() > narrow.overlap_mask.sum()
    assert (narrow.overlap_mask & ~wide.overlap_mask).sum() == 0


def test_sky_is_never_handed_to_the_splats():
    """Sky holes are not occlusion holes — the survey path refuses them too."""

    hole = _hole()
    sky = np.zeros((H, W), dtype=bool)
    sky[16:24, 20:40] = True  # top half of the hole is sky

    own = hole_ownership(hole, _coverage_around(hole), sky_mask=sky)

    assert not (own.splat_mask & sky).any()
    assert own.report["sky_px_dropped"] == int((hole & sky).sum())
    assert own.splat_mask.sum() == int((hole & ~sky).sum())


def test_mismatched_masks_are_refused():
    with pytest.raises(ValueError, match="must match"):
        hole_ownership(_hole(), np.zeros((H + 1, W), dtype=bool))


# ------------------------------------------------------------------- rim


def test_the_rim_interval_comes_from_measured_pixels_only():
    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    rim = rim_depth_interval(_zbuffer(hole), own.splat_mask)

    assert isinstance(rim, RimDepth)
    assert 4.0 <= rim.near_m < rim.far_m <= 11.0
    assert rim.n_ring_px > 0


def test_one_stray_far_pixel_does_not_stretch_the_slab():
    """Percentiles, not min/max: a backdrop pixel in the ring would otherwise
    put the whole seed budget where nothing is."""

    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    z = _zbuffer(hole)
    honest = rim_depth_interval(z, own.splat_mask)

    z[15, 30] = 5000.0  # a slice of far backdrop bleeding into the rim
    contaminated = rim_depth_interval(z, own.splat_mask)

    assert contaminated.far_m < 100.0
    assert contaminated.far_m == pytest.approx(honest.far_m, rel=0.25)


def test_a_hole_with_no_measured_rim_is_refused():
    hole = np.ones((H, W), dtype=bool)  # everything is hole; nothing measured
    z = np.full((H, W), np.inf)
    with pytest.raises(ValueError, match="rim"):
        rim_depth_interval(z, hole)


# ------------------------------------------------------------------ seed


def _seed(**kwargs):
    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    rim = rim_depth_interval(_zbuffer(hole), own.splat_mask)
    params = dict(view_matrix=VIEW, fx=FX, fy=FY, cx=CX, cy=CY)
    params.update(kwargs)
    return own, rim, seed_hole_volume(own.splat_mask, rim, **params)


def test_every_seeded_point_lands_inside_the_measured_interval():
    """The slab bound is the whole point: an unmeasured seed is not a guess."""

    _own, rim, seed = _seed(layers=5, pixel_stride=3)

    depth = -seed.points_world[:, 2]  # camera at origin looking down -Z
    assert depth.min() >= rim.near_m - 1e-9
    assert depth.max() <= rim.far_m + 1e-9
    assert seed.count == seed.report["n_rays"] * seed.report["layers"]
    assert seed.report["source"] == SPLAT_SOURCE


def test_seeded_points_reproject_onto_the_pixels_that_spawned_them():
    """The convention check. A mirrored or transposed matrix still produces a
    plausible-looking cloud, so verify the projection closes the loop."""

    own, _rim, seed = _seed(layers=3, pixel_stride=5)

    pts = seed.points_world
    depth = -pts[:, 2]
    u = pts[:, 0] / depth * FX + CX
    v = -pts[:, 1] / depth * FY + CY

    rows = np.rint(v).astype(int)
    cols = np.rint(u).astype(int)
    assert (rows >= 0).all() and (rows < H).all()
    assert (cols >= 0).all() and (cols < W).all()
    # Every reprojected sample must land back inside the hole it was seeded for.
    assert own.splat_mask[rows, cols].all()


def test_the_seed_is_deterministic_for_a_given_seed_value():
    _o1, _r1, a = _seed(seed=7, pixel_stride=4)
    _o2, _r2, b = _seed(seed=7, pixel_stride=4)
    _o3, _r3, c = _seed(seed=8, pixel_stride=4)

    assert np.array_equal(a.points_world, b.points_world)
    assert not np.array_equal(a.points_world, c.points_world)


def test_stratification_spans_the_slab_rather_than_clumping():
    _own, rim, seed = _seed(layers=8, pixel_stride=4)

    depth = -seed.points_world[:, 2]
    lo, hi = rim.near_m, rim.far_m
    # Each eighth of the slab must actually receive samples.
    edges = np.linspace(lo, hi, 9)
    counts = np.histogram(depth, bins=edges)[0]
    assert (counts > 0).all()


def test_colours_are_carried_per_ray_when_supplied():
    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    rim = rim_depth_interval(_zbuffer(hole), own.splat_mask)
    colors = np.zeros((H, W, 3), dtype=np.float32)
    colors[..., 0] = 1.0

    seed = seed_hole_volume(
        own.splat_mask, rim, view_matrix=VIEW, fx=FX, fy=FY, cx=CX, cy=CY,
        layers=2, pixel_stride=6, colors=colors,
    )
    assert seed.colors.shape == (seed.count, 3)
    assert np.allclose(seed.colors[:, 0], 1.0)
    assert np.allclose(seed.colors[:, 1], 0.0)


def test_a_three_by_three_matrix_is_refused():
    """core builds world math from the full 4x4 only — never the 3x3."""

    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    rim = rim_depth_interval(_zbuffer(hole), own.splat_mask)
    with pytest.raises(ValueError, match="4x4"):
        seed_hole_volume(own.splat_mask, rim, view_matrix=np.eye(3),
                         fx=FX, fy=FY, cx=CX, cy=CY)


def test_an_empty_hole_is_refused():
    rim = RimDepth(near_m=1.0, far_m=2.0, n_ring_px=64)
    with pytest.raises(ValueError, match="empty"):
        seed_hole_volume(np.zeros((H, W), dtype=bool), rim, view_matrix=VIEW,
                         fx=FX, fy=FY, cx=CX, cy=CY)


# ---------------------------------------------------------------- metrics


def test_the_error_split_separates_what_was_shown_from_what_was_not():
    """A whole-frame average lets a fill score well by reproducing what it was
    already shown — OCCLUDED is the number that turns the experiment."""

    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    error = np.zeros((H, W), dtype=np.float64)
    error[own.splat_mask] = 0.4
    error[own.mesh_mask & ~own.splat_mask] = 0.01

    split = split_visible_occluded(error, own.splat_mask, own.mesh_mask)

    assert split["occluded"]["mean"] == pytest.approx(0.4)
    assert split["visible"]["mean"] == pytest.approx(0.01)
    # The trap, made visible: the frame average hides the occluded failure.
    assert split["whole_frame"]["mean"] < split["occluded"]["mean"] / 2
    assert split["occluded"]["n_px"] == int(own.splat_mask.sum())


def test_the_split_accepts_a_colour_error_image():
    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    error = np.zeros((H, W, 3), dtype=np.float64)
    error[own.splat_mask] = 0.3

    split = split_visible_occluded(error, own.splat_mask, own.mesh_mask)
    assert split["occluded"]["mean"] == pytest.approx(0.3)


def test_default_overlap_is_the_documented_constant():
    hole = _hole()
    own = hole_ownership(hole, _coverage_around(hole))
    assert own.report["overlap_radius_px"] == DEFAULT_OVERLAP_PX
