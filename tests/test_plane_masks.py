"""Contracts for per-plane alphas: which pixels does a fitted plane explain?

WHY THIS EXISTS. `extract_planes_ransac` computes a per-plane inlier mask
(`inl_local`) and keeps only its COUNT, so a released `ransac_planes.json`
describes eight planes and cannot say which pixels belong to any of them. A
plane rebuilt from that record is a bare quad, and the plate projects onto the
whole quad — which is exactly the smear seen when orbiting the roundtripped
DSC_2552 scene: a handful of flat planes each painted with everything behind
them.

The load-bearing property is EXCLUSIVITY. Two planes that both claim a pixel
put the same photograph on two surfaces at different depths, and the orbit
shows it as a doubled, sliding ghost. So assignment is nearest-wins and every
test here checks the split, not just the coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.plane_masks import (
    PlaneFrame,
    find_duplicate_planes,
    assign_points_to_planes,
    plane_frames_from_primitives,
    plane_pixel_masks,
)


class _Prim:
    """Stand-in for AtlasProxyPrimitive — the module reads it structurally."""

    def __init__(self, name, transform_matrix, dimensions, metadata=None):
        self.name = name
        self.primitive_type = "plane"
        self.transform_matrix = transform_matrix
        self.dimensions = dimensions
        self.metadata = metadata or {}


def _wall(z, *, w=4.0, h=3.0, x=0.0, y=0.0, name="wall"):
    """A fronto-parallel plane facing +Z at world z, centred on (x, y)."""
    u = (1.0, 0.0, 0.0)
    v = (0.0, 1.0, 0.0)
    n = (0.0, 0.0, 1.0)
    c = (x, y, z)
    transform = tuple(
        (u[i], v[i], n[i], c[i]) for i in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    return _Prim(name, transform, (w, h, 0.0))


def _points(zs, xs=None, ys=None):
    zs = np.asarray(zs, dtype=np.float64)
    xs = np.zeros_like(zs) if xs is None else np.asarray(xs, dtype=np.float64)
    ys = np.zeros_like(zs) if ys is None else np.asarray(ys, dtype=np.float64)
    return np.stack([xs, ys, zs], axis=-1)


CAM = (0.0, 0.0, 0.0)


# ------------------------------------------------------------------- frames


def test_a_plane_frame_is_recovered_from_the_transform_columns():
    """Columns are (u, v, n, c) — `depth_geometry.plane_transform`. Reading
    them as ROWS gives a plausible-looking basis that is silently transposed."""

    frames = plane_frames_from_primitives([_wall(-10.0, w=6.0, h=2.0, x=1.0)])

    assert len(frames) == 1
    f = frames[0]
    assert isinstance(f, PlaneFrame)
    assert np.allclose(f.normal, [0.0, 0.0, 1.0])
    assert np.allclose(f.u, [1.0, 0.0, 0.0])
    assert np.allclose(f.centre, [1.0, 0.0, -10.0])
    assert f.width == pytest.approx(6.0)
    assert f.height == pytest.approx(2.0)


def test_non_plane_primitives_are_skipped():
    box = _Prim("box", np.eye(4).tolist(), (1.0, 1.0, 1.0))
    box.primitive_type = "box"
    assert plane_frames_from_primitives([box, _wall(-5.0)]) != []
    assert len(plane_frames_from_primitives([box, _wall(-5.0)])) == 1


# --------------------------------------------------------------- assignment


def test_a_point_on_a_plane_is_claimed_by_it():
    frames = plane_frames_from_primitives([_wall(-10.0)])
    labels, report = assign_points_to_planes(
        _points([-10.0]), frames, camera_position=CAM)

    assert labels.tolist() == [0]
    assert report["assigned_px"] == 1


def test_a_point_far_off_the_plane_is_claimed_by_nobody():
    """Unassigned is -1, never "nearest anyway". A plane that swallows every
    pixel in the frame IS the smear."""

    frames = plane_frames_from_primitives([_wall(-10.0)])
    labels, _ = assign_points_to_planes(
        _points([-4.0]), frames, camera_position=CAM)

    assert labels.tolist() == [-1]


def test_a_point_beyond_the_quads_edge_is_not_claimed():
    """The rectangle is part of the plane's identity. Without the extent test a
    wall claims everything coplanar with it clear across the frame — the
    "crop to the object" failure."""

    frames = plane_frames_from_primitives([_wall(-10.0, w=4.0, h=3.0)])
    inside = _points([-10.0], xs=[1.0])
    outside = _points([-10.0], xs=[9.0])

    assert assign_points_to_planes(inside, frames, camera_position=CAM)[0].tolist() == [0]
    assert assign_points_to_planes(outside, frames, camera_position=CAM)[0].tolist() == [-1]


def test_two_planes_never_claim_the_same_point():
    """Exclusivity. Both planes are within tolerance of the midpoint; exactly
    one may have it, or the plate lands on two surfaces at once."""

    frames = plane_frames_from_primitives([_wall(-10.0, name="a"),
                                           _wall(-10.2, name="b")])
    labels, report = assign_points_to_planes(
        _points([-10.05]), frames, camera_position=CAM)

    assert labels.tolist() == [0]          # nearest wins
    assert report["contested_px"] == 1     # and the contest is reported


def test_an_exact_tie_goes_to_the_plane_nearer_the_camera():
    """A tie broken by list order lets extraction order decide occlusion."""

    frames = plane_frames_from_primitives([_wall(-12.0, name="far"),
                                           _wall(-8.0, name="near")])
    labels, _ = assign_points_to_planes(
        _points([-10.0]), frames, camera_position=CAM, tolerance_m=5.0)

    assert labels.tolist() == [1]          # index of the near plane


def test_the_tolerance_follows_the_planes_own_depth():
    """Extraction used `max(0.15, 0.02 * median_depth)`; a fixed metric
    tolerance is far too tight on a 60 m backdrop and far too loose at 2 m.
    Same rule here rather than a second invented constant."""

    near = plane_frames_from_primitives([_wall(-2.0)])
    far = plane_frames_from_primitives([_wall(-60.0)])

    off_by = 0.5
    assert assign_points_to_planes(_points([-2.0 - off_by]), near,
                                   camera_position=CAM)[0].tolist() == [-1]
    assert assign_points_to_planes(_points([-60.0 - off_by]), far,
                                   camera_position=CAM)[0].tolist() == [0]


def test_invalid_points_are_never_assigned():
    frames = plane_frames_from_primitives([_wall(-10.0)])
    pts = _points([-10.0, -10.0])
    valid = np.array([True, False])

    labels, report = assign_points_to_planes(
        pts, frames, camera_position=CAM, valid=valid)

    assert labels.tolist() == [0, -1]
    assert report["invalid_px"] == 1


def test_nan_points_are_treated_as_invalid_without_being_told():
    frames = plane_frames_from_primitives([_wall(-10.0)])
    pts = _points([-10.0, np.nan])

    labels, _ = assign_points_to_planes(pts, frames, camera_position=CAM)
    assert labels.tolist() == [0, -1]


def test_no_planes_assigns_nothing_rather_than_raising():
    labels, report = assign_points_to_planes(
        _points([-10.0]), [], camera_position=CAM)
    assert labels.tolist() == [-1]
    assert report["planes"] == 0


# -------------------------------------------------------------- pixel masks


def _grid_scene():
    """A 4x4 raster: left half on a near wall, right half on a far one."""
    zs = np.full((4, 4), -10.0)
    zs[:, 2:] = -20.0
    xs = np.zeros((4, 4))
    xs[:, 2:] = 3.0
    pts = np.stack([xs, np.zeros((4, 4)), zs], axis=-1)
    frames = plane_frames_from_primitives([
        _wall(-10.0, w=4.0, h=8.0, x=0.0, name="near"),
        _wall(-20.0, w=4.0, h=8.0, x=3.0, name="far"),
    ])
    return pts, frames


def test_every_plane_gets_a_mask_in_the_raster_shape():
    pts, frames = _grid_scene()
    masks, report = plane_pixel_masks(pts, frames, camera_position=CAM)

    assert len(masks) == 2
    assert all(m.shape == (4, 4) for m in masks)
    assert masks[0].sum() == 8
    assert masks[1].sum() == 8
    assert report["unassigned_px"] == 0


def test_the_masks_partition_the_raster_and_never_overlap():
    pts, frames = _grid_scene()
    masks, _ = plane_pixel_masks(pts, frames, camera_position=CAM)

    stack = np.stack(masks).astype(np.int32).sum(axis=0)
    assert stack.max() <= 1


def test_a_plane_that_explains_nothing_reports_an_empty_mask(caplog):
    """An empty mask is a finding, not a crash: the plane was fitted on a
    different depth map than the one being assigned."""

    pts, frames = _grid_scene()
    frames = frames + plane_frames_from_primitives([_wall(-90.0, name="ghost")])
    masks, report = plane_pixel_masks(pts, frames, camera_position=CAM)

    assert masks[2].sum() == 0
    assert "ghost" in report["empty_planes"]


def test_a_points_array_that_is_not_a_raster_is_refused():
    frames = plane_frames_from_primitives([_wall(-10.0)])
    with pytest.raises(ValueError, match="HxWx3"):
        plane_pixel_masks(np.zeros((4, 3)), frames, camera_position=CAM)


def test_the_report_says_how_much_of_the_frame_was_explained():
    pts, frames = _grid_scene()
    frames = frames[:1]
    _masks, report = plane_pixel_masks(pts, frames, camera_position=CAM)

    assert report["assigned_px"] == 8
    assert report["unassigned_px"] == 8
    assert report["assigned_fraction"] == pytest.approx(0.5)


def test_a_tilted_plane_recovers_the_basis_that_PRODUCED_it():
    """Mutation-found blind spot: every fixture above uses an axis-aligned
    basis, where reading the transform's axes as ROWS instead of COLUMNS is
    invisible. Build the record with the real producer and tilt it, so a
    transposed read lands the axes somewhere else."""

    from atlas_camera.core.depth_geometry import arbitrary_plane_axes, plane_transform

    normal = np.array([0.4, 0.6, -0.7])
    normal = normal / np.linalg.norm(normal)
    u, v, n = arbitrary_plane_axes(np, normal)
    centre = np.array([1.5, -0.5, -9.0])
    prim = _Prim("tilted", plane_transform(u, v, n, centre), (5.0, 4.0, 0.0))

    frame = plane_frames_from_primitives([prim])[0]

    assert np.allclose(frame.normal, n)
    assert np.allclose(frame.u, u)
    assert np.allclose(frame.v, v)
    assert np.allclose(frame.centre, centre)

    # A point pushed 2 m along the plane's own u axis is inside the 5 m quad
    # and exactly on the plane, so the real basis claims it.
    on_plane = (centre + 2.0 * u).reshape(1, 3)
    labels, _ = assign_points_to_planes(on_plane, [frame], camera_position=CAM)
    assert labels.tolist() == [0]

    # ...while a point 2 m along the NORMAL is off it, whatever the extent says.
    off_plane = (centre + 2.0 * n).reshape(1, 3)
    labels, _ = assign_points_to_planes(off_plane, [frame], camera_position=CAM)
    assert labels.tolist() == [-1]


def test_coverage_is_reported_against_valid_pixels_as_well_as_the_frame():
    """Half of DSC_2552 has no valid depth, so "43.6% assigned" reads as a
    failure when it is really 78.6% of everything that could be assigned."""

    pts, frames = _grid_scene()
    pts = pts.copy()
    pts[0, :, :] = np.nan          # a quarter of the raster has no depth

    _masks, report = plane_pixel_masks(pts, frames, camera_position=CAM)

    assert report["invalid_px"] == 4
    assert report["assigned_px"] == 12
    assert report["assigned_fraction"] == pytest.approx(12 / 16)
    assert report["assigned_fraction_of_valid"] == pytest.approx(1.0)


# ------------------------------------------------------------- duplicates


def test_two_coplanar_planes_are_reported_as_duplicates():
    """The DSC_2552 pair: `projection_plane_01` was the ground refitted, 0.9
    degrees off and 0.247 m away, and it carried 268,633 of the scene's
    299,794 contested pixels."""

    frames = plane_frames_from_primitives([
        _wall(-10.0, name="ground"), _wall(-10.25, name="refit")])

    dupes = find_duplicate_planes(frames, camera_position=CAM)

    assert len(dupes) == 1
    assert dupes[0]["name"] == "refit"
    assert dupes[0]["duplicate_of"] == "ground"
    assert dupes[0]["offset_m"] == pytest.approx(0.25, abs=0.01)


def test_the_earlier_plane_is_the_one_kept():
    """Extraction emits in descending inlier count, so the first of a pair is
    the better-supported fit. Keeping the later one would swap a 172,882-inlier
    ground for its 6,000-inlier shadow."""

    frames = plane_frames_from_primitives([
        _wall(-10.0, name="first"), _wall(-10.2, name="second")])
    assert find_duplicate_planes(frames, camera_position=CAM)[0]["name"] == "second"


def test_planes_far_apart_along_their_normal_are_not_duplicates():
    """A stepped facade is two real walls at the same orientation. The offset
    test is the guard that keeps them."""

    frames = plane_frames_from_primitives([
        _wall(-8.0, name="near"), _wall(-14.0, name="far")])
    assert find_duplicate_planes(frames, camera_position=CAM) == []


def test_perpendicular_planes_are_never_duplicates():
    u = (0.0, 0.0, 1.0)
    v = (0.0, 1.0, 0.0)
    n = (1.0, 0.0, 0.0)
    c = (0.0, 0.0, -10.0)
    side = _Prim("side", tuple((u[i], v[i], n[i], c[i]) for i in range(3))
                 + ((0.0, 0.0, 0.0, 1.0),), (4.0, 3.0, 0.0))
    frames = plane_frames_from_primitives([_wall(-10.0, name="front"), side])

    assert find_duplicate_planes(frames, camera_position=CAM) == []
