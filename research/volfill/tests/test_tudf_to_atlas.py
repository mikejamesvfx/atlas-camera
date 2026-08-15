"""Mapping tests for the VolFill -> Atlas world chain.

No GPU, no weights, no VolFill install: every test builds a synthetic TUDF whose
true surface is known analytically, so a convention error (axis order, half-voxel
offset, y/z sign, view-matrix direction) fails loudly instead of producing a
plausible-looking but wrong point cloud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tudf_to_atlas import (  # noqa: E402
    MOGE_TO_ATLAS_CAM,
    VolFillVolume,
    atlas_camera_to_world,
    load_volume,
    moge_to_atlas_camera,
    surface_points_canonical,
    volfill_to_atlas_world,
)

TRUNC = 3.0


def _volume_from_occupancy(occ_zyx: np.ndarray, bbox_min, extent) -> VolFillVolume:
    """Occupancy (z,y,x) -> a TUDF that is 0 on occupied voxels, TRUNC elsewhere."""
    tudf = np.where(occ_zyx, 0.0, TRUNC).astype(np.float32)
    return VolFillVolume(
        tudf=tudf,
        bbox_min=np.asarray(bbox_min, dtype=np.float64),
        extent=np.asarray(extent, dtype=np.float64),
        truncation_voxels=TRUNC,
    )


def test_single_voxel_recovers_its_own_centre():
    """The half-voxel offset and the (z,y,x) axis order are both load-bearing."""
    R = 8
    occ = np.zeros((R, R, R), dtype=bool)
    # Distinct indices per axis so a transposition cannot pass by symmetry.
    iz, iy, ix = 1, 3, 6
    occ[iz, iy, ix] = True
    bbox_min = np.array([-2.0, -2.0, 4.0])
    extent = np.array([8.0, 8.0, 8.0])
    vol = _volume_from_occupancy(occ, bbox_min, extent)

    pts, vals = surface_points_canonical(vol, threshold=0.5)

    assert pts.shape == (1, 3)
    vs = extent / R  # 1.0 m
    expected = bbox_min + (np.array([ix, iy, iz]) + 0.5) * vs
    np.testing.assert_allclose(pts[0], expected)
    assert vals[0] == pytest.approx(0.0)


def test_threshold_is_in_voxel_units_and_inclusive():
    """`tudf <= t` (visualize.py), not `<` and not the dead `< 0.0` predicate."""
    tudf = np.full((4, 4, 4), TRUNC, dtype=np.float32)
    tudf[0, 0, 0] = 0.5
    tudf[1, 1, 1] = 0.51
    vol = VolFillVolume(tudf=tudf, bbox_min=np.zeros(3), extent=np.full(3, 4.0),
                        truncation_voxels=TRUNC)

    assert surface_points_canonical(vol, threshold=0.5)[0].shape[0] == 1
    assert surface_points_canonical(vol, threshold=0.6)[0].shape[0] == 2
    # The upstream dead-code predicate would select nothing on a [0, trunc] field.
    assert surface_points_canonical(vol, threshold=0.0)[0].shape[0] == 0


def test_moge_to_atlas_flips_y_and_z_only():
    pts = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(moge_to_atlas_camera(pts), [[1.0, -2.0, -3.0]])
    # A point in front of a MoGe camera (+z) lands in front of an Atlas one (-z).
    assert moge_to_atlas_camera(np.array([[0.0, 0.0, 5.0]]))[0, 2] < 0
    # Handedness is preserved: det == +1, so it is a rotation, not a mirror.
    assert np.linalg.det(MOGE_TO_ATLAS_CAM) == pytest.approx(1.0)


def test_moge_to_atlas_applies_scale():
    pts = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(moge_to_atlas_camera(pts, scale=2.0),
                               [[2.0, -4.0, -6.0]])


def test_camera_to_world_uses_view_matrix_as_world_to_camera():
    """view_matrix is WORLD->CAM; the camera origin must map to its world position."""
    cam_pos = np.array([3.0, 1.6, -7.0])
    # Camera yawed 90 deg about world Y.
    c = np.cos(np.pi / 2)
    s = np.sin(np.pi / 2)
    R_cw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c2w = np.eye(4)
    c2w[:3, :3] = R_cw
    c2w[:3, 3] = cam_pos
    view_matrix = np.linalg.inv(c2w)

    np.testing.assert_allclose(
        atlas_camera_to_world(np.zeros((1, 3)), view_matrix)[0], cam_pos, atol=1e-9)
    # A point 10 m in front of the camera (-Z in Atlas camera space).
    world = atlas_camera_to_world(np.array([[0.0, 0.0, -10.0]]), view_matrix)[0]
    np.testing.assert_allclose(world, cam_pos + R_cw @ np.array([0, 0, -10.0]),
                               atol=1e-9)


def test_identity_view_matrix_round_trip_is_exact():
    """Whole chain against an analytically known plane, identity camera."""
    R = 16
    occ = np.zeros((R, R, R), dtype=bool)
    occ[5, :, :] = True                      # a constant-z slab
    bbox_min = np.array([-1.0, -1.0, 2.0])
    extent = np.full(3, 4.0)
    vol = _volume_from_occupancy(occ, bbox_min, extent)

    out = volfill_to_atlas_world(vol, np.eye(4), threshold=0.5)

    vs = extent / R
    expected_z_moge = bbox_min[2] + (5 + 0.5) * vs[2]
    # Identity view matrix: world == Atlas camera space, so z is negated.
    np.testing.assert_allclose(out["points_xyz"][:, 2], -expected_z_moge)
    assert out["metadata"]["n_points"] == R * R
    assert out["metadata"]["voxel_edge_m"] == pytest.approx(0.25)
    assert out["source"] == "volfill"


def test_confidence_is_one_at_the_surface_and_zero_at_the_boundary():
    tudf = np.full((4, 4, 4), TRUNC, dtype=np.float32)
    tudf[0, 0, 0] = 0.0     # exactly on the surface
    tudf[0, 0, 1] = 0.5     # at the selection boundary
    vol = VolFillVolume(tudf=tudf, bbox_min=np.zeros(3), extent=np.full(3, 4.0),
                        truncation_voxels=TRUNC)

    out = volfill_to_atlas_world(vol, np.eye(4), threshold=0.5)
    conf = dict(zip(map(tuple, out["points_xyz"].round(6)), out["confidence"]))
    assert sorted(conf.values()) == [pytest.approx(0.0), pytest.approx(1.0)]


def test_voxel_edge_reports_the_resolution_gate():
    """The exterior-plate resolution problem, in one assertion."""
    # A 200 m-deep street plate: isotropic cube, 256 grid.
    vol = VolFillVolume(tudf=np.zeros((256, 256, 256), dtype=np.float32),
                        bbox_min=np.zeros(3), extent=np.full(3, 200.0),
                        truncation_voxels=TRUNC)
    assert vol.voxel_edge_m == pytest.approx(0.78125)
    # A 4 m interior plate resolves ~1.6 cm.
    vol_in = VolFillVolume(tudf=np.zeros((256, 256, 256), dtype=np.float32),
                           bbox_min=np.zeros(3), extent=np.full(3, 4.0),
                           truncation_voxels=TRUNC)
    assert vol_in.voxel_edge_m == pytest.approx(0.015625)


def test_load_volume_reads_the_on_disk_contract(tmp_path):
    R = 8
    tudf = np.full((R, R, R), TRUNC, dtype=np.float32)
    tudf[2, 3, 4] = 0.1
    np.savez_compressed(tmp_path / f"pred_tudf_{R}.npz", tudf=tudf)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "representation": "tudf",
        "truncation_voxels": TRUNC,
        "field_range": [0.0, TRUNC],
        "field_units": "voxel_units",
        "bbox_min": [-1.0, -2.0, 3.0],
        "extent_xyz": [8.0, 8.0, 8.0],
        "pred_resolution": [R, R, R],
    }), encoding="utf-8")

    vol = load_volume(tmp_path)

    assert vol.resolution == R
    assert vol.truncation_voxels == pytest.approx(TRUNC)
    np.testing.assert_allclose(vol.bbox_min, [-1.0, -2.0, 3.0])
    pts, _ = surface_points_canonical(vol, threshold=0.5)
    np.testing.assert_allclose(pts[0], [-1.0 + 4.5, -2.0 + 3.5, 3.0 + 2.5])


def test_empty_surface_is_not_a_crash():
    vol = VolFillVolume(tudf=np.full((4, 4, 4), TRUNC, dtype=np.float32),
                        bbox_min=np.zeros(3), extent=np.full(3, 4.0),
                        truncation_voxels=TRUNC)
    out = volfill_to_atlas_world(vol, np.eye(4), threshold=0.5)
    assert out["metadata"]["n_points"] == 0
    assert out["points_xyz"].shape == (0, 3)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        moge_to_atlas_camera(np.zeros((4,)))
    with pytest.raises(ValueError):
        atlas_camera_to_world(np.zeros((2, 3)), np.eye(3))
