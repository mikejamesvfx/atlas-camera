"""Tests for shared depth back-projection / plane-fitting primitives.

Focused here: ``detect_sky_mask`` — the heuristic that separates noisy,
spatially-incoherent monocular depth in sky/cloud regions (which otherwise
gets triangulated into jagged, distorted geometry, see test_relief_mesh.py)
from clean, reliable depth on the actual photographed building — and
``primary_camera_validity_mask``, the pure-numpy primitive behind
``AtlasOcclusionMask`` (see test_occlusion_mask.py for the node-level tests).
"""

import numpy as np
import pytest

from atlas_camera.core.depth_geometry import detect_sky_mask, primary_camera_validity_mask


def _building_and_noisy_sky_depth(height=100, width=120, horizon_y=45.0, seed=0):
    """A smooth 'building' depth below the horizon, noisy 'sky' depth above it —
    the exact failure mode Depth Anything shows on featureless sky/clouds.
    """
    rng = np.random.RandomState(seed)
    depth = np.zeros((height, width))
    for row in range(height):
        depth[row, :] = 10.0 + 0.01 * row  # smooth gradient, like a real facade
    sky_rows = int(horizon_y)
    depth[:sky_rows, :] = 40.0 + rng.uniform(-15, 15, size=(sky_rows, width))
    return depth, sky_rows


def test_detect_sky_mask_separates_noisy_sky_from_clean_building():
    horizon_y = 45.0
    depth, sky_rows = _building_and_noisy_sky_depth(horizon_y=horizon_y)

    mask = detect_sky_mask(depth, horizon_y=horizon_y)

    assert mask.shape == depth.shape
    assert mask.dtype == bool
    # Every noisy sky pixel should be flagged...
    assert mask[:sky_rows].mean() > 0.95
    # ...and no clean building pixel should be.
    assert mask[sky_rows:].mean() < 0.02


def test_detect_sky_mask_control_case_no_false_positives_on_uniform_depth():
    # No real sky in this image at all (e.g. a flat wall filling the whole
    # frame) — a tiny amount of sensor-noise-scale jitter shouldn't trigger
    # false positives just because it's above the horizon line.
    height, width = 100, 120
    rng = np.random.RandomState(1)
    depth = np.full((height, width), 10.0) + rng.normal(0, 0.01, size=(height, width))

    mask = detect_sky_mask(depth, horizon_y=45.0)

    assert mask.mean() < 0.05


def test_detect_sky_mask_ignores_region_below_horizon_regardless_of_noise():
    # Even genuinely noisy depth below the horizon (e.g. foliage, gravel)
    # must never be flagged — the mask is deliberately horizon-gated.
    height, width = 100, 120
    rng = np.random.RandomState(2)
    depth = 10.0 + rng.uniform(-3, 3, size=(height, width))

    mask = detect_sky_mask(depth, horizon_y=0.0)  # horizon at the very top row

    assert not mask.any()


def test_detect_sky_mask_requires_numpy(monkeypatch):
    import atlas_camera.core.depth_geometry as dg

    def _raise():
        raise RuntimeError("Depth geometry helpers require numpy. Install with: pip install -e .[vision]")

    monkeypatch.setattr(dg, "_require_numpy", _raise)
    with pytest.raises(RuntimeError, match="numpy"):
        detect_sky_mask(np.zeros((10, 10)), horizon_y=5.0)


def _primary_camera(eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)):
    from atlas_camera.core.camera_math import look_at_view_matrix

    view, _world, _rot = look_at_view_matrix(eye, target)
    return view


def _single_point(point, normal):
    pts = np.zeros((1, 1, 3))
    pts[0, 0] = point
    normals = np.zeros((1, 1, 3))
    normals[0, 0] = normal
    valid = np.ones((1, 1), dtype=bool)
    return pts, valid, normals, valid


_PRIMARY_KWARGS = dict(primary_fx=100.0, primary_fy=100.0, primary_cx=50.0,
                        primary_cy=40.0, primary_width=100, primary_height=80)


def test_validity_mask_false_for_point_in_front_in_frame_head_on():
    view = _primary_camera()
    pts, valid_depth, normals, valid_normal = _single_point((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    mask = primary_camera_validity_mask(
        pts, valid_depth, normals, valid_normal,
        primary_view_matrix=view, angle_threshold_deg=90.0, **_PRIMARY_KWARGS,
    )

    assert mask.shape == (1, 1)
    assert mask.dtype == bool
    assert not mask[0, 0]


def test_validity_mask_true_for_point_behind_primary_camera():
    view = _primary_camera()
    # Point at z=10 is on the far side of the camera (eye at z=5, looking -Z).
    pts, valid_depth, normals, valid_normal = _single_point((0.0, 0.0, 10.0), (0.0, 0.0, 1.0))

    mask = primary_camera_validity_mask(
        pts, valid_depth, normals, valid_normal,
        primary_view_matrix=view, angle_threshold_deg=90.0, **_PRIMARY_KWARGS,
    )

    assert mask[0, 0]


def test_validity_mask_true_for_point_outside_primary_frame():
    view = _primary_camera()
    # Far off to the side — projects way outside the 100x80 frame.
    pts, valid_depth, normals, valid_normal = _single_point((100.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    mask = primary_camera_validity_mask(
        pts, valid_depth, normals, valid_normal,
        primary_view_matrix=view, angle_threshold_deg=90.0, **_PRIMARY_KWARGS,
    )

    assert mask[0, 0]


def test_validity_mask_angle_threshold_gates_grazing_surfaces():
    view = _primary_camera()
    # Normal perpendicular to the view axis: a surface exactly edge-on to the
    # primary camera (e.g. a wall seen from directly along its own face).
    pts, valid_depth, normals, valid_normal = _single_point((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    # 90 degrees (default) never facing-excludes, even at this exact grazing angle.
    mask_90 = primary_camera_validity_mask(
        pts, valid_depth, normals, valid_normal,
        primary_view_matrix=view, angle_threshold_deg=90.0, **_PRIMARY_KWARGS,
    )
    assert not mask_90[0, 0]

    # A stricter gate does exclude it.
    mask_60 = primary_camera_validity_mask(
        pts, valid_depth, normals, valid_normal,
        primary_view_matrix=view, angle_threshold_deg=60.0, **_PRIMARY_KWARGS,
    )
    assert mask_60[0, 0]
