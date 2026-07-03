"""Tests for shared depth back-projection / plane-fitting primitives.

Focused here: ``detect_sky_mask`` — the heuristic that separates noisy,
spatially-incoherent monocular depth in sky/cloud regions (which otherwise
gets triangulated into jagged, distorted geometry, see test_relief_mesh.py)
from clean, reliable depth on the actual photographed building.
"""

import numpy as np
import pytest

from atlas_camera.core.depth_geometry import detect_sky_mask


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
