"""Tests for depth-based camera-height measurement (fills the depth LatentComponent).

The ground-plane fit is pure numpy and is tested against a synthetic flat-ground
depth map with a known camera height. The Depth Anything V2 model path is guarded
by importorskip so it only runs with the [neural] extra installed.
"""

import numpy as np
import pytest

from atlas_camera.core.solver import estimate_ground_height_from_depth


def _flat_ground_depth(height_m, W=512, H=512, focal=500.0):
    """Analytic z-depth of a flat ground plane Y=-height_m seen by a level camera."""
    cx = cy = W / 2.0
    fx = fy = focal
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dir_y = -(vv - cy) / fy            # world up = +Y; level camera (rotation = I)
    depth = np.full((H, W), 60.0)      # sky / no intersection above the horizon
    lower = dir_y < -1e-3              # pixels looking downward
    depth[lower] = height_m * fy / (vv[lower] - cy)
    return depth, fx, fy, cx, cy


def test_recovers_known_camera_height_on_flat_ground():
    depth, fx, fy, cx, cy = _flat_ground_depth(1.6)
    R = np.eye(3)  # level camera, world->cam identity
    result = estimate_ground_height_from_depth(
        depth, rotation=R, fx=fx, fy=fy, cx=cx, cy=cy, horizon_y=depth.shape[0] / 2.0
    )
    assert result["camera_height"] is not None
    assert result["camera_height"] == pytest.approx(1.6, abs=0.15)
    assert result["confidence"] > 0.5  # bottom band is genuinely flat ground


def test_recovers_a_taller_camera_height():
    depth, fx, fy, cx, cy = _flat_ground_depth(4.0)
    result = estimate_ground_height_from_depth(
        depth, rotation=np.eye(3), fx=fx, fy=fy, cx=cx, cy=cy, horizon_y=256.0
    )
    assert result["camera_height"] == pytest.approx(4.0, abs=0.4)


def test_no_ground_returns_none():
    # All-sky (constant far) depth has no downward-facing ground plane.
    depth = np.full((512, 512), 60.0)
    result = estimate_ground_height_from_depth(
        depth, rotation=np.eye(3), fx=500, fy=500, cx=256, cy=256, horizon_y=256.0
    )
    assert result["camera_height"] is None


def test_depth_estimator_end_to_end_if_available(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image

    from atlas_camera.inference.depth_estimator import (
        DEFAULT_RELATIVE_MODEL,
        estimate_depth,
    )

    Image.new("RGB", (256, 192), (100, 130, 160)).save(tmp_path / "img.png")
    result = estimate_depth(tmp_path / "img.png", model_id=DEFAULT_RELATIVE_MODEL, device="cpu")
    assert result.depth.shape == (192, 256)
    assert not result.is_metric
