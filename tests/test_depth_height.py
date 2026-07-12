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


def test_large_map_stride_branch_matches_small_map():
    """The >2MP stride-subsample branch (code-review Important #1): fresh
    geometry code — horizon scaling, fx/s intrinsics, mask upsample — that
    every other test skips because fixtures are sub-2MP. Pin it against the
    identical analytic scene at a sub-threshold resolution."""
    # 1536x1536 = 2.36MP -> stride branch; 1024x1024 = 1.05MP -> direct path.
    big, fx_b, fy_b, cx_b, cy_b = _flat_ground_depth(1.6, W=1536, H=1536, focal=1500.0)
    small, fx_s, fy_s, cx_s, cy_s = _flat_ground_depth(1.6, W=1024, H=1024, focal=1000.0)
    r_big = estimate_ground_height_from_depth(
        big, rotation=np.eye(3), fx=fx_b, fy=fy_b, cx=cx_b, cy=cy_b)
    r_small = estimate_ground_height_from_depth(
        small, rotation=np.eye(3), fx=fx_s, fy=fy_s, cx=cx_s, cy=cy_s)
    assert r_big["camera_height"] == pytest.approx(1.6, abs=0.05)
    assert r_big["camera_height"] == pytest.approx(r_small["camera_height"], abs=0.05)
    # The mask must come back at FULL input resolution, ground below horizon.
    assert r_big["ground_mask"].shape == big.shape
    assert r_big["ground_pixels"] == r_big["ground_mask"].sum()
    assert r_big["ground_mask"][:700, :].sum() == 0        # nothing above horizon
    assert r_big["ground_mask"][900:, :].mean() > 0.9      # solid ground below
    assert r_big["confidence"] > 0.9


def test_stride_branch_float32_input_and_scaled_horizon():
    """float32 input (no premature float64 copy) + an explicit horizon_y must
    scale with the stride rather than clip the candidate pool."""
    depth, fx, fy, cx, cy = _flat_ground_depth(2.5, W=1600, H=1600, focal=1600.0)
    r = estimate_ground_height_from_depth(
        depth.astype(np.float32), rotation=np.eye(3), fx=fx, fy=fy, cx=cx, cy=cy,
        horizon_y=800.0)
    assert r["camera_height"] == pytest.approx(2.5, abs=0.08)
    assert r["ground_mask"].shape == (1600, 1600)


def test_disparity_to_depth_reciprocal_spacing():
    """Code-review Important #2: the disparity conversion is pure numpy and
    must stay pinnable without model weights. Reciprocal semantics: equal
    disparity steps map to UNEQUAL depth steps, growing toward the far end —
    the linear `1 - d` flip this replaced had uniform spacing."""
    from atlas_camera.inference.depth_estimator import (_DISPARITY_FLOOR,
                                                        _disparity_to_depth)

    disp = np.linspace(1.0, 0.0, 11).reshape(1, 11)   # near -> far, equal steps
    depth, meta = _disparity_to_depth(disp.copy(), {})
    row = depth[0]
    # Direction: larger disparity (closer) -> smaller depth value.
    assert row[0] == 0.0 and row[-1] == 1.0
    assert np.all(np.diff(row) >= 0)
    # Reciprocal spacing: steps grow monotonically toward the far end
    # (uniform steps would be the old linear-flip bug).
    steps = np.diff(row[:-1])   # exclude the floored last cell
    assert np.all(np.diff(steps) > 0)
    assert steps[-1] / steps[0] > 5
    # Floor bookkeeping: exactly the cells at/below the floor are recorded.
    assert meta["disparity_floor"] == _DISPARITY_FLOOR
    assert meta["floored_fraction"] == pytest.approx((disp <= _DISPARITY_FLOOR).mean())
    # Degenerate flat map must not divide by zero.
    flat, meta2 = _disparity_to_depth(np.zeros((4, 4)), {})
    assert np.isfinite(flat).all() and meta2["floored_fraction"] == 1.0


def test_record_and_clamp_negative():
    from atlas_camera.inference.depth_estimator import _record_and_clamp_negative

    d = np.array([[1.0, -2.0], [-0.5, 4.0]])
    out, meta = _record_and_clamp_negative(d, {})
    assert meta["negative_fraction"] == pytest.approx(0.5)   # pre-clamp truth
    assert out.min() == 0.0 and out[1, 1] == 4.0
    clean, meta2 = _record_and_clamp_negative(np.ones((2, 2)), {})
    assert meta2["negative_fraction"] == 0.0 and clean.min() == 1.0


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
