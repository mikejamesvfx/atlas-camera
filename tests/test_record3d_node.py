"""AtlasLoadRecord3D — the ComfyUI boundary for a Record3D iPhone capture.

Covers the things that make the node usable in a graph: a real IMAGE tensor, a
metric ATLAS_DEPTH_MAP that AtlasDepthCombine will accept as ``depth_base``,
ARKit confidence gating that produces NaN holes rather than fake surface, and a
report that states the depth resolution gap instead of burying it.
"""

from __future__ import annotations

import json
import zipfile

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes_solve import AtlasLoadRecord3D  # noqa: E402
from atlas_camera.importers.record3d import SOURCE_METHOD  # noqa: E402

RGB_W, RGB_H = 640, 480
DEPTH_W, DEPTH_H = 256, 192


def _jpeg_bytes(width=RGB_W, height=RGB_H, colour=(120, 30, 200)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="JPEG")
    return buf.getvalue()


def _capture(tmp_path, *, frames=1, depth=2.0, conf_pattern=None, with_depth=True):
    meta = {
        "w": RGB_W, "h": RGB_H,
        "dw": DEPTH_W if with_depth else 0,
        "dh": DEPTH_H if with_depth else 0,
        "K": [500.0, 0.0, 0.0, 0.0, 500.0, 0.0, 320.0, 240.0, 1.0],
        "fps": 30.0,
        "poses": [[0.0, 0.0, 0.0, 1.0, 0.0, 1.5, float(i)] for i in range(frames)],
        "deviceType": 14,
    }
    path = tmp_path / "capture.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(meta))
        for i in range(frames):
            zf.writestr(f"rgbd/{i}.jpg", _jpeg_bytes())
            if with_depth:
                zf.writestr(
                    f"rgbd/{i}.depth",
                    np.full((DEPTH_H, DEPTH_W), depth, np.float32).tobytes(),
                )
                conf = (conf_pattern if conf_pattern is not None
                        else np.full((DEPTH_H, DEPTH_W), 2, np.uint8))
                zf.writestr(f"rgbd/{i}.conf", conf.tobytes())
    return str(path)


def test_node_returns_the_five_declared_slots(tmp_path):
    image, solve, depth, conf, report = AtlasLoadRecord3D().load(_capture(tmp_path))

    assert len(AtlasLoadRecord3D.RETURN_TYPES) == 5
    assert image.shape == (1, RGB_H, RGB_W, 3)
    assert image.dtype == torch.float32
    assert solve.source_method == SOURCE_METHOD
    assert depth is not None
    assert conf.shape == (1, RGB_H, RGB_W)
    assert isinstance(report, str)


def test_depth_is_metric_and_upsampled_to_the_colour_frame(tmp_path):
    """Metric is the whole point — AtlasDepthCombine keys off is_metric."""
    _, _, depth, _, _ = AtlasLoadRecord3D().load(_capture(tmp_path, depth=3.25))

    assert depth.is_metric is True
    assert (depth.image_width, depth.image_height) == (RGB_W, RGB_H)
    assert depth.depth.shape == (RGB_H, RGB_W)
    assert float(np.nanmean(depth.depth)) == pytest.approx(3.25)
    assert depth.near == pytest.approx(3.25)
    assert depth.far == pytest.approx(3.25)


def test_native_resolution_mode_leaves_depth_at_the_sensor_size(tmp_path):
    _, _, depth, _, _ = AtlasLoadRecord3D().load(
        _capture(tmp_path), depth_resolution="native")

    assert depth.depth.shape == (DEPTH_H, DEPTH_W)
    assert depth.metadata["native_width"] == DEPTH_W
    assert depth.metadata["resampled_to"] == "native"


def test_upsampling_is_nearest_so_no_detail_is_invented(tmp_path):
    """Every output sample must be a real measurement, not an interpolation.

    A bilinear upsample of a two-valued depth plane produces intermediate
    values along the boundary that read as a real slope downstream. Nearest
    keeps the value set exactly as measured.
    """
    depth_plane = np.full((DEPTH_H, DEPTH_W), 1.0, np.float32)
    depth_plane[:, DEPTH_W // 2:] = 9.0
    path = tmp_path / "step.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps({
            "w": RGB_W, "h": RGB_H, "dw": DEPTH_W, "dh": DEPTH_H,
            "K": [500.0, 0, 0, 0, 500.0, 0, 320.0, 240.0, 1.0],
            "fps": 30.0, "poses": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        }))
        zf.writestr("rgbd/0.jpg", _jpeg_bytes())
        zf.writestr("rgbd/0.depth", depth_plane.tobytes())

    _, _, depth, _, _ = AtlasLoadRecord3D().load(str(path))
    assert set(np.unique(depth.depth)) == {1.0, 9.0}


def test_low_confidence_pixels_become_nan_holes(tmp_path):
    """NaN is the existing 'invalid depth' signal the relief mesh tears around."""
    conf = np.full((DEPTH_H, DEPTH_W), 2, np.uint8)
    conf[:DEPTH_H // 2, :] = 0  # top half low-confidence
    path = _capture(tmp_path, conf_pattern=conf)

    _, _, depth, _, report = AtlasLoadRecord3D().load(path, min_confidence="high")
    nan_fraction = float(np.isnan(depth.depth).mean())

    assert nan_fraction == pytest.approx(0.5, abs=0.02)
    assert depth.metadata["rejected_px"] > 0
    assert "below 'high' confidence" in report


def test_min_confidence_any_keeps_every_pixel(tmp_path):
    conf = np.zeros((DEPTH_H, DEPTH_W), np.uint8)
    _, _, depth, _, _ = AtlasLoadRecord3D().load(
        _capture(tmp_path, conf_pattern=conf), min_confidence="any")

    assert not np.isnan(depth.depth).any()
    assert depth.metadata["rejected_px"] == 0


def test_confidence_mask_is_normalised_to_zero_one(tmp_path):
    conf = np.full((DEPTH_H, DEPTH_W), 1, np.uint8)  # ARKit "medium"
    _, _, _, mask, _ = AtlasLoadRecord3D().load(_capture(tmp_path, conf_pattern=conf))

    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    assert float(mask.mean()) == pytest.approx(0.5)


def test_solve_carries_measured_intrinsics_and_pose(tmp_path):
    _, solve, _, _, _ = AtlasLoadRecord3D().load(_capture(tmp_path))

    assert solve.camera.intrinsics.fx_px == pytest.approx(500.0)
    assert solve.camera.intrinsics.cx_px == pytest.approx(320.0)
    assert solve.camera.extrinsics.camera_position == pytest.approx((0.0, 1.5, 0.0))
    assert solve.known_intrinsics_used is True
    assert solve.debug_metadata["measured_pose"] is True
    assert solve.debug_metadata["canonical_negative_z_applied"] is False


def test_solve_from_the_node_still_serializes(tmp_path):
    """A node-built solve goes on to exporters — solve JSON is a contract."""
    _, solve, _, _, _ = AtlasLoadRecord3D().load(_capture(tmp_path))
    payload = json.loads(solve.to_json())

    assert payload["source_method"] == SOURCE_METHOD
    assert payload["debug_metadata"]["record3d_depth"]["is_metric"] is True


def test_frame_index_selects_a_different_measured_pose(tmp_path):
    path = _capture(tmp_path, frames=3)

    positions = [
        AtlasLoadRecord3D().load(path, frame_index=i)[1].camera.extrinsics.camera_position[2]
        for i in range(3)
    ]
    assert positions == pytest.approx([0.0, 1.0, 2.0])


def test_out_of_range_frame_clamps_and_says_so(tmp_path):
    _, solve, _, _, report = AtlasLoadRecord3D().load(_capture(tmp_path, frames=2), frame_index=99)

    assert solve.debug_metadata["record3d_frame_index"] == 1
    assert "clamped to 1" in report


def test_report_states_the_depth_resolution_gap(tmp_path):
    """The honest headline: this is a metric anchor, not high-res geometry."""
    _, _, _, _, report = AtlasLoadRecord3D().load(_capture(tmp_path))

    assert "256x192" in report
    assert "not a high-resolution surface" in report
    assert "MEASURED" in report


def test_capture_without_depth_still_yields_a_camera_solve(tmp_path):
    image, solve, depth, _, report = AtlasLoadRecord3D().load(
        _capture(tmp_path, with_depth=False))

    assert depth is None
    assert solve.source_method == SOURCE_METHOD
    assert image.shape == (1, RGB_H, RGB_W, 3)
    assert "no depth frames" in report


def test_empty_path_is_refused_with_a_useful_message(tmp_path):
    with pytest.raises(ValueError, match="capture_path is empty"):
        AtlasLoadRecord3D().load("   ")


def test_node_is_registered_under_a_stable_key():
    # Record3D capture is ATLAS_IOS-gated for v1, so the stable key lives in
    # the ios tier rather than the default mapping (same keys either way).
    from atlas_camera.comfy.node_registry import (
        IOS_NODE_CLASS_MAPPINGS,
        IOS_NODE_DISPLAY_NAME_MAPPINGS,
    )

    assert IOS_NODE_CLASS_MAPPINGS["AtlasLoadRecord3D"] is AtlasLoadRecord3D
    assert IOS_NODE_DISPLAY_NAME_MAPPINGS["AtlasLoadRecord3D"] == "Atlas Load Record3D Capture 📱"


def test_depth_output_is_accepted_by_atlas_depth_combine(tmp_path):
    """The intended graph: LiDAR as metric base, monocular as detail source.

    This is the whole reason the importer exists, so the handoff is pinned
    rather than assumed — AtlasDepthCombine must take the node's ATLAS_DEPTH_MAP
    as ``depth_base`` and preserve its metric scale.
    """
    from atlas_camera.comfy.nodes_depth import AtlasDepthCombine
    from atlas_camera.inference.depth_estimator import DepthResult

    _, _, lidar, _, _ = AtlasLoadRecord3D().load(_capture(tmp_path, depth=4.0),
                                                 min_confidence="any")

    rng = np.random.default_rng(0)
    monocular = DepthResult(
        depth=(4.0 + rng.normal(0, 0.2, (RGB_H, RGB_W))).astype(np.float32),
        is_metric=False, model_id="mono/test",
        image_width=RGB_W, image_height=RGB_H, near=3.0, far=5.0,
    )

    combined, report = AtlasDepthCombine().combine(lidar, monocular, mode="high_freq_detail")

    assert combined.is_metric is True  # base flag survives
    assert combined.depth.shape == (RGB_H, RGB_W)
    assert float(np.nanmedian(combined.depth)) == pytest.approx(4.0, rel=0.05)
    assert "record3d/arkit_scene_depth" in report
