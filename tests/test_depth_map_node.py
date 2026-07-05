"""Tests for AtlasDepthMap — the shared metric depth pass for the composable
geometry-derivation nodes (AtlasDeriveReliefMesh/Walls/TowersSpires/
RoofsFacades/InteriorRoom). Monkeypatches estimate_depth exactly like
test_occlusion_mask.py/test_add_patch_view.py do, so this needs no [neural]
extra or model download.
"""

import pytest

from atlas_camera.comfy.nodes import AtlasDepthMap, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from atlas_camera.inference.depth_estimator import DepthResult


def _patch_estimate_depth(monkeypatch, size=64):
    np = pytest.importorskip("numpy")

    def fake(image_path, *, model_id=None, device=None):
        ramp = np.linspace(30.0, 5.0, size)[:, None] * np.ones((1, size), dtype=np.float32)
        return DepthResult(
            depth=ramp.astype(np.float32), is_metric=True, model_id=model_id or "fake",
            image_width=size, image_height=size, near=5.0, far=30.0,
        )

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)
    return fake


def test_node_registered_and_return_types():
    assert NODE_CLASS_MAPPINGS["AtlasDepthMap"] is AtlasDepthMap
    assert "AtlasDepthMap" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasDepthMap.RETURN_TYPES == ("ATLAS_DEPTH_MAP",)


def test_estimate_returns_depth_result(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch, size=64)
    image = torch.rand(1, 64, 64, 3, dtype=torch.float32)

    (result,) = AtlasDepthMap().estimate(image)

    assert isinstance(result, DepthResult)
    assert result.is_metric is True
    assert result.depth.shape == (64, 64)
    assert result.image_width == 64 and result.image_height == 64


def test_estimate_passes_through_model_id(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch, size=32)
    image = torch.rand(1, 32, 32, 3, dtype=torch.float32)

    (result,) = AtlasDepthMap().estimate(
        image, depth_model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")

    assert result.model_id == "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
