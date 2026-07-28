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
    seen = {}

    def fake(image_path, *, model_id=None, device=None, focal_px=None, **kwargs):
        seen["focal_px"] = focal_px
        seen.update(kwargs)
        ramp = np.linspace(30.0, 5.0, size)[:, None] * np.ones((1, size), dtype=np.float32)
        return DepthResult(
            depth=ramp.astype(np.float32), is_metric=True, model_id=model_id or "fake",
            image_width=size, image_height=size, near=5.0, far=30.0,
        )

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)
    return seen


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


def test_optional_solve_input_threads_solved_focal(monkeypatch):
    """The optional `solve` input supplies the GeoCalib focal for DA3METRIC,
    rescaled to the wired image's pixel width; without it focal_px is None."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from types import SimpleNamespace

    seen = _patch_estimate_depth(monkeypatch, size=32)
    image = torch.rand(1, 32, 32, 3, dtype=torch.float32)

    AtlasDepthMap().estimate(image)
    assert seen["focal_px"] is None

    solve = SimpleNamespace(camera=SimpleNamespace(
        intrinsics=SimpleNamespace(fx_px=1000.0, image_width=64)))
    AtlasDepthMap().estimate(image, solve=solve)
    # solve is 64px wide, wired image 32px -> focal halves.
    assert seen["focal_px"] == pytest.approx(1000.0 * 32 / 64)


def test_moge_cost_knobs_reach_estimate_depth_and_default_to_inert(monkeypatch):
    """The appended MoGe dials must actually thread through, at the right defaults.

    A widget that is wired into INPUT_TYPES but never reaches the backend is the
    quiet failure here: the graph shows a knob, turning it changes nothing, and
    nothing errors. So assert both halves — the defaults are the no-op values
    (9 / 0 / "") so existing graphs are untouched, and a non-default value
    actually arrives.
    """
    torch = pytest.importorskip("torch")
    seen = _patch_estimate_depth(monkeypatch, size=32)
    image = torch.rand(1, 32, 32, 3, dtype=torch.float32)

    AtlasDepthMap().estimate(image)
    assert seen["resolution_level"] == 9, "default must be MoGe's own full-detail level"
    assert seen["max_side"] == 0, "default must disable the pre-inference cap"
    assert seen["checkpoint_path"] == "", "default must keep the HuggingFace path"

    AtlasDepthMap().estimate(image, moge_resolution_level=3, moge_max_side=2048,
                             moge_checkpoint_path="/models/moge/model.pt")
    assert seen["resolution_level"] == 3
    assert seen["max_side"] == 2048
    assert seen["checkpoint_path"] == "/models/moge/model.pt"
