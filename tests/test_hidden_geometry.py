"""Tests for the experimental hidden-geometry track: pure-numpy layer
selection (core/hidden_geometry.py), the AtlasPredictHiddenGeometry node
(LaRI inference mocked out), and the guarded-import error. No LaRI clone, no
downloads, no torch model."""

import numpy as np
import pytest

from atlas_camera.core.hidden_geometry import (
    register_layers_to_depth,
    select_hidden_surface,
)


def _synthetic_scene(H=32, W=32, model_scale=0.5):
    """Bg plane at 10 units with a fg box at 2 units in the middle; layered
    stack in the MODEL's units (pipeline = model * 1/model_scale):
      layer 0: visible surface,
      layer 1: fg box's back face (barely behind, must be skipped),
      layer 2: the bg continuation behind the box (the target).
    On bg pixels all layers collapse onto the visible surface (no hidden)."""
    visible = np.full((H, W), 10.0)
    fg = np.zeros((H, W), bool)
    fg[8:24, 8:24] = True
    visible[fg] = 2.0

    layers = np.empty((H, W, 3))
    layers[..., 0] = visible * model_scale
    layers[..., 1] = visible * model_scale + 0.01   # back face / collapse
    layers[..., 2] = visible * model_scale + 0.02
    layers[fg, 1] = (2.0 + 0.3) * model_scale        # box back: +0.3 < margin
    layers[fg, 2] = 10.0 * model_scale               # true bg continuation
    return visible, layers, fg


def test_registration_recovers_scale():
    visible, layers, _ = _synthetic_scene(model_scale=0.5)
    scale, rel_mad, valid = register_layers_to_depth(layers, visible)
    assert scale == pytest.approx(2.0, rel=1e-6)
    assert rel_mad == pytest.approx(0.0, abs=1e-9)
    assert valid.all()


def test_selection_skips_back_face_and_finds_continuation():
    visible, layers, fg = _synthetic_scene()
    hidden, hidden_valid, stats = select_hidden_surface(
        layers, visible, clear_rel=0.2)

    # Only the fg box gets a hidden surface, and it's the bg plane (10 units),
    # NOT the box's back face (2.3 units — inside the clearance margin).
    assert hidden_valid[fg].all()
    assert not hidden_valid[~fg].any()
    np.testing.assert_allclose(hidden[fg], 10.0, rtol=1e-6)
    assert hidden[~fg].max() == 0.0
    # first clearing layer at fg is index 2 (histogram indexes real layers)
    assert stats["layer_used_histogram"][2] == int(fg.sum())
    assert stats["coverage"] == pytest.approx(fg.mean())


def test_adaptive_min_clear_scales_with_scene():
    # Shallow scene: fg 1.5, bg 2.0 — a fixed 20%-of-visible margin (0.3)
    # would reject the true 0.5 separation only if min_clear dominated wrongly;
    # auto min_clear (2% of median visible) must stay small enough to accept.
    H = W = 16
    visible = np.full((H, W), 2.0)
    fg = np.zeros((H, W), bool)
    fg[4:12, 4:12] = True
    visible[fg] = 1.5
    layers = np.stack([visible, np.where(fg, 2.0, visible + 0.001)], axis=-1)
    hidden, hidden_valid, stats = select_hidden_surface(
        layers, visible, clear_rel=0.2)
    assert hidden_valid[fg].all()
    np.testing.assert_allclose(hidden[fg], 2.0, rtol=1e-6)


def test_registration_failure_is_graceful():
    layers = np.zeros((8, 8, 2))
    visible = np.zeros((8, 8))
    hidden, hidden_valid, stats = select_hidden_surface(layers, visible)
    assert not hidden_valid.any()
    assert "warning" in stats


def test_lari_require_raises_informative_error_without_clone():
    from atlas_camera.inference.lari_hidden_geometry import _resolve_lari_root

    with pytest.raises(RuntimeError, match="research use only"):
        _resolve_lari_root("C:/definitely/not/a/lari/clone")


def test_node_patches_depth_and_reports(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from atlas_camera.comfy.nodes import AtlasPredictHiddenGeometry
    from atlas_camera.inference.depth_estimator import DepthResult
    import atlas_camera.inference.lari_hidden_geometry as lhg

    H = W = 32
    visible, layers, fg = _synthetic_scene(H, W, model_scale=0.5)

    def fake_predict(image_path, *, lari_path=None, device=None):
        return lhg.LayeredDepthResult(
            layers=layers.astype(np.float32), image_width=W, image_height=H,
            metadata={"research_only": True})

    monkeypatch.setattr(lhg, "predict_layered_depth", fake_predict)

    depth_in = DepthResult(
        depth=visible.astype(np.float32), is_metric=True, model_id="fake",
        image_width=W, image_height=H, near=2.0, far=10.0)
    image = torch.rand(1, H, W, 3, dtype=torch.float32)

    out, mask, report = AtlasPredictHiddenGeometry().predict(
        depth_in, image, clear_rel=0.2)

    assert out.model_id.endswith("+lari_hidden")
    assert out.metadata["research_only"] is True
    # fg pixels replaced by the bg continuation; bg untouched
    np.testing.assert_allclose(out.depth[fg], 10.0, rtol=1e-5)
    np.testing.assert_allclose(out.depth[~fg], visible[~fg], rtol=1e-6)
    assert mask.shape == (1, H, W)
    assert bool(mask[0][torch.from_numpy(fg)].min() > 0.5)
    assert "RESEARCH-ONLY" in report

    # restrict_mask limits substitution
    restrict = torch.zeros(1, H, W)
    restrict[0, :16, :] = 1.0
    out2, mask2, _ = AtlasPredictHiddenGeometry().predict(
        depth_in, image, clear_rel=0.2, restrict_mask=restrict)
    assert bool(mask2[0, 20:, :].max() == 0)  # nothing below row 16


def test_node_registered():
    from atlas_camera.comfy.nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        AtlasPredictHiddenGeometry,
    )
    assert NODE_CLASS_MAPPINGS["AtlasPredictHiddenGeometry"] is AtlasPredictHiddenGeometry
    assert "research" in NODE_DISPLAY_NAME_MAPPINGS["AtlasPredictHiddenGeometry"]
