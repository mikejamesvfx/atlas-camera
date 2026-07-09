"""Tests for the Depth Anything 3 (DA3) opt-in backend in
atlas_camera.inference.depth_estimator: model-id dispatch, canonical->metric
focal conversion, predicted-intrinsics fallback, resize-to-source, per-family
is_metric semantics, and focal-aware result caching. The depth_anything_3
package is mocked out via _get_da3_model so these run with only numpy/torch/PIL,
no install or download — mirroring test_depth_estimator.py's conventions.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.inference import depth_estimator as de

DA3_METRIC = "depth-anything/DA3METRIC-LARGE"
DA3_MONO = "depth-anything/DA3MONO-LARGE"
DA3_NESTED = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
V2_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def _write_test_image(path, size=(16, 16), color=(10, 20, 30)):
    from PIL import Image
    Image.new("RGB", size, color=color).save(path)


def _patch_fake_da3(monkeypatch, call_counter, *, net_depth=None, focal=400.0):
    """Replace _get_da3_model with a fake whose .inference() returns a
    DA3-shaped prediction, counting actual inference runs."""
    if net_depth is None:
        net_depth = np.linspace(1.0, 3.0, 16, dtype=np.float32).reshape(4, 4)

    class _FakeDA3Model:
        def inference(self, images):
            call_counter["count"] += 1
            return SimpleNamespace(
                depth=np.asarray([net_depth]),
                conf=np.ones((1,) + net_depth.shape, dtype=np.float32),
                intrinsics=np.asarray(
                    [[[focal, 0.0, 2.0], [0.0, focal, 2.0], [0.0, 0.0, 1.0]]]
                ),
                extrinsics=np.zeros((1, 3, 4), dtype=np.float32),
                processed_images=np.zeros(
                    (1,) + net_depth.shape + (3,), dtype=np.uint8
                ),
            )

    def fake_get_da3_model(model_id, device):
        return _FakeDA3Model()

    monkeypatch.setattr(de, "_get_da3_model", fake_get_da3_model)
    return net_depth


def test_da3_dispatch_routes_by_model_id(monkeypatch, tmp_path):
    counter = {"count": 0}
    _patch_fake_da3(monkeypatch, counter)
    de._DEPTH_RESULT_CACHE.clear()

    def exploding_get_model(model_id, device):  # V2 loader must never run
        raise AssertionError("V2 backend touched for a DA3 model id")

    monkeypatch.setattr(de, "_get_model", exploding_get_model)

    img = tmp_path / "photo.png"
    _write_test_image(img)

    result = de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu")
    assert counter["count"] == 1
    assert result.metadata["backend"] == "da3"
    assert result.model_id == DA3_METRIC

    # And a V2 id still routes to the transformers loader (which now explodes).
    with pytest.raises(AssertionError, match="V2 backend touched"):
        de.estimate_depth(str(img), model_id=V2_OUTDOOR, device="cpu")


def test_da3_metric_conversion_with_solve_focal():
    net = np.full((4, 4), 2.0, dtype=np.float32)
    depth, source, f_used = de._da3_metric_from_canonical(
        net, focal_px=1000.0, source_width=1600, processed_width=504,
        predicted_focal=400.0,
    )
    expected_focal = 1000.0 * 504 / 1600
    assert source == "solve"
    assert f_used == pytest.approx(expected_focal)
    np.testing.assert_allclose(depth, net * expected_focal / 300.0)


def test_da3_metric_focal_falls_back_to_predicted_intrinsics(monkeypatch, tmp_path):
    net = np.full((4, 4), 2.0, dtype=np.float32)
    depth, source, f_used = de._da3_metric_from_canonical(
        net, focal_px=None, source_width=1600, processed_width=504,
        predicted_focal=400.0,
    )
    assert source == "predicted"
    assert f_used == pytest.approx(400.0)
    np.testing.assert_allclose(depth, net * 400.0 / 300.0)

    # End-to-end: metadata records which focal was used.
    counter = {"count": 0}
    _patch_fake_da3(monkeypatch, counter, focal=400.0)
    de._DEPTH_RESULT_CACHE.clear()
    img = tmp_path / "photo.png"
    _write_test_image(img)

    predicted = de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu")
    assert predicted.metadata["focal_source"] == "predicted"
    assert predicted.metadata["focal_px_processed"] == pytest.approx(400.0)

    solved = de.estimate_depth(
        str(img), model_id=DA3_METRIC, device="cpu", focal_px=800.0
    )
    assert solved.metadata["focal_source"] == "solve"
    # source width 16, processed width 4 -> focal scales by 4/16.
    assert solved.metadata["focal_px_processed"] == pytest.approx(800.0 * 4 / 16)


def test_da3_metric_assumed_focal_when_model_predicts_no_intrinsics(monkeypatch, tmp_path):
    """DA3METRIC-LARGE is a depth-only head (intrinsics=None, confirmed live) —
    an image-only call must fall back to the assumed normal-lens focal, not raise."""
    net = np.full((4, 4), 2.0, dtype=np.float32)

    class _NoCameraModel:
        def inference(self, images):
            return SimpleNamespace(
                depth=np.asarray([net]), conf=None, intrinsics=None,
                extrinsics=None, processed_images=None,
            )

    monkeypatch.setattr(de, "_get_da3_model", lambda mid, dev: _NoCameraModel())
    de._DEPTH_RESULT_CACHE.clear()
    img = tmp_path / "photo.png"
    _write_test_image(img, size=(4, 4))

    result = de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu")
    assert result.metadata["focal_source"] == "assumed"
    # assumed focal = processed width (4) -> depth = net * 4 / 300
    np.testing.assert_allclose(result.depth, net * 4.0 / 300.0)

    # A solve focal still wins over the assumed fallback.
    solved = de.estimate_depth(
        str(img), model_id=DA3_METRIC, device="cpu", focal_px=600.0
    )
    assert solved.metadata["focal_source"] == "solve"


def test_da3_resizes_to_source_resolution(monkeypatch, tmp_path):
    counter = {"count": 0}
    _patch_fake_da3(monkeypatch, counter)  # 4x4 processed depth
    de._DEPTH_RESULT_CACHE.clear()
    img = tmp_path / "photo.png"
    _write_test_image(img, size=(16, 16))

    result = de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu")
    assert result.depth.shape == (16, 16)
    assert result.image_width == 16 and result.image_height == 16
    assert result.metadata["processed_width"] == 4
    assert result.metadata["processed_height"] == 4


def test_da3_is_metric_per_family(monkeypatch, tmp_path):
    ramp = np.linspace(5.0, 9.0, 16, dtype=np.float32).reshape(4, 4)
    counter = {"count": 0}
    _patch_fake_da3(monkeypatch, counter, net_depth=ramp)
    de._DEPTH_RESULT_CACHE.clear()
    img = tmp_path / "photo.png"
    _write_test_image(img, size=(4, 4))  # matches processed res: no resize

    metric = de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu")
    assert metric.is_metric is True

    nested = de.estimate_depth(str(img), model_id=DA3_NESTED, device="cpu")
    assert nested.is_metric is True
    np.testing.assert_allclose(nested.depth, ramp)  # metres passed through unscaled

    mono = de.estimate_depth(str(img), model_id=DA3_MONO, device="cpu")
    assert mono.is_metric is False
    assert mono.depth.min() == pytest.approx(0.0)
    assert mono.depth.max() == pytest.approx(1.0)
    # DA3MONO is relative DEPTH, not disparity: ordering must be preserved.
    assert np.unravel_index(np.argmax(mono.depth), mono.depth.shape) == \
        np.unravel_index(np.argmax(ramp), ramp.shape)


def test_da3_cache_key_includes_focal_only_when_used(monkeypatch, tmp_path):
    counter = {"count": 0}
    _patch_fake_da3(monkeypatch, counter)
    de._DEPTH_RESULT_CACHE.clear()
    img = tmp_path / "photo.png"
    _write_test_image(img)

    de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu")
    de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu", focal_px=1000.0)
    assert counter["count"] == 2  # focal changes the metric result -> new entry
    de.estimate_depth(str(img), model_id=DA3_METRIC, device="cpu", focal_px=1000.0)
    assert counter["count"] == 2  # identical focal is a cache hit

    # V2 ids ignore focal_px entirely — same cache entry with and without it.
    v2_counter = {"count": 0}

    def fake_get_model(model_id, device):
        def fake_processor(images, return_tensors):
            class _Inputs(dict):
                def to(self, device):
                    return self
            return _Inputs(pixel_values=torch.zeros(1, 3, 8, 8))

        def fake_model(**kwargs):
            v2_counter["count"] += 1
            return SimpleNamespace(
                predicted_depth=torch.rand(1, 4, 4, dtype=torch.float32) + 1.0
            )

        return fake_processor, fake_model

    monkeypatch.setattr(de, "_get_model", fake_get_model)
    de.estimate_depth(str(img), model_id=V2_OUTDOOR, device="cpu")
    de.estimate_depth(str(img), model_id=V2_OUTDOOR, device="cpu", focal_px=1000.0)
    assert v2_counter["count"] == 1


@pytest.mark.skipif(
    not os.environ.get("ATLAS_RUN_DA3_LIVE"),
    reason="live DA3 inference is opt-in: set ATLAS_RUN_DA3_LIVE=1",
)
def test_da3_live_end_to_end(tmp_path):
    pytest.importorskip("depth_anything_3")
    img = tmp_path / "photo.png"
    _write_test_image(img, size=(64, 48))
    result = de.estimate_depth(str(img), model_id=DA3_METRIC, focal_px=60.0)
    assert result.depth.shape == (48, 64)
    assert result.is_metric is True
    assert np.isfinite(result.depth).all()
