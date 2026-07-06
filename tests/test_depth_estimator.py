"""Tests for atlas_camera.inference.depth_estimator: device resolution and
cross-call depth-RESULT memoization (distinct from the model-WEIGHTS cache).
The heavy HF backend is mocked out via _get_model so these run with only
numpy/torch/PIL, no model download.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.inference import depth_estimator as de


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeOutputs:
    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth


def _patch_fake_backend(monkeypatch, call_counter):
    """Replace _get_model with a fake processor+model pair, counting how many
    times the (expensive) model forward pass actually runs."""

    def fake_get_model(model_id, device):
        def fake_processor(images, return_tensors):
            return _FakeInputs(pixel_values=torch.zeros(1, 3, 8, 8))

        def fake_model(**kwargs):
            call_counter["count"] += 1
            return _FakeOutputs(torch.rand(1, 4, 4, dtype=torch.float32) + 1.0)

        return fake_processor, fake_model

    monkeypatch.setattr(de, "_get_model", fake_get_model)


def _write_test_image(path, size=(16, 16), color=(10, 20, 30)):
    from PIL import Image
    Image.new("RGB", size, color=color).save(path)


def test_resolve_device_prefers_explicit_over_autodetect():
    from atlas_camera.inference._common import resolve_device

    class _FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    assert resolve_device("cpu", _FakeTorch) == "cpu"  # explicit wins
    assert resolve_device(None, _FakeTorch) == "cuda"   # else autodetect cuda first


def test_estimate_depth_caches_result_across_calls_on_identical_image(monkeypatch, tmp_path):
    call_counter = {"count": 0}
    _patch_fake_backend(monkeypatch, call_counter)
    de._DEPTH_RESULT_CACHE.clear()

    img_path = tmp_path / "photo.png"
    _write_test_image(img_path)

    result1 = de.estimate_depth(str(img_path), model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", device="cpu")
    result2 = de.estimate_depth(str(img_path), model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", device="cpu")

    assert call_counter["count"] == 1          # second call was a cache hit, no re-inference
    assert result2 is result1                  # same cached object returned


def test_estimate_depth_cache_miss_on_different_image(monkeypatch, tmp_path):
    call_counter = {"count": 0}
    _patch_fake_backend(monkeypatch, call_counter)
    de._DEPTH_RESULT_CACHE.clear()

    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _write_test_image(img_a, color=(10, 20, 30))
    _write_test_image(img_b, color=(200, 100, 50))

    de.estimate_depth(str(img_a), model_id="m", device="cpu")
    de.estimate_depth(str(img_b), model_id="m", device="cpu")

    assert call_counter["count"] == 2  # genuinely different images must not share a cache entry


def test_estimate_depth_cache_is_bounded():
    from atlas_camera.inference._common import bounded_cache_set

    de._DEPTH_RESULT_CACHE.clear()
    for i in range(de._DEPTH_RESULT_CACHE_MAX + 3):
        bounded_cache_set(de._DEPTH_RESULT_CACHE, (str(i), "m", "cpu"), object(), de._DEPTH_RESULT_CACHE_MAX)
    assert len(de._DEPTH_RESULT_CACHE) == de._DEPTH_RESULT_CACHE_MAX
