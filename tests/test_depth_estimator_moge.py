"""MoGe-2 depth backend: id classification, dispatch routing, and (when the
package is installed) a live end-to-end inference. Mirrors test_depth_estimator_da3.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

import atlas_camera.inference.depth_estimator as de
from atlas_camera.inference.depth_estimator import DepthResult, _is_moge_model


def test_moge_model_id_classification():
    assert _is_moge_model("Ruicheng/moge-2-vitl-normal")
    assert _is_moge_model("Ruicheng/moge-2-vitb-normal")
    assert not _is_moge_model("depth-anything/DA3METRIC-LARGE")
    assert not _is_moge_model("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")


def test_moge_dispatch_routes_by_model_id(monkeypatch, tmp_path):
    """A moge id must reach _estimate_depth_moge, never the V2 or DA3 paths."""
    calls = {"moge": 0}

    def fake_moge(image_path, *, model_id, device, focal_px, **kwargs):
        calls["moge"] += 1
        assert focal_px == 500.0  # focal threads through for fov_x
        return DepthResult(depth=np.zeros((4, 4), np.float32), is_metric=True,
                           model_id=model_id, image_width=4, image_height=4)

    def exploding(*a, **k):
        raise AssertionError("wrong backend dispatched")

    monkeypatch.setattr(de, "_estimate_depth_moge", fake_moge)
    monkeypatch.setattr(de, "_estimate_depth_v2", exploding)
    monkeypatch.setattr(de, "_estimate_depth_da3", exploding)

    img = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(img)
    r = de.estimate_depth(str(img), model_id="Ruicheng/moge-2-vitl-normal", focal_px=500.0)
    assert calls["moge"] == 1
    assert r.model_id == "Ruicheng/moge-2-vitl-normal"


def test_moge_cache_key_fragments_on_focal(monkeypatch, tmp_path):
    """fov_x depends on the focal, so distinct focals must not share a cache entry."""
    de._DEPTH_RESULT_CACHE.clear()  # isolate from other tests' shared module cache
    seen = []

    def fake_moge(image_path, *, model_id, device, focal_px, **kwargs):
        seen.append(focal_px)
        return DepthResult(depth=np.zeros((4, 4), np.float32), is_metric=True,
                           model_id=model_id, image_width=4, image_height=4)

    monkeypatch.setattr(de, "_estimate_depth_moge", fake_moge)
    img = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(img)
    mid = "Ruicheng/moge-2-vitl-normal"
    de.estimate_depth(str(img), model_id=mid, focal_px=500.0)
    de.estimate_depth(str(img), model_id=mid, focal_px=500.0)   # cache hit
    de.estimate_depth(str(img), model_id=mid, focal_px=900.0)   # different focal
    assert seen == [500.0, 900.0]   # middle call served from cache


def test_moge_live_end_to_end(tmp_path):  # pragma: no cover - runs only where moge is installed
    pytest.importorskip("moge")
    pytest.importorskip("utils3d")
    img = tmp_path / "scene.png"
    Image.fromarray((np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8))).save(img)
    r = de.estimate_depth(str(img), model_id="Ruicheng/moge-2-vitl-normal", focal_px=200.0)
    d = np.asarray(r.depth)
    assert r.is_metric and d.shape == (96, 128)
    assert np.isfinite(d[np.isfinite(d)]).all()
    assert r.metadata.get("focal_source") == "solve"


def test_moge_cache_key_fragments_on_the_cost_knobs(monkeypatch, tmp_path):
    """resolution_level / max_side / checkpoint_path must split the cache.

    They are quality-and-speed dials, so a second call at a different setting
    that silently returned the FIRST call's map would make the knob look inert —
    the worst failure mode for a dial, because it reads as "no effect" rather
    than as a bug. Focal already fragments (test above); these must too.
    """
    seen = []

    def fake_moge(image_path, *, model_id, device, focal_px,
                  resolution_level=9, max_side=0, checkpoint_path="",
                  tile_side=0, tile_overlap=0.25, report_free_focal=False):
        # Deliberately STRICT (no **kwargs): a stub that swallows unknown
        # arguments hides a caller passing the wrong name, which is exactly how
        # the focal_length_mm/focal_length_mm_hint mix-up got through earlier.
        seen.append((resolution_level, max_side, checkpoint_path, tile_side))
        return DepthResult(depth=np.zeros((4, 4), np.float32), is_metric=True,
                           model_id=model_id, image_width=4, image_height=4)

    monkeypatch.setattr(de, "_estimate_depth_moge", fake_moge)
    de._DEPTH_RESULT_CACHE.clear()

    img = tmp_path / "p.png"
    Image.new("RGB", (4, 4)).save(img)
    mid = "Ruicheng/moge-2-vitl-normal"

    de.estimate_depth(str(img), model_id=mid)                       # defaults
    de.estimate_depth(str(img), model_id=mid)                       # cached — no call
    assert seen == [(9, 0, "", 0)], "identical args must hit the cache"

    de.estimate_depth(str(img), model_id=mid, resolution_level=4)
    de.estimate_depth(str(img), model_id=mid, max_side=2048)
    de.estimate_depth(str(img), model_id=mid, checkpoint_path="/tmp/model.pt")
    de.estimate_depth(str(img), model_id=mid, tile_side=1024)
    assert seen == [(9, 0, "", 0), (4, 0, "", 0), (9, 2048, "", 0),
                    (9, 0, "/tmp/model.pt", 0), (9, 0, "", 1024)]

    # tile_overlap only matters when tiling is on, but it still has to split the
    # cache: two overlaps at the same tile_side are different renders.
    de.estimate_depth(str(img), model_id=mid, tile_side=1024, tile_overlap=0.4)
    assert len(seen) == 6, "tile_overlap did not fragment the cache"


def test_non_moge_models_are_not_fragmented_by_moge_knobs(monkeypatch, tmp_path):
    """The knobs are MoGe-only, so they must NOT split other backends' cache —
    otherwise a DA/V2 re-run would miss the cache for a parameter it ignores."""
    calls = {"n": 0}

    def fake_v2(image_path, *, model_id, device):
        calls["n"] += 1
        return DepthResult(depth=np.zeros((4, 4), np.float32), is_metric=False,
                           model_id=model_id, image_width=4, image_height=4)

    monkeypatch.setattr(de, "_estimate_depth_v2", fake_v2)
    de._DEPTH_RESULT_CACHE.clear()

    img = tmp_path / "q.png"
    Image.new("RGB", (4, 4)).save(img)
    de.estimate_depth(str(img), model_id=de.DEFAULT_RELATIVE_MODEL, resolution_level=3)
    de.estimate_depth(str(img), model_id=de.DEFAULT_RELATIVE_MODEL, max_side=512)
    assert calls["n"] == 1, "MoGe-only knobs must not fragment other backends"
