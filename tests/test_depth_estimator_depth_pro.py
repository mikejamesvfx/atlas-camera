"""Apple Depth Pro backend: id classification, dispatch routing, and the
scene_health focal cross-check fed by its predicted focal. Mirrors
test_depth_estimator_moge's structure; the verdict tests live here too because
the focal_mismatch flag only exists to consume this backend's metadata.
"""

import types

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.scene_health import evaluate_scene_health
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasSolve,
    LatentCamera,
)
from atlas_camera.inference.depth_estimator import _is_depth_pro_model


def test_depth_pro_model_id_classification():
    assert _is_depth_pro_model("apple/DepthPro-hf")
    assert _is_depth_pro_model("apple/DepthPro")
    assert not _is_depth_pro_model("depth-anything/DA3METRIC-LARGE")
    assert not _is_depth_pro_model("Ruicheng/moge-2-vitl-normal")
    assert not _is_depth_pro_model(
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")


def test_depth_pro_dispatch_routes_by_model_id(monkeypatch, tmp_path):
    """An apple/DepthPro id must reach _estimate_depth_depth_pro only."""
    torch = pytest.importorskip("torch")  # noqa: F841 — estimate_depth needs it
    pytest.importorskip("PIL")
    from PIL import Image

    import atlas_camera.inference.depth_estimator as de

    de._DEPTH_RESULT_CACHE.clear()
    calls = {"depth_pro": 0}

    def fake_depth_pro(image_path, *, model_id, device):
        calls["depth_pro"] += 1
        return de.DepthResult(
            depth=np.ones((4, 4), np.float32), is_metric=True,
            model_id=model_id, image_width=4, image_height=4,
            metadata={"predicted_focal_px": 3.5})

    def exploding(*a, **k):
        raise AssertionError("wrong backend dispatched")

    monkeypatch.setattr(de, "_estimate_depth_depth_pro", fake_depth_pro)
    monkeypatch.setattr(de, "_estimate_depth_v2", exploding)
    monkeypatch.setattr(de, "_estimate_depth_da3", exploding)
    monkeypatch.setattr(de, "_estimate_depth_moge", exploding)

    img = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(img)
    # focal_px must NOT fragment the cache (Depth Pro ignores it): two calls
    # with different focals are one inference + one cache hit.
    de.estimate_depth(str(img), model_id="apple/DepthPro-hf", focal_px=500.0)
    r = de.estimate_depth(str(img), model_id="apple/DepthPro-hf", focal_px=900.0)
    assert calls["depth_pro"] == 1
    assert r.metadata["predicted_focal_px"] == 3.5


# ---------------------------------------------------------------------------
# scene_health focal cross-check (the ONLY verdict site for the estimate)
# ---------------------------------------------------------------------------

def _solve(w=800, h=600, fx=700.0):
    eye, target = (0.0, 1.6, 0.0), (0.0, 0.5, -10.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(camera_position=eye, camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=w, image_height=h, fx_px=fx, fy_px=fx,
                           cx_px=w / 2, cy_px=h / 2)
    s = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    s.debug_metadata["scale_source"] = "manual_override"
    return s


def _depth(width=800, height=600, predicted_focal_px=None):
    meta = {}
    if predicted_focal_px is not None:
        meta["predicted_focal_px"] = predicted_focal_px
    return types.SimpleNamespace(
        model_id="apple/DepthPro-hf", is_metric=True, near=0.5, far=50.0,
        image_width=width, image_height=height, metadata=meta,
        depth=np.full((height, width), 5.0, np.float32))


def _codes(report):
    return [f.code for f in report.flags]


def test_focal_agreement_produces_no_flag():
    report = evaluate_scene_health(_solve(fx=700.0),
                                   _depth(predicted_focal_px=700.0))
    assert "focal_mismatch" not in _codes(report)
    assert report.camera["depth_focal_px"] == pytest.approx(700.0)
    assert report.depth["predicted_focal_px"] == pytest.approx(700.0)


def test_focal_two_x_off_flags_warn():
    report = evaluate_scene_health(_solve(fx=700.0),
                                   _depth(predicted_focal_px=1400.0))
    flag = next(f for f in report.flags if f.code == "focal_mismatch")
    assert flag.severity == "warn"
    assert report.level in ("warn", "fail")
    # Message names both values and the depth model, per the parity doctrine.
    assert "700" in flag.message and "1400" in flag.message
    assert "apple/DepthPro-hf" in flag.message


def test_focal_rescales_from_depth_resolution():
    # Prediction at half-resolution depth: 350 px at width 400 == 700 px at
    # the solve's 800 — must agree after rescale.
    report = evaluate_scene_health(
        _solve(fx=700.0), _depth(width=400, height=300, predicted_focal_px=350.0))
    assert "focal_mismatch" not in _codes(report)
    assert report.camera["depth_focal_px"] == pytest.approx(700.0)


def test_no_predicted_focal_no_flag_no_camera_key():
    report = evaluate_scene_health(_solve(), _depth())
    assert "focal_mismatch" not in _codes(report)
    assert "depth_focal_px" not in report.camera


def test_boundary_ratio_inside_band_passes():
    # 1.30x is inside the [0.75, 1.33] band.
    report = evaluate_scene_health(_solve(fx=700.0),
                                   _depth(predicted_focal_px=910.0))
    assert "focal_mismatch" not in _codes(report)
