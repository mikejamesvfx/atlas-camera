"""MoGe pointmap + predicted-intrinsics passthrough (Phase A of the
pointmap-registration track, 2026-08-16).

A stub MoGe model stands in for the real one so the whole path runs without
weights: `out["points"]`, `out["intrinsics"]`, the fov-free second pass, the
tiled-drop rule, and the scene_health rule that an ECHOED focal never reads as
agreement.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

import atlas_camera.inference.depth_estimator as de
from atlas_camera.core.depth_geometry import (
    back_project_normals, opencv_points_to_atlas_cam, opencv_points_to_world,
)

W, H = 12, 8
FREE_FX_NORM = 0.9   # fx in image widths that the "free" head predicts


class _StubMoge:
    """Mimics MoGeModel.infer: depth == points[...,2]; intrinsics echo fov_x
    when given, else the free-head value."""

    def __init__(self):
        self.calls = []

    def infer(self, tensor, fov_x=None, resolution_level=9):
        self.calls.append((fov_x, resolution_level))
        h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
        z = torch.linspace(2.0, 6.0, h)[:, None].expand(h, w).clone()
        x = torch.linspace(-1.0, 1.0, w)[None, :].expand(h, w).clone()
        y = torch.linspace(-0.5, 0.5, h)[:, None].expand(h, w).clone()
        pts = torch.stack([x, y, z], dim=-1)
        mask = torch.ones(h, w, dtype=torch.bool)
        mask[0, 0] = False
        if fov_x is not None:
            fx_norm = 1.0 / (2.0 * math.tan(math.radians(fov_x) / 2.0))
        else:
            fx_norm = FREE_FX_NORM
        k = torch.tensor([[fx_norm, 0.0, 0.5],
                          [0.0, fx_norm * w / h, 0.5],
                          [0.0, 0.0, 1.0]])
        return {"depth": z, "points": pts, "mask": mask, "intrinsics": k}


@pytest.fixture
def stub(monkeypatch, tmp_path):
    de._DEPTH_RESULT_CACHE.clear()
    model = _StubMoge()
    monkeypatch.setattr(de, "_get_moge_model", lambda *a, **k: model)
    monkeypatch.setattr(de, "resolve_device", lambda d, t: "cpu")
    img = tmp_path / "p.png"
    Image.new("RGB", (W, H), (90, 120, 150)).save(img)
    return model, str(img)


def test_points_and_free_intrinsics_are_recorded(stub):
    model, img = stub
    r = de.estimate_depth(img, model_id="Ruicheng/moge-2-vitl-normal")
    assert r.points is not None and r.points.shape == (H, W, 3)
    # depth IS the pointmap's z, and the mask hole is NaN in both.
    finite = np.isfinite(r.depth)
    assert np.allclose(r.points[..., 2][finite], r.depth[finite])
    assert not finite[0, 0] and not np.isfinite(r.points[0, 0]).any()
    assert r.metadata["has_pointmap"] is True
    assert r.metadata["intrinsics_source"] == "moge_fov_head"
    assert r.metadata["predicted_focal_px"] == pytest.approx(FREE_FX_NORM * W, abs=0.01)
    assert r.metadata["predicted_cx_px"] == pytest.approx(0.5 * W)
    assert "points" not in r.summary() and "normal" not in r.summary()
    assert len(model.calls) == 1


def test_fed_focal_is_marked_as_echo_and_free_pass_is_opt_in(stub):
    model, img = stub
    r = de.estimate_depth(img, model_id="Ruicheng/moge-2-vitl-normal", focal_px=20.0)
    assert r.metadata["focal_source"] == "solve"
    assert r.metadata["intrinsics_source"] == "echo_of_solve"
    assert r.metadata["predicted_focal_px"] == pytest.approx(20.0, rel=1e-3)
    assert "predicted_focal_px_free" not in r.metadata
    assert len(model.calls) == 1

    de._DEPTH_RESULT_CACHE.clear()
    r2 = de.estimate_depth(img, model_id="Ruicheng/moge-2-vitl-normal", focal_px=20.0,
                           report_free_focal=True)
    assert r2.metadata["predicted_focal_px_free"] == pytest.approx(FREE_FX_NORM * W, abs=0.01)
    assert r2.metadata["free_focal_resolution_level"] == de.MOGE_FREE_FOCAL_RESOLUTION_LEVEL
    # Second call was fov-free at the reduced level.
    assert model.calls[-1][0] is None
    assert model.calls[-1][1] == de.MOGE_FREE_FOCAL_RESOLUTION_LEVEL
    # Depth itself came from the FED pass (unchanged).
    assert np.allclose(np.nan_to_num(r2.depth), np.nan_to_num(r.depth))


def test_report_free_focal_fragments_the_result_cache(stub):
    model, img = stub
    de.estimate_depth(img, model_id="Ruicheng/moge-2-vitl-normal", focal_px=20.0)
    de.estimate_depth(img, model_id="Ruicheng/moge-2-vitl-normal", focal_px=20.0,
                      report_free_focal=True)
    assert len(model.calls) == 3   # 1 + (1 + free pass), not a cache hit


def test_tiled_run_drops_the_pointmap_rather_than_lie(stub):
    model, img = stub
    r = de.estimate_depth(img, model_id="Ruicheng/moge-2-vitl-normal", tile_side=6,
                          tile_overlap=0.25)
    assert r.points is None
    assert r.metadata["points_dropped_reason"] == "tiled"
    assert r.metadata["has_pointmap"] is False


def test_opencv_to_atlas_camera_matches_backprojection():
    """A pointmap in OpenCV axes maps onto exactly what Atlas back-projects."""
    fx = fy = 30.0
    cx, cy = W / 2.0, H / 2.0
    depth = np.linspace(2.0, 6.0, H)[:, None].repeat(W, axis=1)
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    # OpenCV pointmap: x right, y down, z forward.
    pts_cv = np.stack([(uu - cx) / fx * depth, (vv - cy) / fy * depth, depth], -1)
    cam = opencv_points_to_atlas_cam(pts_cv)
    assert np.allclose(cam[..., 2], -depth)
    assert np.allclose(cam[..., 1], -(vv - cy) / fy * depth)

    # Arbitrary camera pose; world points must equal back_project_normals.
    ang = 0.3
    R = np.array([[math.cos(ang), 0, math.sin(ang)],
                  [0, 1, 0],
                  [-math.sin(ang), 0, math.cos(ang)]])
    pos = np.array([1.0, 1.6, -2.0])
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = pos
    view = np.linalg.inv(c2w)
    world = opencv_points_to_world(pts_cv, view_matrix=view)
    bp = back_project_normals(depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy)
    assert np.allclose(world, bp.pts_world, atol=1e-9)


def _solve(w=800, h=600, fx=700.0):
    import types  # noqa: F401
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.schema import (
        AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera)
    eye, target = (0.0, 1.6, 0.0), (0.0, 0.5, -10.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(camera_position=eye, camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=w, image_height=h, fx_px=fx, fy_px=fx,
                           cx_px=w / 2, cy_px=h / 2)
    s = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    s.debug_metadata["scale_source"] = "manual_override"
    return s


def _depth(meta, width=800, height=600):
    import types
    return types.SimpleNamespace(
        model_id="Ruicheng/moge-2-vitl-normal", is_metric=True, near=0.5, far=50.0,
        image_width=width, image_height=height, metadata=dict(meta),
        depth=np.full((height, width), 5.0, np.float32))


def _codes(report):
    return [f.code for f in report.flags]


def test_scene_health_ignores_echoed_focal_but_uses_free():
    """The verdict lives in scene_health: an echo of the solve is not evidence."""
    from atlas_camera.core.scene_health import evaluate_scene_health

    # Echo of the solve (focal was FED): no comparison, no camera key.
    rep = evaluate_scene_health(_solve(fx=700.0), _depth(
        {"focal_source": "solve", "predicted_focal_px": 700.0}))
    assert "focal_mismatch" not in _codes(rep)
    assert "depth_focal_px" not in rep.camera
    assert rep.depth["focal_source"] == "solve"

    # Free value present and 2x off: flagged, using the FREE value.
    rep = evaluate_scene_health(_solve(fx=700.0), _depth(
        {"focal_source": "solve", "predicted_focal_px": 700.0,
         "predicted_focal_px_free": 1400.0}))
    assert "focal_mismatch" in _codes(rep)
    assert rep.camera["depth_focal_px"] == pytest.approx(1400.0)

    # sh001-style 15% disagreement is inside the band: recorded, not flagged.
    rep = evaluate_scene_health(_solve(fx=6207.0), _depth(
        {"focal_source": "solve", "predicted_focal_px": 6207.0,
         "predicted_focal_px_free": 5278.0}))
    assert "focal_mismatch" not in _codes(rep)
    assert rep.camera["depth_focal_px"] == pytest.approx(5278.0)

    # No solve focal fed: the plain key IS an independent estimate.
    rep = evaluate_scene_health(_solve(fx=700.0), _depth(
        {"focal_source": "predicted", "predicted_focal_px": 1400.0}))
    assert "focal_mismatch" in _codes(rep)
