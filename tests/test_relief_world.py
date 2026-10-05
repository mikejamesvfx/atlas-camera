"""Public relief-world facade: ``atlas.recover_relief_world`` / ``export_relief_world_glb``.

The learned prior and the depth model are replaced with deterministic stubs so the
owner's orchestration is tested without a GPU: a camera 1.6 m above a ground plane,
looking level. The real-model path is proven end to end by Atlas Showcase's GPU bake,
which must stay byte-identical across the move to this facade.
"""

from __future__ import annotations

import io
import json
import math
import struct

import numpy as np
import pytest

import atlas
from atlas_camera import relief_world as rw
from atlas_camera.inference.depth_estimator import DEFAULT_METRIC_OUTDOOR, DepthResult
from atlas_camera.inference.learned_prior import CameraPrior

W, H = 160, 120
FOCAL = 150.0


@pytest.fixture
def photo(tmp_path):
    from PIL import Image

    yy, xx = np.mgrid[0:H, 0:W]
    arr = np.dstack([xx * 255 // W, yy * 255 // H, np.full((H, W), 90)]).astype(np.uint8)
    path = tmp_path / "level_ground.png"
    Image.fromarray(arr, "RGB").save(path)
    return path


@pytest.fixture
def stub_models(monkeypatch):
    calls = {}

    def prior(image_path, *, device=None, weights="pinhole"):
        calls["prior"] = {"image_path": image_path, "device": device}
        fov_h = math.degrees(2 * math.atan(W / 2 / FOCAL))
        fov_v = math.degrees(2 * math.atan(H / 2 / FOCAL))
        return CameraPrior(focal_px=FOCAL, fov_h_deg=fov_h, fov_v_deg=fov_v, roll_deg=0.0,
                           pitch_deg=0.0, up_cam=(0.0, 1.0, 0.0),
                           principal_point_px=(W / 2, H / 2), image_width=W, image_height=H)

    def depth(image_path, *, model_id=DEFAULT_METRIC_OUTDOOR, device=None, focal_px=None, **_):
        calls["depth"] = {"model_id": model_id, "device": device, "focal_px": focal_px}
        rows = np.arange(H, dtype=np.float64)[:, None]
        ground = 1.6 * FOCAL / np.maximum(rows - H / 2, 1e-3)
        z = np.where(rows > H / 2 + 1, np.minimum(ground, 20.0), 20.0) * np.ones((1, W))
        return DepthResult(depth=z.astype(np.float32), is_metric=True, model_id=model_id,
                           image_width=W, image_height=H)

    monkeypatch.setattr("atlas_camera.inference.learned_prior.estimate_camera_prior", prior)
    monkeypatch.setattr("atlas_camera.inference.depth_estimator.estimate_depth", depth)
    return calls


def test_the_facade_is_public_through_import_atlas():
    for name in ("recover_relief_world", "export_relief_world_glb", "ReliefWorld",
                 "ReliefWorldCamera", "ReliefWorldScale", "ReliefWorldMeshStats",
                 "RELIEF_WORLD_INPUT_SUFFIXES"):
        assert name in atlas.__all__ if hasattr(atlas, "__all__") else True
        assert getattr(atlas, name) is getattr(rw, name)


def test_default_depth_model_is_the_depth_estimators_default():
    assert rw.DEFAULT_RELIEF_DEPTH_MODEL == DEFAULT_METRIC_OUTDOOR


def test_recover_relief_world_runs_the_owner_pipeline(photo, stub_models):
    world = atlas.recover_relief_world(photo, depth_model="test/metric", device="cpu",
                                       grid_long_edge=32)

    assert stub_models["prior"]["device"] == "cpu"
    assert stub_models["depth"] == {"model_id": "test/metric", "device": "cpu", "focal_px": FOCAL}

    cam = world.camera
    assert (cam.image_width, cam.image_height) == (W, H)
    assert cam.fx == pytest.approx(FOCAL) and cam.focal_source == "geocalib"
    assert len(cam.view_matrix) == 4 and all(len(r) == 4 for r in cam.view_matrix)
    assert cam.horizon_y == pytest.approx(H / 2)
    assert world.solve.camera.intrinsics.image_width == W

    assert world.scale.depth_model == "test/metric" and world.scale.depth_is_metric is True
    assert world.scale.health.status in ("measured", "manual", "assumed", "unknown")
    assert world.mesh_stats.grid_long_edge == 32
    assert world.mesh_stats.n_faces == world.mesh.stats["n_faces"] > 0
    assert world.capture == {}
    assert world.display_image.size == (W, H)
    assert "mesh=" not in repr(world) and "display_image=" not in repr(world)


def test_the_recorded_camera_describes_the_mesh(photo, stub_models):
    """Every relief vertex samples its own source pixel, so projecting it through the
    returned camera must land on its UV. This is what makes the camera fields a contract."""
    world = atlas.recover_relief_world(photo, grid_long_edge=32)
    cam = world.camera
    vm = np.asarray(cam.view_matrix)
    verts = np.asarray(world.mesh.vertices, dtype=float)
    uvs = np.asarray(world.mesh.uvs, dtype=float)
    p = (vm @ np.c_[verts, np.ones(len(verts))].T).T
    front = -p[:, 2] > 1e-6
    px = cam.fx * p[front, 0] / -p[front, 2] + cam.cx
    py = cam.cy - cam.fy * p[front, 1] / -p[front, 2]
    err = np.hypot(px - uvs[front, 0] * W, py - (1.0 - uvs[front, 1]) * H)
    assert front.mean() > 0.9
    assert np.percentile(err, 95) < 3.0


def test_to_dict_forms_are_json_safe_and_ordered(photo, stub_models):
    world = atlas.recover_relief_world(photo, grid_long_edge=32)
    camera = world.camera.to_dict()
    assert list(camera) == ["fx", "fy", "cx", "cy", "image_width", "image_height",
                            "view_matrix", "pitch_deg", "horizon_y", "focal_source"]
    scale = world.scale.to_dict()
    assert list(scale)[:len(world.scale.health.to_dict())] == list(world.scale.health.to_dict())
    assert list(scale)[-6:] == ["ground_scale", "ground_inliers", "measured_camera_height_m",
                                "measured_height_confidence", "depth_model", "depth_is_metric"]
    assert list(world.mesh_stats.to_dict()) == ["grid_long_edge", "n_faces", "torn_fraction"]
    json.dumps({"camera": camera, "scale": scale, "mesh": world.mesh_stats.to_dict()})


def _glb_image(path):
    raw = open(path, "rb").read()
    json_len = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + json_len])
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    start = 20 + json_len + 8 + view["byteOffset"]
    return gltf["images"][0]["mimeType"], raw[start:start + view["byteLength"]]


def test_export_relief_world_glb(photo, stub_models, tmp_path):
    from PIL import Image

    world = atlas.recover_relief_world(photo, grid_long_edge=32)

    png = atlas.export_relief_world_glb(world, tmp_path / "png")
    assert png.name == "relief_world.glb"
    mime, blob = _glb_image(png)
    assert mime == "image/png" and Image.open(io.BytesIO(blob)).size == (W, H)

    small = world.display_image.resize((80, 60))
    jpeg = atlas.export_relief_world_glb(world, tmp_path / "jpeg", texture=small, name="world",
                                         texture_format="JPEG")
    mime, blob = _glb_image(jpeg)
    assert jpeg.name == "world.glb" and mime == "image/jpeg" and blob[:3] == b"\xff\xd8\xff"
    assert Image.open(io.BytesIO(blob)).size == (80, 60)

    with pytest.raises(TypeError):
        atlas.export_relief_world_glb(world.mesh, tmp_path / "bad")


def test_unsupported_input_is_refused_before_any_model_runs(tmp_path, monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("a model ran for an input that should have been refused")

    monkeypatch.setattr("atlas_camera.inference.learned_prior.estimate_camera_prior", forbidden)
    monkeypatch.setattr("atlas_camera.inference.depth_estimator.estimate_depth", forbidden)
    for name in ("capture.r3d", "notes.txt", "plate.exr"):
        (tmp_path / name).write_bytes(b"x")
        with pytest.raises(ValueError, match="does not accept"):
            atlas.recover_relief_world(tmp_path / name)
    with pytest.raises(ValueError, match="grid_long_edge"):
        (tmp_path / "ok.png").write_bytes(b"x")
        atlas.recover_relief_world(tmp_path / "ok.png", grid_long_edge=0)
