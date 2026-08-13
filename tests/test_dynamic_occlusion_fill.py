"""Occlusion-fill: guide+mask rendering + LTX inpaint plumbing (offline)."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from atlas_camera.core.camera_math import look_at_view_matrix, orbit_camera
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasProjectionScene,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
)
from atlas_camera.dynamic.generators import TemporalGenerationConfig
from atlas_camera.dynamic.ltx_comfy import LTXComfyGenerator
from atlas_camera.dynamic.occlusion_fill import (
    LTX_INPAINT_GREEN,
    render_disocclusion_sequence,
    write_sequences,
)


def _solve_with_wall():
    """Camera at origin looking -Z; a textured quad wall at z=-10."""
    view, world, rot3 = look_at_view_matrix((0.0, 1.6, 0.0), (0.0, 1.6, -10.0))
    cam = LatentCamera(
        intrinsics=build_intrinsics(image_width=128, image_height=96,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 1.6, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view))
    verts = [-6.0, -2.0, -10.0, 6.0, -2.0, -10.0, 6.0, 6.0, -10.0,
             -6.0, 6.0, -10.0]
    faces = [0, 1, 2, 0, 2, 3]
    uvs = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    wall = AtlasProxyPrimitive(
        name="wall", primitive_type="mesh",
        metadata={"vertices": verts, "faces": faces, "uvs": uvs})
    return AtlasSolve(camera=cam, image_width=128, image_height=96,
                      projection_scene=AtlasProjectionScene(
                          proxy_geometry=[wall]))


def test_zero_orbit_has_full_coverage():
    solve = _solve_with_wall()
    source = np.full((96, 128, 3), 128, dtype=np.uint8)
    frames = render_disocclusion_sequence(
        solve, source, [solve.camera.extrinsics.camera_view_matrix],
        resolution=128, hole_dilate_px=0)
    _guide, mask, coverage = frames[0]
    assert coverage < 0.15          # wall fills most of the frame
    assert mask.mean() < 255 * 0.15


def test_orbit_reveals_holes_with_sentinel():
    solve = _solve_with_wall()
    source = np.full((96, 128, 3), 128, dtype=np.uint8)
    from atlas_camera.core.camera_math import ground_lookat_pivot
    pivot = ground_lookat_pivot(solve.camera.extrinsics)
    moved = orbit_camera(solve.camera.extrinsics, pivot,
                         d_azimuth_deg=35.0, d_elevation_deg=0.0)
    frames = render_disocclusion_sequence(
        solve, source, [moved.camera_view_matrix],
        resolution=128, hole_dilate_px=2)
    guide, mask, coverage = frames[0]
    assert coverage > 0.02
    hole = mask > 127
    sentinel = np.round(np.asarray(LTX_INPAINT_GREEN) * 255)
    assert np.allclose(guide[hole][0], sentinel, atol=1)


def test_write_sequences(tmp_path):
    solve = _solve_with_wall()
    source = np.full((96, 128, 3), 128, dtype=np.uint8)
    frames = render_disocclusion_sequence(
        solve, source, [solve.camera.extrinsics.camera_view_matrix] * 3,
        resolution=64)
    guides, masks = write_sequences(frames, tmp_path)
    assert len(guides) == 3 and len(masks) == 3
    with Image.open(masks[0]) as im:
        assert im.mode == "L"


def test_solve_without_meshes_raises():
    view, world, rot3 = look_at_view_matrix((0.0, 1.6, 0.0), (0.0, 1.6, -10.0))
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=build_intrinsics(image_width=64, image_height=64,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_world_matrix=world,
                                   camera_view_matrix=view)),
        image_width=64, image_height=64)
    with pytest.raises(ValueError):
        render_disocclusion_sequence(
            solve, np.zeros((64, 64, 3), np.uint8),
            [view], resolution=64)


# ------------------------------------------------- adapter upload markers

TEMPLATE = {
    "1": {"class_type": "LoadVideo", "inputs": {"file": "{GUIDE_VIDEO}"}},
    "2": {"class_type": "LoadVideo", "inputs": {"file": "{MASK_VIDEO}"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "{PROMPT}"}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
}
OI = {k: {} for k in ("LoadVideo", "CLIPTextEncode", "SaveImage")}


def test_upload_markers_route_two_videos(tmp_path, monkeypatch):
    template = tmp_path / "t.json"
    template.write_text(json.dumps(TEMPLATE), encoding="utf-8")
    gen = LTXComfyGenerator(template_path=str(template))
    captured, uploads = {}, []

    def fake_http(url, payload=None, timeout=60):
        if url.endswith("/object_info"):
            return OI
        if "/history/" in url:
            return {"pid1": {"outputs": {"6": {"images": [
                {"filename": "f.png", "subfolder": "", "type": "output"}]}}}}
        raise AssertionError(url)

    monkeypatch.setattr(gen._C, "http_json", fake_http)
    monkeypatch.setattr(gen._C, "fetch_object_info", lambda h, timeout=120: OI)
    monkeypatch.setattr(
        gen._C, "upload_image",
        lambda path, host: uploads.append(str(path)) or
        ("guide_up.mp4" if "guide" in str(path) else
         "mask_up.mp4" if "mask" in str(path) else "crop.png"))
    monkeypatch.setattr(
        LTXComfyGenerator, "_download",
        lambda self, image, dest: dest.write_bytes(b"png"))

    def fake_queue(api, host, timeout=1800, poll_s=5.0):
        captured["api"] = api
        return {"completed": True, "prompt_id": "pid1", "errors": [],
                "output_nodes": ["6"], "reports": {}}
    monkeypatch.setattr(gen._C, "queue_and_wait", fake_queue)

    pkg = tmp_path / "occ"
    (pkg / "source").mkdir(parents=True)
    (pkg / "source" / "crop.png").write_bytes(b"\x89PNG fake")
    (pkg / "guide.mp4").write_bytes(b"g")
    (pkg / "mask.mp4").write_bytes(b"m")

    class _P:
        source_roi = None
        crop_camera = None

    config = TemporalGenerationConfig(
        prompt="fill the revealed cliff",
        extra={"upload_markers": {"{GUIDE_VIDEO}": pkg / "guide.mp4",
                                  "{MASK_VIDEO}": pkg / "mask.mp4"}})
    result = gen.generate(_P(), pkg, config)
    assert result.status == "ok", result.warnings
    assert captured["api"]["1"]["inputs"]["file"] == "guide_up.mp4"
    assert captured["api"]["2"]["inputs"]["file"] == "mask_up.mp4"
    assert captured["api"]["3"]["inputs"]["text"] == "fill the revealed cliff"
