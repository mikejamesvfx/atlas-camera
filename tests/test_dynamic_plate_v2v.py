"""v2v mode: CLI render pass + adapter video-input plumbing (offline)."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.io import save_solve_json
from atlas_camera.core.schema import AtlasExtrinsics, AtlasSolve, LatentCamera
from atlas_camera.dynamic.cli import main
from atlas_camera.dynamic.generators import TemporalGenerationConfig
from atlas_camera.dynamic.ltx_comfy import LTXComfyGenerator


@pytest.fixture
def shot(tmp_path):
    rng = np.random.default_rng(0)
    img_path = tmp_path / "castle.png"
    Image.fromarray((rng.random((360, 640, 3)) * 255).astype(np.uint8)
                    ).save(img_path)
    matte = np.zeros((360, 640), dtype=np.uint8)
    matte[250:360, :] = 255
    matte_path = tmp_path / "ocean_mask.png"
    Image.fromarray(matte, mode="L").save(matte_path)
    view, world, rot3 = look_at_view_matrix((0.0, 12.0, 0.0), (0.0, 0.0, -50.0))
    solve = AtlasSolve(
        camera=LatentCamera(
            intrinsics=build_intrinsics(image_width=640, image_height=360,
                                        focal_length_mm=32.0),
            extrinsics=AtlasExtrinsics(camera_position=(0.0, 12.0, 0.0),
                                       camera_rotation_matrix=rot3,
                                       camera_world_matrix=world,
                                       camera_view_matrix=view)),
        image_width=640, image_height=360)
    solve_path = tmp_path / "atlas_solve.json"
    save_solve_json(solve, solve_path)
    return {"image": img_path, "matte": matte_path, "solve": solve_path,
            "out": tmp_path / "shot"}


def test_cli_staged_create_then_render(shot, capsys):
    """Staged-by-default: create prepares only; render is a separate pass."""
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]), "--solve", str(shot["solve"]),
               "--out", str(shot["out"]), "--mode", "v2v"])
    out = capsys.readouterr().out
    assert rc == 0, out
    pkg = shot["out"] / "dynamic" / "WATER_0001"
    assert not (pkg / "rendered").exists()   # create did NOT render

    rc = main(["render", "--package", str(pkg),
               "--dolly", "0.5,0.0,-0.5", "--frames", "8"])
    out = capsys.readouterr().out
    assert rc == 0, out
    frames = sorted((pkg / "rendered").glob("frame_*.png"))
    assert len(frames) == 8
    manifest = json.loads((pkg / "manifest.json").read_text())
    assert manifest["metadata"]["rendered_input"]["dolly_m"] == [0.5, 0.0, -0.5]
    # frame 0 is the source pose; final frame differs (camera moved)
    a = np.asarray(Image.open(frames[0])).astype(float)
    b = np.asarray(Image.open(frames[-1])).astype(float)
    assert float(np.abs(a - b).mean()) > 0.5


def test_cli_render_gen_size_downscales(shot, capsys):
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]), "--solve", str(shot["solve"]),
               "--out", str(shot["out"]), "--mode", "v2v"])
    assert rc == 0
    pkg = shot["out"] / "dynamic" / "WATER_0001"
    rc = main(["render", "--package", str(pkg), "--frames", "3",
               "--gen-width", "320", "--gen-height", "96"])
    capsys.readouterr()
    assert rc == 0
    with Image.open(pkg / "rendered" / "frame_0000.png") as im:
        assert im.size == (320, 96)


def test_cli_v2v_inline_render_with_flag(shot, capsys):
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]), "--solve", str(shot["solve"]),
               "--out", str(shot["out"]), "--mode", "v2v", "--render",
               "--dolly", "0.5,0.0,-0.5", "--frames", "8"])
    out = capsys.readouterr().out
    assert rc == 0, out
    pkg = shot["out"] / "dynamic" / "WATER_0001"
    assert len(list((pkg / "rendered").glob("frame_*.png"))) == 8


def test_cli_v2v_bad_dolly_errors(shot, capsys):
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]), "--solve", str(shot["solve"]),
               "--out", str(shot["out"]), "--mode", "v2v", "--render",
               "--dolly", "oops"])
    out = capsys.readouterr().out
    assert rc == 1 and "--dolly" in out


def test_cli_generate_stage_not_available(shot, capsys, monkeypatch):
    monkeypatch.delenv("ATLAS_LTX_TEMPLATE", raising=False)
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]), "--solve", str(shot["solve"]),
               "--out", str(shot["out"])])
    assert rc == 0
    pkg = shot["out"] / "dynamic" / "WATER_0001"
    rc = main(["generate", "--package", str(pkg), "--generator", "ltx"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "status = not_available" in out
    assert (pkg / "generated" / "generation_result.json").exists()


API_TEMPLATE = {
    "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": ""}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "{PROMPT}"}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 0}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
}
OI = {k: {} for k in ("VHS_LoadVideo", "CLIPTextEncode", "KSampler",
                      "SaveImage")}


def _v2v_package(tmp_path):
    pkg = tmp_path / "WATER_0001"
    (pkg / "source").mkdir(parents=True)
    (pkg / "source" / "crop.png").write_bytes(b"\x89PNG fake")
    (pkg / "rendered").mkdir()
    for i in range(3):
        (pkg / "rendered" / f"frame_{i:04d}.png").write_bytes(b"\x89PNG fake")
    return pkg


class _Plate:
    source_roi = None
    crop_camera = None


def test_adapter_v2v_uploads_rendered_video(tmp_path, monkeypatch):
    template = tmp_path / "t.json"
    template.write_text(json.dumps(API_TEMPLATE), encoding="utf-8")
    gen = LTXComfyGenerator(template_path=str(template))
    captured = {}

    def fake_http(url, payload=None, timeout=60):
        if url.endswith("/object_info"):
            return OI
        if "/history/" in url:
            return {"pid1": {"outputs": {"6": {"images": [
                {"filename": "f.png", "subfolder": "", "type": "output"}]}}}}
        raise AssertionError(url)

    monkeypatch.setattr(gen._C, "http_json", fake_http)
    monkeypatch.setattr(gen._C, "fetch_object_info", lambda h, timeout=120: OI)
    uploads = []
    monkeypatch.setattr(gen._C, "upload_image",
                        lambda path, host: uploads.append(path) or
                        ("video.mp4" if path.endswith(".mp4") else "crop.png"))
    monkeypatch.setattr(
        gen, "_encode_rendered_mp4",
        lambda rendered, out, fps: (out.write_bytes(b"mp4"), None)[1])
    monkeypatch.setattr(
        LTXComfyGenerator, "_download",
        lambda self, image, dest: dest.write_bytes(b"png"))

    def fake_queue(api, host, timeout=1800, poll_s=5.0):
        captured["api"] = api
        return {"completed": True, "prompt_id": "pid1", "errors": [],
                "output_nodes": ["6"], "reports": {}}
    monkeypatch.setattr(gen._C, "queue_and_wait", fake_queue)

    pkg = _v2v_package(tmp_path)
    result = gen.generate(_Plate(), pkg,
                          TemporalGenerationConfig(mode="video_to_video",
                                                   prompt="waves"))
    assert result.status == "ok", result.warnings
    assert result.metadata["camera_preservation"] == "atlas_rendered_v2v"
    assert captured["api"]["1"]["inputs"]["video"] == "video.mp4"
    assert any(p.endswith(".mp4") for p in uploads)


def test_adapter_v2v_without_rendered_frames_fails_cleanly(tmp_path, monkeypatch):
    template = tmp_path / "t.json"
    template.write_text(json.dumps(API_TEMPLATE), encoding="utf-8")
    gen = LTXComfyGenerator(template_path=str(template))
    monkeypatch.setattr(gen._C, "http_json",
                        lambda url, payload=None, timeout=60: OI)
    monkeypatch.setattr(gen._C, "fetch_object_info", lambda h, timeout=120: OI)
    monkeypatch.setattr(gen._C, "upload_image", lambda path, host: "crop.png")
    pkg = tmp_path / "WATER_0002"
    (pkg / "source").mkdir(parents=True)
    (pkg / "source" / "crop.png").write_bytes(b"\x89PNG fake")
    result = gen.generate(_Plate(), pkg,
                          TemporalGenerationConfig(mode="video_to_video"))
    assert result.status == "failed"
    assert any("rendered" in w for w in result.warnings)
