"""LTX ComfyUI adapter — fully offline (monkeypatched HTTP)."""
from __future__ import annotations

import json

import pytest

from atlas_camera.dynamic.generators import TemporalGenerationConfig
from atlas_camera.dynamic import ltx_comfy as L
from atlas_camera.dynamic.ltx_comfy import LTXComfyGenerator


API_TEMPLATE = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
    "2": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "{PROMPT}", "clip": ["9", 0]}},
    "3": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "static, watermark", "clip": ["9", 0]}},
    "4": {"class_type": "EmptyLTXVLatentVideo",
          "inputs": {"width": 768, "height": 512, "length": 97}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
}

OI = {name: {} for name in
      ("LoadImage", "CLIPTextEncode", "EmptyLTXVLatentVideo",
       "KSampler", "SaveImage")}


@pytest.fixture
def template(tmp_path):
    p = tmp_path / "ltx_template.json"
    p.write_text(json.dumps(API_TEMPLATE), encoding="utf-8")
    return p


def test_unavailable_without_template():
    gen = LTXComfyGenerator(template_path=None)
    gen.template_path = None
    ok, reason = gen.available()
    assert ok is False
    assert "template" in reason.lower()


def test_unavailable_when_host_down(template, monkeypatch):
    gen = LTXComfyGenerator(template_path=str(template), host="127.0.0.1:9")

    def boom(url, payload=None, timeout=60):
        raise OSError("connection refused")
    monkeypatch.setattr(gen._C, "http_json", boom)
    ok, reason = gen.available()
    assert ok is False and "not reachable" in reason


def test_unavailable_when_node_types_missing(template, monkeypatch):
    gen = LTXComfyGenerator(template_path=str(template))
    partial_oi = {k: v for k, v in OI.items() if k != "EmptyLTXVLatentVideo"}
    monkeypatch.setattr(gen._C, "http_json",
                        lambda url, payload=None, timeout=60: partial_oi)
    ok, reason = gen.available()
    assert ok is False and "EmptyLTXVLatentVideo" in reason


def _wire_success(gen, monkeypatch, tmp_path, *, completed=True, errors=None):
    captured = {}

    def fake_http(url, payload=None, timeout=60):
        if url.endswith("/object_info"):
            return OI
        if "/history/" in url:
            return {"pid1": {"outputs": {"6": {"images": [
                {"filename": f"f_{i:05d}.png", "subfolder": "", "type": "output"}
                for i in range(4)]}}}}
        raise AssertionError(url)

    def fake_queue(api, host, timeout=1800, poll_s=5.0):
        captured["api"] = api
        return {"completed": completed, "prompt_id": "pid1",
                "errors": errors or [], "output_nodes": ["6"], "reports": {}}

    monkeypatch.setattr(gen._C, "http_json", fake_http)
    monkeypatch.setattr(gen._C, "fetch_object_info",
                        lambda host, timeout=120: OI)
    monkeypatch.setattr(gen._C, "upload_image",
                        lambda path, host: "uploaded_crop.png")
    monkeypatch.setattr(gen._C, "queue_and_wait", fake_queue)
    monkeypatch.setattr(
        LTXComfyGenerator, "_download",
        lambda self, image, dest: dest.write_bytes(b"\x89PNG fake"))
    return captured


class _Plate:
    source_roi = None
    crop_camera = None


def _package(tmp_path):
    pkg = tmp_path / "WATER_0001"
    (pkg / "source").mkdir(parents=True)
    (pkg / "source" / "crop.png").write_bytes(b"\x89PNG fake")
    return pkg


def test_generate_success(template, monkeypatch, tmp_path):
    gen = LTXComfyGenerator(template_path=str(template))
    captured = _wire_success(gen, monkeypatch, tmp_path)
    pkg = _package(tmp_path)
    config = TemporalGenerationConfig(prompt="ocean waves", seed=42,
                                      fps=24.0, frame_count=97)
    result = gen.generate(_Plate(), pkg, config)
    assert result.status == "ok", result.warnings
    assert result.frame_count == 4
    assert len(result.frame_paths) == 4
    assert (pkg / "generated" / "frame_0000.png").exists()
    assert result.fps == 24.0
    assert result.seed == 42
    assert result.metadata["camera_preservation"] == "unverified_i2v"
    api = captured["api"]
    assert api["1"]["inputs"]["image"] == "uploaded_crop.png"
    assert api["2"]["inputs"]["text"] == "ocean waves"        # marker node
    assert api["3"]["inputs"]["text"] == "static, watermark"  # negative kept
    assert api["4"]["inputs"]["length"] == 97
    assert api["5"]["inputs"]["seed"] == 42


def test_generate_failure_carries_errors(template, monkeypatch, tmp_path):
    gen = LTXComfyGenerator(template_path=str(template))
    _wire_success(gen, monkeypatch, tmp_path, completed=False,
                  errors=["KSampler (node 5): OOM"])
    pkg = _package(tmp_path)
    result = gen.generate(_Plate(), pkg, TemporalGenerationConfig())
    assert result.status == "failed"
    assert any("OOM" in w for w in result.warnings)


def test_generate_not_available_short_circuits(tmp_path, monkeypatch):
    gen = LTXComfyGenerator(template_path=str(tmp_path / "missing.json"))
    result = gen.generate(_Plate(), tmp_path, TemporalGenerationConfig())
    assert result.status == "not_available"
    assert result.warnings
