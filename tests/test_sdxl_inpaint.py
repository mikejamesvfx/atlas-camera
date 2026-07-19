"""Perspective-conditioning regressions for AtlasSDXLInpaint."""

import pytest

torch = pytest.importorskip("torch")

import atlas_camera.comfy.nodes as nodes_mod
from atlas_camera.comfy.nodes import AtlasSDXLInpaint, _MiniGraphBuilder


_REGISTRY = {name: object for name in (
    "CheckpointLoaderSimple", "CLIPTextEncode", "InpaintModelConditioning",
    "KSampler", "VAEDecode",
)}


def _expand(monkeypatch, preserve_perspective=True):
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: _REGISTRY)
    monkeypatch.setattr(nodes_mod, "_graph_builder", _MiniGraphBuilder)
    image = torch.zeros((1, 64, 48, 3), dtype=torch.float32)
    mask = torch.ones((1, 64, 48), dtype=torch.float32)
    return AtlasSDXLInpaint().expand_sdxl(
        image, mask, "sdxl.safetensors", "matching brick facade",
        "warped, seams", denoise=0.5,
        preserve_perspective=preserve_perspective)


def _clip_texts(expansion):
    return [node["inputs"]["text"] for node in expansion["expand"].values()
            if node["class_type"] == "CLIPTextEncode"]


def test_perspective_guidance_is_appended_by_default(monkeypatch):
    out = _expand(monkeypatch)
    positive, negative = _clip_texts(out)
    assert "exact source camera viewpoint" in positive
    assert "same vanishing directions" in positive
    assert "front elevation" in negative
    assert "orthographic view" in negative
    assert "perspective=preserve" in out["result"][1]


def test_perspective_guidance_can_be_disabled(monkeypatch):
    out = _expand(monkeypatch, preserve_perspective=False)
    assert _clip_texts(out) == ["matching brick facade", "warped, seams"]
    assert "perspective=prompt-only" in out["result"][1]
