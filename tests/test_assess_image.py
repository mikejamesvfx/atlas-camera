"""Tests for the VLM image-assessment pre-flight (inference.assessor +
AtlasAssessImage) — the pause-gated setup advisor at the head of shipped
workflows. The VLM itself is never called here: connectivity failure is
tested against a dead local port, everything else via monkeypatched results.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import atlas_camera.comfy.nodes as nodes_mod
import atlas_camera.inference.assessor as assessor_mod
from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, AtlasAssessImage
from atlas_camera.inference.assessor import (
    ATLAS_ASSESSMENT_SYSTEM_PROMPT,
    AssessmentResult,
    assess_image,
    format_assessment_report,
)

_PAYLOAD = {
    "scene_summary": "Desert vista with buttes; clear layered depth.",
    "viability": {"score_0_10": 7, "max_orbit_deg": 20, "dolly_ok": True,
                  "notes": "bg parallax clean; near ground stretches first"},
    "layers": [
        {"name": "sky", "role": "sky", "notes": "clean horizon"},
        {"name": "bg", "role": "background", "near_pct": 0.35, "far_pct": 1.0,
         "needs_inpaint": True, "fill_occluded": True, "notes": "buttes"},
        {"name": "fg", "role": "foreground", "near_pct": 0.0, "far_pct": 0.35,
         "needs_inpaint": False, "notes": "desert floor"},
    ],
    "recommended_settings": {
        "depth_model": "outdoor", "scene_type": "organic",
        "relief_grid": 256, "depth_edge_rel": 0.5,
        "sky": {"use_sky_dome": True, "sam_prompt": "sky"},
        "patch": {"recommended": True, "suggested_views": ["front-right quarter view"],
                  "notes": "for orbits past 20 deg"},
        "scale_reference": {"present": False, "object": "", "notes": ""},
    },
    "warnings": ["thin rock spires may tear"],
}


def test_node_registered():
    assert NODE_CLASS_MAPPINGS["AtlasAssessImage"] is AtlasAssessImage
    assert AtlasAssessImage.RETURN_TYPES == ("IMAGE", "STRING", "STRING")
    assert AtlasAssessImage.RETURN_NAMES == ("image", "report", "settings_json")


def test_system_prompt_covers_the_settings_surface():
    # The prompt IS the product here — sanity-check the decision rules the
    # node's docstring promises are actually present.
    for token in ("scene_type", "depth_model", "fill_occluded", "embed_matte",
                  "relief_grid", "depth_edge_rel", "max_orbit_deg", "sky",
                  "near_pct", "multi-angle", "OUTPUT FORMAT"):
        assert token in ATLAS_ASSESSMENT_SYSTEM_PROMPT, token


def test_report_formatting_from_payload():
    report = format_assessment_report(_PAYLOAD, provider="ollama", model="gemma3:4b")
    assert "7/10" in report and "20 deg" in report
    assert "bg" in report and "fill_occluded" in report
    assert "SAM prompt: sky" in report
    assert "front-right quarter view" in report
    assert "! thin rock spires may tear" in report
    assert "Continue Workflow" in report


def test_assess_image_fails_soft_when_provider_unreachable(tmp_path):
    from PIL import Image
    img = tmp_path / "t.png"
    Image.new("RGB", (8, 8)).save(img)
    result = assess_image(img, provider="ollama", base_url="http://127.0.0.1:9")
    assert not result.ok
    assert "ATLAS ASSESSMENT UNAVAILABLE" in result.report
    assert "ollama run" in result.report


def _canned(monkeypatch):
    calls = {"n": 0}

    def fake(image_path, **kw):
        calls["n"] += 1
        return AssessmentResult(payload=_PAYLOAD, ok=True, provider="ollama",
                                model="fake", report=format_assessment_report(_PAYLOAD))
    monkeypatch.setattr(assessor_mod, "assess_image", fake)
    nodes_mod._ATLAS_ASSESS_CACHE.clear()
    return calls


def test_node_pauses_image_until_proceed(monkeypatch):
    _canned(monkeypatch)

    class FakeBlocker:
        def __init__(self, message):
            self.message = message

    fake_graph = types.ModuleType("comfy_execution.graph")
    fake_graph.ExecutionBlocker = FakeBlocker
    fake_pkg = types.ModuleType("comfy_execution")
    fake_pkg.graph = fake_graph
    monkeypatch.setitem(sys.modules, "comfy_execution", fake_pkg)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", fake_graph)

    image = torch.rand(1, 32, 32, 3)
    out = AtlasAssessImage().assess(image)
    img_out, report, settings = out["result"]
    assert out["ui"]["text"] == [report]             # report shows on the node
    assert isinstance(img_out, FakeBlocker)          # paused
    assert "7/10" in report                          # report still flows
    assert '"scene_type": "organic"' in settings

    img_out2, _, _ = AtlasAssessImage().assess(image, proceed=True)["result"]
    assert img_out2 is image                         # resumed


def test_node_caches_assessment_across_proceed_flip(monkeypatch):
    calls = _canned(monkeypatch)
    image = torch.rand(1, 32, 32, 3)
    AtlasAssessImage().assess(image, proceed=False)
    AtlasAssessImage().assess(image, proceed=True)
    assert calls["n"] == 1  # flipping proceed must not re-run the VLM


def test_node_falls_back_to_passthrough_outside_comfy(monkeypatch):
    _canned(monkeypatch)
    image = torch.rand(1, 32, 32, 3)
    img_out, _, _ = AtlasAssessImage().assess(image, proceed=False)["result"]
    assert img_out is image  # no ExecutionBlocker importable in the test env


def test_node_tolerates_serialized_button_input(monkeypatch):
    # API-format exports can serialize the ▶ Continue Workflow BUTTON widget
    # as a bogus input key — found in the user's exported workflow.
    _canned(monkeypatch)
    image = torch.rand(1, 32, 32, 3)
    out = AtlasAssessImage().assess(image, proceed=True, **{"▶ Continue Workflow": None})
    assert out["result"][0] is image
