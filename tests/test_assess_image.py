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
    staged_layer_prompts,
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
    "staged_layers": {
        "sky": {"present": True, "sam_prompt": "sky", "notes": "clean horizon"},
        "far": {"present": True, "sam_prompt": "rock formations", "geometry": "card",
                "notes": "buttes"},
        "bg": {"present": True, "sam_prompt": "mesa cliffs", "geometry": "relief", "notes": ""},
        "mid": {"present": False, "sam_prompt": "", "geometry": "relief",
                "notes": "nothing distinct"},
        "fg": {"present": True, "sam_prompt": "desert scrub", "geometry": "ground", "notes": ""},
    },
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
    assert AtlasAssessImage.RETURN_TYPES == (
        "IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING")
    assert AtlasAssessImage.RETURN_NAMES == (
        "image", "report", "settings_json", "sam_prompt_sky", "sam_prompt_far",
        "sam_prompt_bg", "sam_prompt_mid", "sam_prompt_fg",
        "geom_far", "geom_bg", "geom_mid", "geom_fg")


def test_system_prompt_covers_the_settings_surface():
    # The prompt IS the product here — sanity-check the decision rules the
    # node's docstring promises are actually present.
    for token in ("scene_type", "depth_model", "fill_occluded", "embed_matte",
                  "relief_grid", "depth_edge_rel", "max_orbit_deg", "sky",
                  "near_pct", "multi-angle", "OUTPUT FORMAT",
                  # the staged 5-layer plan (sam prompts per fixed layer slot)
                  "staged_layers", "sam_prompt", "present"):
        assert token in ATLAS_ASSESSMENT_SYSTEM_PROMPT, token


def test_staged_layer_prompts_extraction():
    sam = staged_layer_prompts(_PAYLOAD)
    assert sam == {"sky": "sky", "far": "rock formations", "bg": "mesa cliffs",
                   "mid": "", "fg": "desert scrub"}  # absent mid -> "" (row stays bypassed)
    # No/failed assessment: bands empty, sky falls back to the literal "sky"
    # (the always-on sky SAM3 must never receive an empty prompt).
    assert staged_layer_prompts({}) == {"sky": "sky", "far": "", "bg": "", "mid": "", "fg": ""}
    # present=false wins even if the model left a prompt in the field.
    skyless = {"staged_layers": {"sky": {"present": False, "sam_prompt": "sky"},
                                 "fg": {"present": True, "sam_prompt": "office desks"}}}
    sam2 = staged_layer_prompts(skyless)
    assert sam2["sky"] == "sky" and sam2["fg"] == "office desks" and sam2["far"] == ""


def test_staged_layer_geometry_extraction():
    from atlas_camera.inference.assessor import staged_layer_geometry
    geom = staged_layer_geometry(_PAYLOAD)
    assert geom == {"far": "card", "bg": "relief", "mid": "", "fg": "ground"}
    # No assessment -> no recommendations (layer nodes keep their combo).
    assert staged_layer_geometry({}) == {"far": "", "bg": "", "mid": "", "fg": ""}
    # Hallucinated vocabulary must degrade to "" — the wired
    # AtlasCleanPlateLayer errors loudly on unknown values, so this helper
    # may never emit one.
    weird = {"staged_layers": {"far": {"present": True, "sam_prompt": "x",
                                       "geometry": "billboard"}}}
    assert staged_layer_geometry(weird)["far"] == ""


def test_report_formatting_from_payload():
    report = format_assessment_report(_PAYLOAD, provider="ollama", model="gemma3:4b")
    assert "7/10" in report and "20 deg" in report
    assert "bg" in report and "fill_occluded" in report
    assert "SAM prompt: sky" in report
    assert "STAGED 5-LAYER PLAN" in report
    assert 'SAM "rock formations"  · geometry: card' in report
    assert "geometry: ground" in report
    assert "absent — leave this stage bypassed" in report
    assert "front-right quarter view" in report
    assert "! thin rock spires may tear" in report
    assert "Continue Workflow" in report


class _FakeVisionHelper:
    """Minimal stand-in for a provider (lmstudio-shaped by default)."""

    provider = "lmstudio"

    def __init__(self, replies):
        self.replies = list(replies)  # str content OR Exception per call
        self.requests = []
        self.endpoints = []

    def validate_vision_model(self):
        return types.SimpleNamespace(id="fake-vlm")

    def _request_json(self, endpoint, payload):
        self.endpoints.append(endpoint)
        self.requests.append(payload)
        if endpoint == "/api/generate":       # ollama unload ping
            return {}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if self.provider == "ollama":
            return {"message": {"content": reply}}
        return {"choices": [{"message": {"content": reply}}]}


def _fake_provider(monkeypatch, helper):
    monkeypatch.setattr(assessor_mod, "create_multimodal_provider",
                        lambda *a, **k: helper)


def _tmp_image(tmp_path):
    from PIL import Image
    img = tmp_path / "t.png"
    Image.new("RGB", (8, 8)).save(img)
    return img


def test_unparseable_reply_fails_visibly_and_uncached(monkeypatch, tmp_path):
    """A reply that isn't a usable assessment must come back ok=False (so the
    node never caches it — a re-queue actually retries) with the RAW reply in
    the report (found live: lmstudio/gemma returned non-JSON and the artist
    saw only 'Model response was not valid JSON' with nothing to act on)."""
    helper = _FakeVisionHelper(["Sure! Here is my analysis of the photo..."])
    _fake_provider(monkeypatch, helper)
    result = assess_image(_tmp_image(tmp_path), provider="lmstudio")
    assert not result.ok
    assert "ATLAS ASSESSMENT FAILED" in result.report
    assert "Here is my analysis" in result.report      # raw reply shown
    assert "NOT cached" in result.report
    # lmstudio requests structured output (grammar-constrained JSON)
    assert helper.requests[0].get("response_format", {}).get("type") == "json_schema"
    assert helper.requests[0]["max_tokens"] >= 3000


def test_response_format_rejection_falls_back_to_plain(monkeypatch, tmp_path):
    import json as _json
    good = _json.dumps(_PAYLOAD)
    helper = _FakeVisionHelper([
        RuntimeError("400: 'response_format' json_schema unsupported"), good])
    _fake_provider(monkeypatch, helper)
    result = assess_image(_tmp_image(tmp_path), provider="lmstudio")
    assert result.ok
    assert "response_format" in helper.requests[0]
    assert "response_format" not in helper.requests[1]  # retried plain


def test_node_never_caches_a_failed_assessment(monkeypatch):
    calls = {"n": 0}

    def fake(image_path, **kw):
        calls["n"] += 1
        return AssessmentResult(ok=False, report="ATLAS ASSESSMENT FAILED — x")
    monkeypatch.setattr(assessor_mod, "assess_image", fake)
    nodes_mod._ATLAS_ASSESS_CACHE.clear()
    image = torch.rand(1, 32, 32, 3)
    AtlasAssessImage().assess(image)
    AtlasAssessImage().assess(image)
    assert calls["n"] == 2  # each queue retries until an assessment succeeds


def test_openai_provider_factory_and_key_handling(monkeypatch):
    """The 'openai' cloud provider: default endpoint/model, env-var key
    fallback, and an actionable error when no key exists at all."""
    from atlas_camera.inference.multimodal_helper import create_multimodal_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prov = create_multimodal_provider("openai")
    assert prov.provider == "openai"
    assert prov.base_url == "https://api.openai.com/v1"
    assert prov.model == "gpt-4o-mini"
    with pytest.raises(RuntimeError, match="API key"):
        prov.validate_vision_model()

    # Widget key wins; env var is the fallback; explicit model skips /models.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    prov_env = create_multimodal_provider("openai", model="gpt-4o")
    assert prov_env.api_key == "sk-env"
    assert prov_env.validate_vision_model().id == "gpt-4o"  # no network call
    prov_widget = create_multimodal_provider("openai", api_key="sk-widget")
    assert prov_widget.api_key == "sk-widget"


def test_assess_image_openai_sends_structured_output(monkeypatch, tmp_path):
    import json as _json
    helper = _FakeVisionHelper([_json.dumps(_PAYLOAD)])
    helper.provider = "openai"
    _fake_provider(monkeypatch, helper)
    result = assess_image(_tmp_image(tmp_path), provider="openai", api_key="sk-x")
    assert result.ok
    assert helper.requests[0].get("response_format", {}).get("type") == "json_schema"


def test_api_key_widget_appended_last():
    # widgets_values is positional — a new widget anywhere but the END
    # silently corrupts every saved workflow (the documented 2026-07-06 bug).
    from atlas_camera.comfy.nodes import AtlasVLMScaleCues
    assert list(AtlasVLMScaleCues.INPUT_TYPES()["optional"])[-1] == "api_key"
    # AtlasAssessImage: api_key then offload_model, both appended in order.
    assert list(AtlasAssessImage.INPUT_TYPES()["optional"])[-3:] == [
        "api_key", "offload_model", "auto_continue"]
    for cls in (AtlasAssessImage, AtlasVLMScaleCues):
        assert "openai" in cls.INPUT_TYPES()["optional"]["provider"][0], cls.__name__


def test_offload_model_ollama_rides_request_and_pings_unload(monkeypatch, tmp_path):
    import json as _json
    helper = _FakeVisionHelper([_json.dumps(_PAYLOAD)])
    helper.provider = "ollama"
    _fake_provider(monkeypatch, helper)
    r = assess_image(_tmp_image(tmp_path), provider="ollama", offload_model=True)
    assert r.ok
    assert helper.requests[0].get("keep_alive") == 0          # on the chat call
    assert "/api/generate" in helper.endpoints                # explicit unload ping
    assert "MODEL OFFLOAD" in r.report and "keep_alive=0" in r.report
    # Off by default: no keep_alive key, no unload ping, no report line.
    helper2 = _FakeVisionHelper([_json.dumps(_PAYLOAD)])
    helper2.provider = "ollama"
    _fake_provider(monkeypatch, helper2)
    r2 = assess_image(_tmp_image(tmp_path), provider="ollama")
    assert "keep_alive" not in helper2.requests[0]
    assert "/api/generate" not in helper2.endpoints
    assert "MODEL OFFLOAD" not in r2.report


def test_offload_model_lmstudio_ttl_and_llamacpp_honesty(monkeypatch, tmp_path):
    import json as _json

    # lmstudio: ttl rides the request; without the lms CLI the report says so.
    monkeypatch.setattr("shutil.which", lambda name: None)
    helper = _FakeVisionHelper([_json.dumps(_PAYLOAD)])
    _fake_provider(monkeypatch, helper)
    r = assess_image(_tmp_image(tmp_path), provider="lmstudio", offload_model=True)
    assert helper.requests[0].get("ttl") == 2
    assert "MODEL OFFLOAD" in r.report and "ttl" in r.report

    # llamacpp: no unload API exists — the report is honest, never pretends.
    helper2 = _FakeVisionHelper([_json.dumps(_PAYLOAD)])
    helper2.provider = "llamacpp"
    _fake_provider(monkeypatch, helper2)
    r2 = assess_image(_tmp_image(tmp_path), provider="llamacpp", offload_model=True)
    assert "ttl" not in helper2.requests[0]
    assert "not supported" in r2.report


def test_offload_skipped_on_failed_assessment(monkeypatch, tmp_path):
    # A failed assessment keeps the model warm for the retry.
    helper = _FakeVisionHelper(["not json at all, just prose"])
    helper.provider = "ollama"
    _fake_provider(monkeypatch, helper)
    r = assess_image(_tmp_image(tmp_path), provider="ollama", offload_model=True)
    assert not r.ok
    assert "/api/generate" not in helper.endpoints
    assert "MODEL OFFLOAD" not in r.report


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
    # auto_continue defaults ON (advisory mode): the image flows on the very
    # first queue with no ▶ Continue — the solve gate downstream is the
    # first checkpoint (user-requested default, 2026-07-11).
    out_auto = AtlasAssessImage().assess(image)
    assert out_auto["result"][0] is image

    # The hard gate is opt-in via auto_continue=False.
    out = AtlasAssessImage().assess(image, auto_continue=False)
    img_out, report, settings = out["result"][:3]
    assert out["ui"]["text"] == [report]             # report shows on the node
    assert isinstance(img_out, FakeBlocker)          # paused
    assert "7/10" in report                          # report still flows
    assert '"scene_type": "organic"' in settings
    # The staged SAM prompts + geometry recommendations flow UNGATED (the
    # image blocker already pauses everything they feed, via the plate rail).
    assert out["result"][3:8] == ("sky", "rock formations", "mesa cliffs", "", "desert scrub")
    assert out["result"][8:] == ("card", "relief", "", "ground")  # absent mid -> ""
    # ...and ride the ui message so atlas_assess.js can mirror them into
    # linked widgets (linked widget-inputs display stale text otherwise).
    assert out["ui"]["sam_prompts"] == ["sky", "rock formations", "mesa cliffs", "", "desert scrub"]
    assert out["ui"]["sam_geometry"] == ["card", "relief", "", "ground"]

    img_out2 = AtlasAssessImage().assess(image, proceed=True,
                                         auto_continue=False)["result"][0]
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
    img_out = AtlasAssessImage().assess(image, proceed=False, auto_continue=False)["result"][0]
    assert img_out is image  # no ExecutionBlocker importable in the test env


def test_stale_approval_from_a_different_image_rearms_the_gate(monkeypatch):
    """▶ Continue approves THIS image only: a persisted proceed=True whose
    approved_for fingerprint doesn't match the current image must block again
    (found live — a new image sailed through the previous image's approval)."""
    import sys
    import types

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

    from atlas_camera.comfy.nodes import _image_fingerprint
    old_image = torch.rand(1, 32, 32, 3)
    new_image = torch.rand(1, 32, 32, 3)
    old_fp = _image_fingerprint(old_image)

    # Approval matches -> flows.
    out_ok = AtlasAssessImage().assess(old_image, proceed=True, approved_for=old_fp,
                                   auto_continue=False)
    assert out_ok["result"][0] is old_image
    assert out_ok["ui"]["fingerprint"] == [old_fp]

    # Same persisted approval, different image -> blocked + report says why.
    out_stale = AtlasAssessImage().assess(new_image, proceed=True, approved_for=old_fp,
                                      auto_continue=False)
    assert isinstance(out_stale["result"][0], FakeBlocker)
    assert "GATE RE-ARMED" in out_stale["result"][1]

    # Manual override: proceed=True with EMPTY approved_for is unconditional.
    out_manual = AtlasAssessImage().assess(new_image, proceed=True, approved_for="",
                                       auto_continue=False)
    assert out_manual["result"][0] is new_image


def test_node_tolerates_serialized_button_input(monkeypatch):
    # API-format exports can serialize the ▶ Continue Workflow BUTTON widget
    # as a bogus input key — found in the user's exported workflow.
    _canned(monkeypatch)
    image = torch.rand(1, 32, 32, 3)
    out = AtlasAssessImage().assess(image, proceed=True, **{"▶ Continue Workflow": None})
    assert out["result"][0] is image
