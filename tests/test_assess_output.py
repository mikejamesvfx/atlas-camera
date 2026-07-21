"""Terminal AtlasAssessOutput: honest headless provenance + structured QA."""

import hashlib
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, "tests")
from test_inpaint_layers_nodes import _solve

import atlas_camera.comfy.nodes_qa as qa_nodes
import atlas_camera.inference.output_assessor as output_mod
from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, AtlasAssessOutput
from atlas_camera.comfy.headless_evidence import compare_source_structure
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasProxyPrimitive
from atlas_camera.inference.output_assessor import (
    OUTPUT_ASSESSMENT_SYSTEM_PROMPT,
    OutputAssessmentResult,
    assess_output,
)


_PAYLOAD = {
    "verdict": "pass",
    "score_0_100": 91,
    "summary": "Projection is coherent and the recovered view is clean.",
    "checks": {
        "camera_match": {"status": "pass", "evidence": "lines align"},
        "projection_edges": {"status": "pass", "evidence": "smooth"},
        "occlusion_tearing": {"status": "pass", "evidence": "none seen"},
        "layer_inpaint": {"status": "pass", "evidence": "coherent"},
        "color_alpha": {"status": "pass", "evidence": "no seam"},
    },
    "issues": [],
    "strengths": ["stable architecture"],
    "recommended_next_run": ["render a short orbit"],
}


def _healthy_solve():
    solve = _solve()
    solve.debug_metadata["scale_source"] = "manual_override"
    return solve


def _healthy_solve_with_projection_mesh():
    """A full-frame canonical relief proxy suitable for backend evidence."""
    solve = _healthy_solve()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="full_frame_relief",
        primitive_type="mesh",
        metadata={
            "role": PROXY_ROLE,
            "vertices": [-1.0, 0.0, -5.0, 1.0, 0.0, -5.0,
                         1.0, 2.0, -5.0, -1.0, 2.0, -5.0],
            "faces": [0, 1, 2, 0, 2, 3],
            "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "n_vertices": 4,
            "n_faces": 2,
            "torn_fraction": 0.0,
            "stretch_ratio_p95": 1.0,
        }))
    return solve


def test_node_registered_and_prompt_has_headless_and_alpha_rules():
    assert NODE_CLASS_MAPPINGS["AtlasAssessOutput"] is AtlasAssessOutput
    assert AtlasAssessOutput.OUTPUT_NODE is True
    assert AtlasAssessOutput.RETURN_NAMES == (
        "report", "assessment_json", "json_path", "verdict", "image_provenance",
        "assessed_image", "evidence_path")
    for token in ("IMAGE PROVENANCE", "source_image_fallback", "inconclusive",
                  "headless_projection_reconstruction", "unpremultiply RGB",
                  "never color-transform alpha", "matte_coverage",
                  "torn_excessive", "union coverage", "source/reference plate",
                  "coverage matte", "source_comparison"):
        assert token in OUTPUT_ASSESSMENT_SYSTEM_PROMPT


def test_disabled_still_writes_deterministic_headless_report(tmp_path):
    camera = torch.zeros(1, 16, 24, 3)
    source = torch.rand(1, 16, 24, 3)
    path = tmp_path / "qa" / "output.json"
    out = AtlasAssessOutput().assess(
        camera, _healthy_solve(), source_image=source,
        enabled=False, file_path=str(path))
    (report, payload_json, json_path, verdict, provenance,
     assessed_image, evidence_path) = out["result"]
    payload = json.loads(payload_json)

    assert json_path == str(path)
    assert path.is_file()
    assert provenance == "source_image_fallback"
    assert verdict == "inconclusive"
    assert payload["status"] == "disabled"
    assert payload["solve_summary"]["scene_health"]["level"] == "pass"
    assert payload["image"]["camera_view_stats"]["reason"] == "all_zero_unbaked_pass"
    assert "HEADLESS LIMIT" in report
    assert out["ui"]["atlas_output_assessment"] == [payload_json]
    assert evidence_path == payload["image"]["evidence_path"]
    assert assessed_image is source
    assert (tmp_path / "qa" / "output_evidence.png").is_file()
    assert (tmp_path / "qa" / "output_source_reference.png").is_file()


def test_source_fallback_can_never_be_promoted_to_visual_pass(monkeypatch, tmp_path):
    def fake(*_args, **_kwargs):
        return OutputAssessmentResult(
            payload=_PAYLOAD, report="fake pass", provider="lmstudio",
            model="fake-vlm", ok=True)

    monkeypatch.setattr(output_mod, "assess_output", fake)
    qa_nodes._OUTPUT_ASSESS_CACHE.clear()
    out = AtlasAssessOutput().assess(
        torch.zeros(1, 10, 10, 3), _healthy_solve(),
        source_image=torch.rand(1, 10, 10, 3), enabled=True,
        offload_model=False, file_path=str(tmp_path / "fallback.json"))
    data = json.loads(out["result"][1])
    assert out["result"][3] == "inconclusive"
    assert data["vlm"]["assessment"]["verdict"] == "inconclusive"
    for name in ("camera_match", "projection_edges", "occlusion_tearing",
                 "layer_inpaint"):
        assert data["vlm"]["assessment"]["checks"][name]["status"] == "inconclusive"


def test_real_camera_view_combines_vlm_and_scene_health(monkeypatch, tmp_path):
    payload = json.loads(json.dumps(_PAYLOAD))
    payload["verdict"] = "warn"
    payload["score_0_100"] = 72

    monkeypatch.setattr(output_mod, "assess_output", lambda *_a, **_k:
        OutputAssessmentResult(payload=payload, report="visual warning",
                               provider="ollama", model="fake", ok=True))
    qa_nodes._OUTPUT_ASSESS_CACHE.clear()
    out = AtlasAssessOutput().assess(
        torch.rand(1, 12, 18, 3), _healthy_solve(), enabled=True,
        offload_model=False, file_path=str(tmp_path / "real.json"))
    data = json.loads(out["result"][1])
    assert out["result"][3] == "warn"
    assert out["result"][4] == "camera_view"
    assert out["result"][6] == str(tmp_path / "real_evidence.png")
    assert data["status"] == "complete"
    assert data["vlm"]["assessment"]["score_0_100"] == 72


def test_blank_view_reconstructs_and_retains_exact_assessed_image(
        monkeypatch, tmp_path):
    seen = {}

    def fake(image_path, **_kwargs):
        seen["image_path"] = str(image_path)
        seen.update(_kwargs)
        return OutputAssessmentResult(
            payload=_PAYLOAD, report="canonical visual pass",
            provider="lmstudio", model="fake-vlm", ok=True)

    monkeypatch.setattr(output_mod, "assess_output", fake)
    qa_nodes._OUTPUT_ASSESS_CACHE.clear()
    source = torch.zeros(1, 12, 18, 3)
    source[..., 0] = torch.linspace(0.1, 0.9, 18)[None, None, :]
    source[..., 1] = 0.4
    path = tmp_path / "headless.json"
    out = AtlasAssessOutput().assess(
        torch.zeros_like(source), _healthy_solve_with_projection_mesh(),
        source_image=source, enabled=True, offload_model=False,
        file_path=str(path))
    data = json.loads(out["result"][1])
    image = data["image"]

    assert out["result"][4] == "headless_projection_reconstruction"
    assert data["verdict"] == "inconclusive"
    assert data["vlm"]["assessment"]["verdict"] == "inconclusive"
    assert data["vlm"]["assessment"]["checks"]["camera_match"]["status"] == "pass"
    assert data["vlm"]["assessment"]["checks"]["occlusion_tearing"]["status"] == (
        "inconclusive")
    assert image["headless_reconstruction"]["method"] == (
        "canonical_projection_reconstruction_v1")
    assert image["evidence_path"] == str(tmp_path / "headless_evidence.png")
    assert image["coverage_path"] == str(tmp_path / "headless_coverage.png")
    assert image["source_reference_path"] == str(
        tmp_path / "headless_source_reference.png")
    assert seen["image_path"] == image["evidence_path"]
    assert seen["coverage_path"] == image["coverage_path"]
    assert seen["source_reference_path"] == image["source_reference_path"]
    assert path.is_file()
    assert (tmp_path / "headless_evidence.png").is_file()
    assert (tmp_path / "headless_coverage.png").is_file()
    assert (tmp_path / "headless_source_reference.png").is_file()
    assert hashlib.sha256(
        (tmp_path / "headless_evidence.png").read_bytes()).hexdigest() == (
            image["evidence_sha256"])
    assert hashlib.sha256(
        (tmp_path / "headless_source_reference.png").read_bytes()).hexdigest() == (
            image["source_reference_sha256"])
    assert out["result"][5].shape == source.shape
    assert out["result"][6] == image["evidence_path"]
    assert data["solve_summary"]["orbit_coverage"]["poses"]
    assert data["solve_summary"]["source_comparison"]["status"] == (
        "within_tolerance")
    assert "intentionally assigned" in (
        data["solve_summary"]["field_semantics"]["per_layer.matte_coverage"])


def test_no_usable_image_skips_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(output_mod, "assess_output", lambda *_a, **_k:
                        pytest.fail("provider must not be called for an all-zero input"))
    out = AtlasAssessOutput().assess(
        torch.zeros(1, 8, 8, 3), _healthy_solve(), enabled=True,
        file_path=str(tmp_path / "none.json"))
    data = json.loads(out["result"][1])
    assert data["status"] == "no_usable_image"
    assert out["result"][4] == "no_usable_image"


def test_measured_canonical_holes_override_optimistic_vlm_and_raw_metrics(
        monkeypatch, tmp_path):
    solve = _healthy_solve_with_projection_mesh()
    solve.projection_scene.proxy_geometry[0].metadata["uvs"] = [
        0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]
    payload = json.loads(json.dumps(_PAYLOAD))
    payload["issues"] = [{
        "severity": "fail", "category": "geometry_stretch",
        "evidence": "stretch_ratio_p95 7.0 is severe", "action": "retopo"}]
    monkeypatch.setattr(output_mod, "assess_output", lambda *_a, **_k:
        OutputAssessmentResult(payload=payload, report="optimistic raw report",
                               provider="lmstudio", model="fake", ok=True))
    qa_nodes._OUTPUT_ASSESS_CACHE.clear()
    source = torch.rand(1, 20, 30, 3)
    out = AtlasAssessOutput().assess(
        torch.zeros_like(source), solve, source_image=source, enabled=True,
        file_path=str(tmp_path / "grounded.json"))
    data = json.loads(out["result"][1])
    visual = data["vlm"]["assessment"]

    assert data["verdict"] == "fail"
    assert visual["verdict"] == "fail"
    assert visual["checks"]["projection_edges"]["status"] == "fail"
    assert "canonical output holes" in (
        visual["checks"]["projection_edges"]["evidence"])
    assert visual["checks"]["layer_inpaint"]["status"] == "inconclusive"
    assert [issue["category"] for issue in visual["issues"]] == [
        "measured_canonical_coverage"]
    assert any("uncalibrated raw-metric" in item
               for item in visual["grounding_corrections"])
    assert "DETERMINISTIC GROUNDING" in out["result"][0]


def test_measured_holes_fail_even_when_vlm_is_disabled(tmp_path):
    solve = _healthy_solve_with_projection_mesh()
    solve.projection_scene.proxy_geometry[0].metadata["uvs"] = [
        0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]
    source = torch.rand(1, 20, 30, 3)

    out = AtlasAssessOutput().assess(
        torch.zeros_like(source), solve, source_image=source, enabled=False,
        file_path=str(tmp_path / "deterministic_only.json"))
    data = json.loads(out["result"][1])

    assert data["status"] == "disabled"
    assert data["deterministic_output_verdict"] == "fail"
    assert data["verdict"] == "fail"
    assert data["vlm"]["assessment"] is None


def test_source_structure_comparison_detects_wholesale_replacement():
    torch.manual_seed(7)
    source = torch.rand(1, 96, 128, 3)

    identical = compare_source_structure(source, source.clone())
    replaced = compare_source_structure(torch.rand_like(source), source)

    assert identical["status"] == "within_tolerance"
    assert identical["luma_correlation"] == pytest.approx(1.0)
    assert identical["edge_correlation"] == pytest.approx(1.0)
    assert replaced["status"] == "severe"
    assert replaced["luma_correlation"] < 0.70
    assert replaced["edge_correlation"] < 0.40


def test_severe_source_drift_overrides_optimistic_inpaint_assessment():
    payload = json.loads(json.dumps(_PAYLOAD))
    summary = {
        "projection_source_count": 3,
        "scene_health": {"flags": []},
        "headless_evidence": {"coverage_fraction": 1.0},
        "source_comparison": {
            "status": "severe",
            "luma_correlation": 0.5953,
            "edge_correlation": 0.2753,
            "rgb_mean_absolute_error": 0.1508,
            "changed_fraction_gt_0_15": 0.3304,
        },
    }

    grounded = qa_nodes._enforce_deterministic_evidence(
        qa_nodes._enforce_provenance(
            payload, "headless_projection_reconstruction"),
        "headless_projection_reconstruction", summary)

    assert grounded["verdict"] == "fail"
    assert grounded["checks"]["layer_inpaint"]["status"] == "fail"
    assert any(issue.get("category") == "source_structure_drift"
               for issue in grounded["issues"])
    assert grounded["summary"].startswith("Grounded terminal failure:")
    assert grounded["model_summary_raw"] == _PAYLOAD["summary"]
    assert qa_nodes._deterministic_output_floor(
        "headless_projection_reconstruction", summary) == "fail"


class _FakeHelper:
    provider = "lmstudio"

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def validate_vision_model(self):
        return types.SimpleNamespace(id="fake-vlm")

    def _request_json(self, _endpoint, payload):
        self.requests.append(payload)
        return {"choices": [{"message": {"content": self.reply}}]}


def test_inference_request_contains_provenance_and_structured_health(
        monkeypatch, tmp_path):
    from PIL import Image

    image = tmp_path / "frame.png"
    coverage = tmp_path / "coverage.png"
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (32, 64, 96)).save(image)
    Image.new("L", (8, 8), 255).save(coverage)
    Image.new("RGB", (8, 8), (16, 32, 48)).save(source)
    helper = _FakeHelper(json.dumps(_PAYLOAD))
    monkeypatch.setattr(output_mod, "create_multimodal_provider",
                        lambda *_a, **_k: helper)
    result = assess_output(
        image, solve_summary={"scene_health": {"level": "warn"}},
        image_provenance="camera_view", coverage_path=coverage,
        source_reference_path=source, provider="lmstudio")
    assert result.ok and "91/100" in result.report
    request = helper.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    user_text = request["messages"][1]["content"][0]["text"]
    assert "IMAGE_PROVENANCE: camera_view" in user_text
    assert "IMAGE 2: coverage matte" in user_text
    assert "IMAGE 3: source/reference plate" in user_text
    assert '"level":"warn"' in user_text
    assert len(request["messages"][1]["content"]) == 4
