"""Offline contracts for the live agentic-workflow smoke harness."""

import pytest

from tools.smoke_agentic_assessment_workflows import assess_smoke_result


def _result(*, provenance="headless_projection_reconstruction", status="complete",
            vlm_ok=True, visual_verdict="inconclusive"):
    checks = {
        name: {"status": "inconclusive", "evidence": "canonical evidence"}
        for name in (
            "camera_match", "projection_edges", "occlusion_tearing", "layer_inpaint")
    }
    return {
        "completed": True,
        "prompt_id": "prompt-1",
        "errors": [],
        "reports": {
            "14": {
                "json_path": "atlas_debug/result.json",
                "evidence_path": "atlas_debug/result_evidence.png",
                "coverage_path": "atlas_debug/result_coverage.png",
                "source_reference_path": "atlas_debug/result_source_reference.png",
                "assessment": {
                    "status": status,
                    "verdict": "inconclusive",
                    "deterministic_output_verdict": "pass",
                    "image": {
                        "provenance": provenance,
                        "evidence_path": "atlas_debug/result_evidence.png",
                        "coverage_path": "atlas_debug/result_coverage.png",
                        "source_reference_path": (
                            "atlas_debug/result_source_reference.png"),
                        "evidence_sha256": "a" * 64,
                        "source_reference_sha256": "b" * 64,
                    },
                    "solve_summary": {
                        "scene_health": {"level": "pass"},
                        "headless_evidence": {"coverage_fraction": 1.0},
                        "source_comparison": {"status": "within_tolerance"},
                    },
                    "vlm": {
                        "ok": vlm_ok,
                        "provider": "lmstudio",
                        "model": "test-vlm",
                        "assessment": ({"verdict": visual_verdict,
                                        "checks": checks}
                                       if vlm_ok else None),
                        "error_report": None if vlm_ok else "offline",
                    },
                },
            },
        },
    }


def test_smoke_accepts_retained_headless_projection_evidence():
    summary = assess_smoke_result(_result(), require_vlm=True)
    assert summary == {
        "prompt_id": "prompt-1",
        "assessment_node": "14",
        "status": "complete",
        "verdict": "inconclusive",
        "image_provenance": "headless_projection_reconstruction",
        "solve_health": "pass",
        "canonical_coverage": 1.0,
        "source_comparison": "within_tolerance",
        "deterministic_output_verdict": "pass",
        "vlm_ok": True,
        "provider": "lmstudio",
        "model": "test-vlm",
        "json_path": "atlas_debug/result.json",
        "evidence_path": "atlas_debug/result_evidence.png",
        "coverage_path": "atlas_debug/result_coverage.png",
        "source_reference_path": "atlas_debug/result_source_reference.png",
    }


def test_smoke_rejects_missing_terminal_report():
    result = _result()
    result["reports"] = {}
    with pytest.raises(RuntimeError, match="exactly one"):
        assess_smoke_result(result, require_vlm=True)


def test_smoke_rejects_promoted_headless_reconstruction():
    result = _result(visual_verdict="pass")
    with pytest.raises(RuntimeError, match="promoted"):
        assess_smoke_result(result, require_vlm=True)


def test_smoke_accepts_proven_headless_failure():
    summary = assess_smoke_result(
        _result(visual_verdict="fail"), require_vlm=True)
    assert summary["image_provenance"] == "headless_projection_reconstruction"


def test_smoke_rejects_source_plate_fallback_as_inaccurate_output_evidence():
    result = _result(provenance="source_image_fallback")
    with pytest.raises(RuntimeError, match="assessable output evidence"):
        assess_smoke_result(result, require_vlm=True)


def test_smoke_accepts_disabled_deterministic_report():
    result = _result(status="disabled", vlm_ok=False)
    summary = assess_smoke_result(result, require_vlm=False)
    assert summary["status"] == "disabled"
    assert summary["vlm_ok"] is False
