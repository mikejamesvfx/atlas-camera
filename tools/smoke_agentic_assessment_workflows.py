"""Validate and live-run the three shipped agentic output-QA workflows.

Examples::

    python tools/smoke_agentic_assessment_workflows.py --validate-only
    python tools/smoke_agentic_assessment_workflows.py
    python tools/smoke_agentic_assessment_workflows.py --workflow atlas_input_quickstart

The live run opens AtlasSolveGate nodes, keeps VLMs resident between the
sequential workflows, and requires one honest structured AtlasAssessOutput
report per workflow. The smoke requires either a baked camera-view render or a
retained canonical projection reconstruction with hashed output/reference and
a coverage matte; source-plate-only fallbacks are rejected because they are not
output evidence. The shipped variants use Ghost Town and Space Hangar rather
than ComfyUI's cartoon placeholder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_camera.mcp import comfy_http as C


WORKFLOWS = {
    "atlas_input_quickstart": (
        ROOT / "examples" /
        "atlas_input_quickstart_agentic_assessment_workflow.json"),
    "atlas_occlusion_cull_quickstart": (
        ROOT / "examples" /
        "atlas_occlusion_cull_quickstart_agentic_assessment_workflow.json"),
    "atlas_camera_staged_master": (
        ROOT / "examples" /
        "atlas_camera_staged_master_agentic_assessment_workflow.json"),
}

def assess_smoke_result(result: dict[str, Any], *, require_vlm: bool) -> dict[str, Any]:
    """Validate one queue result and return a compact smoke-test summary."""
    if not result.get("completed") or result.get("errors"):
        raise RuntimeError("workflow execution failed: " + "; ".join(
            str(item) for item in result.get("errors") or ["not completed"]))
    reports = result.get("reports") or {}
    if len(reports) != 1:
        raise RuntimeError(
            f"expected exactly one AtlasAssessOutput report, got {len(reports)}")
    node_id, report = next(iter(reports.items()))
    payload = report.get("assessment")
    if not isinstance(payload, dict):
        raise RuntimeError("AtlasAssessOutput did not return structured JSON")

    image_data = payload.get("image") or {}
    provenance = image_data.get("provenance")
    if provenance not in {"camera_view", "headless_projection_reconstruction"}:
        raise RuntimeError(
            f"workflow did not produce assessable output evidence: {provenance!r}")
    evidence_path = str(report.get("evidence_path") or
                        image_data.get("evidence_path") or "")
    if not evidence_path or not image_data.get("evidence_sha256"):
        raise RuntimeError("terminal report did not retain hashed image evidence")
    if provenance == "headless_projection_reconstruction":
        coverage_path = str(report.get("coverage_path") or
                            image_data.get("coverage_path") or "")
        source_path = str(report.get("source_reference_path") or
                          image_data.get("source_reference_path") or "")
        if not coverage_path:
            raise RuntimeError("headless report did not retain its coverage matte")
        if not source_path or not image_data.get("source_reference_sha256"):
            raise RuntimeError("headless report did not retain a hashed source reference")
    vlm = payload.get("vlm") or {}
    solve_summary = payload.get("solve_summary") or {}
    expected_status = "complete" if require_vlm else "disabled"
    if payload.get("status") != expected_status:
        raise RuntimeError(
            f"terminal status {payload.get('status')!r}, expected {expected_status!r}")
    if require_vlm and not vlm.get("ok"):
        raise RuntimeError(
            f"terminal VLM unavailable: {vlm.get('error_report') or 'unknown error'}")

    visual = vlm.get("assessment")
    if provenance == "headless_projection_reconstruction" and require_vlm:
        # Canonical evidence may prove a visible/deterministic failure. It may
        # not be promoted to PASS/WARN because orbit occlusion is unobserved.
        if (not isinstance(visual, dict)
                or visual.get("verdict") not in {"inconclusive", "fail"}):
            raise RuntimeError(
                "canonical headless evidence was promoted to a passing verdict")
        checks = visual.get("checks") or {}
        if (checks.get("occlusion_tearing") or {}).get("status") != "inconclusive":
            raise RuntimeError("canonical evidence promoted orbit occlusion to a visual verdict")

    return {
        "prompt_id": result.get("prompt_id"),
        "assessment_node": str(node_id),
        "status": payload.get("status"),
        "verdict": payload.get("verdict"),
        "image_provenance": provenance,
        "solve_health": (solve_summary.get("scene_health") or {}).get("level"),
        "canonical_coverage": (solve_summary.get("headless_evidence") or {}).get(
            "coverage_fraction"),
        "source_comparison": (solve_summary.get("source_comparison") or {}).get(
            "status"),
        "deterministic_output_verdict": payload.get(
            "deterministic_output_verdict"),
        "vlm_ok": bool(vlm.get("ok")),
        "provider": vlm.get("provider"),
        "model": vlm.get("model"),
        "json_path": report.get("json_path"),
        "evidence_path": evidence_path,
        "coverage_path": str(report.get("coverage_path") or
                             image_data.get("coverage_path") or ""),
        "source_reference_path": str(report.get("source_reference_path") or
                                     image_data.get("source_reference_path") or ""),
    }


def _prepare_api(ui: dict[str, Any], oi: dict[str, Any], *, model: str,
                 run_vlm: bool) -> tuple[dict[str, Any], list[str]]:
    errors, warnings = C.validate_ui(ui, oi)
    if errors:
        raise RuntimeError("workflow validation failed: " + "; ".join(errors))
    api = C.ui_to_api(ui, oi)
    assessors = [node_id for node_id, node in api.items()
                 if node.get("class_type") == "AtlasAssessOutput"]
    if len(assessors) != 1:
        raise RuntimeError(
            f"expected one AtlasAssessOutput node, got {len(assessors)}")

    overrides = C.gate_overrides(ui, oi)
    for node_id, node in api.items():
        node_type = node.get("class_type")
        if node_type == "AtlasAssessOutput":
            overrides[f"{node_id}.enabled"] = run_vlm
            overrides[f"{node_id}.offload_model"] = False
            if model:
                overrides[f"{node_id}.model"] = model
        elif node_type == "AtlasAssessImage":
            # The staged preflight and terminal assessment share one resident
            # model during the smoke run instead of repeatedly unloading it.
            if "offload_model" in node.get("inputs", {}):
                overrides[f"{node_id}.offload_model"] = False
            if model and "model" in node.get("inputs", {}):
                overrides[f"{node_id}.model"] = model
    C.apply_overrides(api, overrides)
    return api, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=C.DEFAULT_HOST)
    parser.add_argument("--workflow", action="append", choices=sorted(WORKFLOWS),
                        help="Run one named workflow (repeatable); default: all three")
    parser.add_argument("--model", default="google/gemma-4-12b-qat")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Exercise deterministic report generation only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    oi = C.fetch_object_info(args.host)
    selected = args.workflow or list(WORKFLOWS)
    summaries = []
    for name in selected:
        path = WORKFLOWS[name]
        ui = json.loads(path.read_text(encoding="utf-8"))
        api, warnings = _prepare_api(
            ui, oi, model=args.model, run_vlm=not args.skip_vlm)
        if args.validate_only:
            summaries.append({
                "workflow": name,
                "validation": "pass",
                "warnings": warnings,
                "api_nodes": len(api),
            })
            continue
        result = C.queue_and_wait(
            api, host=args.host, timeout=args.timeout, poll_s=args.poll)
        summary = assess_smoke_result(result, require_vlm=not args.skip_vlm)
        summary["workflow"] = name
        summary["validation_warnings"] = warnings
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    if args.validate_only:
        print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
