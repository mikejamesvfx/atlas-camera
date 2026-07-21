"""Terminal VLM quality assessment for rendered Atlas camera views.

Unlike :mod:`atlas_camera.inference.assessor` (the pre-flight scene planner),
this module reviews an image produced at the *end* of a workflow together
with Atlas' deterministic scene-health summary.  The VLM is advisory and
fails soft; callers always retain the machine-readable solve diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.inference.multimodal_helper import (
    _image_base64,
    _image_data_url,
    _is_response_format_error,
    _openai_chat_content,
    _parse_model_json,
    create_multimodal_provider,
)


OUTPUT_ASSESSMENT_SYSTEM_PROMPT = """\
You are the final quality-control reviewer for Atlas Camera, a 2.5D camera
projection and layered matte-painting pipeline. Review the supplied camera
view AND the deterministic solve-health JSON included in the user message.

Judge only evidence that is actually available. Look especially for:
- camera/projection mismatch, doubled lines, sliding or warped architecture;
- stretched grazing texels, serrated projection silhouettes, black seams,
  occlusion holes, floating fragments, and discontinuity-edge tearing;
- inconsistent inpaint, repeated texture, layer seams, and missing coverage;
- obvious color/exposure discontinuities. Alpha is data: do not recommend a
  display transform on alpha. If edge filtering is suggested, preserve the
  straight/premultiplied contract (unpremultiply RGB, filter/dilate straight
  RGB, then premultiply; never color-transform alpha).

IMAGE PROVENANCE IS AUTHORITATIVE. If it says source_image_fallback, the
browser/WebGL render was not baked and this image is only the source plate.
In that case projection-edge, occlusion, parallax, and inpaint-output checks
MUST be "inconclusive"; do not invent visual defects or award a clean pass.
Use the solve-health JSON for structural findings and say that a rendered
camera view is still required. A pass requires both a real output image and
healthy deterministic diagnostics.

If provenance says headless_projection_reconstruction, the supplied image is
the canonical recovered-camera output reconstructed from the workflow's real
projection plates, per-pixel mattes, and relief-mesh topology. You MAY judge
camera framing, canonical projection boundaries/holes, inpaint seams, and
colour continuity from it. You MUST keep occlusion_tearing inconclusive as a
visual check because a canonical pose cannot prove orbit/grazing behavior.
Use DETERMINISTIC_SOLVE_HEALTH_JSON.orbit_coverage for objective small-orbit
coverage findings, clearly labeling those findings as geometry-only. Do not
claim WebGL lighting, tone mapping, or interactive occlusion was rendered.

When additional images are supplied, their order and role are stated in the
user message. IMAGE 2 is a coverage matte (white = projected output exists,
black = a canonical hole). IMAGE 3 is the source/reference plate. Compare
IMAGE 1 against IMAGE 3 for unintended subject, architecture, perspective,
or material replacement. Inpaint may plausibly fill masked regions, but a
wholesale unrelated scene or changed camera is a failure, not a strength.

INTERPRET THE TELEMETRY LITERALLY. ``scene_health.flags`` contains the
calibrated findings and severities. Do not manufacture a defect from a raw
number when no corresponding flag was emitted. In particular:
- per-layer ``matte_coverage`` is the fraction of the whole frame deliberately
  owned by that depth band. A small foreground or midground band is normal and
  is NOT missing final coverage; only a ``near_empty_matte`` flag means the
  layer is pathologically empty;
- per-layer ``torn_fraction`` counts grid quads intentionally removed by band
  clipping and can be high on a healthy narrow band; only a
  ``torn_excessive`` flag establishes that defect;
- ``headless_evidence.coverage_fraction`` is the union coverage of the retained
  canonical output and is the correct basis for canonical missing-output
  claims;
- ``source_comparison`` is an exposure-tolerant structural comparison between
  IMAGE 1 and IMAGE 3. A ``severe`` result is a release failure; deliberate
  broad replacement requires an explicit external release review;
- ``camera_looks_up`` and ``scale_unverified`` are WARN findings requiring
  verification, not automatic failures. Do not call a gravity flip visually
  evident unless the supplied output itself establishes a ground-level plate.

Return ONLY this JSON object:
{
  "verdict": "pass|warn|fail|inconclusive",
  "score_0_100": 0,
  "summary": "concise final-output assessment",
  "checks": {
    "camera_match": {"status": "pass|warn|fail|inconclusive", "evidence": "..."},
    "projection_edges": {"status": "pass|warn|fail|inconclusive", "evidence": "..."},
    "occlusion_tearing": {"status": "pass|warn|fail|inconclusive", "evidence": "..."},
    "layer_inpaint": {"status": "pass|warn|fail|inconclusive", "evidence": "..."},
    "color_alpha": {"status": "pass|warn|fail|inconclusive", "evidence": "..."}
  },
  "issues": [
    {"severity": "warn|fail", "category": "...", "evidence": "...", "action": "..."}
  ],
  "strengths": ["..."],
  "recommended_next_run": ["specific setting or validation action"]
}
"""

_MAX_TOKENS = 2200
_VALID_VERDICTS = {"pass", "warn", "fail", "inconclusive"}


@dataclass(slots=True)
class OutputAssessmentResult:
    """Parsed terminal assessment and display-ready report."""

    payload: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    provider: str = ""
    model: str = ""
    ok: bool = False


def _response_format() -> dict[str, Any]:
    check = {"type": "object", "properties": {
        "status": {"type": "string"}, "evidence": {"type": "string"}}}
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "score_0_100": {"type": "number"},
            "summary": {"type": "string"},
            "checks": {"type": "object", "properties": {
                name: check for name in (
                    "camera_match", "projection_edges", "occlusion_tearing",
                    "layer_inpaint", "color_alpha")}},
            "issues": {"type": "array", "items": {"type": "object"}},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "recommended_next_run": {
                "type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "summary", "checks", "issues",
                     "recommended_next_run"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "atlas_output_assessment", "schema": schema, "strict": False}}


def _looks_like_output_assessment(payload: dict[str, Any]) -> bool:
    return (str(payload.get("verdict", "")).lower() in _VALID_VERDICTS
            and isinstance(payload.get("summary"), str)
            and isinstance(payload.get("checks"), dict))


def format_output_assessment_report(
        payload: dict[str, Any], *, provider: str, model: str) -> str:
    verdict = str(payload.get("verdict", "inconclusive")).upper()
    score = payload.get("score_0_100")
    score_text = f"  ·  score {score}/100" if score is not None else ""
    lines = [f"ATLAS OUTPUT ASSESSMENT — {verdict}{score_text}",
             f"provider {provider} / {model}", "",
             str(payload.get("summary") or "No summary returned."), "", "CHECKS"]
    for name, check in (payload.get("checks") or {}).items():
        if not isinstance(check, dict):
            continue
        lines.append(f"  {name}: {str(check.get('status', 'inconclusive')).upper()}"
                     f" — {check.get('evidence', '')}")
    issues = payload.get("issues") or []
    if issues:
        lines += ["", "ISSUES"]
        for issue in issues:
            if isinstance(issue, dict):
                lines.append(
                    f"  ! {str(issue.get('severity', 'warn')).upper()} "
                    f"{issue.get('category', 'quality')}: {issue.get('evidence', '')}"
                    f" — {issue.get('action', '')}")
            else:
                lines.append(f"  ! {issue}")
    actions = payload.get("recommended_next_run") or []
    if actions:
        lines += ["", "NEXT RUN"] + [f"  • {action}" for action in actions]
    corrections = payload.get("grounding_corrections") or []
    if corrections:
        lines += ["", "DETERMINISTIC GROUNDING"] + [
            f"  • {item}" for item in corrections]
    return "\n".join(lines)


def assess_output(
    image_path: str | Path,
    *,
    solve_summary: dict[str, Any],
    image_provenance: str,
    coverage_path: str | Path | None = None,
    source_reference_path: str | Path | None = None,
    provider: str = "ollama",
    model: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    offload_model: bool = False,
    timeout_seconds: float = 180.0,
) -> OutputAssessmentResult:
    """Assess a terminal camera-view image; provider errors fail soft."""

    helper = create_multimodal_provider(
        provider, model=model, base_url=base_url or None, api_key=api_key,
        timeout_seconds=timeout_seconds)
    try:
        model_info = helper.validate_vision_model()
        summary_json = json.dumps(solve_summary, ensure_ascii=False, separators=(",", ":"))
        # A pathological debug payload should not consume the model's entire
        # context window. Health summaries are normally <10 KiB.
        if len(summary_json) > 24000:
            summary_json = summary_json[:24000] + "…[truncated]"
        image_paths = [Path(image_path)]
        image_roles = ["IMAGE 1: retained final/canonical output under review"]
        if coverage_path:
            image_paths.append(Path(coverage_path))
            image_roles.append(
                "IMAGE 2: coverage matte; white means projected output exists, "
                "black means a canonical output hole")
        if source_reference_path:
            image_paths.append(Path(source_reference_path))
            image_roles.append(
                f"IMAGE {len(image_paths)}: source/reference plate for detecting "
                "unintended inpaint replacement or camera/content drift")
        user_text = (
            "Review these Atlas terminal evidence images.\n"
            + "\n".join(image_roles) + "\n"
            f"IMAGE_PROVENANCE: {image_provenance}\n"
            f"DETERMINISTIC_SOLVE_HEALTH_JSON: {summary_json}")
        if extra_instructions.strip():
            user_text += "\nRUN_INTENT: " + extra_instructions.strip()

        if helper.provider == "ollama":
            request = {
                "model": model_info.id,
                "stream": False,
                "messages": [
                    {"role": "system", "content": OUTPUT_ASSESSMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text,
                     "images": [_image_base64(path) for path in image_paths]},
                ],
                "format": "json",
                "options": {"num_predict": _MAX_TOKENS},
            }
            if offload_model:
                request["keep_alive"] = 0
            response = helper._request_json("/api/chat", request)
            content = str(response.get("message", {}).get("content", "")).strip()
        else:
            request = {
                "model": model_info.id,
                "stream": False,
                "messages": [
                    {"role": "system", "content": OUTPUT_ASSESSMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        [{"type": "text", "text": user_text}]
                        + [{"type": "image_url", "image_url": {
                            "url": _image_data_url(path)}} for path in image_paths])},
                ],
                "temperature": 0,
                "max_tokens": _MAX_TOKENS,
            }
            if helper.provider in ("lmstudio", "openai"):
                request["response_format"] = _response_format()
            if offload_model and helper.provider == "lmstudio":
                request["ttl"] = 2
            try:
                response = helper._request_json("/chat/completions", request)
            except RuntimeError as exc:
                if ("response_format" not in request
                        or not _is_response_format_error(str(exc))):
                    raise
                request = dict(request)
                request.pop("response_format", None)
                response = helper._request_json("/chat/completions", request)
            content = _openai_chat_content(response)

        parsed = _parse_model_json(content)
        if not _looks_like_output_assessment(parsed):
            snippet = content.strip()
            if len(snippet) > 1000:
                snippet = snippet[:1000] + " …[truncated]"
            return OutputAssessmentResult(
                payload=parsed, provider=helper.provider, model=model_info.id,
                report=("ATLAS OUTPUT ASSESSMENT FAILED — reply was not usable JSON\n\n"
                        + (snippet or "(empty reply)")), ok=False)

        result = OutputAssessmentResult(
            payload=parsed, provider=helper.provider, model=model_info.id, ok=True,
            report=format_output_assessment_report(
                parsed, provider=helper.provider, model=model_info.id))
        if offload_model:
            # Reuse the provider-specific, verified offload behavior already
            # established by the pre-flight assessor.
            from atlas_camera.inference.assessor import (
                _offload_after_assessment, _vram_free_gb)
            before = _vram_free_gb()
            status = _offload_after_assessment(helper, model_info.id)
            if before is not None and helper.provider != "openai":
                import time
                time.sleep(1.5)
                after = _vram_free_gb()
                if after is not None:
                    status += f"  · VRAM free {before:.1f} → {after:.1f} GB"
            result.report += "\n\nMODEL OFFLOAD: " + status
        return result
    except (RuntimeError, ValueError, OSError) as exc:
        return OutputAssessmentResult(
            provider=provider, model=model, ok=False,
            report=f"ATLAS OUTPUT ASSESSMENT UNAVAILABLE — {exc}")
