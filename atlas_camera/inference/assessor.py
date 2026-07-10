"""Atlas image assessment — a local VLM pre-flight for the whole DMP pipeline.

Runs the user's chosen local vision-language model (Ollama / LM Studio /
llama.cpp — the same provider layer `AtlasVLMScaleCues` uses) over the input
photo with an expert system prompt that encodes Atlas Camera's full settings
knowledge: geometry-derivation strategy choice, depth-model selection, sky
separation, depth-band layering, disocclusion fill, edge mattes, multi-angle
patch planning, and a camera-move viability rubric. The result is a
structured recommendation the artist reads BEFORE the heavy pipeline runs
(`AtlasAssessImage` pauses the graph on its image output until the artist
has applied the recommended settings and clicks ▶ Continue Workflow).

Advisory only, same principle as the scale-cue helper: the VLM never changes
a setting itself — it recommends, the artist decides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.inference.multimodal_helper import (
    _image_base64,
    _image_data_url,
    _openai_chat_content,
    _parse_model_json,
    create_multimodal_provider,
)

# ---------------------------------------------------------------------------
# The expert instruction prompt — Atlas Camera's settings knowledge, written
# for a small local VLM. Keep decision RULES (not just option lists) so a 4-8B
# model can apply them mechanically.
# ---------------------------------------------------------------------------
ATLAS_ASSESSMENT_SYSTEM_PROMPT = """\
You are the Atlas Camera setup assistant. Atlas Camera turns ONE photo into a
2.5D camera-projection matte-painting scene (recovered camera + layered
projection geometry) inside ComfyUI, for export to Nuke/Maya. Your job:
look at the photo and produce a concrete setup plan — which settings, which
layers, and an honest viability assessment of how far a camera move can go.

THE PIPELINE YOU ARE CONFIGURING (in graph order):
1. Camera solve (GeoCalib learned prior) + shared metric depth (Depth Anything V2).
2. Sky separation: a SAM segmentation drives a flat far "sky dome" card.
3. Depth-band layers: the photo splits into metric depth bands (background /
   midground / foreground); occluded bands get INPAINTED clean plates; each
   band gets its own relief mesh; fill_occluded synthesizes geometry behind
   foreground occluders so inpainted content has somewhere to land.
4. Optional multi-angle patch: a Qwen image-edit LoRA generates a novel view
   at a named angle to fill what the original camera never saw.
5. Export: all layers to one Nuke script.

DECISION RULES — apply these to what you SEE in the photo:

depth_model: the default (Depth Anything 3 metric, "DA3METRIC-LARGE") handles
interiors AND exteriors with one model and measurably fewer mesh tears — keep
"outdoor" in the recommendation only as the V2 fallback label. Recommend
"indoor"/"outdoor" (the Depth Anything V2 split) ONLY if DA3 is unavailable
(the [neural-da3] extra is not installed); V2 interiors with the outdoor model
often fail their metric ground fit — flag that when it applies.

scene_type (geometry strategy preset — pick ONE):
- "organic": natural scenes, rocks, general default when unsure.
- "mountains": large-scale terrain (relief quality raised automatically).
- "forests": dense foliage/canopy (also relaxes tear threshold — noisy depth).
- "aerial": drone/high viewpoint over buildings (boxes over relief ground).
- "indoor": orthogonal interiors (room cuboid fitting).
- "outdoor": exterior architecture with sloped roofs (RANSAC planes).
- "simple_walls": plain vertical facades.
- "towers_spires": tall structures whose tops matter (churches, towers).

sky: use_sky_dome true whenever real sky is visible (even a sliver). The SAM
prompt is normally "sky"; suggest a different prompt for unusual cases
(e.g. "sky, clouds" or "ceiling glow" does NOT count — interiors get
use_sky_dome false).

bands (depth layering) — the most important judgment call:
- near_pct/far_pct are positions along the scene's LOG-depth range (0.5 = the
  geometric mean of the depth range, i.e. perceptually mid-scene) — NOT pixel
  percentiles.
- Default 2-band split: foreground 0.0-0.55, background 0.55-1.0 (log-depth
  of the depth distribution).
- Add a third band when there is a distinct midground subject (e.g. a
  building between near ground and far mountains): fg 0-0.20, mid 0.20-0.65,
  bg 0.65-1.0 is a proven starting point.
- CRITICAL RULE: band edges must NOT slice through the main subject — if a
  boundary would cut a building/vehicle/person in half, move the boundary so
  the subject sits whole inside one band.
- A band needs an inpainted clean plate whenever something NEARER occludes
  it. The frontmost band never needs inpainting (wire the original photo).
- fill_occluded: true on every band that has a nearer occluder.
- embed_matte: true on all bands (crisp per-pixel edges; cheap).

relief settings:
- relief_grid 128 is the default. Recommend 256 for detailed/noisy scenes or
  4K plates where edge quality matters; 384+ with depth_edge_rel 1.5 for
  band-clipped meshes on complex subjects (spacecraft, machinery).
- depth_edge_rel 0.5 default; raise toward 1.0-1.5 for foliage/canopy or
  noisy depth (fewer torn holes); keep low for crisp architecture.

camera-move viability — be honest and conservative. Score 0-10 and estimate
max_orbit_deg using this rubric:
- Single continuous surface, subject far away, little occlusion: orbit
  5-15 degrees works from the relief mesh alone; score 7-9.
- Clear layered scene (distinct fg/bg planes) with clean-plate inpainting:
  15-30 degrees; score 6-8. Dolly-in works if disocclusions are filled.
- Heavy occlusion, many overlapping objects, or a dominant very-near
  foreground: 5-10 degrees before holes dominate; score 3-5; recommend
  multi-angle patches for anything further.
- Faces/people close to camera, transparent/reflective surfaces, or dense
  thin structures (rigging, branches against sky): warn — depth will be
  unreliable there; score accordingly.
- Beyond ~30 degrees ALWAYS requires a multi-angle patch (novel view
  generation at a named angle: azimuths every 45deg, elevations
  low-angle/eye-level/elevated/high-angle, distances close-up/medium/wide).

scale: if a known-size object is clearly visible (person, car, door),
mention it as a reference-scale anchor opportunity (person ~1.75m, sedan
~1.45m tall, door ~2.0m) — this beats depth-based scale.

OUTPUT FORMAT — respond with ONLY this JSON object, no prose outside it:
{
  "scene_summary": "one paragraph: what the photo shows, from a matte-painting rigging perspective",
  "viability": {
    "score_0_10": 0,
    "max_orbit_deg": 0,
    "dolly_ok": false,
    "notes": "what limits the move; what breaks first"
  },
  "layers": [
    {"name": "sky", "role": "sky", "notes": "..."},
    {"name": "bg", "role": "background", "near_pct": 0.55, "far_pct": 1.0,
     "needs_inpaint": true, "fill_occluded": true, "notes": "..."}
  ],
  "recommended_settings": {
    "depth_model": "outdoor",
    "scene_type": "organic",
    "relief_grid": 128,
    "depth_edge_rel": 0.5,
    "sky": {"use_sky_dome": true, "sam_prompt": "sky"},
    "patch": {"recommended": false, "suggested_views": [], "notes": "..."},
    "scale_reference": {"present": false, "object": "", "notes": ""}
  },
  "warnings": ["..."]
}
"""

_USER_PROMPT = (
    "Assess this photo for the Atlas Camera 2.5D projection pipeline. Apply "
    "the decision rules from your instructions to what you actually see, and "
    "return ONLY the JSON object in the specified format."
)


@dataclass(slots=True)
class AssessmentResult:
    """Parsed VLM assessment plus a formatted human-readable report."""

    payload: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    provider: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)
    ok: bool = False


def assess_image(
    image_path: str | Path,
    *,
    provider: str = "ollama",
    model: str = "",
    base_url: str | None = None,
    extra_instructions: str = "",
    timeout_seconds: float = 180.0,
) -> AssessmentResult:
    """Run the assessment prompt over one image via a local VLM provider.

    Fails SOFT (ok=False + a report explaining how to start a provider) on
    any connectivity/model error — the pause gating in `AtlasAssessImage`
    still works without an assessment, the artist just gets no advice.
    """
    helper = create_multimodal_provider(
        provider, model=model, base_url=base_url or None, timeout_seconds=timeout_seconds)
    try:
        model_info = helper.validate_vision_model()
        user_text = _USER_PROMPT
        if extra_instructions.strip():
            user_text += "\nAdditional artist instructions: " + extra_instructions.strip()

        if helper.provider == "ollama":
            payload = {
                "model": model_info.id,
                "stream": False,
                "messages": [
                    {"role": "system", "content": ATLAS_ASSESSMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text,
                     "images": [_image_base64(Path(image_path))]},
                ],
                "format": "json",
                "options": {"num_predict": 2200},
            }
            response = helper._request_json("/api/chat", payload)
            content = str(response.get("message", {}).get("content", "")).strip()
        else:  # lmstudio / llamacpp (OpenAI-compatible chat)
            payload = {
                "model": model_info.id,
                "stream": False,
                "messages": [
                    {"role": "system", "content": ATLAS_ASSESSMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url",
                         "image_url": {"url": _image_data_url(Path(image_path))}},
                    ]},
                ],
                "temperature": 0,
                "max_tokens": 2200,
            }
            response = helper._request_json("/chat/completions", payload)
            content = _openai_chat_content(response)

        parsed = _parse_model_json(content)
        result = AssessmentResult(
            payload=parsed, provider=helper.provider, model=model_info.id, ok=True)
        result.warnings = [str(w) for w in parsed.get("warnings", []) if w]
        result.report = format_assessment_report(parsed, provider=helper.provider,
                                                 model=model_info.id)
        return result
    except (RuntimeError, ValueError, OSError) as exc:
        return AssessmentResult(
            ok=False, provider=provider, model=model,
            report=(
                f"ATLAS ASSESSMENT UNAVAILABLE — {exc}\n\n"
                "Start a local VLM provider and re-queue:\n"
                "  ollama:   ollama run gemma3:4b   (default http://127.0.0.1:11434)\n"
                "  lmstudio: load a vision model    (default http://127.0.0.1:1234/v1)\n"
                "  llamacpp: llama-server with a vision model (default http://127.0.0.1:8080/v1)\n\n"
                "You can also toggle `proceed` and continue without an assessment."
            ),
        )


def format_assessment_report(payload: dict[str, Any], *, provider: str = "",
                             model: str = "") -> str:
    """Human-readable report from the assessment JSON — shown in a text node
    while the graph is paused, so the artist can apply the recommendations."""
    lines: list[str] = []
    header = "ATLAS IMAGE ASSESSMENT"
    if model:
        header += f"  ({provider}/{model})"
    lines.append(header)
    lines.append("=" * len(header))

    if payload.get("scene_summary"):
        lines += ["", str(payload["scene_summary"])]

    v = payload.get("viability") or {}
    if v:
        lines += ["", f"VIABILITY  {v.get('score_0_10', '?')}/10   "
                      f"max orbit ~{v.get('max_orbit_deg', '?')} deg   "
                      f"dolly {'OK' if v.get('dolly_ok') else 'limited'}"]
        if v.get("notes"):
            lines.append(f"  {v['notes']}")

    layers = payload.get("layers") or []
    if layers:
        lines += ["", "LAYERS"]
        for l in layers:
            band = ""
            if l.get("near_pct") is not None or l.get("far_pct") is not None:
                band = f"  band {l.get('near_pct', 0)}-{l.get('far_pct', 1)}"
            flags = []
            if l.get("needs_inpaint"):
                flags.append("inpaint")
            if l.get("fill_occluded"):
                flags.append("fill_occluded")
            flag_s = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {l.get('name', '?'):8s} {l.get('role', ''):10s}{band}{flag_s}")
            if l.get("notes"):
                lines.append(f"           {l['notes']}")

    rs = payload.get("recommended_settings") or {}
    if rs:
        lines += ["", "RECOMMENDED SETTINGS"]
        for key in ("depth_model", "scene_type", "relief_grid", "depth_edge_rel"):
            if rs.get(key) is not None:
                lines.append(f"  {key:15s} {rs[key]}")
        sky = rs.get("sky") or {}
        if sky:
            lines.append(f"  sky dome        {'YES, SAM prompt: ' + str(sky.get('sam_prompt', 'sky')) if sky.get('use_sky_dome') else 'no (no sky visible)'}")
        patch = rs.get("patch") or {}
        if patch.get("recommended"):
            views = ", ".join(patch.get("suggested_views") or []) or "artist's choice"
            lines.append(f"  multi-angle patch  YES — {views}")
            if patch.get("notes"):
                lines.append(f"                     {patch['notes']}")
        scale = rs.get("scale_reference") or {}
        if scale.get("present"):
            lines.append(f"  scale reference    {scale.get('object', '?')} — {scale.get('notes', '')}")

    warns = [str(w) for w in payload.get("warnings") or [] if w]
    if warns:
        lines += ["", "WARNINGS"]
        lines += [f"  ! {w}" for w in warns]

    lines += ["", "Apply the settings above, then click ▶ Continue Workflow (or toggle",
              "`proceed`) and Queue — the graph is paused until you do."]
    return "\n".join(lines)
