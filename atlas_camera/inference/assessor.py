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
    _is_response_format_error,
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

STAGED 5-LAYER MASTER PLAN (staged_layers) — ALWAYS produce this block:
the staged master workflow splits every photo into at most FIVE fixed layers,
each its own stage the artist enables one at a time (far to near):
  sky            SAM-segmented sky card
  far  80-100%   farthest depth band
  bg   60-80%
  mid  30-60%
  fg   0-30%     nearest band
Each band is additionally SCOPED by its own SAM segmentation (the layer keeps
band membership AND segment membership), so for EACH of the five layers judge:
- present: does the photo really have distinct content there? NOT every image
  has a sky; an interior never does; a flat vista may have an empty mid band;
  a distant landscape may have nothing in fg. Absent layers get present=false
  and an EMPTY sam_prompt — the artist leaves that stage bypassed.
- sam_prompt: a short segmentation prompt naming the CONCRETE things occupying
  that layer, phrased for a text-prompted segmenter ("rock formations",
  "church tower", "pine trees", "parked cars", "cobblestone street") — never
  abstract words like "background", "midground", or "distant objects".
- geometry (bands only; sky is always a flat card): how this layer's
  projection surface should be built —
  "ground": the layer is a flat horizontal surface the camera stands over
    (desert floor, water, road, plaza/hangar floor). Projected onto the
    exact analytic ground plane: zero depth noise, perfectly smooth.
  "card": a distant or flat-facing layer with negligible internal depth
    relative to its distance (mountains at the horizon, a hangar's flat
    back wall, a city-skyline backdrop). One flat plane at the band's
    depth: never tears.
  "relief": anything with real 3D shape inside the band — the DEFAULT
    whenever unsure. Rule of thumb: if orbiting ~15 degrees would reveal
    parallax INSIDE the layer, it needs "relief"; if the layer would move
    as one rigid poster, "card"; if it is the floor itself, "ground".
- near_pct / far_pct (bands only, OPTIONAL): adjusted band boundaries as
  positions 0-1 along the scene's log-depth range — provide them ONLY when a
  fixed boundary above would slice a layer's main subject in half (say why
  in notes). Keep bands CONTIGUOUS: one band's far_pct must equal the next
  band's near_pct. Omit the fields to accept the fixed slots.

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
  "staged_layers": {
    "sky": {"present": true, "sam_prompt": "sky", "notes": "..."},
    "far": {"present": true, "sam_prompt": "rock formations", "geometry": "card", "notes": "..."},
    "bg":  {"present": true, "sam_prompt": "...", "geometry": "relief",
            "near_pct": 0.55, "far_pct": 0.8,
            "notes": "boundary lowered to 0.55 so the butte sits whole in bg"},
    "mid": {"present": false, "sam_prompt": "", "geometry": "relief", "far_pct": 0.55,
            "notes": "nothing distinct in this band"},
    "fg":  {"present": true, "sam_prompt": "desert floor", "geometry": "ground", "notes": "..."}
  },
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

# Response budget. The staged_layers block grew the output; 2200 was observed
# truncating a verbose 12B model's reply into unparseable JSON (lmstudio/gemma).
_ASSESSMENT_MAX_TOKENS = 3200


def _assessment_response_format() -> dict[str, Any]:
    """OpenAI-style structured-output request for the assessment JSON.

    LM Studio grammar-constrains decoding against this (fixes invalid-JSON
    replies at the source); servers that reject `response_format` fall back
    via the same `_is_response_format_error` chain `analyze_image` uses.
    Deliberately loose (strict=False, open sub-objects) — the report degrades
    gracefully on missing fields, and an over-strict schema is itself a
    failure mode on small local models.
    """
    layer_slot = {"type": "object", "properties": {
        "present": {"type": "boolean"},
        "sam_prompt": {"type": "string"},
        "geometry": {"type": "string"},
        "near_pct": {"type": "number"},
        "far_pct": {"type": "number"},
        "notes": {"type": "string"},
    }, "required": ["present", "sam_prompt"]}
    # "required" matters: LM Studio's grammar treats bare properties as
    # OPTIONAL, and the model nondeterministically omitted the whole
    # staged_layers block until it was required (found live, gemma-4-12b —
    # one call had it perfect, the next skipped it entirely).
    schema = {
        "type": "object",
        "properties": {
            "scene_summary": {"type": "string"},
            "viability": {"type": "object", "properties": {
                "score_0_10": {"type": "number"},
                "max_orbit_deg": {"type": "number"},
                "dolly_ok": {"type": "boolean"},
                "notes": {"type": "string"},
            }},
            "layers": {"type": "array", "items": {"type": "object"}},
            "staged_layers": {"type": "object", "properties": {
                key: layer_slot for key in STAGED_LAYER_KEYS},
                "required": list(STAGED_LAYER_KEYS)},
            "recommended_settings": {"type": "object"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scene_summary", "viability", "staged_layers",
                     "recommended_settings"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "atlas_assessment", "schema": schema, "strict": False}}


# Any real assessment carries at least one of these; a payload with none of
# them is `_parse_model_json`'s prose/garbage fallback (or a reply that was
# valid JSON but not an assessment) — treat both as a failed assessment.
_ASSESSMENT_KEYS = ("scene_summary", "viability", "staged_layers",
                    "recommended_settings", "layers")


def _looks_like_assessment(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in _ASSESSMENT_KEYS)


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
    api_key: str | None = None,
    extra_instructions: str = "",
    offload_model: bool = False,
    timeout_seconds: float = 180.0,
) -> AssessmentResult:
    """Run the assessment prompt over one image via a local VLM provider.

    Fails SOFT (ok=False + a report explaining how to start a provider) on
    any connectivity/model error — the pause gating in `AtlasAssessImage`
    still works without an assessment, the artist just gets no advice.
    """
    helper = create_multimodal_provider(
        provider, model=model, base_url=base_url or None, api_key=api_key,
        timeout_seconds=timeout_seconds)
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
                "options": {"num_predict": _ASSESSMENT_MAX_TOKENS},
            }
            if offload_model:
                payload["keep_alive"] = 0  # unload right after responding
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
                "max_tokens": _ASSESSMENT_MAX_TOKENS,
            }
            # Structured output where the provider supports it (lmstudio +
            # openai cloud; llamacpp stays plain, matching its provider
            # class), with analyze_image's established rejection fallback.
            if helper.provider in ("lmstudio", "openai"):
                payload["response_format"] = _assessment_response_format()
            if offload_model and helper.provider == "lmstudio":
                payload["ttl"] = 2  # LM Studio JIT auto-evict, seconds
            try:
                response = helper._request_json("/chat/completions", payload)
            except RuntimeError as exc:
                if "response_format" not in payload or not _is_response_format_error(str(exc)):
                    raise
                plain_payload = dict(payload)
                plain_payload.pop("response_format", None)
                response = helper._request_json("/chat/completions", plain_payload)
            content = _openai_chat_content(response)

        parsed = _parse_model_json(content)
        if not _looks_like_assessment(parsed):
            # The provider answered, but not with a usable assessment. Fail
            # like a connectivity error (ok=False -> the node never caches
            # it, so re-queuing actually retries) but SHOW the raw reply.
            # Deliberately NO offload here: the model stays warm so the
            # retry is fast.
            return AssessmentResult(
                ok=False, provider=helper.provider, model=model_info.id,
                payload=parsed,
                warnings=[str(w) for w in parsed.get("warnings", []) if w],
                report=format_parse_failure_report(
                    content, provider=helper.provider, model=model_info.id))
        result = AssessmentResult(
            payload=parsed, provider=helper.provider, model=model_info.id, ok=True)
        result.warnings = [str(w) for w in parsed.get("warnings", []) if w]
        result.report = format_assessment_report(parsed, provider=helper.provider,
                                                 model=model_info.id)
        if offload_model:
            # Verify the offload actually freed VRAM (spec-panel finding: a
            # best-effort unload with no verification is unfalsifiable). The
            # unload is out-of-process and slightly async — sample, wait,
            # resample; skip silently when no CUDA is visible.
            before = _vram_free_gb()
            status = _offload_after_assessment(helper, model_info.id)
            if before is not None and helper.provider != "openai":
                import time as _time
                _time.sleep(1.5)
                after = _vram_free_gb()
                if after is not None:
                    status += f"  · VRAM free {before:.1f} → {after:.1f} GB"
            result.report += "\n\nMODEL OFFLOAD: " + status
        return result
    except (RuntimeError, ValueError, OSError) as exc:
        return AssessmentResult(
            ok=False, provider=provider, model=model,
            report=(
                f"ATLAS ASSESSMENT UNAVAILABLE — {exc}\n\n"
                "Start a VLM provider and re-queue:\n"
                "  ollama:   ollama run gemma3:4b   (default http://127.0.0.1:11434)\n"
                "  lmstudio: load a vision model    (default http://127.0.0.1:1234/v1)\n"
                "  llamacpp: llama-server with a vision model (default http://127.0.0.1:8080/v1)\n"
                "  openai:   no local model needed — set api_key (or OPENAI_API_KEY);\n"
                "            any OpenAI-compatible host via base_url (OpenRouter, ...)\n\n"
                "You can also toggle `proceed` and continue without an assessment."
            ),
        )


def _vram_free_gb() -> float | None:
    """Device-wide free VRAM in GB (cudaMemGetInfo is global, so it sees
    other processes' allocations — exactly what an out-of-process ollama/
    LM Studio unload changes). None when CUDA/torch is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9
    except Exception:
        pass
    return None


def _offload_after_assessment(helper, model_id: str) -> str:
    """Best-effort VRAM offload of the assessment VLM, per provider.

    Called only after a SUCCESSFUL assessment (a failed one keeps the model
    warm so the retry is fast). Returns a one-line status for the report —
    offloading is inherently provider-specific and partly out of our hands:
    - ollama: keep_alive=0 rode the chat request already; an explicit
      /api/generate with keep_alive=0 forces the unload even if a server
      config overrode the request-level value.
    - lmstudio: request `ttl` auto-evicts JIT-loaded models; GUI-loaded
      models only unload via the `lms` CLI (tried when on PATH, targeting
      just this model — never `--all`, the user may have an embedding
      model resident that other tooling depends on).
    - llamacpp: llama-server owns its model for the process lifetime; there
      is no unload API. Honest status instead of pretending.
    - openai: cloud — nothing resident locally.
    """
    provider = helper.provider
    if provider == "openai":
        return "nothing to offload (cloud provider)"
    if provider == "llamacpp":
        return ("not supported — llama-server holds its model for the process "
                "lifetime; restart llama-server to free VRAM")
    if provider == "ollama":
        try:
            helper._request_json("/api/generate", {"model": model_id, "keep_alive": 0})
            return f"'{model_id}' unloaded (keep_alive=0)"
        except (RuntimeError, ValueError, OSError):
            return "keep_alive=0 was set on the request; explicit unload call failed (harmless)"
    # lmstudio
    import shutil
    import subprocess
    lms = shutil.which("lms")
    if lms is None:
        return ("ttl=2s set (auto-evicts JIT-loaded models); for GUI-loaded models "
                "install LM Studio's 'lms' CLI on PATH for a guaranteed unload")
    try:
        subprocess.run([lms, "unload", model_id], capture_output=True,
                       timeout=30, check=True)
        return f"'{model_id}' unloaded via lms CLI"
    except Exception as exc:  # CLI identifier mismatch / timeout — degrade to ttl
        return (f"lms unload failed ({type(exc).__name__}); ttl=2s still "
                "auto-evicts JIT-loaded models")


def format_parse_failure_report(raw: str, *, provider: str = "",
                                model: str = "") -> str:
    """Shown on the node when the VLM replied but not with usable JSON —
    the artist needs to SEE what came back, and to know a re-queue retries."""
    snippet = (raw or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + " …[truncated]"
    header = "ATLAS ASSESSMENT FAILED — reply was not a usable assessment"
    if model:
        header += f"  ({provider}/{model})"
    return "\n".join([
        header,
        "=" * len(header),
        "",
        "The VLM responded, but no assessment JSON could be extracted.",
        "Raw reply (start):",
        "",
        snippet or "(empty response)",
        "",
        "This result is NOT cached — just Queue again to retry. If it keeps",
        "happening: try a larger / better instruction-following vision model,",
        "and check the provider's console for context-length truncation.",
        "",
        "You can also toggle `proceed` and continue without an assessment.",
    ])


# The staged master workflow's fixed layer slots, far to near. Order matters:
# it is the AtlasAssessImage output order and the report's display order.
STAGED_LAYER_KEYS = ("sky", "far", "bg", "mid", "fg")

_STAGED_BAND_LABELS = {"sky": "sky card", "far": "80-100%", "bg": "60-80%",
                       "mid": "30-60%", "fg": "0-30%"}


def staged_layer_prompts(payload: dict[str, Any]) -> dict[str, str]:
    """Per-slot SAM3 prompt strings from an assessment's `staged_layers` block.

    Absent layers (present=false, or missing from the payload) yield "" so a
    wired-but-bypassed scope row stays inert — EXCEPT sky, which falls back to
    the literal "sky" (the sky SAM3 always runs in the staged workflow's
    always-on SHARED group; a no-match prompt on a skyless photo just returns
    an empty mask, which is the correct sky mask for that photo anyway).
    """
    staged = payload.get("staged_layers") or {}
    out: dict[str, str] = {}
    for key in STAGED_LAYER_KEYS:
        entry = staged.get(key) or {}
        prompt = str(entry.get("sam_prompt") or "").strip()
        out[key] = prompt if entry.get("present") else ""
    if not out["sky"]:
        out["sky"] = "sky"
    return out


# The band geometry vocabulary AtlasCleanPlateLayer accepts — keep in sync
# with nodes._BAND_GEOMETRY_CHOICES (the node errors loudly on anything else,
# so this helper must never emit an out-of-vocabulary value).
STAGED_GEOMETRY_CHOICES = ("relief", "card", "ground")
STAGED_BAND_KEYS = ("far", "bg", "mid", "fg")


# The staged master's fixed band boundaries (fg|mid, mid|bg, bg|far) as
# log-depth positions — the VLM's near_pct/far_pct suggestions move these.
STAGED_DEFAULT_BOUNDARIES = (0.3, 0.6, 0.8)


def staged_layer_bands(payload: dict[str, Any]) -> dict[str, str]:
    """Watertight per-band near/far percentages from the VLM's optional
    band-boundary suggestions, as `"near_pct=<f> far_pct=<f>"` strings for
    the band nodes' `band_override` inputs ("" everywhere when the payload
    carries no staged plan — the nodes keep their own widgets).

    The three shared boundaries (fg|mid, mid|bg, bg|far) are derived JOINTLY
    — each from the two adjacent bands' suggestions, falling back to the
    fixed slots — so adjacent bands share edges EXACTLY by construction and
    can never reintroduce the metric band-gap defect. A non-monotonic or
    out-of-range suggestion set resets ALL boundaries to the defaults rather
    than emitting a partially-applied plan.
    """
    staged = payload.get("staged_layers") or {}
    if not staged:
        return {key: "" for key in STAGED_BAND_KEYS}

    def suggested(key: str, field: str):
        entry = staged.get(key) or {}
        value = entry.get(field)
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if 0.0 <= v <= 1.0 else None

    def boundary(low_key, high_key, default):
        v = suggested(low_key, "far_pct")
        if v is None:
            v = suggested(high_key, "near_pct")
        return default if v is None else v

    b1 = boundary("fg", "mid", STAGED_DEFAULT_BOUNDARIES[0])
    b2 = boundary("mid", "bg", STAGED_DEFAULT_BOUNDARIES[1])
    b3 = boundary("bg", "far", STAGED_DEFAULT_BOUNDARIES[2])
    # STRICT ordering, strictly inside (0, 1): a boundary at 0.0 or 1.0 (or two
    # equal boundaries) makes some band ZERO-WIDTH and its neighbour unbounded —
    # found live 2026-07-16 when a VLM suggested far=[1.0, 1.0]: the far card
    # collapsed and the bg band ran to infinity, its fill_occluded diffusion
    # smear then outranking (farthest-highest) every real layer below it.
    if not (0.0 < b1 < b2 < b3 < 1.0):
        b1, b2, b3 = STAGED_DEFAULT_BOUNDARIES
    fmt = "near_pct={:.3f} far_pct={:.3f}".format
    return {"fg": fmt(0.0, b1), "mid": fmt(b1, b2),
            "bg": fmt(b2, b3), "far": fmt(b3, 1.0)}


def staged_layer_geometry(payload: dict[str, Any]) -> dict[str, str]:
    """Per-band geometry-type recommendations from `staged_layers`.

    Returns "" (no recommendation — the layer node's own band_geometry combo
    applies) for absent layers, missing fields, and anything outside the
    known vocabulary: a hallucinated value must degrade to the default, not
    crash the wired AtlasCleanPlateLayer downstream.
    """
    staged = payload.get("staged_layers") or {}
    out: dict[str, str] = {}
    for key in STAGED_BAND_KEYS:
        entry = staged.get(key) or {}
        geometry = str(entry.get("geometry") or "").strip().lower()
        out[key] = geometry if (
            entry.get("present") and geometry in STAGED_GEOMETRY_CHOICES) else ""
    return out


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

    staged = payload.get("staged_layers") or {}
    if staged:
        bands = staged_layer_bands(payload)
        lines += ["", "STAGED 5-LAYER PLAN  (SAM prompts are wired to the scope rows)"]
        for key in STAGED_LAYER_KEYS:
            entry = staged.get(key) or {}
            if key == "sky":
                band_label = _STAGED_BAND_LABELS["sky"]
            else:
                # resolved (possibly VLM-adjusted, always watertight) range
                parts = dict(p.split("=") for p in bands.get(key, "").split()) if bands.get(key) else {}
                band_label = (f"{float(parts['near_pct']):.0%}-{float(parts['far_pct']):.0%}"
                              if parts else _STAGED_BAND_LABELS.get(key, ""))
            if entry.get("present"):
                mark, prompt = "+", f'SAM "{entry.get("sam_prompt", "")}"'
                geometry = str(entry.get("geometry") or "").strip().lower()
                if key != "sky" and geometry in STAGED_GEOMETRY_CHOICES:
                    prompt += f"  · geometry: {geometry}"
            else:
                mark, prompt = "-", "absent — leave this stage bypassed"
            lines.append(f"  {mark} {key:4s} {band_label:8s} {prompt}")
            if entry.get("notes"):
                lines.append(f"           {entry['notes']}")
        lines.append("  Un-bypass the + rows' MaskComposite in SAM SCOPE (and each + stage's group).")

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
