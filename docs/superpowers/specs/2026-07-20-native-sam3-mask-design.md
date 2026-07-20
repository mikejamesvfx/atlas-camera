# Native SAM3 via `transformers` — design

**Status:** approved, ready for implementation planning
**Date:** 2026-07-20

## Problem

Atlas's sky/scope segmentation cascade (`AtlasInput`'s auto-graph, via `nodes_viewport.py`'s
`segment()` helper) prefers third-party `SAM3Segment` (comfyui-rmbg) and falls back to Atlas's
own `AtlasSemanticMask` (SegFormer/ADE20K) when it's absent. `SAM3Segment` hard-requires
`triton`, which does not exist on Mac (MPS), CPU-only, or AMD boxes — those users can never load
real SAM3 and always land on the weaker SegFormer fallback, even though nothing about SAM3 itself
requires triton.

[lettidude/LiveActionAOV](https://github.com/lettidude/LiveActionAOV) (MIT core) proves this:
its `passes/matte/sam3.py` loads SAM3 straight from `transformers`
(`Sam3Model`/`Sam3Processor`/`Sam3TrackerVideoModel`/`Sam3TrackerVideoProcessor`), no triton
anywhere, device selection is plain `cuda` or `cpu`.

## Goal

Add a native, `transformers`-only SAM3 mask node to Atlas, and make it the preferred tier in the
sky/scope cascade — ahead of both the third-party `SAM3Segment` (removed from the cascade
entirely) and `AtlasSemanticMask` (kept as the final learned fallback).

## Decisions made during brainstorming

1. **Cascade position:** native SAM3 fully supersedes third-party `SAM3Segment` in Atlas's own
   cascade. New order: **native SAM3 → `AtlasSemanticMask` → heuristic (`None`)**. No middle tier
   for the triton-locked node — there is no scenario where preferring it over native is better.
2. **`AtlasSegmentedSDXLInpaint`'s direct `SAM3Segment` call** (per-instance `Separate` mode, for
   per-building inpaint crops) is **out of scope** for this pass. It's a distinct capability
   (instance separation, not union masking) and isn't the node blocked by triton today. Revisit as
   a fast follow.
3. **Packaging:** new `[sam3]` extra pinning `transformers>=5.5.4,<6` (SAM3 model classes only
   exist from transformers ~5.5). The shared `[neural]` extra keeps its looser `>=4.40` floor —
   `[sam3]` is opt-in and version-gated separately, same pattern as `moge`/`neural-da3`.
4. **Device / MPS:** use `resolve_device` (cuda→mps→cpu autodetect, already correct and shared
   with `semantic_segmenter.py`/`depth_estimator.py`). SAM3's tracker/attention ops are new to
   `transformers` and untested on MPS (LiveActionAOV itself never tries MPS) — wrap inference so a
   `RuntimeError` from an unsupported MPS op triggers one silent reload+retry on `cpu`, noted in
   the report. Graceful degrade, never a hard crash, per the project's existing doctrine.
5. **Node interface:** mirrors `AtlasSemanticMask` exactly — comma-separated concepts, union mask
   + report. SAM3's classification head is single-concept-per-forward internally regardless, so a
   comma list already means "one forward pass per concept, union the results" either way. This
   keeps the new node a drop-in interchangeable alternative to `AtlasSemanticMask` in the cascade,
   and matches the interface `segment()` already expects (single prompt string in, mask out).
6. **Model selection:** single constant `DEFAULT_SAM3_MODEL = "facebook/sam3"`. No combo widget —
   there is nothing to select between yet (YAGNI).
7. **HF gated-repo auth:** `facebook/sam3` is gated on Hugging Face (Meta's SAM-License-1.0,
   commercial use permitted, military/ITAR carved out). Port LiveActionAOV's
   `_wrap_if_gated_repo` pattern: translate the raw `OSError: gated repo` / 401 into an actionable
   message (request access at the model page, `hf auth login` or `HF_TOKEN` env, pointer to
   INSTALL.md).
8. **Auth-miss behavior in `AtlasInput`'s auto-graph:** the build-time capability probe only
   checks "is `transformers>=5.5.4` importable" (cheap, no network, no model load) — it does
   **not** also probe for a cached HF token. If the probe passes but the user hasn't authenticated
   yet, `AtlasInput` still routes to native SAM3 and lets the gated-repo error surface at
   execution. This is consistent with how every other gated/missing-weight case in this repo
   already behaves (GeoCalib, SAM3 itself when comfyui-rmbg errors today) — a clear actionable
   error beats a silent quality downgrade to SegFormer on the artist's very first run, where
   they'd have no idea SAM3 was even in play.
9. **Example workflows:** `examples/atlas_camera_staged_master_workflow.json` (which wires
   `SAM3Segment` manually, not through `AtlasInput`'s auto-graph) is **left untouched**. It's a
   pinned, tested artifact with real calibration history on a CUDA-assumed setup that was never
   blocked by triton in the first place. The quickstart path (`AtlasInput`) is where non-CUDA
   users actually hit the wall, and that's what this pass fixes.
10. **Error-vs-report boundary in `AtlasSAM3Mask.segment()`:** import/version errors
    (`_require_sam3`, i.e. `[sam3]` not installed or `transformers` too old) **raise normally** —
    same as every other `[extra]`-gated node in this repo; it's a setup problem, not a runtime
    condition. The **gated-repo case is caught specifically** and returned as the `report` string
    instead of raising — a gated repo is a one-time auth step, not a broken install, and matching
    LiveActionAOV's own UX (a clear message, not a stack trace) keeps this consistent with that
    precedent.

## Design

### 1. Inference module — `atlas_camera/inference/sam3_segmenter.py`

New leaf module, same shape as `semantic_segmenter.py`:

- Lazy imports **only** `transformers.Sam3Model` / `Sam3Processor` (the single-image,
  concept-conditioned detector). Deliberately **not** `Sam3TrackerVideoModel` /
  `Sam3TrackerVideoProcessor` — Atlas masks stills, never clips, so the video tracker LiveActionAOV
  needs is dead weight here.
- `_require_sam3()` checks `transformers.__version__ >= "5.5.4"` explicitly (import success alone
  isn't enough — an older `transformers` imports fine but lacks the SAM3 model classes). Raises an
  actionable `RuntimeError`: `"Native SAM3 requires transformers>=5.5.4. Install with:\n    pip install -e .[sam3]"`.
- `_wrap_if_gated_repo(repo, exc)` — ported from LiveActionAOV, detects the gated-repo
  `OSError`/401 shape and returns a `RuntimeError` with the request-access + `hf auth login` /
  `HF_TOKEN` guidance, else `None` (caller re-raises the original).
- Model cache: `_SAM3_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]]`, `_SAM3_MODEL_CACHE_MAX
  = 2`, using the existing `bounded_cache_set(..., release_cuda=True)` helper from `_common.py`.
- `_get_sam3(model_id, device)`: loads `Sam3Processor.from_pretrained(repo)` +
  `Sam3Model.from_pretrained(repo)`, `.to(device).eval()`, wrapped so gated-repo errors are
  translated; caches the pair.
- Device handling: `resolve_device(device, torch)` for the initial pick. `_infer_with_mps_retry`
  (or equivalent): run inference; on `RuntimeError` when `device == "mps"`, log/note it, reload the
  model on `"cpu"` (evicting the mps cache entry), retry once, and include the fallback in the
  returned report string. No retry loop beyond one attempt — a second failure propagates.
- `sam3_concept_mask(image, concepts: str, model_id: str = DEFAULT_SAM3_MODEL, device: str | None
  = None, confidence_threshold: float = 0.5) -> tuple[np.ndarray, list[str], float]`:
  - Comma-split `concepts` the same way `match_class_ids` does (strip, drop empties).
  - One SAM3 forward pass per concept token via the processor/model; collect every detected
    instance mask scoring above `confidence_threshold`.
  - Union all per-concept instance masks into a single `(H, W)` bool array at the image's own
    resolution.
  - Return `(mask, matched_concepts, coverage_fraction)` — same shape contract as
    `semantic_class_mask`, so the node's report-formatting code can be near-identical to
    `AtlasSemanticMask.segment`'s.
- `DEFAULT_SAM3_MODEL = "facebook/sam3"` — single module-level constant, no combo tuple.

### 2. Node — `AtlasSAM3Mask` (`atlas_camera/comfy/nodes_inpaint.py`, beside `AtlasSemanticMask`)

```python
class AtlasSAM3Mask:
    """🪄 Native SAM3 concept mask via transformers — no triton/comfyui-rmbg dependency.

    ... (docstring explaining the triton problem this solves, mirroring
    AtlasSemanticMask's docstring shape and cross-referencing it as the
    fallback tier)
    """
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "segment"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "concepts": ("STRING", {"default": "sky",
                    "tooltip": "Comma-separated open-vocabulary concepts (e.g. 'sky', "
                               "'person, vehicle'). The mask is the UNION of all detected "
                               "instances across every concept."}),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    def segment(self, image, concepts="sky", confidence_threshold=0.5, device="auto", **_extra):
        # mirrors AtlasSemanticMask.segment: PIL conversion, dev resolution,
        # call sam3_concept_mask, build the report string, return (mask, report)
        ...
```

- `RETURN_TYPES`/`RETURN_NAMES`/`FUNCTION`/`CATEGORY` match `AtlasSemanticMask` exactly so the two
  are interchangeable at call sites.
- Report on success: `"matched ['sky'] -> 12.3% of frame (facebook/sam3)"` style, matching
  `AtlasSemanticMask`'s format. On a wrapped gated-repo/version error, the report contains the
  actionable guidance (the node should catch its own `RuntimeError` and return it as the report
  string rather than raising, to keep behavior consistent with `AtlasSemanticMask`'s no-match
  case wherever practical — final call left to implementation, see open question below).

### 3. Cascade rewire — `atlas_camera/comfy/nodes_viewport.py` (`AtlasInput`'s `segment()`)

- New capability probe, `_native_sam3_available()`, added to `node_helpers.py` next to
  `_comfy_registry()`: cheap check that `transformers` imports and
  `transformers.__version__ >= "5.5.4"` (packaging-level version compare, e.g.
  `packaging.version.parse` or a simple tuple compare — no torch import, no model load, no
  network).
- `segment()` rewritten:
  ```python
  def segment(image_ref, prompt_value):
      if have_native_sam3:
          return g.node("AtlasSAM3Mask", image=image_ref, concepts=prompt_value).out(0)
      if have_semantic:
          return g.node("AtlasSemanticMask", image=image_ref, classes=prompt_value).out(0)
      return None
  ```
  where `have_native_sam3 = _native_sam3_available()`, evaluated once near the top of `build()`
  alongside the existing `have_semantic` line.
- The old `sam3()` helper, `have_sam`, and the `"SAM3Segment"` registry check are removed
  entirely from this file.
- Docstring/tooltip text referencing "ComfyUI-RMBG (SAM3Segment) — skipped + noted if absent" is
  updated to describe the new native/SegFormer cascade.
- `notes` messaging updated: e.g. `"native SAM3 absent -> AtlasSemanticMask (SegFormer, CPU/MPS) fallback for sky/scope"` when the probe fails, and equivalent adjustments to the "sky SKIPPED" /
  "scope SKIPPED" messages (now naming both native SAM3 and AtlasSemanticMask as the two
  candidates that were unavailable).

### 4. Registration — `atlas_camera/comfy/node_registry.py`

New append-only entry (never renumber/reorder existing ones):
```python
"AtlasSAM3Mask": "Atlas SAM3 Mask 🪄",
```
placed near `AtlasSemanticMask`'s entry for readability (ordering within the dict is not itself
part of the saved-workflow contract — only the key strings are).

### 5. Packaging — `pyproject.toml`

```toml
# Native SAM3 (AtlasSAM3Mask) — transformers-only, no triton/comfyui-rmbg dependency, so it
# works on Mac (MPS) / CPU / AMD where the third-party SAM3Segment node (which hard-requires
# triton) cannot load at all. SAM3's model classes only exist from transformers ~5.5, hence
# the separate, narrower pin from [neural]'s own >=4.40 floor. facebook/sam3 is GATED on
# Hugging Face (Meta's SAM-License-1.0) — one-time `hf auth login` (or HF_TOKEN env) required
# after requesting access at https://huggingface.co/facebook/sam3.
sam3 = [
    "numpy>=1.24",
    "torch>=2.0",
    "transformers>=5.5.4,<6",
]
```

### 6. Explicitly out of scope for this pass

- `AtlasSegmentedSDXLInpaint`'s direct `SAM3Segment` call (`Separate`/per-instance mode).
- `examples/atlas_camera_staged_master_workflow.json`'s manually-wired `SAM3Segment` nodes.
- Any UI/combo for selecting between multiple SAM3 checkpoints (only one exists today).

### 7. Testing

- `tests/test_sam3_segmenter.py` (new): pure/mocked tests —
  - `_require_sam3` raises the actionable `RuntimeError` when a mocked `transformers.__version__`
    is below `5.5.4` (no real download).
  - `_wrap_if_gated_repo` correctly identifies a gated-repo-shaped exception and returns `None`
    for unrelated exceptions (pass-through).
  - Any pure concept-splitting/union logic factored out (mirroring `match_class_ids`'s
    testability) gets direct unit coverage.
- `tests/test_atlas_input.py` (updated): the existing registry-mocked cascade tests
  (`test_sky_and_scope_fall_back_to_semantic_mask_without_sam` etc.) are reworked around the new
  `_native_sam3_available()` probe instead of the `"SAM3Segment" in registry` check, covering all
  three tiers: native available / native absent + SegFormer available / neither available.
- `tests/test_comfy_node_registry.py` picks up the new `AtlasSAM3Mask` registry entry
  automatically (it pins the whole node-catalog surface).
