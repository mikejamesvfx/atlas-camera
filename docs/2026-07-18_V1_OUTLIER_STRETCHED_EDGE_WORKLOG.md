# Atlas Camera — v1 Outlier and Stretched-Edge Worklog

**Date:** 2026-07-18  
**Scope:** RAW/Camera RAW workflows, SDXL inpainting, LaRI hidden geometry, SAM3 segmentation, relief-mesh quality, and v1 projection QA.

## Executive summary

Today’s work moved Atlas from “make a relief mesh and hope the orbit holds” toward an explicit layered-quality pipeline:

1. Detect suspicious depth instead of smoothing it into neighboring geometry.
2. Reject complete grid quads when a diagonal would become a stretched UV wedge.
3. Inpaint occluded buildings independently with SAM3 instance masks and native SDXL inpaint conditioning.
4. Carry mesh quality metrics into proxy metadata and the master debug report.
5. Keep the failures local and measurable so an artist can switch a layer to a card, ground plane, or segmented inpaint rather than raising a global threshold.

The live segmented SDXL workflow completed successfully with no node errors. The final layer-node runtime check is still pending because the running ComfyUI process exposes an older `AtlasCleanPlateLayer` schema than the repository source.

## Implemented code

### Native SDXL inpainting

`AtlasSDXLInpaint` was implemented using the native ComfyUI SDXL path:

`CheckpointLoaderSimple → CLIPTextEncode (positive/negative) → InpaintModelConditioning → KSampler → VAEDecode`

Using `InpaintModelConditioning` was important. A plain VAE-encoded masked image produced flat gray masked regions; the SDXL inpaint conditioning path supplies the concatenated mask/latent conditioning required for coherent reconstruction.

### Segmented building inpainting

Added `AtlasInstanceMask` and `AtlasSegmentedSDXLInpaint`.

The segmented node:

1. Runs SAM3 with `output_mode="Separate"` and a building prompt.
2. Selects individual instance masks.
3. Intersects each instance with the LaRI paint matte.
4. Expands/crops the inpaint region.
5. Runs SDXL inpaint per crop.
6. Stitches each result back into the full plate sequentially.

This avoids the failure mode where one giant SDXL crop invents a single connected mega-structure across multiple buildings.

### Depth outlier mask

Added `AtlasDepthOutlierMask` in `atlas_camera/comfy/nodes.py`.

It computes a local 3×3 median and robust MAD deviation, then optionally dilates the result. The output is a `MASK` plus a report such as:

```text
depth outlier mask: 5 px (5.00%)
```

The mask is wired into both visible and hidden relief branches through `AtlasDeriveReliefMesh` and is OR-ed with any existing exclusion mask.

The purpose is to turn isolated monocular-depth hallucinations into explicit holes rather than allowing one bad pixel to become a stretched frame-spanning shard.

### Quad coherence guard

`build_relief_mesh` now accepts `quad_coherence`.

With this enabled, if either triangle of a grid quad fails a depth, edge-length, or normal-bend test, both triangles are rejected. This is intentionally conservative: a small hole is safer than retaining one surviving diagonal that interpolates a long, stretched UV wedge.

The ComfyUI relief and inpaint-layer paths default to `True`; the low-level core default remains `False` for backward compatibility with callers that intentionally retain partial quads.

The guard is now exposed on:

- `AtlasDeriveReliefMesh`
- `AtlasCleanPlateLayer`
- `AtlasDepthLayerMask`

The latter two share the same setting so hole-mask QA and final layer projection use identical topology decisions.

### Median-depth edge budget

The world-space edge budget in `atlas_camera/core/relief_mesh.py` now uses the triangle’s median depth rather than `dmax`.

Using the farthest corner allowed one hallucinated depth outlier to inflate the allowable edge length. The median is less sensitive to that single bad corner while the existing depth-ratio test still rejects genuine foreground/background discontinuities.

### Stretch diagnostics

Relief meshes now calculate QA-only stretch statistics without changing topology:

- `stretch_ratio_p95`
- `stretch_fraction_gt12`
- `torn_fraction`
- `quad_coherence`

The statistics are carried into `relief_mesh_primitive` metadata and surfaced by `AtlasDebugReport`.

The debug report now flags:

- layers with more than 65% torn quads;
- layers whose p95 world/UV edge ratio exceeds 12;
- zero-vertex layers;
- near-empty mattes;
- band gaps/overlaps and other existing red flags.

This creates a practical fallback trigger: use a card, ground plane, additional clean plate, or segmented inpaint instead of globally relaxing relief thresholds.

## Generated workflows and tools

The main generated workflow is:

[2026-07-18_atlas_raw_quickstart_workflow_hidden_segmented_sdxl.json](../examples/2026-07-18_atlas_raw_quickstart_workflow_hidden_segmented_sdxl.json)

It includes:

- RAW loading and camera metadata;
- depth solving;
- LaRI hidden-geometry prediction;
- local depth outlier masking;
- visible and hidden relief meshes;
- SAM3-separated building instances;
- SDXL per-instance inpainting;
- preview/report nodes;
- quad-coherent relief settings.

Supporting generators and experiments include:

- `tools/build_raw_hidden_segmented_sdxl.py`
- `tools/build_raw_hidden_inpaint_sdxl.py`
- `tools/build_raw_hidden_inpaint_sdxl_native_api.py`
- `tools/build_raw_hidden_inpaint_tagged.py`
- `tools/build_raw_hidden_diagnostic.py`

## Live ComfyUI evidence

### Successful segmented SDXL run

The workflow was submitted through the ComfyUI API and completed successfully:

- `node_errors: {}`
- SAM3 separate building stack: 4 instance slots
- SDXL denoise: 0.65
- Debug flags: 0
- Relief mesh vertex counts observed: approximately 560k and 252k in the first run; approximately 560k and 272k after quad-coherence changes

The generated clean-plate previews were:

- `ComfyUI_temp_lmasz_00001_.png`
- `ComfyUI_temp_rjter_00001_.png`

The segmented result was visibly more coherent than the single giant crop: building textures remained individually plausible instead of collapsing into dense, surreal architecture.

### Current runtime mismatch

The running server exposes:

- `AtlasDepthOutlierMask` — loaded;
- `AtlasDeriveReliefMesh.quad_coherence` — loaded;
- `AtlasCleanPlateLayer.quad_coherence` — not loaded in the observed server schema.

The repository source does contain the updated layer implementation. A full ComfyUI process restart/custom-node reload is required before the final inpaint-layer runtime check can be considered verified.

## Test evidence

Passing targeted tests include:

- 39 relief/outlier tests;
- 11 derive-node tests;
- 21 proxy/geometry-node tests in the final pass;
- 2 dedicated depth-outlier tests.

The dedicated outlier tests cover:

- isolating and dilating a synthetic depth spike;
- not flagging a smooth depth gradient.

The quad-coherence regression verifies that conservative mode never adds faces relative to legacy partial-quad triangulation.

The full repository suite still has unrelated pre-existing issues on this Windows setup:

- permission errors creating/reading `.pytest_tmp` directories;
- a workflow-directory count assertion that predates today’s generated example workflows.

These failures are environmental/repository-baseline issues rather than failures in the new relief logic.

## Engineering interpretation

The main lesson from the screenshot and live tests is that “outlier” and “stretched edge” are different failure classes:

### Outlier

A bad depth sample or a bad hidden-geometry hypothesis produces an implausible world-space point. The correct response is detection and exclusion, followed by a replacement surface where appropriate.

### Stretched edge

A triangle can pass binary validity tests and still have highly anisotropic world-space edges. The correct response is whole-quad rejection, matte-aware coverage, and a quality-based fallback—not a single global increase in `max_edge_factor`.

### Black tear

A black tear can mean the mesh correctly rejected bad geometry but no replacement layer exists. The fix is not to preserve the bad triangle; it is to provide a segmented SDXL plate, card/ground fallback, sky dome, or controlled receding skirt.

## Recommended v1 presentation

For final v1 workflows, prefer this order:

1. Original plate at the recovered camera.
2. Explicit sky dome for sky pixels; do not ask relief to explain clouds or reflective sky.
3. MoGe-2 `vitl-normal` for normals, with the chosen depth model kept separate.
4. Local outlier mask before relief generation.
5. Quad-coherent relief for trusted geometry.
6. SAM3 instance mattes for buildings/foreground subjects.
7. SDXL crop-and-stitch inpaint for meaningful disocclusions.
8. Receding skirts only when a full-resolution matte cuts them and the invented pixels are tagged for downstream regrain/blur.
9. Card or ground fallback when stretch/torn metrics exceed the debug thresholds.

## Further “outside the box” ideas

These are high-value follow-ups rather than silently enabled defaults:

- **Orbit stress testing:** render automated ±3° and ±6° camera moves and score stretch/torn coverage per frame.
- **View-dependent source arbitration:** blend original, trusted relief, segmented inpaint, and sky/card layers according to angle, facing ratio, distance, and hole confidence.
- **Depth ensemble disagreement:** run two depth models and use disagreement as an uncertainty matte.
- **Semantic edge confidence:** combine SAM3 boundaries with depth-gradient boundaries; only permit skirts where both agree.
- **Layer-specific geometry policies:** automatically choose relief/card/ground per layer from stretch and torn statistics.
- **Multi-angle patch capture:** use `ExtractAngle` plates as corrective observations rather than trying to hallucinate every disocclusion from one image.
- **Temporal consistency:** for camera moves, stabilize masks and inpaint textures over time so a good single frame does not shimmer.
- **Exporter parity:** write hole, stretch, outlier, synthesized-depth, and extend mattes into Nuke and Maya exports so downstream artists see the same confidence model as the viewport.

## Files changed today

Primary source changes:

- `atlas_camera/comfy/nodes.py`
- `atlas_camera/core/relief_mesh.py`
- `atlas_camera/core/proxy_geometry.py`
- `tests/test_relief_mesh.py`
- `tests/test_derive_geometry_nodes.py`
- `tests/test_depth_outlier_mask.py`

Documentation and workflow artifacts:

- `docs/V1_EDGE_OUTLIER_CLEANUP.md`
- this worklog;
- the RAW/hidden-geometry/SDXL workflow JSONs and generator scripts listed above.

## Bottom line

The v1 direction is now technically defensible: reject bad geometry, measure the remaining risk, and replace missing coverage with a deliberate layer. The system should not promise perfect parallax from one photograph; it should deliver an honest, layered scene whose failures are local, visible, and artist-controllable.

The final acceptance gate is a fresh ComfyUI reload followed by a live run of the updated `AtlasCleanPlateLayer` path and an orbit stress test using the reported torn/stretch metrics.
