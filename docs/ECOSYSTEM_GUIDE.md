# Atlas Camera — Ecosystem Guide

This is the comprehensive, whole-system companion to [docs/USER_GUIDE.md](USER_GUIDE.md)
(which stays focused and artist-facing on the three core mental models: recovery,
projection, and preview dilation). This guide covers everything USER_GUIDE.md
doesn't: the full node catalog, every geometry-derivation and multi-angle-patch
strategy, the improvements and fixes made in the 2026-07-03/04 work session, and
the new VFX color-managed output path via ComfyUI-OCIO.

If you're new to the project, read USER_GUIDE.md's Part 1–3 first for the core
mental model, then come back here for the rest of the map.

---

## 1. What Atlas Camera Is

Atlas Camera answers one question — *given a single photo, where was the camera,
what lens was it, how big is the scene, and what 3D geometry can I build to paint
that photo onto* — and carries the answer through one consistent through-line:

```
RECOVER  →  DERIVE  →  PROJECT  →  EXPORT
(camera)    (geometry)  (live)     (DCC handoff)
```

- **Recover** — solve a camera (position, orientation, focal length, metric
  scale) from one image, via a classical vanishing-point method or a learned
  neural prior.
- **Derive** — build simple 3D geometry (a depth relief mesh, or fitted
  primitives: walls, planes, room boxes) that receives a camera projection.
- **Project** — paint the source photo onto that geometry from the recovered
  camera's exact point of view, live, in a ComfyUI viewport — the same
  technique matte painters use in Nuke or Maya, but interactive and immediate.
- **Export** — hand the solved camera and/or textured geometry off to Maya,
  Nuke, Blender, USD, or a plain review package, with the projection already
  baked into the mesh's UVs.

### Package layout

```
atlas_camera.core       ← DCC-agnostic schema, solver, math (zero required deps)
atlas_camera.exporters  ← Maya, Blender, Nuke, USD, review package writers
atlas_camera.importers  ← Atlas JSON and USD camera loaders
atlas_camera.comfy      ← ComfyUI node library (26 nodes as of this session)
atlas_camera.ui         ← Optional FastAPI project service + React/Three.js workbench
atlas_camera.reference_data ← Curated scale-reference registry (person/door/car/etc.)
atlas_camera.inference  ← Depth Anything V2, GeoCalib, local VLM helpers
atlas_camera.gaussian   ← Future 3DGS / point-cloud interfaces (placeholder)
```

The core package has **zero required runtime dependencies**. Every optional
capability (numpy/opencv vision math, USD export, the FastAPI UI, the neural
solvers) is guarded by `try/except` with an actionable `pip install -e .[extra]`
message — nothing silently degrades without telling you why.

### Installing

```powershell
pip install -e ".[dev]"                                   # numpy, opencv, pytest
pip install -e ".[neural]"                                 # torch (expected from host env) + GeoCalib
pip install "git+https://github.com/cvg/GeoCalib.git"      # GeoCalib is GitHub-only
pip install -e ".[usd]"                                     # usd-core, for AtlasExportUSD

# Into ComfyUI's own venv (editable — changes to Python source are live, no reinstall):
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install -e .
```

A symlink connects the node pack into ComfyUI:
`<COMFYUI_ROOT>\custom_nodes\AtlasCamera` → `<REPO_ROOT>\atlas_camera\comfy`.

---

## 2. The Node Catalog (37 nodes, category "Atlas Camera")

Grouped by pipeline stage rather than alphabetically — this is the order you'd
actually wire them in.

### Solve

| Node | What it does |
|---|---|
| `AtlasLearnedSolveFromImage` | **Recommended default.** GeoCalib neural prior predicts focal length + gravity direction directly from image content. Robust on AI-generated images (27/33 usable on a test set, vs. 18/33 for VP solving) and reports genuine, meaningful confidence. |
| `AtlasSolveFromImage` | Classical vanishing-point solve — detects and triangulates converging line families. Best on real photographs with clean architectural lines; fragile on AI imagery (locally-plausible-but-globally-inconsistent perspective breaks the RANSAC fit) and reports a constant, uninformative 0.75 confidence regardless of fit quality. |
| `AtlasConstrainedSolve` | Artist-guided solve from explicit line/scale constraints JSON. |
| `AtlasLoadImageSolveCamera` | Legacy file-path-based solve. |
| `AtlasLoadSolveJSON` | Reload a previously exported solve. |

### Scale (tiered — see USER_GUIDE.md Part 1 for the full mental model)

| Node | Tier | What it does |
|---|---|---|
| `AtlasReferenceScaleSolve` | 1 (highest) | Measures camera height from one known-size object's pixel bounding box (person, door, car, shipping container, building story, …). A real single-view geometric measurement, not an inference. |
| `AtlasVLMScaleCues` | 1, auto-suggest | A local vision-LLM (Ollama / LM Studio / llama.cpp) proposes candidate reference objects and their bboxes automatically. **Never auto-applied** — see below. |
| `AtlasApplyScaleReferences` | 1, gate | Applies `AtlasVLMScaleCues`' suggestions to a solve — but **only when `confirm=true`**. With `confirm=false` (default) it just records candidates for review. This mirrors the project's whole-codebase rule: propose, never silently apply. |
| *(built into)* `AtlasLearnedSolveFromImage`, `height_mode=measure_from_depth` | 2 | Depth Anything V2 fits a ground plane below the horizon and reads camera height off it. Medium reliability — AI-image depth is often not perfectly ground-plane-consistent. |
| *(built into)* `AtlasLearnedSolveFromImage`, `height_mode=assume` | 3 (fallback) | Plain assumed eye-height (`camera_height_m`, default 1.6m), always flagged as an assumption. |

### Derive geometry

| Node | What it does |
|---|---|
| `AtlasDeriveProjectionGeometry` | Builds the receiving surfaces for projection: a depth **relief mesh** (default, handles arbitrary/organic shapes), fitted **primitives** (`azimuth_walls` / `ransac_planes` / `room_cuboid` / `vertical_extrusion` — see §3.3), or `both`. The `scene_type` widget (`manual`/`organic`/`indoor`/`outdoor`) is a one-choice convenience preset over `geometry_mode`+`primitive_method`+`depth_model` — it never adds new solving behavior. |
| `AtlasAddPatchView` | Adds an AI-generated novel-view "patch" to fill areas the primary camera couldn't see. See §3.4 — this is the most involved node in the pack. |
| `AtlasDepthAnything` | Standalone monocular depth (Depth Anything V2), metric or relative — mostly useful for inspection/diagnostics. |

### Composable derive & merge (mix strategies per scene region)

An alternative to `AtlasDeriveProjectionGeometry`'s single `scene_type` preset:
estimate depth once and derive each region with the strategy that fits it, then
combine explicitly.

| Node | What it does |
|---|---|
| `AtlasDepthMap` | Shared metric depth estimate (`ATLAS_DEPTH_MAP`) — run **once**, feed the derive nodes below so they agree on metric scale. Distinct from `AtlasDepthAnything` (a lossy preview IMAGE). |
| `AtlasDeriveReliefMesh` · `AtlasDeriveWalls` · `AtlasDeriveTowersSpires` · `AtlasDeriveRoofsFacades` · `AtlasDeriveInteriorRoom` | Per-strategy derive nodes (relief mesh · azimuth walls · vertical-extrusion towers/spires · RANSAC roofs/facades · Manhattan room cuboid). Each consumes a shared `AtlasDepthMap`. |
| `AtlasMergeGeometry` | **Nuke-Merge-node equivalent** — combines two derived solves' proxy geometry (e.g. foreground walls + background relief mesh). `solve_a`'s camera wins; chain instances for 3+-way. Optional `shot_cam` rides along onto the merged solve. |
| `AtlasOcclusionMask` | Frustum / frame / facing-angle validity mask used by multi-angle patch projection (§3.4). |

### Shot format

| Node | What it does |
|---|---|
| `AtlasDefineShotCam` | A project-level render/output camera format (sensor W×H mm + lens mm + long-edge resolution) — like a Nuke/Resolve project setting. Intrinsics-only, no position. Wire into `AtlasMergeGeometry` (attaches to the merged solve) or directly into `AtlasBlockoutViewport` so the render/export conforms to one shot format instead of each photo's own aspect. |

### Decompose / analyze

| Node | What it does |
|---|---|
| `AtlasDecomposeSolve` | Unpacks a solve into `camera`, `confidence`, `source_method`, image dims, raw JSON, `horizon_angle_deg`. |
| `AtlasDecomposeCamera` | Unpacks a camera into `fx/fy/cx/cy`, world position, focal_mm, FOV. |
| `AtlasGroundDepthMap` | Metric ground-plane depth visualization + ground mask. |
| `AtlasGroundMask` | Binary ground/sky mask. |
| `AtlasHorizonMask` | Binary above/below-horizon mask, with feathering. |
| `AtlasVPVisualization` | Overlays detected vanishing points + horizon on the source image. **Empty on the learned/GeoCalib solve path** — it doesn't compute VPs at all, by design. |

### Viewport (interactive, browser-side)

| Node | What it does |
|---|---|
| `AtlasBlockoutViewport` | The live Three.js viewport: 📷 Camera View, 📽 Project (matte-painting mode), ☀ Exposure, 📊 VP/horizon/ground diagram, ℹ camera HUD, 🎥 camera-path authoring, 4 render passes (shaded/depth/normal/mask). Optional `shot_cam` input conforms the render to a shot format. See USER_GUIDE.md Parts 2–3 for the concepts. |
| `AtlasViewportControls` | Optional companion node — moves the viewport's toolbar/panels onto its own node to declutter the viewport (carries no data; a pure UI link). |

### Export

| Node | What it does |
|---|---|
| `AtlasExportSolveJSON` | Raw solve JSON. |
| `AtlasExportReliefMesh` | OBJ+MTL+texture and/or self-contained GLB, projection baked into UVs — imports pre-projected into Maya/Nuke/ZBrush/Blender with zero setup. |
| `AtlasExportBlender` / `AtlasExportNuke` / `AtlasExportUSD` | DCC scene-builder scripts / USD camera. |
| `AtlasExportCameraPathUSD` | Exports the viewport's authored camera-path keyframes as an animated USD camera. |
| `AtlasExportMayaReviewScene` | Maya scene + image card; wire in `AtlasExportReliefMesh`'s `obj_path` to include the real relief mesh instead of placeholder proxies. |
| `AtlasExportReviewPackage` | Full bundle (report + all DCC scripts) for handing off to another artist. |
| `AtlasUSDCameraLoader` | Load a camera back out of a `.usda`. |

---

## 3. Core Concepts Beyond USER_GUIDE.md

### 3.1 Geometry derivation strategies

`AtlasDeriveProjectionGeometry`'s `primitive_method` picks *how* fitted
primitives are built (only relevant when `geometry_mode` is `primitives` or
`both`):

- **`azimuth_walls`** (default) — vertical walls + foreground boxes/cylinders,
  general-purpose. Height is clipped to whatever passes a near-vertical-normal
  filter, so it truncates a sloped roof, spire, or tower (confirmed on real
  church/tower photos).
- **`ransac_planes`** — planes at *any* orientation via a 2D orientation
  histogram + sequential RANSAC. Best for exterior architecture with roofs,
  ramps, or stepped facades.
- **`room_cuboid`** — assumes a Manhattan-orthogonal room: floor + up to 4
  walls + optional ceiling. Best for genuinely box-shaped interiors; produces
  confidently *wrong* (skewed) results on non-orthogonal rooms — pick this
  deliberately, it doesn't auto-detect room shape.
- **`vertical_extrusion`** — same wall detection as `azimuth_walls`, but
  height comes from the image-space silhouette (topmost non-sky pixel per
  column, back-projected at its own depth) instead of a normal filter — the
  Hoiem/Efros/Hebert "Automatic Photo Pop-up" (SIGGRAPH 2005) billboard-cutout
  technique. Reaches towers and sloped roofs `azimuth_walls` truncates, at the
  cost of representing them as a flat vertical plane rather than their true shape.

All four extractors agree on world points and metric scale for a given depth
map (factored into shared `core/depth_geometry.py` helpers) and all populate a
`stats["ground_scale"]` the relief-mesh branch reuses.

### 3.2 Sky-aware depth

Monocular depth models hallucinate noisy, spatially-incoherent depth on
feature-less sky — left unhandled, that noise triangulates into a huge, jagged
mesh chunk that dwarfs the actual scene. `depth_geometry.detect_sky_mask` flags
a pixel as sky when it's above the solved horizon **and** either near the
far-depth percentile or has high local **roughness** (mean-squared discrete
Laplacian — deliberately *not* raw variance, since a genuinely sloped real
surface like a roof or ramp has a near-zero Laplacian despite high raw
variance; using variance would misclassify real architecture as sky).
`build_relief_mesh` excludes sky as a hole rather than distance-clamping it.

### 3.3 Multi-angle patch projection (`AtlasAddPatchView`)

Single-photo projection only textures what the recovered camera actually saw —
occluded and grazing-angle surfaces go black the moment you orbit. The fix is
to add an AI-generated *novel view* of the same scene as a **patch**.

The hard part is registration: a novel-view generation (e.g. via the
Qwen-Image-Edit-2511 "Multiple Angles" LoRA) has no ground-truth transform back
to the original scene. So the patch camera is **constructed, not solved**:
`orbit_camera(primary_extrinsics, pivot, d_azimuth, d_elevation, distance_scale)`
orbits the recovered camera around the scene's ground look-at pivot and
rebuilds the view matrix via an unambiguous `look_at`, guaranteeing it shares
the primary camera's world frame by construction.

**Critical detail:** the LoRA's named angles (`front view`, `right side view`,
etc.) are *absolute*, subject-relative — not relative to your source photo's
own viewing angle. `AtlasAddPatchView` therefore takes both `source_*_view`
(what your source photo actually is) and `patch_*_view` (what you asked the
LoRA for), and applies the **difference** as the orbit. A `flip_azimuth`
toggle corrects mirrored handedness, calibrated by eye. The node then derives
the patch view's own relief geometry in that constructed frame and appends a
`ProjectionSource` (camera + image + geometry + priority) to the solve. The
viewport layers each patch's own projection material over the primary with a
**facing-ratio mask** — patches only paint surfaces they're looking
near-head-on at, falling through to the primary (or empty) elsewhere.

This session's `horizon_row_from_extrinsics` fix (§4.1) specifically improves
this node: it now computes the *actual* horizon row for each constructed patch
camera instead of guessing.

### 3.4 The metric-scale honesty problem

Worth stating plainly, since it shapes several of this session's decisions:
**no tier of the scale system can fully compensate for the fact that
AI-generated images have no metric ground truth.** A door drawn by a diffusion
model isn't constrained to be exactly 2.10m the way a real door is. Reference-object
scale (tier 1) is the most reliable *method*, but its accuracy on synthetic
imagery is bounded by whether the generator actually rendered the reference at
plausible real-world proportions — verified directly this session (§4.3).

---

## 4. This session's improvements (2026-07-03 / 2026-07-04)

### 4.1 Fixed: patch-view horizon calculation

`AtlasAddPatchView` derived each patch's ground scale and relief mesh with
**no horizon estimate at all**, falling back to a generic `height * 0.45`
guess — meaningless for an orbited, non-level patch camera. Added
`horizon_row_from_extrinsics(extrinsics, *, fy, cy)` to `core/camera_math.py`:

```python
def horizon_row_from_extrinsics(extrinsics, *, fy, cy):
    """Image row where the world-horizontal plane's vanishing line falls."""
    rotation = extrinsics.camera_rotation_matrix
    y_up = float(rotation[1][1])
    if abs(y_up) < 1e-6:
        return None  # camera looking straight down/up — degenerate
    y_back = float(rotation[1][2])
    return float(cy) - float(fy) * y_back / y_up
```

Verified via synthetic test: a level camera returns exactly `cy`; downward
tilt moves the row up-frame; straight-down returns `None` (correctly
degenerate). `AtlasAddPatchView.add_patch()` now computes this per-patch and
threads it into both `estimate_ground_scale()` and `build_relief_mesh()`.

### 4.2 Fixed: ground-height noise robustness

`estimate_ground_height_from_depth()` now applies a 3×3 edge-clamped median
filter to the depth map before back-projection, and rejects candidate ground
points across depth discontinuities (the same edge-detection technique
already used in `depth_geometry.back_project_normals`). Validated on a 20-trial
synthetic noise test: mean absolute error dropped 0.0060m → 0.0024m, max error
0.0116m → 0.0035m, std 0.0032 → 0.0006.

**Honest limitation, documented in the code:** this measurably improves
*noise* robustness, but was proven — via direct instrumentation on a real
scene — **not sufficient to fix absolute depth-model scale bias**. If the
depth model itself is systematically wrong about how far away things are (a
real, observed failure mode on AI-generated imagery, not fixable by denoising
its own estimate), tier-2 scale will still be wrong. Tier 1 (reference-object)
remains the only real remedy for that class of error.

### 4.3 Investigated and deliberately *not* fixed: a "wrong surface selection" hypothesis

A speculative theory (that the ground-plane classifier was picking the wrong
surface in a scene with an implausible recovered height) was directly
instrumented and visualized — and proven **wrong**: the classifier correctly
identifies the true ground surface. The real, verified root cause is depth-model
absolute-scale bias, which is not code-fixable at the depth-estimation layer
(see §3.4 and §4.2).

A "plausibility penalty" fix (discount confidence for "unusual" camera
heights) was considered and **explicitly rejected** — it would systematically
penalize legitimate elevated/drone camera shots, which the patch-view
elevation vocabulary (§3.3) explicitly supports as valid. Trading one bug for
a new, less-visible bias was judged worse than reporting the limitation
honestly. This reasoning is preserved as a docstring on
`estimate_ground_height_from_depth`.

A manually-eyeballed reference-object bbox test on an AI-generated coastal
scene (a `door_210cm` reference) produced an implausible 0.78m height — either
bbox imprecision, or (more likely, per §3.4) the generated door simply not
matching real 2.10m proportions. Reported as-is, not oversold.

### 4.4 Fixed: LM Studio VLM integration (previously 100% non-functional)

`AtlasVLMScaleCues`'s `provider="lmstudio"` path always resolved every model
to `"unknown"` and silently failed. Root cause: LM Studio's own
`/api/v1/models` endpoint uses `"key"` as the model identifier field and
`"display_name"` for the human label — not `"id"`/`"model"`/`"name"`, which is
all `_model_info_from_lmstudio()` in `inference/multimodal_helper.py` checked.
Fixed by checking `"key"` first. Confirmed via direct `curl` inspection of a
live LM Studio instance serving `google/gemma-4-12b-qat`.

**Known limitation, not a code bug:** quantized local vision models — confirmed
reproducible across 3 attempts with Gemma 4 12B QAT — can hallucinate field
names outside the requested JSON schema, or degenerate into a repeated-key
decoding loop mid-response. The code already detects and truncates repetition
loops (`_truncate_looping_response`) and closes partial JSON
(`_close_partial_json`), but a hallucinated schema still can't be salvaged.
This is a model/quantization capability limit, not something worth writing
fragile, model-specific parsing workarounds for.

### 4.5 Fixed: cosmetic warning noise

`np.errstate(all="ignore")` alone does **not** suppress numpy's
`RuntimeWarning: All-NaN slice encountered` on `np.nanmedian` — that warning
goes through Python's `warnings` module, not FPU flags. Wrapped the
`build_relief_mesh` 3×3-median call in `relief_mesh.py` in a matching
`warnings.catch_warnings()` block.

### 4.6 Full 26-node learned pipeline validated end-to-end

Ran `atlas_camera_learned_workflow.json`'s complete graph (solve → VLM cues →
apply scale → derive geometry → decompose → all analysis nodes → viewport →
all 5 DCC export formats) against a live ComfyUI instance. One real
environment gap found and fixed along the way: `AtlasExportUSD` needs the
optional `usd-core` package (`pip install usd-core`, ~13.5MB, quick) — not
installed by default in a fresh ComfyUI venv. After installing it, the full
pipeline completed cleanly, producing all outputs: solve JSON, Blender
`build_scene.py`, Nuke projection script, `camera.usda`, a full review package
(report + Maya/Blender/Nuke scripts), and the relief mesh (OBJ+MTL+PNG+GLB).

---

## 5. VFX color-managed output — ComfyUI-OCIO integration (new)

### Why

Atlas Camera's relief-mesh export bakes the source image as an 8-bit,
display-referred sRGB texture — the correct convention for DCC viewports and
game engines, but not what a real VFX comp/lighting pipeline wants. Those
departments work in scene-linear color spaces (typically **ACEScg**) and
expect a properly color-managed EXR plate, not a raw sRGB PNG. ComfyUI itself
has no color management at all — it holds every `IMAGE` tensor as plain
gamma-encoded sRGB in `0..1` (this is documented as ComfyUI-OCIO's own stated
assumption, and matches Atlas Camera's own texture-baking behavior).

[ComfyUI-OCIO](https://github.com/SlavaSexton/ComfyUI-OCIO) (by Slava Sexton)
adds eight Nuke-style OpenColorIO nodes on top of ComfyUI's plain sRGB
working space, backed by OpenColorIO's built-in ACES studio config (~55
colorspaces, including ARRI/RED/Sony camera spaces).

### Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SlavaSexton/ComfyUI-OCIO
pip install opencolorio    # opencv-python-headless / tifffile / Pillow / numpy
                            # are typically already present from other packs
```

Set `OPENCV_IO_ENABLE_OPENEXR=1` in the environment **before** ComfyUI starts
(OpenCV reads this at library-init time, not per-call — setting it after
`import cv2` has already happened does nothing). Confirmed working this
session: OpenColorIO 2.5.2, node pack import time 0.5s, no errors.

Video export (ProRes/DNxHR/h264/hevc) additionally needs a *full* ffmpeg build
on `PATH` — check with `ffmpeg -version`. Stills and sequences (EXR/TIFF/PNG/JPEG)
need nothing beyond the pip install above.

### The eight nodes

| Node | Nuke equivalent | What it does |
|---|---|---|
| `OCIORead` | Read | Load a still/sequence/video off disk, color-managed on the way in. |
| `OCIOWrite` | Write | Color-manage an IMAGE batch and write it to disk — EXR/TIFF/PNG/JPEG stills or sequences, or ProRes/DNxHR/h264/hevc video. |
| `OCIOColorSpace` | OCIOColorSpace | Convert between two named colorspaces. |
| `OCIOLogConvert` | OCIOLogConvert | Linear ↔ log (cineon/acescct/acescc), dependency-free — no OCIO needed. |
| `OCIODisplay` | OCIODisplay | Scene-referred → display-referred view transform. |
| `OCIOCDLTransform` | OCIOCDLTransform | ASC CDL primary grade (slope/offset/power/saturation). |
| `OCIOFileTransform` | OCIOFileTransform | Apply a LUT/CCC/CDL file. |
| `OCIOLookTransform` | OCIOLookTransform | Apply a named OCIO look (e.g. ACES Reference Gamut Compression). |

### The Atlas Camera integration pattern

Because ComfyUI's own working space (`"sRGB - Display"`) is exactly
`OCIOWrite`'s own default `from_colorspace`, no separate `OCIOColorSpace`
conversion step is needed — feed the same source `IMAGE` into both branches:

```
                          ┌─ AtlasExportReliefMesh ─→ sRGB albedo texture (3D dept)
LoadImage ─→ solve ─→ derive ─┤
                          └─ OCIOWrite (sRGB - Display → ACEScg, EXR, 16f) ─→ plate (comp/lighting dept)
```

This produces two complementary, correctly-targeted deliverables from one
graph, instead of overloading a single sRGB texture for both purposes.
Validated end-to-end this session: submitted directly to a live ComfyUI
instance, confirmed successful, and the resulting file's OpenEXR magic number
(`76 2f 31 01`) checked byte-for-byte on disk.

**Path gotcha, worth knowing before you wire this up:** Atlas Camera's export
nodes (`AtlasExportReliefMesh`, etc.) resolve their `output_dir` relative to
ComfyUI's **root** working directory. `OCIOWrite` resolves `output_folder`
relative to ComfyUI's **`output/`** directory instead. The identical relative
path string on both nodes therefore lands in two different places on disk —
use absolute paths on one or both if you need everything co-located.

For a full Nuke/Maya round-trip beyond just writing files, wire an
`OCIOColorSpace` (`sRGB - Display` → `ACEScg`) node ahead of anything that
needs a *linear* `IMAGE` tensor for further ComfyUI-side compositing, rather
than only writing to disk.

See `examples/atlas_camera_vfx_ocio_output_workflow.json` for the full,
validated, ready-to-open graph (with an in-canvas Note explaining all of the
above), or `examples/api_format/atlas_camera_vfx_ocio_output.api.json` for
the scripted-testing equivalent.

---

## 6. Example workflows reference

`examples/*.json` — **UI/litegraph format**, drag-and-drop or load directly in
ComfyUI's browser canvas for interactive, click-around testing:

| File | Demonstrates |
|---|---|
| `atlas_camera_core_projection_workflow.json` | The minimal 6-node core through-line (solve → derive → project → export mesh). Best starting point. |
| `atlas_camera_learned_workflow.json` | The full 26-node learned pipeline — every analysis node, viewport, and all 5 DCC export formats. |
| `atlas_camera_full_workflow.json` | Original 20-node vanishing-point workflow (no neural deps). |
| `atlas_camera_multiangle_patch_proof_workflow.json` | Multi-angle patch calibration proof (§3.3). |
| `atlas_camera_vertical_extrusion_comparison_workflow.json` | Primitive-method comparison. |
| `atlas_camera_reference_scale_workflow.json` **(new)** | Tier-1 reference-object scale demo, with an in-canvas caution note about AI-generated-image limitations (§3.4). |
| `atlas_camera_vlm_scale_cues_workflow.json` **(new)** | `AtlasVLMScaleCues` preset for the now-fixed LM Studio path, with the quantized-model reliability caveat (§4.4) documented in-canvas. |
| `atlas_camera_vfx_ocio_output_workflow.json` **(new)** | The VFX/OCIO integration from §5. |

`examples/api_format/*.json` **(new)** — ComfyUI's raw **API format**
(node-id → `{class_type, inputs}`), no layout data, **cannot be opened in the
browser canvas** — POST directly to `/prompt`, or use the `comfyui` skill's
`comfy_client.py`. All three were run end-to-end against a live ComfyUI
instance and confirmed successful:

| File | Demonstrates |
|---|---|
| `atlas_camera_core_projection.api.json` | Scripted equivalent of the core 6-node workflow. |
| `atlas_camera_learned_full_pipeline.api.json` | Scripted equivalent of the full 26-node pipeline, incl. USD export. |
| `atlas_camera_multiangle_patch_selfcontained.api.json` | Fully self-contained multi-angle patch demo — generates its own source photo and novel-view patch (Qwen-Image-2512 + Qwen-Image-Edit-2511 + Multiple-Angles LoRA), no external image needed. |
| `atlas_camera_vfx_ocio_output.api.json` | Scripted equivalent of the VFX/OCIO workflow. |

---

## 7. Known limitations (consolidated, honest)

- **Classical VP solving is fragile on AI-generated images** (18/33 usable on
  a test set) — prefer `AtlasLearnedSolveFromImage`.
- **No tier of the scale system has true metric ground truth on AI-generated
  imagery** — a generator has no constraint forcing "this door is exactly
  2.10m." Tier 1 (reference object) is the most *reliable method*, not a
  guarantee, and is most trustworthy on real photographs of real objects.
- **Depth-model absolute-scale bias is not fixable by denoising the depth map
  itself** — tier-2 (`measure_from_depth`) scale can still be systematically
  wrong even with the 2026-07-04 noise-robustness improvements (§4.2). This is
  an inherent depth-model limitation, verified by direct instrumentation, not
  a bug in Atlas Camera's own code.
- **`azimuth_walls` truncates sloped roofs/spires/towers** — use
  `ransac_planes` or `vertical_extrusion` for that geometry instead.
- **`room_cuboid` produces confidently wrong results on non-orthogonal
  rooms** — it doesn't detect room shape, it assumes it.
- **Quantized local VLMs are unreliable at structured extraction** — confirmed
  with Gemma 4 12B QAT; expect occasional empty `scale_references` even with
  an obvious reference object in frame.
- **`AtlasVPVisualization`'s VP layer is empty on the learned/GeoCalib solve
  path** by design — it never computes vanishing points on that path.
- **`AtlasExportUSD` needs `usd-core`** (`pip install usd-core`) — not part of
  ComfyUI's default venv.
- **ComfyUI-OCIO video export needs a full ffmpeg build** on `PATH` — many
  bundled/utility ffmpeg installs (screen-capture tools, etc.) lack the
  ProRes/DNxHR/h264/hevc codecs. Stills/sequences are unaffected.
- **`AtlasExportReliefMesh` vs `OCIOWrite` resolve relative output paths
  against different base directories** (§5) — use absolute paths to co-locate.

---

## 8. Quick reference: node → concept

| Node | Concept |
|---|---|
| `AtlasLearnedSolveFromImage` | Recovery — learned camera prior (recommended default) |
| `AtlasSolveFromImage` | Recovery — classical vanishing-point solve |
| `AtlasReferenceScaleSolve`, `AtlasApplyScaleReferences` | Scale tier 1 — reference object |
| `AtlasVLMScaleCues` | Scale tier 1 — automatic suggestions (needs `confirm`) |
| `AtlasDeriveProjectionGeometry` | Derive — relief mesh / primitives, 4 fitting strategies (§3.1) |
| `AtlasAddPatchView` | Derive — multi-angle patch projection (§3.3) |
| `AtlasBlockoutViewport` (📽 Project) | The live camera projection |
| `AtlasExportReliefMesh` | UV-baked mesh export for Maya/Nuke/ZBrush |
| `AtlasBlockoutViewport` (`preview_expand`) | Preview-only geometry dilation |
| `AtlasBlockoutViewport` (☀ / 📊 / ℹ) | Diagnostics — exposure, VP/horizon diagram, camera HUD |
| `OCIOWrite`, `OCIOColorSpace` | VFX output — ACEScg EXR plates alongside DCC exports (§5) |
