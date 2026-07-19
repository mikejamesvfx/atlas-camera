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
atlas_camera.comfy      ← ComfyUI node library (56 nodes; +2 experimental behind ATLAS_EXPERIMENTAL)
atlas_camera.ui         ← Optional FastAPI project service + React/Three.js workbench
atlas_camera.reference_data ← Curated scale-reference registry (person/door/car/etc.)
atlas_camera.inference  ← Depth Anything V2, GeoCalib, local VLM helpers
```

### Is my scale safe to export? (the trust tier, 2026-07-18)

A projection can look perfect while the metric scale is silently wrong —
projection is angular; scale is not. Every solve now carries a
`scale_health` verdict (measured / manual / assumed / unknown + an explicit
safe-to-export flag) derived from the tiered scale cascade's own provenance:
the viewport ℹ HUD shows an orange ⚠ when scale is unverified, the ✅ Solve
Gate report says why, export summaries carry the warning, and
`AtlasSceneHealthGate` 🩺 (the acknowledgement gate before the exporters)
holds the solve on any red flag until you fix it or knowingly continue —
the acknowledged report rides into every export and the `atlas_project.json`
reproducibility manifest. Fix an assumed scale with 📐 `AtlasScaleOverride`
(camera height = floors × ~3.2 m on elevated plates) or a scale reference.

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

## 2. The Node Catalog (68 standard + 4 experimental nodes)

Grouped by pipeline stage rather than alphabetically — this is the order you'd
actually wire them in.

> Count refreshed 2026-07-12. The tables below predate the newest additions —
> the shared-depth layer nodes (`AtlasDepthMap`, `AtlasCleanPlateLayer`,
> `AtlasDepthLayerMask`, `AtlasSkyDomeLayer`), `AtlasAssessImage`,
> `AtlasPredictHiddenGeometry` 🔬, the staged-master support nodes
> (`AtlasSolveGate` ✅, `AtlasScopeMask` 🎯, `AtlasSemanticMask` 🧩,
> `AtlasInpaintCrop`/`AtlasInpaintStitch` ✂, `AtlasDebugReport` 🔍,
> `AtlasLayerPreview` 🎨), and the one-node entry point `AtlasInput` 🎬 —
> which are covered in the dated addenda at the end of this guide and, in
> full, in CLAUDE.md's catalog.

### Solve

| Node | What it does |
|---|---|
| `AtlasLearnedSolveFromImage` | **Recommended default.** GeoCalib neural prior predicts focal length + gravity direction directly from image content. Robust on AI-generated images (27/33 usable on a test set, vs. 18/33 for VP solving) and reports genuine, meaningful confidence. |
| `AtlasSolveFromImage` | Classical vanishing-point solve — detects and triangulates converging line families. Best on real photographs with clean architectural lines; fragile on AI imagery (locally-plausible-but-globally-inconsistent perspective breaks the RANSAC fit) and reports a constant, uninformative 0.75 confidence regardless of fit quality. |
| `AtlasConstrainedSolve` | Artist-guided solve from explicit line/scale constraints JSON. |
| `AtlasLoadImageSolveCamera` | **DEPRECATED** legacy file-path-based solve (`DEPRECATED = True`, hidden from node search). Kept registered so saved workflows load; use `AtlasSolveFromImage` / `AtlasLearnedSolveFromImage` instead. Removal in a later release. |
| `AtlasLoadSolveJSON` | Reload a previously exported solve. |

### Scale (tiered — see USER_GUIDE.md Part 1 for the full mental model)

| Node | Tier | What it does |
|---|---|---|
| `AtlasReferenceScaleSolve` | 1 (highest) | Measures camera height from one known-size object's pixel bounding box (person, door, car, shipping container, building story, …). A real single-view geometric measurement, not an inference. |
| `AtlasVLMScaleCues` | 1, auto-suggest | A local vision-LLM (Ollama / LM Studio / llama.cpp) proposes candidate reference objects and their bboxes automatically. **Never auto-applied** — see below. |
| `AtlasApplyScaleReferences` | 1, gate | Applies `AtlasVLMScaleCues`' suggestions to a solve — but **only when `confirm=true`**. With `confirm=false` (default) it just records candidates for review. This mirrors the project's whole-codebase rule: propose, never silently apply. |
| *(built into)* `AtlasLearnedSolveFromImage`, `height_mode=measure_from_depth` | 2 | Depth Anything V2 fits a ground plane below the horizon and reads camera height off it. Medium reliability — AI-image depth is often not perfectly ground-plane-consistent. |
| *(built into)* `AtlasLearnedSolveFromImage`, `height_mode=assume` | 3 (fallback) | Plain assumed eye-height (`camera_height_m`, default 1.6m), always flagged as an assumption. |
| `AtlasScaleOverride` 📐 | manual dial | The artist's scale override, after any solve. Scale ∝ camera height, so it rescales the whole solve by a `scale` multiplier (10.0 = the "1:10" case for an elevated vista the assumed 1.6 m under-scaled) or to an absolute `camera_height_m`. Every downstream metric follows (geometry, cutoffs, DCC cameras); the projection/view is unchanged. |

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
| `AtlasRegisterPlate` | Registers a source/patch/clean plate as a durable `ATLAS_PLATE_REF`: original path, browser preview, colorspace, bit depth, role, proxy flag, optional LUT metadata. This is the float-safe bridge from ComfyUI preview tensors to final EXR/high-bit-depth plate files. |
| `AtlasAttachSourcePlate` | Attaches a registered plate ref to a solve so viewport/export nodes can keep using browser previews while Nuke/Maya/review/OBJ exporters prefer the original plate path for final projection. |
| `AtlasViewportControls` / Atlas Output Desk | Optional companion/output node. Output 0 remains the legacy detached-controls link; output 1 is `ATLAS_OUTPUT_PROFILE` with OCIO-style intent (config, working/output colorspace, display/view, display trim; the schema still carries look/LUT/exposure/gamma at neutral defaults — the widgets were removed 2026-07-10 as redundant). |

### Inpaint layers (2.5D clean-plate parallax — see §3.5)

| Node | What it does |
|---|---|
| `AtlasDepthLayerMask` | One metric depth band → `(layer_mask, occlusion_mask)`. `occlusion_mask` feeds an external inpaint graph (`INPAINT_ExpandMask` → `INPAINT_InpaintWithModel`) to build that band's clean plate. |
| `AtlasCleanPlateLayer` | Inpainted clean plate + the same depth band → appends a `ProjectionSource` (camera = primary, unchanged; mesh clipped to the band). Chain one per layer. |
| `AtlasBoundedBand` 📏 | Measures a foreground subject's own metric depth extent `W` (P5–P95) from its mask and emits ONE `band_split` cutoff at `near + 2·W`. Feed it into both a foreground layer (`band_side=foreground` → relief clipped, no runaway extrusion) and the background card (`band_side=background` → pushed back behind it). One measured boundary, both layers. |
| `AtlasDepthBandSplit` | One authoritative fg/bg depth boundary (log-depth position or metres) shared by every band node — wire into `band_split` with `band_side` so the layers can't drift apart. |

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
| `AtlasBlockoutViewport` | The live Three.js viewport: 📷 Camera View, 📽 Project (matte-painting mode), ☀ Exposure, 📊 VP/horizon/ground diagram, ℹ camera HUD, 🎥 camera-path authoring, and proxy/LDR render passes (shaded/depth/normal/mask). Optional `shot_cam` input conforms the render to a shot format. See USER_GUIDE.md Parts 2–4 for the concepts. |
| `AtlasViewportControls` | Atlas Output Desk — moves the viewport toolbar/panels onto its own node and emits `ATLAS_OUTPUT_PROFILE` metadata for display-inferred preview and DCC/OCIO handoff. |

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

### 3.5 Inpaint layers — 2.5D clean-plate parallax (`AtlasDepthLayerMask` + `AtlasCleanPlateLayer`)

The classic VFX matte-painting move: split a solved photo into depth layers,
inpaint the region each layer's foreground occluder hides into a **clean
plate**, then project each clean plate onto its own depth-banded geometry. On
a dolly/orbit move, the background layer now reveals inpainted pixels instead
of black holes — solving the orbit-coverage limitation (§3.3's black-reveal
problem) for the *same* camera, with **no angle calibration needed**, unlike
`AtlasAddPatchView`'s multi-angle patches which fill gaps via novel views at
*other* angles.

The design deliberately reuses `ProjectionSource` rather than inventing new
schema — the viewport's per-source projection material already does
everything needed, so these two nodes are pure orchestration:

- `AtlasDepthLayerMask` turns one metric depth band into `(layer_mask,
  occlusion_mask)`. `occlusion_mask` (nearer than the band's near edge) feeds
  an external `INPAINT_ExpandMask` → `INPAINT_InpaintWithModel` graph to build
  that band's clean plate.
- `AtlasCleanPlateLayer` takes the resulting plate and the *same* band,
  clips `build_relief_mesh` to `[near, far]` metres (new `band_min_m`/
  `band_max_m` params — the same "exclude the pixel, don't clamp" hole
  mechanism sky/silhouette exclusion already uses), and appends a
  `ProjectionSource` tagged `metadata["projection_mode"] = "clean_plate"`.
  The camera is the **primary, unchanged** — no `orbit_camera` call anywhere
  in this node, the whole simplification versus patch views.

Both nodes share a private `_resolve_depth_band()` helper so their bands can
never drift apart (the design requires the mask's band and the mesh clip to
match exactly). The one frontend distinction from patch views: ordinary
patches only paint surfaces they see reasonably head-on (`facingThreshold:
0.2`, discarding grazing fragments); a clean-plate layer must paint head-on
*and* grazing exactly like the primary, so `atlas_blockout.js`'s
`buildPatchSources` branches on the serialized `projection_mode` to pass
`facingThreshold: -1` instead, relying on depth + `priority` alone (not
facing angle) to order overlapping layers.

**GPL boundary, deliberately kept clean:** masking/inpainting is never
implemented in `atlas_camera` — it's delegated to external ComfyUI node packs
wired into the graph (`Acly/comfyui-inpaint-nodes`, GPL-3.0; `scraed/LanPaint`,
optional generative tier for hard disocclusions a LaMa/MAT pass smears on).
See `INSTALL.md`'s "Optional Inpaint Integration" section. Graph-level
composition is not linking, so this doesn't touch Atlas's own license.

**Caveats, stated honestly:** inpaint quality is the ceiling (LaMa continues
texture excellently but smears on complex disocclusions — route those to
LanPaint/SDXL); band boundaries are only as good as monocular depth; this is
2.5D parallax, not full 3D reconstruction, so it shines on moderate
dolly/orbit moves and shows its billboard-ish flatness on very large ones.

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

Atlas has two deliberately separate image paths:

- **browser/proxy path** — `AtlasBlockoutViewport` uses JPEG/base64 previews
  and writes proxy/LDR `IMAGE` outputs from `Render Proxy Passes` and
  `Bake Proxy Path`;
- **final plate path** — `AtlasRegisterPlate` records the original EXR or
  high-bit-depth file path, colorspace, role, bit depth, optional LUT, and a
  proxy flag inside `ATLAS_PLATE_REF`, then `AtlasAttachSourcePlate` carries
  that reference on the solve.

That split is what keeps the viewport fast without pretending a browser
canvas is a professional render writer. Comp/lighting departments usually
work in scene-linear color spaces (typically **ACEScg**) and expect the
original float plate to survive into Nuke/Maya/Resolve. Atlas exporters now
prefer file-backed plate refs when available, while falling back to preview
textures only when no durable plate path exists.

ComfyUI itself still has no color management — it holds every `IMAGE` tensor
as plain gamma-encoded sRGB in `0..1` (this is documented as ComfyUI-OCIO's
own stated assumption). Atlas treats those tensors as preview/editorial data
unless a plate ref says otherwise.

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

Use the Atlas Output Desk (`AtlasViewportControls`) to store the intended
output profile: config label/path, working colorspace, output colorspace,
display/view, and display trim. The browser
preview is display-inferred only; final OCIO/LUT fidelity belongs to
ComfyUI-OCIO, Nuke, Maya, Resolve, or another color-managed tool.

For simple ComfyUI-side EXR previews, `OCIOWrite` can consume Atlas proxy
outputs directly. Because ComfyUI's own working space (`"sRGB - Display"`) is
exactly `OCIOWrite`'s own default `from_colorspace`, no separate
`OCIOColorSpace` conversion step is needed for that preview branch:

```
                          ┌─ AtlasBlockoutViewport ─→ Render/Bake Proxy outputs (editorial)
LoadImage ─→ solve ─→ derive ─┤
                          └─ OCIOWrite (sRGB - Display → ACEScg, EXR, 16f) ─→ proxy EXR preview
```

For final projection handoff, register the real source plate and carry that
metadata through the solve:

```
LoadImage ─┬─ AtlasLearnedSolveFromImage ─→ AtlasAttachSourcePlate ─→ derive / viewport / exports
           └─ AtlasRegisterPlate (plate_path=...exr, colorspace=ACEScg) ───────┘

AtlasViewportControls.output_profile ─→ AtlasBlockoutViewport / AtlasExportNuke / AtlasExportMayaReviewScene
```

Exporter behavior with a file-backed plate ref:

- `AtlasExportNuke` creates the Read node from the original plate path and
  annotates/sets colorspace when possible.
- `AtlasExportMayaReviewScene` and the Maya exporter point file nodes at the
  original plate path and store Atlas colorspace/output-profile metadata.
- `AtlasExportReviewPackage` preserves the original source file extension and
  passes the packaged file name through to Maya/Nuke scripts.
- `AtlasExportReliefMesh` writes projection UVs and lets OBJ/MTL reference
  the original EXR/high-bit-depth plate. GLB remains preview/proxy because
  common glTF workflows expect embedded PNG/JPEG-style image payloads.

The older "write a separate OCIO EXR next to an sRGB preview texture" pattern
is still useful for editorial or quick comp checks, and was validated
end-to-end this session: submitted directly to a live ComfyUI instance,
confirmed successful, and the resulting file's OpenEXR magic number
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
| `atlas_input_quickstart_workflow.json` | The 4-node fastest path: LoadImage → 🎬 AtlasInput → Atlas Viewport. Instant relief by default; layers/VLM/sky/scope/inpaint all reachable by widgets on the one node. Start here. |
| `atlas_camera_staged_master_workflow.json` | 🏗 The 5-stage layered master — the same logic with stages, gates (VLM + solve), KJ rails, SAM sky + per-band scope, per-band LaMa clean plates, per-layer debug previews, 🔍 debug JSON, and both DCC layer exports. |
| `atlas_input_ocio_quickstart_workflow.json` | 🎨 The float VFX color-managed handoff (added 2026-07-13): `OCIORead` (ACEScg `.exr`) → `AtlasRegisterPlate` → `AtlasInput` → `AtlasAttachSourcePlate` → Nuke/Maya/USD exporters that read the original EXR at `ACEScg`. Needs ComfyUI-OCIO + opencv-python 4.x + `[usd]`. |

**The shipping catalog was deliberately trimmed to these three on 2026-07-12
(the OCIO quickstart added 2026-07-13)** (release focus). Every workflow this guide's earlier sections mention by name
(core projection, learned pipeline, VP-only, merge scenarios, hidden-geometry
heroes, master DMP variants, OCIO/plate proofs, calibration tests) still
exists in git history — recover any of them with
`git show 10e600b:examples/<name>.json`. The narrative sections below are a
chronicle and intentionally keep the historical file names.

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
| `AtlasDepthLayerMask`, `AtlasCleanPlateLayer` | Derive — inpaint layers, 2.5D clean-plate parallax (§3.5) |
| `AtlasBlockoutViewport` (📽 Project) | The live camera projection |
| `AtlasExportReliefMesh` | UV-baked mesh export for Maya/Nuke/ZBrush |
| `AtlasBlockoutViewport` (`preview_expand`) | Preview-only geometry dilation |
| `AtlasBlockoutViewport` (☀ / 📊 / ℹ) | Diagnostics — exposure, VP/horizon diagram, camera HUD |
| `AtlasRegisterPlate`, `AtlasAttachSourcePlate` | Output — file-backed float-safe plate references (§5) |
| `AtlasViewportControls` | Output — Atlas Output Desk, detached controls, OCIO-style profile (§5) |
| `OCIOWrite`, `OCIOColorSpace` | VFX output — ACEScg EXR plates alongside DCC exports (§5) |

---

## Addendum — the 2026-07-08 session: the complete DMP pipeline

This session closed the loop from "detect where projection fails" to "fix it
end-to-end and hand off to both DCCs." Everything below ships in the hero
workflow `examples/atlas_camera_master_dmp_workflow.json`. Node count is now
45.

### Hole masks and exclusion masks
`ReliefMesh.hole_mask` surfaces the mesh's own gap data (sky/invalid/band
exclusions plus rasterized torn quads, at full plate resolution) as a MASK
output on every relief-mesh node — the literal "where will 📽 Project show
black" signal, available on `AtlasDepthLayerMask` (opt-in
`compute_hole_mask`) BEFORE inpainting so it can drive the inpaint region.
Every relief-mesh node also takes an external `exclude_mask` (e.g. a
ComfyUI-RMBG `SAM3Segment` sky segmentation) which **replaces** the internal
sky heuristic — the heuristic eats tall geometry above the horizon (37% of
monument valley's buttes), a real segmentation doesn't.

### Sky dome (`AtlasSkyDomeLayer` ☁)
The classic DMP sky separation: the SAM sky mask drives a flat far card
(constant forward-Z, `radius_m`) with the segmentation embedded as a
per-pixel edge matte. `edge_extend_px` deterministically smears sky colors
past silhouettes (Nuke-style edge-extend — no inpaint model needed for
narrow reveals); `frame_outpaint_px` pads the plate past the FRAME edges and
widens this one source's camera so small orbits never hit the plate
boundary.

### The edge doctrine: matte + overhang
Geometry silhouettes tear at grid resolution; the fix is per-pixel
**edge mattes** (`ProjectionSource.mask_b64`, cut in the projection shader
at the same projected pixel as the photo) plus **boundary overhang**
(`edge_overhang_cells` — meshes overshoot their mask/band edge so the matte
has something to cut; skyline coverage went 0.475 → 0.001 uncovered).
`AtlasCleanPlateLayer.embed_matte` auto-computes band mattes; mattes ride
into both DCC exports as plate alpha + standalone PNGs.

### Disocclusion fill (`fill_occluded`)
Band clips leave a hole where the foreground occluder stood — the inpainted
plate had pixels there but no geometry. `fill_occluded` diffusion-fills the
depth across the footprint on the decimated grid, so orbit/dolly reveals
inpainted content **on real geometry**. Band layers now default to
`relief_grid=384` / `depth_edge_rel=1.5` (the hangar calibration).

### True depth-shadow occlusion (`AtlasOcclusionMask` Phase 2)
`occlusion_mode="depth_shadow"`: the primary camera's own depth map IS its
shadow map — no render pass, pure numpy. Wire the shared `AtlasDepthMap`
into `primary_depth`; both sides ground-pin to one metric space.

### 📐 Extract Angle + execution pauses
Orbit the viewport to the view a patch should be generated at, click
📐 Extract Angle: the orbit delta is measured about the SAME pivot the
backend's `orbit_camera` uses and snapped to the Qwen Multiple-Angles named
views. The viewport's four `patch_*` STRING outputs stay **paused**
(ExecutionBlocker) until an extraction exists — and extractions are
fingerprinted against the solve+image, so swapping the photo re-arms the
pause instead of running a stale angle. `patch_prompt` feeds the Qwen
generation directly; `patch_view_override` feeds `AtlasAddPatchView`/
`AtlasOcclusionMask` as one wire (ComfyUI's backend rejects STRING→combo
links).

### VLM pre-flight (`AtlasAssessImage` 🧭)
A local VLM (Ollama/LM Studio/llama.cpp) analyzes the photo against an
expert prompt encoding Atlas's full settings knowledge and reports a setup
plan (scene type, depth model, band splits, camera-move viability with a
max-orbit estimate) directly on the node. The whole graph pauses behind it
until ▶ Continue Workflow — the first Queue costs only the assessment.
Advisory-only; works (and resumes) fine with no VLM running.

### All-in-one DCC layer exports
`AtlasExportNukeLayers` 🎞 and `AtlasExportMayaLayers` 🧊 export EVERY
projection layer (sky/plates/patches, each with its own camera) as ONE .nk /
ONE .ma respectively, sharing a single layer-collection path so the two can
never drift. The Maya scene was **verified live in Maya 2027** (mayapy,
37 checks) — which caught and fixed two real bugs: Maya's `projection` node
takes its frustum from `linkedCamera` (it has no focal/aperture attrs), and
Maya's OBJ importer always lands values as centimeters (imported groups get
×100). The same verification repaired the older Maya review-scene exporter,
which carried the same latent projection bug.

### Addendum 2 — same-day follow-up session (MVP pivot)

Product decision: **v1 ships without diffusion patches.** Instead the camera
move is restricted to measured coverage — the viewport's **🧭 Safe Zone**
button probe-renders the projected scene (magenta-sentinel hole counting,
exact to the shader) and clamps the orbit to the scene's real hole-free
envelope. Supporting that: patches became pure **texture projectors onto
existing scene geometry** (`reuse_scene` — the scale-registration problem
dissolves), every plate layer gained the sky's **deterministic edge-extend**
plus an **invented-pixels matte** exported to Nuke for regrain, and mesh
boundary skirts now **recede away from camera with a bevel** (slope in cell
units, 1.0 = 45°). Gate approvals and viewport navigation are
fingerprint-stable across image swaps. Hero recipe:
`examples/atlas_camera_ultra_single_image_workflow.json`. Next planned
lever: `frame_outpaint_px` for band layers (the sky's widened-camera trick),
since Safe Zone measurements show the frame edge — not silhouettes — is the
binding constraint on wide scenes.



## Addendum — the 2026-07-09 session: DA3, the X-ray track, and the five-layer stack

### Depth Anything 3 is the default

`inference/depth_estimator.py` dispatches any `depth-anything/DA3*` model id
to a second backend alongside the transformers V2 path. DA3METRIC emits
*canonical* depth converted to metres as `focal_px_at_processed_res x out / 300`
— and the focal comes from the **camera solve** (GeoCalib or vanishing
points), closing a loop V2's fixed-FOV metric heads structurally can't. All
solve-bearing call sites thread the solved focal; the image-only nodes take
an optional `solve` input for the same. Measured (see the chart below): ~3x fewer relief
tears on 2 of 4 scenes, a usable mesh where V2 shattered to 0 faces, ground
confidence to 0.96. Combo VALUES are append-only (they serialize into saved
workflows); core-library defaults deliberately stay V2 so `[neural]`-only
installs keep working.

![DA3 vs V2 torn fraction](images/chart_da3_vs_v2.svg)

### The experimental hidden-geometry track (research-only)

`AtlasPredictHiddenGeometry` 🔬 predicts per-pixel layered ray intersections
(the surfaces each camera ray pierces), registers the stack to the pipeline
depth via layer-0 median scale, and substitutes the first layer that clears
the occluder by a scene-adaptive margin. Two backends, one contract:
**LaRI** (regression, ~0.2 s, no upstream license — user-cloned only) and
**World Tracing** (diffusion, ~20-34 s, HF-gated, CC BY-NC-ND) — backend
choice is per-scene and flips both ways (canyon: WT 0.113 vs LaRI 0.639;
steep ridge: LaRI 0.134 vs WT 0.305). The node prints registration rel MAD
every run; under 0.2 = trust it.

The v2 architecture (6 calibration rounds) is **mask-membership, not depth
bands**: predicted surfaces behind near occluders are themselves near, so
every band split tried lost 76-97% of predictions. The X-ray layer's
geometry region is the `hidden_mask` (GrowMask -> InvertMask ->
`exclude_mask`), its paint is the `paint_matte`, band uncapped. Fragmented
predictions shred the layer mesh via the world-edge check (immune to
`depth_edge_rel`/grid/dilation — all measured no-ops); the node's coherence
pass fixes it at the source: `fill_gaps` (dual-field Jacobi) + `smooth_px`
**gaussian** smoothing (a median filter is edge-preserving and keeps exactly
the steps that tear: 0.455 vs 0.260 measured). Final hole-in-paint: hangar
0.07, canyon 0.19, jungle 0.26 (from 0.67).

### The five-layer stack (what the six hero workflows build)

![The layer stack](images/layer_stack.svg)

Base relief mesh + backdrop (plate = the **feathered clean-plate composite**
via ImageCompositeMasked, so background geometry never bakes in occluder
pixels) -> matted **foreground layer** from the original photo (SAM x band
mattes where band edges step) -> **X-ray layer** (LaMa-inpainted plate on
predicted geometry) -> **sky dome** on outdoor scenes. Interiors disable the
sky heuristic (SolidMask 0 -> `exclude_mask`). Seeds ship pinned
(`control_after_generate="fixed"` — ComfyUI silently randomizes any widget
named `seed` otherwise).

**Contact/support surfaces use cleanplate depth.** When a foreground removal
reveals a continuous road, floor, shoreline, or facade, run a second depth solve
on the approved cleanplate and project it as a full-range background relief with
`fill_occluded=false`. Keep original depth on the explicit foreground matte.
Using the far side of `AtlasBoundedBand` as broad support geometry can diffuse
the footprint onto the cutoff plane and export a vertical drop; the bounded band
is still appropriate for clipping a foreground that extrudes too far.

Band-layer meshes stay at the calibrated 384 / 1.5 defaults. The measured
tear curve explains both numbers — finer grids reduce spurious tears until
cells approach pixel scale, and the looser edge threshold is safe inside a
band because the band clip already bounds the depth range:

![Torn fraction vs grid density](images/chart_torn_vs_grid.svg)

### Viewport additions

- **🎨 Layers** — opaque per-layer identity tints + legend; black = nothing
  paints, always a finding.
- **🩻 X-ray** — tints invented-geometry pixels (red = LaRI, blue = WT),
  only under 📽 Project.
- Orbit pivot is now the median sampled vertex depth on the camera's central
  ray (a bounding-box center was tail-dominated on full-scene relief
  meshes); 📷 Camera View resets via `{force:true}` (the fingerprint guard
  had silently swallowed explicit resets); the dead primitive/proxy toolbar
  buttons (Box/Plane/Cylinder/Person/Woman/Sedan) were removed.

### The six hero scenes

| | | |
|---|---|---|
| ![cathedral](images/scene_cathedral_nave.jpg) `cathedral` — LaRI, interior | ![hangar](images/scene_scifi_hangar.jpg) `space_hangar` — WT, shallow (clear_rel 0.10) | ![jungle](images/scene_jungle_temple.jpg) `jungle_temple` — WT, sky + SAM fg |
| ![canyon](images/scene_canyon.jpg) `canyon` — WT wins (LaRI misregisters) | ![ridge](images/scene_steep_ridge.jpg) `steep_ridge` — LaRI wins (WT misregisters) | ![valley](images/scene_wide_valley.jpg) `wide_valley` — honest weak case |

The remaining curated scenes (`docs/images/scene_*.jpg`: monument valley,
ghost town, alpine village, sea-cliff castle, snow-capped peaks, the
mountain-ridge figure) anchor the rest of the example catalog — each was
chosen to stress one failure mode; the
[🎞 Examples Catalog](https://claude.ai/code/artifact/186c3a6a-a778-40f0-8f39-fe29cfa6aace)
maps every workflow to its scene.

### Where the full story lives

Three companion pages (same design system, published 2026-07-09):
the [🥞 Build-Up Guide](https://claude.ai/code/artifact/77b10784-a6d5-4def-89bd-84cbfaabc21e)
(the stack taught stage by stage), the
[🎞 Examples Catalog](https://claude.ai/code/artifact/186c3a6a-a778-40f0-8f39-fe29cfa6aace)
(every shipping workflow, scenes, settings, dependencies, professional/OCIO
output), and [📊 Technical Details](https://claude.ai/code/artifact/4781289c-50dd-47fc-8571-1ef67513b7ba)
(the measured numbers as charts). Repo-side: CLAUDE.md (full catalog +
design rules) and THIRD_PARTY.md (license boundaries).
