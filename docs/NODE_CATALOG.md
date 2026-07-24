# Atlas Camera — ComfyUI Node Catalog & Frontend Reference

> Split out of `CLAUDE.md` (2026-07-24) to keep the per-session context lean.
> This file is AUTHORITATIVE for the node catalog, module layout, frontend
> extension (`atlas_blockout.js`) behaviour, API endpoint, and the example
> workflow catalog — update it (not CLAUDE.md) when those change.
> CLAUDE.md keeps only the routing index + hard rules.

## ComfyUI integration (`atlas_camera/comfy/`)



### Module layout (nodes.py modularization, 2026-07-19)



The former 9,110-line `nodes.py` was split into responsibility modules; the 71

node classes (67 standard + 4 experimental) now live in the group modules, and

`nodes.py` is a thin **compatibility façade** (≈180 lines) that re-exports every

class, shared helper, and the registry mappings so `from atlas_camera.comfy.nodes

import X`, `comfy/__init__` (`NODE_CLASS_MAPPINGS` / `_ATLAS_BLOCKOUT_CACHE`), and

saved workflows keep working unchanged. New code should import from the specific

module.



- `node_helpers.py` — the ComfyUI ADAPTER leaf: guarded optional-import shims,

  tensor/PIL/base64 conversion, registry probes, the ExecutionBlocker shim, the

  node-expansion graph builder, and per-execution caches. Depends only on

  `core`/exporters/importers/raw (never on a node class), so it cannot cause a

  cycle. **Reduced 1,591 → ~850 lines by the layering refactor (2026-07-20,

  `docs/dev/node_helpers_layering_plan.md`)**, which moved the host-agnostic

  math into `core/` where the architecture already said it belonged. Everything

  moved is RE-EXPORTED here, so `from ...node_helpers import X` and the

  `comfy.nodes` façade both keep working unchanged — a contract pinned by

  `tests/test_facade_surface.py` (all 155 façade names; verified to FAIL when a

  symbol is dropped, not merely to pass).

  What stayed and why: `_metric_depth_and_validity` and

  `_band_resolution_validity` never mention torch, but they call

  `_resolve_exclude_mask`, which converts a ComfyUI MASK tensor — they are

  TRANSITIVELY host-bound, and moving them would need a signature change rather

  than code motion. `_ground_scale_cached` stays because memoisation is an

  adapter concern, not math. (The first draft of the plan classified all three

  as "pure" by checking direct references only; the dangling-ref check caught it

  before any code moved.)

- `viewport_payload.py` — the viewport WIRE PROTOCOL: `_extract_blockout_camera`

  (231 lines on its own) plus `_fit_long_edge` / `_plate_ref_to_dict` /

  `_output_profile_to_dict`. The serialization boundary between a solved scene

  and `atlas_blockout.js`. Two conventions here are load-bearing: `fx/fy/cx/cy`

  describe the PHOTO (read by `makeProjectionMaterial`) while

  `render_fy`/`render_image_height` are separate keys for the VIEWING camera, so

  a ShotCam cannot corrupt the projection; and `primary_depth_b64` is bit-packed

  R/G/B = high/mid/low bytes of a 24-bit millimetre integer, unpacked as

  `z_mm = R*65536 + G*256 + B` and requiring NEAREST sampling.

- `view_prompts.py` — the Qwen named-view vocabulary (`_AZIMUTH_VIEWS` etc.) and

  its STRING parsers. Module-level on purpose: `AtlasOcclusionMask` must place

  its target camera identically to `AtlasAddPatchView`, and drift between the two

  would silently misalign a precomputed mask from the patch geometry.

- `node_reports.py` — on-node report suffixes (scale trust, scene health) and the

  `atlas_project.json` manifest writer. Rule: a manifest failure must NEVER fail

  an export.

- `fingerprints.py` — content hashes behind every gate approval. Gate widgets

  persist in a saved workflow, so approval must be scoped to WHAT was approved;

  without this a new image sails through the previous image's approval.

- `nodes_solve.py` — loading, registration, camera solving, scale/gravity/pitch/

  roll, assessment, solve/health gates, decompose. Imports `AtlasDebugReport`

  from `nodes_viewport` (the one cross-module class edge — a clean DAG, because

  `AtlasSceneHealthGate` reuses `AtlasDebugReport._matte_coverage`).

- `nodes_depth.py` — depth maps, outlier/ground/horizon/VP masks, MoGe normals,

  band split/bounded band, depth-layer mask.

- `nodes_geometry.py` — projection-geometry derivation, relief/walls/towers/roofs/

  interior, merge, shot cam, patch views, occlusion, and the four experimental

  nodes.

- `nodes_inpaint.py` — crop/stitch, SDXL, SAM3/semantic/scope masks, clean-plate

  layer/stack, sky dome.

- `nodes_export.py` — JSON/review/USD/Maya/Blender/Nuke/camera-path/relief-mesh

  exporters.

- `nodes_viewport.py` — viewport, output-desk controls, debug report, layer

  preview, and the `AtlasInput` node-expansion entry.

- `node_registry.py` — imports every class and builds `NODE_CLASS_MAPPINGS` /

  `NODE_DISPLAY_NAME_MAPPINGS` + the `ATLAS_EXPERIMENTAL` gate (dict literals live

  here). The keys/display names are a saved-workflow contract — never rename or

  reorder an existing entry. `tests/test_comfy_node_registry.py` pins the whole

  surface; `tools/audit_node_usage.py` (read-only) classifies each node's

  reference sites.



Three tests monkeypatch a nodes-module helper on the class's own module

(`_comfy_registry` for `AtlasInput`/`AtlasSDXLInpaint`, `_save_image_tensor_to_tmp`

for `AtlasMogeNormals`); their patch targets follow the class into its new module.



### Setup



A symlink connects the node pack to ComfyUI's custom_nodes directory (paths below

are this machine's; substitute your own ComfyUI install and repo checkout locations):



```

<COMFYUI_ROOT>\custom_nodes\AtlasCamera

    → <REPO_ROOT>\atlas_camera\comfy

```



The package is installed in editable mode in ComfyUI's venv so `import atlas_camera` resolves to this project directory. Changes to Python source are live immediately — no reinstall needed.



### Clone-and-go entry point (repo root `__init__.py`, 2026-07-11)



The repository root is itself a loadable ComfyUI custom node: `git clone` into

`custom_nodes/` works with no pip install (the root `__init__.py` prepends the

checkout to sys.path, re-exports the mappings from `atlas_camera.comfy`, and

sets a RELATIVE `WEB_DIRECTORY = "./atlas_camera/comfy/web"`). The dev

symlink+editable setup never loads that file. `atlas_camera/comfy/__init__.py`'s

own `WEB_DIRECTORY` is now the conventional relative `"./web"` (was absolute —

worked only because os.path.join discards the left side for absolute right

sides). pyproject carries `[tool.comfy]` (PublisherId `miikejamesburns`) for

`comfy node publish`; the publisher must exist on registry.comfy.org first.

Pinned by `tests/test_node_pack_entrypoint.py`, which loads the root file the

way ComfyUI would.



### Double-import guard (critical)



`atlas_camera/comfy/__init__.py` is loaded twice at ComfyUI startup:

1. As `AtlasCamera` custom node (by ComfyUI's node loader)

2. As `atlas_camera.comfy` package (by `from atlas_camera.comfy.nodes import ...` inside the same file)



Both loads hit the same file, causing the aiohttp route `GET /atlas/camera_data/{node_id}` to be registered twice, which raises `RuntimeError: method HEAD is already registered`. The fix in `__init__.py` checks `if not any(r.path == ... for r in _routes)` before registering.



### Node catalog (67 nodes + 4 experimental)



**Menu structure (2026-07-21):** every node lives in an `Atlas Camera/<folder>`

subcategory — the flat top-level "Atlas Camera" list is empty. Folders: `Solve`,

`Scale & Trim`, `Masks & Depth`, `Gates & QA`, `Derive Geometry`, `Inpaint

Layers`, `Patches`, `Export`, `Color`, `Blockout`, `Project`, `Experimental`.

CATEGORY is menu-placement only — changing it never affects a node key or a saved

workflow, unlike a rename. **Three nodes were removed in this same pass** (their

keys are gone, so a saved workflow referencing one will fail to load): the

`AtlasMegaPipeline` 🔬 monolith (unused, crashed), `AtlasLoadImageSolveCamera`

(long-deprecated file-path solve), and `AtlasPitchTrim` (the gravity-mirror

pitch dial — recoverable from git if the flip-repair is wanted back).



**Category: Atlas Camera**



| Node class | Inputs | Outputs | Notes |

|---|---|---|---|

| `AtlasSolveFromImage` | image (IMAGE), ±focal_mm, ±sensor_mm, ±detect_vanishing_points, ±raw_meta (ATLAS_RAW_META) | ATLAS_SOLVE | Geometric VP solve; accepts ComfyUI tensor. VP detection defaults ON. `raw_meta` (from `AtlasLoadRAW` 📷) supplies EXIF focal + measured sensor when the widgets are at defaults — finally implements the "0 = auto-detect or EXIF" tooltip |

| `AtlasLearnedSolveFromImage` | image (IMAGE), ±height_mode, ±camera_height_m, ±depth_model, ±sensor_mm, ±weights, ±device, ±focal_length_mm, ±raw_meta (ATLAS_RAW_META) | ATLAS_SOLVE | Learned GeoCalib prior (focal+gravity). `height_mode=measure_from_depth` fits the ground plane to measure camera height (no assumed eye height) and fills the depth slot. `focal_length_mm` (APPENDED 2026-07-18) / a wired `raw_meta`: trusted EXIF focal REPLACES GeoCalib's estimate (gravity retained) — see the RAW design rule. Needs `[neural]` |

| `AtlasDepthAnything` | image (IMAGE), ±depth_model, ±device | depth_image (IMAGE) | Monocular depth (Depth Anything V2), metric or relative. Needs `[neural]` |

| `AtlasScaleOverride` | ATLAS_SOLVE, ±scale, ±camera_height_m | solve (ATLAS_SOLVE), report (STRING) | � Artist's manual metric-scale dial. Single-image scale is ambiguous (no ground plane / reference → `assumed_default` 1.6m, often ~10× off on elevated vistas). Since scale � camera height, rescales the solve by `scale` (multiplier, 10.0 = the "1:10" case) or to an absolute `camera_height_m` (0 = off) — multiplying the camera position + both matrices' translation columns. EVERY downstream metric follows (geometry distances, � Band Box cutoffs, DCC-export cameras); the projection/view is unchanged (angular). Pure-Python, zero deps; stamps `scale_source="manual_override"`. Composable after any solve. Pinned by `tests/test_scale_override.py` (incl. the estimate_ground_scale ×N flow proof) |

| `AtlasRollTrim` | ATLAS_SOLVE, ±roll_deg | solve (ATLAS_SOLVE), report (STRING) | 🎚 The roll counterpart of the � scale dial (2026-07-16): rotates the recovered camera about its own VIEW AXIS by `roll_deg` — position and view direction INVARIANT (Rz preserves the camera z axis), only up/right spin, so framing is preserved. Levels a solve by eye when GeoCalib's gravity drifts a few degrees on AI plates with no true horizon (measured live: −5.6° solved vs ~−2.6° implied by architecture verticals on a sci-fi interior; the classical VP detector found ZERO VPs on that greebled octagonal chamber, so nothing catches it automatically). Updates view/world/rotation matrices as a rigid family and RECOMPUTES the stored horizon LINE (a rolled camera's horizon is no longer a single image row — the vanishing line of world-horizontal planes, linear in (u,v)) + `camera_estimation.horizon_angle`. Positive = scene turns counter-clockwise on screen (sign pinned by test). Wire BETWEEN the solve and the depth/derive nodes — geometry back-projects through the view matrix; the report warns if the incoming solve already carries proxy geometry. Pure-Python, zero deps; stamps cumulative `debug_metadata["roll_trim_deg"]`. Pinned by `tests/test_roll_trim.py` |

| `AtlasReferenceScaleSolve` | ATLAS_SOLVE, reference_id, bbox_x0/y0/x1/y1, ±height_override_m | solve (ATLAS_SOLVE), camera_height_m | Sets metric camera height from a known-size object (single-view geometry). Composable after any solve. No extra deps |

| `AtlasVLMScaleCues` | image (IMAGE), ±provider, ±model, ±base_url, ±min_confidence, ±api_key | scale_references (STRING json), summary (STRING) | Detects known-size objects with a VLM → scale_references JSON. `provider` selects `ollama`/`lmstudio`/`llamacpp`/`openai` (default `ollama`; `openai` = any OpenAI-compatible CLOUD endpoint for users without local models — needs `api_key` or `OPENAI_API_KEY` env); `base_url` blank uses that provider's own default URL, `model` blank uses its default model. Local providers need a running server; fails soft to `[]` if unreachable |

| `AtlasApplyScaleReferences` | ATLAS_SOLVE, scale_references (STRING), ±confirm, ±min_confidence | solve (ATLAS_SOLVE), camera_height_m, report (STRING) | Rescales metric height from scale_references — **only when `confirm` is on** (else records candidates). No auto-promotion |

| `AtlasDeriveProjectionGeometry` | ATLAS_SOLVE, image (IMAGE), ±depth_model, ±max_walls, ±max_objects, ±device, ±geometry_mode, ±relief_grid, ±primitive_method, ±scene_type | ATLAS_SOLVE | Derives the depth **relief mesh** (default) and/or fitted primitives into `projection_scene.proxy_geometry` (deep-copied solve). `geometry_mode` and `primitive_method` select the derivation strategy — see the "Multiple geometry-derivation strategies" key design rule below. `scene_type` (default `manual`) is a one-choice preset over those two plus `depth_model`: `organic`/`indoor`/`outdoor` — see that same rule. Feeds the blockout's 📽 Project mode. Note: solve JSON exports grow (~1MB) when the mesh is attached. Needs `[neural]` |

| `AtlasAddPatchView` | ATLAS_SOLVE, patch_image (IMAGE), ±patch_azimuth_view, ±patch_elevation_view, ±patch_distance, ±source_azimuth_view, ±source_elevation_view, ±flip_azimuth, ±name, ±depth_model, ±relief_grid, ±priority, ±device | ATLAS_SOLVE | Adds an AI novel-view "patch" to fill areas the primary camera can't see (occluded/grazing). Takes a novel view generated at a defined angle (Qwen-Image-Edit-2511 + Multiple-Angles LoRA, e.g. via the ComfyUI-qwenmultiangle "Qwen Multiangle Camera" node), constructs a patch camera by orbiting the recovered camera around the scene pivot to that view (`camera_math.orbit_camera` — shares the primary's world frame), derives the patch view's own relief geometry (Depth Anything) in that frame, and appends a `ProjectionSource` to the solve. Chain one per angle; the viewport layers them over the primary with a facing-ratio mask. **Angle inputs use the LoRA's exact named views, which are ABSOLUTE (subject-relative), so set both `source_*_view` (what your source photo is) and `patch_*_view` (what you asked the LoRA for) — orbit applied = patch − source.** `flip_azimuth` corrects mirrored handedness. See the "Multi-angle patch projection" design rule. Needs `[neural]` |

| `AtlasOcclusionMask` | ATLAS_SOLVE, target_image (IMAGE), ±patch_azimuth_view, ±patch_elevation_view, ±patch_distance, ±source_azimuth_view, ±source_elevation_view, ±flip_azimuth, ±depth_model, ±device, ±angle_threshold, ±dilate_px, ±soft_edge_px, ±power | occlusion_mask, coverage_mask (both MASK) | Phase 1 (frustum/frame/facing-angle) mask of where the PRIMARY camera's projection is invalid at a target/patch view's surface — white = primary can't cover it, so a patch should fill it there. Places its target camera identically to `AtlasAddPatchView` (same named-view widgets, same `_named_view_orbit_delta` helper — never independently recompute the orbit) so the mask lines up with that node's later patch geometry for the same image. Pure backend/numpy (`depth_geometry.primary_camera_validity_mask`), no browser round-trip, runs headlessly. Intended pipeline: `Solve → AtlasOcclusionMask → ImageCompositeMasked (primary projection + target_image) → AtlasAddPatchView`. Does not yet detect true depth-shadow occlusion (an object hidden behind nearer geometry from the primary's view but still inside its frame/angle limits) — see `docs/dev/atlas_occlusion_mask_implementation_plan.md` for that Phase 2 design. Needs `[neural]` |

| `AtlasConstrainedSolve` | image, constraints_json | ATLAS_SOLVE | Artist-guided; pass scale_constraints for cam height |

| `AtlasLoadSolveJSON` | json_path | ATLAS_SOLVE | Load previously saved solve |

| `AtlasDecomposeSolve` | ATLAS_SOLVE | camera, confidence, source_method, image_width, image_height, solve_json, horizon_angle_deg | horizon_angle_deg comes from debug_metadata |

| `AtlasDecomposeCamera` | ATLAS_CAMERA | fx, fy, cx, cy, cam_x, cam_y, cam_z, focal_mm, fov_h_deg | cam_y must be > 0 for depth map |

| `AtlasGroundDepthMap` | ATLAS_SOLVE, image_width, image_height, near_m, far_m | depth_image (IMAGE), ground_mask (MASK) | Black if cam_y ≤ 0. width/height **0 = auto** (adopt source image size) |

| `AtlasGroundMask` | ATLAS_SOLVE, image_width, image_height | MASK | 1 = ground, 0 = sky. width/height **0 = auto** |

| `AtlasHorizonMask` | ATLAS_SOLVE, image_width, image_height, feather_px | MASK | 1 = above horizon (sky). width/height **0 = auto** |

| `AtlasVPVisualization` | image, ATLAS_SOLVE, ±show_horizon, ±show_vp_lines, ±line_opacity | IMAGE | Pass-through if no VPs detected |

| `AtlasBlockoutViewport` | ATLAS_SOLVE, source_image, resolution, client_data, ±preview_expand, ±primary_depth (ATLAS_DEPTH_MAP), ±controls (ATLAS_VIEWPORT_LINK), ±shot_cam (ATLAS_SHOT_CAM) | shaded, depth, normal, mask, path_frames (all IMAGE), camera_path (ATLAS_CAMERA_PATH) | OUTPUT_NODE; browser-side Three.js. `resolution` = long edge; W×H auto-follows source image aspect and sets the actual WebGL render/intrinsic resolution — unless `shot_cam` resolves (direct wire, or inherited from `solve.shot_cam` attached by `AtlasMergeGeometry`), in which case the render resolution/aspect and viewing-camera FOV conform to that project format instead. See "ShotCam" below for why this is safe (never touches how the source photo is projected onto geometry). The node is also freely resizable by dragging its corner — CSS-only (`container`/`canvasWrap` flex-grow, canvas `height:100%`), no `node.onResize` hook; see "Detached viewport controls" below for why a hooked version froze the tab and why this one can't. Dragging just rescales whatever's already in the WebGL buffer (may blur if you drag far beyond the current `resolution`); bump `resolution` for a sharper render at a given size. `preview_expand` (default 1.0 = off) optionally dilates derived geometry outward from the camera for wider orbit coverage in the grey/undressed preview — display only, never affects exports/measurement, but **actively conflicts with 📽 Project**: dilated geometry has no corresponding photo data, so it renders as empty/black in projected mode the moment you orbit off the exact recovered viewpoint (see "Orbit coverage" below). Leave at 1.0 whenever you intend to use Project. `primary_depth` supplies the packed metric shadow map for ✂ Occlude and MUST be the same shared `AtlasDepthMap` used to derive the displayed relief (never an export preview regenerated with another model). The packed transport is nearest-sampled and capped at 2048px long edge. With the toggle on, depth rejection is an absolute relative-depth mismatch gated to real near/wide depth discontinuities, so both halves of a stretched curtain are removed without allowing a model/retopo mismatch to erase broad smooth facades; large source-texel footprints fade only when they are grazing or on such an edge (the former major/minor anisotropy ratio incorrectly classified ordinary perspective building faces). Each relief mesh also carries a viewport-only per-vertex risk band on the kept side of every deliberately torn quad; a 5×5 binomial prefilter averages single-grid spikes and staircase direction changes across its two inward rings before GLSL interpolates the soft coverage, even when `primary_depth` is absent. The literal boundary is restored after averaging, so it remains fully transparent without moving a vertex or widening the band. GLSL then applies footprint-adaptive coverage dilation: Camera View keeps the original straight RGB opaque farther toward that edge so the feather does not visually enlarge the hole, while a stretched orbit footprint relaxes the dilation and restores the wider cleanup. It never samples neighbouring RGB or attempts to paint outside absent triangles. Mattes, topology boundaries, facing cutoffs, frame borders, and these conservative edge risks feed one derivative-filtered linear coverage/alpha value. Projection RGB stays straight through the display transform, coverage remains untransformed data, and Three.js applies it once at the straight-alpha blend boundary while keeping depth writes enabled. Render Passes button populates client_data. 🎥 Camera Path mode authors a keyframed move via five one-click buttons (Orbit L/R, Pan L/R, Dolly In — fixed 24fps/100 frames, computed from the recovered pose; FBX import may override timing); � Bake Path fills `path_frames` (an IMAGE batch, feeds a Video Combine node) and `camera_path` (raw keyframes, feeds `AtlasExportCameraPathUSD`) into the same `client_data` widget — see "Camera path animation" below. `controls` carries no data — connect an `AtlasViewportControls` node to move every button/panel off this node entirely, leaving it perspective-render-only — see "Detached viewport controls" below |

| `AtlasViewportControls` | *(none)* | controls (ATLAS_VIEWPORT_LINK) | Companion node — connect its output to an `AtlasBlockoutViewport`'s `controls` input to relocate that node's entire toolbar/panel (primitives, 📽/📊/ℹ, 🎥 Camera Path move buttons + FBX import, Render Passes) into this one. No Python computation; `noop()` returns a placeholder string. See "Detached viewport controls" below |

| `AtlasExportSolveJSON` | ATLAS_SOLVE, file_path | STRING | Writes JSON file |

| `AtlasExportBlender` | ATLAS_SOLVE, output_dir | STRING (script_path) | Writes build_scene.py |

| `AtlasExportNuke` | ATLAS_SOLVE, output_dir, ±relief_mesh_obj_path, ±output_profile | script_path, nk_path (both STRING) | Writes both a Nuke Python projection script (`nuke_projection.py`, needs Script Editor) and a native `.nk` scene (`nuke_projection.nk`, drag-and-drop / File > Open ready) describing the identical camera-projection graph. Wire `AtlasExportReliefMesh`'s `obj_path` into `relief_mesh_obj_path` (same pattern as `AtlasExportMayaReviewScene`) to live-project onto the real derived relief mesh (`ReadGeo2`) instead of the default flat 40×40m ground `Card` — see "Nuke camera-projection topology" below |

| `AtlasExportReliefMesh` | ATLAS_SOLVE, image (IMAGE), output_dir, ±grid_long_edge, ±depth_edge_rel, ±depth_model, ±device, ±format, ±fill_interior_holes, ±max_hole_edges, ±fill_depth_near_m, ±fill_depth_far_m, ±retopo_method, ±retopo_target_vertex_count, ±retopo_smooth_iterations, ±retopo_crease_angle, ±retopo_pure_quad | obj_path, glb_path (STRING), preview_solve (ATLAS_SOLVE), report (STRING) | Depth relief mesh → OBJ+MTL+texture and/or GLB (single binary, texture embedded, KHR_materials_unlit). Projection baked into UVs (imports textured into Maya/Nuke/ZBrush/Blender); torn at silhouettes; ground on Y=0; below-ground outliers clamped along the view ray. `fill_interior_holes` (default OFF) caps small interior tear holes in the EXPORT only — the live projection mesh keeps its deliberate tears — so the OBJ/GLB retopos/booleans cleanly in a DCC; `max_hole_edges` (64) separates tear loops from the frame perimeter, and `fill_depth_near_m`/`fill_depth_far_m` (0 = off) scope the fill to a band box (transcribe `AtlasBoundedBand.cutoff_m`). `preview_solve` carries the mesh ACTUALLY written — wire it into an `AtlasBlockoutViewport` to tune the fill without a DCC round-trip (the input solve is untouched, so the live projection mesh keeps its tears); `report` + an on-node render (`web/atlas_export_relief.js`) state what filled and the scope applied. See the interior-hole-fill design rule. Needs `[neural]` |
| `AtlasLiveMeshRepair` | ATLAS_SOLVE, ±backend (auto/cuda/cpu), ±live_fill_holes, ±live_fill_distance_m, ±live_fill_max_hole_edges, ±live_fill_edge_sawteeth, ±cap_enclosed_holes, ±smooth_boundary, ±remove_stretch_factor | solve (ATLAS_SOLVE) | 🔧 LIVE repair of every relief mesh serialized on a solve (scene + all ProjectionSources), placeable anywhere downstream (after AtlasInput / CleanPlateLayer / band nodes). cuda backend recovers the sampling lattice from the mesh's own UV meshgrid and runs the build-time PyTorch conv fill iteratively (`max_hole_edges` scales the pass budget); fills are gated by the depth-ratio + bounded-edge tear tests so they never bridge open silhouettes or reconnect orphaned wild vertices. `cap_enclosed_holes` = ZBrush-style Close Holes for ENCLOSED loops only (channel-tolerant flood; harmonic Jacobi membrane blending the hole's own boundary depths; size-limited by max_hole_edges; open silhouettes/frame can never cap). `smooth_boundary` Taubin-relaxes boundary loops with per-vertex displacement clamped to lattice scale (a depth-jump zigzag loop must not migrate into the void) + exact projective UV regen for moved verts. `remove_stretch_factor` = post-hoc stretched-shard cull, the live twin of the layer nodes' max_edge_factor, self-calibrated from the mesh's own median edge/depth ratio; runs BEFORE the fill so cap walls are never culled. cpu backend = the numpy ear-clip/sawtooth topology fill (max_hole_edges capped at 256 to avoid O(n²) freezes). Display + export both see the repaired mesh (it rewrites the serialized primitive) |
| `AtlasRetopologizeLayer` | ATLAS_SOLVE, ±layer (STRING: ""=primary, name, "*"), ±method (off/quad/decimate/smooth), ±target_vertex_count, ±smooth_iterations, ±crease_angle, ±pure_quad | solve (ATLAS_SOLVE), report (STRING) | 🔷 LIVE retopology for ONE layer's relief mesh (or all) before the viewport — the same `core/mesh_retopo` passes as the Maya/Nuke layer exporters, but rewriting the serialized primitive so 📽 Project and every export see the simplified topology. Reuses `exporters/_layers._retopologize_layer_mesh`: each mesh retopos against its OWN camera (patch/outpainted cameras differ from the primary) and quad/decimate regenerate projection UVs via the exactness-pinned `regenerate_projective_uvs`; smooth preserves counts + UVs. Deliberate revision of the export-only retopo doctrine (see DESIGN_RULES 2026-07-24 amendment). Vertex-count changes consume the layer's serialized `edge_risk` (cleared; viewport tolerates absence). Missing optional deps (pyinstantmeshes / trimesh+scipy / fast-simplification) degrade SOFT — report carries the pip hint, solve passes through |

| `AtlasExportUSD` | ATLAS_SOLVE, output_dir | STRING (usd_path) | Writes camera.usda |

| `AtlasExportCameraPathUSD` | ATLAS_SOLVE, camera_path (ATLAS_CAMERA_PATH), output_dir | STRING (usd_path) | Writes camera_path.usda — a time-sampled (`Usd.TimeCode`-keyed) animated camera from `AtlasBlockoutViewport`'s Camera Path mode. Separate node from `AtlasExportUSD` since it takes a different required input |

| `AtlasDepthMap` | image | ATLAS_DEPTH_MAP | Shared metric depth estimate for the composable geometry-derivation nodes below — estimate once, feed several. Carries the raw `DepthResult` by reference (zero-serialization custom type, same pattern as `ATLAS_SOLVE`); distinct from `AtlasDepthAnything`, whose IMAGE output is a lossy normalized preview that can't be used for metric geometry. Needs `[neural]` |

| `AtlasMogeNormals` | depth (ATLAS_DEPTH_MAP), image (IMAGE), ±normal_model, ±device, ±solve | depth (ATLAS_DEPTH_MAP), report (STRING) | 🧭 Predicted surface normals from MoGe, DECOUPLED from the depth source. Wire BETWEEN `AtlasDepthMap` (any model) and `AtlasCleanPlateLayer`: runs a MoGe `*-normal` model PURELY for its per-pixel normals, discards MoGe's own depth, and attaches those normals (resized to the input depth's resolution, `_resize_normal_field`) onto a COPY of the input `ATLAS_DEPTH_MAP` (`copy.copy`, input never mutated). So you keep V2/DA3 depth (whose far-field behaves on exteriors, where MoGe's runs away) AND get MoGe's cleaner relight normals. Reuses `AtlasCleanPlateLayer`'s existing `depth.normal` channel — no new widget on that node (its capability freeze); the layer's attach still needs `frame_outpaint_px==0` (an outpainted plate's normal map is out of uv-registration). Pass-through if the model returns no normals. `normal_model` (`_MOGE_NORMAL_MODEL_CHOICES`): `moge-2-vitl-normal` (331M, best), `-vitb-normal` (104M, lighter GPU), `-vits-normal` (35M, **CPU/MPS-viable for non-CUDA**); auto-downloaded from HF. Needs `[moge]`. See the decoupled-normals design bullet |

| `AtlasDeriveReliefMesh` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), ±relief_grid, ±relief_quality, ±depth_edge_rel | ATLAS_SOLVE | One job: continuous depth-following relief mesh + backdrop. Fits its own ground scale directly (`relief_mesh.estimate_ground_scale`) rather than borrowing it from a primitive-fitting pass |

| `AtlasDeriveWalls` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), ±max_walls, ±max_objects, ±distance_modes, ±exclude_mask, ±ground_anchor | ATLAS_SOLVE | One job: vertical walls + foreground boxes/cylinders (azimuth_walls). Truncates sloped roofs/spires — use `AtlasDeriveTowersSpires` for those. `distance_modes`/`exclude_mask`: see the skyline design rule |

| `AtlasDeriveTowersSpires` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), ±max_walls, ±max_objects, ±distance_modes, ±exclude_mask, ±ground_anchor, ±roofline_split | ATLAS_SOLVE | One job: walls extruded to the real silhouette top (vertical_extrusion) — reaches towers/spires/sloped roofs `AtlasDeriveWalls` truncates. `distance_modes`/`exclude_mask`: see the skyline design rule |

| `AtlasDeriveRoofsFacades` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), ±max_planes | ATLAS_SOLVE | One job: any-orientation planes via sequential RANSAC (ransac_planes) — sloped roofs, stepped/angled facades |

| `AtlasDeriveInteriorRoom` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP) | ATLAS_SOLVE | One job: Manhattan floor+walls+ceiling (room_cuboid) — orthogonal interiors only |

| `AtlasMergeGeometry` | solve_a (ATLAS_SOLVE), solve_b (ATLAS_SOLVE), ±shot_cam (ATLAS_SHOT_CAM) | ATLAS_SOLVE | Explicit Nuke-Merge-node equivalent — combines two independently-derived solves' geometry (e.g. `AtlasDeriveWalls` foreground + `AtlasDeriveReliefMesh` background). `solve_a`'s camera wins; dedupes the always-emitted `projection_backdrop` plane. Chain multiple instances for 3+-way combination. `shot_cam`, if connected, is attached onto the merged solve (`out.shot_cam`) — pure attachment, never mutates `solve_a`'s camera — so a project format defined once flows downstream to `AtlasBlockoutViewport` without rewiring. See "Composable geometry derivation" and "ShotCam" below |

| `AtlasDefineShotCam` | ±sensor_width_mm, ±sensor_height_mm, ±focal_length_mm, ±resolution | ATLAS_SHOT_CAM | Defines a project-level render/output camera format (sensor mm × 2 + lens mm + long-edge resolution) — like a Nuke/Resolve project format setting. Intrinsics-only, no position. Wire into `AtlasMergeGeometry` (attach) or directly into `AtlasBlockoutViewport` (a direct wire always wins over an inherited one). See "ShotCam" below |

| `AtlasDepthLayerMask` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), ±near_m, ±far_m, ±near_pct, ±far_pct, ±feather_px | layer_mask, occlusion_mask (both MASK) | One depth band → its own pixels (`layer_mask`) + everything nearer that occludes it (`occlusion_mask`, feed into `INPAINT_ExpandMask` → `INPAINT_InpaintWithModel` to build that band's clean plate). Composable: instantiate once per background layer. Metric depth via the same `relief_mesh.estimate_ground_scale` path `AtlasDeriveReliefMesh` uses. Shares `_resolve_depth_band` with `AtlasCleanPlateLayer` so the two nodes' bands can't drift apart — see "Inpaint layers" below |

| `AtlasBoundedBand` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), foreground_mask (MASK), ±extrude_multiplier, ±near_pct, ±far_pct | band_split (ATLAS_BAND_SPLIT), cutoff_m (FLOAT), report (STRING) | � Measures the FOREGROUND's own metric depth extent `W = P(far_pct)−P(near_pct)` over its mask and emits ONE absolute-distance split at `cutoff = near + extrude_multiplier·W` (default 2×). Wire the `band_split` into BOTH clean-plate layers' `band_split` input (with `band_side` set): foreground → `[0, cutoff]` (relief clipped — no runaway extrusion past the guessed distance), background → `[cutoff, +inf]` (card median falls back behind it, pushed back for parallax). Because the split is absolute `split_m` (not a percentile), both layers resolve the identical boundary regardless of their own pixel populations — no band drift, no `band_ref_mask` needed. Composition-only (reuses `AtlasCleanPlateLayer`'s existing input, respects its capability freeze); fails soft to an unclipped sentinel (`_BOUNDED_BAND_NOOP_M`) + report when it can't measure. Needs `[neural]`. See the bounded-band design rule below |

| `AtlasCleanPlateLayer` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), plate_image (IMAGE), ±near/far metres or percentiles, ±name/priority/plate_ref, ±relief_grid/depth_edge_rel, ±exclude_mask/fill_occluded, ±embed_matte/layer_matte, ±edge_extend/skirt/frame_outpaint, ±band_side/band_split/band_geometry/geometry_override/band_ref_mask/band_override, ±tearing controls | solve (ATLAS_SOLVE), hole_mask, extend_mask (MASK) | Inpainted clean plate + geometry depth → appends a `ProjectionSource` with `metadata.projection_mode="clean_plate"`. Camera is the **primary unchanged** — no orbit, unlike `AtlasAddPatchView`. Builds relief/card/ground geometry and exposes the actual hole/extension QA masks. For large subject removals over a continuous surface, feed a SECOND depth solve of the approved cleanplate as a full-range manual background with `fill_occluded=False`; retain original depth only for the explicitly matted foreground. A far bounded band plus `fill_occluded` is for narrow slivers, not broad hidden support, because diffusion can create a cutoff cliff. Chain one per layer; farthest has highest priority. See "Inpaint layers" below |

| `AtlasCleanPlateStack` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), ±plate_1..4 (IMAGE), ±matte_1..4 (MASK), ±name_1..4, ±geometry_1..4, ±grow_px, ±edge_extend_px, ±relief_grid, ±depth_edge_rel, ±mattes_are_transparency | solve (ATLAS_SOLVE), report (STRING) | 🧽 Up to FOUR artist-painted cleanplates + alphas in one node (user-designed, 2026-07-17) — the multi-slot cleanplate injection port: separate the plate in Photoshop (sky/mountains/buildings/road), one full-frame plate + alpha per slot. Slot 1 = FARTHEST; priorities auto farthest-highest (15/10/5/0), behind slots get `edge_extend_px`, the nearest used slot always keeps a clean cut (seam doctrine baked in). Per slot: matte grown by `grow_px` → inverse = geometry `exclude_mask` (mask-membership), raw matte = paint `layer_matte`, `embed_matte` on. Pure composition over `AtlasCleanPlateLayer.add_layer` (capability freeze respected). Incomplete/empty slots skip soft + named in report; nothing wired = deep-copied pass-through. `mattes_are_transparency` inverts LoadImage MASK-output mattes (which mark transparent pixels). Pinned by `tests/test_cleanplate_stack.py` |

| `AtlasInpaintCrop` | image (IMAGE), mask (MASK), ±context_pad_px | cropped_image, cropped_mask, crop_region (ATLAS_CROP_REGION) | ✂ THE LaMa quality lever (see the inpaint-resolution design rule): crops a padded box around the inpaint mask so the inpaint node's fixed internal resolution is spent on the hole's neighborhood, not the whole frame. `context_pad_px` (default 128) is the quality/context tradeoff slider. Union bbox over all holes; empty mask passes through full-frame. Pure tensor orchestration — no inpainting math (GPL boundary intact) |

| `AtlasInpaintStitch` | original_image, inpainted_crop, crop_region, ±mask, ±feather_px | image (IMAGE) | ✂ Pastes the inpainted crop back; resizes mismatched crops (upscale models, mult-of-8 snapping). Whole-rect paste is exact for LaMa/MAT (they return original pixels outside the mask); wire `mask` (+`feather_px`) for generative inpainters that re-render the whole crop |

| `AtlasLayerPreview` | image (IMAGE), mask (MASK), ±layer_index, ±color_hex | image (IMAGE) | 🎨 Cut-out layer preview: plate pixels inside the layer's matte, the layer's 🎨 Layers debug color (opaque) everywhere else — one image showing WHAT the layer projects AND which layer it is. `layer_index` = position in projection_sources (the viewport 🎨 legend order; staged master sky=0/far=1/bg=2/mid=3/fg=4; −1 = primary teal). Palette hand-mirrors `atlas_blockout.js` `LAYER_DEBUG_PALETTE` (`_LAYER_DEBUG_PALETTE_HEX` — keep in sync); `tests/test_frontend_mirrors.py` FAILS on drift (it also pins the scene-preset mirror and numerically executes the JS Catmull-Rom/easing via node). Replaced the strip's separate mask+plate+hole previews (user feedback: show already-cut-out layers) |

| `AtlasScopeMask` | sky_mask (MASK), ±prompt, ±segment_mask (MASK, lazy), ±grow_px, ±min_coverage_pct, ±fallback_mask (MASK, lazy) | exclude_mask (MASK), status (STRING) | 🎯 Per-band scope exclude (`sky ∪ NOT(grow(segment))`) with SELF-DISARMING fallbacks — replaces the staged master's Grow→Invert→MaskComposite rows so scope rows stay permanently active: empty prompt (VLM says layer absent), unwired segment, or segment coverage < `min_coverage_pct` (a SAM no-match — found live at 0.0% on "desert floor and boulder", which used to invert into exclude-EVERYTHING and empty the layer to zero mesh) all return the plain sky mask = band-only behavior. `segment_mask` is lazy: an empty prompt never even executes the segmenter. `status` says which path fired. **Required companion:** any percentile band node consuming a scoped exclude must get the plain sky mask on its `band_ref_mask` (scoped excludes change the depth population and drift band edges apart — see the v8 note). `fallback_mask` (2026-07-12, CV-audit item 10): a geometry-prior segment tried BEFORE the band-only fallback on a SAM no-match — wire an `AtlasSemanticMask`; lazy in two stages (`check_lazy_status` pulls `segment_mask` first, then `fallback_mask` only when the segment's coverage actually no-matches, so the fallback model never runs otherwise) |

| `AtlasSemanticMask` | image (IMAGE), classes (STRING, default "sky"), ±model, ±device | mask (MASK), report (STRING) | 🧩 Named-class semantic mask via SegFormer/ADE20K (b0/b2/b4 combo; b0 ~15MB, CPU-viable) — a promptless, deterministic alternative to SAM3 text prompts: 150 fixed scene classes (sky/floor/building/tree/person/...), mask = union of matched classes. Class matching is EXACT-FIRST per token with substring fallback ("window"→"windowpane" works; "sky" can never bleed into "skyscraper" — found live). Intended roles: native sky-mask source without ComfyUI-RMBG, and `AtlasScopeMask.fallback_mask` geometry prior. No-match returns an empty mask + a report listing available classes. Live-verified: indoor hangar 0% sky / 88% wall∪ceiling∪floor, outdoor plates 10–18% sky. Needs `[neural]` (transformers)

| `AtlasSAM3Mask` | image (IMAGE), concepts (STRING, default "sky"), ±confidence_threshold, ±device, ±output_mode, ±max_instances | mask (MASK), report (STRING) | 🪄 Native SAM3 concept mask via `transformers>=5.5.4` (`[sam3]` extra) — no `triton`/comfyui-rmbg dependency, so it works on CUDA, CPU, and Mac (MPS) alike (MPS is best-effort — a per-concept one-shot retry falls back to CPU automatically if an op isn't yet supported), unlike the third-party `SAM3Segment` node it supersedes in `AtlasInput`'s own cascade. Same interface shape as `AtlasSemanticMask` (comma-separated concepts → union mask + report), which remains the learned fallback tier when `transformers<5.5.4`. **`output_mode` (APPENDED 2026-07-21; `merged` default / `separate`)** exposes the per-instance stack `post_process_instance_segmentation` was always returning — `merged` is one union mask, `separate` is `(N,H,W)`, one instance per slice, ordered **LARGEST FIRST** (SAM3's own score order is unstable between runs, which would silently repoint a saved `AtlasInstanceMask` index at a different object; `max_instances`, 0 = unlimited, caps it). Both views share ONE detection path (`_run_sam3_detector` returns instances; the union is a `|=` in `sam3_concept_mask`) so they cannot drift apart. This is what removed the LAST `triton` dependency from the node pack. `facebook/sam3` is gated on Hugging Face (Meta's SAM-License-1.0) — one-time `hf auth login` after requesting access; a gated-repo failure is caught and returned as the report string rather than raised (a one-time auth step, not a broken install), while version/import errors still raise normally. Inspired by lettidude/LiveActionAOV's `passes/matte/sam3.py`. |

| `AtlasDebugReport` | solve (ATLAS_SOLVE), ±depth, ±file_path, ±status_1..4 (STRING, forceInput), ±vlm_report | report, json_path (STRING) | � OUTPUT_NODE full-stack diagnostic of the layered master scene: camera summary (height derived from the 4×4 view matrix, never `camera_position`), per-ProjectionSource geometry type / verts / band range / matte coverage (decodes the embedded PNG), scope statuses, VLM report — plus red-flag analysis (zero-vertex layers, band GAP/OVERLAP, near-empty mattes, scope FALLBACKs), negative raw-depth fraction >1% (the DA3 watch-item, made measurable). Renders on the node AND writes structured JSON to a STABLE path (default `atlas_debug/master_debug.json`) so external tooling/an AI assistant can read one file instead of autopsying live payloads (built after exactly such a session). JSON carries `schema` (bump on breaking key changes) + `atlas_version`; single writer per path — point concurrent workflows at distinct file_paths |

| `AtlasInput` | image (IMAGE), ±layers, ±mesh, ±mesh_resolution, ±use_vlm, ±vlm_provider, ±vlm_model, ±sky, ±sky_prompt, ±scope_prompts, ±inpaint, ±upscale_model, ±edge_extend_px | solve (ATLAS_SOLVE), image, depth (ATLAS_DEPTH_MAP), sky_mask (MASK), report (STRING) | 🎬 The all-in-one entry node via **NODE EXPANSION** (`comfy_execution.graph_utils.GraphBuilder`): emits the real mini-graph at execution — our nodes by class, third-party SAM3Segment/LaMa by registry name — so inner steps cache individually and missing packs degrade gracefully (skipped + named in `report`, never an error). Defaults = instant relief (layers=0, one grid-512 mesh, VLM/SAM/inpaint off — image→viewport in one queue). layers 2–4 = band clean-plate layers on the proven splits via watertight `band_override` strings; use_vlm inserts AtlasAssessImage (auto_continue, offload) and wires prompts/geometry/bands, forcing 4 bands; sky feeds exclude_mask AND band_ref_mask everywhere (the drift rule); inpaint = per-band expand→✂crop→LaMa(seed pinned 0)→✂stitch with optional upscale model. Band layers bake in the DMP seam doctrine (2026-07-12, two artist passes on the alpine ridge plate): the smear lives on the layers BEHIND — every band except the frontmost gets edge_extend (now the `edge_extend_px` widget, **default 24** — lowered 2026-07-12 from a baked 64 that shredded high-frequency foliage into halos, user-reported on a jungle-temple plate; the front band still stays 0 and the value flows only to the behind bands) / skirt_bevel 1.5 / frame_outpaint 64 (sky card 96/128), the frontmost band keeps a CLEAN cut matte (0/0/0), and band priorities are FARTHEST-HIGHEST so the behind layer wins the watertight seam's depth near-tie (nearest-highest made every band's smear render in front of the layer behind it — striped columns at every seam). See the design bullet + examples/atlas_input_quickstart_workflow.json |

| `AtlasExportReviewPackage` | ATLAS_SOLVE, output_dir | STRING | Full review bundle |

| `AtlasExportMayaReviewScene` | ATLAS_SOLVE, output_dir, ±relief_mesh_obj_path | STRING | Maya scene + image card. Box/cylinder/plane proxies get real dimensions + transforms; wire `AtlasExportReliefMesh`'s `obj_path` into `relief_mesh_obj_path` to also import the real relief mesh (otherwise it's silently omitted, not placeholder-cubed) |

| `AtlasUSDCameraLoader` | usd_path | ATLAS_CAMERA | Load camera from USD |



| `AtlasAssessImage` | image (IMAGE), ±provider, ±model, ±base_url, ±extra_instructions, ±proceed, ±api_key, ±offload_model, ±auto_continue | image (IMAGE, GATED), report, settings_json, sam_prompt_sky/far/bg/mid/fg (5× STRING), geom_far/bg/mid/fg (4× STRING), band_far/bg/mid/fg (4× STRING) | 🧭 VLM pre-flight, wired directly after LoadImage. A VLM (ollama/lmstudio/llamacpp local, or `openai` = any OpenAI-compatible cloud endpoint + api_key; same provider layer as `AtlasVLMScaleCues`) analyzes the photo against `inference/assessor.py`'s `ATLAS_ASSESSMENT_SYSTEM_PROMPT` (the full settings knowledge as decision rules: scene_type, depth model, band design, camera-move viability rubric with max_orbit_deg) and shows the report ON the node (ui.text + `web/atlas_assess.js`). The `image` output returns ExecutionBlocker until `proceed` — first Queue costs only the assessment; ▶ Continue Workflow button resumes. The 5 appended `sam_prompt_*` STRINGs carry the staged master's per-layer SAM3 prompts from the payload's `staged_layers` block (`assessor.staged_layer_prompts`) — wire into SAM3Segment prompt inputs; absent layers yield "" (leave that stage bypassed), sky falls back to literal "sky". They flow UNGATED (everything they feed also consumes the gated image). Assessment cached per image+params (failures never cached). Advisory-only; fails soft without a provider |

| `AtlasAssessOutput` | camera_view (IMAGE), solve (ATLAS_SOLVE), ±source_image, ±depth, ±solve_summary, ±enabled, ±provider/model/base_url/api_key/offload_model, ±file_path, ±fallback_to_source | report, assessment_json, json_path, verdict, image_provenance (STRING), assessed_image (IMAGE), evidence_path (STRING) | 🧪 Terminal VLM + deterministic scene-health review for agentic/headless runs. Connect the final viewport `shaded` output and solve. Browser proxy passes exist only after **Render Proxy Passes**; an all-zero headless pass is detected and reconstructed in the recovered camera from the solve's actual projection plates, straight mattes, and relief topology. The VLM sees the retained output, its union-coverage matte, and the source reference; deterministic coverage and exposure-tolerant structural-drift checks override optimistic model prose. If reconstruction is impossible it may fall back to `source_image`, but projection/culling/inpaint checks are then forcibly `inconclusive` (a source plate can never be promoted to an output-quality pass). Always writes schema-versioned JSON plus hashed evidence/source PNGs, even disabled/provider-unavailable; `atlas_run_workflow(assess_output=True)` enables shipped terminal nodes and returns bounded paths/report JSON inline, never image/base64 history. Canonical headless evidence cannot prove orbit/grazing occlusion, so that visual check remains inconclusive. Assessment-only: no RGB transform or alpha mutation; prompt recommendations preserve unpremultiply → straight-RGB filter/dilate → premultiply and never color-transform alpha |

| `AtlasSolveGate` | solve (ATLAS_SOLVE), source_image (IMAGE), ±proceed, ±approved_for | solve (ATLAS_SOLVE, GATED), report | ✅ Solve-confirm checkpoint — the third gate in the ExecutionBlocker family: wire `solve → viewport` UNGATED (cheap preview) and `solve → gate → heavy stack`; first Queue costs seconds, ✅ Approve Solve (button, `web/atlas_solve_gate.js`) stamps a solve+image fingerprint and re-queues. Re-solve or new photo re-arms. Report = solve summary (focal/FOV/height/pitch/confidence) rendered on the node |

| `AtlasSceneHealthGate` | solve (ATLAS_SOLVE), source_image (IMAGE), ±depth (ATLAS_DEPTH_MAP), ±status_1..4, ±pass_through_on_pass, ±proceed, ±approved_for | solve (ATLAS_SOLVE, GATED), report | 🩺 Gate 4 — the ACKNOWLEDGEMENT gate before the exporters: runs `core.scene_health.evaluate_scene_health` (the same red-flag engine `AtlasDebugReport` renders) and holds the solve on warn/fail until ✅ Acknowledge & Continue (`web/atlas_scene_health_gate.js`; SolveGate mechanics — fingerprint identity, ships closed, RE-ARMED). `pass_through_on_pass` (default ON) = clean scenes flow with zero clicks. EVERY execution stamps `debug_metadata["scene_health"]` (report + acknowledged + evaluated_at) — override a warning, never lose it: the stamp rides exporter summaries, review report.md, and atlas_project.json. Full scenario matrix: gate_state_table.md Gate 4 |

| `AtlasDepthOutlierMask` | depth (ATLAS_DEPTH_MAP), ±rel_threshold, ±mad_threshold, ±dilate_px | mask (MASK), report | 🛡 Local 3×3-median + robust-MAD depth-outlier detector — turns isolated monocular-depth hallucinations into EXPLICIT holes (OR-ed into `exclude_mask`/`outlier_mask` on the relief nodes) instead of letting one bad pixel become a frame-spanning stretched shard. From the outlier/stretched-edge tier (see that design rule) |

| `AtlasSDXLInpaint` | image (IMAGE), mask (MASK), checkpoint, prompts, sampler params | image (IMAGE), report | ✨ Native SDXL inpaint adapter — expands to ComfyUI's stock CheckpointLoader → CLIPTextEncode → `InpaintModelConditioning` → KSampler → VAEDecode (the conditioning path matters: plain VAE-encode produced flat gray fills). Pairs with ✂ AtlasInpaintCrop/Stitch |

| `AtlasInstanceMask` | mask (MASK, an `(N,H,W)` instance stack), instance_index, ±restrict_mask, ±min_coverage | mask (MASK), report | 🎭 SAM3 instance selection — one building/object instance mask at a time, for per-instance inpainting. Feed it `AtlasSAM3Mask` at `output_mode="separate"` (native, no triton) or the third-party `SAM3Segment` at `output_mode=Separate`; the node only ever indexes `m[idx]` on the stack, so it is source-agnostic by construction. Note the two sources ORDER instances differently — Atlas sorts largest-first for stability, SAM3Segment by score — so re-check the index when switching |

| `AtlasSegmentedSDXLInpaint` | image, paint matte, prompt, SDXL params | image (IMAGE), report | � Per-instance crop-and-stitch SDXL inpaint: SAM3-separated instances ∩ the LaRI paint matte, each inpainted in its own crop then stitched sequentially — avoids one giant crop inventing a single connected mega-structure across buildings (live-verified on the D810 NYC plate). **Prefers the native `AtlasSAM3Mask` (`separate`) whenever `_native_sam3_available()`** — the same cascade `AtlasInput.segment()` uses — falling back to the third-party `SAM3Segment` (and its `triton` requirement, i.e. CUDA-only) otherwise. Mind the slot difference between the two: `AtlasSAM3Mask`'s mask is slot 0, `SAM3Segment`'s is slot 1. The report names which path ran |

| `AtlasSkyDomeLayer` | ATLAS_SOLVE, depth (ATLAS_DEPTH_MAP), sky_mask (MASK), plate_image, ±radius_m, ±relief_grid, ±name, ±priority, ±plate_ref, ±edge_extend_px, ±frame_outpaint_px, ±distance_m | solve (ATLAS_SOLVE), hole_mask (MASK) | � The classic DMP sky separation: a real segmentation (e.g. ComfyUI-RMBG `SAM3Segment` prompted "sky") drives a flat constant-forward-Z card at `radius_m` (matches the `projection_backdrop` convention — NOT a literal sphere; `relief_mesh.build_sky_dome_mesh` feeds `build_relief_mesh` a synthetic constant-depth field with `apply_sky_heuristic=False`, which would otherwise re-exclude it). The SAM mask is auto-embedded as the per-pixel edge matte. `edge_extend_px` (default 48) smears sky colors past silhouettes (deterministic Nuke-style edge-extend, quarter-res propagation — NOT an inpaint); `frame_outpaint_px` (default 64) pads the canvas past the FRAME edges and gives this source its own widened camera (cx/cy+P, W/H+2P) so small orbits never hit the plate boundary. `distance_m` (default 0 = legacy: radius_m IS the distance) decouples distance from size: when set, the card sits at `distance_m` and `radius_m` becomes its minimum half-extent (SIZE) — grown via extra outpaint (invented pixels declared in extend_mask; padding memory-capped at half the plate long edge per side), never shrunk below frustum coverage. Distance = parallax, size = orbit/pan slack. Same-camera pose, no orbit |

| `AtlasRenderFix` | images (IMAGE), ±fixer_path, ±docker_image, ±timestep, ±timeout_s | images (IMAGE), report (STRING) | 🔬 EXPERIMENTAL render repair via NVIDIA Fixer (Difix3D+ successor, single-step diffusion) — wire between the viewport's baked `path_frames` and a Video Combine node. Runs in a DOCKER container (cosmos/transformer_engine has no Windows build; image recipe `docker/fixer/Dockerfile`), user-cloned repo via `fixer_path`/`ATLAS_FIXER_PATH`. Weights NVIDIA Open Model License (commercial OK), repo Apache-2.0. See the "Fixer render repair" key design rule |

| `AtlasExportNukeLayers` | ATLAS_SOLVE, output_dir, ±output_profile, ±retopo_method/target/smooth/crease/pure_quad | nk_path, summary (STRING) | 🎞 EVERY `ProjectionSource` as ONE native .nk: per-layer Read + Camera2 (that layer's OWN camera — patches orbit, outpainted skies widen) + Project3D2 + ReadGeo2, merged through a single Scene into one ScanlineRender rendered from the primary camera (wired via the proven Root `onScriptLoad` callback). Assets alongside: plates (edge matte embedded in ALPHA) + standalone mattes + OBJ meshes, via the shared `exporters/_layers.collect_projection_layers`. Optional export-only retopology runs independently on **every layer mesh** and regenerates projective UVs from that layer's own camera before its OBJ is written. Layer overlap resolves by real z-depth. Additive — `AtlasExportNuke` stays the single-projection exporter |

| `AtlasExportMayaLayers` | ATLAS_SOLVE, output_dir, ±output_profile, ±retopo_method/target/smooth/crease/pure_quad | ma_path, summary (STRING) | 🧊 The Maya twin: ONE .ma with per-layer projector cameras as NATIVE nodes (transforms via `_matrix_to_maya_trs` Euler-xyz decomposition, round-trip-tested) + an embedded on-open scriptNode importing the OBJs and building the projection networks. It uses the same per-layer export-only retopology and camera-derived UV regeneration as NukeLayers. **VERIFIED LIVE in Maya 2027 via mayapy** (37 checks), which caught two real bugs now fixed in BOTH Maya exporters: the `projection` node has NO focalLength/aperture attrs (perspective frustum comes from `cameraShape.message → projection.linkedCamera`), and Maya's OBJ importer lands raw values as internal CM regardless of scene unit (imported groups get ×100). Mattes ride plate alpha → `file.outTransparency` → `lambert.transparency`. Same shared layer collection as the Nuke export, so the two DCCs can never drift |



**Category: Atlas Camera/Color** ("Output Desk" — float-safe plate tracking for final DCC/OCIO handoff)



| Node class | Inputs | Outputs | Notes |

|---|---|---|---|

| `AtlasRegisterPlate` | image (IMAGE), ±plate_path, ±colorspace, ±bit_depth, ±role, ±lut_path | image (IMAGE), plate_ref (ATLAS_PLATE_REF) | Pass-through IMAGE; the `ATLAS_PLATE_REF` carries a durable file path/colorspace/bit-depth/LUT for Nuke/Maya/OCIO handoff. Leave `plate_path` blank and it's marked `is_proxy=True` so exporters never mistake a browser/JPEG preview for final EXR data. `bit_depth="auto"` infers `16f/32f` for `.exr` paths, else `8-bit/proxy` |

| `AtlasAttachSourcePlate` | ATLAS_SOLVE, plate_ref (ATLAS_PLATE_REF) | ATLAS_SOLVE | Attaches a registered plate ref onto a solve (`solve.source_plate`) — pure metadata attachment, deep-copies the solve, never touches camera/geometry. Downstream exporters can read `source_plate` for the original/final colorspace instead of assuming the ComfyUI preview |

| `AtlasLoadRAW` | file_path, ±undistort, ±half_size, ±white_balance, ±exposure_ev, ±write_exr, ±output_dir, ±colorspace | image (IMAGE), plate_ref (ATLAS_PLATE_REF), raw_meta (ATLAS_RAW_META), focal_length_mm, sensor_width_mm, report | 📷 Camera RAW loader (NEF/CR2/CR3/RAF/ARW) — replaces the ACR round-trip: one rawpy demosaic → display tensor + scene-linear EXR sidecar (geometrically identical by construction; sidecar tagged `Linear Rec.709 (sRGB)`, NEVER ACEScg), EXIF + `camera_bodies.json` sensor lookup → `raw_meta` for the solve nodes, optional lensfun undistort with graceful-degrade statuses. Occupies the OCIORead slot in the Output Desk chain. Needs `[raw]` (+`[raw-lens]` for undistort). See the RAW design rule |



### Frontend extension (`atlas_camera/comfy/web/atlas_blockout.js`)



Registers as `AtlasCamera.Blockout` ComfyUI extension targeting `AtlasBlockoutViewport` nodes. On node creation it builds a Three.js canvas with a **self-contained orbit controller** (`createOrbitControls` — the examples/jsm `OrbitControls` uses a bare `import ... from "three"` that browsers can't resolve without an import map, so it never loaded; the custom one depends only on the already-loaded THREE), a primitive toolbar (Box/Plane/Cylinder/Person/Clear), scale-reference proxy buttons (� Woman / 🚗 Sedan), a 📷 Camera View reset, and a Render Passes button.



**Default node size — fresh nodes only (added 2026-07-07):** freshly added `AtlasBlockoutViewport` nodes default to **960×720** (`ATLAS_VIEWPORT_DEFAULT_WIDTH/HEIGHT`) instead of LiteGraph's cramped computed ~270×438 — double the 460px the example workflows historically shipped at, per artist request for a usable 3D preview without an immediate manual drag. Saved workflows keep their stored size: an `onConfigure` tracker (installed **synchronously in `nodeCreated`, before the first `await`** — `onConfigure` fires during `graph.configure`, i.e. after the handler's first await suspends, so a late hook would miss it) marks deserialized nodes, and the size bump is skipped for them. Display-only, like all node resizing here — render resolution is still governed solely by the `resolution` widget (a 768 render at 960 display width is a mild CSS upscale; bump `resolution` for sharpness at the new size). Verified live both ways: fresh node → 960×720; example workflow with stored 460×580 → unchanged on load.



**Aspect snap — the preview fills the full node width (added 2026-07-07, same artist request):** `object-fit:contain` letterboxes whenever the canvas box's shape differs from the render aspect, and the letterbox `#111` is the same shade as the WebGL clear color — so a node dragged wide read as "the preview is still small" no matter its size (two indistinguishable darks stack when a `shot_cam` FOV wider than the solved lens adds its own in-render surround on top). Fix: `snapNodeHeightToRenderAspect()` (called only from `resizeViewport`, i.e. on execution when the authoritative `target_width/height` arrive — deliberately NOT from a `node.onResize` hook, preserving the hard-earned "no JS resize hooks" rule) keeps the node's width and sets its **height** to `chrome + width / renderAspect`, where chrome (title + widget rows + any locally-mounted toolbar) is **measured from the live layout** (`node.size[1] − canvasWrap rect height`), so the detached-Output-Desk and local-toolbar cases both come out exact. Between executions a hand-dragged shape may letterbox; the next Queue snaps it back. **Off-screen guard:** executions often finish while the node is scrolled out of view, where ComfyUI hides the DOM widget and every rect measures 0 — the snap then stashes the aspect in `pendingSnapAspect` and the `animate()` loop retries until the widget is laid out again (a single null-check per frame otherwise). Verified live: 960×720 node + 16:9 render → snapped to 960×747, canvas box aspect exactly 1.778 = render aspect, photo edge-to-edge. Note the `shot_cam` in-render surround is orthogonal and remains by design — to make the photo itself fill the frame, drop the shot cam or match its focal to the solved lens (ℹ Info shows it).



**Canvas collapse on first click/orbit-release (`pinDomWidgetFullWidth`, fixed 2026-07-07 — fourth entry in the viewport resize-bug lineage):** artist-reported: viewport renders full-width after a run, then on the first orbit mouse-release the canvas box collapses to ~394px wide while the node stays big. Root cause chain, established by reading the installed frontend's sources out of its sourcemaps (`DomWidgets.vue`, `DomWidget.vue`, `domWidget.ts` — same technique as the earlier `domWidget.ts` extraction): ComfyUI's per-frame DOM-widget layout sizes the host element as **`widget.width ?? node.width`**, and `DomWidget.vue` binds `selectOn: ['focus','click']` listeners that fire on real mouse-release (browsers only auto-generate `click` from trusted input — which is also why synthetic pointerdown/up sequences never reproduced the bug; an explicitly dispatched `click` event does). Something sporadically writes a one-shot stale pixel width (observed live: own-property `width: 394` ≈ this node type's pre-configure computed width) onto the widget object; the value then sits dormant until the next widget-style resync — triggered by exactly that click→selectNode path — after which the collapse is permanent. The writer was never caught in the act (traps on the property and on `addDOMWidget` captured nothing across sessions; plausibly a frontend transient or another extension). Fix: `pinDomWidgetFullWidth()` — `Object.defineProperty` on the widget instance making `width` read as permanently `undefined` with a swallowed setter, so layout always falls through to the live node width no matter who writes. Behavior-neutral elsewhere: litegraph's own reads (`widget.width || nodeWidth` in hit-testing/drawing) fall through identically. Applied to both the viewport and `AtlasViewportControls` DOM widgets. Verified live: full-width after run, after the previously-collapsing click path (node selected), and after adversarially writing `width = 394` + forced redraws — host stays at node width in all three. Debug aid if a variant recurs: `localStorage.ATLAS_VIEWPORT_SIZE_TRACE = "1"` enables `installViewportSizeTrace` (stack-traced logging of suspicious node/widget size writes + ResizeObserver deltas).



**⛶ Fullscreen + UE-style tracking keys (2026-07-12):** toolbar button after 💡 Lights fullscreens `canvasWrap` (canvas + every HUD/diagram/legend overlay — NOT the container, whose toolbar may live in a detached Output Desk; canvasWrap behaves identically in both modes) via the browser Fullscreen API — a pure display change: no node sizing, no widget layout, no canvas-attribute writes; render resolution stays governed by the `resolution` widget and `object-fit:contain` letterboxes, exactly like dragging the node large. Esc exits natively; the `fullscreenchange` listener (label swap + canvas focus) is removed via the CHAINED onRemoved cleanup. **The one real hazard, guarded:** `snapNodeHeightToRenderAspect` would measure SCREEN-sized rects if an execution finishes while fullscreen and persist a garbage node height — it now defers to `pendingSnapAspect` while `document.fullscreenElement === canvasWrap` (verified live: re-queue during fullscreen left node.size untouched; the snap applies on exit). **Tracking keys** live in `createOrbitControls`: ↑/↓ track in/out (view-forward), �/→ track left/right, A/D up/down (user's mapping; W/S + Q/E as UE aliases), Shift = 4×, step scene-scaled (`sph.radius * 1.2 * dt`). Implemented as a public `pan(v)` (the Shift-drag pan math exposed: translate target, `apply()` re-poses the camera → true tracking, orbit clamps untouched) + a self-timed `updateKeys()` in the animate loop. Key listeners are on the CANVAS element only (`tabIndex = -1`, focused on pointerdown / fullscreen entry) with preventDefault/stopPropagation for handled keys exclusively — verified live: keys inert when focus is elsewhere, so ComfyUI hotkeys are never intercepted; (the free-fly controller that used to disable the orbit controller in path mode was removed 2026-07-16 — the orbit controller, and therefore these keys, now stays enabled in Camera Path mode).



**Orbit preserves the recovered camera's ROLL (fixed 2026-07-12):** GeoCalib solves include roll (tilted gravity — measured live at 28.4° on a hazy alpine-ridge photo with no true horizon, where every strong line is a diagonal ridge), and `applyRecoveredView` poses the camera with it — but `createOrbitControls.apply()` rebuilt the pose as `camera.up=(0,1,0); lookAt(target)`, so the FIRST drag snapped the camera level and the whole projected scene visibly rotated by the discarded roll (artist-reported as "the orbit camera rotates anticlockwise when I click"; latent since the controller was written — no earlier test image had meaningful roll). Fix: `syncFromCamera()` measures the signed roll about the view axis (angle from the level lookAt-up — world-up projected perpendicular to the actual quaternion-derived forward — to the actual up; straight-up/down degenerates to 0) and `apply()` re-applies it after `lookAt` via `camera.rotateZ(-rollAngle)` (rotateZ spins about local +z = BACKWARD, hence the negation — sign verified numerically via the vendored bundle: injected 28.4°/−15°/45° all reproduce to 2.4e-6° quaternion error, and re-measuring after apply is stable, so repeated sync/apply cycles can't drift). Verified live via Playwright on the exact reporting solve: a real 60px orbit drag now holds roll at 28.44° while the position genuinely orbits, and 📷 Camera View still restores the exact recovered pose. Orbit clamps, Safe Zone, tracking keys, and path playback are untouched (path playback authors level moves by design).



**Orphaned viewport DOM on workflow switch (`onRemoved` clobber, fixed 2026-07-12 — fifth entry in the viewport DOM-lifecycle lineage):** user-reported on the AtlasInput quickstart: floating slider stubs above the nodes and huge blue/green sheets with mesh-wireframe curls at the page bottom. Root cause: the viewport's cleanup was ASSIGNED (`node.onRemoved = () => {...}`) after `addDOMWidget`, which had already installed ComfyUI's own onRemoved via `useChainCallback` (domWidget.ts) — the assignment clobbered the frontend's DOM-detach step, so every workflow switch/reload left the old viewport's container + WebGL canvas + overlays ORPHANED in the document, rendering in normal flow (toolbar sliders lay out near the top, the body-wide canvas as sheets at the bottom). Diagnosed live via Playwright: a 0×0-rect but `visible` 1280×720 canvas parented in bare divs after `loadGraphData` over a viewport-bearing graph; fix (CHAIN the previous handler + belt-and-braces `container.remove()`) verified live — the same switch now leaves exactly one canvas. Lesson: on any node that calls `addDOMWidget`, NEVER assign lifecycle callbacks afterwards — always chain (`onResize`, `onRemoved`, `onConfigure` alike).



**Three.js loading — vendored local bundle (replaced the CDN chain 2026-07-07):** `loadThree()` imports one committed, self-contained ESM file, `atlas_camera/comfy/web/lib/atlas-three.bundle.js` (three **r185** core + `OBJLoader` + `FBXLoader`; ~770KB minified), built by `npm run build:comfy-three` in `ui/` from `ui/bundle/atlas-three-entry.js` (reuses `ui/`'s own pinned `three@^0.185.0`, so the React workbench and the ComfyUI extension can no longer drift apart in three version). The old chain was quietly broken, confirmed live: ComfyUI does **not** expose its internal three build at `../../lib/three.module.js` (frontend 1.45.20 bundles three r180 only as a hashed Vite chunk, no import map anywhere), so the first-choice import always failed over to unpkg CDN `three@0.163.0` (internet-dependent, 2 years stale) — and the unpkg `examples/jsm` loaders failed outright on their bare `import "three"` specifier, meaning **�/🚗 OBJ proxies and 📥 FBX camera import were silently dead in production** (each wrapped in try/catch → console warning only). The bundle re-exports the full three namespace at top level (`export * from "three"`) so the imported module object *is* THREE, with the two loaders as extra named exports. Verified live on r185 end-to-end: 📽 Project (contrast/saturation correct — the hand-written `atlasLinearToSRGB` encode carries over fine), grey preview, Render Proxy Passes round-trip, and � Woman OBJ loading (50k-vert mesh — restored from dead). Note for future FBX testing: FBXLoader r184+ auto-converts Z-up files to Y-up; the import's frame-0 `alignQuat` normalization should absorb this, but recalibrate by eye per the existing guidance. Upgrading three later = bump `ui/package.json`, rebuild, commit the new bundle; there is deliberately **no CDN fallback** (a broken bundle should fail loudly, not degrade into a version-skewed CDN copy).



The viewport **inherits the recovered camera**: `applyRecoveredView()` sets the Three.js camera to the recovered pose/fov, then initialises the orbit controller *from* it (`syncFromCamera`, pivot = the looked-at ground point) so the default view matches the source photo; dragging orbits, and 📷 Camera View snaps back. For exact photo alignment set the node's `width`/`height` to the source image's aspect ratio (the camera aspect = `target_width/target_height`). The background photo is a plane sized to fill the recovered frustum along the view axis.



**Camera projection (📽 Project, matte-painting mode):** the payload's `proxy_geometry` entries (from `AtlasDeriveProjectionGeometry`) are built as meshes (`buildDerivedProxies`, group `atlas_derived_proxies`, transforms fed verbatim to `Matrix4.set` — row-major both sides). `makeProjectionMaterial` is the GLSL port of `ui/src/ProjectionMaterial.ts` (world pos → recovered-camera pixel → sample source photo; discard behind camera / outside frame) with `depthWrite:true` (multi-proxy occlusion; the ui original was a single-ground overlay) and a **flipY:false** texture (top-left UV origin — never share the background texture, which uses default flipY). The 📽 toggle swaps ALL projectable meshes (derived + user primitives + OBJ proxies) between grey and the shared projection material; `_prevMaterial` is stashed only once so material rebuilds don't lose the original. Clear leaves the derived group alone (Python-owned; regenerates each execution). **Matte-painting property:** texels are assigned by ray, so geometry at slightly-wrong depth still receives exactly the pixels its silhouette subtends — perfect reassembly from Camera View; scale error only shows as parallax when orbiting.



**🎬 Backdrop toggle:** every primitive-fitting derivation strategy (`azimuth_walls`, `vertical_extrusion`, `ransac_planes`, `room_cuboid` — never `relief_mesh`) always emits one extra flat `"projection_backdrop"` plane (`proxy_geometry.py`/`plane_extraction.py`/`room_layout.py`, all via the shared `depth_geometry.build_backdrop_primitive`) sized to cover the whole frustum at the far-depth percentile — a catch-all so 📽 Project never shows raw background behind the fitted primitives. When `geometry_mode="both"` this backdrop plane is also projectable and sits behind/around the actual relief mesh, receiving its own copy of the projected texture. The toggle just sets `.visible` on any mesh named `"projection_backdrop"` (`scene.traverse`, matched by name, not a stashed reference — a plain visibility flag handles both the grey preview AND 📽 Project identically, since an invisible mesh never renders regardless of material), so Project can be limited to painting only the generated mesh. Re-applied inside `setProxies()` after every `buildDerivedProxies()` call, since that rebuilds fresh mesh objects (default `visible=true`) on each execution — without the reapply, toggling it off would silently reset on the next Queue.



**🕳 See-through backdrop — REMOVED (2026-07-14, was 2026-07-13):** briefly, the background source-photo plane (`bgMesh`) was kept visible UNDER 📽 Project (`renderOrder -100000`, `depthTest:false`, enlarged 3× with a soft edge-smear `ShaderMaterial`) so pixels the projection shader discards (matte-cut silhouette, torn quad, out-of-frame, facing-mask) saw THROUGH to the photo instead of the black clear colour, gated by a `🕳 See-through` toolbar toggle. The artist found it too buggy (the enlarged/edge-smeared plane showing through holes read as stretched-photo glitches on orbit/dolly), so the toggle + the `seeThroughOn` state + the see-through fill were removed. Current behavior (the pre-see-through convention): `applyProjection`/the bgMesh build set `bgMesh.visible = !projectionOn` — the photo plane is the grey (Project OFF) backdrop ONLY, and under 📽 Project it is hidden, so discards read as black. The band geometry (bounded card-split, etc.) is the intended cover for off-axis angles. The `bgMesh` itself (still enlarged with the edge-smear material) remains the grey-mode backdrop; only its Project-mode visibility + the toggle are gone. Verified live: no See-through button in the toolbar, `bgMesh.visible` false under Project / true in grey mode, no GL error.



**� Band Box overlay (2026-07-13):** a toolbar toggle that draws a translucent red box (per EVERY bounded foreground layer) whose BACK FACE is pinned to the cutoff plane — ANY clean-plate layer whose `far_m` is FINITE is a bounded foreground (an `AtlasBoundedBand` cutoff `near + N·W`, OR a depth band's own far edge — multi-plane band layers each get their own box + label automatically); the background card's `far_m` is null/+∞. `addBandBoxFor` builds one box per bounded group; the fill/cutoff-plane opacity scales down with the box count (`0.16/N`, `0.42/N`) so a single per-building box stays bold while 3 frame-spanning band boxes read light enough for the scene to show through (the always-visible edges + cutoff plane + label still define each). Each box gets a DISTINCT palette color by depth (near→far: red/amber/cyan/green/violet/yellow) — box fill, edges, cutoff plane, and the label (darkened-color bg + bright border + white text) all share it, so multiple bands are tellable apart. The label distance is the layer's metric `far_m` — verified correct to the solve (the box's cutoff plane sits at exactly `far_m` metres along the recovered camera's forward axis, and the relief is clipped there). **Scale caveat:** `far_m` is only as accurate as the solve's metric scale — a single-image solve with no ground plane to fit (AI cityscape/fantasy vistas) falls back to `scale_source=assumed_default` (1.6 m eye height), which is ~10× small for an elevated vista; fix by `AtlasLearnedSolveFromImage height_mode=assume` + `camera_height_m` as a scale dial, or a reference-scale node. `buildBandBox()` finds that group (smallest finite `far_m` among the `atlas_patch_N` groups) and builds the box **in the RECOVERED camera's frame** (reconstructed from `recoveredData.view_matrix` → `Matrix4` → invert = cam→world), so the box back face sits at view-space `z = −cutoff` (the clip plane) regardless of camera pitch, lateral bounds hug the foreground's own AABB projected to view space, and the front face is the foreground's near depth — the box therefore reaches PAST where the geometry ends, out to the cutoff, showing the headroom. This is deliberate: an AABB that merely hugged the geometry would look identical bounded or not, telling you nothing about the *bound*; pinning to the cutoff is what makes the overlay show the parameter (`extrude_multiplier`) the artist is tuning. A brighter red `PlaneGeometry` at `z = −cutoff` highlights the clip boundary itself (everything in front = foreground relief, everything behind = the pushed-back sky card), with a camera-facing distance label (`cutoff X.X m`) at its top edge — a self-contained canvas-texture `Sprite` (`makeBandLabel`, no CSS2D/font loader), `depthTest:false` so it's always legible. Fill opacity 0.13 (`depthWrite:false`), red `EdgesGeometry` cage + cutoff plane `depthTest:false` with `renderOrder` above the primary's 100000 so they read clearly over the projected photo. Falls back to the plain world AABB if the payload lacks a view matrix. Session-only display state (default off), rebuilt each execution (after `buildPatchSources` in `setProxies`, since the patch groups are fresh objects per run) and disposed on toggle-off. Requires the backend to serialize the band metrics: `_extract_blockout_camera` now adds `near_m`/`far_m`/`band_geometry` to each `projection_sources` entry (from the source's metadata — previously dropped), and `buildPatchSources` tags each group's `userData` with them. Verified live on the card-split: fg `far_m=9.66m`, the box back face lands at view-z −9.66 (past the geometry's −7.66 end), red coverage ~16% of frame, toggles on/off cleanly.



**Viewport diagnostics — exposure, VP/horizon/ground diagram, camera HUD (added 2026-07-02):** three toolbar additions, all pure frontend (no new node widgets, payload-only backend change).

- **☀ Exposure** — `renderer.toneMapping = THREE.ACESFilmicToneMapping` + a slider controlling `renderer.toneMappingExposure` (0.1–3, default 1). Only ever affects the LIT grey (`MeshStandardMaterial`) preview: the projection `ShaderMaterial` writes `gl_FragColor` directly with no tone-mapping GLSL chunk (immune by construction), and `renderAllPasses`' normal/mask override materials are explicitly `toneMapped:false` so the exposure slider can never corrupt those deterministic passes (the custom depth shader is likewise immune — no tonemapping chunk).

- **📊 Diagram** — layered SVG overlay (absolutely positioned over the canvas, `pointer-events:none` so it never blocks orbit dragging) with 3 independently-opacity-dimmable layers: VP fan-lines (image-corner-to-VP-position lines + labeled marker per vanishing point, colored orange/blue/green for left/right/vertical matching `AtlasVPVisualization`'s PIL scheme), horizon (line + confidence label), ground (shaded rect below the horizon split). viewBox uses the solve's native image pixel dimensions, so VP/horizon positions from the payload need no rescaling. **VPs are empty on the learned (GeoCalib) solve path** — it predicts focal+gravity directly, never via classical vanishing points — the layer only populates on the `detect_vanishing_points=True` VP path; horizon/ground work on both. Off-canvas VPs (common — e.g. `(-2297,-392)` on a real test image) are simply clipped by the SVG's default `overflow:hidden`, leaving the converging fan-lines visible at the frame edge.

- **ℹ Info** — HUD panel (top-left, monospace, semi-transparent) showing solved lens (focal mm + FOV°), sensor mm, camera height m, scene depth m (from the backdrop's `distance_m` if derived), confidence %, source method, scale tier — each line only rendered when its value is non-null.

- Backend: `_extract_blockout_camera` gained `vanishing_points` (list of `{position_px, direction_label, confidence}`), `horizon_line` (`{endpoints_px, line_coefficients, confidence}`), and `camera_meta` (`{confidence, source_method, scale_source, focal_mm, sensor_mm, fov_h_deg, camera_height_m, scene_depth_m}`) — all pulled from data already on the solve, no new computation beyond `fov_h_deg` and reading the backdrop primitive's `distance_m`.



**💡 Movable point lights (added 2026-07-06; 3rd light + 0–10 range 2026-07-14):** three `THREE.PointLight`s (was two), added alongside (never replacing) the fixed `HemisphereLight` + key `DirectionalLight`, each with a numeric X/Y/Z position + intensity slider (0–10, doubled from 0–5) + color picker in a toggleable toolbar panel (mirrors the 🎥 Camera Path panel's show/hide pattern) — mounted into the `AtlasViewportControls` Output Desk's own new "Lights" tab when one is linked, exactly like the Path panel's `_atlasPathContainer`. Pure frontend/session-only state (no new node widgets, resets on reload), same category as ☀ Exposure/📊 Diagram/ℹ Info above. Both default to **intensity 0**, so no existing workflow's look changes until an artist explicitly raises one. **On each execution the (unmoved) lights are auto-placed near the recovered geometry, scaled to the scene** (`placeDefaultLights`: pivot = the Box3 centre of all `atlasPatch`/`atlasDerived` meshes — computed AFTER `buildPatchSources` since patch/clean-plate geometry is where band scenes live and `computeGeometryPivot` excludes it — with each light in front of + above the pivot at ~0.36× the camera→pivot distance). The fixed near-origin defaults sat ~scene-depth away at a large `AtlasScaleOverride` (geometry 100 m+), so raising a light did nothing (user-reported "can't see the lights"); this + the scale-aware falloff give a clear relight at any scale (verified: at ×10 a light auto-lands ~55 m from the 150 m geometry → intensity 3 gives max 135/255 change). Editing a light's X/Y/Z **pins** it (`userData.atlasMoved`) so manual placement is never overridden. Real lights, so they light the grey/shaded `MeshStandardMaterial` preview and the "shaded" render pass automatically — no extra wiring needed there. They also drive a **stylized, opt-in "relight" multiply term** in `PROJECTION_FRAGMENT_SHADER` (`atlasRelightTerm`: per-light `NdotL × 1/(1+0.05·(dist/uSceneScale)²) × color × intensity` — the falloff is scale-aware, see the detail-relight note below — summed onto a `vec3(1.0)` base and multiplied against the sampled texture color) using the `vWorldPos`/`vWorldNormal` varyings the facing-ratio discard already computes — **not** physically-correct delighting (the source photo already carries its own real-world lighting; this is a by-eye dodge-and-burn bias), and a deliberate exception to the "projection shader is immune to lighting by construction" rule that ☀ Exposure still follows exactly (exposure/tonemapping remains completely separate from this term). With all lights at intensity 0 the term is an exact no-op (`relight == vec3(1.0)`), so 📽 Project output for every workflow saved before this feature is pixel-identical. Because projection materials are frequently rebuilt (every execution, every patch/clean-plate `ProjectionSource`) rather than mutated, uniforms are pushed by a per-frame `syncProjectionLightUniforms()` (in the `animate()` loop, before `renderer.render()`) that `scene.traverse()`s for any `ShaderMaterial` carrying `uLight1Pos` and copies the lights' live position/color/intensity into it (an array-driven `movableLights.forEach` over `uLight${n}*`, so adding a light is just a longer array + the matching shader uniforms + relight term) — this guarantees every current AND future projection material (primary, patches, clean-plate layers) stays in sync without each call site needing to know about lights. The traverse is skipped whenever all lights have always been off (a `_lightsWereActive` flag forces exactly one final sync on the on→off transition so a previously-lit material's uniforms get zeroed rather than left stale), keeping the default (unused) state at effectively zero extra per-frame cost.



**💡 Detail relight — photo-luminance bump on the projection normal (2026-07-14):** the light relight (`atlasRelightTerm`) uses the GEOMETRY normal (`vWorldNormal`), which on a matte projection is low-frequency (a flat `bg_card` is one constant normal; a relief mesh only broad undulation), so the lights sculpt broad shape but never the surface texture (brick, foliage, rock) that lives in the photo, not the coarse geometry. Depth-derived normals wouldn't help — the geometry IS built from that depth, so its normals already are the depth normals; the missing detail lives in the photo's LUMINANCE. `atlasBumpNormal(N, worldPos, uv, uBumpStrength)` perturbs the relight normal from the sampled photo's luminance as a heightfield: the height gradient is sampled in TEXEL space (`uBumpScale/uImageSize` offsets, default **8 texels** — a 1-texel offset is too fine on a big plate to register any detail, since adjacent-pixel luminance is near-identical; verified live: 1 texel gave max 3/255 normal change, 8–12 texels give ~24/255; zoom-stable, so the detail scale doesn't change as you orbit/dolly, unlike raw `dFdx(luma)`), then mapped tangent→world by a cotangent frame built from screen-space derivatives of `vWorldPos`+`uv` (Schüler's "normal mapping without precomputed tangents" — no tangent attribute needed; WebGL2 gives `dFdx/dFdy` in core, verified compiling on the live context). Only the two `atlasRelightTerm` calls read the perturbed `N`; the facing-ratio discard keeps the true `vWorldNormal` (it's a geometry test, not lighting). `uBumpStrength` default 0 = exact no-op (the geometry normal), so backward-compatible; it + `uBumpScale` are live-synced by `syncProjectionLightUniforms()` alongside the light uniforms (same "materials are rebuilt, push per-frame" reason) and driven by **"Detail" (strength 0–6) + "Scale" (offset 1–32) sliders** in the 💡 Lights panel. Needs a light raised above 0 to show. The relight falloff is **scale-aware** (`uSceneScale` = the recovered camera height / 1.6 m default eye height, applied as `dist/uSceneScale` inside the attenuation), so a light placed proportionally to the scene gives the same relight at any `AtlasScaleOverride` — verified live: at ×10 scale (geometry ~150 m, `uSceneScale=10`) a light 31 m away reads max 53/255 (was ~0 before), and `uSceneScale=1` at the default height reproduces the original `1/(1+0.05·dist²)` exactly, so existing ~1.6 m-camera looks are unchanged. Computed once per material build from `data.camera_position[1]` (rebuilt each execution, so it tracks scale changes; no live-sync needed). Relight-only: never touches the base texture, the matte, or the `renderAllPasses` `MeshNormalMaterial` AOV (that pass overrides materials entirely). **Real predicted-normal relight (MoGe, 2026-07-14):** the follow-on is wired — a REAL per-pixel normal map (MoGe `*-normal` variants) drives the relight instead of only the luminance bump. Pipeline: `_estimate_depth_moge` now captures the predicted normals into `DepthResult.normal` (HxWx3, the MODEL's camera frame — previously only a `has_predicted_normals` flag). `core/normals.py` aligns them to the recovered WORLD frame — MoGe's frame ≠ Atlas's GeoCalib frame (solved gravity/roll), so `align_predicted_normals_to_world` recovers the single rigid rotation by **orthogonal Procrustes** (`procrustes_rotation`, det +1) against the geometry's own gradient normals, trying both sign conventions to survive a global hemisphere flip (a rotation+flip is a reflection Procrustes-proper can't undo) — then `encode_normal_map_b64` writes `(n+1)/2` RGB. `AtlasCleanPlateLayer` computes + embeds it as `ProjectionSource.normal_map_b64` (skipped when frame-outpainted — the map would be out of uv-registration; `pad==0` guard); `_extract_blockout_camera` serializes it; `makeProjectionMaterial` loads it (`NoColorSpace`, `flipY:false`) into `uNormalMap`/`uHasNormalMap`, and the shader uses it as the relight `N` (image-resolution surface orientation, cleaner than the coarse mesh normal) with the luminance bump still adding micro-detail on top. `uHasNormalMap=0` without a normal map (every non-MoGe run) → exact geometry-normal behavior, backward-compatible. Math unit-tested (`tests/test_normals.py` — rotation recovery, flip robustness, encode round-trip) and the node attach (`test_inpaint_layers_nodes.py`); shader verified compiling live. **End-to-end verified live (2026-07-14):** with the layer's `AtlasDepthMap` on `moge-2-vitl-normal`, `fg_buildings` embedded a ~13 MB aligned world-normal map and the shader reported `uHasNormalMap=1` (a frame-outpainted layer like `bg_card` correctly skips on the `pad==0` guard). **Decoupled from the depth source (`AtlasMogeNormals`, 2026-07-14):** because the attach reads `depth.normal` (the same field MoGe-as-depth populates), the relight was initially only available when MoGe was ALSO the depth/geometry model — a problem on exteriors, where MoGe's far-field runs away and V2-Outdoor is the right depth choice. `AtlasMogeNormals` (catalog above) sits between `AtlasDepthMap` and `AtlasCleanPlateLayer`, runs a MoGe `*-normal` pass PURELY for normals, and attaches them (resized to the input depth's resolution) onto a copy of the depth map — so you keep V2/DA3 depth AND get MoGe normals. Verified live: V2-Outdoor depth + `AtlasMogeNormals` → `fg_buildings` still got the 13 MB normal map (`tests/test_moge_normals_node.py` pins the resize/renormalize/non-mutation/pass-through).



**Projection shader was missing its linear→sRGB output encode (fixed 2026-07-06):** discovered while investigating why 📽 Project looked artificially dark/desaturated relative to the source photo — the same symptom that motivated adding movable lights above, but a distinct and more fundamental bug underneath it. `renderer.outputColorSpace = SRGBColorSpace` and every source texture is tagged `colorSpace = SRGBColorSpace`, so `texture2D(uTexture, uv)` in `PROJECTION_FRAGMENT_SHADER` already returns correctly-**decoded linear** values (the GPU auto-decodes an sRGB-tagged texture's bytes to linear float on sample, transparently, regardless of which material/shader does the sampling). But `PROJECTION_FRAGMENT_SHADER` is a raw `THREE.ShaderMaterial` — Three.js only auto-appends its own linear→display-colorspace re-encode chunk (`colorspace_fragment`) into *built-in* material shader templates (`MeshStandardMaterial`, `MeshBasicMaterial`, etc.); a fully custom `ShaderMaterial` never gets it unless the shader author writes the equivalent GLSL by hand. So the shader was writing correctly-decoded linear values straight to `gl_FragColor` with no re-encode — every projected pixel silently skipped the gamma step the display expects, reading as darker/flatter than the source photo (independent of, and in addition to, whatever the ☀ Exposure/💡 Lights sliders were set to). Fixed by adding `atlasLinearToSRGB()` (matches THREE.ShaderChunk's own `LinearTosRGB` formula, exponent `0.41666 ≈ 1/2.4`, verified numerically against the exact sRGB OETF) and applying it to `col.rgb * relight` (clamped to `[0,1]` first, since the encode's `pow()` is undefined for negative/NaN-prone inputs and the relight term can push values above 1) right before the final `gl_FragColor` assignment. This is orthogonal to the ☀ Exposure bullet above, which is still accurate as written — exposure/tonemapping remain genuinely absent from this shader; only the separate color-*space* encode step was missing. Verified live: captured the exact same frame with and without the encode (temporarily patching `material.fragmentShader` + `needsUpdate=true` to compare) — the un-encoded version was visibly crushed/muddy, the encoded version matched the source photo's contrast and saturation.



**Orbit coverage (why the mesh/projection can look like it "disappears" on rotate):** derived geometry only ever covers what the recovered camera could see — a forward-facing cone, inherent to single-photo reconstruction. `createOrbitControls` clamps yaw/pitch to ±80°/±55° around the recovered camera's own direction (`theta0`/`phi0`, wraparound-safe via `atan2(sin,cos)`, re-anchored on every `syncFromCamera()` — i.e. every execution and every 📷 Camera View click) so you can't orbit past the reconstructed cone into empty space in the grey/undressed preview.



**Orbit pivot — geometry centroid, not a ground heuristic:** `applyRecoveredView` sets an initial pivot via `groundPointInView` (where the camera's forward ray crosses Y=0, capped at the solved scene depth) purely as a fallback for before any geometry exists (or for workflows that never run `AtlasDeriveProjectionGeometry` at all) — it's a generic heuristic, not the actual centre of whatever gets reconstructed, and could be visibly off for reconstructions that aren't roughly centred on the camera's exact forward ray. Once `buildDerivedProxies` has real geometry, `setProxies` immediately overrides it with `computeGeometryPivot(data)` — since 2026-07-09 the **recovered camera's central view ray at the MEDIAN sampled vertex depth** of the `atlas_derived_proxies` meshes (≤800 samples/mesh, recovered pose from the payload's `view_matrix` so the pivot never depends on the user's current orbit), still excluding `"projection_backdrop"`; patch/clean-plate sources live in their own `atlas_patch_N` groups and are never included. It was previously a `THREE.Box3` bounding-box CENTER, which is `(min+max)/2` and therefore tail-dominated — fine for clustered fitted primitives, but on a single full-scene relief mesh (the hidden-geometry workflows' base geometry, spanning near-foreground to far-clip plus fill/outpaint skirts) it parked the pivot deep behind the subject (artist-reported). Median vertex depth ≈ the depth of the middle of the visible surface area, since relief grids sample the image uniformly. This only calls `controls.setTarget(...)` + `syncFromCamera()` (re-deriving the orbit sphere from wherever the camera *already* is) — it never moves the camera itself, so recomputing it on every execution (including re-executions from � Bake Path) can't disrupt an in-progress inspection, and is a no-op in effect when the geometry hasn't changed between runs.



**🎯 Manual orbit-pivot offset (toolbar, 2026-07-14):** a `🎯 Pivot` toggle opens a panel with X/Y/Z world-metre inputs + Reset that ADD a manual offset on top of whichever base pivot the auto-logic picked — motivated by `AtlasScaleOverride`: once the scale dial pushes geometry out to 100m+, the auto centroid often isn't the point you want to orbit around. Implemented as a `pivotOffset` `Vector3` applied through a single `targetWithOffset(base)` helper at BOTH `setTarget` sites (the `setProxies` geometry pivot AND `applyRecoveredView`'s ground-point fallback — so it works even on band-only scenes where `computeGeometryPivot` returns null), plus a live `applyPivotOffset()` (re-target `pivotBase + offset` + `syncFromCamera`) on every input change. The input **step auto-scales with the scene** (`lastSceneRadius/40`, radius captured in `placeDefaultLights` from the same geometry Box3 it already computes) so a nudge stays proportional at any scale. Default `(0,0,0)` = the auto pivot exactly (backward-compatible — nothing moves until dialled); session-only frontend state like every other viewport control; the panel mounts beside the 💡 Lights panel (same `lightTarget`, so it follows into the detached Output Desk). Verified live: `Y+5` moved `controls.target` by exactly `+5`, Reset restored the auto pivot, step scaled to the scene. The camera itself never moves (only the orbit target), so an offset can't disrupt an in-progress inspection. **On-screen gizmo:** opening the 🎯 panel shows a marker at the orbit target (`pivotGizmo` — a small sphere + short RGB axis lines, `depthTest:false` / high `renderOrder` so it reads through geometry), so the artist can SEE the pivot while nudging it. `updatePivotGizmo()` (per-frame in `animate()`, AND called immediately from `pivotBtn.onclick` + `applyPivotOffset` so it never waits a frame) positions it at `controls.target` and sizes it by the **camera→pivot distance** (`dist·0.02`, NOT the scene bounds — those include the far backdrop card and would balloon the marker), giving a constant small on-screen size at any scale. Visible only while the panel is open; NOT tagged `atlasDerived`/`atlasPatch` so it never enters the pivot/light-placement/band-box/projection logic; stashed-and-hidden during the deterministic export/Safe-Zone passes AND the � Bake capture so it never leaks into AOVs or baked video. Verified live: marker sits exactly on the target, tracks a nudge immediately, renders as a compact ~5% blob (was filling the frame before the distance-based sizing fix), hidden on close.



`AtlasBlockoutViewport`'s `preview_expand` widget (default 1.0 = off) can optionally dilate the geometry itself for more coverage within that arc, via `proxy_geometry.dilate_proxy_geometry_for_preview()`: for any point with local normal n̂, radiating from the camera position, `p' = pivot + ((p-pivot)·n̂)n̂ + scale·[(p-pivot) - ((p-pivot)·n̂)n̂]` — only the normal-perpendicular offset scales, so a plane widens without drifting in depth, a box/cylinder (no single normal) dilates uniformly, and the relief mesh dilates per-vertex using each vertex's own (genuinely arbitrary) normal. Applied ONLY at `serialize_proxy_geometry()` time (`_extract_blockout_camera` passes `preview_expand`/`preview_pivot=camera_position`) — never mutates the `AtlasProxyPrimitive` objects on the solve, so DCC exports and metric measurements stay accurate regardless of the viewport setting.



**Known interaction, not further fixable without dual geometry (dilation vs. 📽 Project):** dilation and the projection shader are fundamentally in tension, and this is *not* the same bug the clamp fixes. The 📽 Project shader assigns texels by casting each fragment's world position through the RECOVERED (undilated) camera and discarding anything that lands outside the actual photographed frame (`atlas_blockout.js`'s `PROJECTION_FRAGMENT_SHADER`, `discard` on out-of-bounds/behind-camera). Dilated geometry is, by construction, surface area the camera never actually photographed — there is no real pixel data for it, so it correctly discards to empty/black. Because dilation is most visible away from dead-center, this shows up as large black regions after only a *moderate* orbit with Project active — confirmed via live reproduction: with the prior default (`preview_expand=1.4`), a 60px drag left roughly half the frame black; the identical drag at `preview_expand=1.0` showed full coverage (parallax-shifted, as expected). This is why the default was changed to 1.0 — raise it only when inspecting undressed blockout primitives, never while relying on 📽 Project.



(The OBJ scale-proxy meshes, their `/atlas/proxy_model` route, and `examples/models/` were all removed by 2026-07-12 — the viewport buttons that used them went 2026-07-09, and the public release ships without the sample assets; scale checks are the tiered cascade + ℹ Info HUD.)



On each `node.onExecuted`: fetches `GET /atlas/camera_data/{node_id}` to receive the recovered camera dict (`view_matrix`, `fx`, `fy`, `source_image_b64`) and applies it to the Three.js camera. The source image is loaded as a background plane from `cameraData.source_image_b64`.



`Render Passes` encodes four WebGL passes (shaded, depth, normal, mask) as base64 PNG, writes them as JSON into the `client_data` STRING widget, then calls `app.queuePrompt(0, 1)` to send them back to Python.



### API endpoint



```

GET /atlas/camera_data/{node_id}

→ { view_matrix, fx, fy, cx, cy, camera_position,

    image_width, image_height, target_width, target_height,

    focal_mm, sensor_mm, source_image_b64 }

```



Cache stored in `_ATLAS_BLOCKOUT_CACHE` (module-level dict, max 64 entries, LRU-eviction).



The route is registered in `comfy/__init__.py` behind the double-import guard (checks its path is not already registered).



### Example workflows



**Shipping catalog: SIX workflows — three artist-facing bases and three dedicated agentic assessment variants.** `tests/test_example_workflows.py` pins these names so deletion or an unreviewed addition fails loudly. Each `*_agentic_assessment_workflow.json` preserves its base graph and appends one enabled `AtlasAssessOutput` plus a PreviewImage wired to the assessor's exact `assessed_image`, with a distinct stable report path; keeping this terminal off the base avoids spending a VLM call during ordinary interactive queues. The three bases use ComfyUI's bundled `example.png`. The agentic input/staged twins use the normal Atlas `ghosttown.jpg` plate and the occlusion twin uses the normal Space Hangar `moge_hangar_proj.jpg` plate, all already present in the portable input examples. The three bases are `examples/atlas_input_quickstart_workflow.json`, `examples/atlas_camera_staged_master_workflow.json`, and `examples/atlas_occlusion_cull_quickstart_workflow.json`; their agentic twins use the same basename plus `_agentic_assessment`. `tools/smoke_agentic_assessment_workflows.py --validate-only` checks all live schemas, while the command without that flag queues all three agentic variants and requires one structured terminal report plus retained, hashed evidence. Pure HTTP runs cannot bake the browser/WebGL viewport pass, so the node constructs canonical recovered-camera evidence from the solve; it retains the exact output PNG, union-coverage matte, and source reference. Orbit/grazing occlusion remains visually inconclusive without a browser/DCC render. `source_image_fallback` is used only when reconstruction is impossible and cannot earn a projection-quality pass.



**The OCIO/ACEScg + camera-RAW demos are NO LONGER in the repo** (removed in the 0.8.1 trim, along with all of `examples/showcase/`, `examples/experimental/`, and `examples/retopo/`): they require a float plate / camera RAW that is not shipped, so they are distributed as workflow+image bundles from the project website instead. `tests/test_shipping_workflow_paths.py` forbids absolute machine paths, and `tests/test_shipping_workflow_experimental.py` pins the experimental-using set to empty. The OCIO handoff itself still uses `AtlasLoadPlate` 🎞 (Atlas's own OpenImageIO reader — loads the ACEScg `.exr`, converts, emits the `plate_ref` in one node, no ComfyUI-OCIO/opencv-EXR dependency) → `AtlasInput` → `AtlasAttachSourcePlate` → the DCC exporters; that recipe lives in the website bundle. The older removed workflow generators and assets remain recoverable from git history before the 0.8.1 trim (`10e600b` for the 2026-07-12 batch).



- `examples/atlas_input_quickstart_workflow.json` — **the fastest path** (LoadImage → 🎬 AtlasInput → Atlas Viewport + Output Desk + current shipping outputs): instant relief by default, with layers/VLM/native-SAM sky/scope/legacy-fast-inpaint options reachable on the one node. At `layers=0` it writes Solve JSON, USD camera, a Blender script, textured Relief OBJ/GLB, and relief-based Nuke/Maya scenes; the separate Nuke/Maya layer packages activate at `layers>=1`. Every color-aware handoff receives Output Desk OCIO metadata. Its controlled A/B twin, `atlas_occlusion_cull_quickstart_workflow.json`, has the same graph/defaults/exports plus exactly one functional wire — `AtlasInput.depth → AtlasBlockoutViewport.primary_depth` — so Project + ✂ Occlude is tested against the metric depth that made the mesh, never a mismatched retopo/depth pass. Both v2 graphs use distinct relative output roots, disable the outdoor-only sky heuristic for a reliable first queue on bundled `example.png`, and explain that the staged master is the native-SDXL production tier.

- `examples/atlas_camera_staged_master_workflow.json` — � **the five-layer staged master.** Historical v1–v10 details are preserved in git; **v11 (2026-07-21)** replaces the 154-node LaMa/KJ/rgthree graph with five official ComfyUI subgraphs and 31 top-level nodes. Sky contains native `AtlasSAM3Mask → AtlasSkyDomeLayer`; far/bg/mid/fg each contain `AtlasSAM3Mask → AtlasScopeMask → AtlasDepthLayerMask → AtlasInpaintCrop → AtlasSDXLInpaint → AtlasInpaintStitch → AtlasCleanPlateLayer`. The SDXL base checkpoint runs at native 1024 long-edge, 32 DPM++ 2M Karras steps, fixed per-layer seeds, perspective preservation, and a masked 12px stitch feather. VLM prompts/geometry/watertight band overrides wire directly into each instance; no Set/Get rails, shared LaMa/upscaler, or group bypasser remain. The top level keeps the closed solve gate, preview/master viewports, five cutout previews, debug JSON, and SolveJSON/Nuke/Maya/Blender/USD exports. Atlas MCP now expands official v1 subgraphs before live validation/headless execution. Live-verified on portable ComfyUI 0.27.0: closed-gate queue 19s; full open-gate queue 36s; five projection sources; zero execution errors.

- `examples/*_agentic_assessment_workflow.json` — the corresponding automation versions. The terminal receives the final viewport `shaded` image, solve, source image, and depth; the staged version also receives `AtlasDebugReport.report`. Each enables LM Studio assessment by default and writes to its own `atlas_debug/*_agentic_output_assessment.json` path. Live-smoked together on 2026-07-21 against Ghost Town/Space Hangar: all three validated with zero schema warnings, returned complete VLM reports with `headless_projection_reconstruction` provenance, retained output/matte/reference files with verified hashes, and correctly failed the visible defects. Measured union holes were 5.13% (Ghost Town quickstart) and 5.62% (Hangar); staged Ghost Town had full coverage but severe unapproved source drift (luma correlation 0.601, edge correlation 0.249, RGB MAE 0.149, changed fraction 0.324), which deterministically failed `layer_inpaint`. The canonical pass still leaves orbit/grazing occlusion visually inconclusive.



