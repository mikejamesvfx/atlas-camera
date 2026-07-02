# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Install for development (includes numpy, opencv, pytest)
pip install -e ".[dev]"

# Run all tests
python -m pytest -q

# Run a single test file
python -m pytest tests/test_intrinsics.py -v

# Run a specific test
python -m pytest tests/test_intrinsics.py::test_focal_length_to_pixels -v

# Install with UI and vision support
pip install -e ".[ui,vision]"

# Start the FastAPI backend
python -m atlas_camera.ui

# Start the React frontend (separate terminal)
cd ui && npm install && npm run dev

# Install atlas_camera into ComfyUI's venv (editable — run once)
& "C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe" -m pip install -e .

# Verify ComfyUI can import the package
& "C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe" -c "import atlas_camera; print(atlas_camera.__file__)"
```

## Optional dependency groups

- `dev` — numpy, opencv-python, pytest
- `vision` — numpy, opencv-python (runtime vision)
- `image` — Pillow only
- `usd` — usd-core
- `ui` — FastAPI, uvicorn, Pillow, python-multipart
- `neural` — torch + GeoCalib (learned single-image camera prior). GeoCalib is GitHub-only: `pip install "git+https://github.com/cvg/GeoCalib.git"`. torch is expected from the host env (e.g. ComfyUI's venv).

The core package has **zero required runtime dependencies**. All vision, USD, and UI imports are guarded with informative `ImportError` messages.

## Architecture

```
atlas_camera.core       ← DCC-agnostic schema, solver, math (no host deps)
atlas_camera.exporters  ← Maya, Blender, Nuke, USD, review package writers
atlas_camera.importers  ← Atlas JSON and USD camera loaders
atlas_camera.comfy      ← ComfyUI node library (25 nodes, no hard Comfy dep)
atlas_camera.ui         ← Optional FastAPI project service
atlas_camera.reference_data ← Curated scale-reference registry (JSON)
atlas_camera.gaussian   ← Future 3DGS / point-cloud interfaces (placeholder)
atlas_camera.inference  ← Optional local multimodal provider helpers
ui/                     ← React/Vite workbench (Three.js 3D viewport)
examples/               ← Example ComfyUI workflows and test images
```

The public API is `import atlas` (thin facade in `atlas_camera/__init__.py`). The stable package name is `atlas_camera`.

## ComfyUI integration (`atlas_camera/comfy/`)

### Setup

A symlink connects the node pack to ComfyUI's custom_nodes directory:

```
C:\Users\miike\ComfyUI_V91\ComfyUI\custom_nodes\AtlasCamera
    → C:\Users\miike\Desktop\AtlasCamera_Claude\atlas_camera\comfy
```

The package is installed in editable mode in ComfyUI's venv so `import atlas_camera` resolves to this project directory. Changes to Python source are live immediately — no reinstall needed.

### Double-import guard (critical)

`atlas_camera/comfy/__init__.py` is loaded twice at ComfyUI startup:
1. As `AtlasCamera` custom node (by ComfyUI's node loader)
2. As `atlas_camera.comfy` package (by `from atlas_camera.comfy.nodes import ...` inside the same file)

Both loads hit the same file, causing the aiohttp route `GET /atlas/camera_data/{node_id}` to be registered twice, which raises `RuntimeError: method HEAD is already registered`. The fix in `__init__.py` checks `if not any(r.path == ... for r in _routes)` before registering.

### Node catalog (25 nodes)

**Category: Atlas Camera**

| Node class | Inputs | Outputs | Notes |
|---|---|---|---|
| `AtlasLoadImageSolveCamera` | image_path, image_width, image_height | ATLAS_SOLVE | Legacy: file-path-based solve |
| `AtlasSolveFromImage` | image (IMAGE), ±focal_mm, ±sensor_mm, ±detect_vanishing_points | ATLAS_SOLVE | Geometric VP solve; accepts ComfyUI tensor. VP detection defaults ON |
| `AtlasLearnedSolveFromImage` | image (IMAGE), ±height_mode, ±camera_height_m, ±depth_model, ±sensor_mm, ±weights, ±device | ATLAS_SOLVE | Learned GeoCalib prior (focal+gravity). `height_mode=measure_from_depth` fits the ground plane to measure camera height (no assumed eye height) and fills the depth slot. Needs `[neural]` |
| `AtlasDepthAnything` | image (IMAGE), ±depth_model, ±device | depth_image (IMAGE) | Monocular depth (Depth Anything V2), metric or relative. Needs `[neural]` |
| `AtlasReferenceScaleSolve` | ATLAS_SOLVE, reference_id, bbox_x0/y0/x1/y1, ±height_override_m | solve (ATLAS_SOLVE), camera_height_m | Sets metric camera height from a known-size object (single-view geometry). Composable after any solve. No extra deps |
| `AtlasVLMScaleCues` | image (IMAGE), ±provider, ±model, ±base_url, ±min_confidence | scale_references (STRING json), summary (STRING) | Detects known-size objects with a local VLM → scale_references JSON. Needs a running LM Studio/Ollama/llama.cpp server; fails soft to `[]` |
| `AtlasApplyScaleReferences` | ATLAS_SOLVE, scale_references (STRING), ±confirm, ±min_confidence | solve (ATLAS_SOLVE), camera_height_m, report (STRING) | Rescales metric height from scale_references — **only when `confirm` is on** (else records candidates). No auto-promotion |
| `AtlasDeriveProjectionGeometry` | ATLAS_SOLVE, image (IMAGE), ±depth_model, ±max_walls, ±max_objects, ±device, ±geometry_mode, ±relief_grid, ±primitive_method | ATLAS_SOLVE | Derives the depth **relief mesh** (default) and/or fitted primitives (`geometry_mode`: relief_mesh/primitives/both — "both" overlaps the two: enclosure + z-shimmer) into `projection_scene.proxy_geometry` (deep-copied solve). `primitive_method` selects the "primitives" strategy: `azimuth_walls` (default, vertical walls only), `ransac_planes` (any-orientation planes via sequential RANSAC seeded by a 2D orientation histogram — exteriors/roofs/stepped facades), `room_cuboid` (Manhattan floor+≤4 walls+optional ceiling — orthogonal interiors; artist-selected, never auto-picked). Mesh entries carry flat vertices/faces/uvs → viewport BufferGeometry. Feeds the blockout's 📽 Project mode. Note: solve JSON exports grow (~1MB) when the mesh is attached. Needs `[neural]` |
| `AtlasConstrainedSolve` | image, constraints_json | ATLAS_SOLVE | Artist-guided; pass scale_constraints for cam height |
| `AtlasLoadSolveJSON` | json_path | ATLAS_SOLVE | Load previously saved solve |
| `AtlasDecomposeSolve` | ATLAS_SOLVE | camera, confidence, source_method, image_width, image_height, solve_json, horizon_angle_deg | horizon_angle_deg comes from debug_metadata |
| `AtlasDecomposeCamera` | ATLAS_CAMERA | fx, fy, cx, cy, cam_x, cam_y, cam_z, focal_mm, fov_h_deg | cam_y must be > 0 for depth map |
| `AtlasGroundDepthMap` | ATLAS_SOLVE, image_width, image_height, near_m, far_m | depth_image (IMAGE), ground_mask (MASK) | Black if cam_y ≤ 0. width/height **0 = auto** (adopt source image size) |
| `AtlasGroundMask` | ATLAS_SOLVE, image_width, image_height | MASK | 1 = ground, 0 = sky. width/height **0 = auto** |
| `AtlasHorizonMask` | ATLAS_SOLVE, image_width, image_height, feather_px | MASK | 1 = above horizon (sky). width/height **0 = auto** |
| `AtlasVPVisualization` | image, ATLAS_SOLVE, ±show_horizon, ±show_vp_lines, ±line_opacity | IMAGE | Pass-through if no VPs detected |
| `AtlasBlockoutViewport` | ATLAS_SOLVE, source_image, resolution, client_data, ±preview_expand | shaded, depth, normal, mask (all IMAGE) | OUTPUT_NODE; browser-side Three.js. `resolution` = long edge; W×H auto-follows source image aspect. `preview_expand` (default 1.4) dilates derived geometry outward from the camera for wider orbit coverage — display only, never affects exports/measurement. Render Passes button populates client_data |
| `AtlasExportSolveJSON` | ATLAS_SOLVE, file_path | STRING | Writes JSON file |
| `AtlasExportBlender` | ATLAS_SOLVE, output_dir | STRING (script_path) | Writes build_scene.py |
| `AtlasExportNuke` | ATLAS_SOLVE, output_dir | STRING (script_path) | Writes Nuke projection script |
| `AtlasExportReliefMesh` | ATLAS_SOLVE, image (IMAGE), output_dir, ±grid_long_edge, ±depth_edge_rel, ±depth_model, ±device, ±format | obj_path, glb_path (STRING) | Depth relief mesh → OBJ+MTL+texture and/or GLB (single binary, texture embedded, KHR_materials_unlit). Projection baked into UVs (imports textured into Maya/Nuke/ZBrush/Blender); torn at silhouettes; ground on Y=0; below-ground outliers clamped along the view ray. Needs `[neural]` |
| `AtlasExportUSD` | ATLAS_SOLVE, output_dir | STRING (usd_path) | Writes camera.usda |
| `AtlasExportReviewPackage` | ATLAS_SOLVE, output_dir | STRING | Full review bundle |
| `AtlasExportMayaReviewScene` | ATLAS_SOLVE, output_dir | STRING | Maya scene + image card |
| `AtlasUSDCameraLoader` | usd_path | ATLAS_CAMERA | Load camera from USD |

### Frontend extension (`atlas_camera/comfy/web/atlas_blockout.js`)

Registers as `AtlasCamera.Blockout` ComfyUI extension targeting `AtlasBlockoutViewport` nodes. On node creation it builds a Three.js canvas with a **self-contained orbit controller** (`createOrbitControls` — the examples/jsm `OrbitControls` uses a bare `import ... from "three"` that browsers can't resolve without an import map, so it never loaded; the custom one depends only on the already-loaded THREE), a primitive toolbar (Box/Plane/Cylinder/Person/Clear), scale-reference proxy buttons (🧍 Woman / 🚗 Sedan), a 📷 Camera View reset, and a Render Passes button.

The viewport **inherits the recovered camera**: `applyRecoveredView()` sets the Three.js camera to the recovered pose/fov, then initialises the orbit controller *from* it (`syncFromCamera`, pivot = the looked-at ground point) so the default view matches the source photo; dragging orbits, and 📷 Camera View snaps back. For exact photo alignment set the node's `width`/`height` to the source image's aspect ratio (the camera aspect = `target_width/target_height`). The background photo is a plane sized to fill the recovered frustum along the view axis.

**Camera projection (📽 Project, matte-painting mode):** the payload's `proxy_geometry` entries (from `AtlasDeriveProjectionGeometry`) are built as meshes (`buildDerivedProxies`, group `atlas_derived_proxies`, transforms fed verbatim to `Matrix4.set` — row-major both sides). `makeProjectionMaterial` is the GLSL port of `ui/src/ProjectionMaterial.ts` (world pos → recovered-camera pixel → sample source photo; discard behind camera / outside frame) with `depthWrite:true` (multi-proxy occlusion; the ui original was a single-ground overlay) and a **flipY:false** texture (top-left UV origin — never share the background texture, which uses default flipY). The 📽 toggle swaps ALL projectable meshes (derived + user primitives + OBJ proxies) between grey and the shared projection material; `_prevMaterial` is stashed only once so material rebuilds don't lose the original. Clear leaves the derived group alone (Python-owned; regenerates each execution). **Matte-painting property:** texels are assigned by ray, so geometry at slightly-wrong depth still receives exactly the pixels its silhouette subtends — perfect reassembly from Camera View; scale error only shows as parallax when orbiting.

**Viewport diagnostics — exposure, VP/horizon/ground diagram, camera HUD (added 2026-07-02):** three toolbar additions, all pure frontend (no new node widgets, payload-only backend change).
- **☀ Exposure** — `renderer.toneMapping = THREE.ACESFilmicToneMapping` + a slider controlling `renderer.toneMappingExposure` (0.1–3, default 1). Only ever affects the LIT grey (`MeshStandardMaterial`) preview: the projection `ShaderMaterial` writes `gl_FragColor` directly with no tone-mapping GLSL chunk (immune by construction), and `renderAllPasses`' normal/mask override materials are explicitly `toneMapped:false` so the exposure slider can never corrupt those deterministic passes (the custom depth shader is likewise immune — no tonemapping chunk).
- **📊 Diagram** — layered SVG overlay (absolutely positioned over the canvas, `pointer-events:none` so it never blocks orbit dragging) with 3 independently-opacity-dimmable layers: VP fan-lines (image-corner-to-VP-position lines + labeled marker per vanishing point, colored orange/blue/green for left/right/vertical matching `AtlasVPVisualization`'s PIL scheme), horizon (line + confidence label), ground (shaded rect below the horizon split). viewBox uses the solve's native image pixel dimensions, so VP/horizon positions from the payload need no rescaling. **VPs are empty on the learned (GeoCalib) solve path** — it predicts focal+gravity directly, never via classical vanishing points — the layer only populates on the `detect_vanishing_points=True` VP path; horizon/ground work on both. Off-canvas VPs (common — e.g. `(-2297,-392)` on a real test image) are simply clipped by the SVG's default `overflow:hidden`, leaving the converging fan-lines visible at the frame edge.
- **ℹ Info** — HUD panel (top-left, monospace, semi-transparent) showing solved lens (focal mm + FOV°), sensor mm, camera height m, scene depth m (from the backdrop's `distance_m` if derived), confidence %, source method, scale tier — each line only rendered when its value is non-null.
- Backend: `_extract_blockout_camera` gained `vanishing_points` (list of `{position_px, direction_label, confidence}`), `horizon_line` (`{endpoints_px, line_coefficients, confidence}`), and `camera_meta` (`{confidence, source_method, scale_source, focal_mm, sensor_mm, fov_h_deg, camera_height_m, scene_depth_m}`) — all pulled from data already on the solve, no new computation beyond `fov_h_deg` and reading the backdrop primitive's `distance_m`.

**Orbit coverage (why the mesh/projection can look like it "disappears" on rotate, and the two fixes):** derived geometry only ever covers what the recovered camera could see — a forward-facing cone, inherent to single-photo reconstruction. (1) `createOrbitControls` clamps yaw/pitch to ±80°/±55° around the recovered camera's own direction (`theta0`/`phi0`, wraparound-safe via `atan2(sin,cos)`, re-anchored on every `syncFromCamera()` — i.e. every execution and every 📷 Camera View click) so you can't orbit past the reconstructed cone into empty space. (2) `AtlasBlockoutViewport`'s `preview_expand` widget (default 1.4) dilates the geometry itself for more coverage within that arc, via `proxy_geometry.dilate_proxy_geometry_for_preview()`: for any point with local normal n̂, radiating from the camera position, `p' = pivot + ((p-pivot)·n̂)n̂ + scale·[(p-pivot) - ((p-pivot)·n̂)n̂]` — only the normal-perpendicular offset scales, so a plane widens without drifting in depth, a box/cylinder (no single normal) dilates uniformly, and the relief mesh dilates per-vertex using each vertex's own (genuinely arbitrary) normal. Applied ONLY at `serialize_proxy_geometry()` time (`_extract_blockout_camera` passes `preview_expand`/`preview_pivot=camera_position`) — never mutates the `AtlasProxyPrimitive` objects on the solve, so DCC exports and metric measurements stay accurate regardless of the viewport setting.

Proxy meshes live in `examples/models/*.obj` (authored in **centimetres**; the loader scales by 0.01 into the metric world). Loaded via Three.js `OBJLoader` from `GET /atlas/proxy_model/{name}` and dropped on the ground (Y=0) under the camera's view centre — a correctly-sized human/car is the fastest visual check that the solve + camera height are right. Loaded groups are tagged `userData.atlasProxy` so Clear removes them.

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

```
GET /atlas/proxy_model/{name}
→ serves examples/models/{name} (.obj/.mtl only, basename-sanitised) as a FileResponse
```

Both routes are registered in `comfy/__init__.py` behind the double-import guard (each checks its own path is not already registered).

### Example workflows

- `examples/atlas_camera_core_projection_workflow.json` — **simplest / concentrated on the core technology** (6 nodes): Load Image → Learned Solve (GeoCalib) → Derive Projection Geometry (relief mesh) → Blockout Viewport (hit 📽 Project to see the camera projection live) → Export Relief Mesh (OBJ/GLB for Maya/Nuke/ZBrush). No scale-cue/reference nodes, no analysis/mask nodes, no multi-format exports — just the session's core through-line: recover the camera → derive projectable geometry → project → hand off. Best starting point for understanding or demoing the technology. Needs the `[neural]` extra.
- `examples/atlas_camera_learned_workflow.json` — full-featured (26 nodes). Learned pipeline:
  ① Source Image + VLM Scale Cues → ② Learned Solve (GeoCalib) → Apply Scale References (confirm to adopt) → Derive Projection Geometry → Decompose → ③ Analysis (Depth Anything V2 · Ground Depth · Ground/Horizon Masks · VP/Horizon) → ④ Blockout Viewport (derived proxies + 🧍/🚗, 📽 Project, 4 passes) → ⑤ DCC Exports. Needs the `[neural]` extra for the solve/depth/derive nodes; the VLM node additionally needs a local VLM server (fail-soft without one).
- `examples/atlas_camera_full_workflow.json` — original 20-node vanishing-point workflow (no neural deps).

## Core schema (`atlas_camera/core/schema.py`)

- `LatentScene` / `AtlasSolve` — top-level result (aliases; both stable)
- `LatentCamera` / `AtlasCamera` — recovered camera (aliases)
- `LatentComponent` — empty slot for future depth/geometry/lighting/semantics
- `AtlasIntrinsics`, `AtlasExtrinsics`, `AtlasVanishingPoint`, `AtlasHorizon`, `AtlasProjectionScene`, `AtlasProxyPrimitive`

## Coordinate conventions

- World: **right-handed, Y-up** by default
- Image: **origin top-left, x right, y down**
- DCC conversions happen **only at adapter boundaries**, never in core
  - Blender (Z-up): converted in `atlas_camera/exporters/blender_exporter.py`
  - USD stage axis set at export time
  - OpenCV/NumPy: only imported in the `vision` optional layer

## Solve entry points

- `atlas.recover(image_path, method="vanishing_points"|"learned", ...)` → `LatentScene` — primary API
- `atlas_camera.core.solver.solve_still_image_learned(image_path, camera_height="auto"|float, depth_model=..., ...)` — learned GeoCalib prior (needs `[neural]`); robust on AI-generated images. `camera_height="auto"` measures height from depth instead of assuming it
- `atlas_camera.core.solver.estimate_ground_height_from_depth(depth, rotation=..., fx, fy, cx, cy)` — pure-numpy ground-plane fit → camera height + confidence + ground mask
- `atlas_camera.core.relief_mesh.build_relief_mesh(depth, view_matrix=..., fx, fy, cx, cy, ...)` — triangulated/decimated depth mesh, torn at silhouettes, camera projection baked into UVs; `estimate_ground_scale(...)` pins the ground to Y=0. Exported via `exporters.relief_mesh_exporter.export_relief_mesh()` (OBJ+MTL+texture) or `export_relief_mesh_glb()` (self-contained GLB, zero-dep glTF 2.0 writer, texture embedded, KHR_materials_unlit; glTF is Y-up like Atlas — coordinates pass through, only the UV V origin flips)
- `atlas_camera.core.solver.metric_height_from_reference(base_px, top_px, real_height_m, rotation=..., fx, fy, cx, cy)` — single-view metric camera height from one known-size vertical object
- `atlas_camera.core.solver.resolve_reference_scale(refs, ...)` / `apply_reference_scale(solve, refs)` — aggregate multiple reference objects; rescale a solve in place (tier-1 metric scale)
- `solve_still_image_learned(..., scale_references=[...])` — reference objects are tier-1 scale, above depth (tier-2) and the assumed default (tier-3)
- `atlas_camera.inference.depth_estimator.estimate_depth(image_path, model_id=...)` — Depth Anything V2 monocular depth (metric or relative)
- `atlas_camera.core.solver.solve_from_learned_prior(prior, ...)` — pure-numpy builder from a `CameraPrior` (keeps torch out of core)
- `atlas_camera.inference.learned_prior.estimate_camera_prior(image_path)` — run GeoCalib, returns a torch-free `CameraPrior`
- `atlas_camera.core.solver.solve_from_constraints(image_path, constraints_dict)` — artist-guided line constraints
- `tools/solve_image.py` — CLI: auto VP detection + debug overlay + review package
- `tools/solve_constraints.py` — CLI: JSON constraints → review package

## UI architecture

The FastAPI backend (`atlas_camera/ui/api.py`) manages project state in a per-session directory containing `source_image.png`, `atlas_solve.json`, and `constraints.json`. The React workbench owns interactive presentation state (3D viewport toggles, guide drawing, proxy editing) and stores UI-only 3D state under `constraints.viewport3d`. The deterministic solver only reads `line_groups`, `scale_constraints`, and `intrinsics_hint` — `viewport3d` is never used as camera evidence.

## Key design rules

- Core schema is pure Python dataclasses — no external deps.
- Optional deps (`numpy`, `cv2`, `fastapi`, `pxr`) are always guarded by try/except with `pip install -e .[extra]` hints in the error message.
- Adapter boundaries must be explicit: coordinate-system conversions are never silent.
- `LatentComponent` slots (`depth`, `geometry`, `lighting`, `semantics`) default to empty until their solvers exist — review packages describe unsupported components rather than omitting them. `depth` is now populated by `solve_still_image_learned(camera_height="auto")` (Depth Anything V2 map summary + measured camera height + confidence); `geometry`/`lighting`/`semantics` remain empty.
- **View-matrix convention (critical):** geometry/camera math must use the full 4×4 `extrinsics.camera_view_matrix` end-to-end (`cam_to_world = inv(view_matrix)`, row-major, column-vector points, translation in column 3) — the convention proven to match the viewport shader. Never build world math from the 3×3 `camera_rotation_matrix` (it has a transpose ambiguity). `core/proxy_geometry.py`, `core/depth_geometry.py`, `core/plane_extraction.py`, `core/room_layout.py`, and `_ground_depth_compute` all follow this; `THREE.Matrix4.set()` consumes the row-major floats verbatim.
- **Multiple geometry-derivation strategies, always artist-selected, never auto-picked:** `core/depth_geometry.py` factors the shared back-projection/normals/ground-fit/backdrop logic (bit-for-bit consistent with `proxy_geometry.py`'s own copies, which are left untouched to avoid regression risk) so `proxy_geometry.py` (`azimuth_walls`), `plane_extraction.py` (`ransac_planes`), and `room_layout.py` (`room_cuboid`) all agree on world points and metric scale for a given depth map. Every extractor's `stats` dict must include `ground_scale` — required contract with the node's relief-mesh branch, which reuses whichever method's scale was computed. Cross-product normals aren't camera-oriented by construction; `plane_extraction.py`/`room_layout.py` flip them toward the camera globally right after back-projection (sign-invariant under the later ground rescale) so orientation metadata and per-side wall assignment are meaningful.
- Output dimensions **auto-adopt the source image** — never hardcoded. The solve carries `camera.intrinsics.image_width/height`; `AtlasGroundDepthMap`/`GroundMask`/`HorizonMask` take `image_width`/`image_height` where **0 = auto** (via `_solve_image_size`), and `AtlasBlockoutViewport` takes a single `resolution` (long edge) and derives W×H from the source aspect (via `_fit_long_edge`). The blockout frontend resizes its canvas/camera to those `target_width/target_height` on execution, so the viewport inherits the image aspect.
- Camera height / metric scale is **measured, not assumed**, via a tiered cascade (best evidence first), each adopted only above its confidence threshold and otherwise surfaced as a flagged candidate (never silently promoted — matches the LLM-suggestion confirm principle):
  1. **Reference object** (`scale_references` / `AtlasReferenceScaleSolve`) — known-size object (person/door/car from `reference_data`, or explicit height) solved by single-view geometry. Most reliable; `_REFERENCE_ADOPT_CONFIDENCE`. VLM cues auto-feed this tier: `AtlasVLMScaleCues` → `scale_references` JSON → `AtlasApplyScaleReferences` (which rescales **only on `confirm`** — `multimodal_helper.scale_references_from_observation()` maps `SceneScaleCue`s to specs via the registry; LLM cues are never auto-promoted).
  2. **Depth ground-plane** (`camera_height="auto"`) — Depth Anything V2 + `estimate_ground_height_from_depth`; `_HEIGHT_ADOPT_CONFIDENCE`. Unreliable on AI imagery, so usually low-confidence.
  3. **Assumed default** — last resort, flagged. `solve.debug_metadata["scale_source"]` records which tier won.
- The Three.js 3D viewport is a frontend dependency only; it must not become a dependency of `atlas_camera.core`.
- `AtlasDecomposeSolve` is the correct place to expose `horizon_angle_deg` (reads from `solve.debug_metadata`). `AtlasDecomposeCamera` does NOT expose it (camera object has no horizon data).
