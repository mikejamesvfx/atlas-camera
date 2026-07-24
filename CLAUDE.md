# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.



**This file is a lean ROUTER** (split 2026-07-24 -- it had grown to ~61k tokens
and loaded whole into every session). The deep, authoritative references now
live in two tracked docs; read the relevant one BEFORE editing (routing index
at the bottom):

- [docs/NODE_CATALOG.md](docs/NODE_CATALOG.md) -- full node catalog, `comfy/`
  module layout, `atlas_blockout.js` frontend reference, example workflows.
- [docs/DESIGN_RULES.md](docs/DESIGN_RULES.md) -- every design rule's full
  write-up with its "found live" provenance. Update THERE, not here.

For the artist/TD-facing explanation of camera recovery, matte-painting
projection, and preview dilation, see [docs/USER_GUIDE.md](docs/USER_GUIDE.md).
For the full ecosystem map, see [docs/ECOSYSTEM_GUIDE.md](docs/ECOSYSTEM_GUIDE.md).
For the single-photo -> Nuke camera-move / X-ray marketing pipeline, see
[docs/CAMERA_MOVES.md](docs/CAMERA_MOVES.md).

**`docs/dev/` and `docs/artifacts/` are local-only** (gitignored, and excluded
from the published Registry archive). docs/DESIGN_RULES.md cites them as
provenance -- those paths resolve in a working checkout that has them on disk,
not in a fresh clone. The published tree carries the user-facing docs only.

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



# Install neural extras (GeoCalib learned prior + Depth Anything V2) — needed

# by AtlasLearnedSolveFromImage, AtlasDepthAnything, AtlasDeriveProjectionGeometry,

# AtlasExportReliefMesh. torch must already be present in the target env (e.g.

# ComfyUI's venv); GeoCalib is GitHub-only and not on PyPI.

pip install -e ".[neural]"

pip install "git+https://github.com/cvg/GeoCalib.git"



# Start the FastAPI backend

python -m atlas_camera.ui



# Start the React frontend (separate terminal)

cd ui && npm install && npm run dev



# Install atlas_camera into ComfyUI's venv (editable — run once)

# Replace <COMFYUI_ROOT> with your local ComfyUI install path.

& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install -e .



# Verify ComfyUI can import the package

& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -c "import atlas_camera; print(atlas_camera.__file__)"

```



## Optional dependency groups



- `dev` — numpy, opencv-python, pytest

- `vision` — numpy, opencv-python (runtime vision)

- `image` — Pillow only

- `usd` — usd-core

- `ui` — FastAPI, uvicorn, Pillow, python-multipart

- `neural` — torch + GeoCalib (learned single-image camera prior). GeoCalib is GitHub-only: `pip install "git+https://github.com/cvg/GeoCalib.git"`. torch is expected from the host env (e.g. ComfyUI's venv).

- `sam3` — native SAM3 segmentation (`transformers>=5.5.4`, no triton). Powers `AtlasSAM3Mask`, the preferred segmenter in `AtlasInput`'s sky/scope cascade. `facebook/sam3` is gated on Hugging Face — see INSTALL.md for the one-time auth steps.



The core package has **zero required runtime dependencies**. All vision, USD, and UI imports are guarded with informative `ImportError` messages.



## Architecture



```

atlas_camera.core       � DCC-agnostic schema, solver, math (no host deps)

atlas_camera.plate      � Colour-managed float plate I/O + pixel ops (OpenImageIO;

                          EXR/DPX read+write with OCIO conversion, edge-extend,

                          mask flood). Needs [oiio]; no ComfyUI dependency

atlas_camera.raw        � Camera RAW decode/metadata/undistort (rawpy, lensfun,

                          EXIF→intrinsics). Needs [raw]

atlas_camera.exporters  � Maya, Blender, Nuke, USD, review package writers

atlas_camera.importers  � Atlas JSON and USD camera loaders

atlas_camera.comfy      � ComfyUI node library (67 nodes + 4 experimental, no hard Comfy dep;

                          nodes.py is a façade over node_helpers / node_registry / nodes_*

                          responsibility modules — see "Module layout" below)

atlas_camera.inference  � Optional local multimodal provider helpers, depth/normal

                          backends, SAM3, GeoCalib

atlas_camera.mcp        � Optional stdio MCP server exposing a running ComfyUI to

                          MCP-capable assistants. Stdlib + the mcp SDK only. Needs [mcp]

atlas_camera.datasets   � Benchmark dataset loaders (COLMAP, DTU, ETH3D) for

                          accuracy evaluation — not part of the node runtime

atlas_camera.ui         � Optional FastAPI project service

atlas_camera.reference_data � Curated scale-reference registry (JSON)

atlas_camera.utils      � Tiny shared path helpers

ui/                     � React/Vite workbench (Three.js 3D viewport)

examples/               � Example ComfyUI workflows and test images



**Dependency direction is one-way and enforced by the 2026-07-20 layering

refactor:** `comfy/` may import anything; nothing outside `comfy/` may import it.

`core`, `plate` and `raw` are host-agnostic and load with zero ComfyUI modules —

which is what makes their math unit-testable without a ComfyUI install.

```



The public API is `import atlas` (thin facade in `atlas_camera/__init__.py`). The stable package name is `atlas_camera`.



## ComfyUI integration — see docs/NODE_CATALOG.md

The full node catalog (70 nodes + 4 experimental), `comfy/` module layout,
setup/symlink instructions, double-import guard, `atlas_blockout.js` frontend
reference, `/atlas/camera_data` endpoint, and the example-workflow catalog all
live in [docs/NODE_CATALOG.md](docs/NODE_CATALOG.md). Read the relevant part
BEFORE editing anything under `atlas_camera/comfy/` or `comfy/web/`.

Quick facts that must never drift (details in the catalog):
- `nodes.py` is a compatibility FACADE over `node_helpers` / `node_registry` /
  `nodes_*` responsibility modules; import from the specific module in new
  code. `tests/test_facade_surface.py` pins all facade names.
- Registered node keys + display names are a saved-workflow contract
  (`tests/test_comfy_node_registry.py` pins the surface; currently 70 + 4).
- `comfy/__init__.py` loads twice at startup — route registration sits behind
  a double-import guard; keep it there.
- Shipping example workflows are pinned by `tests/test_example_workflows.py`
  (+ `test_shipping_workflow_paths.py`: no absolute machine paths; workflow
  ids are UUIDs). Review = add the name to the pin.

## Core schema (`atlas_camera/core/schema.py`)



- `LatentScene` / `AtlasSolve` — top-level result (aliases; both stable)

- `LatentCamera` / `AtlasCamera` — recovered camera (aliases)

- `LatentComponent` — empty slot for future depth/geometry/lighting/semantics

- `AtlasIntrinsics`, `AtlasExtrinsics`, `AtlasVanishingPoint`, `AtlasHorizon`, `AtlasProjectionScene`, `AtlasProxyPrimitive`



## Coordinate conventions



- World: **right-handed, Y-up** by default

- **Recovered camera faces world −Z** (canonicalized 2026-07-10 — `solver._face_camera_toward_negative_z`, applied in BOTH solve paths): yaw is unobservable from a single image, so the facing is a free convention, and −Z matches Maya/Nuke default cameras. Before this, every DCC import needed a manual −180° Y rotation (found by a real Maya lineup). The flip is a world-side RotY(180) — gravity/pitch untouched (LEFT-multiply on the cam_to_world block; the transposed side rotates in the CAMERA frame and inverts pitch — guarded by tests)

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



## Hard rules (never violate — full write-ups in docs/DESIGN_RULES.md)

Layering & data
- `comfy/` may import anything; NOTHING outside `comfy/` imports it. `core` /
  `plate` / `raw` stay host-agnostic (unit-testable without ComfyUI).
- DCC coordinate conversions happen ONLY at adapter boundaries, never in core.
- The 4x4 `camera_view_matrix` is the world-math convention end-to-end; never
  build world math from the 3x3 rotation (transpose ambiguity).
- `_json_ready` must serialize everything on a solve (incl. os.PathLike) —
  solve JSON is a contract; a manifest failure must NEVER fail an export.
- Scale/health verdicts come ONLY from `core.scene_health` — never re-derive
  trust ad hoc.

Nodes & widgets
- Combo VALUES are append-only; never rename/reorder registered node keys,
  display names, or existing combo values (they serialize into workflows).
- New widgets are APPENDED last (positional `widgets_values`).
- `AtlasCleanPlateLayer` is capability-frozen: no new widgets — new features
  land as companion nodes feeding its existing inputs.
- ComfyUI's backend rejects STRING->combo links: expose `*_override` STRING
  inputs instead (the `patch_view_override` pattern).
- Any persisted widget that gates execution needs a content fingerprint, and
  any silent branch-skip needs a visible explanation (gate doctrine;
  `docs/dev/gate_state_table.md`).
- Derive nodes CLOBBER prior PROXY_ROLE geometry (deliberate);
  `AtlasMergeGeometry` is the one explicit combiner.

Geometry & projection
- Relief-mesh tears are load-bearing. Never fix a black tear by raising a
  global threshold — the fix is a deliberate layer (card/ground/sky/inpaint).
- An explicit `exclude_mask` REPLACES the internal sky heuristic; need both ->
  OR them externally. Scoped excludes shift band percentiles: give percentile
  band nodes the plain sky mask on `band_ref_mask` (drift rule).
- Seam doctrine: edge-extend smear lives on the layers BEHIND; the frontmost
  band keeps a clean cut; band priorities are FARTHEST-highest.
- Export-only transforms (interior hole fill, retopo) never touch the live
  projection mesh or `solve.proxy_geometry`.
- Depth model doctrine: exterior -> V2-Metric-Outdoor, interior -> MoGe (or
  V2-Indoor); DA3 installs with `pip --no-deps`. Depth is SHARED via
  `ATLAS_DEPTH_MAP` so branches agree on metric scale.

Frontend (atlas_blockout.js)
- NEVER assign lifecycle callbacks after `addDOMWidget` — always chain
  (`onResize` / `onRemoved` / `onConfigure`); assignment orphans DOM on
  workflow switch.
- No JS resize hooks for canvas sizing — CSS only (`height:100%` chain,
  `min-width:0`, `object-fit:contain`); render resolution is governed solely
  by the `resolution` widget.
- three.js comes ONLY from the vendored `web/lib/atlas-three.bundle.js`
  (rebuild via `npm run build:comfy-three` in `ui/`); no CDN fallback.
- A raw ShaderMaterial gets NO automatic colourspace encode: the projection
  shader applies `atlasLinearToSRGB` by hand; exposure/tonemapping stay out
  of it entirely.
- Projection materials are REBUILT every execution: dynamic uniforms are
  pushed per-frame by `syncProjectionLightUniforms`, never only at build.
- JS mirrors of Python tables (`SCENE_TYPE_PRESETS`, palette, Catmull-Rom)
  are accepted hand-sync duplication — `tests/test_frontend_mirrors.py` pins
  them.

## Routing index — read the deep doc BEFORE editing

| Touching | Read first |
|---|---|
| Any node's inputs/outputs/behaviour | docs/NODE_CATALOG.md (its table row) |
| `atlas_camera/comfy/` module structure | docs/NODE_CATALOG.md (module layout) |
| `atlas_blockout.js` / viewport UI | docs/NODE_CATALOG.md (frontend extension) + docs/DESIGN_RULES.md (viewport bullets) |
| Relief mesh / tearing / repair | docs/DESIGN_RULES.md (tearing, outlier tier, mesh_repair) |
| Bands / clean plates / inpaint layers | docs/DESIGN_RULES.md (inpaint layers, bounded band, seam doctrine) |
| Depth models / sky mask | docs/DESIGN_RULES.md (DA3/MoGe/V2, sky-aware depth) |
| Patches / occlusion / camera path | docs/DESIGN_RULES.md (multi-angle patch, camera path) |
| Exporters (Nuke/Maya/USD/EXR) | docs/DESIGN_RULES.md (Nuke topology, layer exports, manifest) |
| RAW / OCIO / plates | docs/DESIGN_RULES.md (RAW import, P0 trust tier) + `atlas_camera/plate/oiio_io.py` docstrings |
| Gates / VLM assess | docs/DESIGN_RULES.md (gates, AtlasAssessImage) + docs/dev/gate_state_table.md |
| Example workflows | docs/NODE_CATALOG.md (example workflows) + the pin tests |

For the artist/TD view: docs/USER_GUIDE.md; ecosystem map:
docs/ECOSYSTEM_GUIDE.md; camera moves: docs/CAMERA_MOVES.md.

## graphify



This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.



Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.

- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.

- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.

- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

