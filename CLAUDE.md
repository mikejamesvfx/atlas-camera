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

The core package has **zero required runtime dependencies**. All vision, USD, and UI imports are guarded with informative `ImportError` messages.

## Architecture

```
atlas_camera.core       ← DCC-agnostic schema, solver, math (no host deps)
atlas_camera.exporters  ← Maya, Blender, Nuke, USD, review package writers
atlas_camera.importers  ← Atlas JSON and USD camera loaders
atlas_camera.comfy      ← ComfyUI node library (18 nodes, no hard Comfy dep)
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

### Node catalog (18 nodes)

**Category: Atlas Camera**

| Node class | Inputs | Outputs | Notes |
|---|---|---|---|
| `AtlasLoadImageSolveCamera` | image_path, image_width, image_height | ATLAS_SOLVE | Legacy: file-path-based solve |
| `AtlasSolveFromImage` | image (IMAGE), ±focal_mm, ±sensor_mm | ATLAS_SOLVE | Primary: accepts ComfyUI tensor |
| `AtlasConstrainedSolve` | image, constraints_json | ATLAS_SOLVE | Artist-guided; pass scale_constraints for cam height |
| `AtlasLoadSolveJSON` | json_path | ATLAS_SOLVE | Load previously saved solve |
| `AtlasDecomposeSolve` | ATLAS_SOLVE | camera, confidence, source_method, image_width, image_height, solve_json, horizon_angle_deg | horizon_angle_deg comes from debug_metadata |
| `AtlasDecomposeCamera` | ATLAS_CAMERA | fx, fy, cx, cy, cam_x, cam_y, cam_z, focal_mm, fov_h_deg | cam_y must be > 0 for depth map |
| `AtlasGroundDepthMap` | ATLAS_SOLVE, image_width, image_height, near_m, far_m | depth_image (IMAGE), ground_mask (MASK) | Black if cam_y ≤ 0 |
| `AtlasGroundMask` | ATLAS_SOLVE, image_width, image_height | MASK | 1 = ground, 0 = sky |
| `AtlasHorizonMask` | ATLAS_SOLVE, image_width, image_height, feather_px | MASK | 1 = above horizon (sky) |
| `AtlasVPVisualization` | image, ATLAS_SOLVE, ±show_horizon, ±show_vp_lines, ±line_opacity | IMAGE | Pass-through if no VPs detected |
| `AtlasBlockoutViewport` | ATLAS_SOLVE, source_image, width, height, client_data | shaded, depth, normal, mask (all IMAGE) | OUTPUT_NODE; browser-side Three.js; Render Passes button populates client_data |
| `AtlasExportSolveJSON` | ATLAS_SOLVE, file_path | STRING | Writes JSON file |
| `AtlasExportBlender` | ATLAS_SOLVE, output_dir | STRING (script_path) | Writes build_scene.py |
| `AtlasExportNuke` | ATLAS_SOLVE, output_dir | STRING (script_path) | Writes Nuke projection script |
| `AtlasExportUSD` | ATLAS_SOLVE, output_dir | STRING (usd_path) | Writes camera.usda |
| `AtlasExportReviewPackage` | ATLAS_SOLVE, output_dir | STRING | Full review bundle |
| `AtlasExportMayaReviewScene` | ATLAS_SOLVE, output_dir | STRING | Maya scene + image card |
| `AtlasUSDCameraLoader` | usd_path | ATLAS_CAMERA | Load camera from USD |

### Frontend extension (`atlas_camera/comfy/web/atlas_blockout.js`)

Registers as `AtlasCamera.Blockout` ComfyUI extension targeting `AtlasBlockoutViewport` nodes. On node creation it builds a Three.js canvas with OrbitControls, a primitive toolbar (Box/Plane/Cylinder/Person/Clear), and a Render Passes button.

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

### Example workflow

`examples/atlas_camera_full_workflow.json` — full 20-node workflow covering all five groups:
① Source Image → ② Camera Solve → ③ Analysis Passes → ④ Blockout Viewport → ⑤ DCC Exports

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

- `atlas.recover(image_path, ...)` → `LatentScene` — primary API
- `atlas_camera.core.solver.solve_from_constraints(image_path, constraints_dict)` — artist-guided line constraints
- `tools/solve_image.py` — CLI: auto VP detection + debug overlay + review package
- `tools/solve_constraints.py` — CLI: JSON constraints → review package

## UI architecture

The FastAPI backend (`atlas_camera/ui/api.py`) manages project state in a per-session directory containing `source_image.png`, `atlas_solve.json`, and `constraints.json`. The React workbench owns interactive presentation state (3D viewport toggles, guide drawing, proxy editing) and stores UI-only 3D state under `constraints.viewport3d`. The deterministic solver only reads `line_groups`, `scale_constraints`, and `intrinsics_hint` — `viewport3d` is never used as camera evidence.

## Key design rules

- Core schema is pure Python dataclasses — no external deps.
- Optional deps (`numpy`, `cv2`, `fastapi`, `pxr`) are always guarded by try/except with `pip install -e .[extra]` hints in the error message.
- Adapter boundaries must be explicit: coordinate-system conversions are never silent.
- `LatentComponent` slots (`depth`, `geometry`, `lighting`, `semantics`) default to empty until their solvers exist — review packages describe unsupported components rather than omitting them.
- The Three.js 3D viewport is a frontend dependency only; it must not become a dependency of `atlas_camera.core`.
- `AtlasDecomposeSolve` is the correct place to expose `horizon_angle_deg` (reads from `solve.debug_metadata`). `AtlasDecomposeCamera` does NOT expose it (camera object has no horizon data).
