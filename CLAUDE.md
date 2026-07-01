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
atlas_camera.comfy      ← ComfyUI node wrappers (no hard Comfy dep)
atlas_camera.ui         ← Optional FastAPI project service
atlas_camera.reference_data ← Curated scale-reference registry (JSON)
atlas_camera.gaussian   ← Future 3DGS / point-cloud interfaces (placeholder)
atlas_camera.inference  ← Optional local multimodal provider helpers
ui/                     ← React/Vite workbench (Three.js 3D viewport)
```

The public API is `import atlas` (thin facade in `atlas_camera/__init__.py`). The stable package name is `atlas_camera`.

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
