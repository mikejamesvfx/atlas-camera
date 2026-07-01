# Project Index: Atlas Camera

Generated: 2026-07-01

## 📁 Project Structure

```
AtlasCamera_Claude/
├── atlas_camera/          # Core Python package (pip install -e ".[dev]")
│   ├── __init__.py        # Public façade: atlas.recover(), all schema exports
│   ├── core/              # DCC-agnostic solver, schema, math (zero runtime deps)
│   ├── exporters/         # Maya, Blender, Nuke, USD, review package writers
│   ├── importers/         # Atlas JSON and USD camera loaders
│   ├── comfy/             # ComfyUI node wrappers (no hard Comfy dependency)
│   ├── datasets/          # ETH3D / DTU / COLMAP benchmark adapters
│   ├── gaussian/          # Future 3DGS placeholder interfaces
│   ├── inference/         # Optional local multimodal provider helpers
│   ├── reference_data/    # Curated scale-reference JSON registry
│   └── ui/                # FastAPI project service (optional, requires [ui])
├── ui/                    # React/Vite/Three.js artist workbench
│   └── src/               # TypeScript source (App.tsx, Viewport3D.tsx, api.ts)
├── tools/                 # CLI scripts (not installed as package commands)
├── tests/                 # pytest suite (20 files)
├── docs/                  # Architecture, roadmap, DCC, UI docs
├── pyproject.toml
└── CLAUDE.md
```

## 🚀 Entry Points

| Entry point | Path | Purpose |
|---|---|---|
| Python API | `atlas_camera/__init__.py` → `atlas.recover()` | Recover LatentScene from image |
| Constrained solve | `atlas_camera/core/solver.py` → `solve_from_constraints()` | Artist-guided line solve |
| Review package CLI | `tools/solve_image.py` | Auto VP + debug overlay + package |
| Constraint CLI | `tools/solve_constraints.py` | JSON constraints → review package |
| FastAPI server | `atlas_camera/ui/__main__.py` | `python -m atlas_camera.ui` |
| React dev server | `ui/` | `npm run dev` (port 5173 → proxies 8787) |
| Benchmark CLI | `tools/benchmark_datasets.py` | ETH3D accuracy evaluation |

## 📦 Core Modules

### `atlas_camera/core/schema.py`
Key types: `LatentScene` / `AtlasSolve`, `LatentCamera` / `AtlasCamera`, `AtlasIntrinsics`, `AtlasExtrinsics`, `AtlasVanishingPoint`, `AtlasHorizon`, `AtlasProjectionScene`, `AtlasProxyPrimitive`, `LatentComponent`, `ConfidenceModel`
All types are pure Python dataclasses. `LatentScene`/`LatentCamera` are the canonical names; `AtlasSolve`/`AtlasCamera` are stable aliases.

### `atlas_camera/core/solver.py`
Exports: `solve_still_image()`, `solve_from_constraints()`
Orchestrates: image loading → vanishing-point detection → intrinsics → extrinsics → projection scene → `LatentScene`.

### `atlas_camera/core/vanishing_points.py`
Exports: `VanishingPointDetector`, `fit_vanishing_point_from_lines()`, `draw_debug_overlay()`, `normalize_line_segment()`
Optional deps: `numpy`, `cv2` (guarded; raises `RuntimeError` with install hint if absent).

### `atlas_camera/core/intrinsics.py`
Exports: `build_intrinsics()` — focal length / sensor-to-pixel conversion.

### `atlas_camera/core/camera_math.py`
Exports: `FOCAL_FALLBACK_CONFIDENCE_PENALTY` and camera-from-VP math utilities.

### `atlas_camera/core/confidence.py`
Exports: `ConfidenceModel` — relative heuristic confidence (not calibrated probability). Keys: `horizon`, `vp1`, `vp2`, `vp3`, `focal`, `extrinsics`, `sensor`.

### `atlas_camera/core/extrinsics.py`
Exports: `atlas_y_up_to_blender_z_up()` and other axis-conversion helpers used at adapter boundaries.

### `atlas_camera/core/io.py`
Exports: `save_solve_json()`, JSON serialisation helpers.

### `atlas_camera/exporters/review_package.py`
Exports: `build_review_package()` → `ReviewPackageResult` with `package_dir` and `files` dict.
Output: `source_image.png`, `debug_overlay.png`, `atlas_solve.json`, `maya_open_scene.py`, Blender/Nuke placeholders, `report.md`, optional USD.

### `atlas_camera/exporters/`
- `maya_exporter.py` → `write_maya_scene_script()`
- `blender_exporter.py` → `write_blender_scene_script()` (Y-up → Z-up at boundary)
- `nuke_exporter.py` → `write_nuke_projection_script()`
- `usd_exporter.py` → `USDExporter` (lazy `pxr` import)

### `atlas_camera/reference_data/registry.py`
Exports: `ScaleReference`, `get_scale_reference()`, `search_scale_references()`
Data: bundled JSON (`atlas_camera/reference_data/*.json`), queried by `reference_id` (e.g. `door_210cm`, `person_175cm`).

### `atlas_camera/inference/multimodal_helper.py`
Exports: `SceneScaleCue`, `GuidanceResponse`, `send_guidance_request()`
Provider: LM Studio (`http://127.0.0.1:1234/v1`) or Ollama. Suggestions are advisory only — they do not mutate the deterministic solve.

### `atlas_camera/datasets/`
- `eth3d.py` → `load_eth3d_dataset()`
- `colmap.py` → `ColmapCamera`, `ColmapImage`
- `dtu.py` → DTU dataset adapter
- `benchmark.py` → `BenchmarkOptions`, `run_benchmark()`

### `atlas_camera/ui/`
- `api.py` — FastAPI app (`app`); REST endpoints: `/api/projects`, `/api/projects/{id}/solve`, `/api/projects/{id}/constraints`, `/api/projects/{id}/export`, `/api/references`, `/api/llm/*`
- `project.py` — project filesystem operations (`create_project`, `solve_project`, `save_constraints`, `export_review_package`, `llm_guidance_project`)
- `__main__.py` — uvicorn launcher, `--port` flag, serves built React from `ui/dist/`

### `atlas_camera/comfy/nodes.py`
ComfyUI node scaffolds. No hard Comfy dependency; nodes call `atlas_camera.core` functions.

### `atlas_camera/gaussian/placeholder.py`
Future 3DGS / point-cloud registration interfaces. Currently stubs.

## 🖥️ React Frontend (`ui/src/`)

| File | Purpose |
|---|---|
| `App.tsx` | Main UI shell: tool modes (select/left/right/vertical/scale), constraint editing, solve trigger, LLM guidance |
| `Viewport3D.tsx` | Three.js 3D viewport: frustum, ground grid, image plate, proxy objects, axis guides |
| `api.ts` | Typed wrappers for all FastAPI endpoints |
| `types.ts` | Shared TypeScript types mirroring Python schema |
| `viewport3dState.ts` | Viewport display toggles and proxy-object state (stored in `constraints.viewport3d`) |
| `viewport3dMath.ts` | Camera frustum / projection math utilities |
| `gridGeometry.ts` | Three.js grid geometry helpers |

## 🔧 Configuration

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, optional dep groups (`dev`, `vision`, `image`, `usd`, `ui`), pytest config |
| `requirements.txt` | Documents that no pinned deps exist; use `pip install -e ".[extras]"` |
| `ui/package.json` | Vite + React + Three.js + Vitest |

## 📚 Documentation

| File | Topic |
|---|---|
| `docs/ARCHITECTURE.md` | Layer model, coordinate conventions, adapter boundary rules |
| `docs/PROJECT_VISION.md` | Long-term direction: LatentScene, depth, geometry, lighting |
| `docs/UI_WORKBENCH.md` | Artist guide for the local 3D lineup workbench |
| `docs/DCC_EXPORTS.md` | Maya, Blender, Nuke, USD handoff details |
| `docs/ROADMAP.md` | Milestones and placeholder features |
| `docs/COMFY_WORKFLOW.md` | ComfyUI integration guide |
| `docs/GAUSSIAN_SPLATS.md` | Future 3DGS registration design |
| `docs/MIGRATION_NOTES.md` | `AtlasSolve` → `LatentScene` rename history |

## 🧪 Test Coverage (20 files)

| Test file | What it covers |
|---|---|
| `test_schema.py` | Dataclass serialisation / round-trip |
| `test_intrinsics.py` | Focal length / pixel conversion |
| `test_vanishing_points.py` | Line geometry and VP fitting |
| `test_camera_math.py` | Camera-from-VP math |
| `test_camera_solver_vanishing_points.py` | Full solver with VP |
| `test_confidence_contract.py` | ConfidenceModel keys and clamping |
| `test_projection_scene.py` | Proxy primitive and scene helpers |
| `test_artist_guided_constraints.py` | `solve_from_constraints()` |
| `test_reference_data.py` | Scale-reference registry lookups |
| `test_review_package.py` | Review package output files |
| `test_solve_image_cli.py` | `tools/solve_image.py` integration |
| `test_solve_constraints_cli.py` | `tools/solve_constraints.py` integration |
| `test_maya_exporter.py` | Maya script output |
| `test_usd_camera_loader.py` | USD import (skipped if usd-core absent) |
| `test_gaussian_placeholder.py` | Gaussian stub interface |
| `test_dataset_adapters.py` | ETH3D / COLMAP adapters |
| `test_latent_api.py` | `atlas.recover()` public API |
| `test_multimodal_helper.py` | LLM guidance helpers |
| `test_ui_backend.py` | FastAPI endpoints (TestClient) |
| `conftest.py` | Shared fixtures |

## 🔗 Key Dependencies

| Dependency | Group | Purpose |
|---|---|---|
| `numpy>=1.24` | dev, vision | VP detection math |
| `opencv-python>=4.8` | dev, vision | Image loading, Hough lines, overlay rendering |
| `pillow>=10` | image, ui | Image size fallback when cv2 absent |
| `fastapi>=0.110` | ui | REST API server |
| `uvicorn>=0.27` | ui | ASGI server |
| `usd-core>=22.11` | usd | USD import/export |
| `pytest>=8` | dev | Test runner |

## 📝 Quick Start

```powershell
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q

# Full UI stack
pip install -e ".[ui,vision]"
python -m atlas_camera.ui      # backend: http://127.0.0.1:8787
cd ui && npm install && npm run dev   # frontend dev server
```
