# ATLAS

## Recover the Latent World

Atlas is an open-source platform for recovering the hidden 3D structure implied
by a single image. The current milestone focuses on the first recoverable
component, the `LatentCamera`, and packages it with projection evidence,
confidence, proxy geometry, and DCC handoff data.

> **Status: beta (`release/beta-0.2`, v0.3.0).** Deterministic core + a 46-node
> ComfyUI pack for single-image camera recovery, matte-painting projection,
> layered 2.5D clean-plate rigs, and DCC handoff. See
> [Current Status](#current-status) for what's implemented vs. placeholder.

## Two distributions

- **`main` — the working version.** Everything above the experimental line:
  camera solve, geometry derivation, the layered DMP rig, viewport, and all
  DCC exporters. Runs on any ComfyUI install — the core package has zero
  required runtime dependencies, and the vision/depth features need only the
  `[neural]` extra. No Docker, no research-licensed models.
- **`experimental` — the 🔬 tier enabled.** Same codebase with two extra
  nodes registered by default: `AtlasRenderFix` (NVIDIA Fixer render repair —
  needs Docker + an NVIDIA GPU) and `AtlasPredictHiddenGeometry` (LaRI /
  World Tracing X-ray depth — research-only upstream licenses, user-cloned).
  Their setup lives in [INSTALL.md](INSTALL.md).

The switch is one env var: `ATLAS_EXPERIMENTAL=1` before launching ComfyUI
registers the experimental nodes on *any* branch (`=0` hides them on
`experimental`). The branches differ only in that default.

It is designed for:

- Matte painters
- Environment artists
- Concept-to-3D workflows
- AI-generated image projection workflows
- DMP camera lineup
- DCC handoff

Atlas is not an AI image generator, a sequence camera tracker, a photogrammetry
package, a depth model, or a ComfyUI-only node. It uses those technologies where
appropriate while keeping the deterministic core focused on recovering a
reusable latent scene representation.

## MVP Workflow

```text
image -> LatentCamera -> debug overlay -> review package -> Maya / Blender / Nuke / USD
```

The current milestone provides a clean Python package, portable `LatentScene`
schema aliases, basic intrinsics helpers, DCC-agnostic projection scene data,
an optional local 3D lineup workbench, and a review package builder.

## Future Workflow

```text
image + 3DGS scene prior -> camera registration -> projection handoff
```

3D Gaussian Splat and point-cloud camera registration are future research hooks,
not implemented production features.

## Core Principles

- Core schema is DCC-agnostic.
- Atlas core defaults to right-handed Y-up coordinates.
- Image coordinates use origin top-left, x right, y down.
- Maya, Blender, Nuke, USD, and ComfyUI are adapters around the core.
- Coordinate conversions must happen explicitly at import/export boundaries.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

Run the optional local web UI:

```powershell
pip install -e ".[ui,vision]"
python -m atlas_camera.ui
```

Open `http://127.0.0.1:8787` for the FastAPI-backed UI, or start the React
workbench during development:

```powershell
cd ui
npm install
npm run dev
```

If `8787` is already in use, run the backend on another port:

```powershell
python -m atlas_camera.ui --port 8788
```

The workbench opens on a 3D camera lineup surface. Use Select mode to orbit or
inspect the scene, then switch to left/right/vertical/scale guide tools when
you need the 2D source-image draw layer. The 3D panel can show the source image
plate, camera frustum, ground grid, axis guides, horizon, artist guide lines,
and editable proxy objects. View settings and proxy objects are stored in
`constraints.viewport3d` alongside the existing artist constraints.

Create a metadata-only prototype solve and review package:

```python
import atlas
from atlas_camera.exporters.review_package import build_review_package

scene = atlas.recover(
    "concept.png",
    image_size=(1920, 1080),
    intrinsics_hint={"focal_length_mm": 35.0, "sensor_width_mm": 36.0},
)
result = build_review_package(scene, "review_packages")
print(result.package_dir)
```

The stable package name remains `atlas_camera`, and the concise `import atlas`
facade mirrors the public API for vision-facing examples.

Or run the one-command MVP workflow with vanishing-point detection and a debug
overlay:

```powershell
python tools\solve_image.py --image concept.png --output-dir review_packages --package-name atlas_review_001
```

This writes a portable review package containing `source_image.png`,
`debug_overlay.png`, `atlas_solve.json`, `maya_open_scene.py`, DCC placeholder
scripts, and `report.md`.

For artist-guided line constraints stored in JSON:

```powershell
python tools\solve_constraints.py --image concept.png --constraints constraints.json --output-dir review_packages
```

List curated scale references:

```powershell
python tools\list_references.py --query person
```

Run a metadata-only ETH3D benchmark against an externally downloaded dataset
root:

```powershell
python tools\benchmark_datasets.py --dataset eth3d --root C:\path\to\eth3d_scene --limit 10
```

This writes JSON and CSV reports under `validation_output/`. Large datasets
should stay outside the repository, for example under `external_datasets/` or
another local path.

Optional local multimodal guidance can run through LM Studio, llama.cpp, or
Ollama from the UI. The workbench defaults to LM Studio at
`http://127.0.0.1:1234/v1`; select Ollama in the provider control if you want
to use an Ollama-hosted model. Install Ollama separately, then pull a local
vision model:

```powershell
ollama pull gemma3:4b
```

The UI's Guide action sends the current source image plus Atlas solve context
to the selected local provider. Model outputs are stored as advisory guidance
only and do not mutate the deterministic camera solve.

Artist-guided line constraints can drive the same solver when automatic
detection needs help:

```python
from atlas_camera.core.solver import solve_from_constraints

solve = solve_from_constraints(
    "concept.png",
    {
        "image_width": 1920,
        "image_height": 1080,
        "line_groups": {
            "left": [
                ((100, 500), (900, 300)),
                ((120, 650), (900, 420)),
            ],
            "right": [
                ((1000, 300), (1800, 500)),
                ((1000, 420), (1780, 650)),
            ],
        },
        "scale_constraints": [
            {
                "reference_id": "door_210cm",
                "image_points": [[850, 760], [850, 410]],
            }
        ],
        "focal_length_mm": 35.0,
        "sensor_width_mm": 36.0,
    },
)
```

Scale constraints are stored as explicit review landmarks and height-guide proxy
geometry. They do not yet solve metric depth from a single image.

## ComfyUI Node Pack

The flagship interface is a **47-node ComfyUI pack** (category **Atlas Camera**)
that runs the whole pipeline as a graph. Install it into ComfyUI's own venv
(editable, so source changes are live):

```powershell
# Symlink the node pack into ComfyUI (once):
#   <COMFYUI>\custom_nodes\AtlasCamera  ->  <REPO>\atlas_camera\comfy
& "<COMFYUI>\venv\Scripts\python.exe" -m pip install -e ".[neural]"
```

The `[neural]` extra adds the Depth Anything V2 depth models + the GeoCalib
learned prior (GeoCalib is GitHub-only:
`pip install "git+https://github.com/cvg/GeoCalib.git"`; torch is expected from
ComfyUI's env). **Depth Anything 3 is the default depth model** since v0.3 —
measurably fewer relief-mesh tears and focal-conditioned metric depth using the
*solved* focal; it needs the separate `[neural-da3]` extra (see
[INSTALL.md](INSTALL.md) — into a ComfyUI venv install `--no-deps`). Every
`depth_model` combo keeps the V2 models available.

**Core through-line — recover → derive → project → hand off:**

```text
Load Image → Learned Solve (GeoCalib) → Derive Projection Geometry
          → Atlas Viewport (📽 Project) → Export Relief Mesh / DCC
```

Node tracks:

- **Solve** — `AtlasSolveFromImage` (geometric vanishing points, no deps) and
  `AtlasLearnedSolveFromImage` (GeoCalib learned prior — robust on AI-generated
  images; `height_mode=measure_from_depth` measures camera height from depth).
- **Scale** — tiered metric-scale cascade: known-size reference objects
  (`AtlasReferenceScaleSolve`), local-VLM scale cues (`AtlasVLMScaleCues` →
  `AtlasApplyScaleReferences`, confirm-to-adopt), then depth, then a flagged
  default. Never auto-promoted.
- **Derive geometry** — `AtlasDeriveProjectionGeometry` (relief mesh and/or
  fitted primitives, artist-selected strategy), plus a **composable** track:
  `AtlasDepthMap` + `AtlasDeriveReliefMesh`/`Walls`/`TowersSpires`/
  `RoofsFacades`/`InteriorRoom`, combined with **`AtlasMergeGeometry`**
  (a Nuke-Merge-node equivalent) to mix strategies per scene region.
- **Shot format** — `AtlasDefineShotCam` sets a project-level render/output
  camera (sensor + lens + resolution), attachable via `AtlasMergeGeometry` so the
  viewport/exporters conform to one shot format.
- **DMP layer stack** — the classic 2.5D clean-plate rig as nodes:
  `AtlasSkyDomeLayer` (SAM-driven sky separation with deterministic
  edge-extend/frame-outpaint), `AtlasDepthLayerMask` + `AtlasCleanPlateLayer`
  (depth-banded clean-plate layers with disocclusion fill, per-pixel edge
  mattes, beveled skirts), `AtlasDepthBandSplit` (one authoritative fg/bg
  boundary), and hole masks everywhere (the literal "where projection shows
  black" signal). Inpainting stays graph-level (LaMa/LanPaint/FLUX packs).
- **Viewport** — `AtlasBlockoutViewport`: browser-side Three.js preview with a
  recovered-camera inherit, **📽 Project** (matte-painting projection onto
  geometry), 🎥 camera-path authoring with presets + baked-frame output,
  🧭 measured safe-zone orbit clamps, 📐 patch-angle extraction, 💡 relight
  preview, 🩻 hidden-geometry provenance overlay, and 4 render passes.
- **Multi-angle fill** — `AtlasAddPatchView` / `AtlasOcclusionMask` project
  extra LoRA-generated views to fill what the single recovered camera can't see.
- **🔬 Experimental hidden geometry** — `AtlasPredictHiddenGeometry` predicts
  the surfaces *behind* occluders (layered ray intersections: LaRI, fast
  regression, or World Tracing, generative diffusion — both research-only,
  user-installed) and patches them into an "X-ray" depth map that band layers
  turn into real reveal geometry. See
  [docs/dev/hidden_geometry_training_free_research.md](docs/dev/hidden_geometry_training_free_research.md).
- **Pre-flight** — `AtlasAssessImage` gates the graph behind a local-VLM
  scene assessment (advisory, ▶ Continue to proceed).
- **Output desk** — `AtlasRegisterPlate` / `AtlasAttachSourcePlate` track the
  real float plate (EXR/ACEScg) past the browser preview into every exporter.
- **Export** — Relief mesh (OBJ/GLB, projection baked into UVs), Blender, Nuke
  (.py + native .nk), per-layer Nuke/Maya scene exports, USD, Maya review
  scene, camera-path USD, and a full review package.

Ready-to-load example workflows are in [`examples/`](examples/) — start with
`atlas_camera_core_projection_workflow.json` (the 6-node core),
`atlas_camera_learned_workflow.json` (full pipeline), or the calibrated
per-scene `atlas_camera_hidden_geometry_*_workflow.json` demos (cathedral /
hangar / jungle / canyon / ridge / valley — X-ray reveals + dolly-in camera
move + mp4 bake). See [docs/ECOSYSTEM_GUIDE.md](docs/ECOSYSTEM_GUIDE.md) for
the full node catalog.

## Current Status

Implemented:

- Portable dataclass schema for `AtlasSolve`, `AtlasCamera`, intrinsics,
  extrinsics, vanishing points, horizon, and projection scenes.
- First-class `LatentScene` and `LatentCamera` schema names, with `AtlasSolve`
  and `AtlasCamera` retained as compatibility aliases.
- Empty `LatentComponent` slots for future depth, geometry, lighting, and
  semantic recovery.
- `atlas.recover(...)` for the project-vision API.
- Intrinsics helper for focal length and sensor-to-pixel conversion.
- Vanishing-point detection with optional OpenCV/NumPy vision dependencies.
- Camera estimation from two orthogonal vanishing points.
- Debug overlay rendering for detected lines, vanishing points, horizon, and
  estimated camera metadata.
- Artist-guided line constraints via `solve_from_constraints(...)`.
- Curated local scale-reference registry with `reference_id` support in guided
  constraints.
- Review package output folder with JSON, Maya script, Blender placeholder,
  Nuke placeholder, report, and optional USD files.
- Lazy USD import/export boundary.
- ComfyUI node scaffolds with no hard Comfy dependency.
- Explicit 3DGS placeholder interfaces.
- Optional FastAPI + React local UI for artist-guided still-image lineup,
  constraint editing, Three.js 3D camera/proxy inspection, solve review, local
  multimodal guidance, and review-package export.
- **47-node ComfyUI pack** (category "Atlas Camera") covering the full
  recover → derive → layer → project → export pipeline as a graph.
- **Learned camera recovery** (GeoCalib prior, `method="learned"`) — robust on
  AI-generated images where geometric vanishing points fail.
- **Depth Anything 3 (default) + Depth Anything V2** monocular depth backends +
  ground-plane camera-height measurement (`camera_height="auto"`), with
  sky-aware masking. DA3 converts canonical depth to metres with the *solved*
  focal (measured: ~3× fewer relief-mesh tears vs V2 on the test set).
- **Layered 2.5D DMP rig** — sky dome, depth-band clean-plate layers with
  disocclusion fill, per-pixel edge mattes, edge-extend/frame-outpaint,
  hole-mask honesty signals, and one authoritative band split.
- **🔬 Experimental hidden-geometry prediction** (research-only): LaRI /
  World Tracing layered ray intersections → "X-ray" depth maps → real reveal
  geometry behind occluders, with a viewport provenance overlay and six
  calibrated per-scene demo workflows.
- **Composable projection-geometry derivation** (shared depth map + per-strategy
  derive nodes: relief mesh, walls, towers/spires, roofs/facades, interior room)
  combined with a **merge node**, plus a project-level **shot-camera** format.
- **Relief-mesh export** (OBJ/MTL + GLB, camera projection baked into UVs; imports
  textured into Maya/Nuke/ZBrush/Blender).
- **Interactive browser viewport** with live camera-projection ("matte-painting")
  preview and four render passes (shaded/depth/normal/mask).
- **Camera-path animation** authoring in the viewport with USD export.
- **Multi-angle patch projection** + occlusion mask to fill single-camera gaps.
- Tiered, confirm-to-adopt metric-scale cascade (reference object → local-VLM
  cue → depth → flagged default); LLM/VLM suggestions never auto-promoted.

Placeholder:

- Metric depth fitting from scale/object-height constraints.
- Optional object detector for scale-reference suggestions.
- Production Blender, Nuke, Houdini exporters.
- 3DGS / point-cloud pose estimation.
- Robust production tuning for varied real-world images.

## Documentation

- [Install guide](INSTALL.md) — including the `[neural-da3]` and
  research-only hidden-geometry setup
- [Changelog](CHANGELOG.md) — release notes per beta branch
- [Third-party notices](THIRD_PARTY.md) — license boundaries, incl. the
  research-only hidden-geometry backends
- [User guide](docs/USER_GUIDE.md) — now with the 2026-07-09 five-layer-stack
  section
- [Ecosystem guide](docs/ECOSYSTEM_GUIDE.md) — full node catalog
- [Project vision](docs/PROJECT_VISION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [UI workbench](docs/UI_WORKBENCH.md)
- [DCC exports](docs/DCC_EXPORTS.md) — rewritten 2026-07-09 around the
  verified Nuke/Maya topology
- [Comfy workflow](docs/COMFY_WORKFLOW.md)
- [Hidden-geometry research](docs/dev/hidden_geometry_training_free_research.md)
- [DA3 depth-backend test plan](docs/dev/da3_backend_test_plan.md)
- [Gaussian splats](docs/GAUSSIAN_SPLATS.md)
- [Roadmap](docs/ROADMAP.md)
- [Migration notes](docs/MIGRATION_NOTES.md)
