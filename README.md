# ATLAS

## Recover the Latent World

Atlas is an open-source platform for recovering the hidden 3D structure implied
by a single image. The current milestone focuses on the first recoverable
component, the `LatentCamera`, and packages it with projection evidence,
confidence, proxy geometry, and DCC handoff data.

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

Placeholder:

- Metric depth fitting from scale/object-height constraints.
- Optional object detector for scale-reference suggestions.
- Production Blender, Nuke, Houdini exporters.
- 3DGS / point-cloud pose estimation.
- Robust production tuning for varied real-world images.

## Documentation

- [Project vision](docs/PROJECT_VISION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [UI workbench](docs/UI_WORKBENCH.md)
- [DCC exports](docs/DCC_EXPORTS.md)
- [Comfy workflow](docs/COMFY_WORKFLOW.md)
- [Gaussian splats](docs/GAUSSIAN_SPLATS.md)
- [Roadmap](docs/ROADMAP.md)
- [Migration notes](docs/MIGRATION_NOTES.md)
