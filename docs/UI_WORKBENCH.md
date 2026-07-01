# UI Workbench

The optional local UI is an artist-facing inspection workspace for still-image
camera lineup. It combines a React/Vite frontend with the FastAPI project
service in `atlas_camera.ui`.

The UI is not part of the deterministic core package. It reads and writes local
project files, calls backend solve/analyze/export endpoints, and presents the
resulting camera evidence for review.

## Running The Workbench

Start the backend:

```powershell
pip install -e ".[ui,vision]"
python -m atlas_camera.ui
```

Use `--host`, `--port`, or `--reload` when needed:

```powershell
python -m atlas_camera.ui --port 8788
python -m atlas_camera.ui --reload
```

For frontend development, start Vite in a second terminal:

```powershell
cd ui
npm install
npm run dev
```

The Vite server proxies `/api` requests to `http://127.0.0.1:8787`.

## Primary Workflow

1. Load or open a project with a source image.
2. Inspect the image in the 3D lineup viewport.
3. Use guide tools to draw left, right, vertical, and scale constraints on the
   temporary 2D draw layer.
4. Run Analyze to inspect camera matrices, readiness, confidence, and local
   vision pre-analysis when configured.
5. Add or edit 3D proxy objects for visual scale and projection-prep review.
6. Run Solve and Export when the guide families and scale evidence are ready.

Select mode leaves the 3D viewport interactive. Guide tools bring the source
image draw layer forward so pointer events create 2D constraints.

## 3D Viewport

The 3D viewport is implemented in `ui/src/Viewport3D.tsx` with Three.js. It can
display:

- source image plate
- ground grid
- RGB-style axis guides
- solved or preview camera frustum
- artist guide families projected into the scene view
- solved horizon line when available
- editable proxy objects

Supported view modes:

- `image_match`
- `perspective`
- `top`
- `front`
- `side`

The viewport is a browser-only dependency. Three.js must remain isolated to the
frontend and must not be imported by `atlas_camera.core`.

## Proxy Objects

Proxy objects are rough layout aids for scale and projection review. Current
presets include:

- person card
- box
- floor plane
- wall plane
- corridor volume
- unit box
- custom box

Each proxy stores an id, type, label, source, position, rotation, scale, and
lock flag. Proxy state is editable in the right-side `3D Lineup` inspector.
Local LLM image-reading scale candidates can be converted into suggested proxy
objects, but those suggestions remain advisory.

Proxy objects do not affect deterministic camera solving. To make scale matter
to the solve contract, add an explicit `scale_constraints` entry or a future
supported geometry constraint.

## Persisted UI State

The workbench stores viewport state under `constraints.viewport3d`:

```json
{
  "schema_version": 1,
  "display": {
    "active_mode": "image_match",
    "show_image": true,
    "show_grid": true,
    "show_axes": true,
    "show_frustum": true,
    "show_guides": true,
    "show_proxies": true,
    "show_horizon": true,
    "image_opacity": 0.78,
    "grid_scale": 1,
    "lock_camera_to_view": false
  },
  "proxy_objects": [],
  "camera_overrides": {},
  "selected_proxy_id": null
}
```

`ui/src/viewport3dState.ts` owns normalization, preset creation, proxy
selection, and local LLM candidate conversion. The backend currently preserves
unknown constraint keys when saving, so this UI state round-trips with the rest
of the project constraints.

## Solver Boundary

The solver consumes the deterministic constraint fields:

- `image_width`
- `image_height`
- `line_groups`
- `scale_constraints`
- `intrinsics_hint`

The solver may carry full constraints into debug metadata for review, but
`viewport3d` is not evidence. This boundary keeps UI inspection state from
silently changing camera results.

## Verification

Before shipping UI changes, run:

```powershell
cd ui
npm test
npm run build
```

The production build may warn that the main chunk is larger than 500 kB because
Three.js is bundled into the workbench. That warning is expected until the 3D
viewport is code-split.
