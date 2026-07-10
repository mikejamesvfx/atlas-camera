# Atlas Camera — ComfyUI Integration

> **Vintage note (2026-07-09):** this page's two-track mental model is still
> correct, but the catalog has grown to **47 nodes** — including the shared-depth
> layer nodes, the sky dome, the Output Desk color track, and the experimental
> hidden-geometry 🔬 node — and the default depth model is now DA3METRIC-LARGE.
> For the current catalog see CLAUDE.md; for the layered-workflow story see
> USER_GUIDE.md's 2026-07-09 section and ECOSYSTEM_GUIDE.md's addenda.

## Overview

The `atlas_camera.comfy` package adds ComfyUI nodes split across two tracks:

- **Track 1 — Python-only nodes**: Solve, decompose, analysis masks, VP visualization, per-DCC exports. No browser dependency.
- **Track 2 — AtlasBlockoutViewport**: A Three.js viewport embedded in the ComfyUI node panel. The recovered camera is applied to the Three.js camera so the scene pre-aligns with the source photo. Artist places blockout geometry, clicks Render Passes, and four IMAGE outputs (shaded / depth / normal / mask) flow into the graph.

---

## Installation

```powershell
# 1. Create symlink (run once as admin or with Developer Mode on)
New-Item -ItemType SymbolicLink `
  -Path "C:\Users\miike\ComfyUI_V91\ComfyUI\custom_nodes\AtlasCamera" `
  -Target "C:\Users\miike\Desktop\AtlasCamera_Claude\atlas_camera\comfy"

# 2. Install atlas_camera package into ComfyUI's Python environment
& "C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe" `
  -m pip install -e "C:\Users\miike\Desktop\AtlasCamera_Claude"

# 3. Verify
& "C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe" `
  -c "import atlas_camera; print(atlas_camera.__file__)"
# Expected: ...AtlasCamera_Claude\atlas_camera\__init__.py
```

Start ComfyUI normally. All 18 Atlas Camera nodes appear under the **Atlas Camera** category in the node browser (right-click canvas → Add Node → Atlas Camera).

---

## Example workflow

Load `examples/atlas_camera_full_workflow.json` (drag-and-drop into ComfyUI).

Five colour-coded groups:

| Group | Key nodes |
|---|---|
| ① Source Image | `LoadImage` |
| ② Camera Solve | `AtlasSolveFromImage` → `AtlasDecomposeSolve` → `AtlasDecomposeCamera` |
| ③ Analysis Passes | `AtlasGroundDepthMap`, `AtlasGroundMask`, `AtlasHorizonMask`, `AtlasVPVisualization` |
| ④ Atlas Viewport | `AtlasBlockoutViewport` + 4× `PreviewImage` |
| ⑤ DCC Exports | `AtlasExportSolveJSON`, `AtlasExportBlender`, `AtlasExportNuke`, `AtlasExportUSD`, `AtlasExportReviewPackage` |

---

## Choosing a good source image

The auto-solve (`AtlasSolveFromImage`) needs strong perspective cues. Best results come from:

- **Real photographs** (not AI renders) — AI images often lack consistent perspective geometry
- **Exterior shots at eye height (~1.6 m)** with a visible ground plane
- **At least one clear vanishing direction** — a road, building face, or tiled floor
- **Horizon visible** in the upper third of the frame

Difficult cases (expect `cam_y ≈ 0` or degenerate output):
- AI-generated images
- Interior shots with heavy occlusion
- Fisheye or strongly tilted cameras
- Industrial/pipe scenes without a clear ground plane

For difficult images use `AtlasConstrainedSolve` with a `scale_constraints` JSON hint (see below).

---

## Node reference

### `AtlasSolveFromImage`

Accepts a ComfyUI IMAGE tensor. Saves to a temp PNG, runs `solve_still_image`, returns `ATLAS_SOLVE`.

Optional inputs `focal_length_mm` and `sensor_width_mm` override EXIF-based intrinsics. Leave at `0.0` / `36.0` for auto-detect.

### `AtlasConstrainedSolve`

Use this when auto-solve produces degenerate results. The `constraints_json` widget accepts the same JSON format as the React UI's `constraints.json`:

```json
{
  "scale_constraints": [
    {"type": "camera_height_m", "value": 1.6}
  ],
  "line_groups": [],
  "intrinsics_hint": {}
}
```

Passing `camera_height_m` forces the solver to place the camera 1.6 m above the Y=0 ground plane, which is essential for the depth map to produce non-black output.

### `AtlasDecomposeSolve`

Breaks `ATLAS_SOLVE` into 7 typed outputs:

| Output | Type | Notes |
|---|---|---|
| `camera` | ATLAS_CAMERA | Pass to `AtlasDecomposeCamera` or blockout |
| `confidence` | FLOAT | 0.0–1.0; < 0.3 = low quality solve |
| `source_method` | STRING | e.g. `"vp_auto"`, `"constrained"` |
| `image_width` | INT | Original image pixel width |
| `image_height` | INT | Original image pixel height |
| `solve_json` | STRING | Full JSON representation |
| `horizon_angle_deg` | FLOAT | Roll of horizon in degrees (from debug_metadata) |

### `AtlasDecomposeCamera`

Exposes intrinsic and extrinsic floats. Critical output: **`cam_y`** — must be > 0 for `AtlasGroundDepthMap` to produce visible output.

### `AtlasGroundDepthMap`

Numpy port of the GLSL `DEPTH_FRAGMENT_SHADER` (`ui/src/ProjectionMaterial.ts`). Casts per-pixel rays from camera through the image plane and intersects them with the Y=0 ground plane. Output is a warm→cool heatmap (red=near, blue=far).

**Output is all black when `cam_y ≤ 0`**. If the solve placed the camera on or below the ground plane, use `AtlasConstrainedSolve` with `camera_height_m`.

Default `near_m=1.0`, `far_m=50.0`. Adjust to match your scene scale.

### `AtlasVPVisualization`

Draws VP convergence lines (left VP = orange, right VP = blue, vertical VP = green) and horizon line (yellow) as a PIL overlay. If the solve found no VPs or no horizon, the node returns the input image unchanged — this is not an error.

### `AtlasBlockoutViewport`

**Usage flow:**
1. Connect `ATLAS_SOLVE` and source `IMAGE`. Queue the prompt.
2. The Three.js viewport initialises inside the node with the recovered camera applied.
3. The source photo appears as a background reference plane in the 3D scene.
4. Click **Box / Plane / Cylinder / Person** to place geometry. Orbit with mouse drag, zoom with scroll.
5. Click **Render Proxy Passes** → four browser/WebGL proxy passes are base64-encoded into `client_data` and the prompt is re-queued automatically.
6. `shaded / depth / normal / mask` IMAGE outputs are now populated as proxy/LDR outputs.

**First run:** All four outputs are black placeholder tensors (expected — `client_data` is empty until step 5).

**Camera paths:** Use 🎥 Camera Path mode in the viewport, add keyframes, then click **Bake Proxy Path**. The `path_frames` output is an IMAGE batch for editorial/video-preview workflows; `camera_path` is the raw keyframe data for `AtlasExportCameraPathUSD`.

**Output Desk:** Add `AtlasViewportControls` and connect its first output to the viewport's `controls` input to move the toolbar/panels out of the viewport. The same node's second output, `output_profile`, carries OCIO-style metadata for labels, preview trims, and DCC/export handoff.

**Proxy warning:** viewport passes and baked path frames are browser preview data. Use `AtlasRegisterPlate` / `AtlasAttachSourcePlate` when the final source image exists as an EXR or other high-bit-depth plate.

**Three.js loading:** The extension imports a vendored local bundle, `atlas_camera/comfy/web/lib/atlas-three.bundle.js` — three.js r185 core plus `OBJLoader`/`FBXLoader` in one self-contained ESM file, committed to the repo. No internet access or npm step is needed at runtime. To upgrade three.js: bump `three` in `ui/package.json`, then `cd ui && npm install && npm run build:comfy-three` (entry: `ui/bundle/atlas-three-entry.js`) and commit the rebuilt bundle.

### `AtlasRegisterPlate`

Passes an `IMAGE` through unchanged and emits an `ATLAS_PLATE_REF` containing:

- the original file path, when supplied;
- a JPEG preview for the browser;
- colorspace, bit-depth, role, LUT path, and metadata;
- an explicit `is_proxy` flag when no final file path is available.

Use this before `AtlasAttachSourcePlate` for source plates, or feed the
`plate_ref` into patch/clean-plate nodes so DCC exporters can use the real
EXR/high-bit-depth file instead of the browser preview.

### `AtlasAttachSourcePlate`

Attaches an `ATLAS_PLATE_REF` to a solve. Exporters prefer this file-backed
plate path over copied PNG/JPEG previews, while the viewport continues to use
the lightweight preview payload.

### `AtlasViewportControls` / Atlas Output Desk

This node has two outputs:

| Output | Type | Notes |
|---|---|---|
| `controls` | ATLAS_VIEWPORT_LINK | Backward-compatible link used by the browser extension to relocate viewport controls. |
| `output_profile` | ATLAS_OUTPUT_PROFILE | OCIO-style intent: config, working/output colorspace, display/view/look, LUT, exposure, gamma, and display trim. |

The browser preview is display-inferred only. Final OCIO/LUT fidelity belongs
to ComfyUI-OCIO, Nuke, Maya, Resolve, or another color-managed tool.

### Export nodes

All export nodes write files to disk and return the path as a STRING. They produce no visual output in ComfyUI.

| Node | Output file | Location |
|---|---|---|
| `AtlasExportSolveJSON` | `atlas_solve.json` | `file_path` widget |
| `AtlasExportBlender` | `build_scene.py` | `output_dir/build_scene.py` |
| `AtlasExportNuke` | `nuke_projection.py` + `nuke_projection.nk` | `output_dir/` |
| `AtlasExportUSD` | `camera.usda` | `output_dir/camera.usda` |
| `AtlasExportReviewPackage` | Full bundle dir | `output_dir/` |

Nuke/Maya/review exports prefer attached file-backed plate refs when present
and annotate colorspace/output-profile intent. `AtlasExportReliefMesh` can
reference an external EXR plate from its OBJ/MTL; GLB remains a proxy format
with embedded PNG-style texture data.

---

## API endpoint

The `AtlasBlockoutViewport` node caches the recovered camera after each Python execution and exposes it for the browser extension:

```
GET /atlas/camera_data/{node_id}
Content-Type: application/json

{
  "view_matrix": [[r00,r01,r02,tx], ...],  // 4×4 row-major Atlas view matrix
  "fx": 1234.5, "fy": 1234.5,
  "cx": 960.0,  "cy": 540.0,
  "camera_position": [x, y, z],
  "image_width": 1920, "image_height": 1080,
  "target_width": 512, "target_height": 512,
  "focal_mm": 35.0, "sensor_mm": 36.0,
  "source_image_b64": "data:image/jpeg;base64,..."
}
```

Cache is capped at 64 entries (LRU-evict oldest) to prevent unbounded growth in long sessions.

---

## Known issues and debugging status (2026-07-02)

### 1. Depth map black for auto-solve

**Symptom:** `AtlasGroundDepthMap` outputs a solid black image.

**Root cause:** `cam_y ≤ 0` in the solved camera. The depth compute function requires the camera to be above the Y=0 ground plane (`valid = (abs(ry) > 1e-5) & (cam_y > 0)`). Without a scale constraint, the solver may place the camera at the coordinate origin.

**Debug:** Connect `AtlasDecomposeCamera.cam_y` to a Primitive node to read its value.

**Fix:** Use `AtlasConstrainedSolve` with `{"scale_constraints": [{"type": "camera_height_m", "value": 1.6}]}`.

**Status:** Known, by design. Auto-solve with no scale reference cannot determine camera height.

---

### 2. VP visualization passes through unchanged

**Symptom:** `AtlasVPVisualization` preview shows the source image with no overlaid lines.

**Root cause:** `solve.vanishing_points` is empty (no VPs detected) or all VPs project outside the image bounds. Common for AI-generated images or heavily occluded scenes.

**Fix:** Use a real photograph with strong rectilinear geometry. Alternatively, draw guide lines in the React UI and export `constraints.json` for use with `AtlasConstrainedSolve`.

**Status:** Not a bug — expected behaviour for low-cue images.

---

### 3. Blockout viewport — Three.js canvas not visible

**Symptom:** The `AtlasBlockoutViewport` node shows the button toolbar but no 3D canvas above it.

**Likely causes:**
- Three.js bundle failed to load (`lib/atlas-three.bundle.js` missing from the extension's web dir — e.g. a partial checkout; rebuild with `cd ui && npm run build:comfy-three`)
- `addDOMWidget` not supported in this version of ComfyUI (requires ComfyUI ≥ 0.2.x)
- JavaScript console error during `buildNodeUI()`

**Debug:** Open browser DevTools → Console while ComfyUI is running. Look for `[AtlasBlockout]` prefixed errors.

**Status:** Under test. The canvas area appears dark (black) which may be correct — Three.js initialises with a dark grey background. Adding a Box primitive and orbiting should confirm.

---

### 4. aiohttp HEAD route conflict on startup

**Symptom:** ComfyUI startup traceback: `RuntimeError: Added route will never be executed, method HEAD is already registered`.

**Root cause:** `atlas_camera/comfy/__init__.py` is executed twice — once as `AtlasCamera` (ComfyUI custom node) and once as `atlas_camera.comfy` (Python package import). Both runs attempt to register `GET /atlas/camera_data/{node_id}`, which also registers HEAD; the second registration conflicts.

**Fix (applied):** Guard in `__init__.py`:
```python
if not any(getattr(r, "path", None) == _ATLAS_ROUTE_PATH for r in _routes):
    @_routes.get(_ATLAS_ROUTE_PATH)
    async def _atlas_get_camera_data(...): ...
```

**Status:** Fixed. If this error reappears, check whether another custom node has registered a route with the same path pattern.

---

### 5. Blockout Render Passes cycle (first run always blank)

**Symptom:** After clicking Render Passes the blockout outputs remain black.

**Root cause:** This is the expected two-pass cycle:
- Pass 1: `AtlasBlockoutViewport.render()` is called with empty `client_data` → returns blank tensors, writes camera data to cache.
- Browser: Extension detects `node.onExecuted`, fetches camera from `/atlas/camera_data/{id}`, applies to Three.js camera, background loads.
- User: Places geometry, clicks Render Passes → fills `client_data`, auto-queues prompt.
- Pass 2: `render()` is called with populated `client_data` → decodes base64 → returns real tensors.

**Status:** By design. The blank first pass is intentional.

---

## Testing checklist for co-worker

Work through these in order. Each step depends on the previous passing.

- [ ] **ComfyUI starts without errors** — no aiohttp traceback, Atlas Camera nodes visible in node browser
- [ ] **Symlink resolves** — `custom_nodes\AtlasCamera` → `atlas_camera\comfy` (check in Explorer or `Get-Item`)
- [ ] **Package importable** — `& venv\Scripts\python.exe -c "import atlas_camera; print(atlas_camera.__file__)"` returns project path
- [ ] **Load example workflow** — drag `examples/atlas_camera_full_workflow.json` into ComfyUI, all nodes appear with correct connections
- [ ] **Auto-solve with good image** — load an exterior real photograph; queue; `AtlasVPVisualization` preview shows VP lines overlaid
- [ ] **cam_y > 0** — wire `AtlasDecomposeCamera.cam_y` to a Primitive node; value should be > 0.1 for a ground-level photo
- [ ] **Depth map non-black** — `AtlasGroundDepthMap` preview shows warm-to-cool gradient on ground pixels, black on sky
- [ ] **Blockout camera applies** — after first queue, Three.js viewport should show the source image background and camera roughly aligned to the photo
- [ ] **Blockout render passes** — click Box, orbit the scene, click Render Passes; second queue populates all four `PreviewImage` outputs
- [ ] **Export files written** — check that `atlas_solve.json` and `atlas_exports/` exist in the ComfyUI working directory after a successful queue

---

## File map

```
atlas_camera/comfy/
  __init__.py          ← WEB_DIRECTORY, API route registration (with double-import guard)
  nodes.py             ← All 18 node classes + helpers (_extract_blockout_camera, _ground_depth_compute, etc.)
  web/
    atlas_blockout.js  ← ComfyUI frontend extension (Three.js viewport, camera apply, render passes)

examples/
  atlas_camera_full_workflow.json   ← Load this in ComfyUI to test everything
```
