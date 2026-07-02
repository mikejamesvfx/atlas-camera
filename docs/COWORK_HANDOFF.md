# Atlas Camera — Co-Worker Handoff (2026-07-02)

This document is written for a fresh Claude Code instance picking up testing of the ComfyUI integration.

---

## What this project is

**Atlas Camera** is a camera-recovery tool for VFX/3D artists. Given a still photo, it estimates the camera's focal length, position, and orientation (vanishing-point solve). Artists use this to match-move blockout geometry or set up DCC projection scenes.

The ComfyUI integration adds 18 nodes so the solve can live inside a ComfyUI graph alongside Stable Diffusion, ControlNet, etc.

---

## Repository location

```
C:\Users\miike\Desktop\AtlasCamera_Claude\
```

GitHub: `miikejamesburns/AtlasCamera` (main branch, up to date as of 2026-07-02).

---

## Environment

| What | Where |
|---|---|
| Project repo | `C:\Users\miike\Desktop\AtlasCamera_Claude\` |
| ComfyUI install | `C:\Users\miike\ComfyUI_V91\ComfyUI\` |
| ComfyUI Python | `C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe` |
| Custom nodes dir | `C:\Users\miike\ComfyUI_V91\ComfyUI\custom_nodes\` |
| Symlink | `custom_nodes\AtlasCamera` → `AtlasCamera_Claude\atlas_camera\comfy` |

The package is installed editable in ComfyUI's venv — changes to `.py` files are live without reinstall. JS changes require ComfyUI restart.

---

## Current state: what works, what needs testing

### Confirmed working
- All 18 nodes load without error in ComfyUI node browser
- `AtlasSolveFromImage` runs and returns `ATLAS_SOLVE`
- `AtlasDecomposeSolve` / `AtlasDecomposeCamera` decompose the solve correctly
- `AtlasVPVisualization` returns image (with or without VP overlays)
- `AtlasExport*` nodes write files to disk
- Example workflow JSON loads cleanly: `examples/atlas_camera_full_workflow.json`
- aiohttp double-import crash is fixed

### Needs testing / not yet verified
1. **`AtlasGroundDepthMap` producing non-black output** — requires a real photo where the solver finds `cam_y > 0`. Has only been tested with an AI-generated pipe scene (which gave `cam_y ≈ 0` → black output). Needs a real exterior photo.
2. **`AtlasBlockoutViewport` Three.js canvas rendering** — toolbar buttons visible; canvas area dark. Unclear if Three.js initialised (dark background) or failed silently. Need to add geometry and attempt Render Passes.
3. **Blockout source image background** — after a solve runs, the source photo should load as a reference plane behind the Three.js geometry. Not yet confirmed.
4. **`AtlasConstrainedSolve` with scale constraint** — untested. Should fix the depth map for difficult images.
5. **`AtlasGroundMask` / `AtlasHorizonMask`** — not yet previewed.

---

## Testing instructions

### Step 1 — Verify startup

Start ComfyUI. The console should show Atlas Camera loading without errors.  
In the node browser, confirm **Atlas Camera** category exists with all 18 nodes.

If you see `RuntimeError: method HEAD is already registered` → the double-import guard in `atlas_camera/comfy/__init__.py` failed. Read that file; the guard should look like:
```python
if not any(getattr(r, "path", None) == _ATLAS_ROUTE_PATH for r in _routes):
    @_routes.get(_ATLAS_ROUTE_PATH)
    ...
```

### Step 2 — Load and run example workflow

1. Open `examples/atlas_camera_full_workflow.json` (drag into ComfyUI).
2. In the `LoadImage` node, upload a **real exterior photo** with a visible ground plane (street level, building corner, or similar). Avoid AI-generated images.
3. Queue the prompt.

Expected results:
- `AtlasVPVisualization` preview → source image with coloured VP convergence lines (orange=left VP, blue=right VP, yellow=horizon)
- `AtlasGroundDepthMap` preview → warm-to-cool heatmap on ground area, black on sky
- `AtlasBlockoutViewport` → dark canvas with toolbar buttons

If `AtlasGroundDepthMap` is still black, debug `cam_y` by connecting `AtlasDecomposeCamera.cam_y` to a `Primitive` node and reading its value. If it is `0.0`, the auto-solve failed to find a scale reference — see Step 3.

### Step 3 — Test with scale constraint (if depth map is black)

Replace `AtlasSolveFromImage` with `AtlasConstrainedSolve` and set `constraints_json` to:

```json
{
  "scale_constraints": [{"type": "camera_height_m", "value": 1.6}],
  "line_groups": [],
  "intrinsics_hint": {}
}
```

This tells the solver the camera was 1.6 m above ground (eye height), which forces `cam_y > 0` and makes the depth map work.

### Step 4 — Test blockout viewport

1. After a successful solve (Step 2 or 3), the `AtlasBlockoutViewport` node should show a dark Three.js canvas area above the toolbar.
2. Open browser DevTools (F12) → Console. Filter by `AtlasBlockout`. If Three.js failed to load you'll see an error here.
3. Click **Box** → a grey box should appear in the canvas.
4. Drag to orbit the view. The ground grid should be visible.
5. Click **Render Passes** → the prompt re-queues automatically.
6. After re-queue, the four `PreviewImage` nodes (Shaded / Depth / Normal / Mask) should show renders of the box.

### Step 5 — Verify exports

After a successful queue, check that these files exist in the ComfyUI working directory (usually the ComfyUI root):

- `atlas_solve.json`
- `atlas_exports/build_scene.py` (Blender)
- `atlas_exports/camera.usda` (USD — requires `usd-core` installed in the venv)
- `atlas_review_packages/` (review bundle)

USD export will silently fail without `usd-core`. Install with:
```powershell
& "C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe" -m pip install usd-core
```

---

## Key files for debugging

| File | Purpose |
|---|---|
| `atlas_camera/comfy/nodes.py` | All Python node logic. `_extract_blockout_camera()` (line ~108), `_ground_depth_compute()` (line ~135), `AtlasBlockoutViewport.render()` (line ~731) |
| `atlas_camera/comfy/__init__.py` | Route registration + double-import guard |
| `atlas_camera/comfy/web/atlas_blockout.js` | Three.js frontend extension. `buildNodeUI()`, `applyRecoveredCamera()`, `renderAllPasses()` |
| `atlas_camera/core/solver.py` | `solve_still_image()`, `solve_from_constraints()` |
| `atlas_camera/core/schema.py` | `AtlasSolve`, `AtlasCamera`, `AtlasIntrinsics`, `AtlasExtrinsics` dataclasses |

---

## Architecture notes that matter for testing

**Camera convention:** Atlas uses a row-major 4×4 view matrix, camera looks along -Z, Y-up world. When applying to Three.js, the matrix is inverted (view → camToWorld), then decomposed into position/quaternion.

**cam_y is the key diagnostic.** If it is 0 or negative, depth map is black. If it is a very large number (e.g. 1000), the solver picked up a degenerate scale. Good values for a 1.6 m camera height are roughly 0.8–3.0 depending on solver units.

**Blockout two-pass cycle:** First queue always outputs blank images (client_data empty). The browser extension loads the camera after `node.onExecuted`. User adds geometry, clicks Render Passes → auto-queues second pass with populated client_data → real images out.

**`AtlasDecomposeSolve` outputs `horizon_angle_deg`** (from `solve.debug_metadata`). `AtlasDecomposeCamera` does NOT have this output — horizon data only lives on the solve, not the camera.

---

## What NOT to change during testing

- Do not modify `atlas_camera/core/` — the solver and schema are stable.
- Do not modify `ui/src/` — the React frontend is independent of the ComfyUI work.
- If you find a bug in `nodes.py` or `atlas_blockout.js`, fix it there and restart ComfyUI (JS) or just re-queue (Python).
