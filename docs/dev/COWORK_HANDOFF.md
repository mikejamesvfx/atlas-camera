# Atlas Camera — Co-Worker Handoff (updated 2026-07-02, end of session)

This document is written for a fresh Claude Code instance picking up testing of the
ComfyUI integration. It was originally written at the *start* of 2026-07-02 describing
an 18-node integration; a long session of feature work followed (learned camera prior,
metric-scale cascade, depth-derived projection geometry, matte-painting projection,
relief-mesh export, viewport diagnostics). This revision brings it back in sync — see
[CLAUDE.md](../CLAUDE.md) for the authoritative, continuously-maintained architecture
reference and [docs/USER_GUIDE.md](USER_GUIDE.md) for the artist/TD-facing walkthrough
of what the tool actually does. This file stays useful as the narrower "what's been
manually verified inside a live ComfyUI session vs. only unit-tested" status board.

---

## What this project is

**Atlas Camera** is a camera-recovery tool for VFX/3D artists, purpose-built to be robust
on **AI-generated images**, not just real photographs. Given a still photo, it estimates
the camera's focal length, position, and orientation, and (via a tiered, confidence-gated
cascade) a metric scale for the scene. From there it can derive simple 3D geometry from a
monocular depth map and **project the source photo onto it from the recovered camera** —
the matte-painting technique — live inside a ComfyUI viewport, or exported as a UV-baked
mesh for Maya/Nuke/ZBrush/Blender.

The ComfyUI integration now registers **25 nodes** (see CLAUDE.md's node catalog table for
the full, current list — do not trust a node count anywhere else, including this doc, over
the actual `NODE_CLASS_MAPPINGS` in `atlas_camera/comfy/nodes.py`).

---

## Repository location

Paths below are illustrative — substitute your own checkout/install locations. Do not bake
absolute paths into this file again; that was a real portability bug fixed in CLAUDE.md
during this session and it isn't worth reintroducing here.

```
<REPO_ROOT>\                                    # this repository
<COMFYUI_ROOT>\custom_nodes\AtlasCamera          # symlink -> <REPO_ROOT>\atlas_camera\comfy
```

GitHub: `miikejamesburns/atlasCamera`.

---

## Environment

| What | Where |
|---|---|
| Project repo | `<REPO_ROOT>` |
| ComfyUI install | `<COMFYUI_ROOT>` |
| ComfyUI Python | `<COMFYUI_ROOT>\venv\Scripts\python.exe` |
| Custom nodes dir | `<COMFYUI_ROOT>\custom_nodes\` |
| Symlink | `custom_nodes\AtlasCamera` → `<REPO_ROOT>\atlas_camera\comfy` |

The package is installed editable in ComfyUI's venv — changes to `.py` files are live
without reinstall. JS changes require a ComfyUI restart. Nodes under the `[neural]` extra
(learned solve, Depth Anything, derive-projection-geometry, relief-mesh export) additionally
need `pip install -e ".[neural]"` plus GeoCalib (`pip install "git+https://github.com/cvg/GeoCalib.git"`)
in that same venv.

---

## Current state: what works, what needs testing

### Confirmed working (unit/headless-verified — `python -m pytest -q`, 159 passed / 1 skipped)
Every node's underlying core logic (solving, scale cascade, geometry derivation, relief
mesh, exporters) is covered by the test suite and passes. This is strong evidence the
*math* is correct; it is not the same as confirming the *ComfyUI node UI/UX* works — see
below for what's actually been run inside a live ComfyUI session this session.

### Confirmed working inside a live ComfyUI session (manually verified this session)
- Full pipeline: `LoadImage` → `AtlasLearnedSolveFromImage` → `AtlasDeriveProjectionGeometry`
  (relief mesh) → `AtlasBlockoutViewport` (📽 Project) → `AtlasExportReliefMesh`, using
  `examples/atlas_camera_core_projection_workflow.json`. Two real bugs were found and fixed
  via this live testing (not from unit tests): the blockout viewport never received solve
  data (missing `"ui"` payload key — see the "ComfyUI's `"ui"` payload requirement" note in
  CLAUDE.md) and the projected mesh/backdrop appearing to "disappear" on orbit (fixed via
  always including the backdrop regardless of `geometry_mode`, plus yaw/pitch orbit clamping).
- The original 18-node vanishing-point workflow (`examples/atlas_camera_full_workflow.json`)
  was confirmed loading and running at the start of this session (pre-dating the work below).

### Not yet manually verified in a live ComfyUI session (unit-tested only)
These were built and pass their own test suites, but have not been confirmed by a human
clicking through them in the actual ComfyUI UI:
1. `AtlasReferenceScaleSolve` / `AtlasApplyScaleReferences` / `AtlasVLMScaleCues` end-to-end
   in the node graph (the VLM node additionally needs a live local LM Studio/Ollama/llama.cpp
   server — untested against a real one in this session).
2. `AtlasDeriveProjectionGeometry` with `primitive_method="ransac_planes"` or `"room_cuboid"`
   inside the live viewport (verified headlessly against real test images — see
   `tests/test_plane_extraction.py`, `tests/test_room_layout.py` — but not clicked through
   in ComfyUI with the `primitives`/`both` `geometry_mode`).
3. The three viewport diagnostics added late in this session — ☀ Exposure slider, 📊 VP/
   horizon/ground SVG diagram, ℹ camera metadata HUD — were verified via `node --check` and
   standalone JS math simulations only (no live browser available in that part of the
   session). Worth a real click-through, especially the VP diagram on the classical
   vanishing-point solve path (it's expected to render empty on the learned/GeoCalib path
   by design — see CLAUDE.md's diagnostics section).
4. `AtlasExportReliefMesh`'s GLB output opened in an actual DCC (Maya/Nuke/ZBrush/Blender) —
   verified structurally (valid glTF 2.0, correct binary layout) but not opened in a DCC.

---

## Testing instructions

### Step 1 — Verify startup

Start ComfyUI. The console should show Atlas Camera loading without errors. In the node
browser, confirm the **Atlas Camera** category exists with 25 nodes.

If you see `RuntimeError: method HEAD is already registered` → the double-import guard in
`atlas_camera/comfy/__init__.py` failed. See CLAUDE.md's "Double-import guard (critical)"
section — the fix pattern is `if not any(r.path == ... for r in _routes): ...` around each
route registration, and there are now two routes guarded that way (`camera_data`,
`proxy_model`).

### Step 2 — Load and run the minimal workflow

1. Open `examples/atlas_camera_core_projection_workflow.json` (drag into ComfyUI). This is
   the smallest workflow that exercises the session's core contribution — see CLAUDE.md's
   "Example workflows" section for what each of the three example workflows covers.
2. In `LoadImage`, upload any image — the learned (GeoCalib) solve path this workflow uses
   is specifically designed to be robust on AI-generated images, unlike the classical VP path.
3. Queue the prompt.

Expected: `AtlasBlockoutViewport` shows the source photo as a background plane with the
recovered camera pose; clicking 📽 Project shows the derived relief mesh textured with the
source photo, aligned from the 📷 Camera View angle.

### Step 3 — Test the metric-scale cascade

Swap in `AtlasReferenceScaleSolve` after the solve node, or set
`AtlasLearnedSolveFromImage`'s `height_mode` to `measure_from_depth`, to exercise scale
tiers 1 and 2 respectively (see CLAUDE.md's "Camera height / metric scale" key design rule
for the full tiered cascade). Check `solve.debug_metadata["scale_source"]` (via
`AtlasDecomposeSolve`) to confirm which tier actually won.

### Step 4 — Test blockout viewport diagnostics (not yet live-verified — see above)

1. After a solve, open the ☀/📊/ℹ toolbar controls on `AtlasBlockoutViewport`.
2. ☀ Exposure should only affect the lit grey preview, never the 📽 projected photo or the
   depth/normal/mask render passes.
3. 📊 Diagram should overlay horizon/ground on any solve, and VP fan-lines only on the
   classical vanishing-point path (`AtlasSolveFromImage`, `detect_vanishing_points=True`).
4. ℹ Info HUD should show solved lens/height/confidence/scale-tier text.

### Step 5 — Verify exports

After a successful queue, check for output files (location depends on the export node's
`output_dir` widget, typically under the ComfyUI working directory):

- `atlas_solve.json` (`AtlasExportSolveJSON`)
- `atlas_exports/build_scene.py` (Blender), `.../camera.usda` (USD — needs `usd-core`),
  `.../*.obj`+`.mtl`+texture or `.glb` (`AtlasExportReliefMesh`)
- review package bundle (`AtlasExportReviewPackage`)

---

## Key files for debugging

| File | Purpose |
|---|---|
| `atlas_camera/comfy/nodes.py` | All Python node logic (25 node classes + `NODE_CLASS_MAPPINGS`). Key helpers: `_extract_blockout_camera()`, `_ground_depth_compute()`, `AtlasBlockoutViewport.render()` — grep for these rather than trusting line numbers, which drift as the file grows |
| `atlas_camera/comfy/__init__.py` | Route registration (`/atlas/camera_data/{node_id}`, `/atlas/proxy_model/{name}`) + double-import guard |
| `atlas_camera/comfy/web/atlas_blockout.js` | Three.js frontend extension — camera sync, orbit controls, projection material, diagnostics overlay |
| `atlas_camera/core/solver.py` | `solve_still_image()`, `solve_still_image_learned()`, `solve_from_constraints()`, the metric-scale cascade |
| `atlas_camera/core/proxy_geometry.py`, `depth_geometry.py`, `plane_extraction.py`, `room_layout.py`, `relief_mesh.py` | The three geometry-derivation strategies and the shared depth back-projection helpers they're built from |
| `atlas_camera/core/schema.py` | `AtlasSolve`, `AtlasCamera`, `AtlasIntrinsics`, `AtlasExtrinsics`, `AtlasProxyPrimitive` dataclasses |

---

## Architecture notes that matter for testing

**Camera convention:** always use the full 4×4 `extrinsics.camera_view_matrix`
(`cam_to_world = inv(view_matrix)`, row-major, column-vector points), never the 3×3
`camera_rotation_matrix` — it has a transpose ambiguity. This is CLAUDE.md's single most
load-bearing convention; every geometry-derivation module added this session depends on it.

**Metric scale is measured, not assumed**, via the tiered cascade (reference object → depth
ground-plane → assumed default). `solve.debug_metadata["scale_source"]` records which tier
won — check this before trusting a solve's absolute measurements.

**Blockout viewport requires the `"ui"` payload key.** `AtlasBlockoutViewport.render()` must
return `{"ui": {...}, "result": (...)}`, not a plain tuple, or ComfyUI never fires the
`executed` websocket event and the frontend never receives camera data. This was a real bug
found and fixed this session — if a future node's viewport data mysteriously never arrives,
check this first.

**`AtlasDecomposeSolve` outputs `horizon_angle_deg`** (from `solve.debug_metadata`).
`AtlasDecomposeCamera` does NOT have this output — horizon data only lives on the solve, not
the camera.

---

## What NOT to change during testing

- Do not modify `ui/src/` casually — the React frontend is independent of the ComfyUI work
  and has its own test suite (`ui/` — Vitest).
- `atlas_camera/core/` is **no longer off-limits** — it was extended substantially and
  correctly this session (geometry derivation, relief mesh, learned-prior solving, metric
  scale). The earlier version of this doc said not to touch it; that guidance predates this
  session's work and should not be followed. If you touch it, run the full suite
  (`python -m pytest -q --ignore=tests/test_usd_exporter.py`) before and after — it should
  stay at 159 passed / 1 skipped (the skip is a real `usd-core`-dependent test, unrelated).
- Do not bind `python -m atlas_camera.ui` to a non-loopback host without `--allow-remote` —
  the local UI's `project_dir` is an intentionally arbitrary client-supplied path (a file
  picker, by design) with no further access control; see `atlas_camera/ui/__main__.py`.
