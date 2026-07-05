# Atlas Camera — single-image camera recovery + matte-painting projection geometry (AtlasCamera pack)

**26 nodes** that recover a camera from ONE photo (real or AI-generated), derive projectable 3D geometry from it,
project the photo onto that geometry live (matte-painting style), and hand the result off to Maya / Nuke / Blender /
USD. Pack: **`AtlasCamera`** (category **`Atlas Camera`** + `Atlas Camera/Export`), author **Miike Burns**
(https://github.com/miikejamesburns/atlasCamera). Built to fill a real gap: the registry has no pack that turns a
single still into a solved camera + textured projection proxies for VFX handoff (`search_custom_nodes "camera
projection"` / `"matte painting"` return nothing comparable). Zero-required-dep core; the solve/depth/derive/patch/
relief nodes need the **`[neural]`** extra (torch + GeoCalib + Depth Anything V2).

**The through-line (what this pack is for):** `LoadImage` → **AtlasLearnedSolveFromImage** (recover the camera) →
**AtlasDeriveProjectionGeometry** (depth → relief mesh / fitted primitives) → **AtlasBlockoutViewport** (hit 📽
Project to see the photo projected onto the 3D live) → **AtlasExportReliefMesh** / **AtlasExportMayaReviewScene**
(OBJ+GLB / Maya scene with camera + projected geometry). Everything hangs off the recovered camera; the same solve
object flows the whole way.

**Why it works even on wrong-ish depth (the matte-painting property):** projection assigns texels by *ray* through
the recovered camera, invariant to distance along the ray — geometry at slightly wrong depth still receives exactly
the pixels its silhouette subtends, so the image reassembles perfectly from Camera View; scale error only shows as
parallax when you orbit. This is why AI-generated images (non-physical perspective) still give usable projection
setups.

**Install:** symlink `atlas_camera/comfy` into ComfyUI's `custom_nodes/AtlasCamera`, then in ComfyUI's venv:
`pip install -e ".[neural]"` and `pip install "git+https://github.com/cvg/GeoCalib.git"` (torch expected from the
host venv). Restart ComfyUI (node classes + the browser Three.js extension load at startup). Verify:
`get_node_info AtlasLearnedSolveFromImage`.

## Two custom wire types the agent MUST respect

- **`ATLAS_SOLVE`** — the recovered scene (camera + horizon + confidence + derived `projection_scene` geometry +
  any patch `projection_sources`). An opaque Python object that flows **only between Atlas nodes**
  (solve → scale → derive → add-patch → viewport / export / decompose). You CANNOT wire it into a stock ComfyUI
  node. To read scalars out of it, use **AtlasDecomposeSolve** / **AtlasDecomposeCamera**.
- **`ATLAS_CAMERA`** — a decomposed camera object (from AtlasDecomposeSolve or AtlasUSDCameraLoader), consumed by
  AtlasDecomposeCamera. Also Atlas-only.
- **`IMAGE` / `MASK`** are standard ComfyUI types — solve/derive/viewport/patch nodes take a normal `IMAGE`
  (from `LoadImage`, `VAEDecode`, etc.); the depth/mask/VP-viz nodes emit standard `IMAGE`/`MASK` you can feed
  anywhere (ControlNet, compositing, PreviewImage).

**Scale is measured, not assumed (tiered, best evidence first):** (1) reference object
(AtlasReferenceScaleSolve / VLM→AtlasApplyScaleReferences), (2) depth ground-plane
(AtlasLearnedSolveFromImage `height_mode=measure_from_depth`), (3) assumed default (flagged). Metric scale only
affects real-world sizing; the projection itself is scale-invariant.

**I/O status:** node I/O below is **confirmed from the pack source (each node's `INPUT_TYPES`) 2026-07-03**; for the
exact live inputs/outputs/defaults of the installed version, pull `get_node_info <ClassType>` — this entry holds the
durable semantics `/object_info` can't give (what each field is FOR, how to wire the pipeline, gotchas).

---

## Solve — recover the camera (start here)

### AtlasLearnedSolveFromImage  (display: "Atlas Learned Solve (GeoCalib) 🧠")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera` | **needs `[neural]`**
- **purpose:** recover the camera (focal + gravity/orientation) from one image using the learned GeoCalib prior.
  The **robust default for AI-generated images** (Flux/Qwen/SD) whose perspective is only locally consistent.
- **inputs:**
  - `image` (`IMAGE`) — the source photo (ComfyUI tensor; accepts `LoadImage`/`VAEDecode` directly).
  - `height_mode` (combo) — `assume` uses `camera_height_m`; `measure_from_depth` fits the ground plane in the
    Depth-Anything map to MEASURE camera height (no assumed eye height) and fills the solve's depth slot.
  - `camera_height_m` (FLOAT, default 1.6) — used only when `height_mode=assume`.
  - `depth_model` (combo Outdoor/Indoor) — only used by `measure_from_depth`. Match the scene.
  - `sensor_width_mm` (FLOAT, default 36) — for focal-mm reporting.
  - `weights` (combo, default `pinhole`), `device` (auto/cuda/cpu).
- **outputs:** `ATLAS_SOLVE` — the recovered scene; feed to AtlasDeriveProjectionGeometry / scale / decompose /
  viewport / export.
- **how it works:** GeoCalib predicts focal + gravity directly (no vanishing points), then Atlas builds the Y-up
  right-handed camera; `measure_from_depth` reconciles metric height from the depth ground fit.
- **strengths:** works where geometric VP solving fails — AI renders, low-line-count scenes, organic environments.
- **anti-patterns:** VP-based diagnostics are empty on this path (no vanishing points computed — that's expected;
  AtlasVPVisualization/the viewport VP layer stay blank). Needs `[neural]`.
- **placement:** the front of every Atlas pipeline for photographic/AI input.

### AtlasSolveFromImage  (display: "Atlas Solve Camera from Image")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera`
- **purpose:** deterministic geometric solve via vanishing-point detection. Fast, dependency-light, but fragile on
  AI images (contradictory perspective) — prefer AtlasLearnedSolveFromImage for those.
- **inputs:** `image` (`IMAGE`); `focal_length_mm`/`sensor_width_mm` (optional hints); `detect_vanishing_points`
  (BOOLEAN, **default ON** — off yields a metadata-only solve with no usable camera).
- **outputs:** `ATLAS_SOLVE`. **placement:** front of the pipeline for clean architectural real photos; populates
  the viewport's VP/horizon diagram.

### AtlasConstrainedSolve  (display: "Atlas Constrained Solve")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera`
- **purpose:** artist-guided solve from a JSON of line groups / scale constraints (manual matchmove-lite for hard
  shots). **inputs:** `image`, `constraints_json` (STRING), `±focal_length_mm`/`±sensor_width_mm`.
  **outputs:** `ATLAS_SOLVE`.

### AtlasLoadSolveJSON  (display: "Atlas Load Solve JSON")  /  AtlasLoadImageSolveCamera (legacy, file-path solve)
- Load a previously saved solve (`json_path` → `ATLAS_SOLVE`); the legacy loader solves from an image path + size.

## Scale — pin metric camera height (optional, composable after any solve)

### AtlasReferenceScaleSolve  (display: "Atlas Reference-Object Scale 📏")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera` | no extra deps
- **purpose:** set metric camera height from a known-size object (person/door/car from the reference registry, or
  an explicit height) via single-view geometry — the most reliable scale tier.
- **inputs:** `solve` (`ATLAS_SOLVE`), `reference_id` (STRING, a registry id), `bbox_x0/y0/x1/y1` (the object's
  pixel bbox), `±height_override_m`. **outputs:** `ATLAS_SOLVE` (rescaled), `camera_height_m` (FLOAT).
- **placement:** between solve and derive when you have a scale reference in-frame.

### AtlasVLMScaleCues → AtlasApplyScaleReferences  (displays: "Atlas VLM Scale Cues 👁", "Atlas Apply Scale References ✅")
- **VLM cues:** `image` → detects known-size objects with a local VLM (`provider` ollama/lmstudio/llamacpp) →
  `scale_references` (STRING json) + `summary`. Fails soft to `[]` if no local server.
- **Apply:** `solve` + `scale_references` → rescales metric height **only when `confirm` is ON** (else records
  candidates; LLM cues are never auto-promoted). Outputs `solve`, `camera_height_m`, `report`.

## Derive — turn the solve into projectable 3D

### AtlasDeriveProjectionGeometry  (display: "Atlas Derive Projection Geometry 📽")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera` | **needs `[neural]`** (re-runs metric depth internally)
- **purpose:** derive the depth **relief mesh** (default) and/or fitted primitives into the solve's
  `projection_scene`, so the viewport/exporters have geometry to project onto.
- **inputs:**
  - `solve` (`ATLAS_SOLVE`), `image` (`IMAGE`, the SAME source photo — the normalized depth IMAGE from
    AtlasDepthAnything is NOT usable here; this node runs metric depth itself).
  - `geometry_mode` (combo `relief_mesh` (default) / `primitives` / `both`) — output kind.
  - `primitive_method` (combo, when primitives) — `azimuth_walls` (vertical walls, truncates sloped roofs) /
    `ransac_planes` (any-orientation planes — exteriors) / `room_cuboid` (Manhattan interiors) /
    `vertical_extrusion` (walls extruded to the real image silhouette — towers/spires/roofs). **Artist-picked,
    never auto-detected.**
  - `scene_type` (combo `manual`/`organic`/`indoor`/`outdoor`) — one-choice preset over
    geometry_mode+primitive_method+depth_model.
  - `depth_model` (Outdoor/Indoor), `max_walls`, `max_objects`, `relief_grid`, `device`.
- **outputs:** `ATLAS_SOLVE` (deep-copied, geometry attached; JSON grows ~1MB with the mesh).
- **anti-patterns:** feeding the AtlasDepthAnything output as `image` (it's normalized/unusable for metric geometry
  — feed the original photo). Picking the wrong `primitive_method` for the shot (e.g. `room_cuboid` on a
  non-orthogonal exterior → skewed walls).
- **placement:** after solve (+ optional scale), before the viewport / relief export.

### AtlasAddPatchView  (display: "Atlas Add Patch View (multi-angle) 🩹")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera` | **needs `[neural]`**
- **purpose:** fill areas the primary camera can't see (occluded/grazing → black on orbit) by adding an AI
  novel-view "patch": a view of the same scene generated at a defined angle (Qwen-Image-Edit-2511 Multiple-Angles
  LoRA, e.g. via the `ComfyUI-qwenmultiangle` "Qwen Multiangle Camera" node). Constructs a patch camera by orbiting
  the recovered camera to that view, derives the patch's own relief geometry in that frame, appends it as a
  projection source. Chain one per angle (left/right/above).
- **inputs:** `solve` (`ATLAS_SOLVE`), `patch_image` (`IMAGE`, the novel view). **The LoRA angles are ABSOLUTE
  (subject-relative), so set BOTH** `patch_*_view` (what you asked the LoRA for — must match the Qwen Multiangle
  Camera node) **and** `source_*_view` (what your source photo already is; "front view" for a straight-on shot) —
  orbit applied = patch − source. `flip_azimuth` corrects mirrored left/right. `name`, `depth_model`, `relief_grid`,
  `priority`, `device`.
- **outputs:** `ATLAS_SOLVE` with a `projection_source` appended; the viewport layers it over the primary with a
  facing-ratio mask.
- **anti-patterns:** setting the patch's named view but leaving `source_azimuth_view` wrong for an oblique source
  (the orbit is patch−source, not the absolute angle). This is estimation, not photogrammetry — expect
  matte-painting-plausible fill, verified in the viewport.
- **placement:** after AtlasDeriveProjectionGeometry, before the viewport; the patch IMAGE comes from a
  Qwen-Image-Edit + Multiple-Angles-LoRA subgraph.

## Project & inspect — the browser viewport

### AtlasBlockoutViewport  (display: "Atlas Viewport 🧊")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera/Blockout` | **OUTPUT_NODE**
- **purpose:** a browser-side Three.js 3D viewport initialised to the recovered camera. **Hit 📽 Project** to swap
  the derived geometry (+ any patch sources) between grey blockout and the live camera projection of the source
  photo. Also: 🧍/🚗 correctly-sized proxies (fastest scale check), Camera View reset, VP/horizon/ground diagram,
  info HUD, 4-pass render (shaded/depth/normal/mask).
- **inputs:** `solve` (`ATLAS_SOLVE`), `source_image` (`IMAGE`, the original photo — the projection texture),
  `resolution` (INT, long edge; W×H auto-follows the source aspect), `client_data` (STRING, filled by the
  "Render Passes" button), `±preview_expand` (default 1.0 = off; >1 dilates geometry for wider grey-preview orbit
  but **conflicts with 📽 Project** — leave at 1.0 when projecting).
- **outputs:** `shaded`, `depth`, `normal`, `mask` (all `IMAGE`, populated by the browser "Render Passes" button →
  feed ControlNet / compositing / PreviewImage).
- **how it works:** on execution fetches the recovered camera + source image + geometry from a backend cache and
  builds the scene; orbit clamped to the reconstructed cone; each patch source projects its own camera+image with
  a facing-ratio discard. **Requires a browser refresh after any JS update.**
- **anti-patterns:** `preview_expand > 1.0` with 📽 Project (dilated geometry has no real pixels → black on orbit).
  Wiring the depth/normal/mask outputs before clicking "Render Passes" (they're empty until then).
- **placement:** the interactive terminus of the pipeline (validate the solve/geometry/projection before export).

## Export — hand off to DCC

### AtlasExportReliefMesh  (display: "Atlas Export Relief Mesh (OBJ) 🗻")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera/Export` | **OUTPUT_NODE** | **needs `[neural]`**
- **purpose:** triangulate the metric depth into a world-space relief mesh, torn at silhouettes, **camera
  projection baked into per-vertex UVs** — imports already textured with the source photo into
  Maya/Nuke/ZBrush/Blender, ready to retopo/reproject.
- **inputs:** `solve`, `image` (original photo), `output_dir`; `±grid_long_edge`, `±depth_edge_rel`, `±depth_model`,
  `±device`, `±format` (both/obj/glb). **outputs:** `obj_path`, `glb_path` (STRING). OBJ = +MTL+PNG; GLB =
  single binary, texture embedded, KHR_materials_unlit. Ground on Y=0.
- **placement:** after derive; the `obj_path` also feeds AtlasExportMayaReviewScene's `relief_mesh_obj_path`.

### AtlasExportMayaReviewScene  (display: "Atlas Export Maya Review Scene")
- **pack:** `AtlasCamera` | **category:** `Atlas Camera` | **OUTPUT_NODE**
- **purpose:** write a Maya Python scene-builder (camera + image plane + camera-projection shader network + real
  box/cylinder/plane proxies with correct dimensions/transforms; imports the relief mesh when wired).
- **inputs:** `solve`, `output_dir`, `±relief_mesh_obj_path` (wire AtlasExportReliefMesh's `obj_path` here to
  include the real relief mesh, else it's omitted). **outputs:** `STRING` (maya_open_scene.py path).

### AtlasExportUSD / AtlasExportBlender / AtlasExportNuke  (Atlas Camera/Export, OUTPUT_NODE)
- `solve` + `output_dir` → a `usd_path` / Blender `build_scene.py` / Nuke `nuke_projection.py`. Camera + (Nuke)
  a Project3D card. Single-projection today.

### AtlasExportSolveJSON / AtlasExportReviewPackage  (OUTPUT_NODE)
- Persist the solve JSON, or a full review bundle (JSON + Maya/Blender/Nuke scripts + optional USD + report.md).

## Analysis & decompose — read data out (all take an ATLAS_SOLVE, emit stock types)

| Node | I/O | Purpose |
|------|-----|---------|
| `AtlasDepthAnything` | `image` → `depth_image` (IMAGE) | Monocular depth (Depth Anything V2), metric or relative. Needs `[neural]`. **Do not feed its output into AtlasDeriveProjectionGeometry** (normalized). |
| `AtlasGroundDepthMap` | `solve`, W, H, near, far → `depth_image`, `ground_mask` | Ray-plane ground depth. Black if cam height ≤ 0. W/H **0 = auto**. |
| `AtlasGroundMask` / `AtlasHorizonMask` | `solve`, W, H (+feather) → `MASK` | 1 = ground / 1 = sky. W/H **0 = auto**. |
| `AtlasVPVisualization` | `image`, `solve` → `IMAGE` | Draws VP/horizon overlay (pass-through / empty on the learned solve path). |
| `AtlasDecomposeSolve` | `solve` → `camera` (ATLAS_CAMERA), confidence, source_method, W, H, solve_json, `horizon_angle_deg` | The correct place to read `horizon_angle_deg`. |
| `AtlasDecomposeCamera` | `camera` (ATLAS_CAMERA) → fx, fy, cx, cy, cam_x/y/z, focal_mm, fov_h_deg | Read intrinsics/position scalars. `cam_y` must be > 0 for a valid ground depth. |
| `AtlasUSDCameraLoader` | `usd_path`, W, H → ATLAS_CAMERA | Load a camera from USD. |

## bugs / lags + fixes
- **New nodes / JS not appearing:** node class attrs are read at ComfyUI startup and the Three.js extension loads
  in the browser once — **restart ComfyUI after install/update, hard-refresh the browser** for viewport changes.
- **Everything downstream black:** almost always the solve failed to produce a camera (e.g. `AtlasSolveFromImage`
  with `detect_vanishing_points` off, or an AI image the geometric solver mis-pitched) — use
  AtlasLearnedSolveFromImage. Confirm with AtlasDecomposeCamera (`cam_y` > 0, sane `focal_mm`).
- **Export nodes silently do nothing when their output is unwired:** all export nodes + the viewport are
  `OUTPUT_NODE` so ComfyUI keeps them alive even as terminal leaves (fixed; do not re-wire just to force execution).

## anti-patterns (pack-wide)
- Wiring `ATLAS_SOLVE` / `ATLAS_CAMERA` into a stock node — they're Atlas-only; decompose to scalars/IMAGE first.
- Feeding the AtlasDepthAnything IMAGE into AtlasDeriveProjectionGeometry/AtlasExportReliefMesh (feed the ORIGINAL
  photo; those nodes run metric depth themselves).
- Expecting VP diagrams on the learned (GeoCalib) solve path (empty by design).
- `preview_expand > 1.0` while using 📽 Project.

## placement (where this pack slots in a graph)
This is a **recovery + projection-prep** pack, not a generator: it sits AFTER an image exists (a `LoadImage` of a
real photo, or a `VAEDecode` from any generation graph) and BEFORE a DCC handoff. Canonical wiring:
`LoadImage/VAEDecode → AtlasLearnedSolveFromImage → [AtlasReferenceScaleSolve] → AtlasDeriveProjectionGeometry →
[AtlasAddPatchView ×N] → AtlasBlockoutViewport (📽) and/or AtlasExportReliefMesh → AtlasExportMayaReviewScene`.
Example workflows ship in the repo's `examples/` (core projection, learned full pipeline, multi-angle patch proof).

## author
Pack by **Miike Burns** (https://github.com/miikejamesburns/atlasCamera). This library entry: confirmed from the
pack source `INPUT_TYPES` 2026-07-03; pull `get_node_info <ClassType>` for the exact live I/O of the installed
build.
