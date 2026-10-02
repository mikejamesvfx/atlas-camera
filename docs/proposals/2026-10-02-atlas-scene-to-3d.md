# AtlasSceneTo3D — the layered Atlas scene as ComfyUI 3D sockets

Status: **built** 2026-10-02 (`482038a`, HDR vertex PLY `d3887a1`) — this file records the
design and, below, what was actually built where it differs.

## Why

Core ComfyUI's **Save 3D (Advanced)** (and Load3D / Preview3D Advanced) take
three sockets: `model_3d`, `model_3d_info`, `camera_info`. Atlas already holds
everything those carry — world-space layered geometry, per-layer plates, a solved
camera — but only exposes it through its own viewport and DCC exporters. One node
that emits the three sockets puts an Atlas scene into ComfyUI's native 3D tooling
(save, preview, hand to any 3D-consuming node) without a DCC round trip.

## Decisions (user, 2026-10-02)

| Question | Decision |
|---|---|
| Where | **New node** `AtlasSceneTo3D` (solve in). Viewport outputs untouched. Runs headless. |
| Camera | **Recovered solve camera** (deterministic, matches the plate). |
| Model | **Full layered scene**: relief, clean-plate near/far bands, sky domes, generated objects. |
| Textures | **Full-res PNG in the GLB** + **float EXR sidecars** per layer. |

GLB cannot embed EXR: glTF images are PNG/JPEG (+ WebP/AVIF/KTX2 by extension);
no EXR extension exists, and the ComfyUI viewer (three.js GLTFLoader) would not
load one. So the GLB stays viewable and the EXRs ride beside it, referenced.

## The three sockets (core V135 contracts)

- **`model_3d`** — `FILE_3D_GLB`, a `comfy_api` `File3D` (path or BytesIO).
- **`model_3d_info`** — `LOAD3D_MODEL_INFO`: `list[{position, quaternion, scale}]`,
  right-handed Y-up world. Atlas geometry is already world-space → one identity
  entry per model file.
- **`camera_info`** — `LOAD3D_CAMERA`: `{position, target, zoom, cameraType,
  quaternion, fov (VERTICAL deg), aspect, near, far}`, right-handed, Y-up,
  looking down −Z — **Atlas's own convention, no axis conversion**.
  - position / quaternion from `inv(camera_view_matrix)` (full 4x4, never the 3x3)
  - fov = `2·atan(H / 2fy)`, aspect = `W/H`, zoom 1, `perspective`
  - target = the orbit pivot `/atlas/camera_data` already computes (reuse the helper)
  - near ≈ 0.05 m, far = 1.5 × farthest layer depth (sky domes sit at 300–320 m)

## Node (as built)

`AtlasSceneTo3D` 🧊 — `atlas_camera/comfy/nodes_scene3d.py`, standard tier,
`Atlas/10 · Export`.

| Inputs | |
|---|---|
| `solve` (ATLAS_SOLVE) | the layered solve (e.g. the import node's output) |
| `source_image` (IMAGE) | full-res primary plate — paints the primary relief and generated objects |
| ±`write_exr` (BOOLEAN, True) | one float EXR sidecar per layer plate |
| ±`filename_prefix` (STRING, `atlas/scene`) | under ComfyUI's output dir (`./output/` outside ComfyUI) |
| ±`max_glb_mb` (INT, 1024) | size budget — see *Budgets* |

| Outputs | |
|---|---|
| `model_3d` (FILE_3D_GLB) | → Save 3D (Advanced) `model_3d` |
| `model_3d_info` (LOAD3D_MODEL_INFO) | → `model_3d_info` (identity) |
| `camera_info` (LOAD3D_CAMERA) | → `camera_info` (recovered solve camera) |
| `glb_path`, `report` (STRING) | path + per-layer and per-sidecar account |

**Dropped from the proposal, deliberately:** `clean_plate` (each ProjectionSource
already carries its own full-res plate — see below), `exr_colorspace` (the
colourspace follows the plate's provenance, not a widget), `layers` (no consumer
yet; appendable later without breaking saved graphs).

## What goes in the GLB

One glTF **node + mesh per layer primitive** (`atlas_camera/exporters/scene_glb.py`),
each with its own unlit material:

| Layer | Geometry | Texture |
|---|---|---|
| primary relief | `projection_relief_mesh` | `source_image`, projective UVs |
| generated object (Pixal3D) | `pixal3d_object` | `source_image` on seen faces + linear `COLOR_0` on the hidden side (separate untextured primitive) |
| clean-plate near / far card, sky domes | each ProjectionSource's mesh | that source's `image_b64` |
| analytic backdrop plane | skipped | — |

**Plate per layer — corrected.** The proposal ranked a source's `image_b64` last as a
"preview". It is not: `AtlasCleanPlateLayer` / `AtlasSkyDomeLayer` encode the
FULL-resolution plate there, *already frame-outpainted and edge-extended* — the exact
pixels the layer projects, and the canvas its baked UVs address. A separately wired
full-res IMAGE lacks that padding and would misalign every outpainted layer, so
`image_b64` is the source of truth for ProjectionSources. Order as built:

1. `plate_ref` — when it is a real (non-proxy) float file of the same size: its own
   values and colourspace tag go to the EXR (`scene_referred: true`);
2. otherwise the layer's own plate (`image_b64` for sources, `source_image` for the
   primary scene), linearised with the sRGB EOTF and tagged `Linear Rec.709 (sRGB)`,
   reported as "linearised display plate, NOT scene-referred".

**Sidecars** (named in each material's `extras`): one half EXR per plate; plus, when
`AtlasHDRVertexTransfer` ran, a binary **float-colour PLY** per generated object
(`…_vertex_hdr.ply`, ACEScg linear — glTF `COLOR_0` is a 0..1 quantity).

## Budgets and failure behaviour

- **Size.** Measured on the 8K machine plate, 6 layers: **230–239 MB** GLB. Above
  `max_glb_mb` (default 1024) the node refuses with the measured size and the fix
  (fewer layers / smaller plate) instead of handing the viewer a file it cannot load;
  above 200 MB the report warns. 0 disables the budget.
- **No camera / no geometry.** No usable camera raises; a solve with no mesh layer
  raises "no layer had geometry to write".
- **A sidecar or the manifest failing never fails the GLB** — named in the report.
- **Older ComfyUI** (no `File3D` / `LOAD3D_*`): the node still writes the GLB and
  sidecars and returns a duck-typed file object; `glb_path` + `report` stay usable.

## As built vs plan

1. Multi-layer writer: a NEW writer; the single-mesh `export_relief_mesh_glb` was left
   untouched rather than refactored behind a byte-identical pin (no benefit for the risk).
2. `core/load3d_camera.py` as planned.
3. EXR sidecars via `plate/oiio_io.write_exr`; manifest via `_write_export_manifest`.
4. Node module is `nodes_scene3d.py` (not `nodes_export.py`).
5. Pins, catalog row, audit; wired into the Pixal3D object research workflow
   (under `research/`) as group 8 → Save 3D (Advanced).

## Acceptance

| Criterion | Status |
|---|---|
| camera_info rebuilt as a three.js camera reprojects like the solve, ≤ 0.01 px | **pass** (test) |
| GLB: one node per layer, unlit, `extras.exr`, COLOR_0 only on generated objects | **pass** (test) |
| EXR sidecar linearisation and tag | **pass** (test) |
| Save 3D (Advanced) saves the GLB — **GUI** | **pass** (`output/3d/atlas_scene_00001.glb`, 230 MB) |
| Save 3D (Advanced) saves the GLB — **headless** (Atlas MCP runner) | **pass** (`atlas_scene_00002.glb`, 239 MB) after the runner fix: `viewport_state` (LOAD_3D) owns a `widgets_values` slot (`comfy_http.FRONTEND_WIDGET_TYPES`), and output nodes ComfyUI refuses at queue time are reported as `NOT RUN ...` errors |

## Tests

- camera_info round trip: project a world point with the solve and with a three.js-style
  camera built from camera_info (fov/aspect/quaternion) → same pixel (≤0.01 px).
- GLB: one node per layer, materials unlit, `extras.exr` present, COLOR_0 on the
  generated object only, textures PNG at source resolution.
- EXR sidecars: float32/half, colourspace attribute, linearisation correct on a ramp.
- Node: headless run on a synthetic layered solve; Save 3D (Advanced) accepts the outputs
  (live check in V135).

## Risks

- **Size**: full-res PNG × ~5 layers of an 8K plate → a 250–400 MB GLB; the browser
  viewer may struggle. The decision stands; the report states the size, and a
  `max_texture` widget can be appended later without breaking saved graphs.
- **Per-layer UVs** for frame-outpaint layers must use the widened camera — verify
  before trusting the clean-plate textures.
- `File3D` / `LOAD3D_*` are core V135 types; on older ComfyUI the node must degrade
  to `glb_path` + report rather than fail to register.
