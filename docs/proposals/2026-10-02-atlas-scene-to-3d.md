# AtlasSceneTo3D — the layered Atlas scene as ComfyUI 3D sockets

Status: proposal (scoped 2026-10-02, not built)

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

## Node

`AtlasSceneTo3D` 🧊➡️ (standard tier, `Atlas/10 · Export`; registry 127 → 128)

| Inputs | |
|---|---|
| `solve` (ATLAS_SOLVE) | the layered solve (e.g. the import node's output) |
| `source_image` (IMAGE) | full-res primary plate |
| ±`clean_plate` (IMAGE) | full-res clean plate, for layers projected in `clean_plate` mode |
| ±`write_exr` (BOOLEAN, default True) | EXR sidecars per layer |
| ±`exr_colorspace` (combo) | `Linear Rec.709 (sRGB)` default; `ACEScg` when the plate came float (plate_ref) |
| ±`layers` (combo) | `all` default (room for `primary`/`no_sky` later — append-only) |
| ±`filename_prefix` (STRING) | `atlas/scene` |

| Outputs | |
|---|---|
| `model_3d` (FILE_3D_GLB) | → Save 3D (Advanced) `model_3d` |
| `model_3d_info` (LOAD3D_MODEL_INFO) | → `model_3d_info` |
| `camera_info` (LOAD3D_CAMERA) | → `camera_info` |
| `glb_path`, `report` (STRING) | path + what was written and from which plate |

## What goes in the GLB

One glTF **node + mesh per layer primitive**, each with its own material:

| Layer | Geometry | Texture |
|---|---|---|
| primary relief | `projection_relief_mesh` | source plate, projective UVs |
| generated object (Pixal3D) | `pixal3d_object` | source plate on seen faces + `COLOR_0` vertex colour on the hidden side (existing split) |
| clean-plate near / far card | each `ProjectionSource` mesh | clean plate, that source's OWN projective UVs (frame-outpaint layers use widened intrinsics — verify the baked UVs include the pad) |
| sky domes | dome meshes | that source's sky plate |
| analytic backdrop plane | skipped (or a quad) | — |

All materials `KHR_materials_unlit` (a projection is lighting-baked), textures
**full-resolution PNG** (decision). Each material's `extras` names its EXR sidecar.

**Plate resolution order per layer** (stated in the report, never silent):
1. `plate_ref` — float-safe on-disk plate (RAW/EXR loads) → EXR copied/converted via
   `atlas_camera.plate` (OIIO/OCIO), colourspace tag preserved;
2. a wired full-res IMAGE matched by projection mode (`source_photo` →
   `source_image`, `clean_plate` → `clean_plate`);
3. the source's `image_b64` preview — **flagged as preview resolution**.

EXR from an 8-bit display IMAGE is honest about what it is: sRGB EOTF → linear
Rec.709, tagged `Linear Rec.709 (sRGB)`, flagged "linearised display plate, not
scene-referred". Only a `plate_ref` source earns `ACEScg`.

## Build plan

1. **core / exporters** — `exporters/scene_glb.py`: multi-node GLB writer. Refactor the
   buffer/accessor/material building out of `export_relief_mesh_glb` so the single-mesh
   path is a one-layer call of the new writer (byte-identical output pinned by test),
   keeping the generated-object COLOR_0 split and ribbon handling.
2. **core** — `core/load3d_camera.py`: pure `solve -> camera_info dict`
   (+ `model_3d_info`), unit-tested against the convention table above.
3. **exporters** — EXR sidecars via the existing `plate/oiio_io` writers; manifest via
   `_write_export_manifest` (a manifest failure never fails the export).
4. **comfy** — `nodes_export.py: AtlasSceneTo3D`; `File3D` imported lazily from
   `comfy_api` (comfy/ layer only), duck-typed fallback for tests.
5. **pins** — registry 128, facade, NODE_CATALOG row, feature audit, an example wired
   into the Pixal3D object research workflow (under `research/`) → Save 3D (Advanced).

## Tests

- camera_info round trip: project a world point with the solve and with a three.js-style
  camera built from camera_info (fov/aspect/quaternion) → same pixel (≤0.01 px).
- GLB: one node per layer, materials unlit, `extras.exr` present, COLOR_0 on the
  generated object only, textures PNG at source resolution; single-layer path
  byte-identical to the old exporter.
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
