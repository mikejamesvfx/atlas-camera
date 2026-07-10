# Architecture

Atlas separates the portable latent-scene model from host-specific adapters.
The current implementation is still packaged as `atlas_camera` for compatibility,
with a small `atlas` facade for the vision-facing API.

## Layers

- `atlas_camera.core`: DCC-agnostic schema, intrinsics, extrinsics, validators,
  vanishing-point utilities, projection scene helpers, and recovery entry
  points. `LatentCamera` and `LatentScene` are the primary schema classes;
  `AtlasCamera` and `AtlasSolve` remain stable compatibility aliases.
- `atlas_camera.exporters`: Maya, Blender, Nuke, USD, and review package output.
- `atlas_camera.importers`: Atlas JSON and USD camera loading.
- `atlas_camera.comfy`: ComfyUI node wrappers around the core.
- `atlas_camera.reference_data`: local curated scale-reference registry.
- `atlas_camera.inference`: optional local multimodal provider helpers and
  future object-detector interfaces.
- `atlas_camera.gaussian`: future 3DGS / point-cloud scene-prior interfaces.
- `atlas_camera.ui`: optional FastAPI project service for image-backed local
  UI sessions.
- `ui/`: optional React/Vite workbench. It owns interactive presentation state
  such as 2D guide drawing, 3D viewport display options, and proxy-object
  editing. It must call backend endpoints and should not reimplement the
  deterministic solver.

## Core Conventions

- Right-handed coordinates by default.
- Y-up world axis by default.
- Image coordinates: origin top-left, x right, y down.
- Camera model: pinhole first, distortion metadata optional.

## Adapter Boundary

Maya, Blender, Nuke, Houdini, USD, OpenCV, Kornia, and Comfy conventions must not
silently leak into the core. Convert explicitly at import or export boundaries.

Examples:

- Blender is Z-up, so Y-up to Z-up conversion belongs in
  `atlas_camera.exporters.blender_exporter`.
- USD export should set stage up axis and encode camera metadata at export time.
- Comfy nodes should call core functions rather than own core data structures.

## Artist-Guided Constraints

`solve_from_constraints(...)` accepts simple line groups for the two horizontal
vanishing directions:

```python
{
    "image_width": 1920,
    "image_height": 1080,
    "line_groups": {
        "left": [((x1, y1), (x2, y2)), ...],
        "right": [((x1, y1), (x2, y2)), ...],
        "vertical": [((x1, y1), (x2, y2)), ...],
    },
}
```

Callers may also supply explicit `vanishing_points` instead of line groups.
The core fits vanishing points and then reuses the same camera-from-VP solve
path as automatic detection.

`tools/solve_constraints.py` reads this JSON shape and builds a complete review
package, including a debug overlay when OpenCV is available.

Scale references can be supplied as explicit known-height guides:

```python
{
    "scale_constraints": [
        {
            "reference_id": "door_210cm",
            "image_points": [[850, 760], [850, 410]],
        }
    ]
}
```

These references are stored as solve landmarks and projection-scene
`height_guide` proxy primitives. They are review/DCC guides only until metric
depth fitting is implemented.

## Local UI and 3D Workbench

The optional local UI stores all artist-authored inputs in the project
`constraints.json` file. The deterministic solver reads the established
constraint fields:

- `image_width`
- `image_height`
- `line_groups`
- `scale_constraints`
- `intrinsics_hint`

The React workbench may also persist UI-only 3D state under
`constraints.viewport3d`. This state includes:

- active 3D view mode: image match, orbit, top, front, or side
- display toggles for image plate, ground grid, axes, frustum, guides,
  horizon, and proxies
- camera preview overrides
- editable proxy objects with position, rotation, scale, source, and lock state
- selected proxy id

`viewport3d` is an inspection and layout aid. The core solver keeps it in debug
metadata when constraints are attached, but it does not use viewport state as
camera evidence. Promote a proxy object into deterministic solve input only by
adding an explicit scale constraint or future supported geometry constraint.

The 3D viewport is implemented with Three.js in the frontend. It is an optional
browser dependency and must not become a dependency of `atlas_camera.core`.

## Inference Helpers

The inference layer is optional. Local multimodal providers can suggest visible
scale anchors, perspective cues, risk notes, and `reference_id` values such as
`person_175cm`, `sedan_car`, or `eiffel_tower_tip_330m`. Future object-detector
interfaces should follow the same rule: produce suggestions with confidence and
uncertainty, but do not directly alter the deterministic camera solve without
artist or pipeline confirmation.

As of 2026-07-09 the layer hosts four model families behind guarded imports:
GeoCalib (learned camera prior), Depth Anything V2 and **Depth Anything 3**
(monocular depth — DA3 is the default; its canonical depth is converted to
metres with the *solved* focal), and the experimental layered-ray backends
(LaRI, World Tracing — user-cloned, research-only; see THIRD_PARTY.md). Each
wrapper returns torch-free dataclasses so torch never leaks into
`atlas_camera.core`.

## The ProjectionSource layer model (2026-07 addition)

A solve is no longer just a camera plus derived geometry: `LatentScene` carries
a list of `ProjectionSource` entries — each one a **camera + plate + optional
per-pixel mattes + optional own geometry + priority**. This single schema
object powers every layered workflow:

- **clean-plate bands** (same camera as the primary, depth-band-clipped mesh,
  inpainted plate),
- **sky domes** (same pose, widened outpaint camera, SAM matte),
- **X-ray layers** (patched hidden depth, LaMa plate, prediction mattes),
- **multi-angle patches** (constructed orbit camera, novel-view plate).

The viewport builds one projection material per source (facing-ratio and matte
discards, priority ordering), and the DCC layer exporters materialise the same
list through one shared collection (`exporters/_layers.py`) so Nuke and Maya
can never drift. The design rule: new layer types should be new *metadata* on
`ProjectionSource`, not new schema classes.

## LatentScene Direction

The first concrete recovered object is the camera. Future scene components
should follow the same contract:

- `value`
- `confidence`
- `editable`
- `serializable`
- `exportable`

Depth, geometry, lighting, semantics, and projection workspaces should attach to
the scene without introducing host-application assumptions into the core.
The current schema includes explicit `depth`, `geometry`, `lighting`, and
`semantics` component slots. They default to empty `LatentComponent` values until
their solvers are implemented, which lets review packages describe unsupported
components without pretending they were recovered.
