# Proposal — a public "photograph to relief world" facade

- Date: 2026-09-13
- Status: **PROPOSED. Not accepted, not implemented.** For the Atlas Camera owner to decide.
- From: the `capability.showcase.bake_world` consumer (atlas-showcase), registered in Atlas Nexus
  with `depends.apps[atlas-camera].surface: internal`
- Record of the debt: atlas-showcase `docs/CAMERA_DEPENDENCY_DEBT.md`

## Problem

Atlas Camera's public API is `import atlas` (`CLAUDE.md`). The Showcase bake reaches past it into
eleven internal symbols to perform one operation:

| symbol | visibility |
|---|---|
| `raw.pipeline.import_raw` | internal module |
| `inference.learned_prior.estimate_camera_prior` | internal module, not in `inference.__all__` |
| `core.solver.solve_from_learned_prior` | internal module |
| `core.camera_spec.CameraSpec.from_solve` | internal module |
| `inference.depth_estimator.estimate_depth` | internal module |
| `core.solver._resize_depth` | **underscore-private** |
| `core.relief_mesh.estimate_ground_scale` | internal module |
| `core.solver.estimate_ground_height_from_depth` | internal module |
| `core.relief_mesh.build_relief_mesh` | internal module |
| `core.scene_health.scale_health` | internal module |
| `exporters.relief_mesh_exporter.export_relief_mesh_glb` | internal module |

It also repeats Camera's own orchestration: the horizon-row formula, the depth resize and the
ground-fit sequence are the steps `solve_still_image_learned` already performs in its tier-2
block. A consumer re-deriving domain choreography is how a consumer quietly becomes a second
implementation.

## What the consumer needs

One operation:

> **Photograph → metric relief world.** Given one photograph (RAW, or JPEG/PNG, with EXIF focal
> when present), recover the camera with the learned prior, estimate metric depth at the solve's
> resolution, fit ground scale, build the relief mesh, and export a self-contained textured GLB —
> returning the solved camera (intrinsics, world→camera view matrix, pitch, horizon row), scale
> health with its measured ground evidence, and mesh statistics.

## Why the existing public surface does not cover it

- `atlas.recover(image, method="learned")` performs the learned solve and, with
  `camera_height="auto"`, the depth and ground fit internally — but does not forward
  `focal_length_mm_hint`, sensor size or `depth_model`, returns no depth array, pitch or horizon
  row as a stable interface, builds no relief mesh and exports nothing.
- MCP `atlas_export_scene` exports saved solves to DCC formats through a running ComfyUI; it has
  no relief-GLB format, and the Showcase bake is ComfyUI-free by design.
- `core.relief_mesh._relief_mesh_from_solve` reads a mesh a ComfyUI node already derived onto a
  solve; it builds nothing.

## Proposal

**Do not make the eleven symbols public individually.** That would freeze eleven internal
signatures, and the consumer would still own the orchestration.

Add **one** coherent public operation to the `atlas` facade, with a typed result, for example:

```python
world = atlas.relief_world(
    "photo.NEF",
    depth_model=None,          # Camera's default metric model
    device=None,
    grid_long_edge=192,
)
world.solve            # AtlasSolve (already public)
world.camera           # fx, fy, cx, cy, image size, view_matrix, pitch_deg, horizon_y, focal_source
world.scale            # ScaleHealth + ground_scale, inliers, measured height and confidence, depth_is_metric
world.mesh             # ReliefMesh (+ stats)
world.display_image    # the decoded frame the solve used
world.capture          # EXIF make/model/lens/datetime

world.export_glb(out_dir, texture_max=4096, texture_format="PNG")   # feature/glb-jpeg-texture
```

Names, return type and module placement are Camera's decision. The requirements are:

1. EXIF focal hint and sensor size are honoured, as the current bake does.
2. The depth model is selectable.
3. The result exposes what a consumer records — camera, scale evidence, mesh statistics — as a
   stable interface, not as attributes reached into.
4. GLB export belongs to the same surface, including the `texture_format` choice if that feature
   is accepted.
5. It runs without ComfyUI.

## Closing condition

`tools/bake_showcase.py` imports only the public facade; atlas-showcase's
`docs/CAMERA_DEPENDENCY_DEBT.md` is closed with the Camera commit that introduced it; and the
Nexus manifest changes `surface: internal` to `surface: public`.

## Not proposed

- No change to existing public or node surfaces.
- No shared package.
- No new ComfyUI node.
