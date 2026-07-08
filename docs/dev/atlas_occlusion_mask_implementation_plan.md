> **STATUS (2026-07-08): Phase 1 shipped earlier; Phase 2 is DONE** — but not
> via the browser render pass this plan assumed. The primary camera's own
> depth map IS the shadow map (`primary_camera_validity_mask`'s
> `primary_depth_map`/`depth_bias_rel` params; `AtlasOcclusionMask`'s
> `occlusion_mode="depth_shadow"` + `primary_depth` input), pure numpy and
> headless. Both sides ground-pin via `estimate_ground_scale` so the depth
> comparison happens in one metric space. See CLAUDE.md's depth-shadow
> design rule.

# Atlas Occlusion Mask Implementation Plan

## Context

The old Glyph Software Mattepainting Toolkit used "shadow occlusion" to decide where one camera projection could not see geometry, then allowed lower-priority projections to fill those masked regions. Atlas can adopt the same idea for ComfyUI: generate a mask from the source camera's missing/invalid projection areas, use it in an image composite node, then feed the composited multi-angle image back into `AtlasAddPatchView`.

This document captures two phases:

- Phase 1: a simple projection-validity mask.
- Phase 2: a true MPTK-style depth-shadow occlusion upgrade.

## Phase 1 - AtlasOcclusionMask

### Goal

Add a new ComfyUI node named `AtlasOcclusionMask` that outputs a mask where the primary/source camera projection is invalid from a target/patch view. The mask can drive Comfy's image composite nodes.

This phase is the simple version. It marks areas as missing when they are:

- behind the primary/source projection camera
- outside the source image frame
- too grazing according to an angle threshold

It does not yet detect true self-occlusion behind nearer geometry.

### Node Shape

Add the Python node in `atlas_camera/comfy/nodes.py`.

Inputs:

- `solve`: `ATLAS_SOLVE`
- `target_image`: `IMAGE`, used to inherit composite size
- `client_data`: browser round-trip string, following the `AtlasBlockoutViewport` pattern
- `source_azimuth_view`
- `source_elevation_view`
- `patch_azimuth_view`
- `patch_elevation_view`
- `patch_distance`
- `flip_azimuth`
- `dilate_px`
- `soft_edge_px`
- `power`
- `angle_threshold`

Outputs:

- `occlusion_mask`: `MASK`, white where patch/composite should be applied
- `coverage_mask`: `MASK`, optional inverse/debug mask
- `debug_image`: `IMAGE`, optional colored preview for tuning

### Camera And Payload Plan

Reuse the same target-camera construction as `AtlasAddPatchView`. Ideally factor the shared orbit logic into a helper so both nodes agree exactly.

The browser payload should include:

- primary/source camera view matrix and intrinsics
- target/patch camera view matrix and intrinsics
- target output width and height from `target_image`
- serialized `proxy_geometry`
- mask controls

Either add a new route such as:

```text
GET /atlas/occlusion_data/{node_id}
```

or generalize the existing camera-data route enough to support both viewport and mask nodes.

### Frontend Pass

Extend `atlas_camera/comfy/web/atlas_blockout.js` with a small `AtlasCamera.OcclusionMask` extension.

This does not need a full viewport. A compact node widget with a `Render Mask` button is enough.

Rendering logic:

1. Build derived proxy geometry.
2. Render from the target/patch camera.
3. Use a custom shader/material that projects each fragment back into the primary camera.
4. Mark primary coverage valid when:
   - fragment is in front of the primary camera
   - projected UV is inside the source frame
   - facing angle passes `angle_threshold`
5. Output:
   - white = primary missing, patch should fill
   - black = primary covers this surface

### Mask Controls

`dilate_px`

Expands the white mask after rendering. This matches the old MPTK idea of expanding the shadow/missing area to conceal edge bleeding.

`soft_edge_px`

Blurs the dilated mask for compositing.

`power`

Remaps feather density after blur. Higher values should make the patch contribution more solid near the feathered edge.

`angle_threshold`

Range: `0` to `90` degrees.

Default `90` means only frustum, behind-camera, and out-of-frame failures are masked. Lower values also mask grazing surfaces where projection smear is likely.

### Intended Comfy Workflow

```text
Solve
  -> Derive Projection Geometry
  -> AtlasOcclusionMask
```

Then:

```text
initial projected/inferred image
  + multi-angle patch
  + occlusion_mask
  -> ImageCompositeMasked
```

Then:

```text
composited patch
  -> AtlasAddPatchView
  -> AtlasBlockoutViewport
```

This makes the patch image smarter before it becomes a projection source.

### Known Limitation

The simple version will not detect that a target-view point was hidden behind another object from the primary/source camera if it still projects inside the source frame and passes the facing test.

That requires the Phase 2 depth-shadow comparison.

### Tests

Python tests:

- node registration and display name
- default widget values
- target camera helper matches `AtlasAddPatchView`
- blank `client_data` returns correctly sized zero masks
- decoded mask tensors have correct shape

Browser or fixture tests:

- outside-frame area becomes white
- valid primary projection stays black
- `dilate_px` increases mask area
- `soft_edge_px` creates gray edge pixels
- `power` increases feather density

## Phase 2 - True MPTK-Style Depth-Shadow Occlusion

### Goal

Upgrade `AtlasOcclusionMask` from projection-validity masking to true projection-camera visibility masking.

The simple mask asks:

```text
Can the primary camera project a valid pixel onto this target-view fragment?
```

The MPTK-style mask asks:

```text
Was this fragment actually visible to the primary camera,
or was it hidden behind nearer geometry?
```

### Core Idea

Render Atlas proxy geometry from the primary/source camera into a depth texture.

Then, when rendering the target/patch view mask:

1. Transform each target-view fragment's world position into the primary camera.
2. Project it into primary image coordinates.
3. Compare its primary-camera depth against the stored primary depth map.
4. If the fragment is farther than the depth map at that pixel, it was occluded from the primary camera.
5. Mark it white in the occlusion mask so a patch projection can fill it.

The final mask condition becomes:

```text
outside_source_frame
OR behind_source_camera
OR grazing_angle
OR primary_depth_occluded
```

### New Controls

`occlusion_mode`

Options:

- `simple`
- `depth_shadow`

Default to `simple` initially.

`depth_bias`

Prevents false occlusion from tiny depth precision mismatches.

`occlusion_dilate_px`

Expands the true occlusion boundary.

`occlusion_soft_edge_px`

Feathers only the occlusion boundary.

`shadow_samples`

Optional multi-sample taps around the projected primary pixel for smoother boundaries.

`backface_policy`

Controls whether backfaces count as occluders.

`occluder_scope`

Controls what contributes to the primary depth pass:

- derived geometry only
- user proxies too
- all projectable meshes

### Why This Matters

This reproduces the old toolkit's central behavior: the projection camera behaves like a light. Areas in shadow from that camera are not trusted, and lower-priority projections can fill them.

It solves cases the simple mask misses:

- the side of a wall hidden behind a foreground object
- interior recesses not visible in the original photo
- back sides or overlapping geometry that still project inside the original image frame
- patch image areas that should appear only where the primary camera had no line of sight

### Browser Implementation Shape

The first implementation should stay browser-side because the viewport already owns the exact Three.js geometry and transforms.

Passes:

1. Build the same proxy geometry used by the projection viewport.
2. Create a render target for primary depth, matching the primary source aspect/resolution.
3. Render depth from the primary camera.
4. Render the target-view mask with a shader that receives:
   - primary view matrix
   - primary intrinsics/projection data
   - primary depth texture
   - depth bias
   - facing threshold
5. Post-process the mask for dilation, blur, and power.

### Debug Outputs

True occlusion quality depends on proxy geometry quality, so debug outputs are important:

- primary depth map
- target occlusion mask
- projected primary UV debug
- depth-difference heatmap
- final compositing mask

### Caveat

The true occlusion mask is only as reliable as the proxy geometry. A bad relief mesh or primitive fit can create false occlusions or miss real ones. The node should therefore keep `simple` as the default and make `depth_shadow` an artist-enabled upgrade.

### Rollout

Phase 1 gives a useful frustum/facing/grazing mask.

Phase 2 adds `occlusion_mode = simple | depth_shadow`.

This keeps the workflow forgiving while opening the door to a real MPTK-style projection layering system.
