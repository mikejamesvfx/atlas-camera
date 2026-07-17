> ARCHIVED 2026-07-18 — Angle Patch MVP shipped 2026-07-17

# Angle Patch MVP

This prototype implements the Photoshop round trip discussed for the canonical Ghost Town workflow.

## Nodes

- **Atlas Extract Angle Patch → Photoshop** (`AtlasExtractAnglePatch`)
  - Inputs: solve, viewport/proxy plate, matte, `patch_exact`, output directory.
  - Optional passes: depth and normal.
  - Outputs: cropped patch image, cropped matte, manifest path, typed `ATLAS_PATCH` package.
- **Atlas Import Angle Patch ← Photoshop** (`AtlasImportAnglePatch`)
  - Inputs: the typed package and optional edited image/matte tensors.
  - Outputs: image, matte, exact pose string, enriched package.

## Photoshop contract

Extraction writes:

```text
<output>/<name>/patch.png             the crop you edit in Photoshop
<output>/<name>/patch_matte.png
<output>/<name>/plate_full.png        the FULL frame (used for paste-back — do not edit)
<output>/<name>/patch_depth.png       optional
<output>/<name>/patch_normal.png      optional
<output>/<name>/atlas_angle_patch.json
```

**Registration rule (load-bearing):** the crop exists only as the Photoshop
convenience. `AtlasAddPatchView` needs a FULL frame — its ProjectionSource
samples uv across the whole patch-camera frustum, so a bare crop would
stretch and misregister. The import node therefore pastes the edited crop
back into `plate_full.png` at the manifest's `crop_bbox_xyxy` and returns
full-frame tensors. Photoshop must not resize the crop canvas (the import
node errors loudly if it did). The manifest stores the CAMERA block only,
never the full solve (a layered solve embeds megabytes of base64 plates).

The sidecar preserves the exact `azimuth_deg`, `elevation_deg`, and `distance_scale` string from the viewport. After editing `patch.png` in Photoshop, pass the edited image and matte to the import node, then wire:

```text
Import.patch_image  → AtlasAddPatchView.patch_image
Import.patch_exact  → AtlasAddPatchView.exact_view_override
```

This is intentional: the MVP does not re-estimate a camera or snap to a named 45° view. `AtlasAddPatchView` remains the reprojection stage and reconstructs the original extraction pose exactly.

## Current limitations

- The node packages the incoming viewport/proxy plate; it does not yet rasterize a full-quality 3D extract itself.
- Depth and normal passes currently use ComfyUI `IMAGE` tensors and are written as PNG previews; float-safe EXR writing should be the next engineering step.
- The crop is the padded non-zero matte bounding box. A future version should support an explicit artist rectangle and multi-region patches.
- Reviewed + hardened 2026-07-17 (Claude): full-frame paste-back on import (the crop-reprojection registration bug), camera-only manifest, real package version, honest colorspace_written field, registration regression test.
- Color metadata is recorded but not transformed. ACES/OCIO conversion belongs in the next pass.

Focused tests cover crop bounds, manifest creation, exact-angle preservation, import, and empty-matte rejection.

## Planned: one-button Photoshop launch with OCIO (user request, 2026-07-17)

- **`ATLAS_PHOTOSHOP`** env var → full path to the Photoshop executable —
  specifically **Photoshop (Beta)**, which carries full OCIO v2 color
  management, so the ACEScg float round trip stops being a proxy/LDR
  compromise.
- **"🎨 Open in Photoshop" button** on the extract node (or the viewport
  toolbar next to 📐): one click launches Photoshop with the just-written
  patch. Implementation shape: the existing aiohttp route pattern
  (`comfy/__init__.py`, like `/atlas/camera_data`) gains a
  `POST /atlas/open_photoshop` that `subprocess.Popen`s
  `[ATLAS_PHOTOSHOP, patch_path]` — path-validated to files inside the
  extraction output dir only.
- **"Open as OCIO":** write the patch as float EXR (ACEScg) when the OCIO
  path is available, and launch Photoshop with the `OCIO` env var pointed at
  the project's ACES config so PS Beta's OCIO pipeline picks it up and the
  artist paints in managed color — the manifest's `colorspace_intent` then
  finally matches `colorspace_written`.
- Import side unchanged: the paste-back contract already carries any bit
  depth once the EXR writer lands (the planned float step above).
