# Atlas marketing cleanplate bundle

These are approved presentation assets derived from the three ACEScg showcase
plates.  The `atlas_marketing_ocio_*` workflows preserve the normal Atlas
camera/depth solve, semantic matte, OCIO, layered projection, retopology,
Nuke/Maya, USD, and debug paths, but feed the approved 4K sRGB cleanplate into
the generated-background layer.  The large **HERO OUTPUT** preview is the best
node to frame for a workflow screenshot.

The cleanplate is depth-solved independently and supplies continuous hidden
support geometry under the removed object. The original depth is used only by
the SAM-matted foreground object. Do not replace this with a foreground/far
band split: that places the revealed road/headland at the far cutoff and makes
the car or castle float over a vertical drop during off-axis moves.

## Deliverables

- `cleanplates/*_marketing_cleanplate_4k.png` — 3840×2160 cleanplates.
- `before_after/*_before_after_4k.png` — 3840×1080 source/cleanplate pairs.
- `workflow_previews/*_workflow_hero_preview.png` — images captured from the
  actual completed ComfyUI `PreviewImage` nodes.
- `workflows/atlas_marketing_ocio_*_dcc_workflow.json` — run-verified graphs.

The same bundle is copied to
`ComfyUI/output/AtlasMarketing/`, and the workflow JSONs are installed in the
user's normal ComfyUI workflow directory.

## Final edit prompts

- **Ocean castle:** remove the complete castle; reconstruct an uninterrupted
  grassy rocky headland, ocean, waves, and horizon while locking the original
  camera, lighting, foreground rocks, and coastal perspective.
- **Space hangar:** remove the spacecraft and landing gear; continue the rear
  wall, light frame, reflective floor panels, seams, and reflections through
  the original one-point perspective.
- **Ghost town:** remove the car, fallen sign, log, and debris; continue the
  dirt road, ruts, gravel, dry grass, shadows, and right storefront edge while
  preserving the original buildings and vanishing point.

The cleanplates were created with the built-in image editing model, reviewed,
then upscaled with Lanczos filtering to the source plate's 3840×2160 delivery
size.  They are marketing-quality raster cleanplates, not replacements for the
original linear ACEScg EXRs.
