# Parked / future development

Ideas assessed and deliberately deferred — with enough context that whoever
picks one up doesn't re-derive the analysis. Ordered roughly by value.

## Percentile band mapping for depth-band sliders (parked 2026-07-10, pre-release)

**Problem (real, user-hit):** monocular depth crams the far field into a
sliver of the metric range, so the linear `near_pct`/`far_pct` sliders on
`AtlasDepthLayerMask`/`AtlasCleanPlateLayer` have wildly uneven pixel-space
effect — 0.70→0.80 can jump half the image while 0.0→0.3 does almost nothing
(it's why the staged master's far band sits at 0.72–0.80+ and the shipped
splits are 80/60/30 rather than even quarters).

**Wrong fix:** a fixed logarithmic transfer (the int8→float color-encoding
analogy). The skew is per-image, not a constant like display gamma — a log
curve tuned for one plate over/under-corrects on the next.

**Right fix:** a **percentile mapping** over the actual depth map's histogram
CDF: `near_pct 0.8` = "the depth above which the farthest 20% of *pixels*
sit". Image-adaptive and matches artist intent directly.

**Implementation constraints (why it was parked so close to release):**
- Must be a NEW appended widget (`band_mapping`: `linear`/`percentile`,
  default `linear`) — changing the meaning of the existing serialized pct
  values silently breaks every calibrated saved workflow, including the
  staged master. The append-only widget rule applies.
- Lives in the shared `_resolve_depth_band` helper (both nodes must stay in
  lockstep — that helper exists precisely so bands can't drift apart).
- Needs tests + re-verification of the calibrated examples; a real day-plus,
  not a one-line equation.

**Current mitigation:** per-band mask previews + the staged master's
stage-by-stage rhythm make linear splits calibratable by eye in a couple of
queue cycles.

## Other parked items (tracked in CLAUDE.md / memory, listed here for one view)

- **Render-conditioned patch v2 fine-tune** — Fixer training pair tooling +
  data plan shipped (`tools/generate_fixer_training_pairs.py`,
  `docs/dev/fixer_finetune_data_plan.md`); the training run itself not done.
- **Unanchored-wall plausibility cap** — walls whose ground contact is
  occluded (e.g. behind a fence) keep legacy depth-derived heights/widths;
  per-shot answer today is inpainting the ground line before solving.
- **Sky-spike sliver cosmetics** — thin sky slivers at silhouette junctions.
- **CI workflow** — parked on local branch `ci-workflow`; needs a
  workflow-scoped PAT (or add `.github/workflows/tests.yml` via the web UI).
- **`DA3NESTED-GIANT-LARGE-1.1` evaluation** — exposed but unevaluated;
  future role is the patch-registration spike, not mono depth.
- **Band-layer `depth_edge_rel` recalibration under DA3** — can the 1.5
  default come back down with DA3's cleaner depth?

## Peak-RAM of the full band + inpaint pipeline (parked 2026-07-13, smoke-test)

**Observed:** a full `AtlasInput` run (`layers=4` + `inpaint`) on a 7680×4512
plate hit `Unable to allocate 529 MiB for a (4512,7680,4) float array` — a host
**RAM** OOM (not VRAM), with ~26 GB free. The same image runs fine at a lighter
config, and a 9216×3840 plate ran fine the same session, so it's a peak-memory
ceiling, not a bug. Root cause: several full-resolution RGBA-float plates/mattes
(~0.5 GB each) plus per-band relief meshes and LaMa crops are live at once
across the 4 bands. **Deferred fix ideas:** free each band's full-res plate/mask
as soon as its `ProjectionSource` is built (don't hold all bands' plates
simultaneously); process bands more strictly sequentially in the node-expansion;
or cap the working-plate resolution for the mask/matte ops (the mesh already
decimates). For now it's a **doc note** (see `docs/CAMERA_MOVES.md` →
Performance) — lower `mesh_resolution`/`layers` or free host RAM.
