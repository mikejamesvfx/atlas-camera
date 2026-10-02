# matrixZone SDR -> HDR — full-resolution HDR plates from LTX-2.5

Status: proposal (scoped 2026-10-02, not built)

## Why

The LTX-2.5 SDR->HDR IC-LoRA (Lightricks, scene-embedding conditioned, cfg 1,
8 distilled steps) runs at the resolution it is fed, and a 22B video model will
not take an 8K frame. The current research graph caps the plate to ~0.9 MP, so
the EXR it writes is ~1280 px wide — a preview, not a plate. matrixZone is the
existing answer to "make LTX deliver 8K": split the frame into a regular grid of
exact crops at sizes the model was trained on, run each, place them back.

Design of record for matrixZone itself: `atlas-unreal/Docs/ATLAS_MATRIXZONE.md`
(rev 3, schema `matrixZone/1.2`) and `atlas_bridge.matrixzone.plan()`. This
proposal applies it to a STILL plate and to a CONVERSION, not a generation.

## What carries over unchanged

- **A zone is an exact crop.** One optical centre, shifted principal point; a
  zone result goes back as a 2D placement, never a reprojection or a resample.
- **Padded render, 128-clean axes; zones 64-clean; generation 32-clean;
  frames `% 8 == 1`.** Same arithmetic, same planner — ported into
  `atlas_camera/core/matrixzone.py` and pinned against the bridge's worked
  numbers (UHD 8K -> render 7680x4352, zone renderRect 3904x2240, 128 px
  interior overlap), so the two repos cannot drift.
- **Overlap on interior edges only, 128 px, feathered.**
- **Global-first is the default.**

## What is different for SDR -> HDR

**The seam risk is tonal, not compositional.** The conversion invents no
objects, so a zone cannot disagree about *what* is there — but each zone's
highlight reconstruction is the model's guess about *how bright* it is. Two
neighbours can expand the same sky to different radiance, and a 128 px feather
turns that into a soft gradient that is still wrong.

**So global-first here means a radiance anchor, not a composition pass:**

```
plate (8K, display sRGB)
  -> pad to the planned render (edge-extend; a still has no Unreal overscan)
  -> GLOBAL: whole render at a trained low tier (render/4)  -> HDR_global
  -> ZONES: each renderRect crop at the zone tier            -> HDR_zone[i]
  -> per zone, in LOG radiance:  zone = lowpass(HDR_global, upsampled) + highpass(HDR_zone[i])
  -> place at plateRect, feather the overlap in log space, crop the padding
  -> one ACEScg half EXR at full plate resolution
```

The global pass decides every zone's low-frequency radiance once; zones only add
detail. That is exactly matrixZone's division of labour, moved from composition
to tone. The split frequency is a parameter (default: the overlap width).

## "As a sequence" — two ways to feed the zones

LTX wants a clip. A still zone becomes one of:

| mode | frames | runs | cross-zone consistency |
|---|---|---|---|
| **`per_zone_clip`** | each zone repeated 9x (8k+1) | one LTX run per zone (ComfyUI list mapping — the split node emits a LIST, the chain runs once per element, the stitch node collects) | none from the model; all from the global anchor |
| **`zones_as_frames`** | the zones themselves ARE the frames, serpentine order so consecutive frames are spatial neighbours, padded to 8k+1 by ping-pong | ONE run | the model's temporal attention sees every zone — tone may agree across zones for free |

`zones_as_frames` is the interesting one and the riskier one: neighbouring
frames are different pictures, which the model may read as cuts (temporal
flicker, or tone drifting along the scan). It needs one equal zone size, which
the planner already guarantees. Both modes are measurable with the same seam
metric, so the spike runs both and the numbers decide the default.

## Nodes

- **`AtlasMatrixZoneSplit`** — plate IMAGE, grid, zone tier, overlap, mode ->
  zone IMAGE (LIST for `per_zone_clip`, one 8k+1 batch for `zones_as_frames`),
  the global low-tier IMAGE, and an `ATLAS_MATRIXZONE` handle (the plan:
  render size, plateOrigin, zone renderRects/plateRects, scan order, mode).
- the existing LTX SDR->HDR chain (group 7), minus its 0.9 MP cap, run on the
  zones and once on the global image. Its `hdr_linear` output (ACEScg linear,
  unbounded) is what gets stitched — not the preview, not the per-frame EXRs.
- **`AtlasMatrixZoneStitch`** — zone HDR (list or batch) + global HDR + handle ->
  full-resolution ACEScg EXR (half, via `plate/oiio_io.write_exr`), a tonemapped
  preview IMAGE, and a report with the seam metric before and after the anchor.

Pure maths in `core/matrixzone.py` (plan, crop, place, log-space feather,
frequency split) — numpy only, no ComfyUI.

## The seam metric (what "it works" means)

Per interior seam: median and p95 of |log2 luminance| difference between the two
zones over their shared overlap, BEFORE the anchor and blend. Reported per seam
and gated: a seam above a threshold is named in the report, never silently
blended away. The same metric compares `per_zone_clip` vs `zones_as_frames`.

## Gates, in order

1. **Split -> stitch identity, no model.** Feed the SDR zones straight to the
   stitcher: zero difference outside overlap bands, identical inside them (the
   matrixZone gate 2 for pixels instead of depth).
2. **Planner parity.** `core/matrixzone.plan` reproduces `atlas_bridge` on its
   two worked cases, number for number.
3. **VRAM / tier.** Which zone tier the V135 box runs with 9 frames at 22B bf16:
   the zone renderRect tier (3904x2240) or only the 1080p tier (1920x1088,
   which means a 4x4 grid on 8K). Measured, not assumed.
4. **One zone end to end**, then the grid, with the seam metric on every seam.
5. **Mode shoot-out**: `per_zone_clip` vs `zones_as_frames`, same plate, seam
   metric + visual. The winner becomes the default.
6. **Anchor on/off**: same zones, with and without the global low-frequency
   anchor. If the anchor does not lower the seam metric it is not kept.

## Honesty in the output

The EXR is tagged ACEScg and labelled in the report as a MODEL RECONSTRUCTION of
highlight radiance from a display-referred plate — not photographed HDR. Same
doctrine as the generated-object hidden side.

## Open (the user's calls)

- Grid / zone tier for 8K: 2x2 at the 4K tier (matrixZone default, if VRAM
  allows) vs 4x4 at the 1080p tier.
- Default sequence mode — decided by gate 5's numbers.
- Whether the anchor's split frequency is exposed or fixed.
