# matrixZone SDR -> HDR — full-resolution HDR plates from LTX-2.5

Status: **built** 2026-10-02 (`505137f`, destripe / ring / vertex transfer `d3887a1`,
separate workflows `db3b18e`); first live run on the 8K machine plate the same day.
Sections below the gates record what was built and measured.

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
- the LTX SDR->HDR chain, minus its 0.9 MP cap, run on the zones and once on the
  global image (as built: its own workflow, see *Architecture as built*). Its `hdr_linear` output (ACEScg linear,
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

## Gates, in order (with pass thresholds)

The pre-blend seam metric (per-seam |log2 luminance| disagreement between the two
zones' raw results) is a DIAGNOSTIC, not a gate: in textured regions each zone
re-renders fine detail differently and the metric reports that, while the 128 px
log-space feather hides it (live: textured seams 0.2-0.55 stops pre-blend, no visible
seam at the worst junction). What a viewer sees is the stitched plate, so the gates
use the **stitched-plate step test**: per seam segment, the per-row log2-luminance step
between 48 px strips either side; with the SDR input wired, the SDR's own step on the same
line is subtracted first (same structure, no seams, so structure cancels and a tonal offset
remains). Divided by the p90 of the same score on random lines of that plate (the content
baseline; a median baseline flagged 11 of 24 segments on structure). Built as `core.matrixzone.seam_step_test` and printed in
every `AtlasMatrixZoneStitch` report; a seam over 1.5x is flagged by name, never refused
(structure on the line scores high too, so the visual check stays the deciding half).

| gate | pass threshold | status (8K machine plate, 4x4) |
|---|---|---|
| 1 split -> stitch identity, no model | max rel. error < 1e-5 | **pass** (test) |
| 2 planner parity with `atlas_bridge` | numbers identical on the UHD worked case | **pass** (test) |
| 3 VRAM / tier | 2x2 (4K-tier zones) completes on the target GPU | **pass**: 2x2 on a 32 GB RTX 5090, 14 min 27 s cold; seam step test PASS (worst 1.20x, 4 seams); **2x2 is the default** |
| 4 end to end | every seam segment <= 1.5x the random-line p90, AND no visible seam at the worst-scoring junction | **pass with one flagged junction**: SDR-controlled, 3 of 24 segments flagged (worst `z22\|z23` 2.34x), all at or next to x=5760 / y=3384, visually clean (structure continuous, no ghosting); recorded as content |
| 5 mode shoot-out | the mode with the lower worst seam step wins; tie -> `per_zone_clip` | **open** — `per_zone_clip` is the provisional default |
| 6 anchor on/off | anchor kept only if it lowers the worst stitched-plate step | **pass**: sky seams 0.29 -> 0.04 stops pre-blend; kept |

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

## Budgets and failure behaviour

- **Runtime.** 4x4 = 17 LTX clips (global + 16 zones): **22 min 48 s** cold on the
  V135 box (~80 s per clip incl. model load). 2x2 = 5 clips of ~4x the pixels: **14 min 27 s** cold.
- **A zone fails or comes back wrong.** The stitch refuses — naming the zone — when a
  zone result is empty, non-finite, or the list length does not match the split; it
  never stitches around a hole. A zone returned at the wrong size is resampled and
  reported.
- **Re-runs.** ComfyUI caches every completed clip, so changing only the stitch
  settings (anchor, split, destripe) re-runs the stitch alone.
- **Interrupt.** ComfyUI interrupts between steps; a long LTX step completes first
  (observed: 73 s from Stop to "Processing interrupted").

## Architecture as built

- **Separate workflows** (user decision): SDR->HDR is not part of building a scene; it
  runs on a plate BEFORE the solve or on renders AFTER it.
  Two research workflows under `research/`: the HDR **still** workflow (still -> matrixZone ->
  EXR) and the HDR **clip** workflow (video -> EXR sequence + HLG). The Pixal3D
  scene workflow carries no LTX nodes.
- **Destripe** (`AtlasMatrixZoneStitch` `sdr_plate` + `destripe`): LTX-2.5 adds faint
  vertical stripes in every zone (~0.018 stops; also present with no zones, so the
  model, not the tiling). Measured against the SDR input per zone row, high-passed at
  257 px, divided out in two passes: **0.085 -> 0.016 stops** live.
- **Local destripe** (second pass, `destripe_local`): seen in Nuke on the 2x2 plate —
  streaks in the SDR-clipped bright sky and thin lines in part of the lower sky, both
  invisible to the band pass (it measures unclipped pixels, one profile per band).
  Per overlapping 256-row window, from flat pixels only, weighted by SDR flatness:
  correction p99 0.10 stops, max 0.73 (unweighted it reached 2.6 on the machine).
- **Outpaint ring smear** (clean-plate / sky layers): edge replication across a
  1024 px frame-outpaint ring read as stripes; ring ripple **5.70% -> 0.22%** live,
  real plate unchanged at 0.55%.
- **Hidden side "after" the solve** — `AtlasHDRVertexTransfer`. Generated objects'
  hidden sides are vertex colour; the model cannot see them, and laying them out in
  UVs would give it a texture atlas with no scene context to judge highlights by. The
  plate's own SDR->HDR curve is fitted from the pixel-aligned pair (ACEScg, log-binned,
  monotone) and applied to the vertex colours; live: 23,331 vertices, 184 above 1.0.

## Honesty in the output

The EXR is tagged ACEScg and labelled in the report as a MODEL RECONSTRUCTION of
highlight radiance from a display-referred plate — not photographed HDR. Same
doctrine as the generated-object hidden side.

## Open (the user's calls)

- ~~Grid / zone tier for 8K~~ — decided 2026-10-02: **2x2 at the 4K tier** (fits, faster,
  4 seams all passing). Destripe bands are capped at a quarter of the plate height:
  the stripes drift down a 2256-row zone, and one band per zone row left 0.085 stops.
- Default sequence mode — decided by gate 5's numbers (provisional: `per_zone_clip`).
- Whether the anchor's split frequency is exposed or fixed.
