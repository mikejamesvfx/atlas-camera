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

**Windowed since 2026-10-03 (review OV-1).** The score was one |median| over a whole
zone edge, so a seam +2 stops on its upper half and -2 on its lower half cancelled to
0.00x and passed. Each segment is now scored in ~256 px windows along the line
(`SEAM_WINDOW_PX`): per window |median(step_hdr - step_sdr)| (plain: median |step|), the
segment scores its WORST window, and the report adds the window p95 and the worst
window's span. The random-line baseline uses the same length, the same windows and the
same worst-window score, so a long seam is not flagged for having more windows. On the
synthetic uniform half-stop seam the ratio moved 36.8x -> 35.4x, so the 1.5x bar stands;
**every live seam figure below (1.20x, 2.34x, 0.88x) is PRE-WINDOWING and must be
re-measured on the live plates** (V-1). The worst windowed seam is also written into the
EXR as `atlas:seam_worst`.

**SDR compared in ACEScg (review F-5).** The post-passes used to compare the Rec.709-linear
SDR with the ACEScg HDR and weigh ACEScg with Rec.709 luminance. Destripe, local
destripe, the seam control and `seam_metrics` now use AP1 luminance (0.2723, 0.6741,
0.0537), with the SDR weighted as if moved to ACEScg; `sdr_detail_transfer` converts the
SDR to ACEScg (the same Bradford matrix as `core.hdr_transfer`) before its per-channel
ratio, so a saturated colour no longer picks up a hue shift (synthetic R/G/B ramp: chroma
error 0.17 -> 1e-7). Grey content is unchanged; the live 8K old-vs-new Nuke comparison is
pending (V-1).

| gate | pass threshold | status (8K machine plate, 4x4) |
|---|---|---|
| 1 split -> stitch identity, no model | max rel. error < 1e-5 | **pass** (test) |
| 2 planner parity with `atlas_bridge` | numbers identical on the UHD worked case | **pass** (test); interior zones are now clamped into the render (see *Planner clamp*) — the UHD pins are byte-identical, `atlas_bridge` not yet mirrored |
| 3 VRAM / tier | 2x2 (4K-tier zones) completes on the target GPU | **pass**: 2x2 on a 32 GB RTX 5090, 14 min 27 s cold; seam step test PASS (worst 1.20x pre-windowing, 4 seams; re-measure); **2x2 is the default** |
| 4 end to end | every windowed seam segment <= 1.5x the random-line p90, AND no visible seam at the worst-scoring junction | **pass with one flagged junction (pre-windowing, re-measure)**: SDR-controlled, 3 of 24 segments flagged (worst `z22\|z23` 2.34x), all at or next to x=5760 / y=3384, visually clean (structure continuous, no ghosting); recorded as content |
| 5 mode shoot-out | the mode with the lower worst seam step wins; tie -> `per_zone_clip` | **open** — `per_zone_clip` is the provisional default |
| 6 anchor on/off | anchor kept only if it lowers the worst stitched-plate step | **pass**: sky seams 0.29 -> 0.04 stops pre-blend; kept |

1. **Split -> stitch identity, no model.** Feed the SDR zones straight to the
   stitcher: zero difference outside overlap bands, identical inside them (the
   matrixZone gate 2 for pixels instead of depth).
2. **Planner parity.** `core/matrixzone.plan` reproduces `atlas_bridge` on its
   two worked cases, number for number.

   *Planner clamp (2026-10-03, review F-3).* On a tall/narrow grid the 64-clean zone
   can be far larger than its cell, and the centred start of an INTERIOR zone fell off
   the render (3840x2160, 1x8, overlap 512: `z10` at y=-3, `z60` ending at 2179 >
   2176). Interior starts are now clamped to [0, render - zone], which keeps the cell
   covered, and `plan_still` raises if any renderRect still leaves the render. A
   3200-plan sweep changed only the 316 plans that were off canvas; the UHD worked
   cases are byte-identical. `atlas_bridge` still carries the old arithmetic (TODOS.md).
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
  never stitches around a hole. It also refuses a non-finite global pass, and a
  `zones_as_frames` sequence shorter than the split's zones need (naming both counts;
  it used to clamp the index and stitch another zone's pixels). A zone returned at the
  wrong size is resampled and reported; so is an `sdr_plate` at a different size than
  the plate (both sizes named), and image-batch frames the split dropped.
- **The EXR fails to write.** The failure is report line 1 (`EXR NOT WRITTEN: ...`) and
  `exr_path` is empty; an output prefix that leaves the output directory is a node
  error, not a note.
- **Memory.** Synthetic 4K plate, 2x2: `core.stitch` peak 1012 -> 446 MiB and the
  full stitch node with `sdr_plate` 1506 -> 986 MiB (tracemalloc; working set 2237 ->
  1714 MiB) after the 2026-10-03 trims (anchor once per zone, float32 accumulators and
  resizes, luminance-only seam state, float32 sRGB decode); identity gate unchanged.
  Still clips are MATERIALISED (repeat). A zero-copy expand() was tried and reverted:
  the LTX-2.5 VAE encode on a stride-0 batch measured 166.8 s vs 1.46 s at
  1088x1920x9, and the live 2x2 8K run went 14.5 -> 30 min (2026-10-03).
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
- **Detail from SDR** (`sdr_detail_transfer`, the fix that held in Nuke): band + local
  destripe still left visible lines (thin, and wider ones across cloud texture). The SDR has
  the same content and no stripes, so it guides: per channel the log ratio hdr/sdr is
  smoothed by a guided filter (r 64, eps 0.01) on SDR log luminance and the plate rebuilt as
  sdr x 2^ratio; clipped highlights (SDR max > 0.85-0.97, feathered) keep the HDR. Rejected
  on the way: a thin-line column pass (invented streaks beside bright clouds) and a 6 px
  base/detail split (left the wider lines, softened texture until the slope was an
  amplitude ratio). 16 sampling steps instead of 8: no change in stripes, +28% runtime.
  Live in ComfyUI (2026-10-03, default 2x2 workflow): seam step test worst 0.88x
  (PRE-WINDOWING whole-edge score, before the ACEScg SDR comparison -- re-measure), matches
  the offline reference to 0.004 stops median; confirmed in Nuke.
  **Re-measured 2026-10-16 (V-1 closed for the 2x2 default):** windowed and
  SDR-controlled in ACEScg, the same plate scores **1.14x** (worst `z00|z01`,
  1.306 stops against a 1.146-stop random-line p90) and the cathedral plate
  **0.82x**. Both PASS the 1.5x gate with nothing flagged. The score went UP
  (0.88 -> 1.14) because windowing stopped a bright half and a dark half of one
  edge cancelling, exactly as the windowing change intended.
- **Outpaint ring smear** (clean-plate / sky layers): edge replication across a
  1024 px frame-outpaint ring read as stripes; ring ripple **5.70% -> 0.22%** live,
  real plate unchanged at 0.55%.
- **Hidden side "after" the solve** — `AtlasHDRVertexTransfer`. Generated objects'
  hidden sides are vertex colour; the model cannot see them, and laying them out in
  UVs would give it a texture atlas with no scene context to judge highlights by. The
  plate's own SDR->HDR curve is fitted from the pixel-aligned pair (ACEScg, log-binned,
  monotone) and applied to the vertex colours; live: 23,331 vertices, 184 above 1.0.

## What the zones buy — the zone ablation, run 2026-10-16

Design: `docs/designs/matrixzone-hdr-zone-ablation.md` (approved 2026-10-03).
The question: `sdr_detail_transfer` rebuilds every UNclipped pixel as
`sdr * 2**ratio_smooth` through a guided filter at radius 64 px, so outside the
clipped highlights the zone clips survive only through that filter's slope and
offset. How much of the delivered EXR actually depends on the 4K-tier zones,
and not on the render/4 global pass?

**Answer: most of it. The zones are not redundant outside the clipped area, and
"crops only where the SDR clips" would be SLOWER than the full run, not faster.**

### How it was measured

`AtlasMatrixZoneStitch` now writes its inputs when `ATLAS_MATRIXZONE_DUMP_DIR`
is set — `global.npy`, `zone_NN.npy` (each clip already reduced to its one
image, exactly what the stitch consumes), `sdr.npy`, `sdr_global.npy` (the SDR
the global pass was given, rebuilt with the split's own torch ops),
`plan.json` and `params.json` (every stitch / destripe / transfer value the run
used). Unset, nothing is written. The env-var read and the file writing live in
the comfy node, never in `core/` (review R2-6); `core.matrixzone` gained only
`sdr_transfer_keep_mask`, the feathered mask factored out of
`sdr_detail_transfer` so the analysis and the function share one definition
(review R2-7 is met by `params.json`, and the handle round-trips through JSON
losslessly — pinned).

`tools/matrixzone_zone_ablation.py` replays the dump offline through the SAME
chain (stitch -> band destripe -> local destripe -> `sdr_detail_transfer` ->
seam step test), replacing only the ZONE INPUTS and keeping the dumped global
pass in the global slot, so everything downstream is byte-for-byte the live
one. `tests/test_matrixzone_zone_ablation.py` pins that a V1 replay reproduces
the live node's plate to a max relative error below 1e-5.

*Specification resolved while executing (the design left R2-1 open).* V3 blends
in log2 at the zone inputs, before the stitch, with a 128 px feather around
`dilate(C, 128)`. V4 does the same, with the curve applied to the full-
resolution SDR instead of the upsampled global; V4 keeps the dumped global pass
because approach C fits its curve FROM that pass, so a shipped C still runs it.
`sdr_detail_transfer` runs on every variant's result.

*Timing source (review R2-5).* ComfyUI's websocket stream, timestamped client
side: `executing` gives node boundaries and `progress` resets once per sampler
run, so the five LTX clips are separable. The cathedral run decomposes cleanly;
the machine run's websocket dropped mid-load and its per-clip split is partly
derived (below), which is why the runtime model is quoted with its inputs.

### The two live runs

| | machine plate | cathedral plate |
|---|---|---|
| file | `atlasMachine_8k.jpg` | `atlas_cathedral_nave.png` |
| plate / render | 7680x4512 / 7680x4608 | 3840x2160 / 3840x2176 |
| zone / global | 3904x2368 (4K tier) / 1920x1152 | 1984x1152 / 960x544 |
| total | **875.9 s cold** (14 min 36 s) | **193.1 s warm** |
| model load -> first sampler step | 191.9 s | 51.7 s |
| sampler per clip | not separable (see above) | 12.9 s global; 28.7 / 20.0 / 17.7 / 19.4 s zones |
| clipped `keep > 0.5` | 14.2% | 2.2% |
| seam worst (windowed, SDR-controlled) | 1.14x | 0.82x |

The second plate was to have scattered small highlights rather than the machine
plate's sky band, and the design left the choice to the user "from the V91
CameraRaw captures". **ComfyUI_V91 no longer exists on disk.** Of what is in
V135's `input/`, the clipped mask was measured on every plate at or above 4K:
the machine plate's largest blob is 88.4% of its clipped area, `DSC_2245` 99.7%,
`8245563064` 66.9%, `mj_qwen21_i2v_CP_5k_01` 75.3% — all sky bands — while the
cathedral nave's largest blob is 30.5% with 58 blobs of 100 px or more. So the
cathedral is the scattered plate. Caveat: a 2x2 on a 4K plate puts the global
pass at 960x544 instead of 1920x1152, a WEAKER anchor than the headline case,
which biases toward the zones mattering — so the cathedral agreeing with the
machine plate is the conservative direction, not a free confirmation.

### Regions (plate resolution, disjoint, covering)

`C` is `sdr_transfer_keep_mask > 0.5` (where the HDR's own pixels are kept),
`T` is `dilate(keep > 0, 128) \ C` (the guided filter's 2r reach), `U` the rest.

| plate | C | T | U |
|---|---|---|---|
| machine 8K | 14.2% | 40.2% | 45.6% |
| cathedral 4K | 2.2% | 34.4% | 63.4% |

That by itself dents the design's premise 3: on the machine plate "only where
the SDR clips" already means C + T = 54% of the plate.

### |log2 luminance| against V1, in stops (p50 / p95 / p99)

| plate | variant | C | T | U |
|---|---|---|---|---|
| machine 8K | V2 global-ratio | 0.041 / 0.394 / 0.953 | 0.070 / 0.817 / 1.474 | 0.119 / 0.660 / **1.182** |
| | V3 global + clipped crops | 0.001 / 0.022 / 0.059 | 0.022 / 0.405 / 0.910 | 0.116 / 0.634 / **1.131** |
| | V4 global curve | 0.001 / 0.018 / 0.049 | 0.017 / 0.173 / 0.355 | 0.058 / 0.375 / **0.637** |
| cathedral 4K | V2 global-ratio | 0.270 / 1.406 / 2.283 | 0.099 / 0.420 / 0.645 | 0.097 / 0.444 / **0.724** |
| | V3 global + clipped crops | 0.009 / 0.054 / 0.119 | 0.026 / 0.230 / 0.418 | 0.096 / 0.441 / **0.722** |
| | V4 global curve | 0.008 / 0.030 / 0.056 | 0.013 / 0.091 / 0.186 | 0.036 / 0.214 / **0.369** |

Share of each region moved by more than 0.05 / 0.10 stops — how WIDESPREAD the
change is, which a percentile does not say:

| plate | variant | C | T | U |
|---|---|---|---|---|
| machine 8K | V2 | 42.7% / 19.6% | 57.5% / 41.6% | **73.9% / 55.8%** |
| | V3 | 1.5% / 0.1% | 30.6% / 19.0% | 73.3% / 54.9% |
| | V4 | 1.0% / 0.1% | 20.7% / 10.4% | 53.3% / 36.1% |
| cathedral 4K | V2 | 88.5% / 78.1% | 71.7% / 49.5% | **72.2% / 48.9%** |
| | V3 | 6.0% / 1.6% | 35.4% / 19.3% | 71.9% / 48.3% |
| | V4 | 1.3% / 0.1% | 13.4% / 4.2% | 39.2% / 18.5% |

Drop the zones (V2) and about three quarters of the UNCLIPPED plate moves by
more than the 0.05-stop bar, and about half by more than 0.10 stops — on both
plates, at 14% clipped and at 2% clipped. The median in U is 0.12 / 0.10 stops,
so this is not a tail: the whole region shifts. Codex's challenge to premise 2
is upheld, and more strongly than it was put: the guided filter reduces the
dependency on the zones, it does not come close to removing it. Looked at as a
picture, the V1 - V2 difference is structured — it rides on the machine's
metalwork and the grass, not a flat offset — so it is not something a grade
takes out.

V3 is V2 in U by construction and tracks it there; what it shows is that the
clipped core is reproduced from zone pixels alone (C p99 0.059 / 0.119 stops).
V4 is the best of the three everywhere and still 7-13x over the bar in U. Its
own fit says why: one monotone SDR->HDR curve leaves a median residual of
**0.197 stops** (machine) / **0.278 stops** (cathedral) against the global pass
it was fitted on, so the conversion is not a global tone curve.

### The seam score does not discriminate here

| plate | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| machine 8K | 1.14x | 1.30x | 1.29x | 1.36x |
| cathedral 4K | 0.82x | 0.99x | 0.79x | 0.82x |

Nothing is flagged (gate 1.5x) on any variant of either plate. V2 is seam-free
by construction — one continuous field cropped four ways — and still scores
HIGHER than V1 on both plates, because the step at the seam barely moves
(machine: 1.306 -> 1.353 stops) while the random-line baseline falls
(1.146 -> 1.044). At these levels the ratio is reading content, not seams, so
the seam test cannot rank these variants and was not used to. For V3 and V4 it
could only ever show a seam where a seam line crosses `dilate(C, 128)`; V4's
fill is a global function of the SDR and cannot produce one at all.

### Approach B would be slower

A greedy cover of `dilate(C, 128)` with zone-tier crops on 64-clean origins:

| plate | crops | crop area vs render | projected B | vs live |
|---|---|---|---|---|
| machine 8K | 6 | 1.57x | 1220 s | **1.35x** |
| cathedral 4K | 6 | 1.64x | 212 s | **1.29x** |

Six crops to cover what four zones already cover. The reviewer's R2-4 arithmetic
is confirmed with real masks: the clipped band touches all four quadrants, the
cover needs MORE zone-tier crops than the grid has zones, and the 1/3-of-live
bar was never reachable. The machine projection uses load 192 s (measured),
global 80 s and per-zone 158 s (derived: total minus load, with the global
clip assumed to cost what the 4x4 run's 1080p-tier clips did); the cathedral
projection uses its measured 51.7 / 17.0 / 23.9 s.

### Outcome

**Decision rule, branch 4 on both plates: no variant passes; the zones change
the output materially.** The design's branch 4 asks for a blind A/B of V1 vs V2
in Nuke before concluding the zones are worth their runtime — **that human
verdict is still PENDING**, and the harness has written the pack for it:
`<dump>/ablation/ab/` holds `A.exr`, `B.exr` and their -3/0/+3 sweeps with the
assignment randomised by the harness and hidden in `ab/key.json` (judge first,
read the key after). Approach B is independently dead on the runtime model, so
the A/B now decides only whether the zone tier is worth 14 min against the
global pass alone — not which speedup to build.

What this does NOT establish: the ablation measures DIFFERENCE, not
correctness. V2 differing from V1 by a stop says the quarter-tier global pass
and the zone-tier conversion disagree, not which is right. That is precisely
what the blind A/B is for. V3 also remains an UPPER BOUND on approach B — its
crop pixels come from the full run, and fresh smaller crops would see less
scene context.

**The thresholds were provisional and one of them was wrong.** `U p99 <= 0.05`
and `T p99 <= 0.10` stood up — nothing came near them, so they were never the
deciding edge. `seam worst <= 1.0x` did not: V1, the reference, scores 1.14x on
the machine plate, so the bar sat below the thing it was measuring against.
Recalibrated for any future re-score: compare against V1's own windowed worst on
that plate, and keep the shipped 1.5x gate as the absolute flag.

Reproduce: set `ATLAS_MATRIXZONE_DUMP_DIR`, run the HDR still workflow, then
`python tools/matrixzone_zone_ablation.py <dump> --write-exr --ab V1,V2`. The
harness takes 208 s on the 8K plate and 49 s on the 4K one (the design hoped
for under 2 minutes; the 8K case is 3.5).

## Comp may need paint fixes

The conversion occasionally invents soft dark/coloured spots (~30 px) on flat sky
that are not in the SDR. They are rare and a generic detector mostly flags real
content (surf, JPEG blocking the model cleaned up), so the stitch does not remove
them: expect comp to paint them out. Stripes inside SDR-clipped highlights (~14 %
of the plate) are only touched by the local destripe and may also need a comp pass.

## Honesty in the output

The EXR is tagged ACEScg and labelled in the report as a MODEL RECONSTRUCTION of
highlight radiance from a display-referred plate — not photographed HDR. Same
doctrine as the generated-object hidden side. The file says so too: `atlas:content`
carries the label, `atlas:matrixzone_params` (JSON) records how it was made (grid, zone
size and tier, mode, clip frames, overlap, anchor and split, which clean-up passes RAN
next to their widget values, SDR clip window, guided-filter radius and eps),
`atlas:seam_worst` the worst windowed seam (scored in memory before the half encode)
and `atlas:version` the Atlas version. The report's range line counts non-finite
pixels instead of printing nan.

## Open (the user's calls)

- ~~Grid / zone tier for 8K~~ — decided 2026-10-02: **2x2 at the 4K tier** (fits, faster,
  4 seams all passing). Destripe bands are capped at a quarter of the plate height:
  the stripes drift down a 2256-row zone, and one band per zone row left 0.085 stops.
- Default sequence mode — decided by gate 5's numbers (provisional: `per_zone_clip`).
- Whether the anchor's split frequency is exposed or fixed.
- **Blind A/B, V1 vs V2** (the zone ablation's branch 4). The packs are written
  and unlabelled: `output/atlas_mz_dump_machine8k/ablation/ab/` and
  `output/atlas_mz_dump_cathedral4k/ablation/ab/`. Look at `A.exr` and `B.exr`
  in Nuke at -3/0/+3, write the verdict down, then read `ab/key.json`. If the
  zone-tier plate is not judged better, the 4K-tier zones are not worth 14 min
  and the global pass alone becomes the default; if it is, the speedup track
  closes and the full run stands as built.
