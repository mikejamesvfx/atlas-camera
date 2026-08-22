# Affinity bridge — OCIO-correct generative-fill roundtrip, scored

**Date:** 2026-08-21 · **Status:** **RUNG A ACHIEVED LIVE** — full
agent-driven roundtrip executed against Affinity 3.2.3: ACEScg EXR loaded,
AI subject-select + generative fill run, EXR exported, scored, and stitched
into a full-res cleanplate. Details in "Live run" below; the original spike
notes are kept underneath.

## Live run (2026-08-21, DSC_2552 car removal)

Connection: Affinity's MCP is a local **SSE server on IPv6 loopback only** —
`http://[::1]:6767/sse` (a client resolving localhost to 127.0.0.1 sees
nothing). Protocol `2025-11-25`. 11 tools; everything substantive goes through
`execute_script` (JavaScript SDK) after reading the `preamble` doc topic —
which is enforced **per session**: every fresh SSE session must re-read it
before any other doc read works.

Working pipeline (car ROI, 2255×963 crop of the 7380×4928 NEF-decoded
ACEScg master; filesystem access is Desktop-only, so the roundtrip folder is
`Desktop/atlas_affinity_roundtrip/`):

1. `app.documents.load(<roi>.exr)` — EXR opens under the app's OCIO config.
2. `doc.selectSubject()` (AI) + `createGrowShrinkRasterSelection(12)`.
3. `doc.generativeEditImage('empty asphalt parking lot pavement, … no car,
   no vehicle')` — car AND its shadow removed, lines continued.
4. Export preset `'OpenEXR 32-bit linear'` → filled EXR back on Desktop.
5. `tools/affinity_roundtrip_score.py` + Atlas-side matte + stitch into the
   full-res master → `DSC_2552_cleanplate_affinity_acescg.exr`.

Scores (falsification gates, DSC_2552):

| variant | containment | seam ratio (gate 1.25) |
|---|---|---|
| raw fill, card-mask brief | 0.5888 FAIL | 0.9976 PASS |
| matted tight (card+15px) | 1.0000 PASS | 1.6130 FAIL |
| matted wide (card+45px, feather 12) | 1.0000 PASS | 1.2917 FAIL (marginal) |

Reading: the fill itself joins cleanly (0.9976) but **Canva's generative edit
repaints far beyond the selection** — changed pixels measured up to 636 px
from the mask (p99 of strong changes: 564 px). Confinement must therefore be
Atlas-side matting, and the matte boundary through half-erased shadow is what
the tight-matte seam honestly measures. For the shipped cleanplate the raw
fill was blended at the ROI border only (32 px ramp) — the in-fill seam is
the passing 0.9976 one.

SDK gotchas recorded (also pushed via `add_sdk_hint`):

- `setRasterSelectionFromPolygon` wants the native handle: pass
  `polygon.handle` — the JS wrapper forgets to unwrap (SDK bug).
- `Polygon.create()` + `addPointXY(...)` + `close()`; there is no
  `Polygon.createFromPoints` (that's `Spline`).
- Silhouette-shaped selections make the model REDRAW the object; prompts
  naming the object ("the shadow of a car") likewise regenerate it. The
  reliable removal recipe is selection + an explicit negative ("no car, no
  vehicle") — or `selectSubject` and let the app own the mask.
- Scripted polygon-selection + fill misbehaved with several documents open
  (`Document.close()` is NOT_IMPLEMENTED, so stale docs accumulate);
  `selectSubject` on `documents.current` with a single doc open was solid.

Division of labour that this run settled: **Affinity selects and paints;
Atlas decodes RAW, owns colorimetry, confines (matte), judges (gates), and
stitches.** The authorised mask stays Atlas-side because the judge must be
independent of the editor.

---

## Original spike notes (pre-live, kept for provenance)

## What this is

Atlas exports colour-managed ACEScg EXR plates with per-object masks. Affinity
(affinity.studio, Canva) is the first mainstream paint package in this pipeline
with three things Photoshop's bridge lacked: a real OCIO config loader (the
machine's `fn-nukecg` ACES 1.3 / OCIO v2.1 config is already selected in its
Color settings), correct EXR alpha association options, and a built-in **MCP
server** (Settings → Model Context Protocol → Enable Affinity MCP, with
per-capability permissions: desktop file access, network, saved scripts, Canva
AI Studio features).

The bridge: one Claude session drives BOTH MCPs — Atlas's
(`python -m atlas_camera.mcp`, 12 tools) and Affinity's — so an inpaint
roundtrip is: export EXR + mask → Affinity opens under OCIO → generative fill
constrained to the mask → save to a NEW path → Atlas re-scores the edit with
the same falsification metrics that falsified the hole-splat run. An external
fill is held to exactly the standard an internal one is.

## The scoring leg (built, measured)

`tools/affinity_roundtrip_score.py` — original EXR + edited EXR + authorised
mask in, gate table + JSON report out. Metrics from
`atlas_camera/core/plate_falsification.py`:

- **alpha** = the pixels that actually changed (not the request mask — an edit
  that strayed is measured by where it strayed)
- **containment** (definitional): changed pixels inside the authorised mask
- **seam_gradient_ratio** (empirical, gate 1.25): the join at the changed
  region's rim, self-referenced against the plate's own rim busyness

Measured on the DSC_2552 release master (7380×4928, ACEScg), synthetic edits:

| edit | containment | seam ratio | decision |
|---|---|---|---|
| noise fill inside `mask_car.png` | 1.0000 PASS | 2.8252 FAIL | **rejected** (exit 2) |
| feathered darken, binary mask | 0.9329 FAIL | 0.9996 PASS | **rejected** — the feather itself spilled |
| feathered darken, mask dilated 15 px | 1.0000 PASS | 0.9996 PASS | **accepted** (exit 0) |

Two findings worth keeping:

1. **A feather is spill unless the authorised mask includes it.** The
   containment gate caught a "clean" feathered edit painting outside a binary
   mask. The brief for any external fill must hand over the mask *dilated by
   the feather radius*, or the fill will be honestly rejected.
2. **The do-nothing baseline is unbeatable on seam for 2D edits.** A do-nothing
   composite's rim IS the plate, so its seam ratio is 1.0 exactly; a clean real
   edit measured 0.9996 and reads as infinitesimally "worse". The baseline
   comparison stays in the JSON (it is what falsifies *geometry* candidates),
   but the script's decision and exit code come from the calibrated gates.

## Affinity MCP surface — the recorded rung

Rungs (one knob per rung, strictest first):

- **A — full automation:** Claude opens, masks, fills, saves via Affinity MCP
  tools. *Blocked in this session:* no Affinity MCP server is registered with
  this Claude Code install, and tool enumeration cannot run non-interactively.
- **B — agent opens/saves, human clicks the fill.** Same blocker.
- **C — scripted file roundtrip, manual Affinity leg:** **this is the rung
  reached.** The file contract works today: hand Affinity the absolute paths
  (`master_acescg.exr` or a card EXR + its mask PNG), edit under the loaded
  OCIO config, save to a NEW path (never overwrite a release asset — the
  episode ledger is content-addressed), then run the scorer.

To unlock rung A/B (user action, one-time):

1. In Affinity: Settings → Model Context Protocol → Enable Affinity MCP
   (already on, per 2026-08-21 screenshots). It runs a **local server**.
2. Register it with Claude Code from an interactive session (`claude mcp add`
   — consult Affinity's MCP connection article for the exact endpoint/command;
   public docs describe a local server + scripting-panel tool surface).
3. In the session, enumerate `tools/list` and record the surface verbatim
   here. Decision gates for rung A: can it open a path / apply a mask
   selection / run generative fill / export EXR with ACEScg + correct alpha
   association?

## Demo script (rung C, works today)

```bash
python tools/affinity_roundtrip_score.py \
  --original <release>/DSC_2552/assets/master_acescg.exr \
  --edited   <scratch>/master_edited_affinity.exr \
  --mask     <scratch>/mask_car_dilated.png \
  --out      reports/affinity_roundtrip_report.json
```

To see the accepted edit in the scene: wire the edited EXR through
`AtlasCleanPlateLayer.plate_ref` in the verify graph
(`atlas_roundtrip_verify.json`) — a plate is a LAYER, no episode surgery.
