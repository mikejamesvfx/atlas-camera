# Affinity showcase workflows — RAW in, generative cleanplate out, gated

**Date:** 2026-08-21 · **Affinity:** 3.2.3.4646 Win32 · **ComfyUI:** 0.33.1,
121 Atlas nodes · **Cameras:** Fujifilm X-H2 (40 MP, XF16-55mm), Sony A7S II.

Four shipping example workflows built on real camera RAW from
`docs/dev/RansacPlaneDev/rawphotos`, plus two new tools that make an external
generative edit reproducible and judgeable. Everything below was executed
live; every number is measured, not asserted.

## Headline

**Two complete Affinity round-trips passed both falsification gates**, which
the previous Cafe run never managed (its tight matte failed seam at 1.6130 and
its wide matte at 1.2917):

| run | containment (gate 1.0) | seam ratio (gate 1.25) | decision | looks right? |
|---|---|---|---|---|
| boiler, whole-frame fill, raw | 0.3740 FAIL | 0.9992 PASS | rejected | — |
| boiler, confined (drop 320 / dilate 45 / feather 12) | **1.0000 PASS** | **0.9987 PASS** | **accepted** | partly — see §4 |
| street, whole-frame fill, raw | ~0 | — | rejected | no — new building |
| street, **ROI crop** + confined (drop 40 / dilate 30 / feather 10) | **1.0000 PASS** | **0.9986 PASS** | **accepted** | **yes** |

## 1 · The finding that changes the recipe

`generativeEditImage(prompt)` is **image-to-image regeneration, not
inpainting** — even after `selectSubject()`. There is no strictly-confined
inpaint API: `DocumentCommand` exposes `GenerateImage` and
`GenerativeEditImage` and nothing else, so a selection is a *hint*, never a
boundary.

Handed a whole frame, twice, same call, same version:

* **boiler plate** — composition roughly preserved, containment **0.3740**.
  The treeline, road and paving were all repainted, so they no longer line up
  with the master outside the object silhouette.
* **street plate** — the entire photograph was replaced by a different,
  generic warehouse. Nothing of the original scene survived.

So the boiler result was **luck, not behaviour**. The only reliable
confinement is geometric: crop a tight ROI, edit that, composite it back.
Inside a 38.3%-of-frame ROI the same call removed a bollard and a traffic cone
cleanly while leaving the building, signage, kerb and overhead wires intact.

> **Recipe:** `paint_roi_export.py` → Affinity → `paint_confine_plate.py
> --roi-manifest` → `paint_roundtrip_score.py`. Skipping the crop is how you
> get a new photograph.

## 2 · Colour: Affinity mislabels its EXR export

A plate handed over tagged `lin_rec709_scene` comes back tagged
`oiio:ColorSpace=ACEScg` **with the pixel values untouched** — 1.6% of the
frame returned bit-identical and 17.2% inside 1e-4, which no global colour
transform permits. A least-squares 3×3 fit made the residual *worse* than no
transform at all (0.0683 vs 0.0557), confirming the difference is content, not
colorimetry.

`read_plate` believes a file that self-describes, so reading the export on
`input_colorspace='auto'` converts Rec.709-linear data as though it were
ACEScg and shifts every primary. **Every showcase graph names
`AtlasLoadPlate.input_colorspace` explicitly.** Never `auto` on Affinity output.

## 3 · Two measurement traps found while gating

**The dwab codec outweighed the edit.** `write_exr(bit_depth='half')` selects
a lossy DCT codec by default, which moves essentially every pixel past the
scorer's 1e-4 change threshold. The first confined boiler plate scored
9,918,844 changed pixels against an 8,305,439-pixel edit. `paint_confine_plate.py`
therefore defaults to `float` (zip, lossless) — the plate exists to be measured.

**`torn_fraction` can be negative.** It is `1 − emitted_faces / whole_quad_slots`,
and with `sub_quad_boundary` on a cut quad emits faces from *partial* quads, so
the count exceeds the slot count. A live cleanplate layer reported **−0.0932**.
The QA gate already preferred `torn_fraction_whole_quad`; the MCP viewport
census did not, so the two disagreed about the same mesh. Fixed in
`atlas_camera/mcp/comfy_http.py` with two tests.
*(Side benefit: a negative value is positive proof `sub_quad_boundary` is
actually cutting — which the previous session's open thread #1 could not
establish.)*

## 4 · Gates are necessary, not sufficient

The confined **boiler** plate passes both gates and is still visibly wrong:
the regenerated treeline inside the silhouette does not line up with the
treeline outside it, and a soft luminance ghost follows the old silhouette.

`containment` says the edit stayed in bounds. `seam_gradient_ratio` says it
joins cleanly at the rim. **Neither says the interior content is right** — and
a hazy blend has a *low* rim gradient, so a smooth wrong answer scores well.
This is the counterexample the Cafe run never produced, because the car was
small relative to frame. Look at the picture.

The **street** plate, cropped to an ROI, passes the same gates *and* is
correct. That is the difference the crop buys.

## 5 · `raw_meta` is worth a whole workflow

Measured on the same boiler frame, same solver:

| input | focal reported | sensor | confidence |
|---|---|---|---|
| JPEG only | 26.8 mm | 36 mm assumed (full-frame) | 0.877 |
| RAF + `raw_meta` | **20.6 mm** | **23.5 mm measured** | **0.94** |

And on the night plate, where no RAW exists: **40.5 mm estimated against
53 mm actual EXIF — a 24% error — at confidence 0.777.**

lensfun also *applied* a real distortion profile for the `XF16-55mmF2.8 R LM WR`
on the X-H2, contrary to the node tooltip's pessimism about Fuji X bodies.

## 6 · Four correct refusals — a capture spec, measured

`atlas_raw_multiview_affinity_patch_workflow` **ships on placeholder paths and
does not run on the reference assets.** Every RAW set in the shoot was fed to
it and every one was refused, each for a different and correct reason. The
engine never fabricated a rig:

| outcome code | what it measured |
|---|---|
| `metadata_mismatch` | focal 20.6 / 24.9 / 23.4 mm, orientation 1 vs 8, dimensions [3876,2589] vs [2589,3876] |
| `insufficient_overlap` | "photos 2-3 have 31 mutual matches; at least 48 are required" |
| `dynamic_scene_contamination` | 247 matches but consensus in **5 of 16** grid cells — all in wind-moving tree canopy. Match *count* is not match *coverage*. |
| `ambiguous_motion_model` | 49 matches collapsed to 7 essential inliers at `angle_deg` 164 — nonsense survived as raw matches, geometry did not |

The reference photos are bracket and variation frames, not an overlapping
survey. Shipping the graph pointed at a set that cannot register would teach
the wrong lesson, so it ships pointed at nothing and carries these four numbers
as its capture spec instead.

Same story on the night burst: twelve frames in three seconds at 53 mm measured
`angle_deg` **0.359666** between frames. That is an *exposure* burst, not a
baseline — a burst is not a multi-view set just because it has many frames.
Workflow 4 was rebuilt as an honest single-frame night showcase.

## 7 · What shipped

**Workflows** (`examples/`, generated by
`tools/build_affinity_showcase_workflows.py`, all validated against live
`/object_info`):

| workflow | status |
|---|---|
| `atlas_raw_affinity_cleanplate_workflow` | **runs green** — 4 layers, boiler matte 31.3%, cleanplate_bg 100% |
| `atlas_raw_street_affinity_declutter_workflow` | **runs green** — 5 layers, confined Affinity cleanplate wired as `plate_depth` |
| `atlas_burst_night_affinity_relight_workflow` | **runs green, zero flags** |
| `atlas_raw_multiview_affinity_patch_workflow` | validates; placeholder paths by design (§6) |

**Tools** (written as `affinity_*`; renamed to vendor-neutral `paint_*` the
next day when the bridge gained a second vendor — the logic now lives in
`atlas_camera/paint/` with these as thin CLIs. See
`docs/development/paint-bridges.md`.)

* `tools/paint_roi_export.py` — cuts the ROI crop + manifest that makes an
  external generative edit survivable.
* `tools/paint_confine_plate.py` — composites the edit back through a
  dropped/dilated/feathered matte, and writes the **authorised mask** the
  scorer must be given (a feather is spill unless the authorised mask includes
  it). `--drop-px` grows the matte *downward*, which is where a ground-standing
  object's legs, footings and contact shadow are.

**Fix** — `atlas_camera/mcp/comfy_http.py` census prefers
`torn_fraction_whole_quad` (+2 tests). Pin counts updated in
`tests/test_example_workflows.py` and `tests/test_mcp_comfy_http.py` (12 → 16).

**SDK hints pushed upstream** via `add_sdk_hint` (6 total): the
`FileExportOptions` wrapper requirement, `createGrowShrinkRasterSelection`
not existing on `Document`, path-escaping vs `PERMISSION_DENIED`, the
regeneration-not-inpainting finding, the absence of a confined inpaint API,
and `Document.close()` still `NOT_IMPLEMENTED`.

## 8 · Open threads

1. The boiler cleanplate's interior ghost (§4). A per-channel gain/offset fit
   on the blend annulus would likely kill it; not attempted — it is colour
   surgery and the seam gate already passes.
2. `n_filled_cells` still reads 0 across the census. The producer fix
   (`d4e412b`) is on disk via the junction but the running ComfyUI process
   predates it — carried over from the previous session, still needs a restart
   to confirm.
3. SAM3 over-reach, twice: "concrete footing pad" returned the entire brick
   plinth (50.8% of frame) while missing the black steel legs; "bollard, sign
   post" also claimed the red shop signage. No semantic gating, as recorded.
4. The street `street_furniture` layer is full-band terrain at 1.6% matte, cut
   in the shader — the geometry-level mask restriction thread (previous
   session's #2) is unchanged.
5. Band-overlap flags on both cleanplate graphs are expected (the cleanplate
   sits behind a matte-cut hero at farthest-highest priority) but are still
   reported as flags.
6. Camera height 4.0003 m on the street plate looks like the elevated-vantage
   artifact; an `AtlasScaleOverride` would pin it.
