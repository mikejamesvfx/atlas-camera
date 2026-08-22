# Paint-bridge provenance — where every constant came from

The vendor table in `atlas_camera/paint/vendors.py` carries numbers. This file
carries the runs those numbers came from, so a future change can tell a
measurement from a guess. Nothing here is a capability claim from a datasheet;
every figure is from a live run on a real plate.

The doctrine all of it serves:

> **The paint package selects and paints. Atlas decodes RAW, owns colorimetry,
> confines the edit, judges it against gates, and stitches the result.**
> The authorised mask stays Atlas-side because the judge must be independent of
> the editor.

---

## Affinity (Canva) 3.2.3.4646 — measured 2026-08-21

### `generativeEditImage` is regeneration, not inpainting

`DocumentCommand` exposes only `GenerateImage` and `GenerativeEditImage`. There
is **no confined-inpaint call in the SDK**, so a raster selection is a *hint*,
never a boundary — even after `selectSubject()`.

Two whole-frame runs, same call, same app version:

| plate | result | containment |
|---|---|---|
| boiler (3876×2589) | composition roughly preserved; treeline, road and paving all repainted | **0.3740** |
| street (2589×3876) | **the entire photograph was replaced by a different, generic warehouse** | ≈0 |

**So the boiler result was luck, not behaviour.** One usable-looking output from
a whole-frame call is not evidence that the call is bounded.

The earlier Cafe run recorded the same effect more mildly — changed pixels
measured up to **636 px** past the mask, p99 of strong changes **564 px** —
which was already enough to require Atlas-side matting, but not enough to show
that the composition itself could be replaced.

**Consequence:** `needs_roi_crop = True` for Affinity. Geometric confinement is
the only reliable kind: hand the model a crop, and everything outside it is
untouched by construction rather than by hope.

### Confinement results

| run | containment | seam (gate 1.25) | decision | correct by eye? |
|---|---|---|---|---|
| boiler, raw whole-frame fill | 0.3740 FAIL | 0.9992 PASS | rejected | — |
| boiler, confined (drop 320 / dilate 45 / feather 12) | 1.0000 PASS | 0.9987 PASS | accepted | **partly** |
| street, raw whole-frame fill | ≈0 | — | rejected | no |
| street, **ROI crop** (38.3% of frame) + confined (drop 40 / dilate 30 / feather 10) | 1.0000 PASS | 0.9986 PASS | accepted | **yes** |

`dilate_px = 45`, `feather_px = 12` come from the boiler run. They are tuned to
Affinity's measured spill and are **not universal**.

### Gates are necessary, not sufficient

The confined boiler plate passes **both** gates and is still visibly wrong: a
soft luminance ghost follows the old silhouette, and the regenerated treeline
inside it does not line up with the treeline outside.

The reason is structural, not incidental — **a hazy blend has a LOW rim
gradient, so a smooth wrong answer scores well.** `containment` says the edit
stayed in bounds; `seam_gradient_ratio` says it joins cleanly at the rim.
Neither says the interior content is right. Look at the picture.

### A feather is spill unless the authorised mask includes it

The containment gate rejected a "clean" feathered darken at **0.9329** because
the feather itself painted outside a binary mask. Dilating the mask by the
feather radius took the same edit to **1.0000**.

This is why `paint_confine_plate` emits the *authorised mask* — the full support
of the drop+dilate+feather ramp — and why the scorer must be handed that mask
rather than the raw object mask.

### The do-nothing baseline is unbeatable on seam

A do-nothing composite's rim IS the plate, so its seam ratio is exactly 1.0,
and a clean real edit measured 0.9996 reads as infinitesimally "worse". The
baseline comparison stays in the JSON because it is what falsifies *geometry*
candidates; for a 2D edit the decision comes from the candidate's own
calibrated gates.

### Colour: Affinity mislabels its EXR export

A plate handed over tagged `lin_rec709_scene` came back tagged
`oiio:ColorSpace = ACEScg` **with the pixel values untouched**: 1.6% of the
frame bit-identical, 17.2% inside 1e-4. A least-squares 3×3 fit made the
residual *worse* (0.0683 vs 0.0557), proving the difference is content, not
colorimetry.

`read_plate` gives a file's declared tag **unconditional** precedence — it
overrides even an explicitly passed `input_colorspace` — so naming the
colourspace downstream is **inert** as a defence. The real protection is the
Atlas-side **re-tag** in `paint_confine_plate`, which writes the confined plate
with the original's colourspace.

### SDK friction (recorded, and pushed upstream via `add_sdk_hint`)

* `Document.export(path, opts)` needs a `FileExportOptions` **wrapper** with a
  `.handle`; the SDK ships no wrapper class, so build one inline.
* `setRasterSelectionFromPolygon` wants `polygon.handle`, not the wrapper.
* `createGrowShrinkRasterSelection` is not a method on `Document`.
* `Document.close()` is `NOT_IMPLEMENTED`, so documents accumulate across a
  scripted session.
* Filesystem access is Desktop-only; paths take forward slashes, and a mangled
  path reports `PERMISSION_DENIED` rather than "not found".

---

## Adobe Photoshop (Beta) 27.4.0 — capabilities established 2026-08-21, behaviour NOT yet measured

Everything recorded for Photoshop so far comes from inspecting the shipped
binary and one live COM session. **No generative fill has been run or scored.**
See `reports/photoshop_bridge_probe.md` for the full probe.

`needs_roi_crop` is therefore **`None` — unmeasured**, and
`vendors.require_roi_decision` refuses to choose, forcing an explicit
`--roi`/`--no-roi`. `dilate_px` / `feather_px` are carried over from Affinity as
a starting point and are explicitly provisional.

The open question this bridge exists to answer: `syntheticFill` exposes a
`syntheticFillMode` enum with an **`inpaint`** value, which Affinity had no
equivalent of. Whether it is genuinely bounded by the selection is settled only
by running it on a full uncropped frame and scoring containment against the
dilated+feathered authorised mask — on **both plates, twice each**, because one
pass is exactly how the Affinity boiler result misled.

---

## Bugs found while building the shared core

Recorded here because each was found by comparing two things that were supposed
to agree, and each would otherwise have silently changed what a score means.

* **The SciPy-free dilation fallback disagreed with SciPy.** It was a separable
  BOX dilation built from `np.roll` — a square rather than a disc (56 extra
  corner pixels at r=6 on a single point), and `roll` **wraps**, so a mask
  touching the left edge grew onto the right edge of the frame. The authorised
  region therefore depended on whether SciPy happened to be installed. Replaced
  with an exact non-wrapping disc decomposition, pinned by a parity test on
  border-touching masks.
* **OIIO dropped the colourspace tag on every EXR written under a studio
  config.** It only persists `oiio:ColorSpace` when the config supplies a
  `colorInteropID`; fn-nuke_cg v1.0.0 does not. An untagged `.exr` read on
  `auto` guesses ACEScg, so a Rec.709-linear RAW sidecar would come back
  quietly wrong — and the re-tag defence above silently stopped working. Fixed
  with an `atlas:ColorSpace` attribute Atlas writes unconditionally.
* **`half` bit depth outweighed the edit being measured.** `write_exr`'s auto
  compression selects the lossy dwab DCT codec for `half`, which moves
  essentially every pixel past the scorer's 1e-4 change threshold: the first
  confined boiler plate scored 9,918,844 changed pixels against an
  8,305,439-pixel edit. Gate-bound plates are written `float`/zip.
* **`torn_fraction` could be negative.** It counts emitted faces against
  whole-quad slots, and a `sub_quad_boundary` cut emits faces from partial
  quads, so a live cleanplate layer reported **−0.0932**. The QA gate already
  preferred `torn_fraction_whole_quad`; the MCP census did not, so the two
  disagreed about the same mesh.

---

## SAM3 has no semantic gating

Not a paint-package finding, but it shapes every mask handed to one. Asking for
a broader concept list does not refine a matte, it unions whatever the head
matches:

* `"concrete footing pad"` on the boiler plate returned **the entire brick
  plinth — 50.8% of frame** — while still missing the black steel legs.
* `"bollard, sign post"` on the street plate also claimed the red shop signage.

The reliable fix for a ground-standing object is geometric, not semantic:
`--drop-px` extends the matte straight **down**, where legs, footings and
contact shadow actually are, with none of the sideways bloat an equivalent
dilation costs.
