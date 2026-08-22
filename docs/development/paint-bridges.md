# External paint-package bridges

How Atlas hands a plate to a paint package, gets it back, and decides whether
to believe it. Vendor-neutral; the per-vendor measurements live in
[reports/paint_bridge_provenance.md](../../reports/paint_bridge_provenance.md)
and [reports/photoshop_bridge_probe.md](../../reports/photoshop_bridge_probe.md).

## The doctrine

> **The paint package selects and paints. Atlas decodes RAW, owns colorimetry,
> confines the edit, judges it against gates, and stitches the result.**

The authorised mask stays Atlas-side because **the judge must be independent of
the editor**. A package that both chose the region and painted it cannot also
be the thing that certifies it stayed inside.

## Not to be confused with the angle-patch bridge

Atlas has two Photoshop paths, and they have different invariants. Do not merge
them; forcing one set of constraints onto the other breaks both.

| | `AtlasExtractAnglePatch` / `AtlasImportAnglePatch` | the paint bridge |
|---|---|---|
| carries | a camera POSE | a full-raster plate |
| format | 8-bit sRGB PNG proxy | float EXR, colour-managed |
| resize | **forbidden** (reprojection depends on it) | n/a, ROI offset is in the manifest |
| judged by | reprojection | falsification gates |

## The pipeline

```
tools/paint_roi_export.py       # crop + manifest  (only if the vendor needs it)
  -> the paint package           # selects and paints
tools/paint_confine_plate.py    # composite back through the authorised ramp
tools/paint_roundtrip_score.py  # gate it
```

Logic lives in `atlas_camera/paint/`; the `tools/` entries are argument parsing
only, so everything is importable and testable without a paint package
installed.

## Confinement is a construction, not a hope

`confine` composites `edited` over `original` through a **drop → dilate →
feather** ramp, and outside that ramp the output is **bit-identical** to the
original. That is what makes containment 1.0 a property of the code rather
than a property of the vendor's good behaviour.

* **`--drop-px` grows the mask straight DOWN.** A ground-standing object's
  legs, footings and contact shadow are below it, not around it, so
  gravity-directed growth covers them without the sideways bloat an equivalent
  dilation costs. Widening the segmenter's concept list instead is a trap —
  SAM3 has no semantic gating.
* **The authorised mask is the ramp's full SUPPORT**, not the object mask. A
  feather is spill unless the authorised mask includes it: the gate correctly
  rejected a "clean" feathered edit at 0.9329 for exactly this. Hand the
  scorer the mask `confine` wrote.

## Whether a vendor needs the ROI crop is MEASURED

`atlas_camera/paint/vendors.py` is tri-state on purpose:

* `True` — the package regenerates past its selection (Affinity, measured).
* `False` — genuinely bounded; only ever set from a passing containment
  measurement at full resolution.
* `None` — **unmeasured**, and the tools then refuse to choose, forcing an
  explicit `--roi`/`--no-roi`.

That refusal is the point. It is what stops "Photoshop has an `inpaint` mode so
it is probably bounded" from becoming a fact by being typed into a default.

## Colour: three rules that each cost a run to learn

1. **A declared tag wins unconditionally.** `read_plate` gives a file's
   `oiio:ColorSpace` precedence over an explicitly passed `input_colorspace`.
   So naming `AtlasLoadPlate.input_colorspace` is **inert** as a defence
   against a mislabelled vendor export. The real defence is the Atlas-side
   **re-tag** in `confine`, which writes the confined plate with the
   original's colourspace.
2. **Write gate-bound plates as `float`.** `write_exr(bit_depth='half')`
   selects the lossy dwab DCT codec, which moves essentially every pixel past
   the scorer's 1e-4 threshold — once scoring 9,918,844 changed pixels against
   an 8,305,439-pixel edit. `half` is for delivery copies nothing is gated on.
3. **Record the OCIO config identity, not just a colourspace name.** A plate
   tagged `ACEScg` under two different configs is two different plates. Every
   manifest and score report carries `config_path` + `config_sha256`, so a
   claim that two applications shared a config is checkable.
   `atlas_camera/paint/ocio.py` scopes `$OCIO` **per process** — never
   machine-wide, which would silently change every existing Atlas read.

Related, and load-bearing: OIIO only persists `oiio:ColorSpace` when the active
config supplies a `colorInteropID` for the space. Older studio configs do not,
and OIIO then writes **no tag at all** — so Atlas writes its own
`atlas:ColorSpace` unconditionally and reads it as a fallback. Without it, an
untagged `.exr` read on `auto` guesses ACEScg, and a Rec.709-linear RAW sidecar
comes back quietly wrong.

## Gates are necessary, NOT sufficient

| gate | what it says | threshold |
|---|---|---|
| `containment` | the edit stayed inside its brief | spill ≤ 0.01 |
| `seam_gradient_ratio` | it joins the plate cleanly at the rim | ≤ 1.25 |
| `sky_violation` | it did not paint into sky | ≤ 0.005 |

None of them says the interior content is right. A confined boiler plate passed
both available gates while being visibly wrong, because **a hazy blend has a
LOW rim gradient, so a smooth wrong answer scores well.**

Every accepted run therefore needs a recorded visual check beside the numbers,
and a report must never present a green gate as a correct result.

Two artefacts are always produced, and this is not optional:

1. the **raw return** — diagnostic, scored, recording the vendor's actual
   containment behaviour;
2. the **confined composite** — what ships, containment 1.0 by construction.

Shipping only the composite hides the vendor's real behaviour behind Atlas's
own safety net.

## Adding a vendor

1. Add a `VendorProfile` with `needs_roi_crop=None` and honest `provenance`.
2. Reach the package however it allows (Photoshop: COM + ExtendScript, see
   `atlas_camera/paint/photoshop/`). Keep script generation pure and testable;
   keep the transport thin.
3. Run a full-frame fill on **two plates, twice each**, and score the raw
   return. One pass is not evidence — that is exactly how the Affinity boiler
   result misled.
4. Record every number, including the failures, and only then set
   `needs_roi_crop` and re-measure `dilate_px` / `feather_px`.
