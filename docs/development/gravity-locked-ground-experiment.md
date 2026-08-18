# Gravity-locked ground — experiment report

**Date:** 2026-08-18
**Plate:** `DSC_2245.NEF` — NIKON D810, 24 mm prime, 7380×4928, EXIF focal, `undistort=applied`. Eye-level NYC street: road filling the lower half, a parked Nissan Rogue with clean tyre contacts, a sedan, pedestrians, kerb line.
**Graph under test:** `research/atlas_hero_02_photo_to_editable_scene_workflow_ground.json`
**Instruments:** `atlas_camera/core/ground_consensus.py`, `tests/test_ground_consensus.py`, `tools/ground_consensus_probe.py`
**Raw measurements:** `docs/dev/ground_consensus/report.json` (gitignored)

---

## Verdict

**NO-GO on the hypothesis as posed. GO on a different, larger finding.**

The hypothesis was that RANSAC is being asked to solve ground orientation and
ground height together, and that locking the normal to gravity would fix the
wrong vertical offset. Both halves fail on measurement:

- **The normal is already gravity-locked**, in every single-image path.
- **The height estimator is not where the error is.** Every knob it has —
  near-field ROI, exclusion mask, weighting, choice of robust estimator —
  moves the answer by **≤4%** on the real plate. The error is **48%**.

The 48% lives somewhere else, and the probe found it: the graph runs **two
depth models** and adopts the camera height from the one that is wrong, then
rescales the geometry from the one that is roughly right to match it.

---

## 1. Audit — what the code actually does

Read from the implementation, not the docs.

```
CURRENT GROUND NORMAL SOURCE
    Hard-coded world +Y. Never fitted.
    proxy_geometry.py:519, plane_extraction.py:131, room_layout.py:129
    all emit n_g = [0, 1, 0]. `ground_normal_min` (|n_y| > 0.90) only SELECTS
    candidate pixels; the selected normals are never averaged or SVD-fitted.
    The only true ground RANSAC is multiview_solver._fit_ground_plane:678,
    on the multi-view SfM path, unreachable from a single-image street solve.

CURRENT GROUND OFFSET SOURCE
    48-bin histogram mode of candidate world_y over the 1-99 percentile range,
    then a median refine of the points within plane_tolerance.
    solver.py:839-847, depth_geometry.py:283-292, relief_mesh.py:200-209.
    camera_height = -y0; scale = cam_y / (cam_y - y0).

CURRENT SAMPLE MASK
    inner & (v > horizon_y) & (|n_y| > 0.90) & ~depth_edge
      & isfinite(world_y) & (depth > 0)          -- solver.py:824

CURRENT OUTLIER REJECTION
    depth > 1e-4 and finite; 3x3 nanmedian pre-filter (_median_filter_3x3:663);
    depth-discontinuity edge mask |dd| > 0.05*2*d; normal gate |n_y| > 0.90;
    1/99 percentile clip, then |y - y0| < max(0.15, 0.03 * span).
    No robust M-estimator anywhere.

CURRENT DISTANCE WEIGHTING
    None. Every candidate pixel casts one unweighted vote into the histogram
    and one unweighted sample into the median, at every call site.
    The single weighted ground computation in the repo is the WALL anchor
    blend, proxy_geometry._anchor_wall_to_ground:213-250 -- and the graph
    under test has ground_anchor=False.

CURRENT GROUND VALIDATION
    Candidate floors (200 / 300 / 80 depending on site); camera_height < 0.3 m
    rejected; degenerate-offset guards; confidence = ground coverage of the
    bottom 20% image band, adopted at >= 0.30 (_HEIGHT_ADOPT_CONFIDENCE:655).
    NOT validated: the depth model's absolute metric scale, the planarity of
    the accepted ground, or the ground normal itself.
```

`solver.py:724-742` already documents a street scene at confidence 0.374 whose
height was ~70% too large, and explicitly says no candidate filtering can
detect that from a single depth map.

---

## 2. The failure mechanism, measured

### 2.1 Two depth models, one unmeasured ratio

The graph runs both:

| node | model | what it feeds |
|---|---|---|
| 2 `AtlasLearnedSolveFromImage` (`height_mode='measure_from_depth'`) | DA-V2-Metric-Outdoor | **the adopted camera height** |
| 3 `AtlasDepthMap` | MoGe (`moge-2-vitl-normal`) | **all the geometry** |

`estimate_ground_scale` / `fit_ground_and_scale` then rescale the MoGe world
about the camera so MoGe's own ground lands on Y=0 — anchored to the **V2**
height. Nothing in the pipeline ever compares the two.

Measured on the plate:

| model | camera height | confidence |
|---|---|---|
| V2-Metric-Outdoor (adopted) | **2.694 m** | 0.799 |
| MoGe (geometry) | **1.822 m** | 0.806 |

Ratio **1.479**. Applied world rescale **1.5056×**. Both models clear the 0.30
adopt threshold by a wide margin, and both are confident. Confidence measures
plane agreement, not scale — exactly as its own docstring warns.

### 2.2 An independent arbiter says V2 is the wrong one

The plate contains a Nissan Rogue (badge legible), nominal height **1.684 m**.
`solver.metric_height_from_reference` (tier-1 scale) recovers camera height
from one known-size vertical object with no assumed eye height. Four pixel
picks, to expose marking error rather than hide it:

| pick | camera height |
|---|---|
| rogue_rear_hi | 1.819 m |
| rogue_rear_axle | 1.904 m |
| rogue_rear_lo | 2.006 m |
| rogue_front_axle | 2.144 m |

Median **≈1.96 m**, honest band **1.82–2.14 m**.

- MoGe **1.822 m** — inside the band, −7% vs median.
- V2 **2.694 m** — **+38%** vs median, outside the band on every pick.

### 2.3 What that does to the geometry

Back-projecting the Rogue's roof and its rear-wheel ground contact through the
MoGe depth map:

```
MoGe-measured SUV height        1.493 m   (true 1.684 m, -11%)
after the pipeline's 1.5056x    2.248 m   (true 1.684 m, +33%)
range to SUV (MoGe)            10.00 m
```

The pipeline takes a world that was 11% short and makes it **33% too tall, in
the wrong direction**, by anchoring it to the model that was wrong. That is the
reported symptom — ground that will not sit under cars, geometry at the wrong
scale — and it is a cross-model anchoring failure, not a plane fit.

### 2.4 The plane fits its own depth map fine

Signed residual of every candidate ground pixel against the fitted plane,
binned by range (`residual_profile`, over candidates the fit *threw away* as
well as the ones it kept — inlier statistics cannot show this):

| range | MoGe median / p90 abs | V2 median / p90 abs |
|---|---|---|
| 5–10 m | +0.010 / 0.162 m | −0.050 / 0.133 m |
| 10–20 m | +0.012 / 0.173 m | +0.001 / 0.152 m |
| 20–40 m | −0.034 / 0.532 m | +0.134 / 1.838 m |
| 40 m+ | **+1.103 / 21.27 m** | **+4.377 / 22.65 m** |

Inside 40 m the plane tracks the observed road to within a few centimetres in
both models. Past 40 m both are unusable. So the ground is not "floating"
relative to its own evidence; the whole world is at the wrong scale.

### 2.5 The normal, measured rather than assumed

`probe_ground_normal` fits a ground normal two independent ways — the
component-wise median of the candidate pixels' own normals, and the smallest
singular vector of the near-field candidate points — and reports the angle from
world +Y. It never applies anything.

| depth | median-of-normals | SVD (near half) | the two fits disagree | near-field planarity |
|---|---|---|---|---|
| MoGe | 1.77° | 1.67° | 0.32° | 0.013 |
| V2 | 0.78° | 1.38° | 1.74° | 0.055 |

The road is gravity-aligned to within **1.8°**. Orientation is provably
innocent on this plate — which is the evidence the gravity-lock premise was
missing, and it makes the offset the only remaining suspect.

---

## 3. The candidate estimator, and why it does not help

`core/ground_consensus.py` implements the gravity-locked scalar consensus in
full: near-field ROI (rectangular or trapezoidal), explicit exclusion mask, five
weightings, six robust estimators all computed side by side, and a tolerance
derived from the data's own weighted MAD instead of `0.03 * span`.

Swept over the plate (reference truth ≈1.96 m):

| depth | cell | median | mad_median | mode | ransac1d | tolerance |
|---|---|---|---|---|---|---|
| V2 | full, no mask | 2.673 | 2.699 | 2.695 | 2.724 | 0.340 |
| V2 | full + SAM3 sky+clutter | 2.678 | 2.700 | 2.698 | 2.731 | 0.328 |
| V2 | bottom 45%, centre 70% | 2.735 | 2.749 | 2.754 | 2.754 | 0.244 |
| V2 | bottom 30%, centre 50% | 2.757 | 2.762 | 2.766 | 2.768 | 0.123 |
| MoGe | full, no mask | 1.811 | 1.823 | 1.818 | 1.792 | 0.421 |
| MoGe | full + SAM3 sky+clutter | 1.813 | 1.823 | 1.817 | 1.791 | 0.414 |
| MoGe | bottom 45%, centre 70% | 1.840 | 1.844 | 1.839 | 1.819 | 0.342 |
| MoGe | bottom 30%, centre 50% | 1.853 | 1.850 | 1.831 | 1.835 | 0.261 |

**Within** a model, the full knob space spans ≤0.10 m (≤4%). **Between**
models, 0.87 m (48%). The estimator is not the lever.

Two secondary results worth keeping:

- **The near-field ROI genuinely tightens the tolerance** — V2 0.340→0.123 m,
  MoGe 0.421→0.261 m. That does not move the height, but it changes which
  pixels are classified as ground, so it is real and it is cheap.
- **The SAM3 mask moves the height by 0.001–0.015 m here.** Sky covers 19.1%
  and car/person/bicycle 4.4%, but on this framing the clutter is largely not
  producing horizontal-normal ground candidates anyway. The graph computes the
  sky mask at node 17 and wires it to nothing; on this plate that costs almost
  nothing, which is itself the finding.

### What the synthetic tests found

`tests/test_ground_consensus.py` (32 cases, all green) defends the maths and
records two results that changed the shape of this report:

- **A flat surface at the wrong height *is* the estimator's real weakness.**
  Given a large horizontal sheet 0.9 m above the road — a car roof, a raised
  plaza — the shipping estimator returns **0.700 m against a true 1.600 m
  (−56%)**, and its confidence *rises* with contamination (0.56 → 0.81). An
  exclusion mask recovers it exactly. `|n_y| > 0.90` cannot reject a horizontal
  contaminant, and the shipping estimator has no mask input to reject it with.
  This mechanism is real; it just is not what dominates *this* plate.
- **Distant road pixels do not outvote near ones.** Under perspective the near
  ground occupies *more* of the frame (measured: near 8890 / mid 8890 / far
  8636 candidates), and the horizon-compressed rows are edge-rejected on top of
  that. A mode- or median-like reduction ignores a far tail even when the road
  bends 3 m away over its length. The far field's influence is through
  **spread**, not votes — it inflates `span`, hence the tolerance and the
  histogram bin width. The test is kept as an explicit negative result.

---

## 4. Success criteria, scored

| # | criterion | result |
|---|---|---|
| 1 | ground orientation stable and gravity-consistent | **PASS** — 1.8° from +Y; it was never the problem |
| 2 | implied camera height physically plausible | **FAIL for the adopted path** — V2 2.694 m vs 1.82–2.14 m reference |
| 3 | near-field road residual improves materially | **NO** — already ±0.03 m inside 40 m; nothing to win |
| 4 | ground passes beneath foreground vehicle contacts | **FAIL at pipeline scale** — SUV lands 33% too tall after the 1.5056× rescale |
| 5 | analytic ground extends behind occlusions | out of scope this round |
| 6 | no regression to observed MoGe geometry | **PASS** — nothing shipping was touched |

The experiment is not a success on its own terms. It is a success at finding
why the ground is wrong.

---

## 5. Failure cases confirmed or excluded

| candidate | verdict on this plate |
|---|---|
| MoGe near-field absolute scale wrong | **partly** — SUV 1.493 m vs 1.684 m, −11% |
| V2 and MoGe disagree, so the anchor is wrong | **CONFIRMED, dominant** — ratio 1.479, world rescaled 1.5056× |
| visible surface not planar | excluded inside 40 m (±0.03 m); true past 40 m |
| camera on a slope or kerb | excluded — 1.8° |
| gravity wrong | excluded — 1.8° |
| no clean ground visible | excluded — 500k+ candidates |
| multiple ground levels | not observed |
| camera-height scale unresolved | **this is the finding** |

**Is a camera-height prior needed?** No — and it would have been actively
harmful. Both models' answers (1.82 m, 2.69 m) straddle a 1.0–2.2 m handheld
prior, so a prior would have rejected the *correct* one on some plates and
accepted the wrong one here. `height_prior` in `ground_consensus` reports and
never clamps, for the same reason `solver.py:735-739` rejected a plausibility
penalty: elevated and drone shots are legitimate inputs. What actually
adjudicated this plate was a **known-size object**, which is tier-1 scale and
already shipping.

---

## 6. Recommendation

**NO-GO** on replacing or reworking the height estimator. The normal lock
already exists; the scalar reduction is already robust; the remaining knobs are
worth ≤4% against a 48% error.

**GO** on the cross-model anchor, and the smallest production change is a new
node rather than an edit to any existing one — that keeps every positional
`widgets_values` contract and every default untouched, and leaves the current
estimator and RANSAC exactly where they are.

Proposed, in priority order:

1. **Measure the ratio and surface it.** Whenever a solve's adopted camera
   height comes from one depth model and the geometry from another, compute
   both ground heights and report `h_ratio` through `core.scene_health` (the
   only place trust verdicts may come from). A ratio of 1.48 is not a warning
   about the ground — it is a warning that the scene scale is unresolved. Today
   this is silent.
2. **A new experimental node (🔬, `EXPERIMENTAL_NODE_CLASS_MAPPINGS`,
   `node_registry.py:421`)** that emits a ground plane as its own primitive:
   - gravity-locked normal by default, with the measured tilt reported;
   - `ground_tilt_deg` / `ground_roll_deg` manual overrides, applied to the
     **primitive's own `transform_matrix`** via `depth_geometry.plane_transform`
     — never to the world, since world +Y *is* the solve's gravity and rotating
     it would lean every facade. A separately rotated ground exports to a DCC as
     one rotated object;
   - `camera_height` override, since on this evidence that is the parameter that
     actually matters;
   - near-field ROI and an `exclude_mask` input, both cheap and both real;
   - `provenance="inferred"` / `trust` tags, reusing the keys already flowing
     end-to-end (`nodes_geometry.py:5436`, forwarded at `blender/measured.py:185`).
3. **Prefer a reference object when one is visible.** It settled this plate in
   one measurement and it is already implemented.

Not recommended: distance weighting, a new robust estimator, semantic ground
segmentation, or any change to the existing `estimate_ground_height_from_depth`.
Measured, they buy ≤4%.

---

## 7. Reproducing

```bash
XFORMERS_DISABLED=1 PYTHONPATH=. python tools/ground_consensus_probe.py \
    --raw C:/Users/miike/Pictures/atlas_raws/atlas_raws/DSC_2245.NEF \
    --references docs/dev/ground_consensus/references.json \
    --stages ABCE

python -m pytest tests/test_ground_consensus.py -q     # 32 passed
```

`XFORMERS_DISABLED=1` is required for MoGe on a GPU newer than xformers' built
kernels (sm_120 here); without it MoGe's DINOv2 raises
`NotImplementedError: No operator found for memory_efficient_attention_forward`.
Depth maps, the solve and the SAM3 masks are cached under
`docs/dev/ground_consensus/cache/`, so re-runs are seconds.

Nothing shipping was modified. No node was registered, no workflow altered, no
default changed, and the RANSAC paths are untouched.
