# Flash3D and VolFill for Atlas hidden-geometry recovery

Investigation, 2026-08-15. Research only — no production code was changed. All
work lives under `research/volfill/`; nothing under `atlas_camera/` was touched.

> **Status: measured, both truth passes done.** Licence audit, architecture
> trace, coordinate mapping, runtime, step sweep, a 12-plate evaluation, the G5
> photograph-truth scoring pass and the Blender labelled-visibility scene are all
> complete. Remaining: the multi-angle diagnostic renders and artist review,
> which the generated multi-viewport workflow enables.

---

## Executive summary

**Flash3D: REJECT.** Three independent commercial blockers, any one of which is
fatal, plus a runtime that cannot be built on current Atlas hardware. Not
benchmarked — there is no point measuring something unusable.

- The repository has **no LICENSE file** (`GET /repos/eldar/flash3d` →
  `license: null`). Per the standing rule, that is *not safe to redistribute*
  until proven otherwise. Last upstream push 2025-06-02.
- It pulls `diff-gaussian-rasterization` (the Inria/graphdeco fork) — **non-commercial
  research licence**.
- Its encoder is UniDepth (`models/model.py` → `UniDepthExtended`) — **CC BY-NC 4.0**.
- It pins `torch==2.2.2` / `xformers==0.0.25.post1` / CUDA 11.8. Atlas hardware is
  an RTX 5090 (sm_120, Blackwell) on torch 2.11+cu130. Torch 2.2.2 has no
  Blackwell support and the CUDA rasterizer will not compile for sm_120.

**VolFill: PURSUE, experimental tier, with a named gate.** Licence chain clean
end-to-end (MIT code, MIT weights). Runs on sm_120 out of the box at **~6 s and
5.3 GiB per plate — 4× faster than Wan VACE** and producing geometry rather than
pixels. The canonical→Atlas-world round-trip is closed-form and passes on 11 of
12 plates. Resolution is scene-dependent (4.8–244 cm voxels) and a **depth band
recovers 7.9×** on deep exteriors.

**Against real photographed truth it works.** On the sh001 rig (two surveyed
poses, 14.600 m apart), VolFill's *invented* geometry explains genuinely hidden
structure at **1.026 m median against a 1.055 m rig noise floor**, versus
**2.536 m** for the current Atlas relief-mesh baseline — a **2.5× improvement**,
at the limit of what the answer key can resolve. Where frame 2 can adjudicate,
invented geometry sits **0.678 m from real photographed structure**.

Two gates, both real:

1. **Fidelity, self-detectable — two independent checks.** Predicted surface
   agrees with the visible surface at only visF 0.35–0.41 **even on VolFill's own
   demo scenes**, and about a third of plates diverge outright. Usefully,
   divergence announces itself twice over, both without ground truth:
   **occluded fraction >85% ⟺ visF <0.2**, and **invented geometry rendering in
   FRONT of the visible surface at zero camera offset** (1.3% on a sound plate,
   61.6% on a broken one — physically impossible for hidden geometry). Gate on
   both.
2. **Domain — a caution, not a gate.** On synthetic/CG imagery the MoGe stage
   degrades (a textured Blender scene registers 54% off metric; `golden_corridor`
   produced *zero* output). But the degradation is INVISIBLE to every run-time
   signal: synthetic volumes score visF 0.37–0.38 and pass the divergence gate,
   indistinguishable from photographs. Prefer photographic plates; do not expect
   a gate to catch a CG one.

So: a gated hypothesis generator for photographic plates, not a trusted geometry
source and not for CG input.

The decisive context is not in either paper. **Atlas already built this feature
and unshipped it over licensing.**

---

## What Atlas already has

`atlas_camera/core/hidden_geometry.py` is, today, the "Atlas Layered Rays"
representation the brief proposes as future work:

- `register_layers_to_depth` — median-ratio registration of a layered stack's
  layer-0 into Atlas depth units, returning `rel_mad` as a quality signal.
- `select_hidden_surface` — per-pixel *first-clearing-layer* selection with a
  scene-adaptive margin. Deliberately not a fixed layer index: for a solid
  occluder, layer 1 is usually its own back face.
- `fill_hidden_gaps` — Jacobi diffusion of scattered predictions into one coherent
  surface, because fragmented depth shreds the downstream relief mesh.

Two backends were written against it: LaRI (`inference/lari_hidden_geometry.py`,
~0.2 s regression) and World-Tracing DiT (`inference/wt_hidden_geometry.py`,
~17 s at 20 steps).

**`AtlasPredictHiddenGeometry` is no longer registered.** Verified against
`node_registry.NODE_CLASS_MAPPINGS`: the standard tier (see
[NODE_CATALOG](../NODE_CATALOG.md) for the current count) holds zero
hidden-geometry keys. The class body survives in `comfy/nodes_geometry.py` as dead
code, and downstream consumers still reference its outputs
(`comfy/node_helpers.py:264`, `comfy/nodes_inpaint.py:1353`). It was removed
because **both backends were legally unusable** — LaRI ships no licence at all,
World-Tracing is CC BY-NC-ND 4.0 with gated checkpoints.

So the question this investigation actually answers is narrower and much more
actionable than the brief assumed. Atlas does not need a new subsystem, a new
representation, or a new adapter boundary. It needs a **hidden-geometry backend
whose licence permits shipping**. That is exactly what VolFill is.

---

## Licence audit

Full table and method: `research/volfill/out/license_audit.md`. Summary:

| Component | Licence | Commercial |
|---|---|---|
| VolFill code | MIT (text read in full) | ✅ |
| `volfill_dit.pth`, `volfill_vae.pth` | MIT (HF card) | ✅ |
| `third_party/moge` | MIT + Apache-2.0 (DINOv2) | ✅ |
| `third_party/trellis` | MIT | ✅ |
| MoGe-v2 weights | MIT, not gated | ✅ |
| `spconv-cu130`, `cumm-cu130` | Apache-2.0 | ✅ |
| `utils3d` | MIT | ✅ |
| Training data | **unstated — UNKNOWN** | ⚠ |

### The LaRI gate — cleared

VolFill's README says it "builds on LaRI", and LaRI ships no licence. Top-level
MIT cannot cure copied unlicensed code, so this needed a file-level check rather
than a reading of the acknowledgements.

`research/volfill/audit_provenance.py` tokenises every source file (comments and
docstrings stripped) and scores every VolFill file against every LaRI file with
`difflib.SequenceMatcher` on the token stream — structure survives renaming and
reformatting, so a lightly-edited copy still scores high.

Best match across all 15 non-trivial files: **0.306**, and that is a 377-token
file against a generic DINOv2 MLP block. Near-copies score 0.85+. **No
LaRI-derived code in the inference path.** The lineage is architectural, not
textual.

This is materially different from Atlas's existing backends: LaRI and
World-Tracing *must* be user-cloned and can never be vendored. VolFill could be
vendored or pip-installed outright.

### Residual risks

1. **VolFill does not bundle its vendored licences.** `third_party/` contains no
   LICENSE or NOTICE files and the TRELLIS sources carry no copyright headers, so
   VolFill is itself technically out of MIT compliance with the code it vendors.
   Curable on our side by adding the upstream texts — but it must actually be done.
2. **Non-PyPI wheel index.** `spconv-cu130` / `cumm-cu130` come from
   `ratharog.github.io/cumm-spconv`, a personal GitHub Pages index. Upstream is
   Apache-2.0, but these binaries are third-party builds. Pin versions and hashes,
   or build from source.
3. **Training data unstated.** `volfill/amodal/datasets/scannetpp_tudf.py` implies
   ScanNet++, whose own terms are research-oriented. The *weights* are MIT; if
   training-data terms matter for a given deployment, that needs the authors'
   answer. Flagged, not resolved.

---

## Architecture trace — VolFill

Two-stage latent generative model: a hybrid 3D VAE (TRELLIS sparse conv) compresses
a 256³ TUDF to a 16³ latent; a latent DiT with flow matching generates it,
conditioned on frozen MoGe-v2 image features **and** a visible-geometry latent.

| Concern | Where | Behaviour |
|---|---|---|
| Visible geometry | `preprocess/visible_tudf_prep.py` | MoGe points → occupancy → EDT → TUDF |
| Canonical frame | `estimate_isotropic_bounds` | axis-aligned isotropic bbox, **no rotation** |
| Conditioning | `model/conditioner/moge_conditioner.py` | MoGe frozen, always `.eval()` |
| Sampling | `amodal/flow_matching.py` | Euler ODE, default 50 steps, cfg 3.0 |
| Decode | `model/vae/latent_vae_sparse_decoder.py` | latent → dense 256³ |
| Extraction | `visualize.py` | `tudf <= threshold`, or marching cubes |

### The representation

On disk, `pred_tudf_256.npz['tudf']` is `(256, 256, 256)` float32 with **array axes
`(z, y, x)`**, values in `[0, truncation_voxels]` (default 3.0) in **voxel units**.
The in-memory tensor is `[-1, 1]`; `_save_result` denormalizes on the way out.

`metadata.json` carries `bbox_min` (xyz metres), `extent_xyz`, `truncation_voxels`,
`field_range`, `pred_resolution`.

The field is **unsigned** — there is no inside/outside, only distance-to-surface.
Consequences for Atlas: surfaces extract cleanly (threshold or marching cubes at a
small positive level), but occupancy and watertightness do not come for free, and
a thin surface reads the same as a thin shell. For Atlas's purposes — hidden
*surfaces* to project onto and export — unsigned is sufficient; for solid modelling
it is not.

**Upstream bug, avoided:** `_tudf_to_pointcloud` selects `tudf < self.tudf_threshold`
with default `0.0`, which matches nothing on a non-negative field. It is dead code
(commented out at its only call site) and `visualize.py` is the live path. Do not
copy that predicate — `research/volfill/tests/test_tudf_to_atlas.py` pins the
correct `<=` semantics.

---

## Atlas camera compatibility — solved, closed-form

This was expected to be the hard part. It is not.

VolFill's "canonical frame" is **not a learned canonicalization**. Traced in
`estimate_isotropic_bounds`: it takes the MoGe-v2 visible point cloud in *metric
camera space*, computes an axis-aligned bbox (1st/99th percentile by default,
`robust_percentile=1.0`), centres it, and inflates it to an isotropic cube with a
10% margin. **Translation and uniform scale only — no rotation.**

So the whole chain is arithmetic with exactly one estimated scalar:

```
voxel_size  = extent_xyz / 256                       # isotropic by construction
p_moge_cam  = bbox_min + (idx_xyz + 0.5) * voxel_size    # exact, from metadata.json
p_atlas_cam = diag(1, -1, -1) @ (p_moge_cam * s)         # exact, axis flip
p_world     = inv(camera_view_matrix) @ [p_atlas_cam, 1] # exact
```

- **MoGe camera space** is OpenCV: x right, y **down**, +z **forward**.
- **Atlas camera space** is OpenGL-style per `core/relief_mesh.py:182-184`
  (`x=(u-cx)/fx*d`, `y=-(v-cy)/fy*d`, `z=-d`): x right, y **up**, **−z** forward.
- So `diag(1, -1, -1)`; `det = +1`, a rotation, not a mirror — handedness preserved.
- `camera_view_matrix` is row-major **world→cam**; `cam_to_world = inv(view_matrix)`
  (`core/relief_mesh.py:11-12`). Built from the 4×4 only, never the 3×3 — Atlas
  hard rule.

`s` (MoGe → Atlas depth scale) is measured with Atlas's **existing**
`core.hidden_geometry.register_layers_to_depth`, which already performs exactly this
median-ratio fit and returns `rel_mad` as its quality signal. No second estimator
was written.

Implementation: `research/volfill/tudf_to_atlas.py`. Conversion lives at the adapter
boundary only, per the Atlas layering rule; nothing in `atlas_camera/` imports it.

**Verification: 11/11 tests pass** (`research/volfill/tests/`, no GPU, no weights).
They pin the failure modes that would otherwise pass silently: axis transposition
(distinct indices per axis), the half-voxel offset, the y/z sign flip, handedness,
view-matrix direction (camera origin → its own world position), threshold
inclusivity, and the empty-surface case.

---

## Resolution gate

Because the bbox is an isotropic cube over *all* visible points, voxel edge =
`2·half_scale / 256`. This scales with scene depth, and Atlas plates are deep:

| Scene | Extent | Voxel edge |
|---|---|---|
| Interior / desk | 4 m | **1.6 cm** |
| Room / object | 10 m | 3.9 cm |
| Courtyard | 40 m | 15.6 cm |
| Street plate (DSC_2289 class) | 200 m | **78 cm** |

At 78 cm per voxel a street plate cannot represent a doorway reveal, a window
recess, or a railing — the geometry Atlas is trying to recover is smaller than the
quantum. This is pinned as an assertion in the test suite rather than left as prose.

Those are the *predicted* figures that motivated the experiment. The measured
edges across 12 plates (4.8–244 cm) and the depth-band result that recovers 7.9×
are in **Resolution — measured, and the lever that works** below; where the two
sections differ, the measured one wins. Notably `robust_percentile=1.0` trims the
far 1% before fitting, so a facade-dominated street (sh001) lands at 6.1 cm on
the full frame — better than this prediction suggested.

---

## Runtime — measured

Isolated environment at `research/volfill/.venv` (py3.11, torch 2.10.0+cu130).
Two deviations from upstream, both recorded:

- **`triton==3.6.0` removed** — no Windows wheel exists (Linux-only on PyPI). It
  is a `torch.compile` dependency; eager inference does not need it. Pinned in
  `research/volfill/requirements-win.txt`.
- spconv/cumm resolve to the upstream-pinned 2.4.0 / 0.9.1; cp311 Windows wheels
  exist on the rathaROG index.

**sm_120 works.** Upstream targets "CUDA 13.0 / RTX 40-series" and Blackwell is
untested there, but torch 2.10+cu130 ships `sm_120` in its arch list and a
spconv sparse conv executed on the 5090 in 0.114 s. No source rebuild needed.

| Metric | Value |
|---|---|
| GPU | RTX 5090, 32 GiB, driver 610.62 |
| Model load, cold (weights download) | 103 s, once |
| Model load, warm | ~12 s |
| MoGe visible pass | 3.0–4.2 s |
| Flow-matching sampling, 50 steps | **2.4–2.8 s** |
| End-to-end, warm | **~6 s/plate** |
| Peak VRAM | **5.2–5.3 GiB**, flat across every plate and step count |

Step sweep (sh001, fixed seed):

| steps | sampling | peak VRAM |
|---|---|---|
| 50 | 5.64 s | 5.3 GiB |
| 25 | 4.62 s | 5.3 GiB |
| 16 | 4.12 s | 5.3 GiB |
| 8 | 4.08 s | 5.3 GiB |
| 4 | 3.74 s | 5.3 GiB |

(Those figures include the MoGe pass; net sampling is ~2.6 s at 50 steps.)
**Sampling is not the bottleneck** — a fixed ~3.5 s of MoGe plus dense 256³ VAE
decode dominates, so dropping 50→4 steps saves under 2 s. There is no reason to
run cheap: use 50.

Against the existing slate (`docs/dev/occlusion_arms_2026-08-14`): Wan VACE
CausVid 4-step is 40 s cold / 25.5 s warm, WT-DiT ~17 s at 20 steps. VolFill at
~6 s warm is **4× faster than VACE** and produces geometry rather than pixels.

## Plate evaluation — 12 plates

`visF` = F-score at 2 voxels of the predicted surface against VolFill's own
visible TUDF; `pres` = mean distance from visible surface to nearest predicted
surface, in voxel edges; `round-trip` = the canonical→camera mapping proof
(PASS when the median lands within one voxel edge).

| plate | voxel cm | visF | occluded % | pres (voxels) | round-trip |
|---|---|---|---|---|---|
| ghosttown | 20.5 | 0.38 | 66.4 | 3.29 | PASS |
| cathedral_nave | 31.9 | 0.35 | 64.1 | 4.10 | PASS |
| sh001_street | 6.1 | 0.37 | 76.0 | 2.42 | PASS |
| dsc2289_street | 68.0 | 0.36 | 68.8 | 3.60 | PASS |
| jungleruins | 13.9 | 0.18 | 77.9 | 7.10 | FAIL |
| portal | 4.8 | 0.13 | 88.5 | 5.15 | PASS |
| templecity | 184.1 | 0.14 | 90.9 | 6.59 | PASS |
| newyork_birdseye | 56.9 | 0.18 | 88.5 | 3.69 | PASS |
| coastal_alley | 42.8 | 0.01 | 99.6 | 10.09 | PASS |
| oceancastle | 244.1 | 0.12 | 90.8 | 7.73 | PASS |
| scifi_hangar | 16.7 | 0.25 | 75.9 | 3.66 | PASS |
| golden_corridor | 1.0 | 0.00 | 0.0 | — | degenerate |

**Round-trip: 11/12 PASS.** The mapping is byte-identical code across every
plate, so `jungleruins` failing at 2.68× floor is not a coordinate bug — it is
foliage, where MoGe's points are noisy and sparse so the voxelized visible
surface diverges from the raw pointmap used as reference. Consistent with the
long-standing Atlas finding that foliage fragments hidden-geometry predictions.

**`golden_corridor` produced ZERO surface voxels.** Extent collapsed to 2.56 m —
MoGe failed on the synthetic low-texture corridor, so VolFill had nothing to
condition on. A real failure mode: no texture, no visible geometry, no amodal
completion.

### The in-domain control — the important negative

Visible F-scores of 0.35–0.38 looked like out-of-domain failure. **Running
VolFill's own shipped demo scenes refutes that:**

| scene | voxel cm | visF | pres (voxels) | occluded % |
|---|---|---|---|---|
| VolFill scene1 | 0.9 | 0.41 | 2.23 | 64.7 |
| VolFill scene2 | 1.4 | **0.05** | **8.17** | **96.1** |
| VolFill scene3 | 1.4 | 0.41 | 2.27 | 70.7 |

Atlas's good plates (0.35–0.38) sit at **near-parity with VolFill's own demos
(0.41)**, and VolFill's *own* scene2 fails as hard as `coastal_alley`. So the
failures are not caused by Atlas plates being exteriors — the same failure occurs
in-domain. "Trained on interiors" is a much weaker constraint than assumed.

### A free self-diagnostic gate

Across all 15 volumes, in-domain and out, one relation holds without exception:

> **occluded fraction > 85% ⟺ visF < 0.2**

When the predicted surface stops intersecting the visible surface, the prediction
has diverged from the plate. That is computable **with no ground truth at all**,
from data every run already produces — a falsifiable gate in the style of the
occlusion slate's G-gates, and arguably the most portable result here.

## Resolution — measured, and the lever that works

Voxel edge tracks scene extent (`2·half_scale/256`), and measured edges span
**4.8 cm to 244 cm** across the plate set. Uniform rescaling (e.g. a VFX 1/10
working scale) cannot help: the bbox is FIT to the data, so voxel and feature
shrink together, and Atlas's own scale is applied on the return leg (the `s`
scalar) and never reaches VolFill. Only reducing **extent** helps.

Two candidate constraints, tested as a 2×2 on the worst plate (dsc2289, 25 m band):

| arm | extent | voxel | visF |
|---|---|---|---|
| neither | 174.0 m | 68.0 cm | 0.36 |
| ROI crop only | 169.4 m | 66.2 cm | **0.09** |
| **depth band only** | 22.1 m | **8.65 cm** | **0.28** |
| ROI × band | 18.7 m | 7.31 cm | **0.01** |

**Depth band yes, ROI crop no.** The band gives a **7.9× resolution gain** while
keeping fidelity broadly intact (0.36→0.28). The ROI crop buys almost no extra
resolution — the isotropic cube is sized by the longest axis, which on a street
is depth, not width — and **collapses fidelity**, because cropping strips the
surrounding context the amodal prior conditions on. The ROI×band mesh is 98.7%
invented, i.e. squarely in the divergence regime above.

This converts the exterior verdict from "reject" to "workable with a depth band".

---

## Ground truth, pass 1 — the G5 photograph rig

`research/volfill/g5_geometric_truth.py`. The sh001 rig photographs one scene
from two surveyed poses **14.600 m** apart, so surfaces hidden behind occluders
in frame 1 are directly visible in frame 2. Frame 2's reconstruction is therefore
an answer key for exactly the geometry frame 1 could not see.

Answer key at stride 6: **242,866 hidden points**, 56,980 already-visible,
815,150 out of frustum (a 14.6 m step reveals mostly new content).

The **noise floor** is the number everything must be read against: where BOTH
frames saw the same surface, their reconstructions disagree by a median of
**1.055 m**. That is the limit of what this key can resolve.

### Recall — is hidden truth explained? (truth → candidate)

| candidate | median | mean | p90 |
|---|---|---|---|
| baseline relief mesh | 2.536 m | 2.482 m | 3.587 m |
| VolFill, invented only | **1.026 m** | 1.335 m | 2.808 m |
| VolFill, all surface | 0.932 m | 1.253 m | 2.639 m |
| baseline + VolFill invented | **0.981 m** | 1.110 m | 1.892 m |

**VolFill's invented geometry lands at the noise floor** (1.026 m vs 1.055 m)
while the baseline relief mesh — which cannot represent hidden structure at all —
sits **2.5× worse**. This is the first hard evidence that the model recovers real
occluded structure rather than plausible fiction.

### Precision — is what it invented actually there? (invented → truth)

| subset | median | mean |
|---|---|---|
| all invented → hidden truth | 7.910 m | 8.645 m |
| all invented → any frame-2 point | 3.628 m | 6.101 m |
| **invented inside frame 2's view → any frame-2 point** | **0.678 m** | 1.092 m |
| invented inside frame 2's view → hidden truth | 1.763 m | 3.433 m |

The first two rows are unfair and are reported only to be explicit about it:
VolFill predicts the whole isotropic cube, much of which frame 2 never observed
either, so the key cannot adjudicate it. **44.1%** of invented geometry does land
inside frame 2's view, and there it sits **0.678 m from real photographed
structure — below the rig's own 1.055 m noise floor.**

Registration: MoGe → solve scale 2.4847 at `rel_mad` 0.080 (healthy).

**Verdict on this pass: VolFill genuinely recovers hidden structure.** Where the
answer key can judge, the invented surface is as close to reality as the key can
measure, and it beats the current Atlas baseline by 2.5×.

## Ground truth, pass 2 — the Blender labelled-visibility scene

`research/volfill/blender_truth_scene.py` (runs inside Blender 5.2 headless) and
`blender_truth_eval.py`. A courtyard built around four occlusion cases — a broad
slab hiding a back wall, a box hiding a cylinder, a **thin post**, and a grazing
low step — plus door/crate scale cues. 1.71 M area-weighted surface samples in
Atlas world coordinates, so truth is **exact**: no depth model in the loop, and
the brief's visibility classes are decidable rather than estimated.

Truth classes from the render camera: **395,104 VISIBLE, 222,631 OCCLUDED,
1,094,353 OUT_OF_FRUSTUM**. Per-object occlusion runs 0.9% (thin post — it
occludes, it is barely occluded) to 22.9% (the stacked crate behind the slab).

`UNSUPPORTED` is scored on the PREDICTION, not the truth: predicted points
further than max(3 voxels, 0.15 m) from any real surface — pure invention.

### The confound, found and partly fixed

The first build of the scene was untextured, and the result was alarming until
the registration number explained it: **MoGe → truth scale 2.73**, when a metric
predictor on a metric scene should read ~1.0.

| | untextured | + procedural texture and scale cues |
|---|---|---|
| MoGe → truth scale (ideal 1.0) | 2.73 | **1.54** |
| UNSUPPORTED | 94.8% | **61.3%** |
| prediction → truth, median | 0.844 m | **0.674 m** |
| recall of OCCLUDED truth, median | 1.053 m | **0.823 m** |
| recall of VISIBLE truth, median | 0.801 m | 0.742 m |

Adding surface detail halved the unsupported fraction and nearly halved the scale
error. So the first run was largely measuring **MoGe collapsing on a flat
synthetic render** — the same failure class as the `golden_corridor` plate, which
produced zero surface voxels. The harness is sound; the input domain was the
problem.

Even textured, scale is still **54% off** and **61% of the prediction is
unsupported**, against a real photograph's 0.678 m precision. The honest reading:

> **The pipeline works on photographs and is unreliable on synthetic/CG imagery.**

That is a practical Atlas finding beyond VolFill, because Atlas is routinely
pointed at AI-generated marketing plates and CG renders. It is consistent with
`golden_corridor` (synthetic, zero output) and with the known `assumed_default`
scale failures on AI cityscape plates. Anything monocular in the stack inherits
it.

**Verdict on this pass: inconclusive for VolFill, conclusive about the domain.**
The exact-truth harness is built and reusable, but a synthetic scene cannot
currently adjudicate a MoGe-conditioned predictor. To use it properly the scene
needs photoreal assets and lighting, not procedural noise on primitives.


---

## Multi-angle diagnostic renders

`research/volfill/diagnostic_renders.py`. Nine orbit offsets
(0, +/-2, +/-5, +/-10, +/-20 deg) x four passes: visible relief mesh
(plate-textured), invented geometry (flat), combined (z-buffered), and invented
geometry ramped by distance BEHIND the visible surface.

The offsets **orbit the scene pivot** rather than panning. A camera rotating
about its own centre produces no parallax, so nothing behind an occluder would
ever be exposed and the exercise would show nothing.

A dedicated per-vertex-colour rasterizer is used instead of
`core.projection_render.render_scene`, which renders textured meshes through UVs
and cannot express provenance or behind-distance shading. The flat pass is the
important one: a textured novel view **flatters** a prediction, because the plate
projects onto whatever geometry exists and wrong geometry still looks like the
photograph.

### Two defects the renders caught that the metrics did not

**1. Invented sky.** VolFill fills its entire cube, including the region MoGe
masked out as sky — where there is no surface at all. On sh001 this smothered the
frame at +20 deg. Rejecting invented geometry that projects into MoGe's sky mask
removes **18.9%** of it and drops the 0-degree in-front fraction from 3.25% to
**1.06%**. Atlas is sky-aware throughout; a hidden-geometry backend must be too.

**2. The provenance label was wrong.** "Invented" was first measured against
VolFill's own visible TUDF — a sparse 6 cm voxelization — so a grazing road
surface read as invented even though Atlas's relief mesh covers it perfectly. The
useful question is *what does this ADD to what Atlas already has*, so provenance
is now measured against the **relief mesh** itself.

### A second self-diagnostic gate, purely geometric

At **zero offset**, genuine hidden geometry must sit BEHIND the visible surface,
so the fraction of invented geometry rendering in FRONT of it should be ~0. It is
a physical consistency check that needs no ground truth and no second model:

| plate | in-front @ 0 deg | registration `rel_mad` | reading |
|---|---|---|---|
| sh001 (real-truth plate) | **1.27%** | 0.080 | sound |
| dsc2289, band-clipped | **61.55%** | 0.443 | **misregistered / diverged** |

61% of the prediction in front of the visible surface is physically impossible
for hidden geometry. The band-clipped dsc2289 volume is broken, and neither the
voxel-size win (8.65 cm) nor its visF (0.28) revealed it — only the render did.

This joins the >85%-occluded rule as a second no-truth gate. (The `rel_mad`
column tracks it on both plates, but that needs a second depth source and n=2 is
not enough to claim it as a general rule — noted, not asserted.)

### What the geometry actually looks like

On sh001 the behind-distance pass reads as **coherent architecture**: facade
planes, a doorway recess, bollards in silhouette, with the re-predicted ground
correctly at zero-behind and the deep facade continuation far behind. It is not
noise, and it is not a smeared curtain. In-front fraction rises monotonically
with orbit angle (1.3% at 0 deg to 43-67% at +/-20 deg) exactly as genuine hidden
structure should when parallax brings it around an occluder edge.

Artefacts: `out/diag_sh001_novel/` and `out/diag_dsc2289/`, plus contact sheets.


---

## The rusty-boiler capture set — object scale, and the band knee

Shots sh003/sh004/sh005 (Fuji X-H2 RAF, 2026-08-09) and sh006 (older Sony JPG):
a riveted twin-vessel boiler in a park. A strongly self-occluding OBJECT — the
front vessel hides the rear one, and the far side of each cylinder is hidden —
which is much closer to VolFill's training regime than a street.

Straight off the plate, every shot scored badly: visF **0.08–0.24**, occluded
fraction **82–95%** (above the divergence line on nearly all of them), 6 of 11
round-trips FAIL. The cause is in the voxel sizes — **12–58 cm for an object
roughly 3 m across**. The isotropic bbox was being sized by the distant treeline,
not by the subject.

### The band sweep finds a knee, and it is not where intuition puts it

sh004_0, `--max-depth` swept:

| band | voxel | visF | occluded % | round-trip |
|---|---|---|---|---|
| 3 m | — | — | — | **clips everything** (`No valid points remain`) |
| 4 m | **1.70 cm** | 0.12 | 90.1% | PASS but diverged |
| 5 m | 2.59 cm | 0.23 | 75.7% | PASS |
| 6 m | 2.79 cm | 0.26 | 77.8% | PASS |
| **8 m** | 3.39 cm | **0.42** | **61.9%** | **PASS — peak** |
| 12 m | 4.73 cm | 0.35 | 74.8% | PASS |
| 20 m | 8.21 cm | 0.24 | 83.0% | PASS |
| none | 31.44 cm | 0.14 | 89.7% | FAIL |

An **inverted U**. The 8 m band gives 9.2× finer voxels and **visF 0.42 — the
best score in this entire evaluation**, above VolFill's own demo scenes (0.41).

The photographer recalls standing 4–5 m from the boiler, and the 3 m failure
confirms the subject starts past 3 m. Yet banding AT the subject is as bad as no
band at all (0.12 vs 0.14) and reads as diverged, **while producing the finest
voxels of the whole sweep (1.70 cm)**. Resolution improves monotonically as the
band tightens; fidelity peaks and then collapses.

Two rules fall out — and the FIRST one did not survive replication (see
"Band knee, replicated" below; it holds for this scene only):

1. ~~**Band to roughly 2x the subject distance.**~~ **REFUTED at n=3.** True for
   the boiler it was derived from, false for the street and the ghost town. The
   real rule is that the knee is scene-specific and must be swept. What survives
   is the SHAPE: an inverted U, with a real cost to clipping too tight — the band
   strips the ground plane and surround the amodal prior conditions on, the same
   context-vs-resolution tension the ROI crop exposed.
2. **Voxel size alone is a trap.** The best-resolution arm is among the worst
   arms. Without the occluded-fraction gate, a tuner optimising voxel size would
   walk straight into the divergence regime.

Applying the 8 m band across shots: all round-trips PASS at **2.1–3.4 cm**, but
fidelity splits by shot — sh004_1 0.42 and sh006_0 0.38 are good, sh003_1 0.19 and
sh005_0 0.13 are poor, and the gate correctly flags sh005_0 (88.7% occluded).
Per-shot band tuning is required; one global value will not do.

## Scoring as a PROJECTION surface, not a reconstruction

Atlas projects texture onto approximate geometry and moves the camera modestly. It
is not doing reconstruction, and metric Chamfer therefore over-penalises it in a
computable way. Decompose the error at each point into **radial** (along the source
camera's view ray) and **lateral**:

    screen_error_px ~= (radial * sin(t) + lateral * cos(t)) / depth * focal

A radial error is invisible from the source camera — the texture travels down the
same ray — and only appears in proportion to the move.

G5 hidden truth, 36.7 m median depth, 6207 px focal, 5178 px wide
(`research/volfill/projection_error.py`):

| arm | total | radial | lateral | @2 deg | @5 deg | @20 deg |
|---|---|---|---|---|---|---|
| **VolFill invented** | 1.025 m | 0.724 m | 0.462 m | **85 px (1.64%)** | 94 px (1.81%) | 130 px (2.51%) |
| baseline relief | 2.530 m | 2.154 m | 1.054 m | 192 px (3.70%) | 211 px (4.07%) | 298 px (5.75%) |

**71% of VolFill's error is radial** — the component projection forgives. Naive
total-error-to-pixels gives 173 px; the projection-correct figure at 2 degrees is
**85 px**. The reframing roughly halves the effective error and puts VolFill at
**1.6–2.5% of frame width across the whole +/-2 to +/-20 degree range**.

The caveat: at small offsets `sin(2 deg) = 0.035` discounts radial error to almost
nothing, so screen error becomes **95% lateral** — and lateral error does not
shrink with the move. There is a ~78 px floor that camera-move restraint cannot
buy back, so shrinking the move stops helping below about 5 degrees.

VolFill stays ~2.3x better than the baseline under either metric, so the ranking
is robust; the projection framing moves both into far more usable territory than
the Chamfer numbers implied.


---

## The isosurface is a DOUBLE WALL — and that settles the adapter design

Reported live from the viewport as "every 2nd triangle flipped". The first
explanation — that an unsigned field gives marching cubes no consistent
orientation — was **wrong**. Measured on the sh001 volume:

```
78.3% of rays cross the 0.5 level exactly TWICE
crossings are EVEN on 98.6% of rays (2, 4, 6, 8, 10)
band thickness: median exactly 2.0 voxels = 12.1 cm at a 6.1 cm voxel
```

An unsigned field has no interior, so its level set at `t > 0` is a CLOSED SHELL
offset `+/-t` either side of the true surface: a front sheet and a back sheet, 2t
apart. Even crossing counts are the signature. Half the triangles face away
because they are the BACK WALL, not because their winding is wrong.

Consequences:

1. **Double-siding was the wrong fix.** It quadrupled an already-doubled mesh.
2. **Every Chamfer from marching cubes carries a ~1 voxel (6.1 cm) placement
   error**, because samples sit half a wall off the true surface. Measured
   afterwards with the ray-march adapter: the effect on RECALL is +0.082 m in
   the double wall's favour, i.e. it mildly FLATTERED the numbers rather than
   penalising them (twice the points within a voxel of the surface makes a
   nearest-neighbour search easier). No conclusion moves either way.
3. **The correct extraction is a ray-march, and it lands on machinery Atlas
   already owns.** Marching the volume along Atlas camera rays yields ordered
   crossings per pixel, which IS the layered-ray representation
   `core/hidden_geometry.py` was written to consume.

### The integration this implies

```
volume -> ray-march along Atlas camera rays  -> ordered crossings per pixel
       -> pair crossings, take midpoints     -> true surface, +/-t bias removed
       -> select_hidden_surface()            -> first layer clearing the occluder
       -> occlusion matte                    -> where hidden geometry is authoritative
       -> union with the relief mesh
       -> AtlasRetopologizeLayer             -> ONE single-sided projection surface
```

Single-sided by construction (keep the camera-facing crossing), no `double_sided`
workaround, no double-wall bias, photographed and inferred surface combined in one
retopologised mesh with regenerated projective UVs. Steps 3 and 6 are already
written, calibrated and tested; only the ray-march and the union are new.

This also revives `AtlasPredictHiddenGeometry` — shelved purely because its two
backends were unlicensed — rather than growing a parallel volume path beside it.

**Recommendation strengthened: build the ray-march adapter, not a second
pipeline.** `AtlasLoadHiddenVolume` stays what it is: a research bridge for
getting a volume in front of an artist's eye.


---

## Acceptance criteria (added after the 2026-08-15 spec panel)

The panel's first finding was that this evaluation ranked arms without ever
declaring what PASSING means, which makes "we might have cracked this"
unfalsifiable. Stated here, in projection terms, because Atlas projects texture
onto approximate geometry rather than reconstructing it:

| # | Criterion | Threshold | Measured (best arm) |
|---|---|---|---|
| A1 | Median screen error at +/-10 deg | **< 2.5% of frame width** | 2.07% (VolFill) vs 4.66% (relief baseline) |
| A2 | Invented fraction | **<= 85%** | 62% (boiler, 8 m band) |
| A3 | In-front fraction at 0 deg offset | **< 5%** | 1.27% (sh001) |
| A4 | Canonical->world round trip | **median <= 1 voxel edge** | 11/12 plates |
| A5 | Input class | **photographic — CAUTION, not enforceable** | see below |

A1 is the headline: it is the only one expressed in what a comp actually sees.
A2 and A3 are cheap proxies computable with no ground truth, and they exist
because A1 needs truth that most plates do not have. A volume passing A2 and A3
is a candidate; only A1 against real truth confirms it.

**A6 (added): the volume must contain something.** At least 2% of camera rays
must hit a predicted surface. An empty volume scores 0% invented and therefore
PASSED the divergence gate looking sound — measured on `golden_corridor`, where
the depth stage collapsed, VolFill emitted zero surface voxels, and A2 read it as
clean. Now a hard refusal on both extraction paths.

**Enforcement.** A2 is now a hard gate in `AtlasLoadHiddenVolume`
(`max_invented_fraction`, `on_divergence`), default **refuse** — a volume that
trips it emits nothing and says why. The previous build printed a warning and
emitted the geometry anyway, which let a diverged volume reach the viewport
indistinguishable from a sound one.

## RAW path corrections (same panel)

The session was titled "geometry recovery from camera RAWs" and the first build
of the review graph bypassed Atlas's RAW importer entirely: plates were decoded
with bare `rawpy`, the EXIF focal was printed and discarded, the solve was told
`sensor_width_mm = 36.0` (full frame — the X-H2 is APS-C **23.5 mm**, wrong by
1.53x) and `focal_length_mm = 0.0` (estimate), and no lens correction ran.

Now routed through `AtlasLoadRAW(undistort=True)`:

| plate | lens profile | measured focal | sensor |
|---|---|---|---|
| sh001 / DSCF3915 | XF16-55mm F2.8 R LM WR on X-H2 | **18.70 mm** | 23.50 mm |
| sh004 / DSCF3931 | XF16-55mm F2.8 R LM WR on X-H2 | **20.60 mm** | 23.50 mm |

Lensfun undistortion **applied** on both, and the volumes were re-derived from
the undistorted decodes so plate, intrinsics and geometry share one space.

### Redistort ST maps — delivery, not an afterthought

Undistorting is a one-way door without the inverse, so
`atlas_camera/raw/redistort.py` builds the Nuke ST map that puts a rectilinear
render back onto the original distorted plate.

The inverse is computed by fixed-point iteration on the lensfun remap grid, not
by scatter — a gap-filled scatter would invent geometry at the frame edge, which
is exactly where distortion is largest.

| plate | inversion residual (solvable px) | outside frame |
|---|---|---|
| sh001 | **0.00064 px** | 3.42% |
| machine | **0.00356 px** | 2.31% |

Convergence is judged over SOLVABLE pixels only. A real correction samples from
an inset region (a 5178-wide plate had `coords` spanning x in [25.9, 3849]), so
distorted pixels beyond that inset have no in-frame solution and can never
converge; scoring them made a correct inversion report a 104 px "residual" that
was entirely corner pixels with no answer. Those pixels are flagged in the ST
map's alpha rather than clamped — clamping is how a comp stretches an edge pixel
across a corner.

Round trip on real data (undistort then redistort, valid pixels): mean 3.90/255.
That residual is double-bilinear resampling loss, not mapping error — in
delivery the render is re-distorted ONCE rather than round-tripped.

Conventions pinned by `tests/test_redistort_stmap.py`: channels (u, v, 0, alpha),
normalized [0, 1], **V flipped for Nuke's bottom-left origin**, written as raw
float EXR with no colour transform — an ST map is DATA, and any transfer curve
corrupts the coordinates.


---

## Band knee, replicated across three scenes (panel follow-up)

The knee was originally measured on ONE plate and generalised as "band to ~2x
subject distance". Swept on two further scenes:

| scene | subject distance | knee | visF at knee | voxel at knee |
|---|---|---|---|---|
| rusty boiler | ~4-5 m (photographer's recall) | **8 m** | 0.42 | 3.4 cm |
| sh001 street | ~10-20 m | **6 m** | 0.65 | 2.0 cm |
| ghost town | mixed | **40 m** | 0.35 | 12.9 cm |

**The 2x rule is refuted.** It holds for the boiler and fails on both other
scenes — the street's knee is BELOW its subject distance, the ghost town's is far
beyond. The inverted-U shape replicates; the multiplier does not. Sweep per
scene; do not assume.

**Caveat on comparing visF across bands.** visF scores the prediction against the
visible TUDF *within the same canonical bbox*, so a tighter band evaluates a
smaller region with a different denominator. Scores are comparable WITHIN a band
sweep as a ranking signal, but sh001's 0.65 at 6 m is not a like-for-like
improvement on its 0.41 at 25 m — it is agreement measured over less scene. The
band sweep finds a knee; it does not prove the tight band is globally better.

## Divergence gate, validated on held-out data (panel follow-up)

The gate was fitted on the same 15 volumes it was claimed to generalise from. It
has now been checked against **26 volumes produced later and never used to derive
it** (the boiler set, every band arm, the ROIxband arms):

| gate shape | decisive volumes | agreement |
|---|---|---|
| single threshold at 85% | 26/26 | **88.5%** (TP 13, TN 10, FP 2, FN 1) |
| **pass <82% / inspect 82-88% / refuse >88%** | 20/26 | **100%** |
| pass <80% / inspect / refuse >90% | 18/26 | 100% |

The single threshold's three errors ALL sat between 82% and 88%. So the honest
shape is three states, not two: the rule is sharp away from the boundary and
genuinely ambiguous at it, and pretending otherwise forces a wrong call on ~12%
of volumes.

Implemented in `AtlasLoadHiddenVolume` as `max_invented_fraction` (refuse, 0.88)
plus `inspect_invented_fraction` (0.82). A volume in the band is emitted and
TAGGED `needs_inspection` — neither silently passed nor silently dropped.

**Corrects an earlier overclaim in this report:** ">85% occluded <=> visF <0.2,
without exception" was true only on the derivation set.


---

## The ray-march adapter, built — and the bias correction it enabled

`atlas_camera/core/volume_raymarch.py` (+ 9 tests). Marches the distance field
along the RECOVERED camera's rays and pairs consecutive crossings into their
MIDPOINT, which is where the surface physically is. Output is `(H, W, L)`
front-to-back forward depth — Atlas's existing layered-ray representation — so
`core.hidden_geometry.select_hidden_surface` consumes it with no translation.
A test pins that end-to-end.

Marching the ray rather than the shell fixes three things at once: one sample per
real surface instead of two walls, consistent orientation (single-sided by
construction, so no `double_sided` workaround), and correct placement instead of
a `+/-threshold` offset.

**Independent confirmation of the double wall.** On the real sh001 volume the
marcher reports `odd_crossing_fraction = 0.0%` — every ray that entered a shell
also left it. That is the same conclusion the crossing histogram reached, via a
different mechanism.

### The bias correction — and a correction to this report

Re-scoring G5 with paired midpoints against the same photographed answer key:

| candidate | median | mean | p90 |
|---|---|---|---|
| baseline relief mesh | 2.536 m | 2.482 m | 3.587 m |
| marching cubes (all earlier numbers) | **0.932 m** | 1.253 m | 2.639 m |
| ray-march, crossings paired | **1.014 m** | 1.310 m | 2.634 m |

**This report previously claimed the double wall made every Chamfer pessimistic.
That was wrong in sign.** For a RECALL metric (truth -> nearest candidate),
emitting both walls scatters twice as many points within +/-1 voxel of the
surface, which makes finding a near neighbour EASIER. The double wall was mildly
flattering the recall figures, not penalising them.

The difference is 0.082 m = 1.35 voxels, inside the measurement's own resolution,
and both arms sit at the 1.055 m noise floor. **No conclusion changes**: VolFill
still recovers photographed hidden structure ~2.5x better than the current Atlas
baseline.

So the adapter's value is not a better score. It is that the geometry is placed
correctly, is single-sided without a workaround, and arrives in the form the
calibrated consumer already takes. `AtlasLoadHiddenVolume` still meshes with
marching cubes for the viewport; wiring the marcher through
`select_hidden_surface` and `build_relief_mesh` (one relief surface per layer,
which would drop marching cubes entirely) is the next step and is NOT built.


---

## The intrinsics bypass, fully closed

The spec panel found the workflow bypassing Atlas's RAW importer. Fixing that
properly turned up two further defects, one of them a real Atlas bug.

### 1. A portrait-frame sensor bug in `resolve_sensor_size` (Atlas, not research)

Every consumer computes `fx = focal_mm / sensor_width_mm * image_width_px`, so
"sensor width" must mean the sensor extent along the IMAGE's width. The
`camera_db` tier returned the body's PHYSICAL dimensions — an X-H2 is
23.5 x 15.6 mm however it is held — so a portrait plate got the long edge as its
width:

| sh001 (portrait 5178 x 7752) | sensor width used | resulting fx |
|---|---|---|
| unoriented (the bug) | 23.5 mm | **4120 px — 34% out** |
| oriented | **15.6 mm** | **6207 px** |
| the metric rig solve, from a surveyed 14.6 m baseline | — | **6207 px** |

The oriented value reproduces the measured rig EXACTLY from EXIF alone. The other
sensor tiers derive from `image_width_px` already and were never affected; only
the registry tier carried the body's own orientation. Now transposed to match the
frame, with a warning rather than silently — `tests/test_sensor_orientation.py`
(5 tests) pins it, including the 6207-vs-4120 case.

This also settles an earlier open question in this report. MoGe's free-fov
estimate of 5278 px was compared against the solve's 6207 px and called a "15%
disagreement, one of them is wrong". The oriented sensor confirms 6207 is
correct, so **MoGe was the one in error**.

### 2. The volumes were built on MoGe's guess of the camera

`run_volfill.py` called MoGe with no `fov_x`, so it predicted its own camera and
the canonical bbox, the metric scale and every extracted surface inherited that
guess. `--fov-x-deg` / `--raw` now pass the measured FOV through.

| plate | intrinsics | voxel | visF | invented |
|---|---|---|---|---|
| sh001 | MoGe estimate | 6.07 cm | 0.41 | 71.8% |
| sh001 | **measured (45.28 deg)** | **5.68 cm** | **0.43** | **67.0%** |
| machine | MoGe estimate | 3.37 cm | 0.36 | 68.7% |
| machine | **measured (59.40 deg)** | **3.03 cm** | **0.37** | **64.9%** |

Consistent on both plates and on all three measures — finer voxels (the bbox
fits correctly scaled geometry), better agreement with the visible surface, less
invented fraction. The gain is **modest, not transformative**: MoGe's guess was
wrong but not catastrophic here. It is free, and it is the correct thing to do
when the camera is sitting in the file's EXIF.

`metadata.json` now records `fov_x_deg` and `intrinsics_source`
(`measured_raw` / `measured_arg` / `moge_estimate`) so a volume states whether
its camera was known or guessed.


---

## A5 could not be enforced — and the attempt refuted the claim behind it

A5 said "photographic input only", justified by three synthetic failures. Trying
to enforce it produced two negative results worth more than the gate would have
been.

### The failure is invisible to every no-truth signal

Scoring the synthetic volumes on the same self-consistency measures used
everywhere else in this report:

| volume | invented % | visF | A2 verdict |
|---|---|---|---|
| SYNTHETIC, untextured Blender | 73.5 | 0.38 | pass |
| SYNTHETIC, textured Blender | 75.8 | 0.37 | pass |
| PHOTO sh001 | 67.0 | 0.43 | pass |
| PHOTO boiler | 64.9 | 0.37 | pass |

**The synthetic plates are indistinguishable from the photographs.** Both pass
A2, both pass A4 (round-trip at 1.14–1.17x the quantisation floor).

**This corrects an earlier claim in this report.** The A5 row previously read
"synthetic/CG fails A2 and A4". It does not. The distinction that was conflated:
measured against its OWN visible surface a synthetic volume looks fine, and the
failure only appears against EXTERNAL ground truth (Blender: scale 54% off,
61–95% of prediction unsupported). Every gate available at run time is of the
first kind.

### The obvious detector does not work either

MoGe's own predicted focal, compared against the true camera:

| volume | true hFOV | MoGe hFOV | disagreement |
|---|---|---|---|
| PHOTO sh001 (free-running) | 45.3 deg | 52.6 deg | **16.3%** |
| SYNTHETIC textured (free-running) | 54.4 deg | 50.9 deg | **6.4%** |
| SYNTHETIC untextured (free-running) | 54.4 deg | 63.1 deg | 15.9% |

The textured synthetic render is MORE accurate than the photograph. A detector
built on this signal would be noise.

### Verdict

A5 is demoted from an enforceable criterion to a **documented caution**: the
evidence that CG degrades the pipeline is real (three cases, including one that
produced nothing at all), but no run-time signal separates it, so a gate would be
theatre. Enforcing it would need either provenance the plate does not carry, or a
CG classifier — and three samples cannot support one.

What the attempt DID find was a genuine hole, now closed: **an empty volume
passed the divergence gate.** `golden_corridor` scored 0% invented, which read as
perfect agreement, because there was nothing to disagree with. Both extraction
paths now refuse on a minimum-coverage floor (A6).


---

## Delivered: the full path, wired and validated in the viewport

The adapter is no longer inert. `AtlasLoadHiddenVolume` (experimental tier) now
runs end to end:

```
volume -> ray-march -> layered rays
       -> select_hidden_surface   (per-pixel first-clearing-layer, calibrated)
       -> fill_hidden_gaps        (bounded by restrict_mask)
       -> occlusion matte         (MASK output)
       -> ONE merged relief surface, photographed + inferred together
```

`AtlasExportNuke` gained `raw_meta` and writes the redistort ST map as a
first-class manifest artifact.

### Confirmed by eye, which is what the whole evaluation was for

The review workflow's control lane did its job: `coastal_alley` (visF 0.01,
99.6% invented) rendered as shredded debris, while the boiler and sh001 rendered
as coherent geometry — a closed cylindrical vessel with both flue-tube bores and
the rear vessel's end plate, and a building with legible "36 MARY PARADE"
signage, roller door, kerb and road markings. **The gate separates usable from
unusable, not degrees of bad.**

Note the metrics UNDERSTATED this. visF 0.43 reads as mediocre and ray-wise
agreement is ~4 voxels, yet the viewport shows projectable architecture. For a
surface that texture is thrown onto, "approximately right and continuous" beats
"metrically precise and fragmented" — the reconstruction-style metrics were
penalising the wrong axis, exactly as the projection reframing predicted.

### How much hidden geometry exists is a property of the PLATE

| | sh001 street (flat facade) | rusty boiler (self-occluding) |
|---|---|---|
| layers per hit | 1.12 | 1.61 |
| pixels clearing the occluder | 0.94% | 8.04% |
| band mask overlap | 0% un-inverted, 100% inverted | 100% un-inverted |
| substituted | 65.6% (inverted) | 60.7% |
| occlusion matte | — | 43.5% |

A flat facade has ~one surface per ray and little to recover; a self-occluding
object has real depth complexity. The combined path reports "nothing to recover
here" rather than manufacturing something, which is the correct behaviour.

**The band-mask orientation is per-plate**, found by the artist before the tool
admitted it: the same foreground band caught 100% of the boiler's cleared
selection and 0% of sh001's. Now a widget (`invert_restrict_mask`), and the
report prints how much of the cleared selection the mask actually catches so the
failure names its own fix. No "auto" mode was added deliberately — it would hide
the signal that found the problem.

### Verdict

**VolFill: adopt at experimental tier for photographic plates**, behind the
divergence gate, with a per-scene depth band. It recovers real occluded structure
(2.5x better than the relief baseline against surveyed photographic truth, at the
rig's noise floor), runs at ~6 s and 5.3 GiB, and is MIT code with MIT weights —
which is what makes it shippable where LaRI and World-Tracing were not.


## Technical matrix

| Criterion | Flash3D | VolFill |
|---|---|---|
| Hidden geometry type | layered Gaussians behind the visible layer | complete amodal scene volume |
| Representation | 3D Gaussians | 256³ unsigned TUDF |
| Feed-forward | yes | no |
| Iterative sampling | none | flow matching, 50 steps default |
| Camera controllability | UniDepth-internal | **external — bbox is data, not learned** |
| Metric scale | relative | **metric (MoGe-v2)** |
| Explicit mesh possible | via conversion | **yes — marching cubes** |
| Hidden geometry quality | not measured (rejected) | **1.026 m vs 2.536 m baseline at a 1.055 m noise floor** (photo truth) |
| Visible geometry preservation | not measured | visF 0.35–0.41 good plates, <0.2 on ~1/3 |
| Runtime | will not build on sm_120 | **~6 s/plate warm** (2.6 s sampling) |
| Peak VRAM | — | **5.3 GiB**, flat |
| Modern CUDA compatibility | **no** (torch 2.2.2 / CUDA 11.8) | **yes** (torch 2.10 / CUDA 13.0) |
| Atlas camera integration | — | **closed-form, tested** |
| Atlas World evaluation | — | G5 rig + visible-TUDF split |
| Code licence | **none** | MIT |
| Weight licence | unstated | **MIT** |
| Dependency risk | NC rasterizer + NC depth model | non-PyPI wheel index |
| Commercial suitability | **unusable** | **usable** |
| Recommended role | none | gated hypothesis generator for occluded surface, behind the >85% divergence check |

---

## Conclusions

### Flash3D

1. **Genuine hidden layers?** Not assessed — rejected on licensing before merit.
2. **Extractable without rendering?** Gaussian means are points, so in principle
   yes; moot.
3. **Alignable to an Atlas camera?** Unknown; UniDepth predicts its own intrinsics.
4. **Blender-usable geometry?** Would require Gaussian→surface conversion.
5. **Speed?** Not measured; cannot build for sm_120.
6. **Licensing safe?** **No.** Three independent blockers.
7. **Better than extending Atlas depth/projection?** Unanswerable and irrelevant
   given (6).

### VolFill

1. **Does the TUDF recover meaningful occluded structure?** **Yes, on
   photographs — measured.** Against the sh001 two-pose rig its invented geometry
   explains really-photographed hidden structure at 1.026 m median (rig noise
   floor 1.055 m) versus 2.536 m for the relief-mesh baseline, and invented
   geometry inside frame 2's view sits 0.678 m from real structure. Caveats:
   ~a third of plates diverge (detectable via the >85% rule), and on synthetic/CG
   input the MoGe stage collapses.
2. **Transformable into Atlas world space?** **Yes — closed-form, no rotation to
   recover, verified by 11 tests.** This was the biggest anticipated risk and it
   is fully retired.
3. **Is 256³ enough at Atlas scales?** Scene-dependent — measured 4.8 cm
   (portal) to 244 cm (oceancastle). A **25 m depth band takes the worst street
   plate from 68 cm to 8.65 cm (7.9×) at only 0.36→0.28 fidelity cost**. An ROI
   crop does NOT help (isotropic cube tracks the longest axis) and destroys
   fidelity (→0.01). Band yes, crop no.
4. **How fast before quality collapses?** Wrong question here: 50→4 steps saves
   under 2 s because MoGe plus the dense 256³ decode dominate. Always run 50.
5. **Beats VACE on geometry correctness?** On the same G5 fixture, VACE's fill
   scored 44.2/255 (13.8 dB) against the photograph — a *pixel* metric, because
   VACE produces pixels. VolFill is measured where VACE cannot be: it puts
   invented **geometry** within the rig's noise floor of really-photographed
   hidden structure, at **4× the speed**. Different quantities, but the
   structural claim of the brief holds — geometry first, and faster.
6. **Beats LaRI conceptually/practically?** **Conceptually and legally, yes** — a
   full amodal volume rather than per-ray layers, under a licence that permits
   shipping where LaRI's absence of one does not. Practically undecided: LaRI is
   ~0.2 s against VolFill's ~6 s, and no head-to-head on the same plates was run
   (LaRI needs a user clone, and its output is layered rays rather than a volume,
   so a fair comparison needs the ray-march adapter first).
7. **Blender-useful meshes?** Yes in principle — marching cubes on an unsigned
   field at a small positive level; needs cleanup and is not watertight.
8. **Licences commercially acceptable?** **Yes**, with the three residual risks
   above (missing vendored notices, non-PyPI wheels, unstated training data).

---

### Where VolFill is the WRONG tool — parametric objects (noted, not built)

The boiler set makes a case against itself. Both vessels show their end plates,
and a circle projects to an ellipse: fit the ellipse, eigendecompose the cone
matrix `K^T C K` with known intrinsics, and the circle's 3D pose — centre, radius,
axis — falls out in closed form. Cylinder length comes from where the silhouette
terminates. The amodal problem collapses to a parametric fit with no neural
prediction at all, and the concentric flue-tube ellipses over-constrain it so pose
can be solved and validated in one step. (The cone decomposition is two-fold
ambiguous; gravity from the solve, ground contact, or the second ellipse resolves
it.)

That result is exact, watertight, cleanly topologised, arbitrary-resolution and
artist-editable — and every weakness measured on this set (82–95% occluded
fractions, the band knee, per-shot tuning) simply does not arise.

It does not generalise: it needs the object to BE a primitive with a visible
defining feature. The sh001 street, the foliage, the ghost town and the cathedral
do not reduce to fitted cylinders, and that non-parametric structure behind
occluders is exactly VolFill's niche. So this does not displace VolFill; it marks
out the case where VolFill should not be reached for.

Atlas already holds the pieces — `AtlasProxyPrimitive` supports
`primitive_type="cylinder"`, `AtlasBlockoutMassing` sits in the experimental tier,
the derive-node pattern exists for walls/roofs/towers, and the Blender MCP bridge
is connected. Building and scoring it against the same truth is the brief's
"Atlas + Blender procedural completion" arm. **Deferred by decision 2026-08-15 —
recorded so the routing rule is not lost, not scheduled.**


## Architectural recommendation — *provisional, pending runtime*

Leaning **Option B (integrate a VolFill adapter)**, for a reason that only became
visible by reading the Atlas tree rather than the papers: Atlas already built,
calibrated and shipped a hidden-geometry pipeline, then **withdrew it purely
because both backends were unlicensed or non-commercial**. The representation,
the registration, the layer selection, the coherence pass and the downstream
consumers all still exist. VolFill is MIT code with MIT weights conditioned on a
depth model Atlas already prefers.

Option D (Atlas-native predictor) stays the long game and the findings feed it —
but it needs training data and a training run, and there is a licensed backend
available now.

The provisional shape is *not* "port VolFill into the pipeline". The TUDF is a
volume; `core/hidden_geometry.py` consumes layered rays. The natural adapter
converts a predicted volume into that existing contract by **ray-marching the TUDF
along Atlas camera rays** and recording ordered surface crossings per pixel — which
lands exactly in `select_hidden_surface`'s input format and reuses every
calibrated behaviour downstream. Atlas keeps owning the camera and the semantics;
VolFill supplies only occupancy.

Blocking conditions before recommending integration: sm_120 must actually run;
occluded-region quality must beat the relief+ribbon baseline; and the exterior
resolution gate must be resolved or the role explicitly scoped to interiors.

---

## Checkpoint handling

Never committed. `research/volfill/.gitignore` excludes `.venv/`, `repos/`, `out/`,
`checkpoints/`, `*.npz`, `*.pth`.

| Model | Source | Licence | Cache |
|---|---|---|---|
| `volfill_dit.pth` | HF `TuanNgo/VolFill` | MIT | HF hub cache |
| `volfill_vae.pth` | HF `TuanNgo/VolFill` | MIT | HF hub cache |
| `moge-2-vitl`, `-normal` | HF `Ruicheng/…` | MIT | HF hub cache |

Auto-downloaded on first run. If promoted, they belong in the ComfyUI/Atlas
external model directories, not in the package.

---

## Reproduction

```powershell
# Mapping tests — no GPU, no weights, no VolFill install
cd research/volfill
python -m pytest tests -q --basetemp=<writable-temp>

# Provenance audit
python audit_provenance.py

# Inference (isolated venv only)
.venv/Scripts/python.exe run_volfill.py --image <plate> --out out/<name> --steps 50

# Round-trip + visible/occluded split (Atlas env)
python roundtrip_eval.py out/<name> --out out/<name>_score.json
```
