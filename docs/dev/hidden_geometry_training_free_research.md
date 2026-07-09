# Training-free hidden geometry for the depth-shadow problem — research notes

*2026-07-09. Companion to `docs/ATLASresearch-report.md` (the monocular-occlusion
grounding document). Written for the "predict hidden geometry behind camera
shadow rays, CUDA-only is fine" idea.*

## Where Atlas stands against the grounding document

The report's core discipline is already Atlas policy, mostly by convergent
evolution:

| Report's principle | Atlas today |
|---|---|
| n·v is an orientation cue, not a hidden-surface solver | `uFacingThreshold` facing-ratio discard in the projection shader; `primary_camera_validity_mask`'s facing test |
| Visibility needs the z-buffer test, not just front-facing | `occlusion_mode="depth_shadow"` (the primary's depth map as its own shadow map) |
| Hidden geometry = hypothesis with confidence, never fact | tiered scale cascade, `scale_source` provenance, artist-confirm principle, `hole_mask` honesty |
| Separate measured / derived / hypothesized | solve (measured/derived) vs `fill_occluded`'s `filled_mask` + `extend_mask_b64` ("invented pixels" mattes shipped to the DCC) |
| Enclosed occlusions are not solvable from boundary normals | `fill_occluded`'s Jacobi diffusion only converges from in-band neighbors; unreachable cells stay holes |

Where the report and Atlas genuinely diverge: the report is **CPU/NumPy-first**
and therefore stops at plane-hypothesis completion. Atlas's real deployment is
a CUDA ComfyUI venv running GeoCalib + Depth Anything 3 — the "optional learned
prior plug-in" stage of the report's own roadmap is already our baseline. That
changes which methods are on the table.

## What "training-free" buys us in 2026 (zero-shot occluded-geometry predictors)

The dream — predict the surfaces *behind* the first ray hit, per pixel, no
training on our side — is now a real model class:

### 1. LaRI — Layered Ray Intersections (ICML 2026) ★ primary candidate
- **What it does:** single feed-forward pass predicts *layered point maps* —
  for each pixel, the ordered stack of surfaces its camera ray intersects
  (layer 1 = visible, layers 2..k = occluded), plus a **ray stopping index**
  saying how many layers are valid per pixel. This is literally "geometry
  hidden by camera shadow rays."
- **Practicals:** code + HuggingFace checkpoints public (`lari_obj_16k_pointmap`,
  `lari_scene_pointmap`); MoGe-based backbone; single-image input; Gradio +
  CLI demos; needs PyTorch3D (CUDA — fits the constraint). Regression model —
  deterministic, fast, no diffusion sampling.
- **Risks:** scene-level checkpoint trained on SCRREAM (indoor, small scenes) —
  generalization to 4K outdoor AI-generated plates is THE open question (same
  domain-gap question we just resolved favorably for DA3, but this is a much
  smaller training corpus). Processing resolution will be ~MoGe-scale
  (hundreds of px), so hidden geometry comes back low-frequency. License not
  stated in the README — **check before shipping**.
- Sources: [arXiv 2504.18424](https://arxiv.org/abs/2504.18424) ·
  [project page](https://ruili3.github.io/lari/) ·
  [GitHub](https://github.com/ruili3/lari)

### 2. World Tracing — WT-DiT (June 2026) ★ the quality ceiling
- **What it does:** same pixel-aligned layered representation (ordered stack of
  camera-space 3D points per pixel, front-to-back), but *generative* — a
  diffusion transformer with factorized layer attention, flow-matching in
  pixel space. Claims better occluded-surface modeling and **planar-structure
  preservation on out-of-distribution inputs** (exactly our failure surface),
  and explicitly advertises "training-free integration" with downstream mesh
  generators.
- **Risks:** brand-new (arXiv 2606.13652); diffusion = slower + stochastic;
  code availability unconfirmed. Watch it; don't build on it yet.
- Sources: [arXiv 2606.13652](https://arxiv.org/abs/2606.13652) ·
  [project page](https://haoz19.github.io/world-tracing-page/)

### 3. Adjacent options, noted and deprioritized
- **Amodal Depth Anything** (ICCV 2025): relative depth of occluded *object*
  parts, driven by amodal segmentation masks (pix2gestalt-style). Object-centric
  and needs per-occlusion masks — a per-object tool, not a per-ray scene
  answer. [arXiv 2412.02336](https://arxiv.org/abs/2412.02336) ·
  [project page](https://zhyever.github.io/Amodal-Depth-Anything/)
- **RaySt3R** (2025): zero-shot object completion by predicting novel depth
  maps from virtual query views — interesting mental model (query the scene
  from where the *hole* is visible) but object-level.
  [arXiv 2506.05285](https://arxiv.org/html/2506.05285)
- **The report's own plane-hypothesis generator** (boundary-ring plane fits +
  Manhattan snapping + scoring): still worth having as the zero-dependency
  fallback tier — pure numpy, honest, and composable with everything above.
  It is the "assumed" tier of hidden geometry, exactly like the scale cascade's
  assumed default.

## How layered ray intersections would slot into Atlas

The elegance the dream sensed is real, and it comes from **pixel alignment**:
LaRI/WT layer 1 is a normal visible-surface point map, dense on exactly the
pixels where we already have trusted DA3 metric depth. That solves the
registration problem that killed patch-derived geometry (`own_depth` — no
scalar aligns a hallucinated view's depth to the primary's world):

1. **Register once, on the visible layer.** Fit scale/shift (or per-pixel
   robust scale, median over valid pixels — the same closed-form trick
   `own_depth` already implements) between LaRI layer-1 depth and our DA3
   metric depth. Layers 2..k ride the *same* transform into the primary's
   metric world **by construction** — they share the camera rays.
2. **Consume layer 2 exactly where `hole_mask`/depth-shadow says we're blind.**
   The disocclusion hole behind a foreground occluder is, per ray, "the next
   surface the ray would have hit" — which is literally layer 2. Feed it as
   the depth field for the background band's mesh inside the occluder
   footprint, replacing `fill_occluded`'s Jacobi diffusion (smooth guess) with
   a *predicted* surface. The LaMa/inpaint clean plate still supplies color —
   this upgrades only the geometry under it, so it composes with the existing
   `AtlasCleanPlateLayer` pipeline rather than replacing it.
3. **Keep the report's honesty rules.** Predicted-hidden depth is tier-"learned
   hypothesis": mark it in `filled_mask`-style provenance, keep it out of
   metric measurement, score it (grazing |n·v| penalty at the source
   silhouette, agreement with the Jacobi solution and with boundary plane
   fits, ray-stopping-index confidence), and let the artist see/veto — the
   same confirm principle as VLM scale cues.
4. **Enclosed occlusions stay flagged.** A learned layer-2 there is pure prior
   (the report's Princeton counterexample stands); confidence should reflect
   distance-to-boundary like the report suggests.

Notable non-goal: this does NOT need the patch/Qwen track — no novel view, no
angle calibration, no generative image. It attacks the *geometry* half of
disocclusion only, which is currently the weaker half (we can inpaint pixels
well; we invent their depth crudely).

## Recommended next step (a spike, not a feature)

1. Clone LaRI into the ComfyUI venv (CUDA + PyTorch3D — check the LICENSE
   file first), scene checkpoint.
2. Run it on the 4 hero 4K test images + the hangar; visualize layer-2 points
   inside each image's measured `hole_mask`/depth-shadow regions.
3. Score against the existing machinery: does layer-2, registered via layer-1,
   land *behind* the occluder and *near* the Jacobi-diffused surface where the
   diffusion is trustworthy (wide-open boundaries), and does it do something
   *better* where diffusion fails (deep/structured disocclusions)?
4. Decision gate: only if layer-2 beats diffusion on real disocclusion reveals
   (orbit test in the viewport) does it earn a node
   (`AtlasPredictHiddenGeometry` → feeds `fill_mask`-style depth into
   `build_relief_mesh`, provenance-marked).

World Tracing is the watch-list successor: same representation, likely better
OOD behavior, not yet practical. If the LaRI spike proves the *plumbing*, the
model behind it is swappable — the integration contract is just "layered
point map + ray stopping index," which both share.

## SPIKE RESULTS (2026-07-09, ran on this machine)

Setup: LaRI scene checkpoint (`lari_scene_pointmap.pth`, 5 layers), ComfyUI
venv (torch 2.9.1+cu130 — **no PyTorch3D needed for inference**, it's only in
their dataset/metrics code; `tools.py` needs rembg so the spike replicated its
helpers instead). Registered layer-1 to DA3METRIC metric depth (solved focal)
by robust median scale. Scripts + visual strips + JSON in the session
scratchpad (`lari_spike.py`, `lari_spike2.py`, `lari_spike_out/`).

Findings, in decreasing order of importance:

1. **Layer-2 is the occluder's own BACK FACE, not the background.** LaRI
   enumerates entry/exit ray intersections: for a solid object the order is
   front → back → surface behind. The disocclusion surface Atlas wants is
   "first layer that clears the occluder," usually layers 3–5. Any
   integration must do per-pixel layer selection, not take a fixed layer.
2. **Sign gotcha:** the public demo negates `pts3d`, but the SCENE
   checkpoint's raw z is already positive-forward — copying the demo verbatim
   gives 100% invalid depths.
3. **Registration works.** Layer-1 vs DA3 log-depth correlation 0.80–0.98
   across all 6 test images; median-scale registration is stable on the
   architectural scenes (rel-MAD ~0.12), noisier on canyon landscapes
   (~0.6, driven by sky/far regions).
4. **Domain gap is decisive.** Cathedral nave (indoor architecture ≈ LaRI's
   training domain): hidden-surface composite fires on **76% of foreground
   pixels** with plausible continuation depth (occluder 33m → hidden 50m →
   global bg median 68m; the visual strip shows the nave columns cleanly
   deleted and filled with aisle/wall depth — genuinely the dreamed
   behavior). Outdoor monument-valley-style scenes: partial (54% coverage,
   shallow-biased) down to **total collapse** (all 5 layers span 4.6→5.0m,
   composite fires on 0.8% of pixels). The sci-fi hangar sits between —
   real layering (49% of fg pixels have layer-4 behind the split) but the
   spike's fixed 20%-clearance margin was too strict for its shallow
   fg/bg separation; a scene-adaptive margin is needed.
5. **Fast:** single forward pass, ~0.2s at 512px on this GPU. Whole-image
   layered geometry for the cost of one depth pass.

**Verdict vs the decision gate:** does NOT clear "beats Jacobi diffusion
everywhere" — outdoor terrain can collapse, and diffusion never does. DOES
clear "useful hypothesis generator on indoor/architectural scenes," which is
where Atlas's room workflows live and where diffusion is weakest (structured
interiors behind columns/furniture are exactly what smooth diffusion gets
wrong). Recommendation: an **experimental, research-only node** —
per-pixel "first clearing layer" depth + a confidence output (registration
rel-MAD, layer separation, coverage) feeding `build_relief_mesh`'s existing
`fill_mask` path, honestly provenance-marked, artist-vetoable.

## BUILT (2026-07-09): `AtlasPredictHiddenGeometry` 🔬 (research-only)

The recommendation above was implemented the same day (user-approved, aware of
the license situation): `core/hidden_geometry.py` (pure-numpy registration +
per-pixel first-clearing-layer selection with the scene-adaptive margin),
`inference/lari_hidden_geometry.py` (guarded LaRI import — user-cloned repo
via `lari_path`/`ATLAS_LARI_PATH`, GeoCalib pattern, nothing vendored), and
the node (category `Atlas Camera/Experimental`): input `ATLAS_DEPTH_MAP` +
IMAGE → patched "X-ray" `ATLAS_DEPTH_MAP` (occluders replaced by predicted
hidden depth) + `hidden_mask` provenance MASK + confidence report. Verified
live against the real clone + DA3 depth: cathedral registers at rel-MAD 0.118
with the nave columns cleanly deleted from the patched depth; monument valley
self-reports "poor" registration as designed. **Known footgun, guarded in the
report:** unrestricted, LaRI also predicts through-wall structure at VISIBLE
background pixels — for band workflows always wire the foreground band's
`layer_mask` into `restrict_mask` so only real occluders are substituted.
Tests: `tests/test_hidden_geometry.py` (7, all mocked/pure-numpy).

**Licensing blocker for shipping:** the LaRI repo has **NO license file** —
default all-rights-reserved, stricter than CC BY-NC. We cannot vendor or
redistribute code/weights. The only shippable shape is the GeoCalib pattern:
the node soft-fails unless the user has installed LaRI themselves, with a
clear "research-only, unlicensed upstream" warning in the tooltip/README.
Worth opening an issue asking the authors for a license before building
anything beyond the experiment.
