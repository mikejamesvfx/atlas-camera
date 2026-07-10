# Fixer fine-tune data plan — Atlas-generated training pairs

Status: **designed, not built** (2026-07-10). The packaging stage exists
(`tools/generate_fixer_training_pairs.py`); this doc specifies where the
pairs come from and what has to be built to mass-produce them.

## Why fine-tune at all

The pretrained Fixer checkpoint already earns its keep (spike 2026-07-10,
evidence artifact 🔧): strong on shredded-mesh/hard-tear artifacts, modest on
the DMP rig's softened smears, mild global softening everywhere. All three
observations have the same root: Fixer was trained on NeRF/3DGS artifact
distributions, not Atlas's. A fine-tune on Atlas's *own* artifacts (projected
relief meshes: comb-tooth skirt edges, band seams, matte-cut boundaries,
stretched-texel fans) targets exactly the residual gap — and Fixer ships its
full trainer (`src/train_pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py`,
`--pretrained_path` to start from the shipped checkpoint, best-practice
hyperparameters in the README: lr 2e-5, timestep 250, 576×1024, LPIPS 0.3).

## The pair recipe (Difix3D+'s own strategy, transposed to Atlas)

A training pair is {degraded render, real photo} at the SAME camera. A single
photo has no ground truth off its recovered view — so pairs come from
**multi-view sources**, using Atlas itself as the degrader:

1. Take a posed multi-view sequence (RealEstate10K-style: video frames +
   per-frame cameras; or any video clip run through a SfM/pose stage).
2. Pick frame A. Run the real Atlas pipeline on it alone: learned solve →
   relief mesh / DMP rig derivation.
3. Express camera B's pose relative to A's recovered camera (convert the
   dataset's world-to-camera convention into an `orbit_camera`-reachable
   delta, or a raw extrinsic for the projection).
4. Render A's projection from B's pose → the **degraded** input, carrying
   genuine Atlas artifacts (tears, skirts, smears, frame-edge reveals).
5. Ground truth = the dataset's real frame at B.
6. Package with `tools/generate_fixer_training_pairs.py` (letterboxing both
   sides identically to 576×1024).

Every step except 4 exists today. Step 4 — rendering the projected scene from
an arbitrary pose — currently lives ONLY in the browser viewport
(Three.js `PROJECTION_FRAGMENT_SHADER`); there is no headless renderer.

## The one real build item: headless projection rendering

Options, in preference order:

- **A. Browser-farm the bakes (no new code).** Drive ComfyUI + the viewport
  via the existing bake path (camera-path keyframes at dataset poses →
  `path_frames`). Zero renderer risk, exact shader parity with production —
  but throughput is a browser tab, and pose-exact keyframing from dataset
  extrinsics needs a small JS/client_data injection hook. Right choice for a
  first ~1k-pair pilot.
- **B. Numpy/moderngl software rasterizer of the relief mesh** with the
  projection shader's rules ported (frustum/behind-camera discard, facing
  threshold, matte sampling). Fast and headless at scale, but a SECOND
  implementation of the projection semantics — the exact "two copies drift"
  hazard CLAUDE.md warns about, so it must be conformance-tested against
  browser bakes on golden frames before any mass generation.
- **C. Nuke-render the exported .nk layers** (the exporters already build the
  full projection graph): headless via `nuke -x`, professional-grade
  sampling — but requires a Nuke license on the render machine and is the
  slowest per frame.

## Volume & curation targets

- Difix trained at ~80k pairs; a LoRA-style/full fine-tune from the shipped
  checkpoint plausibly moves at 5–20k pairs (their README encourages starting
  from defaults; `--max_train_steps 10000`, batch 1×8 GPU in their recipe —
  budget accordingly for the single RTX 5090: gradient accumulation, days not
  hours).
- Curate by artifact class, matched to Atlas reality: bare relief tears,
  DMP-rig band seams + skirt smears, sky-dome edges, frame-edge reveals
  (label these — they may want exclusion or a dedicated outpaint head later).
- Keep per-sequence near-duplicates out of the test split (the packaging tool
  already interleaves the split for this reason).

## Success criteria (pre-registered, so the fine-tune can fail honestly)

Re-run the spike harness (same two baked orbits, same metrics) with the
fine-tuned checkpoint vs. pretrained:

1. Hard-variant hole-fill fraction ≥ pretrained (no regression), with
   filled-texture softening visibly reduced in crops.
2. Softened-variant smear repair visibly better at equal timestep.
3. Temporal ratio stays ≤ 1.0.
4. A held-out multi-view test set shows LPIPS/FID improvement over
   pretrained on REAL ground truth (the metric the spike could never have).
