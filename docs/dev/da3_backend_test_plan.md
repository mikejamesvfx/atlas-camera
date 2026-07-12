# DA3 backend test plan (V2 vs Depth Anything 3 A/B)

Goal: decide, with numbers and eyes, whether `depth-anything/DA3METRIC-LARGE`
should replace `Depth-Anything-V2-Metric-Outdoor-Large-hf` as a default.

**DECISION 2026-07-09: defaults flipped to DA3METRIC-LARGE** (all node combo
defaults + method kwargs + the indoor/outdoor `scene_type` presets, which now
both point at the one DA3 metric model). Basis: the Part-1 automated A/B below
(3/4 images clearly better on AI-generated 4K test images — the domain-gap
concern didn't materialize) plus artist visual confirmation in the side-by-side
comparison workflow. Core-library defaults (`estimate_depth`'s signature,
`solve_still_image_learned`'s fallback) deliberately stay V2 so `[neural]`-only
installs keep working without `[neural-da3]`. **Still open:** the band-layer
recalibration — the 384/1.5 tuning was calibrated against DA-V2 noise; with
DA3's cleaner depth, test whether `depth_edge_rel` can come back down.

## What DA3 changes (and what it can't)

- Global metric scale is largely re-normalized by Atlas anyway
  (`estimate_ground_scale` / `fit_ground_and_scale` pin the fitted ground to
  Y=0), so the wins to look for are **local/relative depth quality** (tears,
  band splits, silhouettes), **tier-2 camera height** (the one consumer of
  absolute metric accuracy), and the removal of the indoor/outdoor model split.
- `DA3METRIC-LARGE` scales canonical depth by the **solved focal** when the
  node has a solve (`focal_source: "solve"`), closing the FOV-assumption loop
  DA-V2's metric heads can't. Image-only nodes (`AtlasDepthMap`,
  `AtlasDepthAnything`) fall back to an assumed normal-lens focal
  (`focal_source: "assumed"` — DA3METRIC is a depth-only head and predicts no
  intrinsics, confirmed live; ground-pinning re-normalizes the scale).
- DA3 processes at ~504px (upper-bound resize), coarser than V2's dynamic
  sizing — watch for softened silhouettes after the bicubic upsample.

## Part 1 — automated A/B (`tools/compare_depth_backends.py`)

```powershell
# From the repo root, in a venv with [neural] + depth_anything_3 installed:
python tools/compare_depth_backends.py path/to/test_images/*.png --json da3_ab.json
```

The script runs the learned solve ONCE per image (shared across backends), then
per backend reports:

| Metric | Source | What "better" looks like |
|---|---|---|
| `torn_fraction` | `build_relief_mesh` stats | Lower = fewer spurious mesh holes |
| `ground_scale` + inliers | `estimate_ground_scale` | Scale near 1.0 for a metric model; more inliers = cleaner ground |
| `camera_height` + confidence | `estimate_ground_height_from_depth` | Plausible eye height (~1.4–1.8 m on eye-level shots); higher confidence |
| `near`/`far` | `DepthResult` | Plausible metric range for the scene |
| `focal_source` | DA3 metadata | `solve` when focal was threaded |
| runtime (s) | wall clock | — |

Record the table for ≥4 of the 4K test images (mix: exterior terrain, interior,
architecture with spires, foreground subject).

## Part 2 — manual ComfyUI A/B

Run each workflow twice — once at the default V2 Outdoor, once with every
`depth_model` combo set to `depth-anything/DA3METRIC-LARGE` — all other widgets
fixed:

1. `examples/atlas_camera_core_projection_workflow.json` (hangar interior) —
   viewport 📽 Project: silhouette tearing on the ship/gantries, hole coverage
   on orbit; relief-mesh OBJ export sanity.
2. `examples/atlas_camera_learned_workflow.json` (cathedral nave) — measured
   camera height + `scale_source` in the decomposed solve; depth preview
   banding/detail; ground/horizon masks unchanged (they don't use depth).
3. `examples/atlas_camera_hole_mask_workflow.json` — hole_mask coverage:
   fewer spurious tears inside continuous surfaces = better local depth.

Per run, check:

- **Depth preview** (`AtlasDepthAnything`): detail on thin structures, noise on
  flat walls/sky.
- **Relief mesh in viewport**: torn areas under 📽 Project after a moderate
  orbit; 🧭 Safe Zone envelope (bigger measured arc = genuinely better
  coverage).
- **Camera height** (`AtlasDecomposeSolve` / ℹ Info HUD): plausibility and
  `scale_source` tier.
- **`focal_source`** in the depth `LatentComponent` summary (solve-bearing
  nodes should say `solve`).
- **Runtime/VRAM** deltas (ComfyUI console / task manager).

## Part 3 — decision gates

- **Adopt-as-option (already done):** DA3 stays in the combos as opt-in.
- **Recalibrate:** if DA3 wins on tears/bands, re-run the band calibration
  (can `depth_edge_rel` come back down from 1.5? does `relief_grid` 384 still
  hold?) before recommending it in any preset.
- **Flip presets:** only if DA3 wins on BOTH the automated metrics and the
  manual checks across AI-generated images specifically — then update
  `_SCENE_TYPE_PRESETS` indoor/outdoor entries (one model can serve both) and
  the CLAUDE.md guidance.
- **DA3MONO / DA3NESTED:** MONO is relative-only (tier-2 height becomes
  up-to-scale — expect the solver warning); NESTED is CC BY-NC and 1.4B params
  — evaluate only for the parked multi-view patch-registration spike (v2 of
  the patches track), not as a mono default.

## Results log

| Date | Image/workflow | V2 torn_frac | DA3 torn_frac | V2 height (conf) | DA3 height (conf) | Notes |
|---|---|---|---|---|---|---|
| 2026-07-09 | 4K img 11-42-57 (long lens, 10281px focal) | 0.4692 | 0.4689 | — | 0.46m (0.33) | Parity on tearing; both ground scales far from 1 (7.3 / 3.5) — long-lens vista is hard for both |
| 2026-07-09 | 4K img 11-50-46 (pitch 22°) | 1.0000 (0 faces) | 0.7700 (4240 faces) | — (0.00) | — (0.00) | V2 mesh shattered completely; DA3 kept a usable mesh. Neither found ground |
| 2026-07-09 | 4K img 11-56-01 | 0.3134 | **0.1055** | 2.57m (0.46) | 1.24m (0.76) | DA3 ~3× fewer holes, +30% faces, ground scale 1.30 vs 0.62. DA3 near_m was −11.4 (negative raw values — didn't break ground fit/mesh) |
| 2026-07-09 | 4K img 12-00-58 (wide, 1390px focal) | 0.4081 | **0.1416** | 4.67m (0.71) | 0.98m (0.96) | DA3 ~3× fewer holes, height conf 0.96, ground scale 1.64 vs 0.34 |
| 2026-07-09 | Manual: depth-backend comparison workflow (artist eyes) | | | | | **User-confirmed: general tearing/quality visibly better with DA3METRIC-LARGE** in the side-by-side viewports |

Both backends ran with `focal_px` from the shared GeoCalib solve
(`focal_source: "solve"`), grid 128 / depth_edge_rel 0.5, cuda. Raw JSON from
the run: `tools/compare_depth_backends.py --json` output, 2026-07-09 session.
