# Response to the Atlas Engineering Recommendations (2026-07-17)

**Date:** 2026-07-18
**Refers to:** [ATLAS_ENGINEERING_RECOMMENDATIONS.md](ATLAS_ENGINEERING_RECOMMENDATIONS.md)
(external review, same batch as the project-wide stale-code report — see
`stale_code_report_response.md` for that one).

The review's central thesis — *"Atlas can produce a convincing image while
still being metrically wrong"* — was proven live the day after it was written:
a real D810 bird's-eye NEF solved with a perfect-looking projection while the
metric scale silently fell to the assumed 1.6 m tier (~30× small). That
incident is the acceptance test for the accepted work below.

## Already shipped (the review had partial visibility)

| Review asks for | Reality |
|---|---|
| Three canonical workflows (Quickstart / Production / Research) | Shipped — `examples/showcase/atlas_canonical_*`, exactly that naming (likely prompted by this review) |
| Plate-suitability preflight | `AtlasAssessImage` 🧭 (scene type, depth model, band plan, camera-move rubric) |
| Spatial diagnostics (observed/inferred pixels, restricted masks, holes) | `hole_mask`, `extend_mask` (invented pixels), 🩻 hidden-mask overlay, `AtlasDebugReport` machine-readable flags |
| Scale provenance tiers | The tiered cascade + `scale_source` provenance (CLAUDE.md "measured, not assumed" rule) — compressed into debug metadata, which is the accepted gap below |
| Gate/approval infrastructure | The ExecutionBlocker gate family + `docs/dev/gate_state_table.md` doctrine |
| Example-catalog / temp-root complaints | Resolved in the stale-code pass — the "failing catalog test" was untracked local clutter in the portable install clone |
| Benchmark scene set | The 9 showcase plates cover the exact scene list (street / birdseye / interior / sky-castle / organic / occlusion); what's missing is metric *tracking* over them (P2, deferred) |
| USD as first-class | USD camera + review-package USD trio ship today |

## Accepted — the P0 reliability & trust tier (this effort)

1. **Scale trust surfacing** — `core/scene_health.scale_health()`: status
   (measured / manual / assumed / unknown) + `safe_to_export`, stamped into
   every solve JSON, shown as a HUD warning, in the solve-gate report, in
   exporter summaries and `report.md`.
2. **Confidence vector** — `scale` + `depth` keys appended to
   `ConfidenceModel`'s fixed key set, populated per solve path, riding
   `confidence_detail` into every export.
3. **Scene-health gate** — `AtlasSceneHealthGate` 🩺: the `AtlasDebugReport`
   red-flag engine factored into `core/scene_health.evaluate_scene_health()`
   and put in front of exporters as a ship-closed approval gate
   (override-able, but the warning is stamped into `debug_metadata` and
   survives into every artifact — the review's own "override a warning, but
   not lose the warning" rule).
4. **`atlas_project.json` manifest** — versioned reproducibility bundle
   (plate checksum, solve fingerprint, model IDs, seeds, scale + health,
   settings, artifact list) written by the review package and every export
   node; identity hash embedded as a comment in `.nk`/`.ma` outputs.

## Pushed back, with rationale

- **Per-solve "reprojection error"** (appears throughout the review as a
  metric): a single-image solve has no correspondences independent of the
  fit — reprojecting the very lines/priors that produced the camera scores
  ~perfect by construction. The honest per-solve residuals already exist:
  the `vp1..3`/`horizon` confidence metrics (VP path) and GeoCalib's
  uncertainty mapping (learned path). A meaningful reprojection error needs
  a second view or ground truth — that is the P2 benchmark plan's job
  (patch registration already reports rel-MAD where a second view exists).
- **Automated Maya/Nuke import smoke tests in CI**: no DCC licenses in CI.
  The substitute doctrine stands: syntax-level exporter tests
  (`test_nuke_exporter.py`, `test_maya_exporter.py`) plus the repo's
  live-verification rule (both exporters were verified in real Nuke 16.1v3 /
  Maya 2027 via mayapy, with findings baked into code comments). Recorded as
  accepted risk; revisit if headless DCC access ever lands in CI.
- **P1–P3 (project manifest UX, plugin interfaces, remote execution,
  standalone review viewer, benchmark metric tracking)**: deferred, not
  rejected — out of scope for this tier; the P0 groundwork (manifest schema,
  health engine) is designed to be what those build on.

## Status

Implemented on branch `claude/atlas-trust-tier` (milestones M0–M7); this note
is updated with "shipped in <version>" markers as milestones land.
