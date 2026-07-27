# Atlas Camera — feature audit

Generated 2026-07-27 by `tools/build_feature_audit.py`. 
Machine-gathered evidence from `tools/audit_node_usage.py`; judgements from 
`tools/feature_audit_verdicts.py`. Regenerate with 
`python tools/build_feature_audit.py` (`--check` verifies freshness).

## What counts as evidence

A node has **product evidence** when a shipping workflow uses it, a test
exercises it specifically, or an MCP handler depends on it.

Two exclusions are deliberate and are the point of this document:

* **Registry and façade pin tests do not count.** They name every
  registered node by construction, so before this rework all nodes read as
  "referenced" and the signal was worthless.
* **Documentation does not count.** Documenting a node proves intent, not
  use — and all but one node is documented, so counting docs would flatten
  the signal to nothing.

**Absence of evidence is not a defect.** Every zero-evidence node was
executed in-process during the baseline probe and every one returned
meaningful output. Nothing here is DELETEd on suspicion.

Run the audit from the main checkout, never a git worktree: the tool
imports `atlas_camera` through the editable install while scanning files
relative to itself, so a worktree silently audits a different tree.

## Counts

* standard: **82**
* experimental: **4**
* legacy: **0**
* total registered: **86**
* standard nodes with no product evidence: **15**

| Verdict | Nodes |
|---|---:|
| `KEEP_CORE` | 66 |
| `HOLD_NEEDS_EVIDENCE` | 14 |
| `KEEP_EXPERIMENTAL` | 4 |
| `LEGACY_GATE` | 2 |

## Verdict legend

| Verdict | Meaning |
|---|---|
| `KEEP_CORE` | product evidence exists; stays in the standard registry |
| `KEEP_EXPERIMENTAL` | stays behind `ATLAS_EXPERIMENTAL` |
| `MIGRATE_CAPABILITY` | node goes, capability moves to a supported node first |
| `DEPRECATE` | still registered, marked superseded, removal scheduled |
| `LEGACY_GATE` | moved behind `ATLAS_LEGACY_NODES` this cycle |
| `DELETE` | removed outright |
| `HOLD_NEEDS_EVIDENCE` | zero product evidence, no proven duplicate — keep, revisit |

## Matrix

| Name | Module | Tier | Workflows | Dedicated tests | Live exec | Meaningful output | MCP | Docs | Overlapping replacement | Known defect | Compat risk | Verdict | Migration action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `AtlasAddPatchView` | atlas_camera/comfy/nodes_geometry.py:2498 | standard | 0 | 5 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasApplyLUT` | atlas_camera/comfy/nodes_solve.py:291 | standard | 1 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasApplyScaleReferences` | atlas_camera/comfy/nodes_solve.py:1643 | standard | 0 | 0 | ok | ok | 0 | 4 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasAssessImage` | atlas_camera/comfy/nodes_solve.py:1166 | standard | 3 | 4 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasAssessOutput` | atlas_camera/comfy/nodes_qa.py:348 | standard | 4 | 3 | not_attempted | not_attempted | 2 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasAttachSourcePlate` | atlas_camera/comfy/nodes_solve.py:471 | standard | 4 | 1 | not_attempted | not_attempted | 0 | 6 | — | — | — | **KEEP_CORE** | — |
| `AtlasBlockoutViewport` | atlas_camera/comfy/nodes_viewport.py:120 | standard | 14 | 10 | not_attempted | not_attempted | 0 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasBoundedBand` | atlas_camera/comfy/nodes_depth.py:658 | standard | 1 | 1 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasCleanPlateLayer` | atlas_camera/comfy/nodes_inpaint.py:707 | standard | 4 | 10 | not_attempted | not_attempted | 1 | 6 | — | — | — | **KEEP_CORE** | — |
| `AtlasCleanPlateStack` | atlas_camera/comfy/nodes_inpaint.py:1279 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasConstrainedSolve` | atlas_camera/comfy/nodes_solve.py:719 | standard | 0 | 0 | ok | ok | 0 | 3 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasDeband` | atlas_camera/comfy/nodes_solve.py:120 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasDebugReport` | atlas_camera/comfy/nodes_viewport.py:416 | standard | 3 | 3 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDecomposeCamera` | atlas_camera/comfy/nodes_solve.py:1741 | standard | 0 | 0 | ok | ok | 0 | 4 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasDecomposeSolve` | atlas_camera/comfy/nodes_solve.py:1713 | standard | 0 | 0 | ok | ok | 0 | 4 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasDefineShotCam` | atlas_camera/comfy/nodes_geometry.py:2234 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasDefocus` | atlas_camera/comfy/nodes_solve.py:227 | standard | 1 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthAnything` | atlas_camera/comfy/nodes_depth.py:37 | standard | 0 | 0 | ok | ok | 0 | 3 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasDepthBandSplit` | atlas_camera/comfy/nodes_depth.py:614 | standard | 0 | 1 | not_attempted | not_attempted | 1 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthCombine` | atlas_camera/comfy/nodes_depth.py:328 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthDetailEnhance` | atlas_camera/comfy/nodes_depth.py:252 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthLayerMask` | atlas_camera/comfy/nodes_depth.py:750 | standard | 2 | 4 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthMap` | atlas_camera/comfy/nodes_depth.py:87 | standard | 9 | 4 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthOutlierMask` | atlas_camera/comfy/nodes_depth.py:137 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDeriveInteriorRoom` | atlas_camera/comfy/nodes_geometry.py:2111 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustu… | — | **KEEP_CORE** | — |
| `AtlasDeriveProjectionGeometry` | atlas_camera/comfy/nodes_geometry.py:47 | standard | 0 | 3 | not_attempted | not_attempted | 0 | 4 | — | Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustu… | — | **KEEP_CORE** | — |
| `AtlasDeriveReliefMesh` | atlas_camera/comfy/nodes_geometry.py:732 | standard | 2 | 6 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDeriveRoofsFacades` | atlas_camera/comfy/nodes_geometry.py:2072 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustu… | — | **KEEP_CORE** | — |
| `AtlasDeriveTowersSpires` | atlas_camera/comfy/nodes_geometry.py:1994 | standard | 0 | 4 | not_attempted | not_attempted | 0 | 2 | — | Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustu… | — | **KEEP_CORE** | — |
| `AtlasDeriveWalls` | atlas_camera/comfy/nodes_geometry.py:1922 | standard | 1 | 5 | not_attempted | not_attempted | 0 | 3 | — | Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustu… | — | **KEEP_CORE** | — |
| `AtlasExportBlender` | atlas_camera/comfy/nodes_export.py:386 | standard | 8 | 1 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportCameraPathUSD` | atlas_camera/comfy/nodes_export.py:647 | standard | 0 | 2 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportMayaLayers` | atlas_camera/comfy/nodes_export.py:566 | standard | 8 | 2 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportMayaReviewScene` | atlas_camera/comfy/nodes_export.py:72 | standard | 5 | 1 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportNuke` | atlas_camera/comfy/nodes_export.py:419 | standard | 5 | 1 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportNukeLayers` | atlas_camera/comfy/nodes_export.py:482 | standard | 8 | 2 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportPlateEXR` | atlas_camera/comfy/nodes_export.py:687 | standard | 1 | 1 | not_attempted | not_attempted | 0 | 0 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportReliefMesh` | atlas_camera/comfy/nodes_export.py:106 | standard | 6 | 4 | not_attempted | not_attempted | 0 | 6 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportReviewPackage` | atlas_camera/comfy/nodes_export.py:30 | standard | 1 | 1 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportSolveJSON` | atlas_camera/comfy/nodes_export.py:50 | standard | 9 | 1 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportUSD` | atlas_camera/comfy/nodes_export.py:358 | standard | 9 | 1 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasExtractAnglePatch` | atlas_camera/comfy/nodes_geometry.py:2280 | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 0 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasGrade` | atlas_camera/comfy/nodes_solve.py:183 | standard | 1 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasGravityOverride` | atlas_camera/comfy/nodes_solve.py:1027 | standard | 0 | 0 | ok | ok | 0 | 0 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasGroundDepthMap` | atlas_camera/comfy/nodes_depth.py:417 | standard | 0 | 0 | ok | ok | 0 | 4 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasGroundMask` | atlas_camera/comfy/nodes_depth.py:455 | standard | 0 | 0 | ok | ok | 0 | 3 | AtlasGroundDepthMap, output 1 (ground_mask) | — | low — in no shipping workflow, no test, no MCP consumer | **LEGACY_GATE** | rewire consumers to AtlasGroundDepthMap slot 1; gate the node |
| `AtlasHorizonMask` | atlas_camera/comfy/nodes_depth.py:482 | standard | 0 | 0 | ok | ok | 0 | 3 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasImportAnglePatch` | atlas_camera/comfy/nodes_geometry.py:2402 | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 0 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasInpaintCrop` | atlas_camera/comfy/nodes_inpaint.py:342 | standard | 2 | 2 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasInpaintStitch` | atlas_camera/comfy/nodes_inpaint.py:411 | standard | 2 | 2 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasInput` | atlas_camera/comfy/nodes_viewport.py:608 | standard | 6 | 3 | not_attempted | not_attempted | 0 | 9 | — | — | — | **KEEP_CORE** | — |
| `AtlasInstanceMask` | atlas_camera/comfy/nodes_inpaint.py:600 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasLayerPreview` | atlas_camera/comfy/nodes_viewport.py:550 | standard | 3 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasLearnedSolveFromImage` | atlas_camera/comfy/nodes_solve.py:751 | standard | 9 | 3 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasLiveMeshRepair` | atlas_camera/comfy/nodes_geometry.py:900 | standard | 3 | 3 | not_attempted | not_attempted | 0 | 1 | AtlasPlanarHolePatch (per named layer) -> AtlasRetopologizeLayer(boundary_smooth_iterations) | CPU sawtooth path could emit non-manifold geometry and hang the pivot walk (fixed 2026-07-27 in core/mesh_repair.py, since the same code is reachable from tw… | medium — present in 3 shipping workflows, all of which are migrated in the same cycle; saved user graphs still resolve with ATLAS_LEGACY_NODES=1 for one migr… | **LEGACY_GATE** | boundary smoothing migrated to AtlasRetopologizeLayer; rewire workflows to the hole-patch chain; gate the node |
| `AtlasLoadPlate` | atlas_camera/comfy/nodes_solve.py:366 | standard | 0 | 0 | ok | ok | 0 | 2 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasLoadRAW` | atlas_camera/comfy/nodes_solve.py:491 | standard | 1 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasLoadSolveJSON` | atlas_camera/comfy/nodes_solve.py:1695 | standard | 0 | 0 | ok | ok | 1 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasMergeGeometry` | atlas_camera/comfy/nodes_geometry.py:2147 | standard | 1 | 3 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasMogeNormals` | atlas_camera/comfy/nodes_depth.py:181 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasOcclusionMask` | atlas_camera/comfy/nodes_geometry.py:3023 | standard | 0 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasPathGuidedHoleRepair` | atlas_camera/comfy/nodes_geometry.py:1585 | standard | 1 | 2 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasPlanarHolePatch` | atlas_camera/comfy/nodes_geometry.py:1272 | standard | 2 | 3 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasPlanarRewarp` | atlas_camera/comfy/nodes_planar.py:147 | standard | 0 | 0 | ok | ok | 0 | 1 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasPlanarUnwarp` | atlas_camera/comfy/nodes_planar.py:44 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasPredictHiddenGeometry` | atlas_camera/comfy/nodes_geometry.py:379 | experimental | 0 | 3 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasReferenceScaleSolve` | atlas_camera/comfy/nodes_solve.py:1121 | standard | 0 | 0 | ok | ok | 1 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasRegisterPlate` | atlas_camera/comfy/nodes_solve.py:64 | standard | 3 | 2 | not_attempted | not_attempted | 0 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasRenderFix` | atlas_camera/comfy/nodes_geometry.py:618 | experimental | 0 | 3 | not_attempted | not_attempted | 1 | 2 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasRetopologizeLayer` | atlas_camera/comfy/nodes_geometry.py:1095 | standard | 2 | 2 | not_attempted | not_attempted | 0 | 2 | — | method='smooth' Taubin-relaxes every vertex and then keeps the existing UVs on the grounds that topology is unchanged — but moving a vertex changes where it … | — | **KEEP_CORE** | — |
| `AtlasRollTrim` | atlas_camera/comfy/nodes_solve.py:912 | standard | 0 | 1 | not_attempted | not_attempted | 1 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasSAM3Mask` | atlas_camera/comfy/nodes_inpaint.py:237 | standard | 4 | 4 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasSDXLInpaint` | atlas_camera/comfy/nodes_inpaint.py:477 | standard | 2 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasScaleOverride` | atlas_camera/comfy/nodes_solve.py:826 | standard | 1 | 2 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasSceneHealthGate` | atlas_camera/comfy/nodes_solve.py:1479 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasScopeMask` | atlas_camera/comfy/nodes_inpaint.py:40 | standard | 3 | 3 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasSegmentedSDXLInpaint` | atlas_camera/comfy/nodes_inpaint.py:638 | standard | 0 | 0 | ok | ok | 0 | 5 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasSemanticMask` | atlas_camera/comfy/nodes_inpaint.py:182 | standard | 0 | 3 | not_attempted | not_attempted | 0 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasSkyDomeLayer` | atlas_camera/comfy/nodes_inpaint.py:1407 | standard | 4 | 4 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasSolveFromImage` | atlas_camera/comfy/nodes_solve.py:673 | standard | 0 | 0 | ok | ok | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasSolveGate` | atlas_camera/comfy/nodes_solve.py:1382 | standard | 4 | 2 | not_attempted | not_attempted | 2 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasStereoRender` | atlas_camera/comfy/nodes_viewport.py:1039 | standard | 0 | 0 | ok | ok | 0 | 1 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasUSDCameraLoader` | atlas_camera/comfy/nodes_solve.py:45 | standard | 0 | 0 | ok | ok | 0 | 3 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasVLMScaleCues` | atlas_camera/comfy/nodes_solve.py:1580 | standard | 0 | 1 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasVPVisualization` | atlas_camera/comfy/nodes_depth.py:534 | standard | 0 | 0 | ok | ok | 0 | 3 | — | — | low — nothing depends on it today | **HOLD_NEEDS_EVIDENCE** | find a shipping workflow or a dedicated test, or schedule deprecation next cycle |
| `AtlasViewportControls` | atlas_camera/comfy/nodes_viewport.py:54 | standard | 11 | 2 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |

## Appendix — nodes held rather than cut

Each of these executes correctly and returns meaningful output; what
it lacks is a consumer. That is a reason to find evidence or schedule
deprecation, not a reason to delete.

* **`AtlasApplyScaleReferences`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasConstrainedSolve`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasDecomposeCamera`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasDecomposeSolve`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasDepthAnything`** — Deliberately NOT a duplicate of AtlasDepthMap: it emits a min-max normalized IMAGE preview and destroys the metric payload, which is why it cannot feed geometry. Distinct contract, no consumer.
* **`AtlasGravityOverride`** — The only registered node documented NOWHERE — no workflow, no dedicated test, no MCP consumer, and zero hits in docs/. Executes correctly.
* **`AtlasGroundDepthMap`** — Keeps its HOLD despite being AtlasGroundMask's replacement: it is the superset of the two, so gating it as well would leave no way to get a ground mask at all.
* **`AtlasHorizonMask`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasLoadPlate`** — Near-superset of AtlasRegisterPlate for file-backed plates, but AtlasRegisterPlate covers the case it cannot: tagging an in-graph IMAGE tensor with no file (is_proxy=True). Not a merge candidate.
* **`AtlasPlanarRewarp`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasSegmentedSDXLInpaint`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasStereoRender`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasUSDCameraLoader`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code
* **`AtlasVPVisualization`** — executed in-process 2026-07-27 and returned meaningful output (reports/live_probe_baseline.json, F1) — working code without a product consumer, not dead code

## Appendix — known defects

* **`AtlasDeriveInteriorRoom`** (KEEP_CORE) — Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustum corner hits the plane. Deliberate and test-pinned (backdrop-only on a flat/all-sky scene), but the four derive nodes have no report output, so the fallback cannot be surfaced — and their only guard, a None camera, is a silent no-op.
* **`AtlasDeriveProjectionGeometry`** (KEEP_CORE) — Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustum corner hits the plane. Deliberate and test-pinned (backdrop-only on a flat/all-sky scene), but the four derive nodes have no report output, so the fallback cannot be surfaced — and their only guard, a None camera, is a silent no-op.
* **`AtlasDeriveRoofsFacades`** (KEEP_CORE) — Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustum corner hits the plane. Deliberate and test-pinned (backdrop-only on a flat/all-sky scene), but the four derive nodes have no report output, so the fallback cannot be surfaced — and their only guard, a None camera, is a silent no-op.
* **`AtlasDeriveTowersSpires`** (KEEP_CORE) — Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustum corner hits the plane. Deliberate and test-pinned (backdrop-only on a flat/all-sky scene), but the four derive nodes have no report output, so the fallback cannot be surfaced — and their only guard, a None camera, is a silent no-op.
* **`AtlasDeriveWalls`** (KEEP_CORE) — Emits an unconditional projection_backdrop plane even when extraction finds nothing, with a hardcoded 60 m fallback depth and invented extents when no frustum corner hits the plane. Deliberate and test-pinned (backdrop-only on a flat/all-sky scene), but the four derive nodes have no report output, so the fallback cannot be surfaced — and their only guard, a None camera, is a silent no-op.
* **`AtlasLiveMeshRepair`** (LEGACY_GATE) — CPU sawtooth path could emit non-manifold geometry and hang the pivot walk (fixed 2026-07-27 in core/mesh_repair.py, since the same code is reachable from two nodes that are staying)
* **`AtlasRetopologizeLayer`** (KEEP_CORE) — method='smooth' Taubin-relaxes every vertex and then keeps the existing UVs on the grounds that topology is unchanged — but moving a vertex changes where it projects, so projective registration degrades to 2.9e-2 against a 1.1e-3 build baseline (26x). Pinned by tests/test_retopologize_layer.py::test_method_smooth_deregisters_uvs_and_boundary_pass_reduces_it. The boundary_smooth_iterations pass regenerates UVs for what it moves and reduces the error; fixing 'smooth' itself needs its own change.
