# Atlas Camera — feature audit

Generated 2026-09-04 by `tools/build_feature_audit.py`.
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

* standard: **121**
* experimental: **10**
* legacy: **2**
* total registered: **135**
* standard nodes with no product evidence: **0**

| Verdict | Nodes |
|---|---:|
| `KEEP_CORE` | 121 |
| `KEEP_EXPERIMENTAL` | 10 |
| `LEGACY_GATE` | 2 |
| `IOS_GATE` | 2 |

## Verdict legend

| Verdict | Meaning |
|---|---|
| `KEEP_CORE` | product evidence exists; stays in the standard registry |
| `KEEP_EXPERIMENTAL` | stays behind `ATLAS_EXPERIMENTAL` |
| `MIGRATE_CAPABILITY` | node goes, capability moves to a supported node first |
| `DEPRECATE` | still registered, marked superseded, removal scheduled |
| `LEGACY_GATE` | moved behind `ATLAS_LEGACY_NODES` this cycle |
| `IOS_GATE` | held behind `ATLAS_IOS` (iOS/Record3D capture); a v2 capability |
| `DELETE` | removed outright |
| `HOLD_NEEDS_EVIDENCE` | zero product evidence, no proven duplicate — keep, revisit |

## Matrix

| Name | Module | Tier | Workflows | Dedicated tests | Live exec | Meaningful output | MCP | Docs | Overlapping replacement | Known defect | Compat risk | Verdict | Migration action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `AtlasAddPatchView` | atlas_camera/comfy/nodes_geometry.py | standard | 4 | 18 | not_attempted | not_attempted | 0 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasAgentHandoff` | atlas_camera/comfy/nodes_agent.py | experimental | 1 | 2 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasAnchorDepth` | atlas_camera/comfy/nodes_ltx.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasApplyDepthCalibration` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasApplyLUT` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasApplyScaleReferences` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 3 | ok | ok | 0 | 4 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasAssessImage` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 6 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasAssessOutput` | atlas_camera/comfy/nodes_qa.py | standard | 1 | 2 | not_attempted | not_attempted | 2 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasAttachSourcePlate` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 6 | — | — | — | **KEEP_CORE** | — |
| `AtlasBlenderImportMeshes` | atlas_camera/comfy/nodes_geometry.py | experimental | 1 | 3 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasBlenderMassing` | atlas_camera/comfy/nodes_geometry.py | experimental | 1 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasBlockoutMassing` | atlas_camera/comfy/nodes_geometry.py | experimental | 0 | 6 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasBlockoutViewport` | atlas_camera/comfy/nodes_viewport.py | standard | 19 | 18 | not_attempted | not_attempted | 1 | 8 | — | — | — | **KEEP_CORE** | — |
| `AtlasBoundedBand` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 2 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasCameraMovePreset` | atlas_camera/comfy/nodes_fill.py | standard | 2 | 4 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasCardMask` | atlas_camera/comfy/nodes_inpaint.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasCleanPlateLayer` | atlas_camera/comfy/nodes_inpaint.py | standard | 5 | 11 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasCleanPlateStack` | atlas_camera/comfy/nodes_inpaint.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasCompleteDepth` | atlas_camera/comfy/nodes_completion.py | experimental | 0 | 2 | not_attempted | not_attempted | 1 | 2 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasCompositeCrop` | atlas_camera/comfy/nodes_fill.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasConstrainedSolve` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasCropROI` | atlas_camera/comfy/nodes_fill.py | standard | 1 | 4 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasCropSourcePhoto` | atlas_camera/comfy/nodes_fill.py | standard | 1 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDeband` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDebugReport` | atlas_camera/comfy/nodes_viewport.py | standard | 6 | 4 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDecomposeCamera` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | ok | ok | 0 | 4 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasDecomposeSolve` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasDefineShotCam` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasDefocus` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthAnything` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasDepthBandSplit` | atlas_camera/comfy/nodes_depth.py | standard | 1 | 1 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthCombine` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthDetailEnhance` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthLayerMask` | atlas_camera/comfy/nodes_depth.py | standard | 1 | 4 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthMap` | atlas_camera/comfy/nodes_depth.py | standard | 13 | 5 | not_attempted | not_attempted | 1 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasDepthOutlierMask` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasDeriveInteriorRoom` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because th… | — | **KEEP_CORE** | — |
| `AtlasDeriveProjectionGeometry` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 5 | not_attempted | not_attempted | 0 | 4 | — | FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because th… | — | **KEEP_CORE** | — |
| `AtlasDeriveReliefMesh` | atlas_camera/comfy/nodes_geometry.py | standard | 8 | 15 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasDeriveRoofsFacades` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because th… | — | **KEEP_CORE** | — |
| `AtlasDeriveTowersSpires` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 4 | not_attempted | not_attempted | 0 | 2 | — | FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because th… | — | **KEEP_CORE** | — |
| `AtlasDeriveWalls` | atlas_camera/comfy/nodes_geometry.py | standard | 3 | 5 | not_attempted | not_attempted | 0 | 3 | — | FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because th… | — | **KEEP_CORE** | — |
| `AtlasDirectorTake` | atlas_camera/comfy/nodes_director.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasDisocclusionGuide` | atlas_camera/comfy/nodes_viewport.py | standard | 0 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasEquirectMultiView` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 3 | not_attempted | not_attempted | 0 | 2 | — | — | low — new this cycle, nothing to migrate | **KEEP_CORE** | none; keep |
| `AtlasExportBlender` | atlas_camera/comfy/nodes_export.py | standard | 2 | 1 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportCameraPathUSD` | atlas_camera/comfy/nodes_export.py | standard | 1 | 2 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportMayaLayers` | atlas_camera/comfy/nodes_export.py | standard | 0 | 1 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportMayaReviewScene` | atlas_camera/comfy/nodes_export.py | standard | 1 | 0 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportNuke` | atlas_camera/comfy/nodes_export.py | standard | 3 | 2 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportNukeLayers` | atlas_camera/comfy/nodes_export.py | standard | 1 | 1 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportPlateEXR` | atlas_camera/comfy/nodes_export.py | standard | 1 | 3 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportPlateHandoff` | atlas_camera/comfy/nodes_world_plate.py | standard | 2 | 2 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportReliefMesh` | atlas_camera/comfy/nodes_export.py | standard | 5 | 5 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportReviewPackage` | atlas_camera/comfy/nodes_export.py | standard | 1 | 2 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportScenePackage` | atlas_camera/comfy/nodes_export.py | standard | 3 | 4 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportSolveJSON` | atlas_camera/comfy/nodes_export.py | standard | 4 | 1 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExportUSD` | atlas_camera/comfy/nodes_export.py | standard | 1 | 0 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasExtractAnglePatch` | atlas_camera/comfy/nodes_geometry.py | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasFaceScaleReference` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasFillOccluded` | atlas_camera/comfy/nodes_fill.py | standard | 1 | 3 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasFitDepthCalibration` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasGrade` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasGravityCompass` | atlas_camera/comfy/nodes_solve.py | standard | 2 | 3 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasGravityOverride` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | ok | ok | 0 | 2 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasGroundDepthMap` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | ok | ok | 0 | 4 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasGroundMask` | atlas_camera/comfy/nodes_depth.py | legacy | 0 | 2 | ok | ok | 0 | 2 | AtlasGroundDepthMap, output 1 (ground_mask) | — | low — in no shipping workflow, no test, no MCP consumer | **LEGACY_GATE** | rewire consumers to AtlasGroundDepthMap slot 1; gate the node |
| `AtlasGroundPlane` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasHorizonMask` | atlas_camera/comfy/nodes_depth.py | standard | 1 | 1 | ok | ok | 0 | 3 | — | FIXED 2026-07-27, found by writing this cycle's coverage. The node returned the GROUND as sky — the exact inverse of its docstring and of its 'Sky Mask' disp… | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasImportAnglePatch` | atlas_camera/comfy/nodes_geometry.py | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasInpaintCrop` | atlas_camera/comfy/nodes_inpaint.py | standard | 2 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasInpaintStitch` | atlas_camera/comfy/nodes_inpaint.py | standard | 2 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasInput` | atlas_camera/comfy/nodes_viewport.py | standard | 10 | 7 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasInstanceMask` | atlas_camera/comfy/nodes_inpaint.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasInterpassGate` | atlas_camera/comfy/nodes_fill.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasLayerPlan` | atlas_camera/comfy/nodes_completion.py | standard | 1 | 1 | not_attempted | not_attempted | 1 | 2 | — | — | low — new this cycle, nothing to migrate | **KEEP_CORE** | none; keep |
| `AtlasLayerPreview` | atlas_camera/comfy/nodes_viewport.py | standard | 2 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasLearnedSolveFromImage` | atlas_camera/comfy/nodes_solve.py | standard | 6 | 5 | not_attempted | not_attempted | 1 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasLiveMeshRepair` | atlas_camera/comfy/nodes_geometry.py | legacy | 0 | 6 | not_attempted | not_attempted | 0 | 2 | AtlasPlanarHolePatch (per named layer) -> AtlasRetopologizeLayer(boundary_smooth_iterations) | CPU sawtooth path could emit non-manifold geometry and hang the pivot walk (fixed 2026-07-27 in core/mesh_repair.py, since the same code is reachable from tw… | medium — present in 3 shipping workflows, all of which are migrated in the same cycle; saved user graphs still resolve with ATLAS_LEGACY_NODES=1 for one migr… | **LEGACY_GATE** | boundary smoothing migrated to AtlasRetopologizeLayer; rewire workflows to the hole-patch chain; gate the node |
| `AtlasLoadCameraPath` | atlas_camera/comfy/nodes_ltx.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | low — new this cycle, nothing to migrate | **KEEP_CORE** | none; keep |
| `AtlasLoadDynamicPlate` | atlas_camera/comfy/nodes_dynamic.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasLoadHiddenVolume` | atlas_camera/comfy/nodes_hidden_volume.py | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasLoadPlate` | atlas_camera/comfy/nodes_solve.py | standard | 3 | 4 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasLoadRAW` | atlas_camera/comfy/nodes_solve.py | standard | 5 | 9 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasLoadRecord3D` | — | ios | 0 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **IOS_GATE** | — |
| `AtlasLoadSolveJSON` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 0 | ok | ok | 1 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasMaskedSurfaceReconstruct` | atlas_camera/comfy/nodes_geometry.py | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasMembraneComposite` | atlas_camera/comfy/nodes_fill.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasMergeGeometry` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 4 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasMogeNormals` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasMoveBudget` | atlas_camera/comfy/nodes_completion.py | standard | 0 | 4 | not_attempted | not_attempted | 1 | 2 | — | — | low — new this cycle, nothing to migrate | **KEEP_CORE** | none; keep |
| `AtlasMultiViewSolve` | atlas_camera/comfy/nodes_multiview.py | standard | 2 | 2 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasMultiViewSolveBurst` | atlas_camera/comfy/nodes_multiview.py | standard | 2 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasOcclusionGraph` | atlas_camera/comfy/nodes_completion.py | standard | 4 | 4 | not_attempted | not_attempted | 1 | 2 | — | — | low — new this cycle, nothing to migrate | **KEEP_CORE** | none; keep |
| `AtlasOcclusionMask` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 4 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasOpenRealPlate` | atlas_camera/comfy/nodes_world_plate.py | standard | 2 | 2 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasOutpaintDepth` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasPathFrameIndex` | atlas_camera/comfy/nodes_fill.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasPathGuidedHoleRepair` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 1 | not_attempted | not_attempted | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasPlanarHolePatch` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 7 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasPlaneMattes` | atlas_camera/comfy/nodes_geometry.py | standard | 3 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasPlateLayer` | atlas_camera/comfy/nodes_inpaint.py | standard | 1 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasProject` | atlas_camera/comfy/nodes_project.py | standard | 4 | 2 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasReadLockedPlatePlan` | atlas_camera/comfy/nodes_world_plate.py | standard | 2 | 2 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasRealPlateToScene` | atlas_camera/comfy/nodes_world_plate.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasRecordPlateAttempt` | atlas_camera/comfy/nodes_world_plate.py | standard | 2 | 2 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasReferenceScaleSolve` | atlas_camera/comfy/nodes_solve.py | standard | 1 | 2 | ok | ok | 1 | 7 | — | — | — | **KEEP_CORE** | — |
| `AtlasRefineOcclusionSeams` | atlas_camera/comfy/nodes_geometry.py | experimental | 0 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_EXPERIMENTAL** | — |
| `AtlasRegisterPlate` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 6 | — | — | — | **KEEP_CORE** | — |
| `AtlasReliefGeometry` | atlas_camera/comfy/nodes_ltx.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasRetopologizeLayer` | atlas_camera/comfy/nodes_geometry.py | standard | 5 | 8 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasRollTrim` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | not_attempted | not_attempted | 1 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasSAM3Mask` | atlas_camera/comfy/nodes_inpaint.py | standard | 5 | 5 | not_attempted | not_attempted | 1 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasSDXLInpaint` | atlas_camera/comfy/nodes_inpaint.py | standard | 2 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasScaleOverride` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | not_attempted | not_attempted | 1 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasSceneHealthGate` | atlas_camera/comfy/nodes_solve.py | standard | 5 | 2 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasSceneScale` | atlas_camera/comfy/nodes_geometry.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasScopeMask` | atlas_camera/comfy/nodes_inpaint.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasSegmentedSDXLInpaint` | atlas_camera/comfy/nodes_inpaint.py | standard | 0 | 1 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasSemanticMask` | atlas_camera/comfy/nodes_inpaint.py | standard | 0 | 3 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasShootList` | atlas_camera/comfy/nodes_completion.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **KEEP_CORE** | — |
| `AtlasSkyDomeLayer` | atlas_camera/comfy/nodes_inpaint.py | standard | 1 | 4 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasSolveBurstPatchCrops` | atlas_camera/comfy/nodes_multiview.py | standard | 1 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasSolveFromImage` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 0 | ok | ok | 1 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasSolveGate` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 4 | not_attempted | not_attempted | 2 | 5 | — | — | — | **KEEP_CORE** | — |
| `AtlasSolvePatchViews` | atlas_camera/comfy/nodes_geometry.py | standard | 1 | 1 | not_attempted | not_attempted | 0 | 2 | — | — | — | **KEEP_CORE** | — |
| `AtlasSplitEquirect` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | low — new this cycle, nothing to migrate | **KEEP_CORE** | none; keep |
| `AtlasStereoRender` | atlas_camera/comfy/nodes_viewport.py | standard | 0 | 3 | ok | ok | 0 | 2 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasStreamRecord3D` | — | ios | 0 | 1 | not_attempted | not_attempted | 0 | 3 | — | — | — | **IOS_GATE** | — |
| `AtlasUSDCameraLoader` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 1 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasUnrealCameraPath` | atlas_camera/comfy/nodes_ltx.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasUnrealDepthGeometry` | atlas_camera/comfy/nodes_ltx.py | standard | 0 | 1 | not_attempted | not_attempted | 0 | 1 | — | — | — | **KEEP_CORE** | — |
| `AtlasVLMScaleCues` | atlas_camera/comfy/nodes_solve.py | standard | 0 | 2 | not_attempted | not_attempted | 0 | 4 | — | — | — | **KEEP_CORE** | — |
| `AtlasVPVisualization` | atlas_camera/comfy/nodes_depth.py | standard | 0 | 1 | ok | ok | 0 | 3 | — | — | low — nothing depended on it before either | **KEEP_CORE** | none; keep |
| `AtlasViewportControls` | atlas_camera/comfy/nodes_viewport.py | standard | 13 | 2 | not_attempted | not_attempted | 0 | 5 | — | — | — | **KEEP_CORE** | — |

## Appendix — nodes held rather than cut

Each of these executes correctly and returns meaningful output; what
it lacks is a consumer. That is a reason to find evidence or schedule
deprecation, not a reason to delete.


## Appendix — capabilities REMOVED, not replaced

Retiring `AtlasLiveMeshRepair` to the legacy tier keeps one of its four
capabilities and drops three. They are listed here so the migration is
not mistaken for an equivalence:

| Capability | Status | Where it went |
|---|---|---|
| Boundary Taubin smoothing | **migrated** | `AtlasRetopologizeLayer(boundary_smooth_iterations)`, verbatim implementation, UVs regenerated |
| CUDA 2D grid hole fill | **removed from the repair path** | `core/mesh_repair.repair_relief_mesh_grid_cuda` is no longer reachable from a default-tier REPAIR node, but it is NOT dead: `core/move_budget.seal_relief_mesh` calls it to seal a mesh before measuring disocclusion, reached from `AtlasMoveBudget` (registered unconditionally), and `AtlasLiveMeshRepair` still calls it on the legacy tier. The repair replacement, `AtlasPlanarHolePatch`, is a different algorithm: per-component plane fitting with reports and gates, not a grid convolution |
| Harmonic enclosed-hole cap | **removed from the repair path** | same function, same two surviving callers; the membrane fill for sealed pockets has no equivalent in the planar patch |
| Post-hoc stretch cull (`remove_stretch_factor`) | **removed from the default tier** | `core/mesh_repair.remove_stretched_faces` is still called by `AtlasLiveMeshRepair` on the legacy tier. Deliberately NOT appended to `AtlasRetopologizeLayer`: every shipping workflow set it to 0.0, it has no node-level test, and `max_edge_factor` on the layer/derive nodes covers the same test at build time with the depth map in hand |

All three operated *downstream, on an already-built
solve*. What survives on the build path is unaffected: CPU hole fill and
sawtooth bridging still run via `apply_live_mesh_repair` from
`AtlasDeriveReliefMesh` and `AtlasDeriveProjectionGeometry`, and
`apply_interior_hole_fill` still backs the relief-mesh exporter.

Re-exposing any of the three is a one-widget append if evidence appears.


## Appendix — known defects

* **`AtlasDeriveInteriorRoom`** (KEEP_CORE) — FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because the node had no report output. Now: the primitive records backdrop_depth_source/backdrop_extents_source, an appended `backdrop` widget defaults to measured_only (invented backdrops are dropped and reported; 'always' restores the old behaviour and says the plane is invented), and an appended `report` output carries the explanation. The fx<=0 guard now reports instead of silently no-opping.
* **`AtlasDeriveProjectionGeometry`** (KEEP_CORE) — FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because the node had no report output. Now: the primitive records backdrop_depth_source/backdrop_extents_source, an appended `backdrop` widget defaults to measured_only (invented backdrops are dropped and reported; 'always' restores the old behaviour and says the plane is invented), and an appended `report` output carries the explanation. The fx<=0 guard now reports instead of silently no-opping.
* **`AtlasDeriveRoofsFacades`** (KEEP_CORE) — FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because the node had no report output. Now: the primitive records backdrop_depth_source/backdrop_extents_source, an appended `backdrop` widget defaults to measured_only (invented backdrops are dropped and reported; 'always' restores the old behaviour and says the plane is invented), and an appended `report` output carries the explanation. The fx<=0 guard now reports instead of silently no-opping.
* **`AtlasDeriveTowersSpires`** (KEEP_CORE) — FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because the node had no report output. Now: the primitive records backdrop_depth_source/backdrop_extents_source, an appended `backdrop` widget defaults to measured_only (invented backdrops are dropped and reported; 'always' restores the old behaviour and says the plane is invented), and an appended `report` output carries the explanation. The fx<=0 guard now reports instead of silently no-opping.
* **`AtlasDeriveWalls`** (KEEP_CORE) — FIXED 2026-07-27. Previously emitted a projection_backdrop plane even with no valid depth — hardcoded 60 m, invented extents, no way to explain it because the node had no report output. Now: the primitive records backdrop_depth_source/backdrop_extents_source, an appended `backdrop` widget defaults to measured_only (invented backdrops are dropped and reported; 'always' restores the old behaviour and says the plane is invented), and an appended `report` output carries the explanation. The fx<=0 guard now reports instead of silently no-opping.
* **`AtlasHorizonMask`** (KEEP_CORE) — FIXED 2026-07-27, found by writing this cycle's coverage. The node returned the GROUND as sky — the exact inverse of its docstring and of its 'Sky Mask' display name. ax+by+c=0 names the same line for (a,b,c) and (-a,-b,-c), and the two producers disagreed: solver.py's learned path (the primary path for AI images) emits (0, 1, -horizon_y), for which the node's `signed` grows downward; the VP path builds the line via vanishing_points.line_from_points, whose sign flips with the ORDER of the two vanishing points, so that path's polarity was not even deterministic. The node now canonicalizes to b <= 0 before evaluating. Safe to change precisely because the audit had found the node has no consumers, so no saved graph depended on the inverted output. Pinned by tests/test_node_layer_contracts.py.
* **`AtlasLiveMeshRepair`** (LEGACY_GATE) — CPU sawtooth path could emit non-manifold geometry and hang the pivot walk (fixed 2026-07-27 in core/mesh_repair.py, since the same code is reachable from two nodes that are staying)
