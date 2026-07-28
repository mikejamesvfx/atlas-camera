"""Hand-authored verdicts for the feature audit — the judgement half.

`tools/audit_node_usage.py` gathers evidence mechanically. It cannot decide
whether a node should stay, because "nothing references it" and "it should go"
are different claims: most zero-evidence nodes in this repo execute perfectly
and simply have no consumer yet. Those judgements live here, in a plain dict
so they are reviewable in a diff, next to the evidence that justifies them.

Rules (applied by `build_feature_audit.py` in this order):

1. an explicit entry in ``VERDICTS`` wins;
2. otherwise ``experimental`` -> KEEP_EXPERIMENTAL, ``legacy`` -> LEGACY_GATE;
3. otherwise product evidence -> KEEP_CORE, none -> HOLD_NEEDS_EVIDENCE.

Only the seven allowed verdicts may appear:

    KEEP_CORE             product evidence exists; stays in the standard registry
    KEEP_EXPERIMENTAL     stays behind ATLAS_EXPERIMENTAL
    MIGRATE_CAPABILITY    node goes, but a capability moves to a supported node first
    DEPRECATE             still registered, marked superseded, removal scheduled
    LEGACY_GATE           moved behind ATLAS_LEGACY_NODES this cycle
    DELETE                removed outright
    HOLD_NEEDS_EVIDENCE   zero product evidence, no proven duplicate — keep, revisit

NOTHING is assigned DELETE this cycle. Every one of the zero-evidence nodes was
executed in-process during the 2026-07-27 baseline probe and every one returned
meaningful output (see reports/live_probe_baseline.json, finding F1). Absence
from the examples folder is not evidence of deadness.
"""

VERDICT_VALUES = frozenset({
    "KEEP_CORE",
    "KEEP_EXPERIMENTAL",
    "MIGRATE_CAPABILITY",
    "DEPRECATE",
    "LEGACY_GATE",
    "DELETE",
    "HOLD_NEEDS_EVIDENCE",
})

#: node key -> explicit judgement. Anything absent is defaulted (see module doc).
VERDICTS: dict[str, dict] = {
    "AtlasLiveMeshRepair": {
        "verdict": "LEGACY_GATE",
        "overlapping_replacement":
            "AtlasPlanarHolePatch (per named layer) -> AtlasRetopologizeLayer"
            "(boundary_smooth_iterations)",
        "known_defect":
            "CPU sawtooth path could emit non-manifold geometry and hang the "
            "pivot walk (fixed 2026-07-27 in core/mesh_repair.py, since the "
            "same code is reachable from two nodes that are staying)",
        "compatibility_risk":
            "medium — present in 3 shipping workflows, all of which are "
            "migrated in the same cycle; saved user graphs still resolve with "
            "ATLAS_LEGACY_NODES=1 for one migration cycle",
        "migration_action":
            "boundary smoothing migrated to AtlasRetopologizeLayer; rewire "
            "workflows to the hole-patch chain; gate the node",
        "evidence": [
            "superseded at the workflow level by AtlasPlanarHolePatch + "
            "AtlasPathGuidedHoleRepair (scoped masks, normal-guided plane "
            "fitting, smallest-first ordering, scale-aware gates, reports)",
            "remove_stretch_factor is 0.0 in every shipping workflow — never "
            "exercised",
            "its CPU repair path warns about freezing on complex torn meshes",
        ],
        "notes":
            "Gating removes the only caller of repair_relief_mesh_grid_cuda "
            "and remove_stretched_faces. What is genuinely lost is the CUDA "
            "grid repair, the harmonic enclosed-hole cap and the post-hoc "
            "stretch cull applied downstream on an already-built solve. What "
            "survives: CPU hole-fill + sawtooth via apply_live_mesh_repair "
            "(still called by AtlasDeriveReliefMesh and "
            "AtlasDeriveProjectionGeometry) and apply_interior_hole_fill in "
            "the relief-mesh exporter.",
    },
    "AtlasGroundMask": {
        "verdict": "LEGACY_GATE",
        "overlapping_replacement": "AtlasGroundDepthMap, output 1 (ground_mask)",
        "compatibility_risk": "low — in no shipping workflow, no test, no MCP consumer",
        "migration_action":
            "rewire consumers to AtlasGroundDepthMap slot 1; gate the node",
        "evidence": [
            "nodes_depth.py calls _ground_depth_compute(solve, w, h, 1.0, 50.0) "
            "and discards the rgb — the mask is the same array",
            "depth_geometry.py: near/far only drive the colour ramp, never the mask",
            "measured live 2026-07-27 across 5 camera configurations "
            "(default, fx=150, elevated eye, raised target, 640x480): "
            "bit-identical every time, and varying near/far 1/50 vs 5/500 does "
            "not change the mask (reports/live_probe_baseline.json, F2)",
        ],
    },

    # --- the 2026-07-24 HOLD cohort, resolved 2026-07-27 --------------------
    # All 14 now carry dedicated node-layer coverage
    # (tests/test_node_layer_contracts.py), so the audit's own rule promotes
    # them to KEEP_CORE. Recorded explicitly rather than left to the default,
    # because the interesting part is WHY they were held and what closing the
    # gap turned up: two of them (AtlasStereoRender, AtlasPlanarRewarp) looked
    # covered but were not — tests/test_stereo_render.py and
    # tests/test_planar_projection.py exercise the CORE math and never touch
    # the node classes — and writing the tests exposed a real inverted-output
    # bug in AtlasHorizonMask (see DEFECTS).
    **{
        name: {
            "verdict": "KEEP_CORE",
            "compatibility_risk": "low — nothing depended on it before either",
            "migration_action": "none; keep",
            "evidence": [
                "executed in-process 2026-07-27 and returned meaningful output "
                "(reports/live_probe_baseline.json, F1)",
                "dedicated node-layer contract test added 2026-07-27 "
                "(tests/test_node_layer_contracts.py): output arity/order "
                "against RETURN_TYPES, routed values, and the documented "
                "fail-soft or error path",
            ],
        }
        for name in (
            "AtlasApplyScaleReferences", "AtlasConstrainedSolve",
            "AtlasDecomposeCamera", "AtlasDecomposeSolve", "AtlasDepthAnything",
            "AtlasGravityOverride", "AtlasGroundDepthMap", "AtlasHorizonMask",
            "AtlasLoadPlate", "AtlasPlanarRewarp", "AtlasSegmentedSDXLInpaint",
            "AtlasStereoRender", "AtlasUSDCameraLoader", "AtlasVPVisualization",
        )
    },

    # --- the unseen-geometry track, merged 2026-07-28 -----------------------
    # Shipped WITH workflow evidence and dedicated tests on day one — the
    # thing every node in the HOLD cohort lacked. Recorded explicitly so the
    # contrast is legible next cycle.
    **{
        name: {
            "verdict": "KEEP_CORE",
            "compatibility_risk": "low — new this cycle, nothing to migrate",
            "migration_action": "none; keep",
            "evidence": [
                "shipped with example workflows (atlas_unseen_geometry_test, "
                "atlas_layered_segmentation, atlas_auto_layered_inpaint)",
                "dedicated tests: test_occlusion_graph.py, test_move_budget.py, "
                "test_layer_plan_node.py, test_depth_completion.py",
            ],
        }
        for name in ("AtlasOcclusionGraph", "AtlasMoveBudget", "AtlasLayerPlan")
    },
    "AtlasEquirectMultiView": {
        "verdict": "KEEP_CORE",
        "compatibility_risk": "low — new this cycle, nothing to migrate",
        "migration_action": "none; keep",
        "evidence": [
            "dedicated tests: tests/test_equirect.py — shared optical centre, "
            "ring azimuths, median-not-mean height consolidation, one shared "
            "ground scale, per-view geometry",
        ],
        "notes":
            "Exists because AtlasAddPatchView cannot express a panorama: it builds "
            "patch cameras with camera_math.orbit_camera, which MOVES the eye "
            "(rotates its offset from a ground pivot and re-aims), while panorama "
            "views share one optical centre and differ only in rotation. Chaining "
            "patch nodes therefore registers the geometry in the wrong place — and "
            "eleven of them killed the ComfyUI server outright, since each link "
            "deep-copies a solve and holds its own depth map. This walks the ring "
            "sequentially with one shared eye, one shared ground scale and one "
            "median-consolidated height.",
    },
    "AtlasSplitEquirect": {
        "verdict": "KEEP_CORE",
        "compatibility_risk": "low — new this cycle, nothing to migrate",
        "migration_action": "none; keep",
        "evidence": [
            "dedicated tests: tests/test_equirect.py — geometry (direction, "
            "latitude polarity, seam wrap) plus node-layer contract",
        ],
        "notes":
            "Atlas's answer to ComfyUI core's MoGePanoramaInference. Core merges "
            "12 views into one equirect depth map, which disables per-view metric "
            "scale and drops MoGe-2 normals; this keeps each crop a separate "
            "perspective solve, so both survive. It also feeds the existing "
            "AtlasAddPatchView rather than adding a parallel multi-camera path — "
            "and does so through exact_view_override, because equirect angles are "
            "measured rather than estimated.",
    },
}

# Node-specific notes layered on top of the bulk entries above.
VERDICTS["AtlasMoveBudget"]["notes"] = (
    "Answers \"how far can this camera move before the unseen geometry "
    "shows?\" by bisecting the move against the occlusion graph, rather than "
    "leaving it to the artist's eye.")
VERDICTS["AtlasGravityOverride"]["notes"] = (
    "Was the only registered node with NO evidence of any kind — no workflow, "
    "no test, no MCP consumer, and zero hits in docs/. Now pinned on the one "
    "property its name claims: the override is ABSOLUTE, so applying it twice "
    "is idempotent (a relative nudge would drift on the second application). "
    "It is also the ONLY in-graph repair for the GeoCalib gravity flip since "
    "the one-click AtlasPitchTrim was removed as unused in 23a9179, and the "
    "camera_looks_up health warning now names it — so the node a usage-signal "
    "cull would most likely have deleted is the one that fixes the user's RAW "
    "plates. Worth remembering the next time zero evidence looks like zero "
    "value.")
VERDICTS["AtlasGroundDepthMap"]["notes"] = (
    "AtlasGroundMask's supported replacement, and the equivalence the legacy "
    "gate rests on is now a test rather than only a measurement: output 1 is "
    "asserted bit-identical to the gated node, and near/far are asserted not "
    "to move the mask.")
VERDICTS["AtlasDepthAnything"]["notes"] = (
    "Deliberately NOT a duplicate of AtlasDepthMap: it emits a min-max "
    "normalized IMAGE preview and destroys the metric payload, which is why "
    "it cannot feed geometry. Distinct contract. Weights are network- and "
    "multi-GB, so the unit test pins the bad-model-id error path and leaves "
    "the happy path to the live probe.")
VERDICTS["AtlasLoadPlate"]["notes"] = (
    "Near-superset of AtlasRegisterPlate for file-backed plates, but "
    "AtlasRegisterPlate covers the case it cannot: tagging an in-graph IMAGE "
    "tensor with no file (is_proxy=True). Not a merge candidate.")

#: Known defects worth recording against nodes that are otherwise KEEP_CORE.
DEFECTS: dict[str, str] = {
    # AtlasRetopologizeLayer's method='smooth' UV defect was FIXED 2026-07-27
    # (it now regenerates projective UVs; 2.9e-2 -> 1.2e-3 against a 1.1e-3
    # build baseline). Kept out of DEFECTS deliberately — the regression is
    # pinned by tests/test_retopologize_layer.py::
    # test_method_smooth_regenerates_projective_uvs.
    "AtlasDeriveWalls":
        "FIXED 2026-07-27. Previously emitted a projection_backdrop plane even "
        "with no valid depth — hardcoded 60 m, invented extents, no way to "
        "explain it because the node had no report output. Now: the primitive "
        "records backdrop_depth_source/backdrop_extents_source, an appended "
        "`backdrop` widget defaults to measured_only (invented backdrops are "
        "dropped and reported; 'always' restores the old behaviour and says the "
        "plane is invented), and an appended `report` output carries the "
        "explanation. The fx<=0 guard now reports instead of silently no-opping.",
    "AtlasHorizonMask":
        "FIXED 2026-07-27, found by writing this cycle's coverage. The node "
        "returned the GROUND as sky — the exact inverse of its docstring and of "
        "its 'Sky Mask' display name. ax+by+c=0 names the same line for "
        "(a,b,c) and (-a,-b,-c), and the two producers disagreed: solver.py's "
        "learned path (the primary path for AI images) emits (0, 1, -horizon_y), "
        "for which the node's `signed` grows downward; the VP path builds the "
        "line via vanishing_points.line_from_points, whose sign flips with the "
        "ORDER of the two vanishing points, so that path's polarity was not even "
        "deterministic. The node now canonicalizes to b <= 0 before evaluating. "
        "Safe to change precisely because the audit had found the node has no "
        "consumers, so no saved graph depended on the inverted output. Pinned by "
        "tests/test_node_layer_contracts.py.",
}
for _n in ("AtlasDeriveTowersSpires", "AtlasDeriveRoofsFacades",
           "AtlasDeriveInteriorRoom", "AtlasDeriveProjectionGeometry"):
    DEFECTS[_n] = DEFECTS["AtlasDeriveWalls"]
