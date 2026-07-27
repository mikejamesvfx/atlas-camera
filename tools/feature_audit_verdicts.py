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

    # --- proven-working, no consumer: keep, find evidence or retire later ----
    **{
        name: {
            "verdict": "HOLD_NEEDS_EVIDENCE",
            "compatibility_risk": "low — nothing depends on it today",
            "migration_action":
                "find a shipping workflow or a dedicated test, or schedule "
                "deprecation next cycle",
            "evidence": [
                "executed in-process 2026-07-27 and returned meaningful output "
                "(reports/live_probe_baseline.json, F1) — working code without "
                "a product consumer, not dead code",
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
}

# Node-specific notes layered on top of the bulk HOLD entries above.
VERDICTS["AtlasGravityOverride"]["notes"] = (
    "The only registered node documented NOWHERE — no workflow, no dedicated "
    "test, no MCP consumer, and zero hits in docs/. Executes correctly.")
VERDICTS["AtlasGroundDepthMap"]["notes"] = (
    "Keeps its HOLD despite being AtlasGroundMask's replacement: it is the "
    "superset of the two, so gating it as well would leave no way to get a "
    "ground mask at all.")
VERDICTS["AtlasDepthAnything"]["notes"] = (
    "Deliberately NOT a duplicate of AtlasDepthMap: it emits a min-max "
    "normalized IMAGE preview and destroys the metric payload, which is why "
    "it cannot feed geometry. Distinct contract, no consumer.")
VERDICTS["AtlasLoadPlate"]["notes"] = (
    "Near-superset of AtlasRegisterPlate for file-backed plates, but "
    "AtlasRegisterPlate covers the case it cannot: tagging an in-graph IMAGE "
    "tensor with no file (is_proxy=True). Not a merge candidate.")

#: Known defects worth recording against nodes that are otherwise KEEP_CORE.
DEFECTS: dict[str, str] = {
    "AtlasRetopologizeLayer":
        "method='smooth' Taubin-relaxes every vertex and then keeps the "
        "existing UVs on the grounds that topology is unchanged — but moving a "
        "vertex changes where it projects, so projective registration degrades "
        "to 2.9e-2 against a 1.1e-3 build baseline (26x). Pinned by "
        "tests/test_retopologize_layer.py::"
        "test_method_smooth_deregisters_uvs_and_boundary_pass_reduces_it. "
        "The boundary_smooth_iterations pass regenerates UVs for what it moves "
        "and reduces the error; fixing 'smooth' itself needs its own change.",
    "AtlasDeriveWalls":
        "Emits an unconditional projection_backdrop plane even when extraction "
        "finds nothing, with a hardcoded 60 m fallback depth and invented "
        "extents when no frustum corner hits the plane. Deliberate and "
        "test-pinned (backdrop-only on a flat/all-sky scene), but the four "
        "derive nodes have no report output, so the fallback cannot be "
        "surfaced — and their only guard, a None camera, is a silent no-op.",
}
for _n in ("AtlasDeriveTowersSpires", "AtlasDeriveRoofsFacades",
           "AtlasDeriveInteriorRoom", "AtlasDeriveProjectionGeometry"):
    DEFECTS[_n] = DEFECTS["AtlasDeriveWalls"]
