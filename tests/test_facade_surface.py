"""Phase 0 safety net for the node_helpers layering refactor.

`comfy/nodes.py` is a COMPATIBILITY FAÇADE. The 2026-07-19 split reduced it
from a 9,110-line monolith to a thin re-export layer so that
`from atlas_camera.comfy.nodes import X` kept working for every class, shared
helper and registry mapping — including for saved workflows and any external
code importing from it.

`docs/dev/node_helpers_layering_plan.md` moves ~850 lines of host-agnostic math
out of `comfy/` and into `core/` across four further phases. Every one of those
moves is pure code motion, so the façade surface must come through UNCHANGED.
This test is the thing that guarantees it: a symbol silently dropped during a
move is exactly the failure a 4-phase refactor produces, and it would not be
caught by any other test in the suite — the node-registry tests pin node KEYS,
not importable NAMES, and the helpers are what is actually moving.

Deliberately pins the underscore-prefixed helpers too. They are "private" by
convention, but seven modules and two test files import them directly, so
during this refactor they are contract in practice.

If this fails:
  * a name is MISSING  -> a move dropped a re-export. Fix the move, not this
    list. That is the bug this test exists to catch.
  * a name is EXTRA    -> something new was added. If intended, add it below.
"""

import atlas_camera.comfy.nodes as nodes

#: Public surface — node classes, registry mappings, shared constants.
FACADE_PUBLIC = {
    "ATLAS_EXPERIMENTAL_DEFAULT", "AtlasAddPatchView", "AtlasApplyScaleReferences",
    "AtlasAssessImage", "AtlasAttachSourcePlate", "AtlasBlockoutViewport",
    "AtlasBoundedBand", "AtlasCleanPlateLayer", "AtlasCleanPlateStack",
    "AtlasConstrainedSolve", "AtlasDebugReport", "AtlasDecomposeCamera",
    "AtlasDecomposeSolve", "AtlasDefineShotCam", "AtlasDepthAnything",
    "AtlasDepthBandSplit", "AtlasDepthLayerMask", "AtlasDepthMap",
    "AtlasDepthOutlierMask", "AtlasDeriveInteriorRoom", "AtlasDeriveProjectionGeometry",
    "AtlasDeriveReliefMesh", "AtlasDeriveRoofsFacades", "AtlasDeriveTowersSpires",
    "AtlasDeriveWalls", "AtlasExportBlender", "AtlasExportCameraPathUSD",
    "AtlasExportMayaLayers", "AtlasExportMayaReviewScene", "AtlasExportNuke",
    "AtlasExportNukeLayers", "AtlasExportReliefMesh", "AtlasExportReviewPackage",
    "AtlasExportSolveJSON", "AtlasExportUSD", "AtlasExtractAnglePatch",
    "AtlasGravityOverride", "AtlasGroundDepthMap", "AtlasGroundMask",
    "AtlasHorizonMask", "AtlasImportAnglePatch", "AtlasInpaintCrop",
    "AtlasInpaintStitch", "AtlasInput", "AtlasInstanceMask",
    "AtlasLayerPreview", "AtlasLearnedSolveFromImage", "AtlasLoadImageSolveCamera",
    "AtlasLoadPlate", "AtlasLoadRAW", "AtlasLoadSolveJSON", "AtlasMegaPipeline",
    "AtlasMergeGeometry", "AtlasMogeNormals", "AtlasOcclusionMask",
    "AtlasPitchTrim", "AtlasPredictHiddenGeometry", "AtlasReferenceScaleSolve",
    "AtlasRegisterPlate", "AtlasRenderFix", "AtlasRollTrim",
    "AtlasSAM3Mask", "AtlasSDXLInpaint", "AtlasScaleOverride",
    "AtlasSceneHealthGate", "AtlasScopeMask", "AtlasSegmentedSDXLInpaint",
    "AtlasSemanticMask", "AtlasSkyDomeLayer", "AtlasSolveFromImage",
    "AtlasSolveGate", "AtlasUSDCameraLoader", "AtlasVLMScaleCues",
    "AtlasVPVisualization", "AtlasViewportControls", "EXPERIMENTAL_NODE_CLASS_MAPPINGS",
    "EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "annotations",
}

#: Underscore helpers that other modules import from the façade in practice.
FACADE_PRIVATE = {
    "_ATLAS_ASSESS_CACHE", "_ATLAS_BLOCKOUT_CACHE", "_ATLAS_BLOCKOUT_CACHE_MAX",
    "_ATLAS_INPUT_BAND_NAMES", "_ATLAS_INPUT_BOUNDARIES", "_AZIMUTH_VIEWS",
    "_BAND_GEOMETRY_CHOICES", "_BORDER_FLOOD_PX", "_BOUNDED_BAND_NOOP_M",
    "_DEPTH_MODEL_CHOICES", "_DISTANCE_VIEWS", "_ELEVATION_VIEWS",
    "_GROUND_SCALE_CACHE", "_IDENTITY_COMMENT_PREFIX", "_LAYER_DEBUG_PALETTE_HEX",
    "_LAYER_DEBUG_PRIMARY_HEX", "_MOGE_NORMAL_MODEL_CHOICES", "_MetricDepthSetup",
    "_MiniGraphBuilder", "_analytic_ground_forward_depth", "_apply_band_split",
    "_b64_png_to_mask", "_band_resolution_validity", "_blockout_cache_set",
    "_clone_solve_with_metadata", "_comfy_registry", "_decode_b64_to_tensor",
    "_depth_map_for_solve", "_execution_blocker", "_experimental_enabled",
    "_extend_edge_colors", "_extract_blockout_camera", "_extrinsics_from_view",
    "_fit_long_edge", "_flood_mask_to_frame_borders", "_format_hole_fill_report",
    "_graph_builder", "_ground_depth_compute", "_ground_scale_cached",
    "_health_summary_suffix", "_horizon_y_from_solve", "_image_fingerprint",
    "_image_tensor_to_pil", "_image_tensor_to_preview_b64", "_mask_to_b64_png",
    "_metric_depth_and_validity", "_named_view_orbit_delta", "_native_sam3_available",
    "_output_profile_to_dict", "_parse_band_override", "_parse_exact_view",
    "_parse_view_prompt", "_pil_to_image_tensor", "_plate_ref_to_dict",
    "_recompute_horizon_line", "_reference_id_choices", "_relief_mesh_from_solve",
    "_replace_proxy_role_geometry", "_require_numpy", "_require_pil",
    "_require_torch", "_resize_normal_field", "_resolve_band_geometry",
    "_resolve_depth_band", "_resolve_exclude_mask", "_resolve_raw_hints",
    "_save_image_tensor_to_tmp", "_scale_summary_suffix", "_seg_coverage",
    "_solve_camera_params", "_solve_fingerprint", "_solve_focal_px_for_image",
    "_solve_image_size", "_solve_with_relief_mesh", "_stamp_raw_provenance",
    "_write_export_manifest",
}


def _surface():
    return {n for n in dir(nodes) if not n.startswith("__")}


def test_facade_surface_is_unchanged():
    """The whole safety net. Names may be ADDED deliberately; none may vanish."""
    expected = FACADE_PUBLIC | FACADE_PRIVATE
    actual = _surface()
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert not missing, (
        f"{len(missing)} name(s) DISAPPEARED from the comfy.nodes façade: {missing}. "
        "A refactor move dropped a re-export — fix the move, not this list.")
    assert not extra, (
        f"{len(extra)} new name(s) on the façade: {extra}. "
        "If intended, add them to FACADE_PUBLIC/FACADE_PRIVATE.")


def test_every_pinned_name_actually_resolves():
    """A name present in dir() but raising on access would still break callers."""
    for name in sorted(FACADE_PUBLIC | FACADE_PRIVATE):
        assert getattr(nodes, name, None) is not None or hasattr(nodes, name), name


def test_registry_mappings_are_reachable_through_the_facade():
    """Saved workflows and comfy/__init__ resolve nodes through these."""
    assert len(nodes.NODE_CLASS_MAPPINGS) >= 68
    assert set(nodes.NODE_DISPLAY_NAME_MAPPINGS) == set(nodes.NODE_CLASS_MAPPINGS)
