"""Characterization tests pinning the ComfyUI node registry surface.

These lock the public contract of ``atlas_camera.comfy.nodes`` so the
mechanical split of ``nodes.py`` into responsibility modules cannot silently
change a registered node key, a display name, the experimental gate, or a
public import. They assert the *current* behavior verbatim — if any of these
fail during the refactor, the split changed the contract and must be fixed.
"""
from __future__ import annotations

import importlib
import os

import atlas_camera.comfy.nodes as nodes


# The exact registered node keys at the time of the nodes.py modularization
# (68 standard + 4 experimental = 72). ComfyUI serializes these keys into saved
# workflows, so this set is a compatibility contract, not an implementation
# detail.
NORMAL_KEYS = {
    "AtlasAddPatchView", "AtlasApplyScaleReferences", "AtlasAssessImage",
    "AtlasAttachSourcePlate", "AtlasBlockoutViewport", "AtlasBoundedBand",
    "AtlasCleanPlateLayer", "AtlasCleanPlateStack", "AtlasConstrainedSolve",
    "AtlasDebugReport", "AtlasDecomposeCamera", "AtlasDecomposeSolve",
    "AtlasDefineShotCam", "AtlasDepthAnything", "AtlasDepthBandSplit",
    "AtlasDepthLayerMask", "AtlasDepthMap", "AtlasDepthOutlierMask",
    "AtlasDeriveInteriorRoom", "AtlasDeriveProjectionGeometry",
    "AtlasDeriveReliefMesh", "AtlasDeriveRoofsFacades", "AtlasDeriveTowersSpires",
    "AtlasDeriveWalls", "AtlasExportBlender", "AtlasExportCameraPathUSD",
    "AtlasExportMayaLayers", "AtlasExportMayaReviewScene", "AtlasExportNuke",
    "AtlasExportNukeLayers", "AtlasExportReliefMesh", "AtlasExportReviewPackage",
    "AtlasExportSolveJSON", "AtlasExportUSD", "AtlasGravityOverride",
    "AtlasGroundDepthMap", "AtlasGroundMask", "AtlasHorizonMask",
    "AtlasInpaintCrop", "AtlasInpaintStitch", "AtlasInput", "AtlasInstanceMask",
    "AtlasLayerPreview", "AtlasLearnedSolveFromImage", "AtlasLoadImageSolveCamera",
    "AtlasLoadRAW", "AtlasLoadSolveJSON", "AtlasMergeGeometry", "AtlasMogeNormals",
    "AtlasOcclusionMask", "AtlasPitchTrim", "AtlasReferenceScaleSolve",
    "AtlasRegisterPlate", "AtlasRollTrim", "AtlasSAM3Mask", "AtlasSDXLInpaint",
    "AtlasScaleOverride",
    "AtlasSceneHealthGate", "AtlasScopeMask", "AtlasSegmentedSDXLInpaint",
    "AtlasSemanticMask", "AtlasSkyDomeLayer", "AtlasSolveFromImage",
    "AtlasSolveGate", "AtlasUSDCameraLoader", "AtlasVLMScaleCues",
    "AtlasVPVisualization", "AtlasViewportControls",
}

EXPERIMENTAL_KEYS = {
    "AtlasExtractAnglePatch", "AtlasImportAnglePatch",
    "AtlasPredictHiddenGeometry", "AtlasRenderFix",
    "AtlasMegaPipeline",
}

# Public helper/constant names some tests import directly from the module; the
# compatibility façade must keep re-exporting them.
FACADE_HELPER_NAMES = (
    "_ATLAS_BLOCKOUT_CACHE", "_image_fingerprint", "_solve_fingerprint",
    "_b64_png_to_mask", "_parse_view_prompt", "_parse_exact_view",
    "_parse_band_override", "_flood_mask_to_frame_borders", "_resolve_depth_band",
    "_relief_mesh_from_solve", "_resize_normal_field", "_write_export_manifest",
    "_scale_summary_suffix", "_health_summary_suffix",
)


def test_normal_registry_keys_exact():
    assert set(nodes.NODE_CLASS_MAPPINGS) == NORMAL_KEYS
    assert len(nodes.NODE_CLASS_MAPPINGS) == 68


def test_experimental_registry_keys_exact():
    assert set(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == EXPERIMENTAL_KEYS
    assert len(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == 5


def test_display_name_mapping_covers_registry():
    # Every registered normal node has a display name, and no extras.
    assert set(nodes.NODE_DISPLAY_NAME_MAPPINGS) == NORMAL_KEYS


def test_mapping_values_are_the_registered_classes():
    # The class object under each key must expose the ComfyUI contract.
    for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "INPUT_TYPES"), key
        assert hasattr(cls, "RETURN_TYPES"), key
        assert hasattr(cls, "FUNCTION"), key
        assert hasattr(cls, "CATEGORY"), key


def test_experimental_gate_off_by_default():
    # Default install ships the gate closed: experimental keys are NOT merged
    # into the standard registry, and stay in their own mapping.
    assert nodes.ATLAS_EXPERIMENTAL_DEFAULT == "0"
    assert not (EXPERIMENTAL_KEYS & set(nodes.NODE_CLASS_MAPPINGS))


def test_experimental_gate_merges_when_enabled(monkeypatch):
    # With ATLAS_EXPERIMENTAL truthy, a fresh import of the registration module
    # (where the gate + dict literals live post-modularization) merges the 4
    # experimental nodes into the standard registry.
    import atlas_camera.comfy.node_registry as registry
    monkeypatch.setenv("ATLAS_EXPERIMENTAL", "1")
    importlib.reload(registry)
    try:
        assert EXPERIMENTAL_KEYS <= set(registry.NODE_CLASS_MAPPINGS)
        assert EXPERIMENTAL_KEYS <= set(registry.NODE_DISPLAY_NAME_MAPPINGS)
    finally:
        monkeypatch.delenv("ATLAS_EXPERIMENTAL", raising=False)
        importlib.reload(registry)  # rebuild the default (gate-off) dicts
        importlib.reload(nodes)     # rebind the façade to the restored mappings


def test_representative_public_class_imports():
    from atlas_camera.comfy.nodes import (  # noqa: F401
        AtlasDepthMap, AtlasLearnedSolveFromImage, AtlasBlockoutViewport,
        AtlasExportNukeLayers, AtlasCleanPlateLayer, AtlasMergeGeometry,
        AtlasPitchTrim, AtlasInput,
    )
    # Experimental classes are importable as symbols even when gated out.
    from atlas_camera.comfy.nodes import (  # noqa: F401
        AtlasPredictHiddenGeometry, AtlasRenderFix,
        AtlasExtractAnglePatch, AtlasImportAnglePatch,
        AtlasMegaPipeline,
    )


def test_facade_reexports_public_helpers():
    for name in FACADE_HELPER_NAMES:
        assert hasattr(nodes, name), name
