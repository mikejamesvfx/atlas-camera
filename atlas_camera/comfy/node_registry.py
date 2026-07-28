"""Central ComfyUI registration for Atlas Camera.

Imports every node class from its responsibility module and builds the
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS the ComfyUI loader reads,
plus the ATLAS_EXPERIMENTAL gate that merges the experimental tier at
import time. The node keys and display names here are a saved-workflow
compatibility contract — never rename or reorder an existing entry.
"""
from __future__ import annotations

import os

from atlas_camera.comfy.nodes_viewport import (
    AtlasViewportControls,
    AtlasBlockoutViewport,
    AtlasStereoRender,
    AtlasDebugReport,
    AtlasLayerPreview,
    AtlasInput,
)
from atlas_camera.comfy.nodes_completion import (
    AtlasCompleteDepth,
    AtlasLayerPlan,
    AtlasMoveBudget,
    AtlasOcclusionGraph,
)
from atlas_camera.comfy.nodes_qa import AtlasAssessOutput
from atlas_camera.comfy.nodes_solve import (
    AtlasLoadPlate,
    AtlasRegisterPlate,
    AtlasDeband,
    AtlasGrade,
    AtlasDefocus,
    AtlasApplyLUT,
    AtlasAttachSourcePlate,
    AtlasLoadRAW,
    AtlasSolveFromImage,
    AtlasConstrainedSolve,
    AtlasLearnedSolveFromImage,
    AtlasScaleOverride,
    AtlasRollTrim,
    AtlasGravityOverride,
    AtlasReferenceScaleSolve,
    AtlasAssessImage,
    AtlasSolveGate,
    AtlasSceneHealthGate,
    AtlasVLMScaleCues,
    AtlasApplyScaleReferences,
    AtlasFaceScaleReference,
    AtlasLoadRecord3D,
    AtlasSplitEquirect,
    AtlasEquirectMultiView,
    AtlasLoadSolveJSON,
    AtlasDecomposeSolve,
    AtlasDecomposeCamera,
    AtlasUSDCameraLoader,
)
from atlas_camera.comfy.nodes_depth import (
    AtlasDepthAnything,
    AtlasDepthMap,
    AtlasDepthOutlierMask,
    AtlasMogeNormals,
    AtlasDepthDetailEnhance,
    AtlasDepthCombine,
    AtlasDepthBandSplit,
    AtlasBoundedBand,
    AtlasDepthLayerMask,
    AtlasGroundDepthMap,
    AtlasGroundMask,
    AtlasHorizonMask,
    AtlasVPVisualization,
)
from atlas_camera.comfy.nodes_geometry import (
    AtlasDeriveProjectionGeometry,
    AtlasDeriveReliefMesh,
    AtlasLiveMeshRepair,
    AtlasRetopologizeLayer,
    AtlasPlanarHolePatch,
    AtlasPathGuidedHoleRepair,
    AtlasDeriveWalls,
    AtlasDeriveTowersSpires,
    AtlasDeriveRoofsFacades,
    AtlasDeriveInteriorRoom,
    AtlasMergeGeometry,
    AtlasDefineShotCam,
    AtlasPredictHiddenGeometry,
    AtlasRenderFix,
    AtlasExtractAnglePatch,
    AtlasImportAnglePatch,
    AtlasAddPatchView,
    AtlasOcclusionMask,
)
from atlas_camera.comfy.nodes_planar import (
    AtlasPlanarUnwarp,
    AtlasPlanarRewarp,
)
from atlas_camera.comfy.nodes_inpaint import (
    AtlasScopeMask,
    AtlasSemanticMask,
    AtlasSAM3Mask,
    AtlasInpaintCrop,
    AtlasInpaintStitch,
    AtlasSDXLInpaint,
    AtlasInstanceMask,
    AtlasSegmentedSDXLInpaint,
    AtlasCleanPlateLayer,
    AtlasCleanPlateStack,
    AtlasSkyDomeLayer,
)
from atlas_camera.comfy.nodes_export import (
    AtlasExportReviewPackage,
    AtlasExportSolveJSON,
    AtlasExportMayaReviewScene,
    AtlasExportReliefMesh,
    AtlasExportUSD,
    AtlasExportBlender,
    AtlasExportNuke,
    AtlasExportNukeLayers,
    AtlasExportMayaLayers,
    AtlasExportCameraPathUSD,
    AtlasExportPlateEXR,
)


# ---------------------------------------------------------------------------
# Node registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # Existing
    "AtlasExportReviewPackage":   AtlasExportReviewPackage,
    "AtlasExportSolveJSON":       AtlasExportSolveJSON,
    "AtlasExportMayaReviewScene": AtlasExportMayaReviewScene,
    "AtlasUSDCameraLoader":       AtlasUSDCameraLoader,
    "AtlasRegisterPlate":         AtlasRegisterPlate,
    "AtlasDeband":                AtlasDeband,
    "AtlasGrade":                 AtlasGrade,
    "AtlasDefocus":               AtlasDefocus,
    "AtlasApplyLUT":              AtlasApplyLUT,
    "AtlasLoadPlate":             AtlasLoadPlate,
    "AtlasAttachSourcePlate":     AtlasAttachSourcePlate,
    "AtlasLoadRAW":               AtlasLoadRAW,
    # Track 1 — solve
    "AtlasSolveFromImage":        AtlasSolveFromImage,
    "AtlasLearnedSolveFromImage": AtlasLearnedSolveFromImage,
    "AtlasScaleOverride":         AtlasScaleOverride,
    "AtlasRollTrim":              AtlasRollTrim,
    "AtlasReferenceScaleSolve":   AtlasReferenceScaleSolve,
    "AtlasVLMScaleCues":          AtlasVLMScaleCues,
    "AtlasAssessImage":           AtlasAssessImage,
    "AtlasAssessOutput":          AtlasAssessOutput,
    "AtlasSolveGate":             AtlasSolveGate,
    "AtlasSceneHealthGate":       AtlasSceneHealthGate,
    "AtlasGravityOverride":       AtlasGravityOverride,
    "AtlasApplyScaleReferences":  AtlasApplyScaleReferences,
    "AtlasDeriveProjectionGeometry": AtlasDeriveProjectionGeometry,
    "AtlasAddPatchView":          AtlasAddPatchView,
    "AtlasOcclusionMask":         AtlasOcclusionMask,
    "AtlasConstrainedSolve":      AtlasConstrainedSolve,
    "AtlasFaceScaleReference":    AtlasFaceScaleReference,
    "AtlasLoadRecord3D":          AtlasLoadRecord3D,
    "AtlasSplitEquirect":         AtlasSplitEquirect,
    "AtlasEquirectMultiView":     AtlasEquirectMultiView,
    "AtlasLoadSolveJSON":         AtlasLoadSolveJSON,
    # Track 1 — decompose
    "AtlasDecomposeSolve":        AtlasDecomposeSolve,
    "AtlasDecomposeCamera":       AtlasDecomposeCamera,
    # Track 1 — image generation
    "AtlasDepthAnything":         AtlasDepthAnything,
    "AtlasGroundDepthMap":        AtlasGroundDepthMap,
    "AtlasHorizonMask":           AtlasHorizonMask,
    "AtlasVPVisualization":       AtlasVPVisualization,
    # Track 1 — export
    "AtlasExportReliefMesh":      AtlasExportReliefMesh,
    "AtlasExportUSD":             AtlasExportUSD,
    "AtlasExportBlender":         AtlasExportBlender,
    "AtlasExportNuke":            AtlasExportNuke,
    "AtlasExportNukeLayers":      AtlasExportNukeLayers,
    "AtlasExportMayaLayers":      AtlasExportMayaLayers,
    # Track 2 — blockout viewport
    "AtlasViewportControls":      AtlasViewportControls,
    "AtlasBlockoutViewport":      AtlasBlockoutViewport,
    "AtlasStereoRender":          AtlasStereoRender,
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   AtlasExportCameraPathUSD,
    "AtlasExportPlateEXR":        AtlasExportPlateEXR,
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              AtlasDepthMap,
    "AtlasMogeNormals":           AtlasMogeNormals,
    "AtlasDepthDetailEnhance":    AtlasDepthDetailEnhance,
    "AtlasDepthCombine":          AtlasDepthCombine,
    "AtlasPlanarUnwarp":          AtlasPlanarUnwarp,
    "AtlasPlanarRewarp":          AtlasPlanarRewarp,
    # Experimental (research-only)
    "AtlasDeriveReliefMesh":      AtlasDeriveReliefMesh,
    "AtlasRetopologizeLayer":     AtlasRetopologizeLayer,
    "AtlasPlanarHolePatch":       AtlasPlanarHolePatch,
    "AtlasPathGuidedHoleRepair":  AtlasPathGuidedHoleRepair,
    "AtlasDeriveWalls":           AtlasDeriveWalls,
    "AtlasDeriveTowersSpires":    AtlasDeriveTowersSpires,
    "AtlasDeriveRoofsFacades":    AtlasDeriveRoofsFacades,
    "AtlasDeriveInteriorRoom":    AtlasDeriveInteriorRoom,
    "AtlasMergeGeometry":         AtlasMergeGeometry,
    # Track 6 — shot format
    "AtlasDefineShotCam":         AtlasDefineShotCam,
    # Track 7 — inpaint layers
    "AtlasDepthBandSplit":        AtlasDepthBandSplit,
    "AtlasBoundedBand":           AtlasBoundedBand,
    "AtlasDepthLayerMask":        AtlasDepthLayerMask,
    "AtlasCleanPlateLayer":       AtlasCleanPlateLayer,
    "AtlasCleanPlateStack":       AtlasCleanPlateStack,
    "AtlasSkyDomeLayer":          AtlasSkyDomeLayer,
    "AtlasInpaintCrop":           AtlasInpaintCrop,
    "AtlasInpaintStitch":         AtlasInpaintStitch,
    "AtlasSDXLInpaint":           AtlasSDXLInpaint,
    "AtlasInstanceMask":          AtlasInstanceMask,
    "AtlasSegmentedSDXLInpaint":  AtlasSegmentedSDXLInpaint,
    "AtlasDepthOutlierMask":      AtlasDepthOutlierMask,
    "AtlasScopeMask":             AtlasScopeMask,
    "AtlasSemanticMask":          AtlasSemanticMask,
    "AtlasSAM3Mask":              AtlasSAM3Mask,
    "AtlasDebugReport":           AtlasDebugReport,
    "AtlasLayerPreview":          AtlasLayerPreview,
    "AtlasInput":                 AtlasInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing
    "AtlasExportReviewPackage":   "Atlas Export Review Package",
    "AtlasExportSolveJSON":       "Atlas Export Solve JSON",
    "AtlasExportMayaReviewScene": "Atlas Export Maya Review Scene",
    "AtlasUSDCameraLoader":       "Atlas USD Camera Loader",
    "AtlasLoadPlate":             "Atlas Load Plate 🎞",
    "AtlasRegisterPlate":         "Atlas Register Plate (Float-Safe) 🎞",
    "AtlasDeband":                "Atlas Deband 🎚",
    "AtlasGrade":                 "Atlas Grade 🎨",
    "AtlasDefocus":               "Atlas Defocus (depth) 🌫",
    "AtlasApplyLUT":              "Atlas Apply LUT (.cube) 🌈",
    "AtlasAttachSourcePlate":     "Atlas Attach Source Plate 🎞",
    "AtlasLoadRAW":               "Atlas Load RAW (NEF/CR2/CR3/RAF/ARW) 📷",
    # Track 1 — solve
    "AtlasSolveFromImage":        "Atlas Solve Camera from Image",
    "AtlasLearnedSolveFromImage": "Atlas Learned Solve (GeoCalib) 🧠",
    "AtlasScaleOverride":         "Atlas Scale Override 📐",
    "AtlasRollTrim":              "Atlas Roll Trim 🎚",
    "AtlasReferenceScaleSolve":   "Atlas Reference-Object Scale 📏",
    "AtlasAssessImage":           "Atlas Assess Image 🧭",
    "AtlasAssessOutput":          "Atlas Assess Output 🧪",
    "AtlasSolveGate":             "Atlas Solve Gate ✅",
    "AtlasSceneHealthGate":       "Atlas Scene Health Gate 🩺",
    "AtlasGravityOverride":       "Atlas Gravity Override 🎚",
    "AtlasVLMScaleCues":          "Atlas VLM Scale Cues 👁",
    "AtlasApplyScaleReferences":  "Atlas Apply Scale References ✅",
    "AtlasDeriveProjectionGeometry": "Atlas Derive Projection Geometry 📽",
    "AtlasAddPatchView":          "Atlas Add Patch View (multi-angle) 🩹",
    "AtlasOcclusionMask":         "Atlas Occlusion Mask 🕳",
    "AtlasConstrainedSolve":      "Atlas Constrained Solve",
    "AtlasFaceScaleReference":    "Atlas Face Scale Reference 🙂",
    "AtlasLoadRecord3D":          "Atlas Load Record3D Capture 📱",
    "AtlasSplitEquirect":         "Atlas Split Equirect 🌐",
    "AtlasEquirectMultiView":     "Atlas Equirect Multi-View 🌐",
    "AtlasLoadSolveJSON":         "Atlas Load Solve JSON",
    # Track 1 — decompose
    "AtlasDecomposeSolve":        "Atlas Decompose Solve",
    "AtlasDecomposeCamera":       "Atlas Decompose Camera",
    # Track 1 — image generation
    "AtlasDepthAnything":         "Atlas Depth Anything V2 🧠",
    "AtlasGroundDepthMap":        "Atlas Ground Depth Map",
    "AtlasHorizonMask":           "Atlas Horizon / Sky Mask",
    "AtlasVPVisualization":       "Atlas VP Visualization",
    # Track 1 — export
    "AtlasExportReliefMesh":      "Atlas Export Relief Mesh (OBJ) 🗻",
    "AtlasExportUSD":             "Atlas Export USD",
    "AtlasExportBlender":         "Atlas Export Blender Scene",
    "AtlasExportNuke":            "Atlas Export Nuke Script",
    "AtlasExportNukeLayers":      "Atlas Export Nuke Layers 🎞",
    "AtlasExportMayaLayers":      "Atlas Export Maya Layers 🧊",
    # Track 2 — blockout viewport
    "AtlasViewportControls":      "Atlas Output Desk 🎛",
    "AtlasBlockoutViewport":      "Atlas Viewport 🧊",
    "AtlasStereoRender":          "Atlas Stereo Render 👓",
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   "Atlas Export Camera Path (USD) 🎥",
    "AtlasExportPlateEXR":        "Atlas Export ACEScg Plate 📤",
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              "Atlas Depth Map 🌊",
    "AtlasMogeNormals":           "Atlas MoGe Normals 🧭",
    "AtlasDepthDetailEnhance":    "Atlas Depth Detail Enhance 🔬",
    "AtlasDepthCombine":          "Atlas Depth Combine ➕",
    "AtlasPlanarUnwarp":          "Atlas Planar Unwarp ▱",
    "AtlasPlanarRewarp":          "Atlas Planar Rewarp ▱",
    "AtlasDeriveReliefMesh":      "Atlas Derive Relief Mesh 🏔",
    "AtlasRetopologizeLayer":     "Atlas Retopologize Layer 🔷",
    "AtlasPlanarHolePatch":       "Atlas Planar Hole Patch ◩",
    "AtlasPathGuidedHoleRepair":  "Atlas Path-Guided Hole Repair 🎥",
    "AtlasDeriveWalls":           "Atlas Derive Walls 🧱",
    "AtlasDeriveTowersSpires":    "Atlas Derive Towers & Spires 🗼",
    "AtlasDeriveRoofsFacades":    "Atlas Derive Roofs & Facades 🏛",
    "AtlasDeriveInteriorRoom":    "Atlas Derive Interior Room 🛋",
    "AtlasMergeGeometry":         "Atlas Merge Geometry 🔀",
    # Track 6 — shot format
    "AtlasDefineShotCam":         "Atlas Define Shot Cam 🎬",
    # Track 7 — inpaint layers
    "AtlasDepthBandSplit":        "Atlas Depth Band Split 🎚",
    "AtlasBoundedBand":           "Atlas Bounded Band 📏",
    "AtlasDepthLayerMask":        "Atlas Depth Layer Mask 🎭",
    "AtlasCleanPlateLayer":       "Atlas Clean Plate Layer 🖼",
    "AtlasCleanPlateStack":       "Atlas Clean Plate Stack 🧽 (up to 4 plates + alphas)",
    "AtlasSkyDomeLayer":          "Atlas Sky Dome Layer ☁",
    "AtlasInpaintCrop":           "Atlas Inpaint Crop ✂",
    "AtlasInpaintStitch":         "Atlas Inpaint Stitch ✂",
    "AtlasSDXLInpaint":           "Atlas SDXL Inpaint (native) ✨",
    "AtlasInstanceMask":          "Atlas Instance Mask (SAM3) 🎭",
    "AtlasSegmentedSDXLInpaint":  "Atlas Segmented SDXL Inpaint 🏢",
    "AtlasDepthOutlierMask":      "Atlas Depth Outlier Mask 🛡",
    "AtlasScopeMask":             "Atlas Scope Mask 🎯",
    "AtlasSemanticMask":          "Atlas Semantic Mask 🧩",
    "AtlasSAM3Mask":              "Atlas SAM3 Mask 🪄",
    "AtlasDebugReport":           "Atlas Debug Report 🔍",
    "AtlasLayerPreview":          "Atlas Layer Preview 🎨",
    "AtlasInput":                 "Atlas Input 🎬",
}

# ---------------------------------------------------------------------------
# Experimental tier (🔬) — heavier external requirements than the core node
# set (user-cloned upstream repos, Docker, CUDA-class GPUs). Registered only
# when the ATLAS_EXPERIMENTAL env var is truthy, so the standard install's
# node menu stays universal and nothing here can confuse a stock ComfyUI.
# Set ATLAS_EXPERIMENTAL=1 (or 0) before launching ComfyUI to override this
# default. A long-lived `experimental` branch used to carry the flipped
# constant instead; it was retired 2026-07-28 because the env var does the
# same job on any branch, while the branch re-conflicted on every edit near
# this line, went stale on every push to main, and could never be green (5
# tests pin the default-closed contract it inverted). Do not recreate it.
NODE_CLASS_MAPPINGS["AtlasOcclusionGraph"] = AtlasOcclusionGraph
NODE_CLASS_MAPPINGS["AtlasMoveBudget"] = AtlasMoveBudget
NODE_CLASS_MAPPINGS["AtlasLayerPlan"] = AtlasLayerPlan
NODE_DISPLAY_NAME_MAPPINGS["AtlasOcclusionGraph"] = "Atlas Occlusion Graph 🕸"
NODE_DISPLAY_NAME_MAPPINGS["AtlasMoveBudget"] = "Atlas Move Budget 📐"
NODE_DISPLAY_NAME_MAPPINGS["AtlasLayerPlan"] = "Atlas Layer Plan 🥞"


ATLAS_EXPERIMENTAL_DEFAULT = "0"

EXPERIMENTAL_NODE_CLASS_MAPPINGS = {
    "AtlasCompleteDepth": AtlasCompleteDepth,
    "AtlasPredictHiddenGeometry": AtlasPredictHiddenGeometry,
    "AtlasRenderFix": AtlasRenderFix,
    "AtlasExtractAnglePatch": AtlasExtractAnglePatch,
    "AtlasImportAnglePatch": AtlasImportAnglePatch,
}

EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasCompleteDepth": "Atlas Complete Depth 🩹 🔬 (experimental)",
    "AtlasPredictHiddenGeometry": "Atlas Predict Hidden Geometry 🔬 (research)",
    "AtlasRenderFix": "Atlas Render Fix 🔬 (experimental)",
    "AtlasExtractAnglePatch": "Atlas Extract Angle Patch 🔬 → Photoshop",
    "AtlasImportAnglePatch": "Atlas Import Angle Patch 🔬 ← Photoshop",
}


def _experimental_enabled() -> bool:
    v = os.environ.get("ATLAS_EXPERIMENTAL", ATLAS_EXPERIMENTAL_DEFAULT)
    return v.strip().lower() not in ("", "0", "false", "off", "no")


if _experimental_enabled():
    NODE_CLASS_MAPPINGS.update(EXPERIMENTAL_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS)

# ---------------------------------------------------------------------------
# Legacy tier (superseded) — a supported replacement exists and the shipping
# workflows have been migrated to it, but saved user graphs must keep
# resolving for one migration cycle. Registered only when ATLAS_LEGACY_NODES
# is truthy, so the default node menu offers ONE obvious way to do each job.
# Same shape as the experimental gate above; the two are independent and a
# node is never in both. Keys and display names stay byte-identical — the
# append-only contract covers renames too, so the deprecation is carried by
# the class docstring, the banner below, the node's own report, and
# docs/FEATURE_AUDIT.md rather than by relabelling.
ATLAS_LEGACY_DEFAULT = "0"

LEGACY_NODE_CLASS_MAPPINGS = {
    "AtlasLiveMeshRepair": AtlasLiveMeshRepair,
    "AtlasGroundMask": AtlasGroundMask,
}

LEGACY_NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasLiveMeshRepair": "Atlas Live Mesh Repair 🔧",
    "AtlasGroundMask": "Atlas Ground Mask",
}

#: key -> the supported replacement, surfaced in the banner and the audit.
LEGACY_REPLACEMENTS = {
    "AtlasLiveMeshRepair":
        "AtlasPlanarHolePatch (layer='*') -> AtlasRetopologizeLayer"
        "(boundary_smooth_iterations)",
    "AtlasGroundMask":
        "AtlasGroundDepthMap, output 1 (ground_mask) — bit-identical",
}


def _legacy_enabled() -> bool:
    v = os.environ.get("ATLAS_LEGACY_NODES", ATLAS_LEGACY_DEFAULT)
    return v.strip().lower() not in ("", "0", "false", "off", "no")


if _legacy_enabled():
    NODE_CLASS_MAPPINGS.update(LEGACY_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LEGACY_NODE_DISPLAY_NAME_MAPPINGS)
    print("[Atlas Camera] ATLAS_LEGACY_NODES=1 — registering superseded nodes: "
          + "; ".join(f"{k} (use: {v})" for k, v in LEGACY_REPLACEMENTS.items()))
