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
    AtlasDebugReport,
    AtlasLayerPreview,
    AtlasInput,
)
from atlas_camera.comfy.nodes_solve import (
    AtlasLoadImageSolveCamera,
    AtlasRegisterPlate,
    AtlasAttachSourcePlate,
    AtlasLoadRAW,
    AtlasSolveFromImage,
    AtlasConstrainedSolve,
    AtlasLearnedSolveFromImage,
    AtlasScaleOverride,
    AtlasRollTrim,
    AtlasGravityOverride,
    AtlasPitchTrim,
    AtlasReferenceScaleSolve,
    AtlasAssessImage,
    AtlasSolveGate,
    AtlasSceneHealthGate,
    AtlasVLMScaleCues,
    AtlasApplyScaleReferences,
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
from atlas_camera.comfy.nodes_inpaint import (
    AtlasScopeMask,
    AtlasSemanticMask,
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
)


# ---------------------------------------------------------------------------
# Node registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # Existing
    "AtlasLoadImageSolveCamera":  AtlasLoadImageSolveCamera,
    "AtlasExportReviewPackage":   AtlasExportReviewPackage,
    "AtlasExportSolveJSON":       AtlasExportSolveJSON,
    "AtlasExportMayaReviewScene": AtlasExportMayaReviewScene,
    "AtlasUSDCameraLoader":       AtlasUSDCameraLoader,
    "AtlasRegisterPlate":         AtlasRegisterPlate,
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
    "AtlasSolveGate":             AtlasSolveGate,
    "AtlasSceneHealthGate":       AtlasSceneHealthGate,
    "AtlasPitchTrim":             AtlasPitchTrim,
    "AtlasGravityOverride":       AtlasGravityOverride,
    "AtlasApplyScaleReferences":  AtlasApplyScaleReferences,
    "AtlasDeriveProjectionGeometry": AtlasDeriveProjectionGeometry,
    "AtlasAddPatchView":          AtlasAddPatchView,
    "AtlasOcclusionMask":         AtlasOcclusionMask,
    "AtlasConstrainedSolve":      AtlasConstrainedSolve,
    "AtlasLoadSolveJSON":         AtlasLoadSolveJSON,
    # Track 1 — decompose
    "AtlasDecomposeSolve":        AtlasDecomposeSolve,
    "AtlasDecomposeCamera":       AtlasDecomposeCamera,
    # Track 1 — image generation
    "AtlasDepthAnything":         AtlasDepthAnything,
    "AtlasGroundDepthMap":        AtlasGroundDepthMap,
    "AtlasGroundMask":            AtlasGroundMask,
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
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   AtlasExportCameraPathUSD,
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              AtlasDepthMap,
    "AtlasMogeNormals":           AtlasMogeNormals,
    # Experimental (research-only)
    "AtlasDeriveReliefMesh":      AtlasDeriveReliefMesh,
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
    "AtlasDebugReport":           AtlasDebugReport,
    "AtlasLayerPreview":          AtlasLayerPreview,
    "AtlasInput":                 AtlasInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing
    "AtlasLoadImageSolveCamera":  "Atlas Load Image / Solve Camera (Deprecated)",
    "AtlasExportReviewPackage":   "Atlas Export Review Package",
    "AtlasExportSolveJSON":       "Atlas Export Solve JSON",
    "AtlasExportMayaReviewScene": "Atlas Export Maya Review Scene",
    "AtlasUSDCameraLoader":       "Atlas USD Camera Loader",
    "AtlasRegisterPlate":         "Atlas Register Plate (Float-Safe) 🎞",
    "AtlasAttachSourcePlate":     "Atlas Attach Source Plate 🎞",
    "AtlasLoadRAW":               "Atlas Load RAW (NEF/CR2/CR3/RAF/ARW) 📷",
    # Track 1 — solve
    "AtlasSolveFromImage":        "Atlas Solve Camera from Image",
    "AtlasLearnedSolveFromImage": "Atlas Learned Solve (GeoCalib) 🧠",
    "AtlasScaleOverride":         "Atlas Scale Override 📐",
    "AtlasRollTrim":              "Atlas Roll Trim 🎚",
    "AtlasReferenceScaleSolve":   "Atlas Reference-Object Scale 📏",
    "AtlasAssessImage":           "Atlas Assess Image 🧭",
    "AtlasSolveGate":             "Atlas Solve Gate ✅",
    "AtlasSceneHealthGate":       "Atlas Scene Health Gate 🩺",
    "AtlasPitchTrim":             "Atlas Pitch Trim 🎚",
    "AtlasGravityOverride":       "Atlas Gravity Override 🎚",
    "AtlasVLMScaleCues":          "Atlas VLM Scale Cues 👁",
    "AtlasApplyScaleReferences":  "Atlas Apply Scale References ✅",
    "AtlasDeriveProjectionGeometry": "Atlas Derive Projection Geometry 📽",
    "AtlasAddPatchView":          "Atlas Add Patch View (multi-angle) 🩹",
    "AtlasOcclusionMask":         "Atlas Occlusion Mask 🕳",
    "AtlasConstrainedSolve":      "Atlas Constrained Solve",
    "AtlasLoadSolveJSON":         "Atlas Load Solve JSON",
    # Track 1 — decompose
    "AtlasDecomposeSolve":        "Atlas Decompose Solve",
    "AtlasDecomposeCamera":       "Atlas Decompose Camera",
    # Track 1 — image generation
    "AtlasDepthAnything":         "Atlas Depth Anything V2 🧠",
    "AtlasGroundDepthMap":        "Atlas Ground Depth Map",
    "AtlasGroundMask":            "Atlas Ground Mask",
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
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   "Atlas Export Camera Path (USD) 🎥",
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              "Atlas Depth Map 🌊",
    "AtlasMogeNormals":           "Atlas MoGe Normals 🧭",
    "AtlasDeriveReliefMesh":      "Atlas Derive Relief Mesh 🏔",
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
    "AtlasDebugReport":           "Atlas Debug Report 🔍",
    "AtlasLayerPreview":          "Atlas Layer Preview 🎨",
    "AtlasInput":                 "Atlas Input 🎬",
}

# ---------------------------------------------------------------------------
# Experimental tier (🔬) — heavier external requirements than the core node
# set (user-cloned upstream repos, Docker, CUDA-class GPUs). Registered only
# when the ATLAS_EXPERIMENTAL env var is truthy, so the standard install's
# node menu stays universal and nothing here can confuse a stock ComfyUI.
# The `experimental` branch ships ATLAS_EXPERIMENTAL_DEFAULT = "1" (that one
# line is the entire branch delta); on any branch, setting
# ATLAS_EXPERIMENTAL=1 (or 0) before launching ComfyUI overrides the default.
ATLAS_EXPERIMENTAL_DEFAULT = "1"

EXPERIMENTAL_NODE_CLASS_MAPPINGS = {
    "AtlasPredictHiddenGeometry": AtlasPredictHiddenGeometry,
    "AtlasRenderFix": AtlasRenderFix,
    "AtlasExtractAnglePatch": AtlasExtractAnglePatch,
    "AtlasImportAnglePatch": AtlasImportAnglePatch,
}

EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS = {
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
