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
    AtlasDisocclusionGuide,
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
    AtlasShootList,
)
from atlas_camera.comfy.nodes_qa import AtlasAssessOutput
from atlas_camera.comfy.nodes_dynamic import AtlasLoadDynamicPlate
from atlas_camera.comfy.nodes_fill import (AtlasInterpassGate,
                                           AtlasMembraneComposite)
from atlas_camera.comfy.nodes_project import AtlasProject
from atlas_camera.comfy.nodes_multiview import (
    AtlasMultiViewSolve,
    AtlasMultiViewSolveBurst,
    AtlasSolveBurstPatchCrops,
)
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
    AtlasGravityCompass,
    AtlasReferenceScaleSolve,
    AtlasAssessImage,
    AtlasSolveGate,
    AtlasSceneHealthGate,
    AtlasVLMScaleCues,
    AtlasApplyScaleReferences,
    AtlasFaceScaleReference,
    AtlasLoadRecord3D,
    AtlasStreamRecord3D,
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
    AtlasOutpaintDepth,
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
    AtlasMaskedSurfaceReconstruct,
    AtlasRefineOcclusionSeams,
    AtlasPathGuidedHoleRepair,
    AtlasDeriveWalls,
    AtlasDeriveTowersSpires,
    AtlasDeriveRoofsFacades,
    AtlasDeriveInteriorRoom,
    AtlasMergeGeometry,
    AtlasDefineShotCam,
    AtlasBlockoutMassing,
    AtlasExtractAnglePatch,
    AtlasImportAnglePatch,
    AtlasSolvePatchViews,
    AtlasAddPatchView,
    AtlasOcclusionMask,
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
    "AtlasGravityCompass":        AtlasGravityCompass,
    "AtlasApplyScaleReferences":  AtlasApplyScaleReferences,
    "AtlasDeriveProjectionGeometry": AtlasDeriveProjectionGeometry,
    "AtlasSolvePatchViews":       AtlasSolvePatchViews,
    "AtlasAddPatchView":          AtlasAddPatchView,
    "AtlasOcclusionMask":         AtlasOcclusionMask,
    "AtlasConstrainedSolve":      AtlasConstrainedSolve,
    "AtlasFaceScaleReference":    AtlasFaceScaleReference,
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
    "AtlasDisocclusionGuide":     AtlasDisocclusionGuide,
    "AtlasStereoRender":          AtlasStereoRender,
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   AtlasExportCameraPathUSD,
    "AtlasExportPlateEXR":        AtlasExportPlateEXR,
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              AtlasDepthMap,
    "AtlasOutpaintDepth":          AtlasOutpaintDepth,
    "AtlasMogeNormals":           AtlasMogeNormals,
    "AtlasDepthDetailEnhance":    AtlasDepthDetailEnhance,
    "AtlasDepthCombine":          AtlasDepthCombine,
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
    "AtlasProject":               AtlasProject,
    "AtlasMultiViewSolve":        AtlasMultiViewSolve,
    "AtlasMultiViewSolveBurst":   AtlasMultiViewSolveBurst,
    "AtlasSolveBurstPatchCrops":  AtlasSolveBurstPatchCrops,
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
    "AtlasGravityCompass":        "Atlas Gravity Compass 🧭",
    "AtlasVLMScaleCues":          "Atlas VLM Scale Cues 👁",
    "AtlasApplyScaleReferences":  "Atlas Apply Scale References ✅",
    "AtlasDeriveProjectionGeometry": "Atlas Derive Projection Geometry 📽",
    "AtlasSolvePatchViews":       "Atlas Solve Patch Views ⌖",
    "AtlasAddPatchView":          "Atlas Add Patch View (multi-angle) 🩹",
    "AtlasOcclusionMask":         "Atlas Occlusion Mask 🕳",
    "AtlasConstrainedSolve":      "Atlas Constrained Solve",
    "AtlasFaceScaleReference":    "Atlas Face Scale Reference 🙂",
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
    "AtlasDisocclusionGuide":     "Atlas Disocclusion Guide 🟣",
    "AtlasStereoRender":          "Atlas Stereo Render 👓",
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   "Atlas Export Camera Path (USD) 🎥",
    "AtlasExportPlateEXR":        "Atlas Export ACEScg Plate 📤",
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              "Atlas Depth Map 🌊",
    "AtlasOutpaintDepth":  "Atlas Outpaint Depth 🪟",
    "AtlasMogeNormals":           "Atlas MoGe Normals 🧭",
    "AtlasDepthDetailEnhance":    "Atlas Depth Detail Enhance 🔬",
    "AtlasDepthCombine":          "Atlas Depth Combine ➕",
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
    "AtlasProject":               "Atlas Project 🎬",
    "AtlasMultiViewSolve":        "Atlas Multi-View RAW Solve 📷📷",
    "AtlasMultiViewSolveBurst":   "Atlas Multi-View Burst Solve 📷🎞️",
    "AtlasSolveBurstPatchCrops":  "Atlas Solve Burst Patch Crops 📷✂️",
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
NODE_CLASS_MAPPINGS["AtlasShootList"] = AtlasShootList
NODE_CLASS_MAPPINGS["AtlasMoveBudget"] = AtlasMoveBudget
NODE_CLASS_MAPPINGS["AtlasLayerPlan"] = AtlasLayerPlan
NODE_DISPLAY_NAME_MAPPINGS["AtlasOcclusionGraph"] = "Atlas Occlusion Graph 🕸"
NODE_DISPLAY_NAME_MAPPINGS["AtlasShootList"] = "Atlas Shoot List 📸"
NODE_DISPLAY_NAME_MAPPINGS["AtlasMoveBudget"] = "Atlas Move Budget 📐"
NODE_DISPLAY_NAME_MAPPINGS["AtlasLayerPlan"] = "Atlas Layer Plan 🥞"


ATLAS_EXPERIMENTAL_DEFAULT = "0"

EXPERIMENTAL_NODE_CLASS_MAPPINGS = {
    "AtlasCompleteDepth": AtlasCompleteDepth,
    "AtlasMaskedSurfaceReconstruct": AtlasMaskedSurfaceReconstruct,
    "AtlasRefineOcclusionSeams": AtlasRefineOcclusionSeams,
    "AtlasBlockoutMassing": AtlasBlockoutMassing,
    "AtlasExtractAnglePatch": AtlasExtractAnglePatch,
    "AtlasImportAnglePatch": AtlasImportAnglePatch,
    "AtlasLoadDynamicPlate": AtlasLoadDynamicPlate,
    "AtlasInterpassGate": AtlasInterpassGate,
    "AtlasMembraneComposite": AtlasMembraneComposite,
}

EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasCompleteDepth": "Atlas Complete Depth 🩹 🔬 (experimental)",
    "AtlasMaskedSurfaceReconstruct": "Atlas Masked Surface Reconstruct 🔬 (NumPy)",
    "AtlasRefineOcclusionSeams": "Atlas Refine Occlusion Seams 🔬 (NumPy underlap)",
    "AtlasBlockoutMassing": "Atlas Blockout Massing 🧱🔬 (experimental)",
    "AtlasExtractAnglePatch": "Atlas Extract Angle Patch 🔬 → Photoshop",
    "AtlasImportAnglePatch": "Atlas Import Angle Patch 🔬 ← Photoshop",
    "AtlasLoadDynamicPlate": "Atlas Load Dynamic Plate 🌊🔬 (experimental)",
    "AtlasInterpassGate": "Atlas Interpass Gate 🚦🔬 (experimental)",
    "AtlasMembraneComposite": "Atlas Membrane Composite 🩹🔬 (experimental)",
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


# ---------------------------------------------------------------------------
# iOS / phone-capture tier (📱) — Record3D file import + live USB stream from an
# iPhone/iPad. Held out of the v1 DEFAULT node menu because the iOS capture app
# is not part of the v1 public release; the nodes and their [record3d] /
# [record3d-stream] extras stay in the tree so an opted-in user (or a later
# release) can turn them on. Independent of the experimental/legacy gates — a
# node is never in more than one tier. Keys and display names stay byte-identical
# to when these lived in the base tier, so a saved graph that used them still
# resolves once the flag is set (the append-only contract).
ATLAS_IOS_DEFAULT = "0"

IOS_NODE_CLASS_MAPPINGS = {
    "AtlasLoadRecord3D": AtlasLoadRecord3D,
    "AtlasStreamRecord3D": AtlasStreamRecord3D,
}

IOS_NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasLoadRecord3D": "Atlas Load Record3D Capture 📱",
    "AtlasStreamRecord3D": "Atlas Record3D Live Stream 📲",
}


def _ios_enabled() -> bool:
    v = os.environ.get("ATLAS_IOS", ATLAS_IOS_DEFAULT)
    return v.strip().lower() not in ("", "0", "false", "off", "no")


if _ios_enabled():
    NODE_CLASS_MAPPINGS.update(IOS_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(IOS_NODE_DISPLAY_NAME_MAPPINGS)
    print("[Atlas Camera] ATLAS_IOS=1 — registering iOS/phone-capture nodes: "
          + ", ".join(IOS_NODE_CLASS_MAPPINGS))


# ---------------------------------------------------------------------------
# Menu taxonomy — the ONE place that decides which folder each node sits in.
# ComfyUI builds the Add-Node menu from each class's CATEGORY, splitting on "/".
# Setting it here (instead of per-class) keeps the whole menu visible and
# editable in one map. Zero-padded numeric prefixes (01..10) force pipeline
# order, because ComfyUI sorts folders as strings; "advanced" sorts after the
# digits and holds every gated tier (experimental + legacy + iOS). CATEGORY is
# metadata only — never serialized into a saved graph — so re-foldering cannot
# break an existing workflow; that is what makes this the one restructure the
# append-only key/display-name contract permits.
_MENU_FOLDERS = {
    "Atlas/01 \u00b7 Input & Camera": (
        "AtlasProject",
        "AtlasInput", "AtlasLoadPlate", "AtlasLoadRAW", "AtlasRegisterPlate",
        "AtlasMultiViewSolve", "AtlasMultiViewSolveBurst",
        "AtlasAttachSourcePlate", "AtlasSolveFromImage",
        "AtlasLearnedSolveFromImage", "AtlasConstrainedSolve", "AtlasSplitEquirect",
        "AtlasEquirectMultiView", "AtlasUSDCameraLoader", "AtlasLoadSolveJSON",
        "AtlasDecomposeCamera", "AtlasDecomposeSolve",
    ),
    "Atlas/02 \u00b7 Orient & Scale": (
        "AtlasGravityCompass", "AtlasGravityOverride", "AtlasRollTrim",
        "AtlasScaleOverride", "AtlasReferenceScaleSolve", "AtlasFaceScaleReference",
        "AtlasVLMScaleCues", "AtlasApplyScaleReferences",
    ),
    "Atlas/03 \u00b7 Depth": (
        "AtlasDepthAnything", "AtlasDepthMap", "AtlasMogeNormals",
        "AtlasDepthDetailEnhance", "AtlasDepthCombine", "AtlasGroundDepthMap",
        "AtlasDepthBandSplit", "AtlasBoundedBand", "AtlasDepthLayerMask",
        "AtlasDepthOutlierMask", "AtlasOutpaintDepth",
    ),
    "Atlas/04 \u00b7 Masks": (
        "AtlasHorizonMask", "AtlasSemanticMask", "AtlasInstanceMask",
        "AtlasSAM3Mask", "AtlasScopeMask", "AtlasOcclusionMask",
    ),
    "Atlas/05 \u00b7 Geometry": (
        "AtlasDeriveProjectionGeometry", "AtlasDeriveReliefMesh", "AtlasDeriveWalls",
        "AtlasDeriveTowersSpires", "AtlasDeriveRoofsFacades",
        "AtlasDeriveInteriorRoom", "AtlasMergeGeometry", "AtlasRetopologizeLayer",
        "AtlasDefineShotCam",
    ),
    "Atlas/06 \u00b7 Patch & Repair": (
        "AtlasAddPatchView", "AtlasSolvePatchViews", "AtlasPlanarHolePatch",
        "AtlasPathGuidedHoleRepair", "AtlasOcclusionGraph", "AtlasLayerPlan",
        "AtlasShootList", "AtlasDisocclusionGuide", "AtlasSolveBurstPatchCrops",
    ),
    "Atlas/07 \u00b7 Clean Plate & Inpaint": (
        "AtlasCleanPlateLayer", "AtlasCleanPlateStack", "AtlasLayerPreview",
        "AtlasSkyDomeLayer", "AtlasInpaintCrop", "AtlasInpaintStitch",
        "AtlasSDXLInpaint", "AtlasSegmentedSDXLInpaint",
    ),
    "Atlas/08 \u00b7 Look & Render": (
        "AtlasBlockoutViewport", "AtlasViewportControls", "AtlasVPVisualization",
        "AtlasStereoRender", "AtlasMoveBudget", "AtlasDebugReport", "AtlasGrade",
        "AtlasDeband", "AtlasDefocus", "AtlasApplyLUT",
    ),
    "Atlas/09 \u00b7 QA & Gates": (
        "AtlasAssessImage", "AtlasAssessOutput", "AtlasSceneHealthGate",
        "AtlasSolveGate",
    ),
    "Atlas/10 \u00b7 Export": (
        "AtlasExportNuke", "AtlasExportNukeLayers", "AtlasExportMayaLayers",
        "AtlasExportMayaReviewScene", "AtlasExportBlender", "AtlasExportUSD",
        "AtlasExportCameraPathUSD", "AtlasExportReliefMesh", "AtlasExportPlateEXR",
        "AtlasExportReviewPackage", "AtlasExportSolveJSON",
    ),
    # Every gated tier lands here: experimental, legacy, and iOS. They only
    # appear in the menu when their flag registers them, but carry the folder
    # regardless so they land in the right place when enabled.
    "Atlas/advanced": (
        "AtlasCompleteDepth", "AtlasBlockoutMassing", "AtlasExtractAnglePatch",
        "AtlasImportAnglePatch", "AtlasMaskedSurfaceReconstruct",
        "AtlasRefineOcclusionSeams", "AtlasLiveMeshRepair", "AtlasGroundMask",
        "AtlasLoadRecord3D", "AtlasStreamRecord3D", "AtlasLoadDynamicPlate",
        "AtlasInterpassGate", "AtlasMembraneComposite",
    ),
}

#: node key -> menu folder (CATEGORY). Derived from the readable folder map above.
MENU_CATEGORY = {key: folder
                 for folder, keys in _MENU_FOLDERS.items() for key in keys}


def _apply_menu_categories() -> None:
    """Stamp each registered class's CATEGORY from MENU_CATEGORY, across all tiers.

    Runs at import so ComfyUI reads the folder off the class when it builds the
    menu. CATEGORY is metadata only (never serialized), so this only moves where
    a node shows up, never what a saved graph resolves to.
    """
    for _m in (NODE_CLASS_MAPPINGS, EXPERIMENTAL_NODE_CLASS_MAPPINGS,
               LEGACY_NODE_CLASS_MAPPINGS, IOS_NODE_CLASS_MAPPINGS):
        for _key, _cls in _m.items():
            _cat = MENU_CATEGORY.get(_key)
            if _cat:
                _cls.CATEGORY = _cat


_apply_menu_categories()


def registry_surface_hash() -> str:
    """Short stable digest of the FULL registered node surface (standard +
    experimental + legacy + iOS keys, sorted). Stamped into comfy-layer
    artifacts (debug report, output assessment) so a consumer can tell which
    registry surface produced them without interpreting Git history. Keys
    only — display names and categories are presentation, and gating env vars
    must not change the hash between two installs of the same build.
    """
    import hashlib

    keys = sorted({**NODE_CLASS_MAPPINGS, **EXPERIMENTAL_NODE_CLASS_MAPPINGS,
                   **LEGACY_NODE_CLASS_MAPPINGS, **IOS_NODE_CLASS_MAPPINGS})
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
