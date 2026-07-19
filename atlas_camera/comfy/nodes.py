"""ComfyUI node library for Atlas Camera."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.solver import solve_from_constraints, solve_still_image
from atlas_camera.exporters.blender_exporter import write_blender_scene_script
from atlas_camera.exporters.nuke_exporter import write_nuke_native_script, write_nuke_projection_script
from atlas_camera.exporters.review_package import build_review_package
from atlas_camera.importers.usd_camera_loader import USDCameraLoader

from atlas_camera.comfy.node_helpers import (
    _DEPTH_MODEL_CHOICES,
    _MOGE_NORMAL_MODEL_CHOICES,
    _ATLAS_BLOCKOUT_CACHE,
    _ATLAS_BLOCKOUT_CACHE_MAX,
    _blockout_cache_set,
    _solve_focal_px_for_image,
    _require_numpy,
    _require_torch,
    _require_pil,
    _image_tensor_to_pil,
    _pil_to_image_tensor,
    _save_image_tensor_to_tmp,
    _resolve_raw_hints,
    _scale_summary_suffix,
    _IDENTITY_COMMENT_PREFIX,
    _write_export_manifest,
    _health_summary_suffix,
    _stamp_raw_provenance,
    _extend_edge_colors,
    _b64_png_to_mask,
    _mask_to_b64_png,
    _image_tensor_to_preview_b64,
    _plate_ref_to_dict,
    _output_profile_to_dict,
    _clone_solve_with_metadata,
    _decode_b64_to_tensor,
    _image_fingerprint,
    _solve_fingerprint,
    _execution_blocker,
    _extract_blockout_camera,
    _ground_depth_compute,
    _reference_id_choices,
    _extrinsics_from_view,
    _recompute_horizon_line,
    _ATLAS_ASSESS_CACHE,
    _solve_camera_params,
    _horizon_y_from_solve,
    _depth_map_for_solve,
    _replace_proxy_role_geometry,
    _MetricDepthSetup,
    _BORDER_FLOOD_PX,
    _flood_mask_to_frame_borders,
    _resolve_exclude_mask,
    _GROUND_SCALE_CACHE,
    _ground_scale_cached,
    _metric_depth_and_validity,
    _resolve_depth_band,
    _parse_band_override,
    _band_resolution_validity,
    _resize_normal_field,
    _AZIMUTH_VIEWS,
    _ELEVATION_VIEWS,
    _DISTANCE_VIEWS,
    _parse_view_prompt,
    _parse_exact_view,
    _named_view_orbit_delta,
    _format_hole_fill_report,
    _solve_with_relief_mesh,
    _relief_mesh_from_solve,
    _solve_image_size,
    _fit_long_edge,
    _apply_band_split,
    _BOUNDED_BAND_NOOP_M,
    _LAYER_DEBUG_PRIMARY_HEX,
    _LAYER_DEBUG_PALETTE_HEX,
    _comfy_registry,
    _MiniGraphBuilder,
    _graph_builder,
    _ATLAS_INPUT_BOUNDARIES,
    _ATLAS_INPUT_BAND_NAMES,
    _seg_coverage,
    _BAND_GEOMETRY_CHOICES,
    _resolve_band_geometry,
    _analytic_ground_forward_depth,
)
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
ATLAS_EXPERIMENTAL_DEFAULT = "0"

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
