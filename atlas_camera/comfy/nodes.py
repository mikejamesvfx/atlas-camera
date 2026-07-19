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

class AtlasExportReviewPackage:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            }
        }

    def export(self, solve, output_dir):
        result = build_review_package(solve, output_dir)
        return (str(result.package_dir),)


class AtlasExportSolveJSON:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_path": ("STRING", {"default": "atlas_solve.json"}),
            }
        }

    def export(self, solve, output_path):
        dest = str(save_solve_json(solve, output_path))
        _write_export_manifest(solve, Path(dest).parent or Path("."),
                               [("solve_json", dest)], "AtlasExportSolveJSON")
        return (dest,)


class AtlasExportMayaReviewScene:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            },
            "optional": {
                "relief_mesh_obj_path": ("STRING", {"default": "",
                    "tooltip": "Optional obj_path output from AtlasExportReliefMesh. When set, the "
                               "relief mesh is imported into the Maya scene instead of being omitted — "
                               "wire AtlasExportReliefMesh's obj_path here to see real derived geometry "
                               "(not just the camera) when opening the scene."}),
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata to embed in the review package."}),
            },
        }

    def export(self, solve, output_dir, relief_mesh_obj_path="", output_profile=None):
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        result = build_review_package(
            solve, output_dir, include_usd=False,
            relief_mesh_obj_path=relief_mesh_obj_path or None,
        )
        return (str(result.files["maya_open_scene"]),)


class AtlasExportReliefMesh:
    """Export a depth relief mesh (OBJ + MTL + texture) for Maya / Nuke / ZBrush.

    Triangulates the metric depth map into a world-space mesh, torn at depth
    silhouettes, with the recovered-camera projection baked into per-vertex UVs —
    the mesh imports already textured with the source photo, ready to retopo /
    reproject. OBJ/MTL references a file-backed source plate when the solve has
    one; otherwise it writes a PNG preview texture. GLB remains a preview/proxy
    payload with embedded PNG texture. Ground lands on Y=0 (scale reconciled to
    the solve's camera height). Requires the [neural] extra.
    """
    RETURN_TYPES = ("STRING", "STRING", "ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("obj_path", "glb_path", "preview_solve", "report")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            },
            "optional": {
                "grid_long_edge": ("INT", {"default": 128, "min": 16, "max": 4096,
                    "tooltip": "Mesh density: grid columns along the longest image edge."}),
                "depth_edge_rel": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Relative depth jump that tears the mesh (silhouette holes)."}),
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "format": (["both", "obj", "glb"], {"default": "both"}),
                "use_solve_mesh": ("BOOLEAN", {"default": True,
                    "tooltip": "Export the relief mesh ALREADY on the solve (from "
                               "AtlasDeriveReliefMesh / AtlasInput) so ALL its edge tuning — "
                               "max_edge_factor, normal_edge_deg, the band near-clip, sky_heuristic "
                               "— carries into the OBJ/GLB exactly, with no widget to re-set. Turn "
                               "OFF to re-derive from depth at this node's own grid/thresholds "
                               "below (e.g. to export a HIGHER-resolution mesh than the viewport). "
                               "Auto-falls-back to re-derive when the solve carries no relief mesh."}),
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Re-derive only (use_solve_mesh off): world-space edge tear "
                               "threshold. Raise to 40-80 on deep/interior scenes to stop combs."}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Re-derive only: 0 = off; tears on surface-normal bend (real creases) "
                               "while leaving smooth grazing surfaces intact."}),
                "fill_interior_holes": ("BOOLEAN", {"default": False,
                    "tooltip": "EXPORT-ONLY (the live viewport projection mesh is never touched). "
                               "Fan-fill small interior tear holes in the OBJ/GLB so it retopologizes "
                               "and booleans cleanly in a DCC. Fills ONLY interior enclosed boundary "
                               "loops — never the outer silhouette/frame boundary — by re-using each "
                               "hole's existing boundary vertices, so projection-baked UVs stay valid. "
                               "Off by default: a torn silhouette is the DMP-correct look."}),
                "max_hole_edges": ("INT", {"default": 64, "min": 3, "max": 4096,
                    "tooltip": "A boundary loop is filled only if its edge count is below this. "
                               "The outer frame is ~the grid perimeter (e.g. ~512 at grid 128), "
                               "interior tears are ~4-30, so 64 separates them by construction."}),
                "fill_depth_near_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Band-box spatial scope: only fill loops whose EVERY boundary "
                               "vertex's forward depth (recovered-camera view space, same axis "
                               "as AtlasBoundedBand's cutoff) lies within [near, far] metres. "
                               "Transcribe off a bounded band's near and cutoff_m. 0 = off "
                               "(edge-count-only mode; the single largest loop is always left open)."}),
                "fill_depth_far_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                    "tooltip": "Band-box far bound (the cutoff). 0 = off (see fill_depth_near_m)."}),
                "retopo_method": (["off", "quad", "decimate", "smooth"],
                    {"default": "off",
                     "tooltip": "EXPORT-ONLY retopology pass on the OBJ/GLB (the live viewport "
                                "projection mesh is never touched). off = no change (default, so "
                                "every saved workflow keeps working). quad = pyinstantmeshes "
                                "orientation-field quad remesh (cleanest DCC handoff; needs the "
                                "pyinstantmeshes package). decimate = quadric decimation via "
                                "fast-simplification (fewer faces, same topology class). smooth = "
                                "trimesh Taubin relax (topology-preserving, UVs kept). quad/decimate "
                                "change the vertex count so projection-baked UVs are REGENERATED "
                                "from the recovered camera (pure numpy) and the retopologized mesh "
                                "stays textured. Runs AFTER any interior hole-fill."}),
                "retopo_target_vertex_count": ("INT", {"default": 2000, "min": 4, "max": 2000000,
                    "tooltip": "Target vertex count for quad / target face count for decimate "
                               "(decimate targets ~2x this in faces). Ignored by smooth."}),
                "retopo_smooth_iterations": ("INT", {"default": 0, "min": 0, "max": 100,
                    "tooltip": "quad: Instant Meshes post-smooth iterations. smooth: Taubin "
                               "relax iterations (the actual smoothing strength). decimate: "
                               "ignored."}),
                "retopo_crease_angle": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "quad only: crease angle (deg) below which adjacent faces are "
                               "treated as one smooth surface in the orientation field."}),
                "retopo_pure_quad": ("BOOLEAN", {"default": False,
                    "tooltip": "quad only: force a pure-quad output (no triangles). False allows "
                               "quad-dominant (triangles where the field can't place a quad)."}),
            },
        }

    def export(self, solve, image, output_dir="atlas_exports", grid_long_edge=128,
               depth_edge_rel=0.5,
               depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
               device="auto", format="both", use_solve_mesh=True,
               max_edge_factor=12.0, normal_edge_deg=0.0,
               fill_interior_holes=False, max_hole_edges=64,
               fill_depth_near_m=0.0, fill_depth_far_m=0.0,
               retopo_method="off", retopo_target_vertex_count=2000,
               retopo_smooth_iterations=0, retopo_crease_angle=30.0,
               retopo_pure_quad=False):
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.exporters.relief_mesh_exporter import (
            export_relief_mesh,
            export_relief_mesh_glb,
        )
        from atlas_camera.inference.depth_estimator import estimate_depth

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        width = int(intr.image_width or image.shape[2])
        height = int(intr.image_height or image.shape[1])
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        if fx <= 0:
            raise ValueError(
                "Relief mesh export needs a solved focal length — run a solve node "
                "(e.g. Atlas Learned Solve) before this node."
            )
        cx = intr.cx_px if intr.cx_px is not None else width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else height / 2.0

        # Prefer the relief mesh ALREADY derived onto the solve — it carries all
        # the edge tuning (max_edge_factor / normal_edge_deg / band near-clip /
        # sky_heuristic) exactly, so the OBJ matches the viewport with no widget
        # to re-set. Re-derive only when asked (higher-res export) or when the
        # solve has no relief mesh (e.g. a bare solve node).
        mesh = _relief_mesh_from_solve(solve) if use_solve_mesh else None
        if mesh is None:
            tmp = _save_image_tensor_to_tmp(image)
            try:
                result = estimate_depth(tmp, model_id=depth_model,
                                        device=None if device == "auto" else device,
                                        # fx is in solve-image pixels; the tmp file is the
                                        # wired tensor's resolution (usually identical).
                                        focal_px=fx * (image.shape[2] / width))
            finally:
                os.unlink(tmp)

            depth_map = result.depth
            if depth_map.shape != (height, width):
                depth_map = _resize_depth(depth_map, width, height)

            horizon_y = None
            if solve.horizon_line and solve.horizon_line.endpoints_px:
                p1, p2 = solve.horizon_line.endpoints_px
                horizon_y = 0.5 * (float(p1[1]) + float(p2[1]))

            scale, scale_info = estimate_ground_scale(
                depth_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=horizon_y,
            )
            mesh = build_relief_mesh(
                depth_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                grid_long_edge=int(grid_long_edge),
                depth_edge_rel=float(depth_edge_rel),
                scale=scale,
                horizon_y=horizon_y,
                max_edge_factor=float(max_edge_factor),
                normal_edge_deg=(float(normal_edge_deg) if float(normal_edge_deg) > 0 else None),
            )
        # EXPORT-ONLY interior hole fill. Never touches the live projection
        # mesh (which keeps its deliberate silhouette tears for DMP); caps the
        # exported OBJ/GLB so it retopologizes/booleans cleanly in a DCC.
        n_filled, filled, faces_added, loops_left = 0, [], 0, 0
        if fill_interior_holes:
            from atlas_camera.core.mesh_repair import (
                apply_interior_hole_fill,
                boundary_edges,
                walk_loops,
            )
            n_before = len(mesh.faces)
            n_filled, filled = apply_interior_hole_fill(
                mesh,
                max_hole_edges=int(max_hole_edges),
                view_matrix=extr.camera_view_matrix,
                depth_near_m=float(fill_depth_near_m),
                depth_far_m=float(fill_depth_far_m),
            )
            faces_added = len(mesh.faces) - n_before
            # What's STILL open is the actionable half: a disappointing fill is
            # usually a too-tight scope, and the count says so at a glance.
            be = boundary_edges(mesh.faces)
            loops_left = len(walk_loops(be, faces=mesh.faces)) if len(be) else 0
        # EXPORT-ONLY retopology (quad / decimate / smooth) — same doctrine as
        # the hole-fill above: never touches the live viewport projection mesh
        # or solve.proxy_geometry. Runs AFTER the hole-fill so it retopologizes
        # the capped mesh. quad/decimate change the vertex count, so the 1:1
        # vertex-UV mapping is regenerated from the recovered camera (pure
        # numpy); smooth preserves topology and keeps the existing UVs.
        retopo_note = ""
        if retopo_method and retopo_method != "off":
            from atlas_camera.core.mesh_retopo import apply_retopo
            rrep = apply_retopo(
                mesh,
                method=str(retopo_method),
                target_vertex_count=int(retopo_target_vertex_count),
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                image_width=width, image_height=height,
                pure_quad=bool(retopo_pure_quad),
                crease_angle=float(retopo_crease_angle),
                smooth_iterations=int(retopo_smooth_iterations),
            )
            if rrep.get("changed"):
                retopo_note = (
                    f"\n\U0001f53b retopo [{rrep.get('method', retopo_method)}]: "
                    f"{rrep.get('in_verts', '?')} → {rrep.get('out_verts', '?')} verts, "
                    f"{rrep.get('in_faces', '?')} → {rrep.get('out_faces', '?')} faces "
                    f"— {rrep.get('note', '')}"
                )
            else:
                retopo_note = (
                    f"\n\U0001f53b retopo [{retopo_method}]: no change "
                    f"— {rrep.get('note', '')}"
                )
        report = _format_hole_fill_report(
            fill_interior_holes, n_filled, filled, faces_added, loops_left,
            max_hole_edges, float(fill_depth_near_m), float(fill_depth_far_m)) \
            + retopo_note + _scale_summary_suffix(solve)
        # The viewport gets the geometry that was ACTUALLY written, off the same
        # widgets — so what an artist tunes here is what lands in Maya/Nuke.
        preview_solve = _solve_with_relief_mesh(solve, mesh)
        texture = _image_tensor_to_pil(image)
        plate = getattr(solve, "source_plate", None)
        texture_path = None
        if plate is not None and getattr(plate, "image_path", None) and not getattr(plate, "is_proxy", True):
            texture_path = plate.image_path
        obj_path = glb_path = ""
        if format in ("both", "obj"):
            obj_path = export_relief_mesh(
                mesh,
                output_dir,
                texture=texture,
                texture_path=texture_path,
            )["obj"]
        if format in ("both", "glb"):
            glb_path = export_relief_mesh_glb(mesh, output_dir, texture=texture)["glb"]
        _write_export_manifest(solve, output_dir,
                               [("relief_obj", obj_path), ("relief_glb", glb_path)],
                               "AtlasExportReliefMesh")
        return {"ui": {"text": [report]},
                "result": (obj_path, glb_path, preview_solve, report)}


class AtlasExportUSD:
    """Export the solved camera as a USD camera asset (.usda)."""
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("usd_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            }
        }

    def export(self, solve, output_dir):
        from atlas_camera.exporters.usd_exporter import USDExporter
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "camera.usda"
        USDExporter().export_camera(solve, dest)
        _write_export_manifest(solve, out, [("usd_camera", str(dest))],
                               "AtlasExportUSD")
        return (str(dest),)


class AtlasExportBlender:
    """Export a Blender Python scene-build script for the recovered camera."""
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("script_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            },
            "optional": {
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata to include in the exported solve context."}),
            },
        }

    def export(self, solve, output_dir, output_profile=None):
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "build_scene.py"
        write_blender_scene_script(solve, dest)
        _write_export_manifest(solve, out, [("blender_script", str(dest))],
                               "AtlasExportBlender")
        return (str(dest),)


class AtlasExportNuke:
    """Export a Nuke Python projection script, plus a native .nk scene, for
    the recovered camera.

    Both files describe the identical camera-projection graph (Read ->
    Project3D2 -> Card or ReadGeo2 -> ScanlineRender, Camera2 feeding both
    the projection and the render camera); the .py needs a Script Editor
    (`exec(open(...).read()); build_projection()`), the .nk opens directly
    via File > Open or drag-and-drop. Both were verified by actually
    building and rendering this graph in Nuke (16.1v3) rather than only
    reading documentation — see nuke_exporter.py's module docstring and
    CLAUDE.md's "Nuke camera-projection topology" note for what that caught
    (Card3D has no xsize/ysize, ScanlineRender has no format knob, and the
    real obj/cam input indices are 1/2, not 0/1) and for the relief-mesh
    case specifically (ReadGeo2 imports OBJ/FBX natively, but does NOT
    auto-apply the OBJ/MTL's own texture — it still needs the live
    Project3D2 projection wired into its own image input, same as Card).
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script_path", "nk_path")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            },
            "optional": {
                "relief_mesh_obj_path": ("STRING", {"default": "",
                    "tooltip": "Optional obj_path output from AtlasExportReliefMesh. When set, the real "
                               "derived relief mesh is imported (ReadGeo2) and live-projected onto instead "
                               "of the default flat 40x40m ground card — wire AtlasExportReliefMesh's "
                               "obj_path here to see real derived geometry in Nuke."}),
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata for Read/colorspace annotations."}),
            },
        }

    def export(self, solve, output_dir, relief_mesh_obj_path="", output_profile=None):
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        py_dest = out / "nuke_projection.py"
        nk_dest = out / "nuke_projection.nk"
        # AtlasExportReliefMesh's obj_path is relative to ComfyUI's own
        # working directory (same convention as this node's own output_dir),
        # not to wherever an artist eventually launches Nuke from - resolve
        # to absolute so the generated script/scene stays portable.
        mesh_path = str(Path(relief_mesh_obj_path).resolve()) if relief_mesh_obj_path else None
        write_nuke_projection_script(solve, py_dest, relief_mesh_obj_path=mesh_path)
        write_nuke_native_script(solve, nk_dest, relief_mesh_obj_path=mesh_path)
        _write_export_manifest(solve, out,
                               [("nuke_script", str(py_dest)),
                                ("nuke_scene", str(nk_dest))],
                               "AtlasExportNuke")
        return (str(py_dest), str(nk_dest))


class AtlasExportNukeLayers:
    """Export EVERY projection layer on a solve (sky dome, clean-plate bands,
    multi-angle patches — each `ProjectionSource`) as ONE native .nk scene:
    per-layer Read (plate) + Camera2 (that layer's own camera) + Project3D2 +
    ReadGeo2 (that layer's mesh, written as OBJ+MTL alongside), all merged
    through a single Scene node into one ScanlineRender rendered from the
    PRIMARY solved camera.

    This is the DCC handoff for the viewport's layered 📽 Project — the same
    stacked-projections model, except layer overlap is resolved by Nuke's
    real z-buffer instead of priority/facing masks (true depth wins; for
    spatially-exclusive layers — bands, sky at radius_m — that matches the
    viewport's result). Plate images come from each source's registered
    non-proxy `plate_ref` when present (float/EXR-safe), else the browser
    preview is decoded to a PNG next to the .nk. Complements — never
    replaces — `AtlasExportNuke`, which stays the single-projection
    (primary camera + one mesh/card) exporter.

    Sources without mesh geometry or a plate are skipped (summarized in the
    second output). Errors loudly when NO exportable layer exists — chain at
    least one AtlasSkyDomeLayer / AtlasCleanPlateLayer / AtlasAddPatchView
    first.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("nk_path", "summary")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports/nuke_layers"}),
            },
            "optional": {
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata for annotations."}),
                "retopo_method": (["off", "quad", "decimate", "smooth"], {
                    "default": "off", "tooltip": "Export-only retopology applied to EVERY layer mesh before Nuke writes its OBJs."}),
                "retopo_target_vertex_count": ("INT", {"default": 2000, "min": 100, "max": 100000, "step": 100}),
                "retopo_smooth_iterations": ("INT", {"default": 0, "min": 0, "max": 20}),
                "retopo_crease_angle": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 1.0}),
                "retopo_pure_quad": ("BOOLEAN", {"default": False}),
            },
        }

    def export(self, solve, output_dir, output_profile=None,
               retopo_method="off", retopo_target_vertex_count=2000,
               retopo_smooth_iterations=0, retopo_crease_angle=30.0,
               retopo_pure_quad=False):
        from atlas_camera.exporters.nuke_exporter import write_nuke_layers_script
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        try:
            result = write_nuke_layers_script(
                solve, output_dir,
                retopo_method=retopo_method,
                retopo_target_vertex_count=retopo_target_vertex_count,
                retopo_smooth_iterations=retopo_smooth_iterations,
                retopo_crease_angle=retopo_crease_angle,
                retopo_pure_quad=retopo_pure_quad,
            )
        except ValueError as exc:
            # The LAYER export needs ProjectionSources (sky / clean-plate bands /
            # patches). A layers=0 single relief mesh has none — don't crash the
            # queue; return a clear pointer. Use AtlasInput layers>=1 for the full
            # DCC handoff, or AtlasExportUSD (camera) for the single-relief case.
            if "No exportable projection layers" not in str(exc):
                raise
            return ("", f"Nuke layer export skipped — {exc}")
        summary = f"{len(result['layers'])} layer(s): {', '.join(result['layers'])}"
        if retopo_method != "off":
            summary += f" | {retopo_method} retopo ≤{int(retopo_target_vertex_count)} verts/layer"
        if result["skipped"]:
            summary += f" | skipped: {'; '.join(result['skipped'])}"
        summary += _scale_summary_suffix(solve) + _health_summary_suffix(solve)
        _write_export_manifest(solve, output_dir,
                               [("nuke_scene", result["nk_path"])],
                               "AtlasExportNukeLayers")
        return (result["nk_path"], summary)


class AtlasExportMayaLayers:
    """Export EVERY projection layer on a solve (sky dome, clean-plate bands,
    multi-angle patches — each `ProjectionSource`) as ONE Maya ASCII scene:
    per-layer projector cameras as native .ma nodes, plus an embedded on-open
    scriptNode that imports each layer's OBJ and builds the proven
    camera-projection shading network (place3dTexture parented to that
    layer's camera -> projection.pm, projType 8 — the same verified setup as
    AtlasExportMayaReviewScene's single projection).

    The Maya twin of `AtlasExportNukeLayers`: identical shared layer
    collection and on-disk assets (plates with edge mattes embedded in
    ALPHA + standalone matte PNGs + OBJ meshes), so a layer that exports to
    Nuke always exports to Maya the same way. Edge mattes drive
    lambert.transparency via the plate's alpha (the mesh's baked UVs match
    the plate frame by construction). Drag/File > Open the .ma; if Maya's
    script security blocks the on-open scriptNode, the OBJs sit next to the
    .ma for manual import.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ma_path", "summary")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports/maya_layers"}),
            },
            "optional": {
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata for annotations."}),
                "retopo_method": (["off", "quad", "decimate", "smooth"], {
                    "default": "off", "tooltip": "Export-only retopology applied to EVERY layer mesh before Maya writes its OBJs."}),
                "retopo_target_vertex_count": ("INT", {"default": 2000, "min": 100, "max": 100000, "step": 100}),
                "retopo_smooth_iterations": ("INT", {"default": 0, "min": 0, "max": 20}),
                "retopo_crease_angle": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 1.0}),
                "retopo_pure_quad": ("BOOLEAN", {"default": False}),
            },
        }

    def export(self, solve, output_dir, output_profile=None,
               retopo_method="off", retopo_target_vertex_count=2000,
               retopo_smooth_iterations=0, retopo_crease_angle=30.0,
               retopo_pure_quad=False):
        from atlas_camera.exporters.maya_exporter import write_maya_layers_scene
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        try:
            result = write_maya_layers_scene(
                solve, output_dir,
                retopo_method=retopo_method,
                retopo_target_vertex_count=retopo_target_vertex_count,
                retopo_smooth_iterations=retopo_smooth_iterations,
                retopo_crease_angle=retopo_crease_angle,
                retopo_pure_quad=retopo_pure_quad,
            )
        except ValueError as exc:
            # See AtlasExportNukeLayers: the LAYER export needs ProjectionSources;
            # a layers=0 single relief mesh has none. Graceful skip, not a crash.
            if "No exportable projection layers" not in str(exc):
                raise
            return ("", f"Maya layer export skipped — {exc}")
        summary = f"{len(result['layers'])} layer(s): {', '.join(result['layers'])}"
        if retopo_method != "off":
            summary += f" | {retopo_method} retopo ≤{int(retopo_target_vertex_count)} verts/layer"
        if result["skipped"]:
            summary += f" | skipped: {'; '.join(result['skipped'])}"
        summary += _scale_summary_suffix(solve) + _health_summary_suffix(solve)
        _write_export_manifest(solve, output_dir,
                               [("maya_scene", result["ma_path"])],
                               "AtlasExportMayaLayers")
        return (result["ma_path"], summary)


# ---------------------------------------------------------------------------
# Track 3 — camera path animation (see AtlasBlockoutViewport's Camera Path mode)
# ---------------------------------------------------------------------------

class AtlasExportCameraPathUSD:
    """Export a keyframed camera path as a time-sampled USD camera (.usda).

    Separate from AtlasExportUSD because it takes a different required input
    (ATLAS_CAMERA_PATH, produced by AtlasBlockoutViewport's Camera Path mode)
    rather than a single static solve pose.
    """
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("usd_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "camera_path": ("ATLAS_CAMERA_PATH",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            }
        }

    def export(self, solve, camera_path, output_dir):
        from atlas_camera.exporters.usd_exporter import USDExporter
        if camera_path is None or not camera_path.keyframes:
            raise ValueError(
                "No camera path yet — open AtlasBlockoutViewport, use 🎥 Camera Path "
                "to add at least one keyframe, then click ⏺ Bake Proxy Path before queuing "
                "this export node."
            )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "camera_path.usda"
        USDExporter().export_camera_animation(camera_path, solve.camera.intrinsics, dest)
        _write_export_manifest(solve, out, [("usd_camera_path", str(dest))],
                               "AtlasExportCameraPathUSD")
        return (str(dest),)


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
