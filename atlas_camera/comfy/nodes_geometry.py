"""Atlas ComfyUI nodes — geometry group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
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

from atlas_camera.comfy.node_helpers import (
    LIVE_FILL_WIDGETS,
    _AZIMUTH_VIEWS,
    _DEPTH_MODEL_CHOICES,
    _DISTANCE_VIEWS,
    _ELEVATION_VIEWS,
    _depth_map_for_solve,
    _horizon_y_from_solve,
    _image_tensor_to_pil,
    _mask_to_b64_png,
    _named_view_orbit_delta,
    _parse_exact_view,
    _parse_view_prompt,
    _pil_to_image_tensor,
    _replace_proxy_role_geometry,
    _hole_mask_after_fill,
    _require_numpy,
    _require_pil,
    _require_torch,
    _resolve_exclude_mask,
    _apply_backdrop_mode,
    BACKDROP_WIDGET,
    _save_image_tensor_to_tmp,
    _solve_camera_params,
    apply_live_mesh_repair,
)





class AtlasDeriveProjectionGeometry:
    """Derive camera-projection proxy geometry (ground/walls/boxes/cylinders/backdrop)
    from a Depth Anything V2 depth map + the solve's recovered camera.

    The blockout viewport builds these primitives and can project the source image
    onto them from the recovered camera — the classic VFX matte-painting setup.
    Requires the [neural] extra (re-runs metric depth internally; the IMAGE from
    AtlasDepthAnything is normalized and unusable for metric geometry).

    ``primitive_method`` selects how "primitives" mode derives geometry
    (only relevant when ``geometry_mode`` includes "primitives"):
    - ``azimuth_walls`` (default) — vertical walls only, general-purpose.
      Height comes from a percentile clip of the 3D points that individually
      pass a near-vertical-normal filter — a sloped roof, spire, or tower
      never qualifies, so on complex facades the wall only ever reflects the
      plain section below it (confirmed on real church/tower photos).
    - ``ransac_planes`` — any-orientation planes (sloped roofs, stepped/angled
      facades) via sequential RANSAC seeded by a 2D normal-orientation
      histogram. Best for exterior/architectural shots.
    - ``room_cuboid`` — Manhattan-aligned floor + up to 4 walls + optional
      ceiling. Best for orthogonal interiors; silently produces skewed walls
      on non-orthogonal rooms (pick a different method for those shots).
    - ``vertical_extrusion`` — same wall orientation/distance detection as
      ``azimuth_walls``, but height comes from the image-space silhouette
      instead: the topmost non-sky pixel per column (see
      ``depth_geometry.detect_sky_mask``), back-projected at that pixel's own
      depth regardless of its local surface normal. A flat vertical
      "billboard" extruded to the real silhouette top, per Hoiem/Efros/
      Hebert's "Automatic Photo Pop-up" (SIGGRAPH 2005) — reaches sloped
      roofs, spires, and towers that ``azimuth_walls`` truncates. Best for
      complex exterior architecture where a single flat wall height is the
      wrong shape but full RANSAC plane-fitting is overkill.

    ``scene_type`` (default "manual") is a one-choice convenience preset over
    the three widgets above, for artists who'd rather pick a shot type than
    reason about geometry_mode/primitive_method/depth_model separately:
    "organic" -> relief_mesh, "indoor" -> primitives+room_cuboid+Indoor depth
    model, "outdoor" -> primitives+ransac_planes+Outdoor depth model. Purely
    a preset — it sets the same three parameters this node already exposes,
    never a new solving code path. "manual" leaves them untouched.

    ``hole_mask`` mirrors the relief mesh's own discarded hole/tear data
    (`ReliefMesh.hole_mask`) whenever ``geometry_mode`` builds one ("both"/
    "relief_mesh") - full source-image resolution, white where no triangle
    covers that pixel. A zero mask when ``geometry_mode="primitives"``, since
    no relief mesh is built to have holes in that mode.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "MASK")
    RETURN_NAMES = ("solve", "hole_mask")
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 32,
                                        "tooltip": "Max foreground boxes/cylinders. Street-level scenes: try 0 — the 2D occupancy clustering merges cars/fences/trees into oversized near-camera boxes that dominate any orbit."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "geometry_mode": (["relief_mesh", "primitives", "both"], {"default": "relief_mesh",
                    "tooltip": "What the viewport receives. relief_mesh = contoured depth mesh "
                               "(recommended); primitives = flat blockout planes/boxes; both "
                               "overlaps the two on the same surfaces (enclosure + z-shimmer)."}),
                "relief_grid": ("INT", {"default": 128, "min": 16, "max": 4096,
                    "tooltip": "Viewport relief-mesh density (long-edge grid columns). Higher = "
                               "fewer/smaller torn holes on noisy AI-image depth (each quad spans "
                               "less real-world area, so it's less likely to straddle a spurious "
                               "depth jump) at the cost of a larger mesh payload sent to the "
                               "browser and a slower/heavier viewport. Overridden by "
                               "relief_quality unless that's set to 'custom'."}),
                "primitive_method": (["azimuth_walls", "ransac_planes", "room_cuboid",
                                       "vertical_extrusion"],
                    {"default": "azimuth_walls",
                     "tooltip": "azimuth_walls (default) = vertical walls only, height clipped "
                                "to the plain wall (truncates sloped roofs/spires/towers). "
                                "ransac_planes = any-orientation planes (roofs, stepped "
                                "facades) — exteriors. room_cuboid = Manhattan floor+walls"
                                "+ceiling — orthogonal interiors. vertical_extrusion = same wall "
                                "orientation as azimuth_walls but height extruded to the real "
                                "image-space silhouette top (reaches towers/spires/sloped roofs "
                                "azimuth_walls truncates). Only affects "
                                "geometry_mode=primitives/both; max_walls is reused as the "
                                "plane budget for ransac_planes and ignored by room_cuboid. "
                                "Ignored when scene_type != manual."}),
                "scene_type": ([
                    "manual", "organic", "mountains", "forests", "aerial",
                    "indoor", "outdoor", "simple_walls", "towers_spires",
                ], {"default": "manual",
                    "tooltip": "The one choice that matters — picks a complete, self-consistent "
                               "combination of geometry_mode/primitive_method/relief_quality/"
                               "depth_edge_rel/max_objects/depth_model for a named shot type, so "
                               "you never have to know which of those five widgets actually does "
                               "anything for your scene (e.g. primitive_method is silently ignored "
                               "whenever geometry_mode=relief_mesh — this picks a combination where "
                               "that can't happen). When this is anything but 'manual', the widgets "
                               "below it grey out and show the values this preset is using.\n"
                               "  organic = smooth relief mesh, general-purpose natural/cluttered "
                               "scenes.\n"
                               "  mountains = relief mesh at high density (terrain/ridgelines need "
                               "more grid resolution than the default to read as continuous rather "
                               "than faceted).\n"
                               "  forests = relief mesh at high density with a relaxed tear "
                               "threshold — dense canopy depth is genuinely noisy at a small scale, "
                               "so the default threshold shreds it into holes; this trades a little "
                               "silhouette accuracy for a filled-in canopy instead of swiss cheese.\n"
                               "  aerial = relief mesh AND primitives together (geometry_mode=both) "
                               "with more foreground objects allowed — buildings read as boxes "
                               "sitting on/above the relief-mesh ground and treeline, the drone/"
                               "top-down shot case.\n"
                               "  indoor = primitives + room_cuboid + the Indoor depth model "
                               "(orthogonal interiors).\n"
                               "  outdoor = primitives + ransac_planes + the Outdoor depth model "
                               "(sloped roofs, stepped facades).\n"
                               "  simple_walls = primitives + azimuth_walls (fast flat-wall "
                               "blockout, general exteriors).\n"
                               "  towers_spires = primitives + vertical_extrusion (reaches tall/"
                               "sloped silhouettes azimuth_walls truncates).\n"
                               "  manual (default) leaves every widget below exactly as set — fully "
                               "backward compatible with workflows saved before this widget existed. "
                               "If AtlasLearnedSolveFromImage's height_mode=measure_from_depth, set "
                               "its own depth_model to match by hand — this preset only reaches "
                               "this node's depth estimation, not the upstream solve node's."}),
                # Appended at the end (not inserted earlier in this dict) so that
                # ComfyUI's positional widgets_values array stays backward
                # compatible: a workflow saved before these two existed just gets
                # its own defaults filled in for these trailing slots, instead of
                # every later value shifting into the wrong widget.
                "relief_quality": (["custom", "low", "medium", "high", "ultra"], {"default": "custom",
                    "tooltip": "Quick-pick override for relief_grid: low=64, medium=256, high=512, "
                               "ultra=1024. 'custom' (default) leaves relief_grid exactly as set "
                               "above — fully backward compatible. Same convenience-preset "
                               "pattern as scene_type: this only sets relief_grid, no new solving "
                               "path. 'ultra' produces a much larger mesh — expect a slower "
                               "viewport and bigger solve JSON exports."}),
                "depth_edge_rel": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Relative depth jump that tears the mesh into a silhouette hole. "
                               "Lower = tears more readily (cleaner silhouettes, more holes on "
                               "noisy depth); higher = tears less (fewer holes, more risk of "
                               "rubber-sheeting a real silhouette onto the background). Same "
                               "parameter and default as AtlasExportReliefMesh."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional external exclusion (e.g. a real sky segmentation from "
                               "SAM/RMBG) which REPLACES the internal sky heuristic before "
                               "triangulation - so it must cover EVERYTHING you want gone. Only "
                               "affects the relief_mesh branch (geometry_mode both/relief_mesh); "
                               "the primitives/wall-fitting branch is unaffected. Any resolution - "
                               "resized to match depth."}),
                **LIVE_FILL_WIDGETS,
            },
        }

    _SCENE_TYPE_PRESETS = {
        "organic": {"geometry_mode": "relief_mesh"},
        "mountains": {"geometry_mode": "relief_mesh", "relief_quality": "high"},
        "forests": {"geometry_mode": "relief_mesh", "relief_quality": "high", "depth_edge_rel": 1.0},
        "aerial": {"geometry_mode": "both", "primitive_method": "azimuth_walls",
                   "relief_quality": "medium", "max_objects": 6},
        "indoor": {"geometry_mode": "primitives", "primitive_method": "room_cuboid",
                   "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"},
        "outdoor": {"geometry_mode": "primitives", "primitive_method": "ransac_planes",
                    "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"},
        "simple_walls": {"geometry_mode": "primitives", "primitive_method": "azimuth_walls"},
        "towers_spires": {"geometry_mode": "primitives", "primitive_method": "vertical_extrusion"},
    }
    _RELIEF_QUALITY_PRESETS = {"low": 64, "medium": 256, "high": 512, "ultra": 1024}

    def derive(self, solve, image,
               depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
               max_walls=4, max_objects=3, device="auto",
               geometry_mode="relief_mesh", relief_grid=128,
               primitive_method="azimuth_walls", scene_type="manual",
               relief_quality="custom", depth_edge_rel=0.5,
               exclude_mask=None,
               live_fill_holes=False, live_fill_distance_m=0.0,
               live_fill_max_hole_edges=256,
               live_fill_edge_sawteeth=False):
        torch = _require_torch()
        np = _require_numpy()
        preset = self._SCENE_TYPE_PRESETS.get(scene_type)
        if preset:
            geometry_mode = preset.get("geometry_mode", geometry_mode)
            primitive_method = preset.get("primitive_method", primitive_method)
            depth_model = preset.get("depth_model", depth_model)
            relief_quality = preset.get("relief_quality", relief_quality)
            depth_edge_rel = preset.get("depth_edge_rel", depth_edge_rel)
            max_objects = preset.get("max_objects", max_objects)
        if relief_quality in self._RELIEF_QUALITY_PRESETS:
            relief_grid = self._RELIEF_QUALITY_PRESETS[relief_quality]
        from atlas_camera.core.plane_extraction import PlaneRansacConfig, extract_planes_ransac
        from atlas_camera.core.proxy_geometry import (
            PROXY_ROLE,
            ProxyDerivationConfig,
            derive_projection_proxies,
            derive_vertical_extrusion_proxies,
            relief_mesh_primitive,
        )
        from atlas_camera.core.relief_mesh import build_relief_mesh
        from atlas_camera.core.room_layout import RoomCuboidConfig, extract_room_cuboid
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.inference.depth_estimator import estimate_depth

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        width = int(intr.image_width or image.shape[2])
        height = int(intr.image_height or image.shape[1])
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx

        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device,
                                    focal_px=(fx * (image.shape[2] / width)) if fx > 0 else None)
        finally:
            os.unlink(tmp)

        if fx <= 0:
            zero = torch.zeros(1, int(image.shape[1]), int(image.shape[2]), dtype=torch.float32)
            return (solve, zero)
        cx = intr.cx_px if intr.cx_px is not None else width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else height / 2.0
        resolved_exclude = _resolve_exclude_mask(exclude_mask, height, width)

        depth_map = result.depth
        if depth_map.shape != (height, width):
            depth_map = _resize_depth(depth_map, width, height)

        horizon_y = _horizon_y_from_solve(solve)

        if primitive_method == "ransac_planes":
            prims, stats = extract_planes_ransac(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                max_planes=max(int(max_walls), 1) * 2,
                horizon_y=horizon_y,
                config=PlaneRansacConfig(),
            )
        elif primitive_method == "room_cuboid":
            prims, stats = extract_room_cuboid(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=horizon_y,
                config=RoomCuboidConfig(),
            )
        elif primitive_method == "vertical_extrusion":
            cfg = ProxyDerivationConfig(max_objects=int(max_objects))
            prims, stats = derive_vertical_extrusion_proxies(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                max_walls=int(max_walls),
                horizon_y=horizon_y,
                config=cfg,
            )
        else:
            cfg = ProxyDerivationConfig(max_objects=int(max_objects))
            prims, stats = derive_projection_proxies(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                max_walls=int(max_walls),
                horizon_y=horizon_y,
                config=cfg,
            )
        stats["primitive_method"] = primitive_method

        hole_mask_arr = np.zeros((height, width), dtype=np.float32)
        keep: list = []
        if geometry_mode in ("both", "primitives"):
            keep.extend(prims)
        else:
            keep.extend(p for p in prims if p.name == "projection_backdrop")
        if geometry_mode in ("both", "relief_mesh"):
            mesh = build_relief_mesh(
                depth_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                grid_long_edge=int(relief_grid),
                depth_edge_rel=float(depth_edge_rel),
                scale=float(stats.get("ground_scale", 1.0)),
                horizon_y=horizon_y,
                exclude_mask=resolved_exclude,
                apply_sky_heuristic=resolved_exclude is None,
                live_fill_holes=bool(live_fill_holes),
                live_fill_edge_sawteeth=bool(live_fill_edge_sawteeth),
            )

            stats["relief_mesh"] = {
                "n_vertices": mesh.stats["n_vertices"],
                "n_faces": mesh.stats["n_faces"],
            }
            apply_live_mesh_repair(
                mesh,
                extr.camera_view_matrix,
                live_fill_holes=live_fill_holes,
                live_fill_distance_m=live_fill_distance_m,
                live_fill_max_hole_edges=live_fill_max_hole_edges,
                live_fill_edge_sawteeth=live_fill_edge_sawteeth,
                stats=stats,
                image_width=intr.image_width,
                image_height=intr.image_height,
            )
            keep.append(relief_mesh_primitive(mesh))
            hole_mask_arr = mesh.hole_mask.astype(np.float32)


        out = _replace_proxy_role_geometry(solve, keep, stats, {
            "depth_model": depth_model,
            "geometry_mode": geometry_mode,
            "scene_type": scene_type,
            "primitive_method": primitive_method,
            "depth_edge_rel": float(depth_edge_rel),
            "relief_grid": int(relief_grid),
            "relief_quality": relief_quality,
            "max_objects": int(max_objects),
            "live_fill_holes": bool(live_fill_holes),
            "live_fill_distance_m": float(live_fill_distance_m),
            "live_fill_max_hole_edges": int(live_fill_max_hole_edges),
            "live_fill_edge_sawteeth": bool(live_fill_edge_sawteeth),
            "derive_node": "AtlasDeriveProjectionGeometry",
        })
        hole_t = torch.from_numpy(hole_mask_arr).unsqueeze(0)
        return (out, hole_t)



class AtlasPredictHiddenGeometry:
    """🔬 EXPERIMENTAL, RESEARCH-ONLY — "X-ray" depth map via LaRI layered ray
    intersections.

    Predicts the surfaces HIDDEN behind foreground occluders (per pixel, the
    first ray intersection that clears the visible surface) and returns a
    patched copy of the input ATLAS_DEPTH_MAP with occluder pixels replaced by
    that predicted hidden depth — a depth map of "the world with the occluders
    removed". Wire the ORIGINAL depth into foreground band layers and this
    node's output into BACKGROUND band layers so disocclusion reveals get
    predicted geometry instead of diffusion-smoothed guesses.

    Hidden depth is a HYPOTHESIS, never a measurement: the report output
    carries registration quality + coverage, and `hidden_mask` marks every
    substituted pixel for provenance. Works best on indoor/architectural
    scenes (the model's training domain — see
    docs/dev/hidden_geometry_training_free_research.md); outdoor terrain can
    collapse to near-zero coverage, in which case the depth passes through
    almost unchanged.

    Requires a user-cloned LaRI repository (github.com/ruili3/lari — NO
    upstream license, research use only; atlas_camera bundles none of it).
    Point `lari_path` (or the ATLAS_LARI_PATH env var) at the clone.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("depth", "hidden_mask", "report", "paint_matte")
    FUNCTION = "predict"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("ATLAS_DEPTH_MAP",),
                "image": ("IMAGE",),
            },
            "optional": {
                "lari_path": ("STRING", {"default": "", "tooltip":
                    "Path to your clone of github.com/ruili3/lari (research-only, "
                    "unlicensed upstream). Blank = the ATLAS_LARI_PATH env var."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "clear_rel": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0,
                    "step": 0.01, "tooltip":
                    "A hidden layer must be at least this fraction of the visible "
                    "depth BEHIND it to count as a separate surface (occluder back "
                    "faces are closer than this and get skipped)."}),
                "min_clear_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0,
                    "step": 0.1, "tooltip":
                    "Absolute clearance floor in the depth map's units. 0 = auto "
                    "(2% of the median visible depth) — the scene-adaptive margin "
                    "shallow scenes need."}),
                "restrict_mask": ("MASK", {"tooltip":
                    "Optional — only substitute hidden depth inside this mask "
                    "(e.g. a foreground band's layer_mask). Without it, every "
                    "confidently-detected occluder is replaced."}),
                "model": (["lari-scene", "world-tracing-scene"],
                    {"default": "lari-scene", "tooltip":
                    "Layered-ray-intersection backend. lari-scene = LaRI (fast "
                    "regression, ~0.2s, unlicensed upstream). world-tracing-scene "
                    "= WT-DiT r69l (diffusion, ~17s/20 steps, CC BY-NC-ND 4.0, "
                    "HF-gated checkpoint). Both are research-only."}),
                "wt_path": ("STRING", {"default": "", "tooltip":
                    "Path to your clone of github.com/haoz19/world-tracing "
                    "(only used by the world-tracing-scene backend). Blank = the "
                    "ATLAS_WT_PATH env var."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip":
                    "Diffusion sampling steps (world-tracing backend only)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                    "tooltip": "Diffusion seed (world-tracing backend only — "
                    "WT is generative; pin this for reproducible hidden geometry)."}),
                "smooth_px": ("INT", {"default": 31, "min": 0, "max": 201,
                    "tooltip": "Gaussian-smooth the substituted hidden depth "
                    "(sigma ≈ 0.75×this, px). Layer-switch seams and fill-block "
                    "steps shred the downstream relief mesh via its world-edge "
                    "check (immune to depth_edge_rel — measured; and a MEDIAN "
                    "filter preserves exactly those steps, also measured). "
                    "0 = off."}),
                "fill_gaps": ("BOOLEAN", {"default": True,
                    "tooltip": "Diffuse the predictions across the WHOLE "
                    "restrict_mask region (needs restrict_mask wired): treats "
                    "scattered per-pixel predictions as samples of ONE coherent "
                    "hidden surface, so the X-ray layer meshes continuously "
                    "instead of shredding on fragmented masks (foliage). "
                    "Filled depth is clamped to stay BEHIND the visible surface."}),
            },
        }

    def predict(self, depth, image, lari_path="", device="auto",
                clear_rel=0.15, min_clear_m=0.0, restrict_mask=None,
                model="lari-scene", wt_path="", steps=20, seed=0,
                smooth_px=31, fill_gaps=True):
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.hidden_geometry import select_hidden_surface
        from atlas_camera.inference.depth_estimator import DepthResult

        tmp = _save_image_tensor_to_tmp(image)
        try:
            if model == "world-tracing-scene":
                from atlas_camera.inference.wt_hidden_geometry import (
                    predict_layered_depth_wt,
                )
                layered = predict_layered_depth_wt(
                    tmp, wt_path=wt_path,
                    device=None if device == "auto" else device,
                    steps=steps, seed=seed)
            else:
                from atlas_camera.inference.lari_hidden_geometry import (
                    predict_layered_depth,
                )
                layered = predict_layered_depth(
                    tmp, lari_path=lari_path,
                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)

        raw = np.asarray(depth.depth, dtype=np.float64)
        H, W = raw.shape
        lt = torch.from_numpy(layered.layers).permute(2, 0, 1)[None]  # (1,L,h,w)
        layers_up = torch.nn.functional.interpolate(
            lt, size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).numpy().astype(np.float64)

        hidden, hidden_valid, stats = select_hidden_surface(
            layers_up, raw, clear_rel=clear_rel,
            min_clear=(min_clear_m if min_clear_m > 0 else None))

        region = None
        if restrict_mask is not None:
            m = restrict_mask
            if m.dim() == 3:
                m = m[0]
            m = torch.nn.functional.interpolate(
                m[None, None].float(), size=(H, W), mode="nearest"
            )[0, 0].numpy() > 0.5
            hidden_valid = hidden_valid & m
            stats["restricted_coverage"] = float(hidden_valid.mean())
            region = m & (raw > 1e-6)

        # Coherence pass (see the smooth_px/fill_gaps tooltips): fragmented
        # per-pixel predictions shred the downstream relief mesh via its
        # world-edge check, so (a) diffuse the predictions into ONE surface
        # across the restrict region, (b) median-smooth the layer-switch
        # seams, (c) clamp the result to stay BEHIND the visible surface.
        if fill_gaps and region is not None and hidden_valid.any():
            from atlas_camera.core.hidden_geometry import fill_hidden_gaps
            n_pred = int(hidden_valid.sum())
            hidden, hidden_valid = fill_hidden_gaps(hidden, hidden_valid, region)
            stats["filled_fraction"] = float(
                (int(hidden_valid.sum()) - n_pred) / max(int(hidden_valid.sum()), 1))
        if smooth_px and int(smooth_px) > 1 and hidden_valid.any():
            try:
                # GAUSSIAN, not median (calibrated 2026-07-09): median is
                # edge-preserving, so it kept the fill's block steps intact and
                # the mesh kept shredding (jungle hole-in-paint 0.455 median vs
                # 0.260 gaussian). The diffusion fill already handles outliers.
                from scipy.ndimage import gaussian_filter
                field = np.where(hidden_valid, hidden, raw)
                hidden = gaussian_filter(field, sigma=0.75 * float(smooth_px))
                stats["smooth_px"] = int(smooth_px)
            except ImportError:
                stats["warning_smooth"] = "scipy unavailable — smoothing skipped"
        # Geometry vs paint are SEPARATE concerns (jungle calibration lesson):
        # the substituted surface must stay CONTINUOUS to mesh (no clamping —
        # clamping filled depth out to a farther visible surface at see-through
        # gaps re-creates the metre-scale seams the fill just removed), while
        # PAINTING is only correct where the hidden surface is genuinely behind
        # a nearer occluder. paint_matte = those pixels; wire it into the
        # X-ray band's layer_matte so see-through gaps discard in the shader
        # (revealing the base mesh's real far content) without fragmenting
        # the geometry.
        paint = hidden_valid & (hidden > raw * 1.02)
        stats["paint_fraction"] = float(paint.mean())

        patched = raw.copy()
        patched[hidden_valid] = hidden[hidden_valid]

        scalar_stats = {k: v for k, v in stats.items()
                        if isinstance(v, (int, float, str))}
        backend = "world-tracing" if model == "world-tracing-scene" else "lari"
        # Provenance for the viewport's 🩻 debug overlay: WHICH pixels were
        # substituted and by WHICH backend, threaded (JSON-safe PNG data URI —
        # DepthResult.metadata must stay summary()-serializable) through
        # AtlasCleanPlateLayer into the ProjectionSource payload.
        provenance = {"hidden_backend": backend}
        if paint.any():
            # The 🩻 tint marks PAINTED hidden surface (paint matte), not the
            # full continuity-filled region — see the geometry-vs-paint note.
            hb64 = _mask_to_b64_png(paint)
            if hb64:
                provenance["hidden_mask_b64"] = hb64
        out = DepthResult(
            depth=patched.astype(np.float32),
            is_metric=depth.is_metric,
            model_id=f"{depth.model_id}+{backend}_hidden",
            image_width=depth.image_width,
            image_height=depth.image_height,
            near=float(patched.min()),
            far=float(patched.max()),
            metadata={**depth.metadata, "research_only": True, **provenance,
                      **{f"hidden_{k}": v for k, v in scalar_stats.items()}},
        )
        mask_t = torch.from_numpy(hidden_valid.astype(np.float32))[None]
        paint_t = torch.from_numpy(paint.astype(np.float32))[None]

        rel_mad = stats.get("registration_rel_mad", float("inf"))
        quality = ("good" if rel_mad < 0.2 else
                   "shaky" if rel_mad < 0.5 else "poor")
        backend_line = (
            "World Tracing r69l — CC BY-NC-ND 4.0, non-commercial; "
            f"diffusion steps {steps}, seed {seed}"
            if model == "world-tracing-scene"
            else "LaRI — upstream repo has NO license; do not use commercially"
        )
        report = (
            f"🔬 RESEARCH-ONLY hidden-geometry prediction ({backend_line}).\n"
            f"registration: scale {stats.get('scale', 0):.3f}, rel MAD "
            f"{rel_mad:.3f} ({quality})\n"
            f"substituted pixels: {int(hidden_valid.sum())} "
            f"({100.0 * float(hidden_valid.mean()):.1f}% of frame)\n"
            f"median hidden-vs-visible separation: "
            f"{stats.get('median_separation') if stats.get('median_separation') is not None else 'n/a'}\n"
            f"layer histogram (index of first clearing layer): "
            f"{stats.get('layer_used_histogram')}\n"
            + ("warning: " + stats["warning"] + "\n" if "warning" in stats else "")
            + ("warning: no restrict_mask wired — substitution covers "
               f"{100.0 * float(hidden_valid.mean()):.0f}% of the frame, "
               "including VISIBLE background surfaces (LaRI predicts "
               "through-wall structure there). For band workflows wire the "
               "foreground band's layer_mask into restrict_mask so only real "
               "occluders are replaced.\n"
               if restrict_mask is None and float(hidden_valid.mean()) > 0.25
               else "")
            + "Hidden depth is a hypothesis — best on indoor/architectural "
              "scenes; verify by orbiting the projected result."
        )
        return (out, mask_t, report, paint_t)


class AtlasRenderFix:
    """🔬 EXPERIMENTAL — repair projected-render artifacts with NVIDIA Fixer.

    Runs the pretrained Fixer model (single-step diffusion, the Difix3D+
    successor, trained to fix rendered-novel-view artifacts) over an IMAGE
    batch — typically `AtlasBlockoutViewport`'s baked `path_frames` before a
    Video Combine node. Spike-verified on this repo's own baked orbits
    (2026-07-10): fills ~1/3 of hard black tear pixels on a bare relief
    mesh, softens stretched-texel smears on the full DMP rig, adds no
    temporal flicker, ~0.3–0.45 s/frame on an RTX 5090 (plus ~1 min model
    load/warmup per queue). Costs/limits: mild overall softening (single-step
    regeneration at an internal 576×1024), and it does NOT outpaint large
    frame-edge reveals — band-layer frame outpainting stays the answer there.

    Unlike the LaRI/WT experimental nodes this runs in a DOCKER CONTAINER
    (the cosmos/transformer_engine stack has no native Windows build):
    build the image once from docker/fixer/Dockerfile, clone
    github.com/nv-tlabs/Fixer (Apache-2.0) with its weights (NVIDIA Open
    Model License — commercial use permitted), and point `fixer_path` (or
    ATLAS_FIXER_PATH) at the clone. See INSTALL.md 'Experimental: Fixer
    Render Repair'. Fails loud with actionable errors when docker/image/
    weights are missing.
    """
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "fix"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip":
                    "Frames to repair — e.g. AtlasBlockoutViewport's baked "
                    "path_frames. Fixer works internally at 576×1024; frames "
                    "near that resolution round-trip with the least "
                    "softening."}),
            },
            "optional": {
                "fixer_path": ("STRING", {"default": "", "tooltip":
                    "Path to your clone of github.com/nv-tlabs/Fixer with "
                    "weights downloaded into models/ (hf download nvidia/Fixer "
                    "--local-dir models). Blank = the ATLAS_FIXER_PATH env "
                    "var."}),
                "docker_image": ("STRING", {"default": "fixer-spike-env",
                    "tooltip": "Inference container image — build once with: "
                    "docker build -t fixer-spike-env -f docker/fixer/Dockerfile "
                    "docker/fixer/"}),
                "timestep": ("INT", {"default": 250, "min": 1, "max": 999,
                    "tooltip": "Fixer's single denoising timestep (upstream "
                    "default 250; the older difix checkpoint used 199)."}),
                "timeout_s": ("INT", {"default": 900, "min": 60, "max": 7200,
                    "tooltip": "Kill the container after this many seconds. "
                    "Budget ~1 min load/warmup + ~0.5 s/frame."}),
            },
        }

    def fix(self, images, fixer_path="", docker_image="fixer-spike-env",
            timestep=250, timeout_s=900):
        import shutil
        import time
        np = _require_numpy()
        torch = _require_torch()
        PILImage = _require_pil()
        from atlas_camera.inference.fixer_render_fix import (
            resolve_fixer_root, run_fixer_on_dir,
        )

        root = resolve_fixer_root(fixer_path)
        exchange = Path(tempfile.mkdtemp(prefix="atlas_fixer_"))
        in_dir = exchange / "in"
        out_dir = exchange / "out"
        in_dir.mkdir()
        try:
            frames = images.cpu().numpy()  # (B,H,W,3) float 0-1
            for i in range(frames.shape[0]):
                arr = (frames[i] * 255.0).clip(0, 255).astype("uint8")
                PILImage.fromarray(arr, mode="RGB").save(
                    in_dir / f"frame_{i:05d}.png")
            t0 = time.time()
            log_tail = run_fixer_on_dir(
                in_dir, out_dir, root, docker_image=docker_image,
                timestep=timestep, timeout_s=timeout_s)
            elapsed = time.time() - t0
            outs = sorted(out_dir.glob("*.png"))
            fixed = []
            for i, f in enumerate(outs):
                arr = np.array(PILImage.open(f).convert("RGB"),
                               dtype=np.float32) / 255.0
                # Fixer returns input resolution, but guard against drift so a
                # mismatched frame can't crash the stack() below.
                if arr.shape[:2] != frames.shape[1:3]:
                    pil = PILImage.fromarray(
                        (arr * 255).astype("uint8")).resize(
                        (frames.shape[2], frames.shape[1]), PILImage.LANCZOS)
                    arr = np.array(pil, dtype=np.float32) / 255.0
                fixed.append(arr)
            out_t = torch.from_numpy(np.stack(fixed, axis=0))
            report = (
                "🔬 EXPERIMENTAL Fixer render repair (weights: NVIDIA Open "
                "Model License; single-step diffusion in Docker).\n"
                f"{len(fixed)} frame(s) at "
                f"{frames.shape[2]}x{frames.shape[1]} in {elapsed:.1f}s "
                f"({elapsed / max(len(fixed), 1):.2f}s/frame incl. "
                f"load+warmup), timestep {timestep}.\n"
                "Known costs: mild softening; large frame-edge reveals are "
                "not outpainted (use band-layer frame outpainting for "
                "those).\n--- container log tail ---\n" + log_tail
            )
            return (out_t, report)
        finally:
            shutil.rmtree(exchange, ignore_errors=True)


class AtlasDeriveReliefMesh:
    """Continuous depth-following relief mesh — one job, so there's no
    geometry_mode/primitive_method combination that silently ignores this
    node's own widgets. Takes an already-estimated ATLAS_DEPTH_MAP
    (AtlasDepthMap) instead of an image, so it can share one depth pass with
    sibling derivation nodes wired from the same photo (see AtlasMergeGeometry
    to combine their outputs). Fits its own ground scale/backdrop directly
    (relief_mesh.estimate_ground_scale + depth_geometry.build_backdrop_primitive)
    rather than borrowing them from a primitive-fitting pass — a relief mesh
    alone never needed the wall/object derivation AtlasDeriveProjectionGeometry's
    relief_mesh mode runs internally just to get those two numbers.

    ``hole_mask`` mirrors `build_relief_mesh`'s own discarded hole/tear data
    (see `ReliefMesh.hole_mask`) - full source-image resolution, white where
    no triangle covers that pixel (sky/invalid/silhouette tear). This is the
    literal "where will Project show black" signal, not a heuristic.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "MASK")
    RETURN_NAMES = ("solve", "hole_mask")
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "relief_grid": ("INT", {"default": 128, "min": 16, "max": 4096,
                    "tooltip": "Mesh density (long-edge grid columns). Higher = fewer/"
                               "smaller torn holes on noisy AI-image depth, at the cost "
                               "of a larger mesh payload and a heavier viewport."}),
                "relief_quality": (["custom", "low", "medium", "high", "ultra"], {"default": "custom",
                    "tooltip": "Quick-pick override for relief_grid: low=64, medium=256, "
                               "high=512, ultra=1024. 'custom' leaves relief_grid as set above."}),
                "depth_edge_rel": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Relative depth jump that tears the mesh into a silhouette "
                               "hole. Lower = tears more readily; higher = tears less but "
                               "risks rubber-sheeting a real silhouette onto the background."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional external exclusion (e.g. a real sky segmentation from "
                               "AtlasSAM3Mask) which REPLACES the internal sky heuristic before "
                               "triangulation - so it must cover EVERYTHING you want gone. Any "
                               "resolution - resized to match depth. MIND THE POLARITY: masking "
                               "the BACKGROUND meshes the hero alone and repairs its own tears; "
                               "masking the SUBJECT leaves a subject-shaped hole and repairs "
                               "BEHIND it. For the second, just name BOTH in one "
                               "AtlasSAM3Mask - `concepts` is already a union, so "
                               "\"sky, machinery\" masks sky AND subject in one node (no "
                               "MaskComposite needed); this input replaces the heuristic, "
                               "it does not add to it. Check the segmenter's report: SAM3 "
                               "is open-vocabulary and the WORD decides everything - on one "
                               "real plate \"machine\" matched nothing while \"machinery\" "
                               "took 26.8% of frame."}),
                "outlier_mask": ("MASK", {
                    "tooltip": "Optional local depth outlier mask from AtlasDepthOutlierMask. "
                               "Those cells become explicit holes instead of stretched shards."}),
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "World-space edge tear threshold: a quad tears when its world edge "
                               "exceeds this x the expected local sample spacing. SEPARATE from "
                               "depth_edge_rel, and often the DOMINANT tear cause on deep / "
                               "narrow-FOV / interior scenes, where grazing walls and receding "
                               "floors span large world distances between adjacent samples and "
                               "trip the default 12x even where the surface is continuous. Raise "
                               "(20-40) to close spurious 'comb' tears; too high (>80) rubber-"
                               "sheets real foreground silhouettes onto the background."}),
                "sky_heuristic": ("BOOLEAN", {"default": True,
                    "tooltip": "Exclude above-horizon far/rough regions as sky before "
                               "triangulation. Correct for OUTDOOR plates; turn OFF for INTERIORS "
                               "(it otherwise eats the ceiling / vault / far wall as 'sky', "
                               "punching large holes). Automatically off when exclude_mask is "
                               "wired (an explicit mask always governs)."}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "0 = off. When set, a THIRD tear test: a triangle tears when its "
                               "corner surface-normals bend by more than this angle. Unlike "
                               "max_edge_factor (which trips on ANY grazing/receding surface), "
                               "this fires only where the surface ORIENTATION changes sharply - a "
                               "real crease or occlusion silhouette - so it tears genuine edges "
                               "while leaving a smoothly-receding wall/floor intact. Pair it with "
                               "a HIGHER max_edge_factor: raise mef to stop comb-tearing continuous "
                               "grazing surfaces, then set ~40-70 here to keep real silhouettes "
                               "torn. Lower = tears more readily."}),
                "quad_coherence": ("BOOLEAN", {"default": True,
                    "tooltip": "Reject both triangles when either half of a grid quad fails. "
                               "Prevents one surviving diagonal from becoming a stretched UV wedge."}),
                **LIVE_FILL_WIDGETS,
            },
        }

    _RELIEF_QUALITY_PRESETS = {"low": 64, "medium": 256, "high": 512, "ultra": 1024}

    def derive(self, solve, depth, relief_grid=128, relief_quality="custom",
               depth_edge_rel=0.5,
               exclude_mask=None, outlier_mask=None,
               max_edge_factor=12.0,
               sky_heuristic=True, normal_edge_deg=0.0, quad_coherence=True,
               live_fill_holes=False, live_fill_distance_m=0.0,
               live_fill_max_hole_edges=256,
               live_fill_edge_sawteeth=False):
        torch = _require_torch()
        np = _require_numpy()
        if relief_quality in self._RELIEF_QUALITY_PRESETS:
            relief_grid = self._RELIEF_QUALITY_PRESETS[relief_quality]
        from atlas_camera.core.depth_geometry import back_project_normals, build_backdrop_primitive
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale

        params = _solve_camera_params(solve, depth)
        if params is None:
            h, w = int(depth.image_height), int(depth.image_width)
            return (solve, torch.zeros(1, h, w, dtype=torch.float32))
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        resolved_exclude = _resolve_exclude_mask(exclude_mask, height, width)
        resolved_outliers = _resolve_exclude_mask(outlier_mask, height, width)
        if resolved_outliers is not None:
            resolved_exclude = (resolved_outliers if resolved_exclude is None else
                                (resolved_exclude | resolved_outliers))

        scale, ground_info = estimate_ground_scale(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=horizon_y)
        bp = back_project_normals(depth_map, view_matrix=extr.camera_view_matrix,
                                   fx=fx, fy=fy, cx=cx, cy=cy)
        scaled_depth = depth_map * scale
        backdrop = build_backdrop_primitive(
            bp=bp, scaled_depth=scaled_depth, valid_depth=bp.valid_depth,
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, scale=scale)
        mesh = build_relief_mesh(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            grid_long_edge=int(relief_grid), depth_edge_rel=float(depth_edge_rel),
            scale=scale, horizon_y=horizon_y, exclude_mask=resolved_exclude,
            max_edge_factor=float(max_edge_factor),
            normal_edge_deg=(float(normal_edge_deg) if float(normal_edge_deg) > 0 else None),
            quad_coherence=bool(quad_coherence),
            apply_sky_heuristic=(resolved_exclude is None) and bool(sky_heuristic),
            live_fill_holes=bool(live_fill_holes),
            live_fill_edge_sawteeth=bool(live_fill_edge_sawteeth))

        stats = {
            "ground_scale": scale, "ground_fit": ground_info,
            "relief_mesh": {
                "n_vertices": mesh.stats["n_vertices"],
                "n_faces": mesh.stats["n_faces"],
                "torn_fraction": mesh.stats.get("torn_fraction", 0.0),
                "quad_coherence": mesh.stats.get("quad_coherence", bool(quad_coherence)),
            },
        }
        # An explicit exclude_mask REPLACES the sky heuristic (line above), and
        # nothing said so out loud: a mask wired with the wrong polarity meshes
        # the sky and still reports success. Record coverage so the polarity is
        # readable after the fact, and say it on the console when the heuristic
        # is the thing being silenced (gate doctrine — no silent branch skip).
        if resolved_exclude is not None:
            suppressed = bool(sky_heuristic)
            stats["exclude_mask"] = {
                "frame_fraction": float(resolved_exclude.mean()),
                "sky_heuristic_suppressed": suppressed,
            }
            if suppressed:
                print(f"[Atlas] AtlasDeriveReliefMesh: exclude_mask covers "
                      f"{100.0 * float(resolved_exclude.mean()):.1f}% of frame and "
                      f"REPLACES the internal sky heuristic (it is not OR'd on top). "
                      f"Sky stays meshed unless this mask already contains it.")
        apply_live_mesh_repair(
            mesh,
            extr.camera_view_matrix,
            live_fill_holes=live_fill_holes,
            live_fill_distance_m=live_fill_distance_m,
            live_fill_max_hole_edges=live_fill_max_hole_edges,
            live_fill_edge_sawteeth=live_fill_edge_sawteeth,
            stats=stats,
            image_width=solve.camera.intrinsics.image_width,
            image_height=solve.camera.intrinsics.image_height,
        )
        prims = [backdrop, relief_mesh_primitive(mesh)]

        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "relief_grid": int(relief_grid), "relief_quality": relief_quality,
            "depth_edge_rel": float(depth_edge_rel), "max_edge_factor": float(max_edge_factor),
            "sky_heuristic": bool(sky_heuristic), "normal_edge_deg": float(normal_edge_deg),
            "quad_coherence": bool(quad_coherence),
            "live_fill_holes": bool(live_fill_holes),
            "live_fill_distance_m": float(live_fill_distance_m),
            "live_fill_max_hole_edges": int(live_fill_max_hole_edges),
            "live_fill_edge_sawteeth": bool(live_fill_edge_sawteeth),
            "derive_node": "AtlasDeriveReliefMesh",
        })
        hole_t = torch.from_numpy(mesh.hole_mask.astype(np.float32)).unsqueeze(0)
        return (out, hole_t)


class AtlasLiveMeshRepair:
    """🔧 SUPERSEDED — use AtlasPlanarHolePatch + AtlasRetopologizeLayer.

    Legacy tier: registered only when ATLAS_LEGACY_NODES is truthy, so saved
    graphs keep resolving for one migration cycle. The replacement chain is

        AtlasPlanarHolePatch(layer='*')  → scoped, reported hole filling
        AtlasRetopologizeLayer(boundary_smooth_iterations=N)  → silhouette

    which does what this node did with per-component gates, a report, and
    created-island masks instead of a global switch. Boundary smoothing was
    migrated verbatim; what does NOT carry over is the CUDA grid repair, the
    harmonic enclosed-hole cap, and the post-hoc stretch cull — see
    docs/FEATURE_AUDIT.md, which states that trade explicitly.

    Original description: apply PyTorch/CUDA 2D grid repair & 3D
    hole-fill/sawtooth repair to any solve or layer mesh; placeable anywhere
    downstream (after `AtlasBoundedBand`, `AtlasCleanPlateLayer`,
    `AtlasDepthLayerMask`, or `AtlasDeriveReliefMesh`).
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "repair"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "backend": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Repair backend. cuda = the PyTorch/CUDA 2D grid conv (recovered "
                               "from the mesh's UV lattice; GPU when available, else torch-CPU) — "
                               "live_fill_max_hole_edges sets how many fill rings it iterates, so "
                               "bigger = wider holes closed. cpu = the numpy face-soup ear-clip / "
                               "sawtooth-bridge (max_hole_edges capped at 256 to avoid freezing on "
                               "heavily-torn meshes; use cuda for large fills). auto = cuda when "
                               "torch is importable, else cpu.",
                }),
                **LIVE_FILL_WIDGETS,
                "cap_enclosed_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ZBrush-style Close Holes for ENCLOSED interior loops (cuda backend "
                               "only): a hole fully surrounded by mesh is capped even when it spans "
                               "a real depth jump — filled as a smooth HARMONIC MEMBRANE blending "
                               "the hole's own boundary depths (never a wall at the farthest depth). "
                               "Enclosure is channel-tolerant: dash-tear clusters reaching the "
                               "outside only through a <=2-cell corridor still count as enclosed. "
                               "SIZE cutoff: only holes whose boundary is within "
                               "live_fill_max_hole_edges are capped — a huge region that merely "
                               "happens to be enclosed stays open. live_fill_distance_m scopes it "
                               "by depth. Open silhouette/frame boundaries can never be capped.",
                }),
                "smooth_boundary": ("INT", {
                    "default": 8, "min": 0, "max": 50,
                    "tooltip": "Taubin-relax every open boundary loop this many iterations — rounds "
                               "the lattice-staircase jaggies on the mesh's outer silhouette without "
                               "shrinking it or touching interior detail. Moved vertices get their "
                               "projection UVs regenerated so 📽 Project stays aligned. 0 = off. "
                               "Works on both backends.",
                }),
                "remove_stretch_factor": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Post-hoc stretched-shard cull — the live twin of the layer nodes' "
                               "max_edge_factor: removes faces whose longest world edge exceeds this "
                               "many times the mesh's own local sample spacing (self-calibrated). "
                               "Lets you build the layer leniently (high max_edge_factor there) and "
                               "prune residual shards here instead. Runs BEFORE hole-fill/cap so "
                               "fresh cap walls are never culled. Lower = more aggressive. 0 = off. "
                               "Both backends.",
                }),
            },
        }

    def repair(self, solve, backend="auto", live_fill_holes=True, live_fill_distance_m=0.0,
               live_fill_max_hole_edges=256, live_fill_edge_sawteeth=True,
               cap_enclosed_holes=True, smooth_boundary=8, remove_stretch_factor=0.0):
        import copy

        import numpy as np

        from atlas_camera.exporters._layers import mesh_from_primitive

        solve_out = copy.deepcopy(solve)

        # Deprecation notice. This node has no STRING output and adding one
        # would break the positional-output contract, so the notice goes to
        # the console AND onto the solve, where AtlasDebugReport surfaces it.
        # Plain list[str] so _json_ready serializes it (solve JSON is a
        # contract).
        _notice = ("AtlasLiveMeshRepair is SUPERSEDED — use "
                   "AtlasPlanarHolePatch(layer='*') -> "
                   "AtlasRetopologizeLayer(boundary_smooth_iterations). "
                   "It is registered only because ATLAS_LEGACY_NODES is set.")
        print(f"[Atlas Camera] {_notice}")
        try:
            solve_out.debug_metadata.setdefault(
                "atlas_deprecations", []).append(_notice)
        except Exception:  # noqa: BLE001 — a notice must never break a graph
            pass

        view_matrix = solve.camera.extrinsics.camera_view_matrix

        # Resolve the pinhole intrinsics the CUDA grid path needs to back-project
        # newly-filled lattice cells. None → the grid path can't run, fall to cpu.
        intr = solve.camera.intrinsics
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or 0.0
        pp = intr.principal_point_px
        cx = intr.cx_px if intr.cx_px is not None else (pp[0] if pp else intr.image_width / 2.0)
        cy = intr.cy_px if intr.cy_px is not None else (pp[1] if pp else intr.image_height / 2.0)

        use_cuda = backend in ("auto", "cuda")
        if use_cuda:
            try:
                import torch  # noqa: F401
            except ImportError:
                if backend == "cuda":
                    use_cuda = False  # requested but unavailable → cpu fallback
                else:
                    use_cuda = False
        if fx <= 0 or fy <= 0:
            use_cuda = False  # no usable intrinsics for ray back-projection

        def _repair_prim_in_place(prim):
            """Reconstruct the ReliefMesh from the primitive's flattened metadata
            (relief_mesh_primitive's inverse), run the chosen repair, then write
            the changed arrays back into the SAME metadata dict."""
            meta = prim.metadata or {}
            if prim.primitive_type != "mesh" or meta.get("source") != "depth_relief_mesh":
                return
            mesh = mesh_from_primitive(prim)
            if mesh is None:
                return
            # mesh_from_primitive omits hole_mask; give the cpu path a 0-sized one
            # so _hole_mask_after_fill early-returns instead of dereferencing None.
            mesh.hole_mask = np.zeros((0, 0), dtype=bool)
            n_verts_before = len(mesh.vertices)
            n_faces_before = len(mesh.faces)

            # Shard cull FIRST: exposes clean boundaries for the fill, and by
            # ordering alone guarantees fresh cap walls are never culled.
            if float(remove_stretch_factor) > 0:
                from atlas_camera.core.mesh_repair import remove_stretched_faces
                remove_stretched_faces(
                    mesh, view_matrix=view_matrix,
                    max_edge_factor=float(remove_stretch_factor))

            if use_cuda:
                from atlas_camera.core.mesh_repair import repair_relief_mesh_grid_cuda
                repair_relief_mesh_grid_cuda(
                    mesh, view_matrix=view_matrix,
                    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                    image_width=int(intr.image_width), image_height=int(intr.image_height),
                    fill_holes=bool(live_fill_holes),
                    fill_sawteeth=bool(live_fill_edge_sawteeth),
                    depth_far_m=float(live_fill_distance_m),
                    max_hole_edges=int(live_fill_max_hole_edges),
                    cap_enclosed=bool(cap_enclosed_holes),
                )
            else:
                # The CPU topology fill ear-clips whole boundary loops (O(n^2)
                # per loop). On a heavily-torn mesh a large max_hole_edges makes
                # it attempt many huge non-convex loops and freeze, so cap it —
                # for large fills use backend=cuda (bounded, iterative, fast).
                cpu_max_edges = min(int(live_fill_max_hole_edges), 256)
                apply_live_mesh_repair(
                    mesh, view_matrix,
                    live_fill_holes=bool(live_fill_holes),
                    live_fill_distance_m=float(live_fill_distance_m),
                    live_fill_max_hole_edges=cpu_max_edges,
                    live_fill_edge_sawteeth=bool(live_fill_edge_sawteeth),
                )

            # Boundary smoothing runs on BOTH backends (topology-level, not
            # grid-level): rounds the lattice-staircase silhouette jaggies and
            # regenerates the moved vertices' projection UVs when intrinsics
            # are usable (fx/fy > 0), so 📽 Project stays aligned.
            n_moved = 0
            if int(smooth_boundary) > 0:
                from atlas_camera.core.mesh_repair import smooth_boundary_loops
                n_moved = smooth_boundary_loops(
                    mesh, iterations=int(smooth_boundary),
                    view_matrix=view_matrix,
                    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                    image_width=int(intr.image_width),
                    image_height=int(intr.image_height),
                )

            if (len(mesh.faces) == n_faces_before
                    and len(mesh.vertices) == n_verts_before and n_moved == 0):
                return  # nothing changed
            meta["faces"] = np.asarray(mesh.faces).reshape(-1).astype(np.int64).tolist()
            meta["n_faces"] = int(len(mesh.faces))
            if len(mesh.vertices) != n_verts_before or n_moved > 0:
                meta["vertices"] = np.round(
                    np.asarray(mesh.vertices, dtype=np.float64).reshape(-1), 3).tolist()
                meta["uvs"] = np.round(
                    np.asarray(mesh.uvs, dtype=np.float64).reshape(-1), 4).tolist()
                meta["n_vertices"] = int(len(mesh.vertices))
                er = getattr(mesh, "edge_risk", None)
                if er is not None:
                    meta["edge_risk"] = np.round(
                        np.asarray(er, dtype=np.float64).reshape(-1), 3).tolist()
                elif meta.get("edge_risk") and len(meta["edge_risk"]) < len(mesh.vertices):
                    # mesh_from_primitive doesn't rebuild edge_risk, so a grown
                    # vertex list would leave the serialized field short — pad
                    # added (fill/cap) verts at full boundary risk 1.0 so the
                    # viewport's per-vertex coverage field never misindexes.
                    meta["edge_risk"] = (list(meta["edge_risk"])
                                         + [1.0] * (len(mesh.vertices) - len(meta["edge_risk"])))
            prim.metadata = meta

        scene = getattr(solve_out, "projection_scene", None)
        for prim in (getattr(scene, "proxy_geometry", None) or []):
            _repair_prim_in_place(prim)

        for src in (getattr(solve_out, "projection_sources", None) or []):
            for prim in (getattr(src, "proxy_geometry", None) or []):
                _repair_prim_in_place(prim)

        return (solve_out,)


class AtlasRetopologizeLayer:
    """🔷 Live retopology for ONE solve layer (or all) — before the viewport.

    Applies the export nodes' retopo passes (quad remesh / quadric decimate /
    Taubin smooth, `core.mesh_retopo`) to the LIVE relief mesh serialized on a
    solve, so the simplified topology shows in 📽 Project and rides every
    export — not just the written OBJ. This is a deliberate revision of the
    old "export-only" doctrine: `regenerate_projective_uvs` (run inside
    `apply_retopo` for vertex-count-changing methods, with the layer's OWN
    camera) restores the 1:1 vertex-UV projection contract exactly, and
    smoothing deliberate tears is the point of reaching for this node.

    `layer` selects the target: "" = the PRIMARY scene relief mesh, a
    ProjectionSource name ("bg", "machine", ...) = that layer only,
    "*" = every relief mesh. Missing optional deps (pyinstantmeshes /
    trimesh + scipy / fast-simplification) degrade soft — the report carries
    the pip hint and the solve passes through untouched.

    edge_risk note: a remesh that changes the vertex count consumes the
    serialized per-vertex coverage field (it indexes the OLD ordering) — it is
    cleared for that layer, which the viewport tolerates (slightly harder edge
    feather there). Taubin smooth keeps counts, so it keeps edge_risk.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "retopo"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "layer": ("STRING", {"default": "",
                    "tooltip": "Which mesh to retopologize: blank = the primary scene relief "
                               "mesh; a layer name (AtlasCleanPlateLayer `name`, e.g. 'bg') = "
                               "that projection source only; '*' = every relief mesh."}),
                "method": (["off", "quad", "decimate", "smooth"], {"default": "decimate",
                    "tooltip": "quad = Instant Meshes quad remesh (pyinstantmeshes); decimate = "
                               "quadric decimation (trimesh + fast-simplification); smooth = "
                               "trimesh Taubin relax (topology unchanged, UVs preserved). Same "
                               "passes as the Maya/Nuke layer exporters, applied LIVE with the "
                               "layer's own camera regenerating projection UVs."}),
                "target_vertex_count": ("INT", {"default": 2000, "min": 100, "max": 200000,
                    "tooltip": "Vertex budget for quad/decimate (ignored by smooth)."}),
                "smooth_iterations": ("INT", {"default": 0, "min": 0, "max": 50,
                    "tooltip": "Taubin iterations (smooth method, and post-smooth for quad)."}),
                "crease_angle": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 90.0,
                    "tooltip": "Quad remesh crease preservation angle (degrees)."}),
                "pure_quad": ("BOOLEAN", {"default": False,
                    "tooltip": "Quad remesh: force pure quads (else quad-dominant)."}),
                "boundary_smooth_iterations": ("INT", {"default": 0, "min": 0, "max": 50,
                    "tooltip": "Taubin-relax every open boundary loop this many iterations "
                               "AFTER the retopo pass — rounds the lattice-staircase jaggies "
                               "on the silhouette without shrinking it or touching interior "
                               "detail. Moved vertices get their projection UVs regenerated "
                               "with THIS layer's camera, so 📽 Project stays aligned; if the "
                               "layer camera has no usable intrinsics the pass is SKIPPED "
                               "(and said so in the report) rather than leaving UVs stale. "
                               "0 = off. Migrated from AtlasLiveMeshRepair's smooth_boundary."}),
            },
        }

    def retopo(self, solve, layer="", method="decimate", target_vertex_count=2000,
               smooth_iterations=0, crease_angle=30.0, pure_quad=False,
               boundary_smooth_iterations=0, **_extra):
        import copy

        import numpy as np

        from atlas_camera.exporters._layers import (
            _retopologize_layer_mesh,
            mesh_from_primitive,
        )

        solve_out = copy.deepcopy(solve)
        sel = str(layer or "").strip()
        lines = []

        def _do(prim, camera, name):
            meta = prim.metadata or {}
            if prim.primitive_type != "mesh" or meta.get("source") != "depth_relief_mesh":
                return
            mesh = mesh_from_primitive(prim)
            if mesh is None:
                lines.append(f"{name}: empty mesh — skipped")
                return
            n_v0 = len(mesh.vertices)
            try:
                report = _retopologize_layer_mesh(
                    mesh, camera, method=str(method),
                    target_vertex_count=int(target_vertex_count),
                    smooth_iterations=int(smooth_iterations),
                    crease_angle=float(crease_angle), pure_quad=bool(pure_quad))
            except (ImportError, ValueError, RuntimeError) as exc:
                # A retopo node must never kill the graph — report the pip
                # hint / config problem and pass the solve through untouched.
                lines.append(f"{name}: SKIPPED — {exc}")
                return

            # Boundary smoothing runs REGARDLESS of the retopo verdict:
            # method="off" + boundary_smooth_iterations>0 is the "don't
            # retopo, just round the silhouette" configuration, and
            # apply_retopo reports changed=False for "off" by construction.
            n_moved = 0
            iters = int(boundary_smooth_iterations)
            if iters > 0:
                intr = getattr(camera, "intrinsics", None)
                extr = getattr(camera, "extrinsics", None)
                width = int(getattr(intr, "image_width", 0) or 0)
                height = int(getattr(intr, "image_height", 0) or 0)
                fx = float(getattr(intr, "fx_px", 0.0) or 0.0)
                # Same fallbacks the layer exporter uses (_layers.py): a
                # constructed patch camera often carries fx_px but no fy_px/cx/cy.
                fy = float(getattr(intr, "fy_px", 0.0) or fx)
                cx = getattr(intr, "cx_px", None)
                cy = getattr(intr, "cy_px", None)
                cx = float(cx) if cx is not None else width / 2.0
                cy = float(cy) if cy is not None else height / 2.0
                view = getattr(extr, "camera_view_matrix", None)
                if view is not None and fx > 0 and fy > 0 and width > 1 and height > 1:
                    from atlas_camera.core.mesh_repair import smooth_boundary_loops
                    n_moved = smooth_boundary_loops(
                        mesh, iterations=iters, view_matrix=view,
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        image_width=width, image_height=height)
                    if n_moved == 0:
                        lines.append(f"{name}: boundary smoothing moved 0 verts "
                                     "(no open loop >= 8 verts — watertight or tiny)")
                else:
                    # Never move vertices we cannot re-register: stale projective
                    # UVs would silently break 📽 Project.
                    lines.append(
                        f"{name}: boundary smoothing SKIPPED — layer camera has no "
                        f"usable intrinsics (fx={fx:g} fy={fy:g} {width}x{height}); "
                        "moving verts would leave projective UVs stale")

            if not report.get("changed") and n_moved == 0:
                lines.append(f"{name}: unchanged — {report.get('note', '')}")
                return
            meta["vertices"] = np.round(
                np.asarray(mesh.vertices, dtype=np.float64).reshape(-1), 3).tolist()
            meta["faces"] = np.asarray(mesh.faces).reshape(-1).astype(np.int64).tolist()
            meta["uvs"] = np.round(
                np.asarray(mesh.uvs, dtype=np.float64).reshape(-1), 4).tolist()
            meta["n_vertices"] = int(len(mesh.vertices))
            meta["n_faces"] = int(len(mesh.faces))
            if len(mesh.vertices) != n_v0 and meta.get("edge_risk"):
                meta["edge_risk"] = []  # indexes the OLD vertex order — consumed by the remesh
            prim.metadata = meta
            lines.append(
                f"{name}: {report.get('method')} "
                f"{report.get('in_verts')}->{report.get('out_verts')} verts, "
                f"{report.get('in_faces')}->{report.get('out_faces')} faces — "
                f"{report.get('note', '')}"
                + (f" +boundary smooth x{iters} ({n_moved} verts moved, UVs "
                   "regenerated)" if n_moved else ""))

        if sel in ("", "*"):
            scene = getattr(solve_out, "projection_scene", None)
            for prim in (getattr(scene, "proxy_geometry", None) or []):
                _do(prim, solve_out.camera, "primary")
        for src in (getattr(solve_out, "projection_sources", None) or []):
            if sel == "*" or (sel and getattr(src, "name", "") == sel):
                for prim in (getattr(src, "proxy_geometry", None) or []):
                    _do(prim, src.camera, getattr(src, "name", "") or "layer")
        if sel and sel != "*" and not lines:
            names = [getattr(s, "name", "?")
                     for s in (getattr(solve_out, "projection_sources", None) or [])]
            lines.append(f"layer '{sel}' not found — available: "
                         f"{', '.join(names) if names else '(none)'}; solve passed through")
        return (solve_out, "\n".join(lines) or "nothing to retopologize")


class AtlasBlenderOrganicFill:
    """🔬 Close organic tears in numpy, then place the fill with headless Blender.

    EXPERIMENTAL: needs a Blender install, which Atlas does not bundle. Selecting
    it without one raises with the download link and every path probed.

    The division of labour, both halves measured on real organic plates:

      numpy `core.mesh_voxel.voxel_remesh` closes every interior tear in ~2.5s
      (22/58/51 tears -> 0 on three plates) and then sits off the measured
      surface. Blender's BVH shrinkwrap puts it back EXACTLY on that surface
      (p95 0.0000m against a calibrated 0.0 floor) in ~2.5s.

    Neither half is optional and neither tool does the other's job. Blender
    cannot close a lateral tear at all — solidify thickens PERPENDICULAR to the
    surface and walls the hole's rim, so volumetric remesh preserves it as a
    tube; measured, the hole only vanished when the voxel was as large as the
    hole itself, collapsing the patch from 66 vertices to 16. numpy succeeds
    only because it works in depth-image space, where a hole is a 2D region to
    flood-fill.

    THE GATE MEASURES MOVEMENT, NOT FINAL POSITION. Distance-to-measured-surface
    is zero BY CONSTRUCTION here — NEAREST_SURFACEPOINT projects onto the
    surface — so gating on it would pass everything. What that cannot see is a
    vertex slid ALONG the surface to the wrong place, so the gate is how far
    shrinkwrap had to DRAG the fill. Measured medians are 0.47-0.64x the median
    edge, so the 3.0x default has real headroom rather than being fitted to pass.

    REJECTION IS A THIRD OUTCOME. Over-limit movement leaves the layer untouched
    and says so with its numbers — a raise would kill a long graph, a silent
    accept would ship bad geometry.

    Route this from `core.fill_policy`: it is the ORGANIC family's filler, and a
    planar scene should get a fitted construction instead.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "fill"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "layer": ("STRING", {
                    "default": "",
                    "tooltip": "Layer name, '' for the primary, '*' for every "
                               "layer. Same selector as AtlasRetopologizeLayer."}),
                "blender_path": ("STRING", {
                    "default": "",
                    "tooltip": "Blender executable. Empty = ATLAS_BLENDER_PATH, "
                               "then PATH, then the platform install dirs "
                               "(newest version wins)."}),
                "voxel_grid": ("INT", {
                    "default": 192, "min": 32, "max": 512,
                    "tooltip": "Raster long edge for the numpy closure pass. "
                               "Higher keeps more detail and costs more."}),
                "thickness_vox": ("INT", {
                    "default": 3, "min": 1, "max": 16,
                    "tooltip": "Shell thickness in depth slabs for the closure "
                               "pass."}),
                "smooth_iterations": ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": "Taubin smoothing inside the closure pass."}),
                "shrinkwrap_limit_scale": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 50.0, "step": 0.5,
                    "tooltip": "Search radius for the snap, in median edge "
                               "lengths. Scale-free so a value tuned on a 10m "
                               "interior still means something on a 2km vista."}),
                "max_move_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "REJECT if shrinkwrap had to move the fill more "
                               "than this many median edges. Measured medians on "
                               "real plates are 0.47-0.64x."}),
                "timeout_s": ("INT", {
                    "default": 600, "min": 30, "max": 7200,
                    "tooltip": "Per-layer Blender budget."}),
                "assessment_json": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "AtlasAssessImage's settings_json. Used for the "
                               "ROUTE decision — this node is the ORGANIC "
                               "family's filler, so a planar scene is skipped "
                               "and fill_occluded=false is honoured as a veto. "
                               "Ignored unless use_assessment is on."}),
                "use_assessment": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Let the assessment decide WHETHER to fill each "
                               "layer. Off by default: an inferred route "
                               "silently skipping a layer is worse than an "
                               "artist choosing to run it."}),
            },
        }

    def fill(self, solve, layer="", blender_path="", voxel_grid=192,
             thickness_vox=3, smooth_iterations=8, shrinkwrap_limit_scale=4.0,
             max_move_scale=3.0, timeout_s=600, assessment_json="",
             use_assessment=False):
        import copy

        from atlas_camera.blender import (
            gate_movement, median_edge_length, shrinkwrap_patch,
        )
        from atlas_camera.core.mesh_voxel import voxel_remesh
        from atlas_camera.exporters._layers import mesh_from_primitive
        np = _require_numpy()

        solve_out = copy.deepcopy(solve)
        sel = str(layer or "").strip()
        lines: list[str] = []

        # --- the assessment decides the ROUTE, never the geometry -----------
        # What fill_policy contributes here is which family a layer belongs to,
        # plus the fill_occluded veto. Its numeric organic settings
        # (live_fill_max_hole_edges, retopo_method) were designed for the
        # mesh-repair path and do NOT map onto this node's voxel/shrinkwrap
        # knobs — inventing a correspondence would be worse than not wiring it,
        # so only boundary_smooth_iterations, which genuinely corresponds, is
        # taken.
        payload = None
        if use_assessment and str(assessment_json or "").strip():
            import json
            try:
                payload = json.loads(assessment_json)
            except (ValueError, TypeError) as exc:
                lines.append(f"assessment_json unreadable ({exc}); "
                             "falling back to the widgets for every layer")
                payload = None

        def _plan_for(name):
            if payload is None:
                return None
            from atlas_camera.core.fill_policy import policy_from_assessment
            return policy_from_assessment(payload, layer=("" if name == "primary"
                                                          else name))

        def _do(prim, camera, name):
            meta = prim.metadata or {}
            if prim.primitive_type != "mesh" or meta.get("source") != "depth_relief_mesh":
                return

            smooth = int(smooth_iterations)
            plan = _plan_for(name)
            if plan is not None:
                from atlas_camera.core.fill_policy import ROUTE_ORGANIC
                if plan.route != ROUTE_ORGANIC:
                    lines.append(
                        f"{name}: SKIPPED by assessment — route={plan.route} "
                        f"(scene_type={plan.scene_type!r}). This node is the "
                        "ORGANIC family's filler; "
                        + ("a planar scene wants a fitted construction "
                           "instead. " if plan.route == "planar" else
                           "the layer was vetoed. ")
                        + (plan.reasons[0] if plan.reasons else ""))
                    return
                # The one setting that genuinely corresponds.
                smooth = int(plan.boundary_smooth_iterations)
            mesh = mesh_from_primitive(prim)
            if mesh is None:
                lines.append(f"{name}: empty mesh — skipped")
                return
            verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
            n_v0 = len(verts)

            intr = getattr(camera, "intrinsics", None)
            extr = getattr(camera, "extrinsics", None)
            width = int(getattr(intr, "image_width", 0) or 0)
            height = int(getattr(intr, "image_height", 0) or 0)
            fx = float(getattr(intr, "fx_px", 0.0) or 0.0)
            fy = float(getattr(intr, "fy_px", 0.0) or fx)
            cx = getattr(intr, "cx_px", None)
            cy = getattr(intr, "cy_px", None)
            cx = float(cx) if cx is not None else width / 2.0
            cy = float(cy) if cy is not None else height / 2.0
            view = getattr(extr, "camera_view_matrix", None)
            if view is None or fx <= 0 or fy <= 0 or width <= 1 or height <= 1:
                # Refuse BEFORE launching Blender. Unlike a smoothing pass there
                # is no "keep the old UVs" fallback here: the remesh replaces
                # every vertex, so without intrinsics the new ones have no UVs
                # at all and 📽 Project would render nothing.
                lines.append(
                    f"{name}: SKIPPED — layer camera has no usable intrinsics "
                    f"(fx={fx:g} fy={fy:g} {width}x{height}); the remesh "
                    "replaces every vertex and projective UVs could not be "
                    "regenerated")
                return

            med = median_edge_length(verts, faces)
            try:
                v1, f1 = voxel_remesh(
                    verts, faces, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                    image_width=width, image_height=height,
                    grid=int(voxel_grid), thickness_vox=int(thickness_vox),
                    smooth_iterations=smooth, close_holes=True)
            except (ValueError, RuntimeError) as exc:
                lines.append(f"{name}: SKIPPED — closure pass failed: {exc}")
                return

            try:
                got = shrinkwrap_patch(
                    v1, f1, verts, faces, blender_path=str(blender_path),
                    limit_scale=float(shrinkwrap_limit_scale),
                    timeout_s=int(timeout_s))
            except RuntimeError as exc:
                # Blender missing / too old / recipe failed. Never kill the
                # graph — report and pass the layer through.
                lines.append(f"{name}: SKIPPED — {exc}")
                return

            rep = got.get("report") or {}
            ok, why = gate_movement(float(rep.get("moved_median_m", 0.0)),
                                    float(rep.get("moved_max_m", 0.0)), med,
                                    max_move_scale=float(max_move_scale))
            if not ok:
                lines.append(f"{name}: {why}")
                return

            new_v = np.asarray(got["vertices"], dtype=np.float64).reshape(-1, 3)
            new_f = np.asarray(got["faces"], dtype=np.int64).reshape(-1, 3)
            from atlas_camera.core.mesh_retopo import regenerate_projective_uvs
            new_uv = regenerate_projective_uvs(
                new_v, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                image_width=width, image_height=height)

            meta["vertices"] = np.round(new_v.reshape(-1), 3).tolist()
            meta["faces"] = new_f.reshape(-1).astype(np.int64).tolist()
            meta["uvs"] = np.round(
                np.asarray(new_uv, dtype=np.float64).reshape(-1), 4).tolist()
            meta["n_vertices"] = int(len(new_v))
            meta["n_faces"] = int(len(new_f))
            if len(new_v) != n_v0 and meta.get("edge_risk"):
                meta["edge_risk"] = []   # indexes the OLD vertex order
            prim.metadata = meta
            lines.append(
                f"{name}: {n_v0}->{len(new_v)} verts, {len(faces)}->{len(new_f)} "
                f"faces; {why}"
                + (f" [assessment: {plan.scene_type} -> organic, smooth x{smooth}]"
                   if plan is not None else ""))

        if sel in ("", "*"):
            scene = getattr(solve_out, "projection_scene", None)
            for prim in (getattr(scene, "proxy_geometry", None) or []):
                _do(prim, solve_out.camera, "primary")
        for src in (getattr(solve_out, "projection_sources", None) or []):
            if sel == "*" or (sel and getattr(src, "name", "") == sel):
                for prim in (getattr(src, "proxy_geometry", None) or []):
                    _do(prim, src.camera, getattr(src, "name", "") or "layer")
        if sel and sel != "*" and not lines:
            names = [getattr(s, "name", "?")
                     for s in (getattr(solve_out, "projection_sources", None) or [])]
            lines.append(f"layer '{sel}' not found — available: "
                         f"{', '.join(names) if names else '(none)'}; "
                         "solve passed through")
        return (solve_out, "\n".join(lines) or "nothing to fill")


class AtlasPlanarHolePatch:
    """Fit conservative local planes into selected relief-mesh holes.

    The node recovers the relief mesh's image-space lattice, groups selected
    missing quads into enclosed components, finds an agreeable surrounding
    normal cluster per island, and averages its normals into one patch plane.
    Accepted patches reuse perimeter indices and generate their interior on
    exact camera rays, so they are part of the relief mesh with projective UVs
    rather than separate wall/box primitives.

    Run this before AtlasRetopologizeLayer.  Retopology will then see the
    relief plus accepted patches as one mesh.  Open frame/sky gaps and
    ambiguous multi-surface boundaries pass through unchanged for a clean
    plate or hidden-geometry layer to handle.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("solve", "remaining_holes", "report", "created_islands")
    FUNCTION = "patch"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "hole_mask": ("MASK", {
                    "tooltip": "Selected relief holes to patch — usually the hole_mask "
                               "output of AtlasDeriveReliefMesh or AtlasCleanPlateLayer, "
                               "optionally through a scope mask. LEAVE UNWIRED to treat "
                               "EVERY hole in the target layer(s) as a candidate, which "
                               "is what you want for a whole-solve sweep; the "
                               "enclosed_only / max_hole_fraction / max_components gates "
                               "still bound what actually gets filled.",
                }),
                "layer": ("STRING", {
                    "default": "",
                    "tooltip": "Blank = primary relief mesh; a ProjectionSource name "
                               "= that layer's relief mesh; '*' = every relief mesh "
                               "(primary + all sources), chaining the solve through "
                               "one pass per layer.",
                }),
                "ring_cells": ("INT", {
                    "default": 2, "min": 1, "max": 12,
                    "tooltip": "How many valid grid rings around each hole supply "
                               "normal and plane-fit support.",
                }),
                "max_components": ("INT", {
                    "default": 64, "min": 1, "max": 1024,
                    "tooltip": "Maximum eligible hole islands to fit, smallest first. "
                               "Frame/oversize rejections do not consume this budget.",
                }),
                "normal_tolerance_deg": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Maximum boundary-normal spread. Lower is safer at "
                               "corners; higher accepts rougher/curved surfaces.",
                }),
                "max_plane_error_m": ("FLOAT", {
                    "default": 0.15, "min": 0.001, "max": 10.0, "step": 0.01,
                    "tooltip": "Maximum 95th-percentile residual in metres for the "
                               "densest plane-position band among agreeing normals.",
                }),
                "max_hole_fraction": ("FLOAT", {
                    "default": 0.05, "min": 0.0001, "max": 1.0, "step": 0.005,
                    "tooltip": "Largest accepted component as a fraction of relief "
                               "grid cells. Keeps large sky/background gaps out.",
                }),
                "enclosed_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reject components touching the image frame. Recommended: "
                               "frame/sky openings are not occlusion holes.",
                }),
                "min_normal_support_fraction": ("FLOAT", {
                    "default": 0.30, "min": 0.10, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum share of surrounding valid normals that must "
                               "agree on one plane angle. Lower values allow one-sided "
                               "facade/occlusion repairs. Agreeing normals are averaged; "
                               "max_plane_error_m then checks their positional fit.",
                }),
                "max_patch_edge_factor": ("FLOAT", {
                    "default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Reject a generated island when its longest triangle "
                               "edge exceeds this multiple of its source-pixel span "
                               "times the boundary's median metres-per-pixel. Prevents "
                               "grazing camera-ray spikes at any image/camera scale.",
                }),
                "max_patch_depth_factor": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Reject a patch whose nearest/farthest camera depth "
                               "extends beyond this multiple of the fitted local "
                               "support depth band. Catches forward and backward Z "
                               "needles independently of the edge-length gate.",
                }),
            },
        }

    def patch(self, solve, hole_mask=None, layer="", **kw):
        """Dispatch: one named layer / the primary scene, or '*' for all.

        ``layer="*"`` sweeps the primary scene mesh and every projection
        source that carries one, chaining the solve through each pass — the
        whole-solve behaviour AtlasLiveMeshRepair used to provide, without
        needing one patch node per layer.

        Mask algebra for the sweep: a hole filled by one layer is gone from
        that layer's ``remaining`` but still present in the others', so the
        global remaining is the INTERSECTION and created is the UNION. That
        keeps ``created == input & ~remaining`` true across the sweep, exactly
        as it is for a single target.
        """
        sel = str(layer or "").strip()
        if sel != "*":
            return self._patch_one(solve, hole_mask, sel, **kw)

        torch = _require_torch()
        np = _require_numpy()
        targets = [""]
        targets += [name for name in
                    (getattr(src, "name", "") for src in
                     (getattr(solve, "projection_sources", None) or []))
                    if name]

        # Reference frame for the combined masks: the SOLVE's own camera.
        # Layers do NOT share a resolution — a frame-outpainted clean plate is
        # padded (measured live: 4640x7808 beside the primary's 4512x7680) and
        # each _patch_one call returns its mask at ITS layer's plate size, so
        # combining them raw is a broadcast error.
        ref_intr = getattr(solve.camera, "intrinsics", None)
        ref_h = int(getattr(ref_intr, "image_height", 0) or 0)
        ref_w = int(getattr(ref_intr, "image_width", 0) or 0)

        def _to_ref(arr):
            a = np.asarray(arr)
            plane = a[0] if a.ndim == 3 else a
            if ref_h > 1 and ref_w > 1 and plane.shape != (ref_h, ref_w):
                from atlas_camera.core.solver import _resize_depth
                plane = _resize_depth(plane.astype(np.float64), ref_w, ref_h)
            return plane[None, ...].astype(np.float32)

        current = solve
        remaining_acc = None
        created_acc = None
        lines = []
        for name in targets:
            current, remaining_t, report, created_t = self._patch_one(
                current, hole_mask, name, **kw)
            label = name or "primary"
            if "no relief mesh" in report:
                continue                      # not every layer carries geometry
            lines.append(f"[{label}] " + report.replace("\n", "\n    "))
            rem = _to_ref(remaining_t.detach().cpu().numpy())
            cre = _to_ref(created_t.detach().cpu().numpy())
            remaining_acc = rem if remaining_acc is None else np.minimum(remaining_acc, rem)
            created_acc = cre if created_acc is None else np.maximum(created_acc, cre)

        if remaining_acc is None:
            return (current, torch.zeros(1, 1, 1),
                    "no layer carries a relief mesh — solve passed through",
                    torch.zeros(1, 1, 1))
        header = f"swept {len(lines)} layer(s): " + ", ".join(
            ln.split("]")[0][1:] for ln in lines)
        return (current, torch.from_numpy(remaining_acc), header + "\n" + "\n".join(lines),
                torch.from_numpy(created_acc))

    def _patch_one(self, solve, hole_mask, layer="", ring_cells=2, max_components=64,
                   normal_tolerance_deg=25.0, max_plane_error_m=0.15,
                   max_hole_fraction=0.05, enclosed_only=True,
                   min_normal_support_fraction=0.30,
                   max_patch_edge_factor=20.0,
                   max_patch_depth_factor=2.0):
        torch = _require_torch()
        np = _require_numpy()
        from atlas_camera.core.planar_hole_patch import (
            PlanarHolePatchConfig,
            patch_planar_holes,
        )
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.exporters._layers import mesh_from_primitive

        out = copy.deepcopy(solve)
        intr = out.camera.intrinsics
        width = int(intr.image_width or 0)
        height = int(intr.image_height or 0)
        fx = float(intr.fx_px or 0.0)
        fy = float(intr.fy_px or fx)
        cx = float(intr.cx_px if intr.cx_px is not None else width / 2.0)
        cy = float(intr.cy_px if intr.cy_px is not None else height / 2.0)
        resolved = _resolve_exclude_mask(hole_mask, height, width)
        if resolved is None:
            # No mask wired: every hole in this layer is a candidate. The
            # per-component gates (enclosed_only, max_hole_fraction,
            # max_components, edge/depth factors) still decide what is
            # actually safe to fill, so this is "sweep the layer", not
            # "fill everything".
            resolved = np.ones((height, width), dtype=bool)
        remaining_t = torch.from_numpy(resolved.astype(np.float32)).unsqueeze(0)
        created_t = torch.zeros_like(remaining_t)
        if min(width, height) <= 1 or min(fx, fy) <= 0.0:
            return (
                out, remaining_t,
                "invalid camera/image dimensions — solve passed through",
                created_t,
            )

        target_name = str(layer or "").strip()
        camera = out.camera
        if target_name:
            source = next(
                (src for src in (getattr(out, "projection_sources", None) or [])
                 if getattr(src, "name", "") == target_name),
                None,
            )
            if source is None:
                names = [getattr(src, "name", "?")
                         for src in (getattr(out, "projection_sources", None) or [])]
                return (
                    out, remaining_t,
                    f"layer '{target_name}' not found — available: "
                    f"{', '.join(names) if names else '(none)'}",
                    created_t,
                )
            primitives = source.proxy_geometry
            camera = source.camera
        else:
            scene = getattr(out, "projection_scene", None)
            primitives = getattr(scene, "proxy_geometry", None) if scene is not None else None

        target_intr = camera.intrinsics
        width = int(target_intr.image_width or width)
        height = int(target_intr.image_height or height)
        fx = float(target_intr.fx_px or fx)
        fy = float(target_intr.fy_px or fx)
        cx = float(target_intr.cx_px if target_intr.cx_px is not None else width / 2.0)
        cy = float(target_intr.cy_px if target_intr.cy_px is not None else height / 2.0)
        resolved = _resolve_exclude_mask(hole_mask, height, width)
        if resolved is None:
            # No mask wired: every hole in this layer is a candidate. The
            # per-component gates (enclosed_only, max_hole_fraction,
            # max_components, edge/depth factors) still decide what is
            # actually safe to fill, so this is "sweep the layer", not
            # "fill everything".
            resolved = np.ones((height, width), dtype=bool)
        remaining_t = torch.from_numpy(resolved.astype(np.float32)).unsqueeze(0)
        created_t = torch.zeros_like(remaining_t)
        if min(width, height) <= 1 or min(fx, fy) <= 0.0:
            return (
                out, remaining_t,
                "invalid target camera/image dimensions — solve passed through",
                created_t,
            )

        primitive_index = next(
            (i for i, prim in enumerate(primitives or [])
             if prim.primitive_type == "mesh"
             and (prim.metadata or {}).get("source") == "depth_relief_mesh"),
            None,
        )
        if primitive_index is None:
            return (
                out, remaining_t,
                "target has no relief mesh — solve passed through",
                created_t,
            )
        primitive = primitives[primitive_index]
        mesh = mesh_from_primitive(primitive)
        if mesh is None:
            return (
                out, remaining_t,
                "target relief mesh is empty — solve passed through",
                created_t,
            )
        edge_risk = (primitive.metadata or {}).get("edge_risk") or []
        if len(edge_risk) == len(mesh.vertices):
            mesh.edge_risk = np.asarray(edge_risk, dtype=np.float32)

        cfg = PlanarHolePatchConfig(
            ring_cells=int(ring_cells),
            max_components=int(max_components),
            normal_tolerance_deg=float(normal_tolerance_deg),
            max_plane_error_m=float(max_plane_error_m),
            max_hole_fraction=float(max_hole_fraction),
            enclosed_only=bool(enclosed_only),
            min_normal_support_fraction=float(min_normal_support_fraction),
            max_patch_edge_factor=float(max_patch_edge_factor),
            max_patch_depth_factor=float(max_patch_depth_factor),
        )
        try:
            patched, remaining, report = patch_planar_holes(
                mesh,
                resolved,
                view_matrix=camera.extrinsics.camera_view_matrix,
                fx=float(camera.intrinsics.fx_px or fx),
                fy=float(camera.intrinsics.fy_px or fy),
                cx=float(camera.intrinsics.cx_px
                         if camera.intrinsics.cx_px is not None else cx),
                cy=float(camera.intrinsics.cy_px
                         if camera.intrinsics.cy_px is not None else cy),
                image_width=width,
                image_height=height,
                config=cfg,
            )
        except ValueError as exc:
            return (out, remaining_t, f"SKIPPED — {exc}", created_t)

        replacement = relief_mesh_primitive(patched, name=primitive.name)
        replacement.metadata["planar_hole_patch"] = report
        primitives[primitive_index] = replacement
        remaining_t = torch.from_numpy(remaining.astype(np.float32)).unsqueeze(0)
        created = np.logical_and(resolved, np.logical_not(remaining))
        created_t = torch.from_numpy(created.astype(np.float32)).unsqueeze(0)
        lines = [
            f"filled {report['components_filled']}/"
            f"{report['components_attempted']} attempted "
            f"({report['components_found']} found"
            + (f", {report['components_budget_skipped']} budget-skipped"
               if report["components_budget_skipped"] else "")
            + ")",
            f"+{report['vertices_added']} verts, +{report['faces_added']} faces, "
            f"-{report['faces_removed']} old faces",
        ]
        accepted_edge_factors = [
            float(item["patch_edge_factor"])
            for item in report["filled"]
            if "patch_edge_factor" in item
        ]
        if accepted_edge_factors:
            lines.append(
                f"accepted edge stretch max {max(accepted_edge_factors):.1f}x/"
                f"{float(max_patch_edge_factor):.1f}x "
                "(source pixels × local m/px)")
        accepted_depth_factors = [
            float(item["patch_depth_factor"])
            for item in report["filled"]
            if "patch_depth_factor" in item
        ]
        if accepted_depth_factors:
            lines.append(
                f"accepted depth excursion max "
                f"{max(accepted_depth_factors):.2f}x/"
                f"{float(max_patch_depth_factor):.2f}x "
                "(local support depth band)")
        reason_counts: dict[str, int] = {}
        for item in report["rejected"]:
            reason = str(item["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            lines.append(f"rejected {count}: {reason}")
        diagnostic_items = sorted([
            item for item in report["rejected"]
            if item["reason"] in {
                "normal consensus below threshold",
                "plane residual exceeds tolerance",
                "generated patch edge exceeds local scale",
                "generated patch depth exceeds local support",
            }
        ], key=lambda item: (
            item["reason"] not in {
                "generated patch edge exceeds local scale",
                "generated patch depth exceeds local support",
            },
            -max(
                float(item.get("patch_edge_factor", 0.0)),
                float(item.get("patch_depth_factor", 0.0)),
            ),
        ))
        for item in diagnostic_items[:3]:
            detail = ""
            if "normal_support_fraction" in item:
                detail += (
                    f" normal={item['normal_support_fraction']:.2f}/"
                    f"{item['required_normal_support_fraction']:.2f}")
            if "plane_error_p95_m" in item:
                detail += (
                    f" p95={item['plane_error_p95_m']:.3f}/"
                    f"{item['max_plane_error_m']:.3f}m")
            if "patch_edge_factor" in item:
                detail += (
                    f" edge={item['patch_edge_factor']:.1f}x/"
                    f"{item['max_patch_edge_factor']:.1f}x"
                    f" px={item['worst_edge_pixel_span']:.1f}"
                    f" local={item['local_support_world_per_pixel']:.6f}m/px")
            if "patch_depth_factor" in item:
                detail += (
                    f" depth={item['patch_depth_factor']:.2f}x/"
                    f"{item['max_patch_depth_factor']:.2f}x"
                    f" generated={item['generated_depth_min_m']:.3f}"
                    f"..{item['generated_depth_max_m']:.3f}m"
                    f" support={item['support_depth_p05_m']:.3f}"
                    f"..{item['support_depth_p95_m']:.3f}m")
            lines.append(
                f"example {item['cells']} cells:{detail} "
                f"({item['reason']})")
        return (out, remaining_t, "\n".join(lines), created_t)


class AtlasPathGuidedHoleRepair:
    """Select exact source-space tear islands from a Camera Path view.

    Candidate planes are rendered with stable island IDs at ``last-offset``.
    This makes both automatic selection and an artist-painted moved-view mask
    deterministic: the brush chooses whole connected source islands rather
    than attempting to unproject pixels through geometry that does not exist.
    Wire ``repair_mask`` into a second AtlasPlanarHolePatch with deliberately
    relaxed acceptance settings.
    """

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("repair_mask", "angle_preview", "visible_islands", "report")
    FUNCTION = "select"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "hole_mask": ("MASK", {
                    "tooltip": "Remaining source-space holes from the first "
                               "AtlasPlanarHolePatch pass.",
                }),
                "camera_path": ("ATLAS_CAMERA_PATH", {
                    "tooltip": "Camera path authored/baked by Atlas Viewport. "
                               "The selected pose is last frame minus offset.",
                }),
            },
            "optional": {
                "path_frames": ("IMAGE", {
                    "tooltip": "Optional indexed viewport frame(s) used only as the "
                               "preview background. Bake Repair Frame stores the final "
                               "frame; candidate IDs remain exact without imagery.",
                }),
                "paint_mask": ("MASK", {
                    "tooltip": "Optional mask painted over angle_preview. Used only "
                               "when selection_mode=paint_overlap; touching a rendered "
                               "candidate selects its complete source-space island.",
                }),
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional exclusion mask, subtracted before connected "
                               "components are found, separating open foreground tears "
                               "from the exterior. Wire the SAME mask you gave the mesh "
                               "node - the two must agree or this one hunts for tears in "
                               "geometry that was never built. The report states its "
                               "frame coverage so you can read the polarity back.",
                }),
                "layer": ("STRING", {
                    "default": "",
                    "tooltip": "Blank = primary relief; otherwise a ProjectionSource name.",
                }),
                "frame_offset_from_end": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "0 = final frame, 1 = one frame before final, etc.",
                }),
                "lens_scale_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "0 = use the Camera Path playback lens. Otherwise a "
                               "focal multiplier: below 1.0 is wider; above 1.0 tighter.",
                }),
                "resolution": ("INT", {
                    "default": 768, "min": 128, "max": 4096, "step": 8,
                    "tooltip": "Long edge of the candidate ID render.",
                }),
                "selection_mode": ([
                    "all_visible", "smallest_visible", "largest_visible",
                    "paint_overlap",
                ], {
                    "tooltip": "Agent mode selects visible fitted islands automatically. "
                               "paint_overlap converts a moved-view brush mask back into "
                               "exact source-space connected islands.",
                }),
                "max_selected_islands": ("INT", {
                    "default": 0, "min": 0, "max": 1024,
                    "tooltip": "0 = all qualifying islands; otherwise cap selection in "
                               "the chosen size order.",
                }),
                "min_visible_pixels": ("INT", {
                    "default": 8, "min": 1, "max": 100000,
                    "tooltip": "Ignore candidates with fewer rasterized pixels at the "
                               "chosen repair angle.",
                }),
                "paint_overlap_fraction": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum painted share of a visible island needed to "
                               "select its full source-space component.",
                }),
                "ring_cells": ("INT", {
                    "default": 2, "min": 1, "max": 12,
                }),
                "max_components": ("INT", {
                    "default": 1024, "min": 1, "max": 4096,
                }),
                "normal_tolerance_deg": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 89.0, "step": 1.0,
                }),
                "max_plane_error_m": ("FLOAT", {
                    "default": 0.45, "min": 0.001, "max": 10.0, "step": 0.01,
                }),
                "max_hole_fraction": ("FLOAT", {
                    "default": 0.04, "min": 0.0001, "max": 1.0, "step": 0.005,
                }),
                "enclosed_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "False allows silhouette/open-edge tears after the "
                               "background/sky has already been excluded upstream.",
                }),
                "min_normal_support_fraction": ("FLOAT", {
                    "default": 0.20, "min": 0.10, "max": 1.0, "step": 0.05,
                }),
            },
        }

    @staticmethod
    def _preview_batch_index(camera_path, batch_size, selected_frame_index):
        """Map a path frame number to its compact baked IMAGE batch slot."""
        count = max(0, int(batch_size))
        if count <= 0:
            return None
        baked_indices = [
            int(value)
            for value in (getattr(camera_path, "baked_frame_indices", None) or [])
        ]
        if baked_indices:
            if len(baked_indices) != count:
                return None
            try:
                return baked_indices.index(int(selected_frame_index))
            except ValueError:
                return None
        # Legacy full-path bakes predate explicit frame-index metadata.
        return max(0, min(count - 1, int(selected_frame_index)))

    @staticmethod
    def _warp_frame_for_lens(frame, output_height, output_width,
                             used_lens_scale, baked_lens_scale):
        """Reframe a baked path image when an explicit lens override is used."""
        torch = _require_torch()
        import torch.nn.functional as functional

        image = frame.permute(0, 3, 1, 2)
        ratio = float(used_lens_scale) / max(float(baked_lens_scale), 1e-6)
        scaled_height = max(1, int(round(output_height * ratio)))
        scaled_width = max(1, int(round(output_width * ratio)))
        resized = functional.interpolate(
            image, size=(scaled_height, scaled_width),
            mode="bilinear", align_corners=False)
        canvas = torch.zeros(
            (1, 3, output_height, output_width),
            dtype=resized.dtype, device=resized.device)
        src_y0 = max(0, (scaled_height - output_height) // 2)
        src_x0 = max(0, (scaled_width - output_width) // 2)
        dst_y0 = max(0, (output_height - scaled_height) // 2)
        dst_x0 = max(0, (output_width - scaled_width) // 2)
        copy_height = min(output_height, scaled_height)
        copy_width = min(output_width, scaled_width)
        canvas[:, :, dst_y0:dst_y0 + copy_height,
               dst_x0:dst_x0 + copy_width] = resized[
                   :, :, src_y0:src_y0 + copy_height,
                   src_x0:src_x0 + copy_width]
        return canvas.permute(0, 2, 3, 1)

    def select(
        self,
        solve,
        hole_mask,
        camera_path,
        path_frames=None,
        paint_mask=None,
        exclude_mask=None,
        layer="",
        frame_offset_from_end=0,
        lens_scale_override=0.0,
        resolution=768,
        selection_mode="all_visible",
        max_selected_islands=0,
        min_visible_pixels=8,
        paint_overlap_fraction=0.02,
        ring_cells=2,
        max_components=1024,
        normal_tolerance_deg=30.0,
        max_plane_error_m=0.45,
        max_hole_fraction=0.04,
        enclosed_only=False,
        min_normal_support_fraction=0.20,
    ):
        torch = _require_torch()
        np = _require_numpy()
        from atlas_camera.core.path_hole_repair import (
            PathHoleRepairConfig,
            build_path_hole_repair,
        )
        from atlas_camera.exporters._layers import mesh_from_primitive

        camera = solve.camera
        target_name = str(layer or "").strip()
        if target_name:
            source = next(
                (item for item in (getattr(solve, "projection_sources", None) or [])
                 if getattr(item, "name", "") == target_name),
                None,
            )
            if source is None:
                empty = torch.zeros_like(hole_mask)
                preview = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
                return (empty, preview, preview[..., 0],
                        f"layer '{target_name}' not found")
            primitives = source.proxy_geometry
            camera = source.camera
        else:
            scene = getattr(solve, "projection_scene", None)
            primitives = (
                getattr(scene, "proxy_geometry", None)
                if scene is not None else None)
        primitive = next(
            (item for item in (primitives or [])
             if item.primitive_type == "mesh"
             and (item.metadata or {}).get("source") == "depth_relief_mesh"),
            None,
        )
        if primitive is None:
            empty = torch.zeros_like(hole_mask)
            preview = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
            return (empty, preview, preview[..., 0],
                    "target has no relief mesh")
        mesh = mesh_from_primitive(primitive)
        if mesh is None:
            empty = torch.zeros_like(hole_mask)
            preview = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
            return (empty, preview, preview[..., 0],
                    "target relief mesh is empty")
        height = int(camera.intrinsics.image_height or 0)
        width = int(camera.intrinsics.image_width or 0)
        resolved = _resolve_exclude_mask(hole_mask, height, width)
        if resolved is None:
            resolved = np.zeros((height, width), dtype=bool)
        painted = None
        if paint_mask is not None:
            painted = paint_mask
            if hasattr(painted, "dim") and painted.dim() == 3:
                painted = painted[0]
            painted = (
                painted.detach().cpu().numpy()
                if hasattr(painted, "detach") else painted)
        excluded = None
        if exclude_mask is not None:
            excluded = _resolve_exclude_mask(exclude_mask, height, width)
        if camera_path is None:
            empty = torch.zeros(1, height, width, dtype=torch.float32)
            preview = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
            return (empty, preview, preview[..., 0],
                    "camera path is empty — choose Orbit/Arc, then queue or bake")

        cfg = PathHoleRepairConfig(
            frame_offset_from_end=int(frame_offset_from_end),
            lens_scale_override=float(lens_scale_override),
            resolution=int(resolution),
            selection_mode=str(selection_mode),
            max_selected_islands=int(max_selected_islands),
            min_visible_pixels=int(min_visible_pixels),
            paint_overlap_fraction=float(paint_overlap_fraction),
            ring_cells=int(ring_cells),
            max_components=int(max_components),
            normal_tolerance_deg=float(normal_tolerance_deg),
            max_plane_error_m=float(max_plane_error_m),
            max_hole_fraction=float(max_hole_fraction),
            enclosed_only=bool(enclosed_only),
            min_normal_support_fraction=float(min_normal_support_fraction),
        )
        try:
            result = build_path_hole_repair(
                mesh, resolved, source_camera=camera,
                camera_path=camera_path, paint_mask=painted,
                exclude_mask=excluded, config=cfg)
        except ValueError as exc:
            empty = torch.zeros(1, height, width, dtype=torch.float32)
            preview = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
            return (empty, preview, preview[..., 0], f"SKIPPED — {exc}")

        id_map = np.asarray(result["view_id_map"], dtype=np.int32)
        out_height, out_width = id_map.shape
        preview_note = ""
        if path_frames is not None and getattr(path_frames, "shape", (0,))[0]:
            batch_index = self._preview_batch_index(
                camera_path, int(path_frames.shape[0]), result["frame_index"])
            if batch_index is not None:
                base = path_frames[
                    batch_index:batch_index + 1].to(dtype=torch.float32)
                if ((int(base.shape[1]), int(base.shape[2]))
                        != (out_height, out_width)):
                    import torch.nn.functional as functional
                    base = functional.interpolate(
                        base.permute(0, 3, 1, 2),
                        size=(out_height, out_width),
                        mode="bilinear", align_corners=False,
                    ).permute(0, 2, 3, 1)
                base = self._warp_frame_for_lens(
                    base, out_height, out_width,
                    result["lens_scale"], result["path_lens_scale"])
            else:
                base = torch.zeros(
                    1, out_height, out_width, 3, dtype=torch.float32)
                preview_note = (
                    f"\npreview background unavailable for path frame "
                    f"{result['frame_index']}; geometry selection remains exact")
        else:
            base = torch.zeros(
                1, out_height, out_width, 3, dtype=torch.float32)

        selected_ids = set(int(value) for value in result["selected_ids"])
        visible = id_map > 0
        colors = np.zeros((out_height, out_width, 3), dtype=np.float32)
        for island_id in result["visible_ids"]:
            island = id_map == int(island_id)
            if int(island_id) in selected_ids:
                color = (1.0, 0.05, 0.72)  # selected: Atlas patch magenta
            else:
                phase = float(island_id) * 2.399963229728653
                color = (
                    0.25 + 0.25 * (math.sin(phase) + 1.0),
                    0.55 + 0.20 * (math.sin(phase + 2.1) + 1.0),
                    0.75 + 0.12 * math.sin(phase + 4.2),
                )
            colors[island] = color
        color_t = torch.from_numpy(colors).unsqueeze(0).to(
            dtype=base.dtype, device=base.device)
        alpha = torch.from_numpy(visible.astype(np.float32)).unsqueeze(
            0).unsqueeze(-1).to(dtype=base.dtype, device=base.device) * 0.78
        preview = base * (1.0 - alpha) + color_t * alpha
        repair_t = torch.from_numpy(
            np.asarray(result["repair_mask"], dtype=np.float32)).unsqueeze(0)
        visible_t = torch.from_numpy(visible.astype(np.float32)).unsqueeze(0)
        return (
            repair_t, preview, visible_t,
            str(result["report"]) + preview_note,
        )


class AtlasDeriveWalls:
    """Vertical wall planes + foreground boxes/cylinders (azimuth_walls) — one
    job, general-purpose exterior blockout. Height is clipped to whatever 3D
    points individually pass a near-vertical-normal filter, so it truncates
    sloped roofs/spires/towers — use AtlasDeriveTowersSpires for those.
    Set max_objects=0 for walls/ground/backdrop only (no foreground boxes)."""
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_objects": ("INT", {"default": 0, "min": 0, "max": 32,
                    "tooltip": "Max foreground boxes/cylinders (e.g. buildings, in an "
                               "aerial/top-down shot). 0 = walls/ground/backdrop only."}),
                "distance_modes": ("INT", {"default": 1, "min": 1, "max": 16,
                    "tooltip": "Walls per azimuth DIRECTION. 1 = classic: one plane at "
                               "the median distance of everything facing that way. A "
                               "street-grid skyline has ~2 facing directions but many "
                               "depths — raise this (with max_walls) so each direction "
                               "splits into one wall per depth mode (building row) "
                               "instead of collapsing the skyline into one slab."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Remove these pixels from wall/object fitting (e.g. a SAM "
                               "segment of everything EXCEPT one building — invert per "
                               "branch to scope each derive to one structure, then chain "
                               "AtlasMergeGeometry). Ground fit/scale/backdrop stay "
                               "full-frame so masked branches share one metric world."}),
                "ground_anchor": ("BOOLEAN", {"default": False,
                    "tooltip": "Wall DISTANCE from ray-through-base-pixel x the analytic "
                               "Y=0 ground plane — pure geometry, immune to monocular "
                               "depth's low-frequency 'banana' warp on tall structures. "
                               "Assumes the building's ground contact is VISIBLE: for "
                               "best accuracy inpaint cars/fences off the ground line "
                               "before solving (most street/architectural photos show "
                               "enough contact as-is; occluded bases are detected and "
                               "fall back to the classic depth-median distance)."}),
                **BACKDROP_WIDGET,
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=0, distance_modes=1,
               exclude_mask=None, ground_anchor=False, backdrop="measured_only"):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_projection_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            # Silent no-op was the old behaviour; say why instead.
            return (solve, "SKIPPED — solve has no usable focal (fx <= 0); "
                           "geometry unchanged")
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        cfg = ProxyDerivationConfig(max_objects=int(max_objects),
                                    wall_distance_modes=int(distance_modes),
                                    ground_anchor=bool(ground_anchor))
        prims, stats = derive_projection_proxies(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_walls=int(max_walls), horizon_y=horizon_y, config=cfg,
            exclude_mask=_resolve_exclude_mask(exclude_mask, height, width))
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "azimuth_walls", "derive_node": "AtlasDeriveWalls",
            "distance_modes": int(distance_modes),
            "ground_anchor": bool(ground_anchor),
        })
        note = _apply_backdrop_mode(out, backdrop)
        return (out, note or "geometry derived; backdrop as measured")


class AtlasDeriveTowersSpires:
    """Vertical wall planes extruded to the real image-space silhouette top
    (vertical_extrusion) — one job, reaches towers/spires/sloped roofs that
    AtlasDeriveWalls' azimuth_walls truncates. Per Hoiem/Efros/Hebert's
    "Automatic Photo Pop-up" (SIGGRAPH 2005) billboard-cutout technique."""
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_objects": ("INT", {"default": 0, "min": 0, "max": 32,
                                        "tooltip": "Max foreground boxes/cylinders. Street-level scenes: try 0 — the 2D occupancy clustering merges cars/fences/trees into oversized near-camera boxes that dominate any orbit."}),
                "distance_modes": ("INT", {"default": 1, "min": 1, "max": 16,
                    "tooltip": "Walls per azimuth DIRECTION. 1 = classic: one plane at "
                               "the median distance of everything facing that way. A "
                               "street-grid skyline has ~2 facing directions but many "
                               "depths — raise this (with max_walls) so each direction "
                               "splits into one wall per depth mode (building row) "
                               "instead of collapsing the skyline into one slab."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Remove these pixels from wall/object fitting (e.g. a SAM "
                               "segment of everything EXCEPT one building — invert per "
                               "branch to scope each derive to one structure, then chain "
                               "AtlasMergeGeometry). Ground fit/scale/backdrop stay "
                               "full-frame so masked branches share one metric world."}),
                "ground_anchor": ("BOOLEAN", {"default": False,
                    "tooltip": "Wall DISTANCE from ray-through-base-pixel x the analytic "
                               "Y=0 ground plane — pure geometry, immune to monocular "
                               "depth's low-frequency 'banana' warp on tall structures. "
                               "Assumes the building's ground contact is VISIBLE: for "
                               "best accuracy inpaint cars/fences off the ground line "
                               "before solving (most street/architectural photos show "
                               "enough contact as-is; occluded bases are detected and "
                               "fall back to the classic depth-median distance)."}),
                "roofline_split": ("BOOLEAN", {"default": False,
                    "tooltip": "Split each wall cluster at silhouette-height steps: a "
                               "row of buildings becomes one plane per roofline (each "
                               "cut to its own top, and with ground_anchor each gets "
                               "its own footprint distance) instead of one rectangle "
                               "spanning sky above the shorter buildings."}),
                **BACKDROP_WIDGET,
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=0, distance_modes=1,
               exclude_mask=None, ground_anchor=False, roofline_split=False, backdrop="measured_only"):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_vertical_extrusion_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            # Silent no-op was the old behaviour; say why instead.
            return (solve, "SKIPPED — solve has no usable focal (fx <= 0); "
                           "geometry unchanged")
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        cfg = ProxyDerivationConfig(max_objects=int(max_objects),
                                    wall_distance_modes=int(distance_modes),
                                    ground_anchor=bool(ground_anchor),
                                    roofline_split=bool(roofline_split))
        prims, stats = derive_vertical_extrusion_proxies(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_walls=int(max_walls), horizon_y=horizon_y, config=cfg,
            exclude_mask=_resolve_exclude_mask(exclude_mask, height, width))
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "vertical_extrusion", "derive_node": "AtlasDeriveTowersSpires",
            "distance_modes": int(distance_modes),
            "ground_anchor": bool(ground_anchor),
            "roofline_split": bool(roofline_split),
        })
        note = _apply_backdrop_mode(out, backdrop)
        return (out, note or "geometry derived; backdrop as measured")


class AtlasDeriveRoofsFacades:
    """Any-orientation planes via sequential RANSAC (ransac_planes) — one
    job, sloped roofs and stepped/angled facades. Best for exterior
    architecture where a single flat wall height is the wrong shape."""
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "max_planes": ("INT", {"default": 8, "min": 1, "max": 16,
                    "tooltip": "Plane budget (roofs, facades, ramps)."}),
                **BACKDROP_WIDGET,
            },
        }

    def derive(self, solve, depth, max_planes=8, backdrop="measured_only"):
        from atlas_camera.core.plane_extraction import PlaneRansacConfig, extract_planes_ransac
        params = _solve_camera_params(solve, depth)
        if params is None:
            # Silent no-op was the old behaviour; say why instead.
            return (solve, "SKIPPED — solve has no usable focal (fx <= 0); "
                           "geometry unchanged")
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        prims, stats = extract_planes_ransac(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_planes=int(max_planes), horizon_y=horizon_y, config=PlaneRansacConfig())
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "ransac_planes", "derive_node": "AtlasDeriveRoofsFacades",
        })
        note = _apply_backdrop_mode(out, backdrop)
        return (out, note or "geometry derived; backdrop as measured")


class AtlasDeriveInteriorRoom:
    """Manhattan-aligned floor + up to 4 walls + optional ceiling
    (room_cuboid) — one job, orthogonal interiors. Produces confidently
    wrong/skewed results on non-orthogonal rooms — pick a different node
    for those shots."""
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {**BACKDROP_WIDGET},
        }

    def derive(self, solve, depth, backdrop="measured_only"):
        from atlas_camera.core.room_layout import RoomCuboidConfig, extract_room_cuboid
        params = _solve_camera_params(solve, depth)
        if params is None:
            # Silent no-op was the old behaviour; say why instead.
            return (solve, "SKIPPED — solve has no usable focal (fx <= 0); "
                           "geometry unchanged")
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        prims, stats = extract_room_cuboid(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=horizon_y, config=RoomCuboidConfig())
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "room_cuboid", "derive_node": "AtlasDeriveInteriorRoom",
        })
        note = _apply_backdrop_mode(out, backdrop)
        return (out, note or "geometry derived; backdrop as measured")


class AtlasMergeGeometry:
    """Explicit combinator for two independently-derived solves' geometry —
    the Nuke-Merge-node equivalent for AtlasDeriveWalls/AtlasDeriveReliefMesh/
    etc. Chain multiple instances for 3+-way combination
    (Merge(fg, bg) -> Merge(that, sky)).

    solve_a's camera/intrinsics become the merged solve's camera — wire both
    branches from the SAME upstream solve so they share a camera; this node
    does not check for or correct a mismatch between solve_a and solve_b.

    Derive nodes never chain on their own (each one strips any prior
    PROXY_ROLE-tagged geometry before adding its own, specifically so a
    re-run never silently accumulates stale geometry) — this node is the one
    explicit, visible place two branches' geometry actually combines.

    Only merges solve_b's PROXY_ROLE-tagged geometry — i.e. only what
    solve_b's own derive node actually added — never solve_b's full
    proxy_geometry list. This was found empirically (live end-to-end run,
    not reasoned in the original design): both branches used to inherit a
    "ground_plane" pass-through entry from their shared upstream solve
    (projection_scene.create_default_projection_scene()'s placeholder,
    tagged role="ground", not PROXY_ROLE) that neither derive node touched
    — naively concatenating solve_b's entire list duplicated that inherited
    entry on top of solve_a's own copy of the exact same thing, even though
    solve_a already provides it via `out`. That specific placeholder has
    since been removed for being confusingly named and having no consumer,
    but this filter stays as the correct general contract: a merge should
    only ever combine what each side's own derive node actually produced.

    Also deduplicates the always-emitted "projection_backdrop" plane: every
    derivation strategy emits exactly one, so merging two PROXY_ROLE lists
    that each have one would still produce two overlapping backdrop planes.
    Keeps solve_a's.

    Optional `shot_cam` (ATLAS_SHOT_CAM, from AtlasDefineShotCam): when
    connected, attached onto the merged solve as `out.shot_cam` — a pure
    attachment, never a mutation of `out.camera`. Geometry is world-space and
    doesn't care about sensor/lens format; only the FINAL render/export
    camera does, and this just lets that format ride along with the merged
    result so it reaches AtlasBlockoutViewport/exporters without having to
    be re-wired in separately. solve_a's own camera intrinsics/extrinsics —
    what any of its projection sources actually use to sample their own
    photos — are completely untouched either way.
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "merge"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve_a": ("ATLAS_SOLVE",),
                "solve_b": ("ATLAS_SOLVE",),
            },
            "optional": {
                "shot_cam": ("ATLAS_SHOT_CAM", {
                    "tooltip": "Optional project/shot camera format (AtlasDefineShotCam) — "
                               "attached to the merged solve for AtlasBlockoutViewport/exporters "
                               "to conform to. Never affects this merge's own geometry/camera."}),
            },
        }

    def merge(self, solve_a, solve_b, shot_cam=None):
        from atlas_camera.core.proxy_geometry import PROXY_ROLE
        out = copy.deepcopy(solve_a)
        seen_backdrop = any(p.name == "projection_backdrop" for p in out.projection_scene.proxy_geometry)
        merged_from_b = 0
        for p in solve_b.projection_scene.proxy_geometry:
            if (p.metadata or {}).get("role") != PROXY_ROLE:
                continue  # pass-through geometry solve_b inherited, not something its derive node added
            if p.name == "projection_backdrop":
                if seen_backdrop:
                    continue
                seen_backdrop = True
            out.projection_scene.proxy_geometry.append(p)
            merged_from_b += 1
        out.projection_scene.debug_metadata["proxy_derivation_merge"] = {
            "solve_a_prims": len(solve_a.projection_scene.proxy_geometry),
            "solve_b_prims_merged": merged_from_b,
            "merged_prims_total": len(out.projection_scene.proxy_geometry),
        }
        if shot_cam is not None:
            out.shot_cam = shot_cam
        return (out,)


class AtlasDefineShotCam:
    """Project-level render/output camera format — sensor width/height (mm)
    + lens (focal length mm) + target resolution, analogous to a Nuke/Resolve
    project format setting. Wire its output into AtlasMergeGeometry (to
    attach it onto a merged solve so it flows downstream automatically) or
    directly into AtlasBlockoutViewport (an explicit direct wire always wins
    over an inherited one) to conform the FINAL render/export to this format,
    regardless of what aspect ratio any individual source photo happened to
    have. Intrinsics-only — no position; camera placement still comes from
    whichever solve's own recovered pose is already in play. Never affects
    how any photo gets projected onto geometry — see AtlasShotCam's own
    docstring in schema.py for why that's safe.
    """
    RETURN_TYPES = ("ATLAS_SHOT_CAM",)
    RETURN_NAMES = ("shot_cam",)
    FUNCTION = "define"
    CATEGORY = "Atlas Camera/Project"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 1000.0,
                    "tooltip": "Shot format sensor width in mm (with sensor_height_mm, defines "
                               "the output aspect ratio — e.g. 36x24 for 3:2, 36x20.25 for 16:9)."}),
                "sensor_height_mm": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 1000.0}),
                "focal_length_mm": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 2000.0,
                    "tooltip": "Shot format lens — the FINAL render/export camera's focal length, "
                               "independent of any individual source photo's own solved lens."}),
                "resolution": ("INT", {"default": 1920, "min": 128, "max": 8192, "step": 8,
                    "tooltip": "Long-edge output resolution; the short edge follows the sensor "
                               "aspect above (same long-edge convention as AtlasBlockoutViewport's "
                               "own resolution widget)."}),
            },
        }

    def define(self, sensor_width_mm=36.0, sensor_height_mm=24.0, focal_length_mm=35.0, resolution=1920):
        from atlas_camera.core.schema import AtlasShotCam
        return (AtlasShotCam(
            sensor_width_mm=float(sensor_width_mm),
            sensor_height_mm=float(sensor_height_mm),
            focal_length_mm=float(focal_length_mm),
            resolution_long_edge_px=int(resolution),
        ),)


class AtlasExtractAnglePatch:
    """Write a Photoshop-friendly patch package from an extracted viewport angle.

    This is the MVP bridge for the ``Extract Angle`` control. The incoming
    ``plate_image`` is normally the viewport's shaded/projection render and
    ``matte`` is the artist-selected region to repair. The node crops both to
    one padded rectangle, writes image/matte/depth/normal passes plus a JSON
    sidecar containing the exact orbit string and source solve, and returns a
    typed package for :class:`AtlasImportAnglePatch`.

    It deliberately does not invent a new camera: ``patch_exact`` is preserved
    byte-for-byte so the downstream ``AtlasAddPatchView.exact_view_override``
    can reconstruct the same pose after Photoshop round-tripping.
    """
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "ATLAS_PATCH")
    RETURN_NAMES = ("patch_image", "patch_matte", "manifest_path", "patch_package")
    FUNCTION = "extract"
    CATEGORY = "Atlas Camera/Patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "plate_image": ("IMAGE",),
                "matte": ("MASK",),
                "patch_exact": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "atlas_exports/angle_patches"}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "normal": ("IMAGE",),
                "name": ("STRING", {"default": "angle_patch"}),
                "padding_px": ("INT", {"default": 128, "min": 0, "max": 2048}),
                "colorspace": (["ACEScg", "sRGB - Display"], {"default": "ACEScg"}),
            },
        }

    def extract(self, solve, plate_image, matte, patch_exact, output_dir,
                depth=None, normal=None, name="angle_patch", padding_px=128,
                colorspace="ACEScg"):
        np = _require_numpy()
        torch = _require_torch()
        PILImage = _require_pil()
        if not patch_exact or not patch_exact.strip():
            raise ValueError("patch_exact is empty; click Extract Angle before exporting a patch.")
        if plate_image.ndim != 4 or plate_image.shape[0] < 1:
            raise ValueError("plate_image must be a non-empty ComfyUI IMAGE batch.")
        rgb = plate_image[0].detach().cpu().numpy().clip(0.0, 1.0)
        mask_arr = matte[0].detach().cpu().numpy().clip(0.0, 1.0)
        if mask_arr.shape != rgb.shape[:2]:
            raise ValueError("matte dimensions must match plate_image dimensions.")
        ys, xs = np.where(mask_arr > 1.0 / 255.0)
        if len(xs) == 0:
            raise ValueError("matte contains no non-zero pixels; select the Photoshop repair region first.")
        pad = max(0, int(padding_px))
        y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "angle_patch")).strip("._") or "angle_patch"
        root = Path(output_dir).expanduser().resolve() / safe_name
        root.mkdir(parents=True, exist_ok=True)

        def save_rgb(arr, path):
            PILImage.fromarray((arr * 255.0).clip(0, 255).astype("uint8"), mode="RGB").save(path, format="PNG")

        patch_rgb = rgb[y0:y1, x0:x1]
        patch_mask = mask_arr[y0:y1, x0:x1]
        image_path = root / "patch.png"
        matte_path = root / "patch_matte.png"
        save_rgb(patch_rgb, image_path)
        PILImage.fromarray((patch_mask * 255.0).clip(0, 255).astype("uint8"), mode="L").save(matte_path, format="PNG")
        # The FULL frame is required for reprojection: AtlasAddPatchView's
        # ProjectionSource samples uv across the whole patch-camera frustum, so
        # the import node must paste the edited crop back into this frame — a
        # bare crop fed downstream would stretch across the frustum and
        # misregister. The crop exists purely as the Photoshop convenience.
        full_path = root / "plate_full.png"
        save_rgb(rgb, full_path)

        pass_paths = {"image": str(image_path), "matte": str(matte_path),
                      "plate_full": str(full_path)}
        for label, tensor in (("depth", depth), ("normal", normal)):
            if tensor is not None:
                arr = tensor[0].detach().cpu().numpy().clip(0.0, 1.0)
                pass_path = root / f"patch_{label}.png"
                save_rgb(arr[y0:y1, x0:x1], pass_path)
                pass_paths[label] = str(pass_path)

        # Camera block only — never the full solve: a layered solve's to_dict()
        # carries megabytes of base64 plates and would balloon the sidecar.
        try:
            camera_dict = solve.camera.to_dict()
        except Exception:
            camera_dict = {}
        from atlas_camera import __version__ as _atlas_version
        manifest = {
            "schema": 1,
            "kind": "atlas_angle_patch",
            "atlas_version": _atlas_version,
            "patch_exact": patch_exact.strip(),
            "source_camera": camera_dict,
            "crop_bbox_xyxy": [x0, y0, x1, y1],
            "padding_px": pad,
            "image_wh": [int(x1 - x0), int(y1 - y0)],
            "full_wh": [int(rgb.shape[1]), int(rgb.shape[0])],
            "colorspace_intent": colorspace,
            "colorspace_written": "sRGB 8-bit PNG (proxy/LDR viewport plate; EXR is the planned float path)",
            "premultiplied": False,
            "photoshop_roundtrip": {
                "edit_image": "patch.png",
                "preserve_matte": "patch_matte.png",
                "write_back_as": "patch_edited.png",
            },
            "passes": pass_paths,
        }
        manifest_path = root / "atlas_angle_patch.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        package = {"manifest": str(manifest_path), "passes": pass_paths, "patch_exact": patch_exact.strip(), "crop_bbox_xyxy": manifest["crop_bbox_xyxy"]}
        return (_pil_to_image_tensor(PILImage.fromarray((patch_rgb * 255).astype("uint8"), mode="RGB")),
                torch.from_numpy(patch_mask.astype("float32")).unsqueeze(0), str(manifest_path), package)


class AtlasImportAnglePatch:
    """Load an edited angle patch, paste it back into the FULL frame, and
    expose the exact pose for reprojection.

    The extraction crop is a Photoshop convenience only — reprojection needs
    the full frame, because ``AtlasAddPatchView``'s ProjectionSource samples
    uv across the whole patch-camera frustum (a bare crop would stretch
    across the frustum and misregister). This node loads ``plate_full.png``,
    pastes the edited crop at the manifest's ``crop_bbox_xyxy``, and returns
    FULL-FRAME image and matte tensors.

    Wire ``patch_image`` into ``AtlasAddPatchView.patch_image`` and
    ``patch_exact`` into its ``exact_view_override`` input. This keeps the
    Photoshop edit in the same camera frame that produced the extraction.
    """
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "ATLAS_PATCH")
    RETURN_NAMES = ("patch_image", "patch_matte", "patch_exact", "patch_package")
    FUNCTION = "import_patch"
    CATEGORY = "Atlas Camera/Patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"patch_package": ("ATLAS_PATCH",)},
            "optional": {
                "edited_image": ("IMAGE", {"tooltip": "Optional Photoshop-edited CROP (same size as patch.png); otherwise patch.png is loaded."}),
                "edited_matte": ("MASK", {"tooltip": "Optional edited CROP matte; otherwise patch_matte.png is loaded."}),
            },
        }

    def import_patch(self, patch_package, edited_image=None, edited_matte=None):
        np = _require_numpy()
        torch = _require_torch()
        PILImage = _require_pil()
        if not isinstance(patch_package, dict) or not patch_package.get("manifest"):
            raise ValueError("patch_package is not an Atlas angle-patch package.")
        manifest_path = Path(str(patch_package["manifest"])).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Atlas angle-patch manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "atlas_angle_patch":
            raise ValueError("manifest kind is not atlas_angle_patch")
        passes = manifest.get("passes", {})
        bbox = manifest.get("crop_bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise ValueError("angle patch manifest has no crop_bbox_xyxy — re-extract with a current Atlas.")
        x0, y0, x1, y1 = (int(v) for v in bbox)

        full_path = Path(passes.get("plate_full", ""))
        if not full_path.is_file():
            raise FileNotFoundError(
                "plate_full.png missing from the patch package — reprojection "
                "needs the full frame to paste the edited crop into. "
                "Re-extract with a current Atlas.")
        full = np.asarray(PILImage.open(full_path).convert("RGB"), dtype=np.float32) / 255.0

        if edited_image is None:
            image_path = Path(passes.get("image", ""))
            if not image_path.is_file():
                raise FileNotFoundError("No edited_image was supplied and patch.png is missing.")
            crop = np.asarray(PILImage.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        else:
            crop = edited_image[0].detach().cpu().numpy()[..., :3].clip(0.0, 1.0)
        want_hw = (y1 - y0, x1 - x0)
        if crop.shape[:2] != want_hw:
            raise ValueError(
                f"edited patch is {crop.shape[1]}x{crop.shape[0]} but the extraction "
                f"crop was {want_hw[1]}x{want_hw[0]} — Photoshop must not resize the "
                "canvas (crop/uncrop changes registration).")
        full[y0:y1, x0:x1] = crop
        image_tensor = torch.from_numpy(full.astype("float32")).unsqueeze(0)

        full_mask = np.zeros(full.shape[:2], dtype=np.float32)
        if edited_matte is None:
            matte_path = Path(passes.get("matte", ""))
            if not matte_path.is_file():
                raise FileNotFoundError("No edited_matte was supplied and patch_matte.png is missing.")
            crop_mask = np.asarray(PILImage.open(matte_path).convert("L"), dtype=np.float32) / 255.0
        else:
            crop_mask = edited_matte[0].detach().cpu().numpy().clip(0.0, 1.0)
        if crop_mask.shape != want_hw:
            raise ValueError(
                f"edited matte is {crop_mask.shape[1]}x{crop_mask.shape[0]} but the "
                f"extraction crop was {want_hw[1]}x{want_hw[0]}.")
        full_mask[y0:y1, x0:x1] = crop_mask
        matte_tensor = torch.from_numpy(full_mask).unsqueeze(0)

        exact = str(manifest.get("patch_exact", "")).strip()
        if not exact:
            raise ValueError("angle patch manifest has no patch_exact camera pose.")
        package = dict(patch_package)
        package["manifest_data"] = manifest
        package["imported"] = True
        return image_tensor, matte_tensor, exact, package


class AtlasAddPatchView:
    """Add an AI novel-view "patch" to fill areas the primary camera can't see.

    Camera projection from a single photo can only texture what the recovered
    camera saw — orbit slightly and occluded/grazing areas go black. This node
    takes a novel view of the same scene generated at a defined angle (the
    Qwen-Image-Edit-2511 Multiple-Angles LoRA — e.g. via the ComfyUI-qwenmultiangle
    "Qwen Multiangle Camera" node), constructs a "patch camera" by orbiting the
    recovered camera around the scene pivot to that view (so it shares the
    primary's world frame — `camera_math.orbit_camera`), derives the patch view's
    own relief geometry in that frame (Depth Anything), and appends it to the
    solve as a ``ProjectionSource``. Chain one per angle; the viewport layers them
    over the primary, filling the occluded areas. Needs the [neural] extra.

    IMPORTANT — the LoRA's angles are ABSOLUTE (subject-relative), not relative to
    your source view: "right side view" = 90° around the *subject's* front, etc.
    So to place the patch camera correctly you must tell this node BOTH the view
    your SOURCE photo represents (``source_*``) and the view the PATCH was
    generated at (``patch_*``, matching what you set in the Qwen Multiangle Camera
    node); the actual orbit = patch − source. If the source is a straight-on
    front shot, leave ``source_azimuth_view`` = "front view" and the patch's named
    view maps directly. ``flip_azimuth`` swaps left/right if the recovered
    camera's handedness comes out mirrored (a one-click calibration fix).
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "add_patch"
    CATEGORY = "Atlas Camera/Patches"

    # Aliases onto the shared module-level dicts (see above) — kept as class
    # attributes since tests/test_add_patch_view.py references
    # AtlasAddPatchView._AZIMUTH_VIEWS/_ELEVATION_VIEWS directly.
    _AZIMUTH_VIEWS = _AZIMUTH_VIEWS
    _ELEVATION_VIEWS = _ELEVATION_VIEWS
    _DISTANCE_VIEWS = _DISTANCE_VIEWS

    @classmethod
    def INPUT_TYPES(cls):
        azimuths = list(cls._AZIMUTH_VIEWS)
        elevations = list(cls._ELEVATION_VIEWS)
        distances = list(cls._DISTANCE_VIEWS)
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "patch_image": ("IMAGE",),
            },
            "optional": {
                "patch_azimuth_view": (azimuths, {"default": "front-right quarter view",
                    "tooltip": "The LoRA azimuth the patch was generated at — MUST match the "
                               "Qwen Multiangle Camera node. Absolute about the subject's front."}),
                "patch_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "The LoRA elevation the patch was generated at (match the LoRA node)."}),
                "patch_distance": (distances, {"default": "medium shot",
                    "tooltip": "The LoRA distance the patch was generated at (match the LoRA node)."}),
                "source_azimuth_view": (azimuths, {"default": "front view",
                    "tooltip": "Which view your SOURCE photo already is, in the LoRA's absolute "
                               "frame. Orbit applied = patch − source. Leave 'front view' for a "
                               "straight-on source."}),
                "source_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "Elevation of the SOURCE photo in the LoRA's frame."}),
                "flip_azimuth": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip left/right if the patch lands on the wrong side "
                               "(recovered-camera handedness) — a calibration convenience."}),
                "name": ("STRING", {"default": "patch"}),
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "relief_grid": ("INT", {"default": 96, "min": 16, "max": 4096,
                    "tooltip": "Patch relief-mesh density (long-edge grid columns)."}),
                "priority": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blend priority among patches (higher wins). The primary photo "
                               "is always highest; patches only fill where it can't see."}),
                "plate_ref": ("ATLAS_PLATE_REF", {
                    "tooltip": "Optional registered final plate reference. Browser still uses image_b64 preview; exporters use this for EXR/float-safe handoff."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "patch_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_prompt output here "
                               "(the '<sks> [azimuth] [elevation] [distance]' string from 📐 "
                               "Extract Angle) — when connected it OVERRIDES the three patch_* "
                               "dropdowns above, so the extracted angle drives both the Qwen "
                               "generation and this node identically with one wire. (A single "
                               "STRING socket because ComfyUI's backend rejects STRING links "
                               "into combo dropdowns.) Errors loudly if the string doesn't "
                               "parse, rather than silently patching at the wrong angle."}),
                "exact_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_exact output here "
                               "('azimuth_deg=.. elevation_deg=.. distance_scale=..' — 📐's RAW "
                               "measured orbit, before named-view snapping). When connected it "
                               "WINS over patch_view_override AND the dropdowns, and "
                               "flip_azimuth is ignored (the raw delta is already in "
                               "orbit_camera's own convention). This is the render-conditioned "
                               "patch loop's channel: a frame baked at the artist's real orbit "
                               "(then repaired by AtlasRenderFix) must project back from the "
                               "IDENTICAL pose — the 45° named-view grid would misregister it. "
                               "Errors loudly if unparseable."}),
                "mask_unseen_only": ("BOOLEAN", {"default": True,
                    "tooltip": "Embed an UNSEEN-AREAS matte on the patch (ProjectionSource."
                               "mask_b64): the patch only paints where the PRIMARY camera's "
                               "projection is invalid at the patch view (behind-camera, out-of-"
                               "frame, and — when primary_depth is wired — hidden behind nearer "
                               "geometry, the true MPTK depth-shadow test). Everywhere the "
                               "primary CAN see keeps the primary's real pixels; the AI patch "
                               "fills only genuine gaps. Also rides into the Nuke/Maya exports "
                               "as the patch plate's alpha."}),
                "unseen_dilate_px": ("INT", {"default": 16, "min": 0, "max": 200,
                    "tooltip": "Dilate the unseen matte so the patch slightly overlaps the "
                               "primary's coverage edge (hides hairline seams at the boundary)."}),
                "primary_depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "STRONGLY RECOMMENDED: the shared AtlasDepthMap of the SOURCE "
                               "photo. Enables (1) overlap-based scale REGISTRATION — the patch "
                               "mesh's metric scale is solved by matching its depth against the "
                               "primary's in the mutually-visible region, so the patch actually "
                               "sits in the primary's world instead of trusting an independent "
                               "(and fragile, on AI-generated views) ground fit; and (2) the true "
                               "depth-shadow term in the unseen matte."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Segmentation of the PATCH image's sky (run SAM3Segment on the "
                               "generated novel view, prompt 'sky'). In reuse_scene mode it keeps "
                               "the patch from painting sky onto scene geometry; in own_depth "
                               "mode it REPLACES the internal sky heuristic during meshing "
                               "(hallucinated near-depth sky otherwise triangulates into "
                               "geometry bulging toward the camera)."}),
                "geometry_source": (["reuse_scene", "own_depth"], {"default": "reuse_scene",
                    "tooltip": "reuse_scene (recommended): the patch derives NO geometry of its "
                               "own — it becomes a pure texture projector onto copies of the "
                               "geometry already in the solve (sky dome, band meshes, derived "
                               "proxies), exactly how a DMP artist projects new paint from a "
                               "second camera onto the SAME geo in Nuke. The scale/registration "
                               "problem dissolves: that geometry is in the primary's world by "
                               "construction, and Qwen scene mismatch shows only as texture "
                               "misalignment, never floating geometry. No depth model runs. "
                               "own_depth: the previous behavior (Depth Anything on the patch + "
                               "overlap registration) — for patches revealing genuinely NEW "
                               "terrain no existing geometry covers. Auto-falls back to "
                               "own_depth when the solve carries no geometry to reuse."}),
            },
        }

    def add_patch(self, solve, patch_image,
                  patch_azimuth_view="front-right quarter view",
                  patch_elevation_view="eye-level shot",
                  patch_distance="medium shot",
                  source_azimuth_view="front view",
                  source_elevation_view="eye-level shot",
                  flip_azimuth=False, name="patch",
                  depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                  relief_grid=96, priority=1.0, plate_ref=None, device="auto",
                  patch_view_override="", exact_view_override="",
                  mask_unseen_only=True, unseen_dilate_px=16,
                  primary_depth=None, exclude_mask=None, geometry_source="reuse_scene"):
        exact_delta = None
        if exact_view_override and exact_view_override.strip():
            exact_delta = _parse_exact_view(exact_view_override)
            if exact_delta is None:
                raise ValueError(
                    f"exact_view_override {exact_view_override!r} does not parse as "
                    "'azimuth_deg=<f> elevation_deg=<f> distance_scale=<f>' — wire "
                    "AtlasBlockoutViewport's patch_exact output here, or disconnect it."
                )
        elif patch_view_override and patch_view_override.strip():
            parsed = _parse_view_prompt(patch_view_override)
            if parsed is None:
                raise ValueError(
                    f"patch_view_override {patch_view_override!r} does not parse as "
                    "'<sks> [azimuth] [elevation] [distance]' — wire AtlasBlockoutViewport's "
                    "patch_prompt output here, or disconnect to use the dropdowns."
                )
            patch_azimuth_view, patch_elevation_view, patch_distance = parsed
        from atlas_camera.core.camera_math import (
            ground_lookat_pivot,
            horizon_row_from_extrinsics,
            orbit_camera,
        )
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.core.schema import (
            AtlasIntrinsics,
            AtlasPlateRef,
            LatentCamera,
            ProjectionSource,
        )
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.inference.depth_estimator import estimate_depth

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        p_w = int(intr.image_width or 0)
        p_h = int(intr.image_height or 0)
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        if fx <= 0 or p_w <= 0:
            # No focal / dims on the primary — can't place a patch; pass through.
            return (solve,)
        cx = intr.cx_px if intr.cx_px is not None else p_w / 2.0
        cy = intr.cy_px if intr.cy_px is not None else p_h / 2.0

        # Absolute LoRA views -> the ACTUAL orbit delta (patch - source), since
        # the LoRA angle is subject-relative, not relative to the source view.
        # An exact override (📐's raw measured floats) is already that delta in
        # orbit_camera's own convention — no view arithmetic, no flip.
        if exact_delta is not None:
            d_azimuth, d_elevation, distance_scale = exact_delta
        else:
            d_azimuth, d_elevation, distance_scale = _named_view_orbit_delta(
                patch_azimuth_view, patch_elevation_view, patch_distance,
                source_azimuth_view, source_elevation_view, flip_azimuth,
            )

        # Patch camera: orbit the recovered camera around the scene pivot (the
        # point it looks at) so the patch shares the primary's world frame.
        pivot = ground_lookat_pivot(extr)
        patch_extr = orbit_camera(
            extr, pivot,
            d_azimuth_deg=float(d_azimuth),
            d_elevation_deg=float(d_elevation),
            distance_scale=float(distance_scale),
        )

        # Patch image dimensions + intrinsics (same angular FOV as the primary,
        # scaled to the patch resolution; principal point centered).
        patch_h = int(patch_image.shape[1])
        patch_w = int(patch_image.shape[2])
        pfx = fx * (patch_w / p_w)
        pfy = fy * (patch_h / p_h)
        pcx = patch_w / 2.0
        pcy = patch_h / 2.0
        patch_intr = AtlasIntrinsics(
            image_width=patch_w,
            image_height=patch_h,
            focal_length_mm=intr.focal_length_mm,
            sensor_width_mm=intr.sensor_width_mm,
            fx_px=pfx, fy_px=pfy, cx_px=pcx, cy_px=pcy,
            lens_model=intr.lens_model,
        )

        # The patch camera is constructed (orbited), not solved, so it carries
        # no solve.horizon_line of its own. Derive its real horizon row exactly
        # (see horizon_row_from_extrinsics) so sky-exclusion during meshing
        # uses this camera's actual tilt instead of the generic height*0.45
        # fallback in estimate_ground_scale / build_relief_mesh.
        patch_horizon_y = horizon_row_from_extrinsics(patch_extr, fy=pfy, cy=pcy)

        np = _require_numpy()
        from atlas_camera.core.depth_geometry import (
            back_project_normals,
            primary_camera_validity_mask,
        )
        from atlas_camera.core.proxy_geometry import PROXY_ROLE

        resolved_exclude = _resolve_exclude_mask(exclude_mask, patch_h, patch_w)

        # --- reuse_scene: the patch is a TEXTURE PROJECTOR onto the geometry
        # the scene already has — the DMP move (project new paint from a
        # second camera onto the SAME geo). Deriving geometry from monocular
        # depth on a HALLUCINATED image can never reliably land in the
        # primary's metric world (scale+shift error plus genuine scene
        # mismatch — per-pixel registration confirmed insufficient in Nuke),
        # so we stop trying: reused geometry is in the primary's world by
        # construction, and any Qwen mismatch shows as texture misalignment,
        # never floating geometry.
        reused_geom = []
        fallback_reason = None
        if geometry_source == "reuse_scene":
            for prim in solve.projection_scene.proxy_geometry:
                if (prim.metadata or {}).get("role") == PROXY_ROLE:
                    reused_geom.append(copy.deepcopy(prim))
            for prev in solve.projection_sources:
                for prim in prev.proxy_geometry:
                    reused_geom.append(copy.deepcopy(prim))
            for i, prim in enumerate(reused_geom):
                prim.name = f"{name}_reuse{i}_{prim.name}"
            if not reused_geom:
                geometry_source = "own_depth"
                fallback_reason = "no scene geometry to reuse"

        depth_map = None
        if geometry_source == "own_depth":
            # Depth -> relief geometry in the patch camera's frame.
            tmp = _save_image_tensor_to_tmp(patch_image)
            try:
                result = estimate_depth(tmp, model_id=depth_model,
                                        device=None if device == "auto" else device,
                                        focal_px=pfx)  # patch-image pixels
            finally:
                os.unlink(tmp)
            depth_map = result.depth
            if depth_map.shape != (patch_h, patch_w):
                depth_map = _resize_depth(depth_map, patch_w, patch_h)

        # --- Patch scale: REGISTER against the primary's metric world when the
        # shared primary depth is available; ground-fit is only the fallback.
        # An independent estimate_ground_scale on an AI-generated novel view is
        # fragile — when it misfits, the whole patch mesh lands at the wrong
        # world scale ("the patch doesn't sit with the main geometry", found
        # live). Registration exploits the OVERLAP both cameras see: scaling
        # about the patch camera makes each point's depth in the PRIMARY
        # camera affine in the scale s — z(s) = z_cam + s·(z_p − z_cam) — so
        # each overlap pixel yields a closed-form s = (m − z_cam)/(z_p − z_cam)
        # against the primary's stored metric depth m, and the median over
        # thousands of pixels is a robust one-parameter alignment.
        primary_metric_map = None
        if primary_depth is not None:
            p_map = _depth_map_for_solve(primary_depth, p_w, p_h)
            p_scale, _ = estimate_ground_scale(
                p_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=_horizon_y_from_solve(solve))
            primary_metric_map = np.asarray(p_map, dtype=np.float64) * float(p_scale)

        if geometry_source == "reuse_scene":
            patch_geom = reused_geom
            mesh = None
            scale = 1.0
            scale_source = "reuse_scene"
            # Unseen matte by FORWARD SPLAT of the primary's real metric
            # points into the patch view — coverage means "the primary has
            # trusted data that lands on this patch pixel"; no hallucinated
            # patch depth is involved at all.
            mask_b64 = None
            if mask_unseen_only and primary_metric_map is not None:
                stride = max(1, int(np.ceil(max(p_w, p_h) / 1536.0)))
                sub = primary_metric_map[::stride, ::stride]
                bp_p = back_project_normals(
                    sub, view_matrix=extr.camera_view_matrix,
                    fx=fx / stride, fy=fy / stride,
                    cx=cx / stride, cy=cy / stride)
                pts = bp_p.pts_world[bp_p.valid_depth]
                qvm = np.asarray(patch_extr.camera_view_matrix, dtype=np.float64)
                Rq, tq = qvm[:3, :3], qvm[:3, 3]
                cam_q = pts @ Rq.T + tq
                zq = -cam_q[:, 2]
                front = zq > 1e-6
                with np.errstate(all="ignore"):
                    uq = pcx + pfx * cam_q[:, 0] / np.where(front, zq, np.nan)
                    vq = pcy - pfy * cam_q[:, 1] / np.where(front, zq, np.nan)
                hit = front & np.isfinite(uq) & np.isfinite(vq) & \
                    (uq >= 0) & (uq < patch_w) & (vq >= 0) & (vq < patch_h)
                coverage = np.zeros((patch_h, patch_w), dtype=bool)
                coverage[vq[hit].astype(np.int64), uq[hit].astype(np.int64)] = True
                # Close splat sparsity (patch pixels between projected
                # samples) so 'seen' isn't undercounted — an undercounted
                # coverage would let the AI patch overwrite real pixels.
                close_px = max(2, int(round(2.0 * patch_w * stride / p_w)))
                for _ in range(close_px):
                    up = np.zeros_like(coverage)
                    up[:-1, :] = coverage[1:, :]
                    dn = np.zeros_like(coverage)
                    dn[1:, :] = coverage[:-1, :]
                    lf = np.zeros_like(coverage)
                    lf[:, :-1] = coverage[:, 1:]
                    rt = np.zeros_like(coverage)
                    rt[:, 1:] = coverage[:, :-1]
                    coverage = coverage | up | dn | lf | rt
                unseen = ~coverage
                if resolved_exclude is not None:
                    unseen &= ~resolved_exclude  # never paint sky onto geometry
                for _ in range(int(unseen_dilate_px)):
                    up = np.zeros_like(unseen)
                    up[:-1, :] = unseen[1:, :]
                    dn = np.zeros_like(unseen)
                    dn[1:, :] = unseen[:-1, :]
                    lf = np.zeros_like(unseen)
                    lf[:, :-1] = unseen[:, 1:]
                    rt = np.zeros_like(unseen)
                    rt[:, 1:] = unseen[:, :-1]
                    unseen = unseen | up | dn | lf | rt
                mask_b64 = _mask_to_b64_png(unseen) or None
            return self._finish_patch(
                solve, patch_image, patch_intr, patch_extr, patch_geom, mesh,
                mask_b64, plate_ref, name, priority,
                d_azimuth, d_elevation, distance_scale,
                patch_azimuth_view, patch_elevation_view, patch_distance,
                source_azimuth_view, flip_azimuth, pivot, depth_model,
                scale_source, scale, fallback_reason, exact_view_override,
                exact_delta)

        scale = None
        scale_source = "ground_fit"
        if primary_metric_map is not None:
            bp_raw = back_project_normals(
                depth_map, view_matrix=patch_extr.camera_view_matrix,
                fx=pfx, fy=pfy, cx=pcx, cy=pcy)
            pvm = np.asarray(extr.camera_view_matrix, dtype=np.float64)
            R, t = pvm[:3, :3], pvm[:3, 3]
            cam_pts = bp_raw.pts_world @ R.T + t
            z_p = -cam_pts[..., 2]
            patch_cam = np.asarray(
                [float(v) for v in patch_extr.camera_position], dtype=np.float64)
            z_cam = float(-(R @ patch_cam + t)[2])
            with np.errstate(all="ignore"):
                px = cx + fx * cam_pts[..., 0] / np.where(z_p > 1e-6, z_p, np.nan)
                py = cy - fy * cam_pts[..., 1] / np.where(z_p > 1e-6, z_p, np.nan)
            in_frame = np.isfinite(px) & np.isfinite(py) & \
                (px >= 0) & (px < p_w) & (py >= 0) & (py < p_h)
            sx = np.clip(np.where(in_frame, px, 0.0), 0, primary_metric_map.shape[1] - 1).astype(np.int64)
            sy = np.clip(np.where(in_frame, py, 0.0), 0, primary_metric_map.shape[0] - 1).astype(np.int64)
            m = primary_metric_map[sy, sx]
            denom = z_p - z_cam
            ok = bp_raw.valid_depth & in_frame & (z_p > 1e-6) & \
                np.isfinite(m) & (m > 1e-4) & (np.abs(denom) > 1e-3)
            if resolved_exclude is not None:
                ok &= ~resolved_exclude  # sky pixels are noise for registration
            with np.errstate(all="ignore"):
                s_samples = (m - z_cam) / denom
            ok &= np.isfinite(s_samples) & (s_samples > 1e-3) & (s_samples < 1e3)
            if int(ok.sum()) >= 500:
                scale = float(np.median(s_samples[ok]))
                scale_source = "primary_registration"

        if scale is None:
            scale, _scale_info = estimate_ground_scale(
                depth_map, view_matrix=patch_extr.camera_view_matrix,
                fx=pfx, fy=pfy, cx=pcx, cy=pcy,
                horizon_y=patch_horizon_y,
            )

        mesh = build_relief_mesh(
            depth_map, view_matrix=patch_extr.camera_view_matrix,
            fx=pfx, fy=pfy, cx=pcx, cy=pcy,
            horizon_y=patch_horizon_y,
            grid_long_edge=int(relief_grid),
            scale=scale,
            exclude_mask=resolved_exclude,
            apply_sky_heuristic=resolved_exclude is None,
        )
        patch_geom = [relief_mesh_primitive(mesh, name=f"{name}_relief_mesh")]

        # Unseen-areas matte: the patch should only paint where the PRIMARY
        # camera's projection is invalid at this patch view — everywhere the
        # primary CAN see keeps its real photographed pixels, and the AI
        # novel view fills only genuine gaps. Same math as AtlasOcclusionMask
        # (frustum/frame + optional depth-shadow), embedded directly as this
        # source's per-pixel edge matte instead of a separate composite step.
        # Uses the REGISTERED scale so the depth-shadow comparison happens in
        # the same metric world the mesh lives in.
        mask_b64 = None
        if mask_unseen_only:
            bp = back_project_normals(
                depth_map * float(scale), view_matrix=patch_extr.camera_view_matrix,
                fx=pfx, fy=pfy, cx=pcx, cy=pcy)
            unseen = primary_camera_validity_mask(
                bp.pts_world, bp.valid_depth, bp.normals, bp.valid_normal,
                primary_view_matrix=extr.camera_view_matrix,
                primary_fx=fx, primary_fy=fy, primary_cx=cx, primary_cy=cy,
                primary_width=p_w, primary_height=p_h,
                angle_threshold_deg=90.0,
                primary_depth_map=primary_metric_map)
            for _ in range(int(unseen_dilate_px)):
                up = np.zeros_like(unseen)
                up[:-1, :] = unseen[1:, :]
                dn = np.zeros_like(unseen)
                dn[1:, :] = unseen[:-1, :]
                lf = np.zeros_like(unseen)
                lf[:, :-1] = unseen[:, 1:]
                rt = np.zeros_like(unseen)
                rt[:, 1:] = unseen[:, :-1]
                unseen = unseen | up | dn | lf | rt
            mask_b64 = _mask_to_b64_png(unseen) or None

        return self._finish_patch(
            solve, patch_image, patch_intr, patch_extr, patch_geom, mesh,
            mask_b64, plate_ref, name, priority,
            d_azimuth, d_elevation, distance_scale,
            patch_azimuth_view, patch_elevation_view, patch_distance,
            source_azimuth_view, flip_azimuth, pivot, depth_model,
            scale_source, scale, fallback_reason, exact_view_override,
            exact_delta)

    def _finish_patch(self, solve, patch_image, patch_intr, patch_extr,
                      patch_geom, mesh, mask_b64, plate_ref, name, priority,
                      d_azimuth, d_elevation, distance_scale,
                      patch_azimuth_view, patch_elevation_view, patch_distance,
                      source_azimuth_view, flip_azimuth, pivot, depth_model,
                      scale_source, scale, fallback_reason,
                      exact_view_override="", exact_delta=None):
        from atlas_camera.core.schema import AtlasPlateRef, LatentCamera, ProjectionSource

        # Encode the novel view as a JPEG data-URI (viewport texture).
        image_b64 = ""
        try:
            pil = _image_tensor_to_pil(patch_image)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

        metadata = {
            "source": ("exact_render_patch" if exact_delta is not None
                       else "multi_angle_lora_patch"),
            "patch_azimuth_view": patch_azimuth_view,
            "patch_elevation_view": patch_elevation_view,
            "patch_distance": patch_distance,
            "source_azimuth_view": source_azimuth_view,
            "exact_view_override": (exact_view_override.strip()
                                    if exact_delta is not None else None),
            "flip_azimuth": bool(flip_azimuth) if exact_delta is None else None,
            "pivot": [float(v) for v in pivot],
            "n_vertices": mesh.stats.get("n_vertices") if mesh is not None else None,
            "n_faces": mesh.stats.get("n_faces") if mesh is not None else None,
            "depth_model": depth_model,
            "scale_source": scale_source,
            "scale": float(scale),
            "n_reused_primitives": len(patch_geom) if scale_source == "reuse_scene" else 0,
        }
        if fallback_reason:
            metadata["geometry_fallback"] = fallback_reason

        source = ProjectionSource(
            camera=LatentCamera(intrinsics=patch_intr, extrinsics=patch_extr, name=name),
            name=name,
            image_b64=image_b64,
            mask_b64=mask_b64,
            plate_ref=plate_ref if isinstance(plate_ref, AtlasPlateRef) else AtlasPlateRef.from_dict(plate_ref),
            proxy_geometry=patch_geom,
            azimuth_deg=float(d_azimuth),      # actual orbit delta applied
            elevation_deg=float(d_elevation),
            distance_scale=float(distance_scale),
            priority=float(priority),
            metadata=metadata,
        )

        out = copy.deepcopy(solve)
        out.projection_sources.append(source)
        return (out,)


class AtlasOcclusionMask:
    """Mask where a target/patch novel view has geometry the PRIMARY camera
    could not validly project onto (behind-camera, outside-frame, or too
    grazing) — white = primary is missing there, so a patch/composite should
    fill it; black = primary already has valid, sufficiently head-on coverage.

    Places the target/patch camera identically to ``AtlasAddPatchView``
    (same named-view widgets, same ``camera_math.orbit_camera`` construction —
    see ``_named_view_orbit_delta``), so the mask lines up with whatever patch
    geometry that node will later derive from the same image. Intended
    pipeline: ``Solve -> AtlasOcclusionMask -> ImageCompositeMasked (primary
    projected image + this target image) -> AtlasAddPatchView``.

    ``occlusion_mode="simple"`` (default) is the Phase-1 mask — frustum/
    frame/facing-angle only. ``occlusion_mode="depth_shadow"`` additionally
    detects true MPTK-style self-occlusion — a surface hidden behind NEARER
    geometry from the primary's view despite projecting inside its frame/
    angle limits — by treating the primary camera as a light and its own
    depth map as the shadow map (`primary_camera_validity_mask`'s
    ``primary_depth_map``; no rasterizer/render pass, still pure numpy and
    headless). Requires ``primary_depth`` connected (an `AtlasDepthMap` run
    on the PRIMARY/source photo — the same shared depth the derive nodes
    use); falls back to simple when it isn't. Both the primary shadow map
    and the target back-projection are ground-pinned to metric via
    `estimate_ground_scale` in this mode, so the depth comparison happens in
    one consistent world scale (simple mode's math is left byte-identical to
    before). ``depth_bias`` is the relative tolerance against depth-precision
    false positives — a point counts as shadowed only when it is more than
    ``depth_bias`` (fraction) farther than the stored primary depth at its
    pixel.
    """
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("occlusion_mask", "coverage_mask")
    FUNCTION = "generate"
    CATEGORY = "Atlas Camera/Patches"

    @classmethod
    def INPUT_TYPES(cls):
        azimuths = list(_AZIMUTH_VIEWS)
        elevations = list(_ELEVATION_VIEWS)
        distances = list(_DISTANCE_VIEWS)
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "target_image": ("IMAGE",),
            },
            "optional": {
                "patch_azimuth_view": (azimuths, {"default": "front-right quarter view",
                    "tooltip": "The LoRA azimuth target_image was generated at — should match "
                               "whatever you'll later pass to AtlasAddPatchView for this image."}),
                "patch_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "The LoRA elevation target_image was generated at."}),
                "patch_distance": (distances, {"default": "medium shot",
                    "tooltip": "The LoRA distance target_image was generated at."}),
                "source_azimuth_view": (azimuths, {"default": "front view",
                    "tooltip": "Which view your SOURCE photo already is, in the LoRA's absolute "
                               "frame. Must match the value you'll use in AtlasAddPatchView."}),
                "source_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "Elevation of the SOURCE photo in the LoRA's frame."}),
                "flip_azimuth": ("BOOLEAN", {"default": False,
                    "tooltip": "Must match the AtlasAddPatchView setting for this patch."}),
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "angle_threshold": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Facing-angle gate in degrees for the PRIMARY camera's coverage. "
                               "90 (default) = only frustum/behind-camera/out-of-frame failures are "
                               "masked. Lower values also mask surfaces too grazing to the primary."}),
                "dilate_px": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "Expand the white (missing) mask region by this many pixels."}),
                "soft_edge_px": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "Blur the dilated mask edge by this many pixels, for compositing."}),
                "power": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Gamma remap after blur; > 1 makes the patch contribution more solid "
                               "near the feathered edge."}),
                "occlusion_mode": (["simple", "depth_shadow"], {"default": "simple",
                    "tooltip": "simple = Phase-1 frustum/frame/facing tests only (unchanged). "
                               "depth_shadow = additionally detect surfaces hidden behind NEARER "
                               "geometry from the primary's view (true MPTK camera-as-light "
                               "shadow test, using primary_depth as the shadow map). Falls back "
                               "to simple when primary_depth isn't connected."}),
                "primary_depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "AtlasDepthMap run on the PRIMARY/source photo — the shadow map "
                               "for depth_shadow mode. Wire the same shared AtlasDepthMap the "
                               "derive nodes already use."}),
                "depth_bias": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "depth_shadow only: relative depth tolerance before a point counts "
                               "as shadowed — guards against monocular depth-precision false "
                               "positives. 0.05 = must be 5% farther than the stored depth."}),
                "patch_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_prompt output here — "
                               "overrides the three patch_* dropdowns with 📐 Extract Angle's "
                               "snapped views, keeping this mask aligned with the same "
                               "AtlasAddPatchView wiring. Errors loudly if unparseable."}),
                "exact_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_exact output here "
                               "(📐's RAW orbit floats) — wins over patch_view_override and the "
                               "dropdowns, flip_azimuth ignored, placing this mask's target "
                               "camera IDENTICALLY to an AtlasAddPatchView driven by the same "
                               "string (the shared never-drift contract). Errors loudly if "
                               "unparseable."}),
            },
        }

    def generate(self, solve, target_image,
                 patch_azimuth_view="front-right quarter view",
                 patch_elevation_view="eye-level shot",
                 patch_distance="medium shot",
                 source_azimuth_view="front view",
                 source_elevation_view="eye-level shot",
                 flip_azimuth=False,
                 depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto",
                 angle_threshold=90.0, dilate_px=0, soft_edge_px=0, power=1.0,
                 occlusion_mode="simple", primary_depth=None, depth_bias=0.05,
                 patch_view_override="", exact_view_override=""):
        exact_delta = None
        if exact_view_override and exact_view_override.strip():
            exact_delta = _parse_exact_view(exact_view_override)
            if exact_delta is None:
                raise ValueError(
                    f"exact_view_override {exact_view_override!r} does not parse as "
                    "'azimuth_deg=<f> elevation_deg=<f> distance_scale=<f>' — wire "
                    "AtlasBlockoutViewport's patch_exact output here, or disconnect it."
                )
        elif patch_view_override and patch_view_override.strip():
            parsed = _parse_view_prompt(patch_view_override)
            if parsed is None:
                raise ValueError(
                    f"patch_view_override {patch_view_override!r} does not parse as "
                    "'<sks> [azimuth] [elevation] [distance]' — wire AtlasBlockoutViewport's "
                    "patch_prompt output here, or disconnect to use the dropdowns."
                )
            patch_azimuth_view, patch_elevation_view, patch_distance = parsed
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.camera_math import ground_lookat_pivot, horizon_row_from_extrinsics, orbit_camera
        from atlas_camera.core.depth_geometry import back_project_normals, primary_camera_validity_mask
        from atlas_camera.inference.depth_estimator import estimate_depth

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        p_w = int(intr.image_width or 0)
        p_h = int(intr.image_height or 0)
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        target_h = int(target_image.shape[1])
        target_w = int(target_image.shape[2])
        if fx <= 0 or p_w <= 0:
            # No focal/dims on the primary — nothing to test against; treat
            # as fully missing so downstream compositing still gets a signal.
            mask = torch.ones(1, target_h, target_w, dtype=torch.float32)
            return (mask, 1.0 - mask)
        cx = intr.cx_px if intr.cx_px is not None else p_w / 2.0
        cy = intr.cy_px if intr.cy_px is not None else p_h / 2.0

        if exact_delta is not None:
            d_azimuth, d_elevation, distance_scale = exact_delta
        else:
            d_azimuth, d_elevation, distance_scale = _named_view_orbit_delta(
                patch_azimuth_view, patch_elevation_view, patch_distance,
                source_azimuth_view, source_elevation_view, flip_azimuth,
            )
        pivot = ground_lookat_pivot(extr)
        target_extr = orbit_camera(
            extr, pivot,
            d_azimuth_deg=d_azimuth, d_elevation_deg=d_elevation,
            distance_scale=distance_scale,
        )

        tfx = fx * (target_w / p_w)
        tfy = fy * (target_h / p_h)
        tcx = target_w / 2.0
        tcy = target_h / 2.0

        tmp = _save_image_tensor_to_tmp(target_image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device,
                                    focal_px=tfx)  # target-image pixels
        finally:
            os.unlink(tmp)
        depth_map = result.depth
        if depth_map.shape != (target_h, target_w):
            from atlas_camera.core.solver import _resize_depth
            depth_map = _resize_depth(depth_map, target_w, target_h)

        # depth_shadow mode: ground-pin BOTH sides to one metric world so the
        # shadow comparison (in the primary's camera space) is meaningful —
        # the same estimate_ground_scale reconciliation AtlasAddPatchView
        # applies to its patch geometry. simple mode's math stays
        # byte-identical to the original Phase-1 behavior.
        primary_metric_map = None
        if occlusion_mode == "depth_shadow" and primary_depth is not None:
            from atlas_camera.core.relief_mesh import estimate_ground_scale

            t_horizon = horizon_row_from_extrinsics(target_extr, fy=tfy, cy=tcy)
            t_scale, _ = estimate_ground_scale(
                depth_map, view_matrix=target_extr.camera_view_matrix,
                fx=tfx, fy=tfy, cx=tcx, cy=tcy, horizon_y=t_horizon)
            depth_map = depth_map * float(t_scale)

            p_map = _depth_map_for_solve(primary_depth, p_w, p_h)
            p_scale, _ = estimate_ground_scale(
                p_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=_horizon_y_from_solve(solve))
            primary_metric_map = np.asarray(p_map, dtype=np.float64) * float(p_scale)

        bp = back_project_normals(
            depth_map, view_matrix=target_extr.camera_view_matrix,
            fx=tfx, fy=tfy, cx=tcx, cy=tcy,
        )
        invalid = primary_camera_validity_mask(
            bp.pts_world, bp.valid_depth, bp.normals, bp.valid_normal,
            primary_view_matrix=extr.camera_view_matrix,
            primary_fx=fx, primary_fy=fy, primary_cx=cx, primary_cy=cy,
            primary_width=p_w, primary_height=p_h,
            angle_threshold_deg=float(angle_threshold),
            primary_depth_map=primary_metric_map,
            depth_bias_rel=float(depth_bias),
        )
        mask = invalid.astype(np.float32)

        # 4-connected binary dilation, one pixel per iteration. np.roll wraps
        # at the image border rather than clamping — negligible in practice
        # since dilate_px is capped at 200 and target images are typically
        # much larger, but a very small image + large dilate_px could wrap.
        for _ in range(int(dilate_px)):
            grown = mask.copy()
            grown = np.maximum(grown, np.roll(mask, 1, axis=0))
            grown = np.maximum(grown, np.roll(mask, -1, axis=0))
            grown = np.maximum(grown, np.roll(mask, 1, axis=1))
            grown = np.maximum(grown, np.roll(mask, -1, axis=1))
            mask = grown

        soft_edge_px = int(soft_edge_px)
        if soft_edge_px > 0:
            # Separable 2D box blur via cumulative sums (horizontal pass then
            # vertical) — numpy-only, no scipy (matches core/ convention).
            radius = soft_edge_px
            for axis in (1, 0):
                padded = np.pad(mask, [(radius, radius) if a == axis else (0, 0)
                                       for a in (0, 1)], mode="edge")
                csum = np.cumsum(padded, axis=axis)
                csum = np.insert(csum, 0, 0, axis=axis)
                n = 2 * radius + 1
                lo = np.take(csum, range(0, csum.shape[axis] - n), axis=axis)
                hi = np.take(csum, range(n, csum.shape[axis]), axis=axis)
                mask = (hi - lo) / n

        mask = np.clip(mask, 0.0, 1.0) ** float(power)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        coverage_t = torch.from_numpy((1.0 - mask).astype(np.float32)).unsqueeze(0)
        return (mask_t, coverage_t)


class AtlasSolvePatchViews:
    """⌖ Which Qwen angle actually SEES the hole — measured, not guessed.

    `AtlasAddPatchView` has always asked the artist to pick a named view from
    dropdowns and hope it revealed the occluded geometry. The mesh already knows:
    rasterize the candidate fill planes from each angle and count what survives
    the z-buffer. That is `core.view_solver`, and this is its Qwen consumer — it
    walks the Multiple-Angles LoRA's own vocabulary and hands back the winning
    view as a `patch_view_override` string, ready to wire straight into
    `AtlasAddPatchView`.

    NAMED VIEWS, NOT EXACT. `exact_view_override` is more precise and is the
    wrong tool here: the LoRA only understands its 8x4x3 named grid, so an exact
    orbit it never trained on is not a view it can render. Precision would buy a
    misregistered patch.

    READ THE `no_angle_sees_it` LINE. An island no candidate reveals is a useful
    negative result, not a failure — Qwen cannot help, and the hole wants a real
    capture (AtlasShootList) or a clean plate. Silently returning the least-bad
    angle would send the artist to a view that invents the geometry instead.

    A SOURCE-VISIBLE GAP NEEDS NO PATCH ANGLE AT ALL. Measured while building the
    solver: a see-through gap scores HIGHEST from the source camera (900 px vs
    502 off-axis) because the fill plane faces it squarely. When the report says
    the source view won, the plate already has those pixels and a generated patch
    would replace real data with invention.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("patch_view_override", "patch_prompt", "view_plan", "report")
    FUNCTION = "solve_views"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "hole_mask": ("MASK",),
            },
            "optional": {
                "source_azimuth_view": (list(_AZIMUTH_VIEWS),
                                        {"default": "front view"}),
                "source_elevation_view": (list(_ELEVATION_VIEWS),
                                          {"default": "eye-level shot"}),
                "flip_azimuth": ("BOOLEAN", {"default": False}),
                "search_distances": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also try close-up/wide (96 candidates instead of "
                               "32). Distance changes visibility far less than "
                               "angle does, and every candidate is a rasterize."}),
                "resolution": ("INT", {
                    "default": 384, "min": 128, "max": 1536,
                    "tooltip": "Rasterize size per candidate. Runs ONCE PER "
                               "CANDIDATE VIEW and the rasterizer is pure-numpy "
                               "O(faces x pixels) — this is the expensive knob."}),
                "min_visible_pixels": ("INT", {
                    "default": 32, "min": 1, "max": 100000,
                    "tooltip": "Below this an island counts as unseen from that "
                               "view. A few grazing pixels is not a patch."}),
                "max_views": ("INT", {
                    "default": 3, "min": 1, "max": 32,
                    "tooltip": "How many ranked views go into view_plan. The "
                               "override output always carries the single best; "
                               "the plan is for multi-angle and agent use."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Hole pixels to ignore — typically sky already "
                               "carried by a dome. Excluded pixels leave the "
                               "ranking entirely."}),
                "max_hole_fraction": ("FLOAT", {
                    "default": 0.35, "min": 0.001, "max": 1.0, "step": 0.005,
                    "tooltip": "Largest island, as a fraction of frame, that a "
                               "fill plane may be fitted to. NOT the repair "
                               "module's 0.04 default: that is tuned for small "
                               "planar patches, and at 0.04 a normal tear (43% "
                               "of frame, measured) fits NOTHING and every angle "
                               "scores zero — which reads as 'Qwen cannot see it' "
                               "when nothing was ever fitted to look at."}),
                "normal_tolerance_deg": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 89.0, "step": 1.0,
                    "tooltip": "How much the hole's surrounding normals may "
                               "disagree before it is judged non-planar."}),
                "max_plane_error_m": ("FLOAT", {
                    "default": 0.45, "min": 0.001, "max": 100.0, "step": 0.01,
                    "tooltip": "Plane-fit residual cap, metres. Only meaningful "
                               "when the solve's metric scale is trusted."}),
            },
        }

    def solve_views(self, solve, hole_mask, source_azimuth_view="front view",
                    source_elevation_view="eye-level shot", flip_azimuth=False,
                    search_distances=False, resolution=384,
                    min_visible_pixels=32, max_views=3, exclude_mask=None,
                    max_hole_fraction=0.35, normal_tolerance_deg=30.0,
                    max_plane_error_m=0.45):
        import json

        from atlas_camera.comfy.nodes import _relief_mesh_from_solve
        from atlas_camera.core.path_hole_repair import (
            PathHoleRepairConfig, build_island_candidates,
        )
        from atlas_camera.core.view_solver import (
            CandidateView, best_view_per_island, rank_views,
        )
        np = _require_numpy()

        mesh = _relief_mesh_from_solve(solve)
        if mesh is None:
            return ("", "", "{}",
                    "AtlasSolvePatchViews: no relief mesh on this solve — run "
                    "AtlasDeriveReliefMesh (or AtlasInput) upstream first.")

        intr = solve.camera.intrinsics
        width = int(intr.image_width or 0)
        height = int(intr.image_height or 0)
        holes = self._mask_to_bool(hole_mask, width, height, np)
        if exclude_mask is not None:
            holes &= ~self._mask_to_bool(exclude_mask, width, height, np)
        if not holes.any():
            return ("", "", "{}",
                    "AtlasSolvePatchViews: hole mask is empty after exclusion — "
                    "nothing to find an angle for.")

        distances = list(_DISTANCE_VIEWS) if search_distances else ["medium shot"]
        candidates = []
        for az in _AZIMUTH_VIEWS:
            for el in _ELEVATION_VIEWS:
                for dist in distances:
                    d_az, d_el, scale = _named_view_orbit_delta(
                        az, el, dist, source_azimuth_view,
                        source_elevation_view, flip_azimuth)
                    candidates.append(CandidateView(
                        d_azimuth_deg=d_az, d_elevation_deg=d_el,
                        distance_scale=scale,
                        # The label round-trips the NAMES back out. The solver
                        # works in deltas and cannot reconstruct which named view
                        # produced one — several combinations reach +45 degrees.
                        label=az + "|" + el + "|" + dist))

        cfg = PathHoleRepairConfig(
            resolution=int(resolution),
            max_hole_fraction=float(max_hole_fraction),
            normal_tolerance_deg=float(normal_tolerance_deg),
            max_plane_error_m=float(max_plane_error_m),
        )
        # Built HERE, not inside rank_views, so the two zero-score causes can be
        # told apart. Fitting nothing and seeing nothing look identical in a
        # ranking, and conflating them reports a config problem as a routing
        # decision — measured live: at the repair module's 0.04 default a 43%
        # tear fitted zero planes and the node confidently said "Qwen cannot
        # see it".
        built = build_island_candidates(
            mesh, holes, source_camera=solve.camera, config=cfg)
        if not len(built["candidate_faces"]):
            islands = len(built["components"])
            return ("", "", json.dumps({"views": [], "no_candidate_planes": True}),
                    "AtlasSolvePatchViews: no fill plane could be FITTED to any "
                    "of the " + str(islands) + " hole island(s), so there was "
                    "nothing to look at from any angle.\n"
                    "This is a fit-tolerance problem, NOT a verdict on the "
                    "angles. Raise max_hole_fraction (currently "
                    + str(round(float(max_hole_fraction), 3)) + ") for large "
                    "tears, or normal_tolerance_deg / max_plane_error_m for "
                    "non-planar ones. A hole that is genuinely not planar wants "
                    "AtlasCompleteDepth or a real capture instead.")

        scores = rank_views(
            mesh, holes, source_camera=solve.camera, candidates=candidates,
            resolution=int(resolution),
            min_visible_pixels=int(min_visible_pixels),
            config=cfg, prebuilt=built,
        )
        useful = [s for s in scores if s.visible_px > 0]
        if not useful:
            return ("", "", json.dumps({"views": [], "unseen": True}),
                    "AtlasSolvePatchViews: planes were fitted, but NO candidate "
                    "angle reveals them (" + str(len(candidates))
                    + " views tried).\n"
                    "That is a routing answer, not a failure — Qwen cannot see "
                    "this geometry from any view it knows, so it would invent "
                    "it. Send the hole to a real capture (AtlasShootList) or a "
                    "clean plate.")

        best = useful[0]
        az, el, dist = best.view.label.split("|")
        override = az + " " + el + " " + dist
        prompt = "<sks> " + override

        covered = best_view_per_island(scores)
        all_ids = {i.island_id for s in scores for i in s.islands}
        plan = {
            "source_view": [source_azimuth_view, source_elevation_view],
            "candidates_tried": len(candidates),
            "views": [
                {
                    "azimuth_view": s.view.label.split("|")[0],
                    "elevation_view": s.view.label.split("|")[1],
                    "distance": s.view.label.split("|")[2],
                    "patch_view_override": s.view.label.replace("|", " "),
                    "visible_px": s.visible_px,
                    "islands_seen": s.islands_seen,
                    "islands": [{"id": i.island_id, "visible_px": i.visible_px,
                                 "cells": i.island_cells} for i in s.islands],
                }
                for s in useful[:max(1, int(max_views))]
            ],
            "island_best_view": {
                str(island_id): score.view.label.replace("|", " ")
                for island_id, score in covered.items()
            },
        }

        source_wins = best.view.label.startswith(
            source_azimuth_view + "|" + source_elevation_view + "|")
        lines = [
            "AtlasSolvePatchViews: best = " + override
            + "  (" + str(best.visible_px) + " px across "
            + str(best.islands_seen) + " island(s); "
            + str(len(candidates)) + " candidates tried)",
            "islands covered by some view: "
            + str(len(covered)) + "/" + str(len(all_ids)),
        ]
        unseen = sorted(all_ids - set(covered))
        if unseen:
            lines.append(
                "no_angle_sees_it: island(s) " + str(unseen)
                + " — Qwen cannot help here; route to AtlasShootList or a clean "
                "plate rather than generating.")
        if source_wins:
            lines.append(
                "NOTE the SOURCE view scored highest, so this is a see-through "
                "gap the plate already photographs. A generated patch would "
                "replace real pixels with invention — prefer "
                "AtlasPlanarHolePatch or a clean plate.")
        lines.append(
            "top " + str(min(len(useful), int(max_views))) + ": "
            + ", ".join(s.view.label.replace("|", " ")
                        + " (" + str(s.visible_px) + "px)"
                        for s in useful[:max(1, int(max_views))]))
        return (override, prompt, json.dumps(plan, indent=2), "\n".join(lines))

    @staticmethod
    def _mask_to_bool(mask, width, height, np):
        """ComfyUI MASK -> (height, width) bool at the SOURCE camera's size."""
        arr = np.asarray(
            mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask,
            dtype=np.float64)
        while arr.ndim > 2:
            arr = arr.max(axis=0)
        if arr.shape != (height, width):
            yi = np.minimum((np.arange(height) * arr.shape[0] / max(height, 1))
                            .astype(int), arr.shape[0] - 1)
            xi = np.minimum((np.arange(width) * arr.shape[1] / max(width, 1))
                            .astype(int), arr.shape[1] - 1)
            arr = arr[np.ix_(yi, xi)]
        return arr > 0.5


class AtlasBlockoutMassing:
    """Grid-aligned placeholder building mass for ground the plate never saw.

    A still shows one side of one block. Behind the foreground, past the frame
    edges and beyond the visible buildings there is nothing, and on a camera
    move that nothing reads as a hole in the world rather than as distance.

    This fills it with cuboids, and the discipline is that almost nothing is
    invented. The ground plane and metric scale come from the solve; the street
    azimuth is FITTED to ground lines back-projected off this plate; heights are
    RESAMPLED from roofs actually observed in frame, so no mass can be taller or
    shorter than something really there. The single invented claim is that a
    building stands in that spot -- which is why these are boxes. Give them
    windows and they start asserting detail nobody measured.

    Placeholders are placed only OUTSIDE the ground the camera could see, so
    they cannot contradict the plate; and never inside measured geometry, which
    is the one failure that cannot be excused as "it is only a blockout".

    Every primitive carries provenance="placeholder" and the lowest trust tier.
    Nothing downstream may promote it.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "blockout"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE", {"tooltip": "The plate. Street lines are "
                                    "detected here and back-projected onto the "
                                    "solved ground plane to fit the grid."}),
                "observed_heights_m": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated roof heights measured from THIS "
                               "plate. Placeholder heights are resampled from "
                               "these, never invented. Empty = the node refuses."}),
            },
            "optional": {
                "ground_mask": ("MASK", {"tooltip": "Where the ground actually is. "
                    "Without it every line below the horizon is treated as a "
                    "ground line, and facade edges pollute the grid fit."}),
                "hole_mask": ("MASK", {"tooltip": "hole_mask from "
                    "AtlasDisocclusionGuide. Ground that the mask reports as "
                    "COVERED is treated as measured and left alone; placeholders "
                    "go only where geometry is actually missing. Without it the "
                    "whole visible ground is assumed covered, which is coarser "
                    "and blocks legitimate infill behind the foreground."}),

                "extend_m": ("FLOAT", {"default": 200.0, "min": 20.0, "max": 2000.0,
                                       "step": 10.0}),
                "street_width_m": ("FLOAT", {"default": 30.0, "min": 0.0,
                                             "max": 120.0, "step": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
                # Appended last on purpose: widgets serialise POSITIONALLY into
                # saved workflows, so inserting one above `seed` would silently
                # re-read every existing graph's values one slot across.
                "cell_m": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 60.0,
                                     "step": 1.0}),
            },
        }

    def blockout(self, solve, image, observed_heights_m, ground_mask=None,
                 hole_mask=None, extend_m=200.0, street_width_m=30.0, seed=0,
                 cell_m=10.0):
        import copy

        import numpy as np

        from atlas_camera.core.block_massing import (
            box_transform, estimate_grid_azimuth, grid_basis, massing_report,
            place_massing)
        from atlas_camera.core.proxy_geometry import PROXY_ROLE
        from atlas_camera.core.schema import AtlasProxyPrimitive
        from atlas_camera.core.solver import _ray_world

        out = copy.deepcopy(solve)

        heights = []
        for chunk in str(observed_heights_m or "").replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = float(chunk)
            except ValueError:
                continue
            if value > 0:
                heights.append(value)
        if not heights:
            # Refuse rather than default. A default height is the node inventing
            # precisely the thing it exists to avoid inventing, and it would look
            # like a measurement in every report downstream.
            return (out, "SKIPPED - no observed_heights_m. Placeholder masses take "
                    "their height from roofs measured on this plate; with none "
                    "supplied there is nothing to resample, and a default would be "
                    "an invented number wearing a measurement's authority.")

        intr = out.camera.intrinsics
        extr = out.camera.extrinsics
        fx = float(intr.fx_px or 0.0)
        fy = float(intr.fy_px or fx)
        if fx <= 0:
            return (out, "SKIPPED - the solve has no focal length, so no pixel "
                    "can be back-projected onto the ground plane.")
        cx = float(intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0)
        cy = float(intr.cy_px if intr.cy_px is not None else intr.image_height / 2.0)
        k1 = float((intr.distortion or {}).get("k1", 0.0) or 0.0)
        # World-math convention: the 4x4 view matrix's rotation block, never the
        # bare 3x3 (transpose-ambiguous at call sites).
        world_to_cam = np.asarray(extr.camera_view_matrix, dtype=np.float64)[:3, :3]
        cam_to_world = world_to_cam.T
        cam_pos = np.asarray(extr.camera_position, dtype=np.float64)
        if cam_pos[1] <= 0:
            return (out, "SKIPPED - the camera sits at or below the ground plane "
                    "(y=%.2f); there is no ground in front of it to extend."
                    % cam_pos[1])

        pil = _image_tensor_to_pil(image)
        if pil is None:
            return (out, "SKIPPED - could not read the image.")
        gray = np.asarray(pil.convert("L"), dtype=np.uint8)
        height_px, width_px = gray.shape
        try:
            import cv2
        except ImportError:
            return (out, "SKIPPED - street-line detection needs OpenCV. Install "
                    "with: pip install -e .[vision]")

        mask = np.zeros_like(gray)
        if ground_mask is not None:
            mask[_resolve_exclude_mask(ground_mask, height_px, width_px)] = 255
            mask_note = "ground_mask supplied"
        else:
            mask[height_px // 2:, :] = 255
            mask_note = ("no ground_mask - using the lower half of the frame, so "
                         "facade edges may pollute the grid fit")
        edges = cv2.Canny(cv2.bitwise_and(gray, mask), 60, 180)
        found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=55,
                                minLineLength=55, maxLineGap=6)
        if found is None:
            return (out, "SKIPPED - no straight lines found on the ground; there "
                    "is no evidence here for a street grid.")

        def undistort(u, v):
            if k1 == 0.0:
                return u, v
            x, y = (u - cx) / fx, (v - cy) / fy
            xu, yu = x, y
            for _ in range(20):
                s = 1.0 + k1 * (xu * xu + yu * yu)
                xu, yu = x / s, y / s
            return xu * fx + cx, yu * fy + cy

        def to_ground(u, v):
            u, v = undistort(u, v)
            ray = _ray_world(u, v, fx, fy, cx, cy, cam_to_world)
            if ray[1] >= -1e-9:
                return None                      # at or above the horizon
            return cam_pos + ray * (-cam_pos[1] / ray[1])

        segments = []
        # reshape, never `found[:, 0]`: HoughLinesP returns (N, 1, 4) on some
        # OpenCV builds and (N, 4) on others, and the indexed form silently
        # yields scalars on the second — found live in ComfyUI, whose cv2 build
        # differs from the dev env's.
        for x0, y0, x1, y1 in np.asarray(found).reshape(-1, 4):
            a = to_ground(float(x0), float(y0))
            b = to_ground(float(x1), float(y1))
            if a is not None and b is not None:
                segments.append([a, b])
        fit = estimate_grid_azimuth(np.asarray(segments)) if segments else None
        if fit is None or not fit.usable:
            why = fit.reason if fit is not None else "no line reached the ground"
            return (out, "SKIPPED - %s (%s)." % (why, mask_note))

        basis_u, basis_v = grid_basis(fit.azimuth_deg)
        seen = np.asarray([s[0] for s in segments] + [s[1] for s in segments])
        seen_u = seen @ basis_u
        seen_v = seen @ basis_v
        # Where may a placeholder go? Only where the plate is NOT the evidence.
        #
        # Without a hole mask the answer is the crude one: everything the camera
        # could see is off limits, as a single bounding rectangle. That is safe
        # but blunt — it also forbids the ground BEHIND a foreground building,
        # which the camera could not see at all and which is exactly where a
        # placeholder belongs.
        #
        # AtlasDisocclusionGuide answers it properly. Its hole_mask marks pixels
        # where the geometry is missing, so ground that back-projects from a
        # COVERED pixel is measured and untouchable, while ground under a hole is
        # fair game. Cells rather than polygons because the mask is per-pixel and
        # the placer wants rectangles; cell_m is that quantisation.
        cell = max(float(cell_m), 1.0)
        occupied = []
        if hole_mask is not None:
            hm = _resolve_exclude_mask(hole_mask, height_px, width_px)
            step = max(1, int(min(height_px, width_px) // 256))
            covered_cells = set()
            hole_cells = set()
            for py in range(0, height_px, step):
                for px in range(0, width_px, step):
                    pt = to_ground(float(px), float(py))
                    if pt is None:
                        continue
                    key = (int(np.floor((pt @ basis_u) / cell)),
                           int(np.floor((pt @ basis_v) / cell)))
                    (hole_cells if hm[py, px] else covered_cells).add(key)
            # A cell containing ANY covered pixel counts as covered: biased
            # toward refusing to build, because a placeholder pushed through
            # photographed surface is worse than a gap left open.
            occupied = [(cu * cell, (cu + 1) * cell, cv * cell, (cv + 1) * cell)
                        for cu, cv in covered_cells]
            mask_note += ("; hole_mask supplied - %d covered ground cells are "
                          "off limits, %d hole cells are open"
                          % (len(covered_cells), len(hole_cells - covered_cells)))
        else:
            occupied = [(float(seen_u.min()), float(seen_u.max()),
                         float(seen_v.min()), float(seen_v.max()))]
            mask_note += ("; no hole_mask - the whole visible ground is assumed "
                          "covered, so nothing is placed behind foreground "
                          "geometry even where the camera never saw the ground")
        region = (float(seen_u.min()) - extend_m, float(seen_u.max()) + extend_m,
                  float(seen_v.min()) - extend_m, float(seen_v.max()) + extend_m)
        bands = []
        if street_width_m > 0:
            mid = 0.5 * (float(seen_v.min()) + float(seen_v.max()))
            bands.append((mid - street_width_m / 2.0, mid + street_width_m / 2.0))

        boxes = place_massing(
            azimuth_deg=fit.azimuth_deg, ground_y=0.0,
            observed_heights_m=heights, region_uv=region,
            occupied_uv=occupied, street_bands_v=bands,
            seed=int(seed), zone="off-frame / occluded")
        if not boxes:
            return (out, "grid fitted at %.2fdeg (%.0f%% coherent) but no room "
                    "remained for placeholders once measured ground and the "
                    "roadway were excluded."
                    % (fit.azimuth_deg, 100 * fit.coherence))

        scene = out.projection_scene
        prims = list(getattr(scene, "proxy_geometry", None) or [])
        start = len(prims)
        for index, box in enumerate(boxes):
            matrix, dims = box_transform(box, fit.azimuth_deg, 0.0)
            prims.append(AtlasProxyPrimitive(
                name="placeholder_mass_%03d" % (index + 1),
                primitive_type="box",
                transform_matrix=matrix,
                dimensions=dims,
                material="atlas_projection_proxy",
                metadata={
                    "role": PROXY_ROLE,
                    "source": "block_massing",
                    # The load-bearing tag. This is not measurement and no
                    # downstream consumer may treat it as one.
                    "provenance": "placeholder",
                    "trust": "placeholder",
                    "grid_azimuth_deg": float(fit.azimuth_deg),
                    "height_m": float(box.height_m),
                    "height_source": "resampled from observed roofs",
                    "zone": box.zone,
                },
            ))
        scene.proxy_geometry = prims
        report = [massing_report(boxes, fit),
                  "appended %d box primitives alongside %d existing (measured "
                  "geometry untouched); %s" % (len(prims) - start, start, mask_note)]
        return (out, "\n".join(report))
