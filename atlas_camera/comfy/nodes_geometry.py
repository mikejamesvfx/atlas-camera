"""Atlas ComfyUI nodes — geometry group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
from __future__ import annotations

import base64
import copy
import io
import json
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
               live_fill_max_hole_edges=64,
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
                               "SAM/RMBG) which REPLACES the internal sky heuristic before "
                               "triangulation - so it must cover EVERYTHING you want gone. Any "
                               "resolution - resized to match depth."}),
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
               live_fill_max_hole_edges=64,
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
        apply_live_mesh_repair(
            mesh,
            extr.camera_view_matrix,
            live_fill_holes=live_fill_holes,
            live_fill_distance_m=live_fill_distance_m,
            live_fill_max_hole_edges=live_fill_max_hole_edges,
            live_fill_edge_sawteeth=live_fill_edge_sawteeth,
            stats=stats,
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



class AtlasDeriveWalls:
    """Vertical wall planes + foreground boxes/cylinders (azimuth_walls) — one
    job, general-purpose exterior blockout. Height is clipped to whatever 3D
    points individually pass a near-vertical-normal filter, so it truncates
    sloped roofs/spires/towers — use AtlasDeriveTowersSpires for those.
    Set max_objects=0 for walls/ground/backdrop only (no foreground boxes)."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
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
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 32,
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
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=3, distance_modes=1,
               exclude_mask=None, ground_anchor=False):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_projection_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
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
        return (out,)


class AtlasDeriveTowersSpires:
    """Vertical wall planes extruded to the real image-space silhouette top
    (vertical_extrusion) — one job, reaches towers/spires/sloped roofs that
    AtlasDeriveWalls' azimuth_walls truncates. Per Hoiem/Efros/Hebert's
    "Automatic Photo Pop-up" (SIGGRAPH 2005) billboard-cutout technique."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
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
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 32,
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
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=3, distance_modes=1,
               exclude_mask=None, ground_anchor=False, roofline_split=False):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_vertical_extrusion_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
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
        return (out,)


class AtlasDeriveRoofsFacades:
    """Any-orientation planes via sequential RANSAC (ransac_planes) — one
    job, sloped roofs and stepped/angled facades. Best for exterior
    architecture where a single flat wall height is the wrong shape."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
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
            },
        }

    def derive(self, solve, depth, max_planes=8):
        from atlas_camera.core.plane_extraction import PlaneRansacConfig, extract_planes_ransac
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
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
        return (out,)


class AtlasDeriveInteriorRoom:
    """Manhattan-aligned floor + up to 4 walls + optional ceiling
    (room_cuboid) — one job, orthogonal interiors. Produces confidently
    wrong/skewed results on non-orthogonal rooms — pick a different node
    for those shots."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
        }

    def derive(self, solve, depth):
        from atlas_camera.core.room_layout import RoomCuboidConfig, extract_room_cuboid
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
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
        return (out,)


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
