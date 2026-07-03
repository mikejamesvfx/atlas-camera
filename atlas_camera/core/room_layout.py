"""Gravity-aligned room-cuboid fitting (Manhattan layout) — interiors.

Produces a coherent room shell — floor, up to 4 orthogonal walls, and an
optional ceiling — instead of the disconnected wall fragments the default
vertical-wall fitter (``proxy_geometry.azimuth_walls``) can produce on
cluttered interiors. Assumes the room is roughly orthogonal (Manhattan
world): the dominant wall azimuth (folded modulo 90°) defines two
perpendicular room axes, and each of the 4 possible sides is fit
independently, so a partially-visible room (open doorway, out-of-frame wall)
still returns whatever sides are actually supported — it never fails.

Select this method (``primitive_method="room_cuboid"`` on
``AtlasDeriveProjectionGeometry``) for orthogonal interiors; prefer
``ransac_planes`` for exteriors/non-orthogonal spaces, or the default
``azimuth_walls`` for general scenes.

Known limitation: the Manhattan assumption silently produces geometrically
wrong (skewed) walls on a genuinely non-orthogonal room rather than failing —
this is by design (the artist picks the method appropriate to the shot, the
method never self-selects); pick a different method for angled/curved rooms.

Numpy-only. Reuses ``depth_geometry`` for back-projection, ground/floor fit +
metric-scale reconciliation, and the always-emitted backdrop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from atlas_camera.core.depth_geometry import (
    arbitrary_plane_axes,
    back_project_normals,
    build_backdrop_primitive,
    fit_ground_and_scale,
    plane_transform,
)
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasProxyPrimitive


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Room-cuboid fitting requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class RoomCuboidConfig:
    """Tunables for gravity-aligned Manhattan room-cuboid fitting."""

    ground_normal_min: float = 0.90
    wall_normal_max: float = 0.25
    depth_edge_rel: float = 0.05
    manhattan_fold_bins: int = 18          # 5° cells over the folded 0-90° range
    manhattan_angular_tolerance_deg: float = 12.0
    min_wall_inliers: int = 1200
    min_ceiling_inliers: int = 800
    min_wall_size_m: float = 0.75
    ceiling_min_height_above_floor_m: float = 1.4
    extent_percentiles: tuple[float, float] = (2.0, 98.0)
    backdrop_depth_percentile: float = 96.0
    backdrop_margin: float = 1.35
    ground_padding: float = 1.1
    ground_min_extent_m: float = 4.0


def extract_room_cuboid(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    horizon_y: float | None = None,
    config: RoomCuboidConfig | None = None,
) -> tuple[list[AtlasProxyPrimitive], dict[str, Any]]:
    """Derive a floor + up to 4 Manhattan-aligned walls + optional ceiling
    (+ backdrop) from a forward-z depth map and the recovered camera.

    Returns ``(primitives, debug_stats)``; ``stats["ground_scale"]`` is always
    present (shared contract with other extraction methods).
    """
    np = _require_numpy()
    cfg = config or RoomCuboidConfig()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    if horizon_y is None:
        horizon_y = height * 0.45

    stats: dict[str, Any] = {"ceiling": False, "walls_found": 0, "manhattan_azimuth_deg": None}
    prims: list[AtlasProxyPrimitive] = []

    bp = back_project_normals(depth, view_matrix=view_matrix, fx=fx, fy=fy,
                              cx=cx, cy=cy, depth_edge_rel=cfg.depth_edge_rel)
    # Flip normals toward the camera globally (cross-product normals aren't
    # camera-oriented by construction); sign-invariant under the later
    # rescale, so safe to do before fit_ground_and_scale.
    to_cam = bp.cam_pos[None, None, :] - bp.pts_world
    flip = np.einsum("ijk,ijk->ij", bp.normals, to_cam) < 0
    bp.normals = np.where(flip[..., None], -bp.normals, bp.normals)

    gf = fit_ground_and_scale(bp, horizon_y=horizon_y,
                              ground_normal_min=cfg.ground_normal_min)
    scale = gf.scale
    stats["ground_scale"] = scale
    stats["floor_inliers"] = gf.inliers

    pts_world = gf.pts_world_scaled
    world_y = pts_world[..., 1]
    scaled_depth = depth * scale

    # -- Floor primitive (same construction as proxy_geometry's ground) --------
    if gf.inliers >= 300:
        gx = pts_world[..., 0][gf.ground_inlier]
        gz = pts_world[..., 2][gf.ground_inlier]
        p_lo, p_hi = cfg.extent_percentiles
        x0, x1 = np.percentile(gx, [p_lo, p_hi])
        z0, z1 = np.percentile(gz, [p_lo, p_hi])
        cx_w, cz_w = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
        ex = max((x1 - x0) * cfg.ground_padding, cfg.ground_min_extent_m)
        ez = max((z1 - z0) * cfg.ground_padding, cfg.ground_min_extent_m)
        u_g, v_g, n_g = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
        prims.append(AtlasProxyPrimitive(
            name="projection_floor",
            primitive_type="plane",
            transform_matrix=plane_transform(u_g, v_g, n_g, (cx_w, 0.0, cz_w)),
            dimensions=(float(ex), float(ez), 0.0),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": "room_cuboid",
                      "inliers": gf.inliers, "depth_scale_applied": scale},
        ))

    # -- Ceiling (optional): second horizontal cluster well above the floor ----
    ceil_cand = (bp.valid_normal & (np.abs(bp.normals[..., 1]) > cfg.ground_normal_min)
                 & (world_y > cfg.ceiling_min_height_above_floor_m) & ~gf.ground_inlier)
    if int(ceil_cand.sum()) >= cfg.min_ceiling_inliers:
        ys = world_y[ceil_cand]
        lo, hi = np.percentile(ys, [1, 99])
        span = float(hi - lo)
        yc = float(np.median(ys)) if span < 1e-3 else None
        if yc is None:
            hist, edges = np.histogram(ys, bins=32, range=(lo, hi))
            peak = int(np.argmax(hist))
            yc = 0.5 * (edges[peak] + edges[peak + 1])
        ctol = max(0.15, 0.03 * max(span, 1e-3))
        refine = np.abs(ys - yc) < ctol
        if int(refine.sum()) >= cfg.min_ceiling_inliers // 2:
            yc = float(np.median(ys[refine]))
        ceil_inlier = ceil_cand & (np.abs(world_y - yc) < ctol)
        if int(ceil_inlier.sum()) >= cfg.min_ceiling_inliers:
            gx = pts_world[..., 0][ceil_inlier]
            gz = pts_world[..., 2][ceil_inlier]
            p_lo, p_hi = cfg.extent_percentiles
            x0, x1 = np.percentile(gx, [p_lo, p_hi])
            z0, z1 = np.percentile(gz, [p_lo, p_hi])
            cx_w, cz_w = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
            ex = max((x1 - x0) * cfg.ground_padding, cfg.ground_min_extent_m)
            ez = max((z1 - z0) * cfg.ground_padding, cfg.ground_min_extent_m)
            # Ceiling faces down into the room: normal = -Y.
            u_c, v_c, n_c = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
            prims.append(AtlasProxyPrimitive(
                name="projection_ceiling",
                primitive_type="plane",
                transform_matrix=plane_transform(u_c, v_c, n_c, (cx_w, yc, cz_w)),
                dimensions=(float(ex), float(ez), 0.0),
                material="atlas_projection_proxy",
                metadata={"role": PROXY_ROLE, "source": "room_cuboid",
                          "inliers": int(ceil_inlier.sum()), "distance_m": float(yc),
                          "depth_scale_applied": scale},
            ))
            stats["ceiling"] = True
            stats["ceiling_height_m"] = yc

    # -- Dominant Manhattan azimuth (vertical-surface candidates) --------------
    n_pixels = height * width
    inlier_floor = max(cfg.min_wall_inliers, int(0.003 * n_pixels))
    # Note: unlike proxy_geometry/plane_extraction, no backdrop-distance filter
    # here. A genuinely enclosed room (the common interior case) has no sky to
    # separate from its own back wall, so the 96th-percentile "backdrop"
    # distance lands right at the back wall's own depth — excluding it. Walls
    # here are constrained to exactly 4 Manhattan directions, so there's no
    # open-ended-surface ambiguity requiring that separation in the first place.
    wall_cand = (bp.valid_normal & (np.abs(bp.normals[..., 1]) < cfg.wall_normal_max)
                 & ~gf.ground_inlier)
    if stats["ceiling"]:
        wall_cand &= world_y < stats["ceiling_height_m"] - 0.1

    if int(wall_cand.sum()) >= inlier_floor:
        nx = bp.normals[..., 0][wall_cand]
        nz = bp.normals[..., 2][wall_cand]
        pts_pool = pts_world[wall_cand]
        normals_pool = bp.normals[wall_cand]

        az = np.arctan2(nx, nz)
        folded = np.mod(az, math.pi / 2)
        bins = cfg.manhattan_fold_bins
        bin_idx = np.clip((folded / (math.pi / 2) * bins).astype(int), 0, bins - 1)
        hist = np.bincount(bin_idx, minlength=bins).astype(np.float64)
        hist_s = (np.roll(hist, 1) + 2 * hist + np.roll(hist, -1)) / 4.0
        best_bin = int(np.argmax(hist_s))
        axis_a_rad = (best_bin + 0.5) * (math.pi / 2) / bins
        stats["manhattan_azimuth_deg"] = math.degrees(axis_a_rad)

        A = np.array([math.sin(axis_a_rad), 0.0, math.cos(axis_a_rad)])
        B = np.array([math.sin(axis_a_rad + math.pi / 2), 0.0, math.cos(axis_a_rad + math.pi / 2)])
        cos_tol = math.cos(math.radians(cfg.manhattan_angular_tolerance_deg))

        sides = [("A", 1.0, A), ("A", -1.0, -A), ("B", 1.0, B), ("B", -1.0, -B)]
        for axis_name, sign, e in sides:
            n_plane = -e  # inward-facing (camera-facing) normal for a wall bounding the room at +e
            dots = normals_pool @ n_plane
            side_cand = dots > cos_tol
            if int(side_cand.sum()) < inlier_floor:
                continue
            offs_e = pts_pool[side_cand] @ e
            d_guess = float(np.percentile(offs_e, 96))
            tol = max(0.15, 0.02 * abs(d_guess))
            inl = np.abs(offs_e - d_guess) < tol
            if int(inl.sum()) < inlier_floor:
                continue
            p_in = pts_pool[side_cand][inl]
            d_final = float(np.median(p_in @ e))
            u_ax, v_ax, n_ax = arbitrary_plane_axes(np, n_plane)
            a = p_in @ u_ax
            b_y = np.clip(p_in @ v_ax, -0.1, None)
            p_lo, p_hi = cfg.extent_percentiles
            a0, a1 = np.percentile(a, [p_lo, p_hi])
            b0, b1 = np.percentile(b_y, [p_lo, p_hi])
            b0 = max(b0, -0.1)
            w_m, h_m = float(a1 - a0), float(b1 - b0)
            if w_m < cfg.min_wall_size_m or h_m < cfg.min_wall_size_m:
                continue
            center = d_final * e + 0.5 * (a0 + a1) * u_ax + 0.5 * (b0 + b1) * v_ax
            prims.append(AtlasProxyPrimitive(
                name=f"projection_wall_{axis_name}_{'pos' if sign > 0 else 'neg'}",
                primitive_type="plane",
                transform_matrix=plane_transform(u_ax, v_ax, n_ax, center),
                dimensions=(w_m, h_m, 0.0),
                material="atlas_projection_proxy",
                metadata={"role": PROXY_ROLE, "source": "room_cuboid",
                          "inliers": int(inl.sum()), "axis": axis_name, "sign": sign,
                          "normal_azimuth_deg": float(math.degrees(math.atan2(n_ax[0], n_ax[2]))),
                          "distance_m": d_final, "depth_scale_applied": scale},
            ))
            stats["walls_found"] += 1

    prims.append(build_backdrop_primitive(
        bp=bp, scaled_depth=scaled_depth, valid_depth=bp.valid_depth,
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, scale=scale,
        backdrop_depth_percentile=cfg.backdrop_depth_percentile,
        backdrop_margin=cfg.backdrop_margin,
    ))

    stats["primitives"] = len(prims)
    return prims, stats
