"""Sequential RANSAC / 3D Hough plane extraction — any-orientation planes.

Generalizes ``proxy_geometry``'s vertical-only azimuth-histogram wall fitter to
planes of ANY orientation: sloped roofs, ramps, and stepped/angled building
facades — the cases exterior/architectural shots need that a purely-vertical
wall fitter cannot represent. Select this method (``primitive_method=
"ransac_planes"`` on ``AtlasDeriveProjectionGeometry``) for exteriors; prefer
``azimuth_walls`` (the default) or ``room_cuboid`` for interiors.

Algorithm: bin per-pixel normals into a 2D (azimuth × elevation) orientation
histogram (the "Hough" step — finds dominant plane orientations at any tilt),
then for each orientation peak run sequential RANSAC (fit → count inliers →
extract a bounded rectangle → remove inliers → repeat at the SAME orientation)
so multiple parallel-but-offset planes (a stepped facade) are all recovered,
not just the single strongest one.

Numpy-only. Reuses ``depth_geometry`` for back-projection, ground fit +
metric-scale reconciliation, and the always-emitted backdrop, so this method
agrees bit-for-bit with the other two on world points and metric scale for a
given depth map + camera.
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
            "Plane extraction requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class PlaneRansacConfig:
    """Tunables for any-orientation sequential-RANSAC plane extraction."""

    ground_normal_min: float = 0.90
    depth_edge_rel: float = 0.05
    azimuth_bins: int = 36               # 10° cells
    elevation_bins: int = 18             # 10° cells, covers -90..90 deg
    angular_tolerance_deg: float = 12.0  # normal-to-peak tolerance for RANSAC subset
    min_plane_inliers: int = 1200
    min_plane_size_m: float = 0.6
    extent_percentiles: tuple[float, float] = (2.0, 98.0)
    backdrop_depth_percentile: float = 96.0
    backdrop_margin: float = 1.35
    ground_padding: float = 1.1
    ground_min_extent_m: float = 4.0
    max_ransac_iters_per_peak: int = 6


def extract_planes_ransac(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_planes: int = 8,
    horizon_y: float | None = None,
    config: PlaneRansacConfig | None = None,
) -> tuple[list[AtlasProxyPrimitive], dict[str, Any]]:
    """Derive ground + any-orientation planes (+ backdrop) via sequential RANSAC
    seeded by a 2D normal-orientation histogram.

    Returns ``(primitives, debug_stats)``; ``stats["ground_scale"]`` is always
    present (required by callers, e.g. the relief-mesh branch of the derive
    node, which shares the same metric scale across methods).
    """
    np = _require_numpy()
    cfg = config or PlaneRansacConfig()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    if horizon_y is None:
        horizon_y = height * 0.45

    stats: dict[str, Any] = {"planes": 0}
    prims: list[AtlasProxyPrimitive] = []

    bp = back_project_normals(depth, view_matrix=view_matrix, fx=fx, fy=fy,
                              cx=cx, cy=cy, depth_edge_rel=cfg.depth_edge_rel)
    # Cross-product normals aren't camera-oriented by construction (they can
    # point either way for a given surface); flip each toward the camera so
    # orientation-histogram bins and reported azimuth/elevation are meaningful.
    # Sign is invariant under the later uniform rescale-about-camera in
    # fit_ground_and_scale, so it's safe to do this before that call.
    to_cam = bp.cam_pos[None, None, :] - bp.pts_world
    flip = np.einsum("ijk,ijk->ij", bp.normals, to_cam) < 0
    bp.normals = np.where(flip[..., None], -bp.normals, bp.normals)

    gf = fit_ground_and_scale(bp, horizon_y=horizon_y,
                              ground_normal_min=cfg.ground_normal_min)
    scale = gf.scale
    stats["ground_scale"] = scale
    stats["ground_inliers"] = gf.inliers

    pts_world = gf.pts_world_scaled
    scaled_depth = depth * scale
    backdrop_d_raw = float(np.percentile(
        scaled_depth[bp.valid_depth], cfg.backdrop_depth_percentile
    )) if bp.valid_depth.any() else 60.0

    # -- Ground primitive (identical construction to proxy_geometry.py) --------
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
            name="projection_ground",
            primitive_type="plane",
            transform_matrix=plane_transform(u_g, v_g, n_g, (cx_w, 0.0, cz_w)),
            dimensions=(float(ex), float(ez), 0.0),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": "ransac_plane_extraction",
                      "inliers": gf.inliers, "depth_scale_applied": scale},
        ))

    # -- Remaining pool: non-ground, in-front-of-backdrop, valid-normal pixels -
    n_pixels = height * width
    inlier_floor = max(cfg.min_plane_inliers, int(0.003 * n_pixels))
    pool = (bp.valid_normal & ~gf.ground_inlier
            & (scaled_depth < backdrop_d_raw * 0.95))

    planes: list[dict[str, Any]] = []
    if int(pool.sum()) >= inlier_floor and max_planes > 0:
        normals = bp.normals
        nx_full, ny_full, nz_full = normals[..., 0], normals[..., 1], normals[..., 2]

        # -- 2D orientation histogram (the "Hough" step) ------------------------
        az_full = np.arctan2(nx_full, nz_full)
        el_full = np.arcsin(np.clip(ny_full, -1.0, 1.0))
        ab, eb = cfg.azimuth_bins, cfg.elevation_bins
        az_idx_full = ((az_full + math.pi) / (2 * math.pi) * ab).astype(int) % ab
        el_idx_full = np.clip(((el_full + math.pi / 2) / math.pi * eb).astype(int), 0, eb - 1)

        az_idx = az_idx_full[pool]
        el_idx = el_idx_full[pool]
        hist2d = np.zeros((ab, eb), dtype=np.float64)
        np.add.at(hist2d, (az_idx, el_idx), 1)

        # Smooth: 3x3 box, circular in azimuth, edge-clamped in elevation.
        smoothed = hist2d.copy()
        for da in (-1, 0, 1):
            for de in (-1, 0, 1):
                if da == 0 and de == 0:
                    continue
                shifted = np.roll(hist2d, da, axis=0)
                if de == -1:
                    shifted = np.concatenate([shifted[:, :1], shifted[:, :-1]], axis=1)
                elif de == 1:
                    shifted = np.concatenate([shifted[:, 1:], shifted[:, -1:]], axis=1)
                smoothed += shifted
        smoothed /= 9.0

        # 2D NMS peak-picking (circular suppression on azimuth only).
        flat_order = np.argsort(smoothed.ravel())[::-1]
        suppressed = np.zeros((ab, eb), dtype=bool)
        peaks: list[tuple[int, int]] = []
        for idx in flat_order:
            a_i, e_i = int(idx // eb), int(idx % eb)
            if suppressed[a_i, e_i] or smoothed[a_i, e_i] * 9.0 < inlier_floor * 0.1:
                continue
            peaks.append((a_i, e_i))
            for da in range(-1, 2):
                for de in range(-1, 2):
                    ea = e_i + de
                    if 0 <= ea < eb:
                        suppressed[(a_i + da) % ab, ea] = True
            if len(peaks) >= max_planes * 2:
                break

        pts_pool = pts_world[pool]
        claimed = np.zeros(int(pool.sum()), dtype=bool)
        depth_pool = scaled_depth[pool]

        for (a_i, e_i) in peaks:
            if len(planes) >= max_planes:
                break
            peak_az = -math.pi + (a_i + 0.5) * 2 * math.pi / ab
            peak_el = -math.pi / 2 + (e_i + 0.5) * math.pi / eb
            peak_n = np.array([
                math.cos(peak_el) * math.sin(peak_az),
                math.sin(peak_el),
                math.cos(peak_el) * math.cos(peak_az),
            ])
            tol_cos = math.cos(math.radians(cfg.angular_tolerance_deg))

            for _ in range(cfg.max_ransac_iters_per_peak):
                if len(planes) >= max_planes:
                    break
                avail = ~claimed
                if int(avail.sum()) < inlier_floor:
                    break
                cand_normals = normals[pool][avail]
                dots = cand_normals @ peak_n
                sel_local = dots > tol_cos
                if int(sel_local.sum()) < inlier_floor:
                    break

                avail_idx = np.where(avail)[0]
                sel_idx = avail_idx[sel_local]
                p_sel = pts_pool[sel_idx]
                n_sel = normals[pool][sel_idx]
                n_mean = n_sel.mean(axis=0)
                n_len = np.linalg.norm(n_mean)
                if n_len < 1e-6:
                    break
                n_mean = n_mean / n_len

                offs = p_sel @ n_mean
                d_off = float(np.median(offs))
                med_depth = float(np.median(depth_pool[sel_idx]))
                tol = max(0.15, 0.02 * med_depth)
                inl_local = np.abs(offs - d_off) < tol
                if int(inl_local.sum()) < inlier_floor:
                    break

                inl_idx = sel_idx[inl_local]
                p_in = pts_pool[inl_idx]
                u_ax, v_ax, n_ax = arbitrary_plane_axes(np, n_mean)
                a_coord = p_in @ u_ax
                b_coord = p_in @ v_ax
                p_lo, p_hi = cfg.extent_percentiles
                a0, a1 = np.percentile(a_coord, [p_lo, p_hi])
                b0, b1 = np.percentile(b_coord, [p_lo, p_hi])
                w_m, h_m = float(a1 - a0), float(b1 - b0)

                # Duplicate-offset guard: same orientation, no new distinct
                # surface found this iteration → stop (avoid infinite loop).
                is_dup = any(
                    abs(d_off - p["d"]) < max(0.3, 0.05 * med_depth)
                    and float(np.dot(n_mean, p["n"])) > tol_cos
                    for p in planes
                )
                claimed[inl_idx] = True  # always remove, even if rejected below

                if w_m < cfg.min_plane_size_m or h_m < cfg.min_plane_size_m or is_dup:
                    continue

                planes.append({
                    "n": n_mean, "u": u_ax, "v": v_ax, "d": d_off,
                    "a_mid": 0.5 * (a0 + a1), "b_mid": 0.5 * (b0 + b1),
                    "w": w_m, "h": h_m, "inliers": int(inl_local.sum()),
                    "azimuth_deg": math.degrees(math.atan2(n_mean[0], n_mean[2])),
                    "elevation_deg": math.degrees(math.asin(np.clip(n_mean[1], -1, 1))),
                })

    planes.sort(key=lambda p: -p["inliers"])
    planes = planes[:max_planes]

    for i, p in enumerate(planes):
        c = p["d"] * p["n"] + p["a_mid"] * p["u"] + p["b_mid"] * p["v"]
        prims.append(AtlasProxyPrimitive(
            name=f"projection_plane_{i + 1:02d}",
            primitive_type="plane",
            transform_matrix=plane_transform(p["u"], p["v"], p["n"], c),
            dimensions=(p["w"], p["h"], 0.0),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": "ransac_plane_extraction",
                      "inliers": p["inliers"],
                      "normal_azimuth_deg": float(p["azimuth_deg"]),
                      "normal_elevation_deg": float(p["elevation_deg"]),
                      "distance_m": float(p["d"]),
                      "depth_scale_applied": scale},
        ))
    stats["planes"] = len(planes)

    prims.append(build_backdrop_primitive(
        bp=bp, scaled_depth=scaled_depth, valid_depth=bp.valid_depth,
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, scale=scale,
        backdrop_depth_percentile=cfg.backdrop_depth_percentile,
        backdrop_margin=cfg.backdrop_margin,
    ))

    stats["primitives"] = len(prims)
    return prims, stats
