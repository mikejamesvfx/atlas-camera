"""Depth-derived proxy geometry for camera-projection (matte-painting) setups.

Given a monocular depth map and the recovered camera, derive simple proxy
geometry — ground plane, vertical wall planes, foreground boxes/cylinders, and a
far backdrop ("cyclorama") — as :class:`AtlasProxyPrimitive` entries. The blockout
viewport projects the source image onto this geometry from the recovered camera,
exactly like matte-painting projections in Nuke/Maya.

Why this works even with imperfect depth (the matte-painting property):
projective texturing assigns texels by *ray* through the recovered camera, which
is invariant to distance along the ray. Geometry at slightly wrong depth still
receives exactly the pixels its silhouette subtends — the image reassembles
perfectly from the camera view; scale errors only appear as parallax when
orbiting away. Additionally, step 3 below rescales the depth-derived world about
the camera so the fitted ground lands exactly on Y=0, pinning the depth map to
the solve's adopted metric camera height.

Convention (critical): everything here takes the full 4×4
``extrinsics.camera_view_matrix`` (row-major, world→cam, column-vector points,
translation in column 3) and uses its inverse — the one convention proven to
match the viewport shader. Never pass the 3×3 ``camera_rotation_matrix``.
Emitted ``transform_matrix`` values use the same convention, so the frontend can
feed the 16 floats directly to ``THREE.Matrix4.set()``.

Numpy-only: no torch, no scipy — safe for `atlas_camera.core`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Proxy geometry derivation requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


PROXY_ROLE = "projection_proxy"


@dataclass(slots=True)
class ProxyDerivationConfig:
    """Tunables for depth-derived proxy geometry. Defaults favour robustness on
    noisy AI-image depth: strict inlier floors, capped counts, graceful fallback
    to ground+backdrop (or backdrop-only)."""

    ground_normal_min: float = 0.90      # |n.y| above → horizontal surface
    wall_normal_max: float = 0.25        # |n.y| below → vertical surface
    depth_edge_rel: float = 0.05         # relative depth jump invalidating a normal
    azimuth_bins: int = 36               # 10° bins for wall-normal clustering
    min_wall_inliers: int = 1500         # absolute floor (also 0.3% of pixels)
    min_wall_size_m: float = 0.75        # reject slivers
    wall_min_width_m: float = 2.0        # narrower vertical fits are objects, not walls
    extent_percentiles: tuple[float, float] = (2.0, 98.0)
    backdrop_depth_percentile: float = 96.0
    backdrop_margin: float = 1.35        # frustum-coverage padding
    ground_padding: float = 1.1
    ground_min_extent_m: float = 4.0
    max_objects: int = 3
    min_object_inliers: int = 1200
    object_cell_m: float = 0.25          # occupancy-grid cell for blob clustering
    object_min_cell_pixels: int = 12     # occupancy threshold per cell
    cylinder_azimuth_spread_deg: float = 90.0
    cylinder_max_residual: float = 0.15  # × radius
    cylinder_radius_range: tuple[float, float] = (0.1, 5.0)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _plane_transform(u: Any, v: Any, n: Any, c: Any) -> tuple:
    """Row-major 4×4 with columns = local axes (u, v, n) and translation c.

    Locals are the THREE.PlaneGeometry frame: local X=u, Y=v, Z=n (plane normal).
    """
    return (
        (float(u[0]), float(v[0]), float(n[0]), float(c[0])),
        (float(u[1]), float(v[1]), float(n[1]), float(c[1])),
        (float(u[2]), float(v[2]), float(n[2]), float(c[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _wall_axes(np: Any, n: Any) -> tuple[Any, Any, Any]:
    """Right-handed (u, v, n) frame for a vertical plane with horizontal normal n.

    u = up × n (in-plane horizontal), v = world up. Then u × v = n holds for
    unit horizontal n, keeping the frame right-handed.
    """
    up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, n)
    u = u / (np.linalg.norm(u) or 1.0)
    return u, up, np.asarray(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------

def derive_projection_proxies(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_walls: int = 4,
    horizon_y: float | None = None,
    config: ProxyDerivationConfig | None = None,
) -> tuple[list[AtlasProxyPrimitive], dict[str, Any]]:
    """Derive proxy geometry from a forward-z depth map and the recovered camera.

    Returns ``(primitives, debug_stats)``. Always emits the backdrop; ground,
    walls, boxes and cylinders drop out gracefully when their fits are poor.
    """
    np = _require_numpy()
    cfg = config or ProxyDerivationConfig()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    if horizon_y is None:
        horizon_y = height * 0.45

    stats: dict[str, Any] = {"walls": 0, "boxes": 0, "cylinders": 0}
    prims: list[AtlasProxyPrimitive] = []

    # -- Step 1: back-project to world (camera pose from the view matrix) ------
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    R_cw = cam_to_world[:3, :3]
    cam_pos = cam_to_world[:3, 3]

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    x = (uu - cx) / fx * depth
    y = -(vv - cy) / fy * depth
    z = -depth
    pts_cam = np.stack([x, y, z], axis=-1)
    pts_world = pts_cam @ R_cw.T + cam_pos

    valid_depth = np.isfinite(depth) & (depth > 1e-4)

    # -- Step 2: normals + validity --------------------------------------------
    du = pts_world[:, 2:, :] - pts_world[:, :-2, :]
    dv = pts_world[2:, :, :] - pts_world[:-2, :, :]
    du = du[1:-1, :, :]
    dv = dv[:, 1:-1, :]
    normals_inner = np.cross(du, dv)
    nrm = np.linalg.norm(normals_inner, axis=-1, keepdims=True)
    normals_inner = normals_inner / np.maximum(nrm, 1e-12)

    normals = np.zeros((height, width, 3), dtype=np.float64)
    normals[1:-1, 1:-1] = normals_inner

    # Depth-discontinuity mask: silhouette edges produce garbage normals.
    ddx = np.abs(depth[:, 2:] - depth[:, :-2])
    ddy = np.abs(depth[2:, :] - depth[:-2, :])
    edge = np.zeros((height, width), dtype=bool)
    edge[:, 1:-1] |= ddx > cfg.depth_edge_rel * 2.0 * np.maximum(depth[:, 1:-1], 1e-6)
    edge[1:-1, :] |= ddy > cfg.depth_edge_rel * 2.0 * np.maximum(depth[1:-1, :], 1e-6)

    inner = np.zeros((height, width), dtype=bool)
    inner[1:-1, 1:-1] = True
    valid_normal = inner & valid_depth & ~edge

    n_y = normals[..., 1]
    world_y = pts_world[..., 1]

    # -- Step 3: ground fit + metric-scale reconciliation ----------------------
    below = vv > horizon_y
    ground_cand = valid_normal & below & (np.abs(n_y) > cfg.ground_normal_min)
    scale = 1.0
    ground_inlier = np.zeros((height, width), dtype=bool)
    y0 = None
    if int(ground_cand.sum()) >= 300:
        ys = world_y[ground_cand]
        lo, hi = np.percentile(ys, [1, 99])
        span = float(hi - lo)
        if span < 1e-3:
            y0 = float(np.median(ys))
        else:
            hist, edges = np.histogram(ys, bins=48, range=(lo, hi))
            peak = int(np.argmax(hist))
            y0 = 0.5 * (edges[peak] + edges[peak + 1])
        tol = max(0.15, 0.03 * max(span, 1e-3))
        refine = np.abs(ys - y0) < tol
        if int(refine.sum()) >= 50:
            y0 = float(np.median(ys[refine]))
        # Rescale the world about the camera so the fitted ground lands on Y=0,
        # pinning the depth map to the solve's adopted metric camera height.
        denom = cam_pos[1] - y0
        if cam_pos[1] > 1e-6 and denom > 1e-6:
            scale = float(cam_pos[1] / denom)
        pts_world = cam_pos + scale * (pts_world - cam_pos)
        world_y = pts_world[..., 1]
        ground_inlier = ground_cand & (np.abs(world_y) < tol * scale)
    stats["ground_scale"] = scale
    stats["ground_inliers"] = int(ground_inlier.sum())

    scaled_depth = depth * scale
    backdrop_d_raw = float(np.percentile(
        scaled_depth[valid_depth], cfg.backdrop_depth_percentile
    )) if valid_depth.any() else 60.0

    # -- Step 4: ground primitive ----------------------------------------------
    if int(ground_inlier.sum()) >= 300:
        gx = pts_world[..., 0][ground_inlier]
        gz = pts_world[..., 2][ground_inlier]
        p_lo, p_hi = cfg.extent_percentiles
        x0, x1 = np.percentile(gx, [p_lo, p_hi])
        z0, z1 = np.percentile(gz, [p_lo, p_hi])
        cx_w, cz_w = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
        ex = max((x1 - x0) * cfg.ground_padding, cfg.ground_min_extent_m)
        ez = max((z1 - z0) * cfg.ground_padding, cfg.ground_min_extent_m)
        u_g = np.array([1.0, 0.0, 0.0])
        n_g = np.array([0.0, 1.0, 0.0])
        v_g = np.array([0.0, 0.0, -1.0])  # u × v = n (right-handed)
        prims.append(AtlasProxyPrimitive(
            name="projection_ground",
            primitive_type="plane",
            transform_matrix=_plane_transform(u_g, v_g, n_g, (cx_w, 0.0, cz_w)),
            dimensions=(float(ex), float(ez), 0.0),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                      "inliers": int(ground_inlier.sum()),
                      "depth_scale_applied": scale},
        ))

    # -- Step 5: wall primitives ------------------------------------------------
    wall_inlier_total = np.zeros((height, width), dtype=bool)
    wall_cand = (valid_normal & (np.abs(n_y) < cfg.wall_normal_max)
                 & (scaled_depth < backdrop_d_raw * 0.95) & ~ground_inlier)
    n_pixels = height * width
    inlier_floor = max(cfg.min_wall_inliers, int(0.003 * n_pixels))
    walls: list[dict[str, Any]] = []

    if int(wall_cand.sum()) >= inlier_floor and max_walls > 0:
        nx = normals[..., 0][wall_cand].copy()
        nz = normals[..., 2][wall_cand].copy()
        pw = pts_world[wall_cand]
        # Flip each normal toward the camera, zero Y, renormalise.
        to_cam = cam_pos[None, :] - pw
        flip = (nx * to_cam[:, 0] + normals[..., 1][wall_cand] * to_cam[:, 1]
                + nz * to_cam[:, 2]) < 0
        nx[flip] = -nx[flip]
        nz[flip] = -nz[flip]
        h_norm = np.sqrt(nx ** 2 + nz ** 2)
        ok = h_norm > 1e-6
        nx, nz, pw = nx[ok] / h_norm[ok], nz[ok] / h_norm[ok], pw[ok]

        azimuth = np.arctan2(nx, nz)  # [-pi, pi]
        bins = cfg.azimuth_bins
        bin_idx = ((azimuth + math.pi) / (2 * math.pi) * bins).astype(int) % bins
        hist = np.bincount(bin_idx, minlength=bins).astype(np.float64)
        # Circular [1,2,1] smoothing.
        hist_s = (np.roll(hist, 1) + 2 * hist + np.roll(hist, -1)) / 4.0

        # NMS peak-picking.
        order = np.argsort(hist_s)[::-1]
        suppressed = np.zeros(bins, dtype=bool)
        peaks: list[int] = []
        for b in order:
            if suppressed[b] or hist_s[b] < inlier_floor:
                continue
            peaks.append(int(b))
            for off in range(-2, 3):
                suppressed[(b + off) % bins] = True
            if len(peaks) >= max_walls * 2:  # extra candidates; filtered below
                break

        up = np.array([0.0, 1.0, 0.0])
        for b in peaks:
            if len(walls) >= max_walls:
                break
            # Cluster: pixels within ±1.5 bins of the peak azimuth (wraparound).
            centre = -math.pi + (b + 0.5) * 2 * math.pi / bins
            dang = np.abs(np.arctan2(np.sin(azimuth - centre), np.cos(azimuth - centre)))
            sel = dang < (1.5 * 2 * math.pi / bins)
            if int(sel.sum()) < inlier_floor:
                continue
            n_mean = np.array([np.mean(nx[sel]), 0.0, np.mean(nz[sel])])
            n_len = np.linalg.norm(n_mean)
            if n_len < 1e-6:
                continue
            n_mean = n_mean / n_len
            p_sel = pw[sel]
            offs = p_sel @ n_mean
            d_off = float(np.median(offs))
            med_depth = float(np.median(scaled_depth[wall_cand][ok][sel])) if sel.any() else 5.0
            tol = max(0.15, 0.02 * med_depth)
            inl = np.abs(offs - d_off) < tol
            if int(inl.sum()) < inlier_floor:
                continue
            p_in = p_sel[inl]
            u_ax, v_ax, _ = _wall_axes(np, n_mean)
            a = p_in @ u_ax
            b_y = np.clip(p_in[:, 1], -0.1, None)
            p_lo, p_hi = cfg.extent_percentiles
            a0, a1 = np.percentile(a, [p_lo, p_hi])
            b0, b1 = np.percentile(b_y, [p_lo, p_hi])
            b0 = max(b0, -0.1)
            w_m, h_m = float(a1 - a0), float(b1 - b0)
            if w_m < cfg.min_wall_size_m or h_m < cfg.min_wall_size_m:
                continue
            if w_m < cfg.wall_min_width_m:
                continue  # narrow vertical fit — an object, not a wall
            walls.append({
                "n": n_mean, "d": d_off, "u": u_ax,
                "a_mid": 0.5 * (a0 + a1), "b_mid": 0.5 * (b0 + b1),
                "w": w_m, "h": h_m, "inliers": int(inl.sum()),
                "med_depth": med_depth,
            })
            # Mark inlier pixels so the object stage skips them.
            plane_dist = np.abs(pts_world @ n_mean - d_off)
            wall_inlier_total |= wall_cand & (plane_dist < tol)

        # Merge near-duplicates (azimuth < 15° apart AND offsets agree).
        merged: list[dict[str, Any]] = []
        for w in sorted(walls, key=lambda d: -d["inliers"]):
            dup = False
            for m in merged:
                cosang = float(np.dot(w["n"], m["n"]))
                if cosang > math.cos(math.radians(15.0)) and \
                        abs(w["d"] - m["d"]) < max(0.3, 0.05 * m["med_depth"]):
                    dup = True
                    break
            if not dup:
                merged.append(w)
        walls = merged[:max_walls]

    for i, w in enumerate(walls):
        u_ax, v_ax, n_ax = _wall_axes(np, w["n"])
        c = w["d"] * n_ax + w["a_mid"] * u_ax + w["b_mid"] * v_ax
        prims.append(AtlasProxyPrimitive(
            name=f"projection_wall_{i + 1:02d}",
            primitive_type="plane",
            transform_matrix=_plane_transform(u_ax, v_ax, n_ax, c),
            dimensions=(w["w"], w["h"], 0.0),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                      "inliers": w["inliers"],
                      "yaw_deg": float(math.degrees(math.atan2(w["n"][0], w["n"][2]))),
                      "distance_m": float(w["d"]),
                      "depth_scale_applied": scale},
        ))
    stats["walls"] = len(walls)

    # -- Steps 6+7: foreground objects (boxes / cylinders) ----------------------
    if cfg.max_objects > 0:
        obj_cand = (valid_depth & inner & ~ground_inlier & ~wall_inlier_total
                    & (scaled_depth < backdrop_d_raw * 0.9)
                    & (world_y > 0.05))
        prims.extend(_derive_objects(
            np, cfg, stats, pts_world, normals, obj_cand, valid_normal, scale))

    # -- Step 8: backdrop (always) ----------------------------------------------
    D = 1.02 * backdrop_d_raw
    fwd = R_cw @ np.array([0.0, 0.0, -1.0])
    fwd_h = np.array([fwd[0], 0.0, fwd[2]])
    fl = np.linalg.norm(fwd_h)
    fwd_h = fwd_h / fl if fl > 1e-6 else np.array([0.0, 0.0, -1.0])
    n_b = -fwd_h
    u_b, v_b, _ = _wall_axes(np, n_b)
    c0 = cam_pos + fwd_h * D  # point on the backdrop plane

    # Exact frame coverage: intersect the four frustum-corner rays with the
    # backdrop plane and enclose the hits (plus margin). A flat margin around the
    # horizontal frustum leaves corner gaps under pitch/roll — this doesn't.
    us: list[float] = []
    ys_: list[float] = []
    for (u_px, v_px) in ((0.0, 0.0), (float(width), 0.0),
                         (0.0, float(height)), (float(width), float(height))):
        d_cam = np.array([(u_px - cx) / fx, -(v_px - cy) / fy, -1.0])
        d_w = R_cw @ d_cam
        denom = float(np.dot(n_b, d_w))
        if abs(denom) < 1e-6:
            continue
        t = float(np.dot(n_b, c0 - cam_pos)) / denom
        if t <= 0:
            continue
        p = cam_pos + min(t, 4.0 * D) * d_w
        us.append(float(np.dot(p - c0, u_b)))
        ys_.append(float(p[1]))
    if not us:  # degenerate (camera looking straight up/down)
        us, ys_ = [-D, D], [0.0, D]
    mfrac = cfg.backdrop_margin - 1.0
    u_lo, u_hi = min(us), max(us)
    y_lo, y_hi = min(ys_), max(ys_)
    u_pad = 0.5 * (u_hi - u_lo) * mfrac + 1.0
    y_pad = 0.5 * (y_hi - y_lo) * mfrac + 1.0
    u_lo -= u_pad
    u_hi += u_pad
    y_lo = min(y_lo - y_pad, -1.0)  # always reach below the ground plane
    y_hi += y_pad
    bw = u_hi - u_lo
    bh = y_hi - y_lo
    c_b = c0 + u_b * (0.5 * (u_lo + u_hi))
    c_b[1] = 0.5 * (y_lo + y_hi)
    prims.append(AtlasProxyPrimitive(
        name="projection_backdrop",
        primitive_type="plane",
        transform_matrix=_plane_transform(u_b, v_b, n_b, c_b),
        dimensions=(float(bw), float(bh), 0.0),
        material="atlas_projection_proxy",
        metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                  "distance_m": float(D), "depth_scale_applied": scale},
    ))

    stats["primitives"] = len(prims)
    return prims, stats


def _derive_objects(np, cfg, stats, pts_world, normals, obj_cand, valid_normal,
                    scale) -> list[AtlasProxyPrimitive]:
    """Cluster remaining close-range pixels into boxes / cylinders via an XZ
    occupancy grid + connected components (pure python BFS at grid resolution)."""
    prims: list[AtlasProxyPrimitive] = []
    if int(obj_cand.sum()) < cfg.min_object_inliers:
        return prims

    px = pts_world[..., 0][obj_cand]
    py = pts_world[..., 1][obj_cand]
    pz = pts_world[..., 2][obj_cand]
    naz = normals[..., 0][obj_cand], normals[..., 2][obj_cand]
    has_norm = valid_normal[obj_cand]

    cell = cfg.object_cell_m
    x_min, z_min = float(px.min()), float(pz.min())
    gx = ((px - x_min) / cell).astype(int)
    gz = ((pz - z_min) / cell).astype(int)
    # Cap grid size (very large scenes → coarser cells).
    max_cells = 400
    if gx.max() >= max_cells or gz.max() >= max_cells:
        f = max(gx.max(), gz.max()) / (max_cells - 1)
        gx = (gx / f).astype(int)
        gz = (gz / f).astype(int)
    gw, gh = int(gx.max()) + 1, int(gz.max()) + 1

    counts = np.zeros((gw, gh), dtype=int)
    np.add.at(counts, (gx, gz), 1)
    occupied = counts >= cfg.object_min_cell_pixels

    # Connected components (4-connectivity BFS at grid resolution).
    labels = np.full((gw, gh), -1, dtype=int)
    comp = 0
    for sx in range(gw):
        for sz in range(gh):
            if not occupied[sx, sz] or labels[sx, sz] >= 0:
                continue
            stack = [(sx, sz)]
            labels[sx, sz] = comp
            while stack:
                ix, iz = stack.pop()
                for dx_, dz_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jx, jz = ix + dx_, iz + dz_
                    if 0 <= jx < gw and 0 <= jz < gh and occupied[jx, jz] \
                            and labels[jx, jz] < 0:
                        labels[jx, jz] = comp
                        stack.append((jx, jz))
            comp += 1
    if comp == 0:
        return prims

    pix_label = labels[gx, gz]  # -1 where cell below occupancy threshold
    sizes = [(int((pix_label == c).sum()), c) for c in range(comp)]
    sizes.sort(reverse=True)

    n_box = n_cyl = 0
    for size, c in sizes:
        if len(prims) >= cfg.max_objects or size < cfg.min_object_inliers:
            break
        sel = pix_label == c
        ox, oy, oz = px[sel], py[sel], pz[sel]
        h_obj = float(np.percentile(oy, 98.0))
        if h_obj < 0.2:
            continue

        # Cylinder test: vertical-normal azimuth spread (curved surface sweeps
        # azimuth; a flat face is constant).
        made_cyl = False
        sel_n = sel & has_norm
        if int(sel_n.sum()) > 100:
            nxs, nzs = naz[0][sel_n], naz[1][sel_n]
            hn = np.sqrt(nxs ** 2 + nzs ** 2)
            okn = hn > 0.5  # reasonably horizontal normals only
            if int(okn.sum()) > 100:
                az = np.arctan2(nxs[okn] / hn[okn], nzs[okn] / hn[okn])
                bins = 36
                occ_bins = np.unique(((az + math.pi) / (2 * math.pi) * bins)
                                     .astype(int) % bins)
                spread_deg = len(occ_bins) * (360.0 / bins)
                if spread_deg > cfg.cylinder_azimuth_spread_deg:
                    cyl = _fit_cylinder(np, cfg, ox, oz)
                    if cyl is not None:
                        r, ccx, ccz = cyl
                        prims.append(AtlasProxyPrimitive(
                            name=f"projection_cylinder_{n_cyl + 1:02d}",
                            primitive_type="cylinder",
                            transform_matrix=_plane_transform(
                                np.array([1.0, 0.0, 0.0]),
                                np.array([0.0, 1.0, 0.0]),
                                np.array([0.0, 0.0, 1.0]),
                                (ccx, h_obj / 2.0, ccz)),
                            dimensions=(2.0 * r, h_obj, 2.0 * r),
                            material="atlas_projection_proxy",
                            metadata={"role": PROXY_ROLE,
                                      "source": "depth_derivation",
                                      "inliers": size, "radius_m": float(r),
                                      "depth_scale_applied": scale},
                        ))
                        n_cyl += 1
                        made_cyl = True

        if made_cyl:
            continue

        # Box: PCA on XZ for yaw, extents from percentiles, base on ground.
        xz = np.stack([ox, oz], axis=1)
        centre = xz.mean(axis=0)
        d0 = xz - centre
        cov = d0.T @ d0 / max(len(d0), 1)
        evals, evecs = np.linalg.eigh(cov)
        e1 = evecs[:, int(np.argmax(evals))]  # principal XZ direction
        u_ax = np.array([e1[0], 0.0, e1[1]])
        u_ax = u_ax / (np.linalg.norm(u_ax) or 1.0)
        v_ax = np.array([0.0, 1.0, 0.0])
        w_ax = np.cross(u_ax, v_ax)
        a = d0 @ e1
        e2 = np.array([-e1[1], e1[0]])
        b = d0 @ e2
        p_lo, p_hi = cfg.extent_percentiles
        a0, a1 = np.percentile(a, [p_lo, p_hi])
        b0, b1 = np.percentile(b, [p_lo, p_hi])
        sx_m = max(float(a1 - a0), 0.1)
        sz_m = max(float(b1 - b0), 0.1)
        # w_ax corresponds to -e2 direction in XZ: cross((e1x,0,e1z),(0,1,0)) =
        # (-e1z, 0, e1x) = (e2x, 0, e2y)... verify: e2 = (-e1z, e1x) matches. Good.
        c_w = np.array([centre[0], 0.0, centre[1]])
        c_w = c_w + u_ax * float(0.5 * (a0 + a1))
        c_w = c_w + w_ax * float(0.5 * (b0 + b1))
        c_w[1] = h_obj / 2.0
        prims.append(AtlasProxyPrimitive(
            name=f"projection_box_{n_box + 1:02d}",
            primitive_type="box",
            transform_matrix=_plane_transform(u_ax, v_ax, w_ax, c_w),
            dimensions=(sx_m, h_obj, sz_m),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                      "inliers": size,
                      "yaw_deg": float(math.degrees(math.atan2(u_ax[0], u_ax[2]))),
                      "depth_scale_applied": scale},
        ))
        n_box += 1

    stats["boxes"] = n_box
    stats["cylinders"] = n_cyl
    return prims


def _fit_cylinder(np, cfg, ox, oz):
    """Kåsa algebraic circle fit on XZ points → (radius, cx, cz) or None."""
    A = np.stack([ox, oz, np.ones_like(ox)], axis=1)
    b = ox ** 2 + oz ** 2
    try:
        coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    ccx, ccz = coeff[0] / 2.0, coeff[1] / 2.0
    r2 = coeff[2] + ccx ** 2 + ccz ** 2
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    lo, hi = cfg.cylinder_radius_range
    if not (lo <= r <= hi):
        return None
    dist = np.sqrt((ox - ccx) ** 2 + (oz - ccz) ** 2)
    residual = float(np.std(dist - r))
    if residual > cfg.cylinder_max_residual * r:
        return None
    return r, float(ccx), float(ccz)


# ---------------------------------------------------------------------------
# Preview-only dilation — widen derived geometry's orbit coverage in the
# blockout viewport without altering the geometry used for measurement or
# DCC export (those read the untouched AtlasProxyPrimitive objects on the
# solve; this only ever runs at viewport-payload serialization time).
#
# Derived geometry only ever covers what the recovered camera could see (a
# forward-facing cone, inherent to reconstruction from a single photo). One
# equation widens that coverage for ANY primitive regardless of its normal:
# for a point p with local surface normal n̂ (arbitrary — a plane's fixed
# normal, a mesh vertex's own normal, or none for a volume), radiating from a
# pivot (the recovered camera position):
#
#   p' = pivot + ((p-pivot)·n̂)n̂ + scale · [(p-pivot) - ((p-pivot)·n̂)n̂]
#
# Only the component of the offset-from-pivot PERPENDICULAR to n̂ is scaled;
# the normal-aligned (depth-from-pivot) component is preserved — a plane's
# footprint grows without drifting toward/away from the camera, a box/
# cylinder (no single normal) dilates uniformly, and a mesh dilates per-
# vertex using each vertex's own, genuinely arbitrary normal.
# ---------------------------------------------------------------------------

def dilate_proxy_geometry_for_preview(
    prims: list[AtlasProxyPrimitive],
    *,
    pivot: Any,
    scale: float,
) -> list[AtlasProxyPrimitive]:
    """Return NEW primitives dilated outward from ``pivot`` by ``scale`` (>1
    grows). Never mutates the input; ``scale <= 1`` returns it unchanged."""
    if scale <= 1.0 + 1e-9:
        return list(prims)
    np = _require_numpy()
    pivot = np.asarray(pivot, dtype=np.float64)
    out: list[AtlasProxyPrimitive] = []
    for prim in prims:
        if prim.primitive_type == "mesh":
            out.append(_dilate_mesh_primitive(prim, pivot, scale))
        elif prim.primitive_type == "plane":
            out.append(_dilate_plane_primitive(prim, pivot, scale))
        else:  # box, cylinder: no single preferred normal — uniform radial dilation
            out.append(_dilate_volume_primitive(prim, pivot, scale))
    return out


def _dilate_plane_primitive(prim: AtlasProxyPrimitive, pivot: Any, scale: float) -> AtlasProxyPrimitive:
    np = _require_numpy()
    M = np.array(prim.transform_matrix, dtype=np.float64)
    u, v, n, c = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
    d = c - pivot
    d_n = float(np.dot(d, n)) * n
    d_t = d - d_n
    c2 = pivot + d_n + scale * d_t
    dims = prim.dimensions
    return AtlasProxyPrimitive(
        name=prim.name, primitive_type=prim.primitive_type,
        transform_matrix=_plane_transform(u, v, n, c2),
        dimensions=(dims[0] * scale, dims[1] * scale, dims[2]),
        material=prim.material,
        metadata={**(prim.metadata or {}), "preview_dilated": True, "preview_scale": float(scale)},
    )


def _dilate_volume_primitive(prim: AtlasProxyPrimitive, pivot: Any, scale: float) -> AtlasProxyPrimitive:
    np = _require_numpy()
    M = np.array(prim.transform_matrix, dtype=np.float64)
    u, v, w, c = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
    c2 = pivot + scale * (c - pivot)
    dims = prim.dimensions
    return AtlasProxyPrimitive(
        name=prim.name, primitive_type=prim.primitive_type,
        transform_matrix=_plane_transform(u, v, w, c2),
        dimensions=tuple(dv * scale for dv in dims),
        material=prim.material,
        metadata={**(prim.metadata or {}), "preview_dilated": True, "preview_scale": float(scale)},
    )


def _dilate_mesh_primitive(prim: AtlasProxyPrimitive, pivot: Any, scale: float) -> AtlasProxyPrimitive:
    np = _require_numpy()
    md = prim.metadata or {}
    verts_flat = md.get("vertices") or []
    faces_flat = md.get("faces") or []
    if not verts_flat or not faces_flat:
        return prim
    verts = np.array(verts_flat, dtype=np.float64).reshape(-1, 3)
    faces = np.array(faces_flat, dtype=np.int64).reshape(-1, 3)

    # Per-vertex normals: area-weighted average of adjacent face normals
    # (cross-product magnitude is proportional to triangle area, so a plain
    # sum before normalising is already area-weighted).
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-12)

    d = verts - pivot[None, :]
    d_n = np.sum(d * vertex_normals, axis=1, keepdims=True) * vertex_normals
    d_t = d - d_n
    verts2 = pivot[None, :] + d_n + scale * d_t

    new_meta = dict(md)
    new_meta["vertices"] = [round(float(x), 3) for x in verts2.reshape(-1)]
    new_meta["preview_dilated"] = True
    new_meta["preview_scale"] = float(scale)
    return AtlasProxyPrimitive(
        name=prim.name, primitive_type=prim.primitive_type,
        transform_matrix=prim.transform_matrix,  # identity; vertices are already world-space
        dimensions=prim.dimensions, material=prim.material,
        metadata=new_meta,
    )


# ---------------------------------------------------------------------------
# Serialization for the blockout payload
# ---------------------------------------------------------------------------

def serialize_proxy_geometry(
    scene: AtlasProjectionScene,
    *,
    preview_expand: float = 1.0,
    preview_pivot: Any = None,
) -> list[dict[str, Any]]:
    """JSON-safe payload entries for derivation proxies (role == projection_proxy).

    ``transform`` is the row-major 4×4 flattened to 16 floats — feed directly to
    ``THREE.Matrix4.set()``. ``mesh`` primitives (the relief mesh) additionally
    carry flat ``vertices`` / ``faces`` / ``uvs`` arrays for a Three.js
    BufferGeometry (vertices are already world-space; transform is identity).

    ``preview_expand`` (>1, needs ``preview_pivot``) dilates the geometry for
    wider blockout-viewport orbit coverage via :func:`dilate_proxy_geometry_for_preview`
    — display-only, never touches the primitives stored on the solve.
    """
    prims = [p for p in scene.proxy_geometry if (p.metadata or {}).get("role") == PROXY_ROLE]
    if preview_expand > 1.0 + 1e-9 and preview_pivot is not None:
        prims = dilate_proxy_geometry_for_preview(prims, pivot=preview_pivot, scale=preview_expand)

    out: list[dict[str, Any]] = []
    for prim in prims:
        flat = [float(v) for row in prim.transform_matrix for v in row]
        meta = {k: v for k, v in (prim.metadata or {}).items()
                if isinstance(v, (str, int, float, bool)) or v is None}
        entry: dict[str, Any] = {
            "name": prim.name,
            "type": prim.primitive_type,
            "transform": flat,
            "dimensions": [float(v) for v in prim.dimensions],
            "material": prim.material,
            "metadata": meta,
        }
        if prim.primitive_type == "mesh":
            md = prim.metadata or {}
            entry["vertices"] = md.get("vertices", [])
            entry["faces"] = md.get("faces", [])
            entry["uvs"] = md.get("uvs", [])
        out.append(entry)
    return out


def relief_mesh_primitive(mesh: Any, *, name: str = "projection_relief_mesh") -> AtlasProxyPrimitive:
    """Wrap a :class:`~atlas_camera.core.relief_mesh.ReliefMesh` as a proxy
    primitive so it rides the solve into the blockout viewport payload.

    Arrays are flattened and rounded (mm / 1e-4 UV) to keep the JSON compact.
    """
    verts = [round(float(v), 3) for v in mesh.vertices.reshape(-1)]
    faces = [int(i) for i in mesh.faces.reshape(-1)]
    uvs = [round(float(v), 4) for v in mesh.uvs.reshape(-1)]
    return AtlasProxyPrimitive(
        name=name,
        primitive_type="mesh",
        dimensions=(0.0, 0.0, 0.0),
        material="atlas_projection_proxy",
        metadata={
            "role": PROXY_ROLE,
            "source": "depth_relief_mesh",
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
            "vertices": verts,
            "faces": faces,
            "uvs": uvs,
        },
    )
