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
    object_max_height_m: float = 6.0     # plausibility ceiling — see its use site
    cylinder_azimuth_spread_deg: float = 90.0
    cylinder_max_residual: float = 0.15  # × radius
    cylinder_radius_range: tuple[float, float] = (0.1, 5.0)
    # Walls per azimuth DIRECTION (1 = classic behavior: one plane at the
    # median distance of everything facing that way). A street-grid skyline
    # has ~2 dominant azimuths but MANY depths — raise this so each azimuth
    # peak splits into one wall per depth mode (building row) instead of
    # collapsing thirty facades into one slab at the median depth.
    wall_distance_modes: int = 1
    # Ground-anchored walls: the wall's DISTANCE comes from ray-through-
    # base-pixel ∩ the analytic Y=0 ground plane — pure geometry, immune to
    # monocular depth's low-frequency "banana" warp (the depth model is
    # demoted to grouping pixels). Requires VISIBLE ground contact; walls
    # whose base is occluded or too near the horizon fall back smoothly to
    # the depth-median distance (see anchor_horizon_frac).
    ground_anchor: bool = False
    # Base pixels within this fraction of the below-horizon image span are
    # ill-conditioned (ray nearly parallel to ground) — the geometric
    # distance fades toward the depth median across this band.
    anchor_horizon_frac: float = 0.15
    # Only refit the wall's ORIENTATION from the base contact line when the
    # footprint spans at least this many metres (short/noisy base lines keep
    # the normal-cluster azimuth and anchor distance only).
    anchor_min_line_m: float = 2.0
    # Roofline segmentation (vertical_extrusion only): split a wall cluster
    # at silhouette-height steps bigger than this fraction of the wall's
    # median height, so a row of buildings becomes one plane per roofline
    # instead of one rectangle spanning sky above the shorter buildings.
    roofline_split: bool = False
    roofline_split_rel: float = 0.25


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
# Shared wall-orientation clustering (azimuth_walls AND vertical_extrusion)
# ---------------------------------------------------------------------------


def _anchor_wall_to_ground(np: Any, cfg: "ProxyDerivationConfig", wall: dict,
                           cam_pos: Any, horizon_y: float | None,
                           height: int) -> dict:
    """Ground-anchor one wall: distance (and, when well-conditioned,
    orientation) from ray-through-base-pixel ∩ the analytic Y=0 plane.

    The ray DIRECTION through a pixel is exact regardless of depth error, so
    where the building visibly meets the ground the footprint drops out of
    pure geometry — monocular depth's low-frequency "banana" never enters.
    Three safeties, each with a reason:

    1. **Occlusion poison-gate** (critical): if the true contact is hidden
       (car/fence), the lowest VISIBLE wall pixel sits above the ground and
       its ray lands far BEHIND the building — a 1.5 m occluder at eye
       height overshoots by hundreds of metres, not a little. Any base point
       whose ray-ground distance exceeds 1.6x its own depth-space distance
       is treated as occluded and dropped; too few survivors = no anchor.
    2. **Horizon conditioning**: rays through pixels near the horizon are
       nearly parallel to the ground — the geometric distance fades toward
       the depth median across cfg.anchor_horizon_frac of the below-horizon
       span, so far facades degrade to today's behavior instead of blowing up.
    3. **Conservative orientation**: the footprint line only replaces the
       normal-cluster azimuth when it spans >= cfg.anchor_min_line_m with a
       tight lateral fit; short/noisy contact lines anchor distance only.

    Returns the wall dict (mutated) with ``anchored``/``anchor_weight`` set.
    Early returns leave any EXISTING anchor state untouched: a roofline
    segment whose own re-anchor can't gather enough clean contact keeps its
    parent cluster's anchor (a failed refinement is not a failed anchor).
    """
    wall.setdefault("anchored", False)
    p_in = wall["pts"]
    cols_in = wall.get("base_cols", wall["cols"])
    rows_in = wall.get("base_rows", wall["rows"])
    p_base = wall.get("base_pts", p_in)
    if cols_in.size < 8 or cam_pos[1] <= 1e-3:
        return wall
    # Per-column base pixel = the lowest pool pixel (largest image row).
    order = np.lexsort((rows_in, cols_in))
    c_s, r_s, p_s = cols_in[order], rows_in[order], p_base[order]
    last = np.r_[c_s[1:] != c_s[:-1], True]
    base_rows, base_pts = r_s[last], p_s[last]
    n_base_cols = int(base_rows.size)

    # Keep only genuine contact candidates: junction filtering (ground
    # inliers, depth-edge invalidation) eats base pixels COLUMN-WISE, and a
    # column whose lowest surviving pixel sits a metre up would poison the
    # gate below. The anchor doesn't need every column — just enough clean
    # contact — so restrict to the lowest 0.6 m band of visible bases and
    # let the coverage fraction feed the fusion weight instead.
    if base_pts.size:
        y_lo = float(np.min(base_pts[:, 1]))
        band = base_pts[:, 1] < y_lo + 0.6
        base_rows, base_pts = base_rows[band], base_pts[band]
    if base_rows.size < 8:
        return wall

    dirs = base_pts - cam_pos[None, :]
    dn = np.linalg.norm(dirs, axis=1)
    ok = dn > 1e-9
    dirs, dn, base_rows = dirs[ok] / dn[ok, None], dn[ok], base_rows[ok]
    down = dirs[:, 1] < -1e-4          # ray must descend to reach Y=0
    if int(down.sum()) < 8:
        return wall
    dirs, dn, base_rows = dirs[down], dn[down], base_rows[down]
    t = -cam_pos[1] / dirs[:, 1]
    sane = t < 1.6 * dn                # occlusion poison-gate (see above)
    if int(sane.sum()) < 8:
        return wall
    g = cam_pos[None, :] + t[sane, None] * dirs[sane]
    base_rows = base_rows[sane]

    hy = horizon_y if horizon_y is not None else 0.45 * height
    span = max(float(height) - hy, 1.0)
    cond = np.clip((base_rows - hy) / max(cfg.anchor_horizon_frac * span, 1e-6),
                   0.0, 1.0)
    coverage = float(sane.sum()) / max(float(n_base_cols), 1.0)
    w_geo = float(np.median(cond)) * min(1.0, 2.0 * coverage)
    if w_geo <= 1e-3:
        return wall

    n = np.asarray(wall["n"], dtype=np.float64)
    # Orientation refit from the footprint line (plan-view PCA), gated.
    gx, gz = g[:, 0], g[:, 2]
    cx_, cz_ = float(np.mean(gx)), float(np.mean(gz))
    X = np.column_stack([gx - cx_, gz - cz_])
    if X.shape[0] >= 8:
        cov = X.T @ X / X.shape[0]
        evals, evecs = np.linalg.eigh(cov)
        line_dir = evecs[:, -1]        # principal (along-wall) direction
        length = float(np.ptp(X @ line_dir))
        lateral_rms = float(np.sqrt(max(evals[0], 0.0)))
        if length >= cfg.anchor_min_line_m and lateral_rms < max(0.15 * length, 0.3):
            n_new = np.array([-line_dir[1], 0.0, line_dir[0]])
            n_new /= np.linalg.norm(n_new) or 1.0
            to_cam = np.array([cam_pos[0] - cx_, 0.0, cam_pos[2] - cz_])
            if float(n_new @ to_cam) < 0:
                n_new = -n_new
            # keep the cluster azimuth unless the refit stays within 25 deg
            if float(n_new @ n) > math.cos(math.radians(25.0)):
                n = n_new
    d_geo = float(np.median(g @ n))
    d_depth = float(np.median(p_in @ n))
    # Contamination gate — the anchor REFINES, it never teleports: street
    # clutter (cars, fences) sharing the wall's azimuth donates its own base
    # pixels, and without this gate a 12 m facade "anchors" to the 2 m car
    # row in front of it (found live on a real street photo). The geometric
    # footprint must land within ±50% of the depth fit or it is rejected as
    # foreign contact.
    if not (0.5 * abs(d_depth) <= abs(d_geo) <= 1.5 * abs(d_depth)):
        return wall
    d_final = w_geo * d_geo + (1.0 - w_geo) * d_depth

    u_ax, v_ax, _ = _wall_axes(np, n)
    b_y = np.clip(p_in[:, 1], -0.1, None)
    p_lo, p_hi = cfg.extent_percentiles
    # The anchored wall's WIDTH comes from its footprint too: the cluster's
    # pixel extents can span same-facing surfaces far beyond this building,
    # and an over-wide plane at a near distance sweeps right up to the
    # camera (found live: a 5.7m facade's slab reached across the street).
    # The base contact line knows where the building starts and ends. Only
    # trusted when contact coverage is decent — poor coverage keeps the
    # classic pixel extents.
    if w_geo >= 0.5:
        a_base = g @ u_ax
        a0, a1 = np.percentile(a_base, [p_lo, p_hi])
        a0, a1 = float(a0) - 0.5, float(a1) + 0.5  # roofs overhang bases a bit
    else:
        a = p_in @ u_ax
        a0, a1 = np.percentile(a, [p_lo, p_hi])
    _b0, b1 = np.percentile(b_y, [p_lo, p_hi])
    # An anchored building SITS ON the ground: extrude from Y=0, not from the
    # lowest surviving wall pixel (near-base pixels are routinely eaten by
    # ground-inlier/depth-edge filtering, leaving walls floating ~1m up).
    b0 = 0.0
    wall.update({
        "n": n, "d": d_final, "u": u_ax,
        "a_mid": 0.5 * float(a0 + a1), "b_mid": 0.5 * float(b0 + b1),
        "w": float(a1 - a0), "h": float(b1 - b0),
        "anchored": True, "anchor_weight": round(w_geo, 3),
    })
    return wall


def _cluster_walls_by_azimuth(
    np: Any,
    cfg: "ProxyDerivationConfig",
    *,
    pts_world: Any,
    normals: Any,
    valid_normal: Any,
    ground_inlier: Any,
    scaled_depth: Any,
    backdrop_d_raw: float,
    cam_pos: Any,
    max_walls: int,
    height: int,
    width: int,
    horizon_y: float | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Cluster near-vertical-normal pixels into wall planes by azimuth peak-picking.

    Extracted verbatim from ``derive_projection_proxies`` (step 5) so both the
    default ``azimuth_walls`` strategy and ``vertical_extrusion`` (which only
    replaces the *height* computation below — see that function) agree on
    wall orientation/distance instead of duplicating this clustering twice.

    Returns ``(walls, wall_inlier_total)``. Each wall dict has
    ``n``/``d``/``u``/``a_mid``/``b_mid``/``w``/``h``/``inliers``/``med_depth``/``cols``
    — ``h``/``b_mid`` come from a 2nd-98th percentile clip of *that wall's own*
    3D inlier points. This is ``azimuth_walls``'s known limitation: normals
    off a sloped roof or spire never pass the ``wall_normal_max`` filter, so
    the height only ever reflects the lower straight wall section. ``cols``
    (each inlier's original image column) exists purely so
    ``vertical_extrusion`` can look up, per column, the silhouette top —
    ``azimuth_walls`` itself ignores it.
    """
    n_y = normals[..., 1]
    wall_inlier_total = np.zeros((height, width), dtype=bool)
    wall_cand = (valid_normal & (np.abs(n_y) < cfg.wall_normal_max)
                 & (scaled_depth < backdrop_d_raw * 0.95) & ~ground_inlier)
    n_pixels = height * width
    inlier_floor = max(cfg.min_wall_inliers, int(0.003 * n_pixels))
    walls: list[dict[str, Any]] = []
    col_idx_full = np.tile(np.arange(width, dtype=np.int64), (height, 1))
    row_idx_full = np.tile(np.arange(height, dtype=np.int64)[:, None], (1, width))

    if int(wall_cand.sum()) >= inlier_floor and max_walls > 0:
        nx = normals[..., 0][wall_cand].copy()
        nz = normals[..., 2][wall_cand].copy()
        pw = pts_world[wall_cand]
        cols = col_idx_full[wall_cand]
        rows = row_idx_full[wall_cand]
        # Flip each normal toward the camera, zero Y, renormalise.
        to_cam = cam_pos[None, :] - pw
        flip = (nx * to_cam[:, 0] + normals[..., 1][wall_cand] * to_cam[:, 1]
                + nz * to_cam[:, 2]) < 0
        nx[flip] = -nx[flip]
        nz[flip] = -nz[flip]
        h_norm = np.sqrt(nx ** 2 + nz ** 2)
        ok = h_norm > 1e-6
        nx, nz = nx[ok] / h_norm[ok], nz[ok] / h_norm[ok]
        pw, cols, rows = pw[ok], cols[ok], rows[ok]

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
            cols_sel = cols[sel]
            rows_sel = rows[sel]
            offs = p_sel @ n_mean
            depth_sel = scaled_depth[wall_cand][ok][sel]

            def _fit_wall_at(d_off, med_depth):
                """Fit/append one plane at offset ``d_off`` (shared tail of the
                classic single-plane path and each distance mode). Returns True
                when a wall was appended."""
                nonlocal wall_inlier_total
                tol = max(0.15, 0.02 * med_depth)
                inl = np.abs(offs - d_off) < tol
                if int(inl.sum()) < inlier_floor:
                    return False
                p_in = p_sel[inl]
                cols_in = cols_sel[inl]
                rows_in = rows_sel[inl]
                u_ax, v_ax, _ = _wall_axes(np, n_mean)
                a = p_in @ u_ax
                b_y = np.clip(p_in[:, 1], -0.1, None)
                p_lo, p_hi = cfg.extent_percentiles
                a0, a1 = np.percentile(a, [p_lo, p_hi])
                b0, b1 = np.percentile(b_y, [p_lo, p_hi])
                b0 = max(b0, -0.1)
                w_m, h_m = float(a1 - a0), float(b1 - b0)
                if w_m < cfg.min_wall_size_m or h_m < cfg.min_wall_size_m:
                    return False
                if w_m < cfg.wall_min_width_m:
                    return False  # narrow vertical fit — an object, not a wall
                # Anchor pool: a much wider offset band than the fit slab.
                # A banana-biased fit centers away from the true footprint,
                # and its ±2% inlier slab then EXCLUDES the very base pixels
                # the ground anchor needs (found by instrumenting the gates:
                # every pooled ray overshot and the poison-gate refused).
                pool = np.abs(offs - d_off) < max(6.0 * tol, 0.6)
                wall = {
                    "n": n_mean, "d": d_off, "u": u_ax,
                    "a_mid": 0.5 * (a0 + a1), "b_mid": 0.5 * (b0 + b1),
                    "w": w_m, "h": h_m, "inliers": int(inl.sum()),
                    "med_depth": med_depth, "cols": cols_in,
                    "rows": rows_in, "pts": p_in, "anchored": False,
                    "base_cols": cols_sel[pool], "base_rows": rows_sel[pool],
                    "base_pts": p_sel[pool],
                }
                if cfg.ground_anchor:
                    wall = _anchor_wall_to_ground(
                        np, cfg, wall, cam_pos, horizon_y, height)
                walls.append(wall)
                # Mark inlier pixels so the object stage skips them.
                plane_dist = np.abs(pts_world @ n_mean - d_off)
                wall_inlier_total |= wall_cand & (plane_dist < tol)
                return True

            modes = max(1, int(getattr(cfg, "wall_distance_modes", 1)))
            if modes == 1:
                # Classic behavior, bit-identical: ONE plane per azimuth peak
                # at the median offset of everything facing this way.
                d_off = float(np.median(offs))
                med_depth = float(np.median(depth_sel)) if sel.any() else 5.0
                _fit_wall_at(d_off, med_depth)
            else:
                # Skyline mode: same-facing facades at different depths are
                # separate walls. Histogram the plane offsets and fit one
                # plane per mode (NMS peak-picking, mirroring the azimuth
                # stage), biggest-mass modes first so dominant facades win
                # the max_walls budget.
                lo_o, hi_o = np.percentile(offs, [1.0, 99.0])
                if hi_o - lo_o < 1e-6:
                    _fit_wall_at(float(np.median(offs)),
                                 float(np.median(depth_sel)) if sel.any() else 5.0)
                else:
                    nb = 64
                    o_hist, o_edges = np.histogram(offs, bins=nb, range=(lo_o, hi_o))
                    o_sm = o_hist.astype(np.float64).copy()
                    o_sm[1:-1] = (o_hist[:-2] + 2 * o_hist[1:-1] + o_hist[2:]) / 4.0
                    order_o = np.argsort(o_sm)[::-1]
                    supp_o = np.zeros(nb, dtype=bool)
                    centers: list[float] = []
                    for ob in order_o:
                        if supp_o[ob] or o_hist[ob] <= 0:
                            continue
                        centers.append(0.5 * (o_edges[ob] + o_edges[ob + 1]))
                        for off_i in range(ob - 2, ob + 3):
                            if 0 <= off_i < nb:
                                supp_o[off_i] = True
                        if len(centers) >= modes:
                            break
                    bin_w = (hi_o - lo_o) / nb
                    for c in centers:
                        if len(walls) >= max_walls:
                            break
                        gather = np.abs(offs - c) < max(2.0 * bin_w, 0.15)
                        if int(gather.sum()) < max(64, inlier_floor // 8):
                            continue
                        d_off = float(np.median(offs[gather]))
                        med_depth = float(np.median(depth_sel[gather]))
                        _fit_wall_at(d_off, med_depth)

        # Merge near-duplicates (azimuth < 15° apart AND offsets agree).
        # In distance-modes mode the dedupe distance tightens to the plane
        # fit's own inlier tolerance — anything closer genuinely IS the same
        # plane, anything farther is a deliberate depth mode. The classic
        # 0.3m/5% threshold would re-merge the very building rows the mode
        # split just separated on depth-compressed scenes (mono-depth crams
        # a whole skyline's far field into a metre or two).
        modes_active = max(1, int(getattr(cfg, "wall_distance_modes", 1))) > 1
        merged: list[dict[str, Any]] = []
        for w in sorted(walls, key=lambda d: -d["inliers"]):
            dup = False
            for m in merged:
                cosang = float(np.dot(w["n"], m["n"]))
                dedupe_d = (max(0.15, 0.02 * m["med_depth"]) if modes_active
                            else max(0.3, 0.05 * m["med_depth"]))
                if cosang > math.cos(math.radians(15.0)) and \
                        abs(w["d"] - m["d"]) < dedupe_d:
                    dup = True
                    break
            if not dup:
                merged.append(w)
        walls = merged[:max_walls]

    return walls, wall_inlier_total


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
    exclude_mask: Any | None = None,
) -> tuple[list[AtlasProxyPrimitive], dict[str, Any]]:
    """Derive proxy geometry from a forward-z depth map and the recovered camera.

    ``exclude_mask`` ((H,W) bool, aligned to ``depth``) removes pixels from the
    WALL and OBJECT stages only — the ground fit / metric scale / backdrop
    always use the full depth map, so several mask-scoped derive branches (one
    SAM segment per building, merged afterwards) land in the SAME metric world
    instead of each branch fitting a different ground scale.

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

def _build_ground_primitive(np: Any, cfg: ProxyDerivationConfig, pts_world: Any, ground_inlier: Any, scale: float) -> AtlasProxyPrimitive | None:
    """Helper to build projection_ground primitive from ground inliers."""
    if int(ground_inlier.sum()) < 300:
        return None
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
    return AtlasProxyPrimitive(
        name="projection_ground",
        primitive_type="plane",
        transform_matrix=_plane_transform(u_g, v_g, n_g, (cx_w, 0.0, cz_w)),
        dimensions=(float(ex), float(ez), 0.0),
        material="atlas_projection_proxy",
        metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                  "inliers": int(ground_inlier.sum()),
                  "depth_scale_applied": scale},
    )


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
    exclude_mask: Any | None = None,
) -> tuple[list[AtlasProxyPrimitive], dict[str, Any]]:
    """Derive proxy geometry from a forward-z depth map and the recovered camera.

    ``exclude_mask`` ((H,W) bool, aligned to ``depth``) removes pixels from the
    WALL and OBJECT stages only — the ground fit / metric scale / backdrop
    always use the full depth map, so several mask-scoped derive branches (one
    SAM segment per building, merged afterwards) land in the SAME metric world
    instead of each branch fitting a different ground scale.

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

    from atlas_camera.core.depth_geometry import (
        back_project_normals,
        fit_ground_and_scale,
        build_backdrop_primitive,
    )

    # -- Steps 1 & 2: back-project to world & compute normals (via depth_geometry)
    bp = back_project_normals(
        depth, view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
        depth_edge_rel=cfg.depth_edge_rel,
    )
    pts_world = bp.pts_world
    normals = bp.normals
    valid_normal = bp.valid_normal
    valid_depth = bp.valid_depth
    cam_pos = bp.cam_pos

    # -- Step 3: ground fit + metric-scale reconciliation (via depth_geometry)
    gf = fit_ground_and_scale(
        bp, horizon_y=horizon_y, ground_normal_min=cfg.ground_normal_min,
    )
    scale = gf.scale
    pts_world = gf.pts_world_scaled
    ground_inlier = gf.ground_inlier
    stats["ground_scale"] = scale
    stats["ground_inliers"] = gf.inliers

    scaled_depth = depth * scale
    backdrop_d_raw = float(np.percentile(
        scaled_depth[valid_depth], cfg.backdrop_depth_percentile
    )) if valid_depth.any() else 60.0

    # -- Step 4: ground primitive ----------------------------------------------
    ground_prim = _build_ground_primitive(np, cfg, pts_world, ground_inlier, scale)
    if ground_prim is not None:
        prims.append(ground_prim)

    # -- Step 5: wall primitives ------------------------------------------------
    wall_valid = valid_normal
    if exclude_mask is not None:
        excl = np.asarray(exclude_mask, dtype=bool)
        if excl.shape != (height, width):
            raise ValueError(
                f"exclude_mask shape {excl.shape} != depth shape {(height, width)}")
        wall_valid = valid_normal & ~excl
    walls, wall_inlier_total = _cluster_walls_by_azimuth(
        np, cfg, pts_world=pts_world, normals=normals, valid_normal=wall_valid,
        ground_inlier=ground_inlier, scaled_depth=scaled_depth,
        backdrop_d_raw=backdrop_d_raw, cam_pos=cam_pos, max_walls=max_walls,
        height=height, width=width, horizon_y=horizon_y,
    )

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
                      "depth_scale_applied": scale,
                      "ground_anchored": bool(w.get("anchored", False)),
                      "anchor_weight": w.get("anchor_weight")},
        ))
    stats["walls"] = len(walls)

    # -- Steps 6+7: foreground objects (boxes / cylinders) ----------------------
    if cfg.max_objects > 0:
        world_y = pts_world[..., 1]
        inner = np.zeros((height, width), dtype=bool)
        inner[1:-1, 1:-1] = True
        obj_cand = (valid_depth & inner & ~ground_inlier & ~wall_inlier_total
                    & (scaled_depth < backdrop_d_raw * 0.9)
                    & (world_y > 0.05))
        if exclude_mask is not None:
            obj_cand = obj_cand & ~np.asarray(exclude_mask, dtype=bool)
        prims.extend(_derive_objects(
            np, cfg, stats, pts_world, normals, obj_cand, valid_normal, scale))

    # -- Step 8: backdrop (always) ----------------------------------------------
    backdrop = build_backdrop_primitive(
        bp=bp, scaled_depth=scaled_depth, valid_depth=valid_depth,
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
        scale=scale, backdrop_depth_percentile=cfg.backdrop_depth_percentile,
        backdrop_margin=cfg.backdrop_margin,
    )
    prims.append(backdrop)

    stats["primitives"] = len(prims)
    return prims, stats



# ---------------------------------------------------------------------------
# Vertical-billboard extrusion (Photo-Pop-up-style silhouette height)
# ---------------------------------------------------------------------------

def derive_vertical_extrusion_proxies(
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
    exclude_mask: Any | None = None,
) -> tuple[list[AtlasProxyPrimitive], dict[str, Any]]:
    """Vertical-billboard wall fitter — Photo-Pop-up-style silhouette extrusion.

    ``exclude_mask`` semantics match ``derive_projection_proxies``: walls and
    objects only; ground fit / scale / backdrop stay full-frame so masked
    branches share one metric world.

    Reuses ``azimuth_walls``'s wall ORIENTATION/DISTANCE clustering (that part
    isn't broken) via the shared ``_cluster_walls_by_azimuth`` helper, but
    replaces its HEIGHT computation. ``azimuth_walls`` sets wall height from a
    percentile clip of 3D points that individually pass the near-vertical
    ``wall_normal_max`` filter — a sloped roof, spire, or bell tower never
    qualifies (its surface genuinely isn't vertical), so the wall only ever
    reflects the lower straight section.

    Instead, per Hoiem/Efros/Hebert's "Automatic Photo Pop-up" (SIGGRAPH
    2005): classify the image as ground / vertical / sky
    (``depth_geometry.detect_sky_mask`` supplies the sky half), then for each
    wall's own image-column range, find the topmost *non-sky* pixel per
    column and take *that pixel's own* back-projected world height — not
    filtered by its local surface normal at all — as a per-column height
    sample. A robust (95th-percentile) height across those columns becomes
    the wall's extruded top, reaching the real silhouette (tower/roofline)
    instead of stopping at the plain wall below it.

    Ground/backdrop/objects are unchanged from ``derive_projection_proxies``;
    only wall height differs. Still emits ``primitive_type="plane"`` — no
    downstream consumer (viewport, Maya/USD exporters) needs to change.
    """
    from atlas_camera.core.depth_geometry import (
        back_project_normals,
        build_backdrop_primitive,
        detect_sky_mask,
        fit_ground_and_scale,
    )

    np = _require_numpy()
    cfg = config or ProxyDerivationConfig()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    if horizon_y is None:
        horizon_y = height * 0.45

    stats: dict[str, Any] = {"walls": 0, "boxes": 0, "cylinders": 0}
    prims: list[AtlasProxyPrimitive] = []

    bp = back_project_normals(depth, view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
                               depth_edge_rel=cfg.depth_edge_rel)
    gf = fit_ground_and_scale(bp, horizon_y=horizon_y, ground_normal_min=cfg.ground_normal_min)
    scale = gf.scale
    pts_world = gf.pts_world_scaled
    ground_inlier = gf.ground_inlier
    stats["ground_scale"] = scale
    stats["ground_inliers"] = gf.inliers

    scaled_depth = depth * scale
    valid_depth = bp.valid_depth
    backdrop_d_raw = float(np.percentile(
        scaled_depth[valid_depth], cfg.backdrop_depth_percentile
    )) if valid_depth.any() else 60.0

    sky_mask = detect_sky_mask(depth, horizon_y=horizon_y)
    not_sky = ~sky_mask

    # -- ground primitive (via helper) ----------------------------------------
    ground_prim = _build_ground_primitive(np, cfg, pts_world, ground_inlier, scale)
    if ground_prim is not None:
        prims.append(ground_prim)


    # -- walls: shared orientation/distance clustering, silhouette height ----
    wall_valid = bp.valid_normal
    if exclude_mask is not None:
        excl = np.asarray(exclude_mask, dtype=bool)
        if excl.shape != (height, width):
            raise ValueError(
                f"exclude_mask shape {excl.shape} != depth shape {(height, width)}")
        wall_valid = bp.valid_normal & ~excl
    walls, wall_inlier_total = _cluster_walls_by_azimuth(
        np, cfg, pts_world=pts_world, normals=bp.normals, valid_normal=wall_valid,
        ground_inlier=ground_inlier, scaled_depth=scaled_depth,
        backdrop_d_raw=backdrop_d_raw, cam_pos=bp.cam_pos, max_walls=max_walls,
        height=height, width=width, horizon_y=horizon_y,
    )

    wall_no = 0
    for w in walls:
        base_y = max(w["b_mid"] - 0.5 * w["h"], -0.1)
        u_cols = np.unique(w["cols"])
        col_tops: dict[int, float] = {}
        for col in u_cols:
            rows = np.where(not_sky[:, col] & valid_depth[:, col])[0]
            if rows.size:
                col_tops[int(col)] = float(pts_world[int(rows.min()), col, 1])

        # Column segments: one per roofline when enabled (split at silhouette
        # height steps), else the whole cluster as a single segment — a row of
        # buildings then stops sharing one rectangle that spans sky above its
        # shorter members.
        segments: list[Any] = []
        if cfg.roofline_split and len(col_tops) >= 8:
            sc = np.array(sorted(col_tops))
            tops = np.array([col_tops[int(c)] for c in sc])
            k = 5  # light moving-median so window AC units don't split walls
            sm = np.array([float(np.median(tops[max(0, j - k // 2):j + k // 2 + 1]))
                           for j in range(len(tops))])
            med_h = max(float(np.median(sm - base_y)), cfg.min_wall_size_m)
            step = np.abs(np.diff(sm))
            cuts = np.where(step > max(cfg.roofline_split_rel * med_h, 1.0))[0]
            lo = 0
            for cut in list(cuts) + [len(sc) - 1]:
                segments.append(sc[lo:cut + 1])
                lo = cut + 1
            segments = [seg for seg in segments if seg.size >= 4]
        if not segments:
            segments = [u_cols]

        for seg_cols in segments:
            seg_set = np.isin(w["cols"], seg_cols)
            if not seg_set.any():
                continue
            p_seg = w["pts"][seg_set]
            u_ax, v_ax, n_ax = _wall_axes(np, w["n"])
            seg_w = w.copy()
            if len(segments) > 1:
                # Per-segment extents (and re-anchor: each building's own
                # base contact line gives its own footprint distance).
                a = p_seg @ u_ax
                p_lo, p_hi = cfg.extent_percentiles
                a0, a1 = np.percentile(a, [p_lo, p_hi])
                if float(a1 - a0) < cfg.wall_min_width_m:
                    continue
                base_set = np.isin(w.get("base_cols", w["cols"]), seg_cols)
                seg_w.update({"a_mid": 0.5 * float(a0 + a1),
                              "w": float(a1 - a0),
                              "cols": w["cols"][seg_set],
                              "rows": w["rows"][seg_set],
                              "pts": p_seg,
                              "base_cols": w.get("base_cols", w["cols"])[base_set],
                              "base_rows": w.get("base_rows", w["rows"])[base_set],
                              "base_pts": w.get("base_pts", w["pts"])[base_set],
                              "inliers": int(seg_set.sum())})
                if cfg.ground_anchor:
                    seg_w = _anchor_wall_to_ground(
                        np, cfg, seg_w, bp.cam_pos, horizon_y, height)
            # Anchored buildings sit ON the ground (near-base pixels are
            # routinely eaten by ground-inlier/depth-edge filtering, leaving
            # unanchored walls floating — keep their measured base).
            seg_base_y = 0.0 if seg_w.get("anchored") else base_y
            if seg_w.get("anchored"):
                # Banana-immune heights: the silhouette top PIXEL is reliable
                # but its own back-projected depth is not (a warped far depth
                # inflated rooftops to 100 m+ on a real street photo, in both
                # variants). Intersect each top pixel's ray with the ANCHORED
                # wall plane instead — like the footprint, height becomes
                # pure geometry.
                n_pl = np.asarray(seg_w["n"], dtype=np.float64)
                d_pl = float(seg_w["d"])
                seg_tops = []
                for cc in seg_cols:
                    ci = int(cc)
                    if ci not in col_tops:
                        continue
                    rr = np.where(not_sky[:, ci] & valid_depth[:, ci])[0]
                    if not rr.size:
                        continue
                    p_top = pts_world[int(rr.min()), ci]
                    ray = p_top - bp.cam_pos
                    rn = float(ray @ n_pl)
                    if abs(rn) < 1e-9:
                        continue
                    t_pl = (d_pl - float(bp.cam_pos @ n_pl)) / rn
                    if t_pl <= 0:
                        continue
                    seg_tops.append(float(bp.cam_pos[1] + t_pl * ray[1]))
            else:
                seg_tops = [col_tops[int(c)] for c in seg_cols if int(c) in col_tops]
            if seg_tops:
                top_y = float(np.percentile(seg_tops, 95.0))
                h_m = max(top_y - seg_base_y, cfg.min_wall_size_m)
            else:
                h_m = w["h"]  # no usable column — clusterer's own height
            b_mid = seg_base_y + 0.5 * h_m
            u_ax, v_ax, n_ax = _wall_axes(np, seg_w["n"])
            c = seg_w["d"] * n_ax + seg_w["a_mid"] * u_ax + b_mid * v_ax
            wall_no += 1
            prims.append(AtlasProxyPrimitive(
                name=f"projection_wall_{wall_no:02d}",
                primitive_type="plane",
                transform_matrix=_plane_transform(u_ax, v_ax, n_ax, c),
                dimensions=(seg_w["w"], h_m, 0.0),
                material="atlas_projection_proxy",
                metadata={"role": PROXY_ROLE, "source": "depth_derivation",
                          "inliers": seg_w["inliers"],
                          "yaw_deg": float(math.degrees(math.atan2(seg_w["n"][0], seg_w["n"][2]))),
                          "distance_m": float(seg_w["d"]),
                          "depth_scale_applied": scale,
                          "ground_anchored": bool(seg_w.get("anchored", False)),
                          "anchor_weight": seg_w.get("anchor_weight"),
                          "roofline_segment": len(segments) > 1,
                          "method": "vertical_extrusion"},
            ))
    stats["walls"] = wall_no

    # -- objects (identical to derive_projection_proxies) --------------------
    if cfg.max_objects > 0:
        inner_mask = np.zeros((height, width), dtype=bool)
        inner_mask[1:-1, 1:-1] = True
        obj_cand = (valid_depth & inner_mask & ~ground_inlier & ~wall_inlier_total
                    & (scaled_depth < backdrop_d_raw * 0.9)
                    & (pts_world[..., 1] > 0.05))
        if exclude_mask is not None:
            obj_cand = obj_cand & ~np.asarray(exclude_mask, dtype=bool)
        prims.extend(_derive_objects(
            np, cfg, stats, pts_world, bp.normals, obj_cand, bp.valid_normal, scale))

    # -- backdrop (always), via the shared depth_geometry.py helper ----------
    prims.append(build_backdrop_primitive(
        bp=bp, scaled_depth=scaled_depth, valid_depth=valid_depth,
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, scale=scale,
        backdrop_depth_percentile=cfg.backdrop_depth_percentile,
        backdrop_margin=cfg.backdrop_margin,
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

    # np.bincount on raveled indices, not np.add.at — np.add.at is numpy's
    # known-slow unbuffered scatter-add path; bincount is the vectorized,
    # much faster equivalent for a plain "count occurrences per bin" histogram.
    flat_idx = gx * gh + gz
    counts = np.bincount(flat_idx, minlength=gw * gh).reshape(gw, gh)
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
        if h_obj > cfg.object_max_height_m:
            # XZ-only occupancy clustering has no notion of height, so a
            # foreground object's footprint can spuriously merge with an
            # unrelated tall structure behind/above it (a distant wall or
            # machinery that wasn't caught by wall-fitting) that happens to
            # share the same X/Z cell — the 98th-percentile height then
            # reflects that tall structure, not the actual foreground
            # object, producing an implausible box (confirmed empirically:
            # a real photo produced a ~14m "foreground" box because points
            # from a background structure leaked into the cluster). Refit
            # using only the points below the plausibility ceiling — this
            # is the same "reject and refit/drop" pattern _fit_cylinder
            # already uses for cylinder_radius_range, just applied to height
            # instead of radius.
            low = oy <= cfg.object_max_height_m
            if int(low.sum()) < cfg.min_object_inliers:
                continue
            sel_idx = np.flatnonzero(sel)[low]
            sel = np.zeros_like(sel)
            sel[sel_idx] = True
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
    new_meta["vertices"] = np.round(verts2.reshape(-1), 3).tolist()
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
            # Per-vertex linear coverage risk for viewport-only soft tear edges.
            # Exporters never consume it; positions/faces/UVs remain unchanged.
            entry["edge_risk"] = md.get("edge_risk", [])
        out.append(entry)
    return out


def relief_mesh_primitive(mesh: Any, *, name: str = "projection_relief_mesh") -> AtlasProxyPrimitive:
    """Wrap a :class:`~atlas_camera.core.relief_mesh.ReliefMesh` as a proxy
    primitive so it rides the solve into the blockout viewport payload.

    Arrays are flattened and rounded (mm / 1e-4 UV) to keep the JSON compact.
    Rounding/flattening is vectorized (np.round + .tolist()) rather than a
    Python-level round()/float() comprehension per scalar — at
    relief_quality="ultra" (grid=1024, up to ~780K vertices) the per-scalar
    Python loop cost seconds of pure interpreter overhead on every execution.
    """
    np = _require_numpy()
    verts = np.round(mesh.vertices.reshape(-1).astype(np.float64), 3).tolist()
    faces = mesh.faces.reshape(-1).astype(np.int64).tolist()
    uvs = np.round(mesh.uvs.reshape(-1).astype(np.float64), 4).tolist()
    edge_risk_array = getattr(mesh, "edge_risk", None)
    edge_risk = (np.round(np.asarray(edge_risk_array).reshape(-1), 3).tolist()
                 if edge_risk_array is not None else [])
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
            "torn_fraction": float((mesh.stats or {}).get("torn_fraction", 0.0)),
            "quad_coherence": bool((mesh.stats or {}).get("quad_coherence", False)),
            "stretch_ratio_p95": float((mesh.stats or {}).get("stretch_ratio_p95", 0.0)),
            "stretch_fraction_gt12": float((mesh.stats or {}).get("stretch_fraction_gt12", 0.0)),
            "vertices": verts,
            "faces": faces,
            "uvs": uvs,
            "edge_risk": edge_risk,
        },
    )
