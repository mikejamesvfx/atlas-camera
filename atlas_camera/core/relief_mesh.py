"""Depth relief mesh — a triangulated, decimated depth map for DCC handoff.

Unlike primitive fitting (proxy_geometry.py), the relief mesh never fails: the
depth map is sampled on a grid, back-projected into the Atlas Y-up world, and
triangulated — with triangles torn at depth discontinuities so foreground
silhouettes don't rubber-sheet onto the background. Each vertex's UV is its own
image pixel position, i.e. **the camera projection is baked into the UVs**: the
exported OBJ + source image texture is already correctly "projected" in Maya /
Nuke / ZBrush with no shader setup, ready to retopologize and reproject.

Convention: same as proxy_geometry — the full 4×4 ``camera_view_matrix``
(row-major, world→cam), ``cam_to_world = inv(view_matrix)``. Numpy-only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Relief mesh construction requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class ReliefMesh:
    """Triangulated depth mesh in Atlas world space (Y-up, metres).

    ``vertices`` (N,3) float32; ``faces`` (M,3) int32, counter-clockwise when
    seen from the recovered camera; ``uvs`` (N,2) float32 with OBJ convention
    (origin bottom-left) — each vertex maps to its own source-image pixel.

    ``hole_mask`` (H,W) bool, always at full source-image resolution
    regardless of ``grid_long_edge`` decimation — ``True`` where no triangle
    covers that pixel (sky/invalid/band-excluded, or the pixel's grid quad
    was torn by the ``depth_edge_rel``/``max_edge_factor`` silhouette test).
    This is the literal answer to "where will 📽 Project show black" — the
    mesh's own hole data, otherwise discarded once triangulation finishes.

    ``filled_mask`` (H,W) bool or None — pixels whose depth was SYNTHESIZED
    by the ``fill_mask`` diffusion fill (see ``build_relief_mesh``) rather
    than measured. These pixels carry real geometry (they are NOT in
    ``hole_mask``), but the depth there is a smooth interpolation of the
    surrounding background, not data — surfaced so artists can QA what was
    invented. None when no fill was requested.
    """

    vertices: Any
    faces: Any
    uvs: Any
    stats: dict[str, Any] = field(default_factory=dict)
    hole_mask: Any = None
    filled_mask: Any = None


def estimate_ground_scale(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    horizon_y: float | None = None,
    stride: int = 4,
) -> tuple[float, dict[str, Any]]:
    """Uniform scale about the camera that lands the depth map's ground on Y=0.

    Compact version of the ground fit in proxy_geometry: back-project (strided),
    per-pixel normals, keep near-horizontal below-horizon surfaces, histogram-mode
    plane height y0, then ``scale = cam_y / (cam_y - y0)``. Returns (scale, info);
    scale is 1.0 when no reliable ground is found.
    """
    np = _require_numpy()
    depth = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    height, width = depth.shape
    if horizon_y is None:
        horizon_y = height * 0.45
    else:
        horizon_y = horizon_y / stride

    vm = np.asarray(view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    R_cw = c2w[:3, :3]
    cam = c2w[:3, 3]
    if cam[1] <= 1e-6:
        return 1.0, {"reason": "camera at/below ground height"}

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64) * stride,
                         np.arange(height, dtype=np.float64) * stride)
    x = (uu - cx) / fx * depth
    y = -(vv - cy) / fy * depth
    z = -depth
    pts = np.stack([x, y, z], axis=-1) @ R_cw.T + cam

    du = pts[:, 2:, :] - pts[:, :-2, :]
    dv = pts[2:, :, :] - pts[:-2, :, :]
    n = np.cross(du[1:-1], dv[:, 1:-1])
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)

    world_y = pts[1:-1, 1:-1, 1]
    below = vv[1:-1, 1:-1] > horizon_y * stride
    cand = below & (np.abs(n[..., 1]) > 0.9) & np.isfinite(world_y)
    if int(cand.sum()) < 80:
        return 1.0, {"reason": "insufficient ground candidates"}

    ys = world_y[cand]
    lo, hi = np.percentile(ys, [1, 99])
    if hi - lo < 1e-3:
        y0 = float(np.median(ys))
    else:
        hist, edges = np.histogram(ys, bins=48, range=(lo, hi))
        peak = int(np.argmax(hist))
        y0 = 0.5 * (edges[peak] + edges[peak + 1])
        tol = max(0.15, 0.03 * float(hi - lo))
        refine = np.abs(ys - y0) < tol
        if int(refine.sum()) >= 40:
            y0 = float(np.median(ys[refine]))

    denom = cam[1] - y0
    if denom <= 1e-6:
        return 1.0, {"reason": "degenerate ground offset"}
    scale = float(cam[1] / denom)
    return scale, {"plane_y": y0, "inliers": int(cand.sum())}


def build_relief_mesh(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    grid_long_edge: int = 96,
    depth_edge_rel: float = 0.5,
    far_clip_percentile: float = 97.0,
    scale: float = 1.0,
    floor_clamp: float | None = -0.25,
    smooth_iterations: int = 2,
    max_edge_factor: float = 12.0,
    horizon_y: float | None = None,
    band_min_m: float | None = None,
    band_max_m: float | None = None,
    exclude_mask: Any = None,
    apply_sky_heuristic: bool = True,
    fill_mask: Any = None,
    edge_overhang_cells: int = 0,
) -> ReliefMesh:
    """Triangulate a forward-z depth map into a world-space relief mesh.

    ``grid_long_edge`` sets sampling density (grid columns along the longest
    image edge). Triangles whose corner depths differ by more than
    ``depth_edge_rel`` (relative), or whose world edges stretch beyond
    ``max_edge_factor`` × the expected local sample spacing, are torn —
    silhouettes become holes instead of stretched shards. Depth is sampled with
    a 3×3 median (kills single-pixel spikes) and smoothed edge-aware for
    ``smooth_iterations`` rounds. Depths above ``far_clip_percentile`` are
    clamped so the sky becomes a smooth distant shell.

    ``horizon_y`` (image row, defaults to ``height * 0.45`` matching
    ``depth_geometry.fit_ground_and_scale``'s own fallback) feeds
    ``depth_geometry.detect_sky_mask``: pixels flagged as noisy/incoherent sky
    are excluded from triangulation entirely (a hole, like any other depth
    discontinuity) rather than merely distance-clamped — without this, noisy
    monocular depth in feature-less sky/cloud regions gets torn into a jagged,
    spatially-huge "mountain" that dwarfs the actual photographed geometry.

    ``band_min_m``/``band_max_m`` (metres, default ``None`` = no clip) exclude
    pixels whose *metric* depth (``depth * scale``) falls outside the band from
    triangulation entirely — the same "exclude the pixel, don't clamp it" hole
    mechanism sky/silhouette exclusion already uses. This is how a single depth
    map is split into independent per-layer meshes for the inpaint-layers
    feature (see ``AtlasCleanPlateLayer``): each layer's mesh only contains its
    own band, so overlapping layers never fight over the same texels.

    ``exclude_mask`` (H,W) bool, default ``None``, ORs an externally-supplied
    exclusion (e.g. a real sky segmentation from SAM/RMBG, run once upstream)
    on top of the internal ``detect_sky_mask`` heuristic — never replaces it,
    only ever excludes more. Caller's responsibility to resize it to the
    depth map's own resolution first; this function does no resampling.

    ``apply_sky_heuristic`` (default ``True``) runs the internal
    ``detect_sky_mask`` roughness/far-percentile heuristic as usual. Set
    ``False`` only when the caller's own depth array IS deliberately
    sky-shaped (see ``build_sky_dome_mesh``, which feeds this function a
    synthetic constant-depth field for exactly the pixels it wants to KEEP —
    the heuristic would otherwise re-exclude that same region, since a
    constant-depth field is precisely what it's designed to flag).

    ``edge_overhang_cells`` (default 0 = off) extends every mesh boundary
    OUTWARD into invalid regions by N grid cells, copying the nearest valid
    depth. Only meaningful together with a per-pixel edge matte
    (``ProjectionSource.mask_b64``): boundary quads straddling an exclusion
    edge (sky mask, band clip) are otherwise torn at grid-quad resolution,
    leaving a stepped uncovered strip along the true silhouette that no
    matte can fix — the matte can only CUT geometry that exists. With the
    overhang, geometry slightly overshoots the boundary and the
    full-resolution matte cuts the exact edge. Never enable without a matte:
    the overhang skirt would show visibly stretched edge texels. Overhung
    cells count as covered in ``hole_mask`` (geometry exists there).

    ``fill_mask`` (H,W) bool, default ``None`` — pixels whose depth should be
    SYNTHESIZED by diffusion from the surrounding valid grid instead of
    trusted or torn into a hole. The disocclusion move: a clean-plate layer's
    band clip leaves a hole exactly where a foreground occluder stood; the
    inpainted plate has pixels there but no geometry. Filling interpolates
    the surrounding background depth across the occluder's footprint (Jacobi
    diffusion on the decimated grid — cheap, never full-res) so the plate's
    inpainted content lands on real geometry. Only currently-invalid cells
    are filled (a fill-flagged cell that got a legitimate in-band value from
    the 3x3 median keeps it); cells with no valid neighbors anywhere stay
    holes. Filled pixels are reported in ``ReliefMesh.filled_mask`` and
    removed from ``hole_mask``.
    """
    from atlas_camera.core.depth_geometry import detect_sky_mask

    np = _require_numpy()
    depth = np.asarray(depth, dtype=np.float64).copy()
    height, width = depth.shape

    if horizon_y is None:
        horizon_y = height * 0.45

    valid_full = np.isfinite(depth) & (depth > 1e-4)
    if apply_sky_heuristic:
        valid_full &= ~detect_sky_mask(depth, horizon_y=horizon_y)
    if exclude_mask is not None:
        valid_full &= ~np.asarray(exclude_mask, dtype=bool)
    if band_min_m is not None or band_max_m is not None:
        metric = depth * float(scale)
        if band_min_m is not None:
            valid_full &= metric >= band_min_m
        if band_max_m is not None:
            valid_full &= metric <= band_max_m
    if far_clip_percentile and valid_full.any():
        far = float(np.percentile(depth[valid_full], far_clip_percentile))
        np.minimum(depth, far, out=depth)

    # Seed the full-resolution hole mask from the per-pixel exclusion test
    # (sky/invalid/band) before grid decimation smooths any of it over; the
    # per-quad tear boundary (below, once ok_a/ok_b are known) is OR'd in
    # afterward.
    hole_mask = ~valid_full

    step = max(1, int(round(max(height, width) / max(grid_long_edge, 2))))
    rows = np.arange(0, height, step)
    cols = np.arange(0, width, step)
    if rows[-1] != height - 1:
        rows = np.append(rows, height - 1)
    if cols[-1] != width - 1:
        cols = np.append(cols, width - 1)
    nr, nc = len(rows), len(cols)

    # 3×3 median sampling: single-pixel depth spikes (common in AI-image depth)
    # otherwise become mesh spikes.
    depth_nan = np.where(valid_full, depth, np.nan)
    samples = []
    for dr in (-1, 0, 1):
        rr = np.clip(rows + dr, 0, height - 1)
        for dc in (-1, 0, 1):
            cc = np.clip(cols + dc, 0, width - 1)
            samples.append(depth_nan[np.ix_(rr, cc)])
    # All-NaN grid cells are expected whenever a cell's full 3x3 neighbourhood
    # falls entirely in sky/invalid regions — handled below via vgrid, not an
    # error. np.errstate only silences FPU-flag warnings; nanmedian's "All-NaN
    # slice" warning goes through the separate `warnings` module, so it needs
    # its own suppression.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmedian(np.stack(samples), axis=0)
    vgrid = np.isfinite(d) & (d > 1e-4)
    d = np.where(vgrid, d, 0.0)

    # Diffusion fill of the fill_mask region (disocclusion): synthesize depth
    # for currently-invalid fill-flagged grid cells by iteratively averaging
    # known 4-neighbors (front propagation + Jacobi relaxation in one loop).
    # Runs on the decimated grid, so cost is bounded by grid_long_edge, not
    # image resolution. Cells that never reach a known neighbor stay invalid.
    filled_grid = None
    if fill_mask is not None:
        fill_target = np.asarray(fill_mask, dtype=bool)[np.ix_(rows, cols)] & ~vgrid
        if fill_target.any():
            known = vgrid.copy()
            for _ in range(200):
                acc = np.zeros_like(d)
                cnt = np.zeros_like(d)
                for shift_r, shift_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = np.roll(d, (shift_r, shift_c), axis=(0, 1))
                    nb_k = np.roll(known, (shift_r, shift_c), axis=(0, 1))
                    border = np.zeros_like(known)
                    if shift_r == 1:
                        border[0, :] = True
                    elif shift_r == -1:
                        border[-1, :] = True
                    if shift_c == 1:
                        border[:, 0] = True
                    elif shift_c == -1:
                        border[:, -1] = True
                    ok = nb_k & ~border
                    acc += np.where(ok, nb, 0.0)
                    cnt += ok
                upd = fill_target & (cnt > 0)
                if not upd.any():
                    break
                prev = d[upd].copy()
                d[upd] = (acc / np.maximum(cnt, 1))[upd]
                newly = upd & ~known
                known |= upd
                if not newly.any():
                    delta = np.abs(d[upd] - prev) / np.maximum(prev, 1e-6)
                    if float(delta.max()) < 1e-4:
                        break
            filled_grid = fill_target & known
            vgrid |= filled_grid

    # Boundary overhang (see docstring): extend valid grid cells outward into
    # invalid space by copying neighbor depths — flat skirts the edge matte
    # cuts back to the true silhouette per-pixel. Runs after the diffusion
    # fill (fills are legitimate interior geometry, not boundary) and before
    # smoothing (so the skirt relaxes with everything else).
    if edge_overhang_cells and int(edge_overhang_cells) > 0:
        for _ in range(int(edge_overhang_cells)):
            acc = np.zeros_like(d)
            cnt = np.zeros_like(d)
            for shift_r, shift_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = np.roll(d, (shift_r, shift_c), axis=(0, 1))
                nb_v = np.roll(vgrid, (shift_r, shift_c), axis=(0, 1))
                border = np.zeros_like(vgrid)
                if shift_r == 1:
                    border[0, :] = True
                elif shift_r == -1:
                    border[-1, :] = True
                if shift_c == 1:
                    border[:, 0] = True
                elif shift_c == -1:
                    border[:, -1] = True
                ok = nb_v & ~border
                acc += np.where(ok, nb, 0.0)
                cnt += ok
            newly = ~vgrid & (cnt > 0)
            if not newly.any():
                break
            d[newly] = (acc / np.maximum(cnt, 1))[newly]
            vgrid |= newly

    # Edge-aware smoothing of the sampled grid: average with neighbours that are
    # depth-consistent (within the tear threshold) — flattens shards without
    # blurring across silhouettes.
    for _ in range(max(0, int(smooth_iterations))):
        acc = np.zeros_like(d)
        cnt = np.zeros_like(d)
        for shift_r, shift_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(d, (shift_r, shift_c), axis=(0, 1))
            nb_v = np.roll(vgrid, (shift_r, shift_c), axis=(0, 1))
            # np.roll wraps — invalidate the wrapped border rows/cols.
            border = np.zeros_like(vgrid)
            if shift_r == 1:
                border[0, :] = True
            elif shift_r == -1:
                border[-1, :] = True
            if shift_c == 1:
                border[:, 0] = True
            elif shift_c == -1:
                border[:, -1] = True
            ok = nb_v & ~border & vgrid & (
                np.abs(nb / np.maximum(d, 1e-6) - 1.0) < depth_edge_rel)
            acc += np.where(ok, nb, 0.0)
            cnt += ok
        has = cnt > 0
        d = np.where(has, 0.5 * d + 0.5 * acc / np.maximum(cnt, 1), d)

    # Back-project the grid into the world (camera pose from the view matrix),
    # then rescale about the camera (ground reconciliation).
    vm = np.asarray(view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    R_cw = c2w[:3, :3]
    cam = c2w[:3, 3]
    uu = cols[None, :].astype(np.float64)
    vv = rows[:, None].astype(np.float64)
    x = (uu - cx) / fx * d
    y = -(vv - cy) / fy * d
    z = -d
    pts = np.stack([np.broadcast_to(x, d.shape),
                    np.broadcast_to(y, d.shape), z], axis=-1) @ R_cw.T + cam
    pts = cam + float(scale) * (pts - cam)

    # Noisy depth (AI-image silhouettes, reflective floors) can punch vertices
    # below the ground plane. Clamp offenders back ALONG THEIR VIEW RAY — which
    # preserves the baked camera projection exactly (texels are assigned by ray).
    if floor_clamp is not None and cam[1] > floor_clamp:
        py = pts[..., 1]
        low = py < floor_clamp
        if low.any():
            s_fix = (cam[1] - floor_clamp) / np.maximum(cam[1] - py[low], 1e-9)
            pts[low] = cam + s_fix[:, None] * (pts[low] - cam)

    # UVs: each vertex is its own image pixel. OBJ vt origin is bottom-left.
    u_uv = cols.astype(np.float64) / max(width - 1, 1)
    v_uv = 1.0 - rows.astype(np.float64) / max(height - 1, 1)
    UU, VV = np.meshgrid(u_uv, v_uv)
    uvs = np.stack([UU, VV], axis=-1)

    # Faces: two triangles per grid quad, CCW as seen from the camera; torn
    # where corner depths disagree by more than depth_edge_rel (relative).
    idx = np.arange(nr * nc).reshape(nr, nc)
    i00, i01 = idx[:-1, :-1], idx[:-1, 1:]
    i10, i11 = idx[1:, :-1], idx[1:, 1:]
    d00, d01 = d[:-1, :-1], d[:-1, 1:]
    d10, d11 = d[1:, :-1], d[1:, 1:]
    v00, v01 = vgrid[:-1, :-1], vgrid[:-1, 1:]
    v10, v11 = vgrid[1:, :-1], vgrid[1:, 1:]

    # World-space corner positions per quad, for the edge-length tear: a depth
    # ratio just under the threshold at 10 m is still a metres-long shard, so
    # cap triangle edges at max_edge_factor × the expected local sample spacing.
    P00, P01 = pts[:-1, :-1], pts[:-1, 1:]
    P10, P11 = pts[1:, :-1], pts[1:, 1:]
    edge_budget = (max_edge_factor * float(scale) * step / min(fx, fy))

    def _tri_ok(da, db, dc, va, vb, vc, pa, pb, pc):
        dmax = np.maximum(np.maximum(da, db), dc)
        dmin = np.minimum(np.minimum(da, db), dc)
        ok = (va & vb & vc & (dmin > 1e-4)
              & ((dmax / np.maximum(dmin, 1e-6) - 1.0) <= depth_edge_rel))
        if max_edge_factor:
            limit = dmax * edge_budget  # expected spacing ≈ depth·step/f
            for e0, e1 in ((pa, pb), (pb, pc), (pa, pc)):
                ok &= np.linalg.norm(e1 - e0, axis=-1) <= np.maximum(limit, 0.05)
        return ok

    ok_a = _tri_ok(d00, d10, d01, v00, v10, v01, P00, P10, P01)
    ok_b = _tri_ok(d10, d11, d01, v10, v11, v01, P10, P11, P01)
    tri_a = np.stack([i00[ok_a], i10[ok_a], i01[ok_a]], axis=1)
    tri_b = np.stack([i10[ok_b], i11[ok_b], i01[ok_b]], axis=1)
    faces = np.concatenate([tri_a, tri_b], axis=0)

    # Rasterize torn quads (partially or fully) back into the full-resolution
    # hole mask. rows/cols are literal pixel indices, so a per-pixel gather
    # against the per-quad "fully covered" test is a single vectorized index
    # op — no per-quad Python loop even at grid_long_edge=1024.
    quad_hole = ~(ok_a & ok_b)
    row_idx = np.clip(np.searchsorted(rows, np.arange(height), side="right") - 1, 0, nr - 2)
    col_idx = np.clip(np.searchsorted(cols, np.arange(width), side="right") - 1, 0, nc - 2)
    hole_mask |= quad_hole[row_idx[:, None], col_idx[None, :]]

    # Filled (synthesized-depth) pixels carry real geometry now: rasterize
    # kept quads touching a filled grid vertex, clear them from hole_mask,
    # and surface them separately so artists can QA what was invented.
    filled_mask_full = None
    if filled_grid is not None and filled_grid.any():
        f00, f01 = filled_grid[:-1, :-1], filled_grid[:-1, 1:]
        f10, f11 = filled_grid[1:, :-1], filled_grid[1:, 1:]
        quad_filled = (f00 | f01 | f10 | f11) & ok_a & ok_b
        filled_mask_full = quad_filled[row_idx[:, None], col_idx[None, :]]
        hole_mask &= ~filled_mask_full

    verts = pts.reshape(-1, 3)
    uvs_flat = uvs.reshape(-1, 2)

    # Compact to referenced vertices only (clean OBJ for DCC import).
    used, remap = np.unique(faces.reshape(-1), return_inverse=True)
    faces = remap.reshape(-1, 3).astype(np.int32)
    verts = verts[used].astype(np.float32)
    uvs_flat = uvs_flat[used].astype(np.float32)

    n_quads = 2 * (nr - 1) * (nc - 1)
    stats = {
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "grid": (int(nr), int(nc)),
        "scale": float(scale),
        "torn_fraction": float(1.0 - len(faces) / max(n_quads, 1)),
        "n_filled_cells": int(filled_grid.sum()) if filled_grid is not None else 0,
    }
    return ReliefMesh(vertices=verts, faces=faces, uvs=uvs_flat, stats=stats,
                      hole_mask=hole_mask, filled_mask=filled_mask_full)


def build_sky_dome_mesh(
    sky_mask: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radius_m: float = 300.0,
    grid_long_edge: int = 96,
    edge_overhang_cells: int = 2,
) -> ReliefMesh:
    """Build a same-camera 'sky dome' patch: a constant-radius shell covering
    only the pixels flagged in ``sky_mask``, UV-baked exactly like a normal
    relief mesh so a clean sky plate projects onto it correctly.

    ``edge_overhang_cells`` (default 2, since the dome ALWAYS ships with its
    segmentation embedded as a per-pixel matte) extends the card slightly
    past the mask boundary so the skyline edge is cut by the full-resolution
    matte instead of stepping at grid-quad resolution — without it, boundary
    quads straddling the silhouette tear away, leaving a stepped uncovered
    strip of background exactly along the skyline.

    The standard DMP move (Nuke and similar): separate sky from real
    (parallax-bearing) geometry so it can be clean-plated and projected onto
    simple, depth-free geometry instead of fighting noisy monocular sky depth
    or tearing at a boundary that's really just "where the segmentation mask
    ends," not a real depth discontinuity.

    Reuses ``build_relief_mesh``'s exact triangulation/tearing/UV-baking math
    by feeding it a SYNTHETIC constant-depth array — ``radius_m`` wherever
    ``sky_mask`` is set, ``0.0`` (invalid) elsewhere — with
    ``apply_sky_heuristic=False`` (the internal heuristic would otherwise
    re-exclude this exact region: a constant-depth field is precisely what
    it's designed to flag) and ``floor_clamp=None``/``scale=1.0`` (ground-
    plane clamping and metric ground-scale rescaling are both meaningless
    for a dome deliberately independent of the scene's real depth). The
    boundary of ``sky_mask`` becomes a natural torn edge via the same
    "invalid grid vertex" mechanism silhouette/sky exclusion already uses —
    no special-casing needed. ``depth_edge_rel`` is omitted entirely: with a
    perfectly constant depth field its ratio test is always satisfied
    trivially (``dmax == dmin``), so it has no effect here.
    """
    np = _require_numpy()
    mask = np.asarray(sky_mask, dtype=bool)
    synthetic_depth = np.where(mask, float(radius_m), 0.0)
    return build_relief_mesh(
        synthetic_depth, view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
        grid_long_edge=grid_long_edge, far_clip_percentile=0.0,
        scale=1.0, floor_clamp=None, horizon_y=None,
        apply_sky_heuristic=False,
        edge_overhang_cells=edge_overhang_cells,
    )
