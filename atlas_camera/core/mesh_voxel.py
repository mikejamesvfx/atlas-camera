"""Voxel solidify + watertight remesh of a relief surface (pure numpy).

Inspired by PyTopo3D's output pipeline (voxelize -> isosurface -> smooth), cut
down to what a matte-painting relief actually needs: a WATERTIGHT solid a DCC
can boolean / a printer can slice, built from a surface that is deliberately
torn and open everywhere.

Pipeline (`voxel_remesh`):
  1. depth-rasterize the mesh from the RECOVERED camera (frustum space — a
     relief mesh is a graph over the image plane, so the frustum grid is its
     natural voxelization axis, unlike PyTopo3D's world-AABB grid);
  2. close interior invalid columns of the depth grid (border-connected
     invalid regions stay open — the frame surround must not become geometry);
  3. occupancy: solid between the surface and a back-extrusion of a few
     voxel slabs (a shell with genuine thickness, not a zero-width sheet);
  4. **naive surface nets** over the padded occupancy — chosen over marching
     cubes deliberately: ~a tenth of the code (no 256-entry case tables),
     inherently watertight on a padded grid, quad-dominant, and its blocky
     dual mesh smooths beautifully;
  5. pure-numpy Taubin smoothing (shrink/inflate, no volume collapse);
  6. unproject grid coords back to world through the camera.

Export-only by doctrine: the live projection mesh keeps its load-bearing
tears; this feeds `mesh_retopo.apply_retopo(method="voxel_remesh")`, which
regenerates projective UVs afterwards. No scipy, no scikit-image, no trimesh —
the `core/` numpy-only rule holds.
"""

from __future__ import annotations

from typing import Any


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "voxel remesh requires numpy. Install with:\n"
            "    pip install -e .[vision]"
        ) from exc
    return np


def render_depth_grid(vertices: Any, faces: Any, view_matrix: Any,
                      fx: float, fy: float, cx: float, cy: float,
                      width: int, height: int) -> Any:
    """Nearest-forward depth of the mesh per raster pixel; NaN where uncovered.

    Same projection convention and z-buffer as core.projection_render, minus
    textures — kept local because this variant needs only depth.
    """
    np = _require_numpy()
    from atlas_camera.core.projection_render import project_points

    depth = np.full((height, width), np.nan)
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        return depth
    px, fwd = project_points(verts, view_matrix, fx, fy, cx, cy)
    tri_px = px[tris]
    tri_fwd = fwd[tris]
    keep = (tri_fwd > 1e-6).all(axis=1)
    mins, maxs = tri_px.min(axis=1), tri_px.max(axis=1)
    keep &= ((maxs[:, 0] >= 0) & (mins[:, 0] <= width - 1)
             & (maxs[:, 1] >= 0) & (mins[:, 1] <= height - 1))
    zbuf = np.full((height, width), np.inf)
    for t in np.nonzero(keep)[0]:
        p = tri_px[t]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())), width - 1)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        y1 = min(int(np.ceil(p[:, 1].max())), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        (ax, ay), (bx, by), (ccx, ccy) = p
        det = (bx - ax) * (ccy - ay) - (ccx - ax) * (by - ay)
        if abs(det) < 1e-12:
            continue
        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w1 = ((gx - ax) * (ccy - ay) - (ccx - ax) * (gy - ay)) / det
        w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        invw = (w0 / tri_fwd[t, 0] + w1 / tri_fwd[t, 1] + w2 / tri_fwd[t, 2])
        d = 1.0 / np.maximum(invw, 1e-12)
        sub = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        visible = inside & (d < zbuf[sub])
        zbuf[sub][visible] = d[visible]
    covered = np.isfinite(zbuf) & (zbuf < np.inf)
    depth[covered] = zbuf[covered]
    return depth


def fill_interior_invalid(depth: Any, *, iterations: int = 256) -> Any:
    """Close interior NaN regions of a depth grid by neighbour diffusion.

    Border-connected invalid regions (the frame surround, sky cut to the top
    edge) are left NaN — only ENCLOSED invalid pockets become geometry, which
    is the hole-filling contract. Jacobi mean fill, seeded from the pocket's
    valid rim.
    """
    np = _require_numpy()
    d = np.asarray(depth, dtype=np.float64).copy()
    invalid = ~np.isfinite(d)
    if not invalid.any():
        return d

    # Flood border-connected invalid via iterative dilation limited to invalid.
    border = np.zeros_like(invalid)
    border[0, :] = invalid[0, :]
    border[-1, :] = invalid[-1, :]
    border[:, 0] = invalid[:, 0]
    border[:, -1] = invalid[:, -1]
    while True:
        grown = border.copy()
        grown[1:, :] |= border[:-1, :]
        grown[:-1, :] |= border[1:, :]
        grown[:, 1:] |= border[:, :-1]
        grown[:, :-1] |= border[:, 1:]
        grown &= invalid
        if (grown == border).all():
            break
        border = grown
    pocket = invalid & ~border
    if not pocket.any():
        return d

    filled = np.where(np.isfinite(d), d, 0.0)
    weight = np.isfinite(d).astype(np.float64)
    for _ in range(int(iterations)):
        num = np.zeros_like(filled)
        den = np.zeros_like(weight)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            num += np.roll(filled * weight, shift, axis=(0, 1))
            den += np.roll(weight, shift, axis=(0, 1))
        upd = pocket & (den > 0)
        filled[upd] = num[upd] / den[upd]
        weight[upd] = 1.0
    reached = pocket & (weight > 0)
    d[reached] = filled[reached]
    return d


def surface_nets(occupancy: Any) -> tuple:
    """Naive surface nets over a boolean grid -> (vertices, faces int Nx3).

    ``occupancy`` is (nz, ny, nx) bool. The grid is padded with an empty
    border internally, so the output is watertight by construction whenever
    the solid is bounded. Vertex coordinates are in SAMPLE space (z, y, x) of
    the unpadded grid (cell centres sit at half-integers, from -0.5 to n-0.5).
    Faces are outward-wound triangles (positive signed volume for a solid).
    """
    np = _require_numpy()
    occ = np.asarray(occupancy, dtype=bool)
    p = np.pad(occ, 1, mode="constant", constant_values=False)
    nz, ny, nx = p.shape

    cell_id = np.full((nz - 1, ny - 1, nx - 1), -1, dtype=np.int64)

    quads = []  # (c0, c1, c2, c3) as cell-index triples, outward order

    def cells_for_edges(kk, jj, ii, axis, flip):
        """The 4 dual cells around each crossing edge, wound outward."""
        if axis == 0:      # edge along z at sample (k, j, i): vary (j-1..j, i-1..i)
            ring = [(kk, jj - 1, ii - 1), (kk, jj - 1, ii),
                    (kk, jj, ii), (kk, jj, ii - 1)]
        elif axis == 1:    # edge along y
            ring = [(kk - 1, jj, ii - 1), (kk, jj, ii - 1),
                    (kk, jj, ii), (kk - 1, jj, ii)]
        else:              # edge along x
            ring = [(kk - 1, jj - 1, ii), (kk - 1, jj, ii),
                    (kk, jj, ii), (kk, jj - 1, ii)]
        return ring[::-1] if flip else ring

    for axis in (0, 1, 2):
        a = p
        b = np.roll(p, -1, axis=axis)
        # valid sample range along the axis excludes the last (rolled) slot
        crossing = a != b
        sl = [slice(None)] * 3
        sl[axis] = slice(0, p.shape[axis] - 1)
        crossing = crossing[tuple(sl)]
        ks, js, is_ = np.nonzero(crossing)
        solid_first = a[tuple(sl)][ks, js, is_]
        for kk, jj, ii, sf in zip(ks, js, is_, solid_first):
            # Outward winding: verified by the signed-volume test (a solid
            # cube must come out positive) — reverse when the SOLID sample is
            # first along the axis.
            ring = cells_for_edges(int(kk), int(jj), int(ii), axis, bool(sf))
            quads.append(ring)

    if not quads:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    verts = []
    tris = []
    for ring in quads:
        ids = []
        for (ck, cj, ci) in ring:
            vid = cell_id[ck, cj, ci]
            if vid < 0:
                vid = len(verts)
                cell_id[ck, cj, ci] = vid
                # cell centre in padded sample coords -> unpadded sample coords
                verts.append((ck + 0.5 - 1.0, cj + 0.5 - 1.0, ci + 0.5 - 1.0))
            ids.append(int(vid))
        tris.append((ids[0], ids[1], ids[2]))
        tris.append((ids[0], ids[2], ids[3]))

    return (np.asarray(verts, dtype=np.float64),
            np.asarray(tris, dtype=np.int64))


def taubin_smooth(vertices: Any, faces: Any, *, iterations: int = 8,
                  lam: float = 0.5, mu: float = -0.53) -> Any:
    """Uniform-weight Taubin smoothing (shrink then inflate) — pure numpy."""
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0 or iterations <= 0:
        return v
    edges = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]],
                                         f[:, [2, 0]]]), axis=1), axis=0)
    deg = np.zeros(len(v))
    np.add.at(deg, edges[:, 0], 1.0)
    np.add.at(deg, edges[:, 1], 1.0)
    deg = np.maximum(deg, 1.0)[:, None]

    def one_pass(pts, factor):
        acc = np.zeros_like(pts)
        np.add.at(acc, edges[:, 0], pts[edges[:, 1]])
        np.add.at(acc, edges[:, 1], pts[edges[:, 0]])
        return pts + factor * (acc / deg - pts)

    for _ in range(int(iterations)):
        v = one_pass(v, lam)
        v = one_pass(v, mu)
    return v


def voxel_remesh(
    vertices: Any,
    faces: Any,
    *,
    view_matrix: Any,
    fx: float, fy: float, cx: float, cy: float,
    image_width: int, image_height: int,
    grid: int = 96,
    thickness_vox: int = 3,
    smooth_iterations: int = 8,
    close_holes: bool = True,
) -> tuple:
    """Solidify + watertight-remesh a relief mesh. Returns (verts_world, faces).

    ``grid`` is the raster long edge (and depth slab count); ``thickness_vox``
    the shell thickness in depth slabs. Raises ValueError when the mesh covers
    nothing from the camera (nothing to solidify).
    """
    np = _require_numpy()
    long_edge = max(int(image_width), int(image_height), 1)
    gw = max(8, int(round(int(image_width) * int(grid) / long_edge)))
    gh = max(8, int(round(int(image_height) * int(grid) / long_edge)))
    sx, sy = gw / float(image_width), gh / float(image_height)
    fx_r, fy_r = float(fx) * sx, float(fy) * sy
    cx_r, cy_r = float(cx) * sx, float(cy) * sy

    depth = render_depth_grid(vertices, faces, view_matrix,
                              fx_r, fy_r, cx_r, cy_r, gw, gh)
    if close_holes:
        depth = fill_interior_invalid(depth)
    valid = np.isfinite(depth)
    if not valid.any():
        raise ValueError("voxel_remesh: mesh covers no raster pixels from the "
                         "recovered camera — nothing to solidify")

    w_lo = float(np.nanmin(depth))
    w_hi = float(np.nanmax(depth))
    nz = int(grid)
    dz = max((w_hi - w_lo), 1e-3) / max(nz - 2 * thickness_vox - 2, 1)
    w0 = w_lo - 1.5 * dz
    slabs = w0 + (np.arange(nz) + 0.5) * dz            # slab centre depths
    thick = float(thickness_vox) * dz

    d = np.where(valid, depth, np.inf)
    occ = ((slabs[:, None, None] >= d[None, :, :])
           & (slabs[:, None, None] <= (d + thick)[None, :, :]))

    v_grid, f_out = surface_nets(occ)
    if len(f_out) == 0:
        raise ValueError("voxel_remesh: solidified occupancy produced no "
                         "surface (grid too coarse?)")
    v_grid = taubin_smooth(v_grid, f_out, iterations=int(smooth_iterations))

    # Grid sample coords -> camera -> world. Sample (z, y, x): pixel centre
    # (x + 0.5, y + 0.5); slab depth via the same affine as `slabs`.
    zc = v_grid[:, 0]
    yc = v_grid[:, 1]
    xc = v_grid[:, 2]
    w = w0 + (zc + 0.5) * dz
    u_px = xc + 0.5
    v_px = yc + 0.5
    x_cam = (u_px - cx_r) / fx_r * w
    y_cam = -(v_px - cy_r) / fy_r * w
    z_cam = -w
    cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(w)], axis=1)
    world = cam @ np.linalg.inv(np.asarray(view_matrix, dtype=np.float64)).T
    return world[:, :3], f_out
