"""A bounded screen-space edge-extension skirt on a torn silhouette.

`build_relief_mesh` tears at depth cliffs, which is correct — the two sides of a
cliff must never be joined — but the tear leaves a hard open rim, and the
alternative already shipped (`soft_visibility`) removes the tear entirely and
lets every cliff cell stretch into a fin whose length is whatever the depth jump
dictates. Measured on the castle plate, those fins run metres past the roofline
and read as straight slabs from any off-axis orbit; the shader fades them in a
textured view, but the geometry is what a DCC receives, and a DCC has no shader.

This module keeps the tear and adds a *separate* ribbon of topology hanging off
the rim:

1. every open rim vertex spawns its own column of ``ribbon_rings + 1`` vertices,
   sharing **no index** with the foreground — ring 0 is a duplicate sitting at
   the rim vertex's own position, so the tear survives topologically as well as
   visually;
2. each ring steps outward in IMAGE space along the rim's outward normal, so the
   ribbon's apparent width is a fixed pixel count regardless of scene depth —
   the reason this is not an extrusion along the view ray. Under a pinhole
   camera every point on a ray through the camera centre projects to the SAME
   pixel, so a ray extrusion has exactly zero screen width and no amount of
   marching along it converges on a pixel target;
3. depth ramps from the rim's own depth toward the inferred behind-surface on a
   quadratic Bézier, monotonically, so the ribbon curls away from camera and
   never back toward it;
4. every ring inherits the rim vertex's UV unchanged. The ribbon is therefore an
   edge-extend CLAMP, not a texture smear — which is what the seam doctrine
   asks for, and it also means texture-derivative fades (`uSoftStretch`) cannot
   see it, hence ``ribbon_t``;
5. ``ribbon_t`` (0 at the rim, 1 at the outer edge) is authored here, at ring
   construction, and is never re-derived downstream from position, UV, depth or
   projected distance. The viewport shader and the exported vertex colour
   evaluate the SAME fade from it, so viewport and DCC agree.

Deliberately NOT a polygon-offset solver. At a tight concave corner neighbouring
outward normals cross and the strip folds; this detects the fold in pixel space
(projected quad area flipping sign against the sheet's majority), shrinks the
local width, and drops the quad if that still fails. Reporting a dropped quad
beats silently exporting inverted geometry.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

#: Bend magnitude beyond which the depth ramp stops being monotonic. The profile
#: is ``B(t) = d0 + (d_bg - d0) * f(t)`` with ``f'(t) = (1 - 2b) + 4bt``, a
#: straight line in ``t``, so it suffices to keep both ends non-negative:
#: ``f'(0) = 1 - 2b >= 0`` bounds b above, ``f'(1) = 1 + 2b >= 0`` bounds it
#: below. Both directions are useful and the sign is the artist's control —
#: NEGATIVE curls away fast and then levels off (the tight lip of the Maya
#: sculpt), POSITIVE dwells at foreground depth and then dives (a flange).
#: Outside +/-0.5 the ribbon would leave the rim moving toward the camera at one
#: end or come back toward it at the other.
RIBBON_BEND_MAX = 0.5

#: How hard the outer rings are pulled onto their neighbours' average, at t=1.
#: A torn lattice rim is a staircase, and a skirt built straight out of it
#: inherits every step as a terrace at its own depth — measured live on a
#: castle as slits between neighbouring strips from the front and separated
#: slats from behind. Smoothing the ring POSITIONS along the rim relaxes those
#: steps into the silhouette they approximate. Ring 0 is never moved (it must
#: stay coincident with the rim vertex it duplicates), and the relaxed offset is
#: renormalized back to the requested length, so the screen-space width contract
#: survives the smoothing exactly.
RIBBON_RELAX = 0.85
RIBBON_RELAX_PASSES = 6

#: How far the skirt may recede, as a multiple of its own WORLD width. Bounding
#: the lateral offset in screen pixels bounds only what the recovered camera
#: sees; the depth run was free to reach ``d_bg``, and where no real background
#: exists behind the rim — 95% of columns on a castle against sky — that is the
#: fallback ``d0 * (1 + depth_edge_rel)``, i.e. +50% of depth. Measured on a
#: 7680px plate at f=6963: a 256 px skirt is ~1.1 m wide at 30 m and was running
#: 15 m deep, invisible edge-on from the recovered camera and an enormous tube
#: the moment you orbit. A transition shell is a thin membrane, so its depth run
#: is tied to its width — which also makes ``ribbon_px`` the single control for
#: apparent length from ANY viewpoint, not just the recovered one.
RIBBON_MAX_DEPTH_SLOPE = 2.0

#: Where the shader's fade begins, in ``ribbon_t``. Mirrored in
#: `atlas_blockout.js` (`RIBBON_FADE_START`) and baked into the exported vertex
#: alpha, so all three evaluate one curve. Starting above 0 keeps the ribbon
#: opaque where it meets the rim — a fade that began at the rim would reintroduce
#: the soft-edged hole the tear exists to avoid.
RIBBON_FADE_START = 0.15


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "transition_ribbon requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


def ribbon_alpha(ribbon_t: Any, *, fade_start: float = RIBBON_FADE_START) -> Any:
    """The fade the viewport shader applies, evaluated on the CPU.

    One definition, three consumers: the GLSL fragment shader, the GLB
    ``COLOR_0`` bake, and the tests. ``smoothstep`` reproduced by hand because
    numpy has none and the shader's is normative.
    """
    np = _require_numpy()
    t = np.asarray(ribbon_t, dtype=np.float64)
    span = max(1.0 - float(fade_start), 1e-6)
    x = np.clip((t - float(fade_start)) / span, 0.0, 1.0)
    return (1.0 - x * x * (3.0 - 2.0 * x)).astype(np.float32)


def plain_unprojector(view_matrix: Any, fx: float, fy: float, cx: float, cy: float):
    """A back-projection for callers that hold a finished mesh, not a depth map.

    `build_relief_mesh` passes its OWN closure, which replays scale-about-camera,
    `floor_clamp` and `band_min_m` in the order the lattice used — that is the
    contract that stops a ribbon vertex drifting from the rim. A retopology pass
    has neither those parameters nor any need for them: its vertices are already
    scaled and already clamped, and the depths handed back to this function were
    recovered from those same vertices. Re-applying the clamps here would move
    points that are already where they belong.
    """
    np = _require_numpy()
    c2w = np.linalg.inv(np.asarray(view_matrix, dtype=np.float64))
    R_cw, cam = c2w[:3, :3], c2w[:3, 3]

    def _unproject(u_px, v_px, depth_value):
        out_shape = np.shape(u_px)
        u = np.asarray(u_px, dtype=np.float64).reshape(-1)
        v = np.asarray(v_px, dtype=np.float64).reshape(-1)
        dd = np.asarray(depth_value, dtype=np.float64).reshape(-1)
        local = np.stack([(u - cx) / fx * dd, -(v - cy) / fy * dd, -dd], axis=-1)
        return (local @ R_cw.T + cam).reshape(out_shape + (3,))

    return _unproject


def _signed_area(p0: Any, p1: Any, p2: Any, np: Any) -> Any:
    """Shoelace area of triangles given as (...,2) pixel arrays."""
    return 0.5 * (
        (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1])
        - (p2[..., 0] - p0[..., 0]) * (p1[..., 1] - p0[..., 1])
    )


def _open_rim(faces: Any, np: Any) -> tuple[Any, Any]:
    """Directed open-boundary edges and each one's opposite triangle vertex.

    Returns ``(edges (E,2), opposite (E,))``. An edge belongs to exactly one
    triangle iff its canonical (lo, hi) key occurs once — the same test as
    `mesh_repair.boundary_edges`, kept vectorized here because the ribbon also
    needs the third vertex to orient "outward", which that helper discards.
    """
    f = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    opposite = np.concatenate([f[:, 2], f[:, 0], f[:, 1]], axis=0)
    keys = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    rim = counts[inverse] == 1
    return directed[rim], opposite[rim]


def build_transition_ribbon(
    *,
    vertices: Any,
    faces: Any,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
    unproject: Callable[[float, float, float], Any],
    depth_edge_rel: float,
    depth_full: Any = None,
    valid_full: Any = None,
    image_width: int = 0,
    image_height: int = 0,
    ribbon_px: float = 64.0,
    ribbon_rings: int = 4,
    ribbon_bend: float = 0.2,
    adaptive: bool = True,
    weld_ring0: bool = True,
    smudge_px: float = 12.0,
    depth_slope: float = RIBBON_MAX_DEPTH_SLOPE,
    rim_normal_passes: int = 6,
    frame_margin_px: float = 2.0,
    max_ribbon_columns: int = 400000,
) -> dict[str, Any]:
    """Grow a bounded edge-extension skirt from the mesh's open silhouette rim.

    ``vertices``/``faces`` are the mesh as built so far, in Atlas world space.
    ``unproject(u_px, v_px, depth)`` must be the SAME back-projection the lattice
    used — passed in rather than re-derived so a ribbon vertex can never drift
    from the rim it hangs off, the same contract `subquad_cut.cut_torn_quads`
    keeps. ``depth`` here is pre-``scale`` forward depth, exactly as that
    callable expects.

    Returns ``{"positions", "ribbon_t", "source_index", "faces", "stats"}``.
    ``source_index`` points at the rim vertex each ribbon vertex inherits its UV
    from — the caller copies the UV rather than recomputing one, which is what
    makes the ribbon an edge clamp. ``faces`` are already offset by
    ``len(vertices)``.
    """
    np = _require_numpy()

    rings = max(1, int(ribbon_rings))
    bend = float(np.clip(float(ribbon_bend), -RIBBON_BEND_MAX, RIBBON_BEND_MAX))
    base = int(len(vertices))
    empty = {
        "positions": np.zeros((0, 3), dtype=np.float64),
        "ribbon_t": np.zeros((0,), dtype=np.float32),
        "source_index": np.zeros((0,), dtype=np.int64),
        "faces": np.zeros((0, 3), dtype=np.int64),
        "stats": {
            "n_columns": 0, "n_rings": rings, "ribbon_px": float(ribbon_px),
            "bend": bend, "n_faces": 0, "n_folded_quads": 0,
            "n_dropped_quads": 0, "budget_truncated": False,
        },
    }
    if not len(faces) or float(ribbon_px) <= 0.0:
        return empty

    verts = np.asarray(vertices, dtype=np.float64)
    # Depth is OPTIONAL, and that is not a convenience: it is used only to probe
    # for a real surface behind the rim, and that probe is the exception rather
    # than the rule — measured 4-6% of columns on a castle against sky and 0% on
    # a 7680px machine plate. Everything else already falls back to the
    # tear-margin ramp. A caller that has the mesh but not the depth map (a
    # retopology pass, which changes the rim and must re-derive the skirt on it)
    # therefore loses almost nothing by omitting it.
    have_depth = depth_full is not None and valid_full is not None
    if have_depth:
        depth_full = np.asarray(depth_full, dtype=np.float64)
        valid_full = np.asarray(valid_full, dtype=bool)
        height, width = depth_full.shape[:2]
    else:
        height, width = int(image_height), int(image_width)
        if height < 2 or width < 2:
            raise ValueError(
                "transition ribbon needs either depth_full/valid_full or an "
                "image_width/image_height to bound the frame")

    # Weld coincident vertices BEFORE reading the rim. `sub_quad_boundary`
    # emits each torn cell's polygons independently, so its crossing vertices
    # are duplicated per cell: the open rim it leaves is not a silhouette curve
    # but thousands of isolated one-edge fragments. Unwelded, every column fails
    # the share test below and normal smoothing has no neighbours to average
    # over, so each fragment becomes an independent blade pointing wherever its
    # own edge happens to face — measured live on a castle at 64 px, a dense
    # spray of loose shards instead of a skirt.
    #
    # Welding on WORLD POSITION is safe for exactly the reason the fragments
    # exist: the near and far sheets meet at the same PIXEL but carry different
    # DEPTHS, so they are never coincident in 3D. This merges same-sheet
    # duplicates and can never re-join a tear. The welded topology is used only
    # to find the rim and its connectivity; the mesh itself is not modified.
    weld_key = np.round(verts / 1e-6).astype(np.int64)
    _, weld_first, weld_inv = np.unique(
        weld_key, axis=0, return_index=True, return_inverse=True)
    weld = weld_first[weld_inv].astype(np.int64)
    welded_faces = weld[np.asarray(faces, dtype=np.int64)]
    degenerate_face = (
        (welded_faces[:, 0] == welded_faces[:, 1])
        | (welded_faces[:, 1] == welded_faces[:, 2])
        | (welded_faces[:, 0] == welded_faces[:, 2])
    )
    n_welded = int(len(verts) - len(weld_first))

    rim_edges, rim_opposite = _open_rim(welded_faces[~degenerate_face], np)
    if not len(rim_edges):
        return empty

    # Project every vertex once. Pixel position and forward depth are recovered
    # from the world point rather than tracked alongside it, so lattice vertices
    # and sub-quad cut vertices — which have no lattice row/column — are handled
    # by one code path. floor_clamp and band_min pushes are ray-preserving, so
    # the pixel is exact; the depth they changed is the depth the ribbon should
    # start from anyway.
    vm = np.asarray(view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    R_cw = c2w[:3, :3]
    cam = c2w[:3, 3]
    rel = verts - cam
    fwd = -(rel @ R_cw[:, 2])
    local_x = rel @ R_cw[:, 0]
    local_y = rel @ R_cw[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        px_u = cx + fx * local_x / fwd
        px_v = cy - fy * local_y / fwd
        vert_depth = fwd / max(float(scale), 1e-9)
    on_screen = np.isfinite(px_u) & np.isfinite(px_v) & (fwd > 1e-6)

    # Drop rim edges that are the plate frame rather than a silhouette. The
    # frame is an open edge for the same topological reason a cliff is, but it
    # is not a depth discontinuity and skirting it would wrap the whole plate.
    def _at_frame(idx: Any) -> Any:
        return (
            (px_u[idx] <= frame_margin_px)
            | (px_u[idx] >= width - 1 - frame_margin_px)
            | (px_v[idx] <= frame_margin_px)
            | (px_v[idx] >= height - 1 - frame_margin_px)
        )

    keep = on_screen[rim_edges[:, 0]] & on_screen[rim_edges[:, 1]]
    keep &= ~(_at_frame(rim_edges[:, 0]) & _at_frame(rim_edges[:, 1]))
    rim_edges = rim_edges[keep]
    rim_opposite = rim_opposite[keep]
    if not len(rim_edges):
        return empty

    # A budget on COLUMNS, applied to the rim edges that generate them — two
    # slots per edge is the bound, so capping edges caps columns.
    truncated = 2 * len(rim_edges) > int(max_ribbon_columns)
    if truncated:
        keep_edges = max(1, int(max_ribbon_columns) // 2)
        rim_edges = rim_edges[:keep_edges]
        rim_opposite = rim_opposite[:keep_edges]

    # Columns. Neighbouring quads SHARE a column so the skirt is continuous
    # rather than a fan of loose quads — but sharing is only safe where exactly
    # two rim edges meet, one arriving and one leaving.
    #
    # This is where the first version was wrong, and the exporter caught it: at a
    # pinch or T-junction three or more rim edges meet at one vertex, so one
    # shared column's internal edge B_k->B_k+1 ends up used by three faces. A
    # directed edge is duplicated, the mesh is non-manifold, and
    # `mesh_repair`'s face-fan walk never closes (observed: "did not close after
    # 72461 rotations" on a castle silhouette). Junction columns are therefore
    # DUPLICATED per incident edge. That leaves a hairline crack at the junction,
    # which is invisible — the two copies are coincident, exactly like ring 0
    # against the rim — and manifold, which the crack-free version was not.
    #
    # The one-in/one-out test is stricter than "degree == 2" on purpose: two
    # edges both leaving the same vertex would still duplicate the directed edge
    # after merging. Boundary half-edges of a consistently wound mesh form
    # directed cycles, so this holds everywhere except the degenerate cases it
    # is here to exclude.
    slot_vertex = rim_edges.reshape(-1)
    n_all = int(len(verts))
    out_deg = np.bincount(rim_edges[:, 0], minlength=n_all)
    in_deg = np.bincount(rim_edges[:, 1], minlength=n_all)
    shareable = (out_deg == 1) & (in_deg == 1)
    mergeable = shareable[slot_vertex]

    col_id = np.empty(len(slot_vertex), dtype=np.int64)
    merged_vertices, merged_inv = np.unique(
        slot_vertex[mergeable], return_inverse=True)
    col_id[mergeable] = merged_inv
    n_merged = int(len(merged_vertices))
    lone = np.nonzero(~mergeable)[0]
    col_id[lone] = n_merged + np.arange(len(lone))
    columns = np.concatenate([merged_vertices, slot_vertex[lone]])
    edge_cols = col_id.reshape(-1, 2)
    n_columns = int(len(columns))

    col_u = px_u[columns]
    col_v = px_v[columns]
    col_d0 = vert_depth[columns]

    # Outward normal, per rim edge, in pixel space: perpendicular to the edge,
    # pointing away from the triangle that owns it. Taking the side from the
    # opposite vertex rather than from the winding keeps this correct whatever
    # the image y convention does to a world-space CCW test.
    ax, ay = px_u[rim_edges[:, 0]], px_v[rim_edges[:, 0]]
    bx, by = px_u[rim_edges[:, 1]], px_v[rim_edges[:, 1]]
    ox, oy = px_u[rim_opposite], px_v[rim_opposite]
    tx, ty = bx - ax, by - ay
    perp = np.stack([-ty, tx], axis=1)
    away = np.stack([0.5 * (ax + bx) - ox, 0.5 * (ay + by) - oy], axis=1)
    flip = np.sum(perp * away, axis=1) < 0.0
    perp[flip] *= -1.0
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)

    # Average the incident edge normals onto each column, so a corner bisects
    # its two edges instead of tearing the strip in two directions.
    normals = np.zeros((n_columns, 2), dtype=np.float64)
    np.add.at(normals, edge_cols[:, 0], perp)
    np.add.at(normals, edge_cols[:, 1], perp)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = nlen[:, 0] < 1e-6
    normals = normals / np.maximum(nlen, 1e-9)

    # Smooth the direction field ALONG the rim before offsetting. A torn lattice
    # rim is a staircase even when the real silhouette is a straight diagonal,
    # and the staircase's normals alternate between axis-aligned directions 90°
    # apart. Offsetting each vertex along its own alternating normal folds the
    # strip at every single step — measured on the diagonal-cliff fixture at
    # 48 px, most columns collapsed to a fold-clamped 4-12 px. Averaging over a
    # few rim neighbours recovers the true silhouette direction, so the clamp is
    # left to handle genuine concave corners rather than quantization. Positions
    # are untouched: ring 0 must stay exactly on the rim vertex it duplicates.
    for _ in range(rim_normal_passes):
        acc = normals.copy()
        np.add.at(acc, edge_cols[:, 0], normals[edge_cols[:, 1]])
        np.add.at(acc, edge_cols[:, 1], normals[edge_cols[:, 0]])
        normals = acc / np.maximum(
            np.linalg.norm(acc, axis=1, keepdims=True), 1e-9)
    degenerate |= np.linalg.norm(normals, axis=1) < 1e-6

    col_degree = np.zeros(n_columns, dtype=np.float64)
    np.add.at(col_degree, edge_cols[:, 0], 1.0)
    np.add.at(col_degree, edge_cols[:, 1], 1.0)
    col_degree = np.maximum(col_degree, 1.0)

    # Behind-surface depth. March outward and keep only samples that are
    # genuinely behind by the SAME relative-discontinuity margin that tore the
    # mesh here — otherwise a normal that happens to point at a nearer tower,
    # chimney or foreground island would curl the ribbon forward.
    jump_rel = max(float(depth_edge_rel), 1e-3)
    fallback = col_d0 * (1.0 + jump_rel)
    if have_depth:
        probes = np.linspace(0.35, 1.6, 12)[None, :] * float(ribbon_px)
        su = np.clip(np.rint(col_u[:, None] + normals[:, 0:1] * probes),
                     0, width - 1).astype(np.int64)
        sv = np.clip(np.rint(col_v[:, None] + normals[:, 1:2] * probes),
                     0, height - 1).astype(np.int64)
        samples = depth_full[sv, su]
        ok = valid_full[sv, su] & np.isfinite(samples)
        ok &= samples >= col_d0[:, None] * (1.0 + jump_rel)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            # An all-rejected row is the expected case at a rim with nothing
            # behind it (sky), not an error — nanmedian's All-NaN warning goes
            # through the warnings module, so errstate alone does not silence it.
            warnings.simplefilter("ignore", RuntimeWarning)
            col_bg = np.nanmedian(np.where(ok, samples, np.nan), axis=1)
        col_bg = np.where(np.isfinite(col_bg), col_bg, fallback)
        col_bg = np.maximum(col_bg, fallback)
        n_measured_bg = int(ok.any(axis=1).sum())
    else:
        col_bg = fallback
        n_measured_bg = 0

    # NOT smoothed along the rim, deliberately. Smoothing `col_bg` looked like
    # the obvious cure for a scalloped skirt and measured as doing NOTHING: the
    # depth cap binds on most columns (all of them at a narrow ribbon_px), so
    # the probed background never reaches the geometry, and where it does reach
    # it the neighbouring values already agree. Tried and removed rather than
    # left in as unfalsifiable insurance — the scallop is an ASPECT-RATIO
    # problem, see `columns_per_ribbon_width` below.

    # Adaptive width on the RELATIVE jump, referenced to the tear threshold.
    # A raw metre difference is scene-scale dependent — 10 m is enormous in one
    # reconstruction and negligible in another — and the tear threshold is
    # already Atlas's normalised statement of "how big a discontinuity counts".
    col_width = np.full(n_columns, float(ribbon_px), dtype=np.float64)
    if adaptive:
        rel_jump = np.abs(col_bg - col_d0) / np.maximum(np.abs(col_d0), 1e-6)
        col_width *= np.clip(rel_jump / jump_rel, 0.5, 2.0)
    col_width[degenerate] = 0.0

    # Bound the depth run to the skirt's own world width. `col_width` is in
    # plate pixels; at depth d0 that subtends `col_width * d0 / f` metres, so
    # the cap is a pure aspect-ratio limit and carries no scene-scale assumption.
    focal = max(0.5 * (abs(float(fx)) + abs(float(fy))), 1e-6)
    slope = max(float(depth_slope), 0.0)
    world_width = col_width * col_d0 / focal
    delta = np.minimum(col_bg - col_d0, slope * world_width)
    n_depth_capped = int(np.sum((col_bg - col_d0) > delta + 1e-9))
    col_bg = col_d0 + np.maximum(delta, 1e-6)

    t_ring = (np.arange(rings + 1, dtype=np.float64) / rings)[None, :]
    # f(t) for the quadratic Bézier on DEPTH: P1 = d0 + (0.5 - bend)*(d_bg - d0).
    # bend is a dwell, not a bulge — it says how long the ribbon stays near
    # foreground depth before falling away, and it is clamped so f' can never
    # go negative and pull the ribbon back toward the camera.
    f_t = 2.0 * (1.0 - t_ring) * t_ring * (0.5 - bend) + t_ring * t_ring
    ring_depth = col_d0[:, None] + (col_bg - col_d0)[:, None] * f_t

    # Fold control runs entirely in pixel space, before any unprojection: at a
    # tight concave corner the neighbouring outward normals cross and the strip
    # turns itself inside out. Shrink the offending columns, then drop what is
    # still inverted.
    # Relaxation ramps in with t: 0 at the rim, RIBBON_RELAX at the outer edge.
    relax_w = RIBBON_RELAX * t_ring

    # Smooth the BASE CURVE once, and slide each ring from the true rim toward
    # it as t grows. Two earlier shapes of this failed for the same reason and
    # it is worth stating: any construction of the form `ring = col + f(...)`
    # carries the rim's own Laplacian through unchanged, so the outer ring can
    # never be smoother than the staircase it came from (measured both times:
    # outer 2.77 px against a rim's 2.83 px). The base term itself has to be
    # attenuated. Ring 0 keeps weight 0 and therefore stays exactly on the rim
    # vertex it duplicates, which is the constraint that cannot move.
    base_u, base_v = col_u.copy(), col_v.copy()
    for _ in range(RIBBON_RELAX_PASSES):
        acc_u = np.zeros_like(base_u)
        acc_v = np.zeros_like(base_v)
        np.add.at(acc_u, edge_cols[:, 0], base_u[edge_cols[:, 1]])
        np.add.at(acc_u, edge_cols[:, 1], base_u[edge_cols[:, 0]])
        np.add.at(acc_v, edge_cols[:, 0], base_v[edge_cols[:, 1]])
        np.add.at(acc_v, edge_cols[:, 1], base_v[edge_cols[:, 0]])
        base_u = 0.5 * base_u + 0.5 * (acc_u / col_degree)
        base_v = 0.5 * base_v + 0.5 * (acc_v / col_degree)
    drift_u = base_u - col_u
    drift_v = base_v - col_v

    scale_local = np.ones(n_columns, dtype=np.float64)
    quad_ok = None
    n_folded = 0
    for _ in range(4):
        offs = col_width[:, None] * scale_local[:, None] * t_ring
        # Base slides toward the smoothed rim as t grows; the outward offset is
        # applied on top, so the radial extent is still exactly `offs` and the
        # width contract survives. The lateral drift is bounded by the staircase
        # amplitude (one grid step), which is why it costs the measured width
        # essentially nothing.
        ring_u = col_u[:, None] + relax_w * drift_u[:, None] + normals[:, 0:1] * offs
        ring_v = col_v[:, None] + relax_w * drift_v[:, None] + normals[:, 1:2] * offs

        pa = np.stack([ring_u[edge_cols[:, 0]], ring_v[edge_cols[:, 0]]], axis=-1)
        pb = np.stack([ring_u[edge_cols[:, 1]], ring_v[edge_cols[:, 1]]], axis=-1)
        areas = _signed_area(pa[:, :-1], pb[:, :-1], pb[:, 1:], np) + _signed_area(
            pa[:, :-1], pb[:, 1:], pa[:, 1:], np
        )
        finite = areas[np.isfinite(areas)]
        ref = np.median(finite) if finite.size else 0.0
        ref_sign = 1.0 if ref >= 0.0 else -1.0
        quad_ok = np.isfinite(areas) & (areas * ref_sign > 0.0)
        bad_edges = ~quad_ok.all(axis=1)
        n_folded = int((~quad_ok).sum())
        if not bad_edges.any():
            break
        scale_local[edge_cols[bad_edges, 0]] *= 0.5
        scale_local[edge_cols[bad_edges, 1]] *= 0.5

    # Unproject the whole ring lattice in ONE call. `unproject` is the mesh's
    # own back-projection, passed in rather than re-derived so ribbon vertices
    # cannot drift from the rim they hang off; it broadcasts, so keeping that
    # single copy costs nothing here. A per-point call is what previously forced
    # a column budget small enough to crop a real silhouette.
    positions = np.asarray(
        unproject(ring_u, ring_v, ring_depth), dtype=np.float64
    ).reshape(n_columns, rings + 1, 3)

    # Ring 0 either duplicates the rim vertex or IS it.
    #
    # Welding is safe, and the rule it appears to break is about a different
    # join: "the two sheets must never share a vertex" is the NEAR sheet against
    # the FAR sheet across a depth cliff. A skirt sharing a vertex with its OWN
    # rim bridges nothing — the far sheet stays exactly as separate as before —
    # and it buys continuous normals and the impossibility of a crack opening
    # between the mesh and the skirt hanging off it.
    #
    # It does change the winding rule. Two faces sharing an edge on a manifold
    # traverse it in OPPOSITE directions, and `_open_rim` hands back each rim
    # edge in the direction its own base triangle uses it. So a welded quad must
    # take the edge reversed, and the winding has to come from that adjacency —
    # not from the projected-sign match used when the skirt is free-floating,
    # which knows nothing about the edge it is being attached to.
    if weld_ring0:
        new_per_column = rings
        ring_index = np.empty((n_columns, rings + 1), dtype=np.int64)
        ring_index[:, 0] = columns
        ring_index[:, 1:] = (
            base + np.arange(n_columns, dtype=np.int64)[:, None] * rings
            + np.arange(rings, dtype=np.int64)[None, :])
        keep_rings = slice(1, rings + 1)
        ea, eb = edge_cols[:, 1], edge_cols[:, 0]   # reversed: see above
    else:
        new_per_column = rings + 1
        ring_index = (base + np.arange(n_columns, dtype=np.int64)[:, None]
                      * (rings + 1) + np.arange(rings + 1, dtype=np.int64)[None, :])
        keep_rings = slice(0, rings + 1)
        ea, eb = edge_cols[:, 0], edge_cols[:, 1]

    ribbon_t = np.broadcast_to(
        t_ring, (n_columns, rings + 1))[:, keep_rings].astype(np.float32)
    source_index = np.repeat(columns, new_per_column).astype(np.int64)
    emit_positions = positions[:, keep_rings, :]

    # Two triangles per surviving quad.
    kk = np.arange(rings)[None, :]
    ia0 = ring_index[ea][:, :-1]
    ib0 = ring_index[eb][:, :-1]
    ia1 = ring_index[ea][:, 1:]
    ib1 = ring_index[eb][:, 1:]
    tris = np.concatenate(
        [
            np.stack([ia0, ib0, ib1], axis=-1)[quad_ok],
            np.stack([ia0, ib1, ia1], axis=-1)[quad_ok],
        ],
        axis=0,
    ).astype(np.int64)

    if len(tris) and not weld_ring0:
        # ONE decision for the whole strip, not one per triangle. Flipping
        # triangles individually to match a global sign is what made this
        # non-manifold the first time: adjacent quads have to be wound
        # consistently WITH EACH OTHER, and a near-degenerate quad's projected
        # sign is arbitrary, so flipping it duplicates a directed edge on the
        # column the two quads share. Reversing every ribbon triangle at once
        # preserves that consistency by construction.
        flat_faces = np.asarray(faces, dtype=np.int64)
        fg_area = _signed_area(
            np.stack([px_u[flat_faces[:, 0]], px_v[flat_faces[:, 0]]], axis=-1),
            np.stack([px_u[flat_faces[:, 1]], px_v[flat_faces[:, 1]]], axis=-1),
            np.stack([px_u[flat_faces[:, 2]], px_v[flat_faces[:, 2]]], axis=-1),
            np,
        )
        fg_finite = fg_area[np.isfinite(fg_area)]
        want = 1.0 if (fg_finite.size and np.median(fg_finite) >= 0.0) else -1.0
        ring_px = np.stack([ring_u.reshape(-1), ring_v.reshape(-1)], axis=-1)
        local = tris - base
        got = _signed_area(ring_px[local[:, 0]], ring_px[local[:, 1]],
                           ring_px[local[:, 2]], np)
        got_finite = got[np.isfinite(got) & (np.abs(got) > 1e-9)]
        have = 1.0 if (got_finite.size and np.median(got_finite) >= 0.0) else -1.0
        if have * want < 0.0:
            tris = tris[:, [0, 2, 1]]

    # Acceptance metric: measure the finished ribbon rather than trusting the
    # request. Screen width is exact by construction — the offset IS in pixels
    # and every clamp downstream is ray-preserving — so a drift here means the
    # construction stopped being screen-space, which is the whole contract.
    # Staircase roughness, measured with the rim adjacency — the only place it
    # exists. This is the Laplacian magnitude relaxation minimizes, evaluated at
    # the rim and at the outer ring, so "did the terracing actually relax" is a
    # reported number rather than a judgement call about a grey render.
    def _roughness(uu: Any, vv: Any) -> float:
        acc_u = np.zeros_like(uu)
        acc_v = np.zeros_like(vv)
        np.add.at(acc_u, edge_cols[:, 0], uu[edge_cols[:, 1]])
        np.add.at(acc_u, edge_cols[:, 1], uu[edge_cols[:, 0]])
        np.add.at(acc_v, edge_cols[:, 0], vv[edge_cols[:, 1]])
        np.add.at(acc_v, edge_cols[:, 1], vv[edge_cols[:, 0]])
        return float(np.median(np.hypot(acc_u / col_degree - uu,
                                        acc_v / col_degree - vv)))

    span = np.hypot(ring_u[:, -1] - ring_u[:, 0], ring_v[:, -1] - ring_v[:, 0])
    # Fold-clamped columns are DELIBERATELY narrower than the request, so
    # folding them into the width metric would report the clamp as width error
    # and hide a real regression behind it. Count them instead.
    unclamped = (~degenerate) & (scale_local >= 1.0)
    live = span[unclamped]
    stats = {
        "n_columns": n_columns,
        "n_rings": rings,
        "ribbon_px": float(ribbon_px),
        "bend": bend,
        "adaptive": bool(adaptive),
        # Carried so a downstream pass that changes the RIM (retopology) can
        # re-derive the same skirt on the new one instead of guessing.
        "depth_edge_rel": float(depth_edge_rel),
        "had_depth": bool(have_depth),
        # Not geometry — a shading width, carried here so the viewport shader
        # and the GLB bake read ONE number rather than each keeping a default.
        "smudge_px": float(smudge_px),
        "n_faces": int(len(tris)),
        "n_rim_edges": int(len(rim_edges)),
        "n_folded_quads": n_folded,
        "n_dropped_quads": int((~quad_ok).sum()) if quad_ok is not None else 0,
        "n_width_clamped": int((~unclamped).sum()),
        "weld_ring0": bool(weld_ring0),
        # How the skirt's width compares to the spacing between the columns
        # that build it. Below 1.0 the skirt is narrower than the gap between
        # its own fingers and reads as a FRINGE of tongues with the quads
        # sagging between them — the "U shapes between extrusions" this was
        # reported as, which `depth_slope` then multiplies. Spacing is a
        # property of the rim (≈ one lattice cell), so the fix is ribbon_px,
        # not bend.
        "column_spacing_px": float(np.median(
            np.hypot(col_u[edge_cols[:, 0]] - col_u[edge_cols[:, 1]],
                     col_v[edge_cols[:, 0]] - col_v[edge_cols[:, 1]]))),
        "columns_per_ribbon_width": float(
            float(ribbon_px) / max(float(np.median(
                np.hypot(col_u[edge_cols[:, 0]] - col_u[edge_cols[:, 1]],
                         col_v[edge_cols[:, 0]] - col_v[edge_cols[:, 1]]))), 1e-6)),
        "n_depth_capped": n_depth_capped,
        "depth_slope_max": slope,
        # World length of the skirt, which is what an ORBIT sees — the screen
        # width above is only what the recovered camera sees, and the two came
        # apart badly before the depth cap.
        "world_len_p50_m": float(np.percentile(
            np.linalg.norm(positions[:, -1] - positions[:, 0], axis=-1)[unclamped], 50))
        if int(unclamped.sum()) else 0.0,
        "world_len_p95_m": float(np.percentile(
            np.linalg.norm(positions[:, -1] - positions[:, 0], axis=-1)[unclamped], 95))
        if int(unclamped.sum()) else 0.0,
        "rim_roughness_px": _roughness(col_u, col_v),
        "outer_roughness_px": _roughness(ring_u[:, -1], ring_v[:, -1]),
        "relax": RIBBON_RELAX,
        "n_columns_measured_bg": n_measured_bg,
        "measured_px_p50": float(np.percentile(live, 50)) if live.size else 0.0,
        "measured_px_p95": float(np.percentile(live, 95)) if live.size else 0.0,
        "budget_truncated": bool(truncated),
    }
    return {
        "positions": emit_positions.reshape(-1, 3),
        "ribbon_t": ribbon_t.reshape(-1),
        "source_index": source_index,
        "faces": tris,
        "stats": stats,
    }
