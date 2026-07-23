"""Mesh-topology interior-hole fill — pure numpy, no external deps.

Atlas relief meshes are torn at depth discontinuities (silhouettes, sky, band
clips) — deliberate, so the live 📽 projection never rubber-sheets background
onto foreground. But the EXPORTED OBJ/GLB handed to a DCC benefits from being
closer to watertight *inside the subject*: small interior tear holes (noise,
fine structure, band-clip seams) become stray open boundaries that block retopo
/ boolean ops / 3D-print prep in Maya/ZBrush/Blender.

This fills ONLY interior enclosed boundary loops — never the outer silhouette /
frame boundary — by walking closed boundary loops and triangulating each
qualifying loop FROM ITS EXISTING VERTICES (no new verts → the 1:1 vertex-UV
mapping the OBJ/GLB writers depend on stays exact, so projection-baked UVs
remain valid).

A fill must not introduce the very defects it exists to remove, so three
properties are load-bearing (each pinned by a test):

*   **Triangulate inside the hole.** Tear footprints on a decimated grid are
    routinely non-convex (L / staircase). A naive fan from ``loop[0]`` is valid
    only when the loop is star-shaped from that vertex; anchored on a reflex
    vertex it emits triangles that spill OUTSIDE the hole over real geometry.
    We ear-clip in the loop's own best-fit plane instead, which handles any
    simple polygon.
*   **Match the surrounding winding.** Loop direction is a property of the walk,
    not of the mesh, so the fill's orientation is aligned explicitly against the
    face adjacent to the loop — otherwise fills come out back-facing.
*   **Emit no degenerate faces.** Grid-aligned tears have collinear vertex runs,
    and triangulating across one yields a zero-area sliver. Such a vertex is
    never clipped as an ear — but it is never *dropped* either: deleting it
    would leave the cap spanning boundary edges that still exist on the mesh,
    i.e. a T-junction, which is no more watertight than the hole was. Every loop
    vertex ends up in the fill. This is safe because a simple polygon with
    positive area always has a strictly-convex ear elsewhere (two-ears theorem),
    and a collinear vertex becomes clippable once its neighbours change.

Boundary loops are recovered with a **pivot walk** over the face fan around each
vertex, not by following the undirected boundary graph. Where two tears meet at
one grid vertex (a "pinch", boundary degree 4) the surface's boundary is really
ONE curve that passes through that vertex twice — the undirected walk wanders
from one hole into the other and bails, dropping both. The pivot walk crosses
interior edges only, so it recovers that curve intact; it is then split at the
repeated vertex back into the two simple loops an artist would call holes.

Two scoping mechanisms, composable:

1.  **Edge-count threshold** (``max_hole_edges``) — a loop is filled only if its
    edge count is below the threshold. The outer frame boundary is ~the grid's
    perimeter (e.g. ~512 edges at grid 128), interior tear loops are ~4–30, so a
    threshold around 64 separates them by construction. When no depth window is
    given, the SINGLE LARGEST loop is ALSO always left open as a belt-and-braces
    "outer boundary" guard (covers the rare case where a genuinely large interior
    tear produces a loop bigger than the threshold would catch, or where the
    threshold is raised deliberately).

2.  **Band-box depth window** (``depth_near_m``/``depth_far_m`` +
    ``view_matrix``) — the artist-supplied foreground region, transcribed off
    ``AtlasBoundedBand``'s ``cutoff_m`` (far) and the band near. A loop is filled
    only if ALL its boundary vertices' forward depth (recovered-camera view
    space, the same axis the band box's cutoff plane lives on) falls within
    ``[near, far]``. The outer frame spans the full depth range (its far corners
    are at background depth, beyond the cutoff) so it is left open
    *automatically* — and background/sky holes outside the window stay open too,
    which is the DMP-correct behavior. This is the cleaner realization of "fill
    holes within the inside bounds of the constructed mesh": the band box IS the
    inside bound. With both bounds 0 the depth filter is disabled (threshold-only
    mode, with the largest-loop guard above).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Cross-products/areas below this count as zero. Relief-mesh vertices are in
# metres and a tear quad at grid 128 on a 4K plate is ~1e-2 m across, so this
# sits far below any real feature yet safely above float64 noise.
_EPS = 1e-12


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Edges that appear in exactly one triangle = the open boundary."""
    f = np.asarray(faces)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)  # canonical (lo, hi) so orientation doesn't matter
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def _directed_face_map(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """``(a, b) -> face index`` for every directed edge of every triangle.

    On a consistently-wound manifold each directed edge belongs to at most one
    face, which is what makes the pivot walk below well defined.
    """
    f = np.asarray(faces)
    fo: dict[tuple[int, int], int] = {}
    for fi in range(len(f)):
        a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
        fo[(a, b)] = fi
        fo[(b, c)] = fi
        fo[(c, a)] = fi
    return fo


def _walk_loops_pivot(faces: np.ndarray) -> list[list[int]]:
    """Boundary loops via the face fan around each vertex.

    A directed edge ``(a, b)`` is a boundary half-edge when its reverse doesn't
    exist. Its successor is found by pivoting around ``b`` across INTERIOR edges
    only until the fan runs out — so at a pinch vertex the walk stays on the
    hole it arrived on. Successor is a bijection over boundary half-edges, so
    the walk decomposes them into disjoint cycles with nothing dropped.
    """
    f = np.asarray(faces)
    fo = _directed_face_map(f)
    bhe = [(a, b) for (a, b) in fo if (b, a) not in fo]

    def _after(fi: int, v: int) -> int:
        """The vertex following ``v`` in face ``fi``."""
        tri = f[fi]
        if int(tri[0]) == v:
            return int(tri[1])
        if int(tri[1]) == v:
            return int(tri[2])
        return int(tri[0])

    def _succ(a: int, b: int) -> tuple[int, int]:
        fi = fo[(a, b)]
        x = _after(fi, b)
        while (x, b) in fo:  # b-x is interior → rotate to the adjacent face
            fi = fo[(x, b)]
            x = _after(fi, b)
        return b, x

    loops: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for start in bhe:
        if start in seen:
            continue
        loop: list[int] = []
        he = start
        while he not in seen:
            seen.add(he)
            loop.append(he[0])
            he = _succ(*he)
        if he == start and len(loop) >= 3:
            loops.extend(_split_at_repeats(loop))
    return loops


def _split_at_repeats(loop: list[int]) -> list[list[int]]:
    """Split a boundary curve into simple cycles at any repeated vertex.

    A pinch makes the boundary pass through one vertex twice, so the raw curve
    is a figure-eight rather than a polygon. Cutting at the repeat recovers the
    individual holes; each boundary edge still lands in exactly one sub-loop.
    """
    out: list[list[int]] = []
    stack: list[int] = []
    at: dict[int, int] = {}
    for v in loop:
        if v in at:
            i = at[v]
            sub = stack[i:]
            if len(sub) >= 3:
                out.append(sub)
            for w in sub:
                at.pop(w, None)
            stack = stack[:i]
        at[v] = len(stack)
        stack.append(v)
    if len(stack) >= 3:
        out.append(stack)
    return out


def walk_loops(bedges: np.ndarray, faces: np.ndarray | None = None) -> list[list[int]]:
    """Walk every closed boundary loop.

    Pass ``faces`` for the robust pivot walk (handles pinch vertices, where two
    tears meet at one vertex and the boundary graph alone is ambiguous). Without
    it, falls back to following the undirected boundary graph: each boundary
    vertex has degree 2 there, so from any unvisited start we follow the
    non-previous neighbour until we return to the start. That fallback bails on
    a non-simple cycle rather than spinning, which silently drops pinched loops.
    """
    if faces is not None:
        return _walk_loops_pivot(faces)

    adj: dict[int, list[int]] = {}
    for a, b in bedges:
        a, b = int(a), int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    done: set[int] = set()
    loops: list[list[int]] = []
    for start in adj:
        if start in done:
            continue
        loop = [start]
        prev, cur = -1, start
        closed = False
        while True:
            nxts = [n for n in adj[cur] if n != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                closed = True
                break
            if nxt in loop:
                break  # non-simple; bail
            loop.append(nxt)
            prev, cur = cur, nxt
        if closed:
            for v in loop:
                done.add(v)
            loops.append(loop)
    return loops


def _loop_forward_depths(loop: list[int], vertices: np.ndarray,
                         view_matrix: np.ndarray) -> np.ndarray:
    """Forward depth (metres, +in front of camera) of each loop vertex."""
    v = np.asarray(vertices, dtype=np.float64)
    pts = v[loop]  # (n, 3)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    vz = (hom @ np.asarray(view_matrix, dtype=np.float64).T)[:, 2]
    return -vz  # camera faces -Z → view-space z is negative forward


def _plane_coords(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project ``pts`` onto their best-fit plane. Returns (2D coords, normal)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[2]
    e1 = vt[0]
    e2 = np.cross(n, e1)
    q = np.stack([(pts - c) @ e1, (pts - c) @ e2], axis=1)
    return q, n


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _point_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                  c: np.ndarray) -> bool:
    d1, d2, d3 = _cross2(a, b, p), _cross2(b, c, p), _cross2(c, a, p)
    neg = (d1 < -_EPS) or (d2 < -_EPS) or (d3 < -_EPS)
    pos = (d1 > _EPS) or (d2 > _EPS) or (d3 > _EPS)
    return not (neg and pos)


def _find_ear(q: np.ndarray, ids: list[int], idx: list[int],
              existing: set[tuple[int, int]]) -> int | None:
    """Index into ``idx`` of a clippable ear, or None.

    Refuses an ear whose new diagonal already exists in the mesh: reusing one
    would put that edge in three faces (non-manifold), which breaks a DCC
    boolean just as surely as the hole did.
    """
    for i in range(len(idx)):
        p, cu, nx = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
        a, b, c = q[p], q[cu], q[nx]
        if _cross2(a, b, c) <= _EPS:
            continue  # reflex, or collinear (its ear would be degenerate)
        if any(_point_in_tri(q[j], a, b, c)
               for j in idx if j not in (p, cu, nx)):
            continue  # another vertex inside → not an ear
        lo, hi = sorted((ids[p], ids[nx]))
        if (lo, hi) in existing:
            continue  # would duplicate a mesh edge → non-manifold
        return i
    return None


def _clip_once(q: np.ndarray, ids: list[int],
               existing: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """One greedy ear-clipping pass over a CCW polygon. [] if it gets stuck."""
    idx = list(range(len(ids)))
    tris: list[tuple[int, int, int]] = []
    guard = 4 * len(ids) * len(ids) + 8
    while len(idx) > 3 and guard > 0:
        guard -= 1
        i = _find_ear(q, ids, idx, existing)
        if i is None:
            return []
        p, cu, nx = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
        tris.append((ids[p], ids[cu], ids[nx]))
        idx.pop(i)
    if len(idx) == 3:
        a, b, c = idx
        if abs(_cross2(q[a], q[b], q[c])) > _EPS:
            tris.append((ids[a], ids[b], ids[c]))
    return tris


def _ear_clip(q: np.ndarray, ids: list[int],
              existing: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Ear-clip a CCW simple polygon given as 2D coords ``q`` + vertex ``ids``.

    Collinear vertices are skipped as ear candidates (their ear would be a
    zero-area sliver) but never removed — dropping one would leave a T-junction
    against the boundary edges it still carries. It becomes clippable once a
    neighbouring ear goes; the two-ears theorem guarantees a strictly-convex ear
    exists meanwhile.

    Greedy clipping is scan-order dependent and can corner itself into needing a
    diagonal that already exists, so each rotation of the polygon is tried before
    giving up. Returns [] if none succeeds — leave that hole open rather than
    emit geometry that is non-manifold or outside the hole.
    """
    for r in range(len(ids)):
        tris = _clip_once(np.roll(q, -r, axis=0), ids[r:] + ids[:r], existing)
        if tris:
            return tris
    return []


def _triangulate_loop(loop: list[int], vertices: np.ndarray,
                      fo: dict[tuple[int, int], int],
                      existing: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Triangulate one boundary loop, wound to match the surrounding mesh."""
    v = np.asarray(vertices, dtype=np.float64)
    q, _ = _plane_coords(v[loop])

    ids = list(loop)
    # Ear clipping needs a known handedness to tell convex from reflex; the
    # walk's direction is arbitrary, so normalise to CCW in this basis first.
    area2 = sum(_cross2(q[0], q[i], q[i + 1]) for i in range(1, len(ids) - 1))
    if area2 < 0:
        ids = ids[::-1]
        q = q[::-1]

    tris = _ear_clip(q, ids, existing)
    if not tris:
        return []

    # Align to the mesh. A boundary edge belongs to exactly one face, so the
    # fill triangle sharing it must traverse it the OPPOSITE way — an exact,
    # per-edge rule. (Comparing the loop's best-fit-plane normal to an adjacent
    # face normal instead looks equivalent but is unreliable: a real tear hole
    # is not planar, and its fitted normal can sit at any angle to the face.)
    # Ear clipping preserves the polygon's own direction along its boundary
    # edges, so one lookup settles every triangle.
    for i in range(len(ids)):
        a, b = ids[i], ids[(i + 1) % len(ids)]
        if (a, b) in fo:  # mesh already traverses a→b; the fill must go b→a
            return [(t[0], t[2], t[1]) for t in tris]
        if (b, a) in fo:  # mesh traverses b→a; the fill's a→b is correct
            return tris
    return tris


def fill_interior_holes(
    faces: np.ndarray,
    *,
    max_hole_edges: int = 64,
    vertices: np.ndarray | None = None,
    view_matrix: np.ndarray | None = None,
    depth_near_m: float = 0.0,
    depth_far_m: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Triangulate qualifying interior boundary loops; return (new_faces, filled_edge_counts).

    A loop is filled iff (a) it is not the single largest loop when no depth
    window is active, (b) its edge count < ``max_hole_edges``, and (c) when a
    depth window is given (far bound > 0) every one of its boundary vertices'
    forward depth lies within ``[depth_near_m, depth_far_m]``. Triangulation
    uses only the loop's existing vertices — no new vertices, so 1:1 vertex-UV
    is preserved (UVs for the existing indices already exist).

    ``vertices`` is required to fill: a correct triangulation (non-convex,
    correctly wound, sliver-free) is not decidable from connectivity alone.
    """
    f = np.asarray(faces)
    be = boundary_edges(f)
    if len(be) == 0 or vertices is None:
        return f, []
    loops = walk_loops(be, faces=f)
    if not loops:
        return f, []

    depth_filter = (
        view_matrix is not None
        and depth_far_m > 0.0
    )
    depths: dict[int, np.ndarray] = {}
    if depth_filter:
        for k, loop in enumerate(loops):
            depths[k] = _loop_forward_depths(loop, vertices, view_matrix)

    fo = _directed_face_map(f)
    existing = {(int(a), int(b)) for a, b in
                np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]],
                                             f[:, [2, 0]]]), axis=1), axis=0)}
    outer = max(range(len(loops)), key=lambda i: len(loops[i]))  # largest = frame
    new_faces: list[tuple[int, int, int]] = []
    filled: list[int] = []
    for k, loop in enumerate(loops):
        n = len(loop)
        if depth_filter:
            # All boundary verts must lie inside the band-box depth window.
            d = depths[k]
            if not (np.all(d >= depth_near_m) and np.all(d <= depth_far_m)):
                continue
        else:
            # No spatial scope → always leave the single largest loop open as
            # the outer-frame guard.
            if k == outer:
                continue
        if n >= max_hole_edges:
            continue
        tris = _triangulate_loop(loop, vertices, fo, existing)
        if not tris:
            continue  # untriangulable (self-intersecting) → leave it open
        new_faces.extend(tris)
        filled.append(n)

    if new_faces:
        f = np.vstack([f, np.asarray(new_faces, dtype=f.dtype)])
    return f, filled


def apply_interior_hole_fill(
    mesh: Any,
    *,
    max_hole_edges: int = 64,
    view_matrix: np.ndarray | None = None,
    depth_near_m: float = 0.0,
    depth_far_m: float = 0.0,
) -> tuple[int, list[int]]:
    """Apply :func:`fill_interior_holes` to a ``ReliefMesh`` in place.

    Returns (n_loops_filled, filled_edge_counts) for the node's report. No-op
    (0, []) when ``max_hole_edges <= 0`` or the mesh carries no faces / no
    boundary (already watertight).
    """
    if max_hole_edges <= 0:
        return 0, []
    faces = getattr(mesh, "faces", None)
    if faces is None or len(faces) == 0:
        return 0, []
    vertices = getattr(mesh, "vertices", None)
    new_faces, filled = fill_interior_holes(
        faces,
        max_hole_edges=int(max_hole_edges),
        vertices=vertices,
        view_matrix=view_matrix,
        depth_near_m=float(depth_near_m),
        depth_far_m=float(depth_far_m),
    )
    if filled:
        mesh.faces = np.asarray(new_faces, dtype=faces.dtype)
    return len(filled), filled

def fill_boundary_sawteeth(
    faces: np.ndarray,
    *,
    vertices: np.ndarray | None = None,
    view_matrix: np.ndarray | None = None,
    depth_far_m: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """Bridge sawtooth notches along boundary loops.

    A "valley" vertex is a STRICT local depth maximum on a boundary loop (both
    neighbours are closer to the camera). Connecting the two neighbours across
    the valley caps the notch with a single triangle from existing vertices,
    preserving UVs. Passes repeat until none qualifies, so consecutive teeth
    smooth progressively. Strictness means two adjacent vertices can never both
    be valleys, so bridges within a pass cannot conflict.

    ``depth_far_m`` scopes the pass: only valleys whose forward depth is
    ``<=`` the bound are bridged. ``0.0`` disables the distance filter.

    Winding is the same exact per-edge rule as :func:`_triangulate_loop`, not a
    geometric test: the pivot walk emits loops in face-winding order, so the
    mesh already traverses ``prev→v`` and ``v→nxt`` — the cap must traverse
    both shared edges the OPPOSITE way, which forces ``(nxt, v, prev)``.
    (An orientation heuristic that can emit ``(prev, v, nxt)`` would duplicate
    those directed edges and break the pivot walk's own invariant.)

    Returns ``(new_faces, valley_depths)`` where ``new_faces`` is the face
    array plus any added triangles and ``valley_depths`` lists each bridged
    valley's forward depth.
    """
    f = np.asarray(faces)
    if vertices is None or len(f) == 0 or view_matrix is None:
        return f, []
    verts = np.asarray(vertices, dtype=np.float64)

    existing = {
        tuple(sorted(map(int, e)))
        for e in np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    }

    valley_depths: list[float] = []
    faces_work = f
    max_passes = 2  # 2 passes is sufficient to bridge primary and secondary staircase corners without expensive graph re-traversals

    for _ in range(max_passes):
        be = boundary_edges(faces_work)
        if len(be) == 0:
            break
        added: list[tuple[int, int, int]] = []
        for loop in walk_loops(be, faces=faces_work):


            n = len(loop)
            if n < 4:
                continue
            d = _loop_forward_depths(loop, verts, view_matrix)
            for i in range(n):
                prev, v, nxt = loop[i - 1], loop[i], loop[(i + 1) % n]
                d_v = d[i]
                if depth_far_m > 0.0 and d_v > depth_far_m:
                    continue
                edge = tuple(sorted((int(prev), int(nxt))))
                if edge in existing:
                    continue  # bridging would put that edge in three faces
                a, b, c = verts[prev], verts[v], verts[nxt]
                cross_vec = np.cross(b - a, c - a)
                if np.linalg.norm(cross_vec) < _EPS:
                    continue  # collinear → zero-area sliver

                is_valley = (d_v > d[i - 1] and d_v > d[(i + 1) % n])
                v1, v2 = a - b, c - b
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                is_equal_depth_step = (
                    abs(d_v - d[i - 1]) < 1e-3 and abs(d_v - d[(i + 1) % n]) < 1e-3
                    and n1 > _EPS and n2 > _EPS and abs(float(np.dot(v1, v2) / (n1 * n2))) < 0.707
                )
                if not (is_valley or is_equal_depth_step):
                    continue


                added.append((int(nxt), int(v), int(prev)))
                valley_depths.append(float(d_v))
                existing.add(edge)

        if not added:
            break
        faces_work = np.vstack([faces_work, np.asarray(added, dtype=f.dtype)])
    return faces_work, valley_depths


def apply_boundary_sawtooth_fill(
    mesh: Any,
    *,
    view_matrix: np.ndarray | None = None,
    depth_far_m: float = 0.0,
) -> tuple[int, list[float]]:
    """Apply :func:`fill_boundary_sawteeth` to a ``ReliefMesh`` in place.

    Returns ``(n_triangles_added, valley_depths)`` for the node's report.
    """
    faces = getattr(mesh, "faces", None)
    vertices = getattr(mesh, "vertices", None)
    if faces is None or len(faces) == 0 or vertices is None:
        return 0, []
    new_faces, depths = fill_boundary_sawteeth(
        faces,
        vertices=vertices,
        view_matrix=view_matrix,
        depth_far_m=float(depth_far_m),
    )
    n_added = len(new_faces) - len(faces)
    if n_added:
        mesh.faces = np.asarray(new_faces, dtype=faces.dtype)
    return n_added, depths


def repair_relief_mesh_grid_cuda(
    mesh: Any,
    *,
    view_matrix: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    fill_holes: bool = True,
    fill_sawteeth: bool = True,
    depth_far_m: float = 0.0,
    depth_edge_rel: float = 0.5,
    max_edge_factor: float = 12.0,
    max_hole_edges: int = 64,
    cap_enclosed: bool = False,
) -> tuple[int, int]:
    """GPU grid hole-fill / boundary-sawtooth on an already-built relief mesh.

    ``build_relief_mesh`` runs its sub-millisecond PyTorch/CUDA 2D grid repair
    (:func:`atlas_camera.core.relief_mesh.repair_relief_grid_cuda`) on the depth
    GRID it is triangulating from. A relief mesh stored on a solve has been
    compacted to referenced vertices only, so that grid is gone — which is why
    the standalone ``AtlasLiveMeshRepair`` node's CPU path uses the numpy
    face-soup :func:`apply_interior_hole_fill` / :func:`apply_boundary_sawtooth_fill`
    instead.

    This recovers the regular sampling lattice from the mesh's own UVs (they are
    a ``meshgrid`` of the sampled rows/cols, so ``np.unique`` on each axis
    reconstructs the grid exactly), rebuilds the ``(nr, nc)`` occupancy +
    forward-depth grids, runs the SAME convolutional kernel on the GPU, then
    materializes every newly-valid cell as a new vertex — back-projected along
    that cell's own camera ray at the neighbour-averaged forward distance, the
    identical ray-preserving construction ``build_relief_mesh`` uses, so a new
    vertex lands consistently with the existing ones — and adds the two
    triangles per closed quad, matching ``build_relief_mesh``'s winding. Existing
    vertices and faces are never moved.

    Mutates ``mesh.vertices`` / ``mesh.uvs`` / ``mesh.faces`` (and
    ``mesh.edge_risk`` when present) in place. Returns
    ``(n_holes_filled, n_sawteeth_filled)``; ``(0, 0)`` when the mesh has no
    faces, the UV lattice can't be recovered, or nothing qualified.
    """
    from atlas_camera.core.relief_mesh import repair_relief_grid_cuda

    if not (fill_holes or fill_sawteeth):
        return 0, 0
    verts = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    uvs = getattr(mesh, "uvs", None)
    if verts is None or faces is None or uvs is None:
        return 0, 0
    verts = np.asarray(verts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if len(verts) == 0 or len(faces) == 0 or len(uvs) != len(verts):
        return 0, 0

    # Recover the sampling lattice from the UVs (rounded to serialization
    # precision so float noise never splits one lattice line into two).
    uq = np.round(uvs[:, 0], 5)
    vq = np.round(uvs[:, 1], 5)
    uu = np.unique(uq)   # ascending column positions
    vv = np.unique(vq)   # ascending row positions (v grows as row index shrinks)
    nc, nr = int(len(uu)), int(len(vv))
    if nr < 2 or nc < 2 or nr * nc > 16_000_000:
        return 0, 0

    col_of = np.searchsorted(uu, uq)
    # v = 1 - row/(H-1): larger v is a SMALLER row index, so flip the axis.
    row_of = (nr - 1) - np.searchsorted(vv, vq)

    # Camera pose (row-major, column-vector points) — the exact convention
    # build_relief_mesh uses for back-projection and its band/near clips.
    vm = np.asarray(view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    R_cw = c2w[:3, :3]
    cam = c2w[:3, 3]
    fwd = -((verts - cam) @ R_cw[:, 2])  # forward distance (metres), per vertex

    vgrid = np.zeros((nr, nc), dtype=bool)
    dgrid = np.zeros((nr, nc), dtype=np.float64)
    cell2vidx = np.full((nr, nc), -1, dtype=np.int64)
    vgrid[row_of, col_of] = True
    dgrid[row_of, col_of] = fwd
    cell2vidx[row_of, col_of] = np.arange(len(verts))

    def _shift(m, dr, dc, fill=False):
        out = np.full_like(m, fill)
        rs_dst = slice(max(dr, 0), nr + min(dr, 0))
        cs_dst = slice(max(dc, 0), nc + min(dc, 0))
        rs_src = slice(max(-dr, 0), nr + min(-dr, 0))
        cs_src = slice(max(-dc, 0), nc + min(-dc, 0))
        out[rs_dst, cs_dst] = m[rs_src, cs_src]
        return out

    # ZBrush-style "Close Holes" for ENCLOSED interior loops (cap_enclosed):
    # a hole fully surrounded by mesh is exactly what Close Holes should cap,
    # yet it routinely spans a real depth jump (front barrel vs machinery
    # behind), which the ratio tear gate below would veto. So: flood-fill the
    # invalid cells from the grid border — whatever the flood can't reach is an
    # ENCLOSED hole — and fill those cells at the MAX (farthest) valid-neighbour
    # depth, i.e. the fill continues the BACK surface away from the camera
    # rather than hanging mid-air at the mean (the artist's "offset away from
    # camera along the surface" rule). The open silhouette/frame boundary is by
    # construction border-connected, so it can never be capped.
    capfill = np.zeros((nr, nc), dtype=bool)
    if cap_enclosed and fill_holes:
        invalid = ~vgrid
        outside = invalid.copy()
        interior = np.zeros((nr, nc), dtype=bool)
        interior[1:-1, 1:-1] = True
        outside &= ~interior  # seed: border-row/col invalid cells
        for _ in range(nr * nc):
            grown = outside
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                grown = grown | _shift(outside, dr, dc)
            grown &= invalid
            if (grown == outside).all():
                break
            outside = grown
        remaining = invalid & ~outside
        # Ring-by-ring max-depth fill: each pass fills enclosed cells that have
        # at least one valid 4-neighbour, at the farthest such depth. Bounded by
        # the hole radius, hard-capped at the grid diameter.
        for _ in range(nr + nc):
            if not remaining.any():
                break
            nb_max = np.full((nr, nc), -np.inf)
            nb_any = np.zeros((nr, nc), dtype=bool)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sv = _shift(vgrid | capfill, dr, dc)
                sd = _shift(dgrid, dr, dc)
                nb_max = np.where(sv, np.maximum(nb_max, sd), nb_max)
                nb_any |= sv
            ring = remaining & nb_any
            if not ring.any():
                break
            dgrid = np.where(ring, nb_max, dgrid)
            capfill |= ring
            remaining &= ~ring
        vgrid_conv = vgrid | capfill
        dgrid_conv = dgrid
    else:
        vgrid_conv = vgrid
        dgrid_conv = dgrid

    # Iterate the conv fill: one pass closes only the innermost ring of any
    # concavity (a cell needs >=3 valid orthogonal neighbours), so a wider notch
    # needs more passes. max_hole_edges ~= a loop perimeter, and a hole of E
    # boundary edges is ~E/4 cells across / ~E/8 rings to close from both sides,
    # so scale the pass budget off it. Early-out the instant a pass fills nothing
    # (the boundary is now convex — straight/convex edges never satisfy the >=3
    # rule, so this cannot balloon the mesh outward and always terminates).
    n_iter = max(1, min(int(max_hole_edges) // 2, 512))
    d_out, v_out = dgrid_conv, vgrid_conv
    n_saw = 0
    n_hole = int(capfill.sum())
    for _ in range(n_iter):
        d_out, v_out, ns, nh = repair_relief_grid_cuda(
            d_out, v_out, fill_sawteeth=bool(fill_sawteeth), fill_holes=bool(fill_holes))
        n_saw += ns
        n_hole += nh
        if ns == 0 and nh == 0:
            break
    newfill = v_out & ~vgrid
    if depth_far_m > 0.0:
        newfill &= (d_out <= float(depth_far_m))
    if not newfill.any():
        return 0, 0

    # New vertices: back-project each filled cell along its own camera ray at the
    # averaged forward distance. Pixel per lattice axis (UV -> source pixel).
    px_axis = uu * max(image_width - 1, 1)
    py_axis = (1.0 - vv) * max(image_height - 1, 1)
    dir_x = (px_axis[None, :] - cx) / fx                       # (1, nc)
    dir_y = -(py_axis[:, None] - cy) / fy                      # (nr, 1)
    dir_cam = np.stack([
        np.broadcast_to(dir_x, (nr, nc)),
        np.broadcast_to(dir_y, (nr, nc)),
        np.full((nr, nc), -1.0),
    ], axis=-1)
    world_dir = dir_cam @ R_cw.T
    new_rows, new_cols = np.where(newfill)
    new_pos = cam + d_out[new_rows, new_cols, None] * world_dir[new_rows, new_cols]
    new_uv = np.stack([uu[new_cols], vv[new_rows]], axis=1)

    n_verts0 = len(verts)
    cell2vidx[new_rows, new_cols] = n_verts0 + np.arange(len(new_rows))

    # World position per cell: back-projected for filled cells, the vertex's
    # ACTUAL stored position for existing cells (respects build's floor/band
    # clamps). Needed for the same edge-length tear test build_relief_mesh runs.
    pos_grid = cam + d_out[..., None] * world_dir
    pos_grid[row_of, col_of] = verts  # exact existing positions

    # Closed quads that touch a newly-filled cell (all four corners now usable).
    usable = vgrid | newfill
    i00, i01 = cell2vidx[:-1, :-1], cell2vidx[:-1, 1:]
    i10, i11 = cell2vidx[1:, :-1], cell2vidx[1:, 1:]
    u00, u01 = usable[:-1, :-1], usable[:-1, 1:]
    u10, u11 = usable[1:, :-1], usable[1:, 1:]
    f00, f01 = newfill[:-1, :-1], newfill[:-1, 1:]
    f10, f11 = newfill[1:, :-1], newfill[1:, 1:]
    quad = (u00 & u01 & u10 & u11) & (f00 | f01 | f10 | f11)

    # Tear gate — a filled cell adjacent to a silhouette has valid neighbours on
    # both sides of a near→far depth jump; without this the fill bridges that
    # jump into a stretched vertical shard. Reject a quad whose corner forward
    # depths disagree by more than depth_edge_rel, or whose world edges exceed
    # max_edge_factor × the local sample spacing (the exact tests _tri_ok uses).
    dq = np.stack([d_out[:-1, :-1], d_out[:-1, 1:], d_out[1:, :-1], d_out[1:, 1:]], axis=-1)
    dmax = dq.max(axis=-1)
    dmin = np.maximum(dq.min(axis=-1), 1e-6)
    ratio_ok = (dmax / dmin - 1.0) <= float(depth_edge_rel)
    # cap_enclosed exemption: a quad touching a cap-filled cell IS the deliberate
    # front-to-back wall that closes an enclosed hole — the gates exist to keep
    # OPEN silhouettes open, and an enclosed loop is by definition not one.
    quad_cap = (capfill[:-1, :-1] | capfill[:-1, 1:]
                | capfill[1:, :-1] | capfill[1:, 1:])
    if max_edge_factor and nc > 1:
        step_px = float(np.median(np.diff(px_axis))) if nc > 1 else 1.0
        budget = float(max_edge_factor) * np.median(dq, axis=-1) * abs(step_px) / max(min(fx, fy), 1e-6)
        budget = np.maximum(budget, 0.05)
        P = pos_grid
        p00, p01 = P[:-1, :-1], P[:-1, 1:]
        p10, p11 = P[1:, :-1], P[1:, 1:]
        def _elen(a, b):
            return np.linalg.norm(a - b, axis=-1)
        edge_ok = ((_elen(p00, p01) <= budget) & (_elen(p00, p10) <= budget)
                   & (_elen(p01, p11) <= budget) & (_elen(p10, p11) <= budget)
                   & (_elen(p10, p01) <= budget))  # shared diagonal
        quad = quad & ((ratio_ok & edge_ok) | quad_cap)
    else:
        quad = quad & (ratio_ok | quad_cap)

    tri_a = np.stack([i00[quad], i10[quad], i01[quad]], axis=1)
    tri_b = np.stack([i10[quad], i11[quad], i01[quad]], axis=1)
    add_faces = np.concatenate([tri_a, tri_b], axis=0)
    if len(add_faces) == 0:
        return 0, 0

    faces_arr = np.asarray(mesh.faces)
    mesh.vertices = np.concatenate(
        [np.asarray(mesh.vertices, dtype=np.float32),
         new_pos.astype(np.float32)], axis=0)
    mesh.uvs = np.concatenate(
        [np.asarray(mesh.uvs, dtype=np.float32),
         new_uv.astype(np.float32)], axis=0)
    mesh.faces = np.concatenate(
        [faces_arr, add_faces.astype(faces_arr.dtype)], axis=0)
    er = getattr(mesh, "edge_risk", None)
    if er is not None and len(np.asarray(er)) == n_verts0:
        mesh.edge_risk = np.concatenate(
            [np.asarray(er, dtype=np.float32),
             np.ones(len(new_rows), dtype=np.float32)], axis=0)
    return int(n_hole), int(n_saw)
