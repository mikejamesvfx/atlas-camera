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
