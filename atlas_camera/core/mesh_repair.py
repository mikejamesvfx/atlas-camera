"""Mesh-topology interior-hole fill — pure numpy, no external deps.

Atlas relief meshes are torn at depth discontinuities (silhouettes, sky, band
clips) — deliberate, so the live 📽 projection never rubber-sheets background
onto foreground. But the EXPORTED OBJ/GLB handed to a DCC benefits from being
closer to watertight *inside the subject*: small interior tear holes (noise,
fine structure, band-clip seams) become stray open boundaries that block retopo
/ boolean ops / 3D-print prep in Maya/ZBrush/Blender.

This fills ONLY interior enclosed boundary loops — never the outer silhouette /
frame boundary — by walking closed boundary-edge loops and fan-triangulating
each qualifying loop FROM ITS EXISTING VERTICES (no new verts → the 1:1
vertex-UV mapping the OBJ/GLB writers depend on stays exact, so projection-baked
UVs remain valid).

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


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Edges that appear in exactly one triangle = the open boundary."""
    f = np.asarray(faces)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)  # canonical (lo, hi) so orientation doesn't matter
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def walk_loops(bedges: np.ndarray) -> list[list[int]]:
    """Walk every closed boundary loop.

    Each boundary vertex has degree 2 on the boundary graph, so from any
    unvisited start we follow the non-previous neighbour until we return to the
    start (a closed loop). A mid-path revisit before closing means the loop
    isn't a simple cycle (shouldn't happen on a manifold-adjacent tear mesh,
    but we bail rather than spin). Vertices are only marked done AFTER a loop
    closes, so the start-vertex check for cycle re-entry stays correct.
    """
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


def fill_interior_holes(
    faces: np.ndarray,
    *,
    max_hole_edges: int = 64,
    vertices: np.ndarray | None = None,
    view_matrix: np.ndarray | None = None,
    depth_near_m: float = 0.0,
    depth_far_m: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Fan-fill qualifying interior boundary loops; return (new_faces, filled_edge_counts).

    A loop is filled iff (a) it is not the single largest loop when no depth
    window is active, (b) its edge count < ``max_hole_edges``, and (c) when a
    depth window is given (both bounds > 0) every one of its boundary vertices'
    forward depth lies within ``[depth_near_m, depth_far_m]``. Fan-triangulation
    uses only the loop's existing vertices — no new vertices, so 1:1
    vertex-UV is preserved (UVs for the existing indices already exist).
    """
    f = np.asarray(faces)
    be = boundary_edges(f)
    if len(be) == 0:
        return f, []
    loops = walk_loops(be)
    if not loops:
        return f, []

    depth_filter = (
        view_matrix is not None
        and vertices is not None
        and depth_near_m > 0.0
        and depth_far_m > 0.0
    )
    depths: dict[int, np.ndarray] = {}
    if depth_filter:
        for k, loop in enumerate(loops):
            depths[k] = _loop_forward_depths(loop, vertices, view_matrix)

    outer = max(range(len(loops)), key=lambda i: len(loops[i]))  # largest = frame
    new_faces: list[list[int]] = []
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
        for i in range(1, n - 1):
            new_faces.append([loop[0], loop[i], loop[i + 1]])
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