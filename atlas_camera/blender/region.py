"""Pick the torn collar to send. Policy is Atlas's; Blender only does geometry.

Sending the whole mesh would be wrong twice over. It is slow — voxel-remeshing
780K vertices takes minutes where a 2K-face collar takes milliseconds — and it
is destructive: measured live, a whole-mesh voxel remesh closed the PLATE
PERIMETER too (974 boundary edges -> 0), turning a matte painting into a
watertight solid. Operating on a collar and keeping the outer ring untouched is
what stops that, and it also bounds the blast radius: whatever comes back can
only replace the collar.

Loop finding reuses `core.mesh_repair`'s tested, pinch-vertex-safe walk rather
than a second implementation — including `_perimeter_loops`, which is what tells
a real tear apart from the plate silhouette.
"""
from __future__ import annotations

from typing import Any

from atlas_camera.core.mesh_repair import (
    _perimeter_loops,
    boundary_edges,
    walk_loops,
)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "atlas_camera.blender requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def _vertex_adjacency(np: Any, faces: Any, n_vertices: int) -> list[set]:
    adj: list[set] = [set() for _ in range(n_vertices)]
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for a, b, c in f:
        adj[a].update((int(b), int(c)))
        adj[b].update((int(a), int(c)))
        adj[c].update((int(a), int(b)))
    return adj


def select_torn_collar(vertices: Any, faces: Any, uvs: Any = None, *,
                       max_hole_edges: int = 384, rings: int = 4,
                       image_width: int | None = None,
                       image_height: int | None = None) -> dict[str, Any]:
    """Faces around every interior tear, split into a patch and a weld anchor.

    Returns ``{tear_loops, patch_faces, target_faces, anchor_vertices,
    collar_vertices, skipped_perimeter, skipped_too_large}`` — face INDEX arrays
    into the caller's own array, so nothing is copied or reordered here.

    ``patch_faces`` deliberately EXCLUDES the outermost ring. That ring stays in
    the untouched mesh and is what the returned geometry welds onto; without a
    preserved anchor the seam has nothing to attach to and reappears as a new
    tear the next time boundary_edges runs.
    """
    np = _require_numpy()
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    loops = walk_loops(boundary_edges(f), faces=f)
    perim = _perimeter_loops(loops, uvs, image_width, image_height)

    tear_loops, too_large = [], 0
    for k, loop in enumerate(loops):
        if k in perim:
            continue
        if len(loop) > int(max_hole_edges):
            too_large += 1
            continue
        tear_loops.append(loop)

    empty = np.zeros(0, dtype=np.int64)
    if not tear_loops:
        return {"tear_loops": [], "patch_faces": empty, "target_faces": empty,
                "anchor_vertices": empty, "collar_vertices": empty,
                "skipped_perimeter": len(perim), "skipped_too_large": too_large}

    rings = max(1, int(rings))
    adj = _vertex_adjacency(np, f, len(verts))

    # BFS outward from the tear rims, recording which ring each vertex landed in
    # so the outermost can be held back as the anchor.
    ring_of: dict[int, int] = {}
    frontier = {int(v) for loop in tear_loops for v in loop}
    for v in frontier:
        ring_of[v] = 0
    for r in range(1, rings + 1):
        nxt: set = set()
        for v in frontier:
            nxt.update(adj[v])
        nxt = {v for v in nxt if v not in ring_of}
        for v in nxt:
            ring_of[v] = r
        if not nxt:
            break
        frontier = nxt

    collar = np.fromiter(ring_of.keys(), dtype=np.int64, count=len(ring_of))
    inner = np.fromiter((v for v, r in ring_of.items() if r < rings),
                        dtype=np.int64)
    anchor = np.fromiter((v for v, r in ring_of.items() if r >= rings),
                         dtype=np.int64)

    in_inner = np.zeros(len(verts), dtype=bool)
    in_inner[inner] = True
    in_collar = np.zeros(len(verts), dtype=bool)
    in_collar[collar] = True

    patch_mask = in_inner[f].all(axis=1)
    # The target is the collar plus a little more measured surface for the
    # shrinkwrap to actually land on — a patch can only snap to geometry it was
    # given, and snapping to itself is meaningless.
    target_mask = in_collar[f].any(axis=1)

    return {
        "tear_loops": tear_loops,
        "patch_faces": np.nonzero(patch_mask)[0],
        "target_faces": np.nonzero(target_mask & ~patch_mask)[0],
        "anchor_vertices": anchor,
        "collar_vertices": collar,
        "skipped_perimeter": len(perim),
        "skipped_too_large": too_large,
    }


def compact(vertices: Any, faces: Any, face_indices: Any) -> tuple:
    """(verts, faces, index_map) for a face subset, reindexed from zero.

    ``index_map`` maps new vertex index -> original, which is what lets the
    result be welded back onto vertices that never left Atlas.
    """
    np = _require_numpy()
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)[np.asarray(face_indices)]
    if not len(f):
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    used = np.unique(f)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[f], used
