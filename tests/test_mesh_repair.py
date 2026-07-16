"""Tests for interior-hole-fill on exported relief meshes (core/mesh_repair.py).

A torn relief mesh has two kinds of open boundary loops: the deliberate outer
silhouette/frame boundary (must stay open — DMP-correct) and small interior
tear holes (export-only fill for clean DCC handoff). These pin the selective
fill: edge-count threshold + largest-loop guard + band-box depth window.
"""

import numpy as np
import pytest

from atlas_camera.core.mesh_repair import (
    apply_interior_hole_fill,
    boundary_edges,
    fill_interior_holes,
    walk_loops,
)


def _grid_mesh(z=-10.0, z_outer=None):
    """4x4 flat grid, two triangles per quad, with the CENTER quad (1,1)
    removed so its 4 edges form an interior hole loop and the grid perimeter
    forms the outer frame loop. ``z_outer`` (if given) places the PERIMETER
    vertices at a different depth than the interior — mirroring a real relief
    mesh whose frame spans near-to-far while an interior tear sits at one
    depth. Vertices are 1:1 with uvs (uv == xy)."""
    z_out = z if z_outer is None else z_outer

    def _z(r, c, n=4):
        on_perim = (r in (0, n - 1)) or (c in (0, n - 1))
        return z_out if on_perim else z

    verts = [[float(c), float(r), _z(r, c)] for r in range(4) for c in range(4)]
    vertices = np.asarray(verts, dtype=np.float64)
    uvs = vertices[:, :2].copy()

    faces = []
    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:  # delete center quad → interior hole
                continue
            a = r * 4 + c
            b = r * 4 + c + 1
            d = (r + 1) * 4 + c
            e = (r + 1) * 4 + c + 1
            faces.append([a, b, e])
            faces.append([a, e, d])
    return vertices, uvs, np.asarray(faces, dtype=np.int64)


def _identity_view():
    return np.eye(4, dtype=np.float64)


class _Mesh:
    """Minimal stand-in for ReliefMesh (vertices/faces/uvs)."""

    def __init__(self, vertices, uvs, faces):
        self.vertices = vertices
        self.uvs = uvs
        self.faces = faces


# ---------------------------------------------------------------------------
# boundary discovery
# ---------------------------------------------------------------------------

def test_boundary_loops_found():
    _, _, faces = _grid_mesh()
    be = boundary_edges(faces)
    assert len(be) == 16  # perimeter (12) + hole (4)
    loops = walk_loops(be)
    assert sorted(len(l) for l in loops) == [4, 12]


# ---------------------------------------------------------------------------
# threshold-only mode (no depth window): largest loop always left open
# ---------------------------------------------------------------------------

def test_fill_interior_keeps_outer_frame():
    _, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64)
    assert filled == [4]
    assert len(new_faces) == len(faces) + 2  # fan of a 4-loop = 2 triangles
    assert len(boundary_edges(new_faces)) == 12  # only the frame remains open


def test_threshold_blocks_hole():
    _, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=3)
    assert filled == []  # 4-edge hole >= 3 → not filled
    assert np.array_equal(new_faces, faces)


def test_largest_loop_guard_even_above_threshold():
    """Raise the threshold so the 12-edge frame would normally be eligible;
    the largest-loop guard must still leave it open in threshold-only mode."""
    _, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=4096)
    assert filled == [4]  # frame skipped as the single largest loop
    assert len(boundary_edges(new_faces)) == 12


# ---------------------------------------------------------------------------
# band-box depth window (the user's proposal: scope fill by the band cutoff)
# ---------------------------------------------------------------------------

def test_depth_window_includes_hole_excludes_frame():
    """Frame perimeter at depth 30, interior hole at depth 10. A [5,15]
    window admits the hole but excludes the frame → only the hole fills."""
    vertices, _, faces = _grid_mesh(z=-10.0, z_outer=-30.0)
    new_faces, filled = fill_interior_holes(
        faces, max_hole_edges=64,
        vertices=vertices, view_matrix=_identity_view(),
        depth_near_m=5.0, depth_far_m=15.0,
    )
    assert filled == [4]
    assert len(boundary_edges(new_faces)) == 12  # frame still open


def test_depth_window_can_admit_frame_when_set_to_it():
    """Same fixture, window [25,35] admits the frame but excludes the hole.
    Window mode has no largest-loop guard, so the frame fills here — this is
    the artist's explicit choice and the reason the window IS the scope."""
    vertices, _, faces = _grid_mesh(z=-10.0, z_outer=-30.0)
    new_faces, filled = fill_interior_holes(
        faces, max_hole_edges=4096,
        vertices=vertices, view_matrix=_identity_view(),
        depth_near_m=25.0, depth_far_m=35.0,
    )
    assert filled == [12]  # only the frame (12-edge); hole at depth 10 excluded


def test_depth_window_excludes_both_fills_nothing():
    vertices, _, faces = _grid_mesh(z=-10.0, z_outer=-30.0)
    new_faces, filled = fill_interior_holes(
        faces, max_hole_edges=64,
        vertices=vertices, view_matrix=_identity_view(),
        depth_near_m=50.0, depth_far_m=60.0,
    )
    assert filled == []
    assert np.array_equal(new_faces, faces)


def test_depth_window_zero_falls_back_to_threshold_only():
    """Both bounds 0 → depth filter disabled → largest-loop guard back on."""
    vertices, _, faces = _grid_mesh(z=-10.0, z_outer=-30.0)
    new_faces, filled = fill_interior_holes(
        faces, max_hole_edges=4096,
        vertices=vertices, view_matrix=_identity_view(),
        depth_near_m=0.0, depth_far_m=0.0,
    )
    assert filled == [4]  # frame skipped by the largest-loop guard


# ---------------------------------------------------------------------------
# apply_interior_hole_fill (node-facing)
# ---------------------------------------------------------------------------

def test_apply_to_mesh_in_place():
    vertices, uvs, faces = _grid_mesh()
    mesh = _Mesh(vertices, uvs, faces)
    n_before = len(mesh.faces)
    n_loops, filled = apply_interior_hole_fill(mesh, max_hole_edges=64)
    assert n_loops == 1 and filled == [4]
    assert len(mesh.faces) == n_before + 2
    # uvs untouched (no new vertices) → 1:1 vertex-uv preserved
    assert mesh.uvs is uvs and len(mesh.uvs) == len(mesh.vertices)


def test_apply_noop_when_disabled():
    vertices, uvs, faces = _grid_mesh()
    mesh = _Mesh(vertices, uvs, faces)
    n = len(mesh.faces)
    assert apply_interior_hole_fill(mesh, max_hole_edges=0) == (0, [])
    assert len(mesh.faces) == n  # disabled → mesh untouched, hole still present
    # idempotent: fill once, then a second pass finds no open loops
    apply_interior_hole_fill(mesh, max_hole_edges=64)
    assert apply_interior_hole_fill(mesh, max_hole_edges=64) == (0, [])


def test_watertight_mesh_is_noop():
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64)
    assert filled == []
    assert np.array_equal(new_faces, faces)


def test_trimesh_validates_fill():
    """Optional trimesh round-trip: after the fill the mesh is a topological
    disk (one open boundary loop = the frame), euler characteristic 1."""
    trimesh = pytest.importorskip("trimesh")
    vertices, _, faces = _grid_mesh()
    new_faces, _ = fill_interior_holes(faces, max_hole_edges=64)
    m = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=False)
    assert abs(m.euler_number - 1) < 1e-6  # disk
    # and filling both loops would close it to a sphere (χ=2). The flat grid
    # sits every vertex at depth 10, so a [5,15] window admits BOTH loops
    # (window mode has no largest-loop guard → the frame fills too).
    closed, _ = fill_interior_holes(faces, max_hole_edges=4096,
                                    vertices=vertices, view_matrix=_identity_view(),
                                    depth_near_m=5.0, depth_far_m=15.0)
    m2 = trimesh.Trimesh(vertices=vertices, faces=closed, process=False)
    assert abs(m2.euler_number - 2) < 1e-6