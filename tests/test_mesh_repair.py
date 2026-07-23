"""Tests for interior-hole-fill on exported relief meshes (core/mesh_repair.py).

A torn relief mesh has two kinds of open boundary loops: the deliberate outer
silhouette/frame boundary (must stay open — DMP-correct) and small interior
tear holes (export-only fill for clean DCC handoff). These pin the selective
fill: edge-count threshold + largest-loop guard + band-box depth window.
"""

import numpy as np
import pytest

from atlas_camera.core.mesh_repair import (
    apply_boundary_sawtooth_fill,
    apply_interior_hole_fill,
    boundary_edges,
    fill_boundary_sawteeth,
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


def _grid(n, drop, z=-10.0):
    """n x n vertex grid, 2 tris/quad, with the ``drop`` set of (r, c) quads
    removed. Lets a test carve an arbitrary tear footprint (L-shaped, pinched,
    …) rather than only the single convex quad ``_grid_mesh`` removes."""
    verts = np.asarray([[float(c), float(r), z]
                        for r in range(n) for c in range(n)], dtype=np.float64)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            if (r, c) in drop:
                continue
            a, b = r * n + c, r * n + c + 1
            d, e = (r + 1) * n + c, (r + 1) * n + c + 1
            faces.append([a, b, e])
            faces.append([a, e, d])
    return verts, np.asarray(faces, dtype=np.int64)


def _fan_areas(loop, vertices):
    """Signed areas of the fill triangles, in the loop's own best-fit plane."""
    pts = np.asarray(vertices, dtype=np.float64)[loop]
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    e1, e2 = vt[0], np.cross(vt[2], vt[0])
    q = np.stack([(pts - c) @ e1, (pts - c) @ e2], axis=1)
    return np.asarray([
        0.5 * ((q[i][0] - q[0][0]) * (q[i + 1][1] - q[0][1])
               - (q[i + 1][0] - q[0][0]) * (q[i][1] - q[0][1]))
        for i in range(1, len(loop) - 1)
    ])


def _added_face_areas(vertices, before, after):
    v = np.asarray(vertices, dtype=np.float64)
    return np.asarray([
        0.5 * np.linalg.norm(np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]]))
        for t in after[len(before):]
    ])


def _face_normals(faces, vertices):
    v = np.asarray(vertices, dtype=np.float64)
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln == 0, 1.0, ln)


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
    vertices, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64,
                                            vertices=vertices)
    assert filled == [4]
    assert len(new_faces) == len(faces) + 2  # a 4-loop triangulates to 2 tris
    assert len(boundary_edges(new_faces)) == 12  # only the frame remains open


def test_threshold_blocks_hole():
    vertices, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=3,
                                            vertices=vertices)
    assert filled == []  # 4-edge hole >= 3 → not filled
    assert np.array_equal(new_faces, faces)


def test_largest_loop_guard_even_above_threshold():
    """Raise the threshold so the 12-edge frame would normally be eligible;
    the largest-loop guard must still leave it open in threshold-only mode."""
    vertices, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=4096,
                                            vertices=vertices)
    assert filled == [4]  # frame skipped as the single largest loop
    assert len(boundary_edges(new_faces)) == 12


def test_no_vertices_fills_nothing():
    """A correct triangulation (non-convex, correctly wound, sliver-free) is
    not decidable from connectivity alone, so the fill needs vertices."""
    _, _, faces = _grid_mesh()
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64)
    assert filled == []
    assert np.array_equal(new_faces, faces)


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
    new_faces, _ = fill_interior_holes(faces, max_hole_edges=64,
                                       vertices=vertices)
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

# ---------------------------------------------------------------------------
# fill QUALITY — a fill must not introduce the very defects (flipped winding,
# zero-area faces, triangles outside the hole) that block the retopo/boolean/
# 3D-print prep this feature exists to enable.
# ---------------------------------------------------------------------------

# An L-shaped (non-convex) tear whose walk anchors on a REFLEX vertex — the
# case a blind fan from loop[0] cannot triangulate. Covers 3 unit quads.
_L_TEAR = {(1, 1), (1, 2), (2, 2)}


def test_nonconvex_hole_fills_inside_the_hole_only():
    """An L-shaped tear must be triangulated INSIDE its own footprint.

    A blind fan from loop[0] is only valid when the loop is star-shaped from
    that vertex; walk order is arbitrary, so a reflex anchor emits triangles
    that spill outside the hole over real geometry (mixed winding).
    """
    verts, faces = _grid(6, _L_TEAR)
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64,
                                            vertices=verts)
    assert filled == [8]

    loops = walk_loops(boundary_edges(faces), faces=faces)
    loop = next(l for l in loops if len(l) == 8)
    areas = _fan_areas(loop, verts)
    # Same sign throughout => every triangle lies inside the loop.
    assert not (areas.min() < -1e-9 and areas.max() > 1e-9), (
        "fill emitted triangles OUTSIDE the hole (mixed winding)")
    # The L covers 3 unit quads; the fill must cover exactly that, once.
    assert np.isclose(np.abs(areas).sum(), 3.0), (
        f"fill covers {np.abs(areas).sum()} != hole area 3.0 (overlap)")


def test_fill_adds_no_degenerate_faces():
    """Grid-aligned tears have collinear runs; fanning across them makes
    zero-area triangles, which are exactly what breaks DCC retopo."""
    verts, faces = _grid(6, _L_TEAR)
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64,
                                            vertices=verts)
    areas = _added_face_areas(verts, faces, new_faces)
    assert len(areas) > 0
    assert (areas > 1e-12).all(), (
        f"{int((areas <= 1e-12).sum())} degenerate (zero-area) fill faces")


def test_fill_winding_matches_surrounding_mesh():
    """Loop walk direction is decided by edge ordering, not by the adjacent
    face winding, so the fill's normals must be explicitly aligned."""
    verts, faces = _grid(6, _L_TEAR)
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64,
                                            vertices=verts)
    assert filled == [8]
    base = _face_normals(np.asarray(faces), verts)
    fill = _face_normals(np.asarray(new_faces[len(faces):]), verts)
    # Flat grid: every existing face points the same way; fills must agree.
    assert np.allclose(fill, base[0], atol=1e-6), (
        f"fill normals {np.unique(fill.round(3), axis=0)} != "
        f"mesh normal {base[0].round(3)} (back-facing fill)")


def test_pinch_vertex_does_not_drop_a_hole():
    """Two tears meeting at one grid vertex give it boundary degree 4. The
    walk must still recover BOTH holes, not wander between them and bail."""
    verts, faces = _grid(5, {(1, 1), (2, 2)})  # share vertex 12
    loops = walk_loops(boundary_edges(faces), faces=faces)
    assert sorted(len(l) for l in loops) == [4, 4, 16], (
        f"pinch dropped a loop: got {sorted(len(l) for l in loops)}")
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64,
                                            vertices=verts)
    assert sorted(filled) == [4, 4], f"only filled {filled}, expected both holes"


def _noisy_relief_mesh(seed=6, n=256):
    """A REAL torn relief mesh: subjects at different depths + depth noise, so
    tear holes are genuinely non-planar and non-convex — unlike the flat grids
    above, which every orientation heuristic passes by luck."""
    from atlas_camera.core.relief_mesh import build_relief_mesh
    rng = np.random.default_rng(seed)
    depth = np.full((n, n), 20.0)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(3):
        cx0, cy0 = rng.integers(50, 200, 2)
        r = rng.integers(20, 55)
        depth[(np.abs(xx - cx0) < r) & (np.abs(yy - cy0) < r)] = 6 + 12 * rng.random()
    depth += rng.normal(0, 0.45, (n, n))
    depth[(xx // 6 % 3 == 0)] += 0.9
    return build_relief_mesh(depth, view_matrix=np.eye(4), fx=250.0, fy=250.0,
                             cx=n / 2, cy=n / 2, grid_long_edge=128,
                             depth_edge_rel=0.5, apply_sky_heuristic=False)


def _max_directed_edge_count(faces):
    """In a consistently-wound manifold every directed edge occurs at most once.
    A back-facing fill traverses a shared edge the same way as its neighbour,
    so that edge shows up twice."""
    f = np.asarray(faces)
    d = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, counts = np.unique(d, axis=0, return_counts=True)
    return int(counts.max())


def test_fill_winding_consistent_on_a_real_relief_mesh():
    """Regression: orienting the fill by comparing the loop's best-fit-plane
    normal against an adjacent face normal passes on a FLAT fixture but is
    unreliable on real tear holes, which are not planar. Pins the exact
    per-edge rule instead."""
    mesh = _noisy_relief_mesh()
    faces = np.asarray(mesh.faces)
    assert _max_directed_edge_count(faces) == 1, "fixture itself is inconsistent"
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64,
                                            vertices=np.asarray(mesh.vertices))
    assert filled, "fixture produced no fillable holes — test is vacuous"
    assert _max_directed_edge_count(new_faces) == 1, (
        "fill is back-facing: a directed edge occurs twice")


def test_fill_never_degrades_a_real_relief_mesh():
    """The contract: filling may leave a hole open, but must never make the
    exported mesh worse than not filling at all."""
    mesh = _noisy_relief_mesh()
    faces = np.asarray(mesh.faces)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    new_faces, filled = fill_interior_holes(faces, max_hole_edges=64, vertices=v)
    assert filled

    def nonmanifold(f):
        e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, c = np.unique(e, axis=0, return_counts=True)
        return int((c > 2).sum())

    assert nonmanifold(new_faces) == nonmanifold(faces), "fill added a non-manifold edge"
    areas = _added_face_areas(v, faces, new_faces)
    assert (areas > 1e-12).all(), "fill added a degenerate face"
    # no vertices invented → the OBJ/GLB 1:1 vertex-UV mapping still holds
    assert int(np.asarray(new_faces).max()) < len(v)

def _sawtooth_strip(n_teeth=4):
    """A flat quad strip whose top boundary alternates between near
    "peaks" (z=-5) and far "bases" (z=-15).  Bases are local depth maxima
    on the boundary loop and should be bridged by fill_boundary_sawteeth."""
    verts = []
    # bottom row y=0, top row follows sawtooth
    for i in range(n_teeth * 2 + 1):
        verts.append([float(i), 0.0, -10.0])  # bottom vertex
    for i in range(n_teeth * 2 + 1):
        is_base = (i % 2) == 1
        z = -15.0 if is_base else -5.0
        verts.append([float(i), 1.0, z])  # top vertex
    vertices = np.asarray(verts, dtype=np.float64)
    faces = []
    n = n_teeth * 2 + 1
    for i in range(n - 1):
        a, b = i, i + 1
        d, e = n + i, n + i + 1
        faces.append([a, b, e])
        faces.append([a, e, d])
    return vertices, np.asarray(faces, dtype=np.int64)


def _assert_manifold_and_wound(faces):
    """Undirected edges in ≤2 faces AND no duplicated directed edge — the
    pivot walk's own invariant. A geometric winding pick can pass the
    undirected check while breaking this one."""
    f = np.asarray(faces)
    de = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, dc = np.unique(de, axis=0, return_counts=True)
    assert (dc == 1).all(), "duplicated directed edge → inconsistent winding"
    _, c = np.unique(np.sort(de, axis=1), axis=0, return_counts=True)
    assert (c <= 2).all(), "non-manifold edge"


def test_fill_boundary_sawteeth_bridges_depth_valleys():
    """Sawtooth bases (farther than both neighbours) get peak-base-peak
    triangles; peaks stay untouched."""
    v, f = _sawtooth_strip(n_teeth=4)
    view = _identity_view()
    new_f, depths = fill_boundary_sawteeth(f, vertices=v, view_matrix=view, depth_far_m=0.0)
    # 4 bases should be bridged
    assert len(new_f) == len(f) + 4
    assert len(depths) == 4
    # every added depth should be the far base depth
    assert all(abs(d - 15.0) < 1e-6 for d in depths)
    _assert_manifold_and_wound(new_f)
    # depth filter: forward depth is -view-z (camera faces -Z), so bases at
    # z=-15 sit at depth 15 — beyond a far bound of 8.0, so nothing fills
    new_f2, depths2 = fill_boundary_sawteeth(f, vertices=v, view_matrix=view, depth_far_m=8.0)
    assert len(new_f2) == len(f)
    assert len(depths2) == 0


def test_fill_boundary_sawteeth_non_planar_notches_stay_wound():
    """Non-planar teeth (bases displaced laterally + vertically) must still
    come out manifold and consistently wound — the flat fixture alone passes
    a geometric winding pick by luck (see the module's own winding lesson)."""
    v, f = _sawtooth_strip(n_teeth=4)
    n = 4 * 2 + 1
    for i in range(n):
        if (i % 2) == 1:  # base vertices: push off the strip plane
            v[n + i, 0] += 0.37
            v[n + i, 1] += 0.61
    new_f, depths = fill_boundary_sawteeth(f, vertices=v, view_matrix=_identity_view(), depth_far_m=0.0)
    assert len(new_f) == len(f) + 4
    assert len(depths) == 4
    _assert_manifold_and_wound(new_f)


def test_apply_boundary_sawtooth_fill_updates_mesh():
    """apply_boundary_sawtooth_fill mutates the mesh faces in place."""
    v, f = _sawtooth_strip(n_teeth=3)
    class M:
        pass
    m = M()
    m.vertices = v
    m.faces = f.copy()
    n_added, _ = apply_boundary_sawtooth_fill(m, view_matrix=_identity_view(), depth_far_m=0.0)
    assert n_added == 3
    assert len(m.faces) == len(f) + 3


def test_fill_boundary_sawteeth_bridges_grid_staircase_corners():
    """Verify that fill_boundary_sawteeth bridges 90-degree grid staircase corners even when depth is uniform along the boundary."""
    from atlas_camera.core.relief_mesh import build_relief_mesh
    h, w = 64, 64
    depth_arr = np.full((h, w), 10.0, dtype=np.float32)
    # create a flat circular rock at 3m
    cy, cx = 32, 32
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy)**2 + (xx - cx)**2 < 12**2
    depth_arr[mask] = 3.0

    view_matrix = np.eye(4)
    mesh = build_relief_mesh(
        depth_arr, view_matrix=view_matrix, fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        grid_long_edge=32, depth_edge_rel=0.5, scale=1.0, apply_sky_heuristic=False
    )
    faces_before = len(mesh.faces)
    n_added, _ = apply_boundary_sawtooth_fill(mesh, view_matrix=view_matrix, depth_far_m=0.0)
    assert n_added > 0, "Grid staircase corners on equal-depth silhouette boundary must be bridged by sawtooth fill"
    assert len(mesh.faces) == faces_before + n_added



def test_derive_relief_mesh_node_stores_repaired_faces_in_solve():
    """Verify that AtlasDeriveReliefMesh serializes the repaired mesh (not the raw unrepaired mesh) into solve.projection_scene."""
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    from atlas_camera.core.schema import (
        AtlasCamera, AtlasExtrinsics, AtlasSolve,
    )

    from atlas_camera.core.intrinsics import build_intrinsics

    from atlas_camera.inference.depth_estimator import DepthResult

    depth_arr = np.full((32, 32), 10.0, dtype=np.float32)
    depth_arr[10:20, 10:20] = 2.0
    depth = DepthResult(depth=depth_arr, is_metric=True, model_id="test", image_width=32, image_height=32)




    intr = build_intrinsics(image_width=32, image_height=32, focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))))
    solve = AtlasSolve(camera=cam, image_width=32, image_height=32)

    # 1. Without live repair
    out_unrepaired = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=32, depth_edge_rel=0.5,
        live_fill_holes=False, live_fill_edge_sawteeth=False
    )
    mesh_unrepaired = _relief_mesh_from_solve(out_unrepaired[0])

    # 2. With live repair
    out_repaired = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=32, depth_edge_rel=0.5,
        live_fill_holes=True, live_fill_max_hole_edges=64, live_fill_distance_m=0.0,
        live_fill_edge_sawteeth=True
    )
    mesh_repaired = _relief_mesh_from_solve(out_repaired[0])

    assert mesh_repaired is not None
    assert mesh_unrepaired is not None
    assert len(mesh_repaired.faces) > len(mesh_unrepaired.faces), "Repaired mesh in solve primitive must contain added faces"


def test_atlas_live_mesh_repair_node_repairs_solve_primitives():
    """Verify that AtlasLiveMeshRepair repairs relief mesh primitives on a solve downstream."""
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh, AtlasLiveMeshRepair
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.inference.depth_estimator import DepthResult

    depth_arr = np.full((32, 32), 10.0, dtype=np.float32)
    depth_arr[10:20, 10:20] = 2.0
    depth = DepthResult(depth=depth_arr, is_metric=True, model_id="test", image_width=32, image_height=32)

    intr = build_intrinsics(image_width=32, image_height=32, focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))))
    solve = AtlasSolve(camera=cam, image_width=32, image_height=32)

    out_unrepaired = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=32, depth_edge_rel=0.5,
        live_fill_holes=False, live_fill_edge_sawteeth=False
    )
    mesh_before = _relief_mesh_from_solve(out_unrepaired[0])

    out_repaired = AtlasLiveMeshRepair().repair(
        out_unrepaired[0], live_fill_holes=True, live_fill_max_hole_edges=64, live_fill_edge_sawteeth=True
    )
    mesh_after = _relief_mesh_from_solve(out_repaired[0])

    assert mesh_after is not None
    assert mesh_before is not None
    assert len(mesh_after.faces) >= len(mesh_before.faces), "AtlasLiveMeshRepair must successfully repair the solve's relief mesh"


def test_atlas_live_mesh_repair_cuda_backend():
    """The cuda backend recovers the grid from the mesh UVs, fills via the conv
    kernel, and materializes new vertices ray-consistently (no NaN, forward-
    distance placement)."""
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh, AtlasLiveMeshRepair
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.inference.depth_estimator import DepthResult

    depth_arr = np.full((48, 48), 10.0, dtype=np.float32)
    depth_arr[16:30, 16:30] = 2.0
    depth = DepthResult(depth=depth_arr, is_metric=True, model_id="test",
                        image_width=48, image_height=48)
    intr = build_intrinsics(image_width=48, image_height=48, focal_length_mm=35.0,
                            sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))))
    solve = AtlasSolve(camera=cam, image_width=48, image_height=48)

    out = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=48, depth_edge_rel=0.5,
        live_fill_holes=False, live_fill_edge_sawteeth=False)
    before = _relief_mesh_from_solve(out[0])

    repaired = AtlasLiveMeshRepair().repair(
        out[0], backend="cuda", live_fill_holes=True, live_fill_edge_sawteeth=True)
    after = _relief_mesh_from_solve(repaired[0])

    assert after is not None and before is not None
    assert len(after.vertices) >= len(before.vertices)
    assert len(after.faces) >= len(before.faces)
    assert np.isfinite(np.asarray(after.vertices)).all(), "new vertices must be finite"
    assert np.asarray(after.faces).max() < len(after.vertices), "faces must index valid vertices"


def test_repair_relief_mesh_grid_cuda_ray_consistency():
    """A vertex materialized by the grid path lands at the neighbour-averaged
    forward distance along its own camera ray — the same construction existing
    vertices satisfy — so re-projecting it reproduces that forward distance."""
    pytest.importorskip("torch")
    from atlas_camera.core.mesh_repair import repair_relief_mesh_grid_cuda
    from atlas_camera.core.relief_mesh import build_relief_mesh

    view = np.array([[1, 0, 0, 0], [0, 1, 0, 5.0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    fx = fy = 60.0
    W = H = 48
    cx = cy = 23.5
    depth = np.full((H, W), 12.0, dtype=np.float64)
    depth[18:28, 18:28] = 3.0
    mesh = build_relief_mesh(depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                             grid_long_edge=48, depth_edge_rel=0.5, smooth_iterations=0)
    n_v0 = len(mesh.vertices)
    n_h, n_s = repair_relief_mesh_grid_cuda(
        mesh, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=W, image_height=H, fill_holes=True, fill_sawteeth=True)

    if len(mesh.vertices) > n_v0:  # at least one cell was filled
        c2w = np.linalg.inv(view)
        R_cw, camp = c2w[:3, :3], c2w[:3, 3]
        new = np.asarray(mesh.vertices[n_v0:], dtype=np.float64)
        fwd = -((new - camp) @ R_cw[:, 2])
        assert (fwd > 0).all(), "materialized vertices sit in front of the camera"
        assert np.isfinite(new).all()


def test_repair_relief_mesh_grid_cuda_no_silhouette_bridge():
    """The fill must never bridge a near->far silhouette into a stretched shard:
    every ADDED triangle's world edges stay bounded, like build_relief_mesh's own
    tear test. Without the gate, the grid fill re-connects the torn silhouette."""
    pytest.importorskip("torch")
    from atlas_camera.core.mesh_repair import repair_relief_mesh_grid_cuda
    from atlas_camera.core.relief_mesh import build_relief_mesh

    view = np.array([[1, 0, 0, 0], [0, 1, 0, 4.0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    fx = fy = 70.0
    W = H = 64
    cx = cy = 31.5
    # A near foreground block (2 m) against a far background (40 m): a hard
    # silhouette. Tearing leaves a boundary the fill would love to bridge.
    depth = np.full((H, W), 40.0, dtype=np.float64)
    depth[24:40, 24:40] = 2.0
    mesh = build_relief_mesh(depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                             grid_long_edge=64, depth_edge_rel=0.5, smooth_iterations=0)
    n_f0 = len(mesh.faces)
    repair_relief_mesh_grid_cuda(
        mesh, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=W, image_height=H, fill_holes=True, fill_sawteeth=True)

    added = np.asarray(mesh.faces[n_f0:])
    if len(added):
        v = np.asarray(mesh.vertices, dtype=np.float64)
        a, b, c = v[added[:, 0]], v[added[:, 1]], v[added[:, 2]]
        el = np.concatenate([np.linalg.norm(a - b, axis=1),
                             np.linalg.norm(b - c, axis=1),
                             np.linalg.norm(c - a, axis=1)])
        # A bridged 2m->40m shard would be ~38 m long; legit local fills (which
        # iterate further as max_hole_edges rises) stay a few metres at most.
        assert el.max() < 15.0, f"fill bridged a silhouette (max edge {el.max():.1f} m)"


def test_repair_relief_mesh_grid_cuda_cap_enclosed():
    """cap_enclosed closes an ENCLOSED hole even across a real depth jump — the
    machine case the plain conv fill correctly refuses — placing fills at the
    FARTHEST neighbour depth (the back surface, away from camera), while the
    open silhouette tear around the block itself stays open."""
    pytest.importorskip("torch")
    from atlas_camera.core.mesh_repair import boundary_edges, repair_relief_mesh_grid_cuda
    from atlas_camera.core.relief_mesh import build_relief_mesh

    view = np.array([[1, 0, 0, 0], [0, 1, 0, 4.0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    fx = fy = 70.0
    W = H = 96
    cx = cy = 47.5
    depth = np.full((H, W), 12.0, dtype=np.float64)
    depth[24:64, 24:64] = 3.0  # near block against far bg = open silhouette tear
    excl = np.zeros((H, W), dtype=bool)
    excl[40:52, 56:72] = True  # enclosed hole STRADDLING the block edge (3m|12m boundary)

    def build():
        return build_relief_mesh(depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                                 grid_long_edge=96, depth_edge_rel=0.5, smooth_iterations=0,
                                 exclude_mask=excl, apply_sky_heuristic=False)

    results = {}
    for cap in (False, True):
        m = build()
        n_v0, n_f0 = len(m.vertices), len(m.faces)
        n_hole, _ = repair_relief_mesh_grid_cuda(
            m, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
            image_width=W, image_height=H, fill_holes=True, fill_sawteeth=True,
            max_hole_edges=256, cap_enclosed=cap)
        c2w = np.linalg.inv(view)
        fwd_new = -((np.asarray(m.vertices[n_v0:], dtype=np.float64) - c2w[:3, 3]) @ c2w[:3, :3][:, 2])
        results[cap] = dict(added=len(m.faces) - n_f0, n_hole=n_hole,
                            bedges=len(boundary_edges(np.asarray(m.faces))),
                            fwd_new=fwd_new, verts=np.asarray(m.vertices))
    r0, r1 = results[False], results[True]
    assert r1["n_hole"] > r0["n_hole"], "cap mode must fill the enclosed depth-jump hole"
    assert r1["added"] > r0["added"]
    assert r1["bedges"] < r0["bedges"], "capping must reduce open boundary edges"
    assert np.isfinite(r1["verts"]).all()
    if len(r1["fwd_new"]):
        # HARMONIC MEMBRANE, not a wall: fills stay strictly within the hole's
        # own boundary depth range and blend across it (the old farthest-depth
        # rule parked everything at 12m — downward walls on layered geometry).
        assert r1["fwd_new"].min() >= 3.0 - 1e-6
        assert r1["fwd_new"].max() <= 12.0 + 1e-6
        assert 4.0 < r1["fwd_new"].mean() < 11.0, "cap must be a blend, not a wall"


def test_repair_relief_mesh_grid_cuda_channel_tolerant_enclosure():
    """A dash hole reaching the outside only through a 1-cell invalid corridor
    still counts as enclosed (the valid mask is morphologically closed before
    the border flood-fill) — the user-reported circled holes that never filled."""
    pytest.importorskip("torch")
    from atlas_camera.core.mesh_repair import repair_relief_mesh_grid_cuda
    from atlas_camera.core.relief_mesh import build_relief_mesh

    view = np.array([[1, 0, 0, 0], [0, 1, 0, 4.0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    fx = fy = 70.0
    W = H = 96
    cx = cy = 47.5
    depth = np.full((H, W), 12.0, dtype=np.float64)
    excl = np.zeros((H, W), dtype=bool)
    excl[40:52, 56:68] = True   # the dash hole
    excl[46, 68:96] = True      # 1-cell corridor to the right frame border
    mesh = build_relief_mesh(depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                             grid_long_edge=96, depth_edge_rel=0.5, smooth_iterations=0,
                             exclude_mask=excl, apply_sky_heuristic=False)
    n_f0 = len(mesh.faces)
    n_hole, _ = repair_relief_mesh_grid_cuda(
        mesh, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=W, image_height=H, fill_holes=True, fill_sawteeth=False,
        max_hole_edges=256, cap_enclosed=True)
    assert n_hole > 0, "channel-connected dash hole must still be capped"
    assert len(mesh.faces) > n_f0
    assert np.isfinite(np.asarray(mesh.vertices)).all()


def test_smooth_boundary_loops_rounds_staircase():
    """Boundary Taubin relaxation shortens a staircase silhouette without
    changing vertex/face counts, and regenerates the moved vertices' UVs to
    exactly the projective bake (regenerate_projective_uvs)."""
    from atlas_camera.core.mesh_repair import boundary_edges, smooth_boundary_loops
    from atlas_camera.core.mesh_retopo import regenerate_projective_uvs
    from atlas_camera.core.relief_mesh import build_relief_mesh

    view = np.array([[1, 0, 0, 0], [0, 1, 0, 4.0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    fx = fy = 70.0
    W = H = 96
    cx = cy = 47.5
    depth = np.full((H, W), 12.0, dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    excl = (xx + yy) > 120  # diagonal cut -> lattice staircase boundary
    mesh = build_relief_mesh(depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
                             grid_long_edge=96, depth_edge_rel=0.5, smooth_iterations=0,
                             exclude_mask=excl, apply_sky_heuristic=False)
    uvs_before = np.asarray(mesh.uvs).copy()

    def boundary_length(m):
        be = boundary_edges(np.asarray(m.faces))
        v = np.asarray(m.vertices, dtype=np.float64)
        return float(np.linalg.norm(v[be[:, 0]] - v[be[:, 1]], axis=1).sum())

    L0 = boundary_length(mesh)
    n_v0, n_f0 = len(mesh.vertices), len(mesh.faces)
    n_moved = smooth_boundary_loops(mesh, iterations=8, view_matrix=view,
                                    fx=fx, fy=fy, cx=cx, cy=cy,
                                    image_width=W, image_height=H)
    assert n_moved > 0
    assert (len(mesh.vertices), len(mesh.faces)) == (n_v0, n_f0)
    assert boundary_length(mesh) < L0, "staircase must shorten"
    # Moved verts' UVs match the projective bake exactly; unmoved untouched.
    uvs_after = np.asarray(mesh.uvs)
    changed = np.any(uvs_after != uvs_before, axis=1)
    assert changed.any()
    idx = np.where(changed)[0]
    expected = regenerate_projective_uvs(
        np.asarray(mesh.vertices, dtype=np.float64)[idx], view_matrix=view,
        fx=fx, fy=fy, cx=cx, cy=cy, image_width=W, image_height=H)
    assert np.allclose(uvs_after[idx], expected, atol=1e-5)


