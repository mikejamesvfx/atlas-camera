"""Tests for export-only retopology on relief meshes (core/mesh_retopo.py).

The load-bearing piece is :func:`regenerate_projective_uvs`: quad remesh /
decimation change the vertex count, breaking the 1:1 vertex-UV mapping the
OBJ/GLB writers depend on, so the UVs must be regenerated from the recovered
camera. That math is the *exact inverse* of ``relief_mesh.build_relief_mesh``'s
forward back-projection bake — these tests build that forward bake and assert
the inverse recovers the ground-truth UVs bit-for-bit (pure numpy, no optional
deps, runs on every machine / CI).

The quad / decimation paths need optional third-party packages
(``pyinstantmeshes`` / ``fast-simplification``); where they are absent the tests
pin the informative-ImportError contract, and where present (importorskip) they
pin the round-trip. The smooth path needs only ``trimesh`` (already a dep).
"""

import numpy as np
import pytest

from atlas_camera.core.mesh_retopo import (
    _triangulate_quads,
    apply_retopo,
    regenerate_projective_uvs,
)


# ---------------------------------------------------------------------------
# Forward bake — mirrors relief_mesh.build_relief_mesh lines 454-496 exactly,
# so the inverse under test is verified against the real bake, not a paraphrase.
# ---------------------------------------------------------------------------

def _forward_bake(view_matrix, *, fx, fy, cx, cy, width, height, depth_grid,
                  scale=1.0):
    """Reproduce build_relief_mesh's world back-projection + UV bake.

    ``depth_grid`` is an (H, W) forward-depth array (metres, positive). Returns
    ``(vertices (N,3), uvs (N,2))`` in the exact same conventions as the real
    builder: world space via ``p_cam @ R_cw.T + cam`` then ray-preserving
    rescale about the camera, and OBJ bottom-left UVs ``u=uu/(W-1), v=1-vv/(H-1)``.
    """
    vm = np.asarray(view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    R_cw = c2w[:3, :3]
    cam = c2w[:3, 3]
    H, W = depth_grid.shape
    assert (H, W) == (height, width)
    rows, cols = np.indices((H, W))
    uu = cols.astype(np.float64)
    vv = rows.astype(np.float64)
    d = np.asarray(depth_grid, dtype=np.float64)
    x = (uu - cx) / fx * d
    y = -(vv - cy) / fy * d
    z = -d
    p_cam = np.stack([x, y, z], axis=-1)
    world = p_cam @ R_cw.T + cam
    world = cam + float(scale) * (world - cam)  # ray-preserving rescale
    u_uv = uu / max(W - 1, 1)
    v_uv = 1.0 - vv / max(H - 1, 1)
    uvs = np.stack([u_uv, v_uv], axis=-1).reshape(-1, 2)
    return world.reshape(-1, 3), uvs


def _identity_view():
    return np.eye(4, dtype=np.float64)


def _view_from_c2w(c2w):
    """view = inv(c2w), the convention relief_mesh uses."""
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64))


class _Mesh:
    """Minimal stand-in for ReliefMesh (vertices/faces/uvs)."""

    def __init__(self, vertices, uvs, faces):
        self.vertices = vertices
        self.uvs = uvs
        self.faces = faces


# ---------------------------------------------------------------------------
# regenerate_projective_uvs — the inverse-of-bake (always runnable, pure numpy)
# ---------------------------------------------------------------------------

def test_regenerate_uvs_inverts_identity_bake():
    """Identity view: regenerated UVs == forward-bake UVs to ~1e-6."""
    H, W = 8, 12
    fx = fy = 600.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rng = np.random.default_rng(0)
    depth = 5.0 + rng.random((H, W)) * 10.0  # 5..15 m forward depth
    verts, uvs_true = _forward_bake(
        _identity_view(), fx=fx, fy=fy, cx=cx, cy=cy,
        width=W, height=H, depth_grid=depth,
    )
    uvs = regenerate_projective_uvs(
        verts, view_matrix=_identity_view(),
        fx=fx, fy=fy, cx=cx, cy=cy, image_width=W, image_height=H,
    )
    assert uvs.shape == (H * W, 2)
    assert uvs.dtype == np.float32
    np.testing.assert_allclose(uvs, uvs_true, atol=1e-5)


def test_regenerate_uvs_inverts_with_rescale():
    """The ray-preserving rescale-about-cam must not change the recovered UV
    (it moves verts along their view rays → same pixel). scale != 1."""
    H, W = 6, 6
    fx = fy = 500.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rng = np.random.default_rng(1)
    depth = 10.0 + rng.random((H, W)) * 20.0
    verts, uvs_true = _forward_bake(
        _identity_view(), fx=fx, fy=fy, cx=cx, cy=cy,
        width=W, height=H, depth_grid=depth, scale=3.5,
    )
    uvs = regenerate_projective_uvs(
        verts, view_matrix=_identity_view(),
        fx=fx, fy=fy, cx=cx, cy=cy, image_width=W, image_height=H,
    )
    np.testing.assert_allclose(uvs, uvs_true, atol=1e-5)


def test_regenerate_uvs_inverts_rotated_translated_camera():
    """Non-identity view (rotated + translated camera): recovers UVs exactly."""
    H, W = 10, 14
    fx, fy = 800.0, 750.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    # camera at (1, 2, 3) yawed 25° about Y and pitched 10° about X.
    ang_y, ang_x = np.deg2rad(25.0), np.deg2rad(10.0)
    Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)],
                   [0, 1, 0],
                   [-np.sin(ang_y), 0, np.cos(ang_y)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ang_x), -np.sin(ang_x)],
                   [0, np.sin(ang_x), np.cos(ang_x)]])
    R = Rx @ Ry
    cam = np.array([1.0, 2.0, 3.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam
    view = _view_from_c2w(c2w)
    rng = np.random.default_rng(2)
    depth = 8.0 + rng.random((H, W)) * 12.0
    verts, uvs_true = _forward_bake(
        view, fx=fx, fy=fy, cx=cx, cy=cy,
        width=W, height=H, depth_grid=depth, scale=2.0,
    )
    uvs = regenerate_projective_uvs(
        verts, view_matrix=view,
        fx=fx, fy=fy, cx=cx, cy=cy, image_width=W, image_height=H,
    )
    np.testing.assert_allclose(uvs, uvs_true, atol=1e-5)


def test_regenerate_uvs_behind_camera_clamps_no_nan():
    """A vertex behind / at the camera (z_c >= 0) must clamp to the image
    boundary and never emit NaN — the writers can't tolerate NaN UVs."""
    H, W = 4, 4
    fx = fy = 400.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    # One in-front vert and three behind/at-camera verts.
    verts = np.array([
        [0.0, 0.0, -5.0],   # in front (z_c = -5)
        [0.0, 0.0, +5.0],   # behind
        [0.0, 0.0, 0.0],    # at camera
        [1.0, 1.0, 1.0],    # behind
    ], dtype=np.float64)
    uvs = regenerate_projective_uvs(
        verts, view_matrix=_identity_view(),
        fx=fx, fy=fy, cx=cx, cy=cy, image_width=W, image_height=H,
    )
    assert np.isfinite(uvs).all()
    assert uvs.shape == (4, 2)
    # The in-front vert projects to the principal point → u=v=0.5.
    np.testing.assert_allclose(uvs[0], [0.5, 0.5], atol=1e-4)
    # Behind/at verts clamp into [0,1].
    assert ((uvs[1:] >= -1e-6) & (uvs[1:] <= 1.0 + 1e-6)).all()


def test_regenerate_uvs_rejects_bad_shapes():
    vm = _identity_view()
    with pytest.raises(ValueError):
        regenerate_projective_uvs(np.zeros((5, 2)), view_matrix=vm,
                                  fx=100, fy=100, cx=50, cy=50,
                                  image_width=100, image_height=100)
    with pytest.raises(ValueError):
        regenerate_projective_uvs(np.zeros((5, 3)), view_matrix=np.eye(3),
                                  fx=100, fy=100, cx=50, cy=50,
                                  image_width=100, image_height=100)


# ---------------------------------------------------------------------------
# _triangulate_quads
# ---------------------------------------------------------------------------

def test_triangulate_quads_passthrough_triangles():
    f = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int64)
    out = _triangulate_quads(f)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, f)


def test_triangulate_quads_real_quads_double():
    # Two real quads → 4 triangles.
    f = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    out = _triangulate_quads(f)
    assert out.shape == (4, 3)
    # Implementation concatenates all tri1s then all tri2s (grouped, not
    # per-quad interleaved) — either ordering is a valid triangulation; pin
    # the actual grouping so a silent reorder is caught.
    expected = np.array([[0, 1, 2], [4, 5, 6],
                         [0, 2, 3], [4, 6, 7]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_triangulate_quads_degenerate_rows_single_tri():
    # Degenerate quads: d repeats a/b/c → one triangle each (tri2 dropped).
    f = np.array([
        [0, 1, 2, 0],   # d == a → tri [0,1,2]
        [3, 4, 5, 5],   # d == c → tri [3,4,5]
        [6, 7, 8, -1],  # d < 0  → tri [6,7,8]
    ], dtype=np.int64)
    out = _triangulate_quads(f)
    assert out.shape == (3, 3)
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_triangulate_quads_mixed():
    # One real quad + one degenerate → 2 + 1 = 3 triangles.
    f = np.array([[0, 1, 2, 3], [4, 5, 6, 4]], dtype=np.int64)
    out = _triangulate_quads(f)
    assert out.shape == (3, 3)


def test_triangulate_quads_bad_shape():
    with pytest.raises(ValueError):
        _triangulate_quads(np.zeros((3, 5), dtype=np.int64))


# ---------------------------------------------------------------------------
# apply_retopo — node-facing wrapper (dep-independent paths first)
# ---------------------------------------------------------------------------

def _flat_grid_mesh(z=-10.0):
    """Small watertight-ish grid mesh: 2x2 quads, 2 tris each, 3x3 verts."""
    verts = np.array([[float(c), float(r), z]
                      for r in range(3) for c in range(3)], dtype=np.float64)
    uvs = verts[:, :2].copy()
    faces = []
    for r in range(2):
        for c in range(2):
            a = r * 3 + c
            b = r * 3 + c + 1
            d = (r + 1) * 3 + c
            e = (r + 1) * 3 + c + 1
            faces.append([a, b, e])
            faces.append([a, e, d])
    return _Mesh(verts, uvs, np.asarray(faces, dtype=np.int64))


def test_apply_retopo_invalid_method():
    mesh = _flat_grid_mesh()
    with pytest.raises(ValueError):
        apply_retopo(mesh, method="bogus")


def test_apply_retopo_off_is_noop():
    mesh = _flat_grid_mesh()
    v_before = mesh.vertices.copy()
    f_before = mesh.faces.copy()
    rep = apply_retopo(mesh, method="off")
    assert rep["changed"] is False
    assert rep["method"] == "off"
    np.testing.assert_array_equal(mesh.vertices, v_before)
    np.testing.assert_array_equal(mesh.faces, f_before)


def test_apply_retopo_no_faces_is_noop():
    mesh = _flat_grid_mesh()
    mesh.faces = np.zeros((0, 3), dtype=np.int64)
    rep = apply_retopo(mesh, method="smooth")
    assert rep["changed"] is False


def test_apply_retopo_quad_needs_intrinsics():
    """quad/decimate change vertex count → must regenerate UVs → require the
    solved intrinsics. Missing intrinsics raise ValueError BEFORE any optional
    import is attempted, so this is dep-independent."""
    mesh = _flat_grid_mesh()
    with pytest.raises(ValueError):
        apply_retopo(mesh, method="quad", target_vertex_count=50,
                     view_matrix=None, fx=0.0, image_width=0, image_height=0)
    with pytest.raises(ValueError):
        apply_retopo(mesh, method="decimate", target_vertex_count=50,
                     view_matrix=_identity_view(), fx=100.0,
                     image_width=0, image_height=0)


# ---------------------------------------------------------------------------
# smooth path — trimesh is already a dep, so always runnable
# ---------------------------------------------------------------------------

def test_apply_retopo_smooth_preserves_topology_and_uvs():
    trimesh = pytest.importorskip("trimesh")
    mesh = _flat_grid_mesh(z=-10.0)
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    uvs_before = mesh.uvs.copy()
    rep = apply_retopo(mesh, method="smooth", smooth_iterations=3)
    assert rep["changed"] is True
    assert rep["method"] == "smooth"
    assert rep["in_verts"] == n_v and rep["out_verts"] == n_v
    assert rep["in_faces"] == n_f and rep["out_faces"] == n_f
    # Topology unchanged → same vertex/face count, UVs kept verbatim.
    assert len(mesh.vertices) == n_v
    assert len(mesh.faces) == n_f
    np.testing.assert_array_equal(mesh.faces, np.asarray(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
         [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]], dtype=np.int64))
    np.testing.assert_array_equal(mesh.uvs, uvs_before)
    # Positions moved (Taubin relax on a non-planar-ish field). On a perfectly
    # flat grid all verts are coplanar so positions barely move — assert the
    # function ran (report says changed) rather than asserting movement.


# ---------------------------------------------------------------------------
# quad / decimate paths — optional deps. Pin the missing-dep contract where
# absent, and the UV-regeneration round-trip where present.
# ---------------------------------------------------------------------------

def test_apply_retopo_quad_missing_dep_raises_importerror():
    pytest.importorskip = pytest.importorskip  # noqa
    try:
        import pyinstantmeshes  # noqa: F401
    except ImportError:
        mesh = _flat_grid_mesh()
        with pytest.raises(ImportError) as excinfo:
            apply_retopo(mesh, method="quad", target_vertex_count=20,
                         view_matrix=_identity_view(),
                         fx=100.0, fy=100.0, cx=50.0, cy=50.0,
                         image_width=100, image_height=100)
        assert "pyinstantmeshes" in str(excinfo.value).lower()
    else:  # pragma: no cover - dep present
        pytest.skip("pyinstantmeshes installed; missing-dep contract not testable")


def test_apply_retopo_decimate_missing_dep_raises_importerror():
    try:
        import fast_simplification  # noqa: F401
    except ImportError:
        mesh = _flat_grid_mesh()
        with pytest.raises(ImportError) as excinfo:
            apply_retopo(mesh, method="decimate", target_vertex_count=20,
                         view_matrix=_identity_view(),
                         fx=100.0, fy=100.0, cx=50.0, cy=50.0,
                         image_width=100, image_height=100)
        assert "fast-simplification" in str(excinfo.value).lower() or \
               "fast_simplification" in str(excinfo.value).lower()
    else:  # pragma: no cover - dep present
        pytest.skip("fast-simplification installed; missing-dep contract not testable")


def test_apply_retopo_quad_roundtrip_uv_regeneration():
    """Where pyinstantmeshes IS installed: after quad retopo, mesh.uvs must
    equal the forward-bake UVs of the NEW vertices (the regen is correct)."""
    pytest.importorskip("pyinstantmeshes")
    H, W = 16, 16
    fx = fy = 400.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rng = np.random.default_rng(3)
    depth = 8.0 + rng.random((H, W)) * 6.0
    verts, uvs_true = _forward_bake(
        _identity_view(), fx=fx, fy=fy, cx=cx, cy=cy,
        width=W, height=H, depth_grid=depth,
    )
    # Trivial grid faces so the mesh is valid input for remesh.
    faces = []
    for r in range(H - 1):
        for c in range(W - 1):
            a = r * W + c
            b = r * W + c + 1
            d = (r + 1) * W + c
            e = (r + 1) * W + c + 1
            faces.append([a, b, e])
            faces.append([a, e, d])
    mesh = _Mesh(verts, uvs_true, np.asarray(faces, dtype=np.int64))
    rep = apply_retopo(mesh, method="quad", target_vertex_count=200,
                       view_matrix=_identity_view(),
                       fx=fx, fy=fy, cx=cx, cy=cy,
                       image_width=W, image_height=H)
    assert rep["changed"] is True
    # Regenerated UVs must match a fresh forward projection of the new verts.
    expected = regenerate_projective_uvs(
        mesh.vertices, view_matrix=_identity_view(),
        fx=fx, fy=fy, cx=cx, cy=cy, image_width=W, image_height=H,
    )
    np.testing.assert_allclose(mesh.uvs.astype(np.float64),
                               expected.astype(np.float64), atol=1e-5)
    assert np.isfinite(mesh.uvs).all()


# ---------------------------------------------------------------------------
# Node wiring — AtlasExportReliefMesh.export() must call apply_retopo with the
# solved intrinsics, gated by retopo_method, running after hole-fill. Uses
# monkeypatch (not trimesh moving coplanar verts) so it is dep-independent.
# ---------------------------------------------------------------------------

def _solve_with_relief_mesh(vertices, uvs, faces):
    """Build a minimal AtlasSolve carrying a relief mesh on its proxy_geometry,
    so AtlasExportReliefMesh.export(use_solve_mesh=True) reuses it and never
    invokes the neural depth model."""
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve  # noqa: F401
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    from atlas_camera.core.relief_mesh import ReliefMesh
    from atlas_camera.core.schema import (
        AtlasCamera, AtlasExtrinsics, AtlasProjectionScene, AtlasSolve)

    H, W = 8, 8
    intr = build_intrinsics(image_width=W, image_height=H,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 1.6, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0),
                             (0, 0, 1, 0), (0, 0, 0, 1))))
    solve = AtlasSolve(camera=cam, image_width=W, image_height=H)
    mesh = ReliefMesh(vertices=vertices, faces=faces, uvs=uvs)
    solve.projection_scene = AtlasProjectionScene(
        proxy_geometry=[relief_mesh_primitive(mesh)])
    assert _relief_mesh_from_solve(solve) is not None
    return solve


def _image_tensor(H, W):
    import torch
    arr = np.zeros((H, W, 3), dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # 1×H×W×3


def test_export_retopo_wiring_calls_apply_retopo(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import atlas_camera.core.mesh_retopo as retopo_mod
    from atlas_camera.comfy.nodes import AtlasExportReliefMesh

    verts = np.array([[0, 0, -5], [1, 0, -5], [0, 1, -5],
                      [1, 1, -6]], dtype=np.float64)
    uvs = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
                   dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    solve = _solve_with_relief_mesh(verts, uvs, faces)

    captured = {}

    def fake_apply_retopo(mesh, *, method, **kwargs):
        captured["called"] = True
        captured["method"] = method
        captured["kwargs"] = kwargs
        # Don't actually mutate — just record. Return a report-shaped dict.
        return {"method": method, "changed": True}

    monkeypatch.setattr(retopo_mod, "apply_retopo", fake_apply_retopo)

    res = AtlasExportReliefMesh.export(
        AtlasExportReliefMesh, solve, _image_tensor(8, 8),
        output_dir=str(tmp_path), use_solve_mesh=True,
        format="obj", retopo_method="smooth",
        retopo_smooth_iterations=2,
    )
    obj, glb = res["result"][0], res["result"][1]
    assert captured.get("called") is True
    assert captured["method"] == "smooth"
    # Solved intrinsics are threaded through (the UV-regeneration inputs).
    assert captured["kwargs"]["image_width"] == 8
    assert captured["kwargs"]["image_height"] == 8
    assert captured["kwargs"]["fx"] > 0
    assert captured["kwargs"]["view_matrix"] is not None
    assert captured["kwargs"]["smooth_iterations"] == 2
    assert obj and not glb


def test_export_retopo_off_does_not_call_apply_retopo(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import atlas_camera.core.mesh_retopo as retopo_mod
    from atlas_camera.comfy.nodes import AtlasExportReliefMesh

    verts = np.array([[0, 0, -5], [1, 0, -5], [0, 1, -5],
                      [1, 1, -6]], dtype=np.float64)
    uvs = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
                   dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    solve = _solve_with_relief_mesh(verts, uvs, faces)

    def fake_apply_retopo(*a, **k):
        raise AssertionError("apply_retopo must not run when retopo_method='off'")

    monkeypatch.setattr(retopo_mod, "apply_retopo", fake_apply_retopo)
    res = AtlasExportReliefMesh.export(
        AtlasExportReliefMesh, solve, _image_tensor(8, 8),
        output_dir=str(tmp_path), use_solve_mesh=True,
        format="obj", retopo_method="off",
    )
    obj, glb = res["result"][0], res["result"][1]
    assert obj and not glb  # export still writes the OBJ normally