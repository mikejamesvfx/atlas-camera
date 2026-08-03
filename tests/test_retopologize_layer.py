"""AtlasRetopologizeLayer 🔷 — live retopo on a single solve layer.

Pins the node contract: the `layer` selector targets exactly one mesh (primary
scene mesh for "", a named ProjectionSource, "*" for all), Taubin smooth moves
positions without changing counts (UVs preserved), missing deps degrade soft
with the solve passed through, and an unknown layer name reports the available
names instead of silently no-opping.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasSolve,
    ProjectionSource,
)


def _relief_solve():
    """A solve carrying a primary relief mesh + one named layer source."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    from atlas_camera.inference.depth_estimator import DepthResult

    depth_arr = np.full((32, 32), 10.0, dtype=np.float32)
    depth_arr[10:20, 10:20] = 2.0
    depth = DepthResult(depth=depth_arr, is_metric=True, model_id="test",
                        image_width=32, image_height=32)
    intr = build_intrinsics(image_width=32, image_height=32,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))))
    solve = AtlasSolve(camera=cam, image_width=32, image_height=32)
    out = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=32, depth_edge_rel=0.5)[0]

    # Named layer source: its own camera + a copy of the primary relief mesh.
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    mesh = _relief_mesh_from_solve(out)
    src = ProjectionSource(camera=copy.deepcopy(cam), name="bg",
                           proxy_geometry=[relief_mesh_primitive(copy.deepcopy(mesh))])
    out.projection_sources.append(src)
    return out


def _primary_verts(solve):
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    return np.asarray(_relief_mesh_from_solve(solve).vertices)


def _layer_verts(solve, name):
    from atlas_camera.exporters._layers import mesh_from_primitive
    src = next(s for s in solve.projection_sources if s.name == name)
    prim = next(p for p in src.proxy_geometry if p.primitive_type == "mesh")
    return np.asarray(mesh_from_primitive(prim).vertices)


def test_smooth_targets_only_named_layer():
    pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from atlas_camera.comfy.nodes_geometry import AtlasRetopologizeLayer

    solve = _relief_solve()
    p0, l0 = _primary_verts(solve), _layer_verts(solve, "bg")
    out, report = AtlasRetopologizeLayer().retopo(
        solve, layer="bg", method="smooth", smooth_iterations=5)
    p1, l1 = _primary_verts(out), _layer_verts(out, "bg")
    assert np.allclose(p1, p0), "primary must be untouched when a layer is named"
    assert l1.shape == l0.shape, "smooth keeps the vertex count"
    assert not np.allclose(l1, l0, atol=1e-4), "smooth must move layer vertices"
    assert "bg" in report and "smooth" in report
    # input solve untouched (deep-copy contract)
    assert np.allclose(_layer_verts(solve, "bg"), l0)


def test_smooth_primary_and_star_selector():
    pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from atlas_camera.comfy.nodes_geometry import AtlasRetopologizeLayer

    solve = _relief_solve()
    p0, l0 = _primary_verts(solve), _layer_verts(solve, "bg")
    out, _ = AtlasRetopologizeLayer().retopo(
        solve, layer="", method="smooth", smooth_iterations=5)
    assert not np.allclose(_primary_verts(out), p0)
    assert np.allclose(_layer_verts(out, "bg"), l0)

    out2, report2 = AtlasRetopologizeLayer().retopo(
        solve, layer="*", method="smooth", smooth_iterations=5)
    assert not np.allclose(_primary_verts(out2), p0)
    assert not np.allclose(_layer_verts(out2, "bg"), l0)
    assert "primary" in report2 and "bg" in report2


def test_missing_dep_degrades_soft(monkeypatch):
    from atlas_camera.comfy.nodes_geometry import AtlasRetopologizeLayer
    import atlas_camera.exporters._layers as layers_mod

    def _boom(*a, **k):
        raise ImportError("Quad remeshing needs 'pyinstantmeshes' — pip install pyinstantmeshes")
    monkeypatch.setattr(layers_mod, "_retopologize_layer_mesh", _boom)

    solve = _relief_solve()
    p0 = _primary_verts(solve)
    out, report = AtlasRetopologizeLayer().retopo(solve, layer="", method="quad")
    assert np.allclose(_primary_verts(out), p0), "solve passes through untouched"
    assert "SKIPPED" in report and "pyinstantmeshes" in report


def test_unknown_layer_reports_available_names():
    from atlas_camera.comfy.nodes_geometry import AtlasRetopologizeLayer

    solve = _relief_solve()
    out, report = AtlasRetopologizeLayer().retopo(solve, layer="nope", method="smooth")
    assert "not found" in report and "bg" in report


# ---------------------------------------------------------------------------
# boundary_smooth_iterations — capability migrated from AtlasLiveMeshRepair
# ---------------------------------------------------------------------------

def _primary_mesh(solve):
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    return _relief_mesh_from_solve(solve)


def _uv_registration_error(solve):
    """max |uv - regenerate_projective_uvs(v)| over every vertex, in UV units.

    This is THE assertion for the migration: boundary smoothing moves
    vertices, so if their projective UVs are not regenerated with the layer's
    own camera the projection silently slides off the geometry.
    """
    from atlas_camera.core.mesh_retopo import regenerate_projective_uvs

    mesh = _primary_mesh(solve)
    cam = solve.camera
    intr, extr = cam.intrinsics, cam.extrinsics
    expected = regenerate_projective_uvs(
        np.asarray(mesh.vertices, dtype=np.float64),
        view_matrix=extr.camera_view_matrix,
        fx=float(intr.fx_px), fy=float(intr.fy_px or intr.fx_px),
        cx=float(intr.cx_px), cy=float(intr.cy_px),
        image_width=int(intr.image_width), image_height=int(intr.image_height))
    return float(np.abs(np.asarray(mesh.uvs, dtype=np.float64) - expected).max())


def test_boundary_smooth_widget_is_appended_last():
    """Widgets are positional in saved workflows — appends only, never inserts."""
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    assert list(AtlasRetopologizeLayer.INPUT_TYPES()["optional"]) == [
        "layer", "method", "target_vertex_count", "smooth_iterations",
        "crease_angle", "pure_quad", "boundary_smooth_iterations",
    ]


def test_boundary_smoothing_runs_with_method_off():
    """The early-return regression: method='off' reports changed=False, but
    'just round the silhouette' is exactly what that configuration means."""
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _relief_solve()
    before = _primary_verts(solve)
    out, report = AtlasRetopologizeLayer().retopo(
        solve, layer="", method="off", boundary_smooth_iterations=8)
    after = _primary_verts(out)
    assert after.shape == before.shape                    # topology untouched
    assert not np.allclose(after, before)                 # but something moved
    assert "boundary smooth" in report


def test_boundary_smoothing_moves_only_boundary_verts():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer
    from atlas_camera.core.mesh_repair import boundary_edges

    solve = _relief_solve()
    mesh0 = _primary_mesh(solve)
    before = np.asarray(mesh0.vertices).copy()
    boundary = set(np.asarray(boundary_edges(np.asarray(mesh0.faces))).reshape(-1).tolist())

    out, _ = AtlasRetopologizeLayer().retopo(
        solve, layer="", method="off", boundary_smooth_iterations=8)
    after = _primary_verts(out)

    moved = {int(i) for i in np.nonzero(~np.isclose(after, before).all(axis=1))[0]}
    assert moved, "expected at least one boundary vertex to move"
    assert moved <= boundary, "an interior vertex moved — smoothing must be boundary-only"


def test_boundary_smoothing_keeps_projective_uvs_registered():
    """Boundary smoothing must not degrade projective registration.

    The BUILD itself carries ~1.1e-3 of UV error, because serialized vertices
    round to 3 dp (metres) and UVs to 4 dp — so the honest bar is "no worse
    than the mesh we were handed", measured, not an absolute epsilon.
    """
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _relief_solve()
    baseline = _uv_registration_error(solve)
    out, _ = AtlasRetopologizeLayer().retopo(
        solve, layer="", method="off", boundary_smooth_iterations=8)
    after = _uv_registration_error(out)
    assert after < baseline + 5e-4          # measured: 1.09e-3 -> 1.30e-3
    assert after < 2e-3


def test_boundary_smoothing_after_decimate_keeps_uvs_registered():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _relief_solve()
    n0 = len(_primary_verts(solve))
    out, _ = AtlasRetopologizeLayer().retopo(
        solve, layer="", method="decimate", target_vertex_count=200,
        boundary_smooth_iterations=8)
    assert len(_primary_verts(out)) < n0     # it really decimated
    assert _uv_registration_error(out) < 2e-3


def test_method_smooth_regenerates_projective_uvs():
    """Regression for a real defect: `smooth` used to leave UVs stale.

    A Taubin relax preserves topology, so the 1:1 vertex-UV INDEX mapping
    survives — which is why the branch originally kept the existing UVs. But
    every vertex moves, and a moved vertex projects somewhere else, so the
    projective registration went stale: measured 2.9e-2 against a 1.1e-3 build
    baseline (26x), i.e. the projection visibly slides off the geometry.
    """
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _relief_solve()
    baseline = _uv_registration_error(solve)

    smooth_only, report = AtlasRetopologizeLayer().retopo(
        copy.deepcopy(solve), layer="", method="smooth", smooth_iterations=2)
    err_smooth = _uv_registration_error(smooth_only)
    assert err_smooth < baseline + 5e-4, "smooth must re-register its UVs"
    assert "UVs regenerated" in report

    with_boundary, _ = AtlasRetopologizeLayer().retopo(
        copy.deepcopy(solve), layer="", method="smooth", smooth_iterations=2,
        boundary_smooth_iterations=8)
    assert _uv_registration_error(with_boundary) < baseline + 5e-4


def test_smooth_without_intrinsics_says_uvs_are_stale():
    """Intrinsics stay OPTIONAL for smooth (unlike quad/decimate), because a
    caller may smooth a mesh carrying no projection — but silence would imply
    the UVs are still good."""
    import numpy as np

    from atlas_camera.core.mesh_retopo import apply_retopo

    class M:
        pass

    m = M()
    m.vertices = np.array([[0., 0., -5.], [1., 0., -5.], [0., 1., -5.],
                           [1., 1., -5.2]], dtype=np.float64)
    m.faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    m.uvs = np.zeros((4, 2), dtype=np.float64)
    report = apply_retopo(m, method="smooth", smooth_iterations=1)
    assert report["changed"] is True
    assert "NOT regenerated" in report["note"] and "stale" in report["note"]


def test_boundary_smoothing_skipped_without_intrinsics():
    """No usable camera -> report it and leave the mesh alone, rather than
    moving verts and silently stranding their UVs."""
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _relief_solve()
    src = solve.projection_sources[0]
    src.camera.intrinsics.fx_px = 0.0
    before = np.asarray(
        (src.proxy_geometry[0].metadata or {}).get("vertices"), dtype=np.float64).copy()

    out, report = AtlasRetopologizeLayer().retopo(
        solve, layer="bg", method="off", boundary_smooth_iterations=8)
    after = np.asarray(
        (out.projection_sources[0].proxy_geometry[0].metadata or {}).get("vertices"),
        dtype=np.float64)

    assert np.array_equal(before, after)                  # untouched
    assert "SKIPPED" in report and "intrinsics" in report  # and said so


def test_boundary_smoothing_zero_is_a_no_op():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _relief_solve()
    before = _primary_verts(solve)
    out, _ = AtlasRetopologizeLayer().retopo(
        solve, layer="", method="off", boundary_smooth_iterations=0)
    assert np.array_equal(_primary_verts(out), before)
