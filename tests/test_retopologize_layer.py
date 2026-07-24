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
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve  # noqa: F401
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
        solve, depth, relief_grid=32, depth_edge_rel=0.5,
        live_fill_holes=False, live_fill_edge_sawteeth=False)[0]

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
