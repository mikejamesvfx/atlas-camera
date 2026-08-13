"""Merging solid volumes into one retopologizable mesh layer.

`AtlasBlockoutMassing` emits one box primitive per building — 97 on a city
plate — and `render_scene` loops per mesh, so the rasterizer slows roughly in
proportion. More importantly every `AtlasRetopologizeLayer` method operates on
serialized MESHES, so without a merge there is nothing for retopo to act on and
placeholder mass can never be simplified or exported as normal geometry.

The invariant that must survive: merging is a TOPOLOGY operation and may never
promote a guess to a measurement, so `provenance="placeholder"` rides through.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.projection_render import (  # noqa: E402
    gather_scene_meshes, merge_volume_primitives)
from atlas_camera.core.schema import (  # noqa: E402
    AtlasCamera, AtlasIntrinsics, AtlasProjectionScene, AtlasProxyPrimitive,
    AtlasSolve)


def _box(name, translate=(0.0, 0.0, 0.0), **meta):
    m = [[1.0, 0.0, 0.0, translate[0]],
         [0.0, 1.0, 0.0, translate[1]],
         [0.0, 0.0, 1.0, translate[2]],
         [0.0, 0.0, 0.0, 1.0]]
    md = {"role": "projection_proxy", "provenance": "placeholder",
          "trust": "placeholder"}
    md.update(meta)
    return AtlasProxyPrimitive(name=name, primitive_type="box",
                               transform_matrix=m, dimensions=(2.0, 4.0, 2.0),
                               metadata=md)


def _solve(prims):
    camera = AtlasCamera(intrinsics=AtlasIntrinsics(
        image_width=100, image_height=100, fx_px=100.0, fy_px=100.0,
        cx_px=50.0, cy_px=50.0))
    s = AtlasSolve(camera=camera)
    s.projection_scene = AtlasProjectionScene(proxy_geometry=list(prims))
    return s


def test_merges_every_box_into_one_mesh():
    solve = _solve([_box(f"b{i}", translate=(i * 5.0, 0.0, 0.0))
                    for i in range(4)])
    n = merge_volume_primitives(solve)
    assert n == 4
    prims = solve.projection_scene.proxy_geometry
    assert len(prims) == 1
    assert prims[0].primitive_type == "mesh"
    # 4 boxes x 8 verts, x 12 faces
    assert len(prims[0].metadata["vertices"]) == 4 * 8 * 3
    assert len(prims[0].metadata["faces"]) == 4 * 12 * 3


def test_placeholder_provenance_survives_the_merge():
    """A topology operation must never promote a guess to a measurement."""
    solve = _solve([_box("b0"), _box("b1")])
    merge_volume_primitives(solve)
    meta = solve.projection_scene.proxy_geometry[0].metadata
    assert meta["provenance"] == "placeholder"
    assert meta["trust"] == "placeholder"
    assert meta["merged_primitive_count"] == 2


def test_face_indices_are_rebased_not_overlapping():
    solve = _solve([_box("b0"), _box("b1", translate=(50.0, 0.0, 0.0))])
    merge_volume_primitives(solve)
    meta = solve.projection_scene.proxy_geometry[0].metadata
    faces = np.asarray(meta["faces"]).reshape(-1, 3)
    verts = np.asarray(meta["vertices"]).reshape(-1, 3)
    assert faces.max() == len(verts) - 1      # second box actually referenced
    assert faces.min() == 0


def test_merged_mesh_is_visible_to_the_rasterizer_with_uvs():
    solve = _solve([_box("b0"), _box("b1")])
    merge_volume_primitives(solve)
    meshes = gather_scene_meshes(solve, with_uvs=True)
    assert len(meshes) == 1
    label, verts, faces, uvs, _tex, _meta = meshes[0]
    assert uvs is not None and len(uvs) == len(verts)


def test_non_volume_primitives_are_left_alone():
    plane = AtlasProxyPrimitive(name="ground", primitive_type="plane",
                                dimensions=(10.0, 10.0, 0.0))
    solve = _solve([_box("b0"), plane])
    assert merge_volume_primitives(solve) == 1
    kinds = sorted(p.primitive_type
                   for p in solve.projection_scene.proxy_geometry)
    assert kinds == ["mesh", "plane"]


def test_no_volumes_is_a_no_op():
    plane = AtlasProxyPrimitive(name="ground", primitive_type="plane",
                                dimensions=(1.0, 1.0, 0.0))
    solve = _solve([plane])
    assert merge_volume_primitives(solve) == 0
    assert len(solve.projection_scene.proxy_geometry) == 1


def test_retopo_node_merges_massing_boxes_end_to_end():
    """The wiring: AtlasBlockoutMassing boxes -> one retopologizable layer."""
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _solve([_box(f"placeholder_mass_{i}", translate=(i * 8.0, 0.0, 0.0))
                    for i in range(5)])
    out, report = AtlasRetopologizeLayer().retopo(
        solve, method="off", merge_volume_primitives=True)
    prims = out.projection_scene.proxy_geometry
    assert [p.primitive_type for p in prims] == ["mesh"]
    assert prims[0].metadata["provenance"] == "placeholder"
    assert "merged 5 volume primitive" in report


def test_retopo_node_leaves_boxes_alone_when_off():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer

    solve = _solve([_box("b0"), _box("b1")])
    out, _ = AtlasRetopologizeLayer().retopo(solve, method="off")
    assert [p.primitive_type for p in out.projection_scene.proxy_geometry] == \
        ["box", "box"]
