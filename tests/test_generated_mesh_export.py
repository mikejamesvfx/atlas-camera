"""Generated-object vertex colour survives payload, layer rebuild and export."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from atlas_camera.core.proxy_geometry import PROXY_ROLE, serialize_proxy_geometry
from atlas_camera.core.relief_mesh import ReliefMesh
from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive
from atlas_camera.exporters._layers import mesh_from_primitive
from atlas_camera.exporters.relief_mesh_exporter import (
    export_relief_mesh,
    export_relief_mesh_glb,
)

# Two triangles sharing an edge: one seen (photo), one hidden (vertex colour).
VERTS = np.array([[0, 0, -5], [1, 0, -5], [0, 1, -5], [1, 1, -6]], dtype=np.float32)
FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
UVS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
COLS = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.9, 0.4]],
                dtype=np.float32)
PW = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)


def _prim():
    return AtlasProxyPrimitive(
        name="pixal3d_object", primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
        material="atlas_projection_proxy",
        metadata={"role": PROXY_ROLE, "source": "pixal3d",
                  "vertices": VERTS.reshape(-1).tolist(),
                  "faces": FACES.reshape(-1).tolist(),
                  "uvs": UVS.reshape(-1).tolist(),
                  "vertex_colors": COLS.reshape(-1).tolist(),
                  "photo_weight": PW.tolist(), "ribbon_t": [], "edge_risk": []})


def _glb_json(path):
    with open(path, "rb") as fh:
        data = fh.read()
    (json_len,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + json_len])


def test_payload_lifts_vertex_colour_and_photo_weight():
    scene = AtlasProjectionScene(proxy_geometry=[_prim()])
    entry = serialize_proxy_geometry(scene)[0]
    assert entry["vertex_colors"] == pytest.approx(COLS.reshape(-1).tolist())
    assert entry["photo_weight"] == pytest.approx(PW.tolist())
    # Arrays never leak into the scalar metadata dict.
    assert "vertex_colors" not in entry["metadata"]


def test_ordinary_mesh_payload_has_empty_colour_fields():
    p = _prim()
    p.metadata.pop("vertex_colors")
    p.metadata.pop("photo_weight")
    entry = serialize_proxy_geometry(AtlasProjectionScene(proxy_geometry=[p]))[0]
    assert entry["vertex_colors"] == [] and entry["photo_weight"] == []


def test_layer_rebuild_round_trips():
    mesh = mesh_from_primitive(_prim())
    assert np.allclose(mesh.vertex_colors, COLS)
    assert np.allclose(mesh.photo_weight, PW)


def test_glb_splits_photo_and_vertex_colour_primitives(tmp_path):
    mesh = ReliefMesh(vertices=VERTS, faces=FACES, uvs=UVS, vertex_colors=COLS,
                      photo_weight=PW)
    gltf = _glb_json(export_relief_mesh_glb(mesh, tmp_path)["glb"])
    prims = gltf["meshes"][0]["primitives"]
    assert len(prims) == 2
    assert [m["name"] for m in gltf["materials"]] == [
        "atlas_relief_projection", "atlas_generated_vertex_colour"]
    assert all("alphaMode" not in m for m in gltf["materials"])
    pos = gltf["accessors"][prims[0]["attributes"]["POSITION"]]
    # The hidden face's 3 vertices are duplicated so the sides share none.
    assert pos["count"] == len(VERTS) + 3
    assert "COLOR_0" in prims[0]["attributes"]
    counts = [gltf["accessors"][p["indices"]]["count"] for p in prims]
    assert counts == [3, 3]


def test_glb_without_colour_is_unchanged(tmp_path):
    mesh = ReliefMesh(vertices=VERTS, faces=FACES, uvs=UVS)
    gltf = _glb_json(export_relief_mesh_glb(mesh, tmp_path)["glb"])
    assert len(gltf["meshes"][0]["primitives"]) == 1
    assert "COLOR_0" not in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_obj_writes_generated_material_and_vertex_colours(tmp_path):
    mesh = ReliefMesh(vertices=VERTS, faces=FACES, uvs=UVS, vertex_colors=COLS,
                      photo_weight=PW)
    out = export_relief_mesh(mesh, tmp_path)
    text = open(out["obj"], encoding="utf-8").read()
    vlines = [ln for ln in text.splitlines() if ln.startswith("v ")]
    assert len(vlines) == len(VERTS) + 3 and all(len(ln.split()) == 7 for ln in vlines)
    assert "usemtl atlas_relief_projection_generated" in text
    # Photo-side vertices are white so a multiplying reader leaves the plate alone.
    assert vlines[0].split()[4:] == ["1.0000", "1.0000", "1.0000"]
    assert "newmtl atlas_relief_projection_generated" in open(out["mtl"]).read()


def test_usd_writes_generated_mesh_with_display_colour(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from atlas_camera.core.schema import (
        AtlasExtrinsics,
        AtlasIntrinsics,
        AtlasSolve,
        LatentCamera,
    )
    from atlas_camera.exporters.usd_exporter import USDExporter

    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=64, image_height=48, focal_length_mm=35.0,
                                   sensor_width_mm=36.0),
        extrinsics=AtlasExtrinsics()))
    solve.projection_scene.proxy_geometry.append(_prim())
    path = USDExporter().export_proxy_scene(solve, tmp_path / "scene.usda")
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/AtlasProjectionScene/pixal3d_object"))
    assert mesh and len(mesh.GetPointsAttr().Get()) == len(VERTS)
    dc = mesh.GetDisplayColorPrimvar()
    assert dc.GetInterpolation() == UsdGeom.Tokens.vertex
    assert len(dc.Get()) == len(VERTS)


def test_usd_prim_names_are_valid_and_unique(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from atlas_camera.core.schema import (
        AtlasExtrinsics,
        AtlasIntrinsics,
        AtlasSolve,
        LatentCamera,
    )
    from atlas_camera.exporters.usd_exporter import USDExporter

    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=64, image_height=48, focal_length_mm=35.0,
                                   sensor_width_mm=36.0),
        extrinsics=AtlasExtrinsics()))
    # Two imports both named "object" -> the same primitive name, plus names
    # that are not Sdf identifiers at all.
    for name in ("pixal3d_object", "pixal3d_object", "2nd obj.v2", "", "atlas_projection_plane"):
        p = _prim()
        p.name = name
        solve.projection_scene.proxy_geometry.append(p)
    path = USDExporter().export_proxy_scene(solve, tmp_path / "scene.usda")
    stage = Usd.Stage.Open(str(path))
    root = stage.GetPrimAtPath("/AtlasProjectionScene")
    names = [c.GetName() for c in root.GetChildren()]
    # The built-in ground plane keeps its name; a primitive claiming it is suffixed.
    assert names == ["atlas_projection_plane", "pixal3d_object", "pixal3d_object_2",
                     "_2nd_obj_v2", "proxy_3", "atlas_projection_plane_2"]
    for n in names[1:]:
        assert len(UsdGeom.Mesh(root.GetChild(n)).GetPointsAttr().Get()) == len(VERTS)
