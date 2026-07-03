"""Tests for the depth relief mesh (DCC handoff geometry).

Analytic depth scenes (same construction as test_proxy_geometry): level camera
at (0, h, 0), ground Y=0, wall at known z — expectations computed in the same
world the mesh is built in.
"""

import numpy as np
import pytest

from atlas_camera.core.relief_mesh import (
    build_relief_mesh,
    estimate_ground_scale,
)
from atlas_camera.exporters.relief_mesh_exporter import (
    export_relief_mesh,
    export_relief_mesh_glb,
)

W = H = 512
FX = FY = 500.0
CX = CY = 256.0
SKY = 60.0
EDGE_REL = 0.5


def _view_matrix(h):
    """Level camera at (0, h, 0), identity rotation (world→cam translation only)."""
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _scene_depth(h=1.6, wall_z=-10.0, wall_h=3.0):
    """Analytic z-depth: min over sky, ground plane Y=0, wall at z=wall_z."""
    _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    depth = np.full((H, W), SKY)

    tg = np.full((H, W), np.inf)
    hit = dy < -1e-6
    tg[hit] = -h / dy[hit]

    tw = np.full((H, W), np.inf)
    t = -wall_z
    y_at = h + dy * t
    vis = (y_at >= 0.0) & (y_at <= wall_h)
    tw[vis] = t

    stacked = np.stack([depth,
                        np.where(np.isfinite(tg), tg, SKY),
                        np.where(np.isfinite(tw), tw, SKY)])
    return stacked.min(axis=0)


def _build(depth, h=1.6, **kw):
    kw.setdefault("depth_edge_rel", EDGE_REL)
    return build_relief_mesh(
        depth, view_matrix=_view_matrix(h), fx=FX, fy=FY, cx=CX, cy=CY, **kw
    )


def test_mesh_is_well_formed():
    mesh = _build(_scene_depth(wall_z=-10.0))
    assert len(mesh.vertices) > 100
    assert len(mesh.faces) > 100
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < len(mesh.vertices)
    assert len(mesh.uvs) == len(mesh.vertices)
    assert mesh.uvs.min() >= 0.0 and mesh.uvs.max() <= 1.0


def test_no_face_spans_a_silhouette():
    # Every kept triangle's camera distances agree within the tear threshold —
    # the wall(10m)/sky(60m) silhouette must be a hole, not a rubber sheet.
    mesh = _build(_scene_depth(wall_z=-10.0, wall_h=3.0))
    cam = np.array([0.0, 1.6, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    tri = dist[mesh.faces]  # (M, 3)
    ratio = tri.max(axis=1) / np.maximum(tri.min(axis=1), 1e-6)
    assert float(ratio.max()) <= 1.0 + EDGE_REL + 0.05
    assert mesh.stats["torn_fraction"] > 0.0  # the silhouette actually tore


def test_ground_lands_on_y_zero_and_nothing_below():
    mesh = _build(_scene_depth(wall_z=-10.0))
    ys = mesh.vertices[:, 1]
    assert float(ys.min()) > -0.2          # nothing under the ground plane
    assert float(np.abs(ys).min()) < 0.05  # ground vertices sit at Y≈0


def test_faces_wind_toward_camera():
    mesh = _build(_scene_depth(wall_z=-10.0))
    cam = np.array([0.0, 1.6, 0.0])
    v = mesh.vertices
    f = mesh.faces[:: max(1, len(mesh.faces) // 200)]  # sample
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    n = np.cross(e1, e2)
    centroid = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    to_cam = cam[None, :] - centroid
    dots = np.einsum("ij,ij->i", n, to_cam)
    assert (dots > 0).mean() > 0.99


def test_scale_rescales_about_camera():
    depth = _scene_depth(wall_z=-10.0)
    m1 = _build(depth, scale=1.0)
    m2 = _build(depth, scale=0.5)
    cam = np.array([0.0, 1.6, 0.0])
    d1 = np.linalg.norm(m1.vertices - cam, axis=1)
    d2 = np.linalg.norm(m2.vertices - cam, axis=1)
    assert np.median(d2 / np.maximum(d1, 1e-6)) == pytest.approx(0.5, abs=0.02)


def test_estimate_ground_scale_reconciles_doubled_depth():
    depth = _scene_depth(wall_z=-10.0) * 2.0
    scale, info = estimate_ground_scale(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY
    )
    assert scale == pytest.approx(0.5, abs=0.05)


def test_floor_clamp_pulls_outliers_along_the_ray():
    # Punch some depth outliers that would land far below the ground plane.
    depth = _scene_depth(wall_z=-10.0)
    depth[400:410, 200:210] *= 3.0  # below-horizon rays, 3x too deep → under ground
    mesh = _build(depth)
    assert float(mesh.vertices[:, 1].min()) >= -0.3  # clamped near the floor


def test_single_pixel_depth_spikes_are_removed():
    # A lone bad depth pixel (common in AI-image depth) must not become a mesh
    # spike — the 3x3 median sampling swallows it.
    depth = _scene_depth(wall_z=-10.0)
    for r, c in ((300, 128), (350, 384), (420, 256)):
        depth[r, c] = 2.0  # spike far in front of the true surface
    mesh = _build(depth)
    cam = np.array([0.0, 1.6, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    assert float(dist.min()) > 2.5  # no vertex pulled to the spike depth


def test_no_stretched_shard_triangles():
    # World-space edge cap: no triangle edge may stretch far beyond the local
    # sample spacing (the source of the spiky shards).
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)
    mesh = _build(depth)
    v = mesh.vertices
    f = mesh.faces
    edges = np.concatenate([
        np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1),
    ])
    cam = np.array([0.0, 1.6, 0.0])
    dmax = float(np.linalg.norm(v - cam, axis=1).max())
    step = max(1, round(max(H, W) / 96))
    assert float(edges.max()) <= 12.0 * dmax * step / FX + 0.06


def test_grid_density_follows_grid_long_edge():
    depth = _scene_depth(wall_z=-10.0)
    small = _build(depth, grid_long_edge=32)
    big = _build(depth, grid_long_edge=128)
    assert len(big.faces) > 4 * len(small.faces)


def test_obj_export_round_trip(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=48)
    texture = Image.new("RGB", (64, 64), (200, 150, 100))
    paths = export_relief_mesh(mesh, tmp_path, texture=texture)

    obj_text = open(paths["obj"], encoding="utf-8").read().splitlines()
    n_v = sum(1 for line in obj_text if line.startswith("v "))
    n_vt = sum(1 for line in obj_text if line.startswith("vt "))
    n_f = sum(1 for line in obj_text if line.startswith("f "))
    assert n_v == len(mesh.vertices)
    assert n_vt == len(mesh.uvs)
    assert n_f == len(mesh.faces)
    # Face indices are 1-based and within range.
    for line in obj_text:
        if line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            assert all(1 <= i <= n_v for i in idx)
            break
    mtl_text = open(paths["mtl"], encoding="utf-8").read()
    assert "map_Kd" in mtl_text
    import os
    assert os.path.isfile(paths["texture"])


def test_relief_mesh_rides_the_proxy_payload():
    import json

    from atlas_camera.core.proxy_geometry import (
        relief_mesh_primitive,
        serialize_proxy_geometry,
    )
    from atlas_camera.core.schema import AtlasProjectionScene

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=32)
    scene = AtlasProjectionScene()
    scene.proxy_geometry.append(relief_mesh_primitive(mesh))
    payload = serialize_proxy_geometry(scene)
    assert len(payload) == 1
    entry = payload[0]
    assert entry["type"] == "mesh"
    assert len(entry["vertices"]) == 3 * len(mesh.vertices)
    assert len(entry["faces"]) == 3 * len(mesh.faces)
    assert len(entry["uvs"]) == 2 * len(mesh.uvs)
    assert max(entry["faces"]) < len(mesh.vertices)
    json.dumps(payload)  # JSON-safe


def test_glb_export_is_valid_gltf2(tmp_path):
    pytest.importorskip("PIL")
    import json
    import struct

    from PIL import Image

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=48)
    texture = Image.new("RGB", (64, 64), (90, 120, 180))
    paths = export_relief_mesh_glb(mesh, tmp_path, texture=texture)

    raw = open(paths["glb"], "rb").read()
    magic, version, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67 and version == 2
    assert total == len(raw)

    json_len, json_type = struct.unpack_from("<II", raw, 12)
    assert json_type == 0x4E4F534A
    gltf = json.loads(raw[20:20 + json_len])
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, bin_off)
    assert bin_type == 0x004E4942
    assert gltf["buffers"][0]["byteLength"] == bin_len

    # Accessors match the mesh; POSITION carries required min/max.
    acc = gltf["accessors"]
    assert acc[0]["type"] == "VEC3"
    assert acc[0]["count"] == len(mesh.vertices)
    assert "min" in acc[0] and "max" in acc[0]
    assert acc[1]["count"] == len(mesh.uvs)
    assert acc[2]["count"] == mesh.faces.size

    # Texture embedded, unlit-tagged material, all bufferViews inside the bin.
    assert gltf["images"][0]["mimeType"] == "image/png"
    assert "KHR_materials_unlit" in gltf["materials"][0]["extensions"]
    for bv in gltf["bufferViews"]:
        assert bv["byteOffset"] + bv["byteLength"] <= bin_len

    # glTF V origin is top-left — flipped from the mesh's OBJ-convention UVs.
    uv_bv = gltf["bufferViews"][acc[1]["bufferView"]]
    uv_bytes = raw[bin_off + 8 + uv_bv["byteOffset"]:
                   bin_off + 8 + uv_bv["byteOffset"] + uv_bv["byteLength"]]
    uvs = np.frombuffer(uv_bytes, dtype=np.float32).reshape(-1, 2)
    assert np.allclose(uvs[:, 1], 1.0 - mesh.uvs[:, 1], atol=1e-6)
