"""AtlasSceneTo3D: layered GLB + EXR sidecars + Load3D camera/model info."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
import pytest

from atlas_camera.core.load3d_camera import (
    identity_model_info,
    load3d_camera_info,
    rotation_to_quaternion,
)
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)
from atlas_camera.exporters.scene_glb import (
    SceneLayer,
    build_scene_layers,
    read_glb_json,
    write_scene_glb,
)

W, H, FX = 320, 200, 260.0


def _view(pitch_deg=-10.0, yaw_deg=20.0, pos=(1.0, 1.6, 2.0)):
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    c2w = np.eye(4)
    c2w[:3, :3] = ry @ rx
    c2w[:3, 3] = pos
    return np.linalg.inv(c2w)


def _quat_to_matrix(q):
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# --- camera ------------------------------------------------------------------

def test_camera_info_reprojects_like_the_solve():
    view = _view()
    info = load3d_camera_info(view_matrix=view, fy=FX, image_width=W, image_height=H)
    # A three.js-style perspective camera rebuilt from the payload alone.
    R = _quat_to_matrix(info["quaternion"])
    pos = np.array([info["position"][k] for k in "xyz"])
    f_from_fov = (H / 2.0) / math.tan(math.radians(info["fov"]) / 2.0)
    pts = np.array([[0.5, 0.3, -6.0], [-2.0, 1.0, -9.0], [3.0, -0.5, -4.0]])
    cam_a = (pts - pos) @ R                     # world -> camera via the payload
    cam_b = pts @ view[:3, :3].T + view[:3, 3]  # world -> camera via the solve
    for cam in (cam_a, cam_b):
        assert (cam[:, 2] < 0).all()
    ua = W / 2 + f_from_fov * cam_a[:, 0] / -cam_a[:, 2]
    ub = W / 2 + FX * cam_b[:, 0] / -cam_b[:, 2]
    va = H / 2 - f_from_fov * cam_a[:, 1] / -cam_a[:, 2]
    vb = H / 2 - FX * cam_b[:, 1] / -cam_b[:, 2]
    assert np.max(np.abs(ua - ub)) < 0.01 and np.max(np.abs(va - vb)) < 0.01
    assert info["aspect"] == pytest.approx(W / H)
    assert info["cameraType"] == "perspective" and info["zoom"] == 1


def test_identity_view_has_identity_quaternion():
    q = rotation_to_quaternion(np.eye(3))
    assert q == pytest.approx({"x": 0, "y": 0, "z": 0, "w": 1})


def test_model_info_is_identity():
    (m,) = identity_model_info(1)
    assert m["quaternion"]["w"] == 1.0 and m["scale"]["x"] == 1.0


def test_camera_info_rejects_3x3():
    with pytest.raises(ValueError, match="4x4"):
        load3d_camera_info(view_matrix=np.eye(3), fy=FX, image_width=W, image_height=H)


# --- GLB ---------------------------------------------------------------------

def _png(color=(200, 100, 50), size=(8, 4)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


QUAD_V = np.array([[0, 0, -5], [1, 0, -5], [0, 1, -5], [1, 1, -6]], dtype=np.float32)
QUAD_F = np.array([[0, 1, 2], [1, 3, 2]])
QUAD_UV = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)


def test_glb_has_one_textured_node_per_layer_and_vertex_colour_only_on_generated(tmp_path):
    layers = [
        SceneLayer("relief", QUAD_V, QUAD_F, QUAD_UV, _png(), extras={"exr": "a.exr"}),
        SceneLayer("pixal3d_object", QUAD_V + [0, 0, 1], QUAD_F, QUAD_UV, _png((10, 20, 30)),
                   vertex_colors=np.full((4, 3), 0.4), photo_weight=np.array([1, 1, 1, 0.0])),
    ]
    out = write_scene_glb(layers, tmp_path / "s.glb")
    g = read_glb_json(out["glb"])
    assert [n["name"] for n in g["nodes"]] == ["relief", "pixal3d_object"]
    assert len(g["images"]) == 2 and all(i["mimeType"] == "image/png" for i in g["images"])
    assert all("KHR_materials_unlit" in m["extensions"] for m in g["materials"])
    relief_prims = g["meshes"][0]["primitives"]
    assert len(relief_prims) == 1 and "COLOR_0" not in relief_prims[0]["attributes"]
    assert g["materials"][relief_prims[0]["material"]]["extras"] == {"exr": "a.exr"}
    obj_prims = g["meshes"][1]["primitives"]
    assert len(obj_prims) == 2 and "COLOR_0" in obj_prims[0]["attributes"]
    hidden = g["materials"][obj_prims[1]["material"]]
    assert "baseColorTexture" not in hidden["pbrMetallicRoughness"]
    assert out["layers"][1]["vertex_colour"] is True


def test_glb_refuses_empty_scene(tmp_path):
    with pytest.raises(ValueError, match="no layer"):
        write_scene_glb([SceneLayer("x", np.zeros((0, 3)), np.zeros((0, 3)))], tmp_path / "e.glb")


# --- assembly + sidecars + node ------------------------------------------------

def _prim(name, verts, extra=None):
    return AtlasProxyPrimitive(
        name=name, primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
        material="atlas_projection_proxy",
        metadata={"role": PROXY_ROLE, "vertices": np.asarray(verts).reshape(-1).tolist(),
                  "faces": QUAD_F.reshape(-1).tolist(),
                  "uvs": QUAD_UV.reshape(-1).tolist(), **(extra or {})})


def _solve():
    intr = AtlasIntrinsics(image_width=W, image_height=H, focal_length_mm=35.0,
                           sensor_width_mm=36.0, fx_px=FX, fy_px=FX, cx_px=W / 2, cy_px=H / 2)
    extr = AtlasExtrinsics(camera_view_matrix=tuple(map(tuple, _view(pitch_deg=-15, yaw_deg=0,
                                                                    pos=(0, 1.6, 0)))))
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    solve.projection_scene.proxy_geometry.append(_prim("projection_relief_mesh", QUAD_V))
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="projection_backdrop", primitive_type="plane", dimensions=(10.0, 10.0, 0.0),
        material="m", metadata={"role": PROXY_ROLE}))
    uri = "data:image/png;base64," + base64.b64encode(_png((40, 90, 160), (W, H))).decode()
    solve.projection_sources.append(ProjectionSource(
        camera=solve.camera, name="clean_plate_geo", image_b64=uri,
        proxy_geometry=[_prim("clean_plate_geo_relief_mesh", QUAD_V - [0, 0, 3])],
        metadata={"projection_mode": "clean_plate", "frame_outpaint_px": 0}))
    return solve


def _primary():
    from PIL import Image
    return Image.new("RGB", (W, H), (230, 200, 120))


def test_layers_cover_primary_and_sources_and_skip_analytic(tmp_path):
    layers, sidecars, notes = build_scene_layers(_solve(), _primary(), exr_dir=None)
    assert [l.name for l in layers] == ["projection_relief_mesh",
                                        "clean_plate_geo/clean_plate_geo_relief_mesh"]
    assert all(l.image_bytes for l in layers) and not sidecars


def test_exr_sidecars_are_tagged_and_named_in_extras(tmp_path):
    pytest.importorskip("OpenImageIO")
    layers, sidecars, notes = build_scene_layers(_solve(), _primary(), exr_dir=tmp_path,
                                                 exr_prefix="t")
    assert {s["plate"] for s in sidecars} == {"primary", "clean_plate_geo"}
    assert all(s["exr_origin"] == "linearised_display_plate" and not s["scene_referred"]
               for s in sidecars)
    import OpenImageIO as oiio
    buf = oiio.ImageBuf(str(tmp_path / "t_primary.exr"))
    px = buf.get_pixels(oiio.FLOAT)
    # sRGB 230/255 -> linear ~0.791 (EOTF, not a straight divide)
    assert px[..., 0].mean() == pytest.approx(((230 / 255 + 0.055) / 1.055) ** 2.4, abs=2e-3)
    assert layers[0].extras["exr"] == "t_primary.exr"


def test_node_writes_glb_and_returns_the_three_sockets(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_scene3d

    monkeypatch.setattr(nodes_scene3d, "_output_paths",
                        lambda prefix: (tmp_path, "scene_00001"))
    img = torch.rand(1, H, W, 3)
    out = nodes_scene3d.AtlasSceneTo3D().export(_solve(), img)
    model, info, cam, glb_path, report = out["result"]
    assert (tmp_path / "scene_00001.glb").is_file() and glb_path.endswith(".glb")
    assert getattr(model, "format", "") == "glb"
    assert info == identity_model_info(1)
    assert set(cam) >= {"position", "target", "quaternion", "fov", "aspect", "near", "far"}
    assert cam["far"] > cam["near"] > 0
    g = read_glb_json(glb_path)
    assert len(g["nodes"]) == 2
    assert "EXR scene_00001_primary.exr" in report and "NOT scene-referred" in report
