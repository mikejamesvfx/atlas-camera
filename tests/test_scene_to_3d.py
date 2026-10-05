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
    assert [layer.name for layer in layers] == ["projection_relief_mesh",
                                        "clean_plate_geo/clean_plate_geo_relief_mesh"]
    assert all(layer.image_bytes for layer in layers) and not sidecars


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

    monkeypatch.setattr(nodes_scene3d, "output_paths",
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


def _hdr_exr(path, w=W, h=H, value=6.0):
    """A float ACEScg plate with radiance far above display white."""
    from atlas_camera.plate.oiio_io import write_exr
    px = np.full((h, w, 3), value, dtype=np.float32)
    write_exr(str(path), px, bit_depth="half", source_colorspace="ACEScg")
    return path


def _exr_pixels(path):
    import OpenImageIO as oiio
    return oiio.ImageBuf(str(path)).get_pixels(oiio.FLOAT)


def test_hdr_plate_replaces_the_primary_exr_only(tmp_path):
    pytest.importorskip("OpenImageIO")
    hdr = _hdr_exr(tmp_path / "hdr.exr")
    out = tmp_path / "out"
    _, sidecars, _ = build_scene_layers(_solve(), _primary(), exr_dir=out, exr_prefix="t",
                                        primary_hdr_path=str(hdr))
    by = {s["plate"]: s for s in sidecars}
    assert by["primary"]["exr_origin"] == "hdr_plate" and by["primary"]["scene_referred"]
    assert by["primary"]["exr_source"] == "hdr.exr"
    assert _exr_pixels(out / "t_primary.exr")[..., 0].mean() == pytest.approx(6.0, rel=1e-3)
    # the other layers keep their own (display) plates
    assert by["clean_plate_geo"]["exr_origin"] == "linearised_display_plate"


def test_a_wrong_size_hdr_plate_falls_back_and_says_so(tmp_path):
    pytest.importorskip("OpenImageIO")
    hdr = _hdr_exr(tmp_path / "small.exr", w=W // 2, h=H // 2)
    _, sidecars, _ = build_scene_layers(_solve(), _primary(), exr_dir=tmp_path / "o",
                                        exr_prefix="t", primary_hdr_path=str(hdr))
    prim = next(s for s in sidecars if s["plate"] == "primary")
    assert prim["exr_origin"] == "linearised_display_plate" and not prim["scene_referred"]
    assert "NOT used" in prim["exr_note"] and f"{W // 2}x{H // 2}" in prim["exr_note"]


def _node_export(tmp_path, monkeypatch, **kw):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_object_mesh, nodes_scene3d

    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    monkeypatch.setattr(nodes_scene3d, "output_paths", lambda prefix: (out, "scene_00001"))
    monkeypatch.setattr(nodes_scene3d, "output_root", lambda: out)
    monkeypatch.setattr(nodes_object_mesh, "_comfy_read_roots", lambda: [("output", out)])
    return out, nodes_scene3d.AtlasSceneTo3D().export(_solve(), torch.rand(1, H, W, 3), **kw)


def test_node_writes_the_hdr_plate_as_the_primary_exr(tmp_path, monkeypatch):
    pytest.importorskip("OpenImageIO")
    out = tmp_path / "output"
    out.mkdir()
    _hdr_exr(out / "hdr_plate_00001.exr")
    out, res = _node_export(tmp_path, monkeypatch, hdr_plate_path="hdr_plate_00001.exr")
    report = res["result"][4]
    assert "EXR scene_00001_primary.exr" in report and "HDR plate hdr_plate_00001.exr" in report
    assert _exr_pixels(out / "scene_00001_primary.exr").max() > 1.0
    extras = [m.get("extras", {}) for m in read_glb_json(res["result"][3])["materials"]]
    assert any(e.get("exr_origin") == "hdr_plate" for e in extras)


@pytest.mark.parametrize("path,write_exr,needle", [
    ("missing.exr", True, "not found"),
    ("C:/elsewhere/hdr.exr" if __import__("os").name == "nt" else "/elsewhere/hdr.exr",
     True, "refused"),
    ("hdr.exr", False, "write_exr is off"),
])
def test_an_unusable_hdr_plate_is_a_visible_warning(tmp_path, monkeypatch, path, write_exr,
                                                    needle):
    pytest.importorskip("OpenImageIO")
    monkeypatch.delenv("ATLAS_ALLOW_ABSOLUTE_READS", raising=False)
    monkeypatch.delenv("ATLAS_PROJECT_ROOT", raising=False)
    _, res = _node_export(tmp_path, monkeypatch, hdr_plate_path=path, write_exr=write_exr)
    report = res["result"][4]
    assert "WARNING: hdr_plate_path" in report and needle in report
    assert "HDR plate" not in report.replace("hdr_plate_path", "")


def test_hdr_plate_path_reruns_when_the_file_changes(tmp_path, monkeypatch):
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_object_mesh
    from atlas_camera.comfy.nodes_scene3d import AtlasSceneTo3D

    monkeypatch.setattr(nodes_object_mesh, "_comfy_read_roots", lambda: [("output", tmp_path)])
    assert AtlasSceneTo3D.IS_CHANGED() == AtlasSceneTo3D.IS_CHANGED(hdr_plate_path="")
    before = AtlasSceneTo3D.IS_CHANGED(hdr_plate_path="p.exr")      # missing
    _hdr_exr(tmp_path / "p.exr")
    appeared = AtlasSceneTo3D.IS_CHANGED(hdr_plate_path="p.exr")
    _hdr_exr(tmp_path / "p.exr", value=9.0, w=W + 2)                # rewritten, new size
    assert len({before, appeared, AtlasSceneTo3D.IS_CHANGED(hdr_plate_path="p.exr")}) == 3


def _inflate_plan(monkeypatch, nbytes):
    """Pretend the planned GLB is ``nbytes`` (the pre-build size check)."""
    from atlas_camera.exporters import scene_glb

    real = scene_glb.plan_scene_glb
    calls = []

    def inflated(layers, **kwargs):
        plan = real(layers, **kwargs)
        calls.append(plan)
        return {**plan, "bytes": int(nbytes)}

    monkeypatch.setattr(scene_glb, "plan_scene_glb", inflated)
    return calls


def _spy_sidecar_writes(monkeypatch):
    """Record every EXR / PLY sidecar write the export attempts."""
    from atlas_camera.exporters import scene_glb

    calls = []
    real_exr, real_ply = scene_glb._write_exr_sidecar, scene_glb.write_float_ply
    monkeypatch.setattr(scene_glb, "_write_exr_sidecar",
                        lambda path, *a: calls.append(path) or real_exr(path, *a))
    monkeypatch.setattr(scene_glb, "write_float_ply",
                        lambda path, *a: calls.append(path) or real_ply(path, *a))
    return calls


@pytest.mark.parametrize("budget, refused", [(1024, True), (0, False), (4096, False),
                                             (16384, False)])
def test_node_refuses_a_glb_over_the_size_budget(tmp_path, monkeypatch, budget, refused):
    torch = pytest.importorskip("torch")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_scene3d
    from atlas_camera.exporters import scene_glb

    _inflate_plan(monkeypatch, 1_500_000_000)
    sidecar_writes = _spy_sidecar_writes(monkeypatch)
    built = []
    real_write = scene_glb.write_scene_glb
    monkeypatch.setattr(scene_glb, "write_scene_glb",
                        lambda *a, **k: built.append(1) or real_write(*a, **k))
    monkeypatch.setattr(nodes_scene3d, "output_paths",
                        lambda prefix: (tmp_path, "scene_00001"))
    node = nodes_scene3d.AtlasSceneTo3D()
    if refused:
        with pytest.raises(ValueError, match=r"1500 MB, over the 1024 MB budget") as exc:
            node.export(_solve(), torch.rand(1, H, W, 3), max_glb_mb=budget)
        msg = str(exc.value)
        assert not (tmp_path / "scene_00001.glb").exists() and not built
        # Per-layer / per-plate MB breakdown ...
        assert "Per layer:" in msg and "projection_relief_mesh" in msg
        assert "plate primary" in msg and "plate clean_plate_geo" in msg
        # ... and refused BEFORE any sidecar was written: none to clean up.
        assert sidecar_writes == [] and "Removed the sidecars" not in msg
        assert not list(tmp_path.iterdir())
    else:
        report = node.export(_solve(), torch.rand(1, H, W, 3),
                             max_glb_mb=budget)["result"][4]
        assert (tmp_path / "scene_00001.glb").is_file() and built
        assert sidecar_writes
        # 16384 is past the uint32 GLB ceiling: clamped to 4294 MB, 1.5 GB fits.
        clamped = "max_glb_mb 16384 is past the GLB format's 4 GiB length limit"
        assert (clamped in report) == (budget == 16384)
        if budget == 16384:
            assert "clamped to 4294 MB" in report


def test_budget_widget_keeps_its_16384_range_and_runtime_clamps():
    from atlas_camera.comfy.nodes_scene3d import (
        AtlasSceneTo3D,
        GLB_FORMAT_MAX_MB,
        _effective_budget_mb,
    )

    opt = AtlasSceneTo3D.INPUT_TYPES()["optional"]["max_glb_mb"][1]
    assert opt["max"] == 16384           # saved graphs up to 16384 keep validating
    assert GLB_FORMAT_MAX_MB * 1_000_000 < 2 ** 32
    budget, note = _effective_budget_mb(16384)
    assert budget == GLB_FORMAT_MAX_MB and "clamped to 4294 MB" in note
    assert _effective_budget_mb(1024) == (1024, "")
    assert _effective_budget_mb(0) == (0, "")


def test_over_budget_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    from atlas_camera.exporters import scene_glb

    _inflate_plan(monkeypatch, 5_000_000_000)          # past uint32, no budget set
    monkeypatch.setattr(scene_glb, "open",
                        lambda *a, **k: pytest.fail("the GLB was opened for writing"),
                        raising=False)
    layers = [SceneLayer("relief", QUAD_V, QUAD_F, QUAD_UV, _png())]
    with pytest.raises(scene_glb.GLBBudgetError, match="4 GiB") as exc:
        scene_glb.write_scene_glb(layers, tmp_path / "big.glb")
    assert not (tmp_path / "big.glb").exists()
    assert exc.value.plan["layers"][0]["name"] == "relief"
    with pytest.raises(scene_glb.GLBBudgetError, match="1 MB budget"):
        scene_glb.write_scene_glb(layers, tmp_path / "big.glb", max_bytes=1_000_000)


# --- review (b): the size plan is EXACT, and the final size is enforced -------

def test_a_huge_layer_name_is_counted_and_refused(tmp_path):
    """Codex repro: the old JSON estimate ignored names, so one triangle with a
    400,000-char name planned 2,636 B and wrote 1,200,872 B past a 1 MB limit."""
    from atlas_camera.exporters import scene_glb

    name = "n" * 400_000
    layers = [SceneLayer(name, QUAD_V[:3], QUAD_F[:1])]
    plan = scene_glb.plan_scene_glb(layers)
    assert plan["bytes"] > 1_000_000 and plan["json_bytes"] > 1_000_000
    with pytest.raises(scene_glb.GLBBudgetError, match="1 MB budget") as exc:
        scene_glb.write_scene_glb(layers, tmp_path / "n.glb", max_bytes=1_000_000)
    assert not (tmp_path / "n.glb").exists()
    msg = str(exc.value)
    assert "glTF JSON 1.2 MB" in msg and len(msg) < 2000    # the name is clipped
    # Unbudgeted it writes, and the plan was the real size to the byte.
    out = scene_glb.write_scene_glb(layers, tmp_path / "n.glb")
    assert out["bytes"] == plan["bytes"] == (tmp_path / "n.glb").stat().st_size


def _plan_cases():
    png, other = _png(), _png((1, 2, 3), (16, 16))
    return {
        "textured": [SceneLayer("relief", QUAD_V, QUAD_F, QUAD_UV, png,
                                extras={"exr": "a.exr", "atlas_plate": "primary"})],
        "vertex_coloured": [SceneLayer("pixal3d_object", QUAD_V, QUAD_F, QUAD_UV, png,
                                       vertex_colors=np.full((4, 3), 0.4),
                                       photo_weight=np.array([1, 1, 1, 0.0]))],
        "shared_image": [SceneLayer("a", QUAD_V, QUAD_F, QUAD_UV, png),
                         SceneLayer("b", QUAD_V + [0, 0, 1], QUAD_F, QUAD_UV, png),
                         SceneLayer("c", QUAD_V - [0, 0, 1], QUAD_F, QUAD_UV, other),
                         SceneLayer("untextured", QUAD_V, QUAD_F)],
    }


@pytest.mark.parametrize("case", ["textured", "vertex_coloured", "shared_image"])
def test_plan_bytes_cover_the_written_file(tmp_path, case):
    from atlas_camera.exporters import scene_glb

    layers = _plan_cases()[case]
    plan = scene_glb.plan_scene_glb(layers)
    path = tmp_path / f"{case}.glb"
    out = scene_glb.write_scene_glb(layers, path)
    size = path.stat().st_size
    assert plan["bytes"] >= size and out["bytes"] == size
    assert plan["bytes"] == size              # exact, not merely conservative
    _check_glb_structure(path)


def test_the_final_written_size_is_enforced_too(tmp_path, monkeypatch):
    """A plan that under-counts (simulated) cannot sneak a GLB past the
    budget: the written file is measured, deleted, and its real size named."""
    from atlas_camera.exporters import scene_glb

    _inflate_plan(monkeypatch, 10)
    layers = [SceneLayer("n" * 50_000, QUAD_V[:3], QUAD_F[:1])]
    with pytest.raises(scene_glb.GLBBudgetError,
                       match=r"written GLB is .* bytes\), over the 1000 byte budget") as exc:
        scene_glb.write_scene_glb(layers, tmp_path / "x.glb", max_bytes=1000)
    assert not (tmp_path / "x.glb").exists()
    assert exc.value.plan["bytes"] > 50_000


def test_a_real_over_budget_scene_writes_no_sidecar(tmp_path, monkeypatch):
    """No monkeypatched plan: a 1.5M-char mesh name makes the GLB really
    exceed 1 MB; the node refuses before any EXR / PLY / GLB is written."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_scene3d

    solve = _solve()
    solve.projection_scene.proxy_geometry[0].name = "x" * 1_500_000
    sidecar_writes = _spy_sidecar_writes(monkeypatch)
    folder = tmp_path / "out"
    monkeypatch.setattr(nodes_scene3d, "output_paths",
                        lambda prefix: (folder, "scene_00001"))
    with pytest.raises(ValueError, match=r"over the 1 MB budget") as exc:
        nodes_scene3d.AtlasSceneTo3D().export(solve, torch.rand(1, H, W, 3), max_glb_mb=1)
    assert sidecar_writes == []
    assert not list(folder.iterdir())
    assert "Per layer:" in str(exc.value) and "glTF JSON" in str(exc.value)


def test_large_glb_report_warns(tmp_path, monkeypatch):
    """Not refused (no budget), but past GLB_WARN_MB: the report says so."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_scene3d
    from atlas_camera.exporters import scene_glb

    real_write = scene_glb.write_scene_glb

    def inflated_write(*a, **k):
        out = real_write(*a, **k)
        return {**out, "bytes": (nodes_scene3d.GLB_WARN_MB + 100) * 1_000_000}

    monkeypatch.setattr(scene_glb, "write_scene_glb", inflated_write)
    monkeypatch.setattr(nodes_scene3d, "output_paths",
                        lambda prefix: (tmp_path, "scene_00001"))
    report = nodes_scene3d.AtlasSceneTo3D().export(
        _solve(), torch.rand(1, H, W, 3), write_exr=False, max_glb_mb=0)["result"][4]
    assert "(300.0 MB," in report
    assert "warning: large GLB - the browser 3D viewer may be slow to load it" in report


# --- one embedded image per source -------------------------------------------

def test_layers_sharing_a_source_share_one_image(tmp_path):
    png = _png()
    layers = [SceneLayer("a", QUAD_V, QUAD_F, QUAD_UV, png, extras={"atlas_plate": "primary"}),
              SceneLayer("b", QUAD_V + [0, 0, 1], QUAD_F, QUAD_UV, png,
                         extras={"atlas_plate": "primary"}),
              SceneLayer("c", QUAD_V - [0, 0, 1], QUAD_F, QUAD_UV, _png((1, 2, 3)))]
    out = write_scene_glb(layers, tmp_path / "s.glb")
    g = read_glb_json(out["glb"])
    assert len(g["images"]) == 2 and len(g["textures"]) == 2 and out["images"] == 2
    tex = [g["materials"][m["primitives"][0]["material"]]["pbrMetallicRoughness"]
           ["baseColorTexture"]["index"] for m in g["meshes"]]
    assert tex[0] == tex[1] != tex[2]
    assert g["images"][0]["name"] == "primary"


def test_scene_with_many_primary_meshes_embeds_the_primary_plate_once(tmp_path):
    solve = _solve()
    solve.projection_scene.proxy_geometry.append(_prim("second_mesh", QUAD_V + [0, 0, 1]))
    layers, _, _ = build_scene_layers(solve, _primary(), exr_dir=None)
    out = write_scene_glb(layers, tmp_path / "s.glb")
    g = read_glb_json(out["glb"])
    assert len(layers) == 3 and len(g["images"]) == 2     # primary + clean plate
    assert layers[0].image_bytes is layers[1].image_bytes
    image_views = [g["bufferViews"][i["bufferView"]]["byteLength"] for i in g["images"]]
    assert sorted(image_views) == sorted([len(layers[0].image_bytes),
                                          len(layers[2].image_bytes)])


# --- dropped layers, corrupt plates, sidecar names ---------------------------

def test_dropped_and_untextured_layers_are_listed(tmp_path):
    layers = [SceneLayer("empty", np.zeros((0, 3)), np.zeros((0, 3))),
              SceneLayer("degenerate", QUAD_V, np.array([[0, 0, 0]]), QUAD_UV, _png()),
              SceneLayer("bad_uvs", QUAD_V, QUAD_F, QUAD_UV[:3], _png()),
              SceneLayer("ok", QUAD_V, QUAD_F, QUAD_UV, _png())]
    out = write_scene_glb(layers, tmp_path / "s.glb")
    assert {d["name"] for d in out["dropped"]} == {"empty", "degenerate"}
    bad = next(x for x in out["layers"] if x["name"] == "bad_uvs")
    assert not bad["textured"] and "UV count 3 != vertex count 4" in bad["untextured_reason"]


def test_corrupt_image_b64_exports_untextured_with_a_note():
    solve = _solve()
    solve.projection_sources[0].image_b64 = "data:image/png;base64," + base64.b64encode(
        b"\x89PNG not really a png").decode()
    layers, _, notes = build_scene_layers(solve, _primary(), exr_dir=None)
    src = [lay for lay in layers if lay.name.startswith("clean_plate_geo/")]
    assert src and src[0].image_bytes is None
    assert any("could not be decoded" in n and "untextured" in n for n in notes)


def test_sidecar_names_are_sanitised_and_unique(tmp_path, monkeypatch):
    from atlas_camera.exporters import scene_glb

    written = []

    def fake_exr(path, pil, plate_ref):
        written.append(path)
        return {"exr": path.name, "exr_colorspace": "x", "exr_origin": "test",
                "scene_referred": False}

    monkeypatch.setattr(scene_glb, "_write_exr_sidecar", fake_exr)
    solve = _solve()
    uri = solve.projection_sources[0].image_b64
    for name in ("../../evil name", "evil/name"):
        solve.projection_sources.append(ProjectionSource(
            camera=solve.camera, name=name, image_b64=uri,
            proxy_geometry=[_prim("m", QUAD_V)], metadata={}))
    _, sidecars, _ = build_scene_layers(solve, _primary(), exr_dir=tmp_path, exr_prefix="t")
    names = [p.name for p in written]
    assert names == ["t_primary.exr", "t_clean_plate_geo.exr", "t_evil_name.exr",
                     "t_evil_name_2.exr"]
    assert all(p.parent == tmp_path for p in written)
    import re
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", n) for n in names)


def test_node_report_lists_drops_and_file3d_fallback(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_scene3d

    solve = _solve()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="no_payload", primitive_type="mesh", dimensions=(1.0, 1.0, 1.0),
        material="m", metadata={"role": PROXY_ROLE}))
    monkeypatch.setattr(nodes_scene3d, "output_paths",
                        lambda prefix: (tmp_path, "scene_00001"))
    report = nodes_scene3d.AtlasSceneTo3D().export(
        solve, torch.rand(1, H, W, 3), write_exr=False)["result"][4]
    assert "no_payload: dropped" in report
    assert "File3D unavailable" in report      # no comfy_api in the test env
    assert "2 embedded plate image(s), one per source" in report


# --- the GLB parses; a float plate_ref rides as a scene-referred EXR -----

def _check_glb_structure(path):
    import struct as _s
    data = path.read_bytes()
    magic, version, total = _s.unpack_from("<III", data, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    jlen, jtype = _s.unpack_from("<II", data, 12)
    blen, btype = _s.unpack_from("<II", data, 20 + jlen)
    assert jtype == 0x4E4F534A and btype == 0x004E4942 and jlen % 4 == 0
    g = read_glb_json(path)
    assert g["buffers"][0]["byteLength"] == blen
    for v in g["bufferViews"]:
        assert v["byteOffset"] % 4 == 0 and v["byteOffset"] + v["byteLength"] <= blen
    return g


def test_written_glb_parses(tmp_path):
    layers, _, _ = build_scene_layers(_solve(), _primary(), exr_dir=None)
    layers.append(SceneLayer("pixal3d_object", QUAD_V + [0, 0, 1], QUAD_F, QUAD_UV,
                             layers[0].image_bytes, vertex_colors=np.full((4, 3), 0.4),
                             photo_weight=np.array([1, 1, 1, 0.0])))
    path = tmp_path / "s.glb"
    write_scene_glb(layers, path)
    _check_glb_structure(path)
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.load(str(path), force="scene")
    assert len(scene.geometry) >= 3
    verts = np.concatenate([np.asarray(g.vertices) for g in scene.geometry.values()])
    assert np.isfinite(verts).all() and len(verts) >= 12


def test_float_plate_ref_writes_a_scene_referred_exr(tmp_path):
    pytest.importorskip("OpenImageIO")
    from types import SimpleNamespace

    from atlas_camera.plate.oiio_io import read_plate, write_exr

    hdr = np.full((H, W, 3), 4.5, dtype=np.float32)                 # > 1: scene-linear
    ref = tmp_path / "plate_acescg.exr"
    write_exr(str(ref), hdr, bit_depth="float", source_colorspace="ACEScg")
    solve = _solve()
    # A layer's float plate_ref (ProjectionSource.plate_ref, e.g. a RAW/EXR
    # registered source) is preferred over its 8-bit display plate.
    solve.projection_sources[0].plate_ref = SimpleNamespace(
        image_path=str(ref), is_proxy=False, colorspace="ACEScg")
    out_dir = tmp_path / "out"
    _, sidecars, notes = build_scene_layers(solve, _primary(), exr_dir=out_dir,
                                            exr_prefix="t")
    assert next(s for s in sidecars if s["plate"] == "primary")["scene_referred"] is False
    prim = next(s for s in sidecars if s["plate"] == "clean_plate_geo")
    assert prim["exr_origin"] == "plate_ref" and prim["scene_referred"] is True, notes
    back = read_plate(str(out_dir / prim["exr"]), output_colorspace=None)
    assert float(np.asarray(back.pixels)[..., :3].mean()) == pytest.approx(4.5, rel=1e-3)
