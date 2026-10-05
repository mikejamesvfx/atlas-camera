"""UV-textured generated objects: the model's own texture paints the hidden side.

Found live 2026-10-05: the Pixal3D object's hidden side looked blotchy next
to the core Pixal3D template, which bakes a 2048 texture. Atlas kept one colour
per vertex (~23k samples on a 50k-face mesh). These tests pin the texture path
end to end: core UV bake -> import keeps uvs + texture -> viewport payload ->
GLB layer with the photo baked into the model's UV texture -> HDR texture EXR.
"""

# The `scene` fixture is imported from test_object_mesh_nodes; pytest injects it
# by parameter name, which ruff reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]

# --- import --------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from test_object_mesh_nodes import _pixal_mesh, scene  # noqa: E402,F401  (fixture)

from atlas_camera.comfy.nodes_object_mesh import (  # noqa: E402
    AtlasImportGeneratedMesh,
    AtlasObjectCrop,
)


def _textured(scene, handle, colour=(0.2, 0.7, 0.3), size=32):
    m = _pixal_mesh(scene, handle, colours=False)
    n = m.vertices.shape[1]
    rng = np.random.default_rng(0)
    m.uvs = torch.from_numpy(rng.uniform(0.05, 0.95, (n, 2)).astype(np.float32))[None]
    m.texture = torch.tensor(colour, dtype=torch.float32).expand(1, size, size, 3).clone()
    return m


def _import(scene, mesh, handle, **kw):
    out = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], mesh, handle, scene["depth"], scene["image"], scene["obj"],
        sky_mask=scene["sky"], **kw)
    prim = [p for p in out[0].projection_scene.proxy_geometry
            if (p.metadata or {}).get("source") == "pixal3d"]
    return out, (prim[-1] if prim else None)


def test_import_keeps_the_models_uvs_and_texture(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    (_, report, _), prim = _import(scene, _textured(scene, handle), handle,
                                   match_colour=False)
    assert prim is not None, report
    md = prim.metadata
    n = len(md["vertices"]) // 3
    assert len(md["texture_uvs"]) == 2 * n
    assert md["texture_b64"].startswith("data:image/png;base64,")
    assert md["texture_size"] == [32, 32]
    assert "UV texture 32x32 kept" in report
    # per-vertex fallback colours come FROM the texture (no PaintMesh upstream)
    vc = np.asarray(md["vertex_colors"]).reshape(-1, 3)
    assert np.allclose(vc, [0.2, 0.7, 0.3], atol=0.01)


def test_a_textured_mesh_is_never_cluster_decimated(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    mesh = _textured(scene, handle)
    n_faces = mesh.faces.shape[1]
    (_, report, _), prim = _import(scene, mesh, handle, max_faces=100)
    assert prim is not None and len(prim.metadata["faces"]) // 3 == n_faces
    assert "not decimated here" in report


def test_the_colour_grade_reaches_the_texture_too(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    (_, report, _), prim = _import(scene, _textured(scene, handle), handle, match_colour=True)
    from PIL import Image
    md = prim.metadata
    png = base64.b64decode(md["texture_b64"].split(",", 1)[1])
    tex = np.asarray(Image.open(io.BytesIO(png)), dtype=np.float32) / 255.0
    vc = np.asarray(md["vertex_colors"]).reshape(-1, 3)
    # texture and per-vertex fallback were graded by the same gain
    assert np.allclose(tex.reshape(-1, 3).mean(0), vc.mean(0), atol=0.02), report


def test_mismatched_uvs_fall_back_to_vertex_colours_with_a_warning(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    mesh = _textured(scene, handle)
    mesh.uvs = mesh.uvs[:, :5]
    (_, report, _), prim = _import(scene, mesh, handle)
    assert "MESH texture ignored" in report
    assert "texture_b64" not in prim.metadata


# --- viewport payload ----------------------------------------------------------------

def test_payload_lifts_the_texture_and_keeps_it_out_of_metadata(scene):
    from atlas_camera.core.proxy_geometry import serialize_proxy_geometry
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    (solve_out, _, _), _ = _import(scene, _textured(scene, handle), handle)
    entry = [e for e in serialize_proxy_geometry(solve_out.projection_scene)
             if e.get("texture_b64")][0]
    assert len(entry["texture_uvs"]) == 2 * (len(entry["vertices"]) // 3)
    assert "texture_b64" not in entry["metadata"]


def test_the_shader_samples_the_generated_texture_in_uniform_control_flow():
    js = (REPO / "atlas_camera/comfy/web/atlas_blockout.js").read_text(encoding="utf-8")
    frag = js[js.index("const PROJECTION_FRAGMENT_SHADER"):]
    sample = frag.index("texture2D(uGenTexture")
    assert sample < frag.index("if (hasVC) {"), "mip derivatives undefined in the branch"
    assert "atlasGenUv: [0.0, 0.0]" in js and "uHasGenTexture:" in js


# --- GLB + HDR export ----------------------------------------------------------------

def _exported_solve(scene, with_curve=False):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    (solve_out, _, _), prim = _import(scene, _textured(scene, handle), handle,
                                      match_colour=False)
    if with_curve:
        prim.metadata["hdr_curve"] = {"log2_x": [-8.0, 0.0], "log2_y": [[-8.0, 1.0]] * 3,
                                      "residual_stops": 0.1, "space": "ACEScg"}
    return solve_out, prim


def _plate():
    from PIL import Image
    return Image.new("RGB", (320, 200), (250, 10, 10))


def test_glb_layer_uses_the_model_uvs_and_a_baked_texture(scene):
    from atlas_camera.exporters.scene_glb import build_scene_layers
    solve_out, prim = _exported_solve(scene)
    layers, _, notes = build_scene_layers(solve_out, _plate(), exr_dir=None)
    layer = next(lay for lay in layers if lay.name == prim.name)
    tl = np.asarray(prim.metadata["texture_uvs"]).reshape(-1, 2)
    assert np.allclose(layer.uvs[:, 0], tl[:, 0]) and np.allclose(layer.uvs[:, 1], 1 - tl[:, 1])
    assert layer.vertex_colors is None
    assert "generated_texture" in layer.extras and not notes
    from PIL import Image
    baked = np.asarray(Image.open(io.BytesIO(layer.image_bytes)), dtype=np.float32) / 255.0
    # red photo baked in somewhere, green model texture kept elsewhere
    assert (baked[..., 0] > 0.8).any() and (baked[..., 1] > 0.6).any()


def test_glb_writes_and_carries_a_texture_hdr_exr(scene, tmp_path):
    pytest.importorskip("OpenImageIO")
    from atlas_camera.exporters.scene_glb import build_scene_layers, write_scene_glb
    solve_out, prim = _exported_solve(scene, with_curve=True)
    layers, sidecars, _ = build_scene_layers(solve_out, _plate(), exr_dir=tmp_path,
                                             exr_prefix="t")
    tex_exr = [s for s in sidecars if s["exr"].endswith("_texture_hdr.exr")]
    assert len(tex_exr) == 1 and (tmp_path / tex_exr[0]["exr"]).is_file()
    layer = next(lay for lay in layers if lay.name == prim.name)
    assert layer.extras["texture_hdr_exr"] == tex_exr[0]["exr"]
    written = write_scene_glb(layers, tmp_path / "t.glb")
    assert written["bytes"] > 0


def test_a_failed_bake_falls_back_to_vertex_colour_with_a_note(scene, monkeypatch):
    from atlas_camera.core import uv_bake
    from atlas_camera.exporters.scene_glb import build_scene_layers

    def boom(*a, **k):
        raise RuntimeError("bake exploded")
    monkeypatch.setattr(uv_bake, "bake_photo_into_uv", boom)
    solve_out, prim = _exported_solve(scene)
    layers, _, notes = build_scene_layers(solve_out, _plate(), exr_dir=None)
    layer = next(lay for lay in layers if lay.name == prim.name)
    assert layer.vertex_colors is not None
    assert any("texture bake failed" in n for n in notes)

