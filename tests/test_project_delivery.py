"""Found live 2026-10-03: Save 3D (Advanced) copies only the GLB into
``output/3d/``, so the bare-name EXR sidecar references in the saved copy
resolved 0 of 5. Two fixes, both pinned here:

* without a project, every sidecar reference also records its path relative
  to ComfyUI's output folder, so a copy anywhere in that tree still finds it;
* with an AtlasProject, the GLB + sidecars (and the matrixZone EXR) land
  TOGETHER in the shot's lane, where a bare name resolves.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from atlas_camera.comfy.nodes_project import AtlasProject
from atlas_camera.exporters.scene_glb import build_scene_layers, read_glb_json

from test_scene_to_3d import H, W, _primary, _solve  # noqa: E402


def _project(tmp_path):
    out = AtlasProject().build("Show", "sh010", "VFX (ACEScg / float)",
                               project_root=str(tmp_path))
    return (out["result"] if isinstance(out, dict) else out)[0]


def _sidecar_refs(glb_path):
    g = read_glb_json(glb_path)
    out = []
    for m in g.get("materials", []):
        ex = m.get("extras") or {}
        if ex.get("exr"):
            out.append(ex)
    return out


def test_sidecar_refs_carry_their_output_root_path(tmp_path):
    pytest.importorskip("OpenImageIO")
    out_root = tmp_path / "output"
    folder = out_root / "atlas"
    folder.mkdir(parents=True)
    layers, sidecars, _ = build_scene_layers(_solve(), _primary(), exr_dir=folder,
                                             exr_prefix="s", output_root=out_root)
    exrs = [layer.extras for layer in layers if layer.extras.get("exr")]
    assert exrs and all(e["exr_output_path"] == f"atlas/{e['exr']}" for e in exrs)
    assert all((out_root / e["exr_output_path"]).is_file() for e in exrs)


def test_no_output_root_path_outside_the_root(tmp_path):
    pytest.importorskip("OpenImageIO")
    layers, _, _ = build_scene_layers(_solve(), _primary(), exr_dir=tmp_path / "elsewhere",
                                      exr_prefix="s", output_root=tmp_path / "output")
    assert all("exr_output_path" not in layer.extras for layer in layers)


def test_a_save3d_copy_in_output_3d_resolves_every_sidecar(tmp_path, monkeypatch):
    """The live repro: copy the GLB to output/3d like Save 3D does, then
    resolve each reference the way a consumer would."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_scene3d

    out_root = tmp_path / "output"
    (out_root / "atlas").mkdir(parents=True)
    monkeypatch.setattr(nodes_scene3d, "output_paths",
                        lambda prefix: (out_root / "atlas", "scene_00001"))
    monkeypatch.setattr(nodes_scene3d, "output_root", lambda: out_root)
    res = nodes_scene3d.AtlasSceneTo3D().export(_solve(), torch.rand(1, H, W, 3))
    glb, report = Path(res["result"][3]), res["result"][4]
    saved = out_root / "3d" / "atlas_scene_00001.glb"
    saved.parent.mkdir()
    shutil.copy(glb, saved)

    refs = _sidecar_refs(saved)
    assert refs
    assert not any((saved.parent / r["exr"]).is_file() for r in refs)     # the bug
    assert all((out_root / r["exr_output_path"]).is_file() for r in refs)  # the fix
    assert "exr_output_path" in report and "Save 3D copy" in report


def test_scene_with_a_project_delivers_glb_and_sidecars_together(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_scene3d

    # the project lives OUTSIDE the output folder here: bare names only
    monkeypatch.setattr(nodes_scene3d, "output_root", lambda: tmp_path / "comfy_output")
    proj = _project(tmp_path)
    res = nodes_scene3d.AtlasSceneTo3D().export(_solve(), torch.rand(1, H, W, 3),
                                                filename_prefix="atlas/scene", project=proj)
    glb = Path(res["result"][3])
    assert glb.parent == proj.subdir("geo")
    assert glb.name == "scene_00001.glb"
    refs = _sidecar_refs(glb)
    assert refs and all((glb.parent / r["exr"]).is_file() for r in refs)
    assert all("exr_output_path" not in r for r in refs)
    assert "project lane" in res["result"][4]
    # a second export counts up instead of overwriting
    res2 = nodes_scene3d.AtlasSceneTo3D().export(_solve(), torch.rand(1, H, W, 3),
                                                 filename_prefix="atlas/scene", project=proj)
    assert Path(res2["result"][3]).name == "scene_00002.glb"


def test_an_over_budget_scene_leaves_the_project_lane_empty(tmp_path, monkeypatch):
    """The budget is planned BEFORE the sidecars: a refused export writes no
    EXR, PLY or GLB into the shot's geo lane (nothing to clean up)."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_scene3d
    from atlas_camera.exporters import scene_glb

    monkeypatch.setattr(nodes_scene3d, "output_root", lambda: tmp_path / "comfy_output")
    monkeypatch.setattr(scene_glb, "_write_exr_sidecar",
                        lambda *a: pytest.fail("a sidecar was written before the refusal"))
    proj = _project(tmp_path)
    solve = _solve()
    solve.projection_scene.proxy_geometry[0].name = "x" * 1_500_000   # GLB JSON > 1 MB
    with pytest.raises(ValueError, match=r"over the 1 MB budget"):
        nodes_scene3d.AtlasSceneTo3D().export(solve, torch.rand(1, H, W, 3),
                                              max_glb_mb=1, project=proj)
    lane = proj.subdir("geo")
    assert not lane.exists() or not list(lane.iterdir())


def test_stitch_with_a_project_writes_the_plate_into_the_plates_lane(tmp_path):
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy.nodes_matrixzone import AtlasMatrixZoneSplit, AtlasMatrixZoneStitch
    from test_matrixzone_nodes import _plate

    proj = _project(tmp_path)
    clips, handle, _ = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "per_zone_clip", 9)
    res = AtlasMatrixZoneStitch().stitch(clips, [handle], filename_prefix=["atlas/hdr_plate"],
                                         project=[proj])
    exr = Path(res["result"][1])
    assert exr.parent == proj.subdir("plates") and exr.name == "hdr_plate_00001.exr"
    assert exr.is_file()


def test_project_sockets_are_appended_last():
    from atlas_camera.comfy.nodes_matrixzone import AtlasMatrixZoneStitch
    from atlas_camera.comfy.nodes_scene3d import AtlasSceneTo3D

    for cls, after in ((AtlasSceneTo3D, ["hdr_plate_path"]), (AtlasMatrixZoneStitch, [])):
        opt = list(cls.INPUT_TYPES()["optional"])
        # project came last when it was added; only later appends may follow it
        assert opt[len(opt) - 1 - len(after):] == ["project", *after]
        assert cls.INPUT_TYPES()["optional"]["project"][0] == "ATLAS_PROJECT"


def test_a_project_lane_inside_output_also_survives_a_save3d_copy(tmp_path, monkeypatch):
    """Found live: the default project root IS ComfyUI's output folder, and a
    Save 3D copy of a project-routed GLB resolved 0 of 6 sidecars."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_scene3d

    monkeypatch.setattr(nodes_scene3d, "output_root", lambda: tmp_path)
    proj = _project(tmp_path)                      # project root == output root
    res = nodes_scene3d.AtlasSceneTo3D().export(_solve(), torch.rand(1, H, W, 3), project=proj)
    glb = Path(res["result"][3])
    saved = tmp_path / "3d" / "copy.glb"
    saved.parent.mkdir()
    shutil.copy(glb, saved)
    refs = _sidecar_refs(saved)
    assert refs and all((tmp_path / r["exr_output_path"]).is_file() for r in refs)


def test_exports_leave_no_reservation_placeholders(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_scene3d

    proj = _project(tmp_path)
    nodes_scene3d.AtlasSceneTo3D().export(_solve(), torch.rand(1, H, W, 3), project=proj)
    assert not list(proj.subdir("geo").glob("*.reserved"))
