"""Tests for the AtlasProject ComfyUI node (atlas_camera.comfy.nodes_project)."""
from __future__ import annotations

from pathlib import Path

from atlas_camera.comfy.nodes_project import AtlasProject
from atlas_camera.core.project import (
    AtlasProject as ProjectCtx,
    MODE_STANDARD,
    MODE_VFX,
)


def test_node_returns_atlas_project_type():
    assert AtlasProject.RETURN_TYPES == ("ATLAS_PROJECT",)
    assert AtlasProject.RETURN_NAMES == ("project",)


def test_node_builds_context_and_tree_in_vfx(tmp_path):
    node = AtlasProject()
    (proj,) = node.build(
        "My Show", "sh010", "VFX (ACEScg / float)", project_root=str(tmp_path)
    )
    assert isinstance(proj, ProjectCtx)
    assert proj.colour.mode == MODE_VFX and proj.colour.managed is True
    assert proj.shot_dir.is_dir()          # create_tree defaults True
    assert proj.manifest_path.is_file()


def test_node_defaults_to_standard_lane(tmp_path):
    node = AtlasProject()
    (proj,) = node.build(
        "P", "S", "Standard (sRGB)", project_root=str(tmp_path), create_tree=False
    )
    assert proj.colour.mode == MODE_STANDARD and proj.colour.managed is False
    assert proj.colour.ocio_config is None


def test_node_unknown_mode_falls_to_standard(tmp_path):
    node = AtlasProject()
    (proj,) = node.build(
        "P", "S", "something odd", project_root=str(tmp_path), create_tree=False
    )
    assert proj.colour.mode == MODE_STANDARD


# --- Exporter project routing (phase 6 step 2) -------------------------------

def test_export_solve_json_routes_into_project_tree(tmp_path, make_atlas_solve):
    from atlas_camera.comfy.nodes_export import AtlasExportSolveJSON
    from atlas_camera.core.project import build_project

    proj = build_project(str(tmp_path), "proj", "sh010", "standard")
    solve = make_atlas_solve()
    (dest,) = AtlasExportSolveJSON().export(solve, "atlas_solve.json", project=proj)
    dest_path = Path(dest)
    assert dest_path.parent == tmp_path / "proj" / "sh010" / "solves"
    assert dest_path.name == "atlas_solve.json"
    assert dest_path.is_file()


def test_export_solve_json_without_project_keeps_legacy_path(tmp_path, make_atlas_solve):
    from atlas_camera.comfy.nodes_export import AtlasExportSolveJSON

    solve = make_atlas_solve()
    legacy = tmp_path / "legacy" / "atlas_solve.json"
    legacy.parent.mkdir(parents=True)
    (dest,) = AtlasExportSolveJSON().export(solve, str(legacy))
    assert Path(dest) == legacy
    assert legacy.is_file()


def test_export_blender_routes_into_blender_lane(tmp_path, make_atlas_solve):
    from atlas_camera.comfy.nodes_export import AtlasExportBlender
    from atlas_camera.core.project import build_project

    proj = build_project(str(tmp_path), "proj", "sh010", "standard")
    solve = make_atlas_solve()
    (script_path,) = AtlasExportBlender().export(
        solve, str(tmp_path / "ignored_dir"), project=proj
    )
    dest = Path(script_path)
    assert dest.parent == tmp_path / "proj" / "sh010" / "blender"
    assert dest.is_file()
    assert not (tmp_path / "ignored_dir").exists()
