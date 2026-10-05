"""Tests for the AtlasProject ComfyUI node (atlas_camera.comfy.nodes_project)."""
from __future__ import annotations

from pathlib import Path

from atlas_camera.comfy.nodes_project import AtlasProject
from atlas_camera.core.project import (
    AtlasProject as ProjectCtx,
    MODE_STANDARD,
    MODE_VFX,
)


def _result(out):
    """The node's result tuple, whether or not it came wrapped with a UI note."""
    return out["result"] if isinstance(out, dict) else out


def test_node_returns_atlas_project_type():
    # slot 0 is the project (saved links are by slot); STRING pipes are appended
    assert AtlasProject.RETURN_TYPES == ("ATLAS_PROJECT",) + ("STRING",) * 6
    assert AtlasProject.RETURN_NAMES == ("project", "project_name", "shot", "colour_mode",
                                         "project_root", "shot_dir", "shot_prefix")
    assert len(AtlasProject.OUTPUT_TOOLTIPS) == len(AtlasProject.RETURN_TYPES)


def test_node_pipes_its_inputs_and_paths_out(tmp_path, monkeypatch):
    from atlas_camera.comfy import nodes_project

    monkeypatch.setattr(nodes_project, "_default_output_root", lambda: str(tmp_path))
    from atlas_camera.comfy import node_helpers
    monkeypatch.setattr(node_helpers, "output_root", lambda: tmp_path.resolve())
    proj, name, shot, mode, root, shot_dir, prefix = AtlasProject().build(
        "My Show", "sh010", "VFX (ACEScg / float)", project_root="", create_tree=False)
    assert (name, shot, mode) == (proj.project, "sh010", "VFX (ACEScg / float)")
    assert Path(root) == Path(proj.root).resolve()
    assert Path(shot_dir) == Path(proj.shot_dir).resolve()
    assert prefix == Path(shot_dir).relative_to(tmp_path.resolve()).as_posix()
    assert "\\" not in prefix


def test_shot_prefix_is_empty_outside_the_output_folder(tmp_path, monkeypatch):
    from atlas_camera.comfy import nodes_project

    monkeypatch.setattr(nodes_project, "_default_output_root", lambda: str(tmp_path / "out"))
    from atlas_camera.comfy import node_helpers
    monkeypatch.setattr(node_helpers, "output_root", lambda: (tmp_path / "out").resolve())
    out = _result(AtlasProject().build("P", "S", "Standard (sRGB)",
                                       project_root=str(tmp_path / "elsewhere"),
                                       create_tree=False))
    assert out[-1] == "" and Path(out[-2]).is_absolute()


def test_root_outside_the_output_folder_is_made_visible(tmp_path, monkeypatch):
    """An absolute project_root is a feature, so it is not confined -- but the
    node says, in the UI, that it writes outside ComfyUI's output folder."""
    from atlas_camera.comfy import node_helpers, nodes_project

    monkeypatch.setattr(nodes_project, "_default_output_root", lambda: str(tmp_path / "out"))
    monkeypatch.setattr(node_helpers, "output_root", lambda: (tmp_path / "out").resolve())
    elsewhere = tmp_path / "elsewhere"
    out = AtlasProject().build("P", "S", "Standard (sRGB)", project_root=str(elsewhere),
                               create_tree=False)
    assert isinstance(out, dict)
    (note,) = out["ui"]["text"]
    assert "outside ComfyUI's output folder" in note and str(elsewhere.resolve()) in note
    # The result tuple is the same one an inside-root build returns.
    assert len(out["result"]) == len(AtlasProject.RETURN_TYPES)
    assert out["result"][4] == str(elsewhere.resolve())


def test_root_inside_the_output_folder_has_no_note(tmp_path, monkeypatch):
    from atlas_camera.comfy import node_helpers, nodes_project

    out_dir = tmp_path / "out"
    monkeypatch.setattr(nodes_project, "_default_output_root", lambda: str(out_dir))
    monkeypatch.setattr(node_helpers, "output_root", lambda: out_dir.resolve())
    for root in ("", str(out_dir / "shows")):
        out = AtlasProject().build("P", "S", "Standard (sRGB)", project_root=root,
                                   create_tree=False)
        assert isinstance(out, tuple) and len(out) == len(AtlasProject.RETURN_TYPES)


def test_node_builds_context_and_tree_in_vfx(tmp_path):
    node = AtlasProject()
    proj = _result(node.build(
        "My Show", "sh010", "VFX (ACEScg / float)", project_root=str(tmp_path)
    ))[0]
    assert isinstance(proj, ProjectCtx)
    assert proj.colour.mode == MODE_VFX and proj.colour.managed is True
    assert proj.shot_dir.is_dir()          # create_tree defaults True
    assert proj.manifest_path.is_file()


def test_node_defaults_to_standard_lane(tmp_path):
    node = AtlasProject()
    proj = _result(node.build(
        "P", "S", "Standard (sRGB)", project_root=str(tmp_path), create_tree=False
    ))[0]
    assert proj.colour.mode == MODE_STANDARD and proj.colour.managed is False
    assert proj.colour.ocio_config is None


def test_node_unknown_mode_falls_to_standard(tmp_path):
    node = AtlasProject()
    proj = _result(node.build(
        "P", "S", "something odd", project_root=str(tmp_path), create_tree=False
    ))[0]
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


def test_export_scene_package_routes_into_scenes_lane(tmp_path, make_atlas_solve):
    """The .atlas package needs a lane of its own, and it must EXIST.

    `subdir` raises on a name that is not in SHOT_SUBDIRS, deliberately, to
    catch an exporter typo before it scatters files somewhere odd — so a node
    that routes to a lane nobody declared does not misfile, it fails outright.
    AtlasExportScenePackage asked for "scenes" while the tuple ended at
    "review", and every project-connected export raised
    `unknown shot subfolder 'scenes'` at execution time. Found live in ComfyUI.
    """
    from atlas_camera.comfy.nodes_export import AtlasExportScenePackage
    from atlas_camera.core.project import build_project

    proj = build_project(str(tmp_path), "proj", "sh010", "standard")
    outcome = AtlasExportScenePackage().export(
        make_atlas_solve(), str(tmp_path / "ignored_dir"), "street_001", project=proj
    )
    # A solve with no plate and no mesh carries complaints, and complaints come
    # back as the UI form rather than a bare tuple.
    (package_path,) = outcome["result"] if isinstance(outcome, dict) else outcome
    dest = Path(package_path)
    assert dest.parent == tmp_path / "proj" / "sh010" / "scenes"
    assert dest.is_file()
    assert not (tmp_path / "ignored_dir").exists()


def test_ensure_tree_creates_the_scenes_lane(tmp_path):
    from atlas_camera.core.project import build_project

    shot_dir = build_project(str(tmp_path), "proj", "sh010", "standard").ensure_tree()
    assert (shot_dir / "scenes").is_dir()
