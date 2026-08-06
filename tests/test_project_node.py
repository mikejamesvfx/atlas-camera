"""Tests for the AtlasProject ComfyUI node (atlas_camera.comfy.nodes_project)."""
from __future__ import annotations

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
