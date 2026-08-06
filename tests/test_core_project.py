"""Tests for the delivery project system (atlas_camera.core.project)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_camera.core.project import (
    ColourPolicy,
    MODE_STANDARD,
    MODE_VFX,
    PROJECT_MANIFEST,
    SHOT_SUBDIRS,
    build_project,
    normalise_mode,
    resolve_root,
    sanitise_name,
)


def test_normalise_mode_defaults_unknown_to_standard():
    assert normalise_mode("vfx") == MODE_VFX
    assert normalise_mode("ACEScg") == MODE_VFX
    assert normalise_mode("float") == MODE_VFX
    for bad in ("", None, "sRGB", "standard", "banana"):
        assert normalise_mode(bad) == MODE_STANDARD


def test_sanitise_name_strips_illegal_and_falls_back():
    assert sanitise_name("  Shot 010  ", fallback="x") == "Shot 010"
    assert sanitise_name("a/b\\c:d", fallback="x") == "a_b_c_d"
    assert sanitise_name("a<<>>b", fallback="x") == "a_b"
    assert sanitise_name("...", fallback="untitled") == "untitled"
    assert sanitise_name(None, fallback="untitled") == "untitled"


def test_colour_policy_standard_is_unmanaged_srgb():
    p = ColourPolicy.from_mode("standard")
    assert p.mode == MODE_STANDARD and p.managed is False
    assert p.working_space == "sRGB" and p.image_ext == "png"
    assert p.ocio_config is None


def test_colour_policy_vfx_is_managed_acescg():
    p = ColourPolicy.from_mode("vfx")
    assert p.mode == MODE_VFX and p.managed is True
    assert p.working_space == "ACEScg" and p.image_ext == "exr"
    assert p.ocio_config


def test_resolve_root_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    env = tmp_path / "env"
    default = tmp_path / "default"
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(env))
    assert resolve_root(str(explicit), default_root=default) == explicit
    assert resolve_root("", default_root=default) == env
    monkeypatch.delenv("ATLAS_PROJECT_ROOT", raising=False)
    assert resolve_root(None, default_root=default) == default
    assert resolve_root(None) == Path.cwd() / "AtlasProjects"


def test_build_project_sanitises_and_derives(tmp_path):
    proj = build_project(str(tmp_path), "My Show", "sh/010", "vfx")
    assert proj.project == "My Show"
    assert proj.shot == "sh_010"
    assert proj.colour.mode == MODE_VFX
    assert proj.project_dir == tmp_path / "My Show"
    assert proj.shot_dir == tmp_path / "My Show" / "sh_010"
    assert proj.manifest_path == tmp_path / "My Show" / PROJECT_MANIFEST


def test_ensure_tree_creates_all_subdirs(tmp_path):
    proj = build_project(str(tmp_path), "P", "S", "standard")
    shot_dir = proj.ensure_tree()
    assert shot_dir.is_dir()
    for name in SHOT_SUBDIRS:
        assert (shot_dir / name).is_dir()


def test_subdir_known_and_unknown(tmp_path):
    proj = build_project(str(tmp_path), "P", "S", "standard")
    assert proj.subdir("nuke") == proj.shot_dir / "nuke"
    created = proj.subdir("maya", create=True)
    assert created.is_dir()
    with pytest.raises(ValueError):
        proj.subdir("bogus")


def test_write_manifest_records_policy_and_accumulates_shots(tmp_path):
    a = build_project(str(tmp_path), "P", "shotA", "vfx")
    a.write_manifest()
    manifest = json.loads(a.manifest_path.read_text(encoding="utf-8"))
    assert manifest["project"] == "P"
    assert manifest["colour"]["mode"] == MODE_VFX
    assert manifest["shots"] == ["shotA"]
    created = manifest["created"]

    b = build_project(str(tmp_path), "P", "shotB", "vfx")
    b.write_manifest()
    manifest2 = json.loads(b.manifest_path.read_text(encoding="utf-8"))
    assert manifest2["shots"] == ["shotA", "shotB"]
    assert manifest2["created"] == created


def test_no_inputs_never_crashes_and_avoids_empty_segments(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_PROJECT_ROOT", raising=False)
    proj = build_project("", "", "", "", default_root=tmp_path)
    assert proj.project == "untitled_project"
    assert proj.shot == "untitled_shot"
    assert proj.colour.mode == MODE_STANDARD
    assert proj.root == tmp_path
    proj.ensure_tree()
    assert proj.shot_dir.is_dir()
