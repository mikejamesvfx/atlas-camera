"""ADR-005 (atlas-nexus): ``atlas_project.json`` names the delivery project, and nothing else.

Three Camera files used the name: the delivery-project record (``core/project.py``), the per-export
reproducibility manifest (now ``atlas_export.json``) and the workbench session (now
``atlas_workbench_session.json``). Co-located, each writer destroyed the others' files. These tests pin
the rename, the content-discriminated legacy reads, and that no writer overwrites a file it does not
own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_camera.core.project import (
    PROJECT_MANIFEST,
    ForeignProjectFileError,
    build_project,
    is_project_record,
)
from atlas_camera.exporters.manifest import (
    LEGACY_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    ForeignManifestError,
    ManifestArtifact,
    build_project_manifest,
    find_export_manifest,
    load_project_manifest,
    write_project_manifest,
)

from test_project_manifest import _solve


def _delivery_record(root: Path) -> Path:
    project = build_project(str(root), "Show", "sh010", "vfx")
    return project.write_manifest()


# ------------------------------------------------------------------ names
def test_the_three_files_have_three_names():
    from atlas_camera.ui.project import LEGACY_PROJECT_META, PROJECT_META

    assert PROJECT_MANIFEST == "atlas_project.json"
    assert MANIFEST_FILENAME == "atlas_export.json"
    assert PROJECT_META == "atlas_workbench_session.json"
    assert LEGACY_MANIFEST_FILENAME == LEGACY_PROJECT_META == "atlas_project.json"


# ------------------------------------------------------------------ export manifest
def test_export_manifest_is_written_under_its_own_name(tmp_path):
    path = write_project_manifest(_solve(), tmp_path,
                                  artifacts=[ManifestArtifact("nuke_scene", "a.nk", "T")])
    assert path.name == "atlas_export.json"
    assert not (tmp_path / "atlas_project.json").exists()
    assert load_project_manifest(path)["artifacts"][0]["path"] == "a.nk"


def test_an_export_beside_a_delivery_record_leaves_the_record_untouched(tmp_path):
    record = _delivery_record(tmp_path)
    before = record.read_bytes()

    write_project_manifest(_solve(), record.parent,
                           artifacts=[ManifestArtifact("nuke_scene", "a.nk", "T")])

    assert record.read_bytes() == before
    assert is_project_record(json.loads(before))
    assert (record.parent / "atlas_export.json").is_file()


def test_a_legacy_export_manifest_is_merged_from_but_never_modified(tmp_path):
    solve = _solve()
    legacy = tmp_path / "atlas_project.json"
    legacy.write_text(json.dumps(build_project_manifest(
        solve, artifacts=[ManifestArtifact("nuke_scene", "old.nk", "T")])), encoding="utf-8")
    before = legacy.read_bytes()

    path = write_project_manifest(solve, tmp_path,
                                  artifacts=[ManifestArtifact("maya_scene", "new.ma", "T")])

    assert legacy.read_bytes() == before
    paths = [a["path"] for a in load_project_manifest(path)["artifacts"]]
    assert paths == ["old.nk", "new.ma"]


@pytest.mark.parametrize("foreign", [
    {"project": "Show", "shots": ["sh010"], "colour": {"mode": "vfx"}},  # delivery record shape
    {"hello": "world"},
    "not json at all",
    {"schema": 99, "solve_fingerprint": "x"},                           # a newer Atlas's manifest
])
def test_a_foreign_file_under_the_export_name_is_refused_not_overwritten(tmp_path, foreign):
    path = tmp_path / "atlas_export.json"
    path.write_text(foreign if isinstance(foreign, str) else json.dumps(foreign), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ForeignManifestError, match="refusing to overwrite"):
        write_project_manifest(_solve(), tmp_path)
    assert path.read_bytes() == before


def test_find_export_manifest_discriminates_by_content(tmp_path):
    assert find_export_manifest(tmp_path) is None

    _delivery_record(tmp_path)
    assert find_export_manifest(tmp_path / "Show") is None, "a delivery record is not an export manifest"

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "atlas_project.json").write_text(
        json.dumps(build_project_manifest(_solve())), encoding="utf-8")
    assert find_export_manifest(legacy_dir) == legacy_dir / "atlas_project.json"

    write_project_manifest(_solve(), legacy_dir)
    assert find_export_manifest(legacy_dir) == legacy_dir / "atlas_export.json"


# ------------------------------------------------------------------ delivery-project record
def test_the_project_record_refuses_a_legacy_export_manifest(tmp_path):
    project = build_project(str(tmp_path), "Show", "sh010", "vfx")
    project.project_dir.mkdir(parents=True)
    project.manifest_path.write_text(json.dumps(build_project_manifest(_solve())), encoding="utf-8")
    before = project.manifest_path.read_bytes()

    with pytest.raises(ForeignProjectFileError, match="refusing to overwrite"):
        project.write_manifest()
    assert project.manifest_path.read_bytes() == before


@pytest.mark.parametrize("content", [
    json.dumps({"project_dir": "C:/somewhere", "source_image": None}),  # legacy workbench session
    "{ not json",
])
def test_the_project_record_refuses_a_session_or_unreadable_file(tmp_path, content):
    project = build_project(str(tmp_path), "Show", "sh010", "vfx")
    project.project_dir.mkdir(parents=True)
    project.manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(ForeignProjectFileError):
        project.write_manifest()
    assert project.manifest_path.read_text(encoding="utf-8") == content


def test_the_project_record_still_updates_itself(tmp_path):
    build_project(str(tmp_path), "Show", "shA", "vfx").write_manifest()
    record = build_project(str(tmp_path), "Show", "shB", "vfx").write_manifest()
    assert json.loads(record.read_text(encoding="utf-8"))["shots"] == ["shA", "shB"]


def test_the_project_node_reports_a_refusal_instead_of_failing(tmp_path):
    from atlas_camera.comfy.nodes_project import AtlasProject

    (tmp_path / "Show").mkdir()
    foreign = tmp_path / "Show" / "atlas_project.json"
    foreign.write_text(json.dumps(build_project_manifest(_solve())), encoding="utf-8")
    before = foreign.read_bytes()

    out = AtlasProject().build("Show", "sh010", "VFX (ACEScg / float)", project_root=str(tmp_path))

    assert isinstance(out, dict), "a refusal should surface as a UI note"
    assert out["result"][0].project == "Show"
    assert "not updated" in out["ui"]["text"][0]
    assert foreign.read_bytes() == before


# ------------------------------------------------------------------ workbench session
def test_the_workbench_never_treats_a_delivery_record_as_its_session(tmp_path):
    from atlas_camera.ui.project import open_project

    record = _delivery_record(tmp_path)
    before = record.read_bytes()

    project = open_project(record.parent)

    assert record.read_bytes() == before
    assert project.source_image is None
    session = json.loads((record.parent / "atlas_workbench_session.json").read_text(encoding="utf-8"))
    assert Path(session["project_dir"]) == record.parent.resolve()


def test_a_legacy_workbench_session_is_read_and_carried_over(tmp_path):
    from atlas_camera.ui.project import open_project

    image = tmp_path / "plate.png"
    image.write_bytes(b"not really a png")
    legacy = tmp_path / "atlas_project.json"
    legacy.write_text(json.dumps({"project_dir": str(tmp_path), "source_image": str(image)}),
                      encoding="utf-8")
    before = legacy.read_bytes()

    project = open_project(tmp_path)

    assert project.source_image == image
    assert legacy.read_bytes() == before
    assert (tmp_path / "atlas_workbench_session.json").is_file()


def test_the_identity_header_name_is_frozen():
    """Shipped .nk/.py/.ma exports carry this marker; renaming files must not rename it."""
    import inspect

    from atlas_camera.comfy import node_reports

    assert 'atlas_project_identity: ' in inspect.getsource(node_reports._write_export_manifest)
