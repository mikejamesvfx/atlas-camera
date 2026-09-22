"""ADR-005 Amendment 1 (atlas-nexus): the delivery-project record gains a schema, keeps what it does not
own, and the standalone workbench stays out of delivery trees."""
from __future__ import annotations

import json

import pytest

from atlas_camera.core.project import (
    PROJECT_RECORD_SCHEMA,
    WORKBENCH_SESSION_FILENAME,
    ForeignProjectFileError,
    build_project,
    find_delivery_project_root,
    is_project_record,
)


def _project(root, shot="sh010"):
    return build_project(str(root), "Show", shot, "vfx")


# ------------------------------------------------------------------ schema
def test_a_new_record_carries_the_string_schema(tmp_path):
    record = json.loads(_project(tmp_path).write_manifest().read_text(encoding="utf-8"))
    assert record["schema"] == PROJECT_RECORD_SCHEMA == "atlas-project/1"
    assert isinstance(record["schema"], str)
    assert is_project_record(record)


def test_a_legacy_record_is_upgraded_and_keeps_what_it_does_not_own(tmp_path):
    project = _project(tmp_path)
    project.project_dir.mkdir(parents=True)
    legacy = {"atlas_version": "0.8.0", "project": "Show", "colour": {"mode": "vfx"},
              "shots": ["sh005"], "created": "2026-08-01T00:00:00+00:00", "updated": "x",
              "production_notes": "keep me", "lane_extensions": ["gen", "comp"]}
    project.manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    record = json.loads(project.write_manifest().read_text(encoding="utf-8"))

    assert record["schema"] == "atlas-project/1"
    assert record["production_notes"] == "keep me"
    assert record["lane_extensions"] == ["gen", "comp"]
    assert record["shots"] == ["sh005", "sh010"]
    assert record["created"] == legacy["created"]
    assert record["atlas_version"] == "0.8.0"


@pytest.mark.parametrize("schema", ["atlas-project/2", "atlas-project/99"])
def test_a_newer_record_is_recognised_but_never_modified(tmp_path, schema):
    project = _project(tmp_path)
    project.project_dir.mkdir(parents=True)
    content = json.dumps({"schema": schema, "project": "Show", "shots": ["sh005"], "colour": {}})
    project.manifest_path.write_text(content, encoding="utf-8")

    assert is_project_record(json.loads(content))
    with pytest.raises(ForeignProjectFileError, match="does not write"):
        project.write_manifest()
    assert project.manifest_path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("schema", [1, True, "something-else/1", ["atlas-project/1"]])
def test_other_schemas_are_not_project_records(schema):
    assert not is_project_record({"schema": schema, "project": "Show", "shots": []})


# ------------------------------------------------------------------ boundary: delivery side
def test_a_delivery_project_refuses_a_workbench_session_directory(tmp_path):
    project = _project(tmp_path)
    project.project_dir.mkdir(parents=True)
    session = project.project_dir / WORKBENCH_SESSION_FILENAME
    session.write_text(json.dumps({"project_dir": str(project.project_dir)}), encoding="utf-8")

    with pytest.raises(ForeignProjectFileError, match="workbench session directory"):
        project.write_manifest()
    assert not project.manifest_path.exists()


def test_find_delivery_project_root_walks_up_from_inside_the_tree(tmp_path):
    project = _project(tmp_path)
    project.write_manifest()
    lane = project.subdir("nuke", create=True)

    assert find_delivery_project_root(lane) == project.project_dir.resolve()
    assert find_delivery_project_root(project.project_dir) == project.project_dir.resolve()
    assert find_delivery_project_root(tmp_path) is None


def test_a_legacy_session_under_the_old_name_is_not_a_delivery_root(tmp_path):
    (tmp_path / "atlas_project.json").write_text(
        json.dumps({"project_dir": str(tmp_path), "source_image": None}), encoding="utf-8")
    assert find_delivery_project_root(tmp_path) is None


def test_an_unreadable_record_counts_as_a_delivery_root(tmp_path):
    (tmp_path / "atlas_project.json").write_text("{ not json", encoding="utf-8")
    assert find_delivery_project_root(tmp_path / "anything" / "below") == tmp_path.resolve()


# ------------------------------------------------------------------ boundary: workbench side
def test_the_workbench_refuses_to_open_inside_a_delivery_tree(tmp_path):
    from atlas_camera.ui.project import WorkbenchInDeliveryProjectError, open_project

    project = _project(tmp_path)
    project.write_manifest()
    inside = project.shot_dir / "review" / "session"

    with pytest.raises(WorkbenchInDeliveryProjectError, match="delivery project"):
        open_project(inside)
    assert not inside.exists(), "nothing may be created before refusing"


def test_the_workbench_opens_outside_delivery_trees(tmp_path):
    from atlas_camera.ui.project import open_project

    _project(tmp_path / "deliveries").write_manifest()
    session_dir = tmp_path / "workbench" / "lineup"

    project = open_project(session_dir)

    assert project.project_dir == session_dir.resolve()
    assert (session_dir / WORKBENCH_SESSION_FILENAME).is_file()


def test_the_workbench_session_name_has_one_definition():
    from atlas_camera.ui import project as ui_project

    assert ui_project.PROJECT_META is WORKBENCH_SESSION_FILENAME
