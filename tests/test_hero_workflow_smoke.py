"""Offline contracts for the hero-workflow release acceptance harness.

The harness itself needs a live ComfyUI, models and disk. These tests drive its
pure assessment function instead, so the RULES stay pinned in CI even though
the run cannot happen there.

The rule that matters: "completed with no errors" is not proof. A node that
exports nothing also completes. Hero 02 caught a NameError in both solver nodes
on its first real run precisely because someone looked at the artifacts.
"""
from __future__ import annotations

import pytest

from tools.smoke_hero_workflows import HEROES, assess_hero_result

SPEC = {
    "artifacts": ["mesh.obj", "build_scene.py"],
    "min_bytes": {"mesh.obj": 1000},
}
OK = {"completed": True, "errors": [], "reports": {"5": {}}, "output_nodes": ["7", "8"]}


def test_a_clean_run_with_all_artifacts_passes():
    out = assess_hero_result(OK, SPEC, sizes={"mesh.obj": 8000, "build_scene.py": 40})
    assert out["completed"] is True
    assert out["artifacts"] == {"mesh.obj": 8000, "build_scene.py": 40}
    assert out["output_nodes"] == ["7", "8"]


def test_execution_errors_fail_loudly():
    bad = {**OK, "completed": False,
           "errors": ["AtlasLearnedSolveFromImage (node 2): NameError: "
                      "name '_solve_summary' is not defined"]}
    with pytest.raises(RuntimeError, match="_solve_summary"):
        assess_hero_result(bad, SPEC, sizes={"mesh.obj": 8000, "build_scene.py": 40})


def test_not_completed_fails_even_with_no_error_list():
    with pytest.raises(RuntimeError, match="not completed"):
        assess_hero_result({"completed": False, "errors": []}, SPEC,
                           sizes={"mesh.obj": 8000, "build_scene.py": 40})


def test_a_missing_artifact_fails_a_completed_run():
    """THE point of the harness. Success plus no output is the failure mode a
    green queue cannot see."""
    with pytest.raises(RuntimeError, match="produced no build_scene.py"):
        assess_hero_result(OK, SPEC, sizes={"mesh.obj": 8000})


def test_a_zero_byte_artifact_fails():
    with pytest.raises(RuntimeError, match="zero-byte"):
        assess_hero_result(OK, SPEC, sizes={"mesh.obj": 8000, "build_scene.py": 0})


def test_an_undersized_artifact_fails():
    """A 40-byte OBJ is a header with no geometry — present, non-zero, useless."""
    with pytest.raises(RuntimeError, match="expected >= 1000B"):
        assess_hero_result(OK, SPEC, sizes={"mesh.obj": 40, "build_scene.py": 40})


# --------------------------------------------------------------- registry


def test_every_registered_hero_names_a_workflow_that_exists():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for slug, spec in HEROES.items():
        assert (root / spec["workflow"]).is_file(), f"{slug}: {spec['workflow']}"
        assert spec["artifacts"], f"{slug}: must assert at least one artifact"
        assert spec["export_dir"], f"{slug}: needs an export_dir to look in"


def test_min_bytes_only_names_declared_artifacts():
    """A floor on an artifact the harness never looks for silently never fires."""
    for slug, spec in HEROES.items():
        unknown = set(spec.get("min_bytes") or {}) - set(spec["artifacts"])
        assert not unknown, f"{slug}: min_bytes names non-artifact(s) {unknown}"
