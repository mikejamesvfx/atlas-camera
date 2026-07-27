"""docs/FEATURE_AUDIT.md + reports/feature_audit.json — the audit's contract.

The audit only stays honest if it cannot drift from the registry: a node added
without a verdict, a verdict naming a node that no longer exists, or a
LEGACY_GATE row pointing at a replacement that itself got culled are all
silent failures otherwise.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def report():
    path = REPO / "reports" / "feature_audit.json"
    assert path.exists(), "run tools/build_feature_audit.py"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_registered_node_has_exactly_one_verdict(report):
    from atlas_camera.comfy import node_registry as reg
    registered = set(reg.NODE_CLASS_MAPPINGS) | set(reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS)
    registered |= set(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}))
    assert set(report["nodes"]) == registered


def test_verdicts_are_from_the_allowed_set(report):
    verdicts = _load("feature_audit_verdicts")
    for key, rec in report["nodes"].items():
        assert rec["verdict"] in verdicts.VERDICT_VALUES, key


def test_hand_authored_verdicts_name_real_nodes():
    """A verdict for a node that no longer exists is a stale judgement."""
    verdicts = _load("feature_audit_verdicts")
    from atlas_camera.comfy import node_registry as reg
    registered = set(reg.NODE_CLASS_MAPPINGS) | set(reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS)
    registered |= set(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}))
    unknown = sorted(set(verdicts.VERDICTS) - registered)
    assert unknown == [], f"verdicts for unregistered nodes: {unknown}"
    unknown_defects = sorted(set(verdicts.DEFECTS) - registered)
    assert unknown_defects == [], f"defects for unregistered nodes: {unknown_defects}"


def test_removal_verdicts_carry_evidence_and_an_action(report):
    """Anything being taken away must say why, and where users go instead."""
    for key, rec in report["nodes"].items():
        if rec["verdict"] in ("LEGACY_GATE", "DEPRECATE", "DELETE", "MIGRATE_CAPABILITY"):
            assert rec["evidence"], f"{key}: no evidence for {rec['verdict']}"
            assert rec["migration_action"], f"{key}: no migration action"


def test_legacy_replacements_point_at_a_registered_standard_node(report):
    """Catches a replacement that was itself culled in the same cycle."""
    from atlas_camera.comfy import node_registry as reg
    for key, rec in report["nodes"].items():
        if rec["verdict"] != "LEGACY_GATE":
            continue
        target = rec["overlapping_replacement"]
        assert target, f"{key}: LEGACY_GATE with no replacement named"
        head = target.split()[0].strip("(),")
        assert head in reg.NODE_CLASS_MAPPINGS, (
            f"{key}: replacement {head!r} is not a registered standard node")


def test_nothing_is_deleted_without_execution_evidence(report):
    """A node that executes and returns meaningful output is not dead code."""
    for key, rec in report["nodes"].items():
        if rec["verdict"] == "DELETE":
            assert rec["live_execution"] == "error", (
                f"{key}: marked DELETE but it executes — that is HOLD_NEEDS_EVIDENCE")


def test_artifacts_are_regenerated_from_current_evidence():
    """The committed report must match a fresh run, so it cannot go stale."""
    builder = _load("build_feature_audit")
    fresh = builder.build()
    committed = json.loads(
        (REPO / "reports" / "feature_audit.json").read_text(encoding="utf-8"))
    assert committed["nodes"] == fresh["nodes"], (
        "reports/feature_audit.json is stale — run tools/build_feature_audit.py")
    md = (REPO / "docs" / "FEATURE_AUDIT.md").read_text(encoding="utf-8")
    assert md == builder.render_markdown(fresh), (
        "docs/FEATURE_AUDIT.md is stale — run tools/build_feature_audit.py")
