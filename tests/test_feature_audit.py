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


_NO_CALLER_CLAIMS = {
    "repair_relief_mesh_grid_cuda": "no node caller",
    "remove_stretched_faces": "no node caller",
}


def _callers_of(function_name: str) -> list[str]:
    """Files under atlas_camera/ that name `function_name`, excluding the
    module that defines it."""
    hits = []
    for path in (REPO / "atlas_camera").rglob("*.py"):
        if path.name == "mesh_repair.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if function_name in text:
            hits.append(str(path.relative_to(REPO)).replace("\\", "/"))
    return sorted(hits)


@pytest.mark.parametrize("function_name", sorted(_NO_CALLER_CLAIMS))
def test_the_appendix_does_not_claim_a_called_function_is_uncalled(function_name):
    """The 'capabilities REMOVED' appendix is PROSE hardcoded in the generator.

    Because the freshness test compares generated against committed, a false
    claim baked into the generator is permanently 'fresh' — it can never go
    stale, only stay wrong. So check the claim against the code instead.

    Both of these were in fact still called when this test was written:
    `repair_relief_mesh_grid_cuda` from `core/move_budget.py` (reached from
    `AtlasMoveBudget`, registered unconditionally) and `remove_stretched_faces`
    from `AtlasLiveMeshRepair` on the legacy tier.
    """
    md = (REPO / "docs" / "FEATURE_AUDIT.md").read_text(encoding="utf-8")
    if f"`core/mesh_repair.{function_name}`" not in md:
        pytest.skip(f"{function_name} is no longer discussed in the appendix")
    callers = _callers_of(function_name)
    if not callers:
        return
    claim = _NO_CALLER_CLAIMS[function_name]
    index = md.find(f"`core/mesh_repair.{function_name}`")
    sentence = md[index:index + 400]
    assert claim not in sentence, (
        f"FEATURE_AUDIT.md says `{function_name}` has {claim}, but it is "
        f"referenced from {callers}")


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


def _without_generated_stamp(markdown: str) -> str:
    """Delegate to the builder's own normalizer.

    That stamp is useful provenance but it must not gate freshness: comparing it
    made the suite go red at every calendar rollover with nothing actually
    stale. Worse, the advertised remedy (re-run the builder) WORKED, so the
    false alarm looked exactly like a real one — which trains people to
    regenerate without reading the diff, and that is how a genuinely stale
    artifact gets waved through. Found live 2026-07-29, a day after the file
    was last written.

    The 2026-07-29 fix defined the normalizer HERE, so it only ever taught the
    test to ignore the stamp — ``build_feature_audit.py --check``, the half the
    docstring advertises to CI, kept comparing the whole markdown and kept
    false-failing on rollover (found live 2026-08-04, a day after the last
    regeneration — the same bug, the same distance from the last write). It now
    lives in the builder and this is a thin alias, so the test exercises the
    shipped code path instead of a copy that cannot go wrong.
    """
    return _load("build_feature_audit").without_generated_stamp(markdown)


def test_artifacts_are_regenerated_from_current_evidence():
    """The committed report must match a fresh run, so it cannot go stale."""
    builder = _load("build_feature_audit")
    fresh = builder.build()
    committed = json.loads(
        (REPO / "reports" / "feature_audit.json").read_text(encoding="utf-8"))
    assert committed["nodes"] == fresh["nodes"], (
        "reports/feature_audit.json is stale — run tools/build_feature_audit.py")
    md = (REPO / "docs" / "FEATURE_AUDIT.md").read_text(encoding="utf-8")
    assert _without_generated_stamp(md) == \
        _without_generated_stamp(builder.render_markdown(fresh)), (
        "docs/FEATURE_AUDIT.md is stale — run tools/build_feature_audit.py")


def test_a_date_rollover_alone_does_not_fail_the_staleness_check():
    """Regression for the above: only CONTENT should count as stale.

    Simulated by editing the stamp rather than the clock, so the test is
    deterministic on any day.
    """
    builder = _load("build_feature_audit")
    fresh = builder.render_markdown(builder.build())
    yesterday = fresh.replace("Generated 2026-", "Generated 2020-", 1)
    assert yesterday != fresh, "the generated stamp is no longer where expected"
    assert _without_generated_stamp(yesterday) == _without_generated_stamp(fresh)
