"""Contract for the read-only node-usage audit (tools/audit_node_usage.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_node_usage", ROOT / "tools" / "audit_node_usage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_covers_every_registered_node():
    audit = _load_audit()
    kinds, names = audit.registered_nodes()
    data = audit.audit()
    assert set(data) == names            # exactly the registered set, nothing invented
    assert len(names) == 110    # 99 standard + 7 experimental + 2 legacy + 2 iOS
    for name, rec in data.items():
        assert rec["kind"] in ("standard", "experimental", "legacy", "ios")
        assert rec["status"] in ("referenced", "registered_only")
        for bucket in ("example_workflows", "tests", "mcp_tools", "repo_tools",
                       "docs", "dedicated_tests"):
            assert isinstance(rec[bucket], list)
        assert isinstance(rec["product_evidence"], bool)


def test_generic_pin_tests_are_not_product_evidence():
    """The whole point of the audit rework.

    Every node's name appears in the registry/façade pins by construction, so
    counting those as evidence made all 86 nodes look "referenced" and hid the
    nodes that nothing actually uses. `dedicated_tests` must exclude them, and
    `product_evidence` must be false for a node whose only test hits are pins.
    """
    audit = _load_audit()
    data = audit.audit()
    for rec in data.values():
        assert not (set(rec["dedicated_tests"]) & audit.GENERIC_TESTS)

    # The rule must actually DO something — one that classified everything the
    # same way would pass the check above while being useless. This used to be
    # asserted as "some standard node has no product evidence", which held only
    # while the audit's 14 unevidenced nodes were still unevidenced; covering
    # them (tests/test_node_layer_contracts.py) legitimately emptied that set,
    # so the assertion was measuring the repo's state rather than the rule.
    # Test the filter itself instead: it must strip pin hits from at least one
    # node, and a node whose ONLY test hits are pins must come out unevidenced.
    assert any(set(r["tests"]) - set(r["dedicated_tests"]) for r in data.values()), (
        "GENERIC_TESTS filtered nothing — the pin tests should hit every node")
    for rec in data.values():
        only_pins = rec["tests"] and not rec["dedicated_tests"]
        if only_pins and not rec["example_workflows"] and not rec["mcp_tools"]:
            assert not rec["product_evidence"]


def test_mcp_and_repo_tools_are_separate_buckets():
    """An MCP handler depending on a node is product evidence; a CLI script
    merely naming it is not. Folding both into one bucket made them
    indistinguishable."""
    audit = _load_audit()
    data = audit.audit()
    for rec in data.values():
        assert all(p.startswith("atlas_camera/mcp/") for p in rec["mcp_tools"])
        assert all(p.startswith("tools/") for p in rec["repo_tools"])


def test_no_standard_node_is_orphaned():
    # The motivating case (originally AtlasPitchTrim): a node absent from every
    # shipped workflow is NOT unused if a test/doc exercises it, and must not be
    # classified as registered-only. Generalized after AtlasPitchTrim's removal
    # left no workflow-absent standard node: every standard node must be
    # referenced somewhere (workflow, test, doc, or mcp tool) — none orphaned.
    audit = _load_audit()
    data = audit.audit()
    orphaned = sorted(n for n, r in data.items()
                      if r["kind"] == "standard" and r["status"] == "registered_only")
    assert orphaned == [], f"standard nodes referenced nowhere: {orphaned}"


def test_experimental_nodes_flagged():
    audit = _load_audit()
    data = audit.audit()
    experimental = {n for n, r in data.items() if r["kind"] == "experimental"}
    assert experimental == {"AtlasMaskedSurfaceReconstruct",
                            "AtlasRefineOcclusionSeams",
                            "AtlasExtractAnglePatch", "AtlasImportAnglePatch",
                                                        "AtlasCompleteDepth", "AtlasBlockoutMassing",
                            "AtlasPathFrameIndex"}


def test_audit_is_read_only(tmp_path):
    # Running the audit must not create or modify any file under the repo.
    audit = _load_audit()
    before = {p: p.stat().st_mtime_ns
              for p in (ROOT / "examples").rglob("*.json")}
    audit.audit()
    after = {p: p.stat().st_mtime_ns
             for p in (ROOT / "examples").rglob("*.json")}
    assert before == after


def test_untracked_files_are_not_product_evidence(tmp_path):
    """A local scratch file must not count toward a node's evidence.

    Regression: only the examples scan refused personal copies ("-edit",
    examples/local/). The text scans had no equivalent guard, so an untracked
    script dropped in tools/ silently became evidence and made the COMMITTED
    audit look stale on the author's machine alone — a failure nobody else could
    reproduce.
    """
    audit = _load_audit()
    root = tmp_path / "tools"
    root.mkdir()
    (root / "shipped.py").write_text("AtlasLoadRecord3D\n", encoding="utf-8")
    (root / "scratch.py").write_text("AtlasLoadRecord3D\n", encoding="utf-8")

    tracked = frozenset({"tools/shipped.py"})
    found = {p.name for p in audit._iter_files(
        root, {".py"}, repo=tmp_path, tracked=tracked)}
    assert found == {"shipped.py"}, "untracked scratch file must be skipped"


def test_no_tracking_information_means_every_file_counts(tmp_path):
    """Degrade to the old behaviour rather than reporting an empty repo.

    A tarball install or a checkout without git has no `git ls-files`, and
    silently finding zero evidence there would be far worse than counting
    everything.
    """
    audit = _load_audit()
    root = tmp_path / "tools"
    root.mkdir()
    (root / "a.py").write_text("x\n", encoding="utf-8")
    (root / "b.py").write_text("x\n", encoding="utf-8")

    found = {p.name for p in audit._iter_files(root, {".py"}, repo=tmp_path, tracked=None)}
    assert found == {"a.py", "b.py"}

    # An EMPTY git result must also mean "no opinion". This repo puts pytest's
    # tmp_path INSIDE the checkout, so `git -C <tmp> ls-files` exits 0 with no
    # output — and reading that literally would filter every file away and
    # report a product with no tests at all.
    assert audit._tracked_files(tmp_path) is None
