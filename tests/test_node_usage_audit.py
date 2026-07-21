"""Contract for the read-only node-usage audit (tools/audit_node_usage.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
    assert len(names) == 72   # 68 standard (incl. AtlasAssessOutput) + 4 experimental
    for name, rec in data.items():
        assert rec["kind"] in ("standard", "experimental")
        assert rec["status"] in ("referenced", "registered_only")
        for bucket in ("example_workflows", "tests", "mcp_tools", "docs"):
            assert isinstance(rec[bucket], list)


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
    assert experimental == {"AtlasPredictHiddenGeometry", "AtlasRenderFix",
                            "AtlasExtractAnglePatch", "AtlasImportAnglePatch"}


def test_audit_is_read_only(tmp_path):
    # Running the audit must not create or modify any file under the repo.
    audit = _load_audit()
    before = {p: p.stat().st_mtime_ns
              for p in (ROOT / "examples").rglob("*.json")}
    audit.audit()
    after = {p: p.stat().st_mtime_ns
             for p in (ROOT / "examples").rglob("*.json")}
    assert before == after
