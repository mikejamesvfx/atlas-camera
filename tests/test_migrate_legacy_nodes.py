"""tools/migrate_legacy_nodes.py — rewiring saved graphs off legacy nodes.

A migrator that emits a subtly broken graph is worse than none: ComfyUI will
load it and fail at execution. These pin the same link-graph invariants
tests/test_example_workflows.py enforces on shipped workflows, plus the
safety properties (dry-run default, -edit skip, refusal to half-wire).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _mig():
    spec = importlib.util.spec_from_file_location(
        "migrate_legacy_nodes", REPO / "tools" / "migrate_legacy_nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _legacy_graph():
    """LoadImage -> AtlasInput -> AtlasLiveMeshRepair -> AtlasBlockoutViewport."""
    return {
        "id": "3f1b0c6e-0000-4000-8000-000000000001",
        "last_node_id": 4, "last_link_id": 3,
        "nodes": [
            {"id": 1, "type": "AtlasInput", "pos": [0, 0], "size": [300, 100],
             "flags": {}, "order": 0, "mode": 0, "inputs": [],
             "outputs": [{"name": "solve", "type": "ATLAS_SOLVE", "links": [1],
                          "slot_index": 0}],
             "properties": {"Node name for S&R": "AtlasInput"},
             "widgets_values": []},
            {"id": 2, "type": "AtlasLiveMeshRepair", "pos": [400, 0],
             "size": [300, 100], "flags": {}, "order": 1, "mode": 0,
             "inputs": [{"name": "solve", "type": "ATLAS_SOLVE", "link": 1}],
             "outputs": [{"name": "ATLAS_SOLVE", "type": "ATLAS_SOLVE",
                          "links": [2], "slot_index": 0}],
             "properties": {"Node name for S&R": "AtlasLiveMeshRepair"},
             "widgets_values": ["auto", True, 0.0, 256, True, True, 8, 0.0]},
            {"id": 3, "type": "AtlasBlockoutViewport", "pos": [800, 0],
             "size": [300, 100], "flags": {}, "order": 2, "mode": 0,
             "inputs": [{"name": "solve", "type": "ATLAS_SOLVE", "link": 2}],
             "outputs": [], "properties": {"Node name for S&R": "AtlasBlockoutViewport"},
             "widgets_values": []},
        ],
        "links": [[1, 1, 0, 2, 0, "ATLAS_SOLVE"], [2, 2, 0, 3, 0, "ATLAS_SOLVE"]],
        "groups": [], "config": {}, "extra": {}, "version": 0.4,
    }


def _assert_link_graph_consistent(d):
    """The exact invariants test_example_workflows.py enforces."""
    nodes = {n["id"]: n for n in d["nodes"]}
    for link in d["links"]:
        lid, sid, sslot, tid, tslot = link[:5]
        assert sid in nodes and tid in nodes, f"link {lid} references a dead node"
        assert lid in (nodes[sid]["outputs"][sslot].get("links") or []), \
            f"link {lid} missing from its origin's outputs"
        assert nodes[tid]["inputs"][tslot].get("link") == lid, \
            f"link {lid} missing from its target's input"
    ids = [n["id"] for n in d["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    assert d["last_node_id"] >= max(ids)
    assert d["last_link_id"] >= max(l[0] for l in d["links"])


def test_replaces_the_legacy_node_with_its_chain():
    d, notes = _mig().migrate_workflow(_legacy_graph())
    types = [n["type"] for n in d["nodes"]]
    assert "AtlasLiveMeshRepair" not in types
    assert "AtlasPlanarHolePatch" in types and "AtlasRetopologizeLayer" in types
    assert notes and "AtlasLiveMeshRepair" in notes[0]


def test_migrated_graph_keeps_every_link_invariant():
    d, _ = _mig().migrate_workflow(_legacy_graph())
    _assert_link_graph_consistent(d)


def test_upstream_and_downstream_are_reconnected_through_the_chain():
    d, _ = _mig().migrate_workflow(_legacy_graph())
    by_type = {n["type"]: n for n in d["nodes"]}
    patch, retopo = by_type["AtlasPlanarHolePatch"], by_type["AtlasRetopologizeLayer"]
    src = {(l[1], l[3]) for l in d["links"]}
    assert (1, patch["id"]) in src          # AtlasInput feeds the chain head
    assert (patch["id"], retopo["id"]) in src
    assert (retopo["id"], 3) in src         # tail feeds the original consumer


def test_migrated_widgets_match_the_live_schema():
    """Widget rows come from live INPUT_TYPES, so an appended widget cannot
    silently desync the emitted node."""
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.mcp.comfy_http import is_widget

    before = {n["id"] for n in _legacy_graph()["nodes"]}
    d, _ = _mig().migrate_workflow(_legacy_graph())
    created = [n for n in d["nodes"] if n["id"] not in before]
    assert created, "migration created no nodes"
    for n in created:                      # only what the migrator emitted
        cls = reg.NODE_CLASS_MAPPINGS[n["type"]]
        it = cls.INPUT_TYPES()
        want = sum(1 for s in ("required", "optional")
                   for _k, v in (it.get(s) or {}).items() if is_widget(v))
        assert len(n.get("widgets_values") or []) == want, n["type"]


def test_migration_carries_the_intended_settings():
    d, _ = _mig().migrate_workflow(_legacy_graph())
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.mcp.comfy_http import is_widget

    def widget(node, name):
        it = reg.NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES()
        names = [n for s in ("required", "optional")
                 for n, spec in (it.get(s) or {}).items() if is_widget(spec)]
        return node["widgets_values"][names.index(name)]

    by_type = {n["type"]: n for n in d["nodes"]}
    # '*' reproduces LiveMeshRepair's whole-solve sweep...
    assert widget(by_type["AtlasPlanarHolePatch"], "layer") == "*"
    assert widget(by_type["AtlasRetopologizeLayer"], "layer") == "*"
    # ...and the migrated capability is boundary smoothing, not retopology.
    assert widget(by_type["AtlasRetopologizeLayer"], "method") == "off"
    assert widget(by_type["AtlasRetopologizeLayer"], "boundary_smooth_iterations") == 8


def test_workflow_identity_is_preserved():
    before = _legacy_graph()
    d, _ = _mig().migrate_workflow(before)
    assert d["id"] == before["id"]          # UUID must not be regenerated


def test_graph_without_legacy_nodes_is_untouched():
    g = _legacy_graph()
    g["nodes"] = [n for n in g["nodes"] if n["type"] != "AtlasLiveMeshRepair"]
    g["links"] = []
    for n in g["nodes"]:
        for o in n.get("outputs") or []:
            o["links"] = []
        for i in n.get("inputs") or []:
            i["link"] = None
    d, notes = _mig().migrate_workflow(g)
    assert notes == []
    assert [n["type"] for n in d["nodes"]] == [n["type"] for n in g["nodes"]]


def test_refuses_to_half_wire_when_a_required_input_is_missing():
    """Better to report and skip than to emit a graph that loads and fails."""
    g = _legacy_graph()
    for n in g["nodes"]:
        if n["type"] == "AtlasLiveMeshRepair":
            n["inputs"][0]["link"] = None
    with pytest.raises(ValueError, match="half-wired|not connected"):
        _mig().migrate_workflow(g)


def test_personal_edit_files_are_skipped(tmp_path, capsys):
    """-edit workflows are personal working copies — never rewritten."""
    p = tmp_path / "atlas_thing_workflow-edit.json"
    p.write_text(json.dumps(_legacy_graph()), encoding="utf-8")
    before = p.read_text(encoding="utf-8")

    mod = _mig()
    import sys
    argv = sys.argv
    sys.argv = ["migrate_legacy_nodes.py", str(p), "--write"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    assert "SKIP" in capsys.readouterr().out
    assert p.read_text(encoding="utf-8") == before      # byte-identical


def test_dry_run_is_the_default(tmp_path, capsys):
    p = tmp_path / "atlas_legacy_workflow.json"
    p.write_text(json.dumps(_legacy_graph()), encoding="utf-8")
    before = p.read_text(encoding="utf-8")

    mod = _mig()
    import sys
    argv = sys.argv
    sys.argv = ["migrate_legacy_nodes.py", str(p)]      # no --write
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "WOULD" in out and "--write" in out
    assert p.read_text(encoding="utf-8") == before      # nothing written


def test_shipping_workflows_have_no_legacy_nodes_left():
    """The cull is only done when nothing shipped still names a gated node."""
    from atlas_camera.comfy import node_registry as reg

    legacy = set(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}))
    offenders = []
    for wf in sorted((REPO / "examples").glob("*.json")):
        data = json.loads(wf.read_text(encoding="utf-8"))
        nodes = data.get("nodes")
        types = ({n.get("type") for n in nodes} if isinstance(nodes, list)
                 else {v.get("class_type") for v in data.values() if isinstance(v, dict)})
        if types & legacy:
            offenders.append(f"{wf.name}: {sorted(types & legacy)}")
    assert offenders == [], f"shipping workflows still use gated nodes: {offenders}"
