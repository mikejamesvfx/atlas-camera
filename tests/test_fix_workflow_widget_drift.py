"""The drift repair tool must fix BOTH of LiteGraph's widget representations.

A save carries each widget twice: positionally in ``widgets_values``, and again
as a converted-widget socket in ``inputs`` (``{"name", "type", "link": null,
"widget": {"name"}}``). The tool only ever topped up the first, so appending a
widget to a node left the two disagreeing -- caught downstream by
tests/test_example_workflows.py, and repaired by hand twice (2026-09-04) before
this was fixed at the source.

Everything is derived from the live ``INPUT_TYPES``, so these tests build their
fixtures from a real registered node rather than inventing a schema that could
drift from the one the tool reads.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fix_workflow_widget_drift import (  # noqa: E402
    ATLAS,
    fix_graph,
    widget_defaults,
    widget_slots,
)

NODE = "AtlasCropROI"          # a node with combos, ints, floats and a mask


def _named_slots(node_type=NODE):
    return [s for s in widget_slots(ATLAS[node_type]) if s.name is not None]


def _graph(node_type=NODE, *, n_values, n_sockets, link_inputs=("solve",)):
    """A save truncated to n_values widget values and n_sockets widget sockets."""
    slots = _named_slots(node_type)
    inputs = [{"name": n, "type": "ATLAS_SOLVE", "link": 1} for n in link_inputs]
    inputs += [{"name": s.name, "type": s.litegraph_type, "link": None,
                "widget": {"name": s.name}} for s in slots[:n_sockets]]
    return {"nodes": [{
        "id": 1, "type": node_type, "inputs": inputs,
        "widgets_values": [s.default for s in widget_slots(ATLAS[node_type])][:n_values],
    }]}


def _sockets(node):
    return [i["widget"]["name"] for i in node["inputs"] if "widget" in i]


def test_widget_defaults_still_returns_just_the_values():
    """The old helper is the public one the sibling tools import; the refactor
    must not change what it returns."""
    assert widget_defaults(ATLAS[NODE]) == [
        s.default for s in widget_slots(ATLAS[NODE])]


def test_it_tops_up_both_representations_together():
    slots = _named_slots()
    g = _graph(n_values=len(slots) - 2, n_sockets=len(slots) - 2)

    fixed, errors = fix_graph(g)

    assert not errors and fixed == 1
    node = g["nodes"][0]
    assert node["widgets_values"] == [s.default for s in widget_slots(ATLAS[NODE])]
    assert _sockets(node) == [s.name for s in slots]


def test_the_appended_sockets_carry_the_litegraph_shape():
    slots = _named_slots()
    g = _graph(n_values=len(slots), n_sockets=len(slots) - 1)

    fix_graph(g)

    added = g["nodes"][0]["inputs"][-1]
    assert added == {"name": slots[-1].name, "type": slots[-1].litegraph_type,
                     "link": None, "widget": {"name": slots[-1].name}}
    assert added["type"] in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")


def test_link_inputs_are_left_exactly_where_they_were():
    """Only widget sockets are appended; a real link must not move or change."""
    slots = _named_slots()
    g = _graph(n_values=len(slots), n_sockets=len(slots) - 2,
               link_inputs=("solve", "source_image"))
    before = [dict(i) for i in g["nodes"][0]["inputs"] if "widget" not in i]

    fix_graph(g)

    after = [i for i in g["nodes"][0]["inputs"] if "widget" not in i]
    assert after == before
    assert g["nodes"][0]["inputs"][:2] == before, "links stay at the head"


def test_a_node_that_serializes_no_widget_sockets_is_left_alone():
    """A compact save with zero widget sockets is VALID, not drift. Inventing a
    full list there is a rewrite, and this tool only ever appends a tail."""
    slots = _named_slots()
    g = _graph(n_values=len(slots) - 1, n_sockets=0)

    fixed, errors = fix_graph(g)

    assert fixed == 1 and not errors          # values still topped up
    assert _sockets(g["nodes"][0]) == []      # sockets untouched


def test_sockets_out_of_class_order_are_reported_not_rewritten():
    slots = _named_slots()
    g = _graph(n_values=len(slots), n_sockets=len(slots) - 2)
    g["nodes"][0]["inputs"][-1]["widget"]["name"] = "not_a_real_widget"

    fixed, errors = fix_graph(g)

    assert fixed == 0
    assert errors and "NOT append-only" in errors[0]


def test_more_sockets_than_the_class_has_is_reported():
    slots = _named_slots()
    g = _graph(n_values=len(slots), n_sockets=len(slots))
    g["nodes"][0]["inputs"].append(
        {"name": "ghost", "type": "INT", "link": None, "widget": {"name": "ghost"}})

    fixed, errors = fix_graph(g)

    assert fixed == 0 and errors


def test_an_already_correct_graph_is_untouched():
    slots = _named_slots()
    g = _graph(n_values=len(slots), n_sockets=len(slots))
    before = json.dumps(g, sort_keys=True)

    fixed, errors = fix_graph(g)

    assert fixed == 0 and not errors
    assert json.dumps(g, sort_keys=True) == before


def test_check_mode_reports_socket_drift_without_writing(tmp_path):
    """--check must SEE inputs drift, or CI passes while the file is broken."""
    slots = _named_slots()
    g = _graph(n_values=len(slots), n_sockets=len(slots) - 1)
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    before = p.read_text(encoding="utf-8")

    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "fix_workflow_widget_drift.py"),
         "--check", str(p)],
        capture_output=True, text=True, cwd=ROOT)

    assert r.returncode == 1, r.stdout
    assert "would top up" in r.stdout
    assert p.read_text(encoding="utf-8") == before, "--check must not write"


def test_it_actually_writes_and_then_comes_back_clean(tmp_path):
    slots = _named_slots()
    g = _graph(n_values=len(slots) - 2, n_sockets=len(slots) - 2)
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")

    tool = str(ROOT / "tools" / "fix_workflow_widget_drift.py")
    assert subprocess.run([sys.executable, tool, str(p)], capture_output=True,
                          text=True, cwd=ROOT).returncode == 0
    again = subprocess.run([sys.executable, tool, "--check", str(p)],
                           capture_output=True, text=True, cwd=ROOT)

    assert again.returncode == 0, again.stdout
    node = json.loads(p.read_text(encoding="utf-8"))["nodes"][0]
    assert _sockets(node) == [s.name for s in slots]


def test_shipping_workflows_have_no_APPENDABLE_socket_drift():
    """Nothing the tool could silently top up may be left sitting in the repo.

    This deliberately does NOT assert a clean exit. Widening the check surfaced
    four shipped workflows whose widget SOCKETS are a subsequence of the class's
    widget order with a GAP in the middle (AtlasDeriveReliefMesh,
    AtlasExportScenePackage) -- the signature of a widget inserted mid-list in
    the node's history rather than appended, which the tool refuses to "fix"
    because guessing which value maps to which widget is exactly how a save gets
    silently rewired. Those are reported for a human, and are a separate problem
    from the append-only drift this tool exists to repair.
    """
    from conftest import is_local_workflow

    files = [p for p in sorted((ROOT / "examples").rglob("*.json"))
             if not is_local_workflow(p)]
    assert files, "examples/ must not be empty or this checks nothing"

    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "fix_workflow_widget_drift.py"),
         "--check", *[str(p) for p in files]],
        capture_output=True, text=True, cwd=ROOT)

    appendable = [ln for ln in r.stdout.splitlines() if "would top up" in ln]
    assert not appendable, (
        "shipped workflows carry drift this tool can repair — run "
        "`python tools/fix_workflow_widget_drift.py examples/*.json`: "
        + "; ".join(appendable))
