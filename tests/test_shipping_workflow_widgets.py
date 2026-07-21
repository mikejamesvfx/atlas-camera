"""Guard against positional-widget drift in the pinned shipping workflows.

ComfyUI serializes each node's widgets into a POSITIONAL ``widgets_values``
array. When a node gains an appended widget, older saved workflows silently
fall out of sync (ComfyUI fills the tail with defaults on load, so nothing
visibly breaks — which is exactly how this drift accumulates unnoticed).

This test derives each Atlas node's expected widget count from its live
``INPUT_TYPES`` (the same ``is_widget`` rule the MCP validator + ComfyUI use)
and asserts the committed shipping workflows match. It only checks ATLAS node
types — third-party/core nodes (SAM3Segment, LoadImage, rgthree, KJ rails,
INPAINT_*) aren't importable in CI and are validated live via the MCP instead.
"""
from __future__ import annotations

import json
from pathlib import Path

from atlas_camera.comfy import node_registry as reg
from atlas_camera.mcp.comfy_http import is_widget

ROOT = Path(__file__).resolve().parents[1]

def _shipping_workflows() -> list[Path]:
    """EVERY LiteGraph workflow under examples/, not a hand-maintained subset.
    The old three-file tuple left examples/showcase/ (22 files) unchecked, so
    append-only drift accumulated there unnoticed — a Mac reviewer's strict
    validator flagged AtlasExportMayaLayers 1≠6 in the hangar showcase. Repair
    with `python tools/fix_workflow_widget_drift.py <files>`. Any non-graph
    JSON is skipped (no top-level "nodes" list)."""
    out = []
    for p in sorted((ROOT / "examples").rglob("*.json")):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(g, dict) and isinstance(g.get("nodes"), list):
            out.append(p)
    return out


ATLAS = {**reg.NODE_CLASS_MAPPINGS, **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS}


def _expected_widget_count(cls) -> int:
    """widgets_values length for a node type: one slot per widget input, plus
    the phantom control_after_generate slot each seed carries."""
    it = cls.INPUT_TYPES()
    items = [(k, v)
             for sec in ("required", "optional")
             for k, v in (it.get(sec) or {}).items()]
    count = 0
    for name, spec in items:
        if is_widget(spec):
            count += 1
            if name in ("seed", "noise_seed"):
                count += 1
    return count


def test_shipping_workflows_have_no_atlas_widget_drift():
    workflows = _shipping_workflows()
    assert workflows, "no shipped workflows discovered under examples/"
    problems = []
    for path in workflows:
        rel = path.relative_to(ROOT / "examples").as_posix()
        wf = json.loads(path.read_text(encoding="utf-8"))
        for node in wf["nodes"]:
            cls = ATLAS.get(node.get("type"))
            if cls is None:
                continue  # third-party / core / virtual — checked live, not here
            want = _expected_widget_count(cls)
            got = len(node.get("widgets_values") or [])
            if want != got:
                problems.append(
                    f"{rel}: {node['type']} id{node['id']} "
                    f"widgets_values {got} != {want} (run "
                    f"tools/fix_workflow_widget_drift.py)")
    assert not problems, "Atlas widget drift:\n" + "\n".join(problems)
