"""No shipped workflow may carry an authoring-machine absolute path.

Every OCIO/RAW/hidden-geometry showcase used to bake the author's own filesystem
paths (``C:\\Users\\miike\\…`` / ``/Users/…``) into node widgets, so a fresh
clone or another OS loaded a workflow pointing at files that do not exist — a
Mac reviewer had to repoint one by hand. This asserts the invariant across
EVERY shipped workflow; repair with:

    python tools/normalize_workflow_paths.py <files>
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_absolute_machine_path(v: object) -> bool:
    if not isinstance(v, str) or len(v) < 3:
        return False
    if v[0].isalpha() and v[1] == ":" and v[2] in "\\/":   # Windows drive
        return True
    return v.startswith("/Users/") or v.startswith("/home/")


def _shipping_workflows() -> list[Path]:
    out = []
    for p in sorted((ROOT / "examples").rglob("*.json")):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(g, dict) and isinstance(g.get("nodes"), list):
            out.append(p)
    return out


def test_no_shipping_workflow_has_absolute_machine_paths():
    workflows = _shipping_workflows()
    assert workflows, "no shipped workflows discovered under examples/"
    problems = []
    for path in workflows:
        rel = path.relative_to(ROOT / "examples").as_posix()
        wf = json.loads(path.read_text(encoding="utf-8"))
        for node in wf["nodes"]:
            wv = node.get("widgets_values")
            if not isinstance(wv, list):
                continue
            for i, v in enumerate(wv):
                if _is_absolute_machine_path(v):
                    problems.append(f"{rel}: {node.get('type')} id{node.get('id')} "
                                    f"widget[{i}] = {v!r}")
    assert not problems, (
        "absolute machine paths in shipped workflows "
        "(run tools/normalize_workflow_paths.py):\n" + "\n".join(problems))
