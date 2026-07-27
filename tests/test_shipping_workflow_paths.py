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

from conftest import is_local_workflow

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
        if is_local_workflow(p):
            continue  # artist working copy — see conftest.is_local_workflow
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


def test_local_workflow_predicate_ignores_the_checkout_location():
    r"""`is_local_workflow` must judge the path RELATIVE to the repo.

    It used to scan the absolute path's parts for "local", which folded the
    checkout's own location into the predicate. A worktree under
    `AppData\Local\Temp` — or any clone beneath `~/.local/share` — made every
    shipping workflow look like an artist's working copy, so the workflow
    suites discovered NOTHING and passed vacuously. Found 2026-07-28 when a
    temp worktree produced 18 unrelated "failures".
    """
    from conftest import is_local_workflow

    shipped = ROOT / "examples" / "atlas_path_guided_hole_repair_workflow.json"
    assert shipped.is_file(), "fixture moved"
    assert not is_local_workflow(shipped)

    # The real markers still work.
    assert is_local_workflow(ROOT / "examples" / "something-edit.json")
    assert is_local_workflow(ROOT / "examples" / "local" / "scratch.json")

    # And the discovery itself must be non-empty — a predicate that hides
    # everything is how this went unnoticed.
    assert _shipping_workflows(), "no shipping workflows discovered"
