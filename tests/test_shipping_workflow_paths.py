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
import pytest
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
    if not workflows:
        # Skip, not fail: this guard exists so the suite cannot pass by testing
        # NOTHING. While examples/ is deliberately empty between the 2026-07-31
        # cull and the replacement set, an empty result is the expected state and
        # the check relights by itself the moment a workflow lands.
        pytest.skip("examples/ is empty between the workflow cull and its rebuild")
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


def test_multiview_raw_workflow_uses_input_relative_placeholders():
    path = ROOT / "examples" / "atlas_multiview_raw_qwen_workflow.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    raw_widgets = [
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "AtlasLoadRAW"
    ]
    assert raw_widgets == [
        "atlas_multiview/photo_01.RAF",
        "atlas_multiview/photo_02.RAF",
        "atlas_multiview/photo_03.RAF",
    ]
    assert all(not _is_absolute_machine_path(value) for value in raw_widgets)


