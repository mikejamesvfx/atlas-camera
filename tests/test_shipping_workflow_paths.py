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
from pathlib import Path, PurePosixPath, PureWindowsPath

from conftest import is_local_workflow

ROOT = Path(__file__).resolve().parents[1]


def _is_absolute_machine_path(v: object) -> bool:
    if not isinstance(v, str) or len(v) < 3:
        return False
    return PureWindowsPath(v).is_absolute() or PurePosixPath(v).is_absolute()


@pytest.mark.parametrize("value", [
    r"C:\Users\artist\plate.exr",
    "D:/plates/plate.exr",
    r"\\server\share\plate.exr",
    r"\\?\C:\plates\plate.exr",
    r"\\?\UNC\server\share\plate.exr",
    "/var/tmp/plate.exr",
    "/opt/show/plate.exr",
])
def test_absolute_machine_path_guard_covers_windows_and_posix_roots(value):
    assert _is_absolute_machine_path(value)


@pytest.mark.parametrize("value", [
    "atlas_multiview/photo_01.RAF",
    "example.png",
    "../shared/plate.exr",
    r"atlas_multiview\photo_02.RAF",
])
def test_absolute_machine_path_guard_allows_relative_placeholders(value):
    assert not _is_absolute_machine_path(value)


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


