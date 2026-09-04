"""`AtlasExportScenePackage`'s widget order is a saved-workflow contract.

`widgets_values` is POSITIONAL. Inserting a widget anywhere but the end
re-points every value after it at the wrong widget, in every graph already
saved — and nothing raises, because the values are all strings.

That happened on 2026-08-26 (`cf76682`, "package layered cleanplate reliefs"):
`cleanplate_path` and `cleanplate_mesh_path` went in BEFORE `observation_id`.
Three shipped workflows had been saved with the five-widget order, and when
`tools/fix_workflow_widget_drift.py` later topped them up to eight — correctly,
by its own append-only rule — the result read `observation_id`'s "obs_001" as
`cleanplate_path`, and exported with NO observation id and a cleanplate path
pointing at an evidence label.

The fix was to move the two widgets back to the end, which is where an
append-only history would have put them, so every save made before the insert
reads correctly again. These tests pin the order so it cannot drift a second
time, and check the shipped graphs actually decode to sane values.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS  # noqa: E402

NODE = "AtlasExportScenePackage"

#: The order a save's positional values are read against. `observation_id` MUST
#: stay at index 4: three shipped workflows and every graph an artist saved
#: before 2026-08-26 put it there.
EXPECTED_WIDGET_ORDER = [
    "output_dir", "scene_id", "plate_path", "relief_mesh_path",
    "observation_id",
    # Appended 2026-08-26 with the layered cleanplate package, and appended is
    # the operative word — they were briefly inserted mid-list instead.
    "cleanplate_path", "cleanplate_mesh_path", "cleanplate_observation_id",
]


def _widget_names(cls):
    from atlas_camera.mcp.comfy_http import is_widget

    it = cls.INPUT_TYPES()
    return [n for sec in ("required", "optional")
            for n, sp in (it.get(sec) or {}).items() if is_widget(sp)]


def test_the_widget_order_is_pinned():
    assert _widget_names(NODE_CLASS_MAPPINGS[NODE]) == EXPECTED_WIDGET_ORDER


def test_the_cleanplate_widgets_come_after_observation_id():
    """Stated separately from the list above, because THIS is the property a
    future edit is likely to break without noticing."""
    order = _widget_names(NODE_CLASS_MAPPINGS[NODE])
    assert order.index("observation_id") < order.index("cleanplate_path")
    assert order.index("observation_id") < order.index("cleanplate_mesh_path")


def _shipped_instances():
    out = []
    for p in sorted((ROOT / "examples").glob("*.json")):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(g, dict) or not isinstance(g.get("nodes"), list):
            continue
        for n in g["nodes"]:
            if n.get("type") == NODE:
                out.append((p.name, n))
    return out


@pytest.mark.parametrize("name,node", _shipped_instances(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_shipped_graphs_decode_to_sane_values(name, node):
    """Read each save the way ComfyUI reads it — positionally — and check the
    values land on widgets they could plausibly belong to."""
    order = _widget_names(NODE_CLASS_MAPPINGS[NODE])
    values = node.get("widgets_values") or []
    assert len(values) == len(order), f"{name}: widget drift"
    got = dict(zip(order, values))

    assert got["observation_id"] == "obs_001", (
        f"{name}: observation_id decoded as {got['observation_id']!r} — the "
        "evidence identity landed on the wrong widget")
    assert got["cleanplate_observation_id"] == "obs_cleanplate"
    # A path widget must hold a path or nothing, never an evidence label.
    for key in ("cleanplate_path", "cleanplate_mesh_path"):
        assert got[key] == "" or "/" in got[key] or "\\" in got[key], (
            f"{name}: {key} decoded as {got[key]!r}, which is not a path")


def test_the_signature_still_matches_the_declared_order():
    """ComfyUI calls by keyword, but the repo pins signature order against
    INPUT_TYPES (tests/test_widget_order_pins.py) — reordering one means
    reordering the other."""
    import inspect

    cls = NODE_CLASS_MAPPINGS[NODE]
    params = list(inspect.signature(getattr(cls, cls.FUNCTION)).parameters)
    params = [p for p in params if p != "self"]
    widgets = _widget_names(cls)
    assert [p for p in params if p in widgets] == widgets
