"""atlas_camera.mcp.comfy_http — the MCP server's stdlib workflow plumbing.

Pure-offline tests over a synthetic /object_info + hand-built UI graphs: the
UI→API flattening (rails, mute/bypass, phantom widget slots), gate overrides,
and the validator. No network, no mcp SDK needed.
"""
import json

import pytest

from atlas_camera.mcp import comfy_http as C

OI = {
    "LoadImage": {
        "input": {"required": {"image": [["a.png", "b.png"], {"image_upload": True}]}},
        "output_name": ["IMAGE", "MASK"], "output": ["IMAGE", "MASK"],
    },
    "AtlasSolveGate": {
        "input": {"required": {"solve": ["ATLAS_SOLVE"], "source_image": ["IMAGE"]},
                  "optional": {"proceed": ["BOOLEAN", {"default": False}],
                               "approved_for": ["STRING", {"default": ""}]}},
        "output_name": ["solve", "report"], "output": ["ATLAS_SOLVE", "STRING"],
    },
    "FakeSolve": {
        "input": {"required": {"image": ["IMAGE"]},
                  "optional": {"seed": ["INT", {"default": 0}],
                               "strength": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}]}},
        "output_name": ["solve"], "output": ["ATLAS_SOLVE"],
    },
    "FakeSink": {
        "input": {"required": {"solve": ["ATLAS_SOLVE"]}},
        "output_name": [], "output": [],
    },
}


def _out(name, type_, links=None):
    return {"name": name, "type": type_, "links": links if links is not None else [], "slot_index": 0}


def _ui():
    """LoadImage → FakeSolve → SetNode rail → GetNode → Gate → FakeSink."""
    nodes = [
        {"id": 1, "type": "LoadImage", "mode": 0, "inputs": [],
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1, 5], "slot_index": 0},
                     {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}],
         "widgets_values": ["a.png", "image"]},
        {"id": 2, "type": "FakeSolve", "mode": 0,
         "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
         "outputs": [_out("solve", "ATLAS_SOLVE", [2])],
         # seed carries the phantom control_after_generate slot
         "widgets_values": [123, "fixed", 1.5]},
        {"id": 3, "type": "SetNode", "mode": 0,
         "inputs": [{"name": "ATLAS_SOLVE", "type": "ATLAS_SOLVE", "link": 2}],
         "outputs": [{"name": "ATLAS_SOLVE", "type": "ATLAS_SOLVE", "links": None}],
         "widgets_values": ["solve_rail"]},
        {"id": 4, "type": "GetNode", "mode": 0, "inputs": [],
         "outputs": [_out("ATLAS_SOLVE", "ATLAS_SOLVE", [3])],
         "widgets_values": ["solve_rail"]},
        {"id": 5, "type": "AtlasSolveGate", "mode": 0,
         "inputs": [{"name": "solve", "type": "ATLAS_SOLVE", "link": 3},
                    {"name": "source_image", "type": "IMAGE", "link": 5},
                    {"name": "proceed", "type": "BOOLEAN", "link": None},
                    {"name": "approved_for", "type": "STRING", "link": None}],
         "outputs": [_out("solve", "ATLAS_SOLVE", [4]), _out("report", "STRING", [])],
         "widgets_values": [False, ""]},
        {"id": 6, "type": "FakeSink", "mode": 0,
         "inputs": [{"name": "solve", "type": "ATLAS_SOLVE", "link": 4}],
         "outputs": [], "widgets_values": []},
    ]
    links = [
        [1, 1, 0, 2, 0, "IMAGE"],
        [2, 2, 0, 3, 0, "ATLAS_SOLVE"],
        [3, 4, 0, 5, 0, "ATLAS_SOLVE"],
        [5, 1, 0, 5, 1, "IMAGE"],
        [4, 5, 0, 6, 0, "ATLAS_SOLVE"],
    ]
    return {"nodes": nodes, "links": links, "groups": []}


def test_ui_to_api_flattens_rails_and_phantom_slots():
    api = C.ui_to_api(_ui(), OI)
    assert set(api) == {"1", "2", "5", "6"}          # Set/Get gone
    assert api["1"]["inputs"]["image"] == "a.png"     # upload slot skipped
    assert api["2"]["inputs"]["seed"] == 123          # control_after_generate skipped
    assert api["2"]["inputs"]["strength"] == 1.5
    assert api["5"]["inputs"]["solve"] == ["2", 0]    # rail resolved to source
    assert api["5"]["inputs"]["proceed"] is False
    assert api["6"]["inputs"]["solve"] == ["5", 0]


def test_gate_overrides_and_apply():
    ui = _ui()
    api = C.ui_to_api(ui, OI)
    ov = C.gate_overrides(ui)
    assert ov == {"5.proceed": True}
    applied = C.apply_overrides(api, ov)
    assert api["5"]["inputs"]["proceed"] is True
    assert applied == ["5.proceed=True"]
    with pytest.raises(KeyError):
        C.apply_overrides(api, {"99.x": 1})


def test_bypassed_node_forwards_same_type_input():
    ui = _ui()
    ui["nodes"][4]["mode"] = 4  # bypass the gate
    api = C.ui_to_api(ui, OI)
    assert "5" not in api
    assert api["6"]["inputs"]["solve"] == ["2", 0]  # forwarded through


def test_muted_terminal_node_drops():
    ui = _ui()
    ui["nodes"][5]["mode"] = 2  # mute the sink (nothing consumes it)
    api = C.ui_to_api(ui, OI)
    assert "6" not in api


def test_muted_producer_with_consumer_raises():
    ui = _ui()
    ui["nodes"][4]["mode"] = 2  # mute the gate while the sink consumes it
    with pytest.raises(RuntimeError, match="MUTED"):
        C.ui_to_api(ui, OI)


def test_validate_clean_graph():
    errs, warns = C.validate_ui(_ui(), OI)
    assert errs == []
    assert warns == []


def test_validate_catches_drift_range_and_rails():
    ui = _ui()
    ui["nodes"][1]["widgets_values"] = [123, "fixed"]       # missing strength
    ui["nodes"][3]["widgets_values"] = ["missing_rail"]     # Get without Set
    errs, _ = C.validate_ui(ui, OI)
    assert any("widgets_values" in e for e in errs)
    assert any("missing_rail" in e for e in errs)
    ui2 = _ui()
    ui2["nodes"][1]["widgets_values"] = [123, "fixed", 9.0]  # strength > max 2.0
    errs2, _ = C.validate_ui(ui2, OI)
    assert any("ABOVE max" in e for e in errs2)


def test_validate_star_inputs_accepted():
    ui = _ui()
    ui["nodes"][5]["inputs"][0]["type"] = "*"
    errs, _ = C.validate_ui(ui, OI)
    assert not any("TYPE" in e for e in errs)


def test_shipping_workflows_flatten_against_recorded_shapes():
    """Every committed showcase/experimental workflow must at least parse and
    resolve its rails structurally (no oi lookups — VIRTUAL/link walk only).
    Full validation runs against a live server; here we pin the JSONs are
    structurally sound (bidirectional links, resolvable rails)."""
    import pathlib
    roots = [pathlib.Path("examples/showcase"), pathlib.Path("examples/experimental")]
    checked = 0
    for root in roots:
        for p in sorted(root.glob("*_workflow.json")) + sorted(root.glob("atlas_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            nodes = {n["id"]: n for n in d["nodes"]}
            sets = {n["widgets_values"][0] for n in d["nodes"] if n["type"] == "SetNode"}
            for n in d["nodes"]:
                if n["type"] == "GetNode":
                    assert n["widgets_values"][0] in sets, f"{p.name}: dangling rail"
            for l in d["links"]:
                lid, sid, sslot, tid, tslot = l[:5]
                assert lid in (nodes[sid]["outputs"][sslot].get("links") or []), f"{p.name}: link {lid}"
                assert nodes[tid]["inputs"][tslot].get("link") == lid, f"{p.name}: link {lid} dst"
            checked += 1
    assert checked >= 12  # 11 showcase + experimental set
