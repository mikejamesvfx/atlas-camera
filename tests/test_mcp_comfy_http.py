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


def test_summarize_viewport_layer_exposes_hidden_support_qa():
    source = {
        "name": "background_clean",
        "priority": 10.0,
        "near_m": 0.0,
        "far_m": None,
        "band_geometry": "relief",
        "mask_b64": "png",
        "normal_map_b64": "",
        "hidden_mask_b64": "",
        "proxy_geometry": [
            {"vertices": [0.0] * 9,
             "metadata": {"n_filled_cells": 0, "torn_fraction": 0.29,
                          "stretch_ratio_p95": 3.2}},
            {"vertices": [0.0] * 6,
             "metadata": {"n_filled_cells": 4, "torn_fraction": 0.12,
                          "stretch_ratio_p95": 2.1}},
        ],
    }

    summary = C.summarize_viewport_layer(source)

    assert summary["verts"] == 5
    assert summary["n_filled_cells"] == 4
    assert summary["torn_fraction_max"] == pytest.approx(0.29)
    assert summary["stretch_ratio_p95_max"] == pytest.approx(3.2)
    assert summary["matte"] is True


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


def test_output_assessment_overrides_and_bounded_report_extraction():
    api = {
        "4": {"class_type": "AtlasAssessOutput", "inputs": {"enabled": False}},
        "5": {"class_type": "PreviewImage", "inputs": {}},
    }
    assert C.output_assessment_overrides(api) == {"4.enabled": True}

    assessment = {"schema": 1, "verdict": "warn", "status": "complete"}
    outputs = {
        "4": {
            "text": ["terminal report"],
            "json_path": ["atlas_debug/output.json"],
            "atlas_output_assessment": [json.dumps(assessment)],
            "images": [{"filename": "must-not-leak.png"}],
        },
        "5": {"images": [{"filename": "preview.png"}]},
    }
    reports = C.collect_output_reports(outputs)
    assert reports == {"4": {
        "text": "terminal report",
        "json_path": "atlas_debug/output.json",
        "evidence_path": "",
        "coverage_path": "",
        "source_reference_path": "",
        "assessment": assessment,
    }}
    assert "images" not in json.dumps(reports)


def test_queue_wait_tolerates_history_commit_race(monkeypatch):
    history_calls = 0

    def fake_http(url, *_args, **_kwargs):
        nonlocal history_calls
        if url.endswith("/prompt"):
            return {"prompt_id": "prompt-race"}
        if url.endswith("/history/prompt-race"):
            history_calls += 1
            if history_calls == 1:
                return {}
            return {"prompt-race": {
                "status": {"completed": True, "messages": []},
                "outputs": {},
            }}
        if url.endswith("/queue"):
            return {"queue_running": [], "queue_pending": []}
        raise AssertionError(url)

    monkeypatch.setattr(C, "http_json", fake_http)
    monkeypatch.setattr(C.time, "sleep", lambda _seconds: None)
    result = C.queue_and_wait({}, timeout=10, poll_s=0)
    assert result["completed"] is True
    assert result["errors"] == []
    assert history_calls == 2


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


def _subgraph_ui():
    """LoadImage -> official v1 subgraph(FakeSolve) -> FakeSink."""
    subgraph_id = "11111111-2222-4333-8444-555555555555"
    load = {
        "id": 1, "type": "LoadImage", "mode": 0, "inputs": [],
        "outputs": [_out("IMAGE", "IMAGE", [1]), _out("MASK", "MASK", [])],
        "widgets_values": ["a.png", "image"],
    }
    instance = {
        "id": 2, "type": subgraph_id, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
        "outputs": [_out("solve", "ATLAS_SOLVE", [2])],
        "properties": {"proxyWidgets": [
            ["1", "seed"], ["1", "control_after_generate"], ["1", "strength"],
        ]},
        "widgets_values": [999, "increment", 1.75],
    }
    sink = {
        "id": 3, "type": "FakeSink", "mode": 0,
        "inputs": [{"name": "solve", "type": "ATLAS_SOLVE", "link": 2}],
        "outputs": [], "widgets_values": [],
    }
    definition = {
        "id": subgraph_id, "version": 1, "name": "Synthetic solve",
        "inputs": [{"id": "in", "name": "image", "type": "IMAGE",
                    "linkIds": [10]}],
        "outputs": [{"id": "out", "name": "solve", "type": "ATLAS_SOLVE",
                     "linkIds": [11]}],
        "nodes": [{
            "id": 1, "type": "FakeSolve", "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": 10}],
            "outputs": [_out("solve", "ATLAS_SOLVE", [11])],
            "widgets_values": [123, "fixed", 1.5],
        }],
        "links": [
            {"id": 10, "origin_id": -10, "origin_slot": 0,
             "target_id": 1, "target_slot": 0, "type": "IMAGE"},
            {"id": 11, "origin_id": 1, "origin_slot": 0,
             "target_id": -20, "target_slot": 0, "type": "ATLAS_SOLVE"},
        ],
    }
    return {
        "nodes": [load, instance, sink],
        "links": [[1, 1, 0, 2, 0, "IMAGE"],
                  [2, 2, 0, 3, 0, "ATLAS_SOLVE"]],
        "groups": [], "definitions": {"subgraphs": [definition]},
    }


def test_official_subgraph_expands_and_applies_proxy_widgets():
    expanded = C.expand_subgraphs(_subgraph_ui(), OI)
    assert [node["type"] for node in expanded["nodes"]] == [
        "LoadImage", "FakeSink", "FakeSolve"]
    fake = next(node for node in expanded["nodes"] if node["type"] == "FakeSolve")
    assert fake["widgets_values"] == [999, "increment", 1.75]
    assert len(expanded["links"]) == 2

    api = C.ui_to_api(_subgraph_ui(), OI)
    fake_id = str(fake["id"])
    assert api[fake_id]["inputs"]["image"] == ["1", 0]
    assert api[fake_id]["inputs"]["seed"] == 999
    assert api[fake_id]["inputs"]["strength"] == 1.75
    assert api["3"]["inputs"]["solve"] == [fake_id, 0]
    assert C.validate_ui(_subgraph_ui(), OI) == ([], [])


def test_shipping_workflows_flatten_against_recorded_shapes():
    """The six shipped UI workflows must parse and resolve their KJ rails
    structurally (no oi lookups — VIRTUAL/link walk only). Full validation
    runs against a live server; here we pin the JSONs are structurally sound
    (bidirectional links, resolvable rails). Was over examples/showcase +
    examples/experimental; those trees were removed in the 0.8.1 trim. This
    now walks the three base workflows and their three agentic QA variants."""
    import pathlib
    root = pathlib.Path("examples")
    checked = 0
    for p in sorted(root.glob("*_workflow.json")):
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
    assert checked == 7
