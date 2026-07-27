import json

import pytest

from tools.compact_path_repair_bake import (
    _upgrade_api_depth_gates,
    _upgrade_ui_depth_gates,
    compact_client_data,
)


def _client_data():
    return json.dumps({
        "path_frames": ["frame-0", "frame-1", "frame-2"],
        "camera_path": {
            "keyframes": [],
            "fps": 24,
            "frame_count": 3,
            "lens_scale": 0.8,
        },
        "atlas_proxy_path": {
            "transport": "jpeg_base64_proxy_ldr",
            "frame_count": 3,
        },
    })


def test_compact_client_data_keeps_path_and_last_preview_only():
    compacted, old_count, selected = compact_client_data(_client_data())
    data = json.loads(compacted)

    assert old_count == 3
    assert selected == 2
    assert data["path_frames"] == ["frame-2"]
    assert data["camera_path"]["frame_count"] == 3
    assert data["camera_path"]["baked_frame_indices"] == [2]
    assert data["atlas_proxy_path"]["stored_frame_count"] == 1
    assert data["atlas_proxy_path"]["frame_indices"] == [2]


def test_compact_client_data_can_select_an_earlier_offset():
    compacted, _, selected = compact_client_data(
        _client_data(), frame_offset_from_end=1)
    data = json.loads(compacted)

    assert selected == 1
    assert data["path_frames"] == ["frame-1"]
    assert data["camera_path"]["baked_frame_indices"] == [1]


def test_compact_client_data_rejects_missing_explicit_frame():
    data = json.loads(_client_data())
    data["path_frames"] = ["frame-2"]
    data["camera_path"]["baked_frame_indices"] = [2]

    with pytest.raises(ValueError, match="path frame 1 is not present"):
        compact_client_data(
            json.dumps(data), frame_offset_from_end=1)


def test_upgrade_ui_depth_gate_appends_widget_only_once():
    workflow = {
        "nodes": [{
            "type": "AtlasPlanarHolePatch",
            "title": "PASS 2 · scoped repair (65× edge)",
            "widgets_values": ["", 2, 512, 30, 0.3, 0.04, True, 0.2, 65],
        }],
    }

    _upgrade_ui_depth_gates(workflow, 2.0)
    _upgrade_ui_depth_gates(workflow, 3.0)

    assert workflow["nodes"][0]["widgets_values"][-1] == 2.0
    assert workflow["nodes"][0]["widgets_values"][-2] == 40.0
    assert "40×" in workflow["nodes"][0]["title"]
    assert len(workflow["nodes"][0]["widgets_values"]) == 10


def test_upgrade_api_depth_gate_preserves_explicit_value():
    prompt = {
        "22": {
            "class_type": "AtlasPlanarHolePatch",
            "inputs": {
                "max_patch_depth_factor": 1.5,
                "max_patch_edge_factor": 65,
            },
            "_meta": {"title": "PASS 2 · scoped repair (65× edge)"},
        },
        "5": {
            "class_type": "AtlasPlanarHolePatch",
            "inputs": {},
        },
    }

    _upgrade_api_depth_gates(prompt, 2.0)

    assert prompt["22"]["inputs"]["max_patch_depth_factor"] == 1.5
    assert prompt["22"]["inputs"]["max_patch_edge_factor"] == 40.0
    assert "40×" in prompt["22"]["_meta"]["title"]
    assert prompt["5"]["inputs"]["max_patch_depth_factor"] == 2.0
