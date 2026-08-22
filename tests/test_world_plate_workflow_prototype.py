from __future__ import annotations

import pytest


import json
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from atlas_camera.comfy import node_registry
from tools.build_world_plate_workflows import reconcile_object_info

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "examples" / "prototypes" / "atlas_world_plate_prototype_workflow.json"
API = ROOT / "examples" / "prototypes" / "atlas_world_plate_prototype_api.json"


def _absolute(value: object) -> bool:
    return isinstance(value, str) and (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute())


def test_two_shot_prototype_is_explicitly_non_shipping_and_contains_required_nodes():
    graph = json.loads(GUI.read_text(encoding="utf-8"))
    assert graph["extra"]["non_shipping"] is True
    types = {node["type"] for node in graph["nodes"]}
    required = {"AtlasOpenRealPlate", "AtlasReadLockedPlatePlan", "AtlasRecordPlateAttempt", "AtlasExportPlateHandoff", "AtlasSAM3Mask", "AtlasInpaintCrop", "AtlasSDXLInpaint", "AtlasInpaintStitch"}
    assert required <= types
    assert {node["properties"]["prototype_shot"] for node in graph["nodes"]} == {"DSC_2245", "DSC_2552"}


def test_prototype_workflow_has_no_absolute_paths_and_gui_api_node_types_match():
    graph = json.loads(GUI.read_text(encoding="utf-8"))
    api = json.loads(API.read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        assert all(not _absolute(value) for value in node.get("widgets_values", []))
    assert {node["type"] for node in graph["nodes"]} == {value["class_type"] for value in api.values()}


def test_prototype_links_are_bidirectional_for_plate_handoffs():
    graph = json.loads(GUI.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    for link_id, source, source_slot, target, target_slot, _ in graph["links"]:
        assert any(link_id == item.get("link") for item in nodes[target].get("inputs", []))
        assert any(link_id in (item.get("links") or []) for item in nodes[source].get("outputs", []))


def test_prototype_link_ids_counters_outputs_and_socket_types_match_registry():
    graph = json.loads(GUI.read_text(encoding="utf-8"))
    assert len({link[0] for link in graph["links"]}) == len(graph["links"])
    assert graph["last_link_id"] >= max(link[0] for link in graph["links"])
    assert graph["last_node_id"] >= max(node["id"] for node in graph["nodes"])
    for node in graph["nodes"]:
        cls = node_registry.NODE_CLASS_MAPPINGS.get(node["type"])
        if cls is None:
            continue
        assert len(node.get("outputs", [])) == len(cls.RETURN_TYPES)
        specs = {}
        declared = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            specs.update(declared.get(section, {}))
        for item in node.get("inputs", []):
            assert item["name"] in specs
            expected = specs[item["name"]][0] if isinstance(specs[item["name"]], tuple) else specs[item["name"]]
            assert item["type"] == expected


def test_prototype_functional_rectangles_do_not_overlap():
    graph = json.loads(GUI.read_text(encoding="utf-8"))
    functional = [node for node in graph["nodes"] if node["type"] != "LoadImage"]
    for left_index, left in enumerate(functional):
        lx, ly = left["pos"]
        lw, lh = left["size"]
        for right in functional[left_index + 1:]:
            rx, ry = right["pos"]
            rw, rh = right["size"]
            assert lx + lw <= rx or rx + rw <= lx or ly + lh <= ry or ry + rh <= ly


def test_builder_has_explicit_offline_fallback_and_rejects_schema_mismatch():
    assert reconcile_object_info() == "offline-local-signatures"
    with __import__("pytest").raises(RuntimeError, match="absent"):
        reconcile_object_info({})


def test_builder_rejects_live_socket_type_and_widget_order_drift():
    names = {"AtlasOpenRealPlate", "AtlasReadLockedPlatePlan", "AtlasRecordPlateAttempt", "AtlasExportPlateHandoff", "AtlasSAM3Mask", "AtlasInpaintCrop", "AtlasSDXLInpaint", "AtlasInpaintStitch"}
    fake = {}
    for name in names:
        cls = node_registry.NODE_CLASS_MAPPINGS[name]
        inputs = cls.INPUT_TYPES()
        fake[name] = {"input": {section: dict(inputs.get(section) or {}) for section in ("required", "optional")}, "output": list(cls.RETURN_TYPES)}
    assert reconcile_object_info(fake) == "live-object-info"
    fake["AtlasSAM3Mask"]["input"]["required"]["image"] = ["MASK"]
    with __import__("pytest").raises(RuntimeError, match="socket type"):
        reconcile_object_info(fake)
    fake["AtlasSAM3Mask"]["input"]["required"]["image"] = ["IMAGE"]
    fake["AtlasSDXLInpaint"]["input"]["required"] = dict(reversed(list(fake["AtlasSDXLInpaint"]["input"]["required"].items())))
    with __import__("pytest").raises(RuntimeError, match="widget order"):
        reconcile_object_info(fake)


def test_builder_rejects_live_widget_defaults_ranges_options_and_flags():
    names = {"AtlasOpenRealPlate", "AtlasReadLockedPlatePlan", "AtlasRecordPlateAttempt", "AtlasExportPlateHandoff", "AtlasSAM3Mask", "AtlasInpaintCrop", "AtlasSDXLInpaint", "AtlasInpaintStitch"}
    fake = {}
    for name in names:
        cls = node_registry.NODE_CLASS_MAPPINGS[name]
        inputs = cls.INPUT_TYPES()
        fake[name] = {"input": {section: dict(inputs.get(section) or {}) for section in ("required", "optional")}, "output": list(cls.RETURN_TYPES)}
    assert reconcile_object_info(fake) == "live-object-info"
    fake["AtlasSAM3Mask"]["input"]["optional"]["confidence_threshold"] = ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01})
    with __import__("pytest").raises(RuntimeError, match="config"):
        reconcile_object_info(fake)


def test_builder_rejects_live_input_add_remove_mismatch():
    names = {"AtlasOpenRealPlate", "AtlasReadLockedPlatePlan", "AtlasRecordPlateAttempt", "AtlasExportPlateHandoff", "AtlasSAM3Mask", "AtlasInpaintCrop", "AtlasSDXLInpaint", "AtlasInpaintStitch"}
    fake = {}
    for name in names:
        cls = node_registry.NODE_CLASS_MAPPINGS[name]
        inputs = cls.INPUT_TYPES()
        fake[name] = {"input": {section: dict(inputs.get(section) or {}) for section in ("required", "optional")}, "output": list(cls.RETURN_TYPES)}
    del fake["AtlasSAM3Mask"]["input"]["optional"]["confidence_threshold"]
    with __import__("pytest").raises(RuntimeError, match="inputs differ"):
        reconcile_object_info(fake)
    fake["AtlasSAM3Mask"]["input"]["optional"]["confidence_threshold"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "lazy": True})
    with __import__("pytest").raises(RuntimeError, match="config"):
        reconcile_object_info(fake)
