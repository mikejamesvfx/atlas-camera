"""Build the non-shipping, two-shot evidence-first prototype graph."""
from __future__ import annotations
import json
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "prototypes"
BRANCHES = {
    "DSC_2245": (1, "car", "episodes/DSC_2245.json", "plans/DSC_2245.locked.json", "result/DSC_2245_attempt.json", "exports/DSC_2245.nuke.json"),
    "DSC_2552": (11, "signs", "episodes/DSC_2552.json", "plans/DSC_2552.locked.json", "result/DSC_2552_attempt.json", "exports/DSC_2552.nuke.json"),
}


def reconcile_object_info(object_info=None, *, server: str | None = None) -> str:
    """Reconcile local workflow assumptions with Comfy's live schema.

    Offline generation deliberately uses the four adapter signatures pinned by
    the local registry. When a snapshot or server is supplied, absence or any
    output/required-input mismatch is a hard failure rather than silently
    emitting a stale graph.
    """
    if object_info is None and server:
        with urllib.request.urlopen(server.rstrip("/") + "/object_info", timeout=30) as fh:
            object_info = json.load(fh)
    if object_info is None:
        return "offline-local-signatures"
    from atlas_camera.comfy import node_registry
    names = {"AtlasOpenRealPlate", "AtlasReadLockedPlatePlan", "AtlasRecordPlateAttempt", "AtlasExportPlateHandoff", "AtlasSAM3Mask", "AtlasInpaintCrop", "AtlasSDXLInpaint", "AtlasInpaintStitch"}
    for name in names:
        local = node_registry.NODE_CLASS_MAPPINGS.get(name)
        live = object_info.get(name)
        if local is None or live is None:
            raise RuntimeError(f"schema reconciliation failed: {name} is absent")
        if tuple(live.get("output", ())) != tuple(local.RETURN_TYPES):
            raise RuntimeError(f"schema reconciliation failed: {name} outputs differ")
        local_inputs = local.INPUT_TYPES()
        live_inputs = live.get("input", {})
        for section in ("required", "optional"):
            local_names = set((local_inputs.get(section) or {}).keys())
            live_names = set((live_inputs.get(section) or {}).keys())
            if local_names != live_names:
                raise RuntimeError(f"schema reconciliation failed: {name}.{section} inputs differ")
            for input_name in local_names:
                local_spec = local_inputs[section][input_name]
                live_spec = live_inputs[section][input_name]
                local_kind = local_spec[0] if isinstance(local_spec, (list, tuple)) else local_spec
                live_kind = live_spec[0] if isinstance(live_spec, (list, tuple)) else live_spec
                if isinstance(local_kind, list):
                    local_kind = "COMBO"
                if isinstance(live_kind, list):
                    live_kind = "COMBO"
                if local_kind != live_kind:
                    raise RuntimeError(f"schema reconciliation failed: {name}.{input_name} socket type differs")
                if _input_signature(local_spec) != _input_signature(live_spec):
                    raise RuntimeError(f"schema reconciliation failed: {name}.{input_name} config differs")
        local_widgets = _local_widget_names(local_inputs)
        live_widgets = _live_widget_names(live_inputs)
        if local_widgets != live_widgets:
            raise RuntimeError(f"schema reconciliation failed: {name} widget order differs")
        workflow = build_gui()
        for node in workflow["nodes"]:
            if node["type"] != name:
                continue
            values = node.get("widgets_values") or []
            if len(values) != len(live_widgets):
                raise RuntimeError(f"schema reconciliation failed: {name} widget count differs")
            _validate_widget_values(name, values, live_inputs, live_widgets)
    return "live-object-info"


def _local_widget_names(inputs):
    names = []
    for section in ("required", "optional"):
        for name, spec in (inputs.get(section) or {}).items():
            if not isinstance(spec, (list, tuple)) or len(spec) <= 1:
                continue
            if isinstance(spec[1], dict) and spec[1].get("forceInput"):
                continue
            if spec[0] not in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO") and not isinstance(spec[0], list):
                continue
            names.append(name)
            if name in ("seed", "noise_seed"):
                names.append("__control_after_generate__")
    return names


def _input_signature(spec):
    """Canonical live/local input contract, excluding presentation text."""
    kind = spec[0] if isinstance(spec, (list, tuple)) else spec
    cfg = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    options = list(kind) if isinstance(kind, list) else list(cfg.get("options") or []) if kind == "COMBO" else None
    default = cfg.get("default", options[0] if options else None)
    def flag(*names):
        for name in names:
            if name in cfg:
                return cfg[name]
        return None
    return (
        "COMBO" if options is not None else kind,
        default,
        cfg.get("min"), cfg.get("max"), cfg.get("step"),
        tuple(options) if options is not None else None,
        bool(flag("forceInput", "force_input") or False),
        bool(flag("defaultInput", "default_input") or False),
        bool(flag("multiline") or False),
        bool(flag("lazy") or False), bool(flag("rawLink", "raw_link") or False),
        bool(flag("dynamicPrompts", "dynamic_prompts") or False),
        flag("round"), bool(flag("advanced") or False), bool(flag("socketless") or False),
        flag("controlAfterGenerate", "control_after_generate", "control_after_generate_mode"),
    )


def _live_widget_names(inputs):
    names = []
    for section in ("required", "optional"):
        for name, spec in (inputs.get(section) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if cfg.get("forceInput"):
                continue
            kind = spec[0]
            if kind in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO") or isinstance(kind, list):
                names.append(name)
                if name in ("seed", "noise_seed"):
                    names.append("__control_after_generate__")
    return names


def _validate_widget_values(name, values, inputs, widget_names):
    index = 0
    for widget_name in widget_names:
        value = values[index]
        index += 1
        if widget_name == "__control_after_generate__":
            continue
        spec = next(spec for section in ("required", "optional") for nm, spec in (inputs.get(section) or {}).items() if nm == widget_name)
        kind = spec[0]
        cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        options = kind if isinstance(kind, list) else cfg.get("options") if kind == "COMBO" else None
        if options is not None and options and value not in options:
            raise RuntimeError(f"schema reconciliation failed: {name}.{widget_name} value is not a live option")
        if kind == "INT" and (isinstance(value, bool) or not isinstance(value, int)):
            raise RuntimeError(f"schema reconciliation failed: {name}.{widget_name} value type differs")
        if kind == "FLOAT" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise RuntimeError(f"schema reconciliation failed: {name}.{widget_name} value type differs")
        if kind == "STRING" and not isinstance(value, str):
            raise RuntimeError(f"schema reconciliation failed: {name}.{widget_name} value type differs")
        if kind == "BOOLEAN" and not isinstance(value, bool):
            raise RuntimeError(f"schema reconciliation failed: {name}.{widget_name} value type differs")


def build_api() -> dict:
    out = {}
    for shot, (base, concept, episode, plan, result, export) in BRANCHES.items():
        load = base + 8
        out.update({
            str(base): {"class_type": "AtlasOpenRealPlate", "inputs": {"episode": episode}},
            str(base + 1): {"class_type": "AtlasReadLockedPlatePlan", "inputs": {"plan": plan, "plate": [str(base), 0]}},
            str(base + 2): {"class_type": "AtlasSAM3Mask", "inputs": {"image": [str(load), 0], "concepts": concept, "confidence_threshold": 0.5, "device": "auto", "output_mode": "merged", "max_instances": 0}},
            str(base + 3): {"class_type": "AtlasInpaintCrop", "inputs": {"image": [str(load), 0], "mask": [str(base + 2), 0], "context_pad_px": 128}},
            str(base + 4): {"class_type": "AtlasSDXLInpaint", "inputs": {"image": [str(base + 3), 0], "mask": [str(base + 3), 1], "checkpoint": "SDXL/sd_xl_base_1.0.safetensors", "positive_prompt": "coherent architecture", "negative_prompt": "blurry, warped", "seed": 0, "steps": 30, "cfg": 5.5, "denoise": 0.85, "grow_mask_by": 8, "max_side": 1024, "preserve_perspective": True}},
            str(base + 5): {"class_type": "AtlasInpaintStitch", "inputs": {"original_image": [str(load), 0], "inpainted_crop": [str(base + 4), 0], "crop_region": [str(base + 3), 2]}},
            str(base + 6): {"class_type": "AtlasRecordPlateAttempt", "inputs": {"plate": [str(base), 0], "result": result, "locked_plan": [str(base + 1), 0]}},
            str(base + 7): {"class_type": "AtlasExportPlateHandoff", "inputs": {"plate": [str(base + 6), 0], "export": export}},
            str(load): {"class_type": "LoadImage", "inputs": {"image": f"prototypes/{shot}.png"}},
        })
    return out


def _node(node_id, class_type, pos, widgets, inputs, outputs, shot):
    return {"id": node_id, "type": class_type, "pos": pos, "size": [260, 180], "flags": {}, "order": node_id, "mode": 0, "inputs": inputs, "outputs": outputs, "properties": {"prototype_shot": shot}, "widgets_values": widgets}


def build_gui() -> dict:
    nodes, links = [], []
    for branch_index, (shot, (base, concept, episode, plan, result, export)) in enumerate(BRANCHES.items()):
        load = base + 8
        lr, lp, lm, lo, li, lc, cs, sd, sm, st, re, ex = base, base + 1, base + 2, base + 20, base + 21, base + 3, base + 4, base + 5, base + 22, base + 6, base + 7, base + 8
        nodes.extend([
            _node(base, "AtlasOpenRealPlate", [0, branch_index * 400], [episode], [], [{"name": "plate", "type": "ATLAS_REAL_PLATE", "links": [lr, lp], "slot_index": 0}, {"name": "episode_json", "type": "STRING", "links": [], "slot_index": 1}, {"name": "report", "type": "STRING", "links": [], "slot_index": 2}], shot),
            _node(base + 1, "AtlasReadLockedPlatePlan", [300, branch_index * 400], [plan], [{"name": "plate", "type": "ATLAS_REAL_PLATE", "link": lr}], [{"name": "locked_plan_json", "type": "STRING", "links": [re], "slot_index": 0}, {"name": "report", "type": "STRING", "links": [], "slot_index": 1}], shot),
            _node(base + 2, "AtlasSAM3Mask", [600, branch_index * 400], [concept, 0.5, "auto", "merged", 0], [{"name": "image", "type": "IMAGE", "link": lm}], [{"name": "mask", "type": "MASK", "links": [lc, cs], "slot_index": 0}, {"name": "report", "type": "STRING", "links": [], "slot_index": 1}], shot),
            _node(base + 3, "AtlasInpaintCrop", [900, branch_index * 400], [128], [{"name": "image", "type": "IMAGE", "link": lo}, {"name": "mask", "type": "MASK", "link": lc}], [{"name": "cropped_image", "type": "IMAGE", "links": [sd], "slot_index": 0}, {"name": "cropped_mask", "type": "MASK", "links": [sm], "slot_index": 1}, {"name": "crop_region", "type": "ATLAS_CROP_REGION", "links": [st + 1], "slot_index": 2}], shot),
            _node(base + 4, "AtlasSDXLInpaint", [1200, branch_index * 400], ["SDXL/sd_xl_base_1.0.safetensors", "coherent architecture", "blurry, warped", 0, 0, 30, 5.5, 0.85, 8, 1024, True], [{"name": "image", "type": "IMAGE", "link": sd}, {"name": "mask", "type": "MASK", "link": sm}], [{"name": "image", "type": "IMAGE", "links": [st], "slot_index": 0}, {"name": "report", "type": "STRING", "links": [], "slot_index": 1}], shot),
            _node(base + 5, "AtlasInpaintStitch", [1500, branch_index * 400], [0], [{"name": "original_image", "type": "IMAGE", "link": li}, {"name": "inpainted_crop", "type": "IMAGE", "link": st}, {"name": "crop_region", "type": "ATLAS_CROP_REGION", "link": st + 1}, {"name": "mask", "type": "MASK", "link": cs}], [{"name": "image", "type": "IMAGE", "links": [], "slot_index": 0}], shot),
            _node(base + 6, "AtlasRecordPlateAttempt", [1800, branch_index * 400], [result], [{"name": "plate", "type": "ATLAS_REAL_PLATE", "link": lp}, {"name": "locked_plan", "type": "STRING", "link": re}], [{"name": "plate", "type": "ATLAS_REAL_PLATE", "links": [ex], "slot_index": 0}, {"name": "result_json", "type": "STRING", "links": [], "slot_index": 1}, {"name": "report", "type": "STRING", "links": [], "slot_index": 2}], shot),
            _node(base + 7, "AtlasExportPlateHandoff", [2100, branch_index * 400], [export], [{"name": "plate", "type": "ATLAS_REAL_PLATE", "link": ex}], [{"name": "plate", "type": "ATLAS_REAL_PLATE", "links": [], "slot_index": 0}, {"name": "export_json", "type": "STRING", "links": [], "slot_index": 1}, {"name": "report", "type": "STRING", "links": [], "slot_index": 2}], shot),
            _node(load, "LoadImage", [600, branch_index * 400 + 260], [f"prototypes/{shot}.png"], [], [{"name": "IMAGE", "type": "IMAGE", "links": [lm, lo, li], "slot_index": 0}, {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}], shot),
        ])
        links.extend([[lr, base, 0, base + 1, 0, "ATLAS_REAL_PLATE"], [lp, base, 0, base + 6, 0, "ATLAS_REAL_PLATE"], [re, base + 1, 0, base + 6, 1, "STRING"], [lm, load, 0, base + 2, 0, "IMAGE"], [lo, load, 0, base + 3, 0, "IMAGE"], [li, load, 0, base + 5, 0, "IMAGE"], [lc, base + 2, 0, base + 3, 1, "MASK"], [cs, base + 2, 0, base + 5, 3, "MASK"], [sd, base + 3, 0, base + 4, 0, "IMAGE"], [sm, base + 3, 1, base + 4, 1, "MASK"], [st, base + 4, 0, base + 5, 1, "IMAGE"], [st + 1, base + 3, 2, base + 5, 2, "ATLAS_CROP_REGION"], [ex, base + 6, 0, base + 7, 0, "ATLAS_REAL_PLATE"]])
    # Normalize link IDs from one authoritative sequence and rebuild every
    # node-side reference. This keeps the hand-authored topology readable
    # while guaranteeing the LiteGraph invariant that IDs are unique.
    node_map = {node["id"]: node for node in nodes}
    normalized = []
    for new_id, (_, source, source_slot, target, target_slot, kind) in enumerate(links, 1):
        normalized.append([new_id, source, source_slot, target, target_slot, kind])
        node_map[target]["inputs"][target_slot]["link"] = new_id
    for node in nodes:
        for output in node.get("outputs", []):
            output["links"] = [link[0] for link in normalized if link[1] == node["id"] and link[2] == output["slot_index"]]
    return {"last_node_id": 19, "last_link_id": len(normalized), "nodes": nodes, "links": normalized, "groups": [], "config": {}, "extra": {"atlas_shipping_set": "v0.3-world-plate-prototype", "non_shipping": True}, "version": 0.4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-info", type=Path, help="saved /object_info JSON snapshot")
    parser.add_argument("--comfy-url", help="running ComfyUI base URL")
    args = parser.parse_args()
    snapshot = json.loads(args.object_info.read_text(encoding="utf-8")) if args.object_info else None
    reconcile_object_info(snapshot, server=args.comfy_url)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("atlas_world_plate_prototype_workflow.json", build_gui()), ("atlas_world_plate_prototype_api.json", build_api())):
        (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

