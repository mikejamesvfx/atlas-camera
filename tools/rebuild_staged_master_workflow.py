"""Rebuild the shipping five-layer master as native ComfyUI subgraphs.

The generator reads the running ComfyUI ``/object_info`` endpoint so every
node is serialized with the live input order.  That deliberately avoids the
positional-widget drift that made the old LaMa/KJ/rgthree workflow brittle.

Usage::

    python tools/rebuild_staged_master_workflow.py
    python tools/rebuild_staged_master_workflow.py --host 127.0.0.1:8188
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples" / "atlas_camera_staged_master_workflow.json"
AGENTIC_OUTPUT = (
    ROOT / "examples" / "atlas_camera_staged_master_agentic_assessment_workflow.json")
WORKFLOW_ID = "3a115c93-dd03-417c-b6c0-b47bf6f3e710"
AGENTIC_WORKFLOW_ID = "bd63d232-688b-5378-ae08-8962cfadc52a"
UUID_NAMESPACE = uuid.UUID("6ac2634a-648f-4db6-93ef-a9cd95741a52")
PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _fetch_object_info(host: str) -> dict:
    with urllib.request.urlopen(f"http://{host}/object_info", timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    # The Atlas portable MCP proxy may JSON-encode the Comfy response once
    # more; accepting both forms keeps the generator useful in either setup.
    return json.loads(data) if isinstance(data, str) else data


def _is_widget(spec) -> bool:
    type_name = spec[0]
    config = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(type_name, list) or type_name == "COMBO":
        return True
    return type_name in PRIMITIVES and not config.get("forceInput")


def _slot_type(spec) -> str:
    return "COMBO" if isinstance(spec[0], list) else str(spec[0])


def _default_value(spec):
    type_name = spec[0]
    config = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in config:
        return config["default"]
    if isinstance(type_name, list):
        return type_name[0] if type_name else ""
    return {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(type_name)


class Graph:
    """Small LiteGraph builder shared by the top graph and definitions."""

    def __init__(self, object_info: dict, *, object_links: bool = False):
        self.oi = object_info
        self.object_links = object_links
        self.nodes: list[dict] = []
        self.links: list = []
        self._node_id = 0
        self._link_id = 0

    def node(self, type_name: str, *, title: str = "", values: dict | None = None,
             size: tuple[int, int] | None = None, mode: int = 0) -> dict:
        if type_name not in self.oi:
            raise KeyError(f"{type_name} is not registered on the running ComfyUI")
        info = self.oi[type_name]
        values = values or {}
        self._node_id += 1
        inputs = []
        widgets = []
        order = info.get("input_order") or {}
        for section in ("required", "optional"):
            specs = info.get("input", {}).get(section) or {}
            names = order.get(section) or list(specs)
            for name in names:
                spec = specs[name]
                inp = {"localized_name": name, "name": name,
                       "type": _slot_type(spec), "link": None}
                if section == "optional" or _is_widget(spec):
                    inp["shape"] = 7
                if _is_widget(spec):
                    inp["widget"] = {"name": name}
                    widgets.append(values.get(name, _default_value(spec)))
                    config = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                    if name in ("seed", "noise_seed"):
                        widgets.append(values.get("control_after_generate", "fixed"))
                    elif config.get("image_upload"):
                        widgets.append(values.get("image_upload", "image"))
                inputs.append(inp)

        output_types = info.get("output") or []
        output_names = info.get("output_name") or output_types
        outputs = [
            {"localized_name": str(name), "name": str(name),
             "type": str(type_), "links": []}
            for name, type_ in zip(output_names, output_types)
        ]
        node = {
            "id": self._node_id,
            "type": type_name,
            "pos": [0, 0],
            "size": list(size or (320, max(100, 74 + 28 * len(widgets)))),
            "flags": {},
            "order": self._node_id - 1,
            "mode": mode,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": type_name},
            "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    @staticmethod
    def _input_slot(node: dict, name: str) -> int:
        return next(i for i, item in enumerate(node.get("inputs") or [])
                    if item["name"] == name)

    @staticmethod
    def _output_slot(node: dict, name: str) -> int:
        return next(i for i, item in enumerate(node.get("outputs") or [])
                    if item["name"] == name)

    def connect(self, source: dict, source_name: str, target: dict,
                target_name: str, type_name: str | None = None) -> int:
        source_slot = self._output_slot(source, source_name)
        target_slot = self._input_slot(target, target_name)
        wire_type = type_name or source["outputs"][source_slot]["type"]
        self._link_id += 1
        link_id = self._link_id
        if self.object_links:
            link = {"id": link_id, "origin_id": source["id"],
                    "origin_slot": source_slot, "target_id": target["id"],
                    "target_slot": target_slot, "type": wire_type}
        else:
            link = [link_id, source["id"], source_slot, target["id"],
                    target_slot, wire_type]
        self.links.append(link)
        source["outputs"][source_slot].setdefault("links", []).append(link_id)
        target["inputs"][target_slot]["link"] = link_id
        return link_id


class Subgraph(Graph):
    def __init__(self, object_info: dict, name: str, description: str):
        super().__init__(object_info, object_links=True)
        self.name = name
        self.description = description
        self.id = str(uuid.uuid5(UUID_NAMESPACE, name))
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.proxy_widgets: list[list[str]] = []
        self.proxy_values: list = []

    def add_input(self, name: str, type_name: str, label: str | None = None) -> int:
        index = len(self.inputs)
        self.inputs.append({
            "id": str(uuid.uuid5(UUID_NAMESPACE, f"{self.name}:input:{name}")),
            "name": name,
            "type": type_name,
            "linkIds": [],
            "localized_name": name,
            "label": label or name,
            "pos": [0, 0],
        })
        return index

    def add_output(self, name: str, type_name: str, label: str | None = None) -> int:
        index = len(self.outputs)
        self.outputs.append({
            "id": str(uuid.uuid5(UUID_NAMESPACE, f"{self.name}:output:{name}")),
            "name": name,
            "type": type_name,
            "linkIds": [],
            "localized_name": name,
            "label": label or name,
            "pos": [0, 0],
        })
        return index

    def input_to(self, input_name: str, target: dict, target_name: str) -> int:
        source_slot = next(i for i, item in enumerate(self.inputs)
                           if item["name"] == input_name)
        target_slot = self._input_slot(target, target_name)
        self._link_id += 1
        link_id = self._link_id
        self.links.append({"id": link_id, "origin_id": -10,
                           "origin_slot": source_slot, "target_id": target["id"],
                           "target_slot": target_slot,
                           "type": self.inputs[source_slot]["type"]})
        self.inputs[source_slot]["linkIds"].append(link_id)
        target["inputs"][target_slot]["link"] = link_id
        return link_id

    def output_from(self, output_name: str, source: dict, source_name: str) -> int:
        target_slot = next(i for i, item in enumerate(self.outputs)
                           if item["name"] == output_name)
        source_slot = self._output_slot(source, source_name)
        self._link_id += 1
        link_id = self._link_id
        self.links.append({"id": link_id, "origin_id": source["id"],
                           "origin_slot": source_slot, "target_id": -20,
                           "target_slot": target_slot,
                           "type": self.outputs[target_slot]["type"]})
        self.outputs[target_slot]["linkIds"].append(link_id)
        source["outputs"][source_slot].setdefault("links", []).append(link_id)
        return link_id

    def expose(self, node: dict, *widget_names: str) -> None:
        info = self.oi[node["type"]]
        widget_positions: dict[str, int] = {}
        index = 0
        order = info.get("input_order") or {}
        for section in ("required", "optional"):
            specs = info.get("input", {}).get(section) or {}
            for name in order.get(section) or list(specs):
                spec = specs[name]
                if not _is_widget(spec):
                    continue
                widget_positions[name] = index
                index += 1
                if name in ("seed", "noise_seed"):
                    widget_positions["control_after_generate"] = index
                    index += 1
                elif len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("image_upload"):
                    widget_positions["image_upload"] = index
                    index += 1
        for name in widget_names:
            self.proxy_widgets.append([str(node["id"]), name])
            self.proxy_values.append(node["widgets_values"][widget_positions[name]])

    def finish(self, layout) -> dict:
        layout.auto_layout({"nodes": self.nodes, "links": self.links}, origin=(260, 80))
        left = min(node["pos"][0] for node in self.nodes)
        right = max(node["pos"][0] + node["size"][0] for node in self.nodes)
        top = min(node["pos"][1] for node in self.nodes)
        input_height = max(60, 48 + 20 * len(self.inputs))
        output_height = max(60, 48 + 20 * len(self.outputs))
        input_box = [left - 210, top, 160, input_height]
        output_box = [right + 70, top, 160, output_height]
        for i, item in enumerate(self.inputs):
            item["pos"] = [input_box[0] + input_box[2] - 24, input_box[1] + 24 + 20 * i]
        for i, item in enumerate(self.outputs):
            item["pos"] = [output_box[0] + 24, output_box[1] + 24 + 20 * i]
        check = layout.inspect({"nodes": self.nodes, "links": self.links})
        if check["overlaps"]:
            raise RuntimeError(f"{self.name} layout overlaps: {check['overlaps']}")
        return {
            "id": self.id,
            "version": 1,
            "state": {"lastGroupId": 0, "lastNodeId": self._node_id,
                      "lastLinkId": self._link_id, "lastRerouteId": 0},
            "revision": 0,
            "config": {},
            "name": self.name,
            "inputNode": {"id": -10, "bounding": input_box},
            "outputNode": {"id": -20, "bounding": output_box},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "widgets": [],
            "nodes": self.nodes,
            "groups": [],
            "links": self.links,
            "extra": {"workflowRendererVersion": "LG"},
            "category": "Atlas Camera/Layers",
            "description": self.description,
        }


def _load_layout_module():
    candidates = [
        Path.home() / ".agents" / "skills" / "comfyui" / "workflow_layout.py",
        Path.home() / ".codex" / "skills" / "comfyui" / "workflow_layout.py",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("ComfyUI skill workflow_layout.py was not found")
    spec = importlib.util.spec_from_file_location("comfy_workflow_layout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_sky(object_info: dict) -> Subgraph:
    graph = Subgraph(
        object_info,
        "1 · SKY LAYER",
        "Segments the sky and adds the farthest projection dome.",
    )
    for name, type_name in (("solve", "ATLAS_SOLVE"), ("depth", "ATLAS_DEPTH_MAP"),
                            ("plate", "IMAGE"), ("sam_prompt", "STRING")):
        graph.add_input(name, type_name)
    for name, type_name in (("solve", "ATLAS_SOLVE"), ("sky_mask", "MASK"),
                            ("hole_mask", "MASK"), ("sam_report", "STRING")):
        graph.add_output(name, type_name)

    sam = graph.node("AtlasSAM3Mask", title="Sky segmentation", values={
        "concepts": "sky", "confidence_threshold": 0.5,
        "device": "auto", "output_mode": "merged", "max_instances": 0,
    })
    sky = graph.node("AtlasSkyDomeLayer", title="Sky dome · priority -10", values={
        "radius_m": 300.0, "relief_grid": 96, "name": "sky", "priority": -10.0,
        "edge_extend_px": 48, "frame_outpaint_px": 64, "distance_m": 0.0,
    })
    graph.input_to("plate", sam, "image")
    graph.input_to("sam_prompt", sam, "concepts")
    graph.input_to("solve", sky, "solve")
    graph.input_to("depth", sky, "depth")
    graph.input_to("plate", sky, "plate_image")
    graph.connect(sam, "mask", sky, "sky_mask")
    graph.output_from("solve", sky, "solve")
    graph.output_from("sky_mask", sam, "mask")
    graph.output_from("hole_mask", sky, "hole_mask")
    graph.output_from("sam_report", sam, "report")
    graph.expose(sam, "confidence_threshold")
    return graph


LAYER_SPECS = (
    ("2 · FAR LAYER", "band_far", 0.80, 1.00, 15.0, 128, 101,
     "Photorealistic continuation of the distant scene behind occluders, matching the source plate's architecture, atmosphere, lighting, material detail, scale, and exact camera perspective."),
    ("3 · BACKGROUND LAYER", "band_bg", 0.60, 0.80, 10.0, 160, 202,
     "Photorealistic continuation of background buildings and landscape behind occluders, with coherent windows, facades, depth, lighting, texture, and the exact source camera perspective."),
    ("4 · MIDGROUND LAYER", "band_mid", 0.30, 0.60, 5.0, 192, 303,
     "Photorealistic continuation of midground structures and surfaces behind occluders, preserving object scale, material texture, illumination, line convergence, and the exact source camera perspective."),
    ("5 · FOREGROUND LAYER", "band_fg", 0.00, 0.30, 0.0, 224, 404,
     "Photorealistic continuation of foreground street, rooftop, ground, and nearby surfaces behind occluders, preserving fine texture, contact detail, lighting, and the exact source camera perspective."),
)

NEGATIVE_PROMPT = (
    "blurry, soft detail, warped geometry, duplicate structures, repeated windows, "
    "floating objects, seams, halos, text, logo, watermark, changed viewpoint, "
    "straight-on facade, front elevation, orthographic view, flat perspective"
)


def _build_depth_layer(object_info: dict, spec: tuple) -> Subgraph:
    title, layer_name, near_pct, far_pct, priority, context_pad, seed, prompt = spec
    graph = Subgraph(
        object_info,
        title,
        "SAM-scoped depth band with cropped native SDXL inpaint, feathered stitch, and clean-plate projection geometry.",
    )
    for name, type_name in (
        ("solve", "ATLAS_SOLVE"), ("depth", "ATLAS_DEPTH_MAP"),
        ("plate", "IMAGE"), ("sky_mask", "MASK"),
        ("sam_prompt", "STRING"), ("geometry_override", "STRING"),
        ("band_override", "STRING"),
    ):
        graph.add_input(name, type_name)
    for name, type_name in (
        ("solve", "ATLAS_SOLVE"), ("clean_plate", "IMAGE"),
        ("layer_mask", "MASK"), ("hole_mask", "MASK"),
        ("scope_status", "STRING"), ("inpaint_report", "STRING"),
    ):
        graph.add_output(name, type_name)

    sam = graph.node("AtlasSAM3Mask", title=f"SAM scope · {layer_name}", values={
        "concepts": layer_name, "confidence_threshold": 0.5,
        "device": "auto", "output_mode": "merged", "max_instances": 0,
    })
    scope = graph.node("AtlasScopeMask", title="Self-disarming layer scope", values={
        "prompt": layer_name, "grow_px": 16, "min_coverage_pct": 0.2,
    })
    mask = graph.node("AtlasDepthLayerMask", title=f"Depth band · {layer_name}", values={
        "near_m": 0.0, "far_m": 0.0, "near_pct": near_pct, "far_pct": far_pct,
        "feather_px": 4, "compute_hole_mask": False, "relief_grid": 384,
        "depth_edge_rel": 1.5, "fill_occluded": False, "band_side": "manual",
        "band_override": "", "quad_coherence": True,
    })
    crop = graph.node("AtlasInpaintCrop", title="Crop SDXL context", values={
        "context_pad_px": context_pad,
    })
    sdxl = graph.node("AtlasSDXLInpaint", title="Native SDXL clean plate", values={
        "checkpoint": "SDXL\\sd_xl_base_1.0.safetensors",
        "positive_prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": seed,
        "control_after_generate": "fixed",
        "steps": 32,
        "cfg": 5.5,
        "denoise": 0.85,
        "grow_mask_by": 8,
        "max_side": 1024,
        "preserve_perspective": True,
    })
    stitch = graph.node("AtlasInpaintStitch", title="Feathered full-frame stitch", values={
        "feather_px": 12,
    })
    clean = graph.node("AtlasCleanPlateLayer", title=f"Projection · {layer_name}", values={
        "near_m": 0.0, "far_m": 0.0, "near_pct": near_pct, "far_pct": far_pct,
        "name": layer_name, "priority": priority, "relief_grid": 384,
        "depth_edge_rel": 1.5, "fill_occluded": True, "embed_matte": True,
        "edge_extend_px": 32, "skirt_bevel": 0.0, "frame_outpaint_px": 0,
        "exclude_choke_cells": 2, "band_side": "manual",
        "band_geometry": "relief", "geometry_override": "",
        "band_override": "", "max_edge_factor": 12.0,
        "normal_edge_deg": 0.0, "quad_coherence": True,
    })

    graph.input_to("plate", sam, "image")
    graph.input_to("sam_prompt", sam, "concepts")
    graph.input_to("sky_mask", scope, "sky_mask")
    graph.input_to("sam_prompt", scope, "prompt")
    graph.connect(sam, "mask", scope, "segment_mask")

    graph.input_to("solve", mask, "solve")
    graph.input_to("depth", mask, "depth")
    graph.input_to("sky_mask", mask, "band_ref_mask")
    graph.input_to("band_override", mask, "band_override")
    graph.connect(scope, "exclude_mask", mask, "exclude_mask")

    graph.input_to("plate", crop, "image")
    graph.connect(mask, "occlusion_mask", crop, "mask")
    graph.connect(crop, "cropped_image", sdxl, "image")
    graph.connect(crop, "cropped_mask", sdxl, "mask")

    graph.input_to("plate", stitch, "original_image")
    graph.connect(sdxl, "image", stitch, "inpainted_crop")
    graph.connect(crop, "crop_region", stitch, "crop_region")
    # AtlasInpaintStitch expects a full-frame mask. Its feather is the final
    # seam guard for a generative inpainter that may re-render crop context.
    graph.connect(mask, "occlusion_mask", stitch, "mask")

    graph.input_to("solve", clean, "solve")
    graph.input_to("depth", clean, "depth")
    graph.input_to("sky_mask", clean, "band_ref_mask")
    graph.input_to("geometry_override", clean, "geometry_override")
    graph.input_to("band_override", clean, "band_override")
    graph.connect(stitch, "image", clean, "plate_image")
    graph.connect(scope, "exclude_mask", clean, "exclude_mask")

    graph.output_from("solve", clean, "solve")
    graph.output_from("clean_plate", stitch, "image")
    graph.output_from("layer_mask", mask, "layer_mask")
    graph.output_from("hole_mask", clean, "hole_mask")
    graph.output_from("scope_status", scope, "status")
    graph.output_from("inpaint_report", sdxl, "report")
    graph.expose(sdxl, "positive_prompt", "negative_prompt", "seed",
                 "control_after_generate", "steps", "cfg", "denoise")
    return graph


def _subgraph_instance(top: Graph, definition: Subgraph, title: str) -> dict:
    top._node_id += 1
    node = {
        "id": top._node_id,
        "type": definition.id,
        "pos": [0, 0],
        "size": [390, max(180, 100 + 28 * len(definition.proxy_values))],
        "flags": {},
        "order": top._node_id - 1,
        "mode": 0,
        "inputs": [
            {"label": item["label"], "localized_name": item["localized_name"],
             "name": item["name"], "type": item["type"], "link": None}
            for item in definition.inputs
        ],
        "outputs": [
            {"label": item["label"], "localized_name": item["localized_name"],
             "name": item["name"], "type": item["type"], "links": []}
            for item in definition.outputs
        ],
        "properties": {"proxyWidgets": definition.proxy_widgets},
        "widgets_values": definition.proxy_values,
        "title": title,
    }
    top.nodes.append(node)
    return node


def _group(layout, nodes: list[dict], title: str, color: str) -> dict:
    x0 = min(node["pos"][0] for node in nodes)
    y0 = min(node["pos"][1] for node in nodes)
    x1 = max(node["pos"][0] + layout.est_size(node)[0] for node in nodes)
    y1 = max(node["pos"][1] + layout.est_size(node)[1] for node in nodes)
    return {"id": str(uuid.uuid5(UUID_NAMESPACE, f"group:{title}")),
            "title": title, "bounding": [x0 - 45, y0 - 88, x1 - x0 + 90, y1 - y0 + 133],
            "color": color, "font_size": 24, "flags": {}}


def build(object_info: dict, layout, *, agentic_assessment: bool = False) -> dict:
    sky_def = _build_sky(object_info)
    layer_defs = [_build_depth_layer(object_info, spec) for spec in LAYER_SPECS]
    definitions = [sky_def, *layer_defs]

    top = Graph(object_info)
    load = top.node(
        "LoadImage",
        title=("SOURCE PLATE · GHOST TOWN QA SAMPLE"
               if agentic_assessment else "SOURCE PLATE"), values={
        "image": "ghosttown.jpg" if agentic_assessment else "example.png",
        "image_upload": "image",
    })
    assess = top.node("AtlasAssessImage", title="0 · VLM 5-layer assessment", values={
        "provider": "lmstudio", "model": "google/gemma-4-12b-qat",
        "base_url": "", "extra_instructions": "", "proceed": False,
        "approved_for": "", "api_key": "", "offload_model": True,
        "auto_continue": True,
    })
    register = top.node("AtlasRegisterPlate", title="Register float-safe plate", values={
        "plate_path": "", "colorspace": "ACEScg", "bit_depth": "auto",
        "role": "source", "lut_path": "",
    })
    solve = top.node("AtlasLearnedSolveFromImage", title="Learned camera solve · GeoCalib", values={
        "height_mode": "measure_from_depth", "camera_height_m": 1.6,
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "sensor_width_mm": 36.0, "weights": "pinhole", "device": "auto",
        "focal_length_mm": 0.0,
    })
    attach = top.node("AtlasAttachSourcePlate", title="Attach source plate")
    gate = top.node("AtlasSolveGate", title="Approve solve, then re-queue", values={
        "proceed": False, "approved_for": "",
    })
    depth = top.node("AtlasDepthMap", title="Shared metric depth", values={
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "device": "auto",
    })
    preview = top.node("AtlasBlockoutViewport", title="SOLVE PREVIEW · approve before layers", values={
        "resolution": 768, "client_data": "", "preview_expand": 1.0,
    }, size=(720, 560))

    sky = _subgraph_instance(top, sky_def, "1 · SKY — segmentation + dome")
    layers = [
        _subgraph_instance(top, definition, spec[0])
        for definition, spec in zip(layer_defs, LAYER_SPECS)
    ]

    controls = top.node("AtlasViewportControls", title="OUTPUT DESK · OCIO metadata", values={
        "config_label": "ACES 2.0 / Studio", "config_path": "",
        "working_colorspace": "ACEScg", "output_colorspace": "ACES - ACEScg",
        "display": "sRGB - Display", "view": "ACES 2.0 SDR-video",
        "display_trim": 1.0,
    })
    master = top.node("AtlasBlockoutViewport", title="MASTER VIEWPORT · five-layer scene", values={
        "resolution": 1464, "client_data": "", "preview_expand": 1.0,
    }, size=(900, 700))

    export_json = top.node("AtlasExportSolveJSON", title="Solve JSON", values={
        "output_path": "output/atlas_staged_master_solve.json",
    })
    export_nuke = top.node("AtlasExportNukeLayers", title="Nuke layers", values={
        "output_dir": "output/atlas_staged_master/nuke", "retopo_method": "off",
        "retopo_target_vertex_count": 2000, "retopo_smooth_iterations": 0,
        "retopo_crease_angle": 30.0, "retopo_pure_quad": False,
    })
    export_maya = top.node("AtlasExportMayaLayers", title="Maya layers", values={
        "output_dir": "output/atlas_staged_master/maya", "retopo_method": "off",
        "retopo_target_vertex_count": 2000, "retopo_smooth_iterations": 0,
        "retopo_crease_angle": 30.0, "retopo_pure_quad": False,
    })
    export_blender = top.node("AtlasExportBlender", title="Blender handoff", values={
        "output_dir": "output/atlas_staged_master/blender",
    })
    export_usd = top.node("AtlasExportUSD", title="USD camera + scene", values={
        "output_dir": "output/atlas_staged_master/usd",
    })
    debug = top.node("AtlasDebugReport", title="MASTER DEBUG · stable JSON", values={
        "file_path": "atlas_debug/master_debug.json",
    })
    assess_output = None
    evidence_preview = None
    if agentic_assessment:
        assess_output = top.node(
            "AtlasAssessOutput", title="TERMINAL QA · agent/headless report", values={
                "enabled": True,
                "provider": "lmstudio",
                "model": "google/gemma-4-12b-qat",
                "base_url": "",
                "extra_instructions": (
                    "Assess the final five-layer camera view and release readiness."),
                "file_path": (
                    "atlas_debug/staged_master_agentic_output_assessment.json"),
                "api_key": "",
                "offload_model": True,
                "fallback_to_source": True,
            })
        evidence_preview = top.node(
            "PreviewImage", title="ASSESSED EVIDENCE · exact VLM image")
    preview_pairs = []
    for index, label in enumerate(("sky", "band_far", "band_bg", "band_mid", "band_fg")):
        layer_preview = top.node("AtlasLayerPreview", title=f"Layer cutout · {label}", values={
            "layer_index": index, "color_hex": "",
        })
        image_preview = top.node("PreviewImage", title=f"PREVIEW · {label}")
        top.connect(layer_preview, "image", image_preview,
                    object_info["PreviewImage"]["input_order"]["required"][0])
        preview_pairs.append((layer_preview, image_preview))

    top.connect(load, "IMAGE", assess, "image")
    top.connect(assess, "image", register, "image")
    top.connect(register, "image", solve, "image")
    top.connect(solve, "ATLAS_SOLVE", attach, "solve")
    top.connect(register, "plate_ref", attach, "plate_ref")
    top.connect(attach, "ATLAS_SOLVE", gate, "solve")
    top.connect(register, "image", gate, "source_image")
    top.connect(attach, "ATLAS_SOLVE", preview, "solve")
    top.connect(register, "image", preview, "source_image")
    top.connect(gate, "solve", depth, "solve")
    top.connect(register, "image", depth, "image")

    top.connect(gate, "solve", sky, "solve")
    top.connect(depth, "depth", sky, "depth")
    top.connect(register, "image", sky, "plate")
    top.connect(assess, "sam_prompt_sky", sky, "sam_prompt")

    previous = sky
    prompt_names = ("sam_prompt_far", "sam_prompt_bg", "sam_prompt_mid", "sam_prompt_fg")
    geometry_names = ("geom_far", "geom_bg", "geom_mid", "geom_fg")
    band_names = ("band_far", "band_bg", "band_mid", "band_fg")
    for layer, prompt_name, geometry_name, band_name in zip(
            layers, prompt_names, geometry_names, band_names):
        top.connect(previous, "solve", layer, "solve")
        top.connect(depth, "depth", layer, "depth")
        top.connect(register, "image", layer, "plate")
        top.connect(sky, "sky_mask", layer, "sky_mask")
        top.connect(assess, prompt_name, layer, "sam_prompt")
        top.connect(assess, geometry_name, layer, "geometry_override")
        top.connect(assess, band_name, layer, "band_override")
        previous = layer

    final_solve = layers[-1]
    top.connect(final_solve, "solve", master, "solve")
    top.connect(register, "image", master, "source_image")
    top.connect(depth, "depth", master, "primary_depth")
    top.connect(controls, "controls", master, "controls")
    top.connect(controls, "output_profile", master, "output_profile")

    for exporter in (export_json, export_nuke, export_maya, export_blender,
                     export_usd):
        top.connect(final_solve, "solve", exporter, "solve")
    for exporter in (export_nuke, export_maya, export_blender):
        top.connect(controls, "output_profile", exporter, "output_profile")
    top.connect(final_solve, "solve", debug, "solve")
    top.connect(depth, "depth", debug, "depth")
    top.connect(assess, "report", debug, "vlm_report")
    for index, layer in enumerate(layers, start=1):
        top.connect(layer, "scope_status", debug, f"status_{index}")
    if assess_output is not None:
        top.connect(master, "shaded", assess_output, "camera_view")
        top.connect(final_solve, "solve", assess_output, "solve")
        top.connect(register, "image", assess_output, "source_image")
        top.connect(depth, "depth", assess_output, "depth")
        top.connect(debug, "report", assess_output, "solve_summary")
        top.connect(assess_output, "assessed_image", evidence_preview, "images")

    top.connect(register, "image", preview_pairs[0][0], "image")
    top.connect(sky, "sky_mask", preview_pairs[0][0], "mask")
    for pair, layer in zip(preview_pairs[1:], layers):
        top.connect(layer, "clean_plate", pair[0], "image")
        top.connect(layer, "layer_mask", pair[0], "mask")

    top_dict = {"nodes": top.nodes, "links": top.links}
    layout.auto_layout(top_dict, origin=(80, 140))
    check = layout.inspect(top_dict)
    if check["overlaps"]:
        raise RuntimeError(f"top-level layout overlaps: {check['overlaps']}")

    input_nodes = [load, assess, register, solve, attach, gate, depth, preview]
    layer_nodes = [sky, *layers]
    output_nodes = [controls, master, export_json, export_nuke, export_maya,
                    export_blender, export_usd, debug,
                    *(node for pair in preview_pairs for node in pair)]
    if assess_output is not None:
        output_nodes.extend((assess_output, evidence_preview))
    groups = [
        _group(layout, input_nodes, "0 · ASSESS + CAMERA SOLVE", "#35536b"),
        _group(layout, layer_nodes, "1–5 · LAYER SUBGRAPHS · open a layer to tune masks and geometry", "#4d436b"),
        _group(layout, output_nodes, "ASSEMBLE · DEBUG · DCC EXPORT", "#375c4a"),
    ]

    finished_defs = [definition.finish(layout) for definition in definitions]
    return {
        "id": AGENTIC_WORKFLOW_ID if agentic_assessment else WORKFLOW_ID,
        "revision": 1,
        "last_node_id": top._node_id,
        "last_link_id": top._link_id,
        "nodes": top.nodes,
        "links": top.links,
        "groups": groups,
        "config": {},
        "extra": {
            "ds": {"scale": 0.62, "offset": [40, 80]},
            "frontendVersion": "1.25.11",
            "workflowRendererVersion": "LG",
            "atlas_staged_master_version": 13,
            "atlas_agentic_assessment": agentic_assessment,
            "atlas_notes": (
                "Five native subgraphs; SDXL crop-inpaint-stitch per layer; "
                "terminal VLM + deterministic headless QA report."
                if agentic_assessment else
                "Five native subgraphs; SDXL crop-inpaint-stitch per layer; "
                "artist-facing workflow without automatic terminal VLM."),
        },
        "version": 0.4,
        "definitions": {"subgraphs": finished_defs},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agentic-output", type=Path, default=AGENTIC_OUTPUT)
    args = parser.parse_args()
    object_info = _fetch_object_info(args.host)
    layout = _load_layout_module()
    for output, agentic_assessment in ((args.output, False),
                                       (args.agentic_output, True)):
        workflow = build(object_info, layout,
                         agentic_assessment=agentic_assessment)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print(layout.inspect(workflow)["summary"])
        for definition in workflow["definitions"]["subgraphs"]:
            print(f"  {definition['name']}: {layout.inspect(definition)['summary']}")


if __name__ == "__main__":
    main()
