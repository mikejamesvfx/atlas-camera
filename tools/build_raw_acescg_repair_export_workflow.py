"""Build the production RAW -> ACEScg -> repaired 3-layer DCC workflow.

The source is the proven ``atlas_raw_3layer_ocio_workflow.json`` graph.  This
builder keeps its file-float OCIO handoff, inserts the native camera-space
surface repair passes created on 2026-08-02, and performs one live retopology
pass before the Maya, Nuke, and Blender exporters.

Usage::

    python tools/build_raw_acescg_repair_export_workflow.py SOURCE_WORKFLOW
"""
from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "examples" / "local" /
    "2026-08-02_DSC_2245_ULTIMATE_RAW_ACESCG_DCC.json"
)
RAW_PATH = Path(r"C:\Users\miike\Pictures\atlas_raws\atlas_raws\DSC_2245.NEF")
LAYOUT = Path.home() / ".agents" / "skills" / "comfyui" / "workflow_layout.py"
OUTPUT_ROOT = "output/DSC_2245_ULTIMATE_RAW_ACESCG"


def _load_layout():
    spec = importlib.util.spec_from_file_location("workflow_layout", LAYOUT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load workflow layout helper: {LAYOUT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(
    node_id: int,
    node_type: str,
    *,
    title: str,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
    widgets: list,
    size: tuple[int, int] = (390, 300),
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "pos": [0, 0],
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": [
            {"name": name, "type": data_type, "link": None}
            for name, data_type in inputs
        ],
        "outputs": [
            {
                "name": name,
                "type": data_type,
                "links": [],
                "slot_index": slot,
            }
            for slot, (name, data_type) in enumerate(outputs)
        ],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets,
        "title": title,
    }


def _note(node_id: int) -> dict:
    node = _node(
        node_id,
        "Note",
        title="ULTIMATE RAW · ACEScg + camera-away repair",
        inputs=[],
        outputs=[],
        widgets=[
            "FINAL PIPELINE\n\n"
            "1. AtlasLoadRAW demosaics the NEF at full resolution and writes a "
            "scene-linear EXR tagged Linear Rec.709.\n"
            "2. AtlasExportPlateEXR converts that FILE directly through OCIO to "
            "a tagged ACEScg half-float EXR. The Comfy image proxy is never used "
            "to write the master plate.\n"
            "3. Background and foreground reliefs each run Masked Surface "
            "Reconstruct, followed by Occlusion Seams in away_from_camera mode.\n"
            "4. Retopology runs AFTER repair: decimate to 50k vertices per relief, "
            "with all smoothing disabled so camera-away displacement stays exact.\n"
            "5. Maya, Nuke, and Blender exporters all receive the same final solve "
            "and the same ACEScg output profile. Their export-only retopo is OFF "
            "to avoid processing the meshes twice.\n\n"
            "ATLAS_EXPERIMENTAL=1 is required for the two repair nodes."
            "\n\nLOCKED CALIBRATION\n"
            "Foreground relief_grid is 384, not source-image resolution. "
            "Do not raise it to 3714: that creates roughly 18.4M candidate "
            "triangles and resolves depth noise rather than useful shape."
        ],
        size=(650, 500),
    )
    node["color"] = "#31515b"
    node["bgcolor"] = "#1c3037"
    return node


def _input_slot(node: dict, name: str) -> int:
    return next(i for i, item in enumerate(node["inputs"])
                if item["name"] == name)


def _output_slot(node: dict, name: str) -> int:
    return next(i for i, item in enumerate(node["outputs"])
                if item["name"] == name)


def _add_link(
    workflow: dict,
    link_id: int,
    source: dict,
    output_name: str,
    target: dict,
    input_name: str,
    data_type: str,
) -> None:
    workflow["links"].append([
        link_id,
        source["id"],
        _output_slot(source, output_name),
        target["id"],
        _input_slot(target, input_name),
        data_type,
    ])


def _rebuild_link_slots(workflow: dict) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in nodes.values():
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for link_id, source, source_slot, target, target_slot, _kind in workflow["links"]:
        nodes[source]["outputs"][source_slot]["links"].append(link_id)
        nodes[target]["inputs"][target_slot]["link"] = link_id


def _repair_nodes(start_id: int, layer: str) -> tuple[dict, dict]:
    reconstruct = _node(
        start_id,
        "AtlasMaskedSurfaceReconstruct",
        title=f"{layer.upper()} · manufacture rim + reconstruct",
        inputs=[("solve", "ATLAS_SOLVE"), ("hole_mask", "MASK")],
        outputs=[
            ("solve", "ATLAS_SOLVE"),
            ("remaining_holes", "MASK"),
            ("created_region", "MASK"),
            ("report", "STRING"),
        ],
        widgets=[layer, 1, 1024 if layer == "fg" else 64, 0.20, True, 128],
        size=(430, 350),
    )
    seams = _node(
        start_id + 1,
        "AtlasRefineOcclusionSeams",
        title=f"{layer.upper()} · dual-sheet camera-away underlap",
        inputs=[("solve", "ATLAS_SOLVE"), ("hole_mask", "MASK")],
        outputs=[
            ("solve", "ATLAS_SOLVE"),
            ("remaining_holes", "MASK"),
            ("created_region", "MASK"),
            ("report", "STRING"),
        ],
        widgets=[layer, 3.0, 8, 0.35, 256, 0.08, 2, "away_from_camera"],
        size=(450, 410),
    )
    return reconstruct, seams


def build(source: Path) -> dict:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"RAW photo not found: {RAW_PATH}")
    workflow = json.loads(source.read_text(encoding="utf-8"))
    workflow["id"] = str(uuid.UUID("b32e5bb4-a3b6-4e69-8ee9-0b808ce14d7b"))
    workflow["revision"] = 0

    # Keep only the requested DCC outputs plus the viewport. The ACEScg plate
    # writer remains upstream because it is the colour-managed source of truth.
    remove_ids = {21, 22, 26, 27}
    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] not in remove_ids
    ]
    workflow["links"] = [
        link for link in workflow["links"]
        if link[1] not in remove_ids and link[3] not in remove_ids
    ]
    nodes = {node["id"]: node for node in workflow["nodes"]}

    nodes[1]["title"] = "📷 DSC_2245.NEF · full-resolution RAW"
    nodes[1]["widgets_values"] = [
        str(RAW_PATH), True, False, "camera", 0.0, True,
        f"{OUTPUT_ROOT}/raw_plates", "Linear Rec.709 (sRGB)",
    ]
    nodes[2]["title"] = "📤 OCIO · Linear Rec.709 → ACEScg EXR"
    nodes[2]["widgets_values"] = [
        "ACEScg", f"{OUTPUT_ROOT}/acescg_plates", "half",
        "DSC_2245_acescg.exr",
    ]
    nodes[3]["widgets_values"] = [
        "RAW → ACEScg (file-float, no proxy round-trip)\n\n"
        "AtlasLoadRAW demosaics the Nikon NEF once and writes a full-resolution "
        "scene-linear EXR sidecar tagged Linear Rec.709 (sRGB), matching rawpy's "
        "actual primaries. AtlasExportPlateEXR then transforms that float file "
        "through the built-in OCIO ACES configuration to a tagged ACEScg half EXR.\n\n"
        "The ACEScg plate_ref is attached to the solve, so Maya, Nuke, and Blender "
        "all reference the correct master plate."
    ]
    nodes[19]["widgets_values"] = [
        "ACES 2.0 / Studio", "", "ACEScg", "ACES - ACEScg",
        "sRGB - Display", "ACES 2.0 SDR-video", 1.0,
    ]
    # AtlasDepthMap gained MoGe-only controls after the reference workflow was
    # authored. They are appended at their defaults; positional workflow widget
    # compatibility means these values must never be inserted before model/device.
    nodes[10]["widgets_values"] = [
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "auto", 9, 0, "", 0, 0.25,
    ]
    # Pin the complete positional arrays instead of patching individual indexes:
    # ComfyUI persists widgets by position, and this prevents an edited source
    # workflow (for example relief_grid accidentally dragged to 3714) from
    # leaking into the generated production graph.
    nodes[13]["widgets_values"] = [
        0.0, 0.0, 0.55, 1.0, 4, False, 384, 1.5,
        False, "manual", "", True,
    ]

    # Measured on DSC_2245: filling the background's occluded footprint changed
    # its matte coverage from 18.6% to 81.0% and its torn fraction from 83.7%
    # to 23.3% before edge-budget tuning.  A 14x world-edge gate was the knee:
    # 22.3% torn with 0.48% extreme stretch, versus 23.3%/0.04% at 12x and
    # 20.4%/1.43% at 24x.  Keep normal tearing off and quad coherence on.
    nodes[17]["widgets_values"] = [
        0.0, 0.0, 0.55, 1.0, "bg", 10.0, 384, 1.5,
        True, True, 24, 1.5, 64, 2, "manual", "relief", "", "",
        14.0, 0.0, True,
    ]
    nodes[18]["widgets_values"] = [
        0.0, 0.0, 0.0, 0.55, "fg", 0.0, 384, 1.5,
        False, True, 0, 0.0, 0, 2, "manual", "relief", "", "",
        14.0, 0.0, True,
    ]
    nodes[23]["widgets_values"] = [f"{OUTPUT_ROOT}/blender"]
    nodes[24]["widgets_values"] = [
        f"{OUTPUT_ROOT}/nuke", "off", 50000, 0, 30.0, False,
    ]
    nodes[25]["widgets_values"] = [
        f"{OUTPUT_ROOT}/maya", "off", 50000, 0, 30.0, False,
    ]

    bg_reconstruct, bg_seams = _repair_nodes(28, "bg")
    fg_reconstruct, fg_seams = _repair_nodes(30, "fg")
    retopo = _node(
        32,
        "AtlasRetopologizeLayer",
        title="FINAL RETOPO · all reliefs, after camera-away repair",
        inputs=[("solve", "ATLAS_SOLVE")],
        outputs=[("solve", "ATLAS_SOLVE"), ("report", "STRING")],
        widgets=["*", "decimate", 50000, 0, 30.0, False, 0],
        size=(460, 390),
    )
    debug = _node(
        34,
        "AtlasDebugReport",
        title="MEASURED RESULT · persistent geometry report",
        inputs=[
            ("solve", "ATLAS_SOLVE"),
            ("depth", "ATLAS_DEPTH_MAP"),
            ("status_1", "STRING"),
            ("status_2", "STRING"),
            ("status_3", "STRING"),
            ("status_4", "STRING"),
            ("vlm_report", "STRING"),
        ],
        outputs=[("report", "STRING"), ("json_path", "STRING")],
        widgets=["atlas_debug/DSC_2245_ultimate_raw_acescg.json"],
        size=(470, 260),
    )
    workflow["nodes"].extend([
        bg_reconstruct, bg_seams, fg_reconstruct, fg_seams,
        retopo, _note(33), debug,
    ])
    nodes.update({node["id"]: node for node in workflow["nodes"]})

    # Replace bg -> fg with bg -> reconstruct -> seams -> fg.
    workflow["links"] = [
        link for link in workflow["links"] if link[0] != 35
    ]

    # Every original final-solve consumer now receives the one live-retopologized
    # solve. Exporter retopology remains explicitly off.
    final_consumers = {20, 23, 24, 25}
    for link in workflow["links"]:
        if link[1] == 18 and link[3] in final_consumers:
            link[1] = retopo["id"]
            link[2] = _output_slot(retopo, "solve")

    next_link = max(link[0] for link in workflow["links"]) + 1

    def connect(source_node, output_name, target_node, input_name, kind):
        nonlocal next_link
        _add_link(
            workflow, next_link, source_node, output_name,
            target_node, input_name, kind,
        )
        next_link += 1

    connect(nodes[17], "solve", bg_reconstruct, "solve", "ATLAS_SOLVE")
    connect(nodes[17], "hole_mask", bg_reconstruct, "hole_mask", "MASK")
    connect(bg_reconstruct, "solve", bg_seams, "solve", "ATLAS_SOLVE")
    connect(bg_reconstruct, "remaining_holes", bg_seams, "hole_mask", "MASK")
    connect(bg_seams, "solve", nodes[18], "solve", "ATLAS_SOLVE")

    connect(nodes[18], "solve", fg_reconstruct, "solve", "ATLAS_SOLVE")
    connect(nodes[18], "hole_mask", fg_reconstruct, "hole_mask", "MASK")
    connect(fg_reconstruct, "solve", fg_seams, "solve", "ATLAS_SOLVE")
    connect(fg_reconstruct, "remaining_holes", fg_seams, "hole_mask", "MASK")
    connect(fg_seams, "solve", retopo, "solve", "ATLAS_SOLVE")
    connect(retopo, "solve", debug, "solve", "ATLAS_SOLVE")
    connect(nodes[10], "depth", debug, "depth", "ATLAS_DEPTH_MAP")
    connect(bg_reconstruct, "report", debug, "status_1", "STRING")
    connect(bg_seams, "report", debug, "status_2", "STRING")
    connect(fg_reconstruct, "report", debug, "status_3", "STRING")
    connect(fg_seams, "report", debug, "status_4", "STRING")

    _rebuild_link_slots(workflow)
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    workflow["groups"] = []

    layout = _load_layout()
    workflow = layout.auto_layout(workflow, origin=(80, 80))
    layout.fit_group(
        workflow,
        title="DSC_2245 · ULTIMATE RAW → ACEScg → repair → Maya / Nuke / Blender",
        color="#344f58",
    )
    report = layout.inspect(workflow)
    if report["overlaps"]:
        raise RuntimeError(f"workflow layout still overlaps: {report['summary']}")
    workflow.setdefault("extra", {})["atlas_layout_report"] = report["summary"]
    workflow["extra"]["atlas_pipeline"] = (
        "RAW Linear Rec.709 sidecar -> OCIO ACEScg plate -> bg reconstruct -> "
        "bg camera-away seams -> fg reconstruct -> fg camera-away seams -> "
        "live decimate retopo -> Maya/Nuke/Blender"
    )
    workflow["extra"]["atlas_calibration"] = {
        "depth_model": "V2-Metric-Outdoor-Large",
        "band_split_log_depth": 0.55,
        "relief_grid": 384,
        "depth_edge_rel": 1.5,
        "max_edge_factor": 14.0,
        "quad_coherence": True,
        "background_fill_occluded": True,
        "foreground_fill_occluded": False,
        "seam_width_cells": 3.0,
        "seam_direction": "away_from_camera",
        "retopo_target_vertices_per_relief": 50000,
        "retopo_smoothing": False,
    }
    return workflow


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("pass atlas_raw_3layer_ocio_workflow.json")
    source = Path(sys.argv[1])
    if not source.is_file():
        raise SystemExit(f"source workflow not found: {source}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workflow = build(source)
    OUTPUT.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(workflow["extra"]["atlas_layout_report"])


if __name__ == "__main__":
    main()
