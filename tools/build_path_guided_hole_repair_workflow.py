"""Build the shipping Camera Path-guided relief-hole repair workflow."""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples" / "atlas_planar_hole_patch_workflow.json"
OUTPUT = ROOT / "examples" / "atlas_path_guided_hole_repair_workflow.json"
LAYOUT = Path.home() / ".agents" / "skills" / "comfyui" / "workflow_layout.py"


def _load_layout():
    spec = importlib.util.spec_from_file_location("workflow_layout", LAYOUT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load workflow layout helper: {LAYOUT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(
    node_id,
    node_type,
    *,
    inputs,
    outputs,
    widgets,
    title=None,
    size=(390, 300),
):
    value = {
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
    }
    if title:
        value["title"] = title
    return value


def _note(node_id, text, title):
    node = _node(
        node_id,
        "Note",
        inputs=[],
        outputs=[],
        widgets=[text],
        title=title,
        size=(620, 430),
    )
    node["color"] = "#384f68"
    node["bgcolor"] = "#1d2b3a"
    return node


def _rebuild_link_slots(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in nodes.values():
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for link_id, source, source_slot, target, target_slot, _type in workflow["links"]:
        nodes[source]["outputs"][source_slot]["links"].append(link_id)
        nodes[target]["inputs"][target_slot]["link"] = link_id


def build():
    workflow = json.loads(BASE.read_text(encoding="utf-8"))
    workflow["id"] = str(uuid.UUID("e140dd66-1b4a-44bd-b6ea-59c6a8f62c3b"))

    # Keep the tested solve/depth/conservative-patch path and its diagnostics;
    # remove the optional wall branch and the old single-pass retopo note.
    removed = {7, 8, 13}
    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] not in removed]
    workflow["links"] = [
        link for link in workflow["links"]
        if link[1] not in removed and link[3] not in removed
    ]

    nodes = {node["id"]: node for node in workflow["nodes"]}
    nodes[5]["title"] = "PASS 1 · conservative enclosed repairs"
    nodes[5]["widgets_values"] = [
        "", 2, 1024, 30.0, 0.30, 0.04, True, 0.20, 28.0, 2.0]
    nodes[6]["title"] = "RETOPO · after both repair passes"
    nodes[10]["title"] = "AUTHOR REPAIR ANGLE · choose Orbit/Arc then Bake"
    nodes[11]["widgets_values"] = [
        "PATH-GUIDED HOLE REPAIR\n\n"
        "1. Queue once to build the relief and conservative first pass.\n"
        "2. In AUTHOR REPAIR ANGLE enable Camera Path and choose Orbit L/R "
        "or Arc L/R.\n"
        "3. Move the playback lens left for a wider view. Click Bake Repair "
        "Frame; it stores only the indexed final frame while the complete "
        "parametric path remains in camera_path.\n"
        "4. Path-Guided Hole Repair uses frame 0 from the end (the final "
        "frame), renders relaxed candidate planes with stable island IDs, "
        "and returns their exact original-image cells.\n"
        "5. Pass 2 applies the higher edge budget only to that source mask. "
        "The final viewport receives the combined repaired mesh.\n\n"
        "For an artist-painted choice, save/paint ANGLE PREVIEW and wire that "
        "mask into paint_mask, then set selection_mode=paint_overlap."
    ]
    nodes[12]["widgets_values"] = [
        "AUTOMATION CONTRACT\n\n"
        "• frame_offset_from_end=0 always means the final Camera Path frame.\n"
        "• lens_scale_override=0 inherits the path playback lens; values below "
        "1.0 widen the lens.\n"
        "• Candidate pixels are IDs, not ray-cast guesses. A selected pixel "
        "maps back to its complete connected source-space tear island.\n"
        "• Wire the same upstream background/sky mask into exclude_mask. "
        "This subtracts the exterior before open-edge components are found.\n"
        "• all_visible is the unattended/agent mode. max_selected_islands=0 "
        "means every visible candidate, ordered smallest-first.\n"
        "• The selector bypasses only the preview edge gate. Pass 2 remains "
        "the geometry gate (40× here), so view selection never forces unsafe "
        "geometry into the mesh.\n"
        "• Background and sky must remain excluded upstream before enabling "
        "open-edge repair."
    ]

    # Viewport 10 authors the path from pass 1, not from the final retopo.
    for link in workflow["links"]:
        if link[0] == 8:
            link[1], link[2], link[3], link[4] = 22, 0, 6, 0
        elif link[0] == 13:
            link[1], link[2], link[3], link[4] = 5, 0, 10, 0

    added = [
        _node(
            21,
            "AtlasPathGuidedHoleRepair",
            inputs=[
                ("solve", "ATLAS_SOLVE"),
                ("hole_mask", "MASK"),
                ("camera_path", "ATLAS_CAMERA_PATH"),
                ("path_frames", "IMAGE"),
                ("paint_mask", "MASK"),
                ("exclude_mask", "MASK"),
            ],
            outputs=[
                ("repair_mask", "MASK"),
                ("angle_preview", "IMAGE"),
                ("visible_islands", "MASK"),
                ("report", "STRING"),
            ],
            widgets=[
                "", 0, 0.0, 1024, "all_visible", 0, 8, 0.02,
                2, 1024, 30.0, 0.45, 0.04, False, 0.20,
            ],
            title="PATH-GUIDED TEAR IDS · final frame, inherited lens",
            size=(460, 560),
        ),
        _node(
            22,
            "AtlasPlanarHolePatch",
            inputs=[("solve", "ATLAS_SOLVE"), ("hole_mask", "MASK")],
            outputs=[
                ("solve", "ATLAS_SOLVE"),
                ("remaining_holes", "MASK"),
                ("report", "STRING"),
                ("created_islands", "MASK"),
            ],
            widgets=[
                "", 2, 1024, 30.0, 0.45, 0.04, False, 0.20, 40.0, 2.0,
            ],
            title="PASS 2 · scoped repair (40× edge, 2× depth)",
            size=(430, 430),
        ),
        _node(
            23,
            "PreviewImage",
            inputs=[("images", "IMAGE")],
            outputs=[("images", "IMAGE")],
            widgets=[],
            title="ANGLE PREVIEW · selected IDs are magenta",
            size=(520, 500),
        ),
        _node(
            24,
            "MaskToImage",
            inputs=[("mask", "MASK")],
            outputs=[("IMAGE", "IMAGE")],
            widgets=[],
            title="VISIBLE ANGLE ISLANDS",
            size=(300, 100),
        ),
        _node(
            25,
            "PreviewImage",
            inputs=[("images", "IMAGE")],
            outputs=[("images", "IMAGE")],
            widgets=[],
            title="VISIBLE ANGLE ISLANDS · white",
            size=(420, 380),
        ),
        _node(
            26,
            "ShowText|pysssss",
            inputs=[("text", "STRING")],
            outputs=[("STRING", "STRING")],
            widgets=[],
            title="PATH REPAIR REPORT · frame/lens/IDs",
            size=(480, 300),
        ),
        _node(
            27,
            "ShowText|pysssss",
            inputs=[("text", "STRING")],
            outputs=[("STRING", "STRING")],
            widgets=[],
            title="PASS 2 REPORT · final acceptance gate",
            size=(480, 300),
        ),
        _node(
            28,
            "AtlasBlockoutViewport",
            inputs=[
                ("solve", "ATLAS_SOLVE"),
                ("source_image", "IMAGE"),
                ("primary_depth", "ATLAS_DEPTH_MAP"),
                ("controls", "ATLAS_VIEWPORT_LINK"),
                ("shot_cam", "ATLAS_SHOT_CAM"),
                ("output_profile", "ATLAS_OUTPUT_PROFILE"),
                ("debug_matte", "MASK"),
                ("patch_mask", "MASK"),
            ],
            outputs=[
                ("shaded", "IMAGE"), ("depth", "IMAGE"),
                ("normal", "IMAGE"), ("mask", "IMAGE"),
                ("path_frames", "IMAGE"),
                ("camera_path", "ATLAS_CAMERA_PATH"),
                ("patch_azimuth_view", "STRING"),
                ("patch_elevation_view", "STRING"),
                ("patch_distance", "STRING"),
                ("patch_prompt", "STRING"),
                ("patch_exact", "STRING"),
                ("patch_render_mask", "MASK"),
            ],
            widgets=[1024, "", 1.0],
            title="FINAL GEOMETRY · inspect repaired + retopologized mesh",
            size=(720, 620),
        ),
        _note(
            29,
            "AGENT LOOP\n\n"
            "An MCP/agent can iterate without brush input:\n"
            "• choose Orbit/Arc in the author viewport\n"
            "• Bake Repair Frame (use Bake Full Path only for video)\n"
            "• inspect PATH REPAIR REPORT\n"
            "• vary frame_offset_from_end and lens_scale_override\n"
            "• cap max_selected_islands for smaller-first batches\n"
            "• inspect PASS 2 REPORT and final viewport\n"
            "• stop when no visible candidate IDs remain or the geometry gate "
            "rejects the remainder\n\n"
            "This makes the chosen camera frame an explicit, reproducible "
            "geometry-repair input rather than hidden viewport state.",
            "MCP / AGENTIC REPAIR LOOP",
        ),
    ]
    workflow["nodes"].extend(added)
    workflow["links"].extend([
        [27, 5, 0, 21, 0, "ATLAS_SOLVE"],
        [28, 5, 1, 21, 1, "MASK"],
        [29, 10, 5, 21, 2, "ATLAS_CAMERA_PATH"],
        [30, 10, 4, 21, 3, "IMAGE"],
        [31, 21, 0, 22, 1, "MASK"],
        [32, 5, 0, 22, 0, "ATLAS_SOLVE"],
        [33, 21, 1, 23, 0, "IMAGE"],
        [34, 21, 2, 24, 0, "MASK"],
        [35, 24, 0, 25, 0, "IMAGE"],
        [36, 21, 3, 26, 0, "STRING"],
        [37, 22, 2, 27, 0, "STRING"],
        [38, 6, 0, 28, 0, "ATLAS_SOLVE"],
        [39, 1, 0, 28, 1, "IMAGE"],
        [40, 3, 0, 28, 2, "ATLAS_DEPTH_MAP"],
        [41, 22, 1, 28, 6, "MASK"],
        [42, 22, 3, 28, 7, "MASK"],
    ])
    _rebuild_link_slots(workflow)

    layout = _load_layout()
    workflow["groups"] = []
    layout.auto_layout(workflow, origin=(80, 100))
    layout.fit_group(
        workflow,
        title="CAMERA PATH-GUIDED RELIEF REPAIR · SOURCE → PATH IDS → SCOPED PATCH → FINAL",
        color="#31536b",
        pad=60,
        title_h=60,
    )
    inspection = layout.inspect(workflow)
    if inspection["overlaps"]:
        raise RuntimeError(f"workflow layout overlaps: {inspection}")

    workflow["last_node_id"] = 29
    workflow["last_link_id"] = 42
    workflow["extra"] = {
        "ds": {"scale": 0.55, "offset": [60, 80]},
        "atlas_example": "path_guided_hole_repair",
        "atlas_agentic_geometry_repair": True,
        "layout_inspection": inspection["summary"],
        "notes": (
            "Camera Path final-frame candidate-ID raster drives an exact "
            "source-space mask for a scoped second planar patch pass."
        ),
    }
    OUTPUT.write_text(
        json.dumps(workflow, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(inspection["summary"])


if __name__ == "__main__":
    build()
