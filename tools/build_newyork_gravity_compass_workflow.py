"""Build the New York bird's-eye flagship Gravity Compass demo workflow."""
from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "examples" / "images" / "newyork_birdseye.jpg"
OUT_GUI = ROOT / "examples" / "local" / "2026-08-02_newyork_gravity_compass_demo.json"
OUT_API = ROOT / "examples" / "local" / "2026-08-02_newyork_gravity_compass_demo_api.json"
LAYOUT = Path.home() / ".agents" / "skills" / "comfyui" / "workflow_layout.py"


def _layout_module():
    spec = importlib.util.spec_from_file_location("atlas_workflow_layout", LAYOUT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workflow layout helper: {LAYOUT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(node_id, node_type, *, title, inputs, outputs, widgets, size):
    return {
        "id": node_id,
        "type": node_type,
        "pos": [0, 0],
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": [{"name": name, "type": kind, "link": None}
                   for name, kind in inputs],
        "outputs": [{"name": name, "type": kind, "links": [], "slot_index": i}
                    for i, (name, kind) in enumerate(outputs)],
        "widgets_values": list(widgets),
        "title": title,
        "properties": {"Node name for S&R": node_type},
    }


def _link(workflow, link_id, source, source_slot, target, target_slot, kind):
    workflow["links"].append(
        [link_id, source["id"], source_slot, target["id"], target_slot, kind])
    source["outputs"][source_slot]["links"].append(link_id)
    target["inputs"][target_slot]["link"] = link_id


def build_gui():
    if not IMAGE.is_file():
        raise FileNotFoundError(IMAGE)

    load = _node(
        1, "AtlasLoadPlate", title="NEW YORK · bird's-eye source plate",
        inputs=[],
        outputs=[("image", "IMAGE"), ("alpha", "MASK"),
                 ("plate_ref", "ATLAS_PLATE_REF"), ("report", "STRING")],
        widgets=[str(IMAGE), "auto", "sRGB - Display", False], size=(390, 300))
    solve = _node(
        2, "AtlasLearnedSolveFromImage", title="LEARNED SOLVE · intentionally review gravity",
        inputs=[("image", "IMAGE"), ("raw_meta", "ATLAS_RAW_META")],
        outputs=[("ATLAS_SOLVE", "ATLAS_SOLVE")],
        # Bird's-eye scale is not the point of this demo. Assume a plausible
        # aerial height so the first queue evaluates gravity without a second
        # depth-model pass; the artist can add a measured scale downstream.
        widgets=["assume", 120.0,
                 "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 36.0, "pinhole", "auto", 0.0], size=(420, 390))
    gate = _node(
        3, "AtlasSolveGate", title="🧭 ORIENTATION COMPASS · gravity + world XYZ",
        inputs=[("solve", "ATLAS_SOLVE"), ("source_image", "IMAGE")],
        outputs=[("solve", "ATLAS_SOLVE"), ("report", "STRING"),
                 ("preview_solve", "ATLAS_SOLVE")],
        widgets=[False, "", False, 0.0, 0.0, False, 0.0], size=(500, 650))
    derive = _node(
        4, "AtlasDeriveProjectionGeometry",
        title="APPROVED AERIAL · derive relief + building proxies",
        inputs=[("solve", "ATLAS_SOLVE"), ("image", "IMAGE"),
                ("exclude_mask", "MASK")],
        outputs=[("solve", "ATLAS_SOLVE"), ("hole_mask", "MASK")],
        widgets=[
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            4, 6, "auto", "both", 256, "azimuth_walls", "aerial",
            "medium", 0.5, False, 0.0, 256, False,
        ], size=(470, 500))
    viewport = _node(
        5, "AtlasBlockoutViewport", title="APPROVED 3D VIEW · updates on compass release",
        inputs=[("solve", "ATLAS_SOLVE"), ("source_image", "IMAGE"),
                ("primary_depth", "ATLAS_DEPTH_MAP"),
                ("controls", "ATLAS_VIEWPORT_LINK"),
                ("shot_cam", "ATLAS_SHOT_CAM"),
                ("output_profile", "ATLAS_OUTPUT_PROFILE"),
                ("debug_matte", "MASK"), ("patch_mask", "MASK")],
        outputs=[("shaded", "IMAGE"), ("depth", "IMAGE"),
                 ("normal", "IMAGE"), ("mask", "IMAGE"),
                 ("path_frames", "IMAGE"), ("camera_path", "ATLAS_CAMERA_PATH"),
                 ("patch_azimuth_view", "STRING"),
                 ("patch_elevation_view", "STRING"),
                 ("patch_distance", "STRING"), ("patch_prompt", "STRING"),
                 ("patch_exact", "STRING"), ("patch_render_mask", "MASK")],
        widgets=[1024, "", 1.0], size=(760, 650))
    debug = _node(
        6, "AtlasDebugReport", title="APPROVED CAMERA + GEOMETRY · diagnostic",
        inputs=[("solve", "ATLAS_SOLVE"), ("depth", "ATLAS_DEPTH_MAP"),
                ("status_1", "STRING"), ("status_2", "STRING"),
                ("status_3", "STRING"), ("status_4", "STRING"),
                ("vlm_report", "STRING")],
        outputs=[("report", "STRING"), ("json_path", "STRING")],
        widgets=["atlas_debug/newyork_gravity_compass.json"], size=(470, 280))
    image_preview = _node(
        7, "PreviewImage", title="SOURCE PLATE · full image reference",
        inputs=[("images", "IMAGE")], outputs=[], widgets=[], size=(430, 500))

    workflow = {
        "id": str(uuid.UUID("41ee7496-82d1-4fe3-a194-b567adb11bf8")),
        "revision": 0,
        "last_node_id": 7,
        "last_link_id": 9,
        "nodes": [load, solve, gate, derive, viewport, debug, image_preview],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "atlas_pipeline": (
                "New York bird's-eye -> learned solve -> embedded Gravity Compass "
                "-> approved aerial projection geometry -> 3D viewport"),
            "atlas_instructions": (
                "Queue once to reveal the compass. Drag gravity or world heading; "
                "release to approve, re-queue once, derive aerial geometry, and update the 3D view."),
        },
        "version": 0.4,
    }
    _link(workflow, 1, load, 0, solve, 0, "IMAGE")
    _link(workflow, 2, solve, 0, gate, 0, "ATLAS_SOLVE")
    _link(workflow, 3, load, 0, gate, 1, "IMAGE")
    _link(workflow, 4, gate, 0, derive, 0, "ATLAS_SOLVE")
    _link(workflow, 5, load, 0, derive, 1, "IMAGE")
    _link(workflow, 6, derive, 0, viewport, 0, "ATLAS_SOLVE")
    _link(workflow, 7, load, 0, viewport, 1, "IMAGE")
    _link(workflow, 8, derive, 0, debug, 0, "ATLAS_SOLVE")
    _link(workflow, 9, load, 0, image_preview, 0, "IMAGE")

    layout = _layout_module()
    layout.auto_layout(workflow, origin=(80, 110))
    layout.fit_group(
        workflow, title="ATLAS FLAGSHIP · NEW YORK GRAVITY COMPASS",
        color="#1d5665", pad=55, title_h=55)
    report = layout.inspect(workflow)
    if report["overlaps"]:
        raise RuntimeError(report["summary"])
    workflow["extra"]["atlas_layout_report"] = report["summary"]
    return workflow


def build_api():
    return {
        "1": {"class_type": "AtlasLoadPlate", "inputs": {
            "file_path": str(IMAGE), "input_colorspace": "auto",
            "output_colorspace": "sRGB - Display", "raw_data": False}},
        "2": {"class_type": "AtlasLearnedSolveFromImage", "inputs": {
            "image": ["1", 0], "height_mode": "assume", "camera_height_m": 120.0,
            "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            "sensor_width_mm": 36.0, "weights": "pinhole", "device": "auto",
            "focal_length_mm": 0.0}},
        "3": {"class_type": "AtlasSolveGate", "inputs": {
            "solve": ["2", 0], "source_image": ["1", 0], "proceed": False,
            "approved_for": "", "apply_override": False,
            "pitch_deg": 0.0, "roll_deg": 0.0,
            "heading_override": False, "heading_deg": 0.0}},
        "4": {"class_type": "AtlasDeriveProjectionGeometry", "inputs": {
            "solve": ["3", 0], "image": ["1", 0],
            "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            "max_walls": 4, "max_objects": 6, "device": "auto",
            "geometry_mode": "both", "relief_grid": 256,
            "primitive_method": "azimuth_walls", "scene_type": "aerial",
            "relief_quality": "medium", "depth_edge_rel": 0.5}},
        "5": {"class_type": "AtlasBlockoutViewport", "inputs": {
            "solve": ["4", 0], "source_image": ["1", 0],
            "resolution": 1024, "client_data": "", "preview_expand": 1.0}},
        "6": {"class_type": "AtlasDebugReport", "inputs": {
            "solve": ["4", 0],
            "file_path": "atlas_debug/newyork_gravity_compass.json"}},
        "7": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }


def main():
    OUT_GUI.parent.mkdir(parents=True, exist_ok=True)
    gui = build_gui()
    OUT_GUI.write_text(json.dumps(gui, indent=2) + "\n", encoding="utf-8")
    OUT_API.write_text(json.dumps(build_api(), indent=2) + "\n", encoding="utf-8")
    print(OUT_GUI)
    print(OUT_API)
    print(gui["extra"]["atlas_layout_report"])


if __name__ == "__main__":
    main()
