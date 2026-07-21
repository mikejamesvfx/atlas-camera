"""Rebuild the lightweight artist and agentic quickstarts from live schemas.

The standard workflow intentionally leaves ``primary_depth`` disconnected;
the occlusion workflow is its controlled A/B twin and adds exactly that one
connection.  Keeping the graphs otherwise equivalent makes the viewport
occlusion behaviour easy to evaluate without conflating it with a different
solve or mesh.

Usage::

    python tools/rebuild_shipping_quickstarts.py
    python tools/rebuild_shipping_quickstarts.py --host 127.0.0.1:8188
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rebuild_staged_master_workflow import Graph, _fetch_object_info, _load_layout_module


ROOT = Path(__file__).resolve().parents[1]
STANDARD_OUTPUT = ROOT / "examples" / "atlas_input_quickstart_workflow.json"
OCCLUSION_OUTPUT = ROOT / "examples" / "atlas_occlusion_cull_quickstart_workflow.json"
STANDARD_AGENTIC_OUTPUT = (
    ROOT / "examples" / "atlas_input_quickstart_agentic_assessment_workflow.json")
OCCLUSION_AGENTIC_OUTPUT = (
    ROOT / "examples" / "atlas_occlusion_cull_quickstart_agentic_assessment_workflow.json")
WORKFLOW_IDS = {
    (False, False): "53b0cf46-4d8e-5467-8ad6-2ad768f43d4e",
    (True, False): "5a2cc223-e3f2-5322-969d-9262fd4c9805",
    (False, True): "06890c21-d10f-506c-a558-df61d28415ca",
    (True, True): "59e7a212-07ae-53fd-b5a9-2e79979f1777",
}


STANDARD_NOTE = """ATLAS INPUT QUICKSTART — the fastest image-to-3D path.

1. Choose a plate in Load Image and queue once.
2. Click Camera View for the recovered view; click Project to inspect the photo projection.
3. Orbit gently. This standard workflow deliberately leaves primary_depth disconnected, so ✂ Occlude is unavailable. Use the occlusion-cull quickstart for the matched A/B test.

AtlasInput defaults to one high-resolution relief mesh (layers=0). Optional upgrades:
• sky_heuristic is off so bundled example.png always produces exportable geometry; turn it on for real outdoor plates after checking the horizon mask
• layers 2–4: watertight depth-band projection layers
• use_vlm: image-specific prompts, geometry choices, and four-band boundaries
• sky / scope_prompts: native AtlasSAM3Mask when [sam3] is installed, with native semantic fallback — no ComfyUI-RMBG or triton dependency
• inpaint: the optional legacy fast LaMa path, if comfyui-inpaint-nodes + big-lama.pt are installed

For release-quality clean plates use atlas_camera_staged_master_workflow.json: four native cropped SDXL inpaint subgraphs, explicit masks, gates, previews, and per-layer controls.

OUTPUT DESK / COLOR
The profile link carries OCIO-style metadata to the viewport and DCC handoffs. Browser display is a preview only. Exported RGB stays associated with its color metadata; alpha/mattes are data and are never display-transformed.

DCC EXPORTS
• At layers=0 the graph writes Solve JSON, USD camera, Blender build script, textured Relief OBJ/GLB, a Nuke relief-projection scene, and a Maya relief review scene.
• The separate Nuke Layers / Maya Layers nodes activate when AtlasInput layers >= 1 and add per-band cameras, geometry, plates, and mattes.
• Retopology is export-only. Leave it off for projection fidelity; select quad only when pyinstantmeshes is installed and a topology-changing DCC handoff is intended."""


OCCLUSION_NOTE = """OCCLUSION-CULL QUICKSTART — matched depth, controlled A/B.

1. Choose a plate and queue once.
2. Click Project, then toggle ✂ Occlude while orbiting.
3. Return to Camera View to judge the source boundary. The shader compares each projected fragment with the same metric primary depth that built the relief mesh, rejects hidden/grazing fragments, averages the depth edge, feathers inward, and extends nearby straight RGB outward under that feather.

This graph differs from atlas_input_quickstart_workflow.json by ONE functional wire: AtlasInput.depth → viewport.primary_depth. Do not substitute depth from another model, retopology pass, resolution, crop, or camera — mismatched depth causes the large false cutouts that this A/B workflow is designed to avoid. Keep preview_expand at 1.0 while Project is active.

AtlasInput defaults to one relief mesh. sky_heuristic is off so bundled example.png remains exportable; turn it on for real outdoor plates after checking the horizon mask. Native sky/scope segmentation uses AtlasSAM3Mask with semantic fallback and does not require ComfyUI-RMBG/triton. For production clean plates and larger moves, use atlas_camera_staged_master_workflow.json with its four cropped native SDXL inpaint subgraphs.

OUTPUT DESK / COLOR
The profile is metadata for DCC handoff; the browser shader is a display preview. Occlusion changes coverage only. It does not color-transform alpha: mattes remain linear data, while any RGB filtering happens on straight color before coverage is applied downstream (unpremultiply → filter/dilate RGB → premultiply).

DCC EXPORTS
At layers=0 the graph writes Solve JSON, USD camera, Blender build script, textured Relief OBJ/GLB, a Nuke relief-projection scene, and a Maya relief review scene. The separate Nuke Layers / Maya Layers packages activate when layers >= 1. Retopology remains export-only and never changes the viewport mesh or its matched primary depth."""


def _note(graph: Graph, text: str) -> dict:
    """Add ComfyUI's frontend-only Note node (absent from /object_info)."""
    graph._node_id += 1
    node = {
        "id": graph._node_id,
        "type": "Note",
        "pos": [0, 0],
        "size": [820, 520],
        "flags": {},
        "order": graph._node_id - 1,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"Node name for S&R": "Note"},
        "widgets_values": [text],
        "title": "READ ME · workflow contract and production handoff",
    }
    graph.nodes.append(node)
    return node


def _group(nodes: list[dict], title: str, color: str) -> dict:
    x0 = min(node["pos"][0] for node in nodes)
    y0 = min(node["pos"][1] for node in nodes)
    x1 = max(node["pos"][0] + node["size"][0] for node in nodes)
    y1 = max(node["pos"][1] + node["size"][1] for node in nodes)
    return {
        "title": title,
        "bounding": [x0 - 45, y0 - 88, x1 - x0 + 90, y1 - y0 + 133],
        "color": color,
        "font_size": 24,
        "flags": {},
    }


def build(object_info: dict, layout, *, occlusion: bool,
          agentic_assessment: bool = False) -> dict:
    graph = Graph(object_info)
    slug = "atlas_occlusion_quickstart" if occlusion else "atlas_input_quickstart"
    title = "✂ OCCLUSION A/B · matched primary depth" if occlusion else "ATLAS INPUT · instant relief"

    sample_image = ("moge_hangar_proj.jpg" if occlusion
                    else "ghosttown.jpg") if agentic_assessment else "example.png"
    sample_label = ("SPACE HANGAR" if occlusion else "GHOST TOWN")
    load = graph.node(
        "LoadImage",
        title=(f"1 · SOURCE PLATE · {sample_label} QA SAMPLE"
               if agentic_assessment else "1 · SOURCE PLATE"), values={
        "image": sample_image, "image_upload": "image",
    }, size=(360, 310))
    atlas_input = graph.node("AtlasInput", title="2 · SOLVE + RELIEF · expand options here", values={
        "layers": 0,
        "mesh": "relief",
        "mesh_resolution": 512,
        "use_vlm": False,
        "vlm_provider": "lmstudio",
        "vlm_model": "",
        "sky": False,
        "sky_prompt": "sky",
        "scope_prompts": "",
        "inpaint": False,
        "upscale_model": "",
        "edge_extend_px": 24,
        "max_edge_factor": 12.0,
        # The shipping contract is a successful first queue on ComfyUI's
        # neutral bundled placeholder.  Its content can be classified almost
        # entirely as above-horizon far field, leaving an empty export mesh.
        # Artists should enable the outdoor-only heuristic on real plates.
        "sky_heuristic": False,
        "normal_edge_deg": 0.0,
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "vlm_scope": True,
    }, size=(440, 590))
    controls = graph.node("AtlasViewportControls", title="OUTPUT DESK · OCIO metadata", values={
        "config_label": "ACES 2.0 / Studio",
        "config_path": "",
        "working_colorspace": "ACEScg",
        "output_colorspace": "ACES - ACEScg",
        "display": "sRGB - Display",
        "view": "ACES 2.0 SDR-video",
        "display_trim": 1.0,
    }, size=(440, 300))
    viewport = graph.node("AtlasBlockoutViewport", title=title, values={
        "resolution": 1280 if occlusion else 1024,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    note_text = OCCLUSION_NOTE if occlusion else STANDARD_NOTE
    if agentic_assessment:
        note_text += """

AGENTIC TERMINAL QA
This variant starts on a real Atlas Ghost Town or Space Hangar plate, appends AtlasAssessOutput plus an exact-evidence preview, and enables its VLM by default. If the browser/WebGL shaded pass is blank in a headless run, Atlas reconstructs the recovered-camera image from the actual projection plates, mattes, and relief UV coverage. Framing, canonical projection edges, inpaint seams, and colour continuity are assessable; orbit/grazing occlusion remains explicitly inconclusive visually and is reported by deterministic geometry coverage. The exact assessed PNG, coverage matte, SHA-256, and stable JSON are retained beside one another and returned inline by atlas_run_workflow. A browser Render Proxy Passes capture or DCC render remains the final orbit/lighting oracle."""
    note = _note(graph, note_text)

    solve_json = graph.node("AtlasExportSolveJSON", title="Solve JSON · portable camera", values={
        "output_path": f"atlas_exports/{slug}/atlas_solve.json",
    }, size=(390, 150))
    nuke_layers = graph.node("AtlasExportNukeLayers", title="Nuke Layers · layers ≥ 1", values={
        "output_dir": f"atlas_exports/{slug}/nuke_layers",
        "retopo_method": "off",
        "retopo_target_vertex_count": 2000,
        "retopo_smooth_iterations": 0,
        "retopo_crease_angle": 30.0,
        "retopo_pure_quad": False,
    }, size=(410, 300))
    maya_layers = graph.node("AtlasExportMayaLayers", title="Maya Layers · layers ≥ 1", values={
        "output_dir": f"atlas_exports/{slug}/maya_layers",
        "retopo_method": "off",
        "retopo_target_vertex_count": 2000,
        "retopo_smooth_iterations": 0,
        "retopo_crease_angle": 30.0,
        "retopo_pure_quad": False,
    }, size=(410, 300))
    blender = graph.node("AtlasExportBlender", title="Blender · projection handoff", values={
        "output_dir": f"atlas_exports/{slug}/blender",
    }, size=(410, 160))
    usd = graph.node("AtlasExportUSD", title="USD · recovered camera", values={
        "output_dir": f"atlas_exports/{slug}/usd",
    }, size=(410, 140))
    relief = graph.node("AtlasExportReliefMesh", title="Relief OBJ/GLB · export-only retopo", values={
        "output_dir": f"atlas_exports/{slug}/relief",
        "grid_long_edge": 128,
        "depth_edge_rel": 0.5,
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "device": "auto",
        "format": "both",
        "use_solve_mesh": True,
        "max_edge_factor": 12.0,
        "normal_edge_deg": 0.0,
        "fill_interior_holes": False,
        "max_hole_edges": 64,
        "fill_depth_near_m": 0.0,
        "fill_depth_far_m": 0.0,
        "retopo_method": "off",
        "retopo_target_vertex_count": 2000,
        "retopo_smooth_iterations": 0,
        "retopo_crease_angle": 30.0,
        "retopo_pure_quad": False,
    }, size=(440, 610))
    nuke_relief = graph.node("AtlasExportNuke", title="Nuke · relief projection · works at layers 0", values={
        "output_dir": f"atlas_exports/{slug}/nuke_relief",
    }, size=(430, 180))
    maya_relief = graph.node("AtlasExportMayaReviewScene", title="Maya · relief review · works at layers 0", values={
        "output_dir": f"atlas_exports/{slug}/maya_relief",
    }, size=(430, 180))
    assess_output = None
    evidence_preview = None
    if agentic_assessment:
        assess_output = graph.node(
            "AtlasAssessOutput", title="TERMINAL QA · agent/headless report", values={
                "enabled": True,
                "provider": "lmstudio",
                "model": "",
                "base_url": "",
                "extra_instructions": (
                    "Review the recovered camera view and release readiness."),
                "file_path": f"atlas_debug/{slug}_agentic_output_assessment.json",
                "api_key": "",
                "offload_model": True,
                "fallback_to_source": True,
            }, size=(520, 430))
        evidence_preview = graph.node(
            "PreviewImage", title="ASSESSED EVIDENCE · exact VLM image",
            size=(520, 420))

    graph.connect(load, "IMAGE", atlas_input, "image")
    graph.connect(atlas_input, "solve", viewport, "solve")
    graph.connect(atlas_input, "image", viewport, "source_image")
    if occlusion:
        graph.connect(atlas_input, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")
    if assess_output is not None:
        graph.connect(viewport, "shaded", assess_output, "camera_view")
        graph.connect(atlas_input, "solve", assess_output, "solve")
        graph.connect(atlas_input, "image", assess_output, "source_image")
        graph.connect(atlas_input, "depth", assess_output, "depth")
        graph.connect(assess_output, "assessed_image", evidence_preview, "images")

    for exporter in (solve_json, nuke_layers, maya_layers, blender, usd,
                     relief, nuke_relief, maya_relief):
        graph.connect(atlas_input, "solve", exporter, "solve")
    graph.connect(atlas_input, "image", relief, "image")
    graph.connect(relief, "obj_path", nuke_relief, "relief_mesh_obj_path")
    graph.connect(relief, "obj_path", maya_relief, "relief_mesh_obj_path")
    for exporter in (nuke_layers, maya_layers, blender, nuke_relief,
                     maya_relief):
        graph.connect(controls, "output_profile", exporter, "output_profile")

    # A compact, stable artist layout.  The standard/occlusion pair uses the
    # exact same coordinates so visual differences come from one depth wire.
    load["pos"] = [100, 180]
    atlas_input["pos"] = [560, 140]
    controls["pos"] = [100, 750]
    viewport["pos"] = [1120, 140]
    note["pos"] = [1160, 940]
    solve_json["pos"] = [2180, 140]
    relief["pos"] = [2180, 400]
    nuke_relief["pos"] = [2740, 400]
    maya_relief["pos"] = [2740, 700]
    blender["pos"] = [2740, 1000]
    usd["pos"] = [2740, 1280]
    nuke_layers["pos"] = [3280, 400]
    maya_layers["pos"] = [3280, 950]
    if assess_output is not None:
        assess_output["pos"] = [3840, 400]
        evidence_preview["pos"] = [4440, 400]

    workflow_graph = {"nodes": graph.nodes, "links": graph.links}
    check = layout.inspect(workflow_graph)
    if check["overlaps"]:
        raise RuntimeError(f"quickstart layout overlaps: {check['overlaps']}")

    core_nodes = [load, atlas_input, controls, viewport, note]
    export_nodes = [solve_json, relief, nuke_relief, maya_relief, blender,
                    usd, nuke_layers, maya_layers]
    if assess_output is not None:
        export_nodes.extend((assess_output, evidence_preview))
    return {
        "id": WORKFLOW_IDS[(occlusion, agentic_assessment)],
        "revision": 1,
        "last_node_id": graph._node_id,
        "last_link_id": graph._link_id,
        "nodes": graph.nodes,
        "links": graph.links,
        "groups": [
            _group(core_nodes, "1 · LOAD → SOLVE → VIEWPORT · read the note below", "#35536b"),
            _group(export_nodes, "2 · SHIPPING OUTPUTS · DEFAULT RELIEF + OPTIONAL LAYER PACKAGES", "#375c4a"),
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.58, "offset": [35, 85]},
            "frontendVersion": "1.25.11",
            "workflowRendererVersion": "LG",
            "atlas_quickstart_version": 4,
            "atlas_occlusion_primary_depth": occlusion,
            "atlas_agentic_assessment": agentic_assessment,
            "atlas_notes": (
                "Matched primary depth is wired for Project + Occlude A/B."
                if occlusion else
                "Fast relief path; use the occlusion twin for matched-depth culling."
            ),
        },
        "version": 0.4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--standard-output", type=Path, default=STANDARD_OUTPUT)
    parser.add_argument("--occlusion-output", type=Path, default=OCCLUSION_OUTPUT)
    parser.add_argument("--standard-agentic-output", type=Path,
                        default=STANDARD_AGENTIC_OUTPUT)
    parser.add_argument("--occlusion-agentic-output", type=Path,
                        default=OCCLUSION_AGENTIC_OUTPUT)
    args = parser.parse_args()
    object_info = _fetch_object_info(args.host)
    layout = _load_layout_module()
    jobs = (
        (args.standard_output, False, False),
        (args.occlusion_output, True, False),
        (args.standard_agentic_output, False, True),
        (args.occlusion_agentic_output, True, True),
    )
    for output, occlusion, agentic_assessment in jobs:
        workflow = build(object_info, layout, occlusion=occlusion,
                         agentic_assessment=agentic_assessment)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print(f"  {layout.inspect(workflow)['summary']}")


if __name__ == "__main__":
    main()
