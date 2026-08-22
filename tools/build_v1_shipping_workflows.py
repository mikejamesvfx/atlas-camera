"""Generate the four v1 shipping example workflows from a live ComfyUI.

The set is deliberately small and each file answers a different question:

===================================  ==========================================
``atlas_input_quickstart``           "Does this thing work?"  One node.
``atlas_quickstart_solve_project_export``  "What are the stages?"  The explicit
                                     chain the front door hides, so a user can
                                     take over any one of them.
``atlas_export_fanout``              "What do I get in my DCC?"  Eight exporters
                                     routed by one ``AtlasProject``.
``atlas_layered_projection``         "How do I matte-paint with it?"  The 2.5D
                                     layer stack.
===================================  ==========================================

Nothing here is hand-authored.  ComfyUI's UI format is redundantly linked (the
top-level ``links`` array AND every node's ``inputs[].link`` /
``outputs[].links`` must agree) and ``widgets_values`` is POSITIONAL, so a node
that gains an appended widget silently shifts every value after it.  Both fail
on load without a word.  The ``Graph`` DSL borrowed from
``rebuild_staged_master_workflow`` reads widget order and defaults from the
running server's ``/object_info``, which is the same source ComfyUI itself
uses, so the generator can never fall out of step with a node.

Two node choices are worth recording, because both were forced by a REQUIRED
input rather than chosen:

* ``AtlasHorizonMask`` supplies ``AtlasSkyDomeLayer.sky_mask`` in the layered
  workflow.  The obvious source is a segmenter, but ``AtlasSAM3Mask`` needs the
  ``[sam3]`` extra and a Hugging Face token for a gated repo, which a fresh
  clone does not have.  The horizon mask is pure solve geometry with zero
  dependencies, so the example stays runnable for everyone.
* ``AtlasBoundedBand`` is NOT in the layered workflow despite being part of the
  layer family, for the same reason: its ``foreground_mask`` is required and
  only a segmenter produces one.  ``AtlasDepthBandSplit`` covers the same beat
  (an ordered near/far split) with no dependency.

Usage::

    python tools/build_v1_shipping_workflows.py
    python tools/build_v1_shipping_workflows.py --host 127.0.0.1:8188

Loading is not acceptance.  Run every result through
``tools/workflow_benchmark.py`` before it lands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rebuild_staged_master_workflow import Graph, _fetch_object_info, _load_layout_module


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# Stable per-file ids, carried over from the workflows these replace so the
# frontend keeps treating each as the same document.  Must be real UUIDs and
# must differ from one another — tests/test_example_workflows pins both rules.
WORKFLOW_IDS = {
    "atlas_input_quickstart_workflow": "1b2f3ca6-51b4-511a-948f-5eb82d480483",
    "atlas_quickstart_solve_project_export_workflow": "07a2ce62-8289-56cd-86db-740e798798d9",
    "atlas_export_fanout_workflow": "abcf0898-2667-533f-a424-46fadd40dce5",
    "atlas_layered_projection_workflow": "ff547cce-9b2b-5ca6-915a-c582c1ab14b5",
    # Hero 02. New file, so no id to carry over — uuid5 of the slug under the
    # DNS namespace, which is reproducible if it ever needs regenerating.
    "atlas_hero_02_photo_to_editable_scene_workflow":
        "77ce20a3-ab7f-56c7-afa5-ce91642e5373",
    # The .atlas producer. Same rule: uuid5 of the slug under the DNS
    # namespace, so it is reproducible if it ever needs regenerating.
    "atlas_photo_to_atlas_scene_workflow":
        "99f2f654-6a3c-5853-b6d6-adc33295a89f",
}

# Exterior plates: the metric outdoor model is the doctrine choice for these
# (docs/development/design-rules.md, depth model doctrine).  Interiors would take MoGe.
OUTDOOR_DEPTH = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def _note(graph: Graph, text: str, title: str, *, size=(820, 520)) -> dict:
    """ComfyUI's frontend-only Note node, which is absent from /object_info."""
    graph._node_id += 1
    node = {
        "id": graph._node_id,
        "type": "Note",
        "pos": [0, 0],
        "size": list(size),
        "flags": {},
        "order": graph._node_id - 1,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"Node name for S&R": "Note"},
        "widgets_values": [text],
        "title": title,
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


def _controls(graph: Graph, *, vfx: bool) -> dict:
    """The output desk.  Its profile is OCIO-style METADATA for the DCC handoff;
    the browser shader is a display preview, never the deliverable."""
    return graph.node("AtlasViewportControls", title="OUTPUT DESK · OCIO metadata", values={
        "config_label": "ACES 2.0 / Studio" if vfx else "sRGB / Standard",
        "config_path": "",
        "working_colorspace": "ACEScg" if vfx else "sRGB",
        "output_colorspace": "ACES - ACEScg" if vfx else "sRGB - Texture",
        "display": "sRGB - Display",
        "view": "ACES 2.0 SDR-video" if vfx else "Un-tone-mapped",
        "display_trim": 1.0,
    }, size=(440, 300))


def _finish(graph: Graph, slug: str, group_specs, layout, notes: str) -> dict:
    """Lay the graph out, then box the groups around where the nodes LANDED.

    Positions are not hand-typed. `est_size` in the layout module derives a
    node's rendered height from its slot and widget counts, and an Atlas node
    with twenty-odd widgets renders far taller than a hand-guessed size — so
    hand coordinates that looked generous overlapped anyway. Auto-layout is
    deterministic (columns by dependency depth, barycentre within a column), so
    the result is stable across regenerations, and the overlap check below is
    then a real assertion rather than a formality.

    Groups are boxed AFTER layout because `_group` reads pos/size, and
    auto_layout writes both back onto the same node objects.
    """
    workflow_graph = layout.auto_layout({"nodes": graph.nodes, "links": graph.links})
    check = layout.inspect(workflow_graph)
    if check["overlaps"]:
        raise RuntimeError(f"{slug} layout overlaps: {check['overlaps']}")
    groups = [_group(nodes, title, color) for title, color, nodes in group_specs]
    return {
        "id": WORKFLOW_IDS[slug],
        "revision": 1,
        "last_node_id": graph._node_id,
        "last_link_id": graph._link_id,
        "nodes": graph.nodes,
        "links": graph.links,
        "groups": groups,
        "config": {},
        "extra": {
            "ds": {"scale": 0.58, "offset": [35, 85]},
            "frontendVersion": "1.25.11",
            "workflowRendererVersion": "LG",
            "atlas_shipping_set": "v1",
            "atlas_notes": notes,
        },
        "version": 0.4,
    }


# ---------------------------------------------------------------------------
# 1 · the front door
# ---------------------------------------------------------------------------

QUICKSTART_NOTE = """ATLAS INPUT QUICKSTART — the front door. One node, plate in, camera and geometry out.

1. Queue once. Load Image starts on ComfyUI's bundled example.png so this runs with nothing to download.
   For a REAL result, swap in your own photo — or grab the Atlas plate pack from mikejamesvfx.com. No images
   ship in this repo, so every shipping workflow starts on the neutral placeholder by construction.
2. Click Camera View for the recovered camera; click Project to inspect the photo projection.
3. Orbit gently. Solve JSON lands in atlas_exports/ and carries the camera anywhere.

WHY layers=0 AND sky_heuristic OFF
layers=0 builds ONE high-resolution relief mesh, which is the fastest honest answer for a first run. The outdoor sky heuristic is off because the bundled example.png classifies almost entirely as above-horizon far field, which would leave an empty export mesh. Turn it ON for real outdoor plates once you have checked the horizon mask.

WHAT TO REACH FOR NEXT
• atlas_quickstart_solve_project_export_workflow.json — the same result with every stage exposed and overridable.
• atlas_export_fanout_workflow.json — Nuke, Maya, Blender, USD and a review package from one solve, routed into a delivery project tree.
• atlas_layered_projection_workflow.json — the 2.5D layer stack a matte painter actually works in.

OPTIONAL UPGRADES ON AtlasInput
• layers 2-4: watertight depth-band projection layers instead of one mesh
• use_vlm: image-specific prompts, geometry choices and band boundaries from a local VLM
• sky / scope_prompts: native AtlasSAM3Mask when [sam3] is installed, with a semantic fallback
• inpaint: the optional LaMa path, if comfyui-inpaint-nodes and big-lama.pt are present

TERMINAL QA is wired but DISABLED. AtlasAssessOutput needs a local VLM (LM Studio or Ollama); set enabled=true once you have one running and it writes a headless verdict for agent-driven runs. Left off, the graph runs green on a fresh clone with no model downloads beyond depth.

COLOUR
The profile link carries OCIO-style metadata to the viewport and the DCC handoffs. The browser is a display preview only. Exported RGB stays associated with its colour metadata; alpha and mattes are data and are never display-transformed."""


def build_quickstart(object_info: dict, layout) -> dict:
    slug = "atlas_input_quickstart_workflow"
    graph = Graph(object_info)

    load = graph.node("LoadImage", title="1 · SOURCE PLATE", values={
        "image": "example.png", "image_upload": "image",
    }, size=(360, 310))
    atlas_input = graph.node("AtlasInput", title="2 · SOLVE + RELIEF · expand options here", values={
        "layers": 0,
        "mesh": "relief",
        "mesh_resolution": 512,
        "use_vlm": False,
        "sky": False,
        "inpaint": False,
        "edge_extend_px": 24,
        "max_edge_factor": 12.0,
        "sky_heuristic": False,
        "normal_edge_deg": 0.0,
        "depth_model": OUTDOOR_DEPTH,
        "vlm_scope": True,
    }, size=(440, 620))
    controls = _controls(graph, vfx=False)
    viewport = graph.node("AtlasBlockoutViewport", title="3 · VIEWPORT · camera view + projection", values={
        "resolution": 1024,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    layer_preview = graph.node("AtlasLayerPreview", title="LAYER PREVIEW · matte over plate", values={
        "layer_index": 0,
        "color_hex": "",
    }, size=(400, 300))
    # AtlasLayerPreview emits an IMAGE and nothing consumed it, so the
    # preview never rendered. A sink is not decoration here.
    layer_preview_sink = graph.node(
        "PreviewImage",
        title="LAYER PREVIEW — without a sink this never rendered 🖼",
        size=(320, 300))
    assess = graph.node("AtlasAssessOutput", title="TERMINAL QA · disabled, needs a local VLM", values={
        "enabled": False,
        "provider": "lmstudio",
        "model": "",
        "base_url": "",
        "extra_instructions": "Review the recovered camera view and release readiness.",
        "file_path": f"atlas_debug/{slug}_assessment.json",
        "api_key": "",
        "offload_model": True,
        "fallback_to_source": True,
    }, size=(520, 430))
    solve_json = graph.node("AtlasExportSolveJSON", title="Solve JSON · portable camera", values={
        "output_path": f"atlas_exports/{slug}/atlas_solve.json",
    }, size=(390, 150))
    note = _note(graph, QUICKSTART_NOTE, "READ ME · what this graph is and where to go next")

    graph.connect(load, "IMAGE", atlas_input, "image")
    graph.connect(atlas_input, "solve", viewport, "solve")
    graph.connect(atlas_input, "image", viewport, "source_image")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")
    graph.connect(atlas_input, "image", layer_preview, "image")
    graph.connect(atlas_input, "sky_mask", layer_preview, "mask")
    graph.connect(layer_preview, "image", layer_preview_sink, "images")
    graph.connect(viewport, "shaded", assess, "camera_view")
    graph.connect(atlas_input, "solve", assess, "solve")
    graph.connect(atlas_input, "image", assess, "source_image")
    graph.connect(atlas_input, "depth", assess, "depth")
    graph.connect(atlas_input, "solve", solve_json, "solve")

    groups = [
        ("1 · LOAD → SOLVE → VIEWPORT · read the note", "#35536b",
         [load, atlas_input, controls, viewport, note]),
        ("2 · INSPECT + EXPORT", "#375c4a",
         [layer_preview, layer_preview_sink, solve_json, assess]),
    ]
    return _finish(graph, slug, groups, layout,
                   "The front door. One node from plate to camera and relief mesh.")


# ---------------------------------------------------------------------------
# 2 · the explicit chain
# ---------------------------------------------------------------------------

STAGES_NOTE = """SOLVE STAGES — the front door taken apart, so you can take over a stage.

atlas_input_quickstart_workflow.json produces this same result from one node. This graph is that node opened up: every stage is a node you can inspect, swap or override.

THE CHAIN
1. AtlasLearnedSolveFromImage — the camera. GeoCalib's learned prior gives focal length, pitch and roll from a single frame. This is a METRIC pinhole camera with a real focal length in mm, not a pose guess.
2. AtlasGravityCompass — the beat a plain depth solve gets wrong. On a steep down-angle (a birds-eye plate shows this best) the compass reports and can override pitch, roll and heading. Leave apply_override off to read the solve; turn it on to correct it.
3. AtlasDepthMap — monocular metric depth, shared downstream so every branch agrees on scale. Exterior plates take the V2 Metric Outdoor model; interiors take MoGe.
4. AtlasDeriveReliefMesh — depth becomes a triangulated mesh, torn at silhouettes, with the camera projection baked into the UVs. The tears are deliberate: a tear is honest missing information, and the fix is a layer, never a raised threshold.
5. AtlasBlockoutViewport — the recovered camera and the projection, live.

WHAT TO OVERRIDE FIRST
• height_mode / camera_height_m on the solve. The solver assumes a standing eye height by default; type a real camera height and the whole scene rescales. The solver assumes, you correct.
• focal_length_mm, if you know the lens. A measured value always beats a learned one.
• depth_model, if the plate is interior — MoGe reads rooms far better than the outdoor metric model.
• relief_grid and depth_edge_rel on the mesh, to trade density against silhouette fidelity.

RESOLUTION
The solver is resolution-independent and has been validated past 8k. The viewport renders up to 8k."""


def build_stages(object_info: dict, layout) -> dict:
    slug = "atlas_quickstart_solve_project_export_workflow"
    graph = Graph(object_info)

    load = graph.node("LoadImage", title="1 · SOURCE PLATE", values={
        "image": "example.png", "image_upload": "image",
    }, size=(360, 310))
    solve = graph.node("AtlasLearnedSolveFromImage", title="2 · CAMERA · learned metric prior", values={
        "height_mode": "assume",
        "camera_height_m": 1.6,
        "depth_model": OUTDOOR_DEPTH,
        "sensor_width_mm": 36.0,
        "device": "auto",
        "focal_length_mm": 0.0,
    }, size=(440, 340))
    compass = graph.node("AtlasGravityCompass", title="3 · GRAVITY COMPASS · read, or override", values={
        "apply_override": False,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "heading_override": False,
        "heading_deg": 0.0,
    }, size=(430, 250))
    depth = graph.node("AtlasDepthMap", title="4 · DEPTH · shared metric scale", values={
        "depth_model": OUTDOOR_DEPTH,
        "device": "auto",
    }, size=(440, 300))
    relief = graph.node("AtlasDeriveReliefMesh", title="5 · RELIEF MESH · torn at silhouettes", values={
        "relief_grid": 256,
        "relief_quality": "custom",
        "depth_edge_rel": 0.5,
        "max_edge_factor": 12.0,
        "sky_heuristic": False,
        "normal_edge_deg": 0.0,
        "quad_coherence": True,
    }, size=(430, 320))
    retopo = graph.node("AtlasRetopologizeLayer", title="ROUND THE SILHOUETTE · boundary smooth ×8 🔷", values={
        "layer": "",
        "method": "off",
        "target_vertex_count": 2000,
        "smooth_iterations": 0,
        "crease_angle": 30.0,
        "pure_quad": False,
        "boundary_smooth_iterations": 8,
        "rebuild_transition_ribbon": True,
        "merge_volume_primitives": False,
    }, size=(430, 340))
    gate = graph.node("AtlasSceneHealthGate", values={
        "pass_through_on_pass": True,
        "proceed": False,
        "approved_for": "",
    }, size=(430, 300))
    controls = _controls(graph, vfx=False)
    viewport = graph.node("AtlasBlockoutViewport", title="6 · VIEWPORT · identical result, every stage visible", values={
        "resolution": 1280,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    solve_json = graph.node("AtlasExportSolveJSON", title="Solve JSON · portable camera", values={
        "output_path": f"atlas_exports/{slug}/atlas_solve.json",
    }, size=(390, 150))
    note = _note(graph, STAGES_NOTE, "READ ME · which stage to take over first")

    graph.connect(load, "IMAGE", solve, "image")
    graph.connect(solve, "solve", compass, "solve")
    graph.connect(load, "IMAGE", depth, "image")
    graph.connect(compass, "solve", depth, "solve")
    graph.connect(compass, "solve", relief, "solve")
    graph.connect(depth, "depth", relief, "depth")
    graph.connect(relief, "solve", retopo, "solve")
    graph.connect(retopo, "solve", viewport, "solve")
    graph.connect(load, "IMAGE", viewport, "source_image")
    graph.connect(depth, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")
    # Gate 4 sits between the scene and the exporter, as it does everywhere.
    graph.connect(retopo, "solve", gate, "solve")
    graph.connect(load, "IMAGE", gate, "source_image")
    graph.connect(gate, "solve", solve_json, "solve")

    groups = [
        ("1 · THE EXPLICIT CHAIN · camera → compass → depth → mesh", "#35536b",
         [load, solve, compass, depth, relief, retopo, controls]),
        ("2 · RESULT · same as the front door, now overridable", "#375c4a",
         [gate, viewport, note, solve_json]),
    ]
    return _finish(graph, slug, groups, layout,
                   "The front door taken apart: every solve stage exposed and overridable.")


# ---------------------------------------------------------------------------
# 3 · the DCC payoff
# ---------------------------------------------------------------------------

FANOUT_NOTE = """EXPORT FAN-OUT — one photograph, eight native DCC deliverables, one project tree.

AtlasProject is the delivery desk. Name the project and shot, pick a colour lane, and every export node wired to its project output writes into <project_root>/<project>/<shot>/ instead of its own loose output_dir. Unwire the project link and each node falls back to its own path exactly as before, so an existing graph never changes behaviour by accident.

THE ONE CHOICE: COLOUR LANE
• Standard (sRGB) — photographers and designers. 8/16-bit, sRGB texture space.
• VFX (ACEScg / float) — the managed float path. ACEScg working space, float EXR plates, colour metadata carried into Nuke, Maya, Blender and USD.
This graph ships on the VFX lane because that is the lane the DCC exports are for. A camera RAW forces the choice, because the file itself cannot tell you which one it is.

WHAT LANDS WHERE
• Nuke — a .nk with the projection camera grouped with the geometry, plus a Python build script.
• Maya — a .ma review scene, camera and relief mesh, ready to look through.
• Blender — a build script; run it and the scene assembles.
• USD — the recovered camera on a stage, axis set at export time.
• Relief mesh — OBJ+MTL and a self-contained GLB, the projection baked into the UVs.
• Review package — the shareable summary bundle.
• Camera path USD — the viewport's baked move, once you have made one.
• Solve JSON — the portable contract every other tool reads.

STARTING FROM A CAMERA RAW
No RAW ships in the repo — one file would be twice the size of the whole tracked tree — so this graph starts from a jpg. To run the real colour path, drop in AtlasLoadRAW 📷 (NEF/CR2/CR3/RAF/ARW) ahead of the solve and make three connections:
• AtlasLoadRAW.image → AtlasInput.image — the developed, undistorted plate.
• AtlasLoadRAW.raw_meta → AtlasInput.raw_meta — EXIF focal plus the sensor width looked up from the camera model. This is the only MEASURED intrinsic in the graph, and a measured focal always beats a learned one; wire it and the solver stops guessing.
• AtlasLoadRAW.plate_ref → AtlasAttachSourcePlate, if you want the float plate tracked by reference into the exports.
Set the project's colour lane to VFX (ACEScg / float) — it already is here — and AtlasLoadRAW writes an undistorted ACEScg EXR alongside the solve.

THE CAMERA PATH EXPORT SHIPS MUTED
AtlasExportCameraPathUSD is the one muted node in this graph, and deliberately so: it raises rather than no-ops when no path has been baked, which is the state every first queue is in. To use it — queue once, open the viewport, add keyframes with 🎥 Camera Path, click ⏺ Bake Proxy Path, then unmute the node (Ctrl+M) and queue again. The move leaves the viewport as USD.

RETOPOLOGY IS EXPORT-ONLY
The retopo widgets on the export nodes never touch the live projection mesh. Leave them off for projection fidelity. Live retopology happens only through AtlasRetopologizeLayer, which regenerates the projective UVs. Quad output additionally needs pyinstantmeshes."""


def build_fanout(object_info: dict, layout) -> dict:
    slug = "atlas_export_fanout_workflow"
    graph = Graph(object_info)

    load = graph.node("LoadImage", title="1 · SOURCE PLATE", values={
        "image": "example.png", "image_upload": "image",
    }, size=(360, 310))
    atlas_input = graph.node("AtlasInput", title="2 · SOLVE + RELIEF", values={
        "layers": 0,
        "mesh": "relief",
        "mesh_resolution": 512,
        "use_vlm": False,
        "sky": False,
        "inpaint": False,
        "edge_extend_px": 24,
        "max_edge_factor": 12.0,
        "sky_heuristic": False,
        "normal_edge_deg": 0.0,
        "depth_model": OUTDOOR_DEPTH,
        "vlm_scope": True,
    }, size=(440, 620))
    project = graph.node("AtlasProject", title="3 · DELIVERY PROJECT · routes every export", values={
        "project": "atlas_demo",
        "shot": "shot010",
        "colour_mode": "VFX (ACEScg / float)",
        "project_root": "",
        "create_tree": True,
    }, size=(430, 240))
    controls = _controls(graph, vfx=True)
    viewport = graph.node("AtlasBlockoutViewport", title="4 · VIEWPORT · make a move, then re-queue", values={
        "resolution": 1280,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    note = _note(graph, FANOUT_NOTE, "READ ME · the project tree and the colour lane", size=(880, 620))

    solve_json = graph.node("AtlasExportSolveJSON", title="Solve JSON · the portable contract", values={
        "output_path": "atlas_solve.json",
    }, size=(390, 150))
    relief = graph.node("AtlasExportReliefMesh", title="Relief OBJ + GLB · projection baked into UVs", values={
        "output_dir": "relief",
        "grid_long_edge": 128,
        "depth_edge_rel": 0.5,
        "depth_model": OUTDOOR_DEPTH,
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
    }, size=(440, 620))
    nuke = graph.node("AtlasExportNuke", title="Nuke · .nk + build script", values={
        "output_dir": "nuke",
    }, size=(430, 180))
    maya = graph.node("AtlasExportMayaReviewScene", title="Maya · .ma review scene", values={
        "output_dir": "maya",
    }, size=(430, 180))
    blender = graph.node("AtlasExportBlender", title="Blender · build script", values={
        "output_dir": "blender",
    }, size=(430, 160))
    usd = graph.node("AtlasExportUSD", title="USD · recovered camera on a stage", values={
        "output_dir": "usd",
    }, size=(430, 140))
    review = graph.node("AtlasExportReviewPackage", title="Review package · shareable bundle", values={
        "output_dir": "review",
    }, size=(430, 140))
    # MUTED (mode 2) on purpose, and this is the only node in the shipping set
    # that is. AtlasExportCameraPathUSD raises rather than no-ops when no path
    # has been baked — "No camera path yet" — which is the state EVERY first
    # queue is in, so leaving it live would make the flagship fan-out red on
    # open. Muted, it is visible and wired and one right-click from working,
    # which is the point of showing it at all. Unmute after baking a path.
    campath = graph.node("AtlasExportCameraPathUSD",
                         title="Camera path USD · MUTED until you bake a path",
                         values={"output_dir": "camera_path"},
                         size=(430, 160), mode=2)

    graph.connect(load, "IMAGE", atlas_input, "image")
    graph.connect(atlas_input, "solve", viewport, "solve")
    graph.connect(atlas_input, "image", viewport, "source_image")
    graph.connect(atlas_input, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")

    exporters = (solve_json, relief, nuke, maya, blender, usd, review, campath)
    for exporter in exporters:
        graph.connect(atlas_input, "solve", exporter, "solve")
        graph.connect(project, "project", exporter, "project")
    graph.connect(atlas_input, "image", relief, "image")
    graph.connect(relief, "obj_path", nuke, "relief_mesh_obj_path")
    graph.connect(relief, "obj_path", maya, "relief_mesh_obj_path")
    graph.connect(viewport, "camera_path", campath, "camera_path")
    for exporter in (nuke, maya, blender):
        graph.connect(controls, "output_profile", exporter, "output_profile")

    groups = [
        ("1 · SOLVE + DELIVERY PROJECT · read the note", "#35536b",
         [load, atlas_input, controls, project, viewport, note]),
        ("2 · EIGHT NATIVE DELIVERABLES · every path routed by the project", "#375c4a",
         list(exporters)),
    ]
    return _finish(graph, slug, groups, layout,
                   "One solve into eight DCC exports, all routed by AtlasProject.")


# ---------------------------------------------------------------------------
# 4 · the 2.5D stack
# ---------------------------------------------------------------------------

LAYERED_NOTE = """LAYERED PROJECTION — one plate becomes an ordered, controllable 2.5D stack.

This is the workflow a matte painter opens. A single relief mesh is one surface; a layered stack is separate surfaces at separate depths, each with its own plate, its own matte and its own priority, which is what gives you real parallax on a move and a layer you can paint on in isolation.

THE STACK, FAR TO NEAR
1. AtlasDepthBandSplit — the near/far cut, as a fraction of the depth range. Everything downstream addresses a band by name rather than by hand-typed metres.
2. AtlasDepthLayerMask — that band as an actual matte, feathered, with its occlusion and hole masks.
3. AtlasDeriveReliefMesh — the base surface. Its tears are load-bearing information about what the camera could not see.
4. AtlasCleanPlateLayer — the mid band as its own layer, carrying its own plate. This is the layer you paint on, and the one that holds parallax when the camera moves.
5. AtlasSkyDomeLayer — the far field on a dome behind everything. Priority is negative so it sits farthest back: band priorities are FARTHEST-highest.
6. AtlasOcclusionGraph — works out which layer hides which, so the stack composites in the right order.
7. AtlasLayerPlan — the readable summary of the finished stack. Read its report before exporting.

SEAM DOCTRINE
Edge-extend smear lives on the layers BEHIND. The frontmost band keeps a clean cut. If a seam shows, raise edge_extend_px on the layer behind it, never on the one in front.

WHY THIS GRAPH USES AtlasHorizonMask FOR THE SKY
AtlasSkyDomeLayer requires a sky mask. The better source is a segmenter, but AtlasSAM3Mask needs the [sam3] extra and a Hugging Face token for a gated repo. The horizon mask is pure solve geometry and needs nothing, so this example runs on a fresh clone. On a real plate, swap in AtlasSAM3Mask with the prompt "sky", or AtlasInput's own sky output, and the dome will follow the actual skyline instead of the horizon line.

AtlasBoundedBand is absent for the same reason: its foreground_mask is required and only a segmenter produces one. Add it once you have segmentation, and it will bound the band to a real object instead of a depth percentile.

WHAT TO TUNE FIRST
• split on the band split — the single most useful dial in this graph.
• near_pct / far_pct on the clean plate layer, to widen or narrow the mid band.
• radius_m on the sky dome, if the horizon parallaxes wrongly on a move.
• feather_px on the layer mask, if a matte edge reads hard."""


def build_layered(object_info: dict, layout) -> dict:
    slug = "atlas_layered_projection_workflow"
    graph = Graph(object_info)

    load = graph.node("LoadImage", title="1 · SOURCE PLATE", values={
        "image": "example.png", "image_upload": "image",
    }, size=(360, 310))
    solve = graph.node("AtlasLearnedSolveFromImage", title="2 · CAMERA", values={
        "height_mode": "assume",
        "camera_height_m": 1.6,
        "depth_model": OUTDOOR_DEPTH,
        "sensor_width_mm": 36.0,
        "device": "auto",
        "focal_length_mm": 0.0,
    }, size=(440, 340))
    depth = graph.node("AtlasDepthMap", title="3 · DEPTH · shared by every layer", values={
        "depth_model": OUTDOOR_DEPTH,
        "device": "auto",
    }, size=(440, 300))
    band = graph.node("AtlasDepthBandSplit", title="4 · BAND SPLIT · the near/far cut", values={
        "split": 0.55,
        "split_m": 0.0,
    }, size=(400, 140))
    layer_mask = graph.node("AtlasDepthLayerMask", title="5 · LAYER MATTE · the band, as a matte", values={
        "near_pct": 0.0,
        "far_pct": 0.55,
        "feather_px": 4,
        "compute_hole_mask": True,
        "relief_grid": 384,
        "depth_edge_rel": 1.5,
        "fill_occluded": False,
        "band_side": "foreground",
        "quad_coherence": True,
    }, size=(430, 460))
    horizon = graph.node("AtlasHorizonMask", title="SKY MASK · pure geometry, zero deps", values={
        "image_width": 1024,
        "image_height": 1024,
        "feather_px": 8,
    }, size=(400, 180))
    relief = graph.node("AtlasDeriveReliefMesh", title="6 · BASE RELIEF · tears are information", values={
        "relief_grid": 256,
        "relief_quality": "custom",
        "depth_edge_rel": 0.5,
        "max_edge_factor": 12.0,
        "sky_heuristic": False,
        "normal_edge_deg": 0.0,
        "quad_coherence": True,
    }, size=(430, 320))
    cleanplate = graph.node("AtlasCleanPlateLayer", title="7 · CLEAN PLATE LAYER · the mid band you paint on", values={
        "near_pct": 0.0,
        "far_pct": 0.55,
        "name": "midground",
        "priority": 0.0,
        "relief_grid": 192,
        "depth_edge_rel": 0.75,
        "fill_occluded": False,
        "embed_matte": True,
        "edge_extend_px": 32,
        "band_side": "foreground",
        "band_geometry": "relief",
    }, size=(440, 700))
    skydome = graph.node("AtlasSkyDomeLayer", title="8 · SKY DOME · farthest, priority -10", values={
        "radius_m": 300.0,
        "relief_grid": 96,
        "name": "sky",
        "priority": -10.0,
        "edge_extend_px": 48,
        "frame_outpaint_px": 64,
        "distance_m": 0.0,
    }, size=(430, 380))
    # layer="*" — every layer in the stack, not just the primary. The
    # layered graph is the one place a stack exists to retopologize.
    retopo = graph.node("AtlasRetopologizeLayer", title="ROUND THE SILHOUETTE · boundary smooth ×8 🔷", values={
        "layer": "*",
        "method": "off",
        "target_vertex_count": 2000,
        "smooth_iterations": 0,
        "crease_angle": 30.0,
        "pure_quad": False,
        "boundary_smooth_iterations": 8,
        "rebuild_transition_ribbon": True,
        "merge_volume_primitives": False,
    }, size=(430, 340))
    occlusion = graph.node("AtlasOcclusionGraph", title="9 · OCCLUSION GRAPH · who hides whom", size=(400, 110))
    plan = graph.node("AtlasLayerPlan", title="10 · LAYER PLAN · read this before exporting", values={
        "include_unoccluded": False,
    }, size=(430, 160))
    layer_preview = graph.node("AtlasLayerPreview", title="LAYER PREVIEW · the matte over the plate", values={
        "layer_index": 0,
        "color_hex": "",
    }, size=(400, 300))
    # AtlasLayerPreview emits an IMAGE and nothing consumed it, so the
    # preview never rendered. A sink is not decoration here.
    layer_preview_sink = graph.node(
        "PreviewImage",
        title="LAYER PREVIEW — without a sink this never rendered 🖼",
        size=(320, 300))
    controls = _controls(graph, vfx=True)
    viewport = graph.node("AtlasBlockoutViewport", title="11 · VIEWPORT · orbit for the parallax", values={
        "resolution": 1280,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    solve_json = graph.node("AtlasExportSolveJSON", title="Solve JSON · the whole stack", values={
        "output_path": f"atlas_exports/{slug}/atlas_solve.json",
    }, size=(390, 150))
    nuke_layers = graph.node("AtlasExportNukeLayers", title="Nuke Layers · per-band cameras, plates, mattes", values={
        "output_dir": f"atlas_exports/{slug}/nuke_layers",
        "retopo_method": "off",
    }, size=(430, 300))
    note = _note(graph, LAYERED_NOTE, "READ ME · the stack, the seam doctrine, what to tune",
                 size=(880, 660))

    graph.connect(load, "IMAGE", solve, "image")
    graph.connect(load, "IMAGE", depth, "image")
    graph.connect(solve, "solve", depth, "solve")
    graph.connect(solve, "solve", horizon, "solve")

    graph.connect(solve, "solve", layer_mask, "solve")
    graph.connect(depth, "depth", layer_mask, "depth")
    graph.connect(band, "band_split", layer_mask, "band_split")

    graph.connect(solve, "solve", relief, "solve")
    graph.connect(depth, "depth", relief, "depth")

    graph.connect(relief, "solve", cleanplate, "solve")
    graph.connect(depth, "depth", cleanplate, "depth")
    graph.connect(load, "IMAGE", cleanplate, "plate_image")
    graph.connect(layer_mask, "layer_mask", cleanplate, "layer_matte")
    graph.connect(band, "band_split", cleanplate, "band_split")

    graph.connect(cleanplate, "solve", skydome, "solve")
    graph.connect(depth, "depth", skydome, "depth")
    graph.connect(horizon, "MASK", skydome, "sky_mask")
    graph.connect(load, "IMAGE", skydome, "plate_image")

    graph.connect(skydome, "solve", retopo, "solve")
    graph.connect(retopo, "solve", occlusion, "solve")
    graph.connect(depth, "depth", occlusion, "depth")
    graph.connect(occlusion, "solve", plan, "solve")

    graph.connect(load, "IMAGE", layer_preview, "image")
    graph.connect(layer_mask, "layer_mask", layer_preview, "mask")
    graph.connect(layer_preview, "image", layer_preview_sink, "images")

    graph.connect(plan, "solve", viewport, "solve")
    graph.connect(load, "IMAGE", viewport, "source_image")
    graph.connect(depth, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")

    graph.connect(plan, "solve", solve_json, "solve")
    graph.connect(plan, "solve", nuke_layers, "solve")
    graph.connect(controls, "output_profile", nuke_layers, "output_profile")

    groups = [
        ("1 · BUILD THE STACK · far to near", "#35536b",
         [load, solve, depth, band, horizon, layer_mask, relief, cleanplate,
          skydome, retopo, occlusion, plan, controls]),
        ("2 · INSPECT + DELIVER", "#375c4a",
         [layer_preview, layer_preview_sink, viewport, note, solve_json,
          nuke_layers]),
    ]
    return _finish(graph, slug, groups, layout,
                   "One plate as an ordered 2.5D layer stack: band, clean plate, sky dome.")



# ---------------------------------------------------------------------------
# 5 · HERO 02 — one photograph becomes an editable DCC scene
# ---------------------------------------------------------------------------

HERO02_NOTE = """HERO 02 — ONE PHOTOGRAPH BECOMES AN EDITABLE 3D SCENE.

This is the Atlas model in one graph: a photograph goes in, a camera-aware
projection scene comes out, and it lands in Blender for you to keep working on.

WHAT THIS IS NOT
Atlas does not reconstruct a perfect hidden world from one photograph, and this
graph does not pretend to. What you get is a USABLE EDITABLE STARTING SCENE:
the camera is recovered, the visible surfaces carry the photograph projected
through it, and everything the camera could not see is yours to build. Geometry
cleanup, depth adjustment, occlusion repair and modelling in the DCC are the
expected next step, not evidence that something failed.

THE STAGES
1 · PHOTOGRAPH — starts on ComfyUI's bundled example.png so this runs with
    nothing to download. Swap in your own photo for a real result.
2 · CAMERA — the learned prior (GeoCalib) reads focal length and gravity from
    image content. Robust on ordinary photographs and on AI imagery, and it
    reports honest confidence.
3 · DEPTH — one shared metric depth estimate. Every stage downstream reads THIS
    map, so they cannot disagree about scale.
4 · PROJECTION GEOMETRY — a relief mesh that follows the depth and TEARS at
    silhouettes. The tears are deliberate: a surface that stretched across them
    would smear the plate.
5 · SCENE HEALTH — the acknowledgement gate. It runs the same red-flag engine
    the debug report uses and holds the solve until you accept what it says.
6 · VIEWPORT — look through the recovered camera, then orbit off it. From the
    recovered viewpoint the plate reassembles exactly; the further you move,
    the more you see what the photograph never recorded.
7 · BLENDER — the geometry as OBJ+MTL with the projection baked into the UVs,
    plus a build_scene.py that assembles camera and mesh. Run the script in
    Blender and continue there.

MEASURED vs INFERRED — worth knowing before you trust a number
• INFERRED: focal length, gravity and camera height all come from a learned
  prior reading one image. There is no measurement in a single photograph.
• INFERRED: depth, and therefore the shape of the relief mesh.
• MEASURED: nothing here, and that is the honest answer for one photograph.
  If you shoot the plate yourself, AtlasLoadRAW gives EXIF focal and a sensor
  lookup, which IS measured — and a measured focal always beats a learned one.
  Scale references (AtlasReferenceScaleSolve) turn an assumed camera height
  into a measured one.
Scene health reports which tier the scale came from. Read it.

KNOWN LIMITATIONS
• One photograph cannot see behind anything. Occluded regions arrive as tears
  or as backdrop, never as recovered geometry.
• Camera height defaults to an assumed 1.6 m. On an elevated or wide vista that
  is often badly wrong — scene health will say so.
• The viewport render is a browser preview, not a final render.
• IN BLENDER, IMPORT THE OBJ FOR TEXTURE. build_scene.py builds the camera, the
  relief mesh and a projection material with correct UVs, but it can only bind
  the plate image when the solve carries a file-backed plate reference — this
  graph feeds the image as a tensor from LoadImage, so that material lands
  UNTEXTURED (grey), which is not an error. atlas_relief_mesh.obj beside it
  carries its own .mtl and diffuse PNG and imports fully textured. Verified in
  Blender 2026-08-17: script → camera + 122k-poly mesh; OBJ → same mesh with
  the 768px projection bound. Add AtlasRegisterPlate → AtlasAttachSourcePlate
  ahead of the solve if you want the script itself textured.

WHERE TO GO NEXT
• atlas_export_fanout_workflow.json — the same solve into Nuke, Maya, USD and a
  review package instead of one DCC.
• atlas_layered_projection_workflow.json — the 2.5D layer stack, when one mesh
  is not enough and you need clean plates behind the foreground."""


def build_hero02(object_info: dict, layout) -> dict:
    """Hero 02: photograph in, editable Blender scene out.

    Deliberately ONE DCC. The fan-out workflow already answers "what do I get
    in every DCC"; this one answers "what does Atlas actually do", and a second
    exporter would only dilute that. Blender because it is free, so a reviewer
    can follow the whole story without a licence.

    The gate sits between the geometry and the exporters because that is its
    documented job — an acknowledgement before anything leaves for a DCC, not a
    checkpoint in the middle of the solve.
    """
    slug = "atlas_hero_02_photo_to_editable_scene_workflow"
    graph = Graph(object_info)

    load = graph.node("LoadImage", title="1 · YOUR PHOTOGRAPH", values={
        "image": "example.png", "image_upload": "image",
    }, size=(360, 310))
    solve = graph.node("AtlasLearnedSolveFromImage", title="2 · CAMERA · learned prior", values={
        "height_mode": "assume",
        "camera_height_m": 1.6,
        "depth_model": OUTDOOR_DEPTH,
        "sensor_width_mm": 36.0,
        "device": "auto",
        "focal_length_mm": 0.0,
    }, size=(440, 340))
    depth = graph.node("AtlasDepthMap", title="3 · DEPTH · one shared metric scale", values={
        "depth_model": OUTDOOR_DEPTH,
        "device": "auto",
    }, size=(440, 300))
    relief = graph.node("AtlasDeriveReliefMesh", title="4 · PROJECTION GEOMETRY · torn at silhouettes", values={
        "relief_grid": 256,
        "relief_quality": "custom",
        "depth_edge_rel": 0.5,
        "sky_heuristic": False,
    }, size=(430, 320))
    gate = graph.node("AtlasSceneHealthGate", title="5 · SCENE HEALTH · acknowledge before export", values={
        "pass_through_on_pass": True,
    }, size=(430, 300))
    controls = _controls(graph, vfx=False)
    viewport = graph.node("AtlasBlockoutViewport", title="6 · VIEWPORT · look through the recovered camera", values={
        "resolution": 1280,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    mesh = graph.node("AtlasExportReliefMesh", title="7a · GEOMETRY · OBJ+MTL, projection baked into the UVs", values={
        "output_dir": f"atlas_exports/{slug}",
        "format": "obj",
    }, size=(430, 320))
    blender = graph.node("AtlasExportBlender", title="7b · BLENDER · run build_scene.py and keep working", values={
        "output_dir": f"atlas_exports/{slug}",
    }, size=(400, 160))
    note = _note(graph, HERO02_NOTE, "READ ME · what this is, and what it is not")

    graph.connect(load, "IMAGE", solve, "image")
    graph.connect(load, "IMAGE", depth, "image")
    graph.connect(solve, "solve", depth, "solve")
    graph.connect(solve, "solve", relief, "solve")
    graph.connect(depth, "depth", relief, "depth")
    # the gate stands between the geometry and everything that leaves for a DCC
    graph.connect(relief, "solve", gate, "solve")
    graph.connect(load, "IMAGE", gate, "source_image")
    graph.connect(depth, "depth", gate, "depth")
    graph.connect(gate, "solve", viewport, "solve")
    graph.connect(load, "IMAGE", viewport, "source_image")
    graph.connect(depth, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")
    graph.connect(gate, "solve", mesh, "solve")
    graph.connect(load, "IMAGE", mesh, "image")
    graph.connect(gate, "solve", blender, "solve")

    groups = [
        ("1 · PHOTOGRAPH → CAMERA → GEOMETRY", "#35536b",
         [load, solve, depth, relief, controls]),
        ("2 · CHECK IT", "#6b5a35", [gate, viewport, note]),
        ("3 · HAND IT TO BLENDER · this is where the artist takes over", "#375c4a",
         [mesh, blender]),
    ]
    return _finish(graph, slug, groups, layout,
                   "One photograph becomes a camera-aware editable scene, handed to Blender.")


SCENE_PACKAGE_NOTE = """PHOTOGRAPH -> .atlas SCENE PACKAGE — the format Atlas Scene opens.

Every other export hands one DCC a camera and the conversation ends there. This hands over the whole
scene, and it is meant to come BACK edited — so the package carries what a REVIEWER needs to judge it
(which plane licensed a construction, what the fit was worth, what the solve said about its own scale),
not only what a renderer needs to draw it.

THREE INPUTS, AND WHY EACH IS WIRED

1. AtlasOcclusionGraph is not decoration. Plane classification is READ from the occlusion graph, which
   already decides it. Without this node upstream every plane arrives unclassified and therefore
   licenses completion_policy "none" — the editor will refuse to extrude on any of them, which is the
   correct outcome for a surface nobody has vouched for, and a baffling one if you did not know why.
   This is the single most common reason a package opens inert.

2. AtlasExportReliefMesh writes GLB and its glb_path feeds relief_mesh_path. GLB rather than OBJ because
   the package should be self-contained: the writer embeds its texture and needs no sidecar MTL. Without
   geometry the package has a camera and planes and nothing to edit.

3. plate_path is given EXPLICITLY, as a path relative to ComfyUI's working directory. It has to be:
   AtlasLearnedSolveFromImage solves from a temporary file and deletes it, so the solve's own image_path
   names something already gone by the time the exporter runs. Point this at your own plate when you
   swap the photograph.

The node COMPLAINS on its face for each of these that is missing, naming the input to connect. An
export that quietly drops the plate produces a scene whose dead features nobody can explain.

WHERE IT LANDS
AtlasProject routes the package into <project_root>/<project>/<shot>/scenes/<scene_id>.atlas — the
scenes lane, alongside plates/, solves/, nuke/, maya/ and the rest. Inside:
  scene.json   the document, validated against the format's own checklist BEFORE it is written
  atlas/       the solve, kept as produced
  imagery/     the plate
  geometry/    the relief mesh
  mattes/      per-layer mattes as FILES with digests, decoded out of the JSON
  history/     the ledger, opened by the export itself

SCALE
The package records the solve's scale verdict verbatim, including whether it is safe to export. A single
photograph has an inherent scale ambiguity; if this reads assumed_default, reach for AtlasScaleOverride
or a scale reference before anyone builds on it."""


def build_scene_package(object_info: dict, layout) -> dict:
    """Photograph in, `.atlas` package out — the scene Atlas Scene opens.

    The occlusion graph is the load-bearing node here and the one a hand-wired
    graph leaves out. `AtlasExportScenePackage` reads plane classification from
    it, and an unclassified plane licenses `none`, so a package built without it
    validates, opens, and does nothing — which is exactly what a real run
    produced before this workflow existed.
    """
    slug = "atlas_photo_to_atlas_scene_workflow"
    graph = Graph(object_info)

    project = graph.node("AtlasProject", title="0 \u00b7 DELIVERY PROJECT \u00b7 where everything lands", values={
        "project": "atlas_scene_demo",
        "shot": "sh010",
        "colour_mode": "Standard (sRGB)",
        # Empty means ComfyUI's own output folder, which is the one path that
        # resolves on any machine — a shipping workflow must carry no absolute
        # path (tests/test_shipping_workflow_paths.py).
        "project_root": "",
        "create_tree": True,
    }, size=(400, 220))
    load = graph.node("LoadImage", title="1 \u00b7 YOUR PHOTOGRAPH", values={
        "image": "example.png", "image_upload": "image",
    }, size=(360, 310))
    solve = graph.node("AtlasLearnedSolveFromImage", title="2 \u00b7 CAMERA \u00b7 learned prior", values={
        "height_mode": "assume",
        "camera_height_m": 1.6,
        "depth_model": OUTDOOR_DEPTH,
        "sensor_width_mm": 36.0,
        "device": "auto",
        "focal_length_mm": 0.0,
    }, size=(440, 340))
    depth = graph.node("AtlasDepthMap", title="3 \u00b7 DEPTH \u00b7 one shared metric scale", values={
        "depth_model": OUTDOOR_DEPTH,
        "device": "auto",
    }, size=(440, 300))
    relief = graph.node("AtlasDeriveReliefMesh", title="4 \u00b7 PROJECTION GEOMETRY \u00b7 torn at silhouettes", values={
        "relief_grid": 256,
        "relief_quality": "custom",
        "depth_edge_rel": 0.5,
        "sky_heuristic": False,
    }, size=(430, 320))
    occlusion = graph.node(
        "AtlasOcclusionGraph",
        title="5 \u00b7 OCCLUSION GRAPH \u00b7 this is what CLASSIFIES the planes",
        size=(430, 140))
    gate = graph.node("AtlasSceneHealthGate", title="6 \u00b7 SCENE HEALTH \u00b7 acknowledge before export", values={
        "pass_through_on_pass": True,
    }, size=(430, 300))
    controls = _controls(graph, vfx=False)
    viewport = graph.node("AtlasBlockoutViewport", title="7 \u00b7 VIEWPORT \u00b7 look through the recovered camera", values={
        "resolution": 1280,
        "client_data": "",
        "preview_expand": 1.0,
    }, size=(900, 680))
    mesh = graph.node("AtlasExportReliefMesh", title="8a \u00b7 GEOMETRY \u00b7 GLB, self-contained, texture embedded", values={
        "output_dir": "atlas_exports/" + slug,
        "format": "glb",
    }, size=(430, 320))
    package = graph.node("AtlasExportScenePackage", title="8b \u00b7 .atlas PACKAGE \u00b7 the whole scene, meant to come back", values={
        "output_dir": "atlas_exports/" + slug,
        "scene_id": "street_001",
        "plate_path": "input/example.png",
        "observation_id": "obs_001",
    }, size=(460, 240))
    note = _note(graph, SCENE_PACKAGE_NOTE, "READ ME \u00b7 why the occlusion graph is not optional")

    graph.connect(load, "IMAGE", solve, "image")
    graph.connect(load, "IMAGE", depth, "image")
    graph.connect(solve, "solve", depth, "solve")
    graph.connect(solve, "solve", relief, "solve")
    graph.connect(depth, "depth", relief, "depth")
    # Classification BEFORE the gate: the package reads its completion policies
    # from what this node decided, and it decides nothing without being run.
    graph.connect(relief, "solve", occlusion, "solve")
    graph.connect(depth, "depth", occlusion, "depth")
    graph.connect(occlusion, "solve", gate, "solve")
    graph.connect(load, "IMAGE", gate, "source_image")
    graph.connect(depth, "depth", gate, "depth")
    graph.connect(gate, "solve", viewport, "solve")
    graph.connect(load, "IMAGE", viewport, "source_image")
    graph.connect(depth, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")
    graph.connect(gate, "solve", mesh, "solve")
    graph.connect(load, "IMAGE", mesh, "image")
    graph.connect(project, "project", mesh, "project")
    graph.connect(gate, "solve", package, "solve")
    graph.connect(project, "project", package, "project")
    # The mesh the package carries is the one just written, not one it makes.
    graph.connect(mesh, "glb_path", package, "relief_mesh_path")

    groups = [
        ("1 \u00b7 PHOTOGRAPH \u2192 CAMERA \u2192 GEOMETRY \u2192 CLASSIFICATION", "#35536b",
         [project, load, solve, depth, relief, occlusion, controls]),
        ("2 \u00b7 CHECK IT", "#6b5a35", [gate, viewport, note]),
        ("3 \u00b7 WRITE THE SCENE \u00b7 this is what Atlas Scene opens", "#375c4a",
         [mesh, package]),
    ]
    return _finish(graph, slug, groups, layout,
                   "One photograph becomes a .atlas package: camera, classified planes, plate, mesh and ledger.")


BUILDERS = {
    "atlas_input_quickstart_workflow": build_quickstart,
    "atlas_quickstart_solve_project_export_workflow": build_stages,
    "atlas_export_fanout_workflow": build_fanout,
    "atlas_layered_projection_workflow": build_layered,
    "atlas_hero_02_photo_to_editable_scene_workflow": build_hero02,
    "atlas_photo_to_atlas_scene_workflow": build_scene_package,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--only", default="", help="build a single slug")
    parser.add_argument(
        "--check", action="store_true",
        help="compare each builder's output against the committed JSON and "
             "exit non-zero on drift, writing nothing")
    args = parser.parse_args()
    object_info = _fetch_object_info(args.host)
    layout = _load_layout_module()
    drifted: list[str] = []

    for slug, builder in BUILDERS.items():
        if args.only and args.only not in slug:
            continue
        workflow = builder(object_info, layout)
        output = EXAMPLES / f"{slug}.json"

        if args.check:
            # Node TYPES, not bytes. Positions and link ids move with the
            # layout pass and a byte comparison would cry drift on every run;
            # a node appearing or vanishing is the thing that actually matters.
            if not output.is_file():
                drifted.append(f"{slug}: no committed file")
                continue
            committed = json.loads(output.read_text(encoding="utf-8"))
            was = sorted(n["type"] for n in committed["nodes"])
            now = sorted(n["type"] for n in workflow["nodes"])
            if was != now:
                only_committed = sorted(set(was) - set(now))
                only_builder = sorted(set(now) - set(was))
                drifted.append(
                    f"{slug}: committed-only={only_committed} "
                    f"builder-only={only_builder}")
            else:
                print(f"ok    {slug}")
            continue

        output.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print(f"  {layout.inspect(workflow)['summary']}")

    if drifted:
        print("\nDRIFT — the builder and the committed workflow disagree:")
        for line in drifted:
            print("  " + line)
        print("\nRegenerating would CHANGE the shipped file. Fix the builder to "
              "match, or regenerate deliberately and review the diff.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
