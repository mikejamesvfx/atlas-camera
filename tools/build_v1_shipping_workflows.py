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
}

# Exterior plates: the metric outdoor model is the doctrine choice for these
# (docs/DESIGN_RULES.md, depth model doctrine).  Interiors would take MoGe.
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

1. Pick a plate in Load Image and queue once.
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
    graph.connect(viewport, "shaded", assess, "camera_view")
    graph.connect(atlas_input, "solve", assess, "solve")
    graph.connect(atlas_input, "image", assess, "source_image")
    graph.connect(atlas_input, "depth", assess, "depth")
    graph.connect(atlas_input, "solve", solve_json, "solve")

    groups = [
        ("1 · LOAD → SOLVE → VIEWPORT · read the note", "#35536b",
         [load, atlas_input, controls, viewport, note]),
        ("2 · INSPECT + EXPORT", "#375c4a", [layer_preview, solve_json, assess]),
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
2. AtlasGravityCompass — the beat a plain depth solve gets wrong. On a steep down-angle (try newyork_Birdseye.png) the compass reports and can override pitch, roll and heading. Leave apply_override off to read the solve; turn it on to correct it.
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

    load = graph.node("LoadImage", title="1 · SOURCE PLATE · GHOST TOWN", values={
        "image": "ghosttown.jpg", "image_upload": "image",
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
        "sky_heuristic": True,
        "normal_edge_deg": 0.0,
        "quad_coherence": True,
    }, size=(430, 320))
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
    graph.connect(solve, "ATLAS_SOLVE", compass, "solve")
    graph.connect(load, "IMAGE", depth, "image")
    graph.connect(compass, "solve", depth, "solve")
    graph.connect(compass, "solve", relief, "solve")
    graph.connect(depth, "depth", relief, "depth")
    graph.connect(relief, "solve", viewport, "solve")
    graph.connect(load, "IMAGE", viewport, "source_image")
    graph.connect(depth, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")
    graph.connect(relief, "solve", solve_json, "solve")

    groups = [
        ("1 · THE EXPLICIT CHAIN · camera → compass → depth → mesh", "#35536b",
         [load, solve, compass, depth, relief, controls]),
        ("2 · RESULT · same as the front door, now overridable", "#375c4a",
         [viewport, note, solve_json]),
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

THE CAMERA PATH EXPORT SHIPS MUTED
AtlasExportCameraPathUSD is the one muted node in this graph, and deliberately so: it raises rather than no-ops when no path has been baked, which is the state every first queue is in. To use it — queue once, open the viewport, add keyframes with 🎥 Camera Path, click ⏺ Bake Proxy Path, then unmute the node (Ctrl+M) and queue again. The move leaves the viewport as USD.

RETOPOLOGY IS EXPORT-ONLY
The retopo widgets on the export nodes never touch the live projection mesh. Leave them off for projection fidelity. Live retopology happens only through AtlasRetopologizeLayer, which regenerates the projective UVs. Quad output additionally needs pyinstantmeshes."""


def build_fanout(object_info: dict, layout) -> dict:
    slug = "atlas_export_fanout_workflow"
    graph = Graph(object_info)

    load = graph.node("LoadImage", title="1 · SOURCE PLATE · OCEAN CASTLE", values={
        "image": "oceancastle.jpg", "image_upload": "image",
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
        "sky_heuristic": True,
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

    load = graph.node("LoadImage", title="1 · SOURCE PLATE · GHOST TOWN", values={
        "image": "ghosttown.jpg", "image_upload": "image",
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
        "sky_heuristic": True,
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
    occlusion = graph.node("AtlasOcclusionGraph", title="9 · OCCLUSION GRAPH · who hides whom", size=(400, 110))
    plan = graph.node("AtlasLayerPlan", title="10 · LAYER PLAN · read this before exporting", values={
        "include_unoccluded": False,
    }, size=(430, 160))
    layer_preview = graph.node("AtlasLayerPreview", title="LAYER PREVIEW · the matte over the plate", values={
        "layer_index": 0,
        "color_hex": "",
    }, size=(400, 300))
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
    graph.connect(solve, "ATLAS_SOLVE", depth, "solve")
    graph.connect(solve, "ATLAS_SOLVE", horizon, "solve")

    graph.connect(solve, "ATLAS_SOLVE", layer_mask, "solve")
    graph.connect(depth, "depth", layer_mask, "depth")
    graph.connect(band, "band_split", layer_mask, "band_split")

    graph.connect(solve, "ATLAS_SOLVE", relief, "solve")
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

    graph.connect(skydome, "solve", occlusion, "solve")
    graph.connect(depth, "depth", occlusion, "depth")
    graph.connect(occlusion, "solve", plan, "solve")

    graph.connect(load, "IMAGE", layer_preview, "image")
    graph.connect(layer_mask, "layer_mask", layer_preview, "mask")

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
          skydome, occlusion, plan, controls]),
        ("2 · INSPECT + DELIVER", "#375c4a",
         [layer_preview, viewport, note, solve_json, nuke_layers]),
    ]
    return _finish(graph, slug, groups, layout,
                   "One plate as an ordered 2.5D layer stack: band, clean plate, sky dome.")


BUILDERS = {
    "atlas_input_quickstart_workflow": build_quickstart,
    "atlas_quickstart_solve_project_export_workflow": build_stages,
    "atlas_export_fanout_workflow": build_fanout,
    "atlas_layered_projection_workflow": build_layered,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--only", default="", help="build a single slug")
    args = parser.parse_args()
    object_info = _fetch_object_info(args.host)
    layout = _load_layout_module()
    for slug, builder in BUILDERS.items():
        if args.only and args.only not in slug:
            continue
        workflow = builder(object_info, layout)
        output = EXAMPLES / f"{slug}.json"
        output.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print(f"  {layout.inspect(workflow)['summary']}")


if __name__ == "__main__":
    main()
