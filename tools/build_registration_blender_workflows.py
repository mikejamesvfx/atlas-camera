"""Generate the two 2026-08-16 example workflows: the Qwen ROI + measured
patch-camera loop, and the Blender measured-primitives bridge.

Same Graph DSL as build_v1_shipping_workflows (redundant links + positional
widgets are generated, never typed), but the node table comes from the LOCAL
registry (`atlas_camera.comfy.node_registry`) rather than a live /object_info,
so a fresh checkout can regenerate these before the server has been restarted
on the new nodes. `--host` switches to the live server when you want the
frontend's own truth.

Usage::

    python tools/build_registration_blender_workflows.py
    python tools/build_registration_blender_workflows.py --host 127.0.0.1:8188
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from build_v1_shipping_workflows import (  # noqa: E402
    _controls, _group, _note, EXAMPLES, OUTDOOR_DEPTH,
)
from rebuild_staged_master_workflow import Graph, _fetch_object_info, _load_layout_module

WORKFLOW_IDS = {
    "atlas_qwen_roi_registered_patch_workflow": "6c1d5b1e-8f4a-5c2b-9d3e-2a7f0b4c9e11",
    "atlas_blender_measured_primitives_workflow": "9e2b7d40-3c6f-5a1d-8b47-5f1e2c9a0d22",
    "atlas_cleanplate_depth_layer_workflow": "4f7a1c2e-6b3d-5e8f-9a10-c2d4e6f8a0b3",
}

MOGE = "Ruicheng/moge-2-vitl-normal"


def local_object_info() -> dict:
    """Synthesize /object_info from the registry (standard + experimental)."""
    os.environ.setdefault("ATLAS_EXPERIMENTAL", "1")
    from atlas_camera.comfy import node_registry as reg
    table = {**reg.NODE_CLASS_MAPPINGS, **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS}
    oi = {}
    for name, cls in table.items():
        it = cls.INPUT_TYPES()
        oi[name] = {
            "input": {k: v for k, v in it.items() if k in ("required", "optional")},
            "input_order": {k: list(v) for k, v in it.items() if k in ("required", "optional")},
            "output": list(getattr(cls, "RETURN_TYPES", ())),
            "output_name": list(getattr(cls, "RETURN_NAMES", getattr(cls, "RETURN_TYPES", ()))),
        }
    # ComfyUI core LoadImage (image_upload combo + upload widget).
    oi["LoadImage"] = {
        "input": {"required": {"image": (["example.png"], {"image_upload": True})}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"],
    }
    return oi


def _finish(graph, slug, group_specs, layout, notes):
    wg = layout.auto_layout({"nodes": graph.nodes, "links": graph.links})
    check = layout.inspect(wg)
    if check["overlaps"]:
        raise RuntimeError(f"{slug} layout overlaps: {check['overlaps']}")
    groups = [_group(nodes, title, color) for title, color, nodes in group_specs]
    return {
        "id": WORKFLOW_IDS[slug], "revision": 1,
        "last_node_id": graph._node_id, "last_link_id": graph._link_id,
        "nodes": graph.nodes, "links": graph.links, "groups": groups, "config": {},
        "extra": {"ds": {"scale": 0.58, "offset": [35, 85]}, "frontendVersion": "1.25.11",
                  "workflowRendererVersion": "LG", "atlas_shipping_set": "v1",
                  "atlas_notes": notes},
        "version": 0.4,
    }


# ---------------------------------------------------------------------------
QWEN_NOTE = """QWEN ROI + MEASURED PATCH CAMERA — the novel-view loop that checks itself.

WHAT RUNS
1 · SOLVE: AtlasInput (layers=0, one relief mesh) on the plate. AtlasDepthMap makes the SHARED metric depth
   with moge_report_free_focal=true, so the 🩺 report can flag a focal mismatch between the solve and MoGe.
2 · WHERE IS THE HOLE: AtlasCameraMovePreset builds a short move; AtlasCropROI (roi_source=auto_largest)
   surveys the END frame's disocclusion holes and returns the largest cluster as a crop rect + generation raster.
   AtlasCropSourcePhoto cuts the PRISTINE photo at that rect (square, a little context) — what a subject-centric
   novel-view model wants instead of the whole plate.
3 · GENERATE (your model): feed `source_crop` to Qwen-Image-Edit-2511 + the Multiple-Angles LoRA
   (`<sks> [azimuth] [elevation] [distance]`, 96 absolute poses). The LoadImage placeholder stands in for the
   result so this graph loads and runs on a fresh clone; replace it with your generated novel view.
4 · REGISTER: AtlasAddPatchView camera_source=register_to_primary MEASURES the novel view's camera —
   MoGe pointmap on the patch + SIFT matches to the full plate → RANSAC similarity against the primary's metric
   depth. flip_azimuth is resolved automatically; inliers / RMS / deviation-from-declared are on the source
   metadata (registration_*). Below the gates it FALLS BACK to the declared orbit and says so.
5 · VIEW + QA: viewport (📷 Camera View / 📽 Project) — an automatic 1280 snapshot pair lands in
   output/atlas_viewport/ after every run — plus AtlasDebugReport for the numbers.

WHY A CROP: diffusion resolution is capped and the LoRA was trained on one clear subject; a whole 36 MP plate
spends the pixel budget everywhere but the hole. Registration against the FULL plate means the crop needs no
origin bookkeeping — the pixels say where it came from.

FIREWALL: generated pixels register TO the photographed world; the primary camera never moves and the patch stays
evidence_type=generated. The registered pose inherits primary_depth's scale (metric when that is; always consistent
with the geometry the patch projects onto)."""


def build_qwen(oi: dict, layout) -> dict:
    slug = "atlas_qwen_roi_registered_patch_workflow"
    g = Graph(oi)
    load = g.node("LoadImage", title="1 · SOURCE PLATE", values={"image": "example.png"}, size=(320, 320))
    atlas_input = g.node("AtlasInput", title="2 · SOLVE + RELIEF", values={
        "layers": 0, "depth_model": OUTDOOR_DEPTH, "sky_heuristic": False,
    }, size=(440, 620))
    depth = g.node("AtlasDepthMap", title="3 · SHARED DEPTH · MoGe + free-focal check", values={
        "depth_model": MOGE, "device": "auto", "moge_report_free_focal": True,
    }, size=(420, 330))
    move = g.node("AtlasCameraMovePreset", title="4 · THE MOVE · where holes open", values={
        "frames": 48,
    }, size=(400, 260))
    crop = g.node("AtlasCropROI", title="5 · CROP ROI · largest hole cluster", values={
        "roi_slot": 1, "roi_source": "auto_largest", "max_gen_long_edge": 1280,
    }, size=(420, 380))
    photo = g.node("AtlasCropSourcePhoto", title="6 · PHOTO CROP → Qwen input", values={
        "pad_frac": 0.25, "square": True,
    }, size=(380, 180))
    novel = g.node("LoadImage", title="7 · QWEN MULTIPLE-ANGLES RESULT · placeholder, replace",
                   values={"image": "example.png"}, size=(320, 320))
    patch = g.node("AtlasAddPatchView", title="8 · ADD PATCH · register_to_primary", values={
        "patch_azimuth_view": "front-right quarter view", "patch_elevation_view": "eye-level shot",
        "patch_distance": "medium shot", "source_azimuth_view": "front view",
        "source_elevation_view": "eye-level shot", "flip_azimuth": False, "name": "qwen_roi_patch",
        "depth_model": MOGE, "relief_grid": 96, "priority": 1.0, "device": "auto",
        "mask_unseen_only": True, "unseen_dilate_px": 16, "geometry_source": "reuse_scene",
        "camera_source": "register_to_primary", "registration_min_inliers": 40,
        "registration_max_residual_m": 0.35, "registration_max_deviation_deg": 25.0,
        "auto_flip_azimuth": True,
    }, size=(460, 640))
    controls = _controls(g, vfx=False)
    viewport = g.node("AtlasBlockoutViewport", title="9 · VIEWPORT · 📽 + auto snapshots", values={
        "resolution": 1024, "client_data": "", "preview_expand": 1.0,
    }, size=(900, 680))
    report = g.node("AtlasDebugReport", title="10 · 🩺 REPORT · registration_* + focal_mismatch", values={
        "file_path": f"atlas_debug/{slug}_debug.json",
    }, size=(420, 200))
    note = _note(g, QWEN_NOTE, "READ ME · Qwen ROI loop with a measured patch camera", size=(900, 620))

    g.connect(load, "IMAGE", atlas_input, "image")
    g.connect(load, "IMAGE", depth, "image")
    g.connect(atlas_input, "solve", depth, "solve")
    g.connect(atlas_input, "solve", move, "solve")
    g.connect(atlas_input, "solve", crop, "solve")
    g.connect(load, "IMAGE", crop, "source_image")
    g.connect(move, "camera_path", crop, "camera_path")
    g.connect(load, "IMAGE", photo, "source_image")
    g.connect(crop, "crop", photo, "crop")
    g.connect(atlas_input, "solve", patch, "solve")
    g.connect(novel, "IMAGE", patch, "patch_image")
    g.connect(depth, "depth", patch, "primary_depth")
    g.connect(load, "IMAGE", patch, "primary_image")
    g.connect(patch, "ATLAS_SOLVE", viewport, "solve")
    g.connect(load, "IMAGE", viewport, "source_image")
    g.connect(depth, "depth", viewport, "primary_depth")
    g.connect(controls, "controls", viewport, "controls")
    g.connect(controls, "output_profile", viewport, "output_profile")
    g.connect(patch, "ATLAS_SOLVE", report, "solve")
    g.connect(depth, "depth", report, "depth")
    groups = [
        ("SOLVE + SHARED DEPTH", "#3f789e", [load, atlas_input, depth]),
        ("FIND THE HOLE → PHOTO CROP", "#88A", [move, crop, photo]),
        ("GENERATE → REGISTER", "#8A8", [novel, patch]),
        ("VIEW + QA", "#b58b2a", [controls, viewport, report]),
    ]
    return _finish(g, slug, groups, layout, QWEN_NOTE)


# ---------------------------------------------------------------------------
BLENDER_NOTE = """BLENDER MEASURED PRIMITIVES — ground plane, footprints and facades assembled in the metric world.

NEEDS: a Blender >= 4.2 install (ATLAS_BLENDER_PATH, PATH, or the platform install dirs) and
ATLAS_EXPERIMENTAL=1 (AtlasBlenderMassing / AtlasBlenderImportMeshes are experimental-tier).

WHAT RUNS
1 · SOLVE: AtlasInput on the plate (a RAW plate through AtlasLoadRAW gives the MEASURED focal; the bundled
   example.png runs on the solved one).
2 · DRAW: in the first viewport use ✏️ Draw to outline building FOOTPRINTS on the ground and FACADES on walls,
   then ✅ Apply — they land on the solve as PROXY_ROLE polygons (viewport_polygon).
3 · MEASURE + MASSING: AtlasDepthMap (MoGe) supplies the metric POINTMAP. AtlasBlenderMassing measures the scene
   from it — sky excluded, ground plane + camera height + extents + dominant planes at MoGe's scale — and seeds a
   SMALL Blender scene (recovered camera, a sky-free point cloud to snap to, the measured planes, your drawn
   polygons; the relief mesh stays home). recipes/massing.py runs headless — ground plane at the MEASURED ground
   level, FLAT polygons extruded up by default_height_m, VERTICAL polygons thickened AWAY from the camera into
   wall_thickness_m slabs, massing boxes as volumes — saves scene.blend and brings the meshes back as PROXY_ROLE
   geometry with projective UVs regenerated for the recovered camera. ~2 s round trip. Appends, never clobbers.
4 · AGENT HANDOFF 🤝: AtlasAgentHandoff PAUSES the graph (blocking, timeout_s) and publishes a brief at
   output/atlas_agent/<node>/brief.json — task, scene.blend, viewport snapshots, measured numbers, allowed tools.
   An external agent (Claude Code via the atlas MCP `atlas_agent_brief`/`atlas_agent_resume`, Hermes, OpenClaw,
   or you with curl / a text editor) operates Blender (blender-mcp GUI or headless), models under `atlas_out`,
   saves, and resumes. auto_import then exports those meshes and appends them. docs/AGENT_HANDOFF.md.
   The bypassed AtlasBlenderImportMeshes remains for a manual GUI round-trip without an agent.
5 · PLATE LAYER 🎞: AtlasPlateLayer puts YOUR clean plate (replace the LoadImage placeholder) on the agent's
   surfaces (`geometry_filter=blender_import, agent_`) as a projection layer from the primary camera — the water and
   hillside behind the foreground now show the clean background when you orbit. Chain another for another plate.
6 · RETOPO + VIEW: AtlasRetopologizeLayer (method off by default — set quad/decimate to reduce), second viewport
   for 📽 Project, Nuke export.

WHY: the missing half of the Blender bridge was IMPORT — nothing read a mesh back into the projection scene.
Wire format is NPZ with NO UVs (Blender importers merge vertices; Atlas regenerates projective UVs on import).
Doctrine: numpy CLOSES, Blender PLACES/HOSTS; the MCP server never subprocesses — an agent drives this graph
through atlas_run_workflow. Rejection is a third outcome: a missing Blender or a failed recipe reports and passes
the solve through, never a raise."""


def build_blender(oi: dict, layout) -> dict:
    slug = "atlas_blender_measured_primitives_workflow"
    g = Graph(oi)
    load = g.node("LoadImage", title="1 · SOURCE PLATE", values={"image": "example.png"}, size=(320, 320))
    atlas_input = g.node("AtlasInput", title="2 · SOLVE + RELIEF", values={
        "layers": 0, "depth_model": OUTDOOR_DEPTH, "sky_heuristic": False,
    }, size=(440, 620))
    depth = g.node("AtlasDepthMap", title="2b · MoGe DEPTH · the metric POINTMAP the seed measures", values={
        "depth_model": MOGE, "device": "auto", "moge_report_free_focal": True,
    }, size=(420, 330))
    controls = _controls(g, vfx=False)
    draw_vp = g.node("AtlasBlockoutViewport", title="3 · DRAW FOOTPRINTS / FACADES · ✏️ then ✅ Apply", values={
        "resolution": 1024, "client_data": "", "preview_expand": 1.0,
    }, size=(900, 680))
    massing = g.node("AtlasBlenderMassing", title="4 · BLENDER MASSING 🧱 · headless", values={
        "blender_path": "", "exchange_dir": "atlas_exports/blender_massing",
        "default_height_m": 3.0, "wall_thickness_m": 0.3, "ground_extent_m": 60.0,
        "footprint_source": "both", "run_recipe": True, "save_blend": True,
        "min_y_m": -0.05, "timeout_s": 300,
        "cloud_max_points": 200000, "include_relief_reference": False,
    }, size=(460, 460))
    agent = g.node("AtlasAgentHandoff", title="4b · AGENT HANDOFF 🤝 · pause, brief, resume", values={
        "task": "Look at the snapshots and the measured scene. In Blender (scene.blend), model the "
                "building volumes and any ground/facade geometry the photo shows but the massing "
                "missed, under the atlas_out collection, in metres (ground is at ground_y_m). Save "
                "and resume with a short note.",
        "exchange_dir": "atlas_exports/blender_massing", "snapshot_node_id": "",
        "tools_allowed": "blender_mcp, blender_headless, atlas_mcp", "blender_path": "",
        "timeout_s": 1800, "on_timeout": "continue", "auto_import": True,
        "expect_fingerprint": True, "min_y_m": -0.05, "poll_s": 1.0,
    }, size=(480, 460))
    imp = g.node("AtlasBlenderImportMeshes", title="5 · IMPORT 📥 · GUI round-trip (blend_file) · bypassed",
                 values={
                     "exchange_dir": "atlas_exports/blender_massing",
                     "blend_file": "", "blender_path": "", "name_prefix": "blender",
                     "expect_fingerprint": True, "min_y_m": -0.05, "max_radius_m": 0.0,
                     "timeout_s": 300,
                 }, size=(460, 320), mode=4)
    clean = g.node("LoadImage", title="5b · CLEAN PLATE · placeholder, replace",
                   values={"image": "example.png"}, size=(320, 320))
    layer = g.node("AtlasPlateLayer", title="5c · PLATE LAYER 🎞 · clean plate on the agent's surfaces", values={
        "geometry_filter": "blender_import, agent_", "name": "clean_plate_layer",
        "priority": 5.0, "move_from_primary": True,
    }, size=(460, 300))
    retopo = g.node("AtlasRetopologizeLayer", title="6 · RETOPO · projective UVs", values={
        "layer": "", "method": "off", "target_vertex_count": 2000,
    }, size=(420, 330))
    view_vp = g.node("AtlasBlockoutViewport", title="7 · VIEWPORT · 📽 Project the measured primitives", values={
        "resolution": 1024, "client_data": "", "preview_expand": 1.0,
    }, size=(900, 680))
    nuke = g.node("AtlasExportNuke", title="8 · NUKE", values={
        "output_dir": f"atlas_exports/{slug}",
    }, size=(400, 200))
    note = _note(g, BLENDER_NOTE, "READ ME · Blender measured primitives", size=(900, 620))

    g.connect(load, "IMAGE", atlas_input, "image")
    g.connect(load, "IMAGE", depth, "image")
    g.connect(atlas_input, "solve", depth, "solve")
    g.connect(depth, "depth", massing, "depth")
    g.connect(depth, "depth", draw_vp, "primary_depth")
    g.connect(atlas_input, "solve", draw_vp, "solve")
    g.connect(load, "IMAGE", draw_vp, "source_image")
    g.connect(controls, "controls", draw_vp, "controls")
    g.connect(controls, "output_profile", draw_vp, "output_profile")
    g.connect(draw_vp, "solve", massing, "solve")
    g.connect(massing, "solve", agent, "solve")
    g.connect(massing, "exchange_dir", agent, "exchange_dir")
    g.connect(depth, "depth", agent, "depth")
    g.connect(agent, "solve", imp, "solve")
    g.connect(imp, "solve", layer, "solve")
    g.connect(clean, "IMAGE", layer, "plate_image")
    g.connect(layer, "solve", retopo, "solve")
    g.connect(retopo, "solve", view_vp, "solve")
    g.connect(load, "IMAGE", view_vp, "source_image")
    g.connect(controls, "output_profile", view_vp, "output_profile")
    g.connect(retopo, "solve", nuke, "solve")
    groups = [
        ("SOLVE + MoGe MEASUREMENT + DRAW", "#3f789e", [load, atlas_input, depth, controls, draw_vp]),
        ("BLENDER 🧱 · measured massing → AGENT 🤝 → import → PLATE LAYER 🎞", "#8A8", [massing, agent, imp, clean, layer]),
        ("RETOPO + VIEW + EXPORT", "#b58b2a", [retopo, view_vp, nuke]),
    ]
    return _finish(g, slug, groups, layout, BLENDER_NOTE)


# ---------------------------------------------------------------------------
CLEANPLATE_NOTE = """CLEAN PLATE GEOMETRY FROM ITS OWN DEPTH — one solve, two plates, no registration guesswork.

THE PROBLEM: the source photo has the machine; the clean plate does not. Solving both and merging never lines up
(the learned solve is not deterministic between two different images). THE ANSWER: never solve the clean plate.
Same viewpoint → same camera → REUSE the source solve. What the clean plate contributes is DEPTH: MoGe on the
clean plate sees the terrain BEHIND the machine (hillside to the water, the water, the far hill), which the
source depth cannot.

WHAT RUNS
1 · SOURCE: LoadImage (placeholder example.png — swap in atlasMachine_8k.jpg from examples/images/) → AtlasInput (layers=0, MoGe) →
   AtlasDepthMap (MoGe, solve wired) = the primary depth.
2 · CLEAN PLATE: LoadImage (placeholder — swap in atlasMachine_8k_CP.jpg) → AtlasDepthMap (MoGe, SAME solve wired) = the plate depth.
3 · OBJECT MASK (bypassed by default): AtlasSAM3Mask "machine" — the object's pixels are excluded from the scale
   registration (their depth differs between the plates by definition). Needs the [sam3] extra; without it the
   registration still works over the whole frame — the median is robust to a smaller object.
4 · AtlasCleanPlateLayer with plate_depth wired: the layer's mesh is built from the CLEAN plate's depth,
   scale-registered to the primary by ONE median ratio over the mutually visible pixels
   (core.hidden_geometry.register_layers_to_depth; the report/metadata carry plate_depth_scale + rel_mad),
   near_pct 0 / far_pct 1 (the whole plate), sky excluded via AtlasInput's sky_mask, projection_mode=clean_plate.
5 · VIEWPORT: orbit — behind the machine you now see the clean plate ON the clean plate's own terrain.

WHY IT IS DETERMINISTIC: no second solve. Camera identical by construction; the only free number is one scale,
measured, reported. For a clean plate shot from a DIFFERENT position use AtlasAddPatchView
camera_source=register_to_primary instead (the plate's camera is MEASURED against the primary's world)."""


def build_cleanplate(oi: dict, layout) -> dict:
    slug = "atlas_cleanplate_depth_layer_workflow"
    g = Graph(oi)
    # example.png placeholders (the shipping rule: the only plate a fresh install
    # has). Swap in atlasMachine_8k.jpg / _CP.jpg from examples/images/.
    src = g.node("LoadImage", title="1 · SOURCE PLATE (with the object) · placeholder", values={"image": "example.png"}, size=(320, 320))
    cp = g.node("LoadImage", title="2 · CLEAN PLATE (same camera, object removed) · placeholder", values={"image": "example.png"}, size=(320, 320))
    atlas_input = g.node("AtlasInput", title="1b · SOLVE (once — the clean plate reuses it)", values={
        "layers": 0, "depth_model": MOGE, "sky_heuristic": False,
    }, size=(440, 620))
    d_src = g.node("AtlasDepthMap", title="1c · PRIMARY DEPTH · MoGe", values={
        "depth_model": MOGE, "device": "auto", "moge_report_free_focal": True}, size=(420, 330))
    d_cp = g.node("AtlasDepthMap", title="2b · CLEAN PLATE DEPTH · MoGe, same solve", values={
        "depth_model": MOGE, "device": "auto"}, size=(420, 330))
    sam = g.node("AtlasSAM3Mask", title="3 · OBJECT MASK · SAM3 'machine' · bypassed (needs [sam3])", values={
        "concepts": "machine, machinery, rusty engine"}, size=(400, 260), mode=4)
    layer = g.node("AtlasCleanPlateLayer", title="4 · CLEAN PLATE LAYER · geometry from the plate's OWN depth", values={
        "near_pct": 0.0, "far_pct": 1.0, "name": "clean_plate_geo", "priority": 5.0,
        "relief_grid": 512, "depth_edge_rel": 1.5, "max_edge_factor": 12.0,
    }, size=(460, 900))
    controls = _controls(g, vfx=False)
    viewport = g.node("AtlasBlockoutViewport", title="5 · VIEWPORT · orbit behind the machine", values={
        "resolution": 1024, "client_data": "", "preview_expand": 1.0,
    }, size=(900, 680))
    note = _note(g, CLEANPLATE_NOTE, "READ ME · clean plate geometry from its own depth", size=(900, 620))
    g.connect(src, "IMAGE", atlas_input, "image")
    g.connect(src, "IMAGE", d_src, "image")
    g.connect(atlas_input, "solve", d_src, "solve")
    g.connect(cp, "IMAGE", d_cp, "image")
    g.connect(atlas_input, "solve", d_cp, "solve")
    g.connect(src, "IMAGE", sam, "image")
    g.connect(atlas_input, "solve", layer, "solve")
    g.connect(d_src, "depth", layer, "depth")
    g.connect(cp, "IMAGE", layer, "plate_image")
    g.connect(d_cp, "depth", layer, "plate_depth")
    g.connect(sam, "mask", layer, "object_mask")
    g.connect(atlas_input, "sky_mask", layer, "exclude_mask")
    g.connect(layer, "solve", viewport, "solve")
    g.connect(src, "IMAGE", viewport, "source_image")
    g.connect(d_src, "depth", viewport, "primary_depth")
    g.connect(controls, "controls", viewport, "controls")
    g.connect(controls, "output_profile", viewport, "output_profile")
    groups = [
        ("SOURCE · solve + depth", "#3f789e", [src, atlas_input, d_src]),
        ("CLEAN PLATE · depth + object mask", "#88A", [cp, d_cp, sam]),
        ("LAYER + VIEW", "#b58b2a", [layer, controls, viewport]),
    ]
    return _finish(g, slug, groups, layout, CLEANPLATE_NOTE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="", help="live ComfyUI host:port; default = local registry")
    args = ap.parse_args()
    oi = _fetch_object_info(args.host) if args.host else local_object_info()
    layout = _load_layout_module()
    for slug, builder in (("atlas_qwen_roi_registered_patch_workflow", build_qwen),
                          ("atlas_blender_measured_primitives_workflow", build_blender),
                          ("atlas_cleanplate_depth_layer_workflow", build_cleanplate)):
        wf = builder(oi, layout)
        out = EXAMPLES / f"{slug}.json"
        out.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")


if __name__ == "__main__":
    main()
