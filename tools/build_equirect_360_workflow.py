"""Generate examples/atlas_equirect_360_workflow.json.

The shipped 360-panorama graph. Every non-obvious default in it was MEASURED on
two 8K panoramas (cobblestone_parish_road, urban_street_02) rather than chosen
by taste — see reports/live_probe_baseline.json and DESIGN_RULES:

  * depth_model = MoGe-2, NOT V2-Metric-Outdoor. The exterior doctrine does not
    survive panorama crops: V2 mis-scaled both plates ~4.4x (5.57 vs 1.22 m,
    6.00 vs 1.40 m) while recovering the FOV perfectly. A metric-scale failure,
    not a geometry one.
  * focal_length_mm is WIRED from the node, not left at 0. The crop's focal is
    constructed from the FOV we asked for, so guessing it costs up to 10 deg of
    FOV error and can drop a view to scale_source=assumed_default.
  * pitch_deg = 0. Tilting the ring down removes the horizon and the ground fit
    stops working entirely (0/6 views measured at -20/-40/-60).

Regenerate:  python tools/build_equirect_360_workflow.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "atlas_equirect_360_workflow.json"

MOGE = "Ruicheng/moge-2-vitl-normal"
# Bare filename on purpose: a shipped workflow must never bake an authoring
# machine's absolute path (tests/test_shipping_workflow_paths.py).
PANORAMA = "urban_street_02_8k.exr"


def _widget_values(class_type: str) -> list:
    """Widget defaults straight from the live class, so a node that gains an
    appended widget cannot silently desync this builder."""
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.mcp.comfy_http import is_widget

    cls = reg.NODE_CLASS_MAPPINGS[class_type]
    it = cls.INPUT_TYPES()
    out = []
    for sec in ("required", "optional"):
        for name, spec in (it.get(sec) or {}).items():
            if not is_widget(spec):
                continue
            cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            default = cfg.get("default")
            if default is None:
                default = spec[0][0] if isinstance(spec[0], list) and spec[0] else ""
            out.append(default)
            if name in ("seed", "noise_seed"):
                out.append("fixed")
    return out


def build() -> dict:
    nodes, links = [], []
    link_id = [0]

    def add(node_id, class_type, pos, size, *, title=None, inputs=None,
            outputs=None, widgets=None):
        nodes.append({
            "id": node_id, "type": class_type, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(nodes), "mode": 0,
            "inputs": inputs or [], "outputs": outputs or [],
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets if widgets is not None else _widget_values(class_type),
            **({"title": title} if title else {}),
        })

    def wire(src_id, src_slot, dst_id, dst_slot, type_name):
        link_id[0] += 1
        lid = link_id[0]
        links.append([lid, src_id, src_slot, dst_id, dst_slot, type_name])
        for n in nodes:
            if n["id"] == src_id:
                n["outputs"][src_slot].setdefault("links", [])
                n["outputs"][src_slot]["links"].append(lid)
            if n["id"] == dst_id:
                n["inputs"][dst_slot]["link"] = lid
        return lid

    def out(name, type_name):
        return {"name": name, "type": type_name, "links": []}

    def inp(name, type_name):
        return {"name": name, "type": type_name, "link": None}

    # 1 — plate. AtlasLoadPlate because ComfyUI's LoadImage cannot read float EXR;
    # OIIO also carries the colourspace (these HDRIs are ACEScg).
    w = _widget_values("AtlasLoadPlate")
    w[0] = PANORAMA
    add(1, "AtlasLoadPlate", (40, 60), (420, 190), title="360 PANORAMA (equirect, 2:1)",
        outputs=[out("image", "IMAGE"), out("alpha", "MASK"),
                 out("plate_ref", "ATLAS_PLATE_REF"), out("report", "STRING")],
        widgets=w)

    # 2 — split. pitch 0 is load-bearing: the ground fit needs the horizon.
    w = _widget_values("AtlasSplitEquirect")
    add(2, "AtlasSplitEquirect", (520, 60), (430, 240), title="🌐 SPLIT — 12 views @ 90°",
        inputs=[inp("equirect", "IMAGE")],
        outputs=[out("view", "IMAGE"), out("exact_view", "STRING"),
                 out("focal_mm", "FLOAT"), out("all_views", "IMAGE"),
                 out("report", "STRING")],
        widgets=w)

    # 3 — solve. MoGe, and the focal WIRED rather than guessed.
    w = _widget_values("AtlasLearnedSolveFromImage")
    it = __import__("atlas_camera.comfy.node_registry", fromlist=["x"]).NODE_CLASS_MAPPINGS[
        "AtlasLearnedSolveFromImage"].INPUT_TYPES()
    names = []
    from atlas_camera.mcp.comfy_http import is_widget
    for sec in ("required", "optional"):
        for name, spec in (it.get(sec) or {}).items():
            if is_widget(spec):
                names.append(name)
    w[names.index("depth_model")] = MOGE
    w[names.index("height_mode")] = "measure_from_depth"
    add(3, "AtlasLearnedSolveFromImage", (1000, 60), (430, 260),
        title="SOLVE — MoGe + EXACT focal",
        inputs=[inp("image", "IMAGE"), inp("focal_length_mm", "FLOAT")],
        outputs=[out("solve", "ATLAS_SOLVE"), out("report", "STRING")],
        widgets=w)

    # 4 — depth, same backend as the solve so scale agrees.
    w = _widget_values("AtlasDepthMap")
    w[0] = MOGE
    add(4, "AtlasDepthMap", (1000, 360), (430, 200), title="DEPTH — MoGe (matches solve)",
        inputs=[inp("image", "IMAGE"), inp("solve", "ATLAS_SOLVE")],
        outputs=[out("depth", "ATLAS_DEPTH_MAP")], widgets=w)

    # 5 — geometry
    add(5, "AtlasDeriveReliefMesh", (1480, 60), (430, 320),
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("depth", "ATLAS_DEPTH_MAP"),
                inp("exclude_mask", "MASK"), inp("outlier_mask", "MASK")],
        outputs=[out("solve", "ATLAS_SOLVE"), out("hole_mask", "MASK")])

    # 6 — viewport
    add(6, "AtlasBlockoutViewport", (1960, 60), (520, 560),
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("source_image", "IMAGE"),
                inp("primary_depth", "ATLAS_DEPTH_MAP"), inp("preview_expand", "BOOLEAN"),
                inp("controls", "ATLAS_VIEWPORT_LINK"), inp("shot_cam", "ATLAS_SHOT_CAM"),
                inp("output_profile", "ATLAS_OUTPUT_PROFILE"), inp("debug_matte", "MASK"),
                inp("patch_mask", "MASK")],
        outputs=[out("shaded", "IMAGE"), out("depth", "IMAGE"), out("normal", "IMAGE"),
                 out("mask", "MASK"), out("path_frames", "IMAGE"),
                 out("camera_path", "ATLAS_CAMERA_PATH"),
                 out("patch_azimuth_view", "STRING"), out("patch_elevation_view", "STRING"),
                 out("patch_distance", "FLOAT"), out("patch_prompt", "STRING"),
                 out("patch_exact", "STRING"), out("patch_render_mask", "MASK")])

    # previews / reports
    add(7, "PreviewImage", (520, 360), (430, 330), title="ALL 12 VIEWS",
        inputs=[inp("images", "IMAGE")], widgets=[])
    add(8, "ShowText|pysssss", (520, 730), (430, 180), title="SPLIT REPORT",
        inputs=[inp("text", "STRING")], widgets=[])
    add(9, "ShowText|pysssss", (1000, 610), (430, 180), title="PLATE REPORT",
        inputs=[inp("text", "STRING")], widgets=[])

    wire(1, 0, 2, 0, "IMAGE")        # plate  -> split
    wire(2, 0, 3, 0, "IMAGE")        # view   -> solve
    wire(2, 2, 3, 1, "FLOAT")        # focal  -> solve   (the whole point)
    wire(2, 0, 4, 0, "IMAGE")        # view   -> depth
    wire(3, 0, 4, 1, "ATLAS_SOLVE")
    wire(3, 0, 5, 0, "ATLAS_SOLVE")
    wire(4, 0, 5, 1, "ATLAS_DEPTH_MAP")
    wire(5, 0, 6, 0, "ATLAS_SOLVE")
    wire(2, 0, 6, 1, "IMAGE")
    wire(4, 0, 6, 2, "ATLAS_DEPTH_MAP")
    wire(2, 3, 7, 0, "IMAGE")        # all_views -> preview
    wire(2, 4, 8, 0, "STRING")
    wire(1, 3, 9, 0, "STRING")

    notes = [
        (20, (40, 300), (420, 620), "360 PANORAMA -> ATLAS", """## Equirect 360 intake

**Load a 2:1 equirectangular panorama** (8192x4096 etc). `AtlasLoadPlate`
because ComfyUI's `LoadImage` cannot read float EXR — OIIO also detects the
colourspace (these HDRIs are usually ACEScg).

Atlas is pinhole end to end, so the panorama is NOT modelled as an equirect
camera. It is cut into 12 perspective crops, each already a valid Atlas camera.

### Why this exists
A single plate has no data for what the camera never saw — the whole reason
`AtlasOcclusionGraph` / `AtlasMoveBudget` / `AtlasPathGuidedHoleRepair` exist.
A 360 capture supplies that coverage as REAL geometry instead of inventing it.

### Working the ring
`view_index` picks which crop leaves `view` / `exact_view`. Step it 0..11 to
walk the ring; `all_views` always carries the whole batch for preview.

To build a multi-camera scene, wire `view` -> `AtlasAddPatchView.patch_image`
and `exact_view` -> its `exact_view_override`. These angles are MEASURED, not
estimated, which is why they bypass the named-view combos."""),
        (21, (1480, 420), (430, 500), "MEASURED DEFAULTS — DO NOT GUESS", """## Every default here was measured

Two 8K panoramas, 12 views each. See reports/live_probe_baseline.json.

### MoGe, not V2-Metric-Outdoor
The exterior doctrine does not survive panorama crops. A 90 deg square crop is
framing V2 was not trained on and it mis-scales ~4.4x:

| plate | V2-Outdoor | MoGe-2 |
|---|---|---|
| parish road | 5.57 m | **1.22 m** |
| urban street 02 | 6.00 m | **1.40 m** |

Both backends were internally consistent (stdev 0.04-0.4), so this is
systematic, not noise. FOV is exact under both — a metric-scale failure only.

### focal_length_mm is WIRED, not 0
The crop's focal is constructed from the FOV we chose, so it is exact. Letting
the solver guess cost 1.6 / 10.2 / 9.1 / 3.8 deg of FOV error across four views,
and dropped one to `scale_source=assumed_default`. Wired: **0.000 deg on all 12**.

### pitch_deg stays 0
Tilting down does not "see more ground". It removes the horizon and the ground
fit stops working — at -20/-40/-60 every view fell back to `assumed_default`
(stdev exactly 0.0000, 0/6 measured).

### All views share ONE height
They share an optical centre, so disagreement is noise — independent fits spread
0.90 m and 0.46 m. Solve ONE view, then give the rest `height_mode=assume` with
that value (the median, if you sample several)."""),
    ]
    for nid, pos, size, title, body in notes:
        add(nid, "Note", pos, size, title=title, widgets=[body])
        nodes[-1]["color"] = "#432"
        nodes[-1]["bgcolor"] = "#653"

    return {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": link_id[0],
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
