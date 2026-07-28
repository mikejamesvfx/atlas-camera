"""Generate examples/atlas_equirect_360_multiview_workflow.json.

The MULTI-CAMERA counterpart to atlas_equirect_360_workflow.json. Same panorama,
same measured defaults, but `AtlasEquirectMultiView` absorbs the split, the
solve, the per-view depth passes and the relief mesh into one node — six nodes
become three. Shipping both makes the difference legible side by side.

It also carries AtlasMoveBudget deliberately. That node refused the solve during
development ("has proxy primitives but no relief mesh") because the primary
geometry reached its ProjectionSource but not projection_scene — a failure only a
live run surfaced. Keeping it in a shipped graph means the regression cannot
return unnoticed.

Regenerate:  python tools/build_equirect_360_multiview_workflow.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "atlas_equirect_360_multiview_workflow.json"

# Bare filename: a shipped workflow must never bake an authoring machine's
# absolute path (tests/test_shipping_workflow_paths.py). AtlasLoadPlate resolves
# a bare name against ComfyUI's input directory.
PANORAMA = "urban_street_02_8k.exr"


def _widget_values(class_type: str) -> list:
    """Widget defaults straight from the live class, so a node that gains an
    appended widget cannot silently desync this builder — and so `n_views` here
    always equals the node's own measured default rather than a copy of it."""
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

    def out(name, type_name):
        return {"name": name, "type": type_name, "links": []}

    def inp(name, type_name):
        return {"name": name, "type": type_name, "link": None}

    w = _widget_values("AtlasLoadPlate")
    w[0] = PANORAMA
    add(1, "AtlasLoadPlate", (40, 60), (420, 190),
        title="360 PANORAMA (equirect, 2:1)",
        outputs=[out("image", "IMAGE"), out("alpha", "MASK"),
                 out("plate_ref", "ATLAS_PLATE_REF"), out("report", "STRING")],
        widgets=w)

    add(2, "AtlasEquirectMultiView", (520, 60), (450, 340),
        title="🌐 MULTI-VIEW — the whole ring, one node",
        inputs=[inp("equirect", "IMAGE")],
        outputs=[out("solve", "ATLAS_SOLVE"), out("all_views", "IMAGE"),
                 out("report", "STRING")])

    add(3, "AtlasBlockoutViewport", (1480, 60), (520, 560),
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

    add(4, "AtlasMoveBudget", (1030, 60), (400, 220),
        title="MOVE BUDGET — also a regression guard",
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("camera_path", "ATLAS_CAMERA_PATH")],
        outputs=[out("solve", "ATLAS_SOLVE"), out("report", "STRING")])

    add(5, "PreviewImage", (520, 440), (450, 340), title="EVERY VIEW IN THE RING",
        inputs=[inp("images", "IMAGE")], widgets=[])
    add(6, "ShowText|pysssss", (1030, 320), (400, 220), title="MULTI-VIEW REPORT",
        inputs=[inp("text", "STRING")], widgets=[])
    add(7, "ShowText|pysssss", (1030, 580), (400, 200), title="MOVE BUDGET REPORT",
        inputs=[inp("text", "STRING")], widgets=[])

    wire(1, 0, 2, 0, "IMAGE")          # panorama -> multi-view
    wire(2, 0, 4, 0, "ATLAS_SOLVE")    # solve    -> move budget
    wire(2, 0, 3, 0, "ATLAS_SOLVE")    # solve    -> viewport
    # all_views is an N-frame BATCH into a single-image input, and that is
    # deliberate: _image_tensor_to_pil takes frame [0], which is the primary view
    # by construction (perspective_view_angles puts yaw 0 first, and the node
    # solves that view as the primary). See the note node.
    wire(2, 1, 3, 1, "IMAGE")
    wire(2, 1, 5, 0, "IMAGE")
    wire(2, 2, 6, 0, "STRING")
    wire(4, 1, 7, 0, "STRING")

    notes = [
        (20, (40, 300), (420, 620), "MULTI-VIEW vs SINGLE-VIEW", """## The whole ring in one node

Compare with **atlas_equirect_360_workflow.json**, the single-view graph. Same
panorama, same measured defaults — but there the split, solve, depth and relief
mesh are four separate nodes operating on ONE crop. Here `AtlasEquirectMultiView`
does all of it for every view in the ring.

### Why not chain AtlasAddPatchView instead
Two reasons, one fatal.

`AtlasAddPatchView` builds patch cameras with `orbit_camera`, which **moves the
eye** — it rotates the camera's offset from a ground pivot and re-aims,
displacing it by ~`2*r*sin(delta/2)`, metres at a typical pivot distance.
Panorama views share ONE optical centre and differ only in direction, so
chaining patch nodes registers their geometry in the wrong place.

And eleven chained patch nodes killed the ComfyUI server: each link deep-copies
a whole solve and holds its own depth map. This node walks the ring
sequentially and releases each depth — 12 views in ~32 s.

### One eye, one scale, one height
Every view shares the recovered optical centre (`distance_scale` is exactly 1.0,
a fact about panoramas). The ground scale is computed once from the primary
rather than refitted per view, and the height is the MEDIAN of the sampled
views. Read the report: it prints every sample, the spread and the furthest
outlier, because a consolidation nobody can audit is how a wrong scale gets
baked in unnoticed."""),
        (21, (1480, 660), (520, 420), "WIRING + THE MEASURED DEFAULTS", """## Two things worth knowing

### `all_views` -> viewport `source_image`
That is an N-frame batch going into a single-image input, and it is correct:
the viewport's `_image_tensor_to_pil` takes frame **[0]**, which is the primary
view by construction — `perspective_view_angles` puts yaw 0 first and the node
solves that view as the primary. Pinned by a test so it cannot silently drift if
view ordering ever changes.

### `n_views` defaults to 4, not 12
Measured, not guessed. Going 2 -> 4 closes the ±90° gap that makes a sideways
dolly disocclude — safe z dolly jumps **4.2x**, 0.233 -> 0.983 m — and the
budget then PLATEAUS:

| views | z dolly | torn frame |
|---|---|---|
| 2 | 0.233 m | 11.7% |
| **4** | **0.983 m** | 11.7% |
| 8 | 0.927 m | 7.4% |
| 12 | 0.906 m | 5.5% |

So four views buys the camera freedom; raising it buys projection COVERAGE, at
one depth pass per view. Raise it when projection quality matters more than
runtime.

### Why AtlasMoveBudget is in a shipped graph
It is not decoration. During development it refused this solve — *"has proxy
primitives but no relief mesh"* — because the primary geometry reached its
ProjectionSource but not `projection_scene`. No unit test caught that; only a
live run did. Keeping it here means the regression cannot come back quietly."""),
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
