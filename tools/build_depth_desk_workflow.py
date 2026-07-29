"""Generate examples/atlas_depth_desk_workflow.json — the depth processing chain.

One graph exercising the depth tools end to end, with visible output at each
consequential step:

    plate -> solve -> depth -> MoGe normals -> detail enhance -> relief -> stereo
                                     |
                                     +-> pad -> outpaint depth -> relief -> stereo

The lower branch is the point of contrast. AtlasOutpaintDepth returns a
`widened_solve` alongside the widened depth, and that branch feeds THAT solve to
its relief mesh — because every geometry node reads width/cx/cy from the solve
rather than from the depth map. Wiring the original solve there instead
misregisters the new ring by exactly the padding while looking entirely
plausible, which is why the graph shows the correct wiring rather than leaving
it to be discovered.

Both branches end in AtlasStereoRender, never AtlasBlockoutViewport: the
viewport renders in the BROWSER via three.js, so a headless queue would save
black frames.

Regenerate:  python tools/build_depth_desk_workflow.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "atlas_depth_desk_workflow.json"

#: Bare filename — a shipped workflow must never bake an authoring machine's
#: absolute path (tests/test_shipping_workflow_paths.py). AtlasLoadPlate resolves
#: a bare name against ComfyUI's input directory.
PLATE = "newyork_Birdseye.png"

PAD_PX = 192


def _widget_values(class_type: str) -> list:
    """Defaults straight from the live class, so a node that gains an appended
    widget cannot silently desync this builder."""
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

    def out(name, t):
        return {"name": name, "type": t, "links": []}

    def inp(name, t):
        return {"name": name, "type": t, "link": None}

    w = _widget_values("AtlasLoadPlate")
    w[0] = PLATE
    add(1, "AtlasLoadPlate", (40, 80), (400, 180), title="PLATE",
        outputs=[out("image", "IMAGE"), out("alpha", "MASK"),
                 out("plate_ref", "ATLAS_PLATE_REF"), out("report", "STRING")],
        widgets=w)

    add(2, "AtlasSolveFromImage", (40, 300), (400, 200), title="SOLVE",
        inputs=[inp("image", "IMAGE")],
        outputs=[out("solve", "ATLAS_SOLVE")])

    add(3, "AtlasDepthMap", (480, 80), (420, 300),
        title="DEPTH — moge_tile_side is the native-resolution lever",
        inputs=[inp("image", "IMAGE"), inp("solve", "ATLAS_SOLVE")],
        outputs=[out("depth", "ATLAS_DEPTH_MAP")])

    add(4, "AtlasMogeNormals", (940, 80), (400, 240),
        title="NORMALS — MoGe's normals, keeping V2's depth",
        inputs=[inp("depth", "ATLAS_DEPTH_MAP"), inp("image", "IMAGE"),
                inp("solve", "ATLAS_SOLVE")],
        outputs=[out("depth", "ATLAS_DEPTH_MAP"), out("report", "STRING")])

    add(5, "AtlasDepthDetailEnhance", (1380, 80), (380, 200),
        title="DETAIL — emboss the normals into depth",
        inputs=[inp("depth", "ATLAS_DEPTH_MAP"), inp("exclude_mask", "MASK")],
        outputs=[out("depth", "ATLAS_DEPTH_MAP"), out("report", "STRING")])

    add(6, "AtlasDeriveReliefMesh", (1800, 80), (400, 320), title="RELIEF",
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("depth", "ATLAS_DEPTH_MAP"),
                inp("exclude_mask", "MASK")],
        outputs=[out("solve", "ATLAS_SOLVE"), out("hole_mask", "MASK")])

    add(7, "AtlasStereoRender", (2240, 80), (380, 220), title="RENDER — server-side",
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("source_image", "IMAGE")],
        outputs=[out("stereo", "IMAGE"), out("left", "IMAGE"),
                 out("right", "IMAGE"), out("report", "STRING")])

    add(8, "SaveImage", (2660, 80), (360, 300), title="MAIN CHAIN",
        inputs=[inp("images", "IMAGE")], widgets=["atlas_depth_desk"])

    # ---- outpaint branch -------------------------------------------------
    # ComfyUI CORE node, so it is not in Atlas's registry and _widget_values
    # cannot resolve it. Its widgets are (left, top, right, bottom, feathering).
    pad_w = [PAD_PX, PAD_PX, PAD_PX, PAD_PX, 0]
    add(10, "ImagePadForOutpaint", (480, 460), (400, 220),
        title=f"PAD {PAD_PX}px — edge-replicated, NOT generated",
        inputs=[inp("image", "IMAGE")],
        outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
        widgets=pad_w)

    add(11, "AtlasOutpaintDepth", (940, 460), (420, 300),
        title="🪟 OUTPAINT DEPTH — emits a WIDENED CAMERA",
        inputs=[inp("depth", "ATLAS_DEPTH_MAP"), inp("widened_image", "IMAGE"),
                inp("solve", "ATLAS_SOLVE")],
        outputs=[out("depth", "ATLAS_DEPTH_MAP"), out("ring_mask", "MASK"),
                 out("report", "STRING"), out("widened_solve", "ATLAS_SOLVE")])

    add(12, "AtlasDeriveReliefMesh", (1800, 460), (400, 320),
        title="RELIEF — fed the WIDENED solve",
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("depth", "ATLAS_DEPTH_MAP"),
                inp("exclude_mask", "MASK")],
        outputs=[out("solve", "ATLAS_SOLVE"), out("hole_mask", "MASK")])

    add(13, "AtlasStereoRender", (2240, 460), (380, 220), title="RENDER — widened",
        inputs=[inp("solve", "ATLAS_SOLVE"), inp("source_image", "IMAGE")],
        outputs=[out("stereo", "IMAGE"), out("left", "IMAGE"),
                 out("right", "IMAGE"), out("report", "STRING")])

    add(14, "SaveImage", (2660, 460), (360, 300), title="OUTPAINTED CHAIN",
        inputs=[inp("images", "IMAGE")], widgets=["atlas_depth_desk_widened"])

    add(20, "ShowText|pysssss", (940, 800), (400, 180), title="NORMALS REPORT",
        inputs=[inp("text", "STRING")], widgets=[])
    add(21, "ShowText|pysssss", (1380, 800), (400, 180), title="DETAIL REPORT",
        inputs=[inp("text", "STRING")], widgets=[])
    add(22, "ShowText|pysssss", (1800, 800), (400, 220), title="OUTPAINT REPORT",
        inputs=[inp("text", "STRING")], widgets=[])

    # main chain
    wire(1, 0, 2, 0, "IMAGE")
    wire(1, 0, 3, 0, "IMAGE")
    wire(2, 0, 3, 1, "ATLAS_SOLVE")
    wire(3, 0, 4, 0, "ATLAS_DEPTH_MAP")
    wire(1, 0, 4, 1, "IMAGE")
    wire(2, 0, 4, 2, "ATLAS_SOLVE")
    wire(4, 0, 5, 0, "ATLAS_DEPTH_MAP")
    wire(2, 0, 6, 0, "ATLAS_SOLVE")
    wire(5, 0, 6, 1, "ATLAS_DEPTH_MAP")
    wire(6, 0, 7, 0, "ATLAS_SOLVE")
    wire(1, 0, 7, 1, "IMAGE")
    wire(7, 0, 8, 0, "IMAGE")

    # outpaint branch
    wire(1, 0, 10, 0, "IMAGE")
    # RAW depth (node 3), NOT the detail-enhanced one (node 5). This node re-runs
    # a raw pass on the widened plate and affine-fits it onto whatever it is
    # given; feeding it post-processed depth means the two differ in
    # high-frequency content, which an affine fit cannot absorb. Measured live on
    # this plate: enhanced input gave a 7.7%-of-scene-depth residual and tripped
    # the node's own warning.
    wire(3, 0, 11, 0, "ATLAS_DEPTH_MAP")
    wire(10, 0, 11, 1, "IMAGE")
    wire(2, 0, 11, 2, "ATLAS_SOLVE")
    # THE point of the branch: the widened solve, not the original.
    wire(11, 3, 12, 0, "ATLAS_SOLVE")
    wire(11, 0, 12, 1, "ATLAS_DEPTH_MAP")
    wire(12, 0, 13, 0, "ATLAS_SOLVE")
    wire(10, 0, 13, 1, "IMAGE")
    wire(13, 0, 14, 0, "IMAGE")

    # reports
    wire(4, 1, 20, 0, "STRING")
    wire(5, 1, 21, 0, "STRING")
    wire(11, 2, 22, 0, "STRING")

    notes = [
        (30, (40, 560), (400, 420), "THE DEPTH CHAIN", """## What each step buys

**AtlasDepthMap** — `moge_tile_side` (0 = off) runs inference on overlapping
tiles at SOURCE resolution instead of letting the model downscale a 36MP plate
to its token budget. Every tile is affine-fitted onto one global pass first,
because monocular depth is scale-ambiguous per input and raw tiles step at
every seam. Costs one pass per tile plus the global one.

**AtlasMogeNormals** — runs a MoGe `*-normal` model purely for its normals and
throws away its depth, so you keep V2's far-field (which behaves on exteriors,
where MoGe's runs away) AND get MoGe's cleaner normals.

**AtlasDepthDetailEnhance** — embosses those normals' high frequencies onto the
depth. Monocular depth is metrically sound but low-frequency: brick courses,
window reveals and rock striations flatten out, and this puts them back."""),
        (31, (2240, 800), (420, 440), "WHY THE LOWER BRANCH EXISTS", """## The widened camera

`frame_outpaint_px` on the clean-plate layer can already widen a plate past the
frame edge — its own tooltip calls the frame-edge reveal *"the binding
constraint on wide scenes"*. But the ring is edge-replicated smear and depth
could not follow it.

**AtlasOutpaintDepth** re-runs depth on the widened plate and stitches it to the
original, affine-fitting the widened pass onto the region they share first: a
monocular model returns a DIFFERENT scale on a different framing, so pasting
would step at the frame boundary.

### Note the wiring
The lower relief mesh is fed `widened_solve`, **not** the original solve. Every
geometry node reads width/cx/cy from the SOLVE rather than from the depth map,
so the original camera would back-project the new ring against the old principal
point — misregistering it by exactly the padding, while looking entirely
plausible.

### Note the OTHER wiring
The outpaint branch takes depth from **AtlasDepthMap (raw)**, not from the
detail-enhanced output. This node re-runs a raw pass on the widened plate and
affine-fits it onto whatever depth it is handed — so handing it POST-PROCESSED
depth means the two differ in high-frequency content, which an affine fit cannot
absorb. Measured on this plate: enhanced input produced a residual of 7.7% of
scene depth and tripped the node's own warning. Raw in, raw out, enhance after.

### The ring is invented
`ImagePadForOutpaint` here is edge replication, not generation. For real content
put a prompt-driven inpaint between the pad and the depth node; the geometry
follows whatever the pixels say."""),
    ]
    for nid, pos, size, title, body in notes:
        add(nid, "Note", pos, size, title=title, widgets=[body])
        nodes[-1]["color"] = "#432"
        nodes[-1]["bgcolor"] = "#653"

    return {
        "id": str(uuid.uuid4()), "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": link_id[0],
        "nodes": nodes, "links": links, "groups": [], "config": {},
        "extra": {}, "version": 0.4,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
