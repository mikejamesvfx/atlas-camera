"""Generate the AUTO layered-inpaint workflow — the whole anti-tear pipeline,
self-contained.

The layered-segmentation workflow ships with an artist-made clean plate. This
one makes the plate IN-GRAPH with the pack's own crop→SDXL-inpaint→stitch trio,
so nothing external is needed beyond an SDXL checkpoint:

  VLM names occluders → SAM3 masks them → AtlasInpaintCrop scopes the inpaint
  to the occlusion event (crop discipline: one giant crop invents a connected
  mega-structure) → AtlasSDXLInpaint paints the occluders out → stitch back →
  the clean plate gets ITS OWN depth solve → full-range AtlasCleanPlateLayer →
  twin move budgets give the verdict. The layered budget must GROW.

VRAM choreography matters here and is encoded in the widgets: the VLM offloads
IMMEDIATELY (offload_ttl_s=2) because SDXL needs the memory right after.

Run: python tools/build_auto_layered_inpaint_workflow.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid

# Pin imports to this tree (a worktree's editable install points at MAIN).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atlas_camera.comfy import node_registry as R  # noqa: E402

PLATE = "source_image.png"
CHECKPOINT = "SDXL/juggernautXL_versionXInpaint.safetensors"
OUT = pathlib.Path(__file__).resolve().parents[1] / "examples" / \
    "atlas_auto_layered_inpaint_workflow.json"


def widget_defaults(node_type: str, overrides: dict | None = None) -> list:
    cls = R.NODE_CLASS_MAPPINGS[node_type]
    spec = cls.INPUT_TYPES()
    overrides = overrides or {}
    out = []
    for section in ("required", "optional"):
        for name, decl in (spec.get(section) or {}).items():
            if not isinstance(decl, (tuple, list)) or not decl:
                continue
            kind, opts = decl[0], (decl[1] if len(decl) > 1 else {})
            if isinstance(opts, dict) and opts.get("forceInput"):
                continue
            if isinstance(kind, (list, tuple)):
                out.append(overrides.get(name, opts.get("default",
                                                        kind[0] if kind else "")))
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out.append(overrides.get(name, opts.get("default", 0)))
                if kind == "INT" and (name in ("seed", "noise_seed")
                                      or opts.get("control_after_generate")):
                    # The frontend inserts a control_after_generate widget
                    # after seed-type INTs. widgets_values is positional, so
                    # omitting its slot shifts every later widget by one —
                    # found live: grow_mask_by=24 landed in denoise (max 1.0)
                    # and the workflow refused to queue.
                    out.append("fixed")
    return out


class Builder:
    def __init__(self):
        self.nodes, self.links = [], []
        self.nid, self.lid = 0, 0

    def add(self, node_type, pos, title, *, widgets=None, overrides=None,
            size=(340, 150)):
        self.nid += 1
        self.nodes.append({
            "id": self.nid, "type": node_type, "pos": list(pos),
            "size": list(size), "flags": {}, "order": self.nid - 1, "mode": 0,
            "inputs": [], "outputs": [],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets if widgets is not None
            else widget_defaults(node_type, overrides),
            "title": title,
        })
        return self.nid

    def note(self, pos, title, text, size=(480, 320)):
        self.nid += 1
        self.nodes.append({
            "id": self.nid, "type": "Note", "pos": list(pos), "size": list(size),
            "flags": {}, "order": self.nid - 1, "mode": 0,
            "inputs": [], "outputs": [], "properties": {},
            "widgets_values": [text], "title": title, "color": "#432",
            "bgcolor": "#653",
        })
        return self.nid

    def link(self, src, src_slot, dst, dst_slot, type_name, *,
             src_name=None, dst_name=None, widget_name=None):
        self.lid += 1
        self.links.append([self.lid, src, src_slot, dst, dst_slot, type_name])
        s = self._node(src)
        while len(s["outputs"]) <= src_slot:
            s["outputs"].append({"name": "", "type": type_name, "links": []})
        s["outputs"][src_slot].update(
            {"name": src_name or type_name, "type": type_name})
        s["outputs"][src_slot].setdefault("links", []).append(self.lid)
        d = self._node(dst)
        while len(d["inputs"]) <= dst_slot:
            d["inputs"].append({"name": "", "type": type_name, "link": None})
        entry = {"name": dst_name or type_name.lower(), "type": type_name,
                 "link": self.lid}
        if widget_name:
            entry["widget"] = {"name": widget_name}
        d["inputs"][dst_slot].update(entry)
        return self.lid

    def _node(self, nid):
        return next(n for n in self.nodes if n["id"] == nid)

    def validate(self):
        by_id = {n["id"]: n for n in self.nodes}
        bad = []
        for lid, src, ss, dst, ds, _t in self.links:
            out_t = by_id[src]["outputs"][ss]["type"]
            in_t = by_id[dst]["inputs"][ds]["type"]
            if out_t != in_t:
                bad.append(f"link {lid}: {by_id[src]['type']}.{out_t} -> "
                           f"{by_id[dst]['type']}.{in_t}")
        if bad:
            raise SystemExit("incompatible links:\n  " + "\n  ".join(bad))

    def dump(self, path, extra):
        self.validate()
        doc = {
            "id": str(uuid.uuid4()), "revision": 0,
            "last_node_id": self.nid, "last_link_id": self.lid,
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "extra": extra, "version": 0.4,
        }
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return doc


def main() -> None:
    b = Builder()

    load = b.add("LoadImage", (60, 220), "1 · SOURCE PLATE",
                 widgets=[PLATE, "image"], size=(320, 314))

    assess = b.add("AtlasAssessImage", (430, 160),
                   "2 · \U0001f9e0 VLM NAMES OCCLUDERS (frees RAM for SDXL)",
                   overrides={"provider": "lmstudio",
                              "model": "google/gemma-4-12b-qat",
                              "proceed": True,
                              "offload_model": True,
                              # SDXL needs the VRAM right after — evict NOW,
                              # JIT reloads on the next plate.
                              "offload_ttl_s": 2},
                   size=(360, 430))

    inp = b.add("AtlasInput", (840, 160), "3 · SOLVE + DEPTH + RELIEF",
                overrides={"depth_model": "Ruicheng/moge-2-vitl-normal"},
                size=(360, 500))

    sam = b.add("AtlasSAM3Mask", (430, 660), "4 · \U0001fa84 SAM3 OCCLUDER MASK")

    crop = b.add("AtlasInpaintCrop", (800, 700),
                 "5 · ✂ CROP THE OCCLUSION EVENT",
                 overrides={"context_pad_px": 96})
    paint = b.add("AtlasSDXLInpaint", (1140, 700),
                  "6 · ✨ PAINT THE OCCLUDERS OUT",
                  overrides={
                      "checkpoint": CHECKPOINT,
                      "positive_prompt": "empty scene, clean background, "
                                         "seamless continuation of the "
                                         "surrounding environment",
                      "negative_prompt": "person, people, car, vehicle, "
                                         "object, text, watermark",
                      "grow_mask_by": 24,
                  }, size=(380, 420))
    stitch = b.add("AtlasInpaintStitch", (1560, 700), "7 · STITCH → CLEAN PLATE")
    cpprev = b.add("PreviewImage", (1560, 900), "CLEAN PLATE (in-graph)",
                   widgets=[], size=(300, 280))

    cdepth = b.add("AtlasDepthMap", (1900, 700),
                   "8 · CLEAN PLATE'S OWN DEPTH (doctrine)",
                   overrides={"depth_model": "Ruicheng/moge-2-vitl-normal"},
                   size=(340, 200))

    walls = b.add("AtlasDeriveWalls", (1250, 120), "9 · FIT PLANES")
    merge = b.add("AtlasMergeGeometry", (1250, 290), "10 · MERGE + RELIEF")
    graph = b.add("AtlasOcclusionGraph", (1250, 450), "11 · \U0001f578 GRAPH")
    plan = b.add("AtlasLayerPlan", (1620, 160), "12 · \U0001f95e LAYER PLAN")

    layer = b.add("AtlasCleanPlateLayer", (2280, 500),
                  "13 · \U0001f96e BACKGROUND LAYER (full range)",
                  overrides={"near_pct": 0.0, "far_pct": 0.0,
                             "fill_occluded": False,
                             "name": "cleanplate_background"},
                  size=(380, 560))

    bud_a = b.add("AtlasMoveBudget", (2280, 130), "14a · \U0001f4d0 BUDGET before")
    bud_b = b.add("AtlasMoveBudget", (2700, 130), "14b · \U0001f4d0 BUDGET after — must GROW")
    view = b.add("AtlasBlockoutViewport", (2700, 420), "15 · VIEWPORT",
                 size=(480, 640))

    b.link(load, 0, assess, 0, "IMAGE", src_name="IMAGE", dst_name="image")
    b.link(assess, 0, inp, 0, "IMAGE", src_name="image", dst_name="image")
    b.link(assess, 0, sam, 0, "IMAGE", src_name="image", dst_name="image")
    b.link(assess, 7, sam, 1, "STRING", src_name="sam_prompt_fg",
           dst_name="concepts", widget_name="concepts")

    b.link(assess, 0, crop, 0, "IMAGE", src_name="image", dst_name="image")
    b.link(sam, 0, crop, 1, "MASK", src_name="mask", dst_name="mask")
    b.link(crop, 0, paint, 0, "IMAGE", src_name="cropped_image", dst_name="image")
    b.link(crop, 1, paint, 1, "MASK", src_name="cropped_mask", dst_name="mask")
    b.link(assess, 0, stitch, 0, "IMAGE", src_name="image",
           dst_name="original_image")
    b.link(paint, 0, stitch, 1, "IMAGE", src_name="image",
           dst_name="inpainted_crop")
    b.link(crop, 2, stitch, 2, "ATLAS_CROP_REGION", src_name="crop_region",
           dst_name="crop_region")
    b.link(stitch, 0, cpprev, 0, "IMAGE", src_name="image", dst_name="images")

    b.link(stitch, 0, cdepth, 0, "IMAGE", src_name="image", dst_name="image")
    b.link(inp, 0, cdepth, 1, "ATLAS_SOLVE", src_name="solve", dst_name="solve")

    b.link(inp, 0, walls, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(inp, 2, walls, 1, "ATLAS_DEPTH_MAP", src_name="depth", dst_name="depth")
    b.link(walls, 0, merge, 0, "ATLAS_SOLVE", src_name="ATLAS_SOLVE",
           dst_name="solve_a")
    b.link(inp, 0, merge, 1, "ATLAS_SOLVE", src_name="solve", dst_name="solve_b")
    b.link(merge, 0, graph, 0, "ATLAS_SOLVE", src_name="ATLAS_SOLVE",
           dst_name="solve")
    b.link(inp, 2, graph, 1, "ATLAS_DEPTH_MAP", src_name="depth", dst_name="depth")
    b.link(graph, 0, plan, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(assess, 7, plan, 1, "STRING", src_name="sam_prompt_fg",
           dst_name="foreground_concepts_override")
    b.link(assess, 5, plan, 2, "STRING", src_name="sam_prompt_bg",
           dst_name="background_concepts_override")

    b.link(plan, 0, layer, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(cdepth, 0, layer, 1, "ATLAS_DEPTH_MAP", src_name="depth",
           dst_name="depth")
    b.link(stitch, 0, layer, 2, "IMAGE", src_name="image",
           dst_name="plate_image")

    b.link(plan, 0, bud_a, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(layer, 0, bud_b, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(layer, 0, view, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(load, 0, view, 1, "IMAGE", src_name="IMAGE", dst_name="source_image")

    b.note((60, 1000), "READ ME · fully automatic layers", "\n".join([
        "AUTO LAYERED INPAINT — the whole anti-tear pipeline, no external",
        "packs, no hand-made plate.",
        "",
        "The clean plate is made IN-GRAPH: SAM3 masks what the VLM named,",
        "the inpaint is CROPPED to the occlusion event (one giant crop",
        "invents a connected mega-structure — crop discipline), SDXL paints",
        "the occluders out, and the result stitches back losslessly.",
        "",
        "VRAM choreography is deliberate: the VLM offloads immediately",
        "(offload_ttl_s=2) because SDXL needs the memory next. On a batch,",
        "raise offload_ttl_s to ~300 so gemma lingers between plates.",
        "",
        "DOCTRINE: the clean plate gets ITS OWN depth solve (node 8). Never",
        "extend the original's far band — vertical cliff, floating",
        "foreground. The layer is full-range (near_pct=0, far_pct=0,",
        "fill_occluded OFF).",
        "",
        "CHECKPOINT: SDXL/juggernautXL_versionXInpaint.safetensors from the",
        "central H: model drive (extra_model_paths.yaml). Any SDXL inpaint",
        "checkpoint works — set it in node 6.",
        "",
        "VERDICT: 14a vs 14b. The layered budget must GROW. If it does not,",
        "look at the clean-plate preview first — a bad inpaint makes a bad",
        "layer, and the budget will say so honestly.",
    ]), size=(560, 620))

    doc = b.dump(OUT, {
        "ds": {"scale": 0.6, "offset": [0, 0]},
        "atlas_auto_layered_inpaint_version": 1,
        "atlas_notes": "Self-contained VLM->SAM3->crop->SDXL->stitch->layer "
                       "pipeline; twin budgets as acceptance.",
    })
    print(f"wrote {OUT.name}: {len(doc['nodes'])} nodes, {len(doc['links'])} links")


if __name__ == "__main__":
    main()
