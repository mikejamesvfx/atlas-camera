"""Generate the layered-segmentation workflow — the anti-tear pipeline.

The measured story this graph encodes (live A/B, 2026-07-25, overpass plate):
filling holes in one surface tears along every fill seam and SHRANK the move
budget; one clean-plate layer that CONTINUES underneath the occluders GREW it
(dolly z +/-1.804 -> +/-2.044 m). Layers, not patches.

Widget values are read from each node's live ``INPUT_TYPES`` (with named
overrides) so the positional ``widgets_values`` arrays cannot drift — and every
link is type-validated before the file is written.

Run: python tools/build_layered_segmentation_workflow.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid

# Pin imports to this tree: run as `python tools/...py`, sys.path[0] is tools/,
# and `atlas_camera` would otherwise resolve from the editable install's MAIN
# checkout — silently generating against the wrong node definitions.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atlas_camera.comfy import node_registry as R  # noqa: E402

PLATE = "source_image.png"
CLEANPLATE = "cleanplate.png"
OUT = pathlib.Path(__file__).resolve().parents[1] / "examples" / \
    "atlas_layered_segmentation_workflow.json"


def widget_defaults(node_type: str, overrides: dict | None = None) -> list:
    """Default widget values in declared order, with named overrides.

    Only widget-rendered entries count; link-only sockets contribute nothing.
    """
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
                continue                       # socket, not a widget
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
            size=(340, 140)):
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

    def note(self, pos, title, text, size=(460, 300)):
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
            # A widget converted to an input socket (STRING prompts fed by
            # upstream outputs) must carry its widget binding or the loader
            # misaligns the remaining positional widgets_values.
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

    # --- plates ------------------------------------------------------------
    load = b.add("LoadImage", (60, 200), "1 · SOURCE PLATE",
                 widgets=[PLATE, "image"], size=(320, 314))
    clean = b.add("LoadImage", (60, 620), "1b · CLEAN PLATE (occluders painted out)",
                  widgets=[CLEANPLATE, "image"], size=(320, 314))

    # --- VLM names the scene ----------------------------------------------
    assess = b.add("AtlasAssessImage", (430, 160),
                   "2 · \U0001f9e0 VLM NAMES THE SCENE",
                   overrides={"provider": "lmstudio",
                              "model": "google/gemma-4-12b-qat",
                              "proceed": True,
                              # gemma at full context holds tens of GB in LM
                              # Studio; ask the provider to release it after
                              # each assessment.
                              "offload_model": True},
                   size=(360, 420))

    # --- shared solve + depth + relief --------------------------------------
    inp = b.add("AtlasInput", (840, 160), "3 · SOLVE + DEPTH + RELIEF",
                overrides={"depth_model": "Ruicheng/moge-2-vitl-normal"},
                size=(360, 500))

    # --- occluder mask (QA view of what the VLM named) ----------------------
    sam = b.add("AtlasSAM3Mask", (430, 640), "4 · \U0001fa84 SAM3 OCCLUDER MASK")
    m2i = b.add("MaskToImage", (430, 830), "mask → image", widgets=[],
                size=(260, 30))
    mprev = b.add("PreviewImage", (430, 900), "OCCLUDER MASK", widgets=[],
                  size=(300, 280))

    # --- geometric analysis: graph + plan -----------------------------------
    walls = b.add("AtlasDeriveWalls", (1250, 120), "5 · FIT PLANES + OBJECTS")
    merge = b.add("AtlasMergeGeometry", (1250, 300), "6 · MERGE + RELIEF")
    graph = b.add("AtlasOcclusionGraph", (1250, 470),
                  "7 · \U0001f578 OCCLUSION GRAPH")
    plan = b.add("AtlasLayerPlan", (1250, 650), "8 · \U0001f95e LAYER PLAN")

    # --- the layer: clean plate on ITS OWN depth solve ----------------------
    cdepth = b.add("AtlasDepthMap", (840, 720),
                   "9 · CLEAN PLATE'S OWN DEPTH (doctrine: never a band extension)",
                   overrides={"depth_model": "Ruicheng/moge-2-vitl-normal"},
                   size=(360, 200))
    layer = b.add("AtlasCleanPlateLayer", (1660, 470),
                  "10 · \U0001f96e BACKGROUND LAYER (full range)",
                  overrides={"near_pct": 0.0, "far_pct": 0.0,
                             "fill_occluded": False,
                             "name": "cleanplate_background"},
                  size=(380, 560))

    # --- verdicts ------------------------------------------------------------
    bud_a = b.add("AtlasMoveBudget", (1660, 120), "11a · \U0001f4d0 BUDGET before")
    bud_b = b.add("AtlasMoveBudget", (2080, 120), "11b · \U0001f4d0 BUDGET after — must GROW")
    view = b.add("AtlasBlockoutViewport", (2080, 400), "12 · VIEWPORT",
                 size=(480, 640))

    # --- wiring --------------------------------------------------------------
    b.link(load, 0, assess, 0, "IMAGE", src_name="IMAGE", dst_name="image")
    # AtlasAssessImage: 0 image .. 3 sky 4 far 5 bg 6 mid 7 fg ...
    b.link(assess, 0, inp, 0, "IMAGE", src_name="image", dst_name="image")
    b.link(assess, 7, sam, 1, "STRING", src_name="sam_prompt_fg",
           dst_name="concepts", widget_name="concepts")
    b.link(assess, 0, sam, 0, "IMAGE", src_name="image", dst_name="image")
    b.link(sam, 0, m2i, 0, "MASK", src_name="mask", dst_name="mask")
    b.link(m2i, 0, mprev, 0, "IMAGE", src_name="IMAGE", dst_name="images")

    # AtlasInput: 0 solve 1 image 2 depth 3 sky 4 report
    b.link(inp, 0, walls, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(inp, 2, walls, 1, "ATLAS_DEPTH_MAP", src_name="depth", dst_name="depth")
    b.link(walls, 0, merge, 0, "ATLAS_SOLVE", src_name="ATLAS_SOLVE",
           dst_name="solve_a")
    b.link(inp, 0, merge, 1, "ATLAS_SOLVE", src_name="solve", dst_name="solve_b")
    b.link(merge, 0, graph, 0, "ATLAS_SOLVE", src_name="ATLAS_SOLVE",
           dst_name="solve")
    b.link(inp, 2, graph, 1, "ATLAS_DEPTH_MAP", src_name="depth",
           dst_name="depth")
    b.link(graph, 0, plan, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(assess, 7, plan, 1, "STRING", src_name="sam_prompt_fg",
           dst_name="foreground_concepts_override")
    b.link(assess, 5, plan, 2, "STRING", src_name="sam_prompt_bg",
           dst_name="background_concepts_override")

    # Clean plate gets its OWN depth solve, camera shared via the solve link.
    b.link(clean, 0, cdepth, 0, "IMAGE", src_name="IMAGE", dst_name="image")
    b.link(inp, 0, cdepth, 1, "ATLAS_SOLVE", src_name="solve", dst_name="solve")

    b.link(plan, 0, layer, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(cdepth, 0, layer, 1, "ATLAS_DEPTH_MAP", src_name="depth",
           dst_name="depth")
    b.link(clean, 0, layer, 2, "IMAGE", src_name="IMAGE",
           dst_name="plate_image")

    b.link(plan, 0, bud_a, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(layer, 0, bud_b, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(layer, 0, view, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(load, 0, view, 1, "IMAGE", src_name="IMAGE", dst_name="source_image")

    b.note((60, 1000), "READ ME · layers, not patches", "\n".join([
        "LAYERED SEGMENTATION — the anti-tear pipeline, measured.",
        "",
        "Why layers: filling holes in ONE surface makes every fill butt",
        "against measured depth and tear along the seam (live measurement:",
        "302k px closed, 222k px of NEW rim tears, budget SHRANK). A",
        "clean-plate layer instead CONTINUES underneath the occluder — the",
        "surfaces overlap, so there is no seam. Same plate, one layer:",
        "dolly z went +/-1.80 m -> +/-2.04 m. The budget GREW.",
        "",
        "THE CLEAN PLATE (node 1b): paint the occluders out with your",
        "inpaint graph of choice (LaMa / SDXL / Photoshop). Atlas does not",
        "inpaint — that is delegated by design (GPL boundary). Use node 4's",
        "SAM3 mask as your inpaint mask; it is cut from the concepts the",
        "VLM named in node 2 (LM Studio + gemma-4-12b-qat preset).",
        "",
        "DOCTRINE (2026-07-19): the clean plate gets its OWN depth solve",
        "(node 9). Never extend the original's far band to cover the",
        "removal — that puts the support at the cutoff and produces a",
        "vertical cliff with floating foreground under orbit.",
        "",
        "VERDICT: 11a vs 11b. The layered budget must GROW; if it does",
        "not, the layer's depth solve or matte is wrong — fix that before",
        "trusting the projection.",
    ]), size=(540, 560))

    doc = b.dump(OUT, {
        "ds": {"scale": 0.65, "offset": [0, 0]},
        "atlas_layered_segmentation_version": 1,
        "atlas_notes": "VLM->SAM3->cleanplate layered pipeline. Plates go in "
                       "ComfyUI/input/; the clean plate is artist-made "
                       "(external inpaint, GPL boundary).",
    })
    print(f"wrote {OUT.name}: {len(doc['nodes'])} nodes, {len(doc['links'])} links")


if __name__ == "__main__":
    main()
