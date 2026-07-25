"""Generate the unseen-geometry test workflow.

Widget values are read from each node's live ``INPUT_TYPES`` rather than being
transcribed by hand, so the positional ``widgets_values`` arrays cannot drift
out of order — the exact failure mode that made saved workflows load values
into the wrong parameters in the AtlasInput/DeriveGeometry signature bugs.

Run: python tools/build_unseen_geometry_workflow.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid

# Run as `python tools/build_...py`, sys.path[0] is tools/, so `atlas_camera`
# would resolve from whatever the editable install points at — which in a git
# worktree is the MAIN checkout, silently generating the workflow against the
# wrong node definitions. Pin it to this tree.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atlas_camera.comfy import node_registry as R  # noqa: E402

PLATE = "atlas_seacliff_castle.png"
OUT = pathlib.Path(__file__).resolve().parents[1] / "examples" / \
    "atlas_unseen_geometry_test_workflow.json"


def widget_defaults(node_type: str) -> list:
    """Default widget values, in declared order (required then optional).

    Only entries ComfyUI renders as widgets are included — a socket-only input
    (one whose spec is a bare type with no default) contributes nothing.
    """
    cls = R.NODE_CLASS_MAPPINGS[node_type]
    spec = cls.INPUT_TYPES()
    out = []
    for section in ("required", "optional"):
        for name, decl in (spec.get(section) or {}).items():
            if not isinstance(decl, (tuple, list)) or not decl:
                continue
            kind, opts = decl[0], (decl[1] if len(decl) > 1 else {})
            if isinstance(kind, (list, tuple)):          # COMBO
                out.append(opts.get("default", kind[0] if kind else ""))
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out.append(opts.get("default", 0))
            # ATLAS_*/IMAGE/MASK are links, not widgets.
    return out


class Builder:
    def __init__(self):
        self.nodes, self.links = [], []
        self.nid, self.lid = 0, 0

    def add(self, node_type, pos, title, *, widgets=None, size=(340, 130)):
        self.nid += 1
        self.nodes.append({
            "id": self.nid, "type": node_type, "pos": list(pos),
            "size": list(size), "flags": {}, "order": self.nid - 1, "mode": 0,
            "inputs": [], "outputs": [],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets if widgets is not None
            else widget_defaults(node_type),
            "title": title,
        })
        return self.nid

    def note(self, pos, title, text, size=(420, 260)):
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
             src_name=None, dst_name=None):
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
        d["inputs"][dst_slot].update(
            {"name": dst_name or type_name.lower(), "type": type_name,
             "link": self.lid})
        return self.lid

    def _node(self, nid):
        return next(n for n in self.nodes if n["id"] == nid)

    def validate(self):
        """Every link's output type must equal its input type.

        ComfyUI does not coerce between socket types — it refuses the
        connection at load time with "Invalid connection", which is how a
        MASK wired straight into PreviewImage's IMAGE input shipped once.
        A generator that emits links is exactly where that belongs.
        """
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

    load = b.add("LoadImage", (60, 260), "1 · SEACLIFF CASTLE PLATE",
                 widgets=[PLATE, "image"], size=(320, 314))
    inp = b.add("AtlasInput", (430, 260), "2 · SOLVE + DEPTH + RELIEF")

    walls = b.add("AtlasDeriveWalls", (800, 200), "3 · FIT PLANES + OBJECTS")
    merge = b.add("AtlasMergeGeometry", (800, 420),
                  "4 · MERGE fitted planes + relief mesh")

    graph = b.add("AtlasOcclusionGraph", (1160, 200),
                  "5 · \U0001f578 WHAT OCCLUDES WHAT")
    budget = b.add("AtlasMoveBudget", (1160, 430),
                   "6 · \U0001f4d0 HOW FAR CAN THE CAMERA GO")
    # AtlasCompleteDepth is deliberately absent — see the READ ME note. It is
    # experimental, so a stock install would fail to load a graph containing it.
    view = b.add("AtlasBlockoutViewport", (1560, 260), "7 · VIEWPORT",
                 size=(480, 640))

    b.link(load, 0, inp, 0, "IMAGE", src_name="IMAGE", dst_name="image")
    # AtlasInput: 0 solve, 1 image, 2 depth, 3 sky_mask, 4 report
    b.link(inp, 0, walls, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(inp, 2, walls, 1, "ATLAS_DEPTH_MAP", src_name="depth", dst_name="depth")
    # Derive nodes clobber prior PROXY_ROLE geometry, so the relief mesh from
    # AtlasInput is merged back in explicitly rather than assumed to survive.
    b.link(walls, 0, merge, 0, "ATLAS_SOLVE", src_name="ATLAS_SOLVE",
           dst_name="solve_a")
    b.link(inp, 0, merge, 1, "ATLAS_SOLVE", src_name="solve", dst_name="solve_b")

    b.link(merge, 0, graph, 0, "ATLAS_SOLVE", src_name="ATLAS_SOLVE",
           dst_name="solve")
    b.link(inp, 2, graph, 1, "ATLAS_DEPTH_MAP", src_name="depth", dst_name="depth")

    b.link(graph, 0, budget, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(budget, 0, view, 0, "ATLAS_SOLVE", src_name="solve", dst_name="solve")
    b.link(load, 0, view, 1, "IMAGE", src_name="IMAGE", dst_name="source_image")

    b.note((60, 640), "READ ME · what this workflow is testing", "\n".join([
        "UNSEEN GEOMETRY TEST — seacliff castle plate.",
        "",
        "5 🕸 OCCLUSION GRAPH  what occludes what, and what may be built.",
        "   Read its report FIRST. A tear it cannot classify licenses NOTHING",
        "   (policy=none) and says so — that refusal is the design, not a bug.",
        "   Needs fitted primitives upstream: node 3 supplies them, and node 4",
        "   merges the relief mesh back, because derive nodes CLOBBER prior",
        "   PROXY_ROLE geometry.",
        "",
        "6 📐 MOVE BUDGET  how far the camera can go before a tear opens.",
        "   Sealed-minus-covered, relative to the source view, so neither the",
        "   backdrop nor pre-existing tearing can mask it.",
        "   pan/tilt unbounded is CORRECT: rotation about the optical centre",
        "   produces no parallax, so it cannot open a tear.",
        "   Author a camera path, wire it in, push past the reported limit —",
        "   tearing should return where the number said it would.",
        "",
        "NOT INCLUDED: 🩹 AtlasCompleteDepth is EXPERIMENTAL and is left",
        "out on purpose. It fills tears exactly from graph-fitted planes, but",
        "the fill butts against measured depth at the hole rim, so re-meshing",
        "tears along every seam. Measured on a real 4K plate: 302k px closed,",
        "222k px of NEW rim tears, and the move budget got WORSE.",
        "Set ATLAS_EXPERIMENTAL=1 to try it, and measure with node 6 first.",
        "The clean-plate layer path (per-segment plates, shared solve) avoids",
        "the seam entirely and is the intended fix.",
    ]), size=(520, 620))

    doc = b.dump(OUT, {
        "ds": {"scale": 0.75, "offset": [0, 0]},
        "atlas_unseen_geometry_version": 1,
        "atlas_notes": "Test workflow for the occlusion graph / move budget / "
                       "depth completion track. Plate must be in ComfyUI/input/.",
    })
    print(f"wrote {OUT.name}: {len(doc['nodes'])} nodes, {len(doc['links'])} links")


if __name__ == "__main__":
    main()
