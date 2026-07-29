"""Give a shipped workflow something you can actually LOOK at.

Most Atlas workflows end in `AtlasBlockoutViewport`, which renders in the
BROWSER via three.js. Interactively that is the product; but run headlessly —
by the MCP server, by an agent, by tools/workflow_benchmark.py — the graph
produces no image at all, so there is nothing to inspect when it goes wrong.

This appends a debug tail to the graph's terminal solve:

    AtlasStereoRender -> PreviewImage      the geometry, rendered server-side
    AtlasMoveBudget   -> ShowText          the camera envelope and tear numbers

PreviewImage rather than SaveImage on purpose: debugging should not silently
fill the output directory with files nobody asked for. Swap with --save if you
want them on disk.

The nodes are added MUTED (mode 2) by default, so a shipped graph costs exactly
what it did before until someone wants them — a stereo render on a 36 MP plate
is not free, and a debug aid that slows every run is one people delete.
Un-mute in ComfyUI (Ctrl-M) to switch them on.

    python tools/add_debug_tail.py examples/atlas_input_quickstart_workflow.json
    python tools/add_debug_tail.py --all --active     # added switched ON
    python tools/add_debug_tail.py --all --check      # report, change nothing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

TAIL_TITLE = "🔬 DEBUG"
MUTED, ACTIVE = 2, 0


#: The lightweight teaching graphs. tests/test_example_workflows pins their
#: node-type set EXACTLY, deliberately — the guard exists so the quickstart
#: cannot drift back to a Nuke/Maya-only handoff. Debug clutter in a workflow
#: whose whole job is to be minimal is the wrong trade, and widening someone
#: else's deliberate contract test to make room for it is worse.
SKIP_PINNED = {
    "atlas_input_quickstart_workflow",
    "atlas_input_quickstart_agentic_assessment_workflow",
    "atlas_occlusion_cull_quickstart_workflow",
    "atlas_occlusion_cull_quickstart_agentic_assessment_workflow",
}


def shipped() -> list:
    return sorted(p for p in EXAMPLES.glob("*.json")
                  if "-edit" not in p.stem and p.stem not in SKIP_PINNED)


def widget_defaults(class_type: str) -> list:
    """Widget defaults straight from the live node class.

    Emitting an empty list instead trips tests/test_shipping_workflow_widgets —
    positional `widgets_values` is a saved-workflow contract, so a node written
    with the wrong arity is drift the moment it lands.
    """
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.mcp.comfy_http import is_widget

    cls = reg.NODE_CLASS_MAPPINGS.get(class_type)
    if cls is None:
        return []
    out = []
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, entry in (spec.get(section) or {}).items():
            if not is_widget(entry):
                continue
            cfg = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            default = cfg.get("default")
            if default is None:
                default = entry[0][0] if isinstance(entry[0], list) and entry[0] else ""
            out.append(default)
            if name in ("seed", "noise_seed"):
                out.append("fixed")
    return out


def has_tail(doc: dict) -> bool:
    return any(str(n.get("title", "")).startswith(TAIL_TITLE) for n in doc["nodes"])


def terminal_solve(doc: dict):
    """The last ATLAS_SOLVE nothing else consumes, plus an IMAGE to texture with."""
    consumed = {(l[1], l[2]) for l in doc.get("links", [])}
    solve = image = None
    for n in doc["nodes"]:
        for slot, o in enumerate(n.get("outputs") or []):
            if o.get("type") == "ATLAS_SOLVE" and (n["id"], slot) not in consumed:
                solve = (n["id"], slot)
            if o.get("type") == "IMAGE" and image is None:
                image = (n["id"], slot)
    if solve is None:                      # consumed everywhere; take any solve
        for n in doc["nodes"]:
            for slot, o in enumerate(n.get("outputs") or []):
                if o.get("type") == "ATLAS_SOLVE":
                    solve = (n["id"], slot)
    return solve, image


def add_tail(doc: dict, *, active: bool = False, save: bool = False) -> str:
    solve, image = terminal_solve(doc)
    if solve is None:
        return "no ATLAS_SOLVE — nothing to debug"
    if has_tail(doc):
        return "already has a debug tail"

    mode = ACTIVE if active else MUTED
    nid = max(n["id"] for n in doc["nodes"]) + 1
    lid = max((l[0] for l in doc.get("links", [])), default=0) + 1
    x = max((n["pos"][0] for n in doc["nodes"]), default=0) + 460
    y = max((n["pos"][1] for n in doc["nodes"]), default=0) - 200

    def node(node_id, cls, pos, size, title, inputs, outputs=None, widgets=None):
        doc["nodes"].append({
            "id": node_id, "type": cls, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(doc["nodes"]), "mode": mode,
            "inputs": inputs, "outputs": outputs or [],
            "properties": {"Node name for S&R": cls},
            "widgets_values": (widgets if widgets is not None
                               else widget_defaults(cls)),
            "title": title, "color": "#323", "bgcolor": "#535",
        })

    def link(src, dst_id, dst_slot, type_name):
        nonlocal lid
        doc["links"].append([lid, src[0], src[1], dst_id, dst_slot, type_name])
        for n in doc["nodes"]:
            if n["id"] == src[0]:
                out = n["outputs"][src[1]]
                # Saved graphs write `"links": null` for an unconnected output,
                # so setdefault is not enough — the key exists and holds None.
                if not isinstance(out.get("links"), list):
                    out["links"] = []
                out["links"].append(lid)
            if n["id"] == dst_id:
                n["inputs"][dst_slot]["link"] = lid
        lid += 1

    made = []
    node(nid, "AtlasMoveBudget", (x, y), (400, 200),
         f"{TAIL_TITLE} move budget",
         [{"name": "solve", "type": "ATLAS_SOLVE", "link": None},
          {"name": "camera_path", "type": "ATLAS_CAMERA_PATH", "link": None}],
         [{"name": "solve", "type": "ATLAS_SOLVE", "links": []},
          {"name": "report", "type": "STRING", "links": []}], widgets=None)
    link(solve, nid, 0, "ATLAS_SOLVE")
    node(nid + 1, "ShowText|pysssss", (x + 440, y), (400, 200),
         f"{TAIL_TITLE} budget report",
         [{"name": "text", "type": "STRING", "link": None}])
    link((nid, 1), nid + 1, 0, "STRING")
    made.append("move budget")

    if image is not None:
        node(nid + 2, "AtlasStereoRender", (x, y + 240), (400, 220),
             f"{TAIL_TITLE} server-side render",
             [{"name": "solve", "type": "ATLAS_SOLVE", "link": None},
              {"name": "source_image", "type": "IMAGE", "link": None}],
             [{"name": "stereo", "type": "IMAGE", "links": []},
              {"name": "left", "type": "IMAGE", "links": []},
              {"name": "right", "type": "IMAGE", "links": []},
              {"name": "report", "type": "STRING", "links": []}],
             widgets=None)
        link(solve, nid + 2, 0, "ATLAS_SOLVE")
        link(image, nid + 2, 1, "IMAGE")
        out_cls = "SaveImage" if save else "PreviewImage"
        node(nid + 3, out_cls, (x + 440, y + 240), (400, 340),
             f"{TAIL_TITLE} geometry preview",
             [{"name": "images", "type": "IMAGE", "link": None}],
             widgets=["atlas_debug"] if save else [])
        link((nid + 2, 0), nid + 3, 0, "IMAGE")
        made.append(f"stereo render -> {out_cls}")

    doc["last_node_id"] = max(n["id"] for n in doc["nodes"])
    doc["last_link_id"] = lid - 1
    return f"added {', '.join(made)} ({'active' if active else 'muted'})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--active", action="store_true",
                    help="add the tail switched ON (default: muted, zero cost)")
    ap.add_argument("--save", action="store_true",
                    help="SaveImage instead of PreviewImage")
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    targets = [Path(p) for p in args.paths] or (shipped() if args.all else [])
    if not targets:
        raise SystemExit("pass workflow paths or --all")

    if args.check:
        for p in targets:
            doc = json.loads(p.read_text(encoding="utf-8"))
            solve, image = terminal_solve(doc)
            state = ("has tail" if has_tail(doc) else
                     "no solve" if solve is None else
                     "would add" + ("" if image else " (budget only — no IMAGE)"))
            print(f"  {p.stem:56} {state}")
        return

    # ALL OR NOTHING. Transform every graph in memory first and only write once
    # they all succeed — a crash partway through leaves a pile of half-edited
    # shipped workflows, which is exactly what happened the first time this ran
    # (18 files modified before it hit one with `"links": null`).
    staged, notes = [], []
    for p in targets:
        doc = json.loads(p.read_text(encoding="utf-8"))
        try:
            note = add_tail(doc, active=args.active, save=args.save)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"{p.name}: {type(exc).__name__}: {exc}\n"
                "nothing was written — fix and re-run") from exc
        notes.append((p.stem, note))
        if note.startswith("added"):
            staged.append((p, doc))

    for p, doc in staged:
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    for stem, note in notes:
        print(f"  {stem:56} {note}")
    print(f"\n  wrote {len(staged)} file(s)")


if __name__ == "__main__":
    main()
