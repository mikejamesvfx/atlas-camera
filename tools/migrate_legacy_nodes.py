"""Rewrite saved workflows off legacy-gated nodes onto their replacements.

`ATLAS_LEGACY_NODES` keeps a superseded node loadable for one migration cycle,
but a saved graph still names it. This tool does the rewire: it removes the
legacy node and splices in the replacement chain, preserving every link-graph
invariant the shipping-workflow tests enforce (both directions of each link,
the id counters, the workflow UUID, node positions).

    python tools/migrate_legacy_nodes.py examples/foo.json            # dry run
    python tools/migrate_legacy_nodes.py examples/foo.json --write

Deliberate safety properties:

* **dry-run by default** — `--write` is required to touch a file;
* **explicit filenames only, never a glob** over `examples/`, so a stray local
  workflow is never rewritten by accident;
* **hard skip on any path whose stem contains `-edit`** — those are personal
  working copies and must not be migrated or committed;
* widget defaults come from the LIVE ``INPUT_TYPES``, not a hardcoded list, so
  an appended widget cannot silently desync the emitted node;
* it refuses to emit a half-wired graph: if a required input of the chain
  cannot be resolved, the file is reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _widget_defaults(node_type: str) -> list:
    """Default widget row for a node, in declared order, from the live schema."""
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.mcp.comfy_http import is_widget

    mapping = {**reg.NODE_CLASS_MAPPINGS,
               **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS,
               **getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {})}
    cls = mapping[node_type]
    spec_types = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}
    out = []
    it = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for _name, spec in (it.get(section) or {}).items():
            if not is_widget(spec):
                continue
            tp = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(tp, list):
                out.append(opts.get("default", tp[0] if tp else ""))
            else:
                out.append(opts.get("default", spec_types.get(tp, "")))
    return out


def _input_slot(node: dict, name: str):
    for i, s in enumerate(node.get("inputs") or []):
        if s.get("name") == name:
            return i
    return None


#: legacy key -> replacement chain. Adding a legacy node here is a data change.
#:
#: ``widgets`` entries are {widget_name: value} overrides applied on top of the
#: live defaults, so an appended widget never shifts a hardcoded row.
MIGRATIONS: dict[str, dict] = {
    "AtlasLiveMeshRepair": {
        "chain": [
            {"type": "AtlasPlanarHolePatch",
             "widgets": {"layer": "*"}},
            {"type": "AtlasRetopologizeLayer",
             "widgets": {"layer": "*", "method": "off",
                         "boundary_smooth_iterations": 8}},
        ],
        # old input name -> (chain index, input name on that node)
        "input_map": {"solve": (0, "solve")},
        # old output slot -> (chain index, output slot)
        "output_map": {0: (1, 0)},
        "note": ("AtlasLiveMeshRepair swept the whole solve; AtlasPlanarHolePatch "
                 "with layer='*' does the same across every relief mesh, and the "
                 "retopo node carries the migrated boundary smoothing. hole_mask "
                 "is left unwired on purpose: that means 'every hole is a "
                 "candidate', which matches the old unscoped behaviour."),
    },
    "AtlasGroundMask": {
        "chain": [{"type": "AtlasGroundDepthMap", "widgets": {}}],
        "input_map": {"solve": (0, "solve")},
        "output_map": {0: (0, 1)},          # MASK -> ground_mask, slot 1
        "note": "AtlasGroundDepthMap output 1 is bit-identical to AtlasGroundMask.",
    },
}


def _widgets_for(entry: dict) -> list:
    """Live defaults with this migration's named overrides applied."""
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.mcp.comfy_http import is_widget

    node_type = entry["type"]
    values = _widget_defaults(node_type)
    overrides = entry.get("widgets") or {}
    if not overrides:
        return values
    cls = reg.NODE_CLASS_MAPPINGS[node_type]
    it = cls.INPUT_TYPES()
    names = [n for section in ("required", "optional")
             for n, spec in (it.get(section) or {}).items() if is_widget(spec)]
    for name, val in overrides.items():
        if name not in names:
            raise KeyError(f"{node_type} has no widget {name!r} (has {names})")
        values[names.index(name)] = val
    return values


def migrate_workflow(data: dict) -> tuple[dict, list[str]]:
    """Return (migrated_copy, notes). Raises ValueError if it cannot rewire."""
    import copy as _copy

    d = _copy.deepcopy(data)
    nodes = {n["id"]: n for n in d.get("nodes", [])}
    notes: list[str] = []
    targets = [n for n in d.get("nodes", []) if n.get("type") in MIGRATIONS]
    if not targets:
        return d, notes

    for old in targets:
        plan = MIGRATIONS[old["type"]]
        next_node = int(d.get("last_node_id", max(nodes) if nodes else 0))
        next_link = int(d.get("last_link_id", 0))

        # Resolve the old node's incoming links before removing anything.
        sources: dict[str, tuple[int, int]] = {}
        for name, _ in plan["input_map"].items():
            slot = _input_slot(old, name)
            lid = (old.get("inputs") or [{}])[slot].get("link") if slot is not None else None
            if lid is None:
                raise ValueError(
                    f"{old['type']} id{old['id']}: required input {name!r} is not "
                    "connected — refusing to emit a half-wired graph")
            link = next(l for l in d["links"] if l[0] == lid)
            sources[name] = (link[1], link[2])

        consumers = [l for l in d["links"] if l[1] == old["id"]]

        # Build the chain.
        created = []
        x, y = old.get("pos", [0, 0])[:2]
        for i, entry in enumerate(plan["chain"]):
            next_node += 1
            created.append({
                "id": next_node, "type": entry["type"],
                "pos": [x + i * 360, y], "size": [330, 170], "flags": {},
                "order": old.get("order", 0) + i, "mode": old.get("mode", 0),
                "inputs": [], "outputs": [],
                "properties": {"Node name for S&R": entry["type"]},
                "widgets_values": _widgets_for(entry),
            })

        from atlas_camera.comfy import node_registry as reg
        from atlas_camera.mcp.comfy_http import is_widget

        def _ensure_input(node, name, link_id):
            cls = reg.NODE_CLASS_MAPPINGS[node["type"]]
            it = cls.INPUT_TYPES()
            for section in ("required", "optional"):
                for n_, spec in (it.get(section) or {}).items():
                    if is_widget(spec) or n_ != name:
                        continue
                    node["inputs"].append({"name": name, "type": spec[0],
                                           "link": link_id})
                    return len(node["inputs"]) - 1
            raise ValueError(f"{node['type']} has no link input {name!r}")

        def _ensure_output(node, slot):
            cls = reg.NODE_CLASS_MAPPINGS[node["type"]]
            names = getattr(cls, "RETURN_NAMES", None) or cls.RETURN_TYPES
            while len(node["outputs"]) <= slot:
                i = len(node["outputs"])
                node["outputs"].append({"name": names[i], "type": cls.RETURN_TYPES[i],
                                        "links": [], "slot_index": i})
            return node["outputs"][slot]

        # Feed the chain head from the old node's sources.
        for name, (ci, in_name) in plan["input_map"].items():
            src_id, src_slot = sources[name]
            next_link += 1
            _ensure_input(created[ci], in_name, next_link)
            d["links"].append([next_link, src_id, src_slot, created[ci]["id"],
                               _input_slot(created[ci], in_name),
                               nodes[src_id]["outputs"][src_slot]["type"]])
            origin = nodes[src_id]["outputs"][src_slot]
            origin["links"] = [x for x in (origin.get("links") or [])
                               if x != (old.get("inputs") or [{}])[
                                   _input_slot(old, name)].get("link")]
            origin["links"].append(next_link)

        # Chain the internal links (each node's slot 0 into the next's solve).
        for i in range(len(created) - 1):
            next_link += 1
            out = _ensure_output(created[i], 0)
            _ensure_input(created[i + 1], "solve", next_link)
            out["links"].append(next_link)
            d["links"].append([next_link, created[i]["id"], 0, created[i + 1]["id"],
                               _input_slot(created[i + 1], "solve"), out["type"]])

        # Re-point the old node's consumers at the chain tail.
        for lid, _sid, sslot, tid, tslot, *rest in consumers:
            ci, cslot = plan["output_map"][sslot]
            next_link += 1
            out = _ensure_output(created[ci], cslot)
            out["links"].append(next_link)
            d["links"].append([next_link, created[ci]["id"], cslot, tid, tslot,
                               out["type"]])
            nodes[tid]["inputs"][tslot]["link"] = next_link

        # Drop the old node and every link that touched it.
        dead = {l[0] for l in d["links"] if l[1] == old["id"] or l[3] == old["id"]}
        d["links"] = [l for l in d["links"] if l[0] not in dead]
        d["nodes"] = [n for n in d["nodes"] if n["id"] != old["id"]] + created
        d["last_node_id"] = next_node
        d["last_link_id"] = next_link
        nodes = {n["id"]: n for n in d["nodes"]}
        notes.append(f"{old['type']} id{old['id']} -> "
                     + " -> ".join(c["type"] for c in created)
                     + f"; {plan['note']}")
    return d, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="workflow JSON files (never a glob)")
    ap.add_argument("--write", action="store_true",
                    help="actually rewrite (default is a dry run)")
    args = ap.parse_args()

    changed = 0
    for raw in args.paths:
        p = Path(raw)
        if "-edit" in p.stem:
            print(f"  SKIP {p.name}: personal working copy (-edit)")
            continue
        if not p.is_file():
            print(f"  SKIP {p.name}: not a file")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        try:
            out, notes = migrate_workflow(data)
        except ValueError as exc:
            print(f"  REFUSED {p.name}: {exc}")
            continue
        if not notes:
            print(f"  --   {p.name}: no legacy nodes")
            continue
        for n in notes:
            print(f"  {'OK  ' if args.write else 'WOULD'} {p.name}: {n}")
        if args.write:
            p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        changed += 1

    verb = "migrated" if args.write else "would migrate"
    print(f"\n{verb} {changed} file(s)"
          + ("" if args.write else " — re-run with --write to apply"))


if __name__ == "__main__":
    main()
