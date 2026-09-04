"""Repair positional-widget drift in shipped ComfyUI workflows.

ComfyUI serializes each node's widgets into a POSITIONAL ``widgets_values``
array. When an Atlas node gains an APPENDED widget (the positional-widgets rule
every node follows), older saved workflows fall out of sync: ComfyUI silently
pads the tail with defaults on load, so nothing visibly breaks and the drift
accumulates unnoticed until a strict validator (the MCP, or another install's
loader) flags it. A Mac reviewer hit exactly this on the arm64 pass.

Because the rule is APPEND-ONLY, the existing ``widgets_values`` are correct for
positions ``[0:got]`` and the fix is purely to append the defaults for the new
trailing widgets ``[got:want]`` — derived from the node's live ``INPUT_TYPES``
(the same ``is_widget`` rule ComfyUI + the MCP validator use). A middle-inserted
widget would NOT be append-only and is reported, never silently "fixed".

A save carries that widget list TWICE, and both drift: positionally in
``widgets_values``, and again as converted-widget SOCKETS in ``inputs``
(``{"name", "type", "link": null, "widget": {"name"}}``). Topping up only the
first leaves the two disagreeing — invisible in ComfyUI, which rebuilds widgets
from the class either way, but a strict reader compares them and it had to be
repaired by hand twice (2026-09-04) before this tool learned to do it. So both
are appended here, in the same append-only spirit:

* a node serializing ZERO widget sockets is a valid compact save, not drift —
  writing a full list there would be a rewrite, so it is left alone;
* an existing socket list must be a PREFIX of the class's widget order, or the
  drift is not append-only and is reported rather than "fixed";
* new sockets go at the END of ``inputs``, which is where they belong precisely
  because the widget that caused the drift was appended last in ``INPUT_TYPES``.

Format is preserved byte-for-byte (see detect_format) so a widget top-up lands
as a tiny diff, not a whole-file reserialization.

    python tools/fix_workflow_widget_drift.py --check examples/**/*.json
    python tools/fix_workflow_widget_drift.py examples/showcase/foo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Import the checkout this tool LIVES in, not a stray editable install that may
# point at a different worktree/branch. Run as a script, sys.path[0] is tools/,
# so a bare `import atlas_camera` would resolve to site-packages — which on a dev
# box is commonly an editable install pinned to the MAIN checkout while you work
# in a git worktree. That silently validated workflows against the wrong node
# signatures (found live: SAM3 read as 3 widgets from the main checkout while
# this worktree's node has 5). Prepending the repo root makes co-located source
# win. tools/ is also added for the sibling port-script import.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_camera.comfy import node_registry as reg  # noqa: E402
from atlas_camera.mcp.comfy_http import is_widget  # noqa: E402
from port_sam3segment_to_atlas import detect_format  # noqa: E402

ATLAS = {**reg.NODE_CLASS_MAPPINGS, **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS}


#: LiteGraph's socket type for a widget input. A combo serializes as "COMBO"
#: whatever its entries are; everything else keeps ComfyUI's own type name.
_LITEGRAPH_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN")


@dataclass(frozen=True)
class WidgetSlot:
    """One ``widgets_values`` position.

    ``name``/``litegraph_type`` are None for the phantom ``control_after_generate``
    slot every seed carries: it occupies a value position but is NOT a socket, so
    it must be counted for ``widgets_values`` and skipped for ``inputs``. Keeping
    both in one ordered list is what stops the two repairs drifting apart.
    """

    name: str | None
    litegraph_type: str | None
    default: object


def widget_slots(cls) -> list[WidgetSlot]:
    """Ordered widget slots as ComfyUI serializes them: required then optional
    in declaration order, one per widget input, plus the phantom seed slot."""
    it = cls.INPUT_TYPES()
    out: list[WidgetSlot] = []
    for sec in ("required", "optional"):
        for name, spec in (it.get(sec) or {}).items():
            if not is_widget(spec):
                continue
            typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            opts = (spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1
                    and isinstance(spec[1], dict) else {})
            if isinstance(typ, (list, tuple)):          # combo -> default or first entry
                default = opts.get("default", typ[0] if typ else None)
                lg = "COMBO"
            else:
                default = opts.get("default")
                if default is None:
                    default = {"INT": 0, "FLOAT": 0.0, "STRING": "",
                               "BOOLEAN": False}.get(typ, "")
                lg = typ if typ in _LITEGRAPH_TYPES else str(typ)
            out.append(WidgetSlot(name, lg, default))
            if name in ("seed", "noise_seed"):
                out.append(WidgetSlot(None, None, "fixed"))
    return out


def widget_defaults(cls) -> list:
    """Just the values, in ``widgets_values`` order (the sibling tools' entry
    point — kept as-is so this refactor is invisible to them)."""
    return [slot.default for slot in widget_slots(cls)]


def _fix_widget_sockets(node: dict, slots: list[WidgetSlot]) -> tuple[bool, str]:
    """Append the missing converted-widget sockets. Returns (changed, error)."""
    named = [s for s in slots if s.name is not None]
    inputs = node.get("inputs")
    if not isinstance(inputs, list):
        return False, ""
    have = [i["widget"]["name"] for i in inputs
            if isinstance(i, dict) and isinstance(i.get("widget"), dict)
            and "name" in i["widget"]]
    if not have:
        # Zero sockets is a valid compact save, not drift (see module docstring).
        return False, ""
    if len(have) > len(named):
        return False, (f"{len(have)} widget sockets > {len(named)} widgets")
    if have != [s.name for s in named[:len(have)]]:
        return False, ("widget sockets are not a prefix of the class's widget "
                       "order — NOT append-only, needs a manual re-save")
    if len(have) == len(named):
        return False, ""
    for slot in named[len(have):]:
        inputs.append({"name": slot.name, "type": slot.litegraph_type,
                       "link": None, "widget": {"name": slot.name}})
    return True, ""


def fix_graph(graph: dict) -> tuple[int, list[str]]:
    """Append missing tail widgets in place, in BOTH representations. Returns
    (nodes_fixed, hard_errors). A node longer than its signature is a
    non-append-only drift and is reported, not touched."""
    fixed, errors = 0, []
    for node in graph.get("nodes", []):
        cls = ATLAS.get(node.get("type"))
        if cls is None:
            continue                       # third-party / core — not ours to judge
        slots = widget_slots(cls)
        want = [s.default for s in slots]
        got = node.get("widgets_values")
        changed = False

        if isinstance(got, list):          # some nodes serialize a dict; leave alone
            if len(got) > len(want):
                errors.append(
                    f"{node['type']} id{node['id']}: {len(got)} > {len(want)} "
                    f"widgets — NOT append-only, needs a manual re-save")
                continue
            if len(got) < len(want):
                node["widgets_values"] = got + want[len(got):]
                changed = True

        sockets_changed, err = _fix_widget_sockets(node, slots)
        if err:
            errors.append(f"{node['type']} id{node['id']}: {err}")
            continue
        if sockets_changed or changed:
            fixed += 1
    return fixed, errors


def process(path: Path, *, check: bool) -> tuple[int, list[str]]:
    raw = path.read_text(encoding="utf-8")
    graph = json.loads(raw)
    if not isinstance(graph, dict) or "nodes" not in graph:
        return 0, []
    indent, ensure_ascii, trailing = detect_format(raw, graph)
    fixed, errors = fix_graph(graph)
    if fixed and not check:
        out = json.dumps(graph, indent=indent, ensure_ascii=ensure_ascii)
        path.write_text(out + ("\n" if trailing else ""), encoding="utf-8")
    return fixed, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing (exit 1 if any remains)")
    args = ap.parse_args(argv)

    total_fixed, any_error, drifted = 0, False, 0
    for path in args.paths:
        if not path.is_file():
            continue
        try:
            fixed, errors = process(path, check=args.check)
        except json.JSONDecodeError as exc:
            print(f"  SKIP  {path.name}: not JSON ({exc})")
            continue
        total_fixed += fixed
        if fixed:
            drifted += 1
            verb = "would top up" if args.check else "topped up"
            print(f"  {'--' if args.check else 'OK'}  {path.name}: {verb} {fixed} node(s)")
        for e in errors:
            any_error = True
            print(f"  FAIL  {path.name}: {e}")
    verb = "would fix" if args.check else "fixed"
    print(f"\n{verb} {total_fixed} node row(s) across {drifted} file(s)")
    if any_error:
        return 1
    if args.check and total_fixed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
