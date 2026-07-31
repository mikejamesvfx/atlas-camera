"""Emit a ComfyUI **UI-format** workflow with its links correct by construction.

Hand-authoring these is how the shipped set acquired its drift bugs. The format
is redundantly linked — the top-level ``links`` array AND every node's
``inputs[].link`` / ``outputs[].links`` must agree — and ``widgets_values`` is
POSITIONAL, so a node that has gained an appended widget silently shifts every
value after it. Both failure modes load without complaint and misbehave later.

So nothing here is typed by hand:

* widget order and defaults are read from a LIVE server's ``/object_info``,
  which is the same source ComfyUI itself uses, so an appended widget can never
  put this generator out of step with the node;
* links are recorded once and written to all three places from that record.

Usage::

    python tools/build_example_workflow.py --spec examples_src/quickstart.py

or import ``Builder`` and drive it from Python. Validate the result with
``tests/test_example_workflows.py`` and then RUN it —
``tools/workflow_benchmark.py`` — because loading is not acceptance. Two live
bugs on 2026-07-31 (a combo value that never reached GeoCalib, an OpenCV shape
that differs by build) passed every offline check and died on execution.
"""
from __future__ import annotations

import json
import urllib.request
import uuid
from typing import Any

DEFAULT_SERVER = "http://127.0.0.1:8188"


def fetch_object_info(server: str = DEFAULT_SERVER) -> dict[str, Any]:
    with urllib.request.urlopen(server.rstrip("/") + "/object_info", timeout=30) as fh:
        return json.load(fh)


def _widget_names_and_defaults(spec: dict[str, Any]) -> list[tuple[str, Any]]:
    """Widget slots in ComfyUI's own order: required first, then optional.

    A widget is an input with an inline config rather than a link type. The
    order here IS the positional order of ``widgets_values``; deriving it from
    the server rather than from a local table is the whole point.
    """
    out: list[tuple[str, Any]] = []
    for section in ("required", "optional"):
        for name, entry in (spec.get("input", {}).get(section) or {}).items():
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            kind = entry[0]
            cfg = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if cfg.get("forceInput"):
                continue                      # a link, never a widget
            if isinstance(kind, list):        # combo
                out.append((name, cfg.get("default", kind[0] if kind else None)))
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out.append((name, cfg.get("default")))
    return out


class Builder:
    """Accumulate nodes and links, then emit a consistent UI workflow."""

    def __init__(self, object_info: dict[str, Any]):
        self.oi = object_info
        self.nodes: list[dict[str, Any]] = []
        self.links: list[list[Any]] = []
        self._next_node = 1
        self._next_link = 1
        self._by_ref: dict[str, dict[str, Any]] = {}

    def add(self, ref: str, class_type: str, *, widgets: dict[str, Any] | None = None,
            pos: tuple[int, int] = (0, 0), title: str | None = None) -> str:
        spec = self.oi.get(class_type)
        if spec is None:
            raise KeyError(
                f"{class_type} is not registered on the server. Experimental "
                "nodes need ATLAS_EXPERIMENTAL=1; a typo looks identical here.")
        slots = _widget_names_and_defaults(spec)
        values = []
        unknown = set(widgets or {}) - {n for n, _ in slots}
        if unknown:
            raise KeyError(
                f"{class_type}: no widget named {sorted(unknown)}. Present: "
                f"{[n for n, _ in slots]}")
        # Validate combo VALUES, not just widget names. Caught live on the first
        # generated workflow: height_mode="auto" is not in the combo, and the
        # name check passed it straight through to a server rejection. A
        # generator that only checks names reproduces the bug it exists to stop.
        combos = {}
        for section in ("required", "optional"):
            for nm, entry in (spec.get("input", {}).get(section) or {}).items():
                if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], list):
                    combos[nm] = list(entry[0])
        for nm, val in (widgets or {}).items():
            if nm in combos and val not in combos[nm]:
                raise ValueError(
                    f"{class_type}.{nm}: {val!r} is not a valid choice. "
                    f"Allowed: {combos[nm]}")
        for name, default in slots:
            values.append((widgets or {}).get(name, default))

        link_inputs = []
        for section in ("required", "optional"):
            for name, entry in (spec.get("input", {}).get(section) or {}).items():
                if not isinstance(entry, (list, tuple)) or not entry:
                    continue
                kind = entry[0]
                cfg = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
                if isinstance(kind, list) or (
                        kind in ("INT", "FLOAT", "STRING", "BOOLEAN")
                        and not cfg.get("forceInput")):
                    continue
                link_inputs.append({"name": name, "type": kind, "link": None})

        outputs = [{"name": n, "type": t, "links": []} for n, t in
                   zip(spec.get("output_name") or [], spec.get("output") or [])]
        node = {
            "id": self._next_node, "type": class_type,
            "pos": list(pos), "size": [340, 120], "flags": {}, "order": len(self.nodes),
            "mode": 0, "inputs": link_inputs, "outputs": outputs,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": values,
        }
        if title:
            node["title"] = title
        self._next_node += 1
        self.nodes.append(node)
        self._by_ref[ref] = node
        return ref

    def link(self, src_ref: str, src_out: str, dst_ref: str, dst_in: str) -> None:
        s, d = self._by_ref[src_ref], self._by_ref[dst_ref]
        si = next((i for i, o in enumerate(s["outputs"]) if o["name"] == src_out), None)
        if si is None:
            raise KeyError(f"{s['type']} has no output {src_out!r}; has "
                           f"{[o['name'] for o in s['outputs']]}")
        di = next((i for i, o in enumerate(d["inputs"]) if o["name"] == dst_in), None)
        if di is None:
            raise KeyError(f"{d['type']} has no link input {dst_in!r}; has "
                           f"{[o['name'] for o in d['inputs']]}")
        lid = self._next_link
        self._next_link += 1
        # written to all three places from one record — the redundancy is the
        # format's, not ours, and this is the only place it is resolved
        s["outputs"][si]["links"].append(lid)
        d["inputs"][di]["link"] = lid
        self.links.append([lid, s["id"], si, d["id"], di, s["outputs"][si]["type"]])

    def build(self, *, revision: int = 0, workflow_id: str | None = None) -> dict[str, Any]:
        """`workflow_id` must be UNIQUE per file. A shared id makes ComfyUI treat
        two workflows as the same document; tests/test_example_workflows pins it,
        and caught a hardcoded constant here that collided the moment a second
        workflow was generated."""
        return {
            "id": workflow_id or str(uuid.uuid4()),
            "revision": revision, "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4,
        }
