"""HTTP + workflow plumbing for the Atlas MCP server.

The canonical home of the UI→API flattening, gate handling, validation and
queue/poll logic that `tools/run_ui_workflow.py` and
`tools/validate_ui_workflow.py` pioneered (those CLIs now import from here).
Standard library only — the MCP server must not drag torch/numpy into its own
process; ComfyUI stays the execution engine.

Conventions this module encodes (learned by running every Atlas workflow
headlessly — see docs/dev/archive/atlas_mcp_server_plan.md):

- KJNodes ``SetNode``/``GetNode`` rails are frontend-only virtual nodes and
  must be resolved through their rail name.
- Official ComfyUI v1 subgraphs are expanded through their ``-10``/``-20``
  boundaries before validation or UI-to-API conversion; instance proxy-widget
  values are applied to the cloned internal nodes.
- Muted nodes (``mode == 2``) drop out of the executed graph; bypassed nodes
  (``mode == 4``) forward their first connected same-type input.
- ``seed``/``noise_seed`` widgets carry a phantom ``control_after_generate``
  slot in ``widgets_values``; ``image_upload`` combos carry a phantom upload
  slot. Both must be skipped when mapping positional values to input names.
- Shipped workflows close their ``AtlasSolveGate`` (``proceed=False``);
  headless runs open them per-run via input overrides.
"""
from __future__ import annotations

import json
from copy import deepcopy
import time
import urllib.request
import uuid

PRIMS = {"INT", "FLOAT", "STRING", "BOOLEAN"}
VIRTUAL = {"SetNode", "GetNode", "Note", "MarkdownNote", "Reroute"}

DEFAULT_HOST = "127.0.0.1:8188"


def http_json(url: str, payload=None, timeout: int = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_object_info(host: str = DEFAULT_HOST, timeout: int = 120):
    return http_json(f"http://{host}/object_info", timeout=timeout)


def summarize_viewport_layer(source: dict) -> dict:
    """Compact one serialized ProjectionSource for MCP/debug consumers."""
    geometry = source.get("proxy_geometry") or []
    verts = sum(len(p.get("vertices") or []) for p in geometry) // 3
    metadata = [p.get("metadata") or {} for p in geometry]
    filled_cells = sum(int(m.get("n_filled_cells") or 0) for m in metadata)
    torn = [float(m["torn_fraction"]) for m in metadata
            if m.get("torn_fraction") is not None]
    stretch = [float(m["stretch_ratio_p95"]) for m in metadata
               if m.get("stretch_ratio_p95") is not None]
    return {
        "name": source.get("name"),
        "priority": source.get("priority"),
        "band_m": [source.get("near_m"), source.get("far_m")],
        "band_geometry": source.get("band_geometry"),
        "verts": verts,
        "n_filled_cells": filled_cells,
        "torn_fraction_max": max(torn) if torn else None,
        "stretch_ratio_p95_max": max(stretch) if stretch else None,
        "matte": bool(source.get("mask_b64")),
        "normal_map": bool(source.get("normal_map_b64")),
        "hidden_provenance": bool(source.get("hidden_mask_b64")),
    }


def is_widget(spec) -> bool:
    """Whether an input spec serializes into positional ``widgets_values``.

    Handles both the legacy combo form (type is a list of options) and the V3
    ``"COMBO"`` string form; ``forceInput`` demotes a primitive to a link.
    """
    t = spec[0]
    cfg = spec[1] if len(spec) > 1 else {}
    if isinstance(t, list):
        return True
    if t == "COMBO" or t in PRIMS:
        return not cfg.get("forceInput")
    return False


def spec_items(oi, type_):
    s = oi[type_]["input"]
    return [(k, v) for sec in ("required", "optional")
            for k, v in (s.get(sec) or {}).items()]


def widget_inputs(oi, type_, values):
    """Positional ``widgets_values`` → ``{input_name: literal}``."""
    out = {}
    vi = 0
    for k, spec in spec_items(oi, type_):
        if not is_widget(spec):
            continue
        if vi >= len(values):
            break
        out[k] = values[vi]
        vi += 1
        cfg = spec[1] if len(spec) > 1 else {}
        if k in ("seed", "noise_seed") or (isinstance(cfg, dict) and cfg.get("image_upload")):
            vi += 1
    return out


def _widget_positions(oi: dict, type_: str) -> dict[str, int]:
    """Map real and phantom widget names to ``widgets_values`` indexes."""
    out = {}
    index = 0
    for name, spec in spec_items(oi, type_):
        if not is_widget(spec):
            continue
        out[name] = index
        index += 1
        cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if name in ("seed", "noise_seed"):
            out["control_after_generate"] = index
            index += 1
        elif cfg.get("image_upload"):
            out["image_upload"] = index
            index += 1
    return out


def expand_subgraphs(ui: dict, oi: dict) -> dict:
    """Expand ComfyUI ``definitions.subgraphs`` into one ordinary UI graph.

    ComfyUI expands official subgraphs in the browser immediately before it
    serializes an API prompt.  Atlas's MCP runner receives the saved UI JSON
    directly, so it must perform the same boundary rewrite itself.  Internal
    node ids are definition-local; every instance gets a fresh deterministic
    numeric id.  Instance proxy widget values are copied back onto the cloned
    internal nodes before normal positional-widget mapping runs.

    This intentionally handles the stable v1 subgraph schema used by current
    ComfyUI (``-10`` input boundary, ``-20`` output boundary).  Nested
    subgraphs are rejected explicitly instead of producing a subtly corrupt
    prompt.
    """
    definitions = {
        item["id"]: item
        for item in ((ui.get("definitions") or {}).get("subgraphs") or [])
    }
    if not definitions:
        return ui

    top_nodes = {node["id"]: node for node in ui.get("nodes") or []}
    instances = {node_id: (node, definitions[node["type"]])
                 for node_id, node in top_nodes.items()
                 if node.get("type") in definitions}
    if not instances:
        return ui

    integer_ids = [node_id for node_id in top_nodes if isinstance(node_id, int)]
    next_node_id = max(integer_ids, default=0)
    expanded_nodes = [deepcopy(node) for node_id, node in top_nodes.items()
                      if node_id not in instances]
    id_maps: dict[object, dict[object, int]] = {}

    for instance_id, (instance, definition) in instances.items():
        mapping = {}
        clones = []
        for internal in definition.get("nodes") or []:
            if internal.get("type") in definitions:
                raise RuntimeError(
                    f"nested subgraph {internal['type']} inside {definition.get('name')} "
                    "is not supported by the Atlas MCP flattener")
            next_node_id += 1
            clone = deepcopy(internal)
            mapping[internal["id"]] = next_node_id
            clone["id"] = next_node_id
            clones.append(clone)

        # A subgraph instance stores only the values that are promoted to its
        # face.  Apply those to the definition clones exactly as the frontend
        # proxy-widget layer does.
        proxies = (instance.get("properties") or {}).get("proxyWidgets") or []
        values = instance.get("widgets_values") or []
        by_local_id = {node["id"]: node for node in clones}
        for proxy, value in zip(proxies, values):
            if not isinstance(proxy, (list, tuple)) or len(proxy) < 2:
                continue
            raw_id, widget_name = proxy[0], proxy[1]
            try:
                local_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            clone_id = mapping.get(local_id)
            clone = by_local_id.get(clone_id)
            if clone is None or clone.get("type") not in oi:
                continue
            position = _widget_positions(oi, clone["type"]).get(widget_name)
            if position is None:
                continue
            clone_values = clone.setdefault("widgets_values", [])
            if position >= len(clone_values):
                raise RuntimeError(
                    f"subgraph {definition.get('name')} proxy {widget_name!r} "
                    f"points past {clone['type']} widgets_values")
            clone_values[position] = value

        id_maps[instance_id] = mapping
        expanded_nodes.extend(clones)

    # The old links on cloned nodes refer to definition-local ids.  Rebuild
    # all link bookkeeping from the canonical top + definition link tables.
    expanded_by_id = {node["id"]: node for node in expanded_nodes}
    for node in expanded_nodes:
        for item in node.get("inputs") or []:
            item["link"] = None
        for item in node.get("outputs") or []:
            item["links"] = []

    top_links = {link[0]: link for link in ui.get("links") or []}
    incoming = {}
    for link in top_links.values():
        incoming[(link[3], link[4])] = link

    def definition_links(definition):
        return definition.get("links") or []

    def source_endpoint(node_id, slot, trail=()):
        if node_id not in instances:
            return node_id, slot
        if node_id in trail:
            raise RuntimeError("subgraph output resolution loop")
        instance, definition = instances[node_id]
        if instance.get("mode", 0) == 2:
            raise RuntimeError(
                f"subgraph instance {node_id} ({definition.get('name')}) is MUTED "
                "but something consumes its output")
        if instance.get("mode", 0) == 4:
            want = instance["outputs"][slot]["type"]
            candidate = next((i for i, item in enumerate(instance.get("inputs") or [])
                              if item.get("type") == want and item.get("link") is not None), None)
            if candidate is None:
                return None
            outer = incoming.get((node_id, candidate))
            return None if outer is None else source_endpoint(
                outer[1], outer[2], trail + (node_id,))
        boundary = [link for link in definition_links(definition)
                    if link.get("target_id") == -20
                    and link.get("target_slot") == slot]
        if len(boundary) != 1:
            raise RuntimeError(
                f"subgraph {definition.get('name')} output slot {slot} has "
                f"{len(boundary)} boundary links; expected exactly one")
        link = boundary[0]
        return id_maps[node_id][link["origin_id"]], link["origin_slot"]

    def target_endpoints(node_id, slot):
        if node_id not in instances:
            return [(node_id, slot)]
        instance, definition = instances[node_id]
        if instance.get("mode", 0) in (2, 4):
            return []
        boundary = [link for link in definition_links(definition)
                    if link.get("origin_id") == -10
                    and link.get("origin_slot") == slot]
        if not boundary:
            raise RuntimeError(
                f"subgraph {definition.get('name')} input slot {slot} has no boundary link")
        return [(id_maps[node_id][link["target_id"]], link["target_slot"])
                for link in boundary]

    rebuilt_links = []
    next_link_id = 0

    def add_link(source_id, source_slot, target_id, target_slot, type_):
        nonlocal next_link_id
        if source_id not in expanded_by_id or target_id not in expanded_by_id:
            raise RuntimeError(
                f"expanded subgraph link has missing endpoint {source_id} -> {target_id}")
        next_link_id += 1
        link = [next_link_id, source_id, source_slot, target_id, target_slot, type_]
        rebuilt_links.append(link)
        expanded_by_id[source_id]["outputs"][source_slot].setdefault("links", []).append(next_link_id)
        expanded_by_id[target_id]["inputs"][target_slot]["link"] = next_link_id

    # Definition-internal links do not cross a boundary and can be copied
    # directly once their scoped ids have been remapped.
    for instance_id, (_, definition) in instances.items():
        mapping = id_maps[instance_id]
        for link in definition_links(definition):
            if link.get("origin_id") in (-10, -20) or link.get("target_id") in (-10, -20):
                continue
            add_link(mapping[link["origin_id"]], link["origin_slot"],
                     mapping[link["target_id"]], link["target_slot"], link["type"])

    # Each top-level link may terminate at multiple internal consumers because
    # one promoted subgraph input can fan out inside its definition.
    for link in ui.get("links") or []:
        source = source_endpoint(link[1], link[2])
        if source is None:
            continue
        for target_id, target_slot in target_endpoints(link[3], link[4]):
            add_link(source[0], source[1], target_id, target_slot, link[5])

    expanded = deepcopy(ui)
    expanded["nodes"] = expanded_nodes
    expanded["links"] = rebuilt_links
    expanded["last_node_id"] = next_node_id
    expanded["last_link_id"] = next_link_id
    expanded.pop("definitions", None)
    return expanded


def ui_to_api(ui: dict, oi: dict) -> dict:
    """Flatten a UI-format workflow to the API format ``/prompt`` executes."""
    ui = expand_subgraphs(ui, oi)
    nodes = {n["id"]: n for n in ui["nodes"]}
    links = {l[0]: l for l in ui["links"]}

    def upstream(link_id):
        seen = 0
        while link_id is not None:
            seen += 1
            if seen > 100:
                raise RuntimeError("link resolution loop")
            _, sid, sslot, _, _, _ = links[link_id]
            src = nodes[sid]
            mode = src.get("mode", 0)
            if src["type"] == "GetNode":
                rail = src["widgets_values"][0]
                setter = next(n for n in ui["nodes"]
                              if n["type"] == "SetNode"
                              and n["widgets_values"][0] == rail)
                link_id = setter["inputs"][0]["link"]
                continue
            if src["type"] == "Reroute":
                link_id = src["inputs"][0]["link"]
                continue
            if mode == 4:
                want = src["outputs"][sslot]["type"]
                nxt = next((i["link"] for i in src.get("inputs", [])
                            if i.get("link") is not None
                            and (i.get("type") == want or want == "*")), None)
                if nxt is None:
                    # A bypassed SOURCE node (e.g. a bypassed LoadImage) has
                    # nothing to forward — the real frontend simply DROPS the
                    # downstream link (an optional input reverts to its
                    # default; a required one fails /prompt validation, which
                    # mirrors what the browser does). Match that.
                    return None
                link_id = nxt
                continue
            if mode == 2:
                raise RuntimeError(
                    f"node {sid} ({src['type']}) is MUTED but something "
                    "consumes its output")
            return sid, sslot
        raise RuntimeError("dangling link")

    api = {}
    for n in ui["nodes"]:
        t = n["type"]
        if t in VIRTUAL or n.get("mode", 0) in (2, 4):
            continue
        if t not in oi:
            raise RuntimeError(f"unknown node type {t} (not registered on this server)")
        inputs = widget_inputs(oi, t, n.get("widgets_values") or [])
        for inp in n.get("inputs", []):
            if inp.get("link") is None:
                continue
            resolved = upstream(inp["link"])
            if resolved is None:      # dropped: bypassed source with no forward
                inputs.pop(inp["name"], None)
                continue
            sid, sslot = resolved
            inputs[inp["name"]] = [str(sid), sslot]
        api[str(n["id"])] = {"class_type": t, "inputs": inputs}
    return api


def gate_overrides(ui: dict, oi: dict | None = None) -> dict:
    """``{"<id>.proceed": True}`` for every AtlasSolveGate in the workflow."""
    if oi is not None:
        ui = expand_subgraphs(ui, oi)
    return {f"{n['id']}.proceed": True
            for n in ui["nodes"] if n["type"] == "AtlasSolveGate"}


def apply_overrides(api: dict, overrides: dict) -> list[str]:
    """Apply ``{"<nodeId>.<input>": value}`` overrides in place."""
    applied = []
    for key, val in (overrides or {}).items():
        nid, _, name = key.partition(".")
        if nid not in api:
            raise KeyError(f"override {key}: node {nid} not in the executed graph")
        api[nid]["inputs"][name] = val
        applied.append(f"{nid}.{name}={val!r}")
    return applied


def queue_and_wait(api: dict, host: str = DEFAULT_HOST,
                   timeout: int = 1800, poll_s: float = 5.0) -> dict:
    """POST the API graph and poll ``/history`` until it finishes.

    Returns ``{completed, prompt_id, errors: [...], output_nodes: [...]}`` —
    node errors carry the verbatim exception message (the single most useful
    diagnostic when driving Atlas headlessly).
    """
    client_id = str(uuid.uuid4())
    resp = http_json(f"http://{host}/prompt",
                     {"prompt": api, "client_id": client_id})
    if resp.get("error"):
        return {"completed": False, "prompt_id": None,
                "errors": [json.dumps(resp, default=str)[:4000]],
                "output_nodes": []}
    pid = resp["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(poll_s)
        hist = http_json(f"http://{host}/history/{pid}", timeout=30)
        if pid not in hist:
            q = http_json(f"http://{host}/queue", timeout=30)
            live = any(item[1] == pid
                       for item in q.get("queue_running", []) + q.get("queue_pending", []))
            if not live:
                return {"completed": False, "prompt_id": pid,
                        "errors": ["prompt vanished from queue with no history entry"],
                        "output_nodes": []}
            continue
        rec = hist[pid]
        status = rec.get("status", {})
        errors = []
        for ev in status.get("messages", []):
            if ev[0] == "execution_error":
                d = ev[1]
                errors.append(f"{d.get('node_type')} (node {d.get('node_id')}): "
                              f"{d.get('exception_type')}: {d.get('exception_message')}")
        return {"completed": bool(status.get("completed")), "prompt_id": pid,
                "errors": errors,
                "output_nodes": sorted(rec.get("outputs", {}).keys(),
                                       key=lambda x: int(x) if x.isdigit() else 0)}
    return {"completed": False, "prompt_id": pid,
            "errors": [f"timeout after {timeout}s"], "output_nodes": []}


def validate_ui(ui: dict, oi: dict) -> tuple[list[str], list[str]]:
    """The drift/link/range/rail checks from ``tools/validate_ui_workflow.py``
    as ``(errors, warnings)`` lists."""
    ui = expand_subgraphs(ui, oi)
    nodes = {n["id"]: n for n in ui["nodes"]}
    errs, warns = [], []

    def wcount(t):
        c = 0
        for k, v in spec_items(oi, t):
            if is_widget(v):
                c += 1
                if k in ("seed", "noise_seed"):
                    c += 1
        return c

    for n in ui["nodes"]:
        if n["type"] not in VIRTUAL and n["type"] not in oi:
            errs.append(f"type {n['type']} unknown")
    for n in ui["nodes"]:
        if n["type"] in VIRTUAL or n["type"] == "LoadImage" or n["type"] not in oi:
            continue
        want, got = wcount(n["type"]), len(n.get("widgets_values") or [])
        if want != got:
            errs.append(f"{n['type']} id{n['id']}: widgets_values {got} != {want}")
    for l in ui["links"]:
        lid, sid, sslot, tid, tslot, _ = l
        if sid not in nodes or tid not in nodes:
            errs.append(f"link {lid} dangling")
            continue
        s, t = nodes[sid], nodes[tid]
        if sslot >= len(s["outputs"]) or tslot >= len(t["inputs"]):
            errs.append(f"link {lid} bad slot")
            continue
        if lid not in (s["outputs"][sslot]["links"] or []):
            errs.append(f"link {lid} not on src")
        if t["inputs"][tslot]["link"] != lid:
            errs.append(f"link {lid} dst mismatch")
        st, tt = s["outputs"][sslot]["type"], t["inputs"][tslot]["type"]
        if s["type"] in VIRTUAL or t["type"] in VIRTUAL:
            continue
        if st != tt and tt != "COMBO" and st != "*" and tt != "*":
            errs.append(f"link {lid}: TYPE {s['type']}.{st} -> {t['type']}.{tt}")
    for n in ui["nodes"]:
        if n["type"] in VIRTUAL or n["type"] not in oi:
            continue
        for k, v in (oi[n["type"]]["input"].get("required") or {}).items():
            if is_widget(v):
                continue
            inp = next((i for i in n["inputs"] if i["name"] == k), None)
            if inp is None:
                errs.append(f"{n['type']} id{n['id']}: missing input {k}")
            elif inp["link"] is None:
                errs.append(f"{n['type']} id{n['id']}: required '{k}' NOT CONNECTED")
    for n in ui["nodes"]:
        if n["type"] in VIRTUAL or n["type"] not in oi:
            continue
        vals = n.get("widgets_values") or []
        vi = 0
        for k, v in spec_items(oi, n["type"]):
            if not is_widget(v):
                continue
            if vi >= len(vals):
                break
            val = vals[vi]
            vi += 1
            if k in ("seed", "noise_seed"):
                vi += 1
            cfg = v[1] if len(v) > 1 else {}
            if v[0] in ("INT", "FLOAT") and isinstance(val, (int, float)) and not isinstance(val, bool):
                lo, hi = cfg.get("min"), cfg.get("max")
                if lo is not None and val < lo:
                    errs.append(f"{n['type']} id{n['id']}: {k}={val} BELOW min {lo}")
                if hi is not None and val > hi:
                    errs.append(f"{n['type']} id{n['id']}: {k}={val} ABOVE max {hi}")
            if v[0] == "COMBO" or isinstance(v[0], list):
                opts = v[1].get("options") if (len(v) > 1 and isinstance(v[1], dict)) else None
                if opts is None and isinstance(v[0], list):
                    opts = v[0]
                if opts and val not in opts:
                    errs.append(f"{n['type']} id{n['id']}: {k}={val!r} not a valid option")
    sets = {n["widgets_values"][0] for n in ui["nodes"] if n["type"] == "SetNode"}
    for n in ui["nodes"]:
        if n["type"] == "GetNode" and n["widgets_values"][0] not in sets:
            errs.append(f"rail '{n['widgets_values'][0]}' has no Set")
    used = {n["widgets_values"][0] for n in ui["nodes"] if n["type"] == "GetNode"}
    for s_ in sorted(sets - used):
        warns.append(f"rail '{s_}' set but never got")
    ids = [n["id"] for n in ui["nodes"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate node ids")
    return errs, warns


def upload_image(path: str, host: str = DEFAULT_HOST) -> str:
    """Upload a local image into ComfyUI's input dir via ``/upload/image``;
    returns the server-side filename LoadImage can reference."""
    import mimetypes
    import os
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        blob = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n"
    ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://{host}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))["name"]
