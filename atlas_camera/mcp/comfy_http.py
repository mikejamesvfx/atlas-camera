"""HTTP + workflow plumbing for the Atlas MCP server.

The canonical home of the UI→API flattening, gate handling, validation and
queue/poll logic that `tools/run_ui_workflow.py` and
`tools/validate_ui_workflow.py` pioneered (those CLIs now import from here).
Standard library only — the MCP server must not drag torch/numpy into its own
process; ComfyUI stays the execution engine.

Conventions this module encodes (learned by running every Atlas workflow
headlessly — see docs/dev/atlas_mcp_server_plan.md):

- KJNodes ``SetNode``/``GetNode`` rails are frontend-only virtual nodes and
  must be resolved through their rail name.
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


def ui_to_api(ui: dict, oi: dict) -> dict:
    """Flatten a UI-format workflow to the API format ``/prompt`` executes."""
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


def gate_overrides(ui: dict) -> dict:
    """``{"<id>.proceed": True}`` for every AtlasSolveGate in the workflow."""
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
