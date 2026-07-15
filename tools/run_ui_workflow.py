"""Convert a UI-format workflow to API format and run it on a live ComfyUI.

Usage:
    python tools/run_ui_workflow.py <workflow_ui.json> [--host 127.0.0.1:8188]
        [--set <nodeId>.<input>=<value> ...]   # override an input (gates, seeds)
        [--convert-only <out_api.json>]        # write the API JSON, don't queue
        [--timeout 1800]

Flattens KJNodes Set/Get rails (frontend-only virtual nodes), maps positional
widgets_values to input names from the live /object_info (the same is_widget
logic as tools/validate_ui_workflow.py), drops muted (mode 2) nodes and
resolves bypassed (mode 4) nodes to their first same-type input, then queues
via /prompt and polls /history until the run finishes, printing every node
error verbatim.
"""
import json
import sys
import time
import urllib.request
import uuid

PRIMS = {"INT", "FLOAT", "STRING", "BOOLEAN"}
VIRTUAL = {"SetNode", "GetNode", "Note", "MarkdownNote", "Reroute"}


def http_json(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def is_widget(spec):
    t = spec[0]
    cfg = spec[1] if len(spec) > 1 else {}
    if isinstance(t, list):
        return True
    if t == "COMBO" or t in PRIMS:
        return not cfg.get("forceInput")
    return False


def spec_items(oi, type_):
    s = oi[type_]["input"]
    return [(k, v) for sec in ("required", "optional") for k, v in (s.get(sec) or {}).items()]


def widget_inputs(oi, type_, values):
    """Positional widgets_values -> {input_name: literal}."""
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
        # seed widgets carry a control_after_generate slot; image_upload
        # widgets carry the upload-button slot — both extra values are
        # frontend-only and must be skipped, not mapped.
        if k in ("seed", "noise_seed") or (isinstance(cfg, dict) and cfg.get("image_upload")):
            vi += 1
    return out


def convert(ui, oi):
    nodes = {n["id"]: n for n in ui["nodes"]}
    links = {l[0]: l for l in ui["links"]}  # id -> [id, src, sslot, dst, dslot, type]

    def upstream(link_id):
        """Resolve a link to its real (non-virtual, non-bypassed) source."""
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
                              if n["type"] == "SetNode" and n["widgets_values"][0] == rail)
                link_id = setter["inputs"][0]["link"]
                continue
            if src["type"] == "Reroute":
                link_id = src["inputs"][0]["link"]
                continue
            if mode == 4:  # bypassed: forward the first connected same-type input
                want = src["outputs"][sslot]["type"]
                nxt = next((i["link"] for i in src.get("inputs", [])
                            if i.get("link") is not None
                            and (i.get("type") == want or want == "*")), None)
                if nxt is None:
                    raise RuntimeError(f"bypassed node {sid} ({src['type']}) has no "
                                       f"same-type input to forward for {want}")
                link_id = nxt
                continue
            if mode == 2:
                raise RuntimeError(f"node {sid} ({src['type']}) is MUTED but something "
                                   "consumes its output")
            return sid, sslot
        raise RuntimeError("dangling link")

    api = {}
    for n in ui["nodes"]:
        t = n["type"]
        if t in VIRTUAL or n.get("mode", 0) in (2, 4):
            continue
        if t not in oi:
            raise RuntimeError(f"unknown node type {t}")
        inputs = widget_inputs(oi, t, n.get("widgets_values") or [])
        for inp in n.get("inputs", []):
            if inp.get("link") is None:
                continue
            sid, sslot = upstream(inp["link"])
            inputs[inp["name"]] = [str(sid), sslot]
        api[str(n["id"])] = {"class_type": t, "inputs": inputs}
    return api


def main():
    args = sys.argv[1:]
    host = "127.0.0.1:8188"
    overrides, convert_only, timeout = [], None, 1800
    path = args.pop(0)
    while args:
        a = args.pop(0)
        if a == "--host":
            host = args.pop(0)
        elif a == "--set":
            overrides.append(args.pop(0))
        elif a == "--convert-only":
            convert_only = args.pop(0)
        elif a == "--timeout":
            timeout = int(args.pop(0))
        else:
            raise SystemExit(f"unknown arg {a}")

    ui = json.load(open(path, encoding="utf-8"))
    oi = http_json(f"http://{host}/object_info", timeout=120)
    api = convert(ui, oi)

    for ov in overrides:
        key, _, raw = ov.partition("=")
        nid, _, name = key.partition(".")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        if nid not in api:
            raise SystemExit(f"--set {ov}: node {nid} not in API graph")
        api[nid]["inputs"][name] = val
        print(f"override: {nid}.{name} = {val!r}")

    if convert_only:
        json.dump(api, open(convert_only, "w", encoding="utf-8"), indent=1)
        print(f"wrote {convert_only} ({len(api)} executing nodes)")
        return

    client_id = str(uuid.uuid4())
    resp = http_json(f"http://{host}/prompt", {"prompt": api, "client_id": client_id})
    if resp.get("error"):
        print("QUEUE REJECTED:")
        print(json.dumps(resp, indent=1)[:8000])
        raise SystemExit(1)
    pid = resp["prompt_id"]
    print(f"queued {pid} ({len(api)} executing nodes)")

    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        hist = http_json(f"http://{host}/history/{pid}", timeout=30)
        if pid not in hist:
            q = http_json(f"http://{host}/queue", timeout=30)
            running = any(item[1] == pid for item in q.get("queue_running", []))
            pending = any(item[1] == pid for item in q.get("queue_pending", []))
            if not running and not pending:
                print("prompt vanished from queue with no history entry")
                raise SystemExit(1)
            continue
        rec = hist[pid]
        status = rec.get("status", {})
        print(f"status: {status.get('status_str')} completed={status.get('completed')}")
        for ev in status.get("messages", []):
            if ev[0] == "execution_error":
                d = ev[1]
                print(f"\nERROR in node {d.get('node_id')} ({d.get('node_type')}):")
                print(f"  {d.get('exception_type')}: {d.get('exception_message')}")
                for ln in (d.get("traceback") or [])[-12:]:
                    print("  " + ln.rstrip())
        outs = rec.get("outputs", {})
        print(f"outputs from {len(outs)} nodes: {sorted(outs.keys(), key=lambda x: int(x) if x.isdigit() else 0)}")
        for nid, o in sorted(outs.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            keys = {k: (v if isinstance(v, (str, int, float)) else f"[{len(v)} items]" if isinstance(v, list) else type(v).__name__) for k, v in o.items()}
            print(f"  node {nid}: {keys}")
        raise SystemExit(0 if status.get("completed") else 1)
    print("TIMEOUT waiting for completion")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
