"""Convert a UI-format workflow to API format and run it on a live ComfyUI.

Usage:
    python tools/run_ui_workflow.py <workflow_ui.json> [--host 127.0.0.1:8188]
        [--set <nodeId>.<input>=<value> ...]   # override an input (gates, seeds)
        [--convert-only <out_api.json>]        # write the API JSON, don't queue
        [--timeout 1800]

Thin CLI over :mod:`atlas_camera.mcp.comfy_http`, the canonical home of the
UI→API flattening (KJ Set/Get rails, muted/bypassed nodes, phantom widget
slots), gate overrides and queue/poll logic. The Atlas MCP server exposes the
same operations as tools.
"""
import json
import sys

sys.path.insert(0, ".")  # repo root (the script is run from the checkout)
from atlas_camera.mcp import comfy_http as C  # noqa: E402


def main():
    args = sys.argv[1:]
    host = C.DEFAULT_HOST
    overrides, convert_only, timeout = {}, None, 1800
    path = args.pop(0)
    while args:
        a = args.pop(0)
        if a == "--host":
            host = args.pop(0)
        elif a == "--set":
            key, _, raw = args.pop(0).partition("=")
            try:
                overrides[key] = json.loads(raw)
            except json.JSONDecodeError:
                overrides[key] = raw
        elif a == "--convert-only":
            convert_only = args.pop(0)
        elif a == "--timeout":
            timeout = int(args.pop(0))
        else:
            raise SystemExit(f"unknown arg {a}")

    ui = json.load(open(path, encoding="utf-8"))
    oi = C.fetch_object_info(host)
    api = C.ui_to_api(ui, oi)
    for line in C.apply_overrides(api, overrides):
        print(f"override: {line}")

    if convert_only:
        json.dump(api, open(convert_only, "w", encoding="utf-8"), indent=1)
        print(f"wrote {convert_only} ({len(api)} executing nodes)")
        return

    print(f"queueing ({len(api)} executing nodes)")
    result = C.queue_and_wait(api, host, timeout=timeout)
    print(f"status: {'success' if result['completed'] else 'error'} "
          f"completed={result['completed']}")
    for e in result["errors"]:
        print("ERROR:", e)
    print(f"outputs from {len(result['output_nodes'])} nodes: {result['output_nodes']}")
    raise SystemExit(0 if result["completed"] else 1)


if __name__ == "__main__":
    main()
