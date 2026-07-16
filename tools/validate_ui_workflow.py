"""Validate a UI-format workflow against an /object_info snapshot.

Usage: python tools/validate_ui_workflow.py <object_info.json> <workflow.json>

Thin CLI over :func:`atlas_camera.mcp.comfy_http.validate_ui` (the canonical
checks: positional-widget drift, link integrity, widget ranges, KJ-rail
resolution). Output format is stable — other tools grep for "ERRORS (0)".
"""
import json
import sys

sys.path.insert(0, ".")  # repo root (the script is run from the checkout)
from atlas_camera.mcp import comfy_http as C  # noqa: E402

oi = json.load(open(sys.argv[1], encoding="utf-8"))
d = json.load(open(sys.argv[2], encoding="utf-8"))
errs, warns = C.validate_ui(d, oi)
print(f"nodes={len(d['nodes'])} links={len(d['links'])} groups={len(d.get('groups') or [])}")
print(f"ERRORS ({len(errs)}):")
for e in errs[:15]:
    print("   ", e)
print(f"WARNINGS ({len(warns)}):")
for x in warns[:8]:
    print("   ", x)
