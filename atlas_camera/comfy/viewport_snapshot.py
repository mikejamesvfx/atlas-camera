"""Automatic end-of-run viewport snapshots — the agent's eyes on a workflow.

WHY. Every workflow ends in the 🏗 viewport, and the only ways to see it were a
human at the browser or the ⬛ Render Proxy Passes button (which re-queues the
graph). An agent driving ComfyUI over the API — `atlas_run_workflow`,
`atlas_inspect_viewport` — got numbers, never pixels, so "did the projection
land" was unanswerable without a screenshot recipe.

WHAT. After each execution of an `AtlasBlockoutViewport` the frontend renders
two offscreen frames FROM THE RECOVERED CAMERA (📷 Camera View pose, not the
artist's orbit) at long-edge 1280 — 📽 Project ON (the product) and OFF (the
grey geometry diagnostic) — hides the draw-tool helpers, and POSTs both here.
This module writes them under ComfyUI's output folder:

    <output>/atlas_viewport/viewport_<node>_projected.png     (stable, overwritten)
    <output>/atlas_viewport/viewport_<node>_geometry.png
    <output>/atlas_viewport/viewport_<node>.json               (sidecar: when, what)
    <output>/atlas_viewport/history/<stamp>_<node>_{projected,geometry}.png

and records the paths on the node's cached camera payload so
`atlas_inspect_viewport` (MCP) can hand an agent the file paths without a
browser. Pure stdlib + base64: the aiohttp route in comfy/__init__.py is a
thin wrapper so this stays unit-testable outside ComfyUI.

The 1280 long edge is fixed on purpose (the user asked for the default size,
every run): it is a review image, not an output — the `resolution` widget
still governs everything that is.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

SNAPSHOT_DIRNAME = "atlas_viewport"
HISTORY_DIRNAME = "history"
SNAPSHOT_LONG_EDGE = 1280
MAX_HISTORY_PER_NODE = 40   # keep the folder bounded; oldest go first

_KINDS = ("projected", "geometry")


def _safe_id(node_id: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(node_id or "unknown"))
    return s[:64] or "unknown"


def _decode_png(b64: str) -> bytes:
    payload = b64.split(",", 1)[1] if "," in b64 else b64
    data = base64.b64decode(payload)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("snapshot is not a PNG")
    return data


def _prune_history(history: Path, node_key: str) -> int:
    files = sorted(history.glob(f"*_{node_key}_*.png"))
    keep = MAX_HISTORY_PER_NODE * len(_KINDS)
    removed = 0
    for old in files[:-keep] if len(files) > keep else []:
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def save_viewport_snapshot(payload: dict[str, Any], *, output_dir: str | Path,
                           now: float | None = None) -> dict[str, Any]:
    """Write the two PNGs + sidecar. Returns the record that goes on the cache.

    ``payload``: ``{"node_id", "projected_b64", "geometry_b64", "width",
    "height", "solve_fingerprint"?, "reason"?, "workflow_name"?}``. Either
    image may be missing (e.g. no solve yet → only geometry). Raises
    ValueError on malformed input; the caller turns that into a 400.
    """
    node_key = _safe_id(payload.get("node_id"))
    images = {k: payload.get(f"{k}_b64") for k in _KINDS}
    images = {k: v for k, v in images.items() if v}
    if not images:
        raise ValueError("no snapshot images in payload")
    root = Path(output_dir) / SNAPSHOT_DIRNAME
    history = root / HISTORY_DIRNAME
    history.mkdir(parents=True, exist_ok=True)
    ts = float(now if now is not None else time.time())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))

    files: dict[str, str] = {}
    hist_files: dict[str, str] = {}
    for kind, b64 in images.items():
        data = _decode_png(str(b64))
        latest = root / f"viewport_{node_key}_{kind}.png"
        latest.write_bytes(data)
        hist = history / f"{stamp}_{node_key}_{kind}.png"
        hist.write_bytes(data)
        files[kind] = str(latest)
        hist_files[kind] = str(hist)
    pruned = _prune_history(history, node_key)

    record = {
        "node_id": str(payload.get("node_id", "")),
        "timestamp": ts,
        "stamp": stamp,
        "width": int(payload.get("width") or 0),
        "height": int(payload.get("height") or 0),
        "long_edge": SNAPSHOT_LONG_EDGE,
        "camera": "recovered (📷 Camera View pose)",
        "solve_fingerprint": payload.get("solve_fingerprint") or "",
        "reason": payload.get("reason") or "executed",
        "workflow_name": payload.get("workflow_name") or "",
        "files": files,
        "history": hist_files,
        "history_pruned": pruned,
    }
    (root / f"viewport_{node_key}.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8")
    return record


def attach_snapshot_to_cache(cache: dict[str, Any], record: dict[str, Any]) -> None:
    """Put the record on the node's camera_data payload (if it is cached) so
    `/atlas/camera_data/{id}` — and therefore `atlas_inspect_viewport` — carry
    the file paths. A missing payload (server restarted, cache LRU'd) is fine:
    the sidecar JSON on disk is the durable copy."""
    node_id = str(record.get("node_id", ""))
    entry = cache.get(node_id)
    if isinstance(entry, dict):
        entry["viewport_snapshot"] = dict(record)


def read_snapshot_record(node_id: Any, *, output_dir: str | Path) -> dict[str, Any] | None:
    """The sidecar for a node, or None."""
    p = Path(output_dir) / SNAPSHOT_DIRNAME / f"viewport_{_safe_id(node_id)}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
