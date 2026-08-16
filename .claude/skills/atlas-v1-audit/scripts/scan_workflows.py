"""Phase 3 — serialized ComfyUI workflows.

Duplicate detection compares NODE GRAPHS, never filenames. Two workflows named
`foo.json` and `foo_v2.json` may be unrelated, and two with different names may
be the same graph — filenames are the least reliable evidence in the tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NEAR_DUPLICATE_THRESHOLD = 0.80


def _node_types(doc) -> list[str]:
    """Both serialization formats: UI (`nodes: [...]`) and API (`{id: {...}}`)."""
    types: list[str] = []
    if isinstance(doc, dict) and isinstance(doc.get("nodes"), list):
        for node in doc["nodes"]:
            if isinstance(node, dict) and isinstance(node.get("type"), str):
                types.append(node["type"])
        return types
    if isinstance(doc, dict):
        for value in doc.values():
            if isinstance(value, dict) and isinstance(value.get("class_type"), str):
                types.append(value["class_type"])
    return types


def _fmt(doc) -> str:
    if isinstance(doc, dict) and isinstance(doc.get("nodes"), list):
        return "ui"
    return "api"


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build(root: Path, cfg: dict) -> dict:
    nodes = common.read_raw(root, "nodes")
    registered = set(nodes["nodes"])
    # Which node types belong to THIS project, derived from the registry's own
    # naming rather than hardcoded. Without it there is no way to tell an
    # unresolved project node from a ComfyUI builtin, since builtins cannot be
    # enumerated. Empty prefix = fall back to exact registry membership.
    prefix = nodes.get("node_prefix", "")

    def ours(node_type: str) -> bool:
        return node_type.startswith(prefix) if prefix else node_type in registered

    workflows: dict[str, dict] = {}
    for rel in common.tracked_files(root):
        if common.categorize(rel) != "COMFY_WORKFLOW":
            continue
        try:
            doc = json.loads((root / rel).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            workflows[rel] = {"path": rel, "broken": True, "error": str(exc),
                              "atlas_types": [], "unresolved": [], "node_count": 0,
                              "format": "unknown"}
            continue
        types = _node_types(doc)
        atlas = sorted({t for t in types if ours(t)})
        workflows[rel] = {
            "path": rel,
            "format": _fmt(doc),
            "node_count": len(types),
            "node_types": sorted(set(types)),
            "atlas_types": atlas,
            "unresolved": sorted(t for t in atlas if t not in registered),
            "broken": False,
        }

    # exact = identical multiset of node types; near = high Jaccard overlap
    exact: dict[str, list[str]] = {}
    for rel, wf in workflows.items():
        if wf["broken"]:
            continue
        sig = json.dumps(sorted(Counter(wf["node_types"]).items()), sort_keys=True)
        exact.setdefault(sig, []).append(rel)

    near: list[dict] = []
    items = [(rel, set(wf["node_types"])) for rel, wf in workflows.items()
             if not wf["broken"]]
    for i, (rel_a, set_a) in enumerate(items):
        for rel_b, set_b in items[i + 1:]:
            score = _overlap(set_a, set_b)
            if score >= NEAR_DUPLICATE_THRESHOLD:
                near.append({"a": rel_a, "b": rel_b, "overlap": round(score, 3)})

    return {
        "workflows": workflows,
        "exact_duplicate_clusters": [sorted(v) for v in exact.values() if len(v) > 1],
        "near_duplicates": sorted(near, key=lambda d: -d["overlap"]),
        "broken": sorted(r for r, w in workflows.items()
                         if w["broken"] or w["unresolved"]),
        "counts": {
            "total": len(workflows),
            "broken": sum(1 for w in workflows.values()
                          if w["broken"] or w["unresolved"]),
        },
    }


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    cfg = common.load_config(Path(__file__).resolve().parents[1])
    payload = build(root, cfg)
    common.write_raw(root, "workflows", payload)
    c = payload["counts"]
    print(f"workflows: {c['total']} found, {c['broken']} broken/unresolved, "
          f"{len(payload['exact_duplicate_clusters'])} exact-duplicate cluster(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
