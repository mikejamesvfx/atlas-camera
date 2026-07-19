"""Read-only usage audit for the registered Atlas ComfyUI nodes.

Classifies every registered node (standard + experimental) by where it is
referenced across the repository:

  * example workflow JSONs (``class_type`` / ``type`` occurrences),
  * tests,
  * MCP server / tools,
  * documentation,

and flags nodes that are registered but otherwise unreferenced.

IMPORTANT — presence is not execution. A node appearing in a workflow file
proves only that the file names it, not that anyone ever queued that graph.
Workflow generators rewrite many files at once, so file mtimes cannot prove
recency either; this tool therefore reports *reference sites*, not a
"used/unused" verdict, and never labels a node unused merely because it is
absent from a workflow. A node with a dedicated test, an MCP handler, or a
public import is exercised even with zero workflow hits.

Optionally, ``--comfyui-host HOST:PORT`` pulls the live ComfyUI ``/history`` and
counts executed ``class_type``s — but that history is transient (cleared on
restart / capped), so it can confirm recent execution yet never disprove it.

The tool is strictly read-only: it never writes or rewrites any workflow.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def registered_nodes() -> tuple[dict, set]:
    """Return ({name: 'standard'|'experimental'}, all_names)."""
    from atlas_camera.comfy import node_registry as reg
    kinds = {k: "standard" for k in reg.NODE_CLASS_MAPPINGS}
    for k in reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS:
        kinds.setdefault(k, "experimental")
    return kinds, set(kinds)


def _iter_files(root: Path, suffixes):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def _workflow_node_types(path: Path) -> set:
    """Node class_type/type strings present in a UI- or API-format workflow."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out = set()
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if isinstance(nodes, list):                      # UI format
        for n in nodes:
            if isinstance(n, dict) and n.get("type"):
                out.add(n["type"])
    elif isinstance(data, dict):                     # API format {id: {class_type}}
        for n in data.values():
            if isinstance(n, dict) and n.get("class_type"):
                out.add(n["class_type"])
    return out


def audit(repo: Path = REPO) -> dict:
    kinds, names = registered_nodes()
    result = {n: {"kind": kinds[n], "example_workflows": [], "tests": [],
                  "mcp_tools": [], "docs": []} for n in names}

    # 1) workflow files (presence only)
    for wf in _iter_files(repo / "examples", {".json"}):
        types = _workflow_node_types(wf)
        rel = str(wf.relative_to(repo)).replace("\\", "/")
        for n in types & names:
            result[n]["example_workflows"].append(rel)

    # 2) text reference scans (word-boundary match on the node name)
    scan = {
        "tests": (repo / "tests", {".py"}),
        "mcp_tools": (repo / "atlas_camera" / "mcp", {".py"}),
        "docs": (repo / "docs", {".md"}),
    }
    # tools/ is scanned into mcp_tools too (audit tool itself excluded)
    tool_files = [p for p in _iter_files(repo / "tools", {".py"})
                  if p.name != "audit_node_usage.py"]
    patterns = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    for bucket, (root, suf) in scan.items():
        for f in _iter_files(root, suf):
            text = f.read_text(encoding="utf-8", errors="ignore")
            rel = str(f.relative_to(repo)).replace("\\", "/")
            for n in names:
                if patterns[n].search(text):
                    result[n][bucket].append(rel)
    for f in tool_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = str(f.relative_to(repo)).replace("\\", "/")
        for n in names:
            if patterns[n].search(text):
                result[n]["mcp_tools"].append(rel)

    for n, rec in result.items():
        for b in ("example_workflows", "tests", "mcp_tools", "docs"):
            rec[b] = sorted(set(rec[b]))
        referenced = any(rec[b] for b in ("example_workflows", "tests",
                                          "mcp_tools", "docs"))
        rec["status"] = "referenced" if referenced else "registered_only"
        rec["in_workflows"] = bool(rec["example_workflows"])
    return result


def _history_counts(host: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(f"http://{host}/history", timeout=30) as r:
        hist = json.loads(r.read().decode("utf-8"))
    counts: dict = {}
    for entry in hist.values():
        prompt = entry.get("prompt")
        graph = prompt[2] if isinstance(prompt, list) and len(prompt) > 2 else {}
        for node in (graph or {}).values():
            ct = node.get("class_type") if isinstance(node, dict) else None
            if ct:
                counts[ct] = counts.get(ct, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="Emit the full audit as JSON.")
    ap.add_argument("--comfyui-host", default=None,
                    help="host:port of a running ComfyUI to read /history (transient!).")
    args = ap.parse_args()

    data = audit()
    if args.comfyui_host:
        try:
            counts = _history_counts(args.comfyui_host)
        except Exception as exc:  # transient/offline — never fatal
            counts = {}
            print(f"# /history unavailable ({exc}); reporting file references only")
        for n, rec in data.items():
            rec["history_executions"] = counts.get(n, 0)

    if args.json:
        print(json.dumps(data, indent=1))
        return

    reg_only = sorted(n for n, r in data.items() if r["status"] == "registered_only")
    no_wf = sorted(n for n, r in data.items() if not r["in_workflows"])
    print(f"registered nodes: {len(data)}  "
          f"({sum(1 for r in data.values() if r['kind']=='experimental')} experimental)")
    print(f"referenced somewhere: {sum(1 for r in data.values() if r['status']=='referenced')}")
    print(f"registered-only (no workflow/test/mcp/doc reference): "
          f"{reg_only or 'none'}")
    print(f"\nnot present in any example workflow (may still be tested/MCP/doc; "
          f"presence != execution): {len(no_wf)}")
    for n in no_wf:
        r = data[n]
        where = []
        if r["tests"]:
            where.append(f"tests={len(r['tests'])}")
        if r["mcp_tools"]:
            where.append(f"mcp/tools={len(r['mcp_tools'])}")
        if r["docs"]:
            where.append(f"docs={len(r['docs'])}")
        print(f"  {n} [{r['kind']}] {', '.join(where) or 'REGISTERED-ONLY'}")


if __name__ == "__main__":
    main()
