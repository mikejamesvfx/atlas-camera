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

# Tests that name EVERY registered node by construction — registry pins,
# façade pins, this tool's own contract test, and the widget-drift sweep.
# A hit in one of these proves the node is REGISTERED, which we already know
# from the registry itself; it says nothing about whether the node is used.
# They are the entire reason a naive scan reports every node as "referenced",
# so they never count toward product evidence.
GENERIC_TESTS = frozenset({
    "tests/test_comfy_node_registry.py",
    "tests/test_facade_surface.py",
    "tests/test_node_usage_audit.py",
    "tests/test_shipping_workflow_widgets.py",
})


def registered_nodes() -> tuple[dict, set]:
    """Return ({name: 'standard'|'experimental'|'legacy'}, all_names)."""
    from atlas_camera.comfy import node_registry as reg
    kinds = {k: "standard" for k in reg.NODE_CLASS_MAPPINGS}
    for k in reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS:
        kinds.setdefault(k, "experimental")
    # getattr so the tool still runs against a checkout without the gate.
    for k in getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}):
        kinds.setdefault(k, "legacy")
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
                  "mcp_tools": [], "repo_tools": [], "docs": []} for n in names}

    # 1) workflow files (presence only). Artists keep personal working copies
    # next to the shipped graphs ("-edit", examples/local/); those are not
    # product evidence and must not inflate a node's workflow bucket. Same
    # rule as tests/conftest.py::is_local_workflow and the migrator.
    for wf in _iter_files(repo / "examples", {".json"}):
        if "-edit" in wf.stem or "local" in {q.lower() for q in wf.parts}:
            continue
        types = _workflow_node_types(wf)
        rel = str(wf.relative_to(repo)).replace("\\", "/")
        for n in types & names:
            result[n]["example_workflows"].append(rel)

    # 2) text reference scans (word-boundary match on the node name)
    # `mcp_tools` and `repo_tools` are SEPARATE buckets. They used to be one,
    # which made "a live MCP handler depends on this node" indistinguishable
    # from "some CLI script mentions the name" — and only the former is
    # evidence that a node is part of the product.
    scan = {
        "tests": (repo / "tests", {".py"}),
        "mcp_tools": (repo / "atlas_camera" / "mcp", {".py"}),
        "repo_tools": (repo / "tools", {".py"}),
        "docs": (repo / "docs", {".md"}),
    }
    patterns = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    # Files that enumerate the whole registry and would otherwise inflate their
    # own bucket: this tool, and the generated audit report (which names every
    # node by construction, so counting it would make every node look
    # documented the moment the report is written).
    self_referential = {"audit_node_usage.py", "feature_audit_verdicts.py",
                        "build_feature_audit.py", "FEATURE_AUDIT.md"}
    for bucket, (root, suf) in scan.items():
        for f in _iter_files(root, suf):
            if f.name in self_referential:
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            rel = str(f.relative_to(repo)).replace("\\", "/")
            for n in names:
                if patterns[n].search(text):
                    result[n][bucket].append(rel)

    buckets = ("example_workflows", "tests", "mcp_tools", "repo_tools", "docs")
    for n, rec in result.items():
        for b in buckets:
            rec[b] = sorted(set(rec[b]))
        # `status` stays exactly two-valued — it means "named anywhere at all",
        # and tests/test_node_usage_audit.py pins that contract.
        rec["status"] = "referenced" if any(rec[b] for b in buckets) else "registered_only"
        rec["in_workflows"] = bool(rec["example_workflows"])

        # Product evidence is the useful signal: a node is part of the product
        # if a shipping workflow uses it, a test exercises it specifically, or
        # an MCP handler depends on it. Docs and repo tools are deliberately
        # excluded — documenting a node proves intent, not use, and every node
        # but one is documented, so counting docs would flatten the signal.
        rec["dedicated_tests"] = [t for t in rec["tests"] if t not in GENERIC_TESTS]
        rec["product_evidence"] = bool(rec["example_workflows"]
                                       or rec["dedicated_tests"]
                                       or rec["mcp_tools"])
        rec["evidence_kinds"] = [k for k, v in (
            ("workflow", rec["example_workflows"]),
            ("dedicated_test", rec["dedicated_tests"]),
            ("mcp", rec["mcp_tools"]),
            ("repo_tool", rec["repo_tools"]),
            ("docs", rec["docs"]),
        ) if v]
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
    ap.add_argument("--repo", default=None,
                    help="repository to audit (default: this file's checkout).")
    ap.add_argument("--comfyui-host", default=None,
                    help="host:port of a running ComfyUI to read /history (transient!).")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else REPO
    # Running from a git worktree silently audits the WRONG tree: sys.path[0]
    # resolves `atlas_camera` through the editable install (the main checkout)
    # while the file scans walk the worktree, so the registry and the files
    # disagree and nodes are reported registered-only that are not.
    try:
        import atlas_camera
        pkg = Path(atlas_camera.__file__).resolve().parent
        if repo not in pkg.parents:
            print(f"# WARNING: imported atlas_camera from {pkg}, but auditing "
                  f"{repo} — run this from the main checkout, not a worktree")
    except Exception:  # noqa: BLE001 — never let the warning break the tool
        pass

    data = audit(repo)
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

    # The signal this tool exists for. A node with no workflow, no test that
    # exercises it specifically, and no MCP consumer has nothing proving it is
    # part of the product — which is a prompt to find evidence or retire it,
    # NOT proof that it is broken (most such nodes run perfectly).
    no_ev = sorted(n for n, r in data.items()
                   if r["kind"] == "standard" and not r["product_evidence"])
    print(f"\nno product evidence (workflow / dedicated test / MCP): {len(no_ev)}")
    for n in no_ev:
        r = data[n]
        extra = []
        if r["docs"]:
            extra.append(f"docs={len(r['docs'])}")
        if r["repo_tools"]:
            extra.append(f"repo_tools={len(r['repo_tools'])}")
        print(f"  {n} ({', '.join(extra) or 'nothing at all'})")


if __name__ == "__main__":
    main()
