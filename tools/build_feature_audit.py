"""Generate reports/feature_audit.json + docs/FEATURE_AUDIT.md.

Kept separate from ``audit_node_usage.py`` because that tool is pinned
read-only (tests/test_node_usage_audit.py::test_audit_is_read_only) — it
gathers evidence and writes nothing. This one joins that evidence to the
hand-authored judgements in ``feature_audit_verdicts.py`` and, when present,
to a live probe, then writes the two tracked artifacts.

Usage:
    python tools/build_feature_audit.py [--check]

``--check`` regenerates in memory and fails if the committed artifacts are
stale, so CI can catch a node added without an audit row.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Run as a script, sys.path[0] is tools/, NOT the repo root — so a bare
# `import atlas_camera` resolves to whatever copy is installed (on a dev box
# that is usually the ComfyUI custom_nodes checkout) and this tool silently
# audits a DIFFERENT tree than the one it lives in.
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _module_of(node_key: str) -> str:
    """Owning module path for a registered node class.

    Deliberately NOT 'path:line'. A line number makes the committed report
    stale on ANY edit to a file that happens to contain nodes — an unrelated
    change 500 lines away shifts every entry below it and turns the freshness
    test permanently red, which trains people to ignore it.
    """
    import inspect

    from atlas_camera.comfy import node_registry as reg
    cls = (reg.NODE_CLASS_MAPPINGS.get(node_key)
           or reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS.get(node_key)
           or getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}).get(node_key))
    if cls is None:
        return ""
    try:
        path = Path(inspect.getsourcefile(cls)).resolve().relative_to(REPO)
        return str(path).replace(chr(92), "/")
    except Exception:  # noqa: BLE001
        return ""


def build() -> dict:
    audit_mod = _load("audit_node_usage")
    verdicts_mod = _load("feature_audit_verdicts")
    data = audit_mod.audit(REPO)

    probe_path = REPO / "reports" / "live_probe_baseline.json"
    probe = {}
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    exec_results = (probe.get("node_execution") or {}).get("results", {})

    nodes: dict[str, dict] = {}
    for key in sorted(data):
        rec = data[key]
        override = verdicts_mod.VERDICTS.get(key, {})
        if override.get("verdict"):
            verdict = override["verdict"]
        elif rec["kind"] == "experimental":
            verdict = "KEEP_EXPERIMENTAL"
        elif rec["kind"] == "legacy":
            verdict = "LEGACY_GATE"
        elif rec["kind"] == "ios":
            verdict = "IOS_GATE"
        elif rec["product_evidence"]:
            verdict = "KEEP_CORE"
        else:
            verdict = "HOLD_NEEDS_EVIDENCE"

        live = exec_results.get(key, {})
        nodes[key] = {
            "module": _module_of(key),
            "tier": rec["kind"],
            # COUNTS, not paths. Storing the matching file PATHS coupled this
            # artifact to the full text of tests/, tools/, mcp/ and docs/: it
            # carried 981 path entries and moved whenever any of those files was
            # renamed, so it changed in 46 of 120 commits while only 13 touched
            # the registry. The paths are still computed live — that is what
            # tests/test_node_usage_audit.py asserts against — they are just not
            # a thing this file has to stay fresh about.
            "example_workflows": len(rec["example_workflows"]),
            "dedicated_tests": len(rec["dedicated_tests"]),
            "tests": len(rec["tests"]),
            "mcp_tools": len(rec["mcp_tools"]),
            "repo_tools": len(rec["repo_tools"]),
            "docs": len(rec["docs"]),
            "product_evidence": rec["product_evidence"],
            "evidence_kinds": rec["evidence_kinds"],
            "live_execution": live.get("execution", "not_attempted"),
            "live_output": live.get("output", "not_attempted"),
            "live_note": live.get("note"),
            "overlapping_replacement": override.get("overlapping_replacement"),
            "known_defect": override.get("known_defect") or verdicts_mod.DEFECTS.get(key),
            "compatibility_risk": override.get("compatibility_risk"),
            "verdict": verdict,
            "evidence": override.get("evidence", []),
            "migration_action": override.get("migration_action"),
            "notes": override.get("notes"),
        }

    counts: dict[str, int] = {}
    for rec in nodes.values():
        counts[rec["tier"]] = counts.get(rec["tier"], 0) + 1
    counts["total"] = len(nodes)
    counts["no_product_evidence"] = sum(
        1 for r in nodes.values() if r["tier"] == "standard" and not r["product_evidence"])

    verdict_counts: dict[str, int] = {}
    for rec in nodes.values():
        verdict_counts[rec["verdict"]] = verdict_counts.get(rec["verdict"], 0) + 1

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "counts": counts,
        "verdict_counts": verdict_counts,
        "live_probe": probe.get("server"),
        "nodes": nodes,
    }


_LEGEND = """\
| Verdict | Meaning |
|---|---|
| `KEEP_CORE` | product evidence exists; stays in the standard registry |
| `KEEP_EXPERIMENTAL` | stays behind `ATLAS_EXPERIMENTAL` |
| `MIGRATE_CAPABILITY` | node goes, capability moves to a supported node first |
| `DEPRECATE` | still registered, marked superseded, removal scheduled |
| `LEGACY_GATE` | moved behind `ATLAS_LEGACY_NODES` this cycle |
| `IOS_GATE` | held behind `ATLAS_IOS` (iOS/Record3D capture); a v2 capability |
| `DELETE` | removed outright |
| `HOLD_NEEDS_EVIDENCE` | zero product evidence, no proven duplicate — keep, revisit |
"""


def without_generated_stamp(markdown: str) -> str:
    """Drop the ``Generated <date>`` line so only CONTENT counts as stale.

    ``render_markdown`` stamps today's UTC date, so a byte comparison of the
    committed markdown against a fresh render fails on any day after the one
    the artifacts were generated. That made ``--check`` report STALE with a
    date-only diff and the advertised remedy (re-run the builder) "fixed" it
    by rewriting the stamp — training the reader to regenerate without reading
    the diff, which is exactly how a genuinely stale artifact gets waved
    through. The staleness check must ignore the stamp; the JSON branch
    already does, by comparing ``["nodes"]`` alone.
    """
    return "\n".join(l for l in markdown.splitlines()
                     if not l.startswith("Generated "))


def render_markdown(report: dict) -> str:
    c = report["counts"]
    out = [
        "# Atlas Camera — feature audit",
        "",
        f"Generated {report['generated_utc']} by `tools/build_feature_audit.py`.",
        "Machine-gathered evidence from `tools/audit_node_usage.py`; judgements from ",
        "`tools/feature_audit_verdicts.py`. Regenerate with ",
        "`python tools/build_feature_audit.py` (`--check` verifies freshness).",
        "",
        "## What counts as evidence",
        "",
        "A node has **product evidence** when a shipping workflow uses it, a test",
        "exercises it specifically, or an MCP handler depends on it.",
        "",
        "Two exclusions are deliberate and are the point of this document:",
        "",
        "* **Registry and façade pin tests do not count.** They name every",
        "  registered node by construction, so before this rework all nodes read as",
        "  \"referenced\" and the signal was worthless.",
        "* **Documentation does not count.** Documenting a node proves intent, not",
        "  use — and all but one node is documented, so counting docs would flatten",
        "  the signal to nothing.",
        "",
        "**Absence of evidence is not a defect.** Every zero-evidence node was",
        "executed in-process during the baseline probe and every one returned",
        "meaningful output. Nothing here is DELETEd on suspicion.",
        "",
        "Run the audit from the main checkout, never a git worktree: the tool",
        "imports `atlas_camera` through the editable install while scanning files",
        "relative to itself, so a worktree silently audits a different tree.",
        "",
        "## Counts",
        "",
        f"* standard: **{c.get('standard', 0)}**",
        f"* experimental: **{c.get('experimental', 0)}**",
        f"* legacy: **{c.get('legacy', 0)}**",
        f"* total registered: **{c['total']}**",
        f"* standard nodes with no product evidence: **{c['no_product_evidence']}**",
        "",
        "| Verdict | Nodes |",
        "|---|---:|",
    ]
    for v, n in sorted(report["verdict_counts"].items(), key=lambda kv: -kv[1]):
        out.append(f"| `{v}` | {n} |")
    out += ["", "## Verdict legend", "", _LEGEND, "## Matrix", ""]

    cols = ("Name | Module | Tier | Workflows | Dedicated tests | Live exec | "
            "Meaningful output | MCP | Docs | Overlapping replacement | "
            "Known defect | Compat risk | Verdict | Migration action")
    out.append(f"| {cols} |")
    out.append("|" + "---|" * 14)

    def cell(x):
        if x is None or x == "":
            return "—"
        s = str(x).replace("|", "\\|").replace("\n", " ")
        return s if len(s) <= 160 else s[:157] + "…"

    for key, r in report["nodes"].items():
        out.append("| " + " | ".join([
            f"`{key}`", cell(r["module"]), r["tier"],
            str(r["example_workflows"]) or "0",
            str(r["dedicated_tests"]),
            r["live_execution"], r["live_output"],
            str(r["mcp_tools"]), str(r["docs"]),
            cell(r["overlapping_replacement"]), cell(r["known_defect"]),
            cell(r["compatibility_risk"]), f"**{r['verdict']}**",
            cell(r["migration_action"]),
        ]) + " |")

    hold = [k for k, r in report["nodes"].items() if r["verdict"] == "HOLD_NEEDS_EVIDENCE"]
    out += ["", "## Appendix — nodes held rather than cut", "",
            "Each of these executes correctly and returns meaningful output; what",
            "it lacks is a consumer. That is a reason to find evidence or schedule",
            "deprecation, not a reason to delete.", ""]
    for k in hold:
        r = report["nodes"][k]
        note = r.get("notes") or (r["evidence"][0] if r["evidence"] else "")
        out.append(f"* **`{k}`** — {note}")

    out += [
        "", "## Appendix — capabilities REMOVED, not replaced", "",
        "Retiring `AtlasLiveMeshRepair` to the legacy tier keeps one of its four",
        "capabilities and drops three. They are listed here so the migration is",
        "not mistaken for an equivalence:", "",
        "| Capability | Status | Where it went |",
        "|---|---|---|",
        "| Boundary Taubin smoothing | **migrated** | `AtlasRetopologizeLayer("
        "boundary_smooth_iterations)`, verbatim implementation, UVs regenerated |",
        "| CUDA 2D grid hole fill | **removed from the repair path** | "
        "`core/mesh_repair.repair_relief_mesh_grid_cuda` is no longer reachable "
        "from a default-tier REPAIR node, but it is NOT dead: `core/move_budget."
        "seal_relief_mesh` calls it to seal a mesh before measuring disocclusion, "
        "reached from `AtlasMoveBudget` (registered unconditionally), and "
        "`AtlasLiveMeshRepair` still calls it on the legacy tier. The repair "
        "replacement, `AtlasPlanarHolePatch`, is a different algorithm: "
        "per-component plane fitting with reports and gates, not a grid convolution |",
        "| Harmonic enclosed-hole cap | **removed from the repair path** | same "
        "function, same two surviving callers; the membrane fill for sealed "
        "pockets has no equivalent in the planar patch |",
        "| Post-hoc stretch cull (`remove_stretch_factor`) | **removed from the "
        "default tier** | `core/mesh_repair.remove_stretched_faces` is still "
        "called by `AtlasLiveMeshRepair` on the legacy tier. "
        "Deliberately NOT appended to `AtlasRetopologizeLayer`: every shipping "
        "workflow set it to 0.0, it has no node-level test, and `max_edge_factor` "
        "on the layer/derive nodes covers the same test at build time with the "
        "depth map in hand |", "",
        "All three operated *downstream, on an already-built",
        "solve*. What survives on the build path is unaffected: CPU hole fill and",
        "sawtooth bridging still run via `apply_live_mesh_repair` from",
        "`AtlasDeriveReliefMesh` and `AtlasDeriveProjectionGeometry`, and",
        "`apply_interior_hole_fill` still backs the relief-mesh exporter.",
        "",
        "Re-exposing any of the three is a one-widget append if evidence appears.",
        "",
    ]

    defects = [(k, r) for k, r in report["nodes"].items() if r["known_defect"]]
    if defects:
        out += ["", "## Appendix — known defects", ""]
        for k, r in defects:
            out.append(f"* **`{k}`** ({r['verdict']}) — {r['known_defect']}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed artifacts are stale")
    args = ap.parse_args()

    report = build()
    md = render_markdown(report)
    json_path = REPO / "reports" / "feature_audit.json"
    md_path = REPO / "docs" / "FEATURE_AUDIT.md"

    if args.check:
        stale = []
        if not json_path.exists() or json.loads(
                json_path.read_text(encoding="utf-8")).get("nodes") != report["nodes"]:
            stale.append(str(json_path))
        if not md_path.exists() or without_generated_stamp(
                md_path.read_text(encoding="utf-8")) != without_generated_stamp(md):
            stale.append(str(md_path))
        if stale:
            raise SystemExit("STALE (run tools/build_feature_audit.py): "
                             + ", ".join(stale))
        print("feature audit is up to date")
        return

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    c = report["counts"]
    print(f"wrote {json_path.relative_to(REPO)} and {md_path.relative_to(REPO)}")
    print(f"  {c['total']} nodes; {c['no_product_evidence']} standard with no product evidence")
    for v, n in sorted(report["verdict_counts"].items()):
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
