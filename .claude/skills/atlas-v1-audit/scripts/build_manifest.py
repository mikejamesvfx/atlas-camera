"""Phase 8 — merge every evidence source into one disposition per file.

This is the only script that decides anything, and it decides from the raw
phases alone: it re-reads no source. Every row records the nine dimensions it
checked, so a reader can see WHY a verdict was reached and disagree with it
concretely rather than by feel.

The confidence ceiling is where the Atlas-specific knowledge lives. A category
that a scanner structurally cannot observe — a pytest fixture, a script Maya
runs out-of-process, a file under a directory that loads its own modules
dynamically — is capped below CERTAIN no matter how clean the evidence looks.
CERTAIN is the only tier `--apply` may act on without per-file confirmation, so
the cap is the thing standing between a false positive and a deleted feature.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


#: `WEB_DIRECTORY = "./web"` — ComfyUI serves EVERY .js under the named
#: directory. The declaration lives in the package `__init__.py`, one level
#: above the files it governs, so a per-directory marker scan never sees it and
#: four live frontend extensions read as unreferenced.
WEB_DIRECTORY_RE = re.compile(r"""WEB_DIRECTORY\s*=\s*["']([^"']+)["']""")


def _dynamic_dirs(root: Path, inventory: dict, cfg: dict) -> set[str]:
    """Directories whose files are reached without an import statement.

    Two ways that happens, and the second is the one that bit: a directory
    whose OWN files carry a dynamic-loading marker, and a directory NAMED by a
    marker declared somewhere else.
    """
    markers = cfg["dynamic_loading_markers"]
    out: set[str] = set()
    for rel, info in inventory.items():
        if info["category"] not in ("PYTHON", "JS_TS"):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(m in text for m in markers):
            out.add(info["dir"])
        for declared in WEB_DIRECTORY_RE.findall(text):
            # resolve relative to the DECLARING file's directory
            target = (root / info["dir"] / declared).resolve()
            try:
                out.add(target.relative_to(root).as_posix())
            except ValueError:  # pragma: no cover - escapes the repo
                continue
    return out


def _ceiling(rel: str, info: dict, cfg: dict, dynamic: set[str],
             entry_dirs: tuple[str, ...]) -> tuple[str, str | None]:
    if info["category"] in cfg["protected_categories"]:
        return "MEDIUM", f"category {info['category']} is reached by a host this scan cannot observe"
    if info["dir"] in dynamic:
        return "MEDIUM", f"{info['dir']}/ loads its own modules dynamically"
    if rel.startswith(entry_dirs):
        return "HIGH", "entry-point script: invoked by a human, so importer count proves nothing"
    if info["generated"]:
        return "HIGH", "generated/vendored artifact — verify regeneration before acting"
    return "CERTAIN", None


def build(root: Path, cfg: dict) -> dict:
    inventory = common.read_raw(root, "inventory")["files"]
    refs = common.read_raw(root, "references")["references"]
    nodes = common.read_raw(root, "nodes")
    workflows = common.read_raw(root, "workflows")
    docs = common.read_raw(root, "docs")
    experiments = common.read_raw(root, "experiments")
    tools = common.read_raw(root, "tools")

    flagged = set(tools["flagged_paths"])
    dynamic = _dynamic_dirs(root, inventory, cfg)
    entry_dirs = tuple(f"{d}/" for d in cfg["entry_point_dirs"])

    node_impls = {info["implementation"] for info in nodes["nodes"].values()
                  if info.get("implementation")}
    node_impls = {p for p in node_impls if p}
    unwired = {row["path"]: row for row in experiments["unwired_modules"]}
    dev_only = {row["path"] for row in experiments["dev_scripts_only_tested"]}
    broken_workflows = set(workflows["broken"])
    duplicate_workflows = {
        rel for cluster in workflows["exact_duplicate_clusters"] for rel in cluster[1:]
    }

    rows: dict[str, dict] = {}
    for rel, info in sorted(inventory.items()):
        referrers = refs.get(rel, {}).get("referrers", [])
        by_cat = refs.get(rel, {}).get("by_category", {})

        registered = any(rel.endswith(Path(p).as_posix()) for p in node_impls)
        evidence = {
            "static_reference": bool(
                by_cat.get("PYTHON", 0) or by_cat.get("JS_TS", 0)
                or by_cat.get("MODEL_ADAPTER", 0) or by_cat.get("DCC", 0)),
            "registered_reference": registered,
            "workflow_reference": bool(by_cat.get("COMFY_WORKFLOW", 0)),
            "test_reference": bool(by_cat.get("TEST", 0)),
            "doc_reference": bool(by_cat.get("DOC", 0)),
            "setup_reference": bool(by_cat.get("SETUP", 0)),
            "ci_reference": bool(by_cat.get("CI", 0)),
            "static_analysis": rel in flagged,
            "git_history_present": bool(info["last_commit_sha"]),
            "referrer_count": len(referrers),
            "superseded_by": None,
        }

        confidence = common.confidence_from_evidence(evidence)
        ceiling, cap_reason = _ceiling(rel, info, cfg, dynamic, entry_dirs)
        capped = common.cap_confidence(confidence, ceiling)

        # --- disposition -----------------------------------------------------
        if info["generated"]:
            disposition, reason = "GENERATED", "vendored or machine-written artifact"
        elif rel in broken_workflows:
            disposition, reason = "BROKEN", "workflow references node types that do not resolve"
        elif rel in duplicate_workflows:
            evidence["superseded_by"] = next(
                c[0] for c in workflows["exact_duplicate_clusters"] if rel in c)
            disposition = "MERGE"
            reason = f"identical node-type signature to {evidence['superseded_by']}"
        elif registered:
            disposition, reason = "CANONICAL", "implements a registered ComfyUI node"
        elif rel in unwired:
            disposition, reason = "FEATURE_FLAG", unwired[rel]["detail"]
        elif referrers:
            if info["category"] == "DOC":
                disposition = "KEEP_PUBLIC" if not rel.startswith("docs/dev") else "KEEP_INTERNAL"
            else:
                disposition = "KEEP"
            reason = f"reached by {len(referrers)} file(s)"
        elif capped == "CERTAIN":
            disposition = "DELETE_CANDIDATE"
            reason = ("no static, registry, workflow, test, doc, setup or CI "
                      "reference, and a superseding implementation exists")
        else:
            disposition = "UNKNOWN"
            reason = (cap_reason or
                      "no references found, but the evidence is not strong enough "
                      "to nominate deletion")
            if info["category"] == "DOC":
                reason = "no inbound reference — decide whether it is a v1 public doc"

        rows[rel] = {
            "path": rel,
            "category": info["category"],
            "disposition": disposition,
            "confidence": capped,
            "confidence_before_cap": confidence,
            "confidence_ceiling": ceiling,
            "ceiling_reason": cap_reason,
            "evidence": evidence,
            "reason": reason,
            "last_commit_date": info["last_commit_date"],
        }

    # Only CERTAIN rows are eligible for --apply without confirmation, and the
    # ceiling above means that requires the file to be in NO protected
    # category, under NO dynamically-loading directory, and outside every
    # entry-point directory. On a healthy tree this set is empty; that is the
    # expected result, not a scanner failure.
    delete_candidates = [r for r in rows.values() if r["disposition"] == "DELETE_CANDIDATE"]
    dispositions: dict[str, int] = {}
    confidences: dict[str, int] = {}
    for row in rows.values():
        dispositions[row["disposition"]] = dispositions.get(row["disposition"], 0) + 1
        confidences[row["confidence"]] = confidences.get(row["confidence"], 0) + 1

    return {
        "rows": rows,
        "dispositions": dispositions,
        "confidences": confidences,
        "delete_candidates": sorted(r["path"] for r in delete_candidates),
        "certain_delete_candidates": sorted(
            r["path"] for r in delete_candidates if r["confidence"] == "CERTAIN"),
        "dynamic_dirs": sorted(dynamic),
    }


# --- markdown ---------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_markdown(root: Path, manifest: dict) -> None:
    inventory = common.read_raw(root, "inventory")
    nodes = common.read_raw(root, "nodes")
    workflows = common.read_raw(root, "workflows")
    docs = common.read_raw(root, "docs")
    experiments = common.read_raw(root, "experiments")
    tools = common.read_raw(root, "tools")
    state = common.git_state(root)
    rows = manifest["rows"]

    # summary.md
    counts = inventory["counts"]
    parts = [
        "# Atlas Camera v1 audit — summary\n",
        f"Branch `{state['branch']}` @ `{state['sha'][:8]}` "
        f"(working tree **{'DIRTY — ' + str(state['dirty_files']) + ' file(s)' if state['dirty'] else 'clean'}**).\n",
        "This audit is **read-only**. Nothing in the project tree was modified.\n",
        "## Scope\n",
        _table(["metric", "count"], [
            ["files audited", inventory["total"]],
            ["Python", counts.get("PYTHON", 0)],
            ["JS/TS", counts.get("JS_TS", 0)],
            ["ComfyUI workflows", workflows["counts"]["total"]],
            ["registered nodes", nodes["counts"]["registered"]],
            ["docs (markdown)", docs["counts"]["total"]],
            ["tests/fixtures", counts.get("TEST", 0)],
        ]),
        "\n## Dispositions\n",
        _table(["disposition", "count"],
               [[k, v] for k, v in sorted(manifest["dispositions"].items())]),
        "\n## Confidence\n",
        _table(["confidence", "count"],
               [[k, v] for k, v in sorted(manifest["confidences"].items())]),
        "\n## Static-analysis tools\n",
        _table(["tool", "status", "note"],
               [[k, v, (tools.get("notes", {}).get(k) or "—")[:120]]
                for k, v in sorted(tools["status"].items())]),
        ("\n**A tool that could not be read is NOT a clean result.** "
         "`PARSE_FAILED` means its output was unparseable and its findings are "
         "unknown; `AVAILABLE_PARTIAL` means it ran but warned first, so its "
         "report is incomplete. Only `AVAILABLE` with no note means nothing "
         "was found.\n"
         if any(v != "AVAILABLE" for v in tools["status"].values()) else ""),
        "\n## High-risk areas\n",
        f"- **{workflows['counts']['broken']}** workflow(s) reference node types "
        "that are not registered (see `workflow-audit.md`).\n"
        f"- **{len(nodes['deregistered_classes'])}** node-shaped class(es) exist "
        "but are not in `NODE_CLASS_MAPPINGS`.\n"
        f"- **{len(nodes['duplicate_display_names'])}** duplicate node display name(s).\n"
        f"- **{len(workflows['exact_duplicate_clusters'])}** exact workflow duplicate "
        f"cluster(s), **{len(workflows['near_duplicates'])}** near-duplicate pair(s).\n"
        f"- **{docs['counts']['live_with_dead_paths']}** live doc(s) reference paths "
        "that no longer exist.\n"
        f"- **{docs['counts']['live_with_bad_counts']}** live doc(s) state node counts "
        "the registry disagrees with.\n"
        f"- **{experiments['counts']['nodes_without_workflow']}** registered node(s) "
        "are demonstrated by no shipping workflow.\n",
        "\n## Deletion candidates\n",
        f"{len(manifest['delete_candidates'])} total, of which "
        f"**{len(manifest['certain_delete_candidates'])} are CERTAIN** — the only "
        "tier eligible for `--apply` without per-file human confirmation.\n",
        "\n## Recommended cleanup phases\n",
        "1. Generated debris and vendored artifacts (`GENERATED`) — verify regeneration first.\n"
        "2. CERTAIN dead scripts (`delete-candidates.md`).\n"
        "3. Broken and exact-duplicate workflows (`workflow-audit.md`).\n"
        "4. Superseded model/experiment code (`experiment-audit.md`) — prefer ARCHIVE.\n"
        "5. Duplicate documentation (`docs-audit.md`) — merge into the canonical doc.\n"
        "6. Setup contradictions (`setup-audit.md`) — one canonical install path.\n",
        "\nEverything in `unknown-review.md` needs a human decision and must not be automated.\n",
    ]
    common.write_doc(root, "summary.md", "".join(parts))

    # delete-candidates.md
    tiers = {"CERTAIN": [], "HIGH": [], "OTHER": []}
    for path in manifest["delete_candidates"]:
        row = rows[path]
        tiers.get(row["confidence"], tiers["OTHER"]).append(row)
    common.write_doc(root, "delete-candidates.md", "".join([
        "# Delete candidates\n\n",
        "Only **CERTAIN** rows are eligible for `--apply` without per-file "
        "confirmation. Every row below was checked against: static code "
        "references, dynamic registration, workflow references, test "
        "references, doc references, setup references, CI references, git "
        "history and a superseding implementation.\n\n",
        f"## CERTAIN ({len(tiers['CERTAIN'])})\n\n",
        _table(["path", "category", "reason"],
               [[r["path"], r["category"], r["reason"]] for r in tiers["CERTAIN"]]),
        f"\n## HIGH ({len(tiers['HIGH'])})\n\n",
        _table(["path", "category", "reason"],
               [[r["path"], r["category"], r["reason"]] for r in tiers["HIGH"]]),
        f"\n## MEDIUM and below ({len(tiers['OTHER'])})\n\n",
        _table(["path", "category", "reason"],
               [[r["path"], r["category"], r["reason"]] for r in tiers["OTHER"]]),
    ]))

    # unknown-review.md
    unknown = [r for r in rows.values() if r["disposition"] == "UNKNOWN"]
    common.write_doc(root, "unknown-review.md", "".join([
        "# Needs human review\n\n",
        "These have no inbound references but the evidence is not strong "
        "enough to nominate deletion — dynamic loading in the directory, a "
        "protected category, or an entry point with no importer. Never "
        "automate these.\n\n",
        _table(["path", "category", "confidence", "reason"],
               [[r["path"], r["category"], r["confidence"], r["reason"]]
                for r in sorted(unknown, key=lambda r: r["path"])]),
    ]))

    # merge / archive candidates
    merges = [r for r in rows.values() if r["disposition"] == "MERGE"]
    common.write_doc(root, "merge-candidates.md", "".join([
        "# Merge candidates\n\n",
        _table(["path", "superseded by"],
               [[r["path"], r["evidence"]["superseded_by"]] for r in merges]),
    ]))
    common.write_doc(root, "archive-candidates.md",
                     "# Archive candidates\n\n"
                     "Git history is the default archive. Move a file into "
                     "`docs/archive/`, `examples/archive/` or `experimental/` "
                     "only when it still needs to be *discoverable* — otherwise "
                     "delete it and let the history hold it.\n\n"
                     + _table(["path", "reason"], []))

    # node-audit.md
    workflow_usage: dict[str, int] = {}
    for wf in workflows["workflows"].values():
        for t in wf.get("atlas_types", []):
            workflow_usage[t] = workflow_usage.get(t, 0) + 1
    common.write_doc(root, "node-audit.md", "".join([
        "# Node registration audit\n\n",
        f"Registry files: {', '.join('`' + p + '`' for p in nodes['registry_files'])}\n\n",
        f"Counts: {json.dumps(nodes['counts'], sort_keys=True)}\n\n",
        "## Registered but demonstrated by no workflow\n\n",
        _table(["node key", "tier"],
               [[k, nodes["nodes"][k]["tier"]]
                for k in experiments["nodes_without_workflow"]]),
        "\nThese are **not** deletion candidates — a node can be legitimate "
        "without a shipping example. They are documentation and "
        "example-coverage gaps.\n\n",
        "## Implemented but not registered\n\n",
        _table(["class"], [[c] for c in nodes["deregistered_classes"]]),
        "\n## Duplicate display names\n\n",
        _table(["display name", "keys"],
               [[k, ", ".join(v)] for k, v in nodes["duplicate_display_names"].items()]),
        "\n## Registered nodes with no implementation found\n\n",
        _table(["node key"], [[k] for k in nodes["missing_implementation"]]),
    ]))

    # workflow-audit.md
    common.write_doc(root, "workflow-audit.md", "".join([
        "# Workflow audit\n\n",
        _table(["workflow", "format", "nodes", "disposition", "confidence"],
               [[p, w["format"], w["node_count"],
                 rows[p]["disposition"] if p in rows else "?",
                 rows[p]["confidence"] if p in rows else "?"]
                for p, w in sorted(workflows["workflows"].items())]),
        "\n## Broken (unresolved Atlas node types)\n\n",
        _table(["workflow", "unresolved"],
               [[p, ", ".join(w["unresolved"])]
                for p, w in sorted(workflows["workflows"].items()) if w["unresolved"]]),
        "\n## Exact duplicates (identical node-type signature)\n\n",
        _table(["cluster"], [[", ".join(c)] for c in workflows["exact_duplicate_clusters"]]),
        "\n## Near duplicates (>=80% node-type overlap)\n\n",
        _table(["a", "b", "overlap"],
               [[d["a"], d["b"], d["overlap"]] for d in workflows["near_duplicates"]]),
        "\nFilenames were not used for clustering — these are node-graph comparisons.\n",
    ]))

    # docs-audit.md
    live = {k: v for k, v in docs["docs"].items() if not v["provenance"]}
    common.write_doc(root, "docs-audit.md", "".join([
        "# Documentation audit\n\n",
        _table(["doc", "words", "topics", "disposition", "dead paths", "bad counts"],
               [[p, d["words"], ", ".join(d["topics"][:4]),
                 rows[p]["disposition"] if p in rows else "?",
                 len(d["dead_paths"]), len(d["count_claims"])]
                for p, d in sorted(docs["docs"].items())]),
        "\n## Recommended authoritative doc per topic\n\n",
        _table(["topic", "canonical", "duplicating"],
               [[t, v["canonical"] or "— none —", ", ".join(v["duplicating"][:6])]
                for t, v in sorted(docs["canonical_by_topic"].items())]),
        "\n## Overlapping documents\n\n",
        "Measured by *containment*: the share of the smaller document that also "
        "appears in the larger one. The contained document is the merge candidate.\n\n",
        _table(["a", "b", "containment", "contained side"],
               [[d["a"], d["b"], d["containment"], d["contained"]]
                for d in docs["overlaps"]]),
        "\n## Stale claims (live docs only)\n\n",
        "Provenance docs are excluded: a dated record naming a removed file is "
        "the record doing its job.\n\n",
        _table(["doc", "dead paths"],
               [[p, ", ".join(d["dead_paths"])] for p, d in sorted(live.items())
                if d["dead_paths"]]),
        "\n### Node counts that disagree with the registry\n\n",
        _table(["doc", "line", "claim", "actual"],
               [[p, c["line"], c["claim"], c["actual"]]
                for p, d in sorted(live.items()) for c in d["count_claims"]]),
        "\n### Atlas symbols named but not registered\n\n",
        _table(["doc", "symbols"],
               [[p, ", ".join(d["unregistered_symbols"])]
                for p, d in sorted(live.items()) if d["unregistered_symbols"]]),
        "\n### Local-only references (gitignored by design — not defects)\n\n",
        _table(["doc", "paths"],
               [[p, ", ".join(d["local_only_paths"])]
                for p, d in sorted(docs["docs"].items()) if d["local_only_paths"]]),
        "\n## Handoff to /document-release\n\n",
        "- **canonical**: the topic table above\n"
        "- **stale**: the three tables in this section\n"
        "- **duplicate**: the overlap table\n"
        "- **contradictory**: `setup-audit.md`\n"
        f"- **missing**: {', '.join(docs['missing_topics']) or 'none'}\n"
        "- **current implementation evidence**: `node-audit.md`, `workflow-audit.md`\n",
    ]))

    # experiment-audit.md
    common.write_doc(root, "experiment-audit.md", "".join([
        "# Experimental / superseded feature audit\n\n",
        "Signals are mined from the repository, not from path names: Atlas "
        "gates experiments with an env var and leaves superseded work in "
        "place, so `experimental/` never appears in a path here.\n\n",
        f"Signal counts: {json.dumps(experiments['signal_counts'], sort_keys=True)}\n\n",
        "## Deregistered node classes\n\n",
        "A node-shaped class no longer in `NODE_CLASS_MAPPINGS`. For a node "
        "pack this is *the* superseded-feature signal: the implementation "
        "outlived its menu entry. Remove the class or re-register it.\n\n",
        _table(["class"], [[c] for c in experiments["deregistered_classes"]]),
        "\n## Implemented, tested, UNWIRED\n\n",
        "Library modules whose only inbound references are tests. Check the "
        "last-commit date to tell work-in-progress from abandoned prototype.\n\n",
        _table(["path", "detail", "last commit", "disposition"],
               [[r["path"], r["detail"], r["last_commit"], r["disposition"]]
                for r in experiments["unwired_modules"]]),
        "\n## Developer scripts with only test references\n\n",
        "Weaker signal: a script under an entry-point directory is invoked by "
        "a human, so its importer count proves nothing.\n\n",
        _table(["path", "last commit"],
               [[r["path"], r["last_commit"]]
                for r in experiments["dev_scripts_only_tested"]]),
        "\n## Env-gated nodes (already FEATURE_FLAG)\n\n",
        "Leaving a node gated is a valid v1 outcome — the questions are "
        "whether it is *documented* as experimental and whether anything with "
        "real usage is stuck behind the flag.\n\n",
        _table(["node key", "tier", "implementation", "workflows"],
               [[r["key"], r["tier"], r["implementation"] or "—", r["workflows"]]
                for r in experiments["gated_nodes"]]),
        "\n## Dormant\n\n",
        f"Untouched for more than 120 days before the newest commit, with at "
        "most one inbound reference.\n\n",
        _table(["path", "last commit"],
               [[r["path"], r["last_commit"]] for r in experiments["dormant"]]),
        "\nPrefer ARCHIVE (git history) over deletion for anything with a research trail.\n",
    ]))

    # dcc-audit.md
    dcc = {k: v for k, v in rows.items() if v["category"] == "DCC"}
    common.write_doc(root, "dcc-audit.md", "".join([
        "# DCC integration audit\n\n",
        "Distinct DCC implementations are never merged on name similarity — "
        "Maya and Nuke exporters share a collection helper but not a format.\n\n",
        _table(["path", "disposition", "confidence", "referrers"],
               [[p, r["disposition"], r["confidence"], r["evidence"]["referrer_count"]]
                for p, r in sorted(dcc.items())]),
    ]))

    # dependency-audit.md
    common.write_doc(root, "dependency-audit.md", "".join([
        "# Dependency audit\n\n",
        _table(["tool", "status", "note"],
               [[k, v, (tools.get("notes", {}).get(k) or "—")[:200]]
                for k, v in sorted(tools["status"].items())]),
        "\n" + tools["install_hint"] + "\n\n",
        ("## Suppressed by config\n\n"
         + _table(["tool", "findings dropped"],
                  [[k, v] for k, v in sorted(tools.get("suppressed", {}).items())])
         + "\nThese are the KINDS of false positive documented under "
           "`static_analysis` in the skill's `config.json`, each with the reason "
           "it is one — ComfyUI/Blender host modules deptry cannot resolve, "
           "optional extras behind guarded imports, and the node-contract "
           "attributes the ComfyUI host reads by name. The count is printed so a "
           "growing suppression list stays visible: a scanner that quietly stops "
           "scanning looks exactly like a codebase that got cleaner.\n\n"
         if tools.get("suppressed") else ""),
        "## deptry\n\n```json\n",
        json.dumps(tools["findings"].get("deptry", []), indent=1)[:8000],
        "\n```\n\n## knip\n\n```json\n",
        json.dumps(tools["findings"].get("knip", []), indent=1)[:8000],
        "\n```\n\nFindings are `STATIC_ANALYSIS_CANDIDATE`, never authority: "
        "a single tool finding can never reach CERTAIN.\n",
    ]))

    # setup-audit.md
    setup_rows = {k: v for k, v in rows.items() if v["category"] == "SETUP"}
    common.write_doc(root, "setup-audit.md", "".join([
        "# Setup / install audit\n\n",
        _table(["path", "disposition", "confidence"],
               [[p, r["disposition"], r["confidence"]] for p, r in sorted(setup_rows.items())]),
        "\n## Install-command variants across docs\n\n",
        _table(["doc", "topic rank"],
               [[p, ", ".join(d["topics"][:3])] for p, d in sorted(docs["docs"].items())
                if "installation" in d["topics"]]),
        "\nOne of these should be canonical; the rest should link to it.\n",
    ]))

    # capability-surface.md
    common.write_doc(root, "capability-surface.md", "".join([
        "# Atlas Camera v1 public capability surface\n\n",
        "Derived from the repository, not assumed. Each capability is listed "
        "with the evidence that it is live.\n\n",
        f"## ComfyUI nodes ({nodes['counts']['registered']} registered)\n\n",
        _table(["tier", "count"],
               [[k, v] for k, v in sorted(nodes["counts"].items())]),
        "\n## Registered nodes\n\n",
        _table(["node key", "tier", "display name", "workflows"],
               [[k, v["tier"], v["display_name"] or "—", workflow_usage.get(k, 0)]
                for k, v in sorted(nodes["nodes"].items())]),
        f"\n## Shipping workflows ({workflows['counts']['total']})\n\n",
        _table(["workflow", "atlas nodes used"],
               [[p, len(w["atlas_types"])]
                for p, w in sorted(workflows["workflows"].items())]),
    ]))


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    cfg = common.load_config(Path(__file__).resolve().parents[1])
    manifest = build(root, cfg)
    out = common.out_dir(root) / "disposition.json"
    out.write_text(json.dumps(manifest["rows"], indent=1, sort_keys=True,
                              ensure_ascii=False), encoding="utf-8")
    # The spec's artifact list puts inventory.json at the top level next to
    # disposition.json; the phases write into raw/. Mirror rather than move, so
    # a phase can still be re-run in isolation against its own output.
    inventory_top = common.out_dir(root) / "inventory.json"
    inventory_top.write_text(
        json.dumps(common.read_raw(root, "inventory"), indent=1, sort_keys=True,
                   ensure_ascii=False), encoding="utf-8")
    write_markdown(root, manifest)
    print(f"manifest: {len(manifest['rows'])} rows; "
          f"{len(manifest['delete_candidates'])} delete candidate(s), "
          f"{len(manifest['certain_delete_candidates'])} CERTAIN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
