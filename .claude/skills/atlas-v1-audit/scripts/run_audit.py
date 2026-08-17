"""Orchestrator for /atlas-v1-audit. READ-ONLY with respect to the project.

The only writes are under `.v1-audit/`. No project file is edited, no file is
deleted, no dependency is changed, no autofix is run. `--apply` is deliberately
NOT implemented here: applying the manifest is a documented human procedure in
SKILL.md, because a script that can delete is a script that can delete by
accident.

    python .claude/skills/atlas-v1-audit/scripts/run_audit.py --bootstrap-venv

Phases run in order and each writes one json into `.v1-audit/raw/`; later
phases read earlier ones, so a failed phase stops the run rather than producing
a manifest built on missing evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent

#: (phase name, script, extra args). Order is a dependency order.
PHASES = [
    ("inventory", "inventory.py", []),
    ("nodes", "scan_nodes.py", []),
    ("workflows", "scan_workflows.py", []),
    ("references", "scan_references.py", []),
    ("docs", "scan_docs.py", []),
    ("experiments", "scan_experiments.py", []),
    ("tools", "scan_tools.py", []),
    ("manifest", "build_manifest.py", []),
]
#: `--quick` still RUNS the slow phase, in a mode that writes a stub result.
#: Skipping it outright would leave `raw/tools.json` missing and the manifest
#: would abort on it — a phase that later phases read cannot simply not run.
QUICK_STUB = {"tools": "--skip"}


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--bootstrap-venv", action="store_true",
                    help="install vulture/deptry/ruff into .v1-audit/.tools-venv "
                         "(~50 MB, never into the project environment)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow static-analysis phase")
    ap.add_argument("--check", action="store_true",
                    help="CI mode: exit non-zero if the tree has a broken "
                         "workflow, an unregistered node class, a live doc with "
                         "a dead path, or a live doc miscounting the registry")
    ap.add_argument("--only", help="run one phase by name and stop")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    state = common.git_state(root)
    if state["dirty"]:
        common.eprint(
            f"note: working tree is DIRTY ({state['dirty_files']} file(s)). "
            "Continuing read-only; the state is recorded in run.json."
        )

    phases = PHASES
    if args.only:
        phases = [p for p in PHASES if p[0] == args.only]
        if not phases:
            raise SystemExit(f"unknown phase: {args.only}\n"
                             f"known: {', '.join(p[0] for p in PHASES)}")

    results = []
    for name, script, extra in phases:
        cmd = [sys.executable, str(SCRIPTS / script), "--root", str(root), *extra]
        if args.quick and name in QUICK_STUB:
            cmd.append(QUICK_STUB[name])
        elif name == "tools" and args.bootstrap_venv:
            cmd.append("--bootstrap-venv")
        start = time.time()
        cp = subprocess.run(cmd)
        elapsed = round(time.time() - start, 1)
        results.append({"phase": name, "rc": cp.returncode, "seconds": elapsed})
        if cp.returncode != 0:
            common.eprint(f"phase {name} failed (rc={cp.returncode}); stopping.")
            break

    run = {
        "root": root.as_posix(),
        "git": state,
        "quick": args.quick,
        "bootstrap_venv": args.bootstrap_venv,
        "phases": results,
    }
    (common.out_dir(root) / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")

    failed = [r for r in results if r["rc"] != 0]
    if failed:
        return 1

    if not args.only:
        print(f"\nwrote {common.OUT_DIRNAME}/ — start with summary.md, then "
              "unknown-review.md. Nothing in the project tree was modified.")

    if args.check:
        return _check(root)
    return 0


def _check(root: Path) -> int:
    """CI gate. Deliberately narrow: only defects with no judgement in them."""
    nodes = common.read_raw(root, "nodes")
    workflows = common.read_raw(root, "workflows")
    docs = common.read_raw(root, "docs")

    problems: list[str] = []
    for path, wf in sorted(workflows["workflows"].items()):
        if wf.get("unresolved"):
            problems.append(f"{path}: unresolved node types "
                            f"{', '.join(wf['unresolved'])}")
    for cls in nodes["deregistered_classes"]:
        problems.append(f"{cls}: node-shaped class not in NODE_CLASS_MAPPINGS")
    for key in nodes["missing_implementation"]:
        problems.append(f"{key}: registered but no implementation found")
    for path, doc in sorted(docs["docs"].items()):
        if doc["provenance"]:
            continue
        for dead in doc["dead_paths"]:
            problems.append(f"{path}: names {dead}, which does not exist")
        for claim in doc["count_claims"]:
            problems.append(f"{path}:{claim['line']}: claims {claim['claim']}, "
                            f"registry has {claim['actual']}")

    if problems:
        print("\n--check FAILED:")
        for line in problems:
            print(f"  {line}")
        return 1
    print("\n--check passed: no broken workflows, no deregistered classes, "
          "no stale live-doc claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
