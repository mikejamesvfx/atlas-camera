"""Phase 6 — experimental and superseded work still living on `main`.

Signals are mined from the repository, not from path names. Atlas gates
experiments with an environment variable and leaves superseded work in place,
so `experimental/` never appears in a path here and a name-based scan would
find nothing.

The four signals, strongest first:

* **DEREGISTERED** — a node-shaped class no longer in `NODE_CLASS_MAPPINGS`.
  For a node pack this is *the* superseded-feature signal: the implementation
  outlived its menu entry. Remove the class or re-register it; leaving it is
  drift.
* **TEST_ONLY_REACHABLE** — a library module whose only inbound references are
  tests. Its own tests keep it green while nothing in the product reaches it.
  Read the last-commit date to tell new-work-not-yet-wired-up from abandoned
  prototype; the audit cannot tell those apart and does not try.
* **DEV_SCRIPT_ONLY_TESTED** — the same shape under an entry-point directory.
  Reported separately and weaker: a human invokes those.
* **GATED_NODE** — registered behind a flag. A valid v1 outcome. The questions
  are whether it is documented as experimental and whether anything with real
  usage is stuck behind the flag.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DORMANT_DAYS = 120


def _parse(date: str | None):
    if not date:
        return None
    try:
        return datetime.fromisoformat(date.replace("Z", "+00:00"))
    except ValueError:
        return None


def build(root: Path, cfg: dict) -> dict:
    inventory = common.read_raw(root, "inventory")["files"]
    refs = common.read_raw(root, "references")["references"]
    nodes = common.read_raw(root, "nodes")
    workflows = common.read_raw(root, "workflows")["workflows"]

    entry_dirs = tuple(f"{d}/" for d in cfg["entry_point_dirs"])
    signals: dict[str, int] = {}

    def bump(name: str) -> None:
        signals[name] = signals.get(name, 0) + 1

    unwired: list[dict] = []
    dev_scripts: list[dict] = []
    for rel, info in sorted(inventory.items()):
        if info["category"] not in ("PYTHON", "MODEL_ADAPTER", "DCC"):
            continue
        referrers = refs.get(rel, {}).get("referrers", [])
        if not referrers:
            continue
        # DOC referrers do not count. A roadmap entry explaining that a module
        # is not yet wired up is EVIDENCE OF the unwired state, not a refutation
        # of it — and letting a doc mention mask the signal hid
        # `core/depth_calibration.py`, the one module this phase exists to find.
        code = [r for r in referrers
                if r in inventory and inventory[r]["category"] != "DOC"]
        if not code:
            continue
        kinds = {inventory[r]["category"] for r in code}
        if kinds != {"TEST"}:
            continue
        row = {
            "path": rel,
            "test_referrers": len(code),
            "last_commit": info["last_commit_date"],
            "detail": f"only inbound references are tests ({len(referrers)} file(s)); "
                      "no product code path reaches it",
        }
        if rel.startswith(entry_dirs):
            row["disposition"] = "KEEP"
            dev_scripts.append(row)
            bump("DEV_SCRIPT_ONLY_TESTED")
        else:
            row["disposition"] = "FEATURE_FLAG"
            unwired.append(row)
            bump("TEST_ONLY_REACHABLE")

    workflow_usage: dict[str, int] = {}
    for wf in workflows.values():
        for node_type in wf.get("atlas_types", []):
            workflow_usage[node_type] = workflow_usage.get(node_type, 0) + 1

    gated = []
    for key, info in sorted(nodes["nodes"].items()):
        if info["tier"] == "standard":
            continue
        bump("GATED_NODE")
        gated.append({
            "key": key,
            "tier": info["tier"],
            "implementation": info["implementation"],
            "workflows": workflow_usage.get(key, 0),
        })

    for _ in nodes["deregistered_classes"]:
        bump("DEREGISTERED")

    newest = max(
        (d for d in (_parse(i["last_commit_date"]) for i in inventory.values()) if d),
        default=datetime.now(timezone.utc),
    )
    cutoff = newest - timedelta(days=DORMANT_DAYS)
    dormant = []
    for rel, info in sorted(inventory.items()):
        stamp = _parse(info["last_commit_date"])
        if stamp and stamp < cutoff and len(refs.get(rel, {}).get("referrers", [])) <= 1:
            dormant.append({"path": rel, "last_commit": info["last_commit_date"]})

    no_workflow = sorted(k for k in nodes["nodes"] if k not in workflow_usage)

    return {
        "signal_counts": signals,
        "deregistered_classes": nodes["deregistered_classes"],
        "unwired_modules": unwired,
        "dev_scripts_only_tested": dev_scripts,
        "gated_nodes": gated,
        "dormant": dormant,
        "nodes_without_workflow": no_workflow,
        "counts": {
            "unwired": len(unwired),
            "gated": len(gated),
            "dormant": len(dormant),
            "nodes_without_workflow": len(no_workflow),
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
    common.write_raw(root, "experiments", payload)
    c = payload["counts"]
    print(f"experiments: {c['unwired']} unwired module(s), {c['gated']} gated node(s), "
          f"{c['dormant']} dormant file(s), "
          f"{c['nodes_without_workflow']} node(s) with no shipping workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
