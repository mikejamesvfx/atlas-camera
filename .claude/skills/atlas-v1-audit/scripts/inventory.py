"""Phase 1 — inventory every tracked file, with git history as evidence.

Disposition is NOT decided here. This phase only answers "what exists, what
kind of thing is it, and when was it last genuinely touched" so later phases
have one shared file list to reason over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _history(root: Path, paths: list[str]) -> dict[str, dict]:
    """One `git log` pass over the whole tree, not one per file.

    Per-file logs cost ~40ms each; on 500 files that is 20 seconds for data a
    single --name-only walk produces in under one.
    """
    out: dict[str, dict] = {}
    log = common.git(
        root, "log", "--all", "--name-only", "--no-renames",
        "--pretty=format:\x01%H\x02%cI",
    )
    sha = date = None
    for line in log.splitlines():
        if line.startswith("\x01"):
            sha, _, date = line[1:].partition("\x02")
            continue
        rel = line.strip()
        if not rel:
            continue
        entry = out.setdefault(rel, {
            "last_commit_sha": sha, "last_commit_date": date,
            "first_commit_date": date, "commit_count": 0,
        })
        entry["commit_count"] += 1
        entry["first_commit_date"] = date  # log walks newest-first
    known = set(paths)
    return {k: v for k, v in out.items() if k in known}


def build(root: Path, cfg: dict) -> dict:
    paths = [p for p in common.tracked_files(root) if not common.is_excluded(p, cfg)]
    hist = _history(root, paths)

    files: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for rel in paths:
        abs_path = root / rel
        category = common.categorize(rel)
        counts[category] = counts.get(category, 0) + 1
        try:
            size = abs_path.stat().st_size
        except OSError:
            size = 0
        h = hist.get(rel, {})
        files[rel] = {
            "path": rel,
            "dir": str(Path(rel).parent).replace("\\", "/"),
            "ext": Path(rel).suffix,
            "size": size,
            "category": category,
            "tracked": True,
            "generated": any(Path(rel).match(g) for g in cfg["generated_globs"]),
            "commit_count": h.get("commit_count", 0),
            "last_commit_sha": h.get("last_commit_sha"),
            "last_commit_date": h.get("last_commit_date"),
            "first_commit_date": h.get("first_commit_date"),
        }
    return {"counts": counts, "files": files, "total": len(files)}


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    cfg = common.load_config(Path(__file__).resolve().parents[1])
    payload = build(root, cfg)
    common.write_raw(root, "inventory", payload)
    print(f"inventory: {payload['total']} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
