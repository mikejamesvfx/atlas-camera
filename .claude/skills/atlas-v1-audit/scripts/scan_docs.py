"""Phase 5 — documentation archaeology.

Cross-references what the docs CLAIM against what the tree contains: paths that
no longer resolve, Atlas symbols that are not registered, node counts that
disagree with the registry, and topic overlap between documents.

Three classes of dead reference are NOT defects and are separated out rather
than reported:

* **provenance** — a dated record naming a file that was later removed. The
  changelog and the design rules exist to say what was true on a date; editing
  them to match today destroys the evidence they were written to preserve.
* **local-only** — a path git ignores by design, which the maintainer has on
  disk and a clone does not.
* **foreign** — a path in another project's tree, named to say where a
  behaviour was mirrored from. Correct as written; can never resolve here.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

#: Extensions are LONGEST-FIRST. Python's alternation is leftmost-match, not
#: longest-match, so `ts` before `tsx` truncates `Viewport3D.tsx` to
#: `Viewport3D.ts` — a file that does not exist — and the doc gets reported as
#: naming a dead path when it is correct. Same trap for js/jsx.
PATH_RE = re.compile(
    r"(?<![\w/])((?:examples|docs|tests|tools|atlas_camera|blender|ui|research"
    r"|reference_data|reports|scripts)/[\w./-]+"
    r"\.(?:json|jsx|tsx|toml|txt|md|py|js|ts))(?![\w])"
)
GIT_SHOW_RE = re.compile(r"git\s+show\s+[0-9a-f]{7,40}:([\w./-]+)")
SYMBOL_RE = re.compile(r"\bAtlas[A-Z][A-Za-z0-9]{2,}\b")
COUNT_RE = re.compile(r"(\d+)\s+(standard|experimental|legacy|iOS|registered)\b")

TOPICS = {
    "installation": ("install", "pip install", "venv", "symlink", "extra"),
    "quickstart": ("quickstart", "getting started", "first run"),
    "camera-solving": ("solve", "intrinsics", "extrinsics", "vanishing point",
                       "focal", "gravity"),
    "geometry": ("relief mesh", "proxy", "primitive", "tear", "retopo", "mesh"),
    "depth": ("depth", "moge", "depth anything", "normal"),
    "dcc-export": ("nuke", "maya", "blender", "usd", "exporter", "export"),
    "workflows": ("workflow", "comfyui", "node graph", "example"),
    "dynamic-plates": ("dynamic plate", "temporal projection", "ltx"),
    "concepts": ("concept", "doctrine", "philosophy", "mental model"),
    "troubleshooting": ("troubleshoot", "if it fails", "error", "fix"),
    "architecture": ("architecture", "layering", "adapter boundary", "package"),
    "release": ("release", "version", "changelog", "beta"),
    "api-reference": ("api reference", "signature", "returns:", "parameters:"),
    "roadmap": ("roadmap", "planned", "future work", "not yet implemented"),
}

STOPWORDS = set("""a an the and or but if then than that this these those of in on at to
for with from by as is are was were be been being it its it's not no do does did
you your we our they their he she them can may might should would could will
""".split())


def _shingles(text: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]
    return {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}


def build(root: Path, cfg: dict) -> dict:
    nodes = common.read_raw(root, "nodes")
    registered = set(nodes["nodes"])
    known_absent = set(cfg.get("known_absent_symbols", []))
    counts = nodes["counts"]

    provenance_docs = set(cfg.get("provenance_docs", []))
    provenance_prefixes = tuple(cfg.get("provenance_doc_prefixes", []))
    foreign_prefixes = tuple(cfg.get("foreign_path_prefixes", []))

    docs: dict[str, dict] = {}
    dead_all: set[str] = set()
    for rel in common.tracked_files(root):
        if common.categorize(rel) != "DOC" or not rel.endswith(".md"):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        at_revision = set(GIT_SHOW_RE.findall(text))
        cited = {m for m in PATH_RE.findall(text)}
        dead = {m for m in cited
                if m not in at_revision
                and not m.startswith(foreign_prefixes)
                and not (root / m).is_file()}
        dead_all |= dead

        symbols = {s for s in SYMBOL_RE.findall(text)}
        unregistered = sorted(s for s in symbols
                              if s not in registered and s not in known_absent)

        claims = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            for number, tier in COUNT_RE.findall(line):
                key = tier.lower()
                if key in counts and int(number) != counts[key]:
                    claims.append({"line": line_no, "claim": f"{number} {tier}",
                                   "actual": counts[key]})

        lower = text.lower()
        topics = sorted(
            (name for name, keys in TOPICS.items()
             if sum(lower.count(k) for k in keys) > 0),
            key=lambda n: -sum(lower.count(k) for k in TOPICS[n]),
        )

        docs[rel] = {
            "path": rel,
            "words": len(text.split()),
            "topics": topics,
            "provenance": rel in provenance_docs or rel.startswith(provenance_prefixes),
            "dead_paths": sorted(dead),
            "unregistered_symbols": unregistered,
            "count_claims": claims,
            "_shingles": None,
        }

    ignored = common.gitignored(root, dead_all)
    for info in docs.values():
        info["local_only_paths"] = sorted(p for p in info["dead_paths"] if p in ignored)
        info["dead_paths"] = sorted(p for p in info["dead_paths"] if p not in ignored)

    # containment overlap: the share of the SMALLER doc that also appears in
    # the larger. The contained side is the merge candidate.
    shingles = {}
    for rel in docs:
        try:
            shingles[rel] = _shingles((root / rel).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            shingles[rel] = set()
    overlaps = []
    keys = sorted(docs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = shingles[a], shingles[b]
            if len(sa) < 20 or len(sb) < 20:
                continue
            small, large = (a, b) if len(sa) <= len(sb) else (b, a)
            share = len(sa & sb) / len(shingles[small])
            if share >= 0.25:
                overlaps.append({"a": a, "b": b, "containment": round(share, 3),
                                 "contained": small})

    canonical: dict[str, dict] = {}
    for topic in TOPICS:
        holders = [r for r, d in docs.items() if topic in d["topics"]]
        if not holders:
            canonical[topic] = {"canonical": None, "duplicating": []}
            continue
        holders.sort(key=lambda r: (docs[r]["topics"].index(topic), -docs[r]["words"]))
        canonical[topic] = {"canonical": holders[0], "duplicating": holders[1:]}

    for info in docs.values():
        info.pop("_shingles", None)

    live = [d for d in docs.values() if not d["provenance"]]
    return {
        "docs": docs,
        "canonical_by_topic": canonical,
        "overlaps": sorted(overlaps, key=lambda d: -d["containment"]),
        "missing_topics": sorted(t for t, v in canonical.items() if not v["canonical"]),
        "counts": {
            "total": len(docs),
            "live_with_dead_paths": sum(1 for d in live if d["dead_paths"]),
            "live_with_bad_counts": sum(1 for d in live if d["count_claims"]),
            "live_with_unregistered_symbols":
                sum(1 for d in live if d["unregistered_symbols"]),
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
    common.write_raw(root, "docs", payload)
    c = payload["counts"]
    print(f"docs: {c['total']} markdown; live docs with dead paths "
          f"{c['live_with_dead_paths']}, with wrong node counts "
          f"{c['live_with_bad_counts']}, naming unregistered symbols "
          f"{c['live_with_unregistered_symbols']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
