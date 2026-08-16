"""Every repo path and every node count a tracked doc states must be true.

`test_path_references.py` is the Python-string half of this net; it reads
string literals under `atlas_camera/` and nothing else. Markdown is the other
half and nothing read it, which is how the 2026-08-16 docs audit found four
live docs asserting things that had not been true for a release:
`docs/ECOSYSTEM_GUIDE.md` and `docs/MCP_SERVER.md` still headlined "68 standard
+ 4 experimental" against a 102 + 10 registry, and `docs/AGENT_HANDOFF.md` plus
`CLAUDE.md` pointed at `blender/` when the package moved to
`atlas_camera/blender/`. All four are the same defect class the audit skill
caught by hand — and that skill is gone, so the check has to live here.

**Provenance is not drift.** A dated entry that names a file the repo later
removed, or a node count that was correct when it was written, is the record
doing its job — CLAUDE.md says so of `docs/DESIGN_RULES.md`, and `CHANGELOG.md`
and the dated `docs/superpowers/` plans and specs work the same way. Those docs
are exempt from both assertions below. Everything else is a live claim and is
held to it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from atlas_camera.comfy import node_registry as _registry

REPO = Path(__file__).resolve().parents[1]

#: Docs that record what WAS true on a date. They deliberately name removed
#: files and superseded counts; "fixing" them destroys the provenance the
#: design rules are cited from. Keep this list short — a doc lands here only
#: because it is a dated record, never because it is inconvenient to update.
PROVENANCE_DOCS = {
    "CHANGELOG.md",
    "docs/DESIGN_RULES.md",
}
PROVENANCE_PREFIXES = ("docs/superpowers/",)

#: Same extension set as the package-wide check in `test_path_references.py`:
#: the families a doc actually promises. Images and source files a reader is
#: told to supply themselves (NODE_CATALOG's `examples/images/*.jpg`) are out
#: of scope by construction rather than by exemption.
PATH_RE = re.compile(
    r"(?<![\w/])((?:examples|docs|tests|tools|atlas_camera|blender|ui|research"
    r"|reference_data)/[\w./-]+\.(?:json|md|py))"
)

#: `git show <sha>:path` cites a file AT a revision. The path is expected to be
#: absent from the working tree — that is the whole point of the citation.
GIT_SHOW_RE = re.compile(r"git\s+show\s+[0-9a-f]{7,40}:([\w./-]+)")

#: Tier names as docs write them -> the registry attribute holding that tier.
TIER_MAPPINGS = {
    "experimental": "EXPERIMENTAL_NODE_CLASS_MAPPINGS",
    "legacy": "LEGACY_NODE_CLASS_MAPPINGS",
    "ios": "IOS_NODE_CLASS_MAPPINGS",
}

#: "102 standard", "10 experimental", "2 iOS", "116 registered".
COUNT_RE = re.compile(
    r"(\d+)\s+(standard|experimental|legacy|iOS|registered)\b"
)


def _tracked_markdown() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=str(REPO), capture_output=True, text=True, check=True,
    ).stdout.split()
    return sorted(out)


def _is_provenance(rel: str) -> bool:
    return rel in PROVENANCE_DOCS or rel.startswith(PROVENANCE_PREFIXES)


def _gitignored(paths) -> set[str]:
    """Ask git, NUL-delimited.

    Text mode translates the separator to CRLF on Windows and git then reads
    the \\r as part of the path, reporting every entry as not-ignored — the
    exact bug this repo's audit helper hit. Carries the load for every
    local-only tree at once (`docs/dev/`, `docs/artifacts/`, `atlas_debug/`,
    `examples/images/`) instead of a prefix list that drifts from .gitignore.
    """
    paths = sorted(paths)
    if not paths:
        return set()
    payload = b"\0".join(p.encode("utf-8") for p in paths) + b"\0"
    try:
        cp = subprocess.run(["git", "check-ignore", "-z", "--stdin"],
                            cwd=str(REPO), input=payload, capture_output=True)
    except (OSError, FileNotFoundError):  # pragma: no cover - git present in CI
        return set()
    return {p for p in cp.stdout.decode("utf-8", "replace").split("\0") if p}


def _registry_counts() -> dict[str, int]:
    """Tier sizes, independent of which env gates happen to be set."""
    gated: set[str] = set()
    counts: dict[str, int] = {}
    for tier, attr in TIER_MAPPINGS.items():
        keys = set(getattr(_registry, attr))
        counts[tier] = len(keys)
        gated |= keys
    # NODE_CLASS_MAPPINGS holds the standard tier plus whatever gates are on.
    counts["standard"] = len(set(_registry.NODE_CLASS_MAPPINGS) - gated)
    counts["registered"] = counts["standard"] + len(gated)
    return counts


def test_the_scan_has_docs_to_read():
    """Guards the test itself: a rename must not turn this into a no-op."""
    docs = [d for d in _tracked_markdown() if not _is_provenance(d)]
    assert len(docs) > 20, f"only {len(docs)} live docs found — scan is broken"


def test_no_live_doc_names_a_repo_path_that_is_gone():
    found: dict[str, list[str]] = {}
    for rel in _tracked_markdown():
        if _is_provenance(rel):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        cited_at_a_revision = set(GIT_SHOW_RE.findall(text))
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in PATH_RE.findall(line):
                if match in cited_at_a_revision or (REPO / match).is_file():
                    continue
                found.setdefault(match, []).append(f"{rel}:{line_no}")

    for ignored in _gitignored(found):
        found.pop(ignored, None)

    assert not found, (
        "live docs name repo paths that do not exist:\n  "
        + "\n  ".join(f"{p}  <- {', '.join(sites)}"
                      for p, sites in sorted(found.items()))
        + "\nEither restore the file, correct the path, or — if the mention is "
          "a historical record — move it into a dated docs/superpowers/ entry."
    )


def test_no_live_doc_miscounts_the_node_registry():
    actual = _registry_counts()
    wrong: list[str] = []
    for rel in _tracked_markdown():
        if _is_provenance(rel):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for number, tier in COUNT_RE.findall(line):
                key = tier.lower()
                if key not in actual or int(number) == actual[key]:
                    continue
                wrong.append(
                    f"{rel}:{line_no} claims {number} {tier}, "
                    f"registry has {actual[key]}"
                )
    assert not wrong, (
        "live docs state node counts the registry disagrees with:\n  "
        + "\n  ".join(wrong)
        + f"\nCurrent registry: {actual}."
    )
