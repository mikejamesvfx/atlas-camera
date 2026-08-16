"""Every repo path an MCP string promises to an agent must actually resolve.

The MCP server's resources are agent-facing playbooks: an assistant reads
`atlas://path-repair` and follows step 1. When step 1 named
`examples/atlas_path_guided_hole_repair_workflow.json`, which the 0.8.1 trim
had removed, the agent's documented entry point was a dead file — and nothing
caught it, because a path inside a Python string is invisible to both the test
suite and to Markdown link checkers.

This is the cheap check that closes that gap: scan the mcp package's string
literals for repo-relative paths and assert each one exists.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MCP_DIR = REPO / "atlas_camera" / "mcp"

#: Referenced paths we require to exist. Deliberately narrow: these are the
#: two families the server actually promises (a shipped workflow, a tracked
#: doc), not every string that happens to contain a slash.
PATH_RE = re.compile(r"\b(examples/[\w./-]+\.json|docs/[\w./-]+\.md)\b")

#: `docs/dev/` and `docs/artifacts/` are gitignored by design (see CLAUDE.md):
#: they do not survive a clone and are excluded from the published Registry
#: archive. Code may still build a path into them at runtime — `atlas://gates`
#: does, guarded by `is_file()` with a prose fallback — so a reference is not a
#: broken promise. The residual risk this exemption accepts: an UNGUARDED
#: mention of a local-only doc still passes. Keep such mentions out of
#: agent-facing resource text; cite a tracked doc instead.
LOCAL_ONLY_PREFIXES = ("docs/dev/", "docs/artifacts/")


def _referenced_paths(py_file: Path) -> set[tuple[str, int]]:
    """(path, lineno) for every repo-relative path inside a string literal."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in PATH_RE.findall(node.value):
                found.add((match, node.lineno))
    # module/function docstrings are Constants too, so they are already covered
    return found


def _mcp_sources() -> list[Path]:
    return sorted(p for p in MCP_DIR.glob("*.py") if p.name != "__init__.py")


def test_the_mcp_package_has_sources_to_scan():
    """Guards the test itself: a rename must not turn this into a no-op."""
    assert _mcp_sources(), f"no .py sources found under {MCP_DIR}"


@pytest.mark.parametrize("source", _mcp_sources(), ids=lambda p: p.name)
def test_every_promised_repo_path_resolves(source):
    missing = []
    for rel, lineno in sorted(_referenced_paths(source)):
        if rel.startswith(LOCAL_ONLY_PREFIXES):
            continue
        if not (REPO / rel).is_file():
            missing.append(f"{source.name}:{lineno} -> {rel}")
    assert not missing, (
        "MCP strings name repo paths that do not exist. An agent following "
        "these instructions hits a dead file:\n  " + "\n  ".join(missing)
        + "\nEither restore the file or rewrite the text to stop naming it."
    )


def test_agent_facing_resources_do_not_cite_local_only_docs():
    """A clone-absent doc is fine in a runtime lookup, not in a playbook.

    `docs/dev/` is the maintainer's local tree. Naming it in a resource body
    hands the agent a path it can never open.
    """
    server = MCP_DIR / "server.py"
    tree = ast.parse(server.read_text(encoding="utf-8"))
    offenders = []

    # The module docstring is the first thing a reader (human or agent) sees,
    # so it is held to the same standard as the resource bodies.
    module_doc = ast.get_docstring(tree) or ""
    for prefix in LOCAL_ONLY_PREFIXES:
        if prefix in module_doc and "gitignored" not in module_doc:
            offenders.append(
                f"the {server.name} module docstring cites {prefix} without "
                "saying it is absent from a clone"
            )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)):
            continue
        target = node.targets[0]
        name = getattr(target, "id", "")
        # the resource bodies are module-level _UPPER_SNAKE string constants
        if not (name.startswith("_") and name.isupper()):
            continue
        text = node.value.value
        if not isinstance(text, str):
            continue
        for prefix in LOCAL_ONLY_PREFIXES:
            if prefix in text:
                offenders.append(f"{name} (line {node.lineno}) cites {prefix}")
    assert not offenders, (
        "Agent-facing resource text cites a gitignored local-only doc:\n  "
        + "\n  ".join(offenders)
    )
