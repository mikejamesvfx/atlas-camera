"""Shared helpers for the /atlas-v1-audit phase scripts.

Every phase writes ONE json into `.v1-audit/raw/` and nothing else. The
manifest phase is the only reader of all of them, and the only writer of the
human-facing markdown. Keeping the phases write-once makes the run resumable
and makes a single phase debuggable in isolation.

Nothing in here may modify a project file. The audit's first contract is that
a read-only run stays read-only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

OUT_DIRNAME = ".v1-audit"
RAW_DIRNAME = "raw"

# --- repository -------------------------------------------------------------


def repo_root(start: Path | None = None) -> Path:
    """The git worktree root. Fails loudly rather than auditing a random cwd."""
    start = start or Path.cwd()
    cp = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(start), capture_output=True, text=True,
    )
    if cp.returncode != 0:
        raise SystemExit(
            f"not a git repository: {start}\n"
            "The audit uses git history as evidence and refuses to run without it."
        )
    return Path(cp.stdout.strip()).resolve()


def git(root: Path, *args: str, check: bool = True) -> str:
    cp = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {cp.stderr.strip()}")
    return cp.stdout


def git_state(root: Path) -> dict:
    """Branch, SHA and dirtiness — recorded, never required to be clean."""
    dirty = git(root, "status", "--porcelain").strip()
    return {
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "sha": git(root, "rev-parse", "HEAD").strip(),
        "dirty": bool(dirty),
        "dirty_files": len(dirty.splitlines()) if dirty else 0,
    }


def tracked_files(root: Path) -> list[str]:
    out = git(root, "ls-files", "-z")
    return sorted(p for p in out.split("\0") if p)


def gitignored(root: Path, paths) -> set[str]:
    """Ask git which paths it ignores, NUL-delimited.

    Text mode translates the separator to CRLF on Windows and git then reads
    the \\r as part of the path, reporting every entry as not-ignored. That is
    a real bug this audit hit; keep the bytes.
    """
    paths = sorted(paths)
    if not paths:
        return set()
    payload = b"\0".join(p.encode("utf-8") for p in paths) + b"\0"
    cp = subprocess.run(["git", "check-ignore", "-z", "--stdin"],
                        cwd=str(root), input=payload, capture_output=True)
    return {p for p in cp.stdout.decode("utf-8", "replace").split("\0") if p}


# --- config -----------------------------------------------------------------

DEFAULT_CONFIG = {
    "exclude_dirs": [
        ".git", "node_modules", "__pycache__", ".pytest_cache", ".pytest_tmp",
        ".ruff_cache", ".mypy_cache", "build", "dist", "coverage",
        ".v1-audit", "graphify-out", ".venv", "venv",
    ],
    "generated_globs": ["**/atlas-three.bundle.js", "**/*.min.js"],
    "foreign_path_prefixes": [],
    "local_only_prefixes": [],
    "provenance_docs": [],
    "provenance_doc_prefixes": [],
    "known_absent_symbols": [],
    "entry_point_dirs": ["tools", "scripts"],
    "dynamic_loading_markers": [
        "NODE_CLASS_MAPPINGS", "importlib", "pkgutil", "__import__",
        "getattr(", "globals()[", "entry_points",
    ],
    "protected_categories": [
        "TEST", "DCC", "MODEL_ADAPTER", "SETUP", "CI", "CONFIG",
    ],
}


def load_config(skill_dir: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    path = skill_dir / "config.json"
    if path.is_file():
        user = json.loads(path.read_text(encoding="utf-8"))
        for key, value in user.items():
            cfg[key] = value
    return cfg


# --- output -----------------------------------------------------------------


def out_dir(root: Path) -> Path:
    d = root / OUT_DIRNAME
    d.mkdir(exist_ok=True)
    (d / RAW_DIRNAME).mkdir(exist_ok=True)
    return d


def write_raw(root: Path, name: str, payload) -> Path:
    """Deterministic json: sorted keys, so two runs diff cleanly."""
    path = out_dir(root) / RAW_DIRNAME / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_raw(root: Path, name: str):
    path = out_dir(root) / RAW_DIRNAME / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"missing phase output: {path}\nRun run_audit.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_doc(root: Path, name: str, text: str) -> Path:
    path = out_dir(root) / name
    path.write_text(text, encoding="utf-8")
    return path


# --- classification ---------------------------------------------------------

#: Broad file families. Order matters — the first match wins, so the specific
#: patterns (a test, a workflow) precede the generic ones (any .py).
_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("CI", re.compile(r"^\.github/|^\.gitlab|^azure-pipelines|^\.circleci/")),
    ("TEST", re.compile(r"(^|/)tests?/|(^|/)test_[^/]+\.py$|_test\.(py|ts|js)$"
                        r"|\.test\.(ts|tsx|js|jsx)$|(^|/)fixtures?/")),
    ("COMFY_WORKFLOW", re.compile(r"^examples/.*\.json$|(^|/)workflows?/.*\.json$")),
    ("DOC", re.compile(r"\.(md|rst|txt)$|^docs/")),
    ("SETUP", re.compile(r"^(pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.txt"
                         r"|package\.json|INSTALL\.md|Dockerfile[^/]*)$"
                         r"|(^|/)docker/")),
    ("DCC", re.compile(r"(^|/)(exporters|importers|blender|maya|nuke|usd)(/|_)"
                       r"|_exporter\.py$|_importer\.py$")),
    ("MODEL_ADAPTER", re.compile(r"(^|/)inference/|(^|/)dynamic/")),
    ("REPORT", re.compile(r"^reports?/")),
    ("EXAMPLE", re.compile(r"^examples?/")),
    ("ASSET", re.compile(r"\.(png|jpg|jpeg|exr|dpx|tif|tiff|gif|svg|ico|pdf|obj|glb|usda?)$")),
    ("CONFIG", re.compile(r"\.(toml|ya?ml|ini|cfg|json)$|^\.[a-z]")),
    ("JS_TS", re.compile(r"\.(js|jsx|ts|tsx|mjs|cjs)$")),
    ("PYTHON", re.compile(r"\.py$")),
]


def categorize(rel: str) -> str:
    for name, pattern in _CATEGORY_RULES:
        if pattern.search(rel):
            return name
    return "UNKNOWN"


# --- confidence -------------------------------------------------------------

CONFIDENCE_ORDER = ["UNKNOWN", "LOW", "MEDIUM", "HIGH", "CERTAIN"]


def cap_confidence(level: str, ceiling: str) -> str:
    """Lower `level` to `ceiling` when it sits above it. Never raises."""
    if CONFIDENCE_ORDER.index(level) <= CONFIDENCE_ORDER.index(ceiling):
        return level
    return ceiling


def confidence_from_evidence(evidence: dict) -> str:
    """How sure are we that NOTHING reaches this file?

    Independent evidence sources agreeing is the only route to CERTAIN, and
    even then a caller may cap it. A single tool finding can never get here:
    `static_analysis` is one input among nine and is never sufficient alone.
    """
    checked = [
        "static_reference", "registered_reference", "workflow_reference",
        "test_reference", "doc_reference", "setup_reference", "ci_reference",
    ]
    present = [k for k in checked if evidence.get(k) is not None]
    reached = [k for k in checked if evidence.get(k)]
    if reached:
        return "HIGH"          # something reaches it — confidently KEEP
    if len(present) < len(checked):
        return "LOW"           # a dimension went unchecked; cannot conclude
    if evidence.get("superseded_by"):
        return "CERTAIN"
    if evidence.get("git_history_present") and evidence.get("static_analysis"):
        return "HIGH"
    return "MEDIUM"


# --- misc -------------------------------------------------------------------


def rel_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def is_excluded(rel: str, cfg: dict) -> bool:
    parts = rel.split("/")
    return any(part in cfg["exclude_dirs"] for part in parts)


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and Atlas node names carry emoji."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
