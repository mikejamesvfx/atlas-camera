"""Phase 7 — third-party static analysis, as ONE evidence source among many.

Vulture, deptry, Ruff and Knip each answer a narrow question well and none of
them knows about ComfyUI registration, serialized workflows, or a DCC host
executing a script out-of-process. Their findings are recorded as
`STATIC_ANALYSIS_CANDIDATE` and can never on their own carry a file to
CERTAIN — `build_manifest.py` requires independent agreement.

Tools are never installed into the project environment. `--bootstrap-venv`
creates `.v1-audit/.tools-venv` (~50 MB, gitignored with the rest of the audit
output) so an audit never changes what the project itself depends on.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PY_TOOLS = ("vulture", "deptry", "ruff")
VENV_DIRNAME = ".tools-venv"
TIMEOUT_S = 300


def _venv_bin(root: Path) -> Path:
    venv = common.out_dir(root) / VENV_DIRNAME
    return venv / ("Scripts" if os.name == "nt" else "bin")


def bootstrap_venv(root: Path) -> dict:
    """Create the audit's own tool venv. Never touches the project env."""
    venv = common.out_dir(root) / VENV_DIRNAME
    if not venv.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = _venv_bin(root) / ("pip.exe" if os.name == "nt" else "pip")
    cp = subprocess.run([str(pip), "install", "--quiet", *PY_TOOLS],
                        capture_output=True, text=True)
    return {"ok": cp.returncode == 0, "stderr": cp.stderr.strip()[-2000:]}


def _resolve(root: Path, tool: str) -> str | None:
    local = _venv_bin(root) / (f"{tool}.exe" if os.name == "nt" else tool)
    if local.is_file():
        return str(local)
    return shutil.which(tool)


#: A launch that never happened. Distinct from "ran and found nothing".
RC_LAUNCH_FAILED = -1


def _launchable(name: str) -> list[str] | None:
    """argv prefix for a possibly-shimmed executable, or None if absent.

    On Windows `npx` is `npx.cmd`. `shutil.which` finds it — so an availability
    check passes — but `CreateProcess` cannot execute a `.cmd` directly, so the
    call raises `WinError 2` and the tool never runs. That combination is how
    knip reported AVAILABLE and zero findings for an entire session: found,
    never launched, and the failure swallowed into an empty string.
    """
    resolved = shutil.which(name)
    if not resolved:
        return None
    if os.name == "nt" and Path(resolved).suffix.lower() in (".cmd", ".bat"):
        return ["cmd", "/c", resolved]
    return [resolved]


def _run(cmd: list[str], root: Path) -> tuple[int, str, str]:
    try:
        cp = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                            timeout=TIMEOUT_S)
        return cp.returncode, cp.stdout, cp.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RC_LAUNCH_FAILED, "", f"{type(exc).__name__}: {exc}"


def _parse_json(text: str):
    """Parse a tool's JSON output. Returns ``(value, error_text)``.

    Tools print human warnings to stdout BEFORE their JSON — knip emits
    `ERROR: Error loading vite.config.ts` when a devDependency is absent, then
    the report. An earlier revision did `json.loads(out)` inside a bare
    `except json.JSONDecodeError: findings[tool] = []`, so that preamble turned
    a real report into ZERO findings and the audit published a clean JS/TS bill
    of health it had never actually read.

    An empty result must therefore mean "the tool found nothing", never "the
    tool could not be understood" — the two are reported separately, because a
    scanner that silently degrades is the exact defect this audit exists to
    find in other people's code.
    """
    stripped = text.strip()
    if not stripped:
        return [], None
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass
    # Retry from the first structural character, skipping any preamble.
    for i, ch in enumerate(stripped):
        if ch in "[{":
            try:
                return json.loads(stripped[i:]), stripped[:i].strip() or None
            except json.JSONDecodeError:
                break
    return None, stripped[:500]


#: `unused method 'run'`, `unused attribute 'CATEGORY'`, …
VULTURE_NAME_RE = re.compile(r"unused \w+ '([^']+)'")


def _apply_suppressions(findings: dict, cfg: dict) -> tuple[dict, dict]:
    """Drop the configured classes of false positive, counting what went.

    Returns ``(findings, suppressed)``. Nothing is dropped without being
    counted — the audit's own "no silent caps" rule applies to itself, and a
    suppression list nobody can see the size of is how a scanner quietly stops
    scanning.
    """
    sa = cfg.get("static_analysis", {})
    suppressed: dict[str, int] = {}

    ignore_names = set(sa.get("vulture", {}).get("ignore_names", []))
    if ignore_names and findings.get("vulture"):
        kept = []
        for row in findings["vulture"]:
            match = VULTURE_NAME_RE.search(row.get("message", ""))
            if match and match.group(1) in ignore_names:
                suppressed["vulture"] = suppressed.get("vulture", 0) + 1
                continue
            kept.append(row)
        findings["vulture"] = kept

    dep = sa.get("deptry", {})
    ignore_modules = set(dep.get("host_provided", [])) | set(
        dep.get("optional_extra", [])) | set(dep.get("research_or_sibling", []))
    ignore_codes = set(dep.get("ignore_codes", []))
    rows = findings.get("deptry")
    if (ignore_modules or ignore_codes) and isinstance(rows, list):
        kept = []
        for row in rows:
            module = row.get("module")
            code = (row.get("error") or {}).get("code")
            if module in ignore_modules or code in ignore_codes:
                suppressed["deptry"] = suppressed.get("deptry", 0) + 1
                continue
            kept.append(row)
        findings["deptry"] = kept

    return findings, suppressed


def build(root: Path, cfg: dict, package: str) -> dict:
    status: dict[str, str] = {}
    findings: dict[str, list] = {}
    #: Why a tool's result is partial or unreadable. Empty findings plus an
    #: empty note is the ONLY combination that means "clean".
    notes: dict[str, str] = {}

    status["git"] = "AVAILABLE" if shutil.which("git") else "NOT_AVAILABLE"
    status["graphify"] = "AVAILABLE" if shutil.which("graphify") else "NOT_AVAILABLE"

    # --- vulture: unused functions/classes/variables/imports ----------------
    exe = _resolve(root, "vulture")
    if not exe:
        status["vulture"] = "NOT_AVAILABLE"
    else:
        status["vulture"] = "AVAILABLE"
        rc, out, _ = _run([exe, package, "--min-confidence", "60"], root)
        rows = []
        for line in out.splitlines():
            # path:line: unused function 'name' (60% confidence)
            parts = line.split(":", 2)
            if len(parts) == 3:
                rows.append({"path": parts[0].replace("\\", "/"),
                             "line": parts[1], "message": parts[2].strip()})
        findings["vulture"] = rows

    # --- deptry: unused / missing / transitive dependencies ------------------
    exe = _resolve(root, "deptry")
    if not exe or not (root / "pyproject.toml").is_file():
        status["deptry"] = "NOT_AVAILABLE" if not exe else "SKIPPED"
    else:
        status["deptry"] = "AVAILABLE"
        report = common.out_dir(root) / "raw" / "_deptry.json"
        _run([exe, ".", "--json-output", str(report)], root)
        try:
            parsed, note = _parse_json(report.read_text(encoding="utf-8"))
        except OSError as exc:
            parsed, note = None, str(exc)
        if parsed is None:
            status["deptry"] = "PARSE_FAILED"
            notes["deptry"] = note
            findings["deptry"] = []
        else:
            findings["deptry"] = parsed
            if note:
                notes["deptry"] = note

    # --- ruff: unused imports and dead constructs ----------------------------
    exe = _resolve(root, "ruff")
    if not exe:
        status["ruff"] = "NOT_AVAILABLE"
    else:
        status["ruff"] = "AVAILABLE"
        rc, out, _ = _run(
            [exe, "check", package, "--select", "F401,F811,F841,ERA001",
             "--output-format", "json", "--no-cache"], root)
        parsed, note = _parse_json(out)
        if parsed is None:
            status["ruff"] = "PARSE_FAILED"
            notes["ruff"] = note
            findings["ruff"] = []
        else:
            findings["ruff"] = parsed
            if note:
                notes["ruff"] = note

    # --- knip: unused JS/TS files, exports, dependencies ---------------------
    ui = root / "ui"
    npx = _launchable("npx")
    if not (ui / "package.json").is_file():
        status["knip"] = "SKIPPED"
    elif npx is None:
        status["knip"] = "NOT_AVAILABLE"
    else:
        findings["knip"] = []
        rc, out, err = _run([*npx, "--yes", "knip", "--reporter", "json"], ui)
        parsed, note = _parse_json(out)
        if rc == RC_LAUNCH_FAILED:
            # The tool did not run AT ALL. Never a clean result.
            status["knip"] = "LAUNCH_FAILED"
            notes["knip"] = err.strip()[:500]
        elif parsed is None:
            # It ran but its output could not be read. Also never clean —
            # saying "0 findings" here is how this phase published a JS/TS bill
            # of health it had never actually parsed.
            status["knip"] = "PARSE_FAILED"
            notes["knip"] = note or err.strip()[:500]
        else:
            findings["knip"] = parsed
            # knip warns before its JSON when it cannot resolve a
            # devDependency, which makes the report PARTIAL rather than clean.
            status["knip"] = "AVAILABLE_PARTIAL" if note else "AVAILABLE"
            if note:
                notes["knip"] = note

    # --- configured suppressions --------------------------------------------
    # Applied AFTER the tools run, never by narrowing what they were asked to
    # look at, so the suppressed count is always knowable. Reported next to the
    # survivors: a suppression list that grows silently is indistinguishable
    # from a codebase that got cleaner.
    findings, suppressed = _apply_suppressions(findings, cfg)

    # Which project files any tool named at all. This is the ONLY thing the
    # manifest consumes from this phase — a boolean per file, never a verdict.
    flagged: set[str] = set()
    for row in findings.get("vulture", []):
        flagged.add(row["path"].lstrip("./"))
    for row in findings.get("ruff", []):
        name = row.get("filename")
        if name:
            try:
                flagged.add(Path(name).resolve().relative_to(root).as_posix())
            except ValueError:
                pass

    return {
        "status": status,
        "notes": notes,
        "suppressed": suppressed,
        "findings": findings,
        "flagged_paths": sorted(flagged),
        "install_hint": (
            "python .claude/skills/atlas-v1-audit/scripts/run_audit.py "
            "--bootstrap-venv   # installs vulture/deptry/ruff into "
            ".v1-audit/.tools-venv, never into the project env"
        ),
    }


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--package", default="atlas_camera")
    ap.add_argument("--bootstrap-venv", action="store_true")
    ap.add_argument("--skip", action="store_true",
                    help="write a stub result without running any tool "
                         "(used by run_audit.py --quick, so the manifest phase "
                         "still has its input)")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    cfg = common.load_config(Path(__file__).resolve().parents[1])
    if args.skip:
        common.write_raw(root, "tools", {
            "status": {t: "SKIPPED" for t in (*PY_TOOLS, "knip", "graphify", "git")},
            "findings": {},
            "flagged_paths": [],
            "install_hint": "run without --quick to use the static-analysis tools",
        })
        print("tools: SKIPPED (--quick)")
        return 0
    if args.bootstrap_venv:
        result = bootstrap_venv(root)
        print(f"tool venv: {'ready' if result['ok'] else 'FAILED'}")
        if not result["ok"]:
            common.eprint(result["stderr"])

    package = args.package if (root / args.package).is_dir() else "."
    payload = build(root, cfg, package)
    common.write_raw(root, "tools", payload)
    print("tools: " + ", ".join(f"{k}={v}" for k, v in sorted(payload["status"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
