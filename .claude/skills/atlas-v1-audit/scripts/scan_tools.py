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


def _run(cmd: list[str], root: Path) -> tuple[int, str, str]:
    try:
        cp = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                            timeout=TIMEOUT_S)
        return cp.returncode, cp.stdout, cp.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def build(root: Path, cfg: dict, package: str) -> dict:
    status: dict[str, str] = {}
    findings: dict[str, list] = {}

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
            findings["deptry"] = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            findings["deptry"] = []

    # --- ruff: unused imports and dead constructs ----------------------------
    exe = _resolve(root, "ruff")
    if not exe:
        status["ruff"] = "NOT_AVAILABLE"
    else:
        status["ruff"] = "AVAILABLE"
        rc, out, _ = _run(
            [exe, "check", package, "--select", "F401,F811,F841,ERA001",
             "--output-format", "json", "--no-cache"], root)
        try:
            findings["ruff"] = json.loads(out) if out.strip() else []
        except json.JSONDecodeError:
            findings["ruff"] = []

    # --- knip: unused JS/TS files, exports, dependencies ---------------------
    ui = root / "ui"
    if not (ui / "package.json").is_file():
        status["knip"] = "SKIPPED"
    elif not shutil.which("npx"):
        status["knip"] = "NOT_AVAILABLE"
    else:
        status["knip"] = "AVAILABLE"
        rc, out, _ = _run(["npx", "--yes", "knip", "--reporter", "json"], ui)
        try:
            findings["knip"] = json.loads(out) if out.strip() else []
        except json.JSONDecodeError:
            findings["knip"] = []

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
