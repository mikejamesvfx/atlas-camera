"""Locate Blender, build its argv, run a recipe headless, read the result back.

Mirrors `inference/fixer_render_fix.py` deliberately: that module is Atlas's
existing answer to "shell out to a big external tool", and copying its shape
means the error taxonomy, the log tail and the tests all transfer.

WHY THIS IS NOT IN THE MCP SERVER. `mcp/server.py:5-6` states the server never
executes — no subprocess, no numpy, ComfyUI stays the engine. A ComfyUI node
calls this instead, and the agent drives that through the existing
`atlas_run_workflow`. The MCP server's property stays intact.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

BLENDER_PATH_ENV = "ATLAS_BLENDER_PATH"

#: Verified against 5.2.0 LTS. The recipe uses `remesh_voxel_*` and shrinkwrap
#: PROJECT/ON_SURFACE, all probed present there. 4.2 is the oldest release those
#: names are known-good on; below it a silent API mismatch is far worse than a
#: refusal, because a renamed attribute fails as "did nothing" not as an error.
MIN_BLENDER = (4, 2)

_VERSION_CACHE: dict[tuple[str, float], tuple[int, ...]] = {}


def _windows_candidates() -> list[Path]:
    roots = [Path(r"C:/Program Files/Blender Foundation"),
             Path(r"C:/Program Files (x86)/Blender Foundation")]
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            out.extend(p / "blender.exe" for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
    return out


def _candidates() -> list[Path]:
    if sys.platform.startswith("win"):
        return _windows_candidates()
    if sys.platform == "darwin":
        return [p / "Contents/MacOS/Blender"
                for p in Path("/Applications").glob("Blender*.app")]
    return [Path("/usr/bin/blender"), Path("/usr/local/bin/blender"),
            Path("/snap/bin/blender")] + \
           [p / "blender" for p in Path.home().glob("blender*")]


def _version_key(path: Path) -> tuple:
    """Sort key from a version-bearing directory name, newest last.

    Two installs side by side is normal (a released LTS plus a beta), and
    picking whichever the filesystem lists first would make the behaviour depend
    on directory order. Parse and take the newest.
    """
    import re
    nums = re.findall(r"(\d+)\.(\d+)", path.parent.name or "")
    return (tuple(int(n) for n in nums[-1]) if nums else (0, 0), str(path))


def resolve_blender_exe(blender_path: str = "") -> Path:
    """Locate Blender: explicit arg > env var > PATH > platform install dirs.

    Same precedence and same tone as `resolve_fixer_root` — the error names the
    widget, the env var, the download, AND every location actually probed, so a
    user is never left guessing where it looked.
    """
    explicit = (blender_path or "").strip() or os.environ.get(
        BLENDER_PATH_ENV, "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise RuntimeError(
            f"blender_path points at nothing executable: {p}\n"
            f"(from {'the widget' if blender_path.strip() else BLENDER_PATH_ENV})")

    which = shutil.which("blender")
    if which:
        return Path(which)

    found = sorted((p for p in _candidates() if p.is_file()), key=_version_key)
    if found:
        return found[-1]

    probed = ["PATH (\"blender\")"] + [str(p) for p in _candidates()]
    raise RuntimeError(
        "Blender not found. This EXPERIMENTAL node runs a geometry recipe in "
        "headless Blender, which Atlas does not bundle:\n"
        "    download Blender 4.2+ from https://www.blender.org/download/\n"
        f"then set the node's blender_path widget (or {BLENDER_PATH_ENV}) to the "
        "executable.\n"
        "Probed and found nothing at:\n  - " + "\n  - ".join(probed))


def blender_version(exe: Path) -> tuple[int, ...]:
    """(major, minor, patch) from `blender --version`, cached on path+mtime."""
    try:
        key = (str(exe), exe.stat().st_mtime)
    except OSError:
        key = (str(exe), 0.0)
    if key in _VERSION_CACHE:
        return _VERSION_CACHE[key]
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not run '{exe} --version': "
                           f"{type(exc).__name__}: {exc}") from exc
    import re
    m = re.search(r"Blender\s+(\d+)\.(\d+)(?:\.(\d+))?", out or "")
    if not m:
        raise RuntimeError(f"could not parse a version from: {out.strip()[:200]!r}")
    ver = tuple(int(g) for g in m.groups() if g is not None)
    _VERSION_CACHE[key] = ver
    return ver


def require_blender(blender_path: str = "") -> tuple[Path, tuple[int, ...]]:
    exe = resolve_blender_exe(blender_path)
    ver = blender_version(exe)
    if ver[:2] < MIN_BLENDER:
        raise RuntimeError(
            f"Blender {'.'.join(map(str, ver))} at {exe} is older than the "
            f"{'.'.join(map(str, MIN_BLENDER))} floor this recipe is verified "
            "against. A renamed modifier attribute fails silently as 'did "
            "nothing' rather than as an error, so this refuses instead.")
    return exe, ver


def recipes_dir() -> Path:
    return Path(__file__).resolve().parent / "recipes"


def build_blender_command(exe: Path, recipe: Path, exchange_dir: Path) -> list[str]:
    """argv for a headless recipe run.

    `--factory-startup` is not optional: without it a user's addons and
    preferences are an unversioned input to a supposedly deterministic
    construction.

    The bare `--` matters — Blender consumes everything before it, and omitting
    it is the classic silent failure where the recipe never sees its arguments.

    NOT included: `--noaudio`. It is not a valid flag in Blender 5.2 (verified
    live — Blender treats it as a filename and reports "Cannot read file").
    """
    return [str(exe), "--background", "--factory-startup",
            "--python", str(recipe), "--", "--exchange", str(exchange_dir)]


def run_recipe(recipe_name: str, exchange_dir: Path, *, blender_path: str = "",
               timeout_s: int = 300) -> dict[str, Any]:
    """Run `recipes/<recipe_name>` headless against `exchange_dir`.

    Every failure is raised with something quotable in it. The rule copied from
    `run_fixer_on_dir`: Blender is extremely verbose on stdout, so never surface
    the whole log — tail it.
    """
    exe, ver = require_blender(blender_path)
    recipe = recipes_dir() / recipe_name
    if not recipe.is_file():
        raise RuntimeError(f"recipe not found: {recipe} (packaging problem — "
                           "recipes/ ships as package data)")
    exchange_dir = Path(exchange_dir)
    exchange_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_blender_command(exe, recipe, exchange_dir)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=int(timeout_s))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Blender recipe {recipe_name} exceeded timeout_s={timeout_s}. "
            "The usual cause is voxel_size_m being too small — doubling it cuts "
            "the cost roughly 8x."
        ) from exc

    err_path = exchange_dir / "error.json"
    if err_path.is_file():
        # The recipe's OWN traceback and stage, which is what a reader needs —
        # not Blender's startup banner.
        try:
            err = json.loads(err_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            err = {}
        raise RuntimeError(
            f"Blender recipe {recipe_name} failed in stage "
            f"{err.get('stage', '?')}: {err.get('type', '?')}: "
            f"{err.get('message', '?')}\n{err.get('traceback', '')}")

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(
            f"Blender exited {proc.returncode} running {recipe_name} "
            f"(no error.json written). Last 40 lines:\n{tail}")

    report_path = exchange_dir / "report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            report = {}
    report.setdefault("blender_exe", str(exe))
    report.setdefault("blender_version", ".".join(map(str, ver)))
    return report
