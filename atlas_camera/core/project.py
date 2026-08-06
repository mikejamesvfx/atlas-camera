"""Atlas delivery project: structured, colour-managed output routing.

One place a user names the project and shot they're working, so every export
lands in a predictable tree under it, developed to the colour the project is set
to, instead of scattering through the ComfyUI install.

This is the *delivery* project (node graph -> disk). It is deliberately separate
from ``atlas_camera.ui.project.AtlasUiProject``, which is the standalone
workbench's per-solve session directory. Different concern, different tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

PROJECT_MANIFEST = "atlas_project.json"

# --- Colour modes -----------------------------------------------------------
# The one explicit choice that separates the two audiences. Default is the
# simple lane; the managed lane is opt-in, so a photographer or designer never
# meets OCIO by accident. RAW is the input that forces the choice: the file
# can't tell you whether it's a VFX plate or a print job, so the user states it.
MODE_STANDARD = "standard"   # sRGB, 8-bit deliverables, no colour management
MODE_VFX = "vfx"             # ACEScg, float / EXR, OCIO-managed (ACES)
COLOUR_MODES = (MODE_STANDARD, MODE_VFX)

# DCC-aware subfolders under a shot. Each exporter writes into its own, with
# paths the DCC can resolve (a Nuke script beside the plates it reads, etc.).
SHOT_SUBDIRS = ("plates", "solves", "nuke", "maya", "usd", "blender", "geo", "review")

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _atlas_version() -> str:
    try:
        from atlas_camera import __version__  # type: ignore
        return str(__version__)
    except Exception:
        return "0.0.0"


def normalise_mode(mode: str | None) -> str:
    """Fold any input to a known colour mode. Unknown or empty -> standard.

    Lenient on purpose: the node feeds a controlled dropdown, but a hand-built
    graph or an old workflow shouldn't crash, it should fall to the safe lane.
    """
    m = (mode or "").strip().lower()
    if m in ("vfx", "aces", "acescg", "float", "managed"):
        return MODE_VFX
    return MODE_STANDARD


def sanitise_name(name: str | None, *, fallback: str) -> str:
    """Make a project/shot name safe as a single path segment.

    Strips, replaces illegal characters and separators with '_', collapses runs,
    trims stray dots/spaces, and falls back when nothing usable is left.
    """
    raw = (name or "").strip()
    cleaned = _ILLEGAL.sub("_", raw)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_. ")
    return cleaned or fallback


@dataclass(frozen=True)
class ColourPolicy:
    """What 'colour' means for a project, derived purely from its mode."""
    mode: str
    managed: bool
    working_space: str
    image_ext: str           # default deliverable container for stills
    ocio_config: str | None  # None in standard; a config id/path in vfx

    @classmethod
    def from_mode(cls, mode: str | None) -> "ColourPolicy":
        if normalise_mode(mode) == MODE_VFX:
            return cls(MODE_VFX, True, "ACEScg", "exr", "built-in:aces")
        return cls(MODE_STANDARD, False, "sRGB", "png", None)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "managed": self.managed,
            "working_space": self.working_space,
            "image_ext": self.image_ext,
            "ocio_config": self.ocio_config,
        }


def resolve_root(root: str | None, *, default_root: Path | None = None) -> Path:
    """Resolve the projects root: explicit arg, then ``$ATLAS_PROJECT_ROOT``,
    then a caller-supplied default (the node passes ComfyUI's output dir), then a
    clear cwd fallback. Never the empty string, never the Comfy install root,
    never a crash.
    """
    for candidate in (root, os.environ.get("ATLAS_PROJECT_ROOT")):
        c = (candidate or "").strip()
        if c:
            return Path(c).expanduser()
    if default_root is not None:
        return Path(default_root)
    return Path.cwd() / "AtlasProjects"


@dataclass(frozen=True)
class AtlasProject:
    """A resolved delivery project: where outputs go and how they're coloured."""
    root: Path
    project: str
    shot: str
    colour: ColourPolicy

    @property
    def project_dir(self) -> Path:
        return self.root / self.project

    @property
    def shot_dir(self) -> Path:
        return self.project_dir / self.shot

    @property
    def manifest_path(self) -> Path:
        return self.project_dir / PROJECT_MANIFEST

    def subdir(self, name: str, *, create: bool = False) -> Path:
        """Path to a DCC subfolder under the shot. Unknown names raise, to catch
        an exporter typo before it scatters files somewhere odd."""
        if name not in SHOT_SUBDIRS:
            raise ValueError(
                f"unknown shot subfolder {name!r}; expected one of {SHOT_SUBDIRS}")
        path = self.shot_dir / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_tree(self) -> Path:
        """Create the shot dir and every DCC subfolder. Returns the shot dir."""
        for name in SHOT_SUBDIRS:
            (self.shot_dir / name).mkdir(parents=True, exist_ok=True)
        return self.shot_dir

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "project": self.project,
            "shot": self.shot,
            "project_dir": str(self.project_dir),
            "shot_dir": str(self.shot_dir),
            "colour": self.colour.to_dict(),
        }

    def write_manifest(self) -> Path:
        """Write / update ``atlas_project.json`` at the project level: the colour
        policy, the shots seen, and stamps. Makes a project self-describing and
        reproducible. First-write ``created`` is preserved on later updates."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data: dict = {}
        if self.manifest_path.is_file():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                data = {}
        shots = set(data.get("shots") or [])
        shots.add(self.shot)
        payload = {
            "atlas_version": data.get("atlas_version") or _atlas_version(),
            "project": self.project,
            "colour": self.colour.to_dict(),
            "shots": sorted(shots),
            "created": data.get("created") or now,
            "updated": now,
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.manifest_path


def build_project(
    root: str | None,
    project: str | None,
    shot: str | None,
    mode: str | None,
    *,
    default_root: Path | None = None,
) -> AtlasProject:
    """Sanitise inputs, resolve the root, and derive the colour policy into a
    ready-to-use ``AtlasProject``. Pure: does not touch disk (call
    ``ensure_tree`` / ``write_manifest`` for that)."""
    return AtlasProject(
        root=resolve_root(root, default_root=default_root),
        project=sanitise_name(project, fallback="untitled_project"),
        shot=sanitise_name(shot, fallback="untitled_shot"),
        colour=ColourPolicy.from_mode(mode),
    )
