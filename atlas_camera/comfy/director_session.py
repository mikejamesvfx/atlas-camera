"""Launching Atlas Director, and remembering what it sent back.

This module is the security boundary of the ComfyUI-to-Director seam. It is one
file so it can be read in one sitting: a route that spawns a process sits on an
unauthenticated localhost port that any page in the browser can reach, and
CLAUDE.md rule 6 bans exactly this channel one process over. The rules, all
enforced here:

  * the executable comes from configuration, never from the request
  * the request carries no argv -- the command line is composed server-side
  * every path this module touches (the session package, a delivered take
    directory) must resolve inside a configured root, never a root taken
    from the request -- a request may only supply a *relative* subpath
    under that root, and is refused (not repaired) if it tries to leave it
  * the session id, and each part of a slate, is allowlisted before it
    touches a path, and refused, not sanitised, when it does not match
  * a package must already exist on disk before Director is launched onto
    it -- this module writes the session manifest, never the .atlas itself;
    the graph writes that via AtlasExportScenePackage first
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from atlas_camera.comfy.nodes_export import _project_routed_dir

#: Mirrors take_ops.py's SLATE_PART. Kept identical on purpose: a session id
#: (and each '/'-separated part of a slate) becomes part of a path in one
#: repo and part of a slate in the other, and two allowlists that drift are
#: one allowlist that does not work.
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")

#: session_id -> {"session_id", "package", "timebase", "slate", "take_dir"}.
#: In-memory: a ComfyUI restart drops it, and the node treats an unknown
#: session as "re-push, or read this slate directly" rather than as an error.
SESSIONS: dict[str, dict] = {}

#: Small cap so an unauthenticated caller spamming /atlas/director/launch
#: cannot grow this dict without bound. Oldest session evicted past the cap.
_SESSION_LIMIT = 50


def validate_session_id(value) -> str:
    text = str(value or "")
    if not SESSION_ID.match(text):
        raise ValueError(
            f"session id is not usable as a path component: {text!r}. "
            "Allowed: letters, digits, underscore, hyphen."
        )
    return text


def _validate_slate(value) -> str:
    """A slate is `<scene>/<shot>/<setup>_takeNN` -- validate each part.

    Refused, not sanitised, same posture as a session id: a slate that fails
    the allowlist tells a downstream reader (Task 6) nothing safe to open.
    """
    text = str(value or "")
    parts = text.split("/")
    if not parts or any(not SESSION_ID.match(part) for part in parts):
        raise ValueError(
            f"slate is not usable as a path: {text!r}. Each '/'-separated "
            "part must match letters, digits, underscore, hyphen."
        )
    return text


def director_root() -> Path:
    """The root every session package and delivered take must resolve inside.

    Configuration only, exactly like the executable: `ATLAS_DIRECTOR_ROOT`
    if set, else ComfyUI's own output directory when this runs inside a
    ComfyUI process. Never taken from the request -- a request-controlled
    root makes any containment check downstream decorative.
    """
    configured = os.environ.get("ATLAS_DIRECTOR_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    try:
        import folder_paths  # type: ignore[import]  # ComfyUI-provided

        return Path(folder_paths.get_output_directory()).resolve()
    except Exception as exc:  # noqa: BLE001 - any failure means "unconfigured"
        raise RuntimeError(
            "no Atlas Director root configured. Set ATLAS_DIRECTOR_ROOT, or "
            "run inside ComfyUI where folder_paths.get_output_directory() is "
            "available. It is deliberately not taken from the request."
        ) from exc


def _validate_relative_subdir(value) -> Path:
    """A request-supplied `output_dir`, usable only as a relative subpath.

    Refused -- never repaired -- if absolute or if any component is `..`:
    an attacker gets no path arithmetic to work with, only a name under the
    configured root.
    """
    text = str(value or "").strip()
    candidate = Path(text) if text else Path()
    if candidate.is_absolute():
        raise ValueError(
            f"output_dir must be relative to the configured root, not absolute: {text!r}"
        )
    if ".." in candidate.parts:
        raise ValueError(f"output_dir may not contain '..': {text!r}")
    return candidate


def session_package_path(project, output_dir, session_id: str) -> Path:
    """Where a session's package goes: <configured root>/[output_dir/]scenes/<id>.atlas.

    `output_dir`, when given, is a relative subpath under the configured
    root (see `_validate_relative_subdir`) -- it is never itself the root.
    """

    session_id = validate_session_id(session_id)
    root = director_root()

    if project is not None:
        # Reachable only from an in-process caller with a real ATLAS_PROJECT
        # object -- the HTTP route refuses a non-null `project` outright
        # before it ever calls launch_session, because an HTTP caller has no
        # such object to legitimately supply (see __init__.py).
        base = Path(_project_routed_dir(project, str(root), "scenes")).resolve()
    else:
        subdir = _validate_relative_subdir(output_dir)
        base = (root / subdir / "scenes").resolve()

    package = (base / f"{session_id}.atlas").resolve()
    # is_relative_to, not a string prefix check: a sibling directory whose
    # name merely starts with the root's name (root ".../scenes", target
    # ".../scenes-evil") must not pass, and this is exact-boundary safe.
    if not package.is_relative_to(root):
        raise ValueError(f"session path escapes the configured root: {session_id!r}")
    return package


def director_executable() -> str:
    """The Director binary, from configuration only."""

    configured = os.environ.get("ATLAS_DIRECTOR_BIN", "").strip()
    if not configured:
        raise RuntimeError(
            "no Atlas Director executable configured. Set ATLAS_DIRECTOR_BIN to "
            "the Director binary. It is deliberately not taken from the request."
        )
    return configured


def _default_spawn(argv: list[str]) -> None:
    subprocess.Popen(argv)  # noqa: S603 - argv is composed here, never supplied


def _remember(session_id: str, session: dict) -> None:
    SESSIONS[session_id] = session
    while len(SESSIONS) > _SESSION_LIMIT:
        oldest = next(iter(SESSIONS))
        if oldest == session_id:
            break
        del SESSIONS[oldest]


def launch_session(body: dict, *, spawn=_default_spawn) -> dict:
    """Open Director on a session package the graph has already exported.

    `body` supplies a session id, an optional relative `output_dir` and the
    timebase. It supplies nothing else that reaches a command line --
    `executable` and `argv` keys, if present, are ignored rather than
    honoured. The `.atlas` package is never written here: the graph writes
    it via `AtlasExportScenePackage` first, and this function only verifies
    it exists before launching Director onto it -- verify, don't create.
    """

    session_id = validate_session_id(body.get("session_id"))
    executable = director_executable()
    package = session_package_path(
        body.get("project"), body.get("output_dir"), session_id
    )
    if not package.exists():
        raise ValueError(
            f"no session package for {session_id!r}; export it from the graph "
            "first (AtlasExportScenePackage) before launching Director"
        )

    timebase = {
        "width": int(body["width"]),
        "height": int(body["height"]),
        "frames": int(body["frames"]),
        "fps": int(body["fps"]),
    }
    # Recorded now, while the package is known to exist and to be what
    # Director is about to open on. Task 6 (AtlasDirectorTake) recomputes
    # this digest at read time and refuses the take if it no longer
    # matches -- "the package changed since Director opened it".
    package_digest = hashlib.sha256(package.read_bytes()).hexdigest()
    session = {
        "session_id": session_id,
        "package": str(package),
        "package_digest": package_digest,
        "timebase": timebase,
        "slate": None,
        "take_dir": None,
    }
    package.with_suffix(".session.json").write_text(
        json.dumps(session, indent=2), encoding="utf-8"
    )

    _remember(session_id, session)
    spawn([executable, "--director-session", str(package)])
    return session


def record_delivery(session_id: str, slate: str, take_dir: str) -> dict:
    """Remember the take a director pushed.

    Idempotent on (session_id, slate) so a retry after a failed push is safe,
    and last-write-wins on a different slate, because the node's promise is
    that what you pushed is what renders. `session_id` lookup happens before
    validation so an unknown session still refuses with KeyError, matching
    the route's 404; `slate` and `take_dir` are validated so a caller who
    merely guesses a live session id cannot plant an arbitrary path for
    whatever later reads `SESSIONS` (Task 6).
    """

    session = SESSIONS[session_id]
    slate = _validate_slate(slate)
    take_dir = _validate_take_dir(take_dir)
    session["slate"] = slate
    session["take_dir"] = take_dir
    return dict(session)


def _validate_take_dir(value) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("take_dir is required")
    root = director_root()
    resolved = Path(text).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"take_dir must resolve under the configured root: {text!r}")
    return text
