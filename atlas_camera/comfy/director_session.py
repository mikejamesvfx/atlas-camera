"""Launching Atlas Director, and remembering what it sent back.

This module is the security boundary of the ComfyUI-to-Director seam. It is one
file so it can be read in one sitting: a route that spawns a process sits on an
unauthenticated localhost port that any page in the browser can reach, and
CLAUDE.md rule 6 bans exactly this channel one process over. The rules, all
enforced here:

  * the executable comes from configuration, never from the request
  * the request carries no argv -- the command line is composed server-side
  * the session path must resolve inside the configured output root
  * the session id is allowlisted before it touches a path, and refused, not
    sanitised, when it does not match
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from atlas_camera.comfy.nodes_export import _project_routed_dir

#: Mirrors take_ops.py's SLATE_PART. Kept identical on purpose: a session id
#: becomes part of a path in one repo and part of a slate in the other, and two
#: allowlists that drift are one allowlist that does not work.
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")

#: session_id -> {"session_id", "package", "timebase", "slate", "take_dir"}.
#: In-memory: a ComfyUI restart drops it, and the node treats an unknown
#: session as "re-push, or read this slate directly" rather than as an error.
SESSIONS: dict[str, dict] = {}


def validate_session_id(value) -> str:
    text = str(value or "")
    if not SESSION_ID.match(text):
        raise ValueError(
            f"session id is not usable as a path component: {text!r}. "
            "Allowed: letters, digits, underscore, hyphen."
        )
    return text


def session_package_path(project, output_dir: str, session_id: str) -> Path:
    """Where a session's package goes: the project tree's scenes/ lane."""

    session_id = validate_session_id(session_id)
    routed = _project_routed_dir(project, output_dir, "scenes")
    # _project_routed_dir only appends the "scenes" lane when a project is
    # connected (<root>/<project>/<shot>/scenes); with no project it passes
    # output_dir through untouched (nodes_export's callers pass an output_dir
    # that is already the scenes folder). A Director session always wants an
    # actual scenes/ subfolder under whatever root it was handed, project or
    # not, so add the lane ourselves in the no-project case.
    root = Path(routed) if project is not None else Path(output_dir) / "scenes"
    root = root.resolve()
    package = (root / f"{session_id}.atlas").resolve()
    # Belt and braces behind the allowlist, the same posture as
    # take_ops.py::take_directory. If the allowlist ever fails, a session must
    # still refuse to write outside the root it was handed.
    if not str(package).startswith(str(root)):
        raise ValueError(f"session path escapes the project root: {session_id!r}")
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


def launch_session(body: dict, *, spawn=_default_spawn) -> dict:
    """Write a session package and open Director on it.

    `body` supplies a session id, an output directory and the timebase. It
    supplies nothing else that reaches a command line -- `executable` and
    `argv` keys, if present, are ignored rather than honoured.
    """

    session_id = validate_session_id(body.get("session_id"))
    executable = director_executable()
    package = session_package_path(
        body.get("project"), body.get("output_dir") or "atlas_scenes", session_id
    )
    package.parent.mkdir(parents=True, exist_ok=True)

    timebase = {
        "width": int(body["width"]),
        "height": int(body["height"]),
        "frames": int(body["frames"]),
        "fps": int(body["fps"]),
    }
    session = {
        "session_id": session_id,
        "package": str(package),
        "timebase": timebase,
        "slate": None,
        "take_dir": None,
    }
    package.with_suffix(".session.json").write_text(
        json.dumps(session, indent=2), encoding="utf-8"
    )

    SESSIONS[session_id] = session
    spawn([executable, "--director-session", str(package)])
    return session


def record_delivery(session_id: str, slate: str, take_dir: str) -> dict:
    """Remember the take a director pushed.

    Idempotent on (session_id, slate) so a retry after a failed push is safe,
    and last-write-wins on a different slate, because the node's promise is
    that what you pushed is what renders.
    """

    session = SESSIONS[session_id]
    session["slate"] = slate
    session["take_dir"] = take_dir
    return dict(session)
