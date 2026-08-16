"""Agent handoff — pause a graph, hand an external agent a BRIEF, resume on its reply.

WHY. After the Blender massing node the useful next step is judgement + hands:
look at the scene, model what is missing in Blender, hand it back. That is an
agent's job (Claude Code through the atlas MCP, Hermes, OpenClaw — anything
with a vision model and tools), not a node's. The node's job is the CONTRACT:
publish everything the agent needs, wait, accept the reply, bring the .blend
back in. No model, no MCP client, no subprocess-to-an-LLM inside ComfyUI —
ComfyUI stays the engine, the agent stays pluggable.

WIRE FORMAT (all under <ComfyUI output>/atlas_agent/<node_id>/):

  brief.json    written by the node when it pauses. Task text, exchange dir,
                seed/blend paths, snapshot PNG paths, measured numbers, allowed
                tools, solve fingerprint, resume instructions, deadline.
  resume.json   written by the agent (MCP `atlas_agent_resume`, or
                `POST /atlas/agent/resume/{node_id}`, or by hand):
                {"status": "done"|"skip"|"fail", "reply": str,
                 "blend_file": str?, "notes": str?}
  history/      every brief/resume pair, timestamped, so a run is auditable.

The pause is BLOCKING by design (the queue waits — that is what a pause is);
`timeout_s` bounds it and `on_timeout` decides whether the graph continues or
fails. A stale resume (written for a previous brief) is refused by the brief's
token, so an old reply can never release a new pause.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

AGENT_DIRNAME = "atlas_agent"
BRIEF_NAME = "brief.json"
RESUME_NAME = "resume.json"
HISTORY_DIRNAME = "history"
RESUME_STATUSES = ("done", "skip", "fail")


def output_root(default: str = "output") -> Path:
    """<ComfyUI output> when running inside ComfyUI, else ``default``."""
    try:
        import folder_paths  # type: ignore[import]
        return Path(folder_paths.get_output_directory())
    except Exception:  # noqa: BLE001
        return Path(default)


def agent_dir(node_id: Any, *, root: str | Path | None = None) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(node_id))[:64]
    return Path(root) / AGENT_DIRNAME / (safe or "unknown") if root is not None \
        else output_root() / AGENT_DIRNAME / (safe or "unknown")


def write_brief(node_id: Any, brief: dict[str, Any], *, root: str | Path | None = None,
                now: float | None = None) -> dict[str, Any]:
    """Write brief.json (+ history copy) and CLEAR any previous resume.json.

    Adds ``token`` (the resume must echo it), ``issued_at``, ``deadline`` and
    the ``resume`` instructions. Returns the brief as written.
    """
    d = agent_dir(node_id, root=root)
    (d / HISTORY_DIRNAME).mkdir(parents=True, exist_ok=True)
    ts = float(now if now is not None else time.time())
    token = uuid.uuid4().hex[:12]
    full = dict(brief)
    full.update({
        "node_id": str(node_id),
        "token": token,
        "issued_at": ts,
        "deadline": ts + float(brief.get("timeout_s") or 0),
        "resume": {
            "how": [
                f"MCP: atlas_agent_resume(node_id={node_id!s}, token='{token}', "
                "status='done', reply='...', blend_file='<path or empty>')",
                f"HTTP: POST /atlas/agent/resume/{node_id} "
                "{\"token\": \"%s\", \"status\": \"done\", \"reply\": \"...\", "
                "\"blend_file\": \"...\"}" % token,
                f"file: write {d / RESUME_NAME} with the same JSON",
            ],
            "statuses": list(RESUME_STATUSES),
        },
    })
    stale = d / RESUME_NAME
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass
    (d / BRIEF_NAME).write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    shutil.copyfile(d / BRIEF_NAME, d / HISTORY_DIRNAME / f"{stamp}_{token}_brief.json")
    return full


def read_brief(node_id: Any, *, root: str | Path | None = None) -> dict[str, Any] | None:
    p = agent_dir(node_id, root=root) / BRIEF_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def write_resume(node_id: Any, payload: dict[str, Any], *,
                 root: str | Path | None = None) -> dict[str, Any]:
    """Validate + write resume.json. Raises ValueError on a bad payload.

    The token must match the CURRENT brief (or the brief must be absent —
    a resume written before the node paused is allowed and simply picked up).
    """
    status = str(payload.get("status") or "done").lower()
    if status not in RESUME_STATUSES:
        raise ValueError(f"status must be one of {RESUME_STATUSES}; got {status!r}")
    brief = read_brief(node_id, root=root)
    token = str(payload.get("token") or "")
    if brief is not None and brief.get("token") and token != brief.get("token"):
        raise ValueError(
            f"token {token!r} does not match the current brief ({brief.get('token')!r}) — "
            "read the brief first; an old reply cannot release a new pause")
    blend = str(payload.get("blend_file") or "").strip()
    rec = {
        "node_id": str(node_id), "token": token, "status": status,
        "reply": str(payload.get("reply") or ""),
        "blend_file": blend, "notes": payload.get("notes") or "",
        "written_at": time.time(),
    }
    d = agent_dir(node_id, root=root)
    d.mkdir(parents=True, exist_ok=True)
    (d / RESUME_NAME).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


def read_resume(node_id: Any, *, root: str | Path | None = None) -> dict[str, Any] | None:
    p = agent_dir(node_id, root=root) / RESUME_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def wait_for_resume(node_id: Any, token: str, *, timeout_s: float, poll_s: float = 1.0,
                    root: str | Path | None = None,
                    sleep: Callable[[float], None] = time.sleep,
                    clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Block until a resume.json with the right token appears, or time out.

    Returns the resume record, or ``{"status": "timeout"}``. A resume with the
    WRONG token is ignored (and archived) rather than honoured.
    """
    d = agent_dir(node_id, root=root)
    t0 = clock()
    while True:
        rec = read_resume(node_id, root=root)
        if rec is not None:
            if not rec.get("token") or rec.get("token") == token:
                _archive_resume(d, rec)
                return rec
            _archive_resume(d, rec, stale=True)
        if clock() - t0 >= float(timeout_s):
            return {"status": "timeout", "reply": "", "blend_file": "",
                    "waited_s": float(clock() - t0)}
        sleep(float(poll_s))


def _archive_resume(d: Path, rec: dict[str, Any], *, stale: bool = False) -> None:
    try:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = "stale_" if stale else ""
        (d / HISTORY_DIRNAME).mkdir(parents=True, exist_ok=True)
        (d / HISTORY_DIRNAME / f"{stamp}_{tag}{rec.get('token', 'notoken')}_resume.json").write_text(
            json.dumps(rec, indent=1, default=str), encoding="utf-8")
        (d / RESUME_NAME).unlink()
    except OSError:
        pass
