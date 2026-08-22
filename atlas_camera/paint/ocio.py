"""OCIO config identity and per-process scoping for the paint bridges.

A colourspace NAME is not a contract. A plate tagged ``ACEScg`` under two
different configs is two different plates, so every bridge manifest and score
report records the config that produced it — path, sha256, and space counts —
and a claim that two applications shared a config becomes checkable instead of
assumed.

``scoped_config`` sets ``$OCIO`` for THIS PROCESS and its children only.
Deliberately not machine-wide: a global ``$OCIO`` silently changes every
existing Atlas read and would invalidate results already recorded against the
built-in config.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from atlas_camera.plate.oiio_io import ColorConfigCache, config_identity

__all__ = ["config_identity", "scoped_config", "describe", "same_config"]


@contextlib.contextmanager
def scoped_config(config_path: str | os.PathLike | None):
    """Point ``$OCIO`` at ``config_path`` for the duration of the block.

    The process-wide ``ColorConfigCache`` is dropped on both entry and exit —
    without that, a process keeps resolving against whichever config it saw
    first and the scoping silently does nothing.

    ``None`` leaves the environment untouched, so a caller can pass an optional
    ``--ocio-config`` straight through.
    """
    if config_path is None:
        yield config_identity()
        return

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"OCIO config not found: {path}. Both ends of a paint bridge must "
            f"resolve the same config or the handoff carries an unverified "
            f"colour assumption.")

    previous = os.environ.get("OCIO")
    os.environ["OCIO"] = str(path)
    ColorConfigCache._cfg = None
    try:
        yield config_identity()
    finally:
        if previous is None:
            os.environ.pop("OCIO", None)
        else:
            os.environ["OCIO"] = previous
        ColorConfigCache._cfg = None


def describe(identity: dict | None = None) -> str:
    """One line naming the active config, for a report header or a CLI banner."""
    identity = identity if identity is not None else config_identity()
    if not identity.get("available"):
        return "OCIO: unavailable (no OpenImageIO)"
    name = identity.get("config_name") or "(unnamed)"
    digest = identity.get("config_sha256") or ""
    where = identity.get("config_path") or "built-in (ocio://default)"
    return (f"OCIO: {name} · {identity.get('n_colorspaces', 0)} spaces / "
            f"{identity.get('n_displays', 0)} displays · {where}"
            + (f" · sha256 {digest[:12]}" if digest else ""))


def same_config(a: dict, b: dict) -> bool:
    """Whether two recorded identities are the same config.

    Compares the sha256 when both have one; falls back to the config name for
    built-ins, which have no file to hash. Two built-ins of the same name from
    different OIIO builds are treated as equal — that is a limit worth knowing
    rather than a claim to hide, and it is why pinning a real config file is
    the recommended setup for a bridge.
    """
    if a.get("config_sha256") and b.get("config_sha256"):
        return a["config_sha256"] == b["config_sha256"]
    return (a.get("config_name") or "") == (b.get("config_name") or "")
