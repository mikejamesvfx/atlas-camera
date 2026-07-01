"""Lightweight validation helpers adapted from Maya2Comfy concepts."""

from __future__ import annotations

from pathlib import Path


class AtlasValidationError(ValueError):
    """Raised when Atlas input or output validation fails."""


def validate_file_path(path: str | Path, *, must_exist: bool = True) -> Path:
    candidate = Path(path).expanduser()
    if must_exist and not candidate.is_file():
        raise AtlasValidationError(f"File does not exist: {candidate}")
    return candidate


def validate_directory_path(
    path: str | Path,
    *,
    create: bool = False,
    must_exist: bool = True,
) -> Path:
    candidate = Path(path).expanduser()
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if must_exist and not candidate.is_dir():
        raise AtlasValidationError(f"Directory does not exist: {candidate}")
    return candidate

