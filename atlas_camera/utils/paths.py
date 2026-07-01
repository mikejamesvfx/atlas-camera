"""Path helpers for Atlas Camera."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamped_package_name(prefix: str = "atlas_review") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

