"""
JSON export — file I/O around LatentCamera.to_json()/from_json().

The actual serialization logic lives on LatentCamera itself
(DECISIONS.md §7: objects own to_<format>()). This module is the thin
file-system wrapper scene-level and CLI code call into, kept separate so
neither core nor the CLI needs to import `pathlib` or do file handling
directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.core.latent_camera import LatentCamera


def write_json(camera: "LatentCamera", path: str | Path) -> None:
    Path(path).write_text(camera.to_json())


def read_json(path: str | Path) -> "LatentCamera":
    from atlas.core.latent_camera import LatentCamera
    return LatentCamera.from_json(Path(path).read_text())
