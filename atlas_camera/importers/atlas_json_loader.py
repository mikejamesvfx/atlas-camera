"""Atlas solve JSON loader."""

from __future__ import annotations

from pathlib import Path

from atlas_camera.core.io import load_solve_json
from atlas_camera.core.schema import AtlasSolve


def load_atlas_solve(path: str | Path) -> AtlasSolve:
    return load_solve_json(path)

