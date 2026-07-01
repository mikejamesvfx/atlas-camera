"""JSON IO for Atlas solves."""

from __future__ import annotations

import json
from pathlib import Path

from atlas_camera.core.schema import AtlasSolve


def save_solve_json(solve: AtlasSolve, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(solve.to_json(indent=2) + "\n", encoding="utf-8")
    return destination


def load_solve_json(path: str | Path) -> AtlasSolve:
    payload = Path(path).read_text(encoding="utf-8")
    return AtlasSolve.from_dict(json.loads(payload))

