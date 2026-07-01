"""DTU MVS SampleSet calibration/projection readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True, slots=True)
class DTUProjection:
    path: Path
    projection_matrix: tuple[tuple[float, float, float, float], ...]


def load_dtu_projections(root: str | Path) -> list[DTUProjection]:
    dataset_root = Path(root)
    projections: list[DTUProjection] = []
    for path in sorted(dataset_root.rglob("*.txt")):
        try:
            projections.append(DTUProjection(path=path, projection_matrix=parse_projection_matrix(path)))
        except ValueError:
            continue
    return projections


def parse_projection_matrix(path: str | Path) -> tuple[tuple[float, float, float, float], ...]:
    text = Path(path).read_text(encoding="utf-8")
    values = [
        float(match.group(0))
        for match in re.finditer(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            text,
        )
    ]
    if len(values) < 12:
        raise ValueError(f"DTU projection file does not contain a 3x4 matrix: {path}")
    matrix = values[:12]
    return (
        (matrix[0], matrix[1], matrix[2], matrix[3]),
        (matrix[4], matrix[5], matrix[6], matrix[7]),
        (matrix[8], matrix[9], matrix[10], matrix[11]),
    )
