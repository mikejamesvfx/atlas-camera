"""Camera-body sensor-dimension registry (EXIF model string -> sensor mm).

Mirrors registry.py's scale-reference pattern: a frozen dataclass over a
packaged JSON file, loaded once per process. Used by the RAW import path to
turn an EXIF camera model into real sensor dimensions so
``build_intrinsics(focal_length_mm=..., sensor_width_mm=...)`` gets measured
values instead of the 36.0 mm full-frame assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class CameraBody:
    id: str
    make: str
    model_aliases: tuple[str, ...]
    sensor_width_mm: float
    sensor_height_mm: float
    mount: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraBody":
        return cls(
            id=str(data["id"]),
            make=str(data["make"]),
            model_aliases=tuple(str(alias) for alias in data["model_aliases"]),
            sensor_width_mm=float(data["sensor_width_mm"]),
            sensor_height_mm=float(data["sensor_height_mm"]),
            mount=data.get("mount"),
            notes=data.get("notes"),
        )


@lru_cache(maxsize=1)
def load_camera_bodies() -> list[CameraBody]:
    data_path = resources.files(__package__).joinpath("camera_bodies.json")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    return [CameraBody.from_dict(item) for item in payload]


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def find_camera_body(make: str | None, model: str | None) -> CameraBody | None:
    """Match an EXIF make/model pair against the registry.

    EXIF strings are messy — "NIKON CORPORATION" + "NIKON D810",
    "Canon" + "Canon EOS R5", "SONY" + "ILCE-7M3" — so matching is
    whitespace-collapsed, casefolded, and tried both with the model as-is
    and with a duplicated leading make word stripped.
    """
    if not model:
        return None
    norm_model = _normalize(model)
    candidates = {norm_model}
    if make:
        first_make_word = _normalize(make).split(" ")[0]
        if first_make_word and norm_model.startswith(first_make_word + " "):
            candidates.add(norm_model[len(first_make_word) + 1:])
    for body in load_camera_bodies():
        aliases = {_normalize(alias) for alias in body.model_aliases}
        if candidates & aliases:
            return body
    return None
