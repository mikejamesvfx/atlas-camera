"""Local scale-reference registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ScaleReference:
    id: str
    label: str
    category: str
    height: float
    units: str = "m"
    width: float | None = None
    depth: float | None = None
    confidence: str = "heuristic"
    source_url: str | None = None
    source_note: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    asset_hint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScaleReference":
        return cls(
            id=str(data["id"]),
            label=str(data["label"]),
            category=str(data.get("category", "uncategorized")),
            height=float(data["height"]),
            units=str(data.get("units", "m")),
            width=float(data["width"]) if data.get("width") is not None else None,
            depth=float(data["depth"]) if data.get("depth") is not None else None,
            confidence=str(data.get("confidence", "heuristic")),
            source_url=data.get("source_url"),
            source_note=data.get("source_note"),
            notes=data.get("notes"),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            asset_hint=data.get("asset_hint"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "height": self.height,
            "units": self.units,
            "width": self.width,
            "depth": self.depth,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "source_note": self.source_note,
            "notes": self.notes,
            "tags": list(self.tags),
            "asset_hint": self.asset_hint,
        }


def load_scale_references() -> list[ScaleReference]:
    data_path = resources.files(__package__).joinpath("common_scale_references.json")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    return [ScaleReference.from_dict(item) for item in payload]


def get_scale_reference(reference_id: str) -> ScaleReference:
    for reference in load_scale_references():
        if reference.id == reference_id:
            return reference
    raise KeyError(f"Unknown scale reference: {reference_id}")


def list_categories() -> list[str]:
    return sorted({reference.category for reference in load_scale_references()})


def search_scale_references(
    query: str | None = None,
    *,
    category: str | None = None,
) -> list[ScaleReference]:
    query_text = (query or "").casefold().strip()
    matches: list[ScaleReference] = []
    for reference in load_scale_references():
        if category and reference.category != category:
            continue
        haystack = " ".join(
            [
                reference.id,
                reference.label,
                reference.category,
                reference.notes or "",
                " ".join(reference.tags),
            ]
        ).casefold()
        if not query_text or query_text in haystack:
            matches.append(reference)
    return matches

