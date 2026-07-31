"""Local scale-reference registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ScaleReference:
    """A known real-world size usable as a metric anchor.

    Two mutually exclusive kinds, and which one an entry is changes the geometry
    used to solve it:

    * ``height`` — a VERTICAL object standing on the ground (person, door, car).
    * ``ground_span_m`` — a HORIZONTAL distance lying ON the ground (rail gauge,
      sleeper pitch, lane width).

    A span is not a short height. Filing railway gauge as ``height: 1.435`` would
    have it solved as an object standing 1.435 m tall and return a plausible,
    wrong camera height with no error raised — which is why the two fields are
    separate and why exactly one is required.
    """

    id: str
    label: str
    category: str
    height: float | None = None
    ground_span_m: float | None = None
    #: How far the MARKED geometry sits above the local walkable ground, when
    #: that is not zero. Railway gauge is measured across the RAILHEADS, which
    #: stand a rail's height (0.159-0.186 m, typically 0.172) above the ballast
    #: — so a solver told "these two points are on the ground" returns the
    #: camera height above the RAILS, not above the ground.
    #:
    #: Measured 2026-07-31: that is a pure datum error of exactly the offset,
    #: with no distortion (agreement to 7e-15 across heights and pitches), and
    #: it is relative — e/h. At 12 m camera height it is 1.4%, at 4.2 m it is
    #: 4.1%, at eye level 1.6 m it is 10.75%. UNCORRECTED, gauge is therefore
    #: WORSE than an assumed door below 8.35 m and worse than an assumed person
    #: below 3.55 m, which inverts the entire reason for preferring it.
    datum_offset_m: float | None = None
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
        reference_id = str(data["id"])
        height = data.get("height")
        span = data.get("ground_span_m")
        if height is None and span is None:
            # Loud at load, not silent at use. An entry with neither would parse
            # fine and then resolve to nothing much later, inside a solve, as a
            # "no real height" skip that names the spec rather than the registry
            # entry that is actually malformed.
            raise ValueError(
                f"scale reference {reference_id!r} has neither 'height' nor "
                "'ground_span_m'; one is required — 'height' for a vertical "
                "object standing on the ground, 'ground_span_m' for a "
                "horizontal distance lying on it")
        return cls(
            id=reference_id,
            label=str(data["label"]),
            category=str(data.get("category", "uncategorized")),
            height=float(height) if height is not None else None,
            ground_span_m=float(span) if span is not None else None,
            datum_offset_m=(float(data["datum_offset_m"])
                            if data.get("datum_offset_m") is not None else None),
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
            "ground_span_m": self.ground_span_m,
            "datum_offset_m": self.datum_offset_m,
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


@lru_cache(maxsize=1)
def load_scale_references() -> list[ScaleReference]:
    """The registry JSON is small and static for the process lifetime — this
    was previously re-read and re-parsed from disk on every single call.
    get_scale_reference/list_categories/search_scale_references all call this
    fresh each time, and multimodal_helper.py's per-VLM-scale-cue resolution
    can call it in a loop, so an un-cached read repeated the same file I/O +
    JSON parse for every cue in a scene. ScaleReference is a frozen dataclass,
    so the cached list's elements can't be mutated by a caller even though
    the list itself is a single shared object."""
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

