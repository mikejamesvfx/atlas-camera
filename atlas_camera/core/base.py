"""Shared recovered-object contracts for Atlas core types."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from atlas_camera.core.confidence import ConfidenceModel

RecoveredObjectT = TypeVar("RecoveredObjectT", bound="RecoveredObject")


@runtime_checkable
class RecoveredObject(Protocol):
    """Minimal shared surface for concrete recovered objects."""

    schema_version: str
    confidence: ConfidenceModel

    def to_dict(self) -> dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls: type[RecoveredObjectT], data: dict[str, Any]) -> RecoveredObjectT:
        ...
