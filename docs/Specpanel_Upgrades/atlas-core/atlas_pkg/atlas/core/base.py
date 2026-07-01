"""
RecoveredObject — the shared contract for everything Atlas recovers
from an image.

See DECISIONS.md §7. Deliberately does NOT expose a generic `.value`
property: that abstraction is clean for a scalar (a depth sample) and
meaningless for a structured object (a camera *is* its value). The
genuine shared surface is confidence, serialization, and export dispatch.

Subclasses implement `to_<format>()` methods for each export target they
support (e.g. `to_maya()`, `to_json()`). Scene-level export
(`scene.export.maya()`) is a thin orchestrator that calls each component's
own method — it does not know about formats itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from atlas.core.confidence import ConfidenceModel

SCHEMA_VERSION = "0.1.0"


class RecoveredObject(ABC):
    """Base class for every object Atlas recovers from an image.

    Required surface:
        confidence       — a ConfidenceModel instance
        schema_version   — class-level constant, NOT computed at runtime
        to_dict/from_dict — serialization round-trip
    """

    schema_version: str = SCHEMA_VERSION

    def __init__(self, confidence: ConfidenceModel) -> None:
        self.confidence = confidence

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict. Must include `schema_version`."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveredObject":
        """Reconstruct from a dict produced by `to_dict`."""
        raise NotImplementedError
