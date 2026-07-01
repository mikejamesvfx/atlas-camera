"""
ConfidenceModel — the uniform confidence schema for recovered objects.

See DECISIONS.md §6.

Two parts, always:
    global_score        — overall recovery confidence, 0.0-1.0
    individual_metrics   — per-parameter confidence, 0.0-1.0 each

These are RELATIVE HEURISTICS, not calibrated probabilities. A score of
0.82 means "more reliable than a 0.6 score from this same recovery run,"
not "82% likely to be within some tolerance of ground truth." This
distinction is load-bearing and must not be blurred in UI copy or docs.

Every RecoveredObject subclass defines its own fixed key set for
individual_metrics. Ad hoc keys invented per-call are not permitted —
that's exactly the inconsistency this schema exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp(v: float) -> float:
    """Confidence is always in [0.0, 1.0], regardless of what the
    inference pipeline hands back. This is enforced at construction,
    not assumed."""
    return max(0.0, min(1.0, float(v)))


@dataclass
class ConfidenceModel:
    global_score: float
    individual_metrics: dict[str, float] = field(default_factory=dict)

    # Fixed key set for LatentCamera. Other RecoveredObject subclasses
    # define their own constant of this name with their own keys.
    LATENT_CAMERA_KEYS = (
        "horizon", "vp1", "vp2", "vp3", "focal", "extrinsics", "sensor",
    )

    def __post_init__(self) -> None:
        self.global_score = _clamp(self.global_score)
        self.individual_metrics = {
            k: _clamp(v) for k, v in self.individual_metrics.items()
        }

    def set_metric(self, key: str, value: float) -> None:
        """Set (or lower) an individual metric, always clamped."""
        self.individual_metrics[key] = _clamp(value)

    def get_metric(self, key: str, default: float = 0.0) -> float:
        return self.individual_metrics.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_score": self.global_score,
            "individual_metrics": dict(self.individual_metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfidenceModel":
        return cls(
            global_score=data["global_score"],
            individual_metrics=dict(data.get("individual_metrics", {})),
        )
