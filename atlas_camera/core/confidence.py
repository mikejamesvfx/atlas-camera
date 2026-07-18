"""Structured confidence metadata for recovered Atlas objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

LATENT_CAMERA_CONFIDENCE_KEYS = (
    "horizon",
    "vp1",
    "vp2",
    "vp3",
    "focal",
    "extrinsics",
    "sensor",
    # APPENDED 2026-07-18 (append-only tuple — serialized dicts must stay
    # loadable both directions): metric-scale trust + depth ground-fit
    # confidence, per the P0 trust tier. Per-layer mask/hidden confidences
    # deliberately live in scene-health per_layer, not on the camera.
    "scale",
    "depth",
)


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class ConfidenceModel:
    """Relative heuristic confidence, not calibrated probability."""

    global_score: float = 0.0
    individual_metrics: dict[str, float] = field(default_factory=dict)
    metric_semantics: str = "relative_heuristic"
    LATENT_CAMERA_KEYS: ClassVar[tuple[str, ...]] = LATENT_CAMERA_CONFIDENCE_KEYS

    def __post_init__(self) -> None:
        self.global_score = clamp_confidence(self.global_score)
        self.individual_metrics = {
            str(key): clamp_confidence(value)
            for key, value in self.individual_metrics.items()
        }

    @classmethod
    def for_latent_camera(
        cls,
        *,
        global_score: float = 0.0,
        defaults: float = 0.0,
        overrides: dict[str, float] | None = None,
    ) -> "ConfidenceModel":
        metrics = {key: defaults for key in LATENT_CAMERA_CONFIDENCE_KEYS}
        metrics.update(overrides or {})
        return cls(global_score=global_score, individual_metrics=metrics)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConfidenceModel":
        if not data:
            return cls()
        return cls(
            global_score=float(data.get("global_score", 0.0)),
            individual_metrics=dict(data.get("individual_metrics", {})),
            metric_semantics=str(data.get("metric_semantics", "relative_heuristic")),
        )

    def with_metric(self, key: str, value: float) -> "ConfidenceModel":
        metrics = dict(self.individual_metrics)
        metrics[key] = value
        return ConfidenceModel(
            global_score=self.global_score,
            individual_metrics=metrics,
            metric_semantics=self.metric_semantics,
        )

    def lower_metric(self, key: str, penalty: float) -> None:
        current = self.individual_metrics.get(key, self.global_score)
        self.individual_metrics[key] = clamp_confidence(current - penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_score": self.global_score,
            "individual_metrics": dict(self.individual_metrics),
            "metric_semantics": self.metric_semantics,
        }
