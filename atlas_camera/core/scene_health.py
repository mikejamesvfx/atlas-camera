"""Derived trust/health evaluation for Atlas solves — pure Python, zero deps.

The single home for "is this scene trustworthy?" logic (per the 2026-07-17
engineering-recommendations response): scale provenance -> safe-to-export
status here in M1; the full scene-health engine (the AtlasDebugReport red-flag
checks, consumed by both that node and AtlasSceneHealthGate) lands here in a
later milestone. Everything is DERIVED — this module never changes solver
behavior, only reads provenance the solvers already record.

Callers on the serialization hot path (LatentScene.to_dict) rely on
scale_health() being exception-proof: any surprise degrades to the
"unknown / not safe" answer, never a raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCALE_STATUS_MEASURED = "measured"
SCALE_STATUS_MANUAL = "manual"
SCALE_STATUS_ASSUMED = "assumed"
SCALE_STATUS_UNKNOWN = "unknown"

# Fixed nominal confidence for the assumed-eye-height fallback: it is a guess,
# not a measurement — kept > 0 only so downstream sorting/plotting behaves.
_ASSUMED_CONFIDENCE = 0.15


@dataclass(slots=True)
class ScaleHealth:
    status: str                     # measured | manual | assumed | unknown
    scale_source: str | None
    confidence: float | None
    camera_height_m: float | None
    safe_to_export: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scale_source": self.scale_source,
            "confidence": self.confidence,
            "camera_height_m": self.camera_height_m,
            "safe_to_export": self.safe_to_export,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScaleHealth | None":
        if not isinstance(data, dict):
            return None
        return cls(
            status=str(data.get("status", SCALE_STATUS_UNKNOWN)),
            scale_source=data.get("scale_source"),
            confidence=(float(data["confidence"])
                        if data.get("confidence") is not None else None),
            camera_height_m=(float(data["camera_height_m"])
                             if data.get("camera_height_m") is not None else None),
            safe_to_export=bool(data.get("safe_to_export", False)),
            detail=str(data.get("detail", "")),
        )


def _camera_height(solve: Any) -> float | None:
    try:
        pos = solve.camera.extrinsics.camera_position
        return float(pos[1])
    except Exception:  # noqa: BLE001 — hand-built/partial solves
        return None


def _depth_is_metric(solve: Any) -> bool | None:
    """True/False when the depth component records it; None when unknown."""
    try:
        depth = getattr(solve, "depth", None)
        value = getattr(depth, "value", None)
        if isinstance(value, dict) and "is_metric" in value:
            return bool(value["is_metric"])
    except Exception:  # noqa: BLE001
        pass
    return None


def scale_health(solve: Any) -> ScaleHealth:
    """Map the solve's recorded scale provenance to a trust status.

    Reads only what the solvers already stamp (``debug_metadata`` +
    ``solve.depth``); patch-view registration provenance (``reuse_scene`` /
    ``ground_fit`` / ``primary_registration`` on ProjectionSource metadata) is
    deliberately out of scope — layers inherit the scene's scale.
    """
    try:
        return _scale_health_inner(solve)
    except Exception:  # noqa: BLE001 — hot path (to_dict); never raise
        return ScaleHealth(
            status=SCALE_STATUS_UNKNOWN, scale_source=None, confidence=None,
            camera_height_m=None, safe_to_export=False,
            detail="Scale provenance could not be evaluated.")


def _scale_health_inner(solve: Any) -> ScaleHealth:
    meta = getattr(solve, "debug_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    source = meta.get("scale_source")
    height = _camera_height(solve)

    if source == "reference_object":
        ref = meta.get("reference_scale") or {}
        conf = float(ref.get("confidence", 0.0)) if isinstance(ref, dict) else 0.0
        return ScaleHealth(
            SCALE_STATUS_MEASURED, source, conf, height, True,
            f"Metric scale measured from reference object(s), "
            f"consistency {conf:.2f}.")

    if source == "depth_ground_plane":
        depth = getattr(solve, "depth", None)
        conf = float(getattr(depth, "confidence", 0.0) or 0.0)
        is_metric = _depth_is_metric(solve)
        if is_metric is False:
            return ScaleHealth(
                SCALE_STATUS_MEASURED, source, conf, height, False,
                "Camera height measured from RELATIVE depth — up-to-scale "
                "only, not metric. Verify with a reference or manual height.")
        return ScaleHealth(
            SCALE_STATUS_MEASURED, source, conf, height, True,
            f"Camera height measured from the depth ground plane, "
            f"confidence {conf:.2f}.")

    if source == "manual_override":
        return ScaleHealth(
            SCALE_STATUS_MANUAL, source, 1.0, height, True,
            "Metric scale set manually (artist decision).")

    if source == "assumed_default":
        detail = "Assumed 1.6 m eye height — NOT measured."
        if height is not None:
            detail = (f"Assumed eye height ({height:g} m) — NOT measured. "
                      "Elevated/AI plates are typically far off; use "
                      "AtlasScaleOverride or a scale reference.")
        return ScaleHealth(
            SCALE_STATUS_ASSUMED, source, _ASSUMED_CONFIDENCE, height, False,
            detail)

    return ScaleHealth(
        SCALE_STATUS_UNKNOWN, source if isinstance(source, str) else None,
        None, height, False,
        "No metric-scale provenance recorded on this solve.")
