"""
LatentCamera — the virtual camera Atlas recovers from a single image's
perspective, horizon, vanishing points, and spatial cues.

This is the FIRST implementation of RecoveredObject (DECISIONS.md §7),
not a bespoke dataclass — LatentDepth, LatentGeometry, etc. will follow
the same shape later.

Coordinate convention: OpenCV-native, right-handed, +Y down, camera
looks down +Z into the scene (DECISIONS.md §1). This is true for
world_matrix, view_matrix, and projection_matrix on every instance,
unconditionally. Exporters are the only place this convention changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.core.base import RecoveredObject, SCHEMA_VERSION
from atlas.core.confidence import ConfidenceModel
from atlas.core import camera_math

Matrix4 = list[list[float]]

_IDENTITY_4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _validate_matrix4(m: Matrix4, name: str) -> None:
    if len(m) != 4 or any(len(row) != 4 for row in m):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape "
                          f"{len(m)}x{len(m[0]) if m else 0}")


@dataclass
class LatentCamera(RecoveredObject):
    """The recovered virtual camera for a single image.

    `focal_length_mm` is Optional: if it could not be recovered directly,
    it is filled via camera_math.estimate_focal_with_fallback and flagged
    with `focal_inferred=True` plus a lowered confidence + a note — never
    silently invented (DECISIONS.md §3).
    """

    image_width: int
    image_height: int
    sensor_width_mm: float
    sensor_height_mm: float
    principal_point_px: tuple[float, float]
    film_offset: tuple[float, float]
    world_matrix: Matrix4
    view_matrix: Matrix4
    projection_matrix: Matrix4
    confidence: ConfidenceModel

    focal_length_mm: float | None = None
    focal_inferred: bool = False
    rotation_euler: tuple[float, float, float] | None = None
    translation: tuple[float, float, float] | None = None
    horizon_line: tuple[float, float, float] | None = None
    vanishing_points: list[tuple[float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seed: int | None = None

    schema_version: str = SCHEMA_VERSION
    coordinate_convention: str = "opencv"

    def __post_init__(self) -> None:
        for m, name in (
            (self.world_matrix, "world_matrix"),
            (self.view_matrix, "view_matrix"),
            (self.projection_matrix, "projection_matrix"),
        ):
            _validate_matrix4(m, name)

    # ------------------------------------------------------------------
    # Construction helper implementing the focal-length fallback contract
    # ------------------------------------------------------------------
    @classmethod
    def with_estimated_focal(
        cls,
        *,
        fov_deg: float,
        sensor_width_mm: float | None,
        sensor_height_mm: float | None = None,
        **kwargs: Any,
    ) -> "LatentCamera":
        """Build a LatentCamera when focal length must be derived from
        FOV. Applies DECISIONS.md §3: fallback sensor only as last
        resort, confidence penalty applied, note written — never silent.
        """
        focal, used_fallback, penalty = camera_math.estimate_focal_with_fallback(
            fov_deg, sensor_width_mm,
        )
        confidence: ConfidenceModel = kwargs.pop("confidence")
        if used_fallback:
            current = confidence.get_metric("focal", default=0.7)
            confidence.set_metric("focal", current - penalty)
            notes = list(kwargs.pop("notes", []))
            notes.append(
                f"Focal length inferred from FOV estimate — sensor "
                f"assumed {camera_math.FALLBACK_SENSOR_WIDTH_MM}mm "
                f"full-frame (no sensor size recovered directly)."
            )
            kwargs["notes"] = notes
        resolved_sensor_w = (
            sensor_width_mm if sensor_width_mm is not None
            else camera_math.FALLBACK_SENSOR_WIDTH_MM
        )
        resolved_sensor_h = (
            sensor_height_mm if sensor_height_mm is not None
            else camera_math.FALLBACK_SENSOR_HEIGHT_MM
        )
        return cls(
            sensor_width_mm=resolved_sensor_w,
            sensor_height_mm=resolved_sensor_h,
            focal_length_mm=focal,
            focal_inferred=used_fallback,
            confidence=confidence,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # RecoveredObject contract
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_convention": self.coordinate_convention,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "focal_length_mm": self.focal_length_mm,
            "focal_inferred": self.focal_inferred,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "principal_point_px": list(self.principal_point_px),
            "film_offset": list(self.film_offset),
            "world_matrix": self.world_matrix,
            "view_matrix": self.view_matrix,
            "projection_matrix": self.projection_matrix,
            "rotation_euler": list(self.rotation_euler) if self.rotation_euler else None,
            "translation": list(self.translation) if self.translation else None,
            "horizon_line": list(self.horizon_line) if self.horizon_line else None,
            "vanishing_points": [list(vp) for vp in self.vanishing_points],
            "confidence": self.confidence.to_dict(),
            "notes": list(self.notes),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentCamera":
        d = dict(data)
        confidence = ConfidenceModel.from_dict(d.pop("confidence"))
        pp = d.pop("principal_point_px")
        fo = d.pop("film_offset")
        rot = d.pop("rotation_euler", None)
        trans = d.pop("translation", None)
        horizon = d.pop("horizon_line", None)
        vps = d.pop("vanishing_points", [])
        schema_version = d.pop("schema_version", SCHEMA_VERSION)
        coord = d.pop("coordinate_convention", "opencv")
        return cls(
            confidence=confidence,
            principal_point_px=tuple(pp),
            film_offset=tuple(fo),
            rotation_euler=tuple(rot) if rot is not None else None,
            translation=tuple(trans) if trans is not None else None,
            horizon_line=tuple(horizon) if horizon is not None else None,
            vanishing_points=[tuple(vp) for vp in vps],
            schema_version=schema_version,
            coordinate_convention=coord,
            **d,
        )

    # ------------------------------------------------------------------
    # Export — objects own to_<format>(); scene-level export is a thin
    # orchestrator over these (DECISIONS.md §7).
    # ------------------------------------------------------------------
    def to_json(self, *, indent: int | None = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "LatentCamera":
        import json
        return cls.from_dict(json.loads(s))

    def to_maya(self, *, camera_name: str | None = None) -> str:
        """Maya ASCII (.ma) text for this camera. Delegates all unit
        conversion to camera_math and all node naming to the frozen
        names in export.maya (DECISIONS.md §2, §4)."""
        from atlas.export import maya as maya_export
        return maya_export.export_latent_camera(self, camera_name=camera_name)
