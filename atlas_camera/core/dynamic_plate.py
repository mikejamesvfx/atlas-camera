"""Dynamic Plates: a dynamic region of a solved still, separable from the camera.

A `DynamicPlate` represents one time-varying region (ocean, clouds, smoke, ...)
of an Atlas-solved still image. Atlas owns the camera and geometry; a temporal
generator supplies temporal content only for the known region. The generated
sequence is projected through the FIXED crop camera onto known receiver
geometry, so the artist's render camera stays free to move — the camera move is
never baked into the generated video.

v0.1 scope: `water` is the supported case; the receiver is a horizontal plane
(large distant body of water, moderate camera move, small wave displacement
relative to scene scale — temporal appearance, not simulated water geometry).
Other semantic types exist as contracts/status placeholders.

Host-agnostic core: stdlib always; numpy only inside the matte helpers via
`_require_numpy` (install with ``pip install -e .[vision]``).
"""
from __future__ import annotations

import json
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.schema import (
    AtlasProxyPrimitive,
    LatentCamera,
    _json_ready,
)

# ---------------------------------------------------------------------------
# Semantic types (spec §4). Plain string constants — the repo has no Enums.
# Lowercase to match every other Atlas status/provenance string. Append-only.
DYNAMIC_REGION_TYPES = (
    "water", "cloud", "smoke", "fire", "foliage", "cloth", "actor", "generic",
)

# Plate lifecycle statuses.
PLATE_STATUS_DRAFT = "draft"
PLATE_STATUS_READY = "ready"
PLATE_STATUS_GENERATED = "generated"
PLATE_STATUS_FAILED = "failed"

# Generator availability sentinel (spec §32).
GENERATOR_NOT_AVAILABLE = "not_available"

# Failure/status codes (spec §31), lowercase snake like scene_health codes.
REGION_INVALID = "region_invalid"
CAMERA_CROP_FAILURE = "camera_crop_failure"
RECEIVER_GEOMETRY_UNAVAILABLE = "receiver_geometry_unavailable"
GENERATOR_UNAVAILABLE = "generator_unavailable"
GENERATION_FAILURE = "generation_failure"
FRAME_SEQUENCE_INCOMPLETE = "frame_sequence_incomplete"
PROJECTION_SETUP_FAILURE = "projection_setup_failure"
EXPORT_FAILURE = "export_failure"

# Editable default prompt for water (spec §18: a preset, not a magic string).
WATER_PROMPT_DEFAULT = (
    "preserve coastline and large-scale composition; animate natural ocean "
    "wave motion, foam, surface variation, specular movement and subtle "
    "swell; no camera movement; no alteration to static land or architecture"
)

_DEFAULT_PROVENANCE = {
    "source_region": "observed",
    "crop_camera": "derived_from_solve",
    "receiver_geometry": "derived_from_solve",
    "generated_frames": "generated",
}


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Dynamic-plate matte utilities require numpy. Install with:\n"
            "    pip install -e .[vision]") from exc
    return np


@dataclass(slots=True)
class ReceiverGeometry:
    """The known scene geometry that receives the temporal projection.

    v0.1: a plane (`primitive` carries the Atlas Y-up world transform +
    dimensions). `path` is a package-relative exported mesh when written to
    disk. The receiver stays part of the Atlas scene — the generated sequence
    attaches to it, never to the artist camera.
    """

    kind: str = "plane"
    primitive: AtlasProxyPrimitive | None = None
    path: str | None = None
    coordinate_system: str = "right_handed"
    up_axis: str = "Y"
    provenance: str = "derived_from_solve"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReceiverGeometry | None":
        if not data:
            return None
        prim = data.get("primitive")
        return cls(
            kind=data.get("kind", "plane"),
            primitive=AtlasProxyPrimitive.from_dict(prim) if prim else None,
            path=data.get("path"),
            coordinate_system=data.get("coordinate_system", "right_handed"),
            up_axis=data.get("up_axis", "Y"),
            provenance=data.get("provenance", "derived_from_solve"),
        )


@dataclass(slots=True)
class DynamicPlate:
    """One dynamic region of a solved still (spec §3).

    The temporal sequence is a time-varying texture projected through the
    fixed `crop_camera` onto `receiver` geometry; all frames share that camera
    and geometry while the artist camera moves independently (spec §11).
    """

    plate_id: str
    semantic_type: str
    source_image: str
    source_width: int
    source_height: int
    matte_path: str | None = None
    matte_bbox: RegionROI | None = None
    source_roi: RegionROI | None = None
    crop_transform: CropTransform | None = None
    source_camera: LatentCamera | None = None
    crop_camera: LatentCamera | None = None
    receiver: ReceiverGeometry | None = None
    frame_rate: float = 24.0
    frame_start: int = 0
    frame_end: int = 0
    generator: str = ""
    generator_config: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    seed: int | None = None
    projection_mode: str = "camera_projection"
    matte_feather_px: float = 0.0
    color_metadata: dict[str, Any] = field(default_factory=lambda: {
        "input_color_space": "sRGB",
        "generator_input_transform": "none",
        "generator_output_color_space": "sRGB",
        "atlas_working_color_space": "sRGB",
    })
    provenance: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_PROVENANCE))
    status: str = PLATE_STATUS_DRAFT
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = "0.1"

    def __post_init__(self) -> None:
        if self.semantic_type not in DYNAMIC_REGION_TYPES:
            raise ValueError(
                f"Unknown dynamic-region type {self.semantic_type!r}; "
                f"expected one of {DYNAMIC_REGION_TYPES}")

    @property
    def frame_count(self) -> int:
        return max(0, int(self.frame_end) - int(self.frame_start) + 1)

    def to_dict(self) -> dict[str, Any]:
        data = _json_ready(self)
        data["schema_version"] = self.schema_version
        data["plate_type"] = "dynamic_plate"
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicPlate":
        src_cam = data.get("source_camera")
        crop_cam = data.get("crop_camera")
        plate = cls(
            plate_id=str(data["plate_id"]),
            semantic_type=str(data["semantic_type"]),
            source_image=str(data.get("source_image", "")),
            source_width=int(data.get("source_width", 0)),
            source_height=int(data.get("source_height", 0)),
            matte_path=data.get("matte_path"),
            matte_bbox=RegionROI.from_dict(data.get("matte_bbox")),
            source_roi=RegionROI.from_dict(data.get("source_roi")),
            crop_transform=CropTransform.from_dict(data.get("crop_transform")),
            source_camera=LatentCamera.from_dict(src_cam) if src_cam else None,
            crop_camera=LatentCamera.from_dict(crop_cam) if crop_cam else None,
            receiver=ReceiverGeometry.from_dict(data.get("receiver")),
            frame_rate=float(data.get("frame_rate", 24.0)),
            frame_start=int(data.get("frame_start", 0)),
            frame_end=int(data.get("frame_end", 0)),
            generator=data.get("generator", ""),
            generator_config=dict(data.get("generator_config", {})),
            prompt=data.get("prompt", ""),
            seed=data.get("seed"),
            projection_mode=data.get("projection_mode", "camera_projection"),
            matte_feather_px=float(data.get("matte_feather_px", 0.0)),
            status=data.get("status", PLATE_STATUS_DRAFT),
            warnings=list(data.get("warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )
        if data.get("color_metadata"):
            plate.color_metadata = dict(data["color_metadata"])
        if data.get("provenance"):
            plate.provenance = dict(data["provenance"])
        return plate
