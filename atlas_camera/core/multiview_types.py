"""Typed, dependency-free contracts for deterministic multi-view registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import os
from typing import Any, Literal

from atlas_camera.core.schema import AtlasPlateRef, AtlasSolve, _json_ready

CaptureMode = Literal["auto", "translated", "rotation_only"]
MatchQuality = Literal["balanced", "conservative", "permissive"]
OutcomeCode = Literal[
    "translated", "rotation_only", "metadata_mismatch",
    "insufficient_overlap", "dynamic_scene_contamination",
    "degenerate_geometry", "scale_unavailable",
    "inconsistent_third_view", "ambiguous_motion_model",
]


@dataclass(frozen=True)
class QualityProfile:
    ratio: float
    min_inliers: int
    reprojection_threshold_px: float
    min_triangulation_angle_deg: float
    min_grid_cells: int
    max_features: int


QUALITY_PROFILES = {
    "conservative": QualityProfile(0.70, 64, 1.0, 1.5, 8, 8000),
    "balanced": QualityProfile(0.75, 48, 1.5, 1.0, 6, 8000),
    "permissive": QualityProfile(0.80, 32, 2.5, 0.5, 4, 10000),
}


@dataclass(frozen=True)
class MultiViewSettings:
    capture_mode: CaptureMode = "auto"
    camera_height_m: float = 0.0
    match_quality: MatchQuality = "balanced"
    seed: int = 0
    #: Optional world-up direction in Atlas camera coordinates for photo 1,
    #: used ONLY when the deterministic vanishing-point anchor fails.  The
    #: caller (adapter layer) supplies it — typically from a learned prior —
    #: so core stays torch-free.  It enters the registration fingerprint via
    #: asdict, and using it is recorded as a diagnostics warning.
    anchor_up_hint: tuple[float, float, float] | None = None
    anchor_up_hint_source: str = ""

    def __post_init__(self) -> None:
        if self.capture_mode not in ("auto", "translated", "rotation_only"):
            raise ValueError(f"capture_mode must be auto, translated, or rotation_only; got {self.capture_mode!r}")
        if self.match_quality not in QUALITY_PROFILES:
            raise ValueError(f"match_quality must be balanced, conservative, or permissive; got {self.match_quality!r}")


@dataclass(frozen=True)
class MultiViewFrame:
    image: Any
    raw_meta: Any
    plate_ref: AtlasPlateRef | None = None
    label: str = ""


@dataclass(frozen=True)
class FeatureSet:
    points_xy: Any
    descriptors: Any
    responses: Any
    stable_indices: Any
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class PairMatches:
    frame_a: int
    frame_b: int
    points_a: Any
    points_b: Any
    indices: Any
    distances: Any
    occupied_grid_cells: int


@dataclass(frozen=True)
class PairModelEvidence:
    frame_a: int
    frame_b: int
    essential_matrix: Any | None
    homography: Any | None
    relative_rotation: Any | None
    translation_direction: Any | None
    essential_inliers: Any
    homography_inliers: Any
    essential_inlier_count: int
    homography_inlier_count: int
    median_essential_error_px: float
    median_homography_error_px: float
    median_triangulation_angle_deg: float
    positive_depth_fraction: float
    essential_occupied_grid_cells: int = -1
    homography_rotation_residual_px: float | None = None


@dataclass
class RegistrationDiagnostics:
    outcome_code: OutcomeCode
    summary: str
    selected_mode: str | None = None
    metadata_checks: list[dict[str, Any]] = field(default_factory=list)
    pair_metrics: list[dict[str, Any]] = field(default_factory=list)
    camera_metrics: list[dict[str, Any]] = field(default_factory=list)
    scale: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass
class RegistrationOutcome:
    solve: AtlasSolve | None
    diagnostics: RegistrationDiagnostics
    overlays: tuple[Any, ...] = ()

    @classmethod
    def failed(cls, code: OutcomeCode, summary: str) -> "RegistrationOutcome":
        return cls(None, RegistrationDiagnostics(code, summary))


_RAW_VALIDATION_FIELDS = (
    "camera_make", "camera_model", "lens_model", "focal_length_mm",
    "sensor_width_mm", "sensor_height_mm", "sensor_source", "orientation",
    "undistort_status", "source_path",
)


def registration_fingerprint(frames: list[MultiViewFrame] | tuple[MultiViewFrame, ...],
                             settings: MultiViewSettings) -> str:
    """Hash every input that can change deterministic registration output."""
    digest = hashlib.sha256()
    digest.update(b"atlas-camera-registration-v1\0")
    _update_json(digest, {"settings": asdict(settings), "seed": settings.seed})
    for index, frame in enumerate(frames):
        _update_json(digest, {"frame": index, "label": frame.label})
        _update_image(digest, frame.image)
        _update_json(digest, _raw_validation_values(frame.raw_meta))
        _update_json(digest, {
            "plate_source_path": (
                frame.plate_ref.image_path if frame.plate_ref is not None else None
            ),
        })
    return digest.hexdigest()


def _update_image(digest: Any, image: Any) -> None:
    shape = tuple(int(value) for value in getattr(image, "shape", ()))
    dtype = str(getattr(image, "dtype", type(image).__name__))
    _update_json(digest, {"shape": shape, "dtype": dtype})
    if hasattr(image, "tobytes"):
        try:
            pixels = image.tobytes(order="C")
        except TypeError:
            pixels = image.tobytes()
    else:
        try:
            pixels = memoryview(image).tobytes()
        except TypeError as exc:
            raise TypeError("frame image must provide contiguous bytes") from exc
    digest.update(len(pixels).to_bytes(8, "big"))
    digest.update(pixels)


def _raw_validation_values(raw_meta: Any) -> dict[str, Any]:
    if raw_meta is None:
        return {field_name: None for field_name in _RAW_VALIDATION_FIELDS}
    if isinstance(raw_meta, dict):
        return {field_name: raw_meta.get(field_name) for field_name in _RAW_VALIDATION_FIELDS}
    return {
        field_name: getattr(raw_meta, field_name, None)
        for field_name in _RAW_VALIDATION_FIELDS
    }


def _update_json(digest: Any, value: Any) -> None:
    def default(item: Any) -> Any:
        if is_dataclass(item):
            return _json_ready(asdict(item))
        if isinstance(item, os.PathLike):
            return os.fspath(item)
        return str(item)

    encoded = json.dumps(
        _json_ready(value), default=default, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
