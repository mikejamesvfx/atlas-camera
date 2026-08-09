"""Deterministic orchestration for two- and three-photo RAW camera rigs.

The classical feature and geometry stages live in their focused modules.  This
module validates trusted RAW evidence, anchors their relative reconstruction to
photo 1's recovered world-up, resolves translated scale from a measured camera
height, and assembles the public Atlas solve.  Optional vision dependencies are
loaded only when the public solver runs.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import hashlib
import math
from typing import Any, Literal, Sequence

from atlas_camera.core import multiview_features as features
from atlas_camera.core import multiview_geometry as geometry
from atlas_camera.core.confidence import ConfidenceModel
from atlas_camera.core.multiview_types import (
    MultiViewFrame,
    MultiViewSettings,
    OutcomeCode,
    PairMatches,
    PairModelEvidence,
    QUALITY_PROFILES,
    RegistrationDiagnostics,
    RegistrationOutcome,
    registration_fingerprint,
)
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasHorizon,
    AtlasIntrinsics,
    AtlasProjectionScene,
    AtlasSolve,
    AtlasVanishingPoint,
    ProjectionSource,
)
from atlas_camera.core.solver import _face_camera_toward_negative_z
from atlas_camera.core.vanishing_points import (
    VanishingPointDetector,
    horizon_from_vanishing_points,
)


_METADATA_TOLERANCE_MM = 0.05
_GROUND_NORMAL_LIMIT_DEG = 20.0
_GROUND_MIN_INLIERS = 24
_GROUND_MIN_FRACTION = 0.15
_GROUND_SAMPLE_BUDGET = 4_096


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "multi-view solve needs numpy — pip install -e .[vision]"
        ) from exc
    return np


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "multi-view solve needs opencv-python — pip install -e .[vision]"
        ) from exc
    return cv2


def _meta_value(raw_meta: Any, field_name: str) -> Any:
    if raw_meta is None:
        return None
    if isinstance(raw_meta, dict):
        return raw_meta.get(field_name)
    return getattr(raw_meta, field_name, None)


def _normalised_text(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().casefold().split())


def _serial_identity(value: Any) -> str | None:
    """Preserve serial identity while removing safe boundary padding."""
    if value is None:
        return None
    serial = str(value).strip("\x00 \t\r\n\v\f")
    return serial or None


def _developed_dimensions(frame: MultiViewFrame) -> tuple[int | None, int | None]:
    width = _meta_value(frame.raw_meta, "width")
    height = _meta_value(frame.raw_meta, "height")
    shape = tuple(getattr(frame.image, "shape", ()))
    if width is None and len(shape) >= 2:
        width = shape[1]
    if height is None and len(shape) >= 2:
        height = shape[0]
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return None, None


def _check(
    field_name: str,
    frame_index: int,
    anchor_value: Any,
    value: Any,
    *,
    tolerance: float | None = None,
    status: str = "mismatch",
) -> dict[str, Any]:
    result = {
        "field": field_name,
        "frame": frame_index + 1,
        "anchor": anchor_value,
        "value": value,
        "status": status,
    }
    if tolerance is not None:
        result["tolerance"] = tolerance
    return result


def _validation_checks(frames: Sequence[MultiViewFrame]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if len(frames) not in (2, 3):
        return [_check("frame_count", 0, "2 or 3", len(frames))]

    for frame_index, frame in enumerate(frames):
        for field_name in ("focal_length_mm", "sensor_width_mm"):
            value = _meta_value(frame.raw_meta, field_name)
            try:
                valid = math.isfinite(float(value)) and float(value) > 0.0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                checks.append(_check(field_name, frame_index, "positive", value))

        raw_dimensions = (
            _meta_value(frame.raw_meta, "width"),
            _meta_value(frame.raw_meta, "height"),
        )
        shape = tuple(getattr(frame.image, "shape", ()))
        image_dimensions = (
            (int(shape[1]), int(shape[0])) if len(shape) >= 2 else (None, None)
        )
        if all(value is not None for value in raw_dimensions):
            try:
                raw_dimensions = tuple(int(value) for value in raw_dimensions)
            except (TypeError, ValueError):
                pass
            if raw_dimensions != image_dimensions:
                checks.append(_check(
                    "developed_dimensions", frame_index,
                    raw_dimensions, image_dimensions,
                ))

        checks.append(_check(
            "capture_datetime", frame_index, None,
            _meta_value(frame.raw_meta, "capture_datetime"),
            status="diagnostic_only",
        ))

    anchor = frames[0]
    text_fields = (
        "camera_make", "camera_model", "lens_model", "undistort_status",
    )
    numeric_fields = (
        "focal_length_mm", "sensor_width_mm", "sensor_height_mm",
    )
    for frame_index, frame in enumerate(frames[1:], start=1):
        for field_name in text_fields:
            anchor_value = _meta_value(anchor.raw_meta, field_name)
            value = _meta_value(frame.raw_meta, field_name)
            if _normalised_text(anchor_value) != _normalised_text(value):
                checks.append(_check(field_name, frame_index, anchor_value, value))

        for field_name in numeric_fields:
            anchor_value = _meta_value(anchor.raw_meta, field_name)
            value = _meta_value(frame.raw_meta, field_name)
            try:
                matches = (
                    anchor_value is not None
                    and value is not None
                    and math.isfinite(float(anchor_value))
                    and math.isfinite(float(value))
                    and abs(float(anchor_value) - float(value))
                    <= _METADATA_TOLERANCE_MM
                )
            except (TypeError, ValueError):
                matches = False
            if anchor_value is None and value is None:
                matches = True
            if not matches:
                checks.append(_check(
                    field_name, frame_index, anchor_value, value,
                    tolerance=_METADATA_TOLERANCE_MM,
                ))

        anchor_orientation = _meta_value(anchor.raw_meta, "orientation")
        orientation = _meta_value(frame.raw_meta, "orientation")
        if anchor_orientation != orientation:
            checks.append(_check(
                "orientation", frame_index, anchor_orientation, orientation,
            ))

        anchor_dimensions = _developed_dimensions(anchor)
        dimensions = _developed_dimensions(frame)
        if anchor_dimensions != dimensions:
            checks.append(_check(
                "developed_dimensions", frame_index,
                anchor_dimensions, dimensions,
            ))

    for field_name in ("body_serial_number", "lens_serial_number"):
        present = [
            (frame_index, _meta_value(frame.raw_meta, field_name))
            for frame_index, frame in enumerate(frames)
            if _serial_identity(_meta_value(frame.raw_meta, field_name))
        ]
        if len(present) >= 2:
            anchor_index, anchor_value = present[0]
            anchor_identity = _serial_identity(anchor_value)
            for frame_index, value in present[1:]:
                if _serial_identity(value) != anchor_identity:
                    item = _check(field_name, frame_index, anchor_value, value)
                    item["comparison_frame"] = anchor_index + 1
                    checks.append(item)
    return checks


def validate_multiview_frames(
    frames: Sequence[MultiViewFrame], settings: MultiViewSettings,
) -> RegistrationDiagnostics | None:
    """Return structured RAW incompatibilities, or ``None`` when valid.

    Capture timestamps are recorded as diagnostic evidence and deliberately do
    not participate in ordering or compatibility.
    """
    del settings  # Settings are part of the public validation contract.
    checks = _validation_checks(tuple(frames))
    mismatches = [item for item in checks if item["status"] == "mismatch"]
    if not mismatches:
        return None
    fields = ", ".join(dict.fromkeys(item["field"] for item in mismatches))
    summary = (
        "multi-view solve requires two or three RAW frames"
        if any(item["field"] == "frame_count" for item in mismatches)
        else f"RAW metadata mismatch: {fields}"
    )
    return RegistrationDiagnostics(
        "metadata_mismatch", summary, metadata_checks=checks,
    )


def _intrinsics_for_frame(frame: MultiViewFrame) -> AtlasIntrinsics:
    width, height = _developed_dimensions(frame)
    focal = float(_meta_value(frame.raw_meta, "focal_length_mm"))
    sensor_width = float(_meta_value(frame.raw_meta, "sensor_width_mm"))
    sensor_height_value = _meta_value(frame.raw_meta, "sensor_height_mm")
    sensor_height = (
        float(sensor_height_value) if sensor_height_value is not None else None
    )
    # RAW develop applies EXIF orientation to the PIXELS, but the physical
    # sensor millimetres stay in landscape.  For a transposed orientation the
    # developed width spans the sensor's SHORT side — divide accordingly or fx
    # is wrong by the sensor aspect (~1.5x), which collapsed every portrait
    # X-H2 essential-matrix consensus to a handful of grid cells (found live
    # 2026-08-09 on the first real acceptance captures).
    orientation = _meta_value(frame.raw_meta, "orientation")
    transposed = orientation in (5, 6, 7, 8)
    if transposed and sensor_height is not None and sensor_height > 0.0:
        sensor_width, sensor_height = sensor_height, sensor_width
    fx = focal * float(width) / sensor_width
    fy = (
        focal * float(height) / sensor_height
        if sensor_height is not None and sensor_height > 0.0 else fx
    )
    cx, cy = float(width) / 2.0, float(height) / 2.0
    return AtlasIntrinsics(
        image_width=int(width),
        image_height=int(height),
        focal_length_mm=focal,
        sensor_width_mm=sensor_width,
        sensor_height_mm=sensor_height,
        principal_point_px=(cx, cy),
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        lens_model="pinhole",
        distortion={},
    )


def _failure(
    code: OutcomeCode,
    summary: str,
    *,
    metadata_checks: list[dict[str, Any]],
    pair_metrics: list[dict[str, Any]] | None = None,
    scale: dict[str, Any] | None = None,
    overlays: Sequence[Any] = (),
) -> RegistrationOutcome:
    return RegistrationOutcome(
        None,
        RegistrationDiagnostics(
            code,
            summary,
            metadata_checks=metadata_checks,
            pair_metrics=list(pair_metrics or ()),
            scale=dict(scale or {}),
        ),
        tuple(overlays),
    )


def _anchor_orientation(
    frame: MultiViewFrame,
    intrinsics: AtlasIntrinsics,
    seed: int,
) -> tuple[Any, AtlasHorizon, list[AtlasVanishingPoint]] | None:
    np = _require_numpy()
    # The detector receives only photographed anchor pixels.  _gray_uint8 is
    # the reviewed deterministic display conversion used by feature evidence.
    detector_image = features._gray_uint8(frame.image)
    result = VanishingPointDetector.detect_vanishing_points(
        detector_image, random_seed=seed,
    )
    first, second = result.get("vp1"), result.get("vp2")
    if first is None or second is None:
        return None
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != (2,) or second.shape != (2,):
        return None
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return None
    ray_a = np.array((
        (first[0] - intrinsics.cx_px) / intrinsics.fx_px,
        -(first[1] - intrinsics.cy_px) / intrinsics.fy_px,
        -1.0,
    ))
    ray_b = np.array((
        (second[0] - intrinsics.cx_px) / intrinsics.fx_px,
        -(second[1] - intrinsics.cy_px) / intrinsics.fy_px,
        -1.0,
    ))
    cosine = abs(float(np.dot(ray_a, ray_b))) / float(
        np.linalg.norm(ray_a) * np.linalg.norm(ray_b)
    )
    # Detector group labels are approximate; reject pairs more than 30 degrees
    # from orthogonal while tolerating real photographed line noise.
    if not math.isfinite(cosine) or cosine > 0.5:
        return None
    try:
        direction_a = np.asarray(ray_a, dtype=np.float64)
        direction_b = np.asarray(ray_b, dtype=np.float64)
        direction_a /= np.linalg.norm(direction_a)
        direction_b /= np.linalg.norm(direction_b)
        # vp1/vp2 are the detector's two horizontal architectural groups.  In
        # Atlas camera coordinates (+X right, +Y up, camera looks down -Z),
        # their cross product is therefore recovered world-up in CAMERA space.
        # Matrix columns below are world axes expressed in camera coordinates,
        # i.e. world-to-camera; transpose once to obtain the schema's
        # camera-to-world transform.
        up = np.cross(direction_b, direction_a)
        up /= np.linalg.norm(up)
        if up[1] < 0.0:
            up = -up
        # EXIF orientation is already applied to the developed pixels, so
        # recovered world-up must sit near image-up.  Beyond 45 degrees the
        # detector has misclassified the architectural groups (found live
        # 2026-08-09: a street facade passed orthogonality yet yielded an up
        # 72.5 degrees off vertical, silently poisoning the whole rig frame
        # and every downstream ground test).  Fail the VP anchor instead so
        # the caller's up hint — when provided — can take over.
        if float(up[1]) < math.cos(math.radians(45.0)):
            return None
        forward = np.cross(direction_a, up)
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        world_to_camera = np.column_stack((right, up, forward))
        anchor_camera_to_world = world_to_camera.T
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    vanishing_points = VanishingPointDetector.to_schema_vanishing_points(result)
    if len(vanishing_points) < 2:
        return None
    horizon = horizon_from_vanishing_points(
        vanishing_points[0], vanishing_points[1],
        image_width=intrinsics.image_width,
    )
    return anchor_camera_to_world, horizon, vanishing_points


def _anchor_orientation_from_up(
    intrinsics: AtlasIntrinsics,
    up_hint: Sequence[float],
) -> tuple[Any, AtlasHorizon, list[AtlasVanishingPoint]] | None:
    """Anchor basis from a caller-supplied world-up in camera coordinates.

    Fallback for scenes with no two orthogonal architectural vanishing points
    (parks, organic subjects).  Builds the same world-to-camera column basis as
    the VP path: world up is the hint, and world -Z faces the camera's view
    direction, matching the recovered-camera-faces-negative-Z convention.  The
    horizon is the image line of rays orthogonal to up.  No vanishing points
    are fabricated.
    """
    np = _require_numpy()
    up = np.asarray(tuple(float(v) for v in up_hint), dtype=np.float64).reshape(-1)
    if up.shape != (3,) or not np.all(np.isfinite(up)):
        return None
    norm = float(np.linalg.norm(up))
    if norm < 1e-9:
        return None
    up = up / norm
    if up[1] < 0.0:
        up = -up
    view = np.array((0.0, 0.0, -1.0))
    world_z = -(view - up * float(view @ up))
    z_norm = float(np.linalg.norm(world_z))
    if z_norm < 1e-9:
        # Camera looking straight along gravity; horizontal facing is undefined.
        return None
    world_z /= z_norm
    world_x = np.cross(up, world_z)
    world_x /= np.linalg.norm(world_x)
    world_to_camera = np.column_stack((world_x, up, world_z))
    anchor_camera_to_world = world_to_camera.T

    # Horizon: pixels whose rays are orthogonal to up.  With the Atlas ray
    # ((x-cx)/fx, -(y-cy)/fy, -1), ray·up = 0 expands to a*x + b*y + c = 0:
    a = float(up[0]) / intrinsics.fx_px
    b = -float(up[1]) / intrinsics.fy_px
    c = (
        -float(up[0]) * intrinsics.cx_px / intrinsics.fx_px
        + float(up[1]) * intrinsics.cy_px / intrinsics.fy_px
        - float(up[2])
    )
    endpoints = None
    if abs(b) > 1e-12:
        y_at = lambda x: (-c - a * x) / b  # noqa: E731
        endpoints = (
            (0.0, float(y_at(0.0))),
            (float(intrinsics.image_width), float(y_at(float(intrinsics.image_width)))),
        )
    horizon = AtlasHorizon(
        line_coefficients=(a, b, c),
        endpoints_px=endpoints,
        confidence=0.5,
    )
    return anchor_camera_to_world, horizon, []


def _required_pairs(frame_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(frame_count), 2))


def _grid_cells(points: Any, mask: Any, width: int, height: int) -> int:
    np = _require_numpy()
    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != len(points) or not np.any(mask):
        return 0
    cells = np.floor(
        points[mask] / np.array((width, height), dtype=np.float64) * 4.0
    ).astype(np.int64)
    cells = np.clip(cells, 0, 3)
    return len({(int(item[0]), int(item[1])) for item in cells})


def _consensus_diagnostics(
    matches: PairMatches,
    evidence: PairModelEvidence,
    intrinsics: AtlasIntrinsics,
) -> tuple[int, list[float] | None, Any]:
    np = _require_numpy()
    candidates: list[tuple[int, int, Any]] = []
    if evidence.essential_matrix is not None:
        candidates.append((
            _grid_cells(
                matches.points_a, evidence.essential_inliers,
                intrinsics.image_width, intrinsics.image_height,
            ),
            int(evidence.essential_inlier_count),
            np.asarray(evidence.essential_inliers, dtype=bool),
        ))
    if evidence.homography is not None:
        candidates.append((
            _grid_cells(
                matches.points_a, evidence.homography_inliers,
                intrinsics.image_width, intrinsics.image_height,
            ),
            int(evidence.homography_inlier_count),
            np.asarray(evidence.homography_inliers, dtype=bool),
        ))
    if not candidates:
        return 0, None, np.zeros(len(matches.points_a), dtype=bool)
    max_cells = max(item[0] for item in candidates)
    _, _, best_mask = min(candidates, key=lambda item: (-item[1], -item[0]))
    accepted = np.asarray(matches.points_a, dtype=np.float64)[best_mask]
    bounding_box = None
    if len(accepted):
        bounding_box = [
            float(np.min(accepted[:, 0])),
            float(np.min(accepted[:, 1])),
            float(np.max(accepted[:, 0])),
            float(np.max(accepted[:, 1])),
        ]
    return max_cells, bounding_box, best_mask


def _pair_metric(matches: PairMatches) -> dict[str, Any]:
    return {
        "frame_a": int(matches.frame_a) + 1,
        "frame_b": int(matches.frame_b) + 1,
        "mutual_matches": int(len(matches.indices)),
        "raw_occupied_grid_cells": int(matches.occupied_grid_cells),
    }


def _select_mode(
    evidence: Sequence[PairModelEvidence], settings: MultiViewSettings,
) -> tuple[Literal["translated", "rotation_only"], tuple[str, ...]]:
    """Select one rig-wide mode plus each pair's pose source.

    Essential-translated and planar-translated pairs agree on TRANSLATED
    capture — a rig may mix them (e.g. a facade pair beside a corner pair).
    Only translated-versus-rotation disagreement is ambiguous.
    """
    profile = QUALITY_PROFILES[settings.match_quality]
    pair_modes = tuple(
        geometry.select_capture_mode(item, settings.capture_mode, profile)
        for item in evidence
    )
    kinds = {
        "translated" if value in ("translated", "translated_planar") else value
        for value in pair_modes
    }
    if len(kinds) != 1:
        raise geometry.MotionModelError(
            "ambiguous_motion_model",
            "required pairs disagree on translated versus rotation-only capture",
        )
    return kinds.pop(), pair_modes


def _compose_anchor_rig(refined: geometry.RefinedRig, anchor_basis: Any) -> tuple[Any, Any, Any]:
    np = _require_numpy()
    basis = np.asarray(anchor_basis, dtype=np.float64)
    opencv_to_atlas = np.diag((1.0, -1.0, -1.0))
    camera_to_world: list[Any] = []
    positions: list[Any] = []
    for rotation, translation in zip(refined.rotations, refined.translations):
        world_to_camera_cv = np.asarray(rotation, dtype=np.float64)
        translation_cv = np.asarray(translation, dtype=np.float64)
        world_to_camera_local = (
            opencv_to_atlas @ world_to_camera_cv @ opencv_to_atlas
        )
        translation_local = opencv_to_atlas @ translation_cv
        local_position = -world_to_camera_local.T @ translation_local
        camera_to_world.append(basis @ world_to_camera_local.T)
        positions.append(basis @ local_position)
    landmarks = (
        np.asarray(refined.landmarks, dtype=np.float64).reshape((-1, 3))
        @ opencv_to_atlas.T
        @ basis.T
    )
    return (
        tuple(np.ascontiguousarray(value, dtype=np.float64) for value in camera_to_world),
        np.asarray(positions, dtype=np.float64),
        landmarks,
    )


def _face_complete_rig(
    camera_to_world: Sequence[Any], positions: Any, landmarks: Any,
) -> tuple[tuple[Any, ...], Any, Any]:
    """Apply the reviewed free-yaw convention once to every rig quantity."""
    np = _require_numpy()
    primary = np.asarray(camera_to_world[0], dtype=np.float64)
    faced_primary = _face_camera_toward_negative_z(primary, np)
    world_rotation = faced_primary @ primary.T
    return (
        tuple(
            world_rotation @ np.asarray(value, dtype=np.float64)
            for value in camera_to_world
        ),
        np.asarray(positions, dtype=np.float64) @ world_rotation.T,
        np.asarray(landmarks, dtype=np.float64) @ world_rotation.T,
    )


def _ground_candidates(
    landmarks: Any,
    accepted_track_ids: Sequence[int],
    tracks: Sequence[geometry.FeatureTrack],
    horizon: AtlasHorizon,
) -> tuple[Any, tuple[int, ...]]:
    """Landmarks below the anchor camera in the recovered world frame.

    Ground candidacy is a WORLD test (Y < 0: below the optical centre, since
    the anchor camera sits at the origin of the Y-up anchor frame), not an
    image-space horizon test.  The VP-derived horizon line proved fragile on
    real captures — a portrait street facade put it at y=9577 in a 3876-tall
    frame, silently discarding every road landmark and failing the solve as
    scale_unavailable (found live 2026-08-09).  The plane fit downstream still
    enforces the up-facing normal, inlier count, and support fraction.
    """
    np = _require_numpy()
    del tracks, horizon  # Retained in the signature for call-site stability.
    del accepted_track_ids
    points: list[Any] = []
    source_indices: list[int] = []
    for landmark_index, point in enumerate(np.asarray(landmarks, dtype=np.float64)):
        if not np.all(np.isfinite(point)):
            continue
        if float(point[1]) < 0.0:
            points.append(point)
            source_indices.append(landmark_index)
    return (
        np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        tuple(source_indices),
    )


def _ground_sample_schedule(
    count: int, fingerprint: str, seed: int,
) -> tuple[tuple[int, int, int], ...]:
    np = _require_numpy()
    total = math.comb(count, 3) if count >= 3 else 0
    target = min(total, _GROUND_SAMPLE_BUDGET)
    if target == total:
        return tuple(combinations(range(count), 3))
    material = f"{fingerprint}:ground-plane:{seed}".encode("ascii")
    private_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(private_seed))
    samples: set[tuple[int, int, int]] = set()
    while len(samples) < target:
        sample = tuple(sorted(int(value) for value in rng.choice(count, 3, replace=False)))
        samples.add(sample)
    return tuple(sorted(samples))


def _fit_ground_plane(
    points: Any, fingerprint: str, seed: int,
) -> dict[str, Any] | None:
    np = _require_numpy()
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) < _GROUND_MIN_INLIERS:
        return None
    centre = np.median(points, axis=0)
    spread = float(np.median(np.linalg.norm(points - centre, axis=1)))
    # 5% of the candidate spread, not 1%: at street scale that is roughly an
    # 8 cm plane tolerance, matching real road crown/texture plus the
    # triangulation noise of far-from-plane points.  1% (~1.6 cm) rejected
    # every genuine road inlier on the first real X-H2 street set (15/109
    # against a 24 minimum; found live 2026-08-09).
    threshold = max(1.0e-8, 0.05 * spread)
    up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    cosine_limit = math.cos(math.radians(_GROUND_NORMAL_LIMIT_DEG))
    best: tuple[Any, ...] | None = None
    for sample in _ground_sample_schedule(len(points), fingerprint, seed):
        first, second, third = (points[index] for index in sample)
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            continue
        normal /= norm
        if float(np.dot(normal, up)) < 0.0:
            normal = -normal
        if float(np.dot(normal, up)) < cosine_limit:
            continue
        offset = -float(np.dot(normal, first))
        residuals = np.abs(points @ normal + offset)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residuals[inliers])) if count else float("inf")
        score = (-count, median, sample)
        if best is None or score < best[0]:
            best = (score, normal.copy(), offset, inliers.copy(), count)
    if best is None:
        return None
    _, normal, offset, inliers, count = best
    fraction = count / len(points)
    anchor_distance = float(offset)  # Anchor is the local reconstruction origin.
    if (
        count < _GROUND_MIN_INLIERS
        or fraction < _GROUND_MIN_FRACTION
        or not math.isfinite(anchor_distance)
        or anchor_distance <= 0.0
    ):
        return None
    return {
        "normal": normal,
        "offset": offset,
        "inliers": inliers,
        "inlier_count": count,
        "valid_landmark_count": len(points),
        "inlier_fraction": fraction,
        "anchor_plane_distance": anchor_distance,
        "threshold": threshold,
    }


def _rotate_vector_to_up(normal: Any) -> Any:
    np = _require_numpy()
    source = np.asarray(normal, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(source, target))
    if sine <= 1.0e-15:
        return np.eye(3, dtype=np.float64)
    skew = np.array((
        (0.0, -cross[2], cross[1]),
        (cross[2], 0.0, -cross[0]),
        (-cross[1], cross[0], 0.0),
    ), dtype=np.float64)
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (
        (1.0 - cosine) / (sine * sine)
    )


def _apply_metric_ground_scale(
    camera_to_world: Sequence[Any],
    positions: Any,
    landmarks: Any,
    ground_points: Any,
    fit: dict[str, Any],
    camera_height_m: float,
) -> tuple[tuple[Any, ...], Any, Any, dict[str, Any]]:
    np = _require_numpy()
    ground_rotation = _rotate_vector_to_up(fit["normal"])
    rotated_camera_to_world = tuple(
        ground_rotation @ np.asarray(value, dtype=np.float64)
        for value in camera_to_world
    )
    rotated_positions = np.asarray(positions, dtype=np.float64) @ ground_rotation.T
    rotated_landmarks = np.asarray(landmarks, dtype=np.float64) @ ground_rotation.T
    rotated_ground = np.asarray(ground_points, dtype=np.float64) @ ground_rotation.T
    plane_y = float(np.median(rotated_ground[fit["inliers"], 1]))
    scale = float(camera_height_m) / float(fit["anchor_plane_distance"])
    scaled_plane_y = scale * plane_y
    offset = np.array((0.0, scaled_plane_y, 0.0), dtype=np.float64)
    scaled_positions = scale * rotated_positions - offset
    scaled_landmarks = scale * rotated_landmarks - offset
    scaled_positions[0, 1] = float(camera_height_m)
    scale_info = {
        "source": "measured_camera_height",
        "camera_height_m": float(camera_height_m),
        "scale_factor": scale,
        "anchor_plane_distance": float(fit["anchor_plane_distance"]),
        "plane_y_before_translation": plane_y,
        "inlier_count": int(fit["inlier_count"]),
        "valid_landmark_count": int(fit["valid_landmark_count"]),
        "inlier_fraction": float(fit["inlier_fraction"]),
        "normal_before_alignment": [float(value) for value in fit["normal"]],
        "normal_limit_deg": _GROUND_NORMAL_LIMIT_DEG,
        "residual_threshold": float(fit["threshold"]),
    }
    return rotated_camera_to_world, scaled_positions, scaled_landmarks, scale_info


def _tuple3(value: Any) -> tuple[float, float, float]:
    return tuple(0.0 if abs(float(item)) < 1.0e-15 else float(item) for item in value)  # type: ignore[return-value]


def _tuple_matrix3(value: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(_tuple3(row) for row in value)


def _extrinsics(camera_to_world: Any, position: Any) -> AtlasExtrinsics:
    np = _require_numpy()
    rotation = np.asarray(camera_to_world, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    world_to_camera = rotation.T
    translation = -world_to_camera @ position
    world_matrix = tuple(
        _tuple3(rotation[row]) + (_tuple3(position)[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    view_matrix = tuple(
        _tuple3(world_to_camera[row]) + (_tuple3(translation)[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    return AtlasExtrinsics(
        camera_position=_tuple3(position),
        camera_rotation_matrix=_tuple_matrix3(rotation),  # type: ignore[arg-type]
        camera_world_matrix=world_matrix,  # type: ignore[arg-type]
        camera_view_matrix=view_matrix,  # type: ignore[arg-type]
        coordinate_system="right_handed",
        up_axis="Y",
        projection_convention=(
            "Deterministic photographed multi-view pinhole camera, image origin top-left."
        ),
    )


def _confidence(
    mode: Literal["translated", "rotation_only"],
    refined: geometry.RefinedRig,
    profile_threshold: float,
) -> float:
    residual = float(refined.reprojection_rmse_px)
    if not math.isfinite(residual):
        return 0.0
    residual_score = max(0.0, min(1.0, 1.0 - residual / (4.0 * profile_threshold)))
    mode_ceiling = 0.95 if mode == "translated" else 0.85
    return float(round(mode_ceiling * residual_score, 12))


def _camera(
    intrinsics: AtlasIntrinsics,
    extrinsics: AtlasExtrinsics,
    name: str,
    seed: int,
    confidence: float,
    mode: Literal["translated", "rotation_only"],
) -> AtlasCamera:
    return AtlasCamera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        name=name,
        confidence=ConfidenceModel.for_latent_camera(
            global_score=confidence,
            overrides={
                "horizon": confidence,
                "vp1": confidence,
                "vp2": confidence,
                "focal": 1.0,
                "extrinsics": confidence,
                "sensor": 1.0,
                "scale": 1.0 if mode == "translated" else 0.0,
            },
        ),
        focal_length_inferred=False,
        seed=seed,
        notes=(
            [] if mode == "translated"
            else ["Rotation-only registration: all cameras share one optical centre."]
        ),
    )


def _landmark_payload(points: Any, track_ids: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": int(track_id),
            "position": _tuple3(point),
            "evidence_type": "photographed",
        }
        for track_id, point in zip(track_ids, points)
    ]


def solve_multiview(
    frames: Sequence[MultiViewFrame], settings: MultiViewSettings,
) -> RegistrationOutcome:
    """Recover one anchored Atlas rig from two or three ordered RAW photos."""
    ordered_frames = tuple(frames)
    metadata_checks = _validation_checks(ordered_frames)
    validation = validate_multiview_frames(ordered_frames, settings)
    if validation is not None:
        return RegistrationOutcome(None, validation)

    intrinsics = tuple(_intrinsics_for_frame(frame) for frame in ordered_frames)
    anchor_warnings: list[str] = []
    anchored = _anchor_orientation(ordered_frames[0], intrinsics[0], settings.seed)
    if anchored is None and settings.anchor_up_hint is not None:
        anchored = _anchor_orientation_from_up(intrinsics[0], settings.anchor_up_hint)
        if anchored is not None:
            source = settings.anchor_up_hint_source or "caller-supplied up hint"
            anchor_warnings.append(
                f"anchor world-up came from {source}: photo 1 has no two "
                "orthogonal architectural vanishing points. Reproducibility "
                "follows the hint's provider, not the VP detector."
            )
    if anchored is None:
        return _failure(
            "degenerate_geometry",
            "Photo 1 needs two clear orthogonal architectural vanishing points; "
            "use clearer architectural lines or artist constraints. Sparse "
            "correspondence cannot safely guess world-up.",
            metadata_checks=metadata_checks,
        )
    anchor_basis, horizon, vanishing_points = anchored

    profile = QUALITY_PROFILES[settings.match_quality]
    fingerprint = registration_fingerprint(ordered_frames, settings)
    extracted = tuple(
        features.extract_features(frame.image, profile) for frame in ordered_frames
    )
    pair_matches: list[PairMatches] = []
    pair_metrics: list[dict[str, Any]] = []
    overlays: list[Any] = []
    for frame_a, frame_b in _required_pairs(len(ordered_frames)):
        matches = features.match_features(
            extracted[frame_a], extracted[frame_b], profile, frame_a, frame_b,
        )
        pair_matches.append(matches)
        pair_metrics.append(_pair_metric(matches))
        overlays.append(features.render_match_overlay(
            ordered_frames[frame_a].image,
            ordered_frames[frame_b].image,
            matches,
        ))
        if len(matches.indices) < profile.min_inliers:
            return _failure(
                "insufficient_overlap",
                f"photos {frame_a + 1}-{frame_b + 1} have "
                f"{len(matches.indices)} mutual matches; at least "
                f"{profile.min_inliers} are required",
                metadata_checks=metadata_checks,
                pair_metrics=pair_metrics,
                overlays=overlays,
            )

    pair_evidence: list[PairModelEvidence] = []
    try:
        for pair_index, matches in enumerate(pair_matches):
            evidence = geometry.fit_pair_models(
                matches,
                intrinsics[matches.frame_a],
                intrinsics[matches.frame_b],
                settings,
                fingerprint,
            )
            pair_evidence.append(evidence)
            cells, bounding_box, best_mask = _consensus_diagnostics(
                matches, evidence, intrinsics[matches.frame_a],
            )
            metric = pair_metrics[pair_index]
            metric.update({
                "essential_inliers": int(evidence.essential_inlier_count),
                "homography_inliers": int(evidence.homography_inlier_count),
                "essential_occupied_grid_cells": int(
                    evidence.essential_occupied_grid_cells
                ),
                "max_consensus_grid_cells": int(cells),
                "median_essential_error_px": float(
                    evidence.median_essential_error_px
                ),
                "median_homography_error_px": float(
                    evidence.median_homography_error_px
                ),
            })
            # A dominant homography consensus is the planar-candidate
            # signature: judge it by the planar gate's relaxed spatial bar,
            # or the early contamination check kills facades before the
            # planar-translated model is ever consulted (found live
            # 2026-08-09: full-res sh001 spans 3 cells, half-size spans 4).
            planar_candidate = (
                evidence.homography is not None
                and evidence.homography_inlier_count * 2 >= len(matches.indices)
            )
            min_cells = (
                max(2, profile.min_grid_cells - 2) if planar_candidate
                else profile.min_grid_cells
            )
            if (
                len(matches.indices) >= 2 * profile.min_inliers
                and cells < min_cells
            ):
                metric["consensus_bounding_box_px"] = bounding_box
                overlays[pair_index] = features.render_match_overlay(
                    ordered_frames[matches.frame_a].image,
                    ordered_frames[matches.frame_b].image,
                    matches,
                    best_mask,
                )
                return _failure(
                    "dynamic_scene_contamination",
                    f"photos {matches.frame_a + 1}-{matches.frame_b + 1} have "
                    "many raw matches but every geometric consensus is confined "
                    "to too few 4x4 grid cells",
                    metadata_checks=metadata_checks,
                    pair_metrics=pair_metrics,
                    overlays=overlays,
                )

        mode, pair_modes = _select_mode(pair_evidence, settings)
        for pair_index, (matches, evidence) in enumerate(zip(
            pair_matches, pair_evidence,
        )):
            pair_mode = pair_modes[pair_index]
            selected_inliers = (
                evidence.essential_inliers
                if pair_mode == "translated" else evidence.homography_inliers
            )
            pair_metrics[pair_index]["pose_source"] = {
                "translated": "essential",
                "translated_planar": "homography_decomposition",
                "rotation_only": "rotation_homography",
            }[pair_mode]
            overlays[pair_index] = features.render_match_overlay(
                ordered_frames[matches.frame_a].image,
                ordered_frames[matches.frame_b].image,
                matches,
                selected_inliers,
            )
        # Planar pairs feed the rig their homography-decomposed pose through
        # the same evidence fields the essential path uses.
        pair_evidence = [
            replace(
                evidence,
                relative_rotation=evidence.planar_rotation,
                translation_direction=evidence.planar_translation_direction,
            ) if pair_modes[pair_index] == "translated_planar" else evidence
            for pair_index, evidence in enumerate(pair_evidence)
        ]
        tracks = geometry.build_tracks(pair_matches, len(ordered_frames))
        initial = geometry.initialise_rig(pair_evidence, mode)
        refined = geometry.refine_rig(
            initial, tracks, intrinsics, mode, profile,
        )
    except geometry.MotionModelError as exc:
        return _failure(
            exc.outcome_code,
            str(exc),
            metadata_checks=metadata_checks,
            pair_metrics=pair_metrics,
            overlays=overlays,
        )
    except (ValueError, ArithmeticError) as exc:
        return _failure(
            "degenerate_geometry",
            f"deterministic rig composition failed: {exc}",
            metadata_checks=metadata_checks,
            pair_metrics=pair_metrics,
            overlays=overlays,
        )

    camera_to_world, positions, world_landmarks = _compose_anchor_rig(
        refined, anchor_basis,
    )
    scale_source = "not_applicable_rotation_only"
    scale_info: dict[str, Any] = {
        "source": scale_source,
        "camera_height_m": None,
    }
    output_track_ids: Sequence[int] = ()
    if mode == "translated":
        if (
            not math.isfinite(float(settings.camera_height_m))
            or settings.camera_height_m <= 0.0
        ):
            return _failure(
                "scale_unavailable",
                "Translated registration needs a positive measured photo-1 "
                "lens-centre height",
                metadata_checks=metadata_checks,
                pair_metrics=pair_metrics,
                scale={"camera_height_m": float(settings.camera_height_m)},
                overlays=overlays,
            )
        ground_points, _ = _ground_candidates(
            world_landmarks, refined.accepted_track_ids, tracks, horizon,
        )
        fit = _fit_ground_plane(ground_points, fingerprint, settings.seed)
        if fit is None:
            return _failure(
                "scale_unavailable",
                "Translated registration has no valid ground plane with a "
                "normal within 20 degrees of recovered up, at least 24 inliers, "
                "15% support, and positive anchor distance",
                metadata_checks=metadata_checks,
                pair_metrics=pair_metrics,
                scale={
                    "camera_height_m": float(settings.camera_height_m),
                    "valid_ground_landmarks": int(len(ground_points)),
                },
                overlays=overlays,
            )
        camera_to_world, positions, world_landmarks, scale_info = (
            _apply_metric_ground_scale(
                camera_to_world,
                positions,
                world_landmarks,
                ground_points,
                fit,
                settings.camera_height_m,
            )
        )
        scale_source = "measured_camera_height"
        output_track_ids = refined.accepted_track_ids
    else:
        positions = _require_numpy().zeros_like(positions)
        world_landmarks = _require_numpy().empty((0, 3), dtype=_require_numpy().float64)

    camera_to_world, positions, world_landmarks = _face_complete_rig(
        camera_to_world, positions, world_landmarks,
    )

    confidence = _confidence(
        mode, refined, profile.reprojection_threshold_px,
    )
    cameras = tuple(
        _camera(
            intrinsics[index],
            _extrinsics(camera_to_world[index], positions[index]),
            ordered_frames[index].label or f"photo_{index + 1}",
            settings.seed,
            confidence,
            mode,
        )
        for index in range(len(ordered_frames))
    )
    landmark_payload = _landmark_payload(world_landmarks, output_track_ids)
    anchor_identity = ordered_frames[0].label or "photo_1"
    projection_sources: list[ProjectionSource] = []
    for index in range(1, len(ordered_frames)):
        frame = ordered_frames[index]
        projection_sources.append(ProjectionSource(
            camera=cameras[index],
            name=frame.label or f"photo_{index + 1}",
            image_b64=(
                frame.plate_ref.preview_b64 if frame.plate_ref is not None else None
            ),
            plate_ref=frame.plate_ref,
            proxy_geometry=[],
            metadata={
                "evidence_type": "photographed",
                "registration_method": "deterministic_sift_calibrated_multiview",
                "capture_mode": mode,
                "selected_capture_mode": mode,
                "reprojection_error_px": float(refined.reprojection_rmse_px),
                "confidence": confidence,
                "scale_source": scale_source,
                "scale_provenance": scale_source,
                "source_order": index + 1,
                "frame_index": index,
                "anchor_identity": anchor_identity,
                "anchor_frame_index": 0,
                "registration_fingerprint": fingerprint,
            },
        ))

    cv2 = _require_cv2()
    np = _require_numpy()
    projection_scene = AtlasProjectionScene(
        landmarks=list(landmark_payload),
        debug_metadata={
            "registration_fingerprint": fingerprint,
            "evidence_type": "photographed",
        },
    )
    debug_metadata = {
        "solve_mode": "deterministic_raw_multiview",
        "capture_mode": mode,
        "scale_source": scale_source,
        "scale": scale_info,
        "registration_fingerprint": fingerprint,
        "registration_method": "deterministic_sift_calibrated_multiview",
        "anchor_frame_index": 0,
        "anchor_identity": anchor_identity,
        "frame_count": len(ordered_frames),
        "photographed_source_count": len(ordered_frames),
        "generated_inputs_used": False,
        "trusted_raw_intrinsics_locked": True,
        "reprojection_rmse_px": float(refined.reprojection_rmse_px),
        "accepted_track_count": len(refined.accepted_track_ids),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "seed": int(settings.seed),
    }
    image_path = None
    if ordered_frames[0].plate_ref is not None:
        image_path = ordered_frames[0].plate_ref.image_path
    if image_path is None:
        image_path = _meta_value(ordered_frames[0].raw_meta, "source_path") or None
    solve = AtlasSolve(
        camera=cameras[0],
        image_path=str(image_path) if image_path else None,
        image_width=intrinsics[0].image_width,
        image_height=intrinsics[0].image_height,
        vanishing_points=vanishing_points,
        horizon_line=horizon,
        confidence=confidence,
        source_method="deterministic_raw_multiview",
        known_intrinsics_used=True,
        projection_scene=projection_scene,
        projection_sources=projection_sources,
        landmarks=landmark_payload,
        debug_metadata=debug_metadata,
        source_plate=ordered_frames[0].plate_ref,
    )
    diagnostics = RegistrationDiagnostics(
        mode,
        f"registered {len(ordered_frames)} photographed RAW frames in {mode} mode",
        selected_mode=mode,
        metadata_checks=metadata_checks,
        pair_metrics=pair_metrics,
        camera_metrics=[
            {
                "frame": index + 1,
                "position": list(camera.extrinsics.camera_position),
                "reprojection_rmse_px": float(refined.reprojection_rmse_px),
                "confidence": confidence,
            }
            for index, camera in enumerate(cameras)
        ],
        scale=scale_info,
        warnings=anchor_warnings + (
            ["Rotation-only capture recovered no translation geometry or metric scale."]
            if mode == "rotation_only" else []
        ),
    )
    return RegistrationOutcome(solve, diagnostics, tuple(overlays))


__all__ = ["solve_multiview", "validate_multiview_frames"]
