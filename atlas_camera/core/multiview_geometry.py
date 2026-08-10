"""Deterministic calibrated two-view motion-model fitting.

NumPy is optional at package-import time and is loaded only when geometry is
requested.  Candidate schedules use a private PCG64 generator derived from the
registration fingerprint; no OpenCV estimator or process-global RNG is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import hashlib
import math
from typing import Any, Literal

from atlas_camera.core.multiview_types import (
    CaptureMode,
    MultiViewSettings,
    OutcomeCode,
    PairMatches,
    PairModelEvidence,
    QualityProfile,
    QUALITY_PROFILES,
)
from atlas_camera.core.schema import AtlasIntrinsics


_ESSENTIAL_SAMPLES = 2_048
_HOMOGRAPHY_SAMPLES = 1_024
_POSITIVE_DEPTH_FRACTION = 0.75
_MINIMAL_RANK_RELATIVE_TOLERANCE = 1.0e-12
_REFINEMENT_DAMPING = (
    1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5,
    1.0e-5, 3.0e-6, 1.0e-6, 1.0e-6,
)


@dataclass(frozen=True)
class FeatureObservation:
    frame_index: int
    feature_index: int
    point_xy: tuple[float, float]


@dataclass(frozen=True)
class FeatureTrack:
    track_id: int
    observations: tuple[FeatureObservation, ...]


@dataclass(frozen=True)
class CameraRig:
    rotations: tuple[Any, ...]
    translations: tuple[Any, ...]
    landmarks: Any
    reprojection_rmse_px: float
    _pair_evidence: tuple[PairModelEvidence, ...] = field(
        default=(), repr=False, compare=False, kw_only=True,
    )
    _tracks: tuple[FeatureTrack, ...] = field(
        default=(), repr=False, compare=False, kw_only=True,
    )
    _intrinsics: tuple[AtlasIntrinsics, ...] = field(
        default=(), repr=False, compare=False, kw_only=True,
    )


@dataclass(frozen=True)
class ClosureMetrics:
    rotation_error_deg: float
    translation_direction_error_deg: float
    median_reprojection_px: float


@dataclass(frozen=True)
class RefinedRig(CameraRig):
    accepted_track_ids: tuple[int, ...]
    closure: ClosureMetrics


class MotionModelError(ValueError):
    """Geometric failure with a stable public outcome code."""

    def __init__(self, outcome_code: OutcomeCode, summary: str) -> None:
        self.outcome_code = outcome_code
        super().__init__(f"{outcome_code}: {summary}")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "multi-view geometry needs numpy — pip install -e .[vision]"
        ) from exc
    return np


def _intrinsic_matrix(intrinsics: AtlasIntrinsics) -> Any:
    np = _require_numpy()
    values = (
        intrinsics.fx_px, intrinsics.fy_px,
        intrinsics.cx_px, intrinsics.cy_px,
    )
    if any(value is None for value in values):
        raise MotionModelError(
            "metadata_mismatch",
            "trusted pixel intrinsics fx_px, fy_px, cx_px, and cy_px are required",
        )
    fx, fy, cx, cy = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
        raise MotionModelError(
            "metadata_mismatch", "trusted pixel intrinsics must be finite and positive",
        )
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _canonical_model(model: Any) -> Any:
    np = _require_numpy()
    result = np.ascontiguousarray(model, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("degenerate model")
    result = result / norm
    nonzero = np.flatnonzero(np.abs(result.reshape(-1)) > 1.0e-15)
    if len(nonzero) and result.reshape(-1)[int(nonzero[0])] < 0.0:
        result = -result
    return np.ascontiguousarray(result, dtype=np.float64)


def _sample_schedule(count: int, sample_size: int, budget: int,
                     fingerprint: str, seed: int, model_name: str) -> tuple[tuple[int, ...], ...]:
    """Precompute a deterministic schedule of distinct sorted samples."""
    np = _require_numpy()
    if count < sample_size:
        return ()
    combination_count = math.comb(count, sample_size)
    target = min(budget, combination_count)
    if combination_count <= budget:
        return tuple(combinations(range(count), sample_size))
    digest = hashlib.sha256(
        f"{fingerprint}{seed}{model_name}".encode("utf-8")
    ).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))
    samples: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(samples) < target:
        sample = tuple(sorted(
            int(value) for value in rng.choice(count, size=sample_size, replace=False)
        ))
        if sample not in seen:
            seen.add(sample)
            samples.append(sample)
    return tuple(samples)


def _model_sample_schedules(count: int, settings: MultiViewSettings,
                            fingerprint: str) -> tuple[
                                tuple[tuple[int, ...], ...],
                                tuple[tuple[int, ...], ...],
                            ]:
    """Build the production essential/homography schedules and exact budgets."""
    return (
        _sample_schedule(
            count, 8, _ESSENTIAL_SAMPLES,
            fingerprint, settings.seed, "essential",
        ),
        _sample_schedule(
            count, 4, _HOMOGRAPHY_SAMPLES,
            fingerprint, settings.seed, "homography",
        ),
    )


def _homogeneous(points_xy: Any) -> Any:
    np = _require_numpy()
    points = np.asarray(points_xy, dtype=np.float64)
    return np.column_stack((points, np.ones(len(points), dtype=np.float64)))


def _hartley_normalize(points_xy: Any) -> tuple[Any, Any]:
    np = _require_numpy()
    points = np.asarray(points_xy, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    centred = points - centroid
    distances = np.linalg.norm(centred, axis=1)
    mean_distance = float(np.mean(distances))
    if not math.isfinite(mean_distance) or mean_distance <= 1.0e-15:
        raise ValueError("degenerate point normalization")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    normalized_h = (transform @ _homogeneous(points).T).T
    return normalized_h[:, :2], transform


def _calibrated_points(points_xy: Any, intrinsic_matrix: Any) -> Any:
    np = _require_numpy()
    rays = (np.linalg.inv(intrinsic_matrix) @ _homogeneous(points_xy).T).T
    return rays[:, :2] / rays[:, 2:3]


def _fit_essential_eight_point(points_a: Any, points_b: Any) -> Any:
    np = _require_numpy()
    norm_a, transform_a = _hartley_normalize(points_a)
    norm_b, transform_b = _hartley_normalize(points_b)
    x_a, y_a = norm_a[:, 0], norm_a[:, 1]
    x_b, y_b = norm_b[:, 0], norm_b[:, 1]
    design = np.column_stack((
        x_b * x_a, x_b * y_a, x_b,
        y_b * x_a, y_b * y_a, y_b,
        x_a, y_a, np.ones(len(norm_a), dtype=np.float64),
    ))
    _, singular_values, vh = np.linalg.svd(design, full_matrices=True)
    if (
        len(singular_values) < 8
        or singular_values[-1]
        <= singular_values[0] * _MINIMAL_RANK_RELATIVE_TOLERANCE
    ):
        raise ValueError("rank-deficient essential sample")
    essential = transform_b.T @ vh[-1].reshape(3, 3) @ transform_a
    u, singular_values, vh = np.linalg.svd(essential)
    average = 0.5 * float(singular_values[0] + singular_values[1])
    essential = u @ np.diag((average, average, 0.0)) @ vh
    return _canonical_model(essential)


def _fit_homography_four_point(points_a: Any, points_b: Any) -> Any:
    np = _require_numpy()
    norm_a, transform_a = _hartley_normalize(points_a)
    norm_b, transform_b = _hartley_normalize(points_b)
    design = np.empty((2 * len(norm_a), 9), dtype=np.float64)
    for index, ((x_value, y_value), (u_value, v_value)) in enumerate(zip(norm_a, norm_b)):
        design[2 * index] = (
            -x_value, -y_value, -1.0, 0.0, 0.0, 0.0,
            u_value * x_value, u_value * y_value, u_value,
        )
        design[2 * index + 1] = (
            0.0, 0.0, 0.0, -x_value, -y_value, -1.0,
            v_value * x_value, v_value * y_value, v_value,
        )
    _, singular_values, vh = np.linalg.svd(design, full_matrices=True)
    if (
        len(singular_values) < 8
        or singular_values[-1]
        <= singular_values[0] * _MINIMAL_RANK_RELATIVE_TOLERANCE
    ):
        raise ValueError("rank-deficient homography sample")
    homography = (
        np.linalg.inv(transform_b) @ vh[-1].reshape(3, 3) @ transform_a
    )
    return _canonical_model(homography)


def _sampson_errors_px(fundamental: Any, points_a: Any, points_b: Any) -> Any:
    np = _require_numpy()
    homogeneous_a = _homogeneous(points_a)
    homogeneous_b = _homogeneous(points_b)
    lines_b = (fundamental @ homogeneous_a.T).T
    lines_a = (fundamental.T @ homogeneous_b.T).T
    residual = np.sum(homogeneous_b * lines_b, axis=1)
    denominator = (
        lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2
        + lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2
    )
    errors = np.full(len(points_a), np.inf, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    errors[valid] = np.abs(residual[valid]) / np.sqrt(denominator[valid])
    return errors


def _symmetric_transfer_errors(homography: Any, points_a: Any, points_b: Any) -> Any:
    np = _require_numpy()
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.full(len(points_a), np.inf, dtype=np.float64)
    homogeneous_a = _homogeneous(points_a)
    homogeneous_b = _homogeneous(points_b)
    predicted_b = (homography @ homogeneous_a.T).T
    predicted_a = (inverse @ homogeneous_b.T).T
    valid = (np.abs(predicted_b[:, 2]) > 1.0e-15) & (np.abs(predicted_a[:, 2]) > 1.0e-15)
    errors = np.full(len(points_a), np.inf, dtype=np.float64)
    projected_b = predicted_b[valid, :2] / predicted_b[valid, 2:3]
    projected_a = predicted_a[valid, :2] / predicted_a[valid, 2:3]
    forward_sq = np.sum((projected_b - points_b[valid]) ** 2, axis=1)
    backward_sq = np.sum((projected_a - points_a[valid]) ** 2, axis=1)
    errors[valid] = np.sqrt(0.5 * (forward_sq + backward_sq))
    return errors


def _score_candidate(model: Any, errors: Any, threshold_px: float,
                     sample: tuple[int, ...]) -> tuple[tuple[Any, ...], Any]:
    np = _require_numpy()
    inliers = np.asarray(errors <= threshold_px, dtype=bool)
    count = int(np.count_nonzero(inliers))
    median = float(np.median(errors[inliers])) if count else float("inf")
    score = (-count, median, sample, _canonical_model(model).tobytes(order="C"))
    return score, inliers


def _fit_best_essential(
    calibrated_a: Any, calibrated_b: Any, pixel_a: Any, pixel_b: Any,
    inverse_intrinsic_a: Any, inverse_intrinsic_b: Any, threshold_px: float,
    schedule: tuple[tuple[int, ...], ...],
) -> tuple[Any | None, Any, float]:
    np = _require_numpy()
    best: tuple[tuple[Any, ...], Any, Any, Any] | None = None
    for sample in schedule:
        sample_indices = np.asarray(sample, dtype=np.int64)
        try:
            model = _fit_essential_eight_point(
                calibrated_a[sample_indices], calibrated_b[sample_indices],
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        fundamental = inverse_intrinsic_b.T @ model @ inverse_intrinsic_a
        errors = _sampson_errors_px(fundamental, pixel_a, pixel_b)
        score, inliers = _score_candidate(model, errors, threshold_px, sample)
        candidate = (score, model, inliers, errors)
        if best is None or score < best[0]:
            best = candidate
    if best is None:
        return None, np.zeros(len(calibrated_a), dtype=bool), float("inf")
    _, model, inliers, errors = best
    median = float(np.median(errors[inliers])) if np.any(inliers) else float("inf")
    return model, inliers, median


def _fit_best_homography(points_a: Any, points_b: Any, threshold_px: float,
                         schedule: tuple[tuple[int, ...], ...]) -> tuple[Any | None, Any, float]:
    np = _require_numpy()
    best: tuple[tuple[Any, ...], Any, Any, Any] | None = None
    for sample in schedule:
        sample_indices = np.asarray(sample, dtype=np.int64)
        try:
            model = _fit_homography_four_point(
                points_a[sample_indices], points_b[sample_indices],
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = _symmetric_transfer_errors(model, points_a, points_b)
        score, inliers = _score_candidate(model, errors, threshold_px, sample)
        candidate = (score, model, inliers, errors)
        if best is None or score < best[0]:
            best = candidate
    if best is None:
        return None, np.zeros(len(points_a), dtype=bool), float("inf")
    _, model, inliers, errors = best
    median = float(np.median(errors[inliers])) if np.any(inliers) else float("inf")
    return model, inliers, median


def _triangulate(calibrated_a: Any, calibrated_b: Any,
                 rotation: Any, translation: Any) -> Any:
    np = _require_numpy()
    projection_a = np.column_stack((np.eye(3), np.zeros(3)))
    projection_b = np.column_stack((rotation, translation))
    points = np.full((len(calibrated_a), 3), np.nan, dtype=np.float64)
    for index, ((x_a, y_a), (x_b, y_b)) in enumerate(zip(calibrated_a, calibrated_b)):
        design = np.stack((
            x_a * projection_a[2] - projection_a[0],
            y_a * projection_a[2] - projection_a[1],
            x_b * projection_b[2] - projection_b[0],
            y_b * projection_b[2] - projection_b[1],
        ))
        _, _, vh = np.linalg.svd(design, full_matrices=True)
        if abs(float(vh[-1, 3])) > 1.0e-15:
            points[index] = vh[-1, :3] / vh[-1, 3]
    return points


def _pose_reprojection_error(points_xyz: Any, calibrated_a: Any, calibrated_b: Any,
                             rotation: Any, translation: Any) -> float:
    np = _require_numpy()
    valid = np.all(np.isfinite(points_xyz), axis=1)
    if not np.any(valid):
        return float("inf")
    points = points_xyz[valid]
    camera_b = (rotation @ points.T).T + translation
    depth_valid = (np.abs(points[:, 2]) > 1.0e-15) & (np.abs(camera_b[:, 2]) > 1.0e-15)
    if not np.any(depth_valid):
        return float("inf")
    projected_a = points[depth_valid, :2] / points[depth_valid, 2:3]
    projected_b = camera_b[depth_valid, :2] / camera_b[depth_valid, 2:3]
    errors = np.concatenate((
        np.linalg.norm(projected_a - calibrated_a[valid][depth_valid], axis=1),
        np.linalg.norm(projected_b - calibrated_b[valid][depth_valid], axis=1),
    ))
    return float(np.median(errors))


def _select_pose_candidate_index(positive_depth_counts: Any,
                                 median_reprojection_errors: Any) -> int:
    """Select a four-pose candidate by cheirality, error, then stable index."""
    if len(positive_depth_counts) != 4 or len(median_reprojection_errors) != 4:
        raise ValueError("essential decomposition must provide four pose candidates")
    return min(range(4), key=lambda candidate_index: (
        -int(positive_depth_counts[candidate_index]),
        float(median_reprojection_errors[candidate_index]),
        candidate_index,
    ))


def _decompose_essential(essential: Any, calibrated_a: Any,
                         calibrated_b: Any) -> tuple[Any, Any, Any, float, float]:
    np = _require_numpy()
    u, _, vh = np.linalg.svd(essential)
    if np.linalg.det(u) < 0.0:
        u[:, -1] *= -1.0
    if np.linalg.det(vh) < 0.0:
        vh[-1, :] *= -1.0
    w_matrix = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    rotations = (u @ w_matrix @ vh, u @ w_matrix.T @ vh)
    direction = u[:, 2]
    candidates = (
        (rotations[0], direction), (rotations[0], -direction),
        (rotations[1], direction), (rotations[1], -direction),
    )
    candidate_results: list[tuple[Any, Any, Any, Any]] = []
    positive_depth_counts: list[int] = []
    median_reprojection_errors: list[float] = []
    for rotation, translation in candidates:
        points_xyz = _triangulate(calibrated_a, calibrated_b, rotation, translation)
        camera_b = (rotation @ points_xyz.T).T + translation
        positive = (
            np.all(np.isfinite(points_xyz), axis=1)
            & (points_xyz[:, 2] > 0.0) & (camera_b[:, 2] > 0.0)
        )
        positive_count = int(np.count_nonzero(positive))
        reprojection = _pose_reprojection_error(
            points_xyz, calibrated_a, calibrated_b, rotation, translation,
        )
        positive_depth_counts.append(positive_count)
        median_reprojection_errors.append(reprojection)
        candidate_results.append((rotation, translation, points_xyz, positive))
    winner_index = _select_pose_candidate_index(
        positive_depth_counts, median_reprojection_errors,
    )
    rotation, translation, points_xyz, positive = candidate_results[winner_index]
    fraction = float(np.count_nonzero(positive) / len(positive)) if len(positive) else 0.0
    if np.any(positive):
        camera_b_center = -rotation.T @ translation
        rays_a = points_xyz[positive]
        rays_b = points_xyz[positive] - camera_b_center
        norms = np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1)
        cosine = np.sum(rays_a * rays_b, axis=1) / norms
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        triangulation_angle = float(np.median(angles))
    else:
        triangulation_angle = 0.0
    return (
        np.ascontiguousarray(rotation, dtype=np.float64),
        np.asarray(translation, dtype=np.float64),
        points_xyz,
        fraction,
        triangulation_angle,
    )


def _planar_pose_candidates(calibrated_homography: Any) -> list[tuple[Any, Any, Any]]:
    """Faugeras SVD decomposition of a calibrated homography into (R, t, n).

    Returns the physically meaningful candidates (positive-determinant branch)
    of H ~ R + t n^T.  Translation is direction-only, like the essential path;
    the plane normal n is expressed in camera A.  Candidate order is
    deterministic: the two sign choices of (eps1, eps3) enumerated in a fixed
    sequence, each contributing one (R, t, n) and its (R, -t, -n) mirror.
    """
    np = _require_numpy()
    u, singular_values, vh = np.linalg.svd(calibrated_homography)
    d1, d2, d3 = (float(value) for value in singular_values)
    if d2 <= 1.0e-12 or not all(math.isfinite(v) for v in (d1, d2, d3)):
        raise ValueError("singular calibrated homography")
    scaled = calibrated_homography / d2
    if float(np.linalg.det(scaled)) < 0.0:
        scaled = -scaled
    u, singular_values, vh = np.linalg.svd(scaled)
    if np.linalg.det(u) < 0.0:
        u[:, -1] *= -1.0
        vh[-1, :] *= -1.0
    d1, d2, d3 = (float(value) for value in singular_values)
    v = vh.T
    span = d1 * d1 - d3 * d3
    if span <= 1.0e-15:
        # d1 == d3: pure rotation (or identity) — no planar translation.
        raise ValueError("homography carries no resolvable planar translation")
    x1 = math.sqrt(max(0.0, (d1 * d1 - d2 * d2) / span))
    x3 = math.sqrt(max(0.0, (d2 * d2 - d3 * d3) / span))
    sin_theta_base = (d1 - d3) * x1 * x3 / d2
    cos_theta = (d1 * x3 * x3 + d3 * x1 * x1) / d2
    candidates: list[tuple[Any, Any, Any]] = []
    for eps1, eps3 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        sin_theta = eps1 * eps3 * sin_theta_base
        rotation_prime = np.array((
            (cos_theta, 0.0, -sin_theta),
            (0.0, 1.0, 0.0),
            (sin_theta, 0.0, cos_theta),
        ), dtype=np.float64)
        normal_prime = np.array((eps1 * x1, 0.0, eps3 * x3), dtype=np.float64)
        translation_prime = (d1 - d3) * np.array(
            (eps1 * x1, 0.0, -eps3 * x3), dtype=np.float64,
        )
        rotation = u @ rotation_prime @ vh
        translation = u @ translation_prime
        normal = v @ normal_prime
        norm = float(np.linalg.norm(translation))
        if not math.isfinite(norm) or norm < 1.0e-12:
            continue
        translation = translation / norm
        candidates.append((rotation, translation, normal))
        candidates.append((rotation, -translation, -normal))
    if not candidates:
        raise ValueError("homography decomposition produced no candidates")
    return candidates


def _decompose_homography_planar(
    homography: Any, intrinsic_a: Any, intrinsic_b: Any,
    calibrated_a: Any, calibrated_b: Any,
) -> tuple[Any, Any, Any, float, float]:
    """Recover (R, t_dir, n, positive_fraction, triangulation_angle_deg).

    Mirrors _decompose_essential's deterministic selection: triangulate the
    homography inliers under every candidate pose, then pick by maximum
    positive-depth count, then median reprojection, then candidate index.  The
    plane normal must face camera A (n_z < 0 in the Atlas-camera looking
    convention is expressed here in OpenCV coords as n pointing toward the
    camera: n[2] > 0 rejected after cheirality — handled implicitly, since a
    plane behind the camera fails positive depth).
    """
    np = _require_numpy()
    calibrated = np.linalg.inv(intrinsic_b) @ homography @ intrinsic_a
    candidates = _planar_pose_candidates(calibrated)
    positive_depth_counts: list[int] = []
    median_reprojection_errors: list[float] = []
    candidate_results: list[tuple[Any, Any, Any, Any, Any]] = []
    for rotation, translation, normal in candidates:
        points_xyz = _triangulate(calibrated_a, calibrated_b, rotation, translation)
        camera_b = (rotation @ points_xyz.T).T + translation
        positive = (
            np.all(np.isfinite(points_xyz), axis=1)
            & (points_xyz[:, 2] > 0.0) & (camera_b[:, 2] > 0.0)
        )
        positive_depth_counts.append(int(np.count_nonzero(positive)))
        median_reprojection_errors.append(_pose_reprojection_error(
            points_xyz, calibrated_a, calibrated_b, rotation, translation,
        ))
        candidate_results.append((rotation, translation, normal, points_xyz, positive))
    winner = min(range(len(candidate_results)), key=lambda index: (
        -positive_depth_counts[index],
        median_reprojection_errors[index],
        index,
    ))
    rotation, translation, normal, points_xyz, positive = candidate_results[winner]
    fraction = float(np.count_nonzero(positive) / len(positive)) if len(positive) else 0.0
    if np.any(positive):
        camera_b_center = -rotation.T @ translation
        rays_a = points_xyz[positive]
        rays_b = points_xyz[positive] - camera_b_center
        norms = np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1)
        cosine = np.sum(rays_a * rays_b, axis=1) / norms
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        triangulation_angle = float(np.median(angles))
    else:
        triangulation_angle = 0.0
    return (
        np.ascontiguousarray(rotation, dtype=np.float64),
        np.asarray(translation, dtype=np.float64),
        np.asarray(normal, dtype=np.float64),
        fraction,
        triangulation_angle,
    )


def _nearest_rotation(calibrated_homography: Any) -> Any:
    np = _require_numpy()
    determinant = float(np.linalg.det(calibrated_homography))
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-15:
        raise ValueError("singular calibrated homography")
    scaled = calibrated_homography / math.copysign(abs(determinant) ** (1.0 / 3.0), determinant)
    u, _, vh = np.linalg.svd(scaled)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return np.asarray(rotation, dtype=np.float64)


def _homography_rotation_evidence(homography: Any | None, intrinsic_a: Any,
                                  intrinsic_b: Any, points_a: Any, points_b: Any,
                                  inliers: Any) -> tuple[Any | None, float]:
    np = _require_numpy()
    if homography is None or not np.any(inliers):
        return None, float("inf")
    try:
        calibrated = np.linalg.inv(intrinsic_b) @ homography @ intrinsic_a
        rotation = _nearest_rotation(calibrated)
        rotation_homography = intrinsic_b @ rotation @ np.linalg.inv(intrinsic_a)
        errors = _symmetric_transfer_errors(
            rotation_homography, points_a[inliers], points_b[inliers],
        )
    except (ValueError, np.linalg.LinAlgError):
        return None, float("inf")
    finite = errors[np.isfinite(errors)]
    residual = float(np.median(finite)) if len(finite) else float("inf")
    return rotation, residual


def _occupied_grid_cells(points_xy: Any, inliers: Any,
                         width: int, height: int) -> int:
    np = _require_numpy()
    accepted = np.asarray(points_xy, dtype=np.float64)[inliers]
    if not len(accepted) or width <= 0 or height <= 0:
        return 0
    cells = np.floor(
        accepted / np.array((width, height), dtype=np.float64) * 4.0
    ).astype(np.int64)
    cells = np.clip(cells, 0, 3)
    return len({(int(cell[0]), int(cell[1])) for cell in cells})


def fit_pair_models(matches: PairMatches, intr_a: AtlasIntrinsics,
                    intr_b: AtlasIntrinsics, settings: MultiViewSettings,
                    fingerprint: str) -> PairModelEvidence:
    """Fit deterministic essential and homography candidates to one pair."""
    np = _require_numpy()
    points_a = np.asarray(matches.points_a, dtype=np.float64)
    points_b = np.asarray(matches.points_b, dtype=np.float64)
    if points_a.ndim != 2 or points_a.shape[1:] != (2,) or points_b.shape != points_a.shape:
        raise MotionModelError(
            "insufficient_overlap", "pair correspondences must be matching N-by-2 arrays",
        )
    if not np.all(np.isfinite(points_a)) or not np.all(np.isfinite(points_b)):
        raise MotionModelError(
            "degenerate_geometry", "pair correspondences contain non-finite coordinates",
        )
    if len(points_a) < 4:
        raise MotionModelError(
            "insufficient_overlap", "at least four pair correspondences are required",
        )
    intrinsic_a = _intrinsic_matrix(intr_a)
    intrinsic_b = _intrinsic_matrix(intr_b)
    calibrated_a = _calibrated_points(points_a, intrinsic_a)
    calibrated_b = _calibrated_points(points_b, intrinsic_b)
    profile = QUALITY_PROFILES[settings.match_quality]
    inverse_intrinsic_a = np.linalg.inv(intrinsic_a)
    inverse_intrinsic_b = np.linalg.inv(intrinsic_b)
    essential_schedule, homography_schedule = _model_sample_schedules(
        len(points_a), settings, fingerprint,
    )
    essential, essential_inliers, median_essential = _fit_best_essential(
        calibrated_a, calibrated_b, points_a, points_b,
        inverse_intrinsic_a, inverse_intrinsic_b,
        profile.reprojection_threshold_px, essential_schedule,
    )
    homography, homography_inliers, median_homography = _fit_best_homography(
        points_a, points_b, profile.reprojection_threshold_px,
        homography_schedule,
    )
    relative_rotation = None
    translation_direction = None
    positive_depth_fraction = 0.0
    median_triangulation_angle_deg = 0.0
    if essential is not None and np.any(essential_inliers):
        try:
            (
                relative_rotation, translation_direction, _,
                positive_depth_fraction, median_triangulation_angle_deg,
            ) = _decompose_essential(
                essential,
                calibrated_a[essential_inliers], calibrated_b[essential_inliers],
            )
        except (ValueError, np.linalg.LinAlgError):
            relative_rotation = None
            translation_direction = None
    _, rotation_residual = _homography_rotation_evidence(
        homography, intrinsic_a, intrinsic_b,
        points_a, points_b, homography_inliers,
    )
    planar_rotation = None
    planar_translation = None
    planar_normal = None
    planar_positive_fraction = 0.0
    planar_triangulation_angle = 0.0
    if homography is not None and np.count_nonzero(homography_inliers) >= 4:
        try:
            (
                planar_rotation, planar_translation, planar_normal,
                planar_positive_fraction, planar_triangulation_angle,
            ) = _decompose_homography_planar(
                homography, intrinsic_a, intrinsic_b,
                calibrated_a[homography_inliers], calibrated_b[homography_inliers],
            )
        except (ValueError, np.linalg.LinAlgError):
            planar_rotation = None
            planar_translation = None
            planar_normal = None
    return PairModelEvidence(
        frame_a=matches.frame_a,
        frame_b=matches.frame_b,
        essential_matrix=essential,
        homography=homography,
        relative_rotation=relative_rotation,
        translation_direction=translation_direction,
        essential_inliers=essential_inliers,
        homography_inliers=homography_inliers,
        essential_inlier_count=int(np.count_nonzero(essential_inliers)),
        homography_inlier_count=int(np.count_nonzero(homography_inliers)),
        median_essential_error_px=median_essential,
        median_homography_error_px=median_homography,
        median_triangulation_angle_deg=median_triangulation_angle_deg,
        positive_depth_fraction=positive_depth_fraction,
        essential_occupied_grid_cells=_occupied_grid_cells(
            points_a, essential_inliers, intr_a.image_width, intr_a.image_height,
        ),
        homography_rotation_residual_px=rotation_residual,
        planar_rotation=planar_rotation,
        planar_translation_direction=planar_translation,
        planar_plane_normal=planar_normal,
        planar_positive_depth_fraction=planar_positive_fraction,
        planar_median_triangulation_angle_deg=planar_triangulation_angle,
        homography_occupied_grid_cells=_occupied_grid_cells(
            points_a, homography_inliers, intr_a.image_width, intr_a.image_height,
        ),
    )


def _translated_passes(evidence: PairModelEvidence,
                       profile: QualityProfile) -> bool:
    return (
        evidence.essential_matrix is not None
        and evidence.relative_rotation is not None
        and evidence.translation_direction is not None
        and evidence.essential_inlier_count >= profile.min_inliers
        and evidence.essential_occupied_grid_cells >= profile.min_grid_cells
        and evidence.median_essential_error_px <= profile.reprojection_threshold_px
        and evidence.positive_depth_fraction >= _POSITIVE_DEPTH_FRACTION
        and evidence.median_triangulation_angle_deg >= profile.min_triangulation_angle_deg
    )


def _planar_translated_passes(evidence: PairModelEvidence,
                              profile: QualityProfile) -> bool:
    """Near-planar scene with genuine translation: homography decomposition.

    Requires the homography to fit well AND to be inconsistent with a pure
    rotation (large rotation residual) — otherwise rotation-only is the honest
    interpretation.  The decomposed pose must clear the same cheirality and
    parallax bars as the essential path.
    """
    rotation_residual = evidence.homography_rotation_residual_px
    total_matches = len(evidence.homography_inliers)
    # A single plane legitimately concentrates its consensus (a facade band
    # under sky and road), so the planar gate accepts two fewer grid cells
    # than the profile — but in exchange the homography inliers must DOMINATE
    # the raw matches (>= 50%), which a moving object's localized consensus
    # essentially never does.  Found live 2026-08-09 on the sh001 street set:
    # 226/361 inliers, 20.5 deg parallax, 3/16 cells.
    relaxed_min_cells = max(2, profile.min_grid_cells - 2)
    return (
        evidence.homography is not None
        and evidence.planar_rotation is not None
        and evidence.planar_translation_direction is not None
        and evidence.homography_inlier_count >= profile.min_inliers
        and evidence.homography_occupied_grid_cells >= relaxed_min_cells
        and total_matches > 0
        and evidence.homography_inlier_count * 2 >= total_matches
        and evidence.median_homography_error_px <= profile.reprojection_threshold_px
        and rotation_residual is not None
        and rotation_residual > profile.reprojection_threshold_px
        and evidence.planar_positive_depth_fraction >= _POSITIVE_DEPTH_FRACTION
        and evidence.planar_median_triangulation_angle_deg
            >= profile.min_triangulation_angle_deg
    )


def _rotation_only_passes(evidence: PairModelEvidence,
                          profile: QualityProfile) -> bool:
    rotation_residual = evidence.homography_rotation_residual_px
    if rotation_residual is None:
        return False
    return (
        evidence.homography is not None
        and evidence.homography_inlier_count >= profile.min_inliers
        and rotation_residual <= profile.reprojection_threshold_px
    )


def _score_summary(evidence: PairModelEvidence) -> str:
    rotation_residual = evidence.homography_rotation_residual_px
    rotation_residual_text = (
        "legacy_unavailable" if rotation_residual is None
        else f"{rotation_residual:.6g}"
    )
    return (
        "essential("
        f"inliers={evidence.essential_inlier_count}, "
        f"cells={evidence.essential_occupied_grid_cells}, "
        f"error_px={evidence.median_essential_error_px:.6g}, "
        f"positive_depth={evidence.positive_depth_fraction:.6g}, "
        f"angle_deg={evidence.median_triangulation_angle_deg:.6g}); "
        "rotation_homography("
        f"inliers={evidence.homography_inlier_count}, "
        f"transfer_error_px={evidence.median_homography_error_px:.6g}, "
        f"rotation_residual_px={rotation_residual_text}); "
        "planar_translated("
        f"decomposed={evidence.planar_rotation is not None}, "
        f"cells={evidence.homography_occupied_grid_cells}, "
        f"positive_depth={evidence.planar_positive_depth_fraction:.6g}, "
        f"angle_deg={evidence.planar_median_triangulation_angle_deg:.6g})"
    )


def select_capture_mode(
    evidence: PairModelEvidence, requested_mode: CaptureMode,
    profile: QualityProfile,
) -> Literal["translated", "translated_planar", "rotation_only"]:
    """Select a supported interpretation or fail with an exact outcome code.

    "translated_planar" is a translated capture whose pair pose comes from
    homography decomposition — a near-planar scene where the essential matrix
    is degenerate.  It ranks after the essential model and before
    rotation-only in auto, and satisfies a forced "translated" request.
    """
    if requested_mode not in ("auto", "translated", "rotation_only"):
        raise ValueError(f"unsupported capture mode {requested_mode!r}")
    translated = _translated_passes(evidence, profile)
    planar = _planar_translated_passes(evidence, profile)
    rotation_only = _rotation_only_passes(evidence, profile)
    summary = _score_summary(evidence)
    if requested_mode == "translated":
        if translated:
            return "translated"
        if planar:
            return "translated_planar"
        raise MotionModelError(
            "degenerate_geometry", f"forced translated model failed checks; {summary}",
        )
    if requested_mode == "rotation_only":
        if rotation_only:
            return "rotation_only"
        raise MotionModelError(
            "ambiguous_motion_model",
            f"forced rotation-only model failed checks; {summary}",
        )
    if translated:
        return "translated"
    if planar:
        return "translated_planar"
    if rotation_only:
        return "rotation_only"
    raise MotionModelError(
        "ambiguous_motion_model", f"neither motion model passed checks; {summary}",
    )


def build_tracks(pair_matches: Any, n_frames: int) -> tuple[FeatureTrack, ...]:
    """Join stable feature identities into deterministic closed tracks."""
    np = _require_numpy()
    if n_frames < 2:
        raise ValueError("at least two frames are required to build tracks")
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    coordinates: dict[tuple[int, int], list[tuple[float, float]]] = {}
    edges: set[frozenset[tuple[int, int]]] = set()

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent = parents.setdefault(node, node)
        while parent != parents[parent]:
            parent = parents[parent]
        while node != parent:
            next_node = parents[node]
            parents[node] = parent
            node = next_node
        return parent

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            return
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        parents[root_b] = root_a

    ordered_pairs = sorted(
        tuple(pair_matches), key=lambda pair: (int(pair.frame_a), int(pair.frame_b)),
    )
    for pair in ordered_pairs:
        frame_a, frame_b = int(pair.frame_a), int(pair.frame_b)
        if (
            frame_a == frame_b or frame_a < 0 or frame_b < 0
            or frame_a >= n_frames or frame_b >= n_frames
        ):
            raise ValueError("pair frame indices must name two distinct input frames")
        points_a = np.asarray(pair.points_a, dtype=np.float64)
        points_b = np.asarray(pair.points_b, dtype=np.float64)
        indices = np.asarray(pair.indices, dtype=np.int64)
        if (
            points_a.ndim != 2 or points_a.shape[1:] != (2,)
            or points_b.shape != points_a.shape
            or indices.shape != (len(points_a), 2)
        ):
            raise ValueError("pair matches must contain aligned N-by-2 points and indices")
        if not np.all(np.isfinite(points_a)) or not np.all(np.isfinite(points_b)):
            raise ValueError("track observations must be finite")
        for point_a, point_b, feature_indices in zip(points_a, points_b, indices):
            node_a = (frame_a, int(feature_indices[0]))
            node_b = (frame_b, int(feature_indices[1]))
            coordinate_a = (float(point_a[0]), float(point_a[1]))
            coordinate_b = (float(point_b[0]), float(point_b[1]))
            coordinates.setdefault(node_a, []).append(coordinate_a)
            coordinates.setdefault(node_b, []).append(coordinate_b)
            union(node_a, node_b)
            edges.add(frozenset((node_a, node_b)))

    components: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in sorted(parents):
        components.setdefault(find(node), []).append(node)
    observation_tuples: list[tuple[FeatureObservation, ...]] = []
    for nodes in components.values():
        nodes = sorted(nodes)
        frames = [node[0] for node in nodes]
        if len(nodes) < 2 or len(frames) != len(set(frames)):
            continue
        if any(len(set(coordinates[node])) != 1 for node in nodes):
            continue
        if len(nodes) >= 3 and any(
            frozenset((first, second)) not in edges
            for first, second in combinations(nodes, 2)
        ):
            continue
        observations = tuple(
            FeatureObservation(node[0], node[1], coordinates[node][0])
            for node in nodes
        )
        observation_tuples.append(observations)
    observation_tuples.sort(key=lambda observations: tuple(
        (item.frame_index, item.feature_index, item.point_xy[0], item.point_xy[1])
        for item in observations
    ))
    return tuple(
        FeatureTrack(track_id, observations)
        for track_id, observations in enumerate(observation_tuples)
    )


def _normalise_direction(direction: Any) -> Any:
    np = _require_numpy()
    result = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1.0e-15:
        raise MotionModelError("degenerate_geometry", "relative translation is unavailable")
    return result / norm


def _relative_pose_map(pair_evidence: Any) -> dict[tuple[int, int], PairModelEvidence]:
    return {
        (int(evidence.frame_a), int(evidence.frame_b)): evidence
        for evidence in sorted(
            tuple(pair_evidence),
            key=lambda value: (int(value.frame_a), int(value.frame_b)),
        )
    }


def initialise_rig(pair_evidence: Any,
                   mode: Literal["translated", "rotation_only"]) -> CameraRig:
    """Compose pair poses into a stable photo-1-anchored camera rig."""
    np = _require_numpy()
    if mode not in ("translated", "rotation_only"):
        raise ValueError(f"unsupported selected mode {mode!r}")
    evidence_tuple = tuple(sorted(
        tuple(pair_evidence),
        key=lambda value: (int(value.frame_a), int(value.frame_b)),
    ))
    if not evidence_tuple:
        raise MotionModelError("degenerate_geometry", "pair evidence is empty")
    frame_count = 1 + max(
        max(int(value.frame_a), int(value.frame_b)) for value in evidence_tuple
    )
    rotations: list[Any | None] = [None] * frame_count
    translations: list[Any | None] = [None] * frame_count
    rotations[0] = np.eye(3, dtype=np.float64)
    translations[0] = np.zeros(3, dtype=np.float64)
    pair_map = _relative_pose_map(evidence_tuple)

    for frame_index in range(1, frame_count):
        direct = pair_map.get((0, frame_index))
        if direct is None or direct.relative_rotation is None:
            raise MotionModelError(
                "degenerate_geometry", f"photo 1 has no pose evidence for frame {frame_index}",
            )
        rotations[frame_index] = np.ascontiguousarray(
            direct.relative_rotation, dtype=np.float64,
        )
        translations[frame_index] = (
            np.zeros(3, dtype=np.float64) if mode == "rotation_only"
            else _normalise_direction(direct.translation_direction)
        )

    if mode == "translated" and frame_count == 3:
        closing = pair_map.get((1, 2))
        if closing is not None and closing.translation_direction is not None:
            if closing.relative_rotation is None:
                raise MotionModelError(
                    "inconsistent_third_view",
                    "closing pair has translation evidence without a relative rotation",
                )
            direction_01 = _normalise_direction(pair_map[(0, 1)].translation_direction)
            direction_02 = _normalise_direction(pair_map[(0, 2)].translation_direction)
            direction_12 = _normalise_direction(closing.translation_direction)
            relative_rotation_12 = np.asarray(
                closing.relative_rotation, dtype=np.float64,
            )
            scale_system = np.column_stack((
                direction_02,
                -(relative_rotation_12 @ direction_01),
                -direction_12,
            ))
            _, _, vh = np.linalg.svd(scale_system)
            scales = vh[-1]
            if scales[1] < 0.0:
                scales = -scales
            if scales[1] > 1.0e-12 and scales[0] > 1.0e-12:
                translations[1] = direction_01
                translations[2] = direction_02 * float(scales[0] / scales[1])

    error_field = (
        "median_essential_error_px" if mode == "translated"
        else "median_homography_error_px"
    )
    finite_errors = [
        float(getattr(value, error_field)) for value in evidence_tuple
        if math.isfinite(float(getattr(value, error_field)))
    ]
    initial_rmse = (
        math.sqrt(sum(value * value for value in finite_errors) / len(finite_errors))
        if finite_errors else float("inf")
    )
    return CameraRig(
        tuple(np.ascontiguousarray(value, dtype=np.float64) for value in rotations),
        tuple(np.asarray(value, dtype=np.float64) for value in translations),
        np.empty((0, 3), dtype=np.float64),
        initial_rmse,
        _pair_evidence=evidence_tuple,
    )


def _skew(vector: Any) -> Any:
    np = _require_numpy()
    x_value, y_value, z_value = np.asarray(vector, dtype=np.float64)
    return np.array([
        [0.0, -z_value, y_value],
        [z_value, 0.0, -x_value],
        [-y_value, x_value, 0.0],
    ], dtype=np.float64)


def _rotation_increment(rotation_vector: Any) -> Any:
    np = _require_numpy()
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    matrix = _skew(vector)
    if angle <= 1.0e-12:
        return np.eye(3, dtype=np.float64) + matrix + 0.5 * (matrix @ matrix)
    sine_scale = math.sin(angle) / angle
    cosine_scale = (1.0 - math.cos(angle)) / (angle * angle)
    return np.eye(3, dtype=np.float64) + sine_scale * matrix + cosine_scale * (matrix @ matrix)


def _triangulate_track(track: FeatureTrack, rotations: tuple[Any, ...],
                       translations: tuple[Any, ...], intrinsic_matrices: tuple[Any, ...]) -> Any | None:
    np = _require_numpy()
    rows: list[Any] = []
    for observation in track.observations:
        frame_index = observation.frame_index
        inverse_intrinsic = np.linalg.inv(intrinsic_matrices[frame_index])
        ray = inverse_intrinsic @ np.array(
            (observation.point_xy[0], observation.point_xy[1], 1.0),
            dtype=np.float64,
        )
        x_value, y_value = ray[0] / ray[2], ray[1] / ray[2]
        projection = np.column_stack((rotations[frame_index], translations[frame_index]))
        rows.extend((
            x_value * projection[2] - projection[0],
            y_value * projection[2] - projection[1],
        ))
    design = np.asarray(rows, dtype=np.float64)
    row_norms = np.linalg.norm(design, axis=1)
    if np.any(row_norms <= 1.0e-15):
        return None
    design /= row_norms[:, None]
    _, _, vh = np.linalg.svd(design, full_matrices=True)
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) <= 1.0e-15:
        return None
    point = np.asarray(homogeneous[:3] / homogeneous[3], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        return None
    for observation in track.observations:
        camera_point = rotations[observation.frame_index] @ point + translations[observation.frame_index]
        if not math.isfinite(float(camera_point[2])) or camera_point[2] <= 1.0e-12:
            return None
    return point


def _project_pixel(point: Any, rotation: Any, translation: Any,
                   intrinsic_matrix: Any) -> Any | None:
    np = _require_numpy()
    camera_point = rotation @ point + translation
    if camera_point[2] <= 1.0e-12:
        return None
    projected = intrinsic_matrix @ camera_point
    pixel = projected[:2] / projected[2]
    return np.asarray(pixel, dtype=np.float64)


def _track_errors(track: FeatureTrack, point: Any, rotations: tuple[Any, ...],
                  translations: tuple[Any, ...], intrinsic_matrices: tuple[Any, ...]) -> list[float]:
    np = _require_numpy()
    errors: list[float] = []
    for observation in track.observations:
        pixel = _project_pixel(
            point, rotations[observation.frame_index],
            translations[observation.frame_index],
            intrinsic_matrices[observation.frame_index],
        )
        if pixel is None:
            return [float("inf")]
        errors.append(float(np.linalg.norm(
            pixel - np.asarray(observation.point_xy, dtype=np.float64),
        )))
    return errors


def _huber_cost_and_weight(residual: Any, delta: float) -> tuple[float, float]:
    np = _require_numpy()
    norm = float(np.linalg.norm(residual))
    if norm <= delta:
        return 0.5 * norm * norm, 1.0
    return delta * (norm - 0.5 * delta), delta / max(norm, 1.0e-15)


def _pose_observations(frame_index: int, tracks: tuple[FeatureTrack, ...],
                       landmarks: dict[int, Any]) -> list[tuple[Any, Any]]:
    observations: list[tuple[Any, Any]] = []
    for track in tracks:
        point = landmarks.get(track.track_id)
        if point is None:
            continue
        for observation in track.observations:
            if observation.frame_index == frame_index:
                observations.append((point, observation))
    return observations


def _pose_cost(frame_index: int, observations: list[tuple[Any, Any]],
               rotation: Any, translation: Any, intrinsic_matrix: Any,
               delta: float) -> float:
    np = _require_numpy()
    total = 0.0
    for point, observation in observations:
        pixel = _project_pixel(point, rotation, translation, intrinsic_matrix)
        if pixel is None:
            return float("inf")
        residual = pixel - np.asarray(observation.point_xy, dtype=np.float64)
        cost, _ = _huber_cost_and_weight(residual, delta)
        total += cost
    return total


def _pose_step(frame_index: int, observations: list[tuple[Any, Any]],
               rotation: Any, translation: Any, intrinsic_matrix: Any,
               delta: float, damping: float,
               rotation_only: bool = False) -> tuple[Any, Any]:
    np = _require_numpy()
    parameter_count = 3 if rotation_only else 6
    normal = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    gradient = np.zeros(parameter_count, dtype=np.float64)
    fx, fy = float(intrinsic_matrix[0, 0]), float(intrinsic_matrix[1, 1])
    valid_count = 0
    for point, observation in observations:
        camera_point = rotation @ point + translation
        x_value, y_value, z_value = camera_point
        if z_value <= 1.0e-12:
            continue
        pixel = np.array((
            fx * x_value / z_value + intrinsic_matrix[0, 2],
            fy * y_value / z_value + intrinsic_matrix[1, 2],
        ), dtype=np.float64)
        residual = pixel - np.asarray(observation.point_xy, dtype=np.float64)
        projection_jacobian = np.array((
            (fx / z_value, 0.0, -fx * x_value / (z_value * z_value)),
            (0.0, fy / z_value, -fy * y_value / (z_value * z_value)),
        ), dtype=np.float64)
        camera_jacobian = -_skew(camera_point)
        if not rotation_only:
            camera_jacobian = np.column_stack((camera_jacobian, np.eye(3, dtype=np.float64)))
        jacobian = projection_jacobian @ camera_jacobian
        _, weight = _huber_cost_and_weight(residual, delta)
        normal += weight * (jacobian.T @ jacobian)
        gradient += weight * (jacobian.T @ residual)
        valid_count += 1
    if valid_count < parameter_count:
        return rotation, translation
    normal += damping * np.eye(parameter_count, dtype=np.float64)
    try:
        increment = np.linalg.solve(normal, -gradient)
    except np.linalg.LinAlgError:
        return rotation, translation
    rotation_delta = _rotation_increment(increment[:3])
    candidate_rotation = rotation_delta @ rotation
    candidate_translation = (
        np.zeros(3, dtype=np.float64) if rotation_only
        else rotation_delta @ translation + increment[3:]
    )
    current_cost = _pose_cost(
        frame_index, observations, rotation, translation,
        intrinsic_matrix, delta,
    )
    candidate_cost = _pose_cost(
        frame_index, observations, candidate_rotation,
        candidate_translation, intrinsic_matrix, delta,
    )
    if candidate_cost < current_cost:
        return (
            np.ascontiguousarray(candidate_rotation, dtype=np.float64),
            np.asarray(candidate_translation, dtype=np.float64),
        )
    return rotation, translation


def _angle_between_vectors(first: Any, second: Any) -> float:
    np = _require_numpy()
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 1.0e-15:
        return float("inf")
    cosine = float(np.dot(first, second) / norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _rotation_error_degrees(first: Any, second: Any) -> float:
    np = _require_numpy()
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


#: Parallax at (and above) which a pairwise translation direction is treated
#: as fully observed.  Below it, direction uncertainty grows as roughly
#: reference/parallax — a 0.5 m baseline against a 30 m night street measures
#: its direction to a few degrees at best, and holding such a rig to the
#: daylight tolerance rejected captures whose pairs were individually superb
#: (470/586/364 inliers at ~0.9 px; found live 2026-08-10).
_REFERENCE_PARALLAX_DEG = 10.0
_MAX_DIRECTION_TOLERANCE_SCALE = 10.0


def _closure_limits(
    delta: float, pair_evidence: Sequence[PairModelEvidence],
) -> tuple[float, float, float]:
    """Closure gates: fixed rotation and pixel bars, observability-scaled direction.

    Rotation closes from the full match set and pixel closure lives in image
    space — neither depends on baseline length, so their limits stay fixed
    (scaled only by the quality profile, as before).  The translation
    DIRECTION limit additionally scales with the weakest pair's parallax:
    direction is only as measurable as the triangulation angle that observed
    it.  Clamped so a degenerate zero-parallax pair cannot open the gate
    arbitrarily wide.
    """
    scale = delta / QUALITY_PROFILES["balanced"].reprojection_threshold_px
    pair_angles: list[float] = []
    for evidence in pair_evidence:
        angle = max(
            float(evidence.median_triangulation_angle_deg or 0.0),
            float(evidence.planar_median_triangulation_angle_deg or 0.0),
        )
        if angle > 0.0:
            pair_angles.append(angle)
    weakest = min(pair_angles) if pair_angles else _REFERENCE_PARALLAX_DEG
    direction_scale = min(
        _MAX_DIRECTION_TOLERANCE_SCALE,
        max(1.0, _REFERENCE_PARALLAX_DEG / weakest),
    )
    return (
        0.5 * scale,
        1.5 * scale * direction_scale,
        2.0 * scale,
    )


def measure_three_view_closure(refined: RefinedRig) -> ClosureMetrics:
    """Measure pose-edge disagreement and closed-track pixel residual."""
    np = _require_numpy()
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for evidence in refined._pair_evidence:
        frame_a, frame_b = int(evidence.frame_a), int(evidence.frame_b)
        if frame_a >= len(refined.rotations) or frame_b >= len(refined.rotations):
            continue
        predicted_rotation = refined.rotations[frame_b] @ refined.rotations[frame_a].T
        if evidence.relative_rotation is not None:
            rotation_errors.append(_rotation_error_degrees(
                predicted_rotation, evidence.relative_rotation,
            ))
        if evidence.translation_direction is not None:
            predicted_translation = (
                refined.translations[frame_b]
                - predicted_rotation @ refined.translations[frame_a]
            )
            translation_errors.append(_angle_between_vectors(
                predicted_translation, evidence.translation_direction,
            ))
    landmark_by_id = {
        track_id: refined.landmarks[index]
        for index, track_id in enumerate(refined.accepted_track_ids)
        if index < len(refined.landmarks)
    }
    closed_errors: list[float] = []
    if len(refined.rotations) >= 3 and refined._intrinsics:
        matrices = tuple(_intrinsic_matrix(value) for value in refined._intrinsics)
        for track in refined._tracks:
            if len(track.observations) != len(refined.rotations):
                continue
            point = landmark_by_id.get(track.track_id)
            if point is not None:
                closed_errors.extend(_track_errors(
                    track, point, refined.rotations, refined.translations, matrices,
                ))
    median_reprojection = (
        float(np.median(np.asarray(closed_errors, dtype=np.float64)))
        if closed_errors else float("inf")
    )
    return ClosureMetrics(
        max(rotation_errors, default=0.0),
        max(translation_errors, default=0.0),
        median_reprojection,
    )


def _resolve_profile(profile: QualityProfile | str) -> QualityProfile:
    if isinstance(profile, QualityProfile):
        return profile
    try:
        return QUALITY_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown quality profile {profile!r}") from exc


def refine_rig(rig: CameraRig, tracks: Any, intrinsics: Any,
               mode: Literal["translated", "rotation_only"],
               profile: QualityProfile | str = "balanced") -> RefinedRig:
    """Run fixed-order deterministic landmark/pose alternating refinement."""
    np = _require_numpy()
    if mode not in ("translated", "rotation_only"):
        raise ValueError(f"unsupported selected mode {mode!r}")
    selected_profile = _resolve_profile(profile)
    delta = float(selected_profile.reprojection_threshold_px)
    ordered_tracks = tuple(sorted(tuple(tracks), key=lambda value: value.track_id))
    intrinsics_tuple = tuple(intrinsics)
    if len(rig.rotations) != len(rig.translations) or len(rig.rotations) != len(intrinsics_tuple):
        raise ValueError("rig poses and intrinsics must have the same frame count")
    matrices = tuple(_intrinsic_matrix(value) for value in intrinsics_tuple)
    rotations = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in rig.rotations)
    translations = tuple(np.asarray(value, dtype=np.float64) for value in rig.translations)
    if mode == "rotation_only":
        translations = tuple(np.zeros(3, dtype=np.float64) for _ in translations)
        # Rotation-only tracks carry directions, not finite landmarks.  Map each
        # anchor ray to a distant point so the same pose-only equations apply.
        directional_landmarks: dict[int, Any] = {}
        accepted: list[FeatureTrack] = []
        inverse_anchor = np.linalg.inv(matrices[0])
        for track in ordered_tracks:
            anchor = next((item for item in track.observations if item.frame_index == 0), None)
            if anchor is None:
                continue
            ray = inverse_anchor @ np.array((*anchor.point_xy, 1.0), dtype=np.float64)
            ray /= ray[2]
            errors = _track_errors(track, ray, rotations, translations, matrices)
            if errors and max(errors) <= 4.0 * delta:
                accepted.append(track)
                directional_landmarks[track.track_id] = ray
        for damping in _REFINEMENT_DAMPING:
            rotation_list, translation_list = list(rotations), list(translations)
            for frame_index in range(1, len(rotations)):
                observations = _pose_observations(
                    frame_index, tuple(accepted), directional_landmarks,
                )
                rotation_list[frame_index], translation_list[frame_index] = _pose_step(
                    frame_index, observations,
                    rotation_list[frame_index], np.zeros(3, dtype=np.float64),
                    matrices[frame_index], delta, damping, rotation_only=True,
                )
            rotations, translations = tuple(rotation_list), tuple(translation_list)
        accepted_tracks = tuple(accepted)
        landmarks_by_id = directional_landmarks
    else:
        accepted_tracks_list: list[FeatureTrack] = []
        landmarks_by_id: dict[int, Any] = {}
        for track in ordered_tracks:
            point = _triangulate_track(track, rotations, translations, matrices)
            if point is None:
                continue
            errors = _track_errors(track, point, rotations, translations, matrices)
            if errors and max(errors) <= 4.0 * delta:
                accepted_tracks_list.append(track)
                landmarks_by_id[track.track_id] = point
        accepted_tracks = tuple(accepted_tracks_list)
        for damping in _REFINEMENT_DAMPING:
            retriangulated: dict[int, Any] = {}
            for track in accepted_tracks:
                point = _triangulate_track(track, rotations, translations, matrices)
                if point is not None:
                    retriangulated[track.track_id] = point
            accepted_tracks = tuple(
                track for track in accepted_tracks if track.track_id in retriangulated
            )
            landmarks_by_id = retriangulated
            rotation_list, translation_list = list(rotations), list(translations)
            for frame_index in range(1, len(rotations)):
                observations = _pose_observations(
                    frame_index, accepted_tracks, landmarks_by_id,
                )
                rotation_list[frame_index], translation_list[frame_index] = _pose_step(
                    frame_index, observations,
                    rotation_list[frame_index], translation_list[frame_index],
                    matrices[frame_index], delta, damping,
                )
            rotations, translations = tuple(rotation_list), tuple(translation_list)
        final_landmarks: dict[int, Any] = {}
        for track in accepted_tracks:
            point = _triangulate_track(track, rotations, translations, matrices)
            if point is not None:
                final_landmarks[track.track_id] = point
        accepted_tracks = tuple(
            track for track in accepted_tracks if track.track_id in final_landmarks
        )
        landmarks_by_id = final_landmarks

    accepted_ids = tuple(track.track_id for track in accepted_tracks)
    landmark_array = np.asarray(
        [landmarks_by_id[track_id] for track_id in accepted_ids], dtype=np.float64,
    ).reshape((-1, 3))
    all_errors: list[float] = []
    for track in accepted_tracks:
        all_errors.extend(_track_errors(
            track, landmarks_by_id[track.track_id], rotations, translations, matrices,
        ))
    rmse = (
        math.sqrt(float(np.mean(np.square(np.asarray(all_errors, dtype=np.float64)))))
        if all_errors else float("inf")
    )
    provisional = RefinedRig(
        rotations, translations, landmark_array, rmse,
        accepted_ids, ClosureMetrics(0.0, 0.0, float("inf")),
        _pair_evidence=rig._pair_evidence,
        _tracks=ordered_tracks,
        _intrinsics=intrinsics_tuple,
    )
    refined = replace(provisional, closure=measure_three_view_closure(provisional))
    # The closure weld needs an independent CLOSING pair (an edge not anchored
    # at photo 1).  Anchor-star rigs have none — their pose-vs-evidence drift
    # is the refinement doing its job, not an inconsistency to reject.
    has_closing_pair = not rig._pair_evidence or any(
        int(evidence.frame_a) != 0 for evidence in rig._pair_evidence
    )
    if mode == "translated" and len(rotations) >= 3 and has_closing_pair:
        rotation_limit, direction_limit, pixel_limit = _closure_limits(
            delta, rig._pair_evidence,
        )
        closure = refined.closure
        if (
            closure.rotation_error_deg > rotation_limit
            or closure.translation_direction_error_deg > direction_limit
            or closure.median_reprojection_px > pixel_limit
        ):
            raise MotionModelError(
                "inconsistent_third_view",
                "three-view closure exceeded deterministic limits "
                f"(rotation {closure.rotation_error_deg:.2f}deg vs {rotation_limit:.2f}, "
                f"direction {closure.translation_direction_error_deg:.2f}deg vs "
                f"{direction_limit:.2f}, pixel {closure.median_reprojection_px:.2f}px "
                f"vs {pixel_limit:.2f})",
            )
    return refined


__all__ = [
    "CameraRig", "ClosureMetrics", "FeatureObservation", "FeatureTrack",
    "MotionModelError", "RefinedRig", "build_tracks", "fit_pair_models",
    "initialise_rig", "measure_three_view_closure", "refine_rig",
    "select_capture_mode",
]
