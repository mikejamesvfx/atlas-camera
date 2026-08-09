"""Deterministic calibrated two-view motion-model fitting.

NumPy is optional at package-import time and is loaded only when geometry is
requested.  Candidate schedules use a private PCG64 generator derived from the
registration fingerprint; no OpenCV estimator or process-global RNG is used.
"""

from __future__ import annotations

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
        f"rotation_residual_px={rotation_residual_text})"
    )


def select_capture_mode(evidence: PairModelEvidence, requested_mode: CaptureMode,
                        profile: QualityProfile) -> Literal["translated", "rotation_only"]:
    """Select a supported interpretation or fail with an exact outcome code."""
    if requested_mode not in ("auto", "translated", "rotation_only"):
        raise ValueError(f"unsupported capture mode {requested_mode!r}")
    translated = _translated_passes(evidence, profile)
    rotation_only = _rotation_only_passes(evidence, profile)
    summary = _score_summary(evidence)
    if requested_mode == "translated":
        if translated:
            return "translated"
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
    if rotation_only:
        return "rotation_only"
    raise MotionModelError(
        "ambiguous_motion_model", f"neither motion model passed checks; {summary}",
    )


__all__ = ["MotionModelError", "fit_pair_models", "select_capture_mode"]
