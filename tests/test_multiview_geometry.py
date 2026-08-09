"""Deterministic calibrated two-view geometry tests."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import math

import numpy as np
import pytest

from atlas_camera.core.multiview_geometry import (
    MotionModelError,
    _decompose_essential,
    _hartley_normalize,
    _model_sample_schedules,
    _sample_schedule,
    _select_pose_candidate_index,
    _sampson_errors_px,
    fit_pair_models,
    select_capture_mode,
)
from atlas_camera.core.multiview_types import (
    MultiViewSettings,
    PairMatches,
    QUALITY_PROFILES,
)
from atlas_camera.core.schema import AtlasIntrinsics


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


def _project(points_xyz: np.ndarray, rotation: np.ndarray,
             translation: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    camera_points = (rotation @ points_xyz.T).T + translation
    homogeneous = (intrinsic @ camera_points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _project_known_scene(*, rotation_y_deg: float,
                         translation: tuple[float, float, float],
                         outliers: int = 0) -> tuple[
                             PairMatches, AtlasIntrinsics, AtlasIntrinsics,
                         ]:
    """Return literal-camera synthetic evidence with deterministic outliers."""
    rng = np.random.Generator(np.random.PCG64(20260809))
    width, height = 1280, 720
    intr_a = AtlasIntrinsics(
        image_width=width, image_height=height,
        fx_px=900.0, fy_px=880.0, cx_px=640.0, cy_px=360.0,
    )
    intr_b = AtlasIntrinsics(
        image_width=width, image_height=height,
        fx_px=930.0, fy_px=910.0, cx_px=632.0, cy_px=354.0,
    )
    matrix_a = np.array([
        [900.0, 0.0, 640.0],
        [0.0, 880.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    matrix_b = np.array([
        [930.0, 0.0, 632.0],
        [0.0, 910.0, 354.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    points_xyz = np.column_stack((
        rng.uniform(-3.0, 3.0, 120),
        rng.uniform(-1.8, 1.8, 120),
        rng.uniform(6.0, 14.0, 120),
    ))
    points_a = _project(points_xyz, np.eye(3), np.zeros(3), matrix_a)
    points_b = _project(
        points_xyz, _rotation_y(rotation_y_deg),
        np.asarray(translation, dtype=np.float64), matrix_b,
    )
    if outliers:
        points_a = np.vstack((points_a, np.column_stack((
            rng.uniform(0.0, width - 1.0, outliers),
            rng.uniform(0.0, height - 1.0, outliers),
        ))))
        points_b = np.vstack((points_b, np.column_stack((
            rng.uniform(0.0, width - 1.0, outliers),
            rng.uniform(0.0, height - 1.0, outliers),
        ))))
    count = len(points_a)
    matches = PairMatches(
        frame_a=0,
        frame_b=1,
        points_a=points_a.astype(np.float32),
        points_b=points_b.astype(np.float32),
        indices=np.column_stack((np.arange(count), np.arange(count))).astype(np.int64),
        distances=np.zeros(count, dtype=np.float32),
        occupied_grid_cells=16,
    )
    return matches, intr_a, intr_b


def test_translated_pair_selects_essential_model_exactly() -> None:
    # Catches a solver that cannot recover a supported baseline amid outliers.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=7.0, translation=(0.8, 0.0, 0.1), outliers=30,
    )

    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "ab" * 32,
    )

    assert select_capture_mode(
        evidence, "auto", QUALITY_PROFILES["balanced"],
    ) == "translated"
    assert evidence.essential_inlier_count >= 80
    assert evidence.median_triangulation_angle_deg >= 1.0
    assert evidence.positive_depth_fraction >= 0.75
    assert np.allclose(
        evidence.relative_rotation.T @ evidence.relative_rotation,
        np.eye(3), atol=1.0e-12,
    )
    assert np.linalg.det(evidence.relative_rotation) == pytest.approx(
        1.0, abs=1.0e-12,
    )


def test_shared_centre_pair_selects_rotation_only() -> None:
    # Catches invented translation for an exact shared-optical-centre capture.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=24.0, translation=(0.0, 0.0, 0.0),
    )

    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "cd" * 32,
    )

    assert select_capture_mode(
        evidence, "auto", QUALITY_PROFILES["balanced"],
    ) == "rotation_only"
    assert evidence.homography_inlier_count == 120


def test_forced_translated_rejects_rotation_only_evidence() -> None:
    # Catches forced mode silently accepting unobservable translation.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=24.0, translation=(0.0, 0.0, 0.0),
    )
    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "ef" * 32,
    )

    with pytest.raises(MotionModelError, match="degenerate_geometry") as caught:
        select_capture_mode(
            evidence, "translated", QUALITY_PROFILES["balanced"],
        )

    assert caught.value.outcome_code == "degenerate_geometry"


def test_pair_model_fit_ignores_ambient_numpy_random_state() -> None:
    # Catches accidental use of NumPy's process-global random generator.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=7.0, translation=(0.8, 0.0, 0.1), outliers=30,
    )
    settings = MultiViewSettings(seed=91)
    np.random.seed(1)
    first = fit_pair_models(matches, intr_a, intr_b, settings, "12" * 32)
    np.random.seed(999999)
    np.random.random(50_000)
    second = fit_pair_models(matches, intr_a, intr_b, settings, "12" * 32)

    for field_name in (
        "essential_matrix", "homography", "relative_rotation",
        "translation_direction", "essential_inliers", "homography_inliers",
    ):
        assert np.array_equal(
            getattr(first, field_name), getattr(second, field_name), equal_nan=True,
        )
    assert first.essential_inlier_count == second.essential_inlier_count
    assert first.homography_inlier_count == second.homography_inlier_count
    assert first.median_essential_error_px == second.median_essential_error_px
    assert first.median_homography_error_px == second.median_homography_error_px
    assert (
        first.median_triangulation_angle_deg
        == second.median_triangulation_angle_deg
    )
    assert first.positive_depth_fraction == second.positive_depth_fraction


@pytest.mark.parametrize(("requested_mode", "outcome_code"), (
    ("auto", "ambiguous_motion_model"),
    ("translated", "degenerate_geometry"),
))
def test_mode_selection_fails_closed_without_spatial_evidence(
    requested_mode: str, outcome_code: str,
) -> None:
    # Catches unavailable grid evidence being promoted to the profile minimum.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=7.0, translation=(0.8, 0.0, 0.1), outliers=30,
    )
    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "34" * 32,
    )

    missing_spatial_evidence = replace(
        evidence, essential_occupied_grid_cells=-1,
    )

    with pytest.raises(MotionModelError, match=outcome_code) as caught:
        select_capture_mode(
            missing_spatial_evidence, requested_mode,
            QUALITY_PROFILES["balanced"],
        )

    assert caught.value.outcome_code == outcome_code


def test_mode_selection_fails_closed_without_rotation_specific_residual() -> None:
    # Catches a planar homography transfer error masquerading as pure rotation.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=24.0, translation=(0.0, 0.0, 0.0),
    )
    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "56" * 32,
    )

    legacy = replace(
        evidence,
        essential_matrix=None,
        relative_rotation=None,
        translation_direction=None,
        essential_inlier_count=0,
        homography_rotation_residual_px=None,
        median_homography_error_px=0.0,
    )

    with pytest.raises(MotionModelError, match="ambiguous_motion_model"):
        select_capture_mode(
            legacy, "rotation_only", QUALITY_PROFILES["balanced"],
        )


def test_sampson_error_is_measured_in_anisotropic_pixel_coordinates() -> None:
    # Catches scalar-average focal conversion changing the inlier threshold.
    fundamental = np.array([
        [-3.97637431e-08, -6.66666667e-07, 6.61556023e-04],
        [6.55957920e-08, 0.0, -4.70874428e-04],
        [-6.90384399e-05, 2.74666667e-03, -1.05400660],
    ], dtype=np.float64)
    points_a = np.array([[826.6666666666666, 340.0]], dtype=np.float64)
    points_b = np.array([[711.27615123, 266.04341422]], dtype=np.float64)

    errors = _sampson_errors_px(fundamental, points_a, points_b)

    assert errors[0] == pytest.approx(0.9003025532, abs=1.0e-6)


def test_collinear_correspondences_produce_no_minimal_model() -> None:
    # Catches rank-deficient minimal samples entering consensus scoring.
    intrinsics = AtlasIntrinsics(
        image_width=1280, image_height=720,
        fx_px=900.0, fy_px=880.0, cx_px=640.0, cy_px=360.0,
    )
    x_values = np.linspace(80.0, 1200.0, 64, dtype=np.float64)
    points_a = np.column_stack((x_values, 0.25 * x_values + 40.0))
    points_b = np.column_stack((1.03 * x_values + 12.0, 0.2575 * x_values + 51.0))
    matches = PairMatches(
        0, 1, points_a, points_b,
        np.column_stack((np.arange(64), np.arange(64))),
        np.zeros(64, dtype=np.float64), 4,
    )

    evidence = fit_pair_models(
        matches, intrinsics, intrinsics, MultiViewSettings(), "78" * 32,
    )

    assert evidence.essential_matrix is None
    assert evidence.homography is None


def test_hartley_normalization_uses_mean_euclidean_radius() -> None:
    # Catches RMS-radius normalization, which is not Hartley's construction.
    points = np.array([
        [-4.0, -1.0],
        [-1.0, 0.0],
        [2.0, 2.0],
        [9.0, 8.0],
    ], dtype=np.float64)

    normalized, _ = _hartley_normalize(points)

    assert np.mean(normalized, axis=0) == pytest.approx((0.0, 0.0), abs=1.0e-12)
    assert np.mean(np.linalg.norm(normalized, axis=1)) == pytest.approx(
        math.sqrt(2.0), abs=1.0e-12,
    )


def test_sample_schedules_pin_budgets_uniqueness_and_combination_caps() -> None:
    # Catches iteration-budget drift, duplicate samples, or off-by-one caps.
    essential, homography = _model_sample_schedules(
        20, MultiViewSettings(), "aa" * 32,
    )
    capped_essential = _sample_schedule(9, 8, 2_048, "aa" * 32, 0, "essential")
    capped_homography = _sample_schedule(5, 4, 1_024, "aa" * 32, 0, "homography")

    assert len(essential) == len(set(essential)) == 2_048
    assert len(homography) == len(set(homography)) == 1_024
    assert capped_essential == tuple(combinations(range(9), 8))
    assert capped_homography == tuple(combinations(range(5), 4))


def test_sample_schedule_seed_material_is_fully_separated() -> None:
    # Catches omission of fingerprint, exposed seed, or model name from SHA256.
    baseline = _sample_schedule(20, 8, 32, "ab" * 32, 7, "essential")

    assert baseline == _sample_schedule(20, 8, 32, "ab" * 32, 7, "essential")
    assert baseline != _sample_schedule(20, 8, 32, "cd" * 32, 7, "essential")
    assert baseline != _sample_schedule(20, 8, 32, "ab" * 32, 8, "essential")
    assert baseline != _sample_schedule(20, 8, 32, "ab" * 32, 7, "homography")


def test_four_pose_selection_orders_cheirality_error_then_index() -> None:
    # Catches a reordering of the load-bearing four-pose winner tuple.
    positive_depth_counts = (99, 100, 100, 100)
    median_reprojection_errors = (0.01, 0.3, 0.2, 0.2)

    assert _select_pose_candidate_index(
        positive_depth_counts, median_reprojection_errors,
    ) == 2


def test_four_pose_exact_tie_selects_first_decomposition_candidate() -> None:
    # Catches unstable candidate choice when cheirality and error both tie.
    essential = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    rotation, translation, _, _, _ = _decompose_essential(
        essential, np.empty((0, 2)), np.empty((0, 2)),
    )

    assert rotation == pytest.approx(np.eye(3), abs=1.0e-12)
    assert translation == pytest.approx((-1.0, 0.0, 0.0), abs=1.0e-12)
