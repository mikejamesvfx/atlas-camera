"""Deterministic calibrated two-view geometry tests."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from atlas_camera.core.multiview_geometry import (
    MotionModelError,
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


def test_mode_selection_accepts_legacy_evidence_without_grid_diagnostic() -> None:
    # Catches a Task 3 field addition breaking callers built against Task 1.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=7.0, translation=(0.8, 0.0, 0.1), outliers=30,
    )
    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "34" * 32,
    )

    assert select_capture_mode(
        replace(evidence, essential_occupied_grid_cells=-1),
        "translated", QUALITY_PROFILES["balanced"],
    ) == "translated"


def test_mode_selection_accepts_legacy_homography_residual_field() -> None:
    # Catches a Task 3 field addition breaking rotation-only classification.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=24.0, translation=(0.0, 0.0, 0.0),
    )
    evidence = fit_pair_models(
        matches, intr_a, intr_b, MultiViewSettings(), "56" * 32,
    )

    assert select_capture_mode(
        replace(evidence, homography_rotation_residual_px=None),
        "rotation_only", QUALITY_PROFILES["balanced"],
    ) == "rotation_only"
