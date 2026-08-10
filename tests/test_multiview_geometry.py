"""Deterministic calibrated two-view geometry tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math

import numpy as np
import pytest

import atlas_camera.core.multiview_geometry as geometry
from atlas_camera.core.multiview_geometry import (
    CameraRig,
    ClosureMetrics,
    FeatureObservation,
    FeatureTrack,
    MotionModelError,
    _decompose_essential,
    _hartley_normalize,
    _model_sample_schedules,
    _pose_step,
    _sample_schedule,
    _select_pose_candidate_index,
    _sampson_errors_px,
    build_tracks,
    fit_pair_models,
    initialise_rig,
    refine_rig,
    select_capture_mode,
)
from atlas_camera.core.multiview_types import (
    MultiViewSettings,
    PairMatches,
    PairModelEvidence,
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


@dataclass(frozen=True)
class _ThreeCameraFixture:
    pairs: tuple[PairMatches, ...]
    evidence: tuple[PairModelEvidence, ...]
    intrinsics: tuple[AtlasIntrinsics, ...]


def _relative_evidence(frame_a: int, frame_b: int,
                       rotations: tuple[np.ndarray, ...],
                       translations: tuple[np.ndarray, ...]) -> PairModelEvidence:
    relative_rotation = rotations[frame_b] @ rotations[frame_a].T
    relative_translation = (
        translations[frame_b]
        - relative_rotation @ translations[frame_a]
    )
    relative_translation /= np.linalg.norm(relative_translation)
    return PairModelEvidence(
        frame_a=frame_a,
        frame_b=frame_b,
        essential_matrix=np.eye(3, dtype=np.float64),
        homography=None,
        relative_rotation=relative_rotation,
        translation_direction=relative_translation,
        essential_inliers=np.ones(140, dtype=bool),
        homography_inliers=np.zeros(140, dtype=bool),
        essential_inlier_count=140,
        homography_inlier_count=0,
        median_essential_error_px=1.2,
        median_homography_error_px=float("inf"),
        median_triangulation_angle_deg=4.0,
        positive_depth_fraction=1.0,
        essential_occupied_grid_cells=16,
    )


def _three_camera_fixture(*, noise_px: float = 0.35, outliers: int = 0,
                          scramble_pair_1_2: bool = False) -> _ThreeCameraFixture:
    rng = np.random.Generator(np.random.PCG64(2026080917))
    width, height = 1280, 720
    intrinsics = tuple(
        AtlasIntrinsics(
            image_width=width, image_height=height,
            fx_px=900.0 + 15.0 * index,
            fy_px=880.0 + 12.0 * index,
            cx_px=640.0 - 3.0 * index,
            cy_px=360.0 + 2.0 * index,
        )
        for index in range(3)
    )
    matrices = tuple(_intrinsic_matrix_for_test(value) for value in intrinsics)
    rotations = (np.eye(3), _rotation_y(4.0), _rotation_y(-3.0))
    translations = (
        np.zeros(3, dtype=np.float64),
        np.array((0.75, 0.02, 0.10), dtype=np.float64),
        np.array((-0.60, 0.03, 0.20), dtype=np.float64),
    )
    points_xyz = np.column_stack((
        rng.uniform(-2.8, 2.8, 140),
        rng.uniform(-1.6, 1.6, 140),
        rng.uniform(7.0, 15.0, 140),
    ))
    image_points = tuple(
        _project(points_xyz, rotations[index], translations[index], matrices[index])
        + rng.normal(0.0, noise_px, size=(len(points_xyz), 2))
        for index in range(3)
    )
    pairs: list[PairMatches] = []
    for pair_number, (frame_a, frame_b) in enumerate(((0, 1), (0, 2), (1, 2))):
        points_a = image_points[frame_a].copy()
        points_b = image_points[frame_b].copy()
        if scramble_pair_1_2 and (frame_a, frame_b) == (1, 2):
            points_b = np.roll(points_b, 37, axis=0)
        indices = np.column_stack((np.arange(140), np.arange(140)))
        if outliers:
            first = 140 + pair_number * outliers
            outlier_indices = np.arange(first, first + outliers)
            points_a = np.vstack((points_a, rng.uniform(
                (0.0, 0.0), (width - 1.0, height - 1.0), size=(outliers, 2),
            )))
            points_b = np.vstack((points_b, rng.uniform(
                (0.0, 0.0), (width - 1.0, height - 1.0), size=(outliers, 2),
            )))
            indices = np.vstack((indices, np.column_stack((outlier_indices, outlier_indices))))
        pairs.append(PairMatches(
            frame_a, frame_b, points_a, points_b, indices,
            np.zeros(len(points_a), dtype=np.float64), 16,
        ))
    evidence = tuple(
        _relative_evidence(frame_a, frame_b, rotations, translations)
        for frame_a, frame_b in ((0, 1), (0, 2), (1, 2))
    )
    return _ThreeCameraFixture(tuple(pairs), evidence, intrinsics)


def _intrinsic_matrix_for_test(intrinsics: AtlasIntrinsics) -> np.ndarray:
    return np.array([
        [intrinsics.fx_px, 0.0, intrinsics.cx_px],
        [0.0, intrinsics.fy_px, intrinsics.cy_px],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _empty_three_camera_rig() -> tuple[CameraRig, tuple[AtlasIntrinsics, ...]]:
    intrinsics = tuple(
        AtlasIntrinsics(
            image_width=1280, image_height=720,
            fx_px=900.0, fy_px=880.0, cx_px=640.0, cy_px=360.0,
        )
        for _ in range(3)
    )
    return CameraRig(
        rotations=tuple(np.eye(3, dtype=np.float64) for _ in range(3)),
        translations=(
            np.zeros(3, dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
            np.array((-1.0, 0.0, 0.0), dtype=np.float64),
        ),
        landmarks=np.empty((0, 3), dtype=np.float64),
        reprojection_rmse_px=float("inf"),
    ), intrinsics


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


def test_three_view_tracks_close_and_refinement_reduces_error() -> None:
    # Catches unstable track joins, uncalibrated DLT, and non-improving BA.
    fixture = _three_camera_fixture(noise_px=0.35, outliers=45)

    tracks = build_tracks(fixture.pairs, n_frames=3)
    initial = initialise_rig(fixture.evidence, "translated")
    refined = refine_rig(
        initial, tracks, fixture.intrinsics, "translated",
    )

    assert refined.reprojection_rmse_px < initial.reprojection_rmse_px
    assert refined.closure.rotation_error_deg < 0.15
    assert refined.closure.translation_direction_error_deg < 0.5
    assert len(refined.accepted_track_ids) >= 120
    assert refined.accepted_track_ids == tuple(sorted(refined.accepted_track_ids))
    assert refined.landmarks.dtype == np.float64
    assert refined.rotations[0] == pytest.approx(np.eye(3), abs=0.0)
    assert refined.translations[0] == pytest.approx(np.zeros(3), abs=0.0)


def test_inconsistent_third_view_has_a_distinct_error() -> None:
    # Catches an independently inconsistent closing pair being silently ignored.
    fixture = _three_camera_fixture(scramble_pair_1_2=True)
    tracks = build_tracks(fixture.pairs, n_frames=3)
    initial = initialise_rig(fixture.evidence, "translated")

    with pytest.raises(MotionModelError, match="inconsistent_third_view") as caught:
        refine_rig(initial, tracks, fixture.intrinsics, "translated")

    assert caught.value.outcome_code == "inconsistent_third_view"


def test_incomplete_closing_pose_has_the_closure_failure_code() -> None:
    # Catches a partially populated closing edge leaking a low-level matrix error.
    fixture = _three_camera_fixture()
    incomplete = replace(fixture.evidence[2], relative_rotation=None)

    with pytest.raises(MotionModelError, match="inconsistent_third_view") as caught:
        initialise_rig((*fixture.evidence[:2], incomplete), "translated")

    assert caught.value.outcome_code == "inconsistent_third_view"


def test_track_builder_rejects_duplicate_frame_components_and_sorts_observations() -> None:
    # Catches union-find components that alias two features from the same frame.
    pair_0_1 = PairMatches(
        0, 1,
        np.array(((30.0, 20.0), (10.0, 10.0))),
        np.array(((31.0, 20.0), (11.0, 10.0))),
        np.array(((3, 7), (1, 5))), np.zeros(2), 2,
    )
    pair_0_2 = PairMatches(
        0, 2,
        np.array(((30.0, 20.0), (10.0, 10.0))),
        np.array(((32.0, 20.0), (12.0, 10.0))),
        np.array(((3, 9), (1, 8))), np.zeros(2), 2,
    )
    pair_1_2 = PairMatches(
        1, 2,
        np.array(((31.0, 20.0), (11.0, 10.0))),
        np.array(((32.0, 20.0), (12.0, 10.0))),
        # The second edge aliases a second frame-2 feature into the component
        # that already contains feature 8, so only the independent first track survives.
        np.array(((7, 9), (5, 10))), np.zeros(2), 2,
    )

    tracks = build_tracks((pair_1_2, pair_0_2, pair_0_1), n_frames=3)

    assert tracks == (
        FeatureTrack(0, (
            FeatureObservation(0, 3, (30.0, 20.0)),
            FeatureObservation(1, 7, (31.0, 20.0)),
            FeatureObservation(2, 9, (32.0, 20.0)),
        )),
    )


@pytest.mark.parametrize("closure_axis", ("rotation", "translation", "pixels"))
def test_valid_closed_tracks_exercise_each_three_view_closure_gate(
    closure_axis: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches closure checks that only reject an empty/invalid track set.
    fixture = _three_camera_fixture(noise_px=0.0)
    pairs = fixture.pairs
    evidence = fixture.evidence
    if closure_axis == "rotation":
        closing = replace(
            evidence[2],
            relative_rotation=_rotation_y(0.75) @ evidence[2].relative_rotation,
        )
        evidence = (*evidence[:2], closing)
    elif closure_axis == "translation":
        # The fixture's 4-degree parallax scales the direction limit to
        # 3.75 degrees (observability-aware closure, 2026-08-10) — perturb
        # beyond it so the gate still has something real to catch.
        closing = replace(
            evidence[2],
            translation_direction=(
                _rotation_y(4.5) @ evidence[2].translation_direction
            ),
        )
        evidence = (*evidence[:2], closing)
    else:
        offsets = np.column_stack((
            8.0 * np.where(np.arange(140) % 2 == 0, -1.0, 1.0),
            8.0 * np.where(np.arange(140) % 4 < 2, -1.0, 1.0),
        ))
        pairs = tuple(
            replace(pair, points_b=np.asarray(pair.points_b) + offsets)
            if pair.frame_b == 2 else pair
            for pair in pairs
        )
    tracks = build_tracks(pairs, n_frames=3)
    assert sum(len(track.observations) == 3 for track in tracks) == 140
    captured: list[ClosureMetrics] = []
    original_measure = geometry.measure_three_view_closure

    def capture_closure(refined: geometry.RefinedRig) -> ClosureMetrics:
        metrics = original_measure(refined)
        captured.append(metrics)
        return metrics

    monkeypatch.setattr(geometry, "measure_three_view_closure", capture_closure)
    initial = initialise_rig(evidence, "translated")

    with pytest.raises(MotionModelError, match="inconsistent_third_view") as caught:
        refine_rig(initial, tracks, fixture.intrinsics, "translated")

    assert caught.value.outcome_code == "inconsistent_third_view"
    assert captured
    if closure_axis == "rotation":
        assert captured[0].rotation_error_deg > 0.5
        assert captured[0].median_reprojection_px < 2.0
    elif closure_axis == "translation":
        # Limit is 3.75 deg here: 1.5 base x (10 / the fixture's 4-deg parallax).
        assert captured[0].translation_direction_error_deg > 3.75
        assert captured[0].median_reprojection_px < 2.0
    else:
        assert captured[0].median_reprojection_px > 2.0


def test_rotation_only_refinement_updates_rotation_and_pins_all_translations() -> None:
    # Catches a six-parameter update leaking translation into shared-centre capture.
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=24.0, translation=(0.0, 0.0, 0.0),
    )
    tracks = build_tracks((matches,), n_frames=2)
    initial_secondary = _rotation_y(23.8)
    rig = CameraRig(
        rotations=(np.eye(3, dtype=np.float64), initial_secondary.copy()),
        translations=(np.ones(3, dtype=np.float64), -np.ones(3, dtype=np.float64)),
        landmarks=np.empty((0, 3), dtype=np.float64),
        reprojection_rmse_px=10.0,
    )

    refined = refine_rig(
        rig, tracks, (intr_a, intr_b), "rotation_only",
    )

    assert all(np.array_equal(value, np.zeros(3)) for value in refined.translations)
    assert np.array_equal(refined.rotations[0], np.eye(3))
    assert not np.array_equal(refined.rotations[1], initial_secondary)
    initial_error = abs(24.0 - 23.8)
    refined_angle = math.degrees(math.atan2(
        refined.rotations[1][0, 2], refined.rotations[1][0, 0],
    ))
    assert abs(24.0 - refined_angle) < initial_error


def test_refinement_uses_exactly_eight_fixed_damping_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches iteration-count, damping-value, or secondary-camera ordering drift.
    rig, intrinsics = _empty_three_camera_rig()
    calls: list[tuple[int, float]] = []

    def record_step(
        frame_index: int, observations: list[tuple[object, object]],
        rotation: np.ndarray, translation: np.ndarray,
        intrinsic_matrix: np.ndarray, delta: float, damping: float,
        rotation_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((frame_index, damping))
        return rotation, translation

    monkeypatch.setattr(geometry, "_pose_step", record_step)
    monkeypatch.setattr(
        geometry, "measure_three_view_closure",
        lambda _: ClosureMetrics(0.0, 0.0, 0.0),
    )

    refine_rig(rig, (), intrinsics, "translated")

    damping = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-6)
    assert calls == [
        (frame_index, value)
        for value in damping
        for frame_index in (1, 2)
    ]


def test_pose_step_rejects_an_equal_cost_candidate() -> None:
    # Catches <= acceptance accidentally admitting a non-decreasing pose step.
    intrinsic = np.array([
        [900.0, 0.0, 640.0],
        [0.0, 880.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    points = [
        np.array((x_value, y_value, 5.0 + index), dtype=np.float64)
        for index, (x_value, y_value) in enumerate((
            (-1.0, -0.5), (-0.5, 0.7), (0.2, -0.8),
            (0.8, 0.6), (1.2, -0.1), (-1.3, 0.2),
        ))
    ]
    observations = []
    for index, point in enumerate(points):
        projected = intrinsic @ point
        pixel = projected[:2] / projected[2]
        observations.append((
            point, FeatureObservation(1, index, (float(pixel[0]), float(pixel[1]))),
        ))

    got_rotation, got_translation = _pose_step(
        1, observations, rotation, translation, intrinsic,
        delta=1.5, damping=1.0e-3,
    )

    assert got_rotation is rotation
    assert got_translation is translation


@pytest.mark.parametrize(("profile_name", "metric_name", "limit"), (
    ("conservative", "rotation", 0.5 * (1.0 / 1.5)),
    ("conservative", "translation", 1.5 * (1.0 / 1.5)),
    ("conservative", "pixels", 2.0 * (1.0 / 1.5)),
    ("balanced", "rotation", 0.5),
    ("balanced", "translation", 1.5),
    ("balanced", "pixels", 2.0),
    ("permissive", "rotation", 0.5 * (2.5 / 1.5)),
    ("permissive", "translation", 1.5 * (2.5 / 1.5)),
    ("permissive", "pixels", 2.0 * (2.5 / 1.5)),
))
def test_quality_profile_scales_each_closure_boundary(
    profile_name: str, metric_name: str, limit: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches a hard-coded balanced gate or >= rejection at the exact boundary.
    rig, intrinsics = _empty_three_camera_rig()
    current = [ClosureMetrics(0.0, 0.0, 0.0)]
    monkeypatch.setattr(
        geometry, "measure_three_view_closure", lambda _: current[0],
    )

    values = {"rotation": 0.0, "translation": 0.0, "pixels": 0.0}
    values[metric_name] = limit
    current[0] = ClosureMetrics(
        values["rotation"], values["translation"], values["pixels"],
    )
    refine_rig(rig, (), intrinsics, "translated", profile_name)

    values[metric_name] = math.nextafter(limit, math.inf)
    current[0] = ClosureMetrics(
        values["rotation"], values["translation"], values["pixels"],
    )
    with pytest.raises(MotionModelError, match="inconsistent_third_view") as caught:
        refine_rig(rig, (), intrinsics, "translated", profile_name)

    assert caught.value.outcome_code == "inconsistent_third_view"


def _project_planar_scene(*, rotation_y_deg: float = 4.0,
                          translation: tuple[float, float, float] = (0.8, 0.05, 0.0),
                          ) -> tuple[PairMatches, AtlasIntrinsics, AtlasIntrinsics,
                                     np.ndarray, np.ndarray]:
    """All landmarks on one plane (z = 8 + 0.15x): the essential-degenerate case."""
    rng = np.random.Generator(np.random.PCG64(20260810))
    width, height = 1280, 720
    intr = AtlasIntrinsics(
        image_width=width, image_height=height,
        fx_px=900.0, fy_px=900.0, cx_px=640.0, cy_px=360.0,
    )
    matrix = np.array([
        [900.0, 0.0, 640.0],
        [0.0, 900.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    xs = rng.uniform(-4.0, 4.0, 160)
    ys = rng.uniform(-2.5, 2.5, 160)
    zs = 8.0 + 0.15 * xs
    points_xyz = np.column_stack((xs, ys, zs))
    rotation = _rotation_y(rotation_y_deg)
    translation_vec = np.asarray(translation, dtype=np.float64)
    points_a = _project(points_xyz, np.eye(3), np.zeros(3), matrix)
    points_b = _project(points_xyz, rotation, translation_vec, matrix)
    count = len(points_a)
    matches = PairMatches(
        frame_a=0, frame_b=1,
        points_a=points_a.astype(np.float32),
        points_b=points_b.astype(np.float32),
        indices=np.column_stack((np.arange(count), np.arange(count))).astype(np.int64),
        distances=np.zeros(count, dtype=np.float32),
        occupied_grid_cells=16,
    )
    return matches, intr, intr, rotation, translation_vec


def test_planar_scene_decomposes_to_the_true_translated_pose() -> None:
    matches, intr_a, intr_b, true_rotation, true_translation = _project_planar_scene()
    settings = MultiViewSettings(match_quality="balanced")
    evidence = fit_pair_models(
        matches, intr_a, intr_b, settings,
        "planar-decomposition-fingerprint",
    )
    assert evidence.planar_rotation is not None
    assert evidence.planar_translation_direction is not None
    assert evidence.planar_plane_normal is not None
    assert evidence.planar_positive_depth_fraction > 0.9
    assert evidence.planar_median_triangulation_angle_deg > 1.0
    assert evidence.homography_occupied_grid_cells >= 6

    rotation_error = np.degrees(np.arccos(np.clip(
        (np.trace(evidence.planar_rotation @ true_rotation.T) - 1.0) / 2.0, -1.0, 1.0,
    )))
    assert rotation_error < 0.5

    direction = np.asarray(evidence.planar_translation_direction, dtype=np.float64)
    expected = true_translation / np.linalg.norm(true_translation)
    assert float(np.dot(direction, expected)) > 0.999

    # The recovered plane normal (camera A frame) matches z = 8 + 0.15x, whose
    # unit normal is (-0.15, 0, 1)/norm up to sign.
    normal = np.asarray(evidence.planar_plane_normal, dtype=np.float64)
    plane = np.array((-0.15, 0.0, 1.0))
    plane /= np.linalg.norm(plane)
    assert abs(float(np.dot(normal, plane))) > 0.999


def test_planar_translated_wins_auto_when_essential_is_degenerate() -> None:
    from atlas_camera.core.multiview_geometry import (
        _planar_translated_passes,
        select_capture_mode,
    )
    matches, intr_a, intr_b, _, _ = _project_planar_scene()
    settings = MultiViewSettings(match_quality="balanced")
    profile = QUALITY_PROFILES["balanced"]
    evidence = fit_pair_models(
        matches, intr_a, intr_b, settings,
        "planar-selection-fingerprint",
    )
    assert _planar_translated_passes(evidence, profile)
    mode = select_capture_mode(evidence, "auto", profile)
    assert mode in ("translated", "translated_planar")
    # Forcing translated must accept the planar interpretation too.
    forced = select_capture_mode(evidence, "translated", profile)
    assert forced in ("translated", "translated_planar")


def test_pure_rotation_still_selects_rotation_only_not_planar() -> None:
    from atlas_camera.core.multiview_geometry import select_capture_mode
    matches, intr_a, intr_b, _, _ = _project_planar_scene(
        rotation_y_deg=6.0, translation=(0.0, 0.0, 0.0),
    )
    settings = MultiViewSettings(match_quality="balanced")
    profile = QUALITY_PROFILES["balanced"]
    evidence = fit_pair_models(
        matches, intr_a, intr_b, settings,
        "pure-rotation-fingerprint",
    )
    assert select_capture_mode(evidence, "auto", profile) == "rotation_only"


def test_closure_direction_limit_scales_with_pair_observability() -> None:
    from atlas_camera.core.multiview_geometry import _closure_limits

    def evidence_with_angle(angle: float) -> PairModelEvidence:
        return PairModelEvidence(
            0, 1, None, None, None, None,
            np.zeros(1, dtype=bool), np.zeros(1, dtype=bool),
            0, 0, float("inf"), float("inf"), angle, 0.0,
        )

    delta = QUALITY_PROFILES["balanced"].reprojection_threshold_px

    # Daylight parallax (>= reference): the original limits, unchanged.
    rot, direction, px = _closure_limits(delta, [evidence_with_angle(20.0)])
    assert rot == pytest.approx(0.5)
    assert direction == pytest.approx(1.5)
    assert px == pytest.approx(2.0)

    # Tiny-baseline night pair (2 deg): direction limit x5, others fixed.
    rot, direction, px = _closure_limits(delta, [evidence_with_angle(2.0)])
    assert rot == pytest.approx(0.5)
    assert direction == pytest.approx(1.5 * 5.0)
    assert px == pytest.approx(2.0)

    # The WEAKEST pair governs.
    _, direction, _ = _closure_limits(
        delta, [evidence_with_angle(20.0), evidence_with_angle(2.0)],
    )
    assert direction == pytest.approx(1.5 * 5.0)

    # Zero/absent parallax cannot open the gate arbitrarily wide.
    _, direction, _ = _closure_limits(delta, [evidence_with_angle(0.0)])
    assert direction == pytest.approx(1.5)  # no measurable angles -> reference
    _, direction, _ = _closure_limits(delta, [evidence_with_angle(0.5)])
    assert direction == pytest.approx(1.5 * 10.0)  # clamped at x10

    # Planar-translated pairs contribute their planar parallax.
    planar = PairModelEvidence(
        0, 1, None, None, None, None,
        np.zeros(1, dtype=bool), np.zeros(1, dtype=bool),
        0, 0, float("inf"), float("inf"), 0.0, 0.0,
        planar_median_triangulation_angle_deg=25.0,
    )
    _, direction, _ = _closure_limits(delta, [planar])
    assert direction == pytest.approx(1.5)
