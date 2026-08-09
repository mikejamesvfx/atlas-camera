"""Public orchestration tests for deterministic photographed camera rigs."""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

import atlas_camera.core.multiview_features as features  # noqa: E402
import atlas_camera.core.multiview_geometry as geometry  # noqa: E402
from atlas_camera.core.multiview_geometry import ClosureMetrics, RefinedRig  # noqa: E402
from atlas_camera.core.multiview_solver import (  # noqa: E402
    solve_multiview,
    validate_multiview_frames,
)
from atlas_camera.core.multiview_types import (  # noqa: E402
    FeatureSet,
    MultiViewFrame,
    MultiViewSettings,
    PairMatches,
    PairModelEvidence,
)
from atlas_camera.core.schema import AtlasPlateRef  # noqa: E402
from atlas_camera.raw.pipeline import RawImportResult  # noqa: E402


_WIDTH = 1280
_HEIGHT = 720
_FOCAL_MM = 35.0
_SENSOR_WIDTH_MM = 36.0
_SENSOR_HEIGHT_MM = 24.0
_FX = _FOCAL_MM * _WIDTH / _SENSOR_WIDTH_MM


def _frame(
    *,
    focal: float | None = _FOCAL_MM,
    sensor_width: float | None = _SENSOR_WIDTH_MM,
    sensor_height: float | None = _SENSOR_HEIGHT_MM,
    make: str | None = "Atlas",
    model: str | None = "A1",
    lens: str | None = "Atlas Prime",
    orientation: int | None = 1,
    width: int = _WIDTH,
    height: int = _HEIGHT,
    undistort_status: str = "applied",
    body_serial: str | None = "BODY-1",
    lens_serial: str | None = "LENS-1",
    capture_datetime: str | None = "2026:08:09 12:00:00",
    label: str = "",
) -> MultiViewFrame:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    metadata = RawImportResult(
        linear_rgb=image,
        display_srgb=image,
        width=width,
        height=height,
        focal_length_mm=focal,
        sensor_width_mm=sensor_width,
        sensor_height_mm=sensor_height,
        sensor_source="camera_db",
        camera_make=make,
        camera_model=model,
        lens_model=lens,
        undistort_applied=True,
        undistort_status=undistort_status,
        orientation=orientation,
        body_serial_number=body_serial,
        lens_serial_number=lens_serial,
        capture_datetime=capture_datetime,
        metadata_source="embedded_jpeg",
    )
    name = label or "photo"
    plate_ref = AtlasPlateRef(
        image_path=f"C:/plates/{name}.exr",
        preview_b64=f"data:image/png;base64,{name}",
        colorspace="ACEScg",
        bit_depth="float32",
        role="source",
        is_proxy=False,
    )
    return MultiViewFrame(image=image, raw_meta=metadata, plate_ref=plate_ref, label=name)


def _anchor_vp_result() -> dict[str, object]:
    return {
        "vp1": np.array((_WIDTH / 2.0 - _FX, _HEIGHT / 2.0)),
        "vp2": np.array((_WIDTH / 2.0 + _FX, _HEIGHT / 2.0)),
        "vp3": np.array((_WIDTH / 2.0, -2000.0)),
        "lines": np.empty((0, 4), dtype=np.float64),
        "left_lines": np.empty((0, 4), dtype=np.float64),
        "right_lines": np.empty((0, 4), dtype=np.float64),
        "vertical_lines": np.empty((0, 4), dtype=np.float64),
        "num_lines_total": 0,
        "image_size": (_HEIGHT, _WIDTH),
    }


def _camera_to_world_from_angles(
    pitch_deg: float, roll_deg: float, yaw_deg: float = 32.0,
) -> np.ndarray:
    pitch, roll, yaw = map(math.radians, (pitch_deg, roll_deg, yaw_deg))
    pitch_rotation = np.array((
        (1.0, 0.0, 0.0),
        (0.0, math.cos(pitch), -math.sin(pitch)),
        (0.0, math.sin(pitch), math.cos(pitch)),
    ))
    roll_rotation = np.array((
        (math.cos(roll), -math.sin(roll), 0.0),
        (math.sin(roll), math.cos(roll), 0.0),
        (0.0, 0.0, 1.0),
    ))
    yaw_rotation = np.array((
        (math.cos(yaw), 0.0, math.sin(yaw)),
        (0.0, 1.0, 0.0),
        (-math.sin(yaw), 0.0, math.cos(yaw)),
    ))
    return yaw_rotation @ pitch_rotation @ roll_rotation


def _project_direction_to_vanishing_point(
    world_direction: tuple[float, float, float],
    camera_to_world: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    camera_direction = camera_to_world.T @ np.asarray(world_direction)
    assert abs(float(camera_direction[2])) > 1.0e-6
    return np.array((
        cx - fx * camera_direction[0] / camera_direction[2],
        cy + fy * camera_direction[1] / camera_direction[2],
    ))


def _feature_set(count: int) -> FeatureSet:
    x_values = np.linspace(40.0, _WIDTH - 40.0, count, dtype=np.float32)
    y_values = np.linspace(430.0, _HEIGHT - 30.0, count, dtype=np.float32)
    points = np.column_stack((x_values, y_values))
    return FeatureSet(
        points_xy=points,
        descriptors=np.zeros((count, 128), dtype=np.float32),
        responses=np.ones(count, dtype=np.float32),
        stable_indices=np.arange(count, dtype=np.int64),
        image_size=(_WIDTH, _HEIGHT),
    )


def _matches(frame_a: int, frame_b: int, count: int, *, clustered: bool = False) -> PairMatches:
    if clustered:
        x_values = np.linspace(80.0, 120.0, count, dtype=np.float32)
        y_values = np.linspace(450.0, 490.0, count, dtype=np.float32)
    else:
        x_values = np.linspace(40.0, _WIDTH - 40.0, count, dtype=np.float32)
        y_values = np.linspace(430.0, _HEIGHT - 30.0, count, dtype=np.float32)
    base_points = np.column_stack((x_values, y_values))
    points_a = base_points + np.array(
        (7.0 * frame_a, 0.5 * frame_a), dtype=np.float32,
    )
    points_b = base_points + np.array(
        (7.0 * frame_b, 0.5 * frame_b), dtype=np.float32,
    )
    indices = np.column_stack((np.arange(count), np.arange(count))).astype(np.int64)
    return PairMatches(
        frame_a, frame_b, points_a, points_b, indices,
        np.zeros(count, dtype=np.float32), 1 if clustered else 16,
    )


def _evidence(matches: PairMatches, mode: str, *, clustered: bool = False) -> PairModelEvidence:
    count = len(matches.indices)
    if mode == "translated":
        essential_inliers = np.ones(count, dtype=bool)
        homography_inliers = np.zeros(count, dtype=bool)
        return PairModelEvidence(
            matches.frame_a,
            matches.frame_b,
            np.eye(3, dtype=np.float64),
            None,
            np.eye(3, dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
            essential_inliers,
            homography_inliers,
            count,
            0,
            0.2,
            float("inf"),
            4.0,
            1.0,
            1 if clustered else 16,
            None,
        )
    return PairModelEvidence(
        matches.frame_a,
        matches.frame_b,
        None,
        np.eye(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        None,
        np.zeros(count, dtype=bool),
        np.ones(count, dtype=bool),
        0,
        count,
        float("inf"),
        0.2,
        0.0,
        0.0,
        0,
        0.2,
    )


def _ground_landmarks(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    x_values = (np.mod(indices, 10.0) - 4.5) * 0.5
    z_values = 5.0 + np.floor(indices / 10.0) * 1.25
    # RefinedRig landmarks are OpenCV-local (+Y down); the solver adapter
    # converts them to Atlas camera coordinates before applying the anchor.
    return np.column_stack((x_values, np.full(count, 1.0), z_values))


def _install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "translated",
    match_count: int = 60,
    clustered: bool = False,
    landmarks: np.ndarray | None = None,
) -> None:
    monkeypatch.setattr(
        "atlas_camera.core.multiview_solver.VanishingPointDetector.detect_vanishing_points",
        lambda _image, *, random_seed: _anchor_vp_result(),
    )
    monkeypatch.setattr(features, "extract_features", lambda _image, _profile: _feature_set(match_count))
    monkeypatch.setattr(
        features,
        "match_features",
        lambda _a, _b, _profile, frame_a, frame_b: _matches(
            frame_a, frame_b, match_count, clustered=clustered,
        ),
    )
    monkeypatch.setattr(
        features,
        "render_match_overlay",
        lambda _a, _b, pair, _mask=None: np.full(
            (4, 4, 3), pair.frame_b, dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        geometry,
        "fit_pair_models",
        lambda pair, _intr_a, _intr_b, _settings, _fingerprint: _evidence(
            pair, mode, clustered=clustered,
        ),
    )

    def fake_refine(rig, tracks, intrinsics, selected_mode, profile):
        frame_count = len(intrinsics)
        rotations = tuple(np.eye(3, dtype=np.float64) for _ in range(frame_count))
        translations = tuple(
            np.zeros(3, dtype=np.float64)
            if index == 0 or selected_mode == "rotation_only"
            else np.array((float(index), 0.0, 0.0), dtype=np.float64)
            for index in range(frame_count)
        )
        selected_landmarks = (
            np.empty((0, 3), dtype=np.float64)
            if selected_mode == "rotation_only"
            else _ground_landmarks(match_count) if landmarks is None else landmarks
        )
        accepted = tuple(track.track_id for track in tracks[: len(selected_landmarks)])
        return RefinedRig(
            rotations,
            translations,
            selected_landmarks,
            0.25,
            accepted,
            ClosureMetrics(0.0, 0.0, 0.2),
            _pair_evidence=rig._pair_evidence,
            _tracks=tuple(tracks),
            _intrinsics=tuple(intrinsics),
        )

    monkeypatch.setattr(geometry, "refine_rig", fake_refine)


def test_metadata_mismatch_fails_before_feature_extraction(monkeypatch) -> None:
    frames = [_frame(focal=23.0, label="one"), _frame(focal=35.0, label="two")]
    monkeypatch.setattr(features, "extract_features", lambda *_: pytest.fail("must not extract"))

    out = solve_multiview(frames, MultiViewSettings())

    assert out.solve is None
    assert out.diagnostics.outcome_code == "metadata_mismatch"
    assert {item["field"] for item in out.diagnostics.metadata_checks} >= {"focal_length_mm"}


def test_validation_collects_all_mismatches_and_capture_time_is_diagnostic_only() -> None:
    frames = [
        _frame(label="anchor", body_serial=None, lens_serial=None),
        _frame(
            label="second",
            make=" atlas  ",
            model="A1",
            lens="Different Lens",
            focal=35.2,
            sensor_width=35.8,
            orientation=6,
            width=640,
            undistort_status="disabled",
            body_serial="BODY-2",
            lens_serial="LENS-2",
            capture_datetime="2026:08:09 12:05:00",
        ),
        _frame(
            label="third",
            body_serial="BODY-3",
            lens_serial="LENS-3",
            capture_datetime="2026:08:09 11:55:00",
        ),
    ]

    diagnostics = validate_multiview_frames(frames, MultiViewSettings())

    assert diagnostics is not None
    assert diagnostics.outcome_code == "metadata_mismatch"
    fields = [item["field"] for item in diagnostics.metadata_checks]
    assert {"lens_model", "focal_length_mm", "sensor_width_mm", "orientation",
            "developed_dimensions", "undistort_status", "body_serial_number",
            "lens_serial_number"} <= set(fields)
    times = [
        item["value"] for item in diagnostics.metadata_checks
        if item["field"] == "capture_datetime"
    ]
    assert times == [
        "2026:08:09 12:00:00",
        "2026:08:09 12:05:00",
        "2026:08:09 11:55:00",
    ]


@pytest.mark.parametrize(
    ("serial_field", "first", "second"),
    (
        ("body_serial_number", "BODY-AbC-123", "BODY-aBc-123"),
        ("lens_serial_number", "LENS-XyZ-987", "LENS-xYz-987"),
    ),
)
def test_validation_treats_case_distinct_serials_as_different_devices(
    serial_field: str, first: str, second: str,
) -> None:
    first_kwargs = {
        "body_serial": first if serial_field == "body_serial_number" else "BODY-1",
        "lens_serial": first if serial_field == "lens_serial_number" else "LENS-1",
    }
    second_kwargs = {
        "body_serial": second if serial_field == "body_serial_number" else "BODY-1",
        "lens_serial": second if serial_field == "lens_serial_number" else "LENS-1",
    }

    diagnostics = validate_multiview_frames(
        [
            _frame(label="one", **first_kwargs),
            _frame(label="two", **second_kwargs),
        ],
        MultiViewSettings(),
    )

    assert diagnostics is not None
    assert diagnostics.outcome_code == "metadata_mismatch"
    assert serial_field in {
        item["field"] for item in diagnostics.metadata_checks
    }


@pytest.mark.parametrize(("frame_count", "summary_text"), ((1, "two or three"), (4, "two or three")))
def test_validation_requires_two_or_three_frames(frame_count: int, summary_text: str) -> None:
    diagnostics = validate_multiview_frames(
        [_frame(label=f"p{index}") for index in range(frame_count)],
        MultiViewSettings(),
    )

    assert diagnostics is not None
    assert diagnostics.outcome_code == "metadata_mismatch"
    assert summary_text in diagnostics.summary


def test_missing_positive_trusted_intrinsics_are_metadata_mismatch() -> None:
    diagnostics = validate_multiview_frames(
        [_frame(label="one"), _frame(label="two", focal=0.0, sensor_width=None)],
        MultiViewSettings(),
    )

    assert diagnostics is not None
    assert diagnostics.outcome_code == "metadata_mismatch"
    assert {item["field"] for item in diagnostics.metadata_checks} >= {
        "focal_length_mm", "sensor_width_mm",
    }


def test_missing_anchor_vanishing_points_never_guesses_world_up(monkeypatch) -> None:
    monkeypatch.setattr(
        "atlas_camera.core.multiview_solver.VanishingPointDetector.detect_vanishing_points",
        lambda _image, *, random_seed: {"vp1": None, "vp2": None},
    )
    monkeypatch.setattr(features, "extract_features", lambda *_: pytest.fail("must not extract"))

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(capture_mode="rotation_only"),
    )

    assert out.solve is None
    assert out.diagnostics.outcome_code == "degenerate_geometry"
    assert "architectural lines" in out.diagnostics.summary
    assert "artist constraints" in out.diagnostics.summary


@pytest.mark.parametrize(("pitch_deg", "roll_deg"), ((-15.0, 0.0), (-12.0, 8.0)))
def test_anchor_horizon_matches_returned_extrinsics_with_asymmetric_focal_axes(
    monkeypatch, pitch_deg: float, roll_deg: float,
) -> None:
    fx, fy = 1200.0, 800.0
    sensor_width = _FOCAL_MM * _WIDTH / fx
    sensor_height = _FOCAL_MM * _HEIGHT / fy
    known_camera_to_world = _camera_to_world_from_angles(pitch_deg, roll_deg)
    vp_result = _anchor_vp_result()
    vp_result.update({
        "vp1": _project_direction_to_vanishing_point(
            (1.0, 0.0, 0.0), known_camera_to_world,
            fx=fx, fy=fy, cx=_WIDTH / 2.0, cy=_HEIGHT / 2.0,
        ),
        "vp2": _project_direction_to_vanishing_point(
            (0.0, 0.0, 1.0), known_camera_to_world,
            fx=fx, fy=fy, cx=_WIDTH / 2.0, cy=_HEIGHT / 2.0,
        ),
    })
    _install_pipeline(monkeypatch, mode="rotation_only")
    monkeypatch.setattr(
        "atlas_camera.core.multiview_solver.VanishingPointDetector.detect_vanishing_points",
        lambda _image, *, random_seed: vp_result,
    )

    out = solve_multiview(
        [
            _frame(
                label="one", sensor_width=sensor_width,
                sensor_height=sensor_height,
            ),
            _frame(
                label="two", sensor_width=sensor_width,
                sensor_height=sensor_height,
            ),
        ],
        MultiViewSettings(capture_mode="rotation_only"),
    )

    assert out.solve is not None
    intrinsics = out.solve.camera.intrinsics
    assert intrinsics.fx_px == pytest.approx(fx)
    assert intrinsics.fy_px == pytest.approx(fy)
    camera_to_world = np.asarray(
        out.solve.camera.extrinsics.camera_world_matrix,
        dtype=np.float64,
    )[:3, :3]
    up_camera = camera_to_world.T @ np.array((0.0, 1.0, 0.0))
    a, b, c = out.solve.horizon_line.line_coefficients
    for image_x in (0.0, _WIDTH / 2.0, float(_WIDTH)):
        horizon_from_vps = -(a * image_x + c) / b
        normalized_x = (image_x - intrinsics.cx_px) / intrinsics.fx_px
        horizon_from_pose = intrinsics.cy_px + intrinsics.fy_px * (
            up_camera[0] * normalized_x - up_camera[2]
        ) / up_camera[1]
        assert horizon_from_pose == pytest.approx(horizon_from_vps, abs=1.0e-8)


def test_insufficient_overlap_fails_before_model_fitting(monkeypatch) -> None:
    _install_pipeline(monkeypatch, match_count=47)
    monkeypatch.setattr(geometry, "fit_pair_models", lambda *_: pytest.fail("must not fit"))

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is None
    assert out.diagnostics.outcome_code == "insufficient_overlap"
    assert len(out.overlays) == 1


def test_clustered_consensus_reports_dynamic_scene_contamination(monkeypatch) -> None:
    _install_pipeline(monkeypatch, match_count=100, clustered=True)

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is None
    assert out.diagnostics.outcome_code == "dynamic_scene_contamination"
    assert out.diagnostics.pair_metrics[0]["consensus_bounding_box_px"] == pytest.approx(
        [80.0, 450.0, 120.0, 490.0],
    )


def test_translated_rig_is_scaled_to_measured_anchor_height(monkeypatch) -> None:
    _install_pipeline(monkeypatch)

    out = solve_multiview(
        [_frame(label="anchor"), _frame(label="second")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.diagnostics.outcome_code == "translated"
    assert out.solve is not None
    assert out.solve.camera.extrinsics.camera_position[1] == pytest.approx(1.43)
    assert out.solve.debug_metadata["scale_source"] == "measured_camera_height"
    assert out.solve.source_plate.image_path == "C:/plates/anchor.exr"
    assert len(out.solve.projection_sources) == 1
    source = out.solve.projection_sources[0]
    assert source.image_b64 == "data:image/png;base64,second"
    assert source.plate_ref.image_path == "C:/plates/second.exr"
    assert source.proxy_geometry == []
    assert source.metadata["evidence_type"] == "photographed"
    assert source.metadata["capture_mode"] == "translated"
    assert source.metadata["source_order"] == 2
    assert source.metadata["anchor_identity"] == "anchor"


def test_success_overlay_marks_only_selected_geometric_inliers(monkeypatch) -> None:
    _install_pipeline(monkeypatch)
    seen_masks: list[object] = []

    def record_overlay(_a, _b, _pair, mask=None):
        seen_masks.append(mask)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(features, "render_match_overlay", record_overlay)

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is not None
    assert len(seen_masks) == 2
    assert seen_masks[0] is None
    assert np.asarray(seen_masks[1], dtype=bool).all()


@pytest.mark.parametrize("camera_height_m", (0.0, -1.0, float("nan"), float("inf")))
def test_translated_requires_a_finite_positive_camera_height(
    monkeypatch, camera_height_m: float,
) -> None:
    _install_pipeline(monkeypatch)
    frames = [_frame(label="one"), _frame(label="two")]

    out = solve_multiview(
        frames, MultiViewSettings(camera_height_m=camera_height_m),
    )


    assert out.solve is None
    assert out.diagnostics.outcome_code == "scale_unavailable"


def test_translated_requires_a_supported_ground_plane(monkeypatch) -> None:
    _install_pipeline(monkeypatch, landmarks=_ground_landmarks(23))
    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is None
    assert out.diagnostics.outcome_code == "scale_unavailable"


def test_rotation_only_keeps_one_optical_centre_and_needs_no_scale(monkeypatch) -> None:
    _install_pipeline(monkeypatch, mode="rotation_only")

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(capture_mode="auto"),
    )

    assert out.diagnostics.outcome_code == "rotation_only"
    assert out.solve is not None
    primary = out.solve.camera.extrinsics.camera_position
    secondary = out.solve.projection_sources[0].camera.extrinsics.camera_position
    assert primary == secondary == (0.0, 0.0, 0.0)
    assert out.solve.debug_metadata["scale_source"] == "not_applicable_rotation_only"
    assert out.solve.landmarks == []


def test_anchor_faces_negative_z_and_relative_baseline_is_preserved(monkeypatch) -> None:
    _install_pipeline(monkeypatch)

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is not None
    primary_world = np.asarray(out.solve.camera.extrinsics.camera_world_matrix)
    secondary_world = np.asarray(
        out.solve.projection_sources[0].camera.extrinsics.camera_world_matrix,
    )
    primary_forward = -primary_world[:3, 2]
    assert primary_forward[2] < 0.0
    assert np.linalg.norm(secondary_world[:3, 3] - primary_world[:3, 3]) == pytest.approx(1.43)


def test_three_frame_inputs_keep_photo_one_as_anchor_and_source_order(monkeypatch) -> None:
    _install_pipeline(monkeypatch)

    out = solve_multiview(
        [_frame(label="one"), _frame(label="two"), _frame(label="three")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is not None
    assert out.solve.source_plate.image_path == "C:/plates/one.exr"
    assert [source.name for source in out.solve.projection_sources] == ["two", "three"]
    assert [source.metadata["source_order"] for source in out.solve.projection_sources] == [2, 3]
    assert len(out.diagnostics.pair_metrics) == 3


def test_repeated_solve_serializes_byte_identically(monkeypatch) -> None:
    _install_pipeline(monkeypatch)
    frames = [_frame(label="one"), _frame(label="two")]
    settings = MultiViewSettings(camera_height_m=1.43, seed=91)

    first = solve_multiview(frames, settings)
    second = solve_multiview(frames, settings)

    assert first.solve is not None
    assert second.solve is not None
    assert first.solve.to_json() == second.solve.to_json()
    assert first.diagnostics.to_dict() == second.diagnostics.to_dict()


def test_solve_json_is_byte_identical_across_fresh_processes() -> None:
    test_path = str(Path(__file__).resolve())
    repository_root = str(Path(__file__).resolve().parents[1])
    script = f"""
import dataclasses
import runpy
import sys

import atlas_camera.core.multiview_geometry as geometry_module
import atlas_camera.core.multiview_solver as solver_module

namespace = runpy.run_path({test_path!r})
monkeypatch = namespace["pytest"].MonkeyPatch()
try:
    np = namespace["np"]
    width, height = 640, 360
    focal_px = namespace["_FOCAL_MM"] * width / namespace["_SENSOR_WIDTH_MM"]
    vp_result = {{
        "vp1": np.array((width / 2.0 - focal_px, height / 2.0)),
        "vp2": np.array((width / 2.0 + focal_px, height / 2.0)),
        "vp3": None,
        "left_lines": [],
        "right_lines": [],
        "vertical_lines": [],
        "num_lines_total": 0,
    }}
    monkeypatch.setattr(
        solver_module.VanishingPointDetector,
        "detect_vanishing_points",
        lambda _image, *, random_seed: vp_result,
    )
    monkeypatch.setattr(
        geometry_module,
        "fit_pair_models",
        lambda pair, *_args: namespace["_evidence"](pair, "rotation_only"),
    )
    image = np.random.default_rng(42).integers(
        0, 256, (height, width, 3), dtype=np.uint8,
    )
    frames = [
        dataclasses.replace(
            namespace["_frame"](label="one", width=width, height=height),
            image=image,
        ),
        dataclasses.replace(
            namespace["_frame"](label="two", width=width, height=height),
            image=image.copy(),
        ),
    ]
    outcome = namespace["solve_multiview"](
        frames,
        namespace["MultiViewSettings"](capture_mode="rotation_only", seed=91),
    )
    if outcome.solve is None:
        raise RuntimeError(outcome.diagnostics.to_dict())
    sys.stdout.write(outcome.solve.to_json())
finally:
    monkeypatch.undo()
"""

    def solve_bytes(hash_seed: str) -> bytes:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            check=True,
        )
        return completed.stdout

    first = solve_bytes("1")
    second = solve_bytes("937")

    assert first.startswith(b"{")
    assert first == second


def test_qwen_pixels_are_not_an_input_to_the_solver_signature() -> None:
    assert list(inspect.signature(solve_multiview).parameters) == ["frames", "settings"]


def test_anchor_orientation_from_up_builds_right_handed_atlas_basis() -> None:
    from atlas_camera.core.multiview_solver import _anchor_orientation_from_up
    from atlas_camera.core.schema import AtlasIntrinsics

    intrinsics = AtlasIntrinsics(
        fx_px=_FX, fy_px=_FX, cx_px=_WIDTH / 2.0, cy_px=_HEIGHT / 2.0,
        image_width=_WIDTH, image_height=_HEIGHT,
    )
    tilt = np.deg2rad(10.0)
    up_hint = (0.0, float(np.cos(tilt)), float(np.sin(tilt)))
    anchored = _anchor_orientation_from_up(intrinsics, up_hint)
    assert anchored is not None
    camera_to_world, horizon, vanishing_points = anchored

    world_to_camera = camera_to_world.T
    assert np.linalg.det(world_to_camera) == pytest.approx(1.0)
    assert world_to_camera.T @ world_to_camera == pytest.approx(np.eye(3))
    assert world_to_camera[:, 1] == pytest.approx(np.array(up_hint))
    assert vanishing_points == []

    # Horizon pixels' rays are orthogonal to up.
    a, b, c = horizon.line_coefficients
    x = 100.0
    y = (-c - a * x) / b
    ray = np.array((
        (x - intrinsics.cx_px) / intrinsics.fx_px,
        -(y - intrinsics.cy_px) / intrinsics.fy_px,
        -1.0,
    ))
    assert float(ray @ np.array(up_hint)) == pytest.approx(0.0, abs=1e-9)

    # A negated hint flips to keep +Y up.
    flipped = _anchor_orientation_from_up(intrinsics, tuple(-v for v in up_hint))
    assert flipped is not None
    assert flipped[0].T[:, 1] == pytest.approx(np.array(up_hint))


def test_anchor_orientation_from_up_rejects_degenerate_hints() -> None:
    from atlas_camera.core.multiview_solver import _anchor_orientation_from_up
    from atlas_camera.core.schema import AtlasIntrinsics

    intrinsics = AtlasIntrinsics(
        fx_px=_FX, fy_px=_FX, cx_px=_WIDTH / 2.0, cy_px=_HEIGHT / 2.0,
        image_width=_WIDTH, image_height=_HEIGHT,
    )
    assert _anchor_orientation_from_up(intrinsics, (0.0, 0.0, 0.0)) is None
    # Looking straight along gravity: horizontal facing undefined.
    assert _anchor_orientation_from_up(intrinsics, (0.0, 0.0, 1.0)) is None
    assert _anchor_orientation_from_up(intrinsics, (float("nan"), 1.0, 0.0)) is None


def test_anchor_up_hint_enters_the_registration_fingerprint() -> None:
    from atlas_camera.core.multiview_types import registration_fingerprint

    frames = ()
    base = MultiViewSettings()
    hinted = MultiViewSettings(anchor_up_hint=(0.0, 1.0, 0.0), anchor_up_hint_source="learned prior (geocalib)")
    assert registration_fingerprint(frames, base) != registration_fingerprint(frames, hinted)


def test_portrait_orientation_swaps_sensor_millimetres_for_intrinsics() -> None:
    from atlas_camera.core.multiview_solver import _intrinsics_for_frame

    landscape = _frame(width=_WIDTH, height=_HEIGHT, orientation=1)
    portrait = _frame(width=_HEIGHT, height=_WIDTH, orientation=8)

    flat = _intrinsics_for_frame(landscape)
    assert flat.fx_px == pytest.approx(_FOCAL_MM * _WIDTH / _SENSOR_WIDTH_MM)
    assert flat.fy_px == pytest.approx(_FOCAL_MM * _HEIGHT / _SENSOR_HEIGHT_MM)

    tall = _intrinsics_for_frame(portrait)
    # Rotated pixels: the developed width spans the sensor's SHORT side.
    assert tall.fx_px == pytest.approx(_FOCAL_MM * _HEIGHT / _SENSOR_HEIGHT_MM)
    assert tall.fy_px == pytest.approx(_FOCAL_MM * _WIDTH / _SENSOR_WIDTH_MM)


def test_measured_baseline_is_the_top_scale_anchor(monkeypatch) -> None:
    _install_pipeline(monkeypatch)
    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(baseline_m=0.8),
    )

    assert out.solve is not None
    assert out.diagnostics.scale["source"] == "measured_baseline"
    assert out.solve.debug_metadata["scale_source"] == "measured_baseline"
    primary = np.asarray(out.solve.camera.extrinsics.camera_position)
    secondary = np.asarray(
        out.solve.projection_sources[0].camera.extrinsics.camera_position
    )
    assert float(np.linalg.norm(secondary - primary)) == pytest.approx(0.8)
    # No camera height entered: the note says the vertical origin is the camera.
    assert any("optical centre" in note for note in out.diagnostics.scale["notes"])


def test_baseline_with_camera_height_seats_photo_one_at_that_height(monkeypatch) -> None:
    _install_pipeline(monkeypatch)
    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(baseline_m=0.8, camera_height_m=1.66),
    )

    assert out.solve is not None
    assert out.diagnostics.scale["source"] == "measured_baseline"
    assert out.solve.camera.extrinsics.camera_position[1] == pytest.approx(1.66)


def test_learned_depth_prior_scales_when_no_measured_anchor_exists(monkeypatch) -> None:
    count = 60
    flat_landmarks = np.column_stack((
        (np.mod(np.arange(count, dtype=np.float64), 10.0) - 4.5) * 0.5,
        np.full(count, 1.0),
        np.full(count, 5.0),
    ))
    _install_pipeline(monkeypatch, landmarks=flat_landmarks)
    depth = np.full((720, 1280), 10.0, dtype=np.float32)

    frames = [
        _frame(label="one"),
        _frame(label="two"),
    ]
    from dataclasses import replace as dataclass_replace
    frames[0] = dataclass_replace(frames[0], metric_depth=depth)
    out = solve_multiview(frames, MultiViewSettings())

    assert out.solve is not None
    scale = out.diagnostics.scale
    assert scale["source"] == "learned_depth_prior"
    # Predicted 10 m over recovered 5 -> uniform scale factor 2.
    assert scale["scale_factor"] == pytest.approx(2.0)
    primary = np.asarray(out.solve.camera.extrinsics.camera_position)
    secondary = np.asarray(
        out.solve.projection_sources[0].camera.extrinsics.camera_position
    )
    assert float(np.linalg.norm(secondary - primary)) == pytest.approx(2.0)
    assert any("monocular depth prior" in warning for warning in out.diagnostics.warnings)


def test_scale_unavailable_names_every_remedy(monkeypatch) -> None:
    _install_pipeline(monkeypatch, landmarks=_ground_landmarks(23))
    out = solve_multiview(
        [_frame(label="one"), _frame(label="two")],
        MultiViewSettings(camera_height_m=1.43),
    )

    assert out.solve is None
    assert out.diagnostics.outcome_code == "scale_unavailable"
    assert "baseline_m" in out.diagnostics.summary
    assert "depth" in out.diagnostics.summary
