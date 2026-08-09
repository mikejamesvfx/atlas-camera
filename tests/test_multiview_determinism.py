"""Fresh-process release gate for deterministic photographed multi-view solves."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


np = pytest.importorskip("numpy")

from atlas_camera.core.multiview_geometry import ClosureMetrics, RefinedRig  # noqa: E402
from atlas_camera.core.multiview_solver import _compose_anchor_rig  # noqa: E402


_AMBIENT_SEEDS = (1, 97, 2026)


def test_compose_anchor_rig_converts_opencv_local_geometry_to_atlas() -> None:
    angle = np.deg2rad(11.0)
    rotation_cv = np.array((
        (np.cos(angle), 0.0, np.sin(angle)),
        (0.0, 1.0, 0.0),
        (-np.sin(angle), 0.0, np.cos(angle)),
    ))
    translations_cv = (
        np.zeros(3, dtype=np.float64),
        np.array((-0.8, 0.05, 0.2), dtype=np.float64),
    )
    landmarks_cv = np.array((
        (0.2, 1.6, 5.0),
        (-0.7, 1.6, 8.0),
    ))
    refined = RefinedRig(
        rotations=(np.eye(3, dtype=np.float64), rotation_cv),
        translations=translations_cv,
        landmarks=landmarks_cv,
        reprojection_rmse_px=0.25,
        accepted_track_ids=(0, 1),
        closure=ClosureMetrics(0.0, 0.0, 0.0),
    )

    rotations_atlas, positions_atlas, landmarks_atlas = _compose_anchor_rig(
        refined,
        np.eye(3, dtype=np.float64),
    )

    assert np.all(landmarks_atlas[:, 1] < positions_atlas[0, 1])
    assert all(np.linalg.det(rotation) == pytest.approx(1.0) for rotation in rotations_atlas)
    opencv_to_atlas = np.diag((1.0, -1.0, -1.0))
    for frame_index in range(2):
        for landmark_cv, landmark_atlas in zip(landmarks_cv, landmarks_atlas):
            camera_atlas = rotations_atlas[frame_index].T @ (
                landmark_atlas - positions_atlas[frame_index]
            )
            camera_cv = opencv_to_atlas @ camera_atlas
            expected_cv = (
                refined.rotations[frame_index] @ landmark_cv
                + refined.translations[frame_index]
            )
            assert camera_cv == pytest.approx(expected_cv)


def test_fresh_processes_emit_identical_multiview_payloads() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    fixture = Path(__file__).with_name("multiview_subprocess_fixture.py")
    outputs: list[bytes] = []

    for ambient_seed in _AMBIENT_SEEDS:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(ambient_seed)
        completed = subprocess.run(
            [sys.executable, str(fixture), str(ambient_seed)],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            check=True,
        )
        outputs.append(completed.stdout)

    assert outputs[0].startswith(b'{"diagnostics":')
    assert outputs[0] == outputs[1] == outputs[2]
