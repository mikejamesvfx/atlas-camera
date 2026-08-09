"""Fresh-process release gate for deterministic photographed multi-view solves."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


np = pytest.importorskip("numpy")
try:
    __import__("cv2")
except ImportError:
    pytest.skip("OpenCV is unavailable", allow_module_level=True)

from atlas_camera.core.multiview_geometry import ClosureMetrics, RefinedRig  # noqa: E402
from atlas_camera.core.multiview_solver import _compose_anchor_rig  # noqa: E402


_AMBIENT_SEEDS = (1, 97, 2026)


def _parse_child_payload(output: bytes) -> dict[str, object]:
    payload = json.loads(output)
    assert isinstance(payload, dict)
    assert set(payload) == {"diagnostics", "solve"}

    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["outcome_code"] == "translated"
    assert diagnostics["selected_mode"] == "translated"
    assert len(diagnostics["camera_metrics"]) == 3
    assert len(diagnostics["pair_metrics"]) == 3
    assert all(metric["mutual_matches"] > 0 for metric in diagnostics["pair_metrics"])
    assert all(metric["essential_inliers"] > 0 for metric in diagnostics["pair_metrics"])

    solve = payload["solve"]
    assert isinstance(solve, dict)
    assert solve["landmarks"]
    assert len(solve["projection_sources"]) == 2
    debug_metadata = solve["debug_metadata"]
    assert debug_metadata["accepted_track_count"] > 0
    assert debug_metadata["frame_count"] == 3
    assert debug_metadata["generated_inputs_used"] is False
    assert debug_metadata["photographed_source_count"] == 3
    return payload


def test_compose_anchor_rig_converts_opencv_local_geometry_to_atlas() -> None:
    rotation_cv = np.array((
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
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

    anchor_basis = np.array((
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ))
    rotations_atlas, positions_atlas, landmarks_atlas = _compose_anchor_rig(
        refined, anchor_basis,
    )

    expected_rotations = (
        anchor_basis,
        np.array((
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
        )),
    )
    expected_positions = np.array(((0.0, 0.0, 0.0), (-0.8, 0.2, 0.05)))
    expected_landmarks = np.array(((-5.0, 0.2, -1.6), (-8.0, -0.7, -1.6)))
    for actual, expected in zip(rotations_atlas, expected_rotations):
        assert actual == pytest.approx(expected)
    assert positions_atlas == pytest.approx(expected_positions)
    assert landmarks_atlas == pytest.approx(expected_landmarks)
    anchor_up = anchor_basis[:, 1]
    assert np.all(landmarks_atlas @ anchor_up < positions_atlas[0] @ anchor_up)
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


def test_collection_skips_when_opencv_is_unavailable(tmp_path: Path) -> None:
    unavailable = tmp_path / "cv2.py"
    unavailable.write_text(
        "raise ImportError('OpenCV intentionally unavailable')\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "test_collection_sentinel.py"
    sentinel.write_text("def test_collection_sentinel():\n    pass\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(tmp_path),
        environment.get("PYTHONPATH"),
    )))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-rs",
            str(Path(__file__)),
            str(sentinel),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        check=True,
        timeout=120,
    )

    combined_output = completed.stdout + completed.stderr
    assert b"1 skipped" in combined_output
    assert b"test_fresh_processes_emit_identical_multiview_payloads" not in combined_output


def test_collection_supports_pytest_8_importorskip_signature(tmp_path: Path) -> None:
    plugin = tmp_path / "pytest8_importorskip.py"
    plugin.write_text(
        """import pytest

_original_importorskip = pytest.importorskip

def _pytest8_importorskip(modname, minversion=None, reason=None):
    return _original_importorskip(modname, minversion=minversion, reason=reason)

def pytest_configure(config):
    pytest.importorskip = _pytest8_importorskip
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(tmp_path),
        environment.get("PYTHONPATH"),
    )))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest8_importorskip",
            "--collect-only",
            "-q",
            str(Path(__file__)),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert b"test_fresh_processes_emit_identical_multiview_payloads" in completed.stdout


@pytest.mark.parametrize(
    "invalid_output",
    (
        b'{"diagnostics":{},"solve":{}} trailing output',
        b'{"diagnostics":',
    ),
)
def test_child_payload_parser_rejects_noncanonical_output(invalid_output: bytes) -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_child_payload(invalid_output)


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
            timeout=120,
        )
        _parse_child_payload(completed.stdout)
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1] == outputs[2]
