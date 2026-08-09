"""Acceptance-runner contracts for deterministic photographed RAW capture."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from atlas_camera.core.multiview_types import (
    RegistrationDiagnostics,
    RegistrationOutcome,
)
from tools.validate_multiview_capture import canonical_json, run_manifest


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _RawResult:
    display_srgb: np.ndarray
    source_path: str
    camera_make: str = "FUJIFILM"
    camera_model: str = "X-H2"
    lens_model: str = "XF16-55mmF2.8 R LM WR"
    focal_length_mm: float = 23.4
    sensor_width_mm: float = 23.5
    sensor_height_mm: float = 15.6
    sensor_source: str = "camera_db"
    orientation: int = 1
    capture_datetime: str = "2026:08:09 10:11:12"
    metadata_source: str = "embedded_jpeg"
    undistort_status: str = "disabled"
    warnings: tuple[str, ...] = ()


def _manifest(tmp_path: Path, **overrides):
    paths = [tmp_path / "left.raf", tmp_path / "right.raf"]
    for path in paths:
        path.write_bytes(b"fixture")
    manifest = {
        "raw_paths": [str(path) for path in paths],
        "camera_height_m": 1.43,
        "capture_mode": "translated",
        "match_quality": "balanced",
        "seed": 7,
    }
    manifest.update(overrides)
    return manifest


def _import_stub(path, **_kwargs):
    value = sum(Path(path).name.encode("utf-8")) % 255
    image = np.full((4, 6, 3), value / 255.0, dtype=np.float32)
    return _RawResult(display_srgb=image, source_path=str(path))


def _translated_stub(frames, settings):
    assert [frame.label for frame in frames] == ["photo_1", "photo_2"]
    assert settings.camera_height_m == pytest.approx(1.43)
    return RegistrationOutcome(
        solve=SimpleNamespace(to_dict=lambda: {"source_method": "deterministic_raw_multiview"}),
        diagnostics=RegistrationDiagnostics(
            "translated",
            "registered 2 photographed RAW frames in translated mode",
            selected_mode="translated",
            scale={"source": "measured_camera_height", "camera_height_m": 1.43},
        ),
        overlays=(np.full((3, 5, 3), 0.5, dtype=np.float32),),
    )


def test_direct_script_entry_point_loads_repo_package():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_multiview_capture.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "capture manifest" in completed.stdout


@pytest.mark.parametrize("camera_height_m", [0.0, -1.0, float("nan"), float("inf")])
def test_translated_height_validation_is_delegated_to_solver_with_evidence(
    tmp_path, monkeypatch, camera_height_m,
):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    manifest = _manifest(tmp_path, camera_height_m=camera_height_m)
    calls = []

    def reject(frames, settings):
        calls.append((frames, settings))
        return RegistrationOutcome(
            solve=None,
            diagnostics=RegistrationDiagnostics(
                "scale_unavailable",
                "solver retained pair evidence before rejecting scale",
                selected_mode="translated",
                pair_metrics=[{"frame_a": 0, "frame_b": 1, "matches": 51}],
                scale={"camera_height_m": camera_height_m},
            ),
            overlays=(np.zeros((2, 3, 3), dtype=np.float32),),
        )

    result = run_manifest(manifest, output_dir=tmp_path, solve_fn=reject)
    assert len(calls) == 1
    assert len(calls[0][0]) == 2
    if math.isnan(camera_height_m):
        assert math.isnan(calls[0][1].camera_height_m)
    else:
        assert calls[0][1].camera_height_m == camera_height_m
    assert result["outcome_code"] == "scale_unavailable"
    assert result["solve"] is None
    assert result["diagnostics"]["pair_metrics"][0]["matches"] == 51
    assert result["overlays"] == ["pair_01.png"]
    assert len(result["frames"]) == 2
    assert json.loads(canonical_json(result))["outcome_code"] == "scale_unavailable"


def test_auto_rotation_only_does_not_require_positive_height(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    manifest = _manifest(tmp_path, capture_mode="auto", camera_height_m=0.0)

    def rotation(_frames, settings):
        assert settings.capture_mode == "auto"
        assert settings.camera_height_m == 0.0
        return RegistrationOutcome(
            solve=SimpleNamespace(to_dict=lambda: {"camera_count": 2}),
            diagnostics=RegistrationDiagnostics(
                "rotation_only", "orientation recovered", selected_mode="rotation_only",
                scale={"source": "not_applicable_rotation_only", "camera_height_m": None},
            ),
        )

    result = run_manifest(manifest, output_dir=tmp_path / "out", solve_fn=rotation)
    assert result["outcome_code"] == "rotation_only"


def test_acceptance_report_hash_is_stable(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    first = run_manifest(_manifest(tmp_path), output_dir=tmp_path / "first",
                         solve_fn=_translated_stub)
    second = run_manifest(_manifest(tmp_path), output_dir=tmp_path / "second",
                          solve_fn=_translated_stub)
    assert canonical_json(first) == canonical_json(second)
    assert (tmp_path / "first" / "registration.json").read_text("utf-8") == canonical_json(first)


def test_three_frame_run_emits_canonical_pair_names_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    manifest = _manifest(tmp_path)
    third = tmp_path / "third.raf"
    third.write_bytes(b"fixture")
    manifest["raw_paths"].append(str(third))

    def solve(frames, _settings):
        overlays = tuple(np.full((2, 3, 3), index / 4, dtype=np.float32)
                         for index in range(3))
        return RegistrationOutcome(
            solve=SimpleNamespace(to_dict=lambda: {"camera_count": 3}),
            diagnostics=RegistrationDiagnostics(
                "translated", "ok", selected_mode="translated",
                scale={"source": "measured_camera_height", "camera_height_m": 1.43},
            ),
            overlays=overlays,
        )

    result = run_manifest(manifest, output_dir=tmp_path / "out", solve_fn=solve)
    assert result["input_count"] == 3
    assert result["overlays"] == ["pair_01.png", "pair_02.png", "pair_12.png"]
    assert result["frames"][0]["camera_make"] == "FUJIFILM"
    assert result["frames"][0]["focal_length_mm"] == pytest.approx(23.4)
    assert result["frames"][0]["orientation"] == 1
    assert result["frames"][0]["capture_datetime"] == "2026:08:09 10:11:12"
    assert all((tmp_path / "out" / name).is_file() for name in result["overlays"])


def test_failed_solve_still_writes_structured_report_and_available_overlay(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)

    def reject(_frames, _settings):
        return RegistrationOutcome(
            solve=None,
            diagnostics=RegistrationDiagnostics(
                "insufficient_overlap", "not enough mutual matches",
                pair_metrics=[{"frame_a": 0, "frame_b": 1, "matches": 9}],
            ),
            overlays=(np.zeros((2, 3, 3), dtype=np.float32),),
        )

    result = run_manifest(_manifest(tmp_path), output_dir=tmp_path / "out", solve_fn=reject)
    assert result["outcome_code"] == "insufficient_overlap"
    assert result["solve"] is None
    assert result["overlays"] == ["pair_01.png"]
    persisted = json.loads((tmp_path / "out" / "registration.json").read_text("utf-8"))
    assert persisted == result


def test_reused_output_removes_only_stale_canonical_overlays_on_import_failure(
    tmp_path, monkeypatch,
):
    output = tmp_path / "out"
    output.mkdir()
    for name in ("pair_01.png", "pair_02.png", "pair_12.png"):
        (output / name).write_bytes(b"stale")
    unrelated = output / "artist_notes.png"
    unrelated.write_bytes(b"keep")

    def fail_import(_path, **_kwargs):
        raise RuntimeError("decode stopped")

    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", fail_import)
    result = run_manifest(_manifest(tmp_path), output_dir=output, solve_fn=_translated_stub)

    assert result["outcome_code"] == "metadata_mismatch"
    assert not any((output / name).exists()
                   for name in ("pair_01.png", "pair_02.png", "pair_12.png"))
    assert unrelated.read_bytes() == b"keep"


def test_partial_overlay_write_failure_leaves_no_canonical_overlay(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    manifest = _manifest(tmp_path)
    third = tmp_path / "third.raf"
    third.write_bytes(b"fixture")
    manifest["raw_paths"].append(str(third))

    def solve(_frames, _settings):
        overlays = tuple(np.zeros((2, 3, 3), dtype=np.float32) for _ in range(3))
        return RegistrationOutcome(
            solve=None,
            diagnostics=RegistrationDiagnostics("insufficient_overlap", "evidence"),
            overlays=overlays,
        )

    from tools import validate_multiview_capture as runner
    real_write = runner._write_overlay
    writes = 0

    def fail_second(path, overlay):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        real_write(path, overlay)

    monkeypatch.setattr(runner, "_write_overlay", fail_second)
    output = tmp_path / "out"
    result = run_manifest(manifest, output_dir=output, solve_fn=solve)

    assert result["outcome_code"] == "insufficient_overlap"
    assert result["overlays"] == []
    assert "overlay artifact write failed" in " ".join(result["warnings"])
    assert not any((output / name).exists()
                   for name in ("pair_01.png", "pair_02.png", "pair_12.png"))


def test_equivalent_relocated_manifests_write_identical_private_reports(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    reports = []
    for root_name in ("author_a", "author_b"):
        capture = tmp_path / root_name / "capture"
        raw_dir = capture / "raw"
        raw_dir.mkdir(parents=True)
        for name in ("left.raf", "right.raf"):
            (raw_dir / name).write_bytes(b"fixture")
        manifest_path = capture / "manifest.json"
        manifest_path.write_text(json.dumps({
            "raw_paths": ["raw/left.raf", "raw/right.raf"],
            "camera_height_m": 1.43,
            "capture_mode": "translated",
            "match_quality": "balanced",
            "seed": 7,
        }), encoding="utf-8")
        reports.append(run_manifest(
            manifest_path, output_dir=capture / "out", solve_fn=_translated_stub,
        ))

    assert canonical_json(reports[0]) == canonical_json(reports[1])
    assert [frame["source_path"] for frame in reports[0]["frames"]] == [
        "raw/left.raf", "raw/right.raf",
    ]
    report_text = canonical_json(reports[0])
    assert str(tmp_path) not in report_text
    assert "author_a" not in report_text


def test_import_failure_summary_does_not_embed_resolved_machine_path(tmp_path):
    manifest = _manifest(tmp_path, raw_paths=[
        str(tmp_path / "private-author" / "missing.raf"),
        str(tmp_path / "private-author" / "also-missing.raf"),
    ])
    result = run_manifest(manifest, output_dir=tmp_path / "out", solve_fn=_translated_stub)
    report_text = canonical_json(result)
    assert result["outcome_code"] == "metadata_mismatch"
    assert str(tmp_path) not in report_text
    assert "private-author" not in report_text


def test_rotation_only_reports_no_translation_or_metric_scale(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.validate_multiview_capture.import_raw", _import_stub)
    manifest = _manifest(tmp_path, capture_mode="rotation_only", camera_height_m=0.0)

    def rotation(_frames, _settings):
        return RegistrationOutcome(
            solve=SimpleNamespace(to_dict=lambda: {"camera_count": 2}),
            diagnostics=RegistrationDiagnostics(
                "rotation_only", "orientation recovered", selected_mode="rotation_only",
                scale={"source": "not_applicable_rotation_only", "camera_height_m": None},
            ),
        )

    result = run_manifest(manifest, output_dir=tmp_path / "out", solve_fn=rotation)
    assert result["outcome_code"] == "rotation_only"
    assert result["translation_recovered"] is False
    assert result["metric_scale_recovered"] is False
    assert "cannot recover translation or metric scale" in " ".join(result["warnings"])


@pytest.mark.parametrize("raw_count", [0, 1, 4])
def test_manifest_requires_exactly_two_or_three_ordered_paths(tmp_path, raw_count):
    manifest = _manifest(tmp_path, raw_paths=[f"image_{index}.raf" for index in range(raw_count)])
    with pytest.raises(ValueError, match="exactly 2 or 3"):
        run_manifest(manifest, output_dir=tmp_path)
