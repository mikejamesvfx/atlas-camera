"""Benchmark Atlas solves against external dataset camera metadata."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from math import acos, atan, degrees, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any

from atlas_camera.core.schema import AtlasIntrinsics, Matrix3
from atlas_camera.core.solver import solve_still_image
from atlas_camera.datasets.colmap import ColmapCamera, ColmapImage
from atlas_camera.datasets.eth3d import load_eth3d_dataset


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    limit: int | None = None
    detect_vanishing_points: bool = False
    include_missing_images: bool = True


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    dataset: str
    image_name: str
    image_path: str
    status: str
    runtime_seconds: float
    source_method: str | None
    confidence: float | None
    gt_camera_model: str
    gt_fx_px: float
    gt_fy_px: float
    gt_cx_px: float
    gt_cy_px: float
    solved_fx_px: float | None
    solved_fy_px: float | None
    solved_cx_px: float | None
    solved_cy_px: float | None
    fx_abs_error_px: float | None
    fy_abs_error_px: float | None
    principal_point_error_px: float | None
    horizontal_fov_error_deg: float | None
    vertical_fov_error_deg: float | None
    orientation_error_deg: float | None
    detected_vanishing_points: int
    detected_lines: int
    error: str | None = None


def benchmark_eth3d(root: str | Path, options: BenchmarkOptions | None = None) -> list[BenchmarkRecord]:
    options = options or BenchmarkOptions()
    dataset = load_eth3d_dataset(root)
    records: list[BenchmarkRecord] = []
    for image in dataset.iter_images()[: options.limit]:
        camera = dataset.cameras[image.camera_id]
        image_path = dataset.image_path(image)
        if not image_path.exists() and not options.include_missing_images:
            continue
        records.append(_benchmark_colmap_image("eth3d", image_path, camera, image, options))
    return records


def write_benchmark_json(records: list[BenchmarkRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2, sort_keys=True)
    return destination


def write_benchmark_csv(records: list[BenchmarkRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkRecord.__dataclass_fields__)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return destination


def _benchmark_colmap_image(
    dataset_name: str,
    image_path: Path,
    camera: ColmapCamera,
    image: ColmapImage,
    options: BenchmarkOptions,
) -> BenchmarkRecord:
    started = perf_counter()
    try:
        solve = solve_still_image(
            image_path,
            image_size=(camera.width, camera.height),
            intrinsics_hint=camera.intrinsics_hint(),
            detect_vanishing_points=options.detect_vanishing_points,
        )
        runtime = perf_counter() - started
        intrinsics = solve.camera.intrinsics
        return _record_from_solve(
            dataset_name,
            image_path,
            camera,
            image,
            intrinsics,
            solve.camera.extrinsics.camera_rotation_matrix,
            runtime,
            status="ok",
            source_method=solve.source_method,
            confidence=solve.confidence,
            detected_vanishing_points=len(solve.vanishing_points),
            detected_lines=int(solve.debug_metadata.get("num_lines_total", 0)),
        )
    except Exception as exc:
        runtime = perf_counter() - started
        return _error_record(dataset_name, image_path, camera, image, runtime, exc)


def _record_from_solve(
    dataset_name: str,
    image_path: Path,
    camera: ColmapCamera,
    image: ColmapImage,
    intrinsics: AtlasIntrinsics,
    solved_rotation: Matrix3,
    runtime: float,
    *,
    status: str,
    source_method: str | None,
    confidence: float | None,
    detected_vanishing_points: int,
    detected_lines: int,
) -> BenchmarkRecord:
    solved_fx = intrinsics.fx_px
    solved_fy = intrinsics.fy_px
    solved_cx = intrinsics.cx_px
    solved_cy = intrinsics.cy_px
    return BenchmarkRecord(
        dataset=dataset_name,
        image_name=image.name,
        image_path=str(image_path),
        status=status,
        runtime_seconds=runtime,
        source_method=source_method,
        confidence=confidence,
        gt_camera_model=camera.model,
        gt_fx_px=camera.fx_px,
        gt_fy_px=camera.fy_px,
        gt_cx_px=camera.cx_px,
        gt_cy_px=camera.cy_px,
        solved_fx_px=solved_fx,
        solved_fy_px=solved_fy,
        solved_cx_px=solved_cx,
        solved_cy_px=solved_cy,
        fx_abs_error_px=_abs_error(solved_fx, camera.fx_px),
        fy_abs_error_px=_abs_error(solved_fy, camera.fy_px),
        principal_point_error_px=_principal_point_error(solved_cx, solved_cy, camera.cx_px, camera.cy_px),
        horizontal_fov_error_deg=_fov_error(camera.width, solved_fx, camera.fx_px),
        vertical_fov_error_deg=_fov_error(camera.height, solved_fy, camera.fy_px),
        orientation_error_deg=_rotation_error_deg(image.camera_to_world_rotation, solved_rotation),
        detected_vanishing_points=detected_vanishing_points,
        detected_lines=detected_lines,
    )


def _error_record(
    dataset_name: str,
    image_path: Path,
    camera: ColmapCamera,
    image: ColmapImage,
    runtime: float,
    error: Exception,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        dataset=dataset_name,
        image_name=image.name,
        image_path=str(image_path),
        status="error",
        runtime_seconds=runtime,
        source_method=None,
        confidence=None,
        gt_camera_model=camera.model,
        gt_fx_px=camera.fx_px,
        gt_fy_px=camera.fy_px,
        gt_cx_px=camera.cx_px,
        gt_cy_px=camera.cy_px,
        solved_fx_px=None,
        solved_fy_px=None,
        solved_cx_px=None,
        solved_cy_px=None,
        fx_abs_error_px=None,
        fy_abs_error_px=None,
        principal_point_error_px=None,
        horizontal_fov_error_deg=None,
        vertical_fov_error_deg=None,
        orientation_error_deg=None,
        detected_vanishing_points=0,
        detected_lines=0,
        error=str(error),
    )


def _abs_error(value: float | None, ground_truth: float) -> float | None:
    if value is None:
        return None
    return abs(value - ground_truth)


def _principal_point_error(
    cx: float | None,
    cy: float | None,
    gt_cx: float,
    gt_cy: float,
) -> float | None:
    if cx is None or cy is None:
        return None
    return sqrt(((cx - gt_cx) ** 2) + ((cy - gt_cy) ** 2))


def _fov_error(image_size_px: int, solved_focal_px: float | None, gt_focal_px: float) -> float | None:
    if solved_focal_px is None:
        return None
    solved = 2.0 * degrees(atan(image_size_px / (2.0 * solved_focal_px)))
    expected = 2.0 * degrees(atan(image_size_px / (2.0 * gt_focal_px)))
    return abs(solved - expected)


def _rotation_error_deg(first: Matrix3, second: Matrix3) -> float:
    relative = _matrix_multiply(_matrix_transpose(first), second)
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return degrees(acos(cos_angle))


def _matrix_transpose(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(matrix[row][col] for row in range(3))
        for col in range(3)
    )  # type: ignore[return-value]


def _matrix_multiply(first: Matrix3, second: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(first[row][index] * second[index][col] for index in range(3))
            for col in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]
