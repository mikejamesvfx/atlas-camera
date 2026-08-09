"""Run one real synthetic translated multi-view solve in a fresh process."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from atlas_camera.core.multiview_solver import solve_multiview  # noqa: E402
from atlas_camera.core.multiview_types import (  # noqa: E402
    MultiViewFrame,
    MultiViewSettings,
)
from atlas_camera.core.schema import AtlasPlateRef  # noqa: E402
from atlas_camera.raw.pipeline import RawImportResult  # noqa: E402


WIDTH = 800
HEIGHT = 600
FOCAL_MM = 35.0
SENSOR_WIDTH_MM = 40.0
SENSOR_HEIGHT_MM = 30.0
FOCAL_PX = FOCAL_MM * WIDTH / SENSOR_WIDTH_MM
SOLVER_SEED = 1907


def _camera_rotation() -> np.ndarray:
    camera = np.array((0.0, 1.6, 0.0), dtype=np.float64)
    target = np.array((1.25, 0.0, 8.0), dtype=np.float64)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    rotation = np.vstack((right, down, forward))
    roll = math.radians(7.0)
    roll_matrix = np.array((
        (math.cos(roll), math.sin(roll), 0.0),
        (-math.sin(roll), math.cos(roll), 0.0),
        (0.0, 0.0, 1.0),
    ))
    return roll_matrix @ rotation


def _ground_texture() -> np.ndarray:
    size = 1800
    rng = np.random.default_rng(271828)
    texture = rng.integers(105, 151, (size, size), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.0)
    for index in range(900):
        centre = tuple(int(value) for value in rng.integers(30, size - 30, 2))
        radius = int(rng.integers(5, 15))
        colour = 10 if index % 2 else 245
        cv2.circle(texture, centre, radius, colour, 2, cv2.LINE_8)
    return cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)


def _render_views() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(314159)
    intrinsic = np.array((
        (FOCAL_PX, 0.0, WIDTH / 2.0),
        (0.0, FOCAL_PX, HEIGHT / 2.0),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
    rotation = _camera_rotation()
    centres = (
        np.array((0.0, 1.6, 0.0), dtype=np.float64),
        np.array((2.20, 1.6, 0.0), dtype=np.float64),
        np.array((0.50, 1.6, 1.50), dtype=np.float64),
    )

    def project(point: np.ndarray, centre: np.ndarray) -> tuple[float, float] | None:
        camera_point = rotation @ (point - centre)
        if camera_point[2] <= 0.0:
            return None
        pixel = intrinsic @ camera_point
        return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])

    ground_texture = _ground_texture()
    texture_size = ground_texture.shape[0]
    texture_to_ground = np.array((
        (16.0 / texture_size, 0.0, -8.0),
        (0.0, 20.0 / texture_size, 0.75),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
    images: list[np.ndarray] = []
    for centre in centres:
        translation = -rotation @ centre
        ground_projection = np.column_stack((
            rotation[:, 0], rotation[:, 2], translation,
        ))
        homography = intrinsic @ ground_projection @ texture_to_ground
        images.append(cv2.warpPerspective(
            ground_texture,
            homography,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(225, 225, 225),
        ))
    # Long ground-grid segments provide the real vanishing-point detector with
    # two orthogonal architectural line families.
    ground_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for z_value in np.linspace(2.5, 19.0, 13):
        ground_lines.append((
            np.array((-9.0, 0.0, z_value)),
            np.array((9.0, 0.0, z_value)),
        ))
    for x_value in np.linspace(-8.0, 8.0, 15):
        ground_lines.append((
            np.array((x_value, 0.0, 2.0)),
            np.array((x_value, 0.0, 20.0)),
        ))
    for image_index, (image, centre) in enumerate(zip(images, centres)):
        if image_index == 0:
            continue
        for start_world, end_world in ground_lines:
            start = project(start_world, centre)
            end = project(end_world, centre)
            if start is not None and end is not None:
                cv2.line(
                    image,
                    tuple(int(round(value)) for value in start),
                    tuple(int(round(value)) for value in end),
                    (45, 45, 45),
                    2,
                    cv2.LINE_8,
                )

    points: list[np.ndarray] = []
    anchor_pixels = [
        (x_value, y_value)
        for y_value in range(20, 84, 16)
        for x_value in range(34, WIDTH - 34, 12)
    ]
    for candidate_index in rng.permutation(len(anchor_pixels)):
        grid_x, grid_y = anchor_pixels[int(candidate_index)]
        x_pixel = float(grid_x + rng.uniform(-3.0, 3.0))
        y_pixel = float(grid_y + rng.uniform(-3.0, 3.0))
        depth = float(rng.uniform(8.0, 20.0))
        camera_point = np.array((
            (x_pixel - WIDTH / 2.0) * depth / FOCAL_PX,
            (y_pixel - HEIGHT / 2.0) * depth / FOCAL_PX,
            depth,
        ))
        candidate = centres[0] + rotation.T @ camera_point
        projections = [project(candidate, centre) for centre in centres]
        if any(value is None for value in projections):
            continue
        if not all(
            14.0 <= value[0] < WIDTH - 14.0
            and 14.0 <= value[1] < HEIGHT - 14.0
            for value in projections
            if value is not None
        ):
            continue
        points.append(candidate)
        if len(points) == 100:
            break
    assert len(points) == 100

    for point in points:
        small = rng.integers(0, 2, (5, 5), dtype=np.uint8) * 245 + 5
        patch = cv2.resize(small, (17, 17), interpolation=cv2.INTER_NEAREST)
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        for image, centre in zip(images, centres):
            projected = project(point, centre)
            assert projected is not None
            x_value, y_value = (int(round(value)) for value in projected)
            image[y_value - 8:y_value + 9, x_value - 8:x_value + 9] = patch

    ground_points: list[np.ndarray] = []
    for _ in range(8000):
        candidate = np.array((
            rng.uniform(-8.0, 8.0),
            0.0,
            rng.uniform(3.0, 20.0),
        ))
        projections = [project(candidate, centre) for centre in centres]
        if any(value is None for value in projections):
            continue
        if not all(
            10.0 <= value[0] < WIDTH - 10.0
            and 10.0 <= value[1] < HEIGHT - 10.0
            for value in projections
            if value is not None
        ):
            continue
        anchor = projections[0]
        if anchor is None or any(
            math.dist(anchor, project(point, centres[0])) < 14.0
            for point in ground_points
        ):
            continue
        ground_points.append(candidate)
        if len(ground_points) == 220:
            break
    assert len(ground_points) == 220
    for point_index, point in enumerate(ground_points):
        small = rng.integers(80, 176, (7, 7), dtype=np.uint8)
        patch = cv2.resize(small, (17, 17), interpolation=cv2.INTER_CUBIC)
        outer, inner = ((245, 10) if point_index % 2 else (10, 245))
        cv2.circle(patch, (8, 8), 5, outer, -1, cv2.LINE_8)
        cv2.circle(patch, (8, 8), 2, inner, -1, cv2.LINE_8)
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        for image, centre in zip(images, centres):
            projected = project(point, centre)
            assert projected is not None
            x_value, y_value = (int(round(value)) for value in projected)
            image[y_value - 8:y_value + 9, x_value - 8:x_value + 9] = patch

    # Restore uninterrupted architectural lines after placing the markers so
    # the Hough/RANSAC anchor evidence remains independent of marker density.
    for image_index, (image, centre) in enumerate(zip(images, centres)):
        if image_index == 0:
            continue
        for start_world, end_world in ground_lines:
            start = project(start_world, centre)
            end = project(end_world, centre)
            if start is not None and end is not None:
                cv2.line(
                    image,
                    tuple(int(round(value)) for value in start),
                    tuple(int(round(value)) for value in end),
                    (45, 45, 45),
                    2,
                    cv2.LINE_8,
                )

    direction_vps: list[tuple[int, int]] = []
    for direction in (
        np.array((1.0, 0.0, 0.0)),
        np.array((0.0, 0.0, 1.0)),
    ):
        camera_direction = rotation @ direction
        homogeneous = intrinsic @ camera_direction
        direction_vps.append(tuple(
            int(round(float(value / homogeneous[2])))
            for value in homogeneous[:2]
        ))
    for vp_index, angles_deg in (
        (0, range(-25, -5, 2)),
        (1, range(10, 56, 4)),
    ):
        vp_x, vp_y = direction_vps[vp_index]
        for angle_deg in angles_deg:
            angle = math.radians(angle_deg)
            offset_x = int(round(10000.0 * math.cos(angle)))
            offset_y = int(round(10000.0 * math.sin(angle)))
            cv2.line(
                images[0],
                (vp_x - offset_x, vp_y - offset_y),
                (vp_x + offset_x, vp_y + offset_y),
                (5, 5, 5), 1, cv2.LINE_8,
            )

    images = [np.ascontiguousarray(image, dtype=np.uint8) for image in images]
    return tuple(images)


def _frame(image: np.ndarray, index: int) -> MultiViewFrame:
    label = f"synthetic_photo_{index}"
    raw_meta = RawImportResult(
        linear_rgb=image,
        display_srgb=image,
        width=WIDTH,
        height=HEIGHT,
        focal_length_mm=FOCAL_MM,
        sensor_width_mm=SENSOR_WIDTH_MM,
        sensor_height_mm=SENSOR_HEIGHT_MM,
        sensor_source="committed_synthetic_fixture",
        camera_make="Atlas",
        camera_model="Determinism Rig",
        lens_model="Atlas Prime 35",
        undistort_applied=True,
        undistort_status="applied",
        orientation=1,
        body_serial_number="DETERMINISM-BODY-1",
        lens_serial_number="DETERMINISM-LENS-1",
        capture_datetime=f"2026:08:09 12:00:0{index}",
        metadata_source="committed_synthetic_fixture",
    )
    plate_ref = AtlasPlateRef(
        image_path=f"fixtures/multiview/{label}.exr",
        preview_b64=f"data:image/png;base64,{label}",
        colorspace="ACEScg",
        bit_depth="float32",
        role="source",
        is_proxy=False,
    )
    return MultiViewFrame(image, raw_meta, plate_ref, label)


def main() -> None:
    ambient_seed = int(sys.argv[1])
    random.seed(ambient_seed)
    np.random.seed(ambient_seed)
    cv2.setRNGSeed(ambient_seed)

    frames = tuple(
        _frame(image, index)
        for index, image in enumerate(_render_views(), start=1)
    )
    outcome = solve_multiview(
        frames,
        MultiViewSettings(
            capture_mode="translated",
            camera_height_m=1.6,
            match_quality="permissive",
            seed=SOLVER_SEED,
        ),
    )
    if outcome.solve is None:
        raise RuntimeError(json.dumps(outcome.diagnostics.to_dict(), sort_keys=True))

    solve = outcome.solve
    diagnostics = outcome.diagnostics.to_dict()
    assert outcome.diagnostics.outcome_code == "translated"
    assert outcome.diagnostics.selected_mode == "translated"
    assert len(solve.projection_sources) == 2
    assert len(outcome.diagnostics.camera_metrics) == 3
    assert len(outcome.diagnostics.pair_metrics) == 3
    assert all(metric["mutual_matches"] > 0 for metric in diagnostics["pair_metrics"])
    assert all(metric["essential_inliers"] > 0 for metric in diagnostics["pair_metrics"])
    assert solve.landmarks
    assert solve.debug_metadata["accepted_track_count"] > 0
    assert solve.debug_metadata["generated_inputs_used"] is False
    assert solve.debug_metadata["photographed_source_count"] == 3
    assert all(
        source.metadata["evidence_type"] == "photographed"
        for source in solve.projection_sources
    )

    payload = {
        "diagnostics": diagnostics,
        "solve": json.loads(solve.to_json()),
    }
    sys.stdout.write(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ))


if __name__ == "__main__":
    main()
