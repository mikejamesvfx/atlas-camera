"""Still-image camera solving entry points."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from atlas_camera.core.camera_math import FOCAL_FALLBACK_CONFIDENCE_PENALTY
from atlas_camera.core.confidence import ConfidenceModel
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.projection_scene import create_default_projection_scene
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasHorizon,
    AtlasProjectionScene,
    AtlasProxyPrimitive,
    AtlasSolve,
    identity_matrix4,
)
from atlas_camera.core.vanishing_points import (
    VanishingPointDetector,
    draw_debug_overlay,
    fit_vanishing_point_from_lines,
    flatten_line_segment,
    normalize_line_segment,
)
from atlas_camera.reference_data import get_scale_reference


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Vanishing-point camera estimation requires numpy. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return np


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Image-based camera solving requires opencv-python. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return cv2


def _image_size_from_pillow(image_path: str | Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Image size was not provided and Pillow is not installed. "
            "Install atlas-camera[image] or pass image_size."
        ) from exc

    with Image.open(image_path) as image:
        return image.size


def _image_size_from_cv2(image_path: str | Path) -> tuple[int, int]:
    image = _load_image_bgr(image_path)
    height, width = image.shape[:2]
    return width, height


def _load_image_bgr(image_path: str | Path) -> Any:
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image for camera solving: {image_path}")
    return image


def _write_debug_overlay(path: str | Path, image: Any) -> Path:
    cv2 = _require_cv2()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Unable to write debug overlay: {destination}")
    return destination


def _matrix3_to_tuple(matrix: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _matrix4_with_rotation_translation(rotation: Any, translation: Any) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), float(translation[0])),
        (float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), float(translation[1])),
        (float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), float(translation[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _world_matrix_with_position(rotation: Any, position: Any) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), float(position[0])),
        (float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), float(position[1])),
        (float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), float(position[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _horizon_endpoints(
    coefficients: tuple[float, float, float],
    image_width: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    a, b, c = coefficients
    if abs(b) < 1e-9:
        return None
    return ((0.0, -c / b), (float(image_width), -(a * image_width + c) / b))


def _json_safe_camera_result(camera_result: dict[str, Any]) -> dict[str, Any]:
    rotation = camera_result["rotation_matrix"]
    translation = camera_result["translation"]
    position = camera_result["camera_position"]
    return {
        "focal_length_px": float(camera_result["focal_length_px"]),
        "focal_length_mm": float(camera_result["focal_length_mm"]),
        "principal_point": [float(value) for value in camera_result["principal_point"]],
        "rotation_matrix": [
            [float(value) for value in row]
            for row in rotation
        ],
        "translation": [float(value) for value in translation],
        "camera_position": [float(value) for value in position],
        "horizon_angle": float(camera_result["horizon_angle"]),
        "fov_horizontal_deg": float(camera_result["fov_horizontal_deg"]),
        "fov_vertical_deg": float(camera_result["fov_vertical_deg"]),
        "focal_source": camera_result.get("focal_source", "unknown"),
        "focal_length_inferred": bool(camera_result.get("focal_length_inferred", False)),
        "focal_assumption": camera_result.get("focal_assumption"),
    }


def _fallback_focal_note(sensor_width_mm: float) -> str:
    return (
        "Focal length was inferred from vanishing-point geometry using an "
        f"explicit {sensor_width_mm:g} mm sensor-width assumption."
    )


def _camera_confidence(
    *,
    global_score: float,
    horizon: float = 0.0,
    vp1: float = 0.0,
    vp2: float = 0.0,
    vp3: float = 0.0,
    focal: float = 0.0,
    extrinsics: float = 0.0,
    sensor: float = 0.0,
) -> ConfidenceModel:
    return ConfidenceModel.for_latent_camera(
        global_score=global_score,
        overrides={
            "horizon": horizon,
            "vp1": vp1,
            "vp2": vp2,
            "vp3": vp3,
            "focal": focal,
            "extrinsics": extrinsics,
            "sensor": sensor,
        },
    )


class CameraFromVanishingPoints:
    """Estimate intrinsics and orientation from two orthogonal vanishing points."""

    @staticmethod
    def estimate_focal_length(vp1: Any, vp2: Any, principal_point: Any) -> float | None:
        np = _require_numpy()
        d1 = vp1 - principal_point
        d2 = vp2 - principal_point
        focal_squared = -np.dot(d1, d2)
        if focal_squared <= 0:
            return None
        return float(np.sqrt(focal_squared))

    @staticmethod
    def estimate_rotation(vp1: Any, vp2: Any, focal_length: float, principal_point: Any) -> Any:
        np = _require_numpy()
        d1 = np.array(
            [vp1[0] - principal_point[0], vp1[1] - principal_point[1], focal_length],
            dtype=np.float64,
        )
        d2 = np.array(
            [vp2[0] - principal_point[0], vp2[1] - principal_point[1], focal_length],
            dtype=np.float64,
        )
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)

        d3 = np.cross(d1, d2)
        d3 = d3 / np.linalg.norm(d3)
        d2 = np.cross(d3, d1)
        d2 = d2 / np.linalg.norm(d2)

        rotation = np.column_stack([d1, d2, d3])
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] = -rotation[:, 2]
        return rotation

    @staticmethod
    def estimate_horizon_line(vp1: Any, vp2: Any) -> tuple[float, float, float]:
        np = _require_numpy()
        p1 = np.array([vp1[0], vp1[1], 1.0])
        p2 = np.array([vp2[0], vp2[1], 1.0])
        horizon = np.cross(p1, p2)
        norm = np.sqrt(horizon[0] ** 2 + horizon[1] ** 2)
        if norm > 1e-10:
            horizon = horizon / norm
        return float(horizon[0]), float(horizon[1]), float(horizon[2])

    @classmethod
    def estimate_camera(
        cls,
        vp1: Any,
        vp2: Any,
        image_width: int,
        image_height: int,
        camera_height: float = 1.6,
        sensor_width_mm: float = 36.0,
        focal_length_mm: float | None = None,
        vp3: Any | None = None,
        principal_point: Any | None = None,
    ) -> dict[str, Any]:
        np = _require_numpy()
        if principal_point is None:
            principal = np.array([image_width / 2.0, image_height / 2.0])
        else:
            principal = np.array(principal_point, dtype=np.float64)

        vp1 = np.array(vp1, dtype=np.float64)
        vp2 = np.array(vp2, dtype=np.float64)
        focal_source = "vanishing_point_orthogonality"
        if focal_length_mm is not None:
            focal_px = focal_length_mm * image_width / sensor_width_mm
            focal_source = "known_focal_length_hint"
        else:
            focal_px = cls.estimate_focal_length(vp1, vp2, principal)
        focal_length_inferred = False
        focal_assumption = None
        if focal_px is None:
            focal_px = image_width * 1.2
            focal_source = "fallback_default"
            focal_length_inferred = True
            focal_assumption = f"sensor_width_mm={sensor_width_mm:g}; focal_px=image_width*1.2"
        focal_px = float(np.clip(focal_px, image_width * 0.3, image_width * 5.0))
        focal_mm = focal_px * sensor_width_mm / image_width

        rotation = cls.estimate_rotation(vp1, vp2, focal_px, principal)
        horizon_a, horizon_b, horizon_c = cls.estimate_horizon_line(vp1, vp2)
        horizon_angle = float(np.degrees(np.arctan2(horizon_a, -horizon_b)))
        camera_position = np.array([0.0, camera_height, 0.0], dtype=np.float64)
        translation = -rotation.T @ camera_position
        fov_h = 2.0 * math.degrees(math.atan(image_width / (2.0 * focal_px)))
        fov_v = 2.0 * math.degrees(math.atan(image_height / (2.0 * focal_px)))

        conversion_info = {
            "focal_length_px": [[focal_px, focal_px]],
            "principal_point": [[float(principal[0]), float(principal[1])]],
            "rotation_matrix": [rotation.tolist()],
            "translation": [translation.tolist()],
            "camera_position": camera_position.tolist(),
            "image_size": [[image_height, image_width]],
            "coordinate_transform_applied": "estimated_from_vanishing_points",
            "intrinsics_info": {
                "focal_length_px": focal_px,
                "focal_length_mm": focal_mm,
                "principal_point": [float(principal[0]), float(principal[1])],
                "fov_horizontal_deg": fov_h,
                "fov_vertical_deg": fov_v,
                "sensor_width_mm": sensor_width_mm,
                "estimation_method": focal_source,
                "focal_length_inferred": focal_length_inferred,
                "focal_assumption": focal_assumption,
            },
        }
        return {
            "focal_length_px": focal_px,
            "focal_length_mm": focal_mm,
            "principal_point": [float(principal[0]), float(principal[1])],
            "rotation_matrix": rotation,
            "camera_position": camera_position,
            "translation": translation,
            "horizon_angle": horizon_angle,
            "fov_horizontal_deg": fov_h,
            "fov_vertical_deg": fov_v,
            "conversion_info": conversion_info,
            "focal_source": focal_source,
            "focal_length_inferred": focal_length_inferred,
            "focal_assumption": focal_assumption,
            "vp3_used": vp3 is not None,
        }


def solve_from_vanishing_points(
    vp1: Any,
    vp2: Any,
    *,
    image_width: int,
    image_height: int,
    image_path: str | Path | None = None,
    camera_height: float = 1.6,
    sensor_width_mm: float = 36.0,
    focal_length_mm: float | None = None,
    vp3: Any | None = None,
    principal_point: Any | None = None,
    vp_result: dict[str, Any] | None = None,
    seed: int = 0,
) -> AtlasSolve:
    camera_result = CameraFromVanishingPoints.estimate_camera(
        vp1=vp1,
        vp2=vp2,
        image_width=image_width,
        image_height=image_height,
        camera_height=camera_height,
        sensor_width_mm=sensor_width_mm,
        focal_length_mm=focal_length_mm,
        vp3=vp3,
        principal_point=principal_point,
    )
    principal = camera_result["principal_point"]
    intrinsics = build_intrinsics(
        image_width=image_width,
        image_height=image_height,
        focal_length_mm=camera_result["focal_length_mm"],
        sensor_width_mm=sensor_width_mm,
        principal_point_px=(principal[0], principal[1]),
        fx_px=camera_result["focal_length_px"],
        fy_px=camera_result["focal_length_px"],
    )
    rotation = camera_result["rotation_matrix"]
    position = camera_result["camera_position"]
    translation = camera_result["translation"]
    extrinsics = AtlasExtrinsics(
        camera_position=tuple(float(value) for value in position),
        camera_rotation_matrix=_matrix3_to_tuple(rotation),
        camera_world_matrix=_world_matrix_with_position(rotation, position),
        camera_view_matrix=_matrix4_with_rotation_translation(rotation.T, translation),
        coordinate_system="right_handed",
        up_axis="Y",
        projection_convention="Vanishing-point pinhole estimate, image origin top-left.",
    )

    horizon = CameraFromVanishingPoints.estimate_horizon_line(vp1, vp2)
    horizon_line = AtlasHorizon(
        line_coefficients=horizon,
        endpoints_px=_horizon_endpoints(horizon, image_width),
        confidence=0.75,
    )

    if vp_result is not None:
        vanishing_points = VanishingPointDetector.to_schema_vanishing_points(vp_result)
        total_lines = int(vp_result.get("num_lines_total", 0))
    else:
        total_lines = 0
        vanishing_points = []

    scene = create_default_projection_scene()
    scene.debug_metadata["solve_mode"] = "vanishing_point_orthogonality"
    global_confidence = 0.75 if len(vanishing_points) >= 2 else 0.5
    focal_inferred = bool(camera_result.get("focal_length_inferred", False))
    focal_confidence = 0.85 if focal_length_mm is not None else 0.75
    if focal_inferred:
        focal_confidence = max(0.0, focal_confidence - FOCAL_FALLBACK_CONFIDENCE_PENALTY)
    camera_notes = [_fallback_focal_note(sensor_width_mm)] if focal_inferred else []

    return AtlasSolve(
        camera=AtlasCamera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name="atlas_estimated_camera",
            confidence=_camera_confidence(
                global_score=global_confidence,
                horizon=horizon_line.confidence,
                vp1=vanishing_points[0].confidence if len(vanishing_points) > 0 else 0.0,
                vp2=vanishing_points[1].confidence if len(vanishing_points) > 1 else 0.0,
                vp3=vanishing_points[2].confidence if len(vanishing_points) > 2 else 0.0,
                focal=focal_confidence,
                extrinsics=0.75,
                sensor=1.0 if focal_length_mm is not None else 0.75,
            ),
            notes=camera_notes,
            focal_length_inferred=focal_inferred,
            seed=seed,
        ),
        image_path=str(image_path) if image_path else None,
        image_width=image_width,
        image_height=image_height,
        vanishing_points=vanishing_points,
        horizon_line=horizon_line,
        confidence=global_confidence,
        source_method="automatic_still_image_vanishing_points",
        known_intrinsics_used=False,
        projection_scene=scene,
        debug_metadata={
            "solver_status": "vanishing-point solve",
            "num_lines_total": total_lines,
            "camera_estimation": _json_safe_camera_result(camera_result),
            "coordinate_note": "Rotation estimate is stored in Atlas right-handed Y-up schema.",
            "seed": seed,
            "notes": camera_notes,
            "warnings": camera_notes if focal_inferred else [],
        },
    )


def _metadata_only_solve(
    image_path: str | Path,
    image_size: tuple[int, int],
    intrinsics_hint: dict[str, Any],
    *,
    source_method: str,
    debug_metadata: dict[str, Any] | None = None,
    projection_scene: AtlasProjectionScene | None = None,
    seed: int = 0,
) -> AtlasSolve:
    intrinsics = build_intrinsics(
        image_width=image_size[0],
        image_height=image_size[1],
        focal_length_mm=intrinsics_hint.get("focal_length_mm"),
        sensor_width_mm=intrinsics_hint.get("sensor_width_mm", 36.0),
        sensor_height_mm=intrinsics_hint.get("sensor_height_mm"),
        principal_point_px=intrinsics_hint.get("principal_point_px"),
        fx_px=intrinsics_hint.get("fx_px"),
        fy_px=intrinsics_hint.get("fy_px"),
    )
    camera = AtlasCamera(
        intrinsics=intrinsics,
        extrinsics=AtlasExtrinsics(camera_world_matrix=identity_matrix4()),
        name="atlas_estimated_camera",
        confidence=_camera_confidence(
            global_score=0.0,
            focal=1.0 if intrinsics_hint.get("focal_length_mm") else 0.0,
            sensor=1.0 if intrinsics_hint.get("sensor_width_mm") else 0.0,
        ),
        seed=seed,
    )
    return AtlasSolve(
        camera=camera,
        image_path=str(image_path),
        image_width=image_size[0],
        image_height=image_size[1],
        confidence=0.0,
        source_method=source_method,
        known_intrinsics_used=bool(intrinsics_hint),
        projection_scene=projection_scene or create_default_projection_scene(),
        debug_metadata={**(debug_metadata or {}), "seed": seed},
    )


def solve_still_image(
    image_path: str | Path,
    *,
    image_size: tuple[int, int] | None = None,
    intrinsics_hint: dict[str, Any] | None = None,
    detect_vanishing_points: bool = False,
    debug_overlay_path: str | Path | None = None,
    camera_height: float = 1.6,
    detection_options: dict[str, Any] | None = None,
    seed: int = 0,
) -> AtlasSolve:
    path = Path(image_path)
    intrinsics_hint = intrinsics_hint or {}

    if not detect_vanishing_points:
        if image_size is None:
            image_size = _image_size_from_pillow(path)
        return _metadata_only_solve(
            path,
            image_size,
            intrinsics_hint,
            source_method="automatic_still_image_metadata_only",
            debug_metadata={
                "solver_status": "metadata-only solve",
                "next_step": "enable detect_vanishing_points=True for image line detection",
            },
            seed=seed,
        )

    image = _load_image_bgr(path)
    height, width = image.shape[:2]
    detection_options = detection_options or {}
    detection_options.setdefault("random_seed", seed)
    vp_result = VanishingPointDetector.detect_vanishing_points(
        image,
        **detection_options,
    )

    vp1 = vp_result.get("vp1")
    vp2 = vp_result.get("vp2")
    camera_result = None
    if vp1 is not None and vp2 is not None:
        solve = solve_from_vanishing_points(
            vp1,
            vp2,
            image_width=width,
            image_height=height,
            image_path=path,
            camera_height=camera_height,
            sensor_width_mm=intrinsics_hint.get("sensor_width_mm", 36.0),
            focal_length_mm=intrinsics_hint.get("focal_length_mm"),
            vp3=vp_result.get("vp3"),
            principal_point=intrinsics_hint.get("principal_point_px"),
            vp_result=vp_result,
            seed=seed,
        )
        solve.known_intrinsics_used = bool(intrinsics_hint)
        camera_result = solve.debug_metadata["camera_estimation"]
    else:
        scene = create_default_projection_scene()
        scene.debug_metadata["solve_mode"] = "vanishing_point_detection_fallback"
        solve = _metadata_only_solve(
            path,
            (width, height),
            intrinsics_hint,
            source_method="automatic_still_image_vanishing_point_fallback",
            projection_scene=scene,
            debug_metadata={
                "solver_status": "insufficient vanishing points",
                "num_lines_total": int(vp_result.get("num_lines_total", 0)),
                "detected_vanishing_points": len(
                    VanishingPointDetector.to_schema_vanishing_points(vp_result)
                ),
            },
            seed=seed,
        )
        solve.vanishing_points = VanishingPointDetector.to_schema_vanishing_points(vp_result)

    if debug_overlay_path is not None:
        overlay = draw_debug_overlay(image, vp_result, camera_result)
        overlay_path = _write_debug_overlay(debug_overlay_path, overlay)
        solve.debug_metadata["debug_overlay_path"] = str(overlay_path)

    return solve


def _first_available(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _constraint_image_size(
    image_path: str | Path,
    constraints: dict[str, Any],
    image_size: tuple[int, int] | None,
) -> tuple[int, int]:
    if image_size is not None:
        return image_size
    if "image_width" in constraints and "image_height" in constraints:
        return int(constraints["image_width"]), int(constraints["image_height"])
    if "image_size" in constraints:
        value = constraints["image_size"]
        if isinstance(value, dict):
            return int(value["width"]), int(value["height"])
        return int(value[0]), int(value[1])
    try:
        return _image_size_from_cv2(image_path)
    except RuntimeError:
        return _image_size_from_pillow(image_path)


def _constraint_intrinsics_hint(
    constraints: dict[str, Any],
    intrinsics_hint: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(constraints.get("intrinsics_hint", {}))
    for key in (
        "focal_length_mm",
        "sensor_width_mm",
        "sensor_height_mm",
        "principal_point_px",
        "fx_px",
        "fy_px",
    ):
        if key in constraints and key not in merged:
            merged[key] = constraints[key]
    merged.update(intrinsics_hint or {})
    return merged


def _point_from_constraint(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _guided_vp_result(constraints: dict[str, Any]) -> dict[str, Any]:
    line_groups = (
        constraints.get("line_groups")
        or constraints.get("vanishing_line_groups")
        or constraints.get("line_constraints")
        or {}
    )
    explicit_vps = constraints.get("vanishing_points") or constraints.get("vps") or {}

    if isinstance(explicit_vps, list):
        vp1 = _point_from_constraint(explicit_vps[0]) if len(explicit_vps) > 0 else None
        vp2 = _point_from_constraint(explicit_vps[1]) if len(explicit_vps) > 1 else None
        vp3 = _point_from_constraint(explicit_vps[2]) if len(explicit_vps) > 2 else None
    else:
        vp1 = _first_available(explicit_vps, ("left", "vp1", "x", "horizontal_left"))
        vp2 = _first_available(explicit_vps, ("right", "vp2", "z", "horizontal_right"))
        vp3 = _first_available(explicit_vps, ("vertical", "vp3", "y"))
        vp1 = _point_from_constraint(vp1) if vp1 is not None else None
        vp2 = _point_from_constraint(vp2) if vp2 is not None else None
        vp3 = _point_from_constraint(vp3) if vp3 is not None else None

    left_lines = _first_available(line_groups, ("left", "vp1", "x", "horizontal_left")) or []
    right_lines = _first_available(line_groups, ("right", "vp2", "z", "horizontal_right")) or []
    vertical_lines = _first_available(line_groups, ("vertical", "vp3", "y")) or []

    left_segments = [normalize_line_segment(line) for line in left_lines]
    right_segments = [normalize_line_segment(line) for line in right_lines]
    vertical_segments = [normalize_line_segment(line) for line in vertical_lines]

    if vp1 is None:
        vp1 = fit_vanishing_point_from_lines(left_segments, direction_label="left").position_px
    if vp2 is None:
        vp2 = fit_vanishing_point_from_lines(right_segments, direction_label="right").position_px
    if vp3 is None and len(vertical_segments) >= 2:
        vp3 = fit_vanishing_point_from_lines(vertical_segments, direction_label="vertical").position_px

    flat_left = [flatten_line_segment(line) for line in left_segments]
    flat_right = [flatten_line_segment(line) for line in right_segments]
    flat_vertical = [flatten_line_segment(line) for line in vertical_segments]
    all_lines = flat_left + flat_right + flat_vertical
    return {
        "vp1": vp1,
        "vp2": vp2,
        "vp3": vp3,
        "lines": all_lines,
        "left_lines": flat_left,
        "right_lines": flat_right,
        "vertical_lines": flat_vertical,
        "num_lines_total": len(all_lines),
        "constraint_source": "artist_guided",
    }


def _scale_measurement_height(measurement: dict[str, Any]) -> float | None:
    for key in ("height", "known_height", "world_height", "height_world"):
        if key in measurement:
            return float(measurement[key])
    return None


def _scale_measurement_segment(measurement: dict[str, Any]) -> list[list[float]] | None:
    value = (
        measurement.get("image_points")
        or measurement.get("image_segment")
        or measurement.get("segment")
        or measurement.get("points")
    )
    if value is None:
        return None
    segment = normalize_line_segment(value)
    return [
        [float(segment[0][0]), float(segment[0][1])],
        [float(segment[1][0]), float(segment[1][1])],
    ]


def _constraint_scale_measurements(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (
        constraints.get("scale_constraints")
        or constraints.get("known_measurements")
        or constraints.get("measurements")
        or []
    )
    if isinstance(raw, dict):
        if "items" in raw:
            raw_items = raw["items"]
        else:
            raw_items = [raw]
    else:
        raw_items = raw

    measurements: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError("Scale measurements must be dictionaries.")
        reference = None
        reference_id = item.get("reference_id")
        if reference_id:
            reference = get_scale_reference(str(reference_id))

        height = _scale_measurement_height(item)
        if height is None and reference is not None:
            height = reference.height
        if height is None:
            raise ValueError(
                "Scale measurements require height, known_height, world_height, or reference_id."
            )
        segment = _scale_measurement_segment(item)
        name = str(
            item.get("name")
            or item.get("label")
            or (reference.id if reference else f"scale_reference_{index + 1}")
        )
        metadata = dict(item.get("metadata", {}))
        dimensions = {
            "height": height,
            "width": item.get("width"),
            "depth": item.get("depth"),
        }
        if reference is not None:
            metadata.update(
                {
                    "reference_id": reference.id,
                    "reference_label": reference.label,
                    "reference_category": reference.category,
                    "reference_confidence": reference.confidence,
                    "source_url": reference.source_url,
                    "source_note": reference.source_note,
                    "reference_notes": reference.notes,
                    "asset_hint": reference.asset_hint,
                    "tags": list(reference.tags),
                }
            )
            dimensions["width"] = dimensions["width"] if dimensions["width"] is not None else reference.width
            dimensions["depth"] = dimensions["depth"] if dimensions["depth"] is not None else reference.depth

        measurements.append(
            {
                "name": name,
                "type": item.get("type", "vertical_height"),
                "height": height,
                "units": item.get(
                    "units",
                    reference.units if reference is not None else constraints.get("units", "world_units"),
                ),
                "dimensions": dimensions,
                "image_segment": segment,
                "confidence": float(item.get("confidence", 1.0)),
                "metadata": metadata,
            }
        )

    if "known_object_height" in constraints:
        measurements.append(
            {
                "name": str(constraints.get("object_name", "known_object")),
                "type": "vertical_height",
                "height": float(constraints["known_object_height"]),
                "units": constraints.get("units", "world_units"),
                "dimensions": {
                    "height": float(constraints["known_object_height"]),
                    "width": None,
                    "depth": None,
                },
                "image_segment": _scale_measurement_segment(
                    {
                        "image_points": constraints.get("object_image_points")
                        or constraints.get("object_image_segment")
                    }
                )
                if (
                    constraints.get("object_image_points")
                    or constraints.get("object_image_segment")
                )
                else None,
                "confidence": 1.0,
                "metadata": {},
            }
        )
    return measurements


def _attach_scale_measurements(solve: AtlasSolve, constraints: dict[str, Any]) -> None:
    measurements = _constraint_scale_measurements(constraints)
    if not measurements:
        solve.debug_metadata["scale_constraints"] = {
            "count": 0,
            "status": "none_supplied",
        }
        return

    for measurement in measurements:
        landmark = {
            "name": measurement["name"],
            "type": measurement["type"],
            "known_height": measurement["height"],
            "units": measurement["units"],
            "dimensions": measurement["dimensions"],
            "image_segment": measurement["image_segment"],
            "confidence": measurement["confidence"],
            "metadata": measurement["metadata"],
            "interpretation": (
                "Explicit artist scale reference. Stored for review and DCC guides; "
                "metric depth fitting is not solved yet."
            ),
        }
        solve.landmarks.append(landmark)
        solve.projection_scene.landmarks.append(landmark)
        width = measurement["dimensions"].get("width") or 0.05
        depth = measurement["dimensions"].get("depth") or 0.05
        solve.projection_scene.proxy_geometry.append(
            AtlasProxyPrimitive(
                name=f"{measurement['name']}_height_guide",
                primitive_type="height_guide",
                dimensions=(float(width), measurement["height"], float(depth)),
                material="atlas_scale_reference",
                metadata={
                    "role": "scale_constraint",
                    "known_height": measurement["height"],
                    "units": measurement["units"],
                    "dimensions": measurement["dimensions"],
                    "source_landmark": measurement["name"],
                    "image_segment": measurement["image_segment"],
                    "reference_id": measurement["metadata"].get("reference_id"),
                    "source_url": measurement["metadata"].get("source_url"),
                    "depth_solved": False,
                },
            )
        )

    solve.debug_metadata["scale_constraints"] = {
        "count": len(measurements),
        "units": sorted({measurement["units"] for measurement in measurements}),
        "status": "recorded_as_landmarks_and_height_guides",
        "metric_depth_solved": False,
        "reference_ids": [
            measurement["metadata"]["reference_id"]
            for measurement in measurements
            if measurement["metadata"].get("reference_id")
        ],
    }


def solve_from_constraints(
    image_path: str | Path,
    constraints: dict[str, Any],
    intrinsics_hint: dict[str, Any] | None = None,
    *,
    image_size: tuple[int, int] | None = None,
    debug_overlay_path: str | Path | None = None,
    seed: int = 0,
) -> AtlasSolve:
    resolved_size = _constraint_image_size(image_path, constraints, image_size)
    hints = _constraint_intrinsics_hint(constraints, intrinsics_hint)
    vp_result = _guided_vp_result(constraints)
    if vp_result["vp1"] is None or vp_result["vp2"] is None:
        raise ValueError("Artist-guided solve requires left/right vanishing points or line groups.")

    solve = solve_from_vanishing_points(
        vp_result["vp1"],
        vp_result["vp2"],
        image_width=resolved_size[0],
        image_height=resolved_size[1],
        image_path=image_path,
        camera_height=float(constraints.get("camera_height", 1.6)),
        sensor_width_mm=float(hints.get("sensor_width_mm", 36.0)),
        focal_length_mm=hints.get("focal_length_mm"),
        vp3=vp_result.get("vp3"),
        principal_point=hints.get("principal_point_px"),
        vp_result=vp_result,
        seed=seed,
    )
    solve.source_method = "artist_guided_constraints"
    solve.known_intrinsics_used = bool(hints)
    solve.confidence = 0.85 if vp_result["num_lines_total"] >= 4 else 0.7
    solve.debug_metadata["solver_status"] = "artist-guided line constraint solve"
    solve.debug_metadata["artist_constraints"] = constraints
    solve.debug_metadata["constraint_summary"] = {
        "left_line_count": len(vp_result["left_lines"]),
        "right_line_count": len(vp_result["right_lines"]),
        "vertical_line_count": len(vp_result["vertical_lines"]),
        "explicit_vanishing_points_used": bool(
            constraints.get("vanishing_points") or constraints.get("vps")
        ),
    }
    solve.debug_metadata["seed"] = seed
    solve.camera.seed = seed
    _attach_scale_measurements(solve, constraints)

    if debug_overlay_path is not None:
        image = _load_image_bgr(image_path)
        overlay = draw_debug_overlay(
            image,
            vp_result,
            solve.debug_metadata.get("camera_estimation"),
        )
        overlay_path = _write_debug_overlay(debug_overlay_path, overlay)
        solve.debug_metadata["debug_overlay_path"] = str(overlay_path)

    return solve


class StillImageCameraEstimator:
    """Thin class wrapper for automatic still-image solving."""

    def solve(
        self,
        image_path: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        intrinsics_hint: dict[str, Any] | None = None,
        detect_vanishing_points: bool = False,
        debug_overlay_path: str | Path | None = None,
        camera_height: float = 1.6,
        detection_options: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> AtlasSolve:
        return solve_still_image(
            image_path,
            image_size=image_size,
            intrinsics_hint=intrinsics_hint,
            detect_vanishing_points=detect_vanishing_points,
            debug_overlay_path=debug_overlay_path,
            camera_height=camera_height,
            detection_options=detection_options,
            seed=seed,
        )
