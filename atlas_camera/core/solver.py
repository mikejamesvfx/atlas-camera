"""Still-image camera solving entry points."""

from __future__ import annotations

import math
import warnings
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

        # d1, d2, d3 are three mutually orthogonal scene directions expressed in
        # camera coordinates (camera x-right, y-up, z-back). Atlas world is Y-up
        # (camera_position=[0,h,0], ground plane Y=0), so exactly one of these
        # must map to world +Y. We must NOT assume it is d3: depending on which
        # vanishing points the detector returns, the vertical scene axis can be
        # any of the three. The vertical scene axis is the one whose camera-space
        # direction is most aligned with the image's vertical (largest |y|).
        candidates = [d1, d2, d3]
        up_idx = int(np.argmax([abs(c[1]) for c in candidates]))
        up = candidates[up_idx].copy()
        if up[1] < 0:  # make world +Y point up in the image
            up = -up
        right, forward = (candidates[i] for i in range(3) if i != up_idx)

        # Assemble [right, up, forward] and restore a right-handed basis.
        rotation = np.column_stack([right, up, forward])
        if np.linalg.det(rotation) < 0:
            rotation = np.column_stack([right, up, -forward])
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


def _rotation_from_up_vector(up: Any) -> Any:
    """Build a world->cam rotation (Atlas Y-up) from the world-up direction in cam coords.

    Yaw is unobservable from a single up-vector, so world +Z (forward) is chosen as
    the camera's forward projected onto the horizontal plane. Columns are the world
    axes [right(+X), up(+Y), forward(+Z)] expressed in camera coordinates.
    """
    np = _require_numpy()
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) or 1.0)
    fwd_cam = np.array([0.0, 0.0, -1.0])  # camera looks along -Z
    forward = fwd_cam - np.dot(fwd_cam, up) * up
    n = np.linalg.norm(forward)
    if n < 1e-6:
        # Degenerate: camera looking straight up/down. Use camera +X to seed yaw.
        forward = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), up) * up
        n = np.linalg.norm(forward)
    forward = forward / n
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) or 1.0)
    rotation = np.column_stack([right, up, forward])
    if np.linalg.det(rotation) < 0:
        rotation = np.column_stack([-right, up, forward])
    return rotation


def _learned_prior_confidence(prior: Any) -> tuple[float, float, float]:
    """Map GeoCalib uncertainties to (global, orientation, focal) confidences in [0,1]."""
    def _from_deg(unc: float | None, scale: float) -> float:
        if unc is None:
            return 0.7
        return float(max(0.25, min(0.98, 1.0 - abs(unc) / scale)))

    roll_c = _from_deg(getattr(prior, "roll_uncertainty_deg", None), 15.0)
    pitch_c = _from_deg(getattr(prior, "pitch_uncertainty_deg", None), 15.0)
    orient_c = min(roll_c, pitch_c)
    focal_unc = getattr(prior, "focal_uncertainty_px", None)
    focal_c = 0.7 if focal_unc is None else float(
        max(0.25, min(0.98, 1.0 - abs(focal_unc) / max(prior.focal_px, 1.0)))
    )
    return (0.5 * orient_c + 0.5 * focal_c), orient_c, focal_c


def solve_from_learned_prior(
    prior: Any,
    *,
    image_path: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    camera_height: float = 1.6,
    sensor_width_mm: float = 36.0,
    seed: int = 0,
) -> AtlasSolve:
    """Build an :class:`AtlasSolve` from a learned :class:`CameraPrior`.

    Pure numpy — takes the plain-Python prior so `atlas_camera.core` stays free of
    torch. The prior supplies focal + gravity; camera height sets metric scale
    (same convention as the vanishing-point path).
    """
    np = _require_numpy()
    width, height = image_size or (prior.image_width, prior.image_height)

    fx_px = prior.focal_px * (width / max(prior.image_width, 1))
    fy_px = fx_px
    focal_mm = fx_px * sensor_width_mm / max(width, 1)
    intrinsics = build_intrinsics(
        image_width=width,
        image_height=height,
        focal_length_mm=focal_mm,
        sensor_width_mm=sensor_width_mm,
        principal_point_px=(width / 2.0, height / 2.0),
        fx_px=fx_px,
        fy_px=fy_px,
    )

    rotation = _rotation_from_up_vector(prior.up_cam)
    camera_position = np.array([0.0, camera_height, 0.0], dtype=np.float64)
    translation = -rotation.T @ camera_position
    extrinsics = AtlasExtrinsics(
        camera_position=tuple(float(v) for v in camera_position),
        camera_rotation_matrix=_matrix3_to_tuple(rotation),
        camera_world_matrix=_world_matrix_with_position(rotation, camera_position),
        camera_view_matrix=_matrix4_with_rotation_translation(rotation.T, translation),
        coordinate_system="right_handed",
        up_axis="Y",
        projection_convention="Learned single-image prior, image origin top-left.",
    )

    # Horizon line from pitch: y-pixel where the horizon crosses image center.
    pitch_rad = math.radians(prior.pitch_deg)
    horizon_y = height / 2.0 + fy_px * math.tan(pitch_rad)
    horizon_line = AtlasHorizon(
        line_coefficients=(0.0, 1.0, -horizon_y),
        endpoints_px=((0.0, horizon_y), (float(width), horizon_y)),
        confidence=_learned_prior_confidence(prior)[1],
    )

    global_c, orient_c, focal_c = _learned_prior_confidence(prior)
    scene = create_default_projection_scene()
    scene.debug_metadata["solve_mode"] = "learned_prior"

    return AtlasSolve(
        camera=AtlasCamera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name="atlas_estimated_camera",
            confidence=_camera_confidence(
                global_score=global_c,
                horizon=orient_c,
                focal=focal_c,
                extrinsics=orient_c,
                sensor=0.75,
            ),
            seed=seed,
        ),
        image_path=str(image_path) if image_path else None,
        image_width=width,
        image_height=height,
        horizon_line=horizon_line,
        confidence=global_c,
        source_method=f"automatic_still_image_learned_prior:{prior.source_model}",
        known_intrinsics_used=False,
        projection_scene=scene,
        debug_metadata={
            "solver_status": "learned-prior solve",
            "learned_prior": prior.to_dict() if hasattr(prior, "to_dict") else {},
            "horizon_angle_deg": float(prior.roll_deg),
            "pitch_deg": float(prior.pitch_deg),
            "seed": seed,
            "notes": [],
            "warnings": [],
        },
    )


# Minimum ground-plane confidence (fraction of the image-bottom band confirmed as
# flat ground) required to *adopt* a depth-measured camera height over the fallback.
_HEIGHT_ADOPT_CONFIDENCE = 0.30

# Minimum reference-object scale consistency required to adopt its metric height.
# Reference objects are a stronger metric anchor than monocular depth, so this is
# more permissive; below it the candidate is surfaced for artist confirmation.
_REFERENCE_ADOPT_CONFIDENCE = 0.20


def _median_filter_3x3(depth: Any, valid: Any) -> Any:
    """Edge-clamped 3x3 median of ``depth`` over ``valid`` pixels (invalid ignored).

    Kills the single-pixel depth spikes common in monocular/AI-image depth
    before they get turned into per-pixel surface normals — a raw ±1-pixel
    finite difference (see ``estimate_ground_height_from_depth``) amplifies
    exactly this kind of noise. Same technique ``relief_mesh.py`` already uses
    for its own 3x3 median sampling; median chosen over Gaussian because it
    doesn't blur the ground/wall boundary the normal-alignment filter depends on.
    """
    np = _require_numpy()
    height, width = depth.shape
    depth_nan = np.where(valid, depth, np.nan)
    samples = []
    rows = np.arange(height)
    cols = np.arange(width)
    for dr in (-1, 0, 1):
        rr = np.clip(rows + dr, 0, height - 1)
        for dc in (-1, 0, 1):
            cc = np.clip(cols + dc, 0, width - 1)
            samples.append(depth_nan[np.ix_(rr, cc)])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filtered = np.nanmedian(np.stack(samples), axis=0)
    return np.where(np.isfinite(filtered), filtered, 0.0)


def estimate_ground_height_from_depth(
    depth: Any,
    *,
    rotation: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    horizon_y: float | None = None,
    plane_tolerance: float | None = None,
    depth_edge_rel: float = 0.05,
) -> dict[str, Any]:
    """Measure camera height above the ground by fitting the ground plane in 3D.

    Back-projects each pixel with its depth into camera space, rotates into the
    Atlas Y-up world (camera at the origin), then finds the dominant horizontal
    plane among below-horizon pixels — the ground. Its offset gives the camera's
    height. Replaces an assumed eye height with a measurement.

    ``rotation`` is the world->cam matrix (3x3). ``depth`` is a HxW forward-distance
    map (metres for a metric model, else up-to-scale). Returns a dict with
    ``camera_height``, ``confidence`` (inlier fraction), ``ground_mask`` and the
    number of ground pixels.

    Two noise-robustness passes, both borrowed from elsewhere in this codebase
    rather than invented fresh: the depth map is 3x3-median-filtered first
    (``relief_mesh.py``'s own fix for single-pixel AI-depth spikes), and pixels
    near a depth discontinuity are excluded before they can vote on the ground
    plane (the edge-rejection ``depth_geometry.back_project_normals`` already
    does for the primitive-fitting paths, but this function's own copy of the
    normal computation lacked). Raw ±1-pixel finite differences are inherently
    noise-sensitive; both passes target that without changing the downstream
    histogram-mode / bottom-band-confidence logic at all.

    Known limitation this function CANNOT fix, confirmed by direct instrumentation
    2026-07-03 (deep single-point-perspective street scene, sea horizon at the
    vanishing point): ``confidence`` here measures whether the candidate pixels
    agree on *a* consistent plane (bottom-band ground coverage) — it says nothing
    about whether the depth map's *absolute metric scale* is trustworthy. On that
    test scene the classifier correctly picked the true near-camera ground (visually
    confirmed) with confidence 0.374 (> the 0.30 adopt threshold), yet the resulting
    height was ~70% too large, because the metric depth model itself reported the
    foreground as several metres farther away than the framing implies — a systematic
    depth-model scale bias on AI-generated imagery, not a classification error. No
    amount of candidate filtering here can detect that from a single depth map with
    no external anchor. A plausibility penalty on the output height was considered
    and rejected: this toolkit explicitly supports non-eye-level cameras (see
    ``AtlasAddPatchView``'s elevation vocabulary — low-angle/elevated/high-angle
    shots), so penalizing "unusual" heights would misfire on legitimate elevated or
    drone shots. The actual fix for scenes where absolute scale matters is tier 1
    (``resolve_reference_scale`` / ``AtlasReferenceScaleSolve``) — a known-size
    reference object anchors real-world scale independent of the depth model's own
    calibration; prefer it whenever available rather than trusting this tier alone.
    """
    np = _require_numpy()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    rotation = np.asarray(rotation, dtype=np.float64)
    cam_to_world = rotation.T

    empty = {
        "camera_height": None, "confidence": 0.0, "ground_pixels": 0,
        "ground_mask": np.zeros((height, width), dtype=bool),
    }

    valid_depth = np.isfinite(depth) & (depth > 1e-4)
    depth = _median_filter_3x3(depth, valid_depth)

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    # Camera-space point at forward distance d (Atlas cam: x-right, y-up, z-back).
    x = (uu - cx) / fx * depth
    y = -(vv - cy) / fy * depth
    z = -depth
    pts_cam = np.stack([x, y, z], axis=-1)          # H×W×3
    pts_world = pts_cam @ cam_to_world.T            # rotate into world (cam at origin)
    world_y = pts_world[..., 1]

    # Per-pixel surface normals from neighbouring 3D points. Ground pixels are the
    # ones whose normal aligns with world up (+Y) — this rejects vertical walls,
    # facades and clutter that a raw height histogram would wrongly latch onto.
    du = pts_world[:, 2:, :] - pts_world[:, :-2, :]   # horizontal tangent
    dv = pts_world[2:, :, :] - pts_world[:-2, :, :]   # vertical tangent
    du = du[1:-1, :, :]
    dv = dv[:, 1:-1, :]
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-9)
    up_align = np.abs(normals[..., 1])                # |normal · world_up|

    inner = np.zeros((height, width), dtype=bool)
    inner[1:-1, 1:-1] = True

    # Depth-discontinuity rejection (same technique + default threshold as
    # depth_geometry.back_project_normals): a normal computed straddling a
    # silhouette edge is meaningless, and un-rejected it can pull the ground
    # histogram toward a wrong offset.
    ddx = np.abs(depth[:, 2:] - depth[:, :-2])
    ddy = np.abs(depth[2:, :] - depth[:-2, :])
    edge = np.zeros((height, width), dtype=bool)
    edge[:, 1:-1] |= ddx > depth_edge_rel * 2.0 * np.maximum(depth[:, 1:-1], 1e-6)
    edge[1:-1, :] |= ddy > depth_edge_rel * 2.0 * np.maximum(depth[1:-1, :], 1e-6)

    if horizon_y is None:
        horizon_y = height * 0.45
    below = (vv > horizon_y)

    horizontal = np.zeros((height, width), dtype=bool)
    horizontal[1:-1, 1:-1] = up_align > 0.90          # near-horizontal surfaces
    candidate = inner & below & horizontal & ~edge & np.isfinite(world_y) & (depth > 0)
    n_below = int((inner & below & (depth > 0)).sum())
    if candidate.sum() < 200 or n_below < 200:
        return empty

    ys = world_y[candidate]
    lo, hi = np.percentile(ys, [1, 99])
    span = float(hi - lo)
    if plane_tolerance is None:
        plane_tolerance = max(0.15, 0.03 * span)

    # Ground offset = the dominant horizontal-surface height (histogram mode). This
    # is a stable central estimate; the bottom-band confidence below decides whether
    # the fit is actually trustworthy for this image. When the surfaces are already
    # near-coplanar (tiny span), the median is the offset.
    if span < 1e-3:
        y0 = float(np.median(ys))
    else:
        hist, edges = np.histogram(ys, bins=48, range=(lo, hi))
        peak = int(np.argmax(hist))
        y0 = 0.5 * (edges[peak] + edges[peak + 1])
    refine = np.abs(ys - y0) < plane_tolerance
    if refine.sum() >= 50:
        y0 = float(np.median(ys[refine]))

    camera_height = float(-y0)
    ground_mask = candidate & (np.abs(world_y - y0) < plane_tolerance)

    # Confidence: fraction of the image-bottom band that is confirmed ground. A
    # true ground shot fills its bottom rows; a mis-scaled/degenerate fit does not.
    band_top = int(height * 0.80)
    band = np.zeros((height, width), dtype=bool)
    band[band_top:, :] = True
    band_valid = int((band & (depth > 0)).sum())
    confidence = float((ground_mask & band).sum()) / float(max(band_valid, 1))

    # Reject physically implausible camera heights (bad monocular depth scale).
    if camera_height < 0.3 or ground_mask.sum() < 300:
        return {**empty, "confidence": confidence, "plane_y": y0,
                "rejected_height": camera_height}

    return {
        "camera_height": camera_height,
        "confidence": confidence,
        "ground_pixels": int(ground_mask.sum()),
        "ground_mask": ground_mask,
        "plane_y": y0,
        "plane_tolerance": plane_tolerance,
    }


def _ray_world(u: float, v: float, fx: float, fy: float, cx: float, cy: float,
               cam_to_world: Any) -> Any:
    """Unit world-space ray (camera at origin) through image pixel (u, v)."""
    np = _require_numpy()
    ray_cam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    ray = cam_to_world @ ray_cam
    return ray / (np.linalg.norm(ray) or 1.0)


def metric_height_from_reference(
    base_px: tuple[float, float],
    top_px: tuple[float, float],
    real_height_m: float,
    *,
    rotation: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict[str, Any]:
    """Recover metric camera height from one vertical reference object of known height.

    Single-view geometry: the object stands on the ground, so its base ray hits the
    ground plane and its top sits ``real_height_m`` directly above the base in world
    Y. With the camera orientation (from GeoCalib) fixing the rays, the object's
    apparent size pins the absolute scale — no assumed eye height.

    Given base/top rays r_b, r_t (world, camera at origin) and heights, solve
    ``alpha*r_b - beta*r_t = [0, -H, 0]`` for depths (alpha, beta); the camera height
    is then ``-alpha * r_b.y`` (the base's depth below the camera).
    """
    np = _require_numpy()
    rotation = np.asarray(rotation, dtype=np.float64)
    cam_to_world = rotation.T
    r_b = _ray_world(base_px[0], base_px[1], fx, fy, cx, cy, cam_to_world)
    r_t = _ray_world(top_px[0], top_px[1], fx, fy, cx, cy, cam_to_world)

    # Base must look downward (below horizon) to intersect the ground below camera.
    if r_b[1] >= -1e-4:
        return {"camera_height": None, "reason": "reference base is above the horizon",
                "residual": None}

    A = np.column_stack([r_b, -r_t])          # 3x2
    b = np.array([0.0, -real_height_m, 0.0])
    (coeffs, *_), *_ = (np.linalg.lstsq(A, b, rcond=None),)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    residual = float(np.linalg.norm(A @ coeffs - b))

    if alpha <= 0 or beta <= 0:
        return {"camera_height": None, "reason": "degenerate reference geometry",
                "residual": residual}

    base_world = alpha * r_b
    camera_height = float(-base_world[1])
    # Confidence: how well the two rays + known height are mutually consistent,
    # normalised by the object's real size.
    consistency = float(max(0.0, 1.0 - residual / max(real_height_m, 1e-3)))
    return {
        "camera_height": camera_height if camera_height > 0 else None,
        "residual": residual,
        "consistency": consistency,
        "base_world": [float(x) for x in base_world],
        "alpha": alpha,
        "beta": beta,
    }


def _reference_segment(spec: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Extract (base_px, top_px) from a reference spec (bbox, segment, or points)."""
    if "base_px" in spec and "top_px" in spec:
        b, t = spec["base_px"], spec["top_px"]
        return (float(b[0]), float(b[1])), (float(t[0]), float(t[1]))
    seg = spec.get("image_segment") or spec.get("image_points") or spec.get("segment")
    if seg is not None:
        (p0, p1) = normalize_line_segment(seg)
        # base = lower point (larger y), top = upper point.
        base, top = (p0, p1) if p0[1] >= p1[1] else (p1, p0)
        return (float(base[0]), float(base[1])), (float(top[0]), float(top[1]))
    bbox = spec.get("bbox_px") or spec.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        # Accept xyxy or xywh; treat as xyxy unless x1<x0 / y1<y0 suggest width/height.
        if x1 < x0 or y1 < y0:
            x1, y1 = x0 + abs(x1), y0 + abs(y1)
        xc = 0.5 * (x0 + x1)
        return (xc, max(y0, y1)), (xc, min(y0, y1))
    return None


def resolve_reference_scale(
    references: list[dict[str, Any]],
    *,
    rotation: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict[str, Any]:
    """Aggregate metric camera height from one or more reference objects.

    Each reference spec provides its pixel extent (``bbox_px``/``image_segment``/
    ``base_px``+``top_px``) and its real height via ``height_m`` or ``reference_id``
    (looked up in the scale-reference registry). Returns the confidence-weighted
    median camera height plus per-reference detail.
    """
    np = _require_numpy()
    results: list[dict[str, Any]] = []
    for spec in references or []:
        seg = _reference_segment(spec)
        if seg is None:
            results.append({"status": "skipped", "reason": "no pixel extent", "spec": spec})
            continue
        height_m = spec.get("height_m") or spec.get("height") or spec.get("known_height")
        reference_id = spec.get("reference_id")
        label = spec.get("label")
        if height_m is None and reference_id:
            ref = get_scale_reference(str(reference_id))
            height_m = ref.height
            label = label or ref.label
        if height_m is None:
            results.append({"status": "skipped", "reason": "no real height", "spec": spec})
            continue
        solved = metric_height_from_reference(
            seg[0], seg[1], float(height_m), rotation=rotation, fx=fx, fy=fy, cx=cx, cy=cy
        )
        solved.update({
            "status": "solved" if solved.get("camera_height") else "rejected",
            "reference_id": reference_id,
            "label": label,
            "real_height_m": float(height_m),
            "confidence": float(spec.get("confidence", 1.0)),
        })
        results.append(solved)

    solved = [r for r in results if r.get("camera_height")]
    if not solved:
        return {"camera_height": None, "confidence": 0.0, "references": results}

    heights = np.array([r["camera_height"] for r in solved], dtype=np.float64)
    weights = np.array(
        [max(0.0, r.get("consistency", 0.5)) * r.get("confidence", 1.0) for r in solved],
        dtype=np.float64,
    )
    if weights.sum() <= 0:
        weights = np.ones_like(heights)
    order = np.argsort(heights)
    h_sorted, w_sorted = heights[order], weights[order]
    cumw = np.cumsum(w_sorted)
    median_h = float(h_sorted[int(np.searchsorted(cumw, 0.5 * cumw[-1]))])
    # Confidence: mean per-reference consistency, penalised by spread across refs.
    spread = float(np.std(heights) / max(np.mean(heights), 1e-6)) if len(heights) > 1 else 0.0
    confidence = float(np.clip(np.mean([r.get("consistency", 0.5) for r in solved]) * (1.0 - min(spread, 1.0)), 0.0, 1.0))
    return {"camera_height": median_h, "confidence": confidence, "references": results}


def solve_still_image_learned(
    image_path: str | Path,
    *,
    image_size: tuple[int, int] | None = None,
    camera_height: float | str = 1.6,
    scale_references: list[dict[str, Any]] | None = None,
    sensor_width_mm: float = 36.0,
    device: str | None = None,
    weights: str = "pinhole",
    depth_model: str | None = None,
    seed: int = 0,
) -> AtlasSolve:
    """Solve a camera from a single image using the learned GeoCalib prior.

    Robust alternative to the vanishing-point path for AI-generated images. The
    torch/geocalib dependency is imported lazily here; install with
    ``pip install -e .[neural]``.

    ``camera_height`` may be a float (assumed metres) or ``"auto"``. Metric scale is
    resolved through a **tiered cascade**, best evidence first:

    1. ``scale_references`` — one or more known-size objects (person/door/car via
       ``reference_id``, or explicit ``height_m``) with pixel extents. Solved by
       single-view geometry; the most reliable metric anchor.
    2. Depth Anything V2 ground-plane fit (when ``camera_height="auto"``), adopted
       only above a confidence threshold. Also fills the ``depth`` LatentComponent.
    3. The assumed float (last resort), always flagged as an assumption.

    Whichever tier wins, the source and confidence are recorded on the solve.
    """
    np = _require_numpy()
    from atlas_camera.inference.learned_prior import estimate_camera_prior

    prior = estimate_camera_prior(image_path, device=device, weights=weights)

    width, height = image_size or (prior.image_width, prior.image_height)
    fx = prior.focal_px * (width / max(prior.image_width, 1))
    fy = fx
    rotation = np.asarray(_rotation_from_up_vector(prior.up_cam))

    measure_height = isinstance(camera_height, str) and camera_height.lower() in (
        "auto", "measure", "depth",
    )
    depth_result = None
    ground = None
    reference_scale = None
    scale_source = "assumed_default"
    resolved_height: float = 1.6 if measure_height else float(camera_height)

    # Tier 1 — known-size reference objects (most reliable metric anchor).
    if scale_references:
        reference_scale = resolve_reference_scale(
            scale_references, rotation=rotation, fx=fx, fy=fy,
            cx=width / 2.0, cy=height / 2.0,
        )
        if reference_scale.get("camera_height") and \
                reference_scale.get("confidence", 0.0) >= _REFERENCE_ADOPT_CONFIDENCE:
            resolved_height = reference_scale["camera_height"]
            scale_source = "reference_object"

    # Tier 2 — monocular-depth ground-plane fit.
    if measure_height:
        from atlas_camera.inference.depth_estimator import (
            DEFAULT_METRIC_OUTDOOR,
            estimate_depth,
        )

        depth_result = estimate_depth(
            image_path, model_id=depth_model or DEFAULT_METRIC_OUTDOOR, device=device
        )
        depth_map = depth_result.depth
        if depth_map.shape != (height, width):
            depth_map = _resize_depth(depth_map, width, height)
        pitch_rad = math.radians(prior.pitch_deg)
        horizon_y = height / 2.0 + fy * math.tan(pitch_rad)
        ground = estimate_ground_height_from_depth(
            depth_map, rotation=rotation, fx=fx, fy=fy,
            cx=width / 2.0, cy=height / 2.0, horizon_y=horizon_y,
        )
        if scale_source != "reference_object" and ground.get("camera_height") and \
                ground.get("confidence", 0.0) >= _HEIGHT_ADOPT_CONFIDENCE:
            resolved_height = ground["camera_height"]
            scale_source = "depth_ground_plane"

    solve = solve_from_learned_prior(
        prior,
        image_path=image_path,
        image_size=image_size,
        camera_height=resolved_height,
        sensor_width_mm=sensor_width_mm,
        seed=seed,
    )

    if depth_result is not None:
        _attach_depth_component(solve, depth_result, ground)
    if reference_scale is not None:
        _attach_reference_scale(solve, reference_scale, adopted=scale_source == "reference_object")
    solve.debug_metadata["scale_source"] = scale_source
    return solve


def rescale_camera_height(solve: AtlasSolve, new_height: float) -> AtlasSolve:
    """Rebuild the extrinsics for a new camera height, preserving orientation.

    Mutates and returns ``solve``. Uses the stored world->cam rotation so only the
    metric scale (camera_position.y and the derived matrices) changes.
    """
    np = _require_numpy()
    extr = solve.camera.extrinsics
    rotation = np.asarray(extr.camera_rotation_matrix, dtype=np.float64)
    position = np.array([0.0, float(new_height), 0.0], dtype=np.float64)
    translation = -rotation.T @ position
    solve.camera.extrinsics = AtlasExtrinsics(
        camera_position=tuple(float(v) for v in position),
        camera_rotation_matrix=_matrix3_to_tuple(rotation),
        camera_world_matrix=_world_matrix_with_position(rotation, position),
        camera_view_matrix=_matrix4_with_rotation_translation(rotation.T, translation),
        coordinate_system=extr.coordinate_system,
        up_axis=extr.up_axis,
        projection_convention=extr.projection_convention,
    )
    return solve


def apply_reference_scale(
    solve: AtlasSolve,
    references: list[dict[str, Any]],
    *,
    adopt: bool = True,
) -> AtlasSolve:
    """Measure metric camera height from reference objects and rescale the solve.

    Reads orientation + intrinsics from ``solve`` (any solver's output), solves the
    single-view metric height, and — when confident — rescales the camera. Records
    the reference-scale detail regardless. Mutates and returns ``solve``.
    """
    intr = solve.camera.extrinsics
    K = solve.camera.intrinsics
    fx = K.fx_px or 0.0
    fy = K.fy_px or fx
    if fx <= 0:
        solve.debug_metadata["reference_scale"] = {"status": "no_focal_length"}
        return solve
    cx = K.cx_px if K.cx_px is not None else K.image_width / 2.0
    cy = K.cy_px if K.cy_px is not None else K.image_height / 2.0
    result = resolve_reference_scale(
        references, rotation=intr.camera_rotation_matrix, fx=fx, fy=fy, cx=cx, cy=cy
    )
    adopted = bool(
        adopt and result.get("camera_height")
        and result.get("confidence", 0.0) >= _REFERENCE_ADOPT_CONFIDENCE
    )
    if adopted:
        rescale_camera_height(solve, result["camera_height"])
        solve.source_method = f"{solve.source_method}+reference_scale"
        solve.debug_metadata["scale_source"] = "reference_object"
    _attach_reference_scale(solve, result, adopted=adopted)
    return solve


def _attach_reference_scale(solve: AtlasSolve, reference_scale: dict[str, Any], *, adopted: bool) -> None:
    """Record the reference-object scale solve on the solve (landmarks + metadata)."""
    ch = reference_scale.get("camera_height")
    conf = float(reference_scale.get("confidence", 0.0))
    solve.debug_metadata["reference_scale"] = {
        "camera_height_m": ch,
        "confidence": conf,
        "adopted": adopted,
        "references": reference_scale.get("references", []),
    }
    for ref in reference_scale.get("references", []):
        if ref.get("camera_height"):
            solve.landmarks.append({
                "name": ref.get("label") or ref.get("reference_id") or "scale_reference",
                "type": "metric_scale_reference",
                "reference_id": ref.get("reference_id"),
                "known_height": ref.get("real_height_m"),
                "implied_camera_height_m": ref.get("camera_height"),
                "consistency": ref.get("consistency"),
                "interpretation": "Metric scale anchor solved by single-view geometry.",
            })


def _resize_depth(depth: Any, width: int, height: int) -> Any:
    """Nearest-neighbour resize of a depth array to (height, width) without cv2."""
    np = _require_numpy()
    src_h, src_w = depth.shape
    ys = (np.linspace(0, src_h - 1, height)).round().astype(int)
    xs = (np.linspace(0, src_w - 1, width)).round().astype(int)
    return depth[np.ix_(ys, xs)]


def _attach_depth_component(solve: AtlasSolve, depth_result: Any, ground: Any) -> None:
    """Populate the depth LatentComponent and record the measured camera height."""
    from atlas_camera.core.schema import LatentComponent

    if depth_result is None:
        return
    ground = ground or {}
    candidate = ground.get("camera_height")
    conf = float(ground.get("confidence", 0.0))
    adopted = bool(candidate) and conf >= _HEIGHT_ADOPT_CONFIDENCE
    source = "depth_ground_plane" if adopted else "assumed_default"

    warnings: list[str] = []
    if not depth_result.is_metric:
        warnings.append("Relative depth model: measured height is up-to-scale, not metric.")
    if candidate is None:
        warnings.append("No reliable ground plane in depth; using fallback height.")
    elif not adopted:
        warnings.append(
            f"Depth suggested camera height {candidate:.2f} m but confidence "
            f"{conf:.2f} < {_HEIGHT_ADOPT_CONFIDENCE:.2f}; using fallback. Confirm manually."
        )

    solve.depth = LatentComponent(
        value=depth_result.summary(),
        confidence=conf,
        editable=False,
        exportable=True,
        metadata={
            "measured_camera_height_m": candidate,
            "camera_height_adopted": adopted,
            "camera_height_source": source,
            "ground_pixels": int(ground.get("ground_pixels", 0)),
            "adopt_confidence_threshold": _HEIGHT_ADOPT_CONFIDENCE,
        },
        warnings=warnings,
    )
    solve.debug_metadata["camera_height_measured"] = candidate if adopted else None
    solve.debug_metadata["camera_height_candidate"] = candidate
    solve.debug_metadata["camera_height_source"] = source


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
