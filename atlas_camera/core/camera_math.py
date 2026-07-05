"""Core camera math and physical-unit conversions.

Exporters call these helpers instead of carrying local conversion formulas.
Atlas core remains right-handed Y-up; these functions only handle camera units.

Also provides orbit/look-at construction for "patch" cameras (novel views placed
around the recovered camera). These are built through an unambiguous ``look_at``
so they match the system's proven convention: ``camera_view_matrix`` is a
row-major world->cam transform (``cam_point = view @ world_point``), the camera
looks along -Z in camera space, world is right-handed Y-up. Pure Python (no
numpy) to keep this module dependency-free like the rest of it.
"""

from __future__ import annotations

import math

from atlas_camera.core.schema import AtlasExtrinsics, Matrix4, Point3D

MM_PER_INCH = 25.4
FALLBACK_SENSOR_WIDTH_MM = 36.0
FALLBACK_SENSOR_HEIGHT_MM = 20.25
FOCAL_FALLBACK_CONFIDENCE_PENALTY = 0.15


def mm_to_inches(mm: float) -> float:
    return float(mm) / MM_PER_INCH


def inches_to_mm(inches: float) -> float:
    return float(inches) * MM_PER_INCH


def derive_sensor_height_mm(intrinsics) -> float:
    """Return sensor height in mm, falling back to aspect-ratio computation."""
    return float(
        intrinsics.sensor_height_mm
        or (intrinsics.sensor_width_mm * intrinsics.image_height / intrinsics.image_width)
    )


def focal_length_to_fov(focal_length_mm: float, sensor_size_mm: float) -> float:
    if focal_length_mm <= 0:
        raise ValueError("Focal length must be positive.")
    if sensor_size_mm <= 0:
        raise ValueError("Sensor size must be positive.")
    return math.degrees(2.0 * math.atan(sensor_size_mm / (2.0 * focal_length_mm)))


def fov_to_focal_length(fov_degrees: float, sensor_size_mm: float) -> float:
    if sensor_size_mm <= 0:
        raise ValueError("Sensor size must be positive.")
    if fov_degrees <= 0 or fov_degrees >= 180:
        raise ValueError("Field of view must be between 0 and 180 degrees.")
    return (sensor_size_mm / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)


def pixel_offset_to_normalized_film_offset(
    pixel_offset: float,
    *,
    aperture_mm: float,
    image_size_px: int,
) -> float:
    """Convert a principal-point pixel offset to a film-aperture fraction."""

    if aperture_mm <= 0:
        raise ValueError("Aperture must be positive.")
    if image_size_px <= 0:
        raise ValueError("Image size must be positive.")
    return float(pixel_offset) / float(image_size_px)


def normalized_film_offset_to_pixel_offset(
    normalized_offset: float,
    *,
    aperture_mm: float,
    image_size_px: int,
) -> float:
    if aperture_mm <= 0:
        raise ValueError("Aperture must be positive.")
    if image_size_px <= 0:
        raise ValueError("Image size must be positive.")
    return float(normalized_offset) * float(image_size_px)


def estimate_focal_with_fallback(
    *,
    fov_degrees: float,
    sensor_width_mm: float | None,
) -> tuple[float, float, bool, float]:
    """Return focal, sensor width, fallback flag, and confidence penalty."""

    used_fallback = sensor_width_mm is None
    width = sensor_width_mm if sensor_width_mm is not None else FALLBACK_SENSOR_WIDTH_MM
    penalty = FOCAL_FALLBACK_CONFIDENCE_PENALTY if used_fallback else 0.0
    return fov_to_focal_length(fov_degrees, width), width, used_fallback, penalty


# ---------------------------------------------------------------------------
# Orbit / look-at camera construction (patch cameras)
# ---------------------------------------------------------------------------

_Vec3 = tuple[float, float, float]


def _vsub(a: _Vec3, b: _Vec3) -> _Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a: _Vec3, b: _Vec3) -> _Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vdot(a: _Vec3, b: _Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vcross(a: _Vec3, b: _Vec3) -> _Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vlen(a: _Vec3) -> float:
    return math.sqrt(_vdot(a, a))


def _vnorm(a: _Vec3) -> _Vec3:
    length = _vlen(a) or 1.0
    return (a[0] / length, a[1] / length, a[2] / length)


def look_at_view_matrix(
    eye: _Vec3,
    target: _Vec3,
    up: _Vec3 = (0.0, 1.0, 0.0),
) -> tuple[Matrix4, Matrix4, tuple[tuple[float, float, float], ...]]:
    """Build (view, world, rotation3) for a camera at ``eye`` aimed at ``target``.

    Matches the Atlas convention exactly: ``view`` is row-major world->cam
    (``cam = view @ world``), camera looks along -Z in camera space, camera
    space is x-right / y-up / z-back. ``world`` is its inverse (cam->world,
    columns = camera axes, translation = eye). ``rotation3`` is the cam->world
    3x3 (columns = camera axes) matching how ``solver.py`` stores
    ``camera_rotation_matrix`` (its transpose is the view rotation).
    """
    z = _vnorm(_vsub(eye, target))            # camera +Z (points back, away from target)
    x = _vcross(up, z)
    if _vlen(x) < 1e-9:                        # up parallel to view axis — pick an alternate
        x = _vcross((0.0, 0.0, 1.0), z)
        if _vlen(x) < 1e-9:
            x = _vcross((1.0, 0.0, 0.0), z)
    x = _vnorm(x)                             # camera +X (right)
    y = _vcross(z, x)                         # camera +Y (up); unit since z,x orthonormal

    view: Matrix4 = (
        (x[0], x[1], x[2], -_vdot(x, eye)),
        (y[0], y[1], y[2], -_vdot(y, eye)),
        (z[0], z[1], z[2], -_vdot(z, eye)),
        (0.0, 0.0, 0.0, 1.0),
    )
    world: Matrix4 = (
        (x[0], y[0], z[0], eye[0]),
        (x[1], y[1], z[1], eye[1]),
        (x[2], y[2], z[2], eye[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    rotation3 = (
        (x[0], y[0], z[0]),
        (x[1], y[1], z[1]),
        (x[2], y[2], z[2]),
    )
    return view, world, rotation3


def ground_lookat_pivot(
    extrinsics: AtlasExtrinsics,
    *,
    fallback_distance: float = 10.0,
) -> Point3D:
    """Where the camera's forward ray meets the ground plane Y=0.

    The natural orbit pivot: the point the recovered camera is looking at. Forward
    is -Z in camera space, i.e. the negated Z column of the cam->world matrix.
    Falls back to a point ``fallback_distance`` ahead when the camera looks level
    or upward (ray never crosses the ground), mirroring the viewport's own
    ``groundPointInView``.
    """
    world = extrinsics.camera_world_matrix
    eye = tuple(float(v) for v in extrinsics.camera_position)
    forward = (-float(world[0][2]), -float(world[1][2]), -float(world[2][2]))
    if forward[1] < -1e-6:
        t = -eye[1] / forward[1]
        return (eye[0] + t * forward[0], 0.0, eye[2] + t * forward[2])
    return (
        eye[0] + fallback_distance * forward[0],
        eye[1] + fallback_distance * forward[1],
        eye[2] + fallback_distance * forward[2],
    )


def horizon_row_from_extrinsics(
    extrinsics: AtlasExtrinsics,
    *,
    fy: float,
    cy: float,
) -> float | None:
    """Image row where the world-horizontal plane's vanishing line falls.

    Exact for any zero-roll ``look_at`` camera (which is what ``orbit_camera``
    always builds): with world-up composed into the camera's right axis via a
    cross product, the right axis is perfectly horizontal (its world-Y
    component is zero), so the horizon is a single image row independent of
    column. Solving ``world_Y(ray_cam) = 0`` for that ray's v-coordinate gives
    ``v = cy - fy * R[1][2] / R[1][1]`` where ``R`` is ``camera_rotation_matrix``
    (columns = camera right/up/back axes in world coordinates).

    Returns ``None`` when the camera looks straight up/down (``R[1][1] == 0``,
    degenerate — no single horizon row exists). Lets patch cameras (which are
    *constructed*, not solved, so they carry no ``horizon_line`` of their own)
    get the same real sky/ground split that the primary solve's derive step
    gets from its solved horizon, instead of the generic ``height * 0.45``
    fallback in ``relief_mesh.py`` / ``depth_geometry.py``.
    """
    rotation = extrinsics.camera_rotation_matrix
    y_up = float(rotation[1][1])
    if abs(y_up) < 1e-6:
        return None
    y_back = float(rotation[1][2])
    return float(cy) - float(fy) * y_back / y_up


def orbit_camera(
    extrinsics: AtlasExtrinsics,
    pivot: Point3D,
    *,
    d_azimuth_deg: float,
    d_elevation_deg: float,
    distance_scale: float = 1.0,
    up: _Vec3 = (0.0, 1.0, 0.0),
) -> AtlasExtrinsics:
    """Orbit the camera around ``pivot`` and re-aim it there (a patch camera).

    The camera's offset from the pivot is converted to spherical coordinates
    (azimuth about world +Y, elevation above the horizontal), the deltas are
    added, the radius is scaled, and the camera is rebuilt via ``look_at`` so it
    faces the pivot with world-up. Positive ``d_azimuth_deg`` orbits toward the
    camera's right; positive ``d_elevation_deg`` raises it. A zero orbit at
    ``distance_scale=1`` reproduces a camera that already looks at ``pivot``
    with Y-up (patch cameras are always re-aimed at the subject — the primary
    solve is never mutated).
    """
    eye = tuple(float(v) for v in extrinsics.camera_position)
    piv = (float(pivot[0]), float(pivot[1]), float(pivot[2]))
    offset = _vsub(eye, piv)
    radius = _vlen(offset)
    if radius < 1e-9:
        offset = (0.0, 0.0, 1.0)
        radius = 1.0

    azimuth = math.atan2(offset[0], offset[2])
    elevation = math.asin(max(-1.0, min(1.0, offset[1] / radius)))
    azimuth += math.radians(d_azimuth_deg)
    elevation += math.radians(d_elevation_deg)
    limit = math.radians(85.0)
    elevation = max(-limit, min(limit, elevation))
    new_radius = radius * float(distance_scale)

    cos_el = math.cos(elevation)
    new_offset = (
        new_radius * cos_el * math.sin(azimuth),
        new_radius * math.sin(elevation),
        new_radius * cos_el * math.cos(azimuth),
    )
    new_eye = _vadd(piv, new_offset)
    view, world, rotation3 = look_at_view_matrix(new_eye, piv, up)

    return AtlasExtrinsics(
        camera_position=new_eye,
        camera_rotation_matrix=rotation3,  # type: ignore[arg-type]
        camera_world_matrix=world,
        camera_view_matrix=view,
        coordinate_system="right_handed",
        up_axis="Y",
        projection_convention="Atlas pinhole camera (orbit-constructed patch view), image origin top-left.",
    )
