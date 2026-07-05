"""Camera intrinsics helpers."""

from __future__ import annotations

from atlas_camera.core.schema import AtlasIntrinsics, AtlasShotCam


def derive_sensor_height_mm(
    sensor_width_mm: float,
    image_width_px: int,
    image_height_px: int,
) -> float:
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("Image dimensions must be positive.")
    return sensor_width_mm * (image_height_px / image_width_px)


def focal_length_mm_to_pixels(
    focal_length_mm: float,
    sensor_size_mm: float,
    image_size_px: int,
) -> float:
    if focal_length_mm <= 0:
        raise ValueError("Focal length must be positive.")
    if sensor_size_mm <= 0:
        raise ValueError("Sensor size must be positive.")
    if image_size_px <= 0:
        raise ValueError("Image size must be positive.")
    return focal_length_mm * image_size_px / sensor_size_mm


def build_intrinsics(
    *,
    image_width: int,
    image_height: int,
    focal_length_mm: float | None = None,
    sensor_width_mm: float = 36.0,
    sensor_height_mm: float | None = None,
    principal_point_px: tuple[float, float] | None = None,
    fx_px: float | None = None,
    fy_px: float | None = None,
) -> AtlasIntrinsics:
    """Build normalized pinhole intrinsics from lens or pixel hints."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if sensor_height_mm is None:
        sensor_height_mm = derive_sensor_height_mm(sensor_width_mm, image_width, image_height)

    if principal_point_px is None:
        principal_point_px = (image_width / 2.0, image_height / 2.0)

    if focal_length_mm is not None:
        fx_px = fx_px or focal_length_mm_to_pixels(
            focal_length_mm,
            sensor_width_mm,
            image_width,
        )
        fy_px = fy_px or focal_length_mm_to_pixels(
            focal_length_mm,
            sensor_height_mm,
            image_height,
        )

    return AtlasIntrinsics(
        image_width=image_width,
        image_height=image_height,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        principal_point_px=principal_point_px,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=principal_point_px[0],
        cy_px=principal_point_px[1],
    )


def _fit_aspect_to_long_edge(
    width_mm: float,
    height_mm: float,
    long_edge_px: int,
    multiple: int = 8,
) -> tuple[int, int]:
    """Pixel (width, height) preserving the sensor's mm aspect ratio, with the
    longer side set to ``long_edge_px`` (rounded to a multiple of 8, matching
    the ComfyUI node's own ``_fit_long_edge`` convention for GPU-friendly
    dimensions)."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Sensor dimensions must be positive.")
    if long_edge_px <= 0:
        raise ValueError("Long edge must be positive.")
    scale = long_edge_px / float(max(width_mm, height_mm))

    def _round(v: float) -> int:
        return max(multiple, int(round(v / multiple)) * multiple)

    return _round(width_mm * scale), _round(height_mm * scale)


def intrinsics_from_shot_cam(shot_cam: AtlasShotCam) -> AtlasIntrinsics:
    """Canonical pinhole intrinsics for a project-level shot format —
    independent of any particular photographed image. Used to conform the
    render/export camera to a shot's own sensor/lens/resolution regardless
    of what any individual solved photo's aspect ratio happened to be."""
    width_px, height_px = _fit_aspect_to_long_edge(
        shot_cam.sensor_width_mm, shot_cam.sensor_height_mm, shot_cam.resolution_long_edge_px
    )
    return build_intrinsics(
        image_width=width_px,
        image_height=height_px,
        focal_length_mm=shot_cam.focal_length_mm,
        sensor_width_mm=shot_cam.sensor_width_mm,
        sensor_height_mm=shot_cam.sensor_height_mm,
    )
