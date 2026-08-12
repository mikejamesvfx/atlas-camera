"""Atlas-rendered crop sequences for video-to-video Dynamic Plates.

The v0.1 shipped mode fed the temporal generator a single still crop
(image-to-video), which leaves camera preservation to the model's goodwill.
This module implements the designed upgrade (spec §19's "best" input): render
the crop REGION along an Atlas camera move first, so the input video already
carries the geometrically correct camera motion and the generator's only job
is surface motion.

For a plane receiver and a still texture this is exact and cheap: the render
camera's view of the plane and the fixed crop camera's projection onto it are
both plane-induced homographies, so frame t is one 3x3 warp

    M_t = H_crop @ inv(H_render_t)

sampled bilinearly (`planar_projection.warp_by_homography`). No rasterizer,
no GPU. Frames where the plane exits the view get alpha=0 and are filled.

Needs numpy (``pip install -e .[vision]``).
"""
from __future__ import annotations

from typing import Any

from atlas_camera.core.dynamic_plate import DynamicPlate
from atlas_camera.core.camera_spec import CameraSpec
from atlas_camera.core.planar_projection import (
    homography_plane_to_image,
    plane_basis_from_primitive,
    warp_by_homography,
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Dynamic-plate rendering requires numpy. Install with:\n"
            "    pip install -e .[vision]") from exc
    return np


def dolly_view_matrices(camera: Any, *, offset: tuple[float, float, float],
                        frame_count: int) -> list[Any]:
    """View matrices for a linear world-space dolly of ``offset`` metres.

    Frame 0 is the source pose; the final frame is fully displaced. Rotation
    is untouched — a dolly, not a re-aim. For a world translation ``d`` the
    view translation becomes ``t' = t - R_view @ d``.
    """
    np = _require_numpy()
    view = np.asarray(camera.extrinsics.camera_view_matrix, dtype=np.float64)
    if view.shape != (4, 4):
        raise ValueError("camera_view_matrix must be 4x4 (the world-math rule)")
    rot = view[:3, :3]
    d = np.asarray(offset, dtype=np.float64)
    frames = []
    denom = max(1, int(frame_count) - 1)
    for index in range(int(frame_count)):
        s = index / denom
        out = view.copy()
        out[:3, 3] = view[:3, 3] - rot @ (d * s)
        frames.append(tuple(tuple(float(x) for x in row) for row in out))
    return frames


def crop_sequence_homographies(plate: DynamicPlate,
                               view_matrices: list[Any]) -> list[Any]:
    """Per-frame 3x3 mapping RENDER pixel -> CROP pixel.

    Both sides are plane-induced homographies through the receiver plane;
    the crop side uses the plate's fixed crop camera (the projection camera —
    exactly the registration the DCC will reproduce).
    """
    np = _require_numpy()
    if plate.crop_camera is None:
        raise ValueError("plate has no crop camera")
    if plate.receiver is None or plate.receiver.primitive is None:
        raise ValueError("plate has no plane receiver")
    basis = plane_basis_from_primitive(plate.receiver.primitive)
    spec = CameraSpec.from_intrinsics(plate.crop_camera.intrinsics)
    h_crop = homography_plane_to_image(
        plate.crop_camera.extrinsics.camera_view_matrix,
        spec.fx, spec.fy, spec.cx, spec.cy, basis)
    out = []
    for view in view_matrices:
        h_render = homography_plane_to_image(
            view, spec.fx, spec.fy, spec.cx, spec.cy, basis)
        out.append(h_crop @ np.linalg.inv(h_render))
    return out


def render_crop_sequence(plate: DynamicPlate, view_matrices: list[Any],
                         crop_image: Any) -> list[tuple[Any, Any]]:
    """Render the crop region from each view matrix.

    ``crop_image`` is the HxW[x3] crop raster (the packaged source/crop.png).
    Returns ``[(rgb, alpha), ...]`` per frame — alpha 0 where the plane is
    out of frame or behind the camera. The output raster matches the crop
    camera raster, so the sequence drops straight into the same projection
    registration as the still.
    """
    np = _require_numpy()
    image = np.asarray(crop_image)
    height, width = image.shape[0], image.shape[1]
    frames = []
    for m in crop_sequence_homographies(plate, view_matrices):
        rgb, alpha = warp_by_homography(image, m, width, height)
        frames.append((rgb, alpha))
    return frames
