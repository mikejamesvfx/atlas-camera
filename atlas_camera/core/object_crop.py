"""Virtual object camera: a square crop that a centred-pinhole model can read.

Pixel-aligned image-to-3D models (Pixal3D) condition on a SQUARE crop and
assume a centred pinhole: principal point at the crop centre, one horizontal
FOV. A plain rectangle cut out of a solved photo breaks that assumption for any
object away from the optical axis -- the crop's principal point is wherever
the solve's ``cx, cy`` lands relative to the rectangle, not its centre.
Shearing the geometry afterwards would keep pixel alignment but bend the
object; feeding a guessed FOV (MoGe on the crop) contradicts the solve the
mesh must later be mapped back through.

So the crop is not cut, it is RE-RENDERED: a virtual camera at the solved
camera's centre is rotated to look at the object, and the photo is resampled
through the pure-rotation homography ``H = B R_v^T A`` (exact for any scene
depth, because both cameras share a centre). In that virtual camera the
centred-pinhole assumption is true by construction, the FOV comes from the
solve, and the inverse map back to the source camera is a rotation.

Conventions (shared with :mod:`atlas_camera.core.move_budget`'s rasterizer):

* camera frame x right, y up, -Z forward; ``u = cx + fx*x/(-z)``,
  ``v = cy - fy*y/(-z)``; pixel INDEX i sits at coordinate i.
* the virtual crop is ``size`` pixels square with its principal point at
  ``(size-1)/2`` in index coordinates. Pixal3D projects to
  ``f*x/(-z) + R/2`` and samples with ``align_corners=False``, which puts pixel
  index j's centre at ``j + 0.5`` -- the same point. Pinned by a test.
* ``R_v`` maps SOURCE-camera vectors into the VIRTUAL camera: ``p_v = R_v p_s``.
* roll: ``gravity`` keeps world-up vertical in the crop (the model's prior is
  upright objects); ``source_camera`` keeps the photo's own up.

Layering: ``core`` only -- numpy, no torch, no ComfyUI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: Pad applied to the object's half-extent, matching ImageCropToMask(1.1),
#: the framing Pixal3D's conditioning was published with.
DEFAULT_PAD = 1.1
DEFAULT_SIZE = 1024
ROLL_MODES = ("gravity", "source_camera")

#: Rays sampled from the mask when sizing the crop; a full 8K mask has tens of
#: millions of pixels and the extent is a max, so a stride subsample is exact
#: up to one stride of pixels.
_MAX_MASK_RAYS = 200_000

#: Below this |cos| between the object direction and world-up the gravity roll
#: is undefined (looking straight down/up), so the source camera's up is used.
_GRAVITY_DEGENERATE_COS = 0.995


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Object crop requires numpy. Install with: pip install -e .[vision]"
        ) from exc


@dataclass(frozen=True)
class ObjectCropCamera:
    """A virtual camera sharing the solve's centre, aimed at one object.

    ``rotation`` is ``R_v`` (3x3, source camera -> virtual camera).
    ``homography`` maps virtual-crop pixel indices to source pixel coordinates;
    ``inverse_homography`` the reverse. ``fov_deg`` is the HORIZONTAL (and,
    the crop being square, vertical) field of view -- the value Pixal3D's
    ``camera_angle_x`` expects.
    """

    rotation: Any
    focal_px: float
    size: int
    fov_deg: float
    half_extent: float
    pad: float
    roll: str
    roll_fallback: bool
    homography: Any
    inverse_homography: Any
    source_width: int
    source_height: int

    @property
    def principal_px(self) -> float:
        return (self.size - 1) / 2.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready handle (the ``ATLAS_OBJECT_CROP`` socket payload)."""
        np = _require_numpy()
        return {
            "kind": "atlas_object_crop",
            "rotation": np.asarray(self.rotation, dtype=np.float64).tolist(),
            "focal_px": float(self.focal_px),
            "size": int(self.size),
            "fov_deg": float(self.fov_deg),
            "half_extent": float(self.half_extent),
            "pad": float(self.pad),
            "roll": self.roll,
            "roll_fallback": bool(self.roll_fallback),
            "homography": np.asarray(self.homography, dtype=np.float64).tolist(),
            "inverse_homography": np.asarray(self.inverse_homography, dtype=np.float64).tolist(),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectCropCamera":
        np = _require_numpy()
        if not isinstance(d, dict) or d.get("kind") != "atlas_object_crop":
            raise ValueError("not an ATLAS_OBJECT_CROP handle")
        return cls(
            rotation=np.asarray(d["rotation"], dtype=np.float64),
            focal_px=float(d["focal_px"]), size=int(d["size"]),
            fov_deg=float(d["fov_deg"]), half_extent=float(d["half_extent"]),
            pad=float(d["pad"]), roll=str(d["roll"]),
            roll_fallback=bool(d.get("roll_fallback", False)),
            homography=np.asarray(d["homography"], dtype=np.float64),
            inverse_homography=np.asarray(d["inverse_homography"], dtype=np.float64),
            source_width=int(d["source_width"]), source_height=int(d["source_height"]),
        )


def _source_pixel_matrix(np: Any, fx: float, fy: float, cx: float, cy: float) -> Any:
    """Camera-frame ray -> homogeneous source pixel (the -Z forward pinhole)."""
    return np.array([[fx, 0.0, -cx],
                     [0.0, -fy, -cy],
                     [0.0, 0.0, -1.0]], dtype=np.float64)


def _virtual_ray_matrix(np: Any, f: float, c: float) -> Any:
    """Homogeneous virtual pixel index -> virtual camera-frame ray (z = -1)."""
    return np.array([[1.0 / f, 0.0, -c / f],
                     [0.0, -1.0 / f, c / f],
                     [0.0, 0.0, -1.0]], dtype=np.float64)


def _mask_rays(np: Any, mask: Any, fx: float, fy: float, cx: float, cy: float) -> Any:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape {m.shape}")
    ys, xs = np.nonzero(m > 0.5)
    if xs.size == 0:
        raise ValueError("object mask is empty: nothing to crop")
    if xs.size > _MAX_MASK_RAYS:
        step = int(math.ceil(xs.size / _MAX_MASK_RAYS))
        xs, ys = xs[::step], ys[::step]
    # Pixel footprint corners, not just centres, so a one-pixel object still
    # has an extent and the pad is measured from the true silhouette edge.
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    u = (xs[:, None] + corners[None, :, 0]).reshape(-1).astype(np.float64)
    v = (ys[:, None] + corners[None, :, 1]).reshape(-1).astype(np.float64)
    rays = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _basis_looking_along(np: Any, forward: Any, up_hint: Any) -> tuple[Any, bool]:
    """Rows (right, up, back) of a camera looking along ``forward``."""
    back = -forward / np.linalg.norm(forward)
    fallback = abs(float(np.dot(up_hint, back))) > _GRAVITY_DEGENERATE_COS
    up = np.array([0.0, 1.0, 0.0]) if fallback else up_hint
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-9:  # forward exactly along source up
        right = np.array([1.0, 0.0, 0.0])
        fallback = True
    right /= np.linalg.norm(right)
    up_v = np.cross(back, right)
    return np.stack([right, up_v, back], axis=0), fallback


def _plane_coords(np: Any, rays_v: Any) -> tuple[Any, Any]:
    w = -rays_v[:, 2]
    if bool((w <= 1e-9).any()):
        raise ValueError("object spans more than a hemisphere of the camera: cannot crop")
    return rays_v[:, 0] / w, rays_v[:, 1] / w


def virtual_object_camera(
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    mask: Any,
    pad: float = DEFAULT_PAD,
    size: int = DEFAULT_SIZE,
    roll: str = "gravity",
    refine_iterations: int = 2,
) -> ObjectCropCamera:
    """Aim a square virtual camera at the masked object.

    ``view_matrix`` is the solve's full 4x4 world->camera matrix; only its
    rotation block is used, and only to find world-up in camera coordinates
    for the gravity roll. The crop is centred on the midpoint of the object's
    extent in the virtual image plane (refined ``refine_iterations`` times,
    because re-aiming moves the extent), and sized so the larger half-extent
    times ``pad`` exactly fills the half-frame.
    """
    np = _require_numpy()
    if roll not in ROLL_MODES:
        raise ValueError(f"roll must be one of {ROLL_MODES}, got {roll!r}")
    if size < 2:
        raise ValueError("size must be at least 2")
    if pad < 1.0:
        raise ValueError("pad must be >= 1.0 (the object must fit the crop)")
    vm = np.asarray(view_matrix, dtype=np.float64)
    if vm.shape != (4, 4):
        raise ValueError(f"view_matrix must be 4x4, got {vm.shape}")

    rays = _mask_rays(np, mask, fx, fy, cx, cy)
    if roll == "gravity":
        up_hint = vm[:3, :3] @ np.array([0.0, 1.0, 0.0])
        up_hint = up_hint / np.linalg.norm(up_hint)
    else:
        up_hint = np.array([0.0, 1.0, 0.0])

    forward = rays.mean(axis=0)
    R_v, fallback = _basis_looking_along(np, forward, up_hint)
    for _ in range(max(0, int(refine_iterations))):
        px, py = _plane_coords(np, rays @ R_v.T)
        mid = np.array([(px.min() + px.max()) / 2.0, (py.min() + py.max()) / 2.0, -1.0])
        forward = R_v.T @ mid
        R_v, fallback = _basis_looking_along(np, forward, up_hint)

    px, py = _plane_coords(np, rays @ R_v.T)
    half = float(max(np.abs(px).max(), np.abs(py).max()))
    half = max(half, 1e-6) * float(pad)
    f_v = (size / 2.0) / half
    c_v = (size - 1) / 2.0
    fov_deg = math.degrees(2.0 * math.atan(half))

    H = _source_pixel_matrix(np, fx, fy, cx, cy) @ R_v.T @ _virtual_ray_matrix(np, f_v, c_v)
    H = H / H[2, 2]
    H_inv = np.linalg.inv(H)
    H_inv = H_inv / H_inv[2, 2]
    return ObjectCropCamera(
        rotation=R_v, focal_px=float(f_v), size=int(size), fov_deg=float(fov_deg),
        half_extent=float(half), pad=float(pad), roll=roll, roll_fallback=bool(fallback),
        homography=H, inverse_homography=H_inv,
        source_width=int(image_width), source_height=int(image_height),
    )


def apply_homography(H: Any, x: Any, y: Any) -> tuple[Any, Any, Any]:
    """Map pixel coordinates through ``H``; returns ``(x', y', w)``.

    ``w`` is the homogeneous divisor. For the crop homographies it is
    proportional to the ray's forward component, so ``w <= 0`` marks a point
    behind the target camera -- callers must treat it as unmapped.
    """
    np = _require_numpy()
    H = np.asarray(H, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xp = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    yp = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    safe = np.where(np.abs(w) > 1e-12, w, 1e-12)
    return xp / safe, yp / safe, w


def crop_sample_grid(crop: ObjectCropCamera) -> tuple[Any, Any, Any]:
    """Source pixel coordinates for every crop pixel index, ``(size, size)``.

    Returns ``(src_x, src_y, valid)``; ``valid`` is False where the crop ray
    falls behind the source camera or outside the source frame.
    """
    np = _require_numpy()
    j, i = np.meshgrid(np.arange(crop.size, dtype=np.float64),
                       np.arange(crop.size, dtype=np.float64))
    sx, sy, w = apply_homography(crop.homography, j, i)
    valid = (w > 0) & (sx >= -0.5) & (sx <= crop.source_width - 0.5) & \
        (sy >= -0.5) & (sy <= crop.source_height - 0.5)
    return sx, sy, valid


def warp_to_crop(crop: ObjectCropCamera, image: Any, *, fill: Any = 0.0) -> Any:
    """Bilinear-resample a source image (HxW or HxWxC) into the crop.

    Reference implementation for tests and torch-free callers; the ComfyUI
    node uses ``grid_sample`` with the same grid.
    """
    np = _require_numpy()
    img = np.asarray(image, dtype=np.float64)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    h, w = img.shape[:2]
    sx, sy, valid = crop_sample_grid(crop)
    x0 = np.clip(np.floor(sx), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(sy), 0, h - 1).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    ax = np.clip(sx - x0, 0.0, 1.0)[..., None]
    ay = np.clip(sy - y0, 0.0, 1.0)[..., None]
    out = (img[y0, x0] * (1 - ax) * (1 - ay) + img[y0, x1] * ax * (1 - ay)
           + img[y1, x0] * (1 - ax) * ay + img[y1, x1] * ax * ay)
    out = np.where(valid[..., None], out, np.asarray(fill, dtype=np.float64))
    return out[..., 0] if squeeze else out


def pixal_project(points_cam: Any, *, fov_deg: float, resolution: int) -> tuple[Any, Any]:
    """Numpy copy of Pixal3D's ``_project_points_to_image`` (V135
    ``comfy/ldm/trellis2/model.py``), returned in PIXEL-INDEX coordinates.

    Pixal3D's continuous ``x_pix = f*x/(-z) + R/2`` with
    ``align_corners=False`` sampling means index ``j`` covers ``[j, j+1)``;
    subtracting 0.5 converts to the index convention every other function in
    this module uses. ``points_cam`` are in a centred camera frame (x right,
    y up, -Z forward).
    """
    np = _require_numpy()
    p = np.asarray(points_cam, dtype=np.float64)
    f = resolution / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    w = -p[..., 2]
    x_pix = f * p[..., 0] / w + resolution / 2.0
    y_pix = -f * p[..., 1] / w + resolution / 2.0
    return x_pix - 0.5, y_pix - 0.5
