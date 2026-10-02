"""The solved camera and scene placement as ComfyUI's Load3D socket payloads.

Core ComfyUI's 3D nodes (Load3D, Preview/Save 3D Advanced) carry the viewer
camera as ``LOAD3D_CAMERA`` and per-model placement as ``LOAD3D_MODEL_INFO``
(V135 ``comfy_api/latest/_io.py``)::

    CameraInfo: position, target, zoom, cameraType, quaternion?, fov? (VERTICAL
                degrees), aspect?, near?, far?
    Model3DTransform: position, quaternion, scale

Both are right-handed, Y-up, with the camera looking down -Z -- the same
convention as Atlas, so nothing is converted: the camera pose comes straight
from ``inv(camera_view_matrix)`` (the full 4x4, never the 3x3 -- transpose
ambiguity) and the geometry is already in world space, so each model's
placement is the identity.

Layering: ``core`` only -- numpy, no ComfyUI.
"""

from __future__ import annotations

import math
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Load3D camera export requires numpy.") from exc


def _vec(v) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def rotation_to_quaternion(R: Any) -> dict[str, float]:
    """Unit quaternion ``{x, y, z, w}`` for a 3x3 rotation (Shepperd's method)."""
    np = _require_numpy()
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return {"x": x / n, "y": y / n, "z": z / n, "w": w / n}


def load3d_camera_info(
    *,
    view_matrix: Any,
    fy: float,
    image_width: int,
    image_height: int,
    target: Any = None,
    near: float = 0.05,
    far: float = 1000.0,
) -> dict[str, Any]:
    """``LOAD3D_CAMERA`` for a solved pinhole camera.

    ``fov`` is the VERTICAL field of view in degrees, ``2*atan(H / 2fy)``.
    ``target`` (the orbit focus) defaults to a point 10 m ahead on the view
    axis; pass the scene pivot when one is known. A principal point off the
    image centre cannot be expressed in this payload -- a three.js perspective
    camera is centred -- so a viewer built from it is exact only for a centred
    principal point; ``principal_offset_px`` is reported for that reason.
    """
    np = _require_numpy()
    vm = np.asarray(view_matrix, dtype=np.float64)
    if vm.shape != (4, 4):
        raise ValueError(f"view_matrix must be 4x4, got {vm.shape}")
    c2w = np.linalg.inv(vm)
    pos = c2w[:3, 3]
    if target is None:
        target = pos + 10.0 * (-c2w[:3, 2])
    fov = math.degrees(2.0 * math.atan(float(image_height) / (2.0 * float(fy))))
    return {
        "position": _vec(pos),
        "target": _vec(np.asarray(target, dtype=np.float64)),
        "zoom": 1,
        "cameraType": "perspective",
        "quaternion": rotation_to_quaternion(c2w[:3, :3]),
        "fov": fov,
        "aspect": float(image_width) / float(image_height),
        "near": float(near),
        "far": float(far),
    }


def identity_model_info(count: int = 1) -> list[dict[str, Any]]:
    """``LOAD3D_MODEL_INFO``: Atlas geometry is world-space, so identity each."""
    return [{"position": {"x": 0.0, "y": 0.0, "z": 0.0},
             "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
             "scale": {"x": 1.0, "y": 1.0, "z": 1.0}} for _ in range(int(count))]
