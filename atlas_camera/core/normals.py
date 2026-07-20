"""Predicted-normal alignment + encoding for the viewport relight.

MoGe (`*-normal` variants) predicts per-pixel surface normals in its OWN camera
frame. To use them as WORLD-space relight normals on Atlas geometry (built in the
recovered GeoCalib camera frame, which carries the solved gravity/roll), we
recover the single rigid rotation between the two frames by orthogonal Procrustes
against the geometry's own (noisy) gradient normals, apply it, and encode the
result as a normal-map data URI the viewport samples per fragment.

Pure NumPy for the math (no torch); Pillow only for the PNG encode. The
predicted normals are cleaner than gradient-of-depth normals, so the relight
reads the true surface orientation at image resolution rather than the coarse
mesh's interpolated normal.
"""

from __future__ import annotations

import base64
import io
from typing import Any


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError("normals alignment needs numpy — pip install -e .[vision]") from exc
    return np


def _require_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError("normal-map resampling needs Pillow — pip install -e .[image]") from exc
    return Image


def world_normals_from_depth(depth, *, view_matrix, fx, fy, cx, cy):
    """Per-pixel WORLD-space normals from a forward-Z depth map, via back-
    projection + a neighbour cross product (same convention as
    ``relief_mesh.estimate_ground_scale``). Returns ``(normals HxWx3 float64,
    valid HxW bool)``. Oriented toward the camera. Noisy near silhouettes — used
    only as the Procrustes target, never as the final relight normal.
    """
    np = _require_numpy()
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    vm = np.asarray(view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    r_cw = c2w[:3, :3]
    cam = c2w[:3, 3]
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    with np.errstate(invalid="ignore"):
        x = (uu - cx) / fx * depth
        y = -(vv - cy) / fy * depth
        z = -depth
    pts = np.stack([x, y, z], axis=-1) @ r_cw.T + cam
    normals = np.zeros((height, width, 3), dtype=np.float64)
    du = pts[:, 2:, :] - pts[:, :-2, :]
    dv = pts[2:, :, :] - pts[:-2, :, :]
    cr = np.cross(du[1:-1], dv[:, 1:-1])
    norm = np.linalg.norm(cr, axis=-1, keepdims=True)
    normals[1:-1, 1:-1] = cr / np.maximum(norm, 1e-12)
    to_cam = cam - pts
    with np.errstate(invalid="ignore"):
        flip = np.sum(normals * to_cam, axis=-1) < 0
    normals[flip] *= -1.0
    valid = np.isfinite(depth) & (np.linalg.norm(normals, axis=-1) > 0.5)
    return normals, valid


def procrustes_rotation(a, b):
    """Orthogonal Procrustes: the proper rotation ``R`` (3x3, det +1) minimizing
    ``||R @ aᵢ − bᵢ||`` over row-vector sets ``a``/``b`` (both ``(N,3)``). So
    ``a @ R.T ≈ b``. The det-sign correction forbids a reflection (a real camera-
    frame change is a rotation)."""
    np = _require_numpy()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = b.T @ a                                  # 3x3 cross-covariance Σ bᵢ aᵢᵀ
    u, _s, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def align_predicted_normals_to_world(predicted, depth, *, view_matrix, fx, fy, cx, cy):
    """Rotate model-frame per-pixel normals (``predicted`` HxWx3) into the
    recovered WORLD frame by Procrustes against the geometry's gradient normals.
    Returns ``(world_normals HxWx3 float32, valid HxW bool)``. Falls back to the
    geometry normals themselves if too few pixels are usable.
    """
    np = _require_numpy()
    pred = np.asarray(predicted, dtype=np.float64)
    pn = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12)
    geom, valid = world_normals_from_depth(
        depth, view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy)
    valid = valid & np.isfinite(pn).all(axis=-1)
    if int(valid.sum()) < 16:
        return geom.astype(np.float32), valid
    b = geom[valid]
    # A model↔world frame change is a proper ROTATION, but the model's normals
    # may also be globally hemisphere-flipped (oriented into the surface) — that
    # composite is a REFLECTION, which orthogonal-Procrustes (det +1) can't undo.
    # So fit BOTH sign conventions and keep whichever aligns better; the flipped
    # case becomes a pure rotation once negated.
    best_world, best_agree = None, -2.0
    for sign in (1.0, -1.0):
        a = sign * pn[valid]
        rot = procrustes_rotation(a, b)
        world = (sign * pn) @ rot.T
        agree = float(np.mean(np.sum(world[valid] * b, axis=-1)))
        if agree > best_agree:
            best_agree, best_world = agree, world
    world = best_world / np.maximum(np.linalg.norm(best_world, axis=-1, keepdims=True), 1e-12)
    return world.astype(np.float32), valid


def encode_normal_map_b64(world_normals, valid=None):
    """Encode HxWx3 WORLD normals as a PNG data URI: ``(n+1)/2`` in RGB (the
    standard normal-map encoding; the viewport samples with ``NoColorSpace`` and
    decodes ``rgb*2-1``). Invalid pixels get world-up ``(0,1,0)`` so a miss reads
    as flat rather than garbage. Requires Pillow."""
    np = _require_numpy()
    from PIL import Image

    n = np.asarray(world_normals, dtype=np.float32)
    rgb = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    if valid is not None:
        rgb[~np.asarray(valid, dtype=bool)] = np.array([127, 255, 127], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Normal-field resampling (phase 2 move from comfy/node_helpers.py).
# --------------------------------------------------------------------------

def _resize_normal_field(normals, target_hw):
    """Resize an (H,W,3) unit-normal field to ``target_hw`` (h, w) and
    renormalize. Bilinear via PIL mode 'F' per channel; nearest-neighbour numpy
    fallback if PIL is unavailable. A no-op when already the right shape."""
    import numpy as np
    th, tw = int(target_hw[0]), int(target_hw[1])
    n = np.asarray(normals, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] < 3:
        raise ValueError(f"expected an (H,W,3) normal field, got {n.shape}")
    if n.shape[:2] == (th, tw):
        out = n[..., :3]
    else:
        try:
            PILImage = _require_pil()
            chans = []
            for c in range(3):
                im = PILImage.fromarray(np.ascontiguousarray(n[..., c]), mode="F")
                chans.append(np.asarray(im.resize((tw, th), PILImage.BILINEAR), dtype=np.float32))
            out = np.stack(chans, axis=-1)
        except Exception:
            ys = np.linspace(0, n.shape[0] - 1, th).astype(int)
            xs = np.linspace(0, n.shape[1] - 1, tw).astype(int)
            out = n[np.ix_(ys, xs)][..., :3].astype(np.float32)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norm, 1e-12)).astype(np.float32)
