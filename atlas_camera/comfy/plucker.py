"""Camera samples to ray maps, for models that condition on camera motion.

Rule 5 applies with full force here: an encoder that turns poses into rays
cannot avoid committing to a handedness and an up-axis, which makes it exactly
the kind of place an axis swap hides. So this module performs NO conversion.
Samples arrive in Atlas canonical space -- right-handed, Y-up, metres, camera
looking down -Z (app/director/export/chan.ts:15) -- and are encoded as they are.

A mirrored ray map is the expensive failure: it looks entirely plausible,
matches on every summary statistic, and gets debugged as a LoRA problem.
"""
from __future__ import annotations

import numpy as np


def _rotation_matrix(quaternion) -> np.ndarray:
    """Quaternion in xyzw order to a 3x3 rotation matrix."""

    x, y, z, w = (float(component) for component in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_space_directions(width: int, height: int, sample: dict) -> np.ndarray:
    """Unit directions through every pixel centre, in camera space."""

    filmback = sample["filmback"]
    focal = float(sample["focalLengthMm"])

    # Pixel centres in normalised device coordinates. +x right, +y up.
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0
    grid_x, grid_y = np.meshgrid(xs, ys)

    directions = np.stack(
        [
            grid_x * float(filmback["widthMm"]) / 2.0,
            grid_y * float(filmback["heightMm"]) / 2.0,
            np.full_like(grid_x, -focal),  # the camera looks down -Z
        ],
        axis=-1,
    )
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def ray_map(samples, width: int, height: int) -> np.ndarray:
    """Ray origins and directions per pixel, per frame.

    Returns `(frames, height, width, 6)`: channels 0-2 the ray origin, 3-5 a
    unit direction. The origin is the camera position, identical across a
    frame's pixels -- kept per-pixel because that is the shape the models want.
    """

    frames = []
    for sample in samples:
        camera_directions = _camera_space_directions(width, height, sample)
        rotation = _rotation_matrix(sample["rotation"])
        world_directions = camera_directions @ rotation.T
        origin = np.array(sample["position"], dtype=np.float64)
        origins = np.broadcast_to(origin, world_directions.shape)
        frames.append(np.concatenate([origins, world_directions], axis=-1))
    return np.stack(frames, axis=0)


def plucker_embedding(rays: np.ndarray) -> np.ndarray:
    """`(o, d)` to `(o x d, d)` -- Plucker coordinates proper.

    The literature's "Plucker embedding" is the moment beside the direction.
    Kept separate from `ray_map` because origins are far easier to eyeball when
    something is wrong, and the moment is one cross product away.
    """

    origins = rays[..., :3]
    directions = rays[..., 3:]
    moments = np.cross(origins, directions)
    return np.concatenate([moments, directions], axis=-1)
