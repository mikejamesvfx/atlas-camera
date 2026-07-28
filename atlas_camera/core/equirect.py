"""Equirectangular (360°) panorama -> perspective views.

Atlas's whole stack is pinhole: ``AtlasIntrinsics`` is fx/fy/cx/cy and every
solver, relief mesh and projection path assumes a perspective camera. So an
equirect plate is NOT modelled as an equirect camera — it is **split into N
perspective views**, each of which is already a valid Atlas camera, and those
views feed the existing multi-camera machinery (``AtlasAddPatchView``) whose
angles are, unusually, EXACTLY KNOWN rather than estimated.

Why bother: a single perspective plate has no data for what the camera never
saw, which is the entire reason ``AtlasOcclusionGraph`` / ``AtlasMoveBudget`` /
``AtlasPathGuidedHoleRepair`` exist. A 360° capture supplies that coverage as
real measured geometry instead of inventing it.

Conventions (matching ``camera_math`` and the rest of core):
  * world is right-handed Y-up; the camera looks along **-Z** in camera space
  * image origin top-left, x right, y **down**
  * equirect longitude 0 is the image's horizontal CENTRE and increases to the
    right; latitude +90° (up) is the TOP row

Pure NumPy — no torch, no ComfyUI. Host-agnostic per the layering rule.
"""

from __future__ import annotations

import math
from typing import Any

from atlas_camera.core.camera_math import look_at_view_matrix


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a core dev dep
        raise RuntimeError(
            "Equirect splitting requires numpy. Install with: pip install -e .[dev]"
        ) from exc
    return np


def perspective_view_angles(
    n_views: int = 12,
    *,
    pitch_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """The ring of (yaw, pitch) angles in degrees, evenly spaced over 360°.

    12 views is ComfyUI core's `MoGePanoramaInference` default and pairs with a
    90° FOV for ~3x horizontal overlap — generous, but overlap is what lets the
    per-view depths agree where they meet. ``yaw_offset_deg`` rotates the whole
    ring, which matters when a seam would otherwise land on the subject.

    Yaw increases to the RIGHT (matching longitude), so view 0 looks along -Z:
    the same direction a default Atlas camera faces, which keeps the primary
    view of a split panorama consistent with an ordinary single-plate solve.
    """
    if int(n_views) < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    step = 360.0 / float(int(n_views))
    return [(yaw_offset_deg + i * step, float(pitch_deg)) for i in range(int(n_views))]


def intrinsics_for_view(size: int, fov_deg: float) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` for a square crop of ``size`` px at ``fov_deg``.

    Square by construction: a non-square crop would need separate h/v FOV and
    buys nothing, since the ring already controls coverage.
    """
    if size < 2:
        raise ValueError(f"size must be >= 2, got {size}")
    if not (0.0 < float(fov_deg) < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    f = (size / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    c = size / 2.0
    return (f, f, c, c)


def view_matrix_for_angles(yaw_deg: float, pitch_deg: float):
    """World->cam matrix for a camera at the ORIGIN looking along (yaw, pitch).

    Built through ``look_at_view_matrix`` rather than composing rotations by
    hand, so the sign and handedness conventions are the proven ones and cannot
    drift from the rest of the system.
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    # Direction the camera looks. yaw=0 -> -Z (Atlas's default facing).
    target = (
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        -math.cos(pitch) * math.cos(yaw),
    )
    view, world, rot3 = look_at_view_matrix((0.0, 0.0, 0.0), target)
    return view, world, rot3


def equirect_to_perspective(
    equirect,
    *,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    size: int = 512,
):
    """Sample one perspective view out of an equirect image.

    ``equirect`` is (H, W, C) with W == 2*H for a full 360x180 panorama (not
    enforced — a partial pano still samples correctly, it simply has no data
    outside its coverage). Returns ``(crop, (fx, fy, cx, cy))`` where ``crop``
    is (size, size, C) in the input's dtype range.

    Bilinear sampling, wrapping in longitude and clamping in latitude: the
    wrap is what keeps the 360° seam invisible, and clamping the poles avoids
    sampling past the top/bottom row where an equirect has no data.
    """
    np = _require_numpy()
    img = np.asarray(equirect)
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3:
        raise ValueError(f"equirect must be (H,W) or (H,W,C), got shape {img.shape}")
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    if src_h < 2 or src_w < 2:
        raise ValueError(f"equirect too small: {img.shape}")

    fx, fy, cx, cy = intrinsics_for_view(int(size), float(fov_deg))

    # Camera-space rays through pixel centres. y is negated because image y runs
    # DOWN while camera +Y is up; z is -1 because the camera looks along -Z.
    j, i = np.meshgrid(np.arange(size, dtype=np.float64),
                       np.arange(size, dtype=np.float64), indexing="xy")
    dx = (j + 0.5 - cx) / fx
    dy = -(i + 0.5 - cy) / fy
    dz = -np.ones_like(dx)

    _view, _world, rot3 = view_matrix_for_angles(yaw_deg, pitch_deg)
    # rot3 is cam->world with COLUMNS = camera axes, so world = rot3 @ cam.
    r = np.asarray(rot3, dtype=np.float64)
    wx = r[0][0] * dx + r[0][1] * dy + r[0][2] * dz
    wy = r[1][0] * dx + r[1][1] * dy + r[1][2] * dz
    wz = r[2][0] * dx + r[2][1] * dy + r[2][2] * dz

    norm = np.sqrt(wx * wx + wy * wy + wz * wz)
    wx, wy, wz = wx / norm, wy / norm, wz / norm

    # Direction -> spherical. lon 0 is straight ahead (-Z) and the image centre.
    lon = np.arctan2(wx, -wz)
    lat = np.arcsin(np.clip(wy, -1.0, 1.0))

    u = (lon / (2.0 * math.pi) + 0.5) * src_w - 0.5
    v = (0.5 - lat / math.pi) * src_h - 0.5

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu = (u - u0)[..., None]
    fv = (v - v0)[..., None]

    # Longitude WRAPS (the seam must be continuous); latitude CLAMPS (there is
    # no data above the top row or below the bottom one).
    u0m, u1m = u0 % src_w, (u0 + 1) % src_w
    v0m = np.clip(v0, 0, src_h - 1)
    v1m = np.clip(v0 + 1, 0, src_h - 1)

    src = img.astype(np.float64)
    top = src[v0m, u0m] * (1.0 - fu) + src[v0m, u1m] * fu
    bot = src[v1m, u0m] * (1.0 - fu) + src[v1m, u1m] * fu
    crop = top * (1.0 - fv) + bot * fv

    if img.shape[2] == 1 and np.asarray(equirect).ndim == 2:
        crop = crop[..., 0]
    return crop.astype(np.asarray(equirect).dtype, copy=False), (fx, fy, cx, cy)


def split_equirect(
    equirect,
    *,
    n_views: int = 12,
    fov_deg: float = 90.0,
    size: int = 512,
    pitch_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
):
    """Split an equirect into ``n_views`` perspective crops around the ring.

    Returns ``(crops, angles, intrinsics)`` — a list of (size,size,C) arrays,
    the parallel list of (yaw, pitch) in degrees, and the shared
    ``(fx, fy, cx, cy)``. The angles are what make these views worth more to
    Atlas than AI novel views: they are measured, not guessed, so they go into
    ``AtlasAddPatchView`` through its exact-view path.
    """
    angles = perspective_view_angles(n_views, pitch_deg=pitch_deg,
                                     yaw_offset_deg=yaw_offset_deg)
    crops = []
    intr: tuple[float, float, float, float] | None = None
    for yaw, pitch in angles:
        crop, intr = equirect_to_perspective(
            equirect, yaw_deg=yaw, pitch_deg=pitch, fov_deg=fov_deg, size=size)
        crops.append(crop)
    return crops, angles, intr


def direction_to_equirect_uv(x: float, y: float, z: float) -> tuple[float, float]:
    """Unit world direction -> normalised equirect (u, v) in [0,1].

    The inverse of the sampling above; exists so tests can round-trip a known
    direction rather than trusting the forward path alone.
    """
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    lon = math.atan2(x, -z)
    lat = math.asin(max(-1.0, min(1.0, y)))
    return (lon / (2.0 * math.pi) + 0.5, 0.5 - lat / math.pi)


def view_camera_at(eye, yaw_deg: float, pitch_deg: float = 0.0):
    """``AtlasExtrinsics`` for a panorama view: the SAME eye, a different rotation.

    This is the whole difference between panorama views and
    ``AtlasAddPatchView``'s patches. That node builds patch cameras with
    ``camera_math.orbit_camera``, which MOVES the camera — it rotates the
    camera's offset from a ground pivot and re-aims, displacing the eye by
    roughly ``2*r*sin(delta/2)``, metres for a typical pivot distance. Panorama
    views share ONE optical centre by construction and differ only in where they
    point, so orbiting them registers their geometry in the wrong place.

    Built via ``look_at_view_matrix`` from ``eye`` toward a target one unit away
    in the (yaw, pitch) direction, so the handedness and -Z facing come from the
    same proven helper as everything else rather than being re-derived here.
    """
    from atlas_camera.core.schema import AtlasExtrinsics

    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    ex, ey, ez = (float(eye[0]), float(eye[1]), float(eye[2]))
    target = (
        ex + math.cos(pitch) * math.sin(yaw),
        ey + math.sin(pitch),
        ez - math.cos(pitch) * math.cos(yaw),
    )
    view, world, rot3 = look_at_view_matrix((ex, ey, ez), target)
    return AtlasExtrinsics(
        camera_position=(ex, ey, ez),
        camera_rotation_matrix=rot3,
        camera_world_matrix=world,
        camera_view_matrix=view,
    )
