"""Keyframed camera path sampling for testing projection under a moving camera.

Pure Python (no numpy), mirroring ``camera_math.py``'s dependency-free style.
This is the server-side source of truth for path interpolation, consumed by
``AtlasBlockoutViewport``'s baked frame decode and by
``usd_exporter.export_camera_animation``. The browser mirrors this math for
live 60fps scrubbing during path authoring (via Three.js's built-in
``CatmullRomCurve3`` + its own easing) — a deliberate duplication, the same
kind already accepted between ``depth_geometry.py`` and ``proxy_geometry.py``
in this codebase, because the JS copy must run every frame without a Python
round-trip. If either side's curve/easing changes, check the other.
"""

from __future__ import annotations

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import AtlasCameraPath, AtlasExtrinsics, Point3D

_Vec3 = tuple[float, float, float]


def _vadd(a: _Vec3, b: _Vec3) -> _Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vscale(a: _Vec3, s: float) -> _Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _catmull_rom(p0: _Vec3, p1: _Vec3, p2: _Vec3, p3: _Vec3, t: float) -> _Vec3:
    """Standard (non-uniform-agnostic, centripetal-free) Catmull-Rom, t in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    out = [0.0, 0.0, 0.0]
    for i in range(3):
        out[i] = 0.5 * (
            2.0 * p1[i]
            + (-p0[i] + p2[i]) * t
            + (2.0 * p0[i] - 5.0 * p1[i] + 4.0 * p2[i] - p3[i]) * t2
            + (-p0[i] + 3.0 * p1[i] - 3.0 * p2[i] + p3[i]) * t3
        )
    return (out[0], out[1], out[2])


def _apply_easing(t: float, easing: str) -> float:
    if easing == "ease_in":
        return t * t
    if easing == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if easing == "ease_in_out":
        return 3.0 * t * t - 2.0 * t * t * t
    return t  # "linear" and unknown values fall back to linear


def sample_camera_path(path: AtlasCameraPath) -> list[AtlasExtrinsics]:
    """Sample ``path`` into one ``AtlasExtrinsics`` per frame in ``0..frame_count-1``.

    - 0 keyframes: returns an empty list.
    - 1 keyframe: that pose repeated for every frame (a static "path").
    - >=2 keyframes: Catmull-Rom through the keyframes' ``position``/``target``
      (endpoints duplicated as phantom control points), with each segment's
      local ``t`` eased by its *starting* keyframe's ``easing`` before the
      spline is evaluated — so sampling exactly at a keyframe's ``frame_index``
      always reproduces that keyframe's ``position``/``target`` exactly
      (t=0 or t=1 passes through regardless of easing).
    """
    frame_count = max(0, int(path.frame_count))
    keyframes = path.keyframes
    if frame_count == 0 or not keyframes:
        return []

    if len(keyframes) == 1:
        kf = keyframes[0]
        view, world, rotation3 = look_at_view_matrix(kf.position, kf.target, kf.up)
        extr = AtlasExtrinsics(
            camera_position=kf.position,
            camera_rotation_matrix=rotation3,  # type: ignore[arg-type]
            camera_world_matrix=world,
            camera_view_matrix=view,
            coordinate_system="right_handed",
            up_axis="Y",
            projection_convention="Atlas pinhole camera (camera-path-constructed), image origin top-left.",
        )
        return [extr for _ in range(frame_count)]

    positions = [kf.position for kf in keyframes]
    targets = [kf.target for kf in keyframes]
    ups = [kf.up for kf in keyframes]
    frame_indices = [kf.frame_index for kf in keyframes]
    easings = [kf.easing for kf in keyframes]

    # Phantom endpoints so the first/last real segment has 4 control points.
    positions = [positions[0]] + positions + [positions[-1]]
    targets = [targets[0]] + targets + [targets[-1]]

    extrinsics: list[AtlasExtrinsics] = []
    for frame in range(frame_count):
        # Clamp outside the keyframed range to the nearest end keyframe.
        if frame <= frame_indices[0]:
            seg = 0
            local_t = 0.0
        elif frame >= frame_indices[-1]:
            seg = len(frame_indices) - 2
            local_t = 1.0
        else:
            seg = 0
            for i in range(len(frame_indices) - 1):
                if frame_indices[i] <= frame <= frame_indices[i + 1]:
                    seg = i
                    break
            span = frame_indices[seg + 1] - frame_indices[seg]
            local_t = (frame - frame_indices[seg]) / span if span else 0.0

        eased_t = _apply_easing(local_t, easings[seg])

        # positions/targets are offset by +1 due to the phantom endpoints above.
        pos = _catmull_rom(
            positions[seg], positions[seg + 1], positions[seg + 2], positions[seg + 3], eased_t
        )
        tgt = _catmull_rom(
            targets[seg], targets[seg + 1], targets[seg + 2], targets[seg + 3], eased_t
        )
        up = ups[seg] if local_t < 0.5 else ups[min(seg + 1, len(ups) - 1)]

        view, world, rotation3 = look_at_view_matrix(pos, tgt, up)
        extrinsics.append(
            AtlasExtrinsics(
                camera_position=pos,
                camera_rotation_matrix=rotation3,  # type: ignore[arg-type]
                camera_world_matrix=world,
                camera_view_matrix=view,
                coordinate_system="right_handed",
                up_axis="Y",
                projection_convention="Atlas pinhole camera (camera-path-constructed), image origin top-left.",
            )
        )
    return extrinsics
