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

import math

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import AtlasCameraPath, AtlasExtrinsics

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


def _catmull_rom_1d(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Scalar Catmull-Rom with the same basis as ``_catmull_rom`` — used for the
    fov channel so a keyframed lens ramp follows the exact same curve shape as
    the position/target channels it plays against."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _vsub(a: _Vec3, b: _Vec3) -> _Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vcross(a: _Vec3, b: _Vec3) -> _Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vdot(a: _Vec3, b: _Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vlen(a: _Vec3) -> float:
    return math.sqrt(_vdot(a, a))


def _rotate_about_axis(v: _Vec3, axis: _Vec3, angle_rad: float) -> _Vec3:
    """Rodrigues rotation of ``v`` about UNIT ``axis``."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    cross = _vcross(axis, v)
    dot = _vdot(axis, v)
    return (
        v[0] * c + cross[0] * s + axis[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + axis[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + axis[2] * dot * (1.0 - c),
    )


# --- 🎬 Cinematic rig-noise (track chatter / jib bounce / resonance) ----------
#
# Mirrored by atlasHash01 / atlasShakeOffsetsJS / atlasApplyShakeToPoseJS in
# atlas_blockout.js — the same accepted hand-sync duplication as the
# Catmull-Rom/easing functions above, pinned by tests/test_frontend_mirrors.py.
# Determinism doctrine: phases come from a 32-bit INTEGER hash of (seed, k) —
# never a float sin-fract hash and never random() at sample time — so the JS
# preview, the JS bake, and this Python export sample bit-identical curves.


def _hash01(n: int) -> float:
    """32-bit integer hash -> [0, 1). Every step masked to 32 bits so the JS
    twin (Math.imul + >>> 0) is bit-exact."""
    n &= 0xFFFFFFFF
    n = ((n ^ 61) ^ (n >> 16)) & 0xFFFFFFFF
    n = (n * 9) & 0xFFFFFFFF
    n = (n ^ (n >> 4)) & 0xFFFFFFFF
    n = (n * 0x27D4EB2D) & 0xFFFFFFFF
    n = (n ^ (n >> 15)) & 0xFFFFFFFF
    return n / 4294967296.0


_TWO_PI = 6.283185307179586


def shake_offsets(
    frame: float, fps: float, intensity: float, seed: int
) -> tuple[float, float, float, float, float, float]:
    """Rig-noise offsets at ``frame`` -> (dx, dy, dz, rx_deg, ry_deg, rz_deg).

    Translations are DIMENSIONLESS fractions of the camera->target distance
    (``apply_shake_to_pose`` multiplies by that distance, so amplitude is
    scene-scale-correct with no extra plumbing); rotations are degrees.
    Continuous in ``frame`` (fractional preview frames and integer bake/export
    frames sample the same signal). Three bands at intensity 1:

    - jib bounce   0.35/0.45/0.55/0.65 Hz — vertical sway + slight pitch/roll
    - track chatter 9.1/11.7/13.9 Hz     — small lateral/axial buzz + yaw
    - resonance    4.3 Hz, beat-modulated by 0.18 Hz — dy + pitch
    """
    if not (intensity > 0.0):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    t = frame / fps if fps > 0.0 else 0.0

    def ph(k: int) -> float:
        return _hash01(((seed * 1013) & 0xFFFFFFFF) + k) * _TWO_PI

    sin = math.sin
    # Jib bounce (low frequency)
    dy = 0.0040 * (sin(_TWO_PI * 0.35 * t + ph(0)) + 0.6 * sin(_TWO_PI * 0.65 * t + ph(1)))
    dx = 0.0012 * sin(_TWO_PI * 0.45 * t + ph(2))
    dz = 0.0
    rx = 0.12 * sin(_TWO_PI * 0.35 * t + ph(3))
    ry = 0.0
    rz = 0.07 * sin(_TWO_PI * 0.55 * t + ph(4))
    # Track chatter (high frequency)
    dx += 0.0007 * (
        sin(_TWO_PI * 9.1 * t + ph(5)) + sin(_TWO_PI * 11.7 * t + ph(6))
        + sin(_TWO_PI * 13.9 * t + ph(7))
    )
    dz += 0.0007 * (
        sin(_TWO_PI * 9.1 * t + ph(8)) + sin(_TWO_PI * 11.7 * t + ph(9))
        + sin(_TWO_PI * 13.9 * t + ph(10))
    )
    ry += 0.03 * sin(_TWO_PI * 11.7 * t + ph(11))
    # Mechanical resonance (beat-modulated mid frequency)
    dy += 0.0015 * sin(_TWO_PI * 4.3 * t + ph(12)) * (0.5 + 0.5 * sin(_TWO_PI * 0.18 * t + ph(13)))
    rx += 0.02 * sin(_TWO_PI * 4.3 * t + ph(14)) * (0.5 + 0.5 * sin(_TWO_PI * 0.18 * t + ph(15)))
    return (
        dx * intensity, dy * intensity, dz * intensity,
        rx * intensity, ry * intensity, rz * intensity,
    )


def apply_shake_to_pose(
    position: _Vec3,
    target: _Vec3,
    up: _Vec3,
    offsets: tuple[float, float, float, float, float, float],
) -> tuple[_Vec3, _Vec3, _Vec3]:
    """Apply ``shake_offsets`` to a pose BEFORE lookAt -> (pos', tgt', up').

    Translation moves position AND target by the same camera-frame vector
    (a rig translates; it does not re-aim at the old target), scaled by the
    camera->target distance. Pitch/yaw rotate the forward vector (Rodrigues),
    roll rotates ``up`` about the new forward. Callers feed the result to
    their existing lookAt (Python ``look_at_view_matrix``, JS ``camera.lookAt``)
    so both sides produce identical world matrices.
    """
    dx, dy, dz, rx_deg, ry_deg, rz_deg = offsets
    if dx == 0.0 and dy == 0.0 and dz == 0.0 and rx_deg == 0.0 and ry_deg == 0.0 and rz_deg == 0.0:
        return position, target, up
    fwd = _vsub(target, position)
    dist = _vlen(fwd)
    if dist <= 1e-12:
        dist = 1.0
    f = _vscale(fwd, 1.0 / dist)
    r = _vcross(f, (0.0, 1.0, 0.0))
    rlen = _vlen(r)
    r = _vscale(r, 1.0 / rlen) if rlen > 1e-6 else (1.0, 0.0, 0.0)
    u = _vcross(r, f)
    trans = (
        (r[0] * dx + u[0] * dy + f[0] * dz) * dist,
        (r[1] * dx + u[1] * dy + f[1] * dz) * dist,
        (r[2] * dx + u[2] * dy + f[2] * dz) * dist,
    )
    pos2 = _vadd(position, trans)
    deg = math.pi / 180.0
    f2 = _rotate_about_axis(f, u, ry_deg * deg)
    f2 = _rotate_about_axis(f2, r, rx_deg * deg)
    tgt2 = _vadd(pos2, _vscale(f2, dist))
    up2 = _rotate_about_axis(up, f2, rz_deg * deg)
    return pos2, tgt2, up2


def _apply_easing(t: float, easing: str) -> float:
    if easing == "ease_in":
        return t * t
    if easing == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if easing == "ease_in_out":
        return 3.0 * t * t - 2.0 * t * t * t
    return t  # "linear" and unknown values fall back to linear


def sample_camera_path(
    path: AtlasCameraPath, *, apply_shake: bool = False
) -> list[AtlasExtrinsics]:
    """Sample ``path`` into one ``AtlasExtrinsics`` per frame in ``0..frame_count-1``.

    - 0 keyframes: returns an empty list.
    - 1 keyframe: that pose repeated for every frame (a static "path").
    - >=2 keyframes: Catmull-Rom through the keyframes' ``position``/``target``
      (endpoints duplicated as phantom control points), with each segment's
      local ``t`` eased by its *starting* keyframe's ``easing`` before the
      spline is evaluated — so sampling exactly at a keyframe's ``frame_index``
      always reproduces that keyframe's ``position``/``target`` exactly
      (t=0 or t=1 passes through regardless of easing).

    ``apply_shake=True`` layers the path's 🎬 Cinematic rig-noise
    (``shake_enabled``/``shake_intensity``/``shake_seed``) onto every frame.
    Default False is DELIBERATE: analysis consumers (move_budget, path repair,
    completion) must see the intended move, not seed-dependent cosmetic jitter
    — only the USD camera export opts in.
    """
    frame_count = max(0, int(path.frame_count))
    keyframes = path.keyframes
    if frame_count == 0 or not keyframes:
        return []

    shaking = bool(apply_shake) and bool(path.shake_enabled) and float(path.shake_intensity) > 0.0

    def _extrinsics_for(pos: _Vec3, tgt: _Vec3, up: _Vec3, frame: int) -> AtlasExtrinsics:
        if shaking:
            pos, tgt, up = apply_shake_to_pose(
                pos, tgt, up,
                shake_offsets(float(frame), float(path.fps),
                              float(path.shake_intensity), int(path.shake_seed)),
            )
        view, world, rotation3 = look_at_view_matrix(pos, tgt, up)
        return AtlasExtrinsics(
            camera_position=pos,
            camera_rotation_matrix=rotation3,  # type: ignore[arg-type]
            camera_world_matrix=world,
            camera_view_matrix=view,
            coordinate_system="right_handed",
            up_axis="Y",
            projection_convention="Atlas pinhole camera (camera-path-constructed), image origin top-left.",
        )

    if len(keyframes) == 1:
        kf = keyframes[0]
        if shaking:
            # A locked-off pose with rig noise is per-frame motion (the JS
            # preview shakes single-keyframe paths too).
            return [
                _extrinsics_for(kf.position, kf.target, kf.up, frame)
                for frame in range(frame_count)
            ]
        extr = _extrinsics_for(kf.position, kf.target, kf.up, 0)
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

        extrinsics.append(_extrinsics_for(pos, tgt, up, frame))
    return extrinsics


def sample_camera_path_fov_deg(path: AtlasCameraPath) -> list[float] | None:
    """Sample the keyframed VERTICAL fov channel — one value per frame, or ``None``.

    Returns ``None`` (not a list) when NO keyframe carries ``fov_deg`` — the
    static-lens case every pre-Vertigo path is in, so callers keep their
    existing fixed-intrinsics behaviour unchanged. When at least one keyframe
    has ``fov_deg``, keyframes without one inherit the nearest PREVIOUS defined
    value (leading gaps back-fill from the first defined) and the filled
    channel interpolates through the exact same phantom-endpoint Catmull-Rom +
    per-segment easing as the position/target channels, so sampling at a
    keyframe's ``frame_index`` reproduces its fov exactly.

    Mirrored by ``sampleFovChannel`` in ``atlas_blockout.js`` (same accepted
    hand-sync duplication as ``sample_camera_path`` itself — pinned by
    ``tests/test_frontend_mirrors.py``).
    """
    frame_count = max(0, int(path.frame_count))
    keyframes = path.keyframes
    if frame_count == 0 or not keyframes:
        return None
    if all(kf.fov_deg is None for kf in keyframes):
        return None

    first_defined = next(kf.fov_deg for kf in keyframes if kf.fov_deg is not None)
    fovs: list[float] = []
    prev = float(first_defined)
    for kf in keyframes:
        if kf.fov_deg is not None:
            prev = float(kf.fov_deg)
        fovs.append(prev)

    if len(keyframes) == 1:
        return [fovs[0]] * frame_count

    padded = [fovs[0]] + fovs + [fovs[-1]]
    frame_indices = [kf.frame_index for kf in keyframes]
    easings = [kf.easing for kf in keyframes]

    out: list[float] = []
    for frame in range(frame_count):
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
        out.append(
            _catmull_rom_1d(padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3], eased_t)
        )
    return out


# ---------------------------------------------------------------------------
# One-click move presets, server-side. Mirrors atlas_blockout.js
# applyMovePreset / computePresetEndPose (hand-sync duplication, pinned by
# tests/test_frontend_mirrors.py — same doctrine as SCENE_TYPE_PRESETS).
# APPEND-ONLY: the move names serialize into saved workflows via the
# AtlasCameraMovePreset combo.
#
# THE PIVOT IS A PARAMETER, NOT A CONSTANT (revised 2026-08-15). It used to
# be hardwired to ``ground_lookat_pivot`` because that is the pivot
# AtlasAddPatchView reconstructs an ``exact_view_override`` delta about, so a
# ground pivot was the only one the delta could reproduce. That bought
# exactness and cost the MOVE: the ground pivot is where the view ray meets
# Y=0, which for a near-level camera is far past the subject (measured live on
# the ghost-town street solve: ~43 m against the viewport's ~9.8 m median-depth
# pivot). Orbit translation is ``2·R·sin(angle/2)``, so the same 15° swung the
# eye 11.2 m instead of 2.6 m — the arcs the artist got were nothing like the
# arcs the ⤴/⤵ buttons give, and the disocclusion they opened was 4x what the
# fill chain was sized for.
#
# The fix is to carry the pivot IN the exact-view string (``pivot=x,y,z``,
# parsed by ``comfy.view_prompts._parse_exact_pivot``) instead of assuming it,
# so any pivot stays exactly reproducible. ``scene_median_depth_pivot`` below
# mirrors the viewport's own choice; ``ground_lookat_pivot`` remains the
# default when no scene geometry exists and the legacy default for every
# consumer that receives no pivot (viewport-extracted angles still measure
# against the payload's ``orbit_pivot``, which IS the ground pivot).
#
# Both pivots lie ON the recovered central view ray, which is why the dolly
# moves reduce to a pure ``distance_scale`` (the viewport needs its own
# on-axis target construction only because its mesh-centre pivot can carry the
# artist's 🎯 offset and so sit off-axis).
# ---------------------------------------------------------------------------
PRESET_MOVE_ANGLE_DEG = 15.0   # JS MOVE_ANGLE_DEG
PRESET_DOLLY_FRAC = 0.2        # JS MOVE_DOLLY_FRAC
PRESET_PUSH_IN_FRAC = 0.35     # JS PUSH_IN_FRAC
PRESET_ARC_DOLLY_FRAC = 0.15   # JS ARC_DOLLY_FRAC
PRESET_FRAME_COUNT = 100       # JS PATH_FRAME_COUNT
PRESET_FPS = 24.0              # JS PATH_FPS
PRESET_DOLLY_PAN_FRAC = 0.18   # JS DOLLY_PAN_FRAC
PRESET_CRANE_FRAC = 0.25       # JS CRANE_FRAC
PRESET_MOVES = ("orbit_left", "orbit_right", "pan_left", "pan_right",
                "dolly_in", "arc_left", "arc_right", "push_in", "vertigo",
                "dolly_pan_left", "dolly_pan_right", "crane_up")


def build_preset_camera_path(
    extrinsics: AtlasExtrinsics,
    move: str,
    *,
    angle_deg: float = PRESET_MOVE_ANGLE_DEG,
    frame_count: int = PRESET_FRAME_COUNT,
    fps: float = PRESET_FPS,
    easing: str = "ease_in_out",
    fov_deg: float | None = None,
    pivot: _Vec3 | None = None,
) -> tuple[AtlasCameraPath, tuple[float, float, float]]:
    """Build a one-click move as an ``AtlasCameraPath``, plus its END delta.

    Returns ``(path, (d_azimuth_deg, d_elevation_deg, distance_scale))`` where
    the delta reproduces the path's FINAL pose via
    ``orbit_camera(extrinsics, pivot, *delta)`` — the exact contract
    ``AtlasAddPatchView.exact_view_override`` consumes. ``pivot`` defaults to
    ``ground_lookat_pivot(extrinsics)``; pass ``scene_median_depth_pivot`` (what
    AtlasCameraMovePreset does) for the viewport's own arc radius, and emit the
    pivot alongside the delta so the re-entry can reproduce it.
    Pan moves swivel in place (the target moves, not the eye), which no orbit
    delta can express — they return the zero delta ``(0, 0, 1)`` and the
    caller should warn that patch re-entry lands at the recovered pose.

    Sign convention (verified against the JS rotation algebra): the JS orbit
    rotates the pivot->eye offset by ``a = radians(angle) * (-1 if left)``
    as ``(x cos a + z sin a, y, -x sin a + z cos a)``; ``orbit_camera`` stores
    azimuth as ``atan2(x, z)`` and adds its delta, which expands to the SAME
    expression — so ``d_azimuth_deg`` equals the JS ``a`` in degrees with no
    sign flip. ``fov_deg`` (the solved vertical fov) is only needed by
    ``vertigo``; omitted, vertigo raises.
    """
    from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera
    from atlas_camera.core.schema import AtlasCameraKeyframe

    if move not in PRESET_MOVES:
        raise ValueError(f"unknown move {move!r} — one of {PRESET_MOVES}")
    frame_count = max(2, int(frame_count))
    last = frame_count - 1
    eye = tuple(float(v) for v in extrinsics.camera_position)
    pivot = (tuple(float(v) for v in pivot) if pivot is not None
             else ground_lookat_pivot(extrinsics))
    sign = -1.0 if move.endswith("_left") else 1.0

    def pose_at(delta):
        moved = orbit_camera(extrinsics, pivot, d_azimuth_deg=delta[0],
                             d_elevation_deg=delta[1], distance_scale=delta[2])
        return tuple(float(v) for v in moved.camera_position)

    def kf(frame, position, target, ease, fov=None):
        return AtlasCameraKeyframe(frame_index=frame, position=position,
                                   target=target, up=(0.0, 1.0, 0.0),
                                   fov_deg=fov, easing=ease)

    if move in ("arc_left", "arc_right"):
        mid_delta = (sign * angle_deg / 2.0, 0.0,
                     1.0 - PRESET_ARC_DOLLY_FRAC / 2.0)
        end_delta = (sign * angle_deg, 0.0, 1.0 - PRESET_ARC_DOLLY_FRAC)
        # THREE keyframes (0 / 60% / last) so the Catmull-Rom genuinely curves
        # through the combined orbit+dolly instead of cutting the chord.
        keyframes = [
            kf(0, eye, pivot, easing),
            kf(int(round(last * 0.6)), pose_at(mid_delta), pivot, "linear"),
            kf(last, pose_at(end_delta), pivot, "linear"),
        ]
    elif move in ("orbit_left", "orbit_right"):
        end_delta = (sign * angle_deg, 0.0, 1.0)
        keyframes = [kf(0, eye, pivot, easing),
                     kf(last, pose_at(end_delta), pivot, "linear")]
    elif move in ("dolly_pan_left", "dolly_pan_right"):
        # A dolly and a pan are two bodies: the base rolls forward along the
        # track it started on, the head turns on top of it. So the eye travels
        # the ORIGINAL view axis while the look direction swivels -- not a
        # curve toward a moved target, which is what arc does.
        a_full = math.radians(angle_deg) * sign
        off = (pivot[0] - eye[0], pivot[1] - eye[1], pivot[2] - eye[2])
        dist = math.sqrt(sum(c * c for c in off)) or 1.0
        fwd = tuple(c / dist for c in off)

        def dolly_pan(fraction_of_angle: float, travel: float):
            a = a_full * fraction_of_angle
            # Same handedness as the JS and as the pan branch below. Both
            # were checked against it: on a camera at (0,1.6,10) looking down
            # -Z, a left move puts the target at x = -2.59, which is
            # camera-left.
            rotated = (off[0] * math.cos(a) - off[2] * math.sin(a), off[1],
                       off[0] * math.sin(a) + off[2] * math.cos(a))
            length = math.sqrt(sum(c * c for c in rotated)) or 1.0
            direction = tuple(c / length for c in rotated)
            moved_eye = tuple(eye[i] + fwd[i] * dist * travel for i in range(3))
            target = tuple(moved_eye[i] + direction[i] * dist for i in range(3))
            return moved_eye, target

        mid_eye, mid_target = dolly_pan(0.5, PRESET_DOLLY_PAN_FRAC / 2.0)
        end_eye, end_target = dolly_pan(1.0, PRESET_DOLLY_PAN_FRAC)
        end_delta = (sign * angle_deg, 0.0, 1.0 - PRESET_DOLLY_PAN_FRAC)
        keyframes = [
            kf(0, eye, pivot, easing),
            kf(int(round(last * 0.6)), mid_eye, mid_target, "linear"),
            kf(last, end_eye, end_target, "linear"),
        ]
    elif move == "crane_up":
        # The body rises, the head keeps looking at the same point, so the tilt
        # comes out of the geometry. World +Y, not camera up: a crane arm is
        # vertical even when the camera is rolled.
        off = (pivot[0] - eye[0], pivot[1] - eye[1], pivot[2] - eye[2])
        dist = math.sqrt(sum(c * c for c in off)) or 1.0
        risen = (eye[0], eye[1] + dist * PRESET_CRANE_FRAC, eye[2])
        end_delta = (0.0, 0.0, 1.0)
        keyframes = [kf(0, eye, pivot, easing),
                     kf(last, risen, pivot, "linear")]
    elif move in ("pan_left", "pan_right"):
        # Swivel in place: rotate the eye->pivot ray about world +Y.
        #
        # The sin terms were transposed here, so this rotated the OPPOSITE way
        # to the JS preview and a baked pan_left travelled right. Measured on a
        # camera at (0,1.6,10) looking down -Z: the JS puts pan_left's target at
        # x = -2.59 (camera-left, correct) and this branch put it at +2.59.
        #
        # Only pan was affected. Orbit goes through `pose_at` and agrees with
        # the JS to the centimetre; this branch hand-rolls its rotation and got
        # the handedness wrong. The frontend-mirror test could not catch it
        # because it compares the move NAMES, not the geometry they produce --
        # `test_pan_matches_the_js_handedness` now does.
        end_delta = (0.0, 0.0, 1.0)     # not expressible as an orbit delta
        a = math.radians(angle_deg) * sign
        off = (pivot[0] - eye[0], pivot[1] - eye[1], pivot[2] - eye[2])
        rotated = (off[0] * math.cos(a) - off[2] * math.sin(a), off[1],
                   off[0] * math.sin(a) + off[2] * math.cos(a))
        new_target = (eye[0] + rotated[0], eye[1] + rotated[1],
                      eye[2] + rotated[2])
        keyframes = [kf(0, eye, pivot, easing),
                     kf(last, eye, new_target, "linear")]
    else:                               # dolly_in / push_in / vertigo
        frac = {"dolly_in": PRESET_DOLLY_FRAC, "push_in": PRESET_PUSH_IN_FRAC,
                "vertigo": PRESET_DOLLY_FRAC}[move]
        end_delta = (0.0, 0.0, 1.0 - frac)
        fov0 = fov1 = None
        if move == "vertigo":
            if fov_deg is None:
                raise ValueError("vertigo needs the solved vertical fov_deg "
                                 "to key the counter-zoom")
            fov0 = float(fov_deg)
            fov1 = math.degrees(2.0 * math.atan(
                math.tan(math.radians(fov0) / 2.0) / (1.0 - frac)))
        keyframes = [kf(0, eye, pivot, easing, fov0),
                     kf(last, pose_at(end_delta), pivot, "linear", fov1)]

    path = AtlasCameraPath(keyframes=keyframes, fps=float(fps),
                           frame_count=frame_count)
    return path, (float(end_delta[0]), float(end_delta[1]),
                  float(end_delta[2]))


# Cap on vertices sampled per primitive, mirroring the JS stride
# (`Math.floor(pos.count / 800)`) — the median of a uniform sample of a
# uniformly-sampled relief grid is the median of the whole grid.
_PIVOT_SAMPLES_PER_MESH = 800


def scene_median_depth_pivot(
    solve,
    *,
    fallback_distance: float | None = None,
) -> _Vec3:
    """The orbit pivot the VIEWPORT uses: the central view ray at the scene's
    median vertex depth. Server-side mirror of ``atlas_blockout.js``'s
    ``computeGeometryPivot`` (+ its ``groundPointInView`` fallback).

    Median vertex depth, not a bounding-box centre: a Box3 centre is
    ``(min+max)/2``, dominated by the tails of a full-scene relief mesh that
    spans near foreground to far clip plus fill/outpaint skirts, which parks
    the pivot deep behind the subject (artist-reported 2026-07-09, same
    reasoning recorded in the JS). Relief grids sample the IMAGE uniformly, so
    the median vertex depth is the depth of the middle of the visible surface
    area — "the middle of what the photo shows".

    Falls back — exactly as the viewport does — to the ground point along the
    view ray with the travel CLAMPED to ``fallback_distance`` (the JS
    ``Math.min(-p.y / dir.y, lookAheadDist)``). That clamp is the whole reason
    the viewport never suffered the far-ground-pivot problem: a near-level
    camera's true ground intersection is tens of metres out and the viewport
    simply refuses to orbit that far. ``fallback_distance`` defaults to the
    scene's own depth scale where one is known — the backdrop's ``distance_m``
    times 1.5, exactly the JS ``scene_depth_m * 1.5`` — else 30.

    Pure Python, no numpy — this module's dependency-free style.
    """
    extr = solve.camera.extrinsics
    eye = tuple(float(v) for v in extr.camera_position)
    world = extr.camera_world_matrix
    fwd = (-float(world[0][2]), -float(world[1][2]), -float(world[2][2]))
    norm = math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2) or 1.0
    fwd = (fwd[0] / norm, fwd[1] / norm, fwd[2] / norm)

    scene = getattr(solve, "projection_scene", None)
    prims = list(getattr(scene, "proxy_geometry", None) or [])
    if fallback_distance is None:
        backdrop = next((p for p in prims
                         if p.name == "projection_backdrop"), None)
        depth_m = (backdrop.metadata or {}).get("distance_m") if backdrop else None
        fallback_distance = float(depth_m) * 1.5 if depth_m else 30.0

    depths: list[float] = []
    for prim in prims:
        if prim.primitive_type != "mesh" or prim.name == "projection_backdrop":
            continue
        verts = (prim.metadata or {}).get("vertices") or []
        n = len(verts) // 3
        if not n:
            continue
        m = prim.transform_matrix
        stride = max(1, n // _PIVOT_SAMPLES_PER_MESH)
        for i in range(0, n, stride):
            vx, vy, vz = verts[3 * i], verts[3 * i + 1], verts[3 * i + 2]
            # Row-major 4x4, same convention as every other transform here.
            wx = m[0][0] * vx + m[0][1] * vy + m[0][2] * vz + m[0][3]
            wy = m[1][0] * vx + m[1][1] * vy + m[1][2] * vz + m[1][3]
            wz = m[2][0] * vx + m[2][1] * vy + m[2][2] * vz + m[2][3]
            d = ((wx - eye[0]) * fwd[0] + (wy - eye[1]) * fwd[1]
                 + (wz - eye[2]) * fwd[2])
            if d > 0.0 and math.isfinite(d):
                depths.append(d)

    if depths:
        depths.sort()
        t = depths[len(depths) // 2]
    elif fwd[1] < -1e-3:
        t = min(-eye[1] / fwd[1], float(fallback_distance))
    else:
        t = float(fallback_distance)
    return (eye[0] + t * fwd[0], eye[1] + t * fwd[1], eye[2] + t * fwd[2])
