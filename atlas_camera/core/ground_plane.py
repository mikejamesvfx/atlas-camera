"""An artist-placed ground plane, built from parameters rather than measured.

This is SUPPORT geometry, not evidence. Every other ground in Atlas is derived
from a depth map (``proxy_geometry._build_ground_primitive``,
``plane_extraction``, ``room_layout``) and inherits that depth map's errors.
The 2026-08-18 gravity-locked ground experiment measured where those errors
actually come from on a real street plate, and the answer was not the plane
fit: the fitted plane tracked the observed road to +/-0.03 m inside 40 m, and
its normal sat 1.8 degrees from gravity. The 48% error came from two depth
models disagreeing about scale, with the wrong one adopted as camera height.
See ``docs/development/gravity-locked-ground-experiment.md``.

The conclusion this module acts on: when the measurement is the thing that is
wrong, the useful control is a ground the artist can place directly.

Two rules it does not break.

**The world never rotates.** World +Y IS the solve's gravity
(``solver._rotation_from_up_vector``). ``tilt_deg`` and ``roll_deg`` turn the
PRIMITIVE's own ``transform_matrix``, so a tilted ground exports to a DCC as
one rotated object while every facade stays plumb. Rotating the world to match
a ground would lean the entire scene.

**Nothing downstream may mistake this for measurement.** The primitive carries
``provenance="artist_placed"`` and ``trust="placeholder"``, reusing the tags
already flowing end-to-end (``comfy/nodes_geometry.py``'s massing node set the
precedent; ``blender/measured.py`` forwards them by name).

Pure stdlib on purpose -- this is a rotation and a 4x4, and the core package
carries no required runtime dependencies.
"""

from __future__ import annotations

import math
from typing import Any

from .schema import AtlasProxyPrimitive

#: Same role tag every projection proxy carries, so the viewport, exporters and
#: retopo pick this plane up through the paths they already walk. Duplicated
#: rather than imported: ``proxy_geometry`` pulls numpy in at import time and
#: this module deliberately does not.
PROXY_ROLE = "projection_proxy"

GROUND_PLANE_SOURCE = "artist_ground_plane"
DEFAULT_NAME = "artist_ground"


def _rot_x(deg: float) -> tuple:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _rot_z(deg: float) -> tuple:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _mul3(a: tuple, b: tuple) -> tuple:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _apply(m: tuple, v: tuple) -> tuple:
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def ground_plane_axes(tilt_deg: float, roll_deg: float) -> tuple:
    """``(u, v, n)`` for a ground plane tilted and rolled about world axes.

    The untilted frame is the one every derived ground already uses:
    ``u=(1,0,0)``, ``v=(0,0,-1)``, ``n=(0,1,0)`` -- the THREE.PlaneGeometry
    local frame (local X=u, Y=v, Z=normal), NOT an XZ quad. Getting that frame
    wrong stands the plane on its edge, which reads as plausible geometry right
    up until a measurement depends on it.

    ``tilt_deg`` rotates about world X, so the far edge lifts or drops.
    ``roll_deg`` rotates about world Z, so the plane banks left or right.
    Applied roll-after-tilt, which keeps roll reading as a horizon bank rather
    than compounding into the tilt.
    """
    rot = _mul3(_rot_z(roll_deg), _rot_x(tilt_deg))
    u = _apply(rot, (1.0, 0.0, 0.0))
    v = _apply(rot, (0.0, 0.0, -1.0))
    n = _apply(rot, (0.0, 1.0, 0.0))
    return u, v, n


def plane_matrix(u: tuple, v: tuple, n: tuple, c: tuple) -> tuple:
    """Row-major 4x4 with columns = local axes and translation ``c``.

    Same layout as ``depth_geometry.plane_transform`` /
    ``proxy_geometry._plane_transform``; kept local so this module stays free
    of the numpy-importing modules.
    """
    return (
        (float(u[0]), float(v[0]), float(n[0]), float(c[0])),
        (float(u[1]), float(v[1]), float(n[1]), float(c[1])),
        (float(u[2]), float(v[2]), float(n[2]), float(c[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def solve_ground_centre(solve: Any) -> tuple:
    """Where to put the plane by default: under the camera, on Y=0.

    After a solve adopts its metric scale the ground IS Y=0 and the camera sits
    at ``camera_height`` above it, so the only open question is where in XZ to
    centre a finite quad. Directly beneath the camera is the answer that puts
    the plane in frame without the artist hunting for it.

    Falls back to the world origin when the solve carries no usable position,
    which is the documented "or 0,0 if easier" behaviour rather than a failure.
    """
    try:
        pos = solve.camera.extrinsics.camera_position
        x, z = float(pos[0]), float(pos[2])
        if math.isfinite(x) and math.isfinite(z):
            return (x, 0.0, z)
    except Exception:
        pass
    return (0.0, 0.0, 0.0)


def build_ground_plane_primitive(
    *,
    width_m: float,
    depth_m: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
    centre: tuple = (0.0, 0.0, 0.0),
    name: str = DEFAULT_NAME,
    provenance: str = "artist_placed",
    trust: str = "placeholder",
    extra_metadata: dict | None = None,
) -> AtlasProxyPrimitive:
    """A finite ground quad at a chosen size, position and orientation.

    ``centre`` is the anchor (normally :func:`solve_ground_centre`); the three
    offsets move the plane from there, with ``offset_y`` raising or lowering it.
    Sizes are clamped to a positive minimum so a zeroed widget yields a
    degenerate-but-valid quad instead of a NaN transform downstream.
    """
    width = max(float(width_m), 1e-3)
    depth = max(float(depth_m), 1e-3)
    u, v, n = ground_plane_axes(float(tilt_deg), float(roll_deg))
    c = (float(centre[0]) + float(offset_x),
         float(centre[1]) + float(offset_y),
         float(centre[2]) + float(offset_z))

    metadata = {
        "role": PROXY_ROLE,
        "source": GROUND_PLANE_SOURCE,
        # The load-bearing tags. This plane is a placement, not a measurement,
        # and no downstream consumer may promote it to one.
        "provenance": provenance,
        "trust": trust,
        "tilt_deg": float(tilt_deg),
        "roll_deg": float(roll_deg),
        "width_m": width,
        "depth_m": depth,
        "centre": [c[0], c[1], c[2]],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return AtlasProxyPrimitive(
        name=name,
        primitive_type="plane",
        transform_matrix=plane_matrix(u, v, n, c),
        # Third component is 0.0 for a plane, matching every other ground
        # primitive in the codebase (a plane has no thickness).
        dimensions=(width, depth, 0.0),
        material="atlas_projection_proxy",
        metadata=metadata,
    )
