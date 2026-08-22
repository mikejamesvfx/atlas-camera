"""Shared extrinsics conversion for the DCC exporters.

Every writer in this package starts from the SAME thing: the Atlas 4x4
``camera_view_matrix`` / ``camera_world_matrix`` (row-major storage,
column-vector convention — ``world_point = M @ point``, translation in the
last COLUMN, right-handed Y-up). What differs is only how each host wants it
spelled. This module is that boundary, and the only place these conversions
are written:

* :func:`euler_degrees` — the rotation block as XYZ Euler degrees, with the
  gimbal-lock branch. Nuke and Maya do NOT agree on the composition order, so
  the caller states it explicitly (see below).
* :func:`translation` — the position, read off the last column.
* :func:`row_vector_flat` — the transpose into the flat 16-value row-vector
  layout Maya's ``cmds.xform -matrix`` and USD's ``Gf.Matrix4d`` both take.
* :func:`blender_matrix_from_atlas` / :func:`blender_point_from_atlas` — the
  Y-up -> Z-up axis swap.

Per the layering rule this is an ADAPTER-side module: it converts, and nothing
in ``core`` may depend on it. It builds world math from the 4x4 only — never
from a bare 3x3, whose transpose is ambiguous.

Rotation order is a genuine trap
--------------------------------
Nuke's ``rot_order XYZ`` composes ``R = Rx(a) @ Ry(b) @ Rz(c)``.
Maya's default rotate order — ALSO written "xyz" in its UI — composes
``C = Rz(c) @ Ry(b) @ Rx(a)``, the exact reverse. Same label, different
matrix. So the constants below name the MATRIX PRODUCT (factors listed
left-to-right) rather than trusting either host's spelling, and callers pass
one explicitly; there is no default. Both return ``(rx, ry, rz)`` in degrees
regardless.
"""

from __future__ import annotations

import math
from typing import Sequence

Matrix4Like = Sequence[Sequence[float]]

#: ``R = Rx(rx) @ Ry(ry) @ Rz(rz)`` — Nuke's ``rot_order XYZ``.
COMPOSITION_XYZ = "Rx@Ry@Rz"
#: ``R = Rz(rz) @ Ry(ry) @ Rx(rx)`` — Maya's default rotate order "xyz".
COMPOSITION_ZYX = "Rz@Ry@Rx"

#: Below this |cos(middle angle)| the outer two angles are not separately
#: observable; the decomposition folds one into the other and zeroes it.
GIMBAL_EPSILON = 1e-6


def translation(matrix: Matrix4Like) -> tuple[float, float, float]:
    """The position stored in the last column of an Atlas 4x4."""
    return (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))


def euler_degrees(
    matrix: Matrix4Like,
    *,
    composition: str,
) -> tuple[float, float, float]:
    """Decompose the rotation block of an Atlas 4x4 into ``(rx, ry, rz)`` degrees.

    ``composition`` must be :data:`COMPOSITION_XYZ` or :data:`COMPOSITION_ZYX`
    — see the module docstring for why the caller has to say which.

    Both branches handle gimbal lock, where the middle angle saturates at ±90
    and the outer two collapse into a single observable sum/difference: the
    convention here is to zero the third angle and fold it into the first.
    Without it the standard extraction reduces to ``atan2(0.0, 0.0) == 0.0``
    and silently discards the camera's whole pitch — which is precisely how
    this used to be wrong for Maya and right for Nuke (fixed 2026-08-01).
    """
    if composition == COMPOSITION_XYZ:
        return _euler_x_first(matrix)
    if composition == COMPOSITION_ZYX:
        return _euler_z_first(matrix)
    raise ValueError(
        f"Unknown Euler composition {composition!r} — expected "
        f"{COMPOSITION_XYZ!r} (Nuke) or {COMPOSITION_ZYX!r} (Maya)."
    )


def _euler_x_first(m: Matrix4Like) -> tuple[float, float, float]:
    """``R = Rx(a) @ Ry(b) @ Rz(c)`` — R[0][2] = sin(b) is the middle angle.

    Round-trip exact (1e-16) on real solves. Used for Nuke's RENDER camera so
    its translate/rotate channels stay unlocked/animatable — Nuke's
    ``useMatrix true`` disables the TRS knobs, so a matrix-driven camera can't
    be keyframed. Pair with ``rot_order XYZ`` on the node.
    """
    sy = max(-1.0, min(1.0, float(m[0][2])))
    b = math.asin(sy)
    if abs(math.cos(b)) > GIMBAL_EPSILON:
        a = math.atan2(-float(m[1][2]), float(m[2][2]))
        c = math.atan2(-float(m[0][1]), float(m[0][0]))
    else:  # gimbal lock: fold c into a
        a = math.atan2(float(m[2][1]), float(m[1][1]))
        c = 0.0
    return math.degrees(a), math.degrees(b), math.degrees(c)


def _euler_z_first(m: Matrix4Like) -> tuple[float, float, float]:
    """``C = Rz(c) @ Ry(b) @ Rx(a)`` — C[2][0] = -sin(b) is the middle angle.

    ``atan2(-C20, hypot(C21, C22))`` rather than ``asin(-C20)``: the hypot form
    is the cos(b) the lock test needs anyway and stays well-conditioned as the
    matrix drifts off orthonormal.
    """
    cos_b = math.hypot(float(m[2][1]), float(m[2][2]))
    b = math.atan2(-float(m[2][0]), cos_b)
    if cos_b > GIMBAL_EPSILON:
        a = math.atan2(float(m[2][1]), float(m[2][2]))
        c = math.atan2(float(m[1][0]), float(m[0][0]))
    else:  # gimbal lock: fold c into a
        a = math.atan2(-float(m[1][2]), float(m[1][1]))
        c = 0.0
    return math.degrees(a), math.degrees(b), math.degrees(c)


def row_vector_flat(
    matrix: Matrix4Like,
    *,
    assume_affine: bool = False,
) -> list[float]:
    """Atlas column-vector 4x4 -> flat 16-value row-vector matrix.

    Atlas is column-vector (``p' = M @ p``, translation in the last COLUMN);
    Maya's ``cmds.xform -matrix`` and USD's ``Gf.Matrix4d`` are both
    row-vector (``p' = p @ M``, translation in the last ROW), so this
    transposes on the way out. No coordinate-axis swap is involved — Maya and
    USD-as-Atlas-writes-it are both right-handed Y-up.

    ``assume_affine`` forces the source's bottom row to ``(0, 0, 0, 1)``, i.e.
    the output's last column. Maya's camera and proxy transforms have always
    been written that way (they are affine by construction); USD passes the
    row through so a caller-supplied projective matrix survives.
    """
    if assume_affine:
        last = (0.0, 0.0, 0.0, 1.0)
    else:
        last = tuple(float(v) for v in matrix[3])
    return [
        matrix[0][0], matrix[1][0], matrix[2][0], last[0],
        matrix[0][1], matrix[1][1], matrix[2][1], last[1],
        matrix[0][2], matrix[1][2], matrix[2][2], last[2],
        matrix[0][3], matrix[1][3], matrix[2][3], last[3],
    ]


def blender_point_from_atlas(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Atlas Y-up point -> Blender Z-up: ``(x, y, z) -> (x, -z, y)``."""
    return (float(x), -float(z), float(y))


def atlas_point_from_blender(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Blender Z-up point -> Atlas Y-up: ``(x, y, z) -> (x, z, -y)``."""
    return (float(x), float(z), -float(y))


def blender_matrix_from_atlas(matrix: Matrix4Like) -> list[list[float]]:
    """Atlas Y-up world matrix -> Blender Z-up: ``M_blender = T @ M_atlas``.

    ``T`` is the same map :func:`blender_point_from_atlas` applies to points —
    new_Y = -old_Z, new_Z = old_Y — so a camera's position AND rotation both
    land correctly. Left-multiplication only: the camera's LOCAL frame is
    -Z-forward / +Y-up in both Atlas and Blender, so the local side is not
    conjugated.
    """
    return [
        [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
        [-matrix[2][0], -matrix[2][1], -matrix[2][2], -matrix[2][3]],
        [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def atlas_matrix_from_blender(matrix: Matrix4Like) -> list[list[float]]:
    """Blender Z-up world matrix -> Atlas Y-up: ``M_atlas = T^-1 @ M_blender``.

    This is the exact inverse of :func:`blender_matrix_from_atlas`. It belongs
    at this adapter boundary so imported Blender truth never leaks Z-up values
    into Atlas World evaluation.
    """
    return [
        [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
        [-matrix[1][0], -matrix[1][1], -matrix[1][2], -matrix[1][3]],
        [0.0, 0.0, 0.0, 1.0],
    ]
