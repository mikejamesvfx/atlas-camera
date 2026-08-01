"""Shared DCC extrinsics conversion — the one place the Euler decomposition,
the row/column flatten-transpose and the Blender Z-up swap are defined.

The gimbal-lock cases below are the reason this module exists: the Nuke writer
had a degenerate branch and the Maya writer did not, so the SAME Atlas 4x4
exported correctly to one DCC and wrongly to the other.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix


def _rx(t):
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def _ry(t):
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])


def _rz(t):
    return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])


def _rot3(matrix):
    return np.array([[float(matrix[i][j]) for j in range(3)] for i in range(3)])


def _recompose_maya(rx_deg, ry_deg, rz_deg):
    """Maya's default rotate order "xyz" composes C = Rz @ Ry @ Rx."""
    a, b, c = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    return _rz(c) @ _ry(b) @ _rx(a)


def _recompose_nuke(rx_deg, ry_deg, rz_deg):
    """Nuke's ``rot_order XYZ`` composes R = Rx @ Ry @ Rz."""
    a, b, c = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    return _rx(a) @ _ry(b) @ _rz(c)


# A camera whose RIGHT axis lands on world ±Z sits exactly on Maya's gimbal
# lock (ry = ±90): the middle angle of the Rz@Ry@Rx chain saturates and rx/rz
# collapse into one another. It is NOT an exotic pose — a plain look_at with
# the default up vector, aimed anywhere in the world XY plane, produces it.
# 30 degrees of pitch is the angle the broken decomposition throws away.
_LOCK_POSES = [
    # (eye, target, label)
    ((5.0 * math.cos(math.radians(30.0)), 5.0 * math.sin(math.radians(30.0)), 0.0),
     (0.0, 0.0, 0.0), "ry=+90, pitched down 30deg"),
    ((-5.0 * math.cos(math.radians(30.0)), 5.0 * math.sin(math.radians(30.0)), 0.0),
     (0.0, 0.0, 0.0), "ry=-90, pitched down 30deg"),
    ((0.0, -4.0, 0.0), (4.0, 0.0, 0.0), "ry=+90, pitched up 45deg"),
]

_ORDINARY_POSES = [
    ((0.0, 2.0, 5.0), (0.0, 0.0, 0.0)),
    ((3.0, 1.5, -4.0), (-1.0, 0.5, 2.0)),
    ((-2.0, 6.0, 1.0), (0.0, 0.0, -8.0)),
]


@pytest.mark.parametrize("eye,target,label", _LOCK_POSES)
def test_maya_euler_survives_gimbal_lock(eye, target, label):
    """The Maya decomposition must round-trip AT the singularity.

    Before the fix the degenerate branch did not exist: rx came from
    ``atan2(m[2][1], m[2][2])`` which is ``atan2(0.0, 0.0) == 0.0`` exactly on
    the lock, silently discarding the camera's whole pitch.
    """
    from atlas_camera.exporters.maya_exporter import _matrix_to_maya_trs

    _view, world, _rot = look_at_view_matrix(eye, target)
    original = _rot3(world)
    assert abs(abs(original[2][0]) - 1.0) < 1e-12, f"{label} is not on the lock"

    t, (rx, ry, rz) = _matrix_to_maya_trs(world)
    assert t == pytest.approx(eye)
    np.testing.assert_allclose(_recompose_maya(rx, ry, rz), original, atol=1e-9)


@pytest.mark.parametrize("eye,target,label", _LOCK_POSES)
def test_nuke_euler_survives_the_same_poses(eye, target, label):
    """The reference behaviour: Nuke already round-trips these matrices."""
    from atlas_camera.exporters.nuke_exporter import _matrix_to_nuke_euler_xyz

    _view, world, _rot = look_at_view_matrix(eye, target)
    rx, ry, rz = _matrix_to_nuke_euler_xyz([list(r) for r in world])
    np.testing.assert_allclose(_recompose_nuke(rx, ry, rz), _rot3(world), atol=1e-9)


@pytest.mark.parametrize("eye,target", _ORDINARY_POSES)
def test_maya_and_nuke_still_round_trip_ordinary_poses(eye, target):
    """The fix must not disturb the non-degenerate path either writer uses."""
    from atlas_camera.exporters.maya_exporter import _matrix_to_maya_trs
    from atlas_camera.exporters.nuke_exporter import _matrix_to_nuke_euler_xyz

    _view, world, _rot = look_at_view_matrix(eye, target)
    original = _rot3(world)

    _t, maya_angles = _matrix_to_maya_trs(world)
    np.testing.assert_allclose(_recompose_maya(*maya_angles), original, atol=1e-9)

    nuke_angles = _matrix_to_nuke_euler_xyz([list(r) for r in world])
    np.testing.assert_allclose(_recompose_nuke(*nuke_angles), original, atol=1e-9)


# ---------------------------------------------------------------------------
# The shared module itself
# ---------------------------------------------------------------------------

_ALL_POSES = [(e, t) for e, t, _ in _LOCK_POSES] + list(_ORDINARY_POSES)


def test_the_two_compositions_are_exposed_and_genuinely_differ():
    """Nuke and Maya do NOT agree on rotation order — Nuke composes
    R = Rx @ Ry @ Rz, Maya composes C = Rz @ Ry @ Rx. Both are spelled "XYZ"
    in their own UI, which is exactly why the shared module names the matrix
    product instead of trusting either label."""
    from atlas_camera.exporters import dcc_transform as dt

    assert dt.COMPOSITION_XYZ != dt.COMPOSITION_ZYX

    _view, world, _rot = look_at_view_matrix((3.0, 1.5, -4.0), (-1.0, 0.5, 2.0))
    xyz = dt.euler_degrees(world, composition=dt.COMPOSITION_XYZ)
    zyx = dt.euler_degrees(world, composition=dt.COMPOSITION_ZYX)
    assert xyz != zyx  # same matrix, different convention, different angles

    original = _rot3(world)
    np.testing.assert_allclose(_recompose_nuke(*xyz), original, atol=1e-9)
    np.testing.assert_allclose(_recompose_maya(*zyx), original, atol=1e-9)


def test_unknown_composition_is_rejected():
    from atlas_camera.exporters import dcc_transform as dt

    with pytest.raises(ValueError):
        dt.euler_degrees([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                         composition="YXZ")


@pytest.mark.parametrize("eye,target", _ALL_POSES)
@pytest.mark.parametrize("composition", ["XYZ", "ZYX"])
def test_both_compositions_round_trip_every_pose(eye, target, composition):
    """One implementation, one gimbal branch — both conventions survive the
    singularity that only one of the two writers used to handle."""
    from atlas_camera.exporters import dcc_transform as dt

    key = dt.COMPOSITION_XYZ if composition == "XYZ" else dt.COMPOSITION_ZYX
    recompose = _recompose_nuke if composition == "XYZ" else _recompose_maya

    _view, world, _rot = look_at_view_matrix(eye, target)
    angles = dt.euler_degrees(world, composition=key)
    np.testing.assert_allclose(recompose(*angles), _rot3(world), atol=1e-9)


@pytest.mark.parametrize("eye,target", _ALL_POSES)
def test_exporters_delegate_to_the_shared_decomposition(eye, target):
    """The whole point of the extraction: no second implementation survives."""
    from atlas_camera.exporters import dcc_transform as dt
    from atlas_camera.exporters.maya_exporter import _matrix_to_maya_trs
    from atlas_camera.exporters.nuke_exporter import _matrix_to_nuke_euler_xyz

    _view, world, _rot = look_at_view_matrix(eye, target)

    t, maya_angles = _matrix_to_maya_trs(world)
    assert t == dt.translation(world)
    assert maya_angles == dt.euler_degrees(world, composition=dt.COMPOSITION_ZYX)

    nuke_angles = _matrix_to_nuke_euler_xyz([list(r) for r in world])
    assert nuke_angles == dt.euler_degrees(world, composition=dt.COMPOSITION_XYZ)


def test_translation_reads_the_last_column():
    from atlas_camera.exporters import dcc_transform as dt

    m = [[1, 0, 0, 7.5], [0, 1, 0, -2.0], [0, 0, 1, 3.25], [0, 0, 0, 1]]
    assert dt.translation(m) == (7.5, -2.0, 3.25)


def test_row_vector_flat_transposes_atlas_into_row_vector_layout():
    """Atlas stores column-vector matrices (translation in the last COLUMN);
    Maya's cmds.xform -matrix and USD's Gf.Matrix4d both want row-vector
    (translation in the last ROW)."""
    from atlas_camera.exporters import dcc_transform as dt

    m = [
        [1.0, 2.0, 3.0, 10.0],
        [4.0, 5.0, 6.0, 20.0],
        [7.0, 8.0, 9.0, 30.0],
        [0.1, 0.2, 0.3, 1.0],
    ]
    assert dt.row_vector_flat(m) == [
        1.0, 4.0, 7.0, 0.1,
        2.0, 5.0, 8.0, 0.2,
        3.0, 6.0, 9.0, 0.3,
        10.0, 20.0, 30.0, 1.0,
    ]
    # The Maya writer forces the bottom row affine (it always fed 0,0,0,1).
    assert dt.row_vector_flat(m, assume_affine=True) == [
        1.0, 4.0, 7.0, 0.0,
        2.0, 5.0, 8.0, 0.0,
        3.0, 6.0, 9.0, 0.0,
        10.0, 20.0, 30.0, 1.0,
    ]


def test_maya_and_usd_share_the_flatten_transpose():
    from atlas_camera.exporters import dcc_transform as dt
    from atlas_camera.exporters.maya_exporter import _maya_matrix_from_atlas

    _view, world, _rot = look_at_view_matrix((3.0, 1.5, -4.0), (-1.0, 0.5, 2.0))
    assert _maya_matrix_from_atlas(world) == dt.row_vector_flat(world, assume_affine=True)


def test_blender_z_up_swap_maps_y_up_to_z_up():
    """Atlas Y-up -> Blender Z-up is T: (x, y, z) -> (x, -z, y)."""
    from atlas_camera.exporters import dcc_transform as dt

    assert dt.blender_point_from_atlas(1.0, 2.0, 3.0) == (1.0, -3.0, 2.0)

    identity = [[1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0]]
    converted = dt.blender_matrix_from_atlas(identity)
    # Camera position must land on the swapped point.
    assert (converted[0][3], converted[1][3], converted[2][3]) == (1.0, -3.0, 2.0)
    assert converted[3] == [0.0, 0.0, 0.0, 1.0]
    # Rotation must be conjugated, not merely permuted: a world +Y direction
    # (Atlas up) has to come out as Blender +Z.
    basis = np.array([[converted[i][j] for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(basis @ np.array([0.0, 1.0, 0.0]),
                               np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_blender_exporter_uses_the_shared_swap():
    from atlas_camera.exporters import dcc_transform as dt

    _view, world, _rot = look_at_view_matrix((3.0, 1.5, -4.0), (-1.0, 0.5, 2.0))
    converted = dt.blender_matrix_from_atlas(world)
    assert converted[1] == [-world[2][0], -world[2][1], -world[2][2], -world[2][3]]
    assert converted[2] == [world[1][0], world[1][1], world[1][2], world[1][3]]


# ---------------------------------------------------------------------------
# Exporter package surface
#
# Lives here rather than in test_review_package.py because it guards a change
# to the exporters package itself: the review package used to route Blender
# through a ONE-element ``_DCC_EXPORTERS`` list typed by a ``DccExporter``
# Protocol, while Maya and Nuke were called directly with kwargs the Protocol
# could not describe. The seam was hypothetical — one adapter, no second
# implementor, no importer outside this package — so it was removed. Nothing
# asserted the Blender script actually landed in the package; now something
# does.
# ---------------------------------------------------------------------------

def test_review_package_still_writes_every_dcc_script(tmp_path):
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasSolve
    from atlas_camera.exporters.review_package import build_review_package

    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"not a real png, only copied")
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(image_width=1280, image_height=720,
                                        focal_length_mm=35.0),
        ),
        image_path=str(image_path),
        image_width=1280,
        image_height=720,
        source_method="test",
    )

    result = build_review_package(solve, tmp_path, include_usd=False)

    for key, filename in (
        ("maya_open_scene", "maya_open_scene.py"),
        ("nuke_cards", "nuke_cards.py"),
        ("blender_open_scene", "blender_open_scene.py"),
    ):
        assert result.files[key] == result.package_dir / filename
        assert (result.package_dir / filename).is_file()
    assert "bpy" in (result.package_dir / "blender_open_scene.py").read_text(
        encoding="utf-8")
