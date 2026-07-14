"""Predicted-normal alignment for the viewport relight (core.normals).

MoGe predicts normals in its own camera frame; these must be rotated into the
recovered world frame before they can drive the world-space relight. The math
(orthogonal Procrustes against the geometry's own normals) is tested here with
synthetic data — no MoGe model needed.
"""

import numpy as np
import pytest

from atlas_camera.core.normals import (
    align_predicted_normals_to_world,
    procrustes_rotation,
    world_normals_from_depth,
)

W = H = 96
FX = FY = 120.0
CX = CY = 48.0


def _view_matrix(h=5.0):
    # level camera at (0, h, 0), identity rotation — world->cam translation only
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -h), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _rot_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rX = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rY = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rZ = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rZ @ rY @ rX


def _varied_depth():
    """A smooth, non-planar surface so its gradient normals point in many
    directions — a planar depth gives rank-deficient normals and an
    underdetermined rotation fit."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return 20.0 + 4.0 * np.sin(uu / 11.0) + 4.0 * np.cos(vv / 9.0) + 2.0 * np.sin((uu + vv) / 7.0)


def test_procrustes_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    r_true = _rot_xyz(0.3, -0.6, 0.9)
    b = a @ r_true.T
    r = procrustes_rotation(a, b)
    assert np.allclose(r, r_true, atol=1e-6)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-6)   # proper rotation, no reflection


def test_world_normals_are_unit_and_face_camera():
    depth = _varied_depth()
    n, valid = world_normals_from_depth(depth, view_matrix=_view_matrix(), fx=FX, fy=FY, cx=CX, cy=CY)
    assert valid.sum() > 1000
    lens = np.linalg.norm(n[valid], axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-6)                  # unit normals
    # camera at (0,5,0), geometry in front (-Z); normals should face toward +Z-ish
    assert (n[valid][:, 2] > 0).mean() > 0.8


def test_align_recovers_world_normals_from_a_rotated_frame():
    """The whole point: predicted normals in a DIFFERENT (rotated) frame must be
    aligned back to the geometry's world normals."""
    depth = _varied_depth()
    vm = _view_matrix()
    geom, valid = world_normals_from_depth(depth, view_matrix=vm, fx=FX, fy=FY, cx=CX, cy=CY)
    # fake a model camera frame: rotate the true world normals by R_true
    r_true = _rot_xyz(0.4, 0.7, -0.5)
    predicted = geom @ r_true.T                              # HxWx3 in the "model" frame

    aligned, av = align_predicted_normals_to_world(predicted, depth, view_matrix=vm, fx=FX, fy=FY, cx=CX, cy=CY)
    # aligned should match the geometry world normals it was rotated from
    dots = np.sum(aligned[av] * geom[av], axis=-1)
    assert dots.mean() > 0.99                                # recovered to <8deg on average
    assert (dots > 0.9).mean() > 0.95


def test_align_survives_a_hemisphere_flip():
    depth = _varied_depth()
    vm = _view_matrix()
    geom, _ = world_normals_from_depth(depth, view_matrix=vm, fx=FX, fy=FY, cx=CX, cy=CY)
    r_true = _rot_xyz(-0.3, 0.5, 0.2)
    predicted = -(geom @ r_true.T)                           # rotated AND flipped inward
    aligned, av = align_predicted_normals_to_world(predicted, depth, view_matrix=vm, fx=FX, fy=FY, cx=CX, cy=CY)
    assert np.sum(aligned[av] * geom[av], axis=-1).mean() > 0.95


def test_encode_normal_map_round_trips():
    Image = pytest.importorskip("PIL.Image")
    import base64
    import io

    from atlas_camera.core.normals import encode_normal_map_b64
    n = np.zeros((4, 4, 3), dtype=np.float32); n[..., 1] = 1.0   # all up
    n[0, 0] = [1.0, 0.0, 0.0]
    uri = encode_normal_map_b64(n)
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    px = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
    dec = px * 2.0 - 1.0
    assert dec[1, 1][1] == pytest.approx(1.0, abs=0.01)         # up decodes to +Y
    assert dec[0, 0][0] == pytest.approx(1.0, abs=0.01)         # +X decodes to +X
