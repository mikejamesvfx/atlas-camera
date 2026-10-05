"""core.uv_bake: UV-space rasterisation and the photo bake (numpy only, runs on CI)."""

from __future__ import annotations

import numpy as np

from atlas_camera.core.uv_bake import (
    bake_photo_into_uv,
    rasterize_uv,
    sample_texture_at_uvs,
)

QUAD_F = np.array([[0, 1, 2], [0, 2, 3]])
QUAD_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])   # top-left convention


def test_rasterize_uv_covers_a_full_quad_with_valid_barycentrics():
    fi, bary = rasterize_uv(QUAD_F, QUAD_UV, 32)
    assert fi.shape == (32, 32) and (fi >= 0).all()
    assert np.allclose(bary.sum(-1), 1.0, atol=1e-5) and (bary >= -1e-6).all()


def test_rasterize_uv_leaves_the_gutter_empty_and_interpolates_position():
    uv = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    fi, bary = rasterize_uv(QUAD_F, uv, 40)
    assert fi[0, 0] == -1 and fi[20, 20] >= 0
    covered = (fi >= 0).mean()
    assert 0.2 < covered < 0.3
    # texel centre u = (col + .5)/W maps back to the same uv through the bary
    rows, cols = np.nonzero(fi >= 0)
    tri = QUAD_F[fi[rows, cols]]
    rec = (bary[rows, cols][:, :, None] * uv[tri]).sum(1)
    assert np.allclose(rec[:, 0], (cols + 0.5) / 40, atol=1e-6)
    assert np.allclose(rec[:, 1], (rows + 0.5) / 40, atol=1e-6)


def _facing_quad(z=-4.0, half=1.0):
    """A world quad facing a camera at the origin looking down -Z."""
    return np.array([[-half, half, z], [half, half, z], [half, -half, z], [-half, -half, z]])


CAM = dict(view_matrix=np.eye(4), fx=50.0, fy=50.0, cx=32.0, cy=32.0)


def test_bake_puts_the_photo_where_the_camera_saw_and_keeps_the_texture_elsewhere():
    gen = np.zeros((16, 16, 3), np.float32)
    gen[..., 2] = 1.0                                   # blue model texture
    photo = np.zeros((64, 64, 3), np.float32)
    photo[..., 0] = 1.0                                 # red photo
    v = _facing_quad()
    seen, st = bake_photo_into_uv(v, QUAD_F, QUAD_UV, gen, photo, np.ones(4), **CAM)
    assert st["photo_fraction"] > 0.95
    assert seen[..., 0].mean() > 0.95 and seen[..., 2].mean() < 0.05
    hidden, _ = bake_photo_into_uv(v, QUAD_F, QUAD_UV, gen, photo, np.zeros(4), **CAM)
    assert np.allclose(hidden, gen)
    behind, st_b = bake_photo_into_uv(_facing_quad(z=4.0), QUAD_F, QUAD_UV, gen, photo,
                                      np.ones(4), **CAM)
    assert np.allclose(behind, gen) and st_b["photo_texels"] == 0


def test_sample_texture_at_uvs_reads_the_texel_under_each_vertex():
    tex = np.zeros((8, 8, 3), np.float32)
    tex[:4, :4] = [1, 0, 0]          # top-left quadrant red
    tex[4:, 4:] = [0, 1, 0]          # bottom-right green
    got = sample_texture_at_uvs(tex, [[0.1, 0.1], [0.9, 0.9]])
    assert np.allclose(got, [[1, 0, 0], [0, 1, 0]], atol=1e-6)


