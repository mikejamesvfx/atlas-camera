"""core.projection_render — stereo eye construction, shifted-sensor
convergence, analytic disparity on a one-plane scene, and the z-buffered
rasterizer's basic contracts.
"""

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.projection_render import (
    converged_cx,
    gather_scene_meshes,
    project_points,
    render_scene,
    stereo_eye_view_matrices,
)

FX = FY = 400.0
W, H = 320, 240
CX, CY = W / 2, H / 2
IO = 0.065


def _view(eye=(0.0, 1.6, 0.0), target=(0.0, 1.6, -10.0)):
    view, _w, _r = look_at_view_matrix(eye, target)
    return np.asarray(view, dtype=np.float64)


def test_eyes_offset_exactly_along_camera_right():
    view = _view()
    left, right = stereo_eye_view_matrices(view, IO)
    eye_l = np.linalg.inv(left)[:3, 3]
    eye_r = np.linalg.inv(right)[:3, 3]
    d = eye_r - eye_l
    assert np.linalg.norm(d) == pytest.approx(IO, rel=1e-12)
    cam_right = np.linalg.inv(view)[:3, 0]
    assert np.dot(d, cam_right) / np.linalg.norm(d) == pytest.approx(1.0, abs=1e-12)
    # Orientation untouched (no toe-in): rotation blocks identical.
    assert np.allclose(np.linalg.inv(left)[:3, :3], np.linalg.inv(view)[:3, :3])
    assert np.allclose(np.linalg.inv(right)[:3, :3], np.linalg.inv(view)[:3, :3])


def test_converged_cx_parallel_and_offset():
    assert converged_cx(FX, CX, IO, 0.0) == (CX, CX)
    cxl, cxr = converged_cx(FX, CX, IO, 5.0)
    off = FX * (IO / 2) / 5.0
    assert cxl == pytest.approx(CX - off) and cxr == pytest.approx(CX + off)


def test_analytic_disparity_on_point():
    """Screen-space disparity of a point at depth z: d = fx*io/z - 2*cx_off."""
    view = _view()
    left, right = stereo_eye_view_matrices(view, IO)
    for z, conv in [(5.0, 0.0), (5.0, 5.0), (12.0, 6.0)]:
        pt = np.array([[0.3, 1.6, -z]])
        cxl, cxr = converged_cx(FX, CX, IO, conv)
        pl, _ = project_points(pt, left, FX, FY, cxl, CY)
        pr, _ = project_points(pt, right, FX, FY, cxr, CY)
        disparity = float(pl[0, 0] - pr[0, 0])
        expected = FX * IO / z - 2.0 * (FX * (IO / 2) / conv if conv > 0 else 0.0)
        assert disparity == pytest.approx(expected, abs=1e-9)
    # At the convergence distance the disparity is exactly zero.
    pt = np.array([[0.0, 1.6, -5.0]])
    cxl, cxr = converged_cx(FX, CX, IO, 5.0)
    pl, _ = project_points(pt, stereo_eye_view_matrices(_view(), IO)[0], FX, FY, cxl, CY)
    pr, _ = project_points(pt, stereo_eye_view_matrices(_view(), IO)[1], FX, FY, cxr, CY)
    assert float(pl[0, 0] - pr[0, 0]) == pytest.approx(0.0, abs=1e-9)


def _quad_mesh(z=-5.0, half=4.0, uv=True):
    """A fronto-parallel textured quad at depth z as gather-format tuple."""
    verts = np.array([
        [-half, 1.6 - half, z], [half, 1.6 - half, z],
        [half, 1.6 + half, z], [-half, 1.6 + half, z]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) if uv else None
    return ("quad", verts, faces, uvs, "primary", {})


def test_render_scene_paints_quad_with_texture():
    tex = np.zeros((64, 64, 3))
    tex[:, :32] = (1.0, 0.0, 0.0)   # left half red
    tex[:, 32:] = (0.0, 1.0, 0.0)   # right half green
    rgb, alpha, stats = render_scene(
        [_quad_mesh()], {"primary": tex}, _view(), FX, FY, CX, CY, W, H)
    assert stats["meshes_rendered"] == 1
    assert float(alpha.mean()) > 0.5           # quad fills most of the frame
    covered = alpha > 0
    # Texture arrived, with left/right halves on the correct sides (uv origin
    # bottom-left; +u maps to +world-x here which projects to +image-x).
    assert rgb[covered].max() > 0.9
    left_cols = rgb[:, : W // 4][alpha[:, : W // 4] > 0]
    right_cols = rgb[:, -W // 4:][alpha[:, -W // 4:] > 0]
    assert left_cols[:, 0].mean() > 0.9 and left_cols[:, 1].mean() < 0.1
    assert right_cols[:, 1].mean() > 0.9 and right_cols[:, 0].mean() < 0.1


def test_render_scene_zbuffer_near_wins():
    tex_near = np.full((8, 8, 3), (1.0, 0.0, 0.0))
    tex_far = np.full((8, 8, 3), (0.0, 0.0, 1.0))
    near = ("near",) + _quad_mesh(z=-3.0, half=1.0)[1:4] + ("near_tex", {})
    far = ("far",) + _quad_mesh(z=-8.0, half=6.0)[1:4] + ("far_tex", {})
    rgb, alpha, _ = render_scene(
        [far, near], {"near_tex": tex_near, "far_tex": tex_far},
        _view(), FX, FY, CX, CY, W, H)
    centre = rgb[H // 2, W // 2]
    assert centre[0] > 0.9 and centre[2] < 0.1     # near (red) wins at centre
    corner_region = rgb[alpha > 0]
    assert (corner_region[:, 2] > 0.9).any()       # far (blue) visible elsewhere


def test_render_scene_behind_camera_skipped_and_holes_stay_zero():
    behind = _quad_mesh(z=+5.0)   # entirely behind the eye
    rgb, alpha, stats = render_scene(
        [behind], {"primary": np.ones((8, 8, 3))}, _view(), FX, FY, CX, CY, W, H)
    assert float(alpha.sum()) == 0.0
    assert float(rgb.sum()) == 0.0


def test_gather_scene_meshes_uv_variant_and_wrapper_parity():
    from atlas_camera.core.schema import (
        AtlasProjectionScene, AtlasProxyPrimitive, AtlasSolve, LatentCamera,
        AtlasIntrinsics,
    )
    verts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    prim = AtlasProxyPrimitive(name="m", primitive_type="mesh", metadata={
        "vertices": verts, "faces": [0, 1, 2],
        "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]})
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=8, image_height=8)))
    solve.projection_scene = AtlasProjectionScene(proxy_geometry=[prim])

    plain = gather_scene_meshes(solve)
    assert len(plain) == 1 and len(plain[0]) == 4          # legacy 4-tuple shape
    rich = gather_scene_meshes(solve, with_uvs=True)
    label, v, f, uvs, tex_label, _meta = rich[0]
    assert label == "m" and tex_label == "primary"
    assert uvs.shape == (3, 2)

    # The headless_evidence wrapper stays behavior-identical.
    from atlas_camera.comfy.headless_evidence import _mesh_arrays
    legacy = _mesh_arrays(solve)
    assert len(legacy) == 1 and legacy[0][0] == "m"
    assert np.array_equal(legacy[0][1], v) and np.array_equal(legacy[0][2], f)


def test_gather_scene_meshes_honours_a_per_primitive_texture_override():
    """A fill-derived patch mesh lives in the PRIMARY scene but carries its
    own generated texture. Without metadata["texture"] winning over the
    container label, the patch silently rendered with the primary plate
    (found live by the G4 driver, 2026-08-14)."""
    from atlas_camera.core.schema import (
        AtlasProjectionScene, AtlasProxyPrimitive, AtlasSolve, LatentCamera,
        AtlasIntrinsics,
    )
    verts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    base = {"vertices": verts, "faces": [0, 1, 2],
            "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]}
    plain = AtlasProxyPrimitive(name="scene", primitive_type="mesh",
                                metadata=dict(base))
    patch = AtlasProxyPrimitive(name="patch", primitive_type="mesh",
                                metadata={**base, "texture": "fill_patch"})
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=8, image_height=8)))
    solve.projection_scene = AtlasProjectionScene(
        proxy_geometry=[plain, patch])
    labels = {m[0]: m[4] for m in gather_scene_meshes(solve, with_uvs=True)}
    assert labels["scene"] == "primary"        # container label untouched
    assert labels["patch"] == "fill_patch"     # override wins
    # and the override renders with ITS texture, not the primary's
    red = np.zeros((8, 8, 3)); red[..., 0] = 1.0
    green = np.zeros((8, 8, 3)); green[..., 1] = 1.0
    meshes = [m for m in gather_scene_meshes(solve, with_uvs=True)
              if m[0] == "patch"]
    quad = _quad_mesh(z=-5.0)
    patch_mesh = ("patch", quad[1], quad[2], quad[3], "fill_patch", {})
    rgb, alpha, _ = render_scene([patch_mesh],
                                 {"primary": red, "fill_patch": green},
                                 _view(), FX, FY, CX, CY, W, H)
    lit = alpha > 0
    assert lit.any()
    assert float(rgb[lit][:, 1].mean()) > 0.9  # green (its own texture)
    assert float(rgb[lit][:, 0].mean()) < 0.1  # not the primary red
