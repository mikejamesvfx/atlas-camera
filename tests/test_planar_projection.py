"""core.planar_projection — homography exactness, round-trip identity, and the
behind-camera/grazing alpha contract, plus the node-level plane_name fail-soft.
"""

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.planar_projection import (
    build_warp_spec,
    fit_visible_rect,
    ground_plane_basis,
    homography_plane_to_image,
    plane_basis_from_primitive,
    rewarp_into_plate,
    unwarp_plate,
    warp_by_homography,
)

W, H = 320, 240
FX = FY = 300.0
CX, CY = W / 2, H / 2


def _view(eye=(0.0, 2.0, 6.0), target=(0.0, 0.0, 0.0)):
    view, _world, _r3 = look_at_view_matrix(eye, target)
    return view


def _project_reference(view, pt):
    """Independent reference projection (headless_evidence convention)."""
    v = np.asarray(view, float)
    cam = v @ np.array([*pt, 1.0])
    w = -cam[2]
    return np.array([CX + FX * cam[0] / w, CY - FY * cam[1] / w]), w


def test_homography_matches_reference_projection():
    basis = ground_plane_basis()
    h = homography_plane_to_image(_view(), FX, FY, CX, CY, basis)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.uniform(-3, 3, 2)
        # plane point: c + a*u + b*v  (ground: u=+Z, v=+X)
        world = np.array([b, 0.0, a])
        expected, w_ref = _project_reference(_view(), world)
        vec = h @ np.array([a, b, 1.0])
        assert vec[2] == pytest.approx(w_ref, rel=1e-9)
        got = vec[:2] / vec[2]
        assert got == pytest.approx(expected, abs=1e-9)


def test_behind_camera_w_is_negative():
    basis = ground_plane_basis()
    h = homography_plane_to_image(_view(), FX, FY, CX, CY, basis)
    # A ground point far BEHIND the eye (eye z=6, looking toward -z): z=+50.
    vec = h @ np.array([50.0, 0.0, 1.0])   # a=50 -> world z=+50
    assert vec[2] < 0


def test_visible_rect_is_in_front_and_finite():
    rect = fit_visible_rect(_view(), FX, FY, CX, CY, ground_plane_basis(), W, H)
    assert rect is not None
    u_min, v_min, u_max, v_max = rect
    assert u_max > u_min and v_max > v_min
    assert all(np.isfinite(rect))
    # The camera looks toward -z from z=6: visible ground sits at z < 6,
    # i.e. plane-u (== world z) below the eye's z.
    assert u_max < 6.0


def test_unwarp_rewarp_round_trip_identity():
    """Rewarping the UNEDITED flat must reproduce the plate on the covered
    region within bilinear tolerance, and leave uncovered pixels bit-exact."""
    rng = np.random.default_rng(3)
    # Smooth plate (bilinear round-trip on noise would legitimately differ).
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    plate = np.stack([
        0.5 + 0.4 * np.sin(xx / 37.0) * np.cos(yy / 29.0),
        0.5 + 0.4 * np.cos(xx / 51.0),
        0.5 + 0.4 * np.sin(yy / 43.0),
    ], axis=-1).astype(np.float32)

    spec = build_warp_spec(_view(), FX, FY, CX, CY, ground_plane_basis(), W, H,
                           max_resolution=1024)
    assert spec is not None
    flat, alpha = unwarp_plate(plate, spec)
    assert flat.shape == (spec.flat_height, spec.flat_width, 3)
    assert 0.1 < float(alpha.mean()) <= 1.0

    out, cov = rewarp_into_plate(flat, plate, spec, feather_px=0)
    untouched = cov < 1e-6
    assert np.array_equal(out[untouched], plate[untouched])
    solid = cov > 0.999
    assert solid.any()
    err = np.abs(out[solid] - plate[solid])
    assert float(err.mean()) < 0.02 and float(np.percentile(err, 95)) < 0.06


def test_edit_actually_lands_in_plate():
    plate = np.zeros((H, W, 3), dtype=np.float32)
    spec = build_warp_spec(_view(), FX, FY, CX, CY, ground_plane_basis(), W, H,
                           max_resolution=512)
    flat, alpha = unwarp_plate(plate, spec)
    edited = flat.copy()
    edited[:] = 1.0  # paint the whole plane white
    out, cov = rewarp_into_plate(edited, plate, spec, feather_px=0)
    assert float(out.max()) > 0.9          # the edit arrived
    assert float(out[cov < 1e-6].max() if (cov < 1e-6).any() else 0.0) == 0.0


def test_warp_by_homography_identity():
    img = np.random.default_rng(1).random((32, 48, 3)).astype(np.float32)
    out, alpha = warp_by_homography(img, np.eye(3), 48, 32)
    assert np.allclose(out, img, atol=1e-6)
    assert np.allclose(alpha, 1.0)


def test_plane_basis_from_primitive_columns():
    from atlas_camera.core.schema import AtlasProxyPrimitive
    t = ((0.0, 0.0, 1.0, 5.0),
         (1.0, 0.0, 0.0, 6.0),
         (0.0, 1.0, 0.0, 7.0),
         (0.0, 0.0, 0.0, 1.0))
    prim = AtlasProxyPrimitive(name="wall_a", primitive_type="plane",
                               transform_matrix=t)
    basis = plane_basis_from_primitive(prim)
    assert basis.u == (0.0, 1.0, 0.0)
    assert basis.v == (0.0, 0.0, 1.0)
    assert basis.n == (1.0, 0.0, 0.0)
    assert basis.c == (5.0, 6.0, 7.0)
    assert basis.name == "wall_a"


def test_unwarp_node_fail_soft_on_unknown_plane_name():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_planar import AtlasPlanarUnwarp
    from atlas_camera.core.schema import (
        AtlasExtrinsics, AtlasIntrinsics, AtlasProjectionScene,
        AtlasProxyPrimitive, AtlasSolve, LatentCamera,
    )
    view, world, rot3 = look_at_view_matrix((0.0, 2.0, 6.0), (0.0, 0.0, 0.0))
    extr = AtlasExtrinsics(camera_position=(0.0, 2.0, 6.0),
                           camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=W, image_height=H, fx_px=FX, fy_px=FY,
                           cx_px=CX, cy_px=CY)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    solve.projection_scene = AtlasProjectionScene(proxy_geometry=[
        AtlasProxyPrimitive(name="facade_a", primitive_type="plane")])
    img = torch.rand(1, H, W, 3)
    flat, mask, spec, report = AtlasPlanarUnwarp().unwarp(
        solve, img, plane_name="nope", max_resolution=512)
    assert spec is not None                      # fell back to ground and worked
    assert "not found" in report and "facade_a" in report
