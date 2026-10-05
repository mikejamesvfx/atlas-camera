"""Virtual object camera: pixel-exact re-rendering for centred-pinhole models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas_camera.core.object_crop import (
    ObjectCropCamera,
    apply_homography,
    crop_sample_grid,
    pixal_project,
    virtual_object_camera,
    warp_to_crop,
)

W, H = 1600, 1000
FX = FY = 1100.0
CX, CY = W / 2.0, H / 2.0


def _view_matrix(cam_pos, pitch_deg=-8.0, yaw_deg=0.0):
    """World->camera 4x4 for a camera at ``cam_pos`` (Y-up, -Z forward)."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    c2w = np.eye(4)
    c2w[:3, :3] = ry @ rx
    c2w[:3, 3] = cam_pos
    return np.linalg.inv(c2w)


def box_mesh(center, size):
    cx, cy, cz = center
    sx, sy, sz = (s / 2.0 for s in size)
    v = np.array([[cx + dx * sx, cy + dy * sy, cz + dz * sz]
                  for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)], dtype=np.float64)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
                 dtype=np.int64)
    return v, f


def project(view, pts):
    cam = pts @ view[:3, :3].T + view[:3, 3]
    w = -cam[:, 2]
    return CX + FX * cam[:, 0] / w, CY - FY * cam[:, 1] / w, cam


def _mask_for(view, verts, faces):
    from atlas_camera.core.move_budget import rasterize_coverage
    alpha, _ = rasterize_coverage(verts, faces, view_matrix=view, fx=FX, fy=FY,
                                  cx=CX, cy=CY, width=W, height=H, backend="numpy")
    return alpha


@pytest.fixture()
def scene():
    view = _view_matrix([0.0, 1.6, 0.0])
    # ~30 degrees off-axis to the right and a little low: an edge-of-frame object.
    verts, faces = box_mesh((3.4, 0.6, -6.0), (1.2, 1.2, 1.0))
    mask = _mask_for(view, verts, faces)
    assert mask.sum() > 500
    crop = virtual_object_camera(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                                 image_width=W, image_height=H, mask=mask)
    return view, verts, faces, mask, crop


def test_crop_matches_pixal_projection_off_axis(scene):
    view, verts, _, _, crop = scene
    su, sv, cam = project(view, verts)
    # Source pixel -> crop pixel through the inverse homography ...
    ju, iv, w = apply_homography(crop.inverse_homography, su, sv)
    assert (w > 0).all()
    # ... must equal Pixal3D's own projection of the same points in the
    # virtual camera frame, half-pixel convention included.
    p_v = cam @ crop.rotation.T
    xp, yp = pixal_project(p_v, fov_deg=crop.fov_deg, resolution=crop.size)
    assert np.max(np.abs(ju - xp)) < 0.01
    assert np.max(np.abs(iv - yp)) < 0.01


def test_principal_point_is_crop_centre_half_pixel_pinned():
    # A point on the virtual optical axis lands at index (R-1)/2 = 511.5 for
    # R = 1024, i.e. continuous R/2 in Pixal3D's [j, j+1) convention.
    xp, yp = pixal_project(np.array([[0.0, 0.0, -3.0]]), fov_deg=40.0, resolution=1024)
    assert xp[0] == pytest.approx(511.5)
    assert yp[0] == pytest.approx(511.5)


def test_object_fits_and_is_centred(scene):
    view, verts, _, mask, crop = scene
    ys, xs = np.nonzero(mask)
    ju, iv, _ = apply_homography(crop.inverse_homography, xs.astype(float), ys.astype(float))
    c = crop.principal_px
    # Fits inside the frame with the pad's margin ...
    assert ju.min() > 0 and iv.min() > 0
    assert ju.max() < crop.size - 1 and iv.max() < crop.size - 1
    # ... and the extent is centred on the axis (refined re-aim).
    assert abs((ju.min() + ju.max()) / 2 - c) < 0.02 * crop.size
    assert abs((iv.min() + iv.max()) / 2 - c) < 0.02 * crop.size
    # Larger extent fills ~1/pad of the frame.
    span = max(ju.max() - ju.min(), iv.max() - iv.min())
    assert span / crop.size == pytest.approx(1.0 / crop.pad, rel=0.03)


def test_fov_matches_focal(scene):
    *_, crop = scene
    f_from_fov = crop.size / (2.0 * math.tan(math.radians(crop.fov_deg) / 2.0))
    assert f_from_fov == pytest.approx(crop.focal_px, rel=1e-9)


def test_gravity_roll_keeps_world_verticals_vertical(scene):
    # A pitched pinhole keystones verticals; gravity roll guarantees the one
    # through the crop's optical axis is exactly vertical (and the rest lean
    # symmetrically about it).
    view, *_ , crop = scene
    c2w = np.linalg.inv(view)
    axis_world = c2w[:3, :3] @ (crop.rotation.T @ np.array([0.0, 0.0, -1.0]))
    p0 = c2w[:3, 3] + 6.0 * axis_world
    pts = np.array([p0, p0 + np.array([0.0, 1.5, 0.0])])
    su, sv, _ = project(view, pts)
    ju, iv, _ = apply_homography(crop.inverse_homography, su, sv)
    assert abs(ju[0] - ju[1]) < 1e-6
    assert iv[1] < iv[0]  # up in world is up in the crop


def test_source_camera_roll_differs_on_rolled_view():
    view = _view_matrix([0.0, 1.6, 0.0], pitch_deg=-35.0)
    verts, faces = box_mesh((3.0, 0.5, -4.0), (1.0, 1.0, 1.0))
    mask = _mask_for(view, verts, faces)
    g = virtual_object_camera(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                              image_width=W, image_height=H, mask=mask, roll="gravity")
    s = virtual_object_camera(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                              image_width=W, image_height=H, mask=mask,
                              roll="source_camera")
    assert not np.allclose(g.rotation, s.rotation, atol=1e-3)


def test_homographies_are_inverse(scene):
    *_, crop = scene
    j = np.array([0.0, 100.0, 511.5, 1023.0])
    i = np.array([5.0, 900.0, 511.5, 0.0])
    sx, sy, _ = apply_homography(crop.homography, j, i)
    jb, ib, _ = apply_homography(crop.inverse_homography, sx, sy)
    assert np.allclose(jb, j, atol=1e-6) and np.allclose(ib, i, atol=1e-6)


def test_warp_reproduces_source_pixels(scene):
    view, verts, faces, mask, crop = scene
    yy, xx = np.mgrid[0:H, 0:W]
    ramp = np.stack([xx / W, yy / H, np.zeros_like(xx, dtype=float)], axis=-1)
    out = warp_to_crop(crop, ramp)
    sx, sy, valid = crop_sample_grid(crop)
    inside = valid & (sx > 1) & (sx < W - 2) & (sy > 1) & (sy < H - 2)
    assert np.allclose(out[inside, 0], sx[inside] / W, atol=1e-6)
    assert np.allclose(out[inside, 1], sy[inside] / H, atol=1e-6)


def test_handle_round_trips(scene):
    *_, crop = scene
    back = ObjectCropCamera.from_dict(crop.to_dict())
    assert np.allclose(back.rotation, crop.rotation)
    assert back.fov_deg == crop.fov_deg and back.size == crop.size


def test_empty_mask_raises():
    view = _view_matrix([0.0, 1.6, 0.0])
    with pytest.raises(ValueError, match="empty"):
        virtual_object_camera(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                              image_width=W, image_height=H, mask=np.zeros((H, W)))


def test_object_wider_than_a_hemisphere_raises():
    # A U-shaped matte along three frame edges of a very wide lens: no single
    # forward direction sees every ray in front of it.
    m = np.zeros((H, W))
    m[:, :3] = 1
    m[:, -3:] = 1
    m[:3, :] = 1
    with pytest.raises(ValueError, match="hemisphere"):
        virtual_object_camera(view_matrix=np.eye(4), fx=50.0, fy=50.0, cx=CX, cy=CY,
                              image_width=W, image_height=H, mask=m)


def test_crop_past_the_frame_edge_is_invalid_and_warps_to_fill():
    view = _view_matrix([0.0, 1.6, 0.0])
    m = np.zeros((H, W))
    m[400:600, W - 40:] = 1                    # object cut by the right frame edge
    crop = virtual_object_camera(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                                 image_width=W, image_height=H, mask=m, size=128)
    sx, sy, valid = crop_sample_grid(crop)
    assert (~valid).any() and valid.any()
    assert (sx[~valid] > W - 0.5).any()         # the invalid part is past the right edge
    matte = warp_to_crop(crop, m, fill=0.0)
    assert float(matte[~valid].max()) == 0.0
    assert float(matte[valid].max()) > 0.5


def test_rejects_3x3_view():
    with pytest.raises(ValueError, match="4x4"):
        virtual_object_camera(view_matrix=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY,
                              image_width=W, image_height=H, mask=np.ones((H, W)))
