"""Receiver-plane builder + crop-camera registration gate (release-blocking).

Spec §35: full-image pixel -> crop pixel -> crop-camera ray -> receiver
intersection must land on the same world point as the original full camera's
ray. Verified analytically here; a visually plausible but misregistered crop
is not acceptable.
"""
from __future__ import annotations

import math

import pytest

from atlas_camera.core.camera_crop import (
    CropTransform,
    RegionROI,
    crop_intrinsics,
    scale_intrinsics,
)
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.dynamic_plate import (
    ReceiverGeometry,
    build_receiver_plane,
    pixel_ray_world,
    write_plane_obj,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, LatentCamera


def _camera(eye=(0.0, 10.0, 0.0), target=(0.0, 0.0, -40.0),
            width=1920, height=1080, focal_mm=32.0) -> LatentCamera:
    view, world, rot3 = look_at_view_matrix(eye, target)
    intr = build_intrinsics(image_width=width, image_height=height,
                            focal_length_mm=focal_mm)
    extr = AtlasExtrinsics(camera_position=tuple(float(v) for v in eye),
                           camera_rotation_matrix=rot3,
                           camera_world_matrix=world,
                           camera_view_matrix=view)
    return LatentCamera(intrinsics=intr, extrinsics=extr)


def _hit_y_plane(origin, direction, plane_y=0.0):
    dy = direction[1]
    assert abs(dy) > 1e-9
    t = (plane_y - origin[1]) / dy
    assert t > 0
    return tuple(origin[i] + t * direction[i] for i in range(3))


# ------------------------------------------------------------ pixel_ray_world

def test_center_pixel_ray_points_at_target():
    cam = _camera()
    intr = cam.intrinsics
    origin, direction = pixel_ray_world(cam, intr.cx_px, intr.cy_px)
    assert origin == pytest.approx((0.0, 10.0, 0.0))
    # forward toward target: normalized (0,-10,-40)
    n = math.sqrt(10.0 ** 2 + 40.0 ** 2)
    assert direction[0] == pytest.approx(0.0, abs=1e-9)
    assert direction[1] == pytest.approx(-10.0 / n)
    assert direction[2] == pytest.approx(-40.0 / n)


def test_center_ray_ground_hit_distance_analytic():
    cam = _camera()
    intr = cam.intrinsics
    origin, direction = pixel_ray_world(cam, intr.cx_px, intr.cy_px)
    hit = _hit_y_plane(origin, direction, 0.0)
    # By similar triangles the forward ray from (0,10,0) toward (0,0,-40)
    # crosses y=0 exactly at the target.
    assert hit == pytest.approx((0.0, 0.0, -40.0), abs=1e-9)


# ------------------------------------------------- registration gate (§35)

def test_crop_camera_registration_gate():
    cam = _camera()
    intr = cam.intrinsics
    roi = RegionROI(x=192, y=560, width=1536, height=520)
    cropped_intr = crop_intrinsics(intr, roi)
    crop_cam = LatentCamera(intrinsics=cropped_intr, extrinsics=cam.extrinsics)
    ct = CropTransform(source_width=1920, source_height=1080, roi=roi,
                       output_width=roi.width, output_height=roi.height)
    for px, py in [(200.0, 600.0), (960.0, 800.0), (1500.5, 1000.25),
                   (192.0, 560.0)]:
        o_full, d_full = pixel_ray_world(cam, px, py)
        expected = _hit_y_plane(o_full, d_full, 0.0)
        cx, cy = ct.full_to_crop(px, py)
        o_crop, d_crop = pixel_ray_world(crop_cam, cx, cy)
        got = _hit_y_plane(o_crop, d_crop, 0.0)
        assert got == pytest.approx(expected, abs=1e-6)


def test_crop_resize_camera_registration_gate():
    cam = _camera()
    intr = cam.intrinsics
    roi = RegionROI(x=0, y=540, width=1920, height=540)
    resized_intr = scale_intrinsics(crop_intrinsics(intr, roi), 960, 270)
    small_cam = LatentCamera(intrinsics=resized_intr, extrinsics=cam.extrinsics)
    ct = CropTransform(source_width=1920, source_height=1080, roi=roi,
                       output_width=960, output_height=270)
    for px, py in [(100.0, 700.0), (1800.0, 1000.0)]:
        o_full, d_full = pixel_ray_world(cam, px, py)
        expected = _hit_y_plane(o_full, d_full, 0.0)
        u, v = ct.full_to_crop(px, py)
        o_s, d_s = pixel_ray_world(small_cam, u, v)
        got = _hit_y_plane(o_s, d_s, 0.0)
        assert got == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------- build_receiver_plane

def test_receiver_plane_encloses_roi_hits():
    cam = _camera()
    # lower band of the image sees the ground plane
    roi = RegionROI(x=0, y=700, width=1920, height=380)
    rec = build_receiver_plane(cam, roi, plane_height=0.0)
    assert isinstance(rec, ReceiverGeometry)
    prim = rec.primitive
    assert prim is not None and prim.primitive_type == "plane"
    tf = prim.transform_matrix
    centre = (tf[0][3], tf[1][3], tf[2][3])
    assert centre[1] == pytest.approx(0.0)
    ex, ez, _ = prim.dimensions
    # every sampled ROI pixel's plane hit lies inside the plane extents
    for px, py in [(0.0, 700.0), (1920.0, 700.0), (960.0, 1080.0),
                   (0.0, 1080.0), (1920.0, 1080.0)]:
        o, d = pixel_ray_world(cam, px, py)
        if d[1] >= -1e-9:
            continue
        hx, _, hz = _hit_y_plane(o, d, 0.0)
        assert abs(hx - centre[0]) <= ex / 2 + 1e-6
        assert abs(hz - centre[2]) <= ez / 2 + 1e-6
    assert prim.metadata["role"] == "dynamic_plate_receiver"


def test_receiver_plane_clamps_horizon_rays():
    # ROI that includes sky pixels (rays that never hit the plane) must not
    # blow up: extents clamp at max_distance.
    cam = _camera(eye=(0.0, 5.0, 0.0), target=(0.0, 4.5, -40.0))
    roi = RegionROI(x=0, y=0, width=1920, height=1080)
    rec = build_receiver_plane(cam, roi, plane_height=0.0, max_distance=200.0)
    ex, ez, _ = rec.primitive.dimensions
    assert ex <= 2 * 200.0 * 1.1 + 1e-6
    assert ez <= 2 * 200.0 * 1.1 + 1e-6


def test_receiver_plane_camera_below_plane_raises():
    cam = _camera(eye=(0.0, -1.0, 0.0), target=(0.0, -1.5, -40.0))
    with pytest.raises(ValueError):
        build_receiver_plane(cam, RegionROI(x=0, y=540, width=1920, height=540),
                             plane_height=0.0)


# ------------------------------------------------------------- write_plane_obj

def test_write_plane_obj(tmp_path):
    cam = _camera()
    rec = build_receiver_plane(cam, RegionROI(x=0, y=700, width=1920, height=380))
    path = write_plane_obj(rec, tmp_path / "receiver.obj")
    text = path.read_text(encoding="utf-8")
    assert text.count("\nv ") + text.startswith("v ") == 4
    assert "vt " in text and "f " in text
    # all four corners on the plane
    ys = [float(line.split()[2]) for line in text.splitlines()
          if line.startswith("v ")]
    assert all(abs(y) < 1e-6 for y in ys)
