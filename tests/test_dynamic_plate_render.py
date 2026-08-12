"""v2v crop-sequence renderer: homography vs ray-chain ground truth."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.camera_spec import CameraSpec
from atlas_camera.core.dynamic_plate import (
    DynamicPlate,
    build_receiver_plane,
    crop_intrinsics_for_plate,
    pixel_ray_world,
)
from atlas_camera.core.dynamic_plate_render import (
    crop_sequence_homographies,
    dolly_view_matrices,
    render_crop_sequence,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, LatentCamera


def _camera(width=640, height=360):
    view, world, rot3 = look_at_view_matrix((0.0, 10.0, 0.0), (0.0, 0.0, -40.0))
    return LatentCamera(
        intrinsics=build_intrinsics(image_width=width, image_height=height,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 10.0, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view))


def _plate():
    cam = _camera()
    roi = RegionROI(x=0, y=200, width=640, height=160)
    return DynamicPlate(
        plate_id="WATER_0001", semantic_type="water",
        source_image="castle.png", source_width=640, source_height=360,
        source_roi=roi,
        crop_transform=CropTransform(source_width=640, source_height=360,
                                     roi=roi, output_width=roi.width,
                                     output_height=roi.height),
        source_camera=cam,
        crop_camera=crop_intrinsics_for_plate(cam, roi),
        receiver=build_receiver_plane(cam, roi))


def test_dolly_first_frame_is_source_pose():
    cam = _camera()
    views = dolly_view_matrices(cam, offset=(0.5, 0.0, -0.3), frame_count=5)
    assert len(views) == 5
    src = cam.extrinsics.camera_view_matrix
    assert np.allclose(views[0], src)
    assert not np.allclose(views[-1], src)


def test_zero_dolly_homography_is_identity():
    plate = _plate()
    views = dolly_view_matrices(plate.crop_camera, offset=(0.0, 0.0, 0.0),
                                frame_count=2)
    for m in crop_sequence_homographies(plate, views):
        m = np.asarray(m) / m[2][2]
        assert np.allclose(m, np.eye(3), atol=1e-9)


def test_homography_matches_ray_chain():
    """M @ render_pixel must equal project_through_crop(ray_hit(render_pixel))."""
    plate = _plate()
    offset = (0.4, 0.0, -0.6)
    views = dolly_view_matrices(plate.crop_camera, offset=offset, frame_count=3)
    homos = crop_sequence_homographies(plate, views)
    spec = CameraSpec.from_intrinsics(plate.crop_camera.intrinsics)
    crop_view = np.asarray(plate.crop_camera.extrinsics.camera_view_matrix)

    def crop_project(P):
        cam = crop_view[:3, :3] @ P + crop_view[:3, 3]
        w = -cam[2]
        return (spec.cx + spec.fx * cam[0] / w, spec.cy - spec.fy * cam[1] / w)

    # last frame: fully displaced camera
    view = np.asarray(views[-1])
    moved = LatentCamera(intrinsics=plate.crop_camera.intrinsics,
                         extrinsics=AtlasExtrinsics(
                             camera_view_matrix=tuple(map(tuple, view)),
                             camera_world_matrix=tuple(map(tuple, np.linalg.inv(view)))))
    m = np.asarray(homos[-1])
    for px, py in [(120.0, 80.0), (320.0, 100.0), (500.0, 140.0)]:
        origin, direction = pixel_ray_world(moved, px, py)
        t = -origin[1] / direction[1]
        P = np.asarray([origin[i] + t * direction[i] for i in range(3)])
        expected = crop_project(P)
        vec = m @ np.asarray([px, py, 1.0])
        got = (vec[0] / vec[2], vec[1] / vec[2])
        assert got == pytest.approx(expected, abs=1e-6)


def test_render_zero_dolly_reproduces_crop():
    plate = _plate()
    rng = np.random.default_rng(0)
    crop = (rng.random((160, 640, 3)) * 255).astype(np.uint8)
    views = dolly_view_matrices(plate.crop_camera, offset=(0.0, 0.0, 0.0),
                                frame_count=1)
    frames = render_crop_sequence(plate, views, crop)
    rgb, alpha = frames[0]
    inside = alpha > 0.5
    assert inside.mean() > 0.95
    diff = np.abs(rgb[inside].astype(np.float64) - crop[inside].astype(np.float64))
    assert float(diff.mean()) < 2.0


def test_render_dolly_moves_content():
    plate = _plate()
    rng = np.random.default_rng(1)
    crop = (rng.random((160, 640, 3)) * 255).astype(np.uint8)
    views = dolly_view_matrices(plate.crop_camera, offset=(1.0, 0.0, 0.0),
                                frame_count=3)
    frames = render_crop_sequence(plate, views, crop)
    first = frames[0][0].astype(np.float64)
    last = frames[-1][0].astype(np.float64)
    assert float(np.abs(first - last).mean()) > 1.0
