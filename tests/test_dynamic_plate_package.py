"""DynamicPlate artifact package writer (spec §24/§25)."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.dynamic_plate import (
    DynamicPlate,
    build_receiver_plane,
    crop_intrinsics_for_plate,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, LatentCamera
from atlas_camera.exporters.dynamic_plate_package import (
    build_dynamic_plate_package,
    load_dynamic_plate,
)


def _camera(width=640, height=360):
    view, world, rot3 = look_at_view_matrix((0.0, 10.0, 0.0), (0.0, 0.0, -40.0))
    return LatentCamera(
        intrinsics=build_intrinsics(image_width=width, image_height=height,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 10.0, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view))


@pytest.fixture
def source_and_matte(tmp_path):
    img_path = tmp_path / "castle.png"
    Image.fromarray(
        (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
    ).save(img_path)
    matte = np.zeros((360, 640), dtype=np.float32)
    matte[240:360, :] = 1.0
    return img_path, matte


def _plate(cam, roi):
    return DynamicPlate(
        plate_id="WATER_0001", semantic_type="water",
        source_image="castle.png", source_width=640, source_height=360,
        matte_bbox=RegionROI(x=0, y=240, width=640, height=120),
        source_roi=roi,
        crop_transform=CropTransform(source_width=640, source_height=360,
                                     roi=roi, output_width=roi.width,
                                     output_height=roi.height),
        source_camera=cam,
        crop_camera=crop_intrinsics_for_plate(cam, roi),
        receiver=build_receiver_plane(cam, roi),
        frame_rate=24.0, frame_start=0, frame_end=95)


def test_package_layout(tmp_path, source_and_matte):
    img_path, matte = source_and_matte
    cam = _camera()
    roi = RegionROI(x=0, y=228, width=640, height=132)
    result = build_dynamic_plate_package(
        _plate(cam, roi), tmp_path / "dynamic", source_image_path=img_path,
        matte=matte)
    pkg = result.package_dir
    assert pkg.name == "WATER_0001"
    for rel in ("manifest.json", "source/crop.png", "source/matte.png",
                "source/context.png", "camera/source_camera.json",
                "camera/crop_camera.json", "geometry/receiver.obj"):
        assert (pkg / rel).exists(), rel
    assert (pkg / "generated").is_dir()
    assert (pkg / "preview").is_dir()
    # crop.png matches the ROI
    with Image.open(pkg / "source/crop.png") as im:
        assert im.size == (640, 132)
    # matte crop matches too
    with Image.open(pkg / "source/matte.png") as im:
        assert im.size == (640, 132)
    assert set(result.files) >= {"manifest", "crop", "matte", "context",
                                 "source_camera", "crop_camera", "receiver"}


def test_manifest_round_trip(tmp_path, source_and_matte):
    img_path, matte = source_and_matte
    cam = _camera()
    roi = RegionROI(x=0, y=228, width=640, height=132)
    result = build_dynamic_plate_package(
        _plate(cam, roi), tmp_path / "dynamic", source_image_path=img_path,
        matte=matte)
    manifest = json.loads((result.package_dir / "manifest.json").read_text())
    assert manifest["plate_id"] == "WATER_0001"
    assert manifest["semantic_type"] == "water"
    assert "created_at" in manifest and "atlas_version" in manifest
    again = load_dynamic_plate(result.package_dir)
    assert again.plate_id == "WATER_0001"
    assert again.source_roi == roi
    assert again.crop_camera.intrinsics.image_height == 132
    assert again.receiver.path == "geometry/receiver.obj"


def test_package_without_matte(tmp_path, source_and_matte):
    img_path, _ = source_and_matte
    cam = _camera()
    roi = RegionROI(x=0, y=228, width=640, height=132)
    result = build_dynamic_plate_package(
        _plate(cam, roi), tmp_path / "dynamic", source_image_path=img_path)
    assert not (result.package_dir / "source/matte.png").exists()
    assert (result.package_dir / "source/crop.png").exists()
