"""DynamicPlate validator + frame-sequence checks (spec §30/§31)."""
from __future__ import annotations

import pytest

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.dynamic_plate import (
    CAMERA_CROP_FAILURE,
    FRAME_SEQUENCE_INCOMPLETE,
    RECEIVER_GEOMETRY_UNAVAILABLE,
    REGION_INVALID,
    DynamicPlate,
    build_receiver_plane,
    crop_intrinsics_for_plate,
    frame_sequence_report,
    validate_dynamic_plate,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, LatentCamera


def _camera(width=1920, height=1080):
    view, world, rot3 = look_at_view_matrix((0.0, 10.0, 0.0), (0.0, 0.0, -40.0))
    return LatentCamera(
        intrinsics=build_intrinsics(image_width=width, image_height=height,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 10.0, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view))


def _valid_plate() -> DynamicPlate:
    cam = _camera()
    roi = RegionROI(x=0, y=700, width=1920, height=380)
    plate = DynamicPlate(
        plate_id="WATER_0001", semantic_type="water",
        source_image="castle.png", source_width=1920, source_height=1080,
        matte_bbox=roi, source_roi=roi,
        crop_transform=CropTransform(source_width=1920, source_height=1080,
                                     roi=roi, output_width=roi.width,
                                     output_height=roi.height),
        source_camera=cam,
        crop_camera=crop_intrinsics_for_plate(cam, roi),
        receiver=build_receiver_plane(cam, roi),
        frame_rate=24.0, frame_start=0, frame_end=95,
        generator="ltx", seed=3)
    return plate


def _codes(issues):
    return {i.code for i in issues}


def test_valid_plate_no_fails():
    issues = validate_dynamic_plate(_valid_plate())
    assert not [i for i in issues if i.severity == "fail"]


def test_roi_outside_image_flagged():
    plate = _valid_plate()
    plate.source_roi = RegionROI(x=1000, y=1000, width=2000, height=2000)
    assert REGION_INVALID in _codes(validate_dynamic_plate(plate))


def test_missing_crop_camera_flagged():
    plate = _valid_plate()
    plate.crop_camera = None
    assert CAMERA_CROP_FAILURE in _codes(validate_dynamic_plate(plate))


def test_nonfinite_crop_camera_flagged():
    plate = _valid_plate()
    plate.crop_camera.intrinsics.fx_px = float("nan")
    assert CAMERA_CROP_FAILURE in _codes(validate_dynamic_plate(plate))


def test_crop_camera_mismatched_roi_flagged():
    plate = _valid_plate()
    plate.crop_camera.intrinsics.image_width = 999
    assert CAMERA_CROP_FAILURE in _codes(validate_dynamic_plate(plate))


def test_missing_receiver_flagged():
    plate = _valid_plate()
    plate.receiver = None
    assert RECEIVER_GEOMETRY_UNAVAILABLE in _codes(validate_dynamic_plate(plate))


def test_bad_fps_flagged():
    plate = _valid_plate()
    plate.frame_rate = 0.0
    assert REGION_INVALID not in _codes([])  # sanity
    issues = validate_dynamic_plate(plate)
    assert any(i.code == "frame_rate_invalid" for i in issues)


def test_matte_shape_mismatch_flagged():
    plate = _valid_plate()
    issues = validate_dynamic_plate(plate, matte_shape=(500, 500))
    assert REGION_INVALID in _codes(issues)


def test_missing_color_metadata_warns():
    plate = _valid_plate()
    plate.color_metadata = {}
    issues = validate_dynamic_plate(plate)
    assert any(i.code == "color_metadata_missing" for i in issues)


# --------------------------------------------------------- frame sequences

def _frames(tmp_path, count, size=(8, 4), skip=None):
    pytest.importorskip("PIL")
    from PIL import Image
    paths = []
    for i in range(count):
        p = tmp_path / f"frame_{i:04d}.png"
        if skip is None or i != skip:
            Image.new("RGB", size).save(p)
        paths.append(p)
    return paths


def test_frame_sequence_complete(tmp_path):
    paths = _frames(tmp_path, 4)
    issues = frame_sequence_report(paths, expected_count=4, expected_size=(8, 4))
    assert issues == []


def test_frame_sequence_missing_frame(tmp_path):
    paths = _frames(tmp_path, 4, skip=2)
    issues = frame_sequence_report(paths, expected_count=4)
    assert FRAME_SEQUENCE_INCOMPLETE in _codes(issues)


def test_frame_sequence_wrong_count(tmp_path):
    paths = _frames(tmp_path, 3)
    issues = frame_sequence_report(paths, expected_count=8)
    assert FRAME_SEQUENCE_INCOMPLETE in _codes(issues)


def test_frame_sequence_dim_mismatch(tmp_path):
    paths = _frames(tmp_path, 2, size=(8, 4))
    issues = frame_sequence_report(paths, expected_count=2, expected_size=(16, 8))
    assert any(i.code == "frame_dimensions_mismatch" for i in issues)


def test_validator_includes_frame_check(tmp_path):
    plate = _valid_plate()
    plate.frame_end = 3
    paths = _frames(tmp_path, 4, skip=1)
    issues = validate_dynamic_plate(plate, frame_paths=paths)
    assert FRAME_SEQUENCE_INCOMPLETE in _codes(issues)
