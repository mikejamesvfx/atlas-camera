"""Blender dynamic-plate script: projector/artist camera split + sequence."""
from __future__ import annotations

import pytest

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.dynamic_plate import (
    DynamicPlate,
    build_receiver_plane,
    crop_intrinsics_for_plate,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, LatentCamera
from atlas_camera.exporters.dcc_transform import blender_matrix_from_atlas
from atlas_camera.exporters.dynamic_plate_blender import (
    write_dynamic_plate_blender_script,
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


def _plate():
    cam = _camera()
    roi = RegionROI(x=0, y=228, width=640, height=132)
    return DynamicPlate(
        plate_id="WATER_0001", semantic_type="water",
        source_image="castle.png", source_width=640, source_height=360,
        source_roi=roi,
        crop_transform=CropTransform(source_width=640, source_height=360,
                                     roi=roi, output_width=roi.width,
                                     output_height=roi.height),
        source_camera=cam,
        crop_camera=crop_intrinsics_for_plate(cam, roi),
        receiver=build_receiver_plane(cam, roi),
        frame_rate=24.0, frame_start=0, frame_end=95)


def test_script_has_both_cameras_and_split(tmp_path):
    plate = _plate()
    path = write_dynamic_plate_blender_script(plate, tmp_path,
                                              tmp_path / "open_scene.py")
    text = path.read_text(encoding="utf-8")
    assert "atlas_projection_camera" in text
    assert "atlas_render_camera" in text
    # ONLY the artist camera becomes the scene camera
    assert text.count("scene.camera =") == 1
    assert "scene.camera = render_camera" in text
    # projection texture coordinates come from the projection camera OBJECT
    assert 'coord.object = projection_camera' in text
    assert 'coord.outputs["Object"]' in text
    assert 'coord.outputs["Camera"]' not in text


def test_script_sequence_settings(tmp_path):
    plate = _plate()
    path = write_dynamic_plate_blender_script(plate, tmp_path,
                                              tmp_path / "open_scene.py")
    text = path.read_text(encoding="utf-8")
    assert 'img.source = "SEQUENCE"' in text
    assert "frame_duration = 96" in text
    assert "use_auto_refresh = True" in text
    assert "frame_end = 95" in text
    assert "fps = 24" in text


def test_script_projection_matrix_is_crop_camera(tmp_path):
    plate = _plate()
    path = write_dynamic_plate_blender_script(plate, tmp_path,
                                              tmp_path / "open_scene.py")
    text = path.read_text(encoding="utf-8")
    expected = blender_matrix_from_atlas(
        plate.crop_camera.extrinsics.camera_world_matrix)
    assert repr(expected) in text
    # crop-camera pixel projection factors: fx'/w', cx'/w'
    intr = plate.crop_camera.intrinsics
    assert repr(intr.fx_px / intr.image_width) in text
    assert repr(intr.cx_px / intr.image_width) in text


def test_script_fallback_to_crop_still(tmp_path):
    plate = _plate()
    path = write_dynamic_plate_blender_script(plate, tmp_path,
                                              tmp_path / "open_scene.py")
    text = path.read_text(encoding="utf-8")
    # no generated frames yet -> falls back to the still crop with a warning
    assert "source/crop.png" in text.replace("\\\\", "/").replace("\\", "/")


def test_receiver_plane_vertices_converted(tmp_path):
    plate = _plate()
    path = write_dynamic_plate_blender_script(plate, tmp_path,
                                              tmp_path / "open_scene.py")
    text = path.read_text(encoding="utf-8")
    assert "receiver_vertices = " in text
    assert "from_pydata" in text
