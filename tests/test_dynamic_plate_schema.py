"""DynamicPlate schema serialization + provenance contracts."""
from __future__ import annotations

import json

import pytest

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.dynamic_plate import (
    DYNAMIC_REGION_TYPES,
    GENERATOR_NOT_AVAILABLE,
    PLATE_STATUS_DRAFT,
    WATER_PROMPT_DEFAULT,
    DynamicPlate,
    ReceiverGeometry,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasProxyPrimitive, LatentCamera


def _camera(width=1920, height=1080):
    return LatentCamera(intrinsics=build_intrinsics(
        image_width=width, image_height=height, focal_length_mm=32.0))


def _plate() -> DynamicPlate:
    roi = RegionROI(x=0, y=600, width=1920, height=480)
    return DynamicPlate(
        plate_id="WATER_0001",
        semantic_type="water",
        source_image="castle.png",
        source_width=1920,
        source_height=1080,
        matte_bbox=RegionROI(x=10, y=620, width=1880, height=440),
        source_roi=roi,
        crop_transform=CropTransform(source_width=1920, source_height=1080,
                                     roi=roi, output_width=960,
                                     output_height=240),
        source_camera=_camera(),
        crop_camera=_camera(1920, 480),
        receiver=ReceiverGeometry(primitive=AtlasProxyPrimitive(
            name="dynamic_plate_receiver", primitive_type="plane")),
        frame_rate=24.0,
        frame_start=0,
        frame_end=95,
        generator="ltx",
        prompt=WATER_PROMPT_DEFAULT,
        seed=7,
    )


def test_region_types_include_spec_set():
    for t in ("water", "cloud", "smoke", "fire", "foliage", "cloth",
              "actor", "generic"):
        assert t in DYNAMIC_REGION_TYPES


def test_unknown_semantic_type_raises():
    with pytest.raises(ValueError):
        DynamicPlate(plate_id="X", semantic_type="lava",
                     source_image="a.png", source_width=10, source_height=10)


def test_serialization_round_trip():
    plate = _plate()
    data = json.loads(plate.to_json())
    again = DynamicPlate.from_dict(data)
    assert again.plate_id == plate.plate_id
    assert again.semantic_type == "water"
    assert again.source_roi == plate.source_roi
    assert again.matte_bbox == plate.matte_bbox
    assert again.crop_transform == plate.crop_transform
    assert again.crop_camera.intrinsics.image_height == 480
    assert again.receiver.primitive.primitive_type == "plane"
    assert again.frame_end == 95
    assert again.seed == 7
    assert again.prompt == WATER_PROMPT_DEFAULT


def test_minimal_from_dict_tolerates_missing():
    plate = DynamicPlate.from_dict({
        "plate_id": "P", "semantic_type": "generic",
        "source_image": "x.png", "source_width": 4, "source_height": 4})
    assert plate.source_roi is None
    assert plate.receiver is None
    assert plate.crop_camera is None
    assert plate.status == PLATE_STATUS_DRAFT
    assert plate.warnings == []


def test_provenance_defaults():
    plate = _plate()
    prov = plate.provenance
    assert prov["source_region"] == "observed"
    assert prov["crop_camera"] == "derived_from_solve"
    assert prov["generated_frames"] == "generated"
    # generated imagery must never be promoted to observed truth
    assert prov["generated_frames"] != "observed"


def test_generator_not_available_constant():
    assert GENERATOR_NOT_AVAILABLE == "not_available"


def test_receiver_geometry_round_trip():
    rec = ReceiverGeometry(primitive=AtlasProxyPrimitive(
        name="r", primitive_type="plane", dimensions=(40.0, 30.0, 0.0)),
        path="geometry/receiver.obj")
    again = ReceiverGeometry.from_dict(rec.to_dict())
    assert again.kind == "plane"
    assert again.path == "geometry/receiver.obj"
    assert again.primitive.dimensions == (40.0, 30.0, 0.0)
    assert again.provenance == "derived_from_solve"
    assert ReceiverGeometry.from_dict(None) is None


def test_color_metadata_present_by_default():
    plate = _plate()
    assert "input_color_space" in plate.color_metadata
    assert "atlas_working_color_space" in plate.color_metadata
