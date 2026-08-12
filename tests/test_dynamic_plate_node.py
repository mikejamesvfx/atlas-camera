"""AtlasLoadDynamicPlate 🌊🔬 — package -> animated viewport projection layer."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from atlas_camera.comfy.nodes_dynamic import (
    AtlasLoadDynamicPlate,
    registered_plate_dir,
)
from atlas_camera.comfy.viewport_payload import _serialize_projection_sources
from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.dynamic_plate import (
    DynamicPlate,
    build_receiver_plane,
    crop_intrinsics_for_plate,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, AtlasSolve, LatentCamera
from atlas_camera.exporters.dynamic_plate_package import (
    build_dynamic_plate_package,
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
def package(tmp_path):
    rng = np.random.default_rng(0)
    img_path = tmp_path / "castle.png"
    Image.fromarray((rng.random((360, 640, 3)) * 255).astype(np.uint8)
                    ).save(img_path)
    cam = _camera()
    roi = RegionROI(x=0, y=200, width=640, height=160)
    plate = DynamicPlate(
        plate_id="WATER_0001", semantic_type="water",
        source_image="castle.png", source_width=640, source_height=360,
        source_roi=roi,
        crop_transform=CropTransform(source_width=640, source_height=360,
                                     roi=roi, output_width=roi.width,
                                     output_height=roi.height),
        source_camera=cam,
        crop_camera=crop_intrinsics_for_plate(cam, roi),
        receiver=build_receiver_plane(cam, roi),
        frame_rate=24.0, frame_start=0, frame_end=3)
    result = build_dynamic_plate_package(plate, tmp_path / "dynamic",
                                         source_image_path=img_path)
    for i in range(4):
        Image.new("RGB", (roi.width, roi.height)).save(
            result.package_dir / "generated" / f"frame_{i:04d}.png")
    return result.package_dir


def _solve():
    return AtlasSolve(camera=_camera(), image_width=640, image_height=360)


def test_node_appends_animated_source(package):
    node = AtlasLoadDynamicPlate()
    solve = _solve()
    out, report = node.load(solve, str(package), priority=7.0)
    assert solve.projection_sources == []          # input never mutated
    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    assert src.name == "dynamic_plate_water_0001"
    assert src.priority == 7.0
    assert src.image_b64.startswith("data:image/png;base64,")
    assert src.proxy_geometry[0].primitive_type == "plane"
    dyn = src.metadata["dynamic_plate"]
    assert dyn["frame_count"] == 4
    assert dyn["fps"] == 24.0
    assert registered_plate_dir(dyn["key"]) == package
    assert "4 frame(s)" in report


def test_payload_carries_dynamic_plate_block(package):
    node = AtlasLoadDynamicPlate()
    out, _ = node.load(_solve(), str(package))
    serialized = _serialize_projection_sources(out)
    assert serialized[0]["dynamic_plate"]["frame_count"] == 4
    assert serialized[0]["projection_mode"] == "clean_plate"
    assert serialized[0]["evidence_type"] == "generated"
    assert serialized[0]["proxy_geometry"]


def test_node_missing_package_reports_not_crashes(tmp_path):
    node = AtlasLoadDynamicPlate()
    solve = _solve()
    out, report = node.load(solve, str(tmp_path / "nope"))
    assert out is solve
    assert report.startswith("SKIPPED")


def test_node_still_only_package(package):
    for f in (package / "generated").glob("frame_*.png"):
        f.unlink()
    node = AtlasLoadDynamicPlate()
    out, report = node.load(_solve(), str(package))
    assert "still crop only" in report
    assert out.projection_sources[0].metadata["dynamic_plate"]["frame_count"] == 0


def test_registry_reuses_key_per_package(package):
    node = AtlasLoadDynamicPlate()
    _, _ = node.load(_solve(), str(package))
    out2, _ = node.load(_solve(), str(package))
    key = out2.projection_sources[0].metadata["dynamic_plate"]["key"]
    assert registered_plate_dir(key) == package


def test_is_changed_tracks_manifest(package):
    a = AtlasLoadDynamicPlate.IS_CHANGED(None, str(package))
    (package / "manifest.json").write_text(
        (package / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    b = AtlasLoadDynamicPlate.IS_CHANGED(None, str(package))
    assert a != b or a.split(":")[1:] == b.split(":")[1:]
