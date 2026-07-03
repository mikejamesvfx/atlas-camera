from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasProxyPrimitive,
    AtlasSolve,
    ProjectionSource,
)


def test_schema_objects_can_serialize_to_json():
    intrinsics = build_intrinsics(
        image_width=1920,
        image_height=1080,
        focal_length_mm=50.0,
        sensor_width_mm=36.0,
    )
    solve = AtlasSolve(
        camera=AtlasCamera(intrinsics=intrinsics),
        image_path="concept.png",
        image_width=1920,
        image_height=1080,
        source_method="test",
    )

    restored = AtlasSolve.from_json(solve.to_json())

    assert restored.camera.intrinsics.fx_px == intrinsics.fx_px
    assert restored.camera.extrinsics.up_axis == "Y"
    assert restored.image_path == "concept.png"
    assert restored.camera.schema_version == "0.2"
    assert "confidence_detail" in restored.to_dict()


def test_y_up_convention_is_recorded():
    intrinsics = build_intrinsics(image_width=100, image_height=50)
    solve = AtlasSolve(camera=AtlasCamera(intrinsics=intrinsics))

    data = solve.to_dict()

    assert data["camera"]["extrinsics"]["up_axis"] == "Y"
    assert data["camera"]["extrinsics"]["coordinate_system"] == "right_handed"


def test_camera_confidence_notes_and_seed_round_trip():
    intrinsics = build_intrinsics(
        image_width=1920,
        image_height=1080,
        focal_length_mm=50.0,
        sensor_width_mm=36.0,
    )
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=intrinsics,
            notes=["Focal length was inferred."],
            focal_length_inferred=True,
            seed=42,
        ),
        image_path="concept.png",
        image_width=1920,
        image_height=1080,
        confidence=0.25,
    )

    restored = AtlasSolve.from_json(solve.to_json())

    assert restored.camera.notes == ["Focal length was inferred."]
    assert restored.camera.focal_length_inferred is True
    assert restored.camera.seed == 42
    assert restored.camera.confidence.global_score == 0.0


def test_projection_sources_default_empty_and_legacy_load():
    # A solve with no patches has an empty list, and legacy JSON that predates
    # the field loads without error (backward compatible).
    intrinsics = build_intrinsics(image_width=1920, image_height=1080, focal_length_mm=35.0)
    solve = AtlasSolve(camera=AtlasCamera(intrinsics=intrinsics))
    assert solve.projection_sources == []

    legacy = solve.to_dict()
    del legacy["projection_sources"]
    restored = AtlasSolve.from_dict(legacy)
    assert restored.projection_sources == []


def test_projection_source_round_trips_through_json():
    intrinsics = build_intrinsics(image_width=1920, image_height=1080, focal_length_mm=35.0)
    patch_cam = AtlasCamera(intrinsics=intrinsics)
    source = ProjectionSource(
        camera=patch_cam,
        name="patch_right",
        image_b64="data:image/png;base64,AAAA",
        proxy_geometry=[AtlasProxyPrimitive(name="patch_mesh", primitive_type="mesh")],
        azimuth_deg=35.0,
        elevation_deg=0.0,
        distance_scale=1.0,
        priority=1.0,
        metadata={"source": "multi_angle_lora"},
    )
    solve = AtlasSolve(camera=AtlasCamera(intrinsics=intrinsics))
    solve.projection_sources.append(source)

    restored = AtlasSolve.from_json(solve.to_json())

    assert len(restored.projection_sources) == 1
    got = restored.projection_sources[0]
    assert got.name == "patch_right"
    assert got.image_b64 == "data:image/png;base64,AAAA"
    assert got.azimuth_deg == 35.0
    assert got.priority == 1.0
    assert got.metadata["source"] == "multi_angle_lora"
    assert len(got.proxy_geometry) == 1
    assert got.proxy_geometry[0].primitive_type == "mesh"
    assert got.camera.intrinsics.image_width == 1920
