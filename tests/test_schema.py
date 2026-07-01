from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasSolve


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
