import atlas
import atlas_camera
from atlas_camera import LatentCamera, LatentComponent, LatentScene
from atlas_camera.core.schema import AtlasCamera, AtlasSolve


def test_latent_aliases_match_stable_schema_names():
    assert LatentCamera is AtlasCamera
    assert LatentScene is AtlasSolve
    assert LatentScene.__name__ == "LatentScene"
    assert LatentCamera.__name__ == "LatentCamera"


def test_recover_returns_current_latent_scene(tmp_path):
    image_path = tmp_path / "concept.png"

    scene = atlas.recover(image_path, image_size=(1920, 1080))

    assert isinstance(scene, LatentScene)
    assert isinstance(scene.camera, LatentCamera)
    assert scene.image_width == 1920
    assert scene.image_height == 1080
    assert scene.source_method == "automatic_still_image_metadata_only"
    assert isinstance(scene.depth, LatentComponent)
    assert isinstance(scene.geometry, LatentComponent)
    assert isinstance(scene.lighting, LatentComponent)
    assert isinstance(scene.semantics, LatentComponent)
    assert scene.projection_workspace is scene.projection_scene


def test_import_atlas_facade_matches_atlas_camera_api():
    assert atlas.recover is atlas_camera.recover
    assert atlas.LatentScene is atlas_camera.LatentScene


def test_old_atlas_solve_json_loads_as_latent_scene():
    old_json = """
    {
      "schema_version": "0.1",
      "camera": {
        "name": "legacy_camera",
        "intrinsics": {
          "image_width": 640,
          "image_height": 360,
          "sensor_width_mm": 36.0
        }
      },
      "confidence": 0.25,
      "source_method": "legacy_fixture"
    }
    """

    scene = LatentScene.from_json(old_json)

    assert isinstance(scene, LatentScene)
    assert scene.camera.name == "legacy_camera"
    assert scene.image_width == 640
    assert scene.image_height == 360
    assert scene.depth == LatentComponent()
    assert scene.projection_workspace is not None


def test_latent_scene_json_round_trips_component_slots():
    scene = atlas.recover("concept.png", image_size=(320, 180))
    scene.depth = LatentComponent(
        value={"map": None},
        confidence=0.1,
        editable=True,
        exportable=False,
        metadata={"status": "placeholder"},
        warnings=["Depth is not solved yet."],
    )

    restored = LatentScene.from_json(scene.to_json())

    assert restored.depth.value == {"map": None}
    assert restored.depth.confidence == 0.1
    assert restored.depth.metadata["status"] == "placeholder"
    assert restored.depth.warnings == ["Depth is not solved yet."]
    assert restored.to_dict()["scene_type"] == "latent_scene"
