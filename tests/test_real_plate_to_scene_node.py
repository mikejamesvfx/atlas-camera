"""Node-level contract for `AtlasRealPlateToScene`.

The roundtrip maths is covered by tests/world/test_plate_scene_roundtrip.py.
What matters here is the ComfyUI boundary: the registered key and display name
are a saved-workflow contract, the node must refuse malformed inputs rather
than half-build a scene, and a builder failure must surface as a plain node
error instead of an internal exception type ComfyUI cannot render.
"""

from __future__ import annotations

import json

import pytest


from atlas_camera.comfy import node_registry as registry
from atlas_camera.comfy.nodes_world_plate import AtlasRealPlateToScene, world_available

# The node registers and validates its own inputs with the public package alone;
# only the tests that actually round-trip an episode need the private World
# schema, so those are marked rather than the whole file skipped.
requires_world = pytest.mark.skipif(
    not world_available(),
    reason="needs the private atlas-world distribution")
from atlas_camera.core.schema import AtlasIntrinsics, AtlasSolve, LatentCamera


def test_the_node_is_registered_under_its_saved_workflow_key():
    assert registry.NODE_CLASS_MAPPINGS["AtlasRealPlateToScene"] is AtlasRealPlateToScene
    assert "AtlasRealPlateToScene" in registry.NODE_DISPLAY_NAME_MAPPINGS
    assert registry.MENU_CATEGORY["AtlasRealPlateToScene"] == "Atlas/11 · Evidence Plate"


def test_the_node_declares_the_evidence_plate_surface():
    assert AtlasRealPlateToScene.RETURN_TYPES == ("ATLAS_SOLVE", "STRING")
    assert AtlasRealPlateToScene.RETURN_NAMES == ("solve", "report")
    assert AtlasRealPlateToScene.CATEGORY == "Atlas/11 · Evidence Plate"

    schema = AtlasRealPlateToScene.INPUT_TYPES()
    assert schema["required"]["plate"] == ("ATLAS_REAL_PLATE",)
    assert schema["required"]["assets_dir"][0] == "STRING"
    # Appended as optional so the two-input form of a saved graph still loads.
    assert "scene_json_path" in schema["optional"]


@requires_world
def test_a_non_plate_input_is_refused():
    with pytest.raises(ValueError, match="ATLAS_REAL_PLATE"):
        AtlasRealPlateToScene().roundtrip({"plate_id": "NOPE"}, "assets")


@requires_world
def test_a_blank_assets_dir_is_refused():
    with pytest.raises(ValueError, match="assets_dir"):
        AtlasRealPlateToScene().roundtrip(_stub_plate(), "   ")


@requires_world
def test_a_builder_failure_surfaces_as_a_node_error(monkeypatch):
    import atlas_world.plate_scene_roundtrip as roundtrip

    def _boom(*args, **kwargs):
        raise roundtrip.PlateRoundtripError("ransac_planes.json is missing")

    monkeypatch.setattr(roundtrip, "build_plate_scene", _boom)
    with pytest.raises(ValueError, match="plate scene roundtrip failed"):
        AtlasRealPlateToScene().roundtrip(_stub_plate(), "assets")


@requires_world
def test_the_report_names_what_was_rebuilt(monkeypatch, tmp_path):
    import atlas_world.plate_scene_roundtrip as roundtrip

    solve = AtlasSolve(
        camera=LatentCamera(
            intrinsics=AtlasIntrinsics(image_width=16, image_height=12, fx_px=12.0, fy_px=12.0)
        )
    )
    result = roundtrip.PlateSceneRoundtrip(
        solve=solve,
        primitives=(),
        sources=(),
        id_map={1: "car"},
        report={"plate_id": "TEST_PLATE", "planes_verified": 8, "cards_rebuilt": 1},
    )
    monkeypatch.setattr(roundtrip, "build_plate_scene", lambda *a, **k: result)

    target = tmp_path / "scene" / "atlas_scene.json"
    out_solve, report = AtlasRealPlateToScene().roundtrip(
        _stub_plate(), str(tmp_path), str(target)
    )

    assert out_solve is solve
    payload = json.loads(report)
    assert payload["status"] == "roundtripped"
    assert payload["planes_verified"] == 8
    assert payload["cards_rebuilt"] == 1
    # scene_json_path is honoured, and the report says where it landed.
    assert payload["written_to"] == str(target)
    assert target.is_file()


def _stub_plate():
    """A RealPlateEpisode good enough for the node's isinstance gate."""

    import hashlib

    from atlas_world.real_plate import (
        ArtifactKind,
        ContentArtifact,
        ModelIdentity,
        ModelPolicy,
        PlateTime,
        RealPlateEpisode,
        SiteContext,
    )

    return RealPlateEpisode(
        plate_id="TEST_PLATE",
        source_artifact=ContentArtifact(
            artifact_id="SOURCE_TEST",
            kind=ArtifactKind.SOURCE,
            media_type="image/x-nikon-nef",
            sha256=hashlib.sha256(b"raw").hexdigest(),
            byte_size=3,
            uri="source/TEST.NEF",
        ),
        capture=PlateTime(None),
        site=SiteContext(None, None),
        model_policy=ModelPolicy(
            (ModelIdentity("SAM3", "test", hashlib.sha256(b"sam3").hexdigest(), "custom"),)
        ),
    )


@requires_world
def test_recording_without_a_written_scene_is_refused():
    """Only a file on disk can be content-addressed into the ledger."""

    with pytest.raises(ValueError, match="needs scene_json_path"):
        AtlasRealPlateToScene().roundtrip(
            _stub_plate(), "assets", "", True
        )


@requires_world
def test_the_ledger_opt_in_is_off_by_default(monkeypatch, tmp_path):
    """Rebuilding a scene must never mint an episode revision by itself."""

    import atlas_world.plate_scene_roundtrip as roundtrip
    from atlas_camera.core.schema import AtlasIntrinsics, AtlasSolve, LatentCamera

    result = roundtrip.PlateSceneRoundtrip(
        solve=AtlasSolve(
            camera=LatentCamera(
                intrinsics=AtlasIntrinsics(image_width=16, image_height=12, fx_px=12.0, fy_px=12.0)
            )
        ),
        primitives=(),
        sources=(),
        id_map={},
        report={"plate_id": "TEST_PLATE"},
    )
    monkeypatch.setattr(roundtrip, "build_plate_scene", lambda *a, **k: result)
    called = []
    monkeypatch.setattr(
        roundtrip, "record_scene_artifact", lambda *a, **k: called.append(a) or _stub_plate()
    )

    _, report = AtlasRealPlateToScene().roundtrip(
        _stub_plate(), str(tmp_path), str(tmp_path / "s.json")
    )
    assert called == []
    assert "recorded_artifact_id" not in json.loads(report)
