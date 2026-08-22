from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "atlas_world",
    reason="the evidence-plate layer ships in the private atlas-world "
           "distribution, which atlas-camera does not depend on",
)

from atlas_world.real_plate import (
    ArtifactKind, ContentArtifact, ModelIdentity, ModelPolicy, PlateTime,
    ProcessingAttempt, RealPlateEpisode, SiteContext,
)
from atlas_world.orchestration import (HiddenRegion, ModelSpec, OrchestrationModelPolicy, Plan, Workspace)
from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.comfy.nodes_world_plate import (
    AtlasOpenRealPlate, AtlasReadLockedPlatePlan,
    AtlasRecordPlateAttempt, AtlasExportPlateHandoff,
)


def _plate() -> RealPlateEpisode:
    source = ContentArtifact.from_bytes(
        artifact_id="SOURCE", kind=ArtifactKind.SOURCE, media_type="image/nef",
        uri="source/plate.nef", data=b"raw",
    )
    model = ModelIdentity("SAM3", "1", "a" * 64, "MIT")
    plate = RealPlateEpisode(
        "PLATE_TEST", source, PlateTime("2026-01-01T00:00:00+00:00"),
        SiteContext(None, None), ModelPolicy((model,)),
    )
    return plate.record_artifact(ContentArtifact.from_bytes(
        artifact_id="MASK", kind=ArtifactKind.DERIVED, media_type="image/png",
        uri="masks/mask.png", data=b"mask",
    ))


def _locked_plan(plate: RealPlateEpisode) -> str:
    models = tuple(ModelSpec(name, "1", "a" * 64, "MIT") for name in (
        "GeoCalib", "Ruicheng/moge-2-vitl-normal", "SAM3", "AtlasSAM3Mask", "LaMa", "SDXL"))
    ws = Workspace("WS", CropTransform(100, 100, RegionROI(0, 0, 100, 100), 100, 100), "test")
    plan = Plan.propose(
        plate.plate_id, scene_graph={"element_ids": []}, depth_topology={},
        workspaces=(ws,), hidden_regions=(HiddenRegion("REGION", "MASK", ("WS",)),),
        model_policy=OrchestrationModelPolicy(models),
    ).review_edit().lock(approved_support=("MASK",), plate=plate)
    return plan.manifest_bytes().decode()


def test_open_plate_is_json_path_boundary_and_never_returns_pixels(tmp_path: Path):
    path = tmp_path / "episode.json"
    path.write_text(_plate().to_json(), encoding="utf-8")
    plate, payload, report = AtlasOpenRealPlate().open(str(path))
    assert plate.plate_id == "PLATE_TEST"
    assert json.loads(payload)["plate_id"] == "PLATE_TEST"
    assert json.loads(report)["observed_pixels_loaded"] is False


def test_open_plate_rejects_missing_path_and_malformed_json():
    with pytest.raises(ValueError, match="episode"):
        AtlasOpenRealPlate().open("does-not-exist.json")
    with pytest.raises(ValueError, match="JSON"):
        AtlasOpenRealPlate().open("{not json")


def test_locked_plan_reader_requires_valid_digest_and_lock_phase():
    payload = {"phase": "lock", "state_digest_version": 1, "state_digest": "bad"}
    with pytest.raises(ValueError, match="digest"):
        AtlasReadLockedPlatePlan().read(json.dumps(payload), _plate())


def test_plan_and_result_accept_portable_manifest_paths_and_bind_plate(tmp_path: Path):
    plate = _plate()
    plan_path = tmp_path / "locked.json"
    plan_path.write_text(_locked_plan(plate), encoding="utf-8")
    locked, _ = AtlasReadLockedPlatePlan().read(str(plan_path), plate)
    assert json.loads(locked)["plate_id"] == plate.plate_id
    attempt = ProcessingAttempt("ATTEMPT_PATH", "segmentation", ModelIdentity("SAM3", "1", "a" * 64, "MIT"), ())
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"attempt": attempt.to_dict()}), encoding="utf-8")
    updated, _, _ = AtlasRecordPlateAttempt().record(plate, str(result_path), str(plan_path))
    assert updated.attempts[-1].attempt_id == "ATTEMPT_PATH"
    foreign = _plate().to_dict()
    foreign["plate_id"] = "PLATE_OTHER"
    with pytest.raises(ValueError, match="different plate"):
        AtlasReadLockedPlatePlan().read(str(plan_path), RealPlateEpisode.from_dict(foreign))


def test_attempt_handoff_appends_typed_immutable_attempt_only():
    plate = _plate()
    attempt = ProcessingAttempt(
        "ATTEMPT_1", "segmentation", ModelIdentity("SAM3", "1", "a" * 64, "MIT"), (),
    )
    result = json.dumps({"attempt": attempt.to_dict()})
    locked = _locked_plan(plate)
    updated, payload, report = AtlasRecordPlateAttempt().record(plate, result, locked)
    assert updated is not plate
    assert updated.attempts[0] == attempt
    assert json.loads(payload)["attempt"]["attempt_id"] == "ATTEMPT_1"
    with pytest.raises(ValueError):
        AtlasRecordPlateAttempt().record(updated, result, locked)


def test_export_handoff_is_append_only_and_rejects_invalid_export():
    plate = _plate()
    export_artifact = ContentArtifact.from_bytes(
        artifact_id="EXPORT", kind=ArtifactKind.EXPORT, media_type="application/json",
        uri="exports/manifest.json", data=b"manifest",
    )
    payload = json.dumps({"export_id": "EXPORT_1", "artifact": export_artifact.to_dict(), "target": "nuke"})
    updated, manifest, _ = AtlasExportPlateHandoff().export(plate, payload)
    assert updated.exports[0].export_id == "EXPORT_1"
    assert json.loads(manifest)["export_id"] == "EXPORT_1"
    with pytest.raises(ValueError):
        AtlasExportPlateHandoff().export(updated, payload)
