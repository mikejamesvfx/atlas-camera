from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

pytest.importorskip(
    "atlas_world",
    reason="the evidence-plate layer ships in the private atlas-world "
           "distribution, which atlas-camera does not depend on",
)

from atlas_world.real_plate import (
    ArtifactKind,
    CardHypothesis,
    ContentArtifact,
    ModelIdentity,
    ModelPolicy,
    PlateTime,
    ProvenanceSource,
    RealPlateEpisode,
    SceneElement,
    SiteContext,
    SpatialHypothesis,
    SpatialTransform,
)
from atlas_camera.exporters.real_plate_nuke import (
    NukeAssetInput,
    NukeCameraInput,
    NukeCardInput,
    NukeDeliveryProfile,
    export_real_plate_nuke,
)
from atlas_camera.exporters.real_plate_nuke import _validate_exr


IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _plate(tmp_path: Path) -> tuple[RealPlateEpisode, dict[str, Path]]:
    source_path = tmp_path / "source.nef"
    master_path = tmp_path / "master.exr"
    aux_path = tmp_path / "aux.exr"
    card_path = tmp_path / "card.exr"
    card_image_path = tmp_path / "card_image.exr"
    source_path.write_bytes(b"source")
    master_path.write_bytes(b"master")
    aux_path.write_bytes(b"aux")
    card_path.write_bytes(b"card")
    card_image_path.write_bytes(b"card-image")
    def art(aid: str, kind: ArtifactKind, path: Path, media: str) -> ContentArtifact:
        return ContentArtifact.from_bytes(artifact_id=aid, kind=kind, media_type=media, uri=f"assets/{path.name}", data=path.read_bytes())
    source = art("SOURCE", ArtifactKind.SOURCE, source_path, "image/x-nef")
    master = art("MASTER", ArtifactKind.DERIVED, master_path, "image/exr")
    aux = art("AUX", ArtifactKind.DERIVED, aux_path, "image/exr")
    card = art("CARD_ASSET", ArtifactKind.DERIVED, card_path, "image/exr")
    card_image = art("CARD_IMAGE", ArtifactKind.DERIVED, card_image_path, "image/exr")
    transform = SpatialTransform(IDENTITY)
    plate = RealPlateEpisode(
        "PLATE_ONE", source, PlateTime("2026-01-01T00:00:00+00:00"), SiteContext(None, None),
        ModelPolicy((ModelIdentity("MODEL", "1", "a" * 64, "MIT"),)),
        elements=(SceneElement("ELEMENT_CAR", "car", ProvenanceSource.OBSERVED),),
        spatial_hypotheses=(SpatialHypothesis("PLANE", transform, ProvenanceSource.SOLVED),),
        card_hypotheses=(CardHypothesis("CARD_CAR", "ELEMENT_CAR", transform, ProvenanceSource.OBSERVED),),
    )
    plate = plate.record_artifact(master).record_artifact(aux).record_artifact(card).record_artifact(card_image)
    return plate, {"source": source_path, "master": master_path, "aux": aux_path, "card": card_path, "card_image": card_image_path}


def _inspect(path: str):
    name = Path(path).name
    common = {"compression": "zip", "data_window": (0, 0, 16, 16), "display_window": (0, 0, 16, 16), "metadata": {"color_space": "ACEScg"}}
    if name == "master.exr":
        return {**common, "channels": {"R": "half", "G": "half", "B": "half", "A": "half"}, "metadata": {"oiio:ColorSpace": "ACEScg", "color_space": "ACEScg"}}
    if name == "card_image.exr":
        return {**common, "channels": {"R": "half", "G": "half", "B": "half", "A": "half"}, "metadata": {"oiio:ColorSpace": "ACEScg", "color_space": "ACEScg"}}
    if name == "generated.exr":
        return {**common, "channels": {"R": "half", "G": "half", "B": "half", "validity": "float"}, "metadata": {"oiio:ColorSpace": "ACEScg", "color_space": "ACEScg"}}
    channels = {key: "float" for key in ("depth.Z", "validity", "object_id", "semantic_id", "card_id", "workspace_id", "generated_support", "disocclusion", "approval", "distortion.u", "distortion.v", "undistortion.u", "undistortion.v", "P_world.red", "P_world.green", "P_world.blue", "N_world.red", "N_world.green", "N_world.blue")}
    if name == "card.exr":
        channels.update({"R": "half", "G": "half", "B": "half", "A": "half", "N_object.red": "float", "N_object.green": "float", "N_object.blue": "float"})
    return {**common, "channels": channels, "metadata": {"oiio:ColorSpace": "ACEScg"}}


def _patch_readers(monkeypatch):
    aux = {name: np.ones((16, 16), dtype=np.float32) for name in ("depth.Z", "validity", "object_id", "semantic_id", "card_id", "workspace_id", "generated_support", "disocclusion", "approval", "distortion.u", "distortion.v", "undistortion.u", "undistortion.v", "P_world.red", "P_world.green", "P_world.blue", "N_world.red", "N_world.green", "N_world.blue")}
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.read_master_plate", lambda path: SimpleNamespace(pixels=np.ones((16, 16, 3), dtype=np.float32)))
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.read_auxiliary_exr", lambda path: aux)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.read_card_auxiliary_exr", lambda path: SimpleNamespace(channels={"N_object.red": np.ones((16, 16), dtype=np.float32), "N_object.green": np.ones((16, 16), dtype=np.float32), "N_object.blue": np.ones((16, 16), dtype=np.float32)}, object_to_world=IDENTITY))
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.read_generated_patch", lambda path: SimpleNamespace(display_window=(0, 0, 16, 16), data_window=(0, 0, 16, 16), source_sha256="a" * 64))


def _assets(paths):
    return (
        NukeAssetInput("SOURCE", "source", paths["source"]),
        NukeAssetInput("MASTER", "master", paths["master"]),
        NukeAssetInput("AUX", "auxiliary", paths["aux"]),
        NukeAssetInput("CARD_IMAGE", "card_image", paths["card_image"]),
        NukeAssetInput("CARD_ASSET", "card_auxiliary", paths["card"]),
    )


def test_native_scene_is_deterministic_and_records_export(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    assets = _assets(paths)
    cards = (NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET"),)
    first = export_real_plate_nuke(plate, tmp_path / "out" / "shot.nk", camera=camera, assets=assets, cards=cards)
    second = export_real_plate_nuke(plate, tmp_path / "out" / "shot.nk", camera=camera, assets=assets, cards=cards)
    assert first.scene_bytes == second.scene_bytes
    assert "Camera2" in first.scene_text and "P_world.red" in first.scene_text
    assert " red red" in first.scene_text
    assert " red P.red" not in first.scene_text and " red N.red" not in first.scene_text
    assert "name Read_Master" in first.scene_text and " disable 0" in first.scene_text
    assert first.episode.exports[-1].target == "nuke_indie"
    assert first.episode.exports[-1].artifact.sha256 == hashlib.sha256(first.scene_bytes).hexdigest()


def test_rejects_stale_hash_and_unknown_card(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    paths["master"].write_bytes(b"changed")
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    assets = (NukeAssetInput("SOURCE", "source", paths["source"]), NukeAssetInput("MASTER", "master", paths["master"]), NukeAssetInput("AUX", "auxiliary", paths["aux"]))
    with pytest.raises(ValueError, match="stale source hash"):
        export_real_plate_nuke(plate, tmp_path / "out.nk", camera=camera, assets=assets, cards=())


def test_proxy_limit_and_transform_are_explicit(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    with pytest.raises(ValueError, match="proxy"):
        NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=5000, height=3333)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    assets = _assets(paths)
    with pytest.raises(ValueError, match="transform"):
        export_real_plate_nuke(plate, tmp_path / "out.nk", camera=camera, assets=assets, cards=(NukeCardInput("CARD_CAR", "CARD_IMAGE", True, ((1.,0.,0.,1.), IDENTITY[1], IDENTITY[2], IDENTITY[3]), "CARD_ASSET"),))


def test_generated_patch_requires_support_matte(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    generated = tmp_path / "generated.exr"
    generated.write_bytes(b"generated")
    generated_artifact = ContentArtifact.from_bytes(artifact_id="GENERATED", kind=ArtifactKind.DERIVED, media_type="image/exr", uri="assets/generated.exr", data=generated.read_bytes())
    plate = plate.record_artifact(generated_artifact)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    assets = _assets(paths) + (NukeAssetInput("GENERATED", "generated_patch", generated),)
    with pytest.raises(ValueError, match="generated patch"):
        export_real_plate_nuke(plate, tmp_path / "out.nk", camera=camera, assets=assets, cards=(NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET"),))


def test_profile_controls_card_enable_without_plate_name_heuristics(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    assets = _assets(paths)
    card = NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET")
    enabled = export_real_plate_nuke(plate, tmp_path / "enabled.nk", camera=camera, assets=assets, cards=(card,), profile=NukeDeliveryProfile())
    disabled = export_real_plate_nuke(plate, tmp_path / "disabled.nk", camera=camera, assets=assets, cards=(card,), profile=NukeDeliveryProfile(("CARD_CAR",)))
    assert "OBSERVED card ENABLED" in enabled.scene_text
    assert "OBSERVED card DISABLED" in disabled.scene_text


def test_asset_permutation_has_identical_bytes(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    source, master, aux, card_image, card_aux = _assets(paths)
    card = NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET")
    first = export_real_plate_nuke(plate, tmp_path / "a.nk", camera=camera, assets=(source, master, aux, card_image, card_aux), cards=(card,))
    second = export_real_plate_nuke(plate, tmp_path / "a.nk", camera=camera, assets=(card_aux, card_image, aux, master, source), cards=(card,))
    assert first.scene_bytes == second.scene_bytes


def test_scene_graph_is_connected_and_source_is_note_only(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    result = export_real_plate_nuke(plate, tmp_path / "graph.nk", camera=camera, assets=_assets(paths), cards=(NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET"),))
    text = result.scene_text
    assert "OBSERVED source provenance only" in text
    assert "Read_Source" not in text
    assert "Scene_Atlas_Cards" in text and "ScanlineRender_Atlas" in text and "Write_Atlas_Preview" in text
    assert "push $N_CARD_CAR" in text and "set N_Camera2_Atlas" in text
    assert "Card3D_Atlas_001" in text and "inputs 1" in text


def test_stack_simulation_scene_cards_only_camera_only_render_and_preview_paths(tmp_path):
    plate, paths = _plate(tmp_path)
    assets = _assets(paths)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    module = __import__("atlas_camera.exporters.real_plate_nuke", fromlist=["_scene_text"])
    text = module._scene_text(plate, camera, list(assets), [NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET")], NukeDeliveryProfile(), tmp_path / "shot_a_preview.exr")
    scene_pushes = text[text.index("set N_CARD_CAR"):text.index("Scene {")]
    assert "push $N_Camera2_Atlas" not in scene_pushes
    assert "push $N_CARD_CAR" in scene_pushes
    render_start = text.index("push $N_Reformat_Atlas_Preview")
    render_end = text.index("ScanlineRender {")
    render_pushes = text[render_start:render_end]
    assert render_pushes.index("push $N_Reformat_Atlas_Preview") < render_pushes.index("push $N_Scene_Atlas_Cards") < render_pushes.index("push $N_Camera2_Atlas")
    assert 'file "' + str((tmp_path / "shot_a_preview.exr").resolve()).replace("\\", "/") + '"' in text
    zero = module._scene_text(plate, camera, list(assets), [], NukeDeliveryProfile(), tmp_path / "zero_preview.exr")
    assert "Scene_Atlas_Cards" in zero and " inputs 0" in zero


def test_tcl_substitution_characters_are_rejected(tmp_path):
    quote = __import__("atlas_camera.exporters.real_plate_nuke", fromlist=["_quote"])._quote
    for value in ("$HOME", "[file exists x]", "name]", "name[", "{name}"):
        with pytest.raises(ValueError, match="unsafe"):
            quote(value)


def test_preview_path_is_collision_free_for_distinct_nk_outputs(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect); _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    card = NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET")
    assets = _assets(paths)
    first = export_real_plate_nuke(plate, tmp_path / "one" / "shot.nk", camera=camera, assets=assets, cards=(card,))
    second = export_real_plate_nuke(plate, tmp_path / "two" / "shot_alt.nk", camera=camera, assets=assets, cards=(card,))
    assert "one/shot_preview.exr" in first.scene_text.replace("\\", "/")
    assert "two/shot_alt_preview.exr" in second.scene_text.replace("\\", "/")
    assert first.scene_bytes != second.scene_bytes


def test_unknown_profile_card_and_missing_card_pair_rejected(tmp_path, monkeypatch):
    plate, paths = _plate(tmp_path)
    monkeypatch.setattr("atlas_camera.exporters.real_plate_nuke.inspect_exr", _inspect)
    _patch_readers(monkeypatch)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    with pytest.raises(ValueError, match="unknown card"):
        export_real_plate_nuke(plate, tmp_path / "bad.nk", camera=camera, assets=_assets(paths), cards=(NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET"),), profile=NukeDeliveryProfile(("MISSING",)))


def test_real_task2_exr_readers_are_accepted(tmp_path):
    pytest.importorskip("OpenImageIO")
    from atlas_world import plate_artifacts as module
    metadata = {"coordinate_system": "metric_right_handed_y_up", "image_origin": "top_left", "depth_convention": "positive_camera_forward", "color_space": "ACEScg", "ocio_config": "ocio://default", "ocio_config_hash": "a" * 64, "ocio_transform": "scene_linear_to_ACEScg"}
    image = np.ones((2, 2, 3), dtype=np.float32)
    master = tmp_path / "master.exr"
    module.write_master_plate(master, image, metadata=metadata)
    channels = {name: np.zeros((2, 2), dtype=np.float32) for name in ("depth.Z", "P_world.red", "P_world.green", "P_world.blue", "N_world.red", "N_world.green", "N_world.blue", "validity", "object_id", "semantic_id", "card_id", "workspace_id", "generated_support", "disocclusion", "approval", "distortion.u", "distortion.v", "undistortion.u", "undistortion.v")}
    channels["depth.Z"][:] = 1; channels["validity"][:] = 1; channels["N_world.green"][:] = 1; channels["approval"][:] = 1
    aux = tmp_path / "aux.exr"; module.write_auxiliary_exr(aux, channels, metadata=metadata)
    card_aux = tmp_path / "card_aux.exr"; card_channels = dict(channels); card_channels.update({"N_object.red": channels["P_world.red"], "N_object.green": channels["N_world.green"], "N_object.blue": channels["P_world.blue"]}); module.write_card_auxiliary_exr(card_aux, card_channels, object_to_world=np.eye(4), metadata=metadata)
    patch = tmp_path / "patch.exr"; module.write_generated_patch(patch, image, canvas_size=(2, 2), data_window=(0, 0, 2, 2), approved_support=np.ones((2, 2), dtype=np.float32), validity=np.ones((2, 2), dtype=np.float32), source_sha256="a" * 64, metadata=metadata)
    assert _validate_exr(master, "master")["data_window"] == (0, 0, 2, 2)
    assert _validate_exr(aux, "auxiliary")["display_window"] == (0, 0, 2, 2)
    assert _validate_exr(card_aux, "card_auxiliary")["data_window"] == (0, 0, 2, 2)
    assert _validate_exr(patch, "generated_patch")["atlas_source_sha256"] == "a" * 64


def test_topology_parser_proves_patch_reformat_and_camera_scene_wiring(tmp_path):
    plate, paths = _plate(tmp_path)
    generated = tmp_path / "generated.exr"; generated.write_bytes(b"generated")
    assets = _assets(paths) + (NukeAssetInput("GENERATED", "generated_patch", generated),)
    camera = NukeCameraInput(IDENTITY, focal_length_mm=35.0, sensor_width_mm=36.0, width=4096, height=2736)
    text = __import__("atlas_camera.exporters.real_plate_nuke", fromlist=["_scene_text"])._scene_text(plate, camera, sorted(assets, key=lambda x: x.artifact_id), [NukeCardInput("CARD_CAR", "CARD_IMAGE", True, IDENTITY, "CARD_ASSET")], NukeDeliveryProfile(), tmp_path / "preview.exr")
    reformat = text.index("Reformat_Atlas_Preview")
    assert text.index("push $N_Merge_Plate_Patches_001") < reformat
    scene = text.index("Scene_Atlas_Cards")
    scanline = text.index("ScanlineRender {", text.index("Scene_Atlas_Cards"))
    scene_region = text[scene:text.index("set N_Scene_Atlas_Cards")]
    assert "inputs 1" in scene_region and "N_Camera2_Atlas" not in scene_region
    render_region = text[text.index("push $N_Reformat_Atlas_Preview"):scanline]
    assert "push $N_Scene_Atlas_Cards" in render_region and "push $N_Camera2_Atlas" in render_region
    assert "raw true" in text and "Write_Atlas_Preview" in text


def test_tcl_unsafe_path_is_rejected(tmp_path, monkeypatch):
    quote = __import__("atlas_camera.exporters.real_plate_nuke", fromlist=["_quote"])._quote
    with pytest.raises(ValueError, match="unsafe"):
        quote("bad\nname")
