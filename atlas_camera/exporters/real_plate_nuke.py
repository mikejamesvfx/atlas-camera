"""Dependency-free native Nuke Indie delivery for evidence-first plate episodes.

This adapter deliberately owns no RAW, OIIO, NumPy, or Nuke imports at module
load.  It validates files at the boundary, writes a plain-text native script,
and records that script through the immutable World plate ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from os import PathLike
from typing import Any, Literal, Mapping

# The evidence-plate layer lives in the private atlas-world distribution, which
# is NOT a dependency of this package: atlas-camera is the public tool and the
# World schema is research. So the import is guarded — the module must import
# cleanly without it, or node_registry cannot load and every unrelated node
# vanishes from ComfyUI alongside this one. Entry points call `_require_world()`,
# which fails loudly with an install hint rather than raising NameError deep
# inside a write.
try:                                       # pragma: no cover - install-shaped
    from atlas_world.plate_artifacts import (
        inspect_exr,
        read_auxiliary_exr,
        read_card_auxiliary_exr,
        read_generated_patch,
        read_master_plate,
    )
    from atlas_world.real_plate import (
        ArtifactKind,
        ContentArtifact,
        PlateExport,
        RealPlateEpisode,
        SpatialTransform,
    )
    _WORLD_IMPORT_ERROR = None
except ImportError as _exc:                # pragma: no cover - install-shaped
    inspect_exr = read_auxiliary_exr = read_card_auxiliary_exr = None
    read_generated_patch = read_master_plate = None
    ArtifactKind = ContentArtifact = PlateExport = None
    RealPlateEpisode = SpatialTransform = None
    _WORLD_IMPORT_ERROR = _exc

WORLD_INSTALL_HINT = (
    "This requires the atlas-world package, which is distributed separately "
    "from atlas-camera. Install it (pip install -e path/to/atlas-world) and "
    "re-run."
)


def world_available() -> bool:
    """Whether the private World package could be imported."""
    return _WORLD_IMPORT_ERROR is None


def _require_world() -> None:
    if _WORLD_IMPORT_ERROR is not None:
        raise RuntimeError("Evidence-plate Nuke export: " + WORLD_INSTALL_HINT)             from _WORLD_IMPORT_ERROR


_IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
_ROLES = {"source", "master", "auxiliary", "card_image", "card_auxiliary", "generated_patch", "redistortion"}
_AUX_REQUIRED = {
    "depth.Z", "validity", "object_id", "semantic_id", "card_id", "workspace_id",
    "generated_support", "disocclusion", "approval", "distortion.u", "distortion.v",
    "undistortion.u", "undistortion.v", "P_world.red", "P_world.green", "P_world.blue",
    "N_world.red", "N_world.green", "N_world.blue",
}

from atlas_camera.plate.oiio_io import ATLAS_COLORSPACE_ATTR


def _matrix(value: Any, label: str) -> tuple[tuple[float, float, float, float], ...]:
    try:
        return SpatialTransform(tuple(tuple(float(x) for x in row) for row in value)).matrix
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an explicit rigid metric transform") from exc


@dataclass(frozen=True, slots=True)
class NukeCameraInput:
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    focal_length_mm: float
    sensor_width_mm: float
    width: int
    height: int
    authoritative_width: int = 7380
    authoritative_height: int = 4928

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_to_world", _matrix(self.camera_to_world, "camera transform"))
        if type(self.focal_length_mm) not in (int, float) or not math.isfinite(self.focal_length_mm) or self.focal_length_mm <= 0:
            raise ValueError("focal_length_mm must be positive")
        if type(self.sensor_width_mm) not in (int, float) or not math.isfinite(self.sensor_width_mm) or self.sensor_width_mm <= 0:
            raise ValueError("sensor_width_mm must be positive")
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0:
            raise ValueError("camera resolution must be positive integers")
        if self.width > 4096 or self.height > 2736:
            raise ValueError("Nuke Indie proxy resolution exceeds 4096x2736")
        if type(self.authoritative_width) is not int or type(self.authoritative_height) is not int or self.authoritative_width <= 0 or self.authoritative_height <= 0:
            raise ValueError("authoritative resolution must be positive integers")


@dataclass(frozen=True, slots=True)
class NukeAssetInput:
    artifact_id: str
    role: Literal["source", "master", "auxiliary", "card_image", "card_auxiliary", "generated_patch", "redistortion"]
    path: str | PathLike[str]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("asset artifact_id is required")
        if self.role not in _ROLES:
            raise ValueError("asset role is unsupported")
        path = Path(self.path)
        if not path.is_file():
            raise ValueError(f"asset path does not exist: {path}")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class NukeCardInput:
    card_id: str
    artifact_id: str
    enabled: bool
    object_to_world: tuple[tuple[float, float, float, float], ...]
    auxiliary_artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_to_world", _matrix(self.object_to_world, "card transform"))
        if not isinstance(self.card_id, str) or not self.card_id or not isinstance(self.artifact_id, str) or not self.artifact_id or not isinstance(self.auxiliary_artifact_id, str) or not self.auxiliary_artifact_id:
            raise ValueError("card image and auxiliary artifact identifiers are required")
        if type(self.enabled) is not bool:
            raise TypeError("card enabled must be bool")


@dataclass(frozen=True, slots=True)
class NukeDeliveryProfile:
    """Explicit shot delivery controls; no filename/plate heuristics."""

    disabled_card_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.disabled_card_ids)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("disabled_card_ids must contain non-empty identifiers")
        if len(set(values)) != len(values):
            raise ValueError("disabled_card_ids must be unique")
        object.__setattr__(self, "disabled_card_ids", values)


@dataclass(frozen=True, slots=True)
class NukeExportResult:
    episode: RealPlateEpisode
    path: Path
    scene_bytes: bytes
    scene_text: str


def _slash(path: Path) -> str:
    return path.resolve().as_posix()


def _quote(path: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in path) or any(char in path for char in "{}\\$[]"):
        raise ValueError("Nuke Tcl text contains unsafe control, brace, or backslash characters")
    return '"' + path.replace('"', '\\"') + '"'



def _declared_colorspace(info: dict) -> str:
    """The colourspace an EXR declares, however it managed to record it.

    OIIO strips `oiio:ColorSpace` on write when the active OCIO config has no
    `colorInteropID` for the space (older studio configs such as fn-nuke_cg
    v1.0.0), so Atlas mirrors the tag into its own attribute. Note the old
    check allowed `""` for "absent" but `dict.get` returns None, so absence
    never actually matched and the validator failed under such a config.
    """
    metadata = info.get("metadata", {}) or {}
    return str(metadata.get("oiio:ColorSpace")
               or metadata.get(ATLAS_COLORSPACE_ATTR) or "")

def _validate_exr(path: Path, role: str) -> Mapping[str, Any]:
    info = inspect_exr(path)
    channels = set(info.get("channels", {}))
    if path.suffix.lower() != ".exr":
        raise ValueError(f"{role} asset must be an EXR")
    if info.get("compression", "").lower() != "zip":
        raise ValueError(f"{role} EXR must use ZIP compression")
    # These are the authoritative Task 2 validators.  They read the complete
    # raster, so channel order, exact storage types, masks, IDs, and metadata
    # cannot be reduced to a names-only inspection.
    if role == "master":
        plate = read_master_plate(path)
        if (info.get("metadata", {}).get("color_space") != "ACEScg"
                or _declared_colorspace(info) not in {"ACEScg", "lin_ap1_scene", ""}):
            raise ValueError("master EXR must declare ACEScg")
        if tuple(getattr(plate, "pixels").shape[:2]) != (info["data_window"][3], info["data_window"][2]):
            raise ValueError("master data window does not match raster dimensions")
    elif role == "auxiliary":
        channels_read = read_auxiliary_exr(path)
        missing = _AUX_REQUIRED - set(channels_read)
        if missing:
            raise ValueError(f"auxiliary EXR missing channels: {sorted(missing)}")
        validity = channels_read["validity"]
        if tuple(validity.shape) != (info["data_window"][3], info["data_window"][2]):
            raise ValueError("auxiliary data window does not match raster dimensions")
    elif role == "card_image":
        plate = read_master_plate(path)
        if (info.get("metadata", {}).get("color_space") != "ACEScg"
                or _declared_colorspace(info) not in {"ACEScg", "lin_ap1_scene", ""}):
            raise ValueError("card image EXR must declare ACEScg")
        if tuple(getattr(plate, "pixels").shape[:2]) != (info["data_window"][3], info["data_window"][2]):
            raise ValueError("card image data window does not match raster dimensions")
    elif role == "card_auxiliary":
        card = read_card_auxiliary_exr(path)
        if not {"N_object.red", "N_object.green", "N_object.blue"}.issubset(card.channels):
            raise ValueError("card EXR is missing N_object channels")
        card_shape = tuple(next(iter(card.channels.values())).shape)
        if card_shape != (info["data_window"][3], info["data_window"][2]):
            raise ValueError("card data window does not match raster dimensions")
        info["atlas_object_to_world"] = card.object_to_world
    elif role == "generated_patch":
        patch = read_generated_patch(path)
        if patch.display_window != info["display_window"] or patch.data_window != info["data_window"]:
            raise ValueError("generated patch windows do not match EXR header")
        if not patch.source_sha256:
            raise ValueError("generated patch is missing atlas:source_sha256")
        info["atlas_source_sha256"] = patch.source_sha256
    if role == "redistortion":
        required = {"distortion.u", "distortion.v"}
        if not required.issubset(channels):
            raise ValueError("redistortion EXR requires distortion.u and distortion.v channels")
        formats = info.get("channels", {})
        if any(str(formats[name]).lower() not in {"float", "float32"} for name in required):
            raise ValueError("redistortion channels must be float32")
    return info


def _artifact_index(episode: RealPlateEpisode) -> dict[str, ContentArtifact]:
    return {episode.source_artifact.artifact_id: episode.source_artifact, **{item.artifact_id: item for item in episode.artifacts}}


def _node(name: str, klass: str, body: list[str]) -> list[str]:
    return [f"{klass} {{", f" name {name}", *body, "}"]


def _matrix_comment(matrix: tuple[tuple[float, float, float, float], ...]) -> str:
    return ";".join(",".join(f"{value:.17g}" for value in row) for row in matrix)


def _matrix_block(matrix: tuple[tuple[float, float, float, float], ...]) -> str:
    return "\\n".join("     {" + " ".join(f"{value:.17g}" for value in row) + "}" for row in matrix)


def _scene_text(episode: RealPlateEpisode, camera: NukeCameraInput, assets: list[NukeAssetInput], cards: list[NukeCardInput], profile: NukeDeliveryProfile, preview_path: Path) -> str:
    lines = ["# Atlas World Evidence-First Plate Prototype", "# CoordinateSystem: metric_right_handed_y_up", "# Units: metres", "# ColourSpace: ACEScg (declared only for EXR assets that carry it)", f"# Authoritative resolution: {camera.authoritative_width}x{camera.authoritative_height} (external immutable assets)", f"# Proxy resolution: {camera.width}x{camera.height}", "# Authoritative channels: P_world.red P_world.green P_world.blue N_world.red N_world.green N_world.blue depth.Z validity object_id semantic_id card_id workspace_id", f"# Plate: {episode.plate_id}", "Root {", f" format \"{camera.width} {camera.height} 0 0 {camera.width} {camera.height} 1 {camera.width} {camera.height} 1\"", "}", ""]
    source_assets = [asset for asset in assets if asset.role == "source"]
    if source_assets:
        source = source_assets[0]
        source_hash = hashlib.sha256(Path(source.path).read_bytes()).hexdigest()
        lines.append(f"# OBSERVED source provenance only (NEF is not read by Nuke): path={_slash(Path(source.path))} sha256={source_hash}")
    for index, asset in enumerate((asset for asset in assets if asset.role != "source"), 1):
        name = {"master": "Read_Master", "auxiliary": "Read_Auxiliary", "card_image": f"Read_CardImage_{index:03d}", "card_auxiliary": f"Read_CardAux_{index:03d}", "generated_patch": f"Read_Generated_{index:03d}", "redistortion": "Read_Redistortion"}[asset.role]
        label = {"master": "OBSERVED ACEScg master", "auxiliary": "SOLVED spatial evidence (raw data)", "card_image": "OBSERVED card image ACEScg", "card_auxiliary": "SOLVED card spatial evidence (raw data)", "generated_patch": "GENERATED patch ACEScg", "redistortion": "SOLVED redistortion (raw data)"}[asset.role]
        label = f"{label} | artifact={asset.artifact_id} | sha256={hashlib.sha256(Path(asset.path).read_bytes()).hexdigest()}"
        color_roles = {"master", "card_image", "generated_patch"}
        raw_flag = " raw true" if asset.role in {"auxiliary", "card_auxiliary", "redistortion"} else ""
        lines.extend(_node(name, "Read", [f" file {_quote(_slash(Path(asset.path)))}", " disable 0", raw_flag, f" label {_quote(label)}", " colorspace ACEScg" if asset.role in color_roles else "",]))
        lines.append(f"set N_{asset.artifact_id} [stack 0]")
        lines.append("")
    # Authoritative master + sparse generated patches remain in a visible 2D
    # preview branch.  Push order mirrors the verified native Nuke stack
    # convention used by the existing exporter.
    generated = [asset for asset in assets if asset.role == "generated_patch"]
    master_id = next(asset.artifact_id for asset in assets if asset.role == "master")
    lines.append(f"push $N_{master_id}")
    current_preview = f"N_{master_id}"
    for patch_index, patch_asset in enumerate(generated, 1):
        lines.append(f"push $N_{patch_asset.artifact_id}")
        lines.append(f"push ${current_preview}")
        lines.extend(_node(f"Merge_Plate_Patches_{patch_index:03d}", "Merge2", [" inputs 2", " operation over", " label \"OBSERVED master + GENERATED sparse patch\"",]))
        lines.append(f"set N_Merge_Plate_Patches_{patch_index:03d} [stack 0]")
        current_preview = f"N_Merge_Plate_Patches_{patch_index:03d}"
    lines.append(f"push ${current_preview}")
    lines.extend(_node("Reformat_Atlas_Preview", "Reformat", [f" format \"{camera.width} {camera.height} 0 0 {camera.width} {camera.height} 1 {camera.width} {camera.height} 1\"", " label \"Nuke Indie preview | authoritative assets external\"",]))
    lines.append("set N_Reformat_Atlas_Preview [stack 0]")
    lines.append("")
    lines.extend(_node("Camera2_Atlas", "Camera2", [" useMatrix true", f" matrix {{\n{_matrix_block(camera.camera_to_world)}\n }}", f" focal {camera.focal_length_mm:.17g}", f" haperture {camera.sensor_width_mm:.17g}", " addUserKnob {20 atlas_camera_to_world l \"Atlas camera-to-world\"}", f" atlas_camera_to_world {_quote(_matrix_comment(camera.camera_to_world))}", " label \"SOLVED camera | metric RH Y-up\"",]))
    lines.append("set N_Camera2_Atlas [stack 0]")
    lines.append("")
    for index, card in enumerate(cards, 1):
        disabled = 0 if card.enabled else 1
        if card.card_id in profile.disabled_card_ids:
            disabled = 1
        card_asset = next(asset for asset in assets if asset.artifact_id == card.artifact_id)
        lines.append(f"push $N_{card_asset.artifact_id}")
        lines.extend(_node(f"Card3D_Atlas_{index:03d}", "Card", [" inputs 1", f" disable {disabled}", " useMatrix true", f" matrix {{\n{_matrix_block(card.object_to_world)}\n }}", " addUserKnob {20 atlas_card_id l \"Atlas card ID\"}", f" atlas_card_id {_quote(card.card_id)}", f" addUserKnob {{20 atlas_object_to_world l \"Atlas object-to-world\"}}", f" atlas_object_to_world {_quote(_matrix_comment(card.object_to_world))}", f" label {_quote(('OBSERVED card DISABLED' if disabled else 'OBSERVED card ENABLED') + ' | ' + card.card_id)}",]))
        lines.append(f"set N_{card.card_id} [stack 0]")
        lines.append("")
    # Recorded card geometry and camera form the native 3D branch.  The
    # explicit push/pop sequence follows the verified Nuke text convention.
    for card_index, card in enumerate(cards, 1):
        lines.append(f"push $N_{card.card_id}")
    lines.extend(_node("Scene_Atlas_Cards", "Scene", [f" inputs {len(cards)}", " label \"OBSERVED cards only\"",]))
    lines.append("set N_Scene_Atlas_Cards [stack 0]")
    lines.append("push $N_Reformat_Atlas_Preview")
    lines.append("push $N_Scene_Atlas_Cards")
    lines.append("push $N_Camera2_Atlas")
    lines.extend(_node("ScanlineRender_Atlas", "ScanlineRender", [" inputs 3", " label \"Atlas restrained parallax preview\"",]))
    lines.append("set N_ScanlineRender_Atlas [stack 0]")
    lines.append("")
    aux_id = next(asset.artifact_id for asset in assets if asset.role == "auxiliary")
    lines.append(f"push $N_{aux_id}")
    lines.extend(_node("Shuffle_P_Aliases", "Shuffle", [" inputs 1", " in P_world", " out P", " red red", " green green", " blue blue", " label \"Authoritative P_world -> P alias\"",]))
    lines.append(f"push $N_{aux_id}")
    lines.extend(_node("Shuffle_N_Aliases", "Shuffle", [" inputs 1", " in N_world", " out N", " red red", " green green", " blue blue", " label \"Authoritative N_world -> N alias\"",]))
    if any(asset.role == "redistortion" for asset in assets):
        redistort = next(asset for asset in assets if asset.role == "redistortion")
        lines.append(f"push $N_{redistort.artifact_id}")
        lines.append("push $N_ScanlineRender_Atlas")
        lines.extend(_node("STMap_Atlas_Redistort", "STMap", [" inputs 2", " uv distortion", " label \"SOLVED redistortion STMap | distortion.u/v\"",]))
    else:
        lines.append("push $N_ScanlineRender_Atlas")
    lines.extend(_node("Write_Atlas_Preview", "Write", [f" file {_quote(_slash(preview_path))}", " file_type exr", " colorspace ACEScg", " label \"PREVIEW only | authoritative assets remain external\"",]))
    lines.append("")
    lines.append("StickyNote { label \"Atlas evidence: source/master/auxiliary are immutable; generated patches are visibly GENERATED and never replace observed pixels.\" }")
    return "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"


def export_real_plate_nuke(episode: RealPlateEpisode, output_path: str | PathLike[str], *, camera: NukeCameraInput, assets: tuple[NukeAssetInput, ...] | list[NukeAssetInput], cards: tuple[NukeCardInput, ...] | list[NukeCardInput], profile: NukeDeliveryProfile | None = None) -> NukeExportResult:
    _require_world()
    if not isinstance(episode, RealPlateEpisode):
        raise TypeError("episode must be a RealPlateEpisode")
    if not isinstance(camera, NukeCameraInput):
        raise TypeError("camera must be an explicit NukeCameraInput")
    profile = profile or NukeDeliveryProfile()
    if not isinstance(profile, NukeDeliveryProfile):
        raise TypeError("profile must be an explicit NukeDeliveryProfile")
    assets = list(assets)
    cards = list(cards)
    if len({asset.artifact_id for asset in assets}) != len(assets):
        raise ValueError("asset identifiers must be unique")
    counts = {role: sum(asset.role == role for asset in assets) for role in _ROLES}
    if counts["master"] != 1 or counts["auxiliary"] != 1:
        raise ValueError("Nuke delivery requires exactly one master and one auxiliary EXR")
    if counts["source"] != 1 or counts["redistortion"] > 1:
        raise ValueError("Nuke delivery requires exactly one source and permits at most one redistortion asset")
    role_order = {role: index for index, role in enumerate(("source", "master", "auxiliary", "card_image", "card_auxiliary", "generated_patch", "redistortion"))}
    assets = sorted(assets, key=lambda item: (role_order[item.role], item.artifact_id))
    cards = sorted(cards, key=lambda item: item.card_id)
    index = _artifact_index(episode)
    inspected: dict[str, Mapping[str, Any]] = {}
    for asset in assets:
        artifact = index.get(asset.artifact_id)
        if artifact is None:
            raise ValueError(f"asset references unknown artifact: {asset.artifact_id}")
        if artifact.kind is ArtifactKind.SOURCE and asset.role != "source":
            raise ValueError("source artifact must be exported as source")
        if artifact.kind is not ArtifactKind.SOURCE and asset.role == "source":
            raise ValueError("source role requires source artifact")
        actual = hashlib.sha256(Path(asset.path).read_bytes()).hexdigest()
        if actual != artifact.sha256:
            raise ValueError(f"stale source hash: expected {artifact.sha256}, got {actual}")
        if asset.role != "source":
            inspected[asset.artifact_id] = _validate_exr(Path(asset.path), asset.role)
    master_asset = next(asset for asset in assets if asset.role == "master")
    aux_asset = next(asset for asset in assets if asset.role == "auxiliary")
    master_info, aux_info = inspected[master_asset.artifact_id], inspected[aux_asset.artifact_id]
    if master_info["data_window"] != aux_info["data_window"] or master_info["display_window"] != aux_info["display_window"]:
        raise ValueError("master and auxiliary authoritative windows must match")
    master_hash = index[master_asset.artifact_id].sha256
    for asset in assets:
        if asset.role == "generated_patch" and inspected[asset.artifact_id].get("atlas_source_sha256") != master_hash:
            raise ValueError("generated patch atlas:source_sha256 does not bind to master")
    hypotheses = {item.card_id: item for item in episode.card_hypotheses}
    image_assets = {asset.artifact_id for asset in assets if asset.role == "card_image"}
    aux_assets = {asset.artifact_id for asset in assets if asset.role == "card_auxiliary"}
    if len(cards) != len(hypotheses) or len(cards) != len(image_assets) or len(cards) != len(aux_assets):
        raise ValueError("card inputs, card image assets, card auxiliary assets, and episode hypotheses must match 1:1")
    card_image_ids = {card.artifact_id for card in cards}
    card_aux_ids = {card.auxiliary_artifact_id for card in cards}
    if card_image_ids != image_assets or card_aux_ids != aux_assets:
        raise ValueError("card image and auxiliary assets must bind exactly one-to-one to card inputs")
    if set(profile.disabled_card_ids) - set(hypotheses):
        raise ValueError("profile disables an unknown card")
    seen_cards: set[str] = set()
    for card in cards:
        if card.card_id in seen_cards:
            raise ValueError("card identifiers must be unique")
        seen_cards.add(card.card_id)
        hypothesis = hypotheses.get(card.card_id)
        if hypothesis is None:
            raise ValueError(f"card references unknown recorded hypothesis: {card.card_id}")
        if hypothesis.object_to_world.matrix != card.object_to_world:
            raise ValueError("card transform does not match recorded hypothesis")
        asset = next((item for item in assets if item.artifact_id == card.artifact_id), None)
        if asset is None or asset.role != "card_image":
            raise ValueError("card must reference a card_image asset")
        if inspected[asset.artifact_id].get("atlas_object_to_world") != card.object_to_world:
            # Card spatial metadata belongs to the paired auxiliary asset.
            aux_asset = next((item for item in assets if item.artifact_id == card.auxiliary_artifact_id), None)
            if aux_asset is None or aux_asset.role != "card_auxiliary" or inspected[aux_asset.artifact_id].get("atlas_object_to_world") != card.object_to_world:
                raise ValueError("card EXR object_to_world metadata does not match recorded transform")
    output = Path(output_path)
    preview_path = output.with_name(output.stem + "_preview.exr")
    text = _scene_text(episode, camera, assets, cards, profile, preview_path)
    payload = text.encode("utf-8")
    path = output
    if path.suffix.lower() != ".nk":
        raise ValueError("Nuke Indie export path must use .nk extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = ContentArtifact.from_bytes(artifact_id=f"EXPORT_NUKE_{episode.plate_id}", kind=ArtifactKind.EXPORT, media_type="application/x-nuke-script", uri=f"exports/{episode.plate_id}.nk", data=payload)
    updated = episode.record_export(PlateExport(f"EXPORT_NUKE_{episode.plate_id}", artifact, "nuke_indie"))
    temp = path.with_name(f".{path.name}.{artifact.sha256}.tmp")
    try:
        temp.write_bytes(payload)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return NukeExportResult(updated, path, payload, text)


__all__ = ["NukeAssetInput", "NukeCameraInput", "NukeCardInput", "NukeDeliveryProfile", "NukeExportResult", "export_real_plate_nuke"]
