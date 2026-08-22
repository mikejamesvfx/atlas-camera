"""Atlas World evidence-first plate handoff nodes.

This module is deliberately a narrow serialization boundary.  It opens World
episode JSON and appends typed attempt/export records through the immutable
World APIs; it does not load images, invoke models, or return observed pixels.
ComfyUI remains an optional adapter and the World package never imports it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# GUARDED: the World schema ships in the private atlas-world distribution,
# which atlas-camera does not depend on. node_registry imports this module
# unconditionally, so a hard import here would take every other Atlas node down
# with it on a machine that has only the public package. See _require_world.
try:                                       # pragma: no cover - install-shaped
    from atlas_world.orchestration import Plan
    from atlas_world.real_plate import (
        ContentArtifact, PlateExport, ProcessingAttempt, RealPlateEpisode,
    )
    _WORLD_IMPORT_ERROR = None
except ImportError as _exc:                # pragma: no cover - install-shaped
    Plan = ContentArtifact = PlateExport = ProcessingAttempt = None
    RealPlateEpisode = None
    _WORLD_IMPORT_ERROR = _exc


def world_available() -> bool:
    """Whether the private World package could be imported."""
    return _WORLD_IMPORT_ERROR is None


def _require_world() -> None:
    """Raise with an install hint when the private World package is absent.

    A node that cannot work must say so in the node, not disappear from the
    menu: a missing node reads to an artist as a broken install of Atlas, while
    an error names the one package that is actually missing.
    """
    if _WORLD_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Atlas evidence-plate nodes require the atlas-world package, which "
            "is distributed separately from atlas-camera. Install it "
            "(pip install -e path/to/atlas-world) and restart ComfyUI."
        ) from _WORLD_IMPORT_ERROR


def _json_object(value: str, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a JSON object")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} JSON is invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _json_object_or_path(value: str, label: str) -> dict[str, Any]:
    """Read JSON content or a user-supplied relative/absolute manifest path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a JSON object or manifest path")
    candidate = Path(value)
    try:
        if candidate.is_file():
            value = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} path could not be read") from exc
    return _json_object(value, label)


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _episode_value(value: str) -> RealPlateEpisode:
    _require_world()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("episode must be a JSON string or path")
    candidate = Path(value)
    try:
        raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    except OSError as exc:
        raise ValueError("episode path could not be read") from exc
    try:
        return RealPlateEpisode.from_dict(_json_object(raw, "episode"))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"episode is invalid: {exc}") from exc


def _ledger_binding(payload: Mapping[str, Any], plate: RealPlateEpisode) -> None:
    if payload.get("phase") != "lock":
        raise ValueError("plan must be in lock phase")
    if not Plan.verify_manifest_digest(payload):
        raise ValueError("plan manifest digest is invalid")
    if payload.get("plate_id") != plate.plate_id or payload.get("ledger_plate_id") != plate.plate_id:
        raise ValueError("locked plan is bound to a different plate")
    if payload.get("ledger_source_content_address") != plate.source_artifact.content_address:
        raise ValueError("locked plan source content address does not match plate")
    current = tuple(sorted((plate.source_artifact.artifact_id, *(a.artifact_id for a in plate.artifacts))))
    declared = tuple(sorted(payload.get("ledger_artifact_ids", ())))
    if declared != current:
        raise ValueError("locked plan artifact ledger does not match plate")


def _plan_models(payload: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    models = payload.get("model_policy")
    if not isinstance(models, list):
        raise ValueError("locked plan model policy is invalid")
    result = set()
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("locked plan model policy is invalid")
        fields = (model.get("model_id"), model.get("version"), model.get("model_hash"), model.get("license"))
        if any(not isinstance(value, str) or not value for value in fields):
            raise ValueError("locked plan model identity is incomplete")
        if len(fields[2]) != 64 or any(c not in "0123456789abcdef" for c in fields[2]):
            raise ValueError("locked plan model hash is invalid")
        result.add(fields)
    return result


class AtlasOpenRealPlate:
    """Open a serialized RealPlateEpisode without opening its media artifact."""

    RETURN_TYPES = ("ATLAS_REAL_PLATE", "STRING", "STRING")
    RETURN_NAMES = ("plate", "episode_json", "report")
    FUNCTION = "open"
    CATEGORY = "Atlas/11 · Evidence Plate"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"episode": ("STRING", {"default": "", "multiline": True})}}

    def open(self, episode: str):
        plate = _episode_value(episode)
        payload = plate.to_json()
        report = _canonical({
            "status": "opened",
            "plate_id": plate.plate_id,
            "source_artifact_id": plate.source_artifact.artifact_id,
            "observed_pixels_loaded": False,
        })
        return plate, payload, report


class AtlasReadLockedPlatePlan:
    """Validate and pass through a Plan manifest at the lock boundary."""

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("locked_plan_json", "report")
    FUNCTION = "read"
    CATEGORY = "Atlas/11 · Evidence Plate"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"default": "", "multiline": True}),
            # Appended after the original widget to preserve saved workflows.
            "plate": ("ATLAS_REAL_PLATE",),
        }}

    def read(self, plan: str, plate: RealPlateEpisode):
        _require_world()
        if not isinstance(plate, RealPlateEpisode):
            raise ValueError("plate must be an ATLAS_REAL_PLATE value")
        payload = _json_object_or_path(plan, "plan")
        _ledger_binding(payload, plate)
        canonical = _canonical(payload)
        return canonical, _canonical({"status": "locked", "plate_id": payload.get("plate_id")})


class AtlasRecordPlateAttempt:
    """Append a typed, immutable ProcessingAttempt and optional derived artifacts."""

    RETURN_TYPES = ("ATLAS_REAL_PLATE", "STRING", "STRING")
    RETURN_NAMES = ("plate", "result_json", "report")
    FUNCTION = "record"
    CATEGORY = "Atlas/11 · Evidence Plate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("ATLAS_REAL_PLATE",),
                "result": ("STRING", {"default": "", "multiline": True}),
                # Appended after the original widgets for workflow compatibility.
                "locked_plan": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }

    def record(self, plate: RealPlateEpisode, result: str, locked_plan: str):
        _require_world()
        if not isinstance(plate, RealPlateEpisode):
            raise ValueError("plate must be an ATLAS_REAL_PLATE value")
        plan_payload = _json_object_or_path(locked_plan, "locked_plan")
        _ledger_binding(plan_payload, plate)
        allowed_models = _plan_models(plan_payload)
        payload = _json_object_or_path(result, "result")
        try:
            attempt = ProcessingAttempt.from_dict(payload["attempt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"result.attempt is invalid: {exc}") from exc
        identity = (attempt.model.model_id, attempt.model.version, attempt.model.sha256, attempt.model.license)
        if identity not in allowed_models:
            raise ValueError("attempt model identity is not approved by locked plan")
        updated = plate
        artifacts = payload.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("result.artifacts must be a list")
        parsed_artifacts = []
        for item in artifacts:
            try:
                artifact = ContentArtifact.from_dict(item)
                if artifact.kind.value != "derived":
                    raise ValueError("attempt artifacts must be derived")
                parsed_artifacts.append(artifact)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"result artifact is invalid: {exc}") from exc
        if tuple(attempt.output_artifact_ids) != tuple(a.artifact_id for a in parsed_artifacts):
            raise ValueError("attempt output IDs must exactly match result artifacts")
        for artifact in parsed_artifacts:
            updated = updated.record_artifact(artifact)
        try:
            updated = updated.record_attempt(attempt)
            output = updated.to_dict()["attempts"][-1]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"attempt cannot be recorded: {exc}") from exc
        out = {"attempt": output, "artifacts": [a.to_dict() for a in parsed_artifacts]}
        return updated, _canonical(out), _canonical({"status": "recorded", "attempt_id": attempt.attempt_id})


class AtlasExportPlateHandoff:
    """Append a typed PlateExport; file writing is owned by the exporter."""

    RETURN_TYPES = ("ATLAS_REAL_PLATE", "STRING", "STRING")
    RETURN_NAMES = ("plate", "export_json", "report")
    FUNCTION = "export"
    CATEGORY = "Atlas/11 · Evidence Plate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("ATLAS_REAL_PLATE",),
                "export": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def export(self, plate: RealPlateEpisode, export: str):
        _require_world()
        if not isinstance(plate, RealPlateEpisode):
            raise ValueError("plate must be an ATLAS_REAL_PLATE value")
        try:
            item = PlateExport.from_dict(_json_object_or_path(export, "export"))
            updated = plate.record_export(item)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"export cannot be recorded: {exc}") from exc
        return updated, _canonical(item.to_dict()), _canonical({"status": "export_recorded", "export_id": item.export_id})


class AtlasRealPlateToScene:
    """Roundtrip an episode's RANSAC planes and card alphas into an ATLAS_SOLVE.

    Both live outside the solve until now: the planes keep their extents and
    metadata in ``ransac_planes.json`` while the ledger keeps only the 4x4, and
    the card alphas live in cropped RGBA EXRs. Every plane transform is verified
    against its index-aligned ledger hypothesis before it is promoted, so a
    tampered side-car fails loudly instead of quietly re-posing the scene.

    Planes CLOBBER the PROXY_ROLE set, matching every other derive node — a
    re-run is idempotent rather than accumulating duplicates. Cards APPEND as
    ProjectionSource layers, which is the additive half of the same doctrine.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "roundtrip"
    CATEGORY = "Atlas/11 · Evidence Plate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("ATLAS_REAL_PLATE",),
                "assets_dir": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "scene_json_path": ("STRING", {"default": "", "multiline": False}),
                # Appended last; recording mints a NEW episode revision, so it
                # is never a side effect of rebuilding a scene.
                "record_in_ledger": ("BOOLEAN", {"default": False}),
            },
        }

    def roundtrip(self, plate: RealPlateEpisode, assets_dir: str, scene_json_path: str = "",
                  record_in_ledger: bool = False):
        _require_world()
        from atlas_world.plate_scene_roundtrip import (
            PlateRoundtripError, build_plate_scene, record_scene_artifact, write_plate_scene,
        )

        if not isinstance(plate, RealPlateEpisode):
            raise ValueError("plate must be an ATLAS_REAL_PLATE value")
        if not isinstance(assets_dir, str) or not assets_dir.strip():
            raise ValueError("assets_dir must name the episode's assets directory")
        target = scene_json_path.strip() if isinstance(scene_json_path, str) else ""
        if record_in_ledger and not target:
            raise ValueError("record_in_ledger needs scene_json_path: only a written "
                             "artifact can be content-addressed into the ledger")
        try:
            result = build_plate_scene(plate, assets_dir.strip())
        except PlateRoundtripError as exc:
            raise ValueError(f"plate scene roundtrip failed: {exc}") from exc

        report = dict(result.report)
        if target:
            report["written_to"] = str(write_plate_scene(result, target))
        if record_in_ledger:
            try:
                plate = record_scene_artifact(
                    plate, target, episode_root=Path(assets_dir.strip()).parent)
            except PlateRoundtripError as exc:
                raise ValueError(f"scene cannot be recorded: {exc}") from exc
            report["recorded_artifact_id"] = plate.artifacts[-1].artifact_id
            report["episode_revision_events"] = len(plate.events)
        report["status"] = "roundtripped"
        return result.solve, _canonical(report)


__all__ = [
    "AtlasOpenRealPlate", "AtlasReadLockedPlatePlan",
    "AtlasRecordPlateAttempt", "AtlasExportPlateHandoff",
    "AtlasRealPlateToScene",
]
