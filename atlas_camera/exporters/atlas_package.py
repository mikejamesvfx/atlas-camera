"""Writing a `.atlas` package — the scene an editor opens.

`atlas_camera.format` decides what the document says; this decides what lands on
disk beside it. The split is deliberate: a headless consumer validating a
document should not have to care that plates are large or that mattes arrive
base64-encoded, and a writer should not be the second place the schema is
defined.

**The package is self-contained.** Every asset the document names is copied in
and referenced by package-relative path. A package pointing at the plate's
original location on the machine that solved it is not a scene, it is a note
about one — it stops working the moment it is moved, zipped or handed over, and
it stops working silently.

**Evidence is copied, never converted.** Mattes arrive as base64 PNG data URIs
in the solve; they are decoded and written as the same PNG bytes, not re-encoded
through an image library. Only the transport changes, so the pixels a producer
made are the pixels an editor reads, and there is no library version in the
middle to change a truncation into a round.

**The solve travels with the scene.** It goes to `atlas/`, because a package
that cannot show what its camera came from cannot be reviewed, and it is a few
kilobytes against a plate's tens of megabytes.
"""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from atlas_camera.format import (
    ATLAS_DIR,
    GEOMETRY_DIR,
    HISTORY_DIR,
    IMAGERY_DIR,
    MATTES_DIR,
    SCENE_DOCUMENT,
    digest_bytes,
    scene_document,
    validate_document,
)
from atlas_camera.format.document import utc_now
from atlas_camera.format.layout import PACKAGE_DIRS


@dataclass(slots=True)
class PackageResult:
    """What was written, and what could not be."""

    package_dir: Path
    document: Path
    files: dict[str, Path] = field(default_factory=dict)
    #: What was asked for and could not be done. A writer that quietly drops an
    #: asset produces a scene whose missing features nobody can explain.
    complaints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_dir": str(self.package_dir),
            "document": str(self.document),
            "files": {key: str(value) for key, value in self.files.items()},
            "complaints": list(self.complaints),
        }


def write_atlas_package(
    solve: Any,
    destination: str | Path,
    *,
    scene_id: str | None = None,
    name: str | None = None,
    plate_path: str | Path | None = None,
    geometry_path: str | Path | None = None,
    entity_id: str = "relief",
    solve_path: str | Path | None = None,
    observation_id: str | None = None,
    write_mattes: bool = True,
) -> PackageResult:
    """Write `solve` as a `.atlas` package at `destination`.

    `plate_path` overrides the solve's own `image_path`, which names wherever
    the plate was when the solve ran and is routinely a temporary file that has
    since gone.
    """

    package = Path(destination)
    for directory in (package, *(package / name_ for name_ in PACKAGE_DIRS)):
        directory.mkdir(parents=True, exist_ok=True)

    complaints: list[str] = []
    files: dict[str, Path] = {}

    relative_plate = _adopt_plate(package, solve, plate_path, complaints, files)
    relative_geometry = _adopt_geometry(package, geometry_path, complaints, files)
    relative_solve = _adopt_solve(package, solve_path, files)

    entities = (
        [_entity_document(entity_id, relative_geometry, observation_id)]
        if relative_geometry
        else []
    )

    document = scene_document(
        solve,
        scene_id=scene_id or package.stem,
        name=name,
        entities=entities,
        plate_path=relative_plate,
        observation_id=observation_id,
        source=_source_block(solve, relative_solve),
    )

    layers: list[dict[str, Any]] = []
    if write_mattes:
        layers = _adopt_mattes(package, solve, document["planes"], complaints, files)
    document["layers"] = layers

    # Validated BEFORE it is written. A package that fails the format's own
    # checklist should never reach an artist's disk, because by the time the
    # editor refuses it they have already been handed it.
    validate_document(document)

    target = package / SCENE_DOCUMENT
    _write_json(target, document)
    _write_ledger(package / HISTORY_DIR / "ledger.jsonl", document)

    return PackageResult(
        package_dir=package, document=target, files=files, complaints=complaints
    )


# -- assets ------------------------------------------------------------------


def _adopt_plate(
    package: Path,
    solve: Any,
    override: str | Path | None,
    complaints: list[str],
    files: dict[str, Path],
) -> str | None:
    candidate = override or getattr(solve, "image_path", None)
    if not candidate:
        # ABSENT is as load-bearing as EXPIRED and just as invisible: a package
        # with no plate opens as an inert scene, and without a note the artist
        # has nothing to read but the silence.
        complaints.append(
            "no plate was given and the solve names none, so the package has "
            "no plate; nothing in the editor can project or be measured "
            "against it. Connect plate_path."
        )
        return None

    source = Path(str(candidate))
    if not source.is_file():
        complaints.append(
            f"the plate {source.name} was named but is not at {source}; the "
            "package has no plate, so nothing in the editor can project or be "
            "measured against it"
        )
        return None

    relative = f"{IMAGERY_DIR}/{source.name}"
    destination = package / relative
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    files["plate"] = destination
    return relative


def _adopt_geometry(
    package: Path,
    candidate: str | Path | None,
    complaints: list[str],
    files: dict[str, Path],
) -> str | None:
    if not candidate:
        complaints.append(
            "no geometry was given, so the package carries a camera and planes "
            "but nothing to edit. Connect relief_mesh_path from "
            "AtlasExportReliefMesh."
        )
        return None
    source = Path(str(candidate))
    if not source.is_file():
        complaints.append(f"the geometry {source} is not there; no entity was written")
        return None
    relative = f"{GEOMETRY_DIR}/{source.name}"
    destination = package / relative
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    files["geometry"] = destination
    return relative


def _adopt_solve(
    package: Path, candidate: str | Path | None, files: dict[str, Path]
) -> str | None:
    if not candidate:
        return None
    source = Path(str(candidate))
    if not source.is_file():
        return None
    relative = f"{ATLAS_DIR}/{source.name}"
    destination = package / relative
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    files["solve"] = destination
    return relative


def _adopt_mattes(
    package: Path,
    solve: Any,
    planes: Sequence[dict[str, Any]],
    complaints: list[str],
    files: dict[str, Path],
) -> list[dict[str, Any]]:
    """Mattes out of the JSON and onto disk, one file per layer.

    Eleven full-frame 8K mattes inline as base64 in a JSON document is a
    document nothing can open twice. The encoding is untouched — single-channel
    8-bit PNG, white keeps — because re-encoding here would put an image
    library's rounding between the producer's pixels and the editor's.
    """

    sources = getattr(solve, "projection_sources", None) or []
    if not sources or not planes:
        return []

    layers: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        payload = _matte_bytes(getattr(source, "mask_b64", None))
        if payload is None:
            continue
        if index >= len(planes):
            complaints.append(
                f"projection source {index} has a matte but no plane to attach it "
                "to; it was not written, because a layer without a surface names "
                "nothing"
            )
            continue

        plane_id = planes[index]["plane_id"]
        relative = f"{MATTES_DIR}/{plane_id}.png"
        destination = package / relative
        destination.write_bytes(payload)
        files[f"matte:{plane_id}"] = destination

        layers.append(
            {
                "layer_id": f"layer_{plane_id}",
                "plane_id": plane_id,
                "matte_path": relative,
                "matte_digest": digest_bytes(payload),
                # A matte is data: alpha is coverage, never colour. Associating
                # one multiplies a coverage signal by itself, and the result
                # reads as a softer matte rather than as an error.
                "alpha_mode": "straight",
                "priority": float(getattr(source, "priority", 0.0) or 0.0),
                "visible": True,
                "metadata": {"source_name": str(getattr(source, "name", "") or "")},
            }
        )
    return layers


def _matte_bytes(value: Any) -> bytes | None:
    """The PNG bytes out of a data URI, or None.

    Decoded, not re-encoded: these are the producer's own bytes and they arrive
    at the editor unchanged.
    """

    if not value:
        return None
    text = str(value)
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        return base64.b64decode(text, validate=True)
    except Exception:  # noqa: BLE001 - a malformed matte is not a crash
        return None


# -- documents ---------------------------------------------------------------


def _entity_document(
    entity_id: str, geometry_path: str, observation_id: str | None
) -> dict[str, Any]:
    """The relief mesh, as the thing a depth solve OBSERVED.

    Not `SOLVED`, and never `USER_CREATED`: a depth model saw this surface in
    the photograph. The tears in it are part of that observation rather than
    damage to be repaired on import, which is why the editor's one construction
    into un-photographed space has to be licensed by a plane.
    """

    return {
        "entity_id": entity_id,
        "name": entity_id,
        "kind": "RELIEF",
        "parent_id": None,
        "placement": {
            "transform": {
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "source_hypothesis_id": None,
            "revision": 0,
        },
        "hypotheses": [],
        "geometry": {
            "path": geometry_path,
            "node": None,
            "format": Path(geometry_path).suffix.lstrip(".").lower() or "glb",
            "metadata": {},
        },
        "material": {},
        "visible": True,
        "observation_state": "OBSERVED",
        "semantic_class": None,
        # Unknown, and left unknown. A confidence invented here is
        # indistinguishable downstream from one that was measured.
        "confidence": None,
        "visibility": None,
        "source_observation_id": observation_id,
        "provenance": [],
        "metadata": {},
        "created_at": utc_now(),
    }


def _source_block(solve: Any, solve_path: str | None) -> dict[str, Any]:
    """Where this scene came from, so a reader can go back to it."""

    block: dict[str, Any] = {"producer": "atlas-camera"}
    version = getattr(solve, "atlas_version", None)
    try:
        from atlas_camera import __version__ as package_version

        block["producer_version"] = str(version or package_version)
    except Exception:  # noqa: BLE001 - a stamp failure never fails an export
        if version:
            block["producer_version"] = str(version)
    method = getattr(solve, "source_method", None)
    if method:
        block["solve_method"] = str(method)
    if solve_path:
        block["solve"] = solve_path
    return block


def _write_json(target: Path, document: dict[str, Any]) -> None:
    """Atomically, so an interrupted write never leaves half a scene."""

    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.write("\n")
    temporary.replace(target)


def _write_ledger(target: Path, document: dict[str, Any]) -> None:
    """One line: this package was produced.

    History is append-only, and the first entry is the import itself. A ledger
    that begins at the artist's first edit cannot say what the scene was before
    they touched it.
    """

    entry = {
        "operation": "produce_package",
        "actor": "atlas-camera",
        "timestamp": document["created_at"],
        "result": {
            "scene_id": document["scene_id"],
            "schema_version": document["schema_version"],
            "planes": len(document.get("planes") or []),
            "layers": len(document.get("layers") or []),
            "entities": len(document.get("entities") or []),
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
