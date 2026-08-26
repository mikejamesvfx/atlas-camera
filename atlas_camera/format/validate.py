"""Is this a well-formed `.atlas` document?

The checklist a producer's output must pass. It runs on the writing side, before
anything reaches disk, because the alternative is an artist discovering the
problem when the editor refuses to open a package they have already been handed.

**It refuses rather than repairs.** Every violation here is a statement that
turned out not to be true, and quietly correcting one destroys the information
that it was wrong. An unknown `completion_policy` is the clearest case: silently
downgrading it to `none` produces a document indistinguishable from one whose
surfaces were classified correctly, so a licence the reader cannot evaluate is
an error rather than a licence it declines to use.
"""

from __future__ import annotations

from typing import Any

from atlas_camera.format.version import READABLE_SCHEMA_VERSIONS

#: `occlusion_graph.COMPLETION_POLICIES`, mirrored so that validation does not
#: drag numpy in behind it. `test_the_policies_match_the_occlusion_graph` fails
#: if the two ever diverge.
COMPLETION_POLICIES = (
    "none",
    "extend_plane",
    "room_envelope",
    "extrude_profile",
    "conservative_proxy",
    "bridge_discontinuity",
    "backdrop",
)

#: `atlas_camera.core.scene_health`'s spellings, unchanged.
SCALE_STATUSES = ("measured", "manual", "assumed", "unknown")

ALPHA_MODES = ("straight", "associated")
PROJECTION_ROLES = ("foreground", "cleanplate_background")


class FormatError(ValueError):
    """Raised when a document does not conform to the format."""


def validate_document(document: dict[str, Any]) -> None:
    """Raise on the first violation, naming what and where."""

    problems = collect_problems(document)
    if problems:
        raise FormatError("; ".join(problems))


def collect_problems(document: dict[str, Any]) -> list[str]:
    """Every violation, so one call reports all of them rather than the first."""

    problems: list[str] = []
    _check_version(document, problems)
    _check_camera(document, problems)
    planes = _check_planes(document, problems)
    _check_layers(document, planes, problems)
    _check_entities(document, problems)
    _check_revisions(document, problems)
    _check_derived(document, problems)
    _check_scale(document, problems)
    return problems


def _check_version(document: dict[str, Any], problems: list[str]) -> None:
    version = str(document.get("schema_version", ""))
    if version not in READABLE_SCHEMA_VERSIONS:
        problems.append(
            f"schema_version {version!r} is not one of {sorted(READABLE_SCHEMA_VERSIONS)}"
        )


def _check_camera(document: dict[str, Any], problems: list[str]) -> None:
    camera = document.get("camera")
    if camera is None:
        return
    if not isinstance(camera, dict):
        problems.append("camera is not an object")
        return
    rotation = camera.get("rotation")
    if not _is_orthonormal(rotation):
        # A rotation that is not orthonormal is not a rotation. Every position
        # derived through it is wrong by an amount nothing reports.
        problems.append("camera.rotation is not a proper orthonormal matrix")
    plate = camera.get("plate_path")
    if plate and _is_absolute(str(plate)):
        problems.append(
            f"camera.plate_path {plate!r} is absolute; package assets are "
            "referenced by package-relative path or the package cannot be moved"
        )


def _check_planes(document: dict[str, Any], problems: list[str]) -> set[str]:
    identifiers: set[str] = set()
    for index, plane in enumerate(document.get("planes") or []):
        where = f"planes[{index}]"
        if not isinstance(plane, dict):
            problems.append(f"{where} is not an object")
            continue

        plane_id = str(plane.get("plane_id", ""))
        if not plane_id:
            problems.append(f"{where} has no plane_id")
        elif plane_id in identifiers:
            problems.append(f"{where} repeats plane_id {plane_id!r}")
        else:
            identifiers.add(plane_id)

        policy = plane.get("completion_policy", "none")
        if policy not in COMPLETION_POLICIES:
            problems.append(
                f"{where} has completion_policy {policy!r}, which is not one of "
                f"{list(COMPLETION_POLICIES)}. An unrecognised policy is a licence "
                "this reader cannot evaluate, and treating it as 'none' would look "
                "identical to a correctly classified surface"
            )

        confidence = plane.get("confidence")
        if confidence is not None and not _in_unit_range(confidence):
            problems.append(f"{where} has confidence {confidence!r}, outside [0, 1]")

        normal = plane.get("normal")
        if not _is_vector3(normal):
            problems.append(f"{where} has no three-component normal")
        elif abs(_length(normal) - 1.0) > 1e-3:
            problems.append(f"{where} normal is not unit length")
    return identifiers


def _check_layers(
    document: dict[str, Any], planes: set[str], problems: list[str]
) -> None:
    for index, layer in enumerate(document.get("layers") or []):
        where = f"layers[{index}]"
        if not isinstance(layer, dict):
            problems.append(f"{where} is not an object")
            continue
        plane_id = str(layer.get("plane_id", ""))
        if plane_id not in planes:
            problems.append(f"{where} names plane {plane_id!r}, which is not in planes")
        alpha = layer.get("alpha_mode")
        if alpha is not None and alpha not in ALPHA_MODES:
            problems.append(f"{where} has alpha_mode {alpha!r}")
        elif alpha == "associated":
            # A matte is data. Premultiplying one multiplies a coverage signal
            # by itself, and the result looks like a softer matte rather than
            # like an error.
            problems.append(
                f"{where} is a matte and declares alpha_mode 'associated'; "
                "mattes are straight, and associated is for float plates"
            )
        if layer.get("matte_path") and _is_absolute(str(layer["matte_path"])):
            problems.append(f"{where} matte_path is absolute")


def _check_entities(document: dict[str, Any], problems: list[str]) -> None:
    entities = document.get("entities") or []
    identifiers = {
        str(entity.get("entity_id"))
        for entity in entities
        if isinstance(entity, dict) and entity.get("entity_id")
    }
    parents: dict[str, str | None] = {}
    for index, entity in enumerate(entities):
        where = f"entities[{index}]"
        if not isinstance(entity, dict):
            problems.append(f"{where} is not an object")
            continue
        entity_id = str(entity.get("entity_id", ""))
        if not entity_id:
            problems.append(f"{where} has no entity_id")
            continue
        parent = entity.get("parent_id")
        parents[entity_id] = str(parent) if parent else None
        if parent and str(parent) not in identifiers:
            problems.append(f"{where} names parent {parent!r}, which is not present")
        geometry = entity.get("geometry")
        if isinstance(geometry, dict) and geometry.get("path"):
            if _is_absolute(str(geometry["path"])):
                problems.append(f"{where} geometry.path is absolute")
        material = entity.get("material") or {}
        projection = material.get("projection") if isinstance(material, dict) else None
        if projection is not None:
            if not isinstance(projection, dict):
                problems.append(f"{where} material.projection is not an object")
                continue
            plate_path = projection.get("plate_path")
            if not isinstance(plate_path, str) or not plate_path:
                problems.append(f"{where} material.projection has no plate_path")
            elif _is_absolute(plate_path):
                problems.append(f"{where} material.projection.plate_path is absolute")
            if not isinstance(projection.get("observation_id"), str) or not projection["observation_id"]:
                problems.append(f"{where} material.projection has no observation_id")
            if projection.get("role") not in PROJECTION_ROLES:
                problems.append(
                    f"{where} material.projection role {projection.get('role')!r} is not one of {list(PROJECTION_ROLES)}"
                )

    for entity_id in parents:
        if _has_cycle(entity_id, parents):
            problems.append(f"entity {entity_id!r} is its own ancestor")


def _check_revisions(document: dict[str, Any], problems: list[str]) -> None:
    block = document.get("revisions") or {}
    if not isinstance(block, dict):
        problems.append("revisions is not an object")
        return
    revisions = block.get("revisions") or []
    by_id = {
        str(item.get("revision_id")): item
        for item in revisions
        if isinstance(item, dict) and item.get("revision_id")
    }
    for revision_id, item in by_id.items():
        derived = item.get("derived_from")
        if derived and str(derived) not in by_id:
            problems.append(
                f"revision {revision_id!r} derives from {derived!r}, which is absent"
            )
        digest = str(item.get("digest", ""))
        if len(digest) != 64:
            problems.append(f"revision {revision_id!r} has no sha256 digest")
    for revision_id in by_id:
        if _revision_cycle(revision_id, by_id):
            problems.append(f"revision {revision_id!r} is its own ancestor")
    for key, value in (block.get("selected") or {}).items():
        if str(value) not in by_id:
            problems.append(f"selected[{key!r}] names absent revision {value!r}")


def _check_derived(document: dict[str, Any], problems: list[str]) -> None:
    for index, artifact in enumerate(document.get("derived") or []):
        if not isinstance(artifact, dict):
            problems.append(f"derived[{index}] is not an object")
            continue
        if not artifact.get("depends_on"):
            # An artifact that depends on nothing can never go stale, which
            # means it can never be known to be current either.
            problems.append(f"derived[{index}] has an empty depends_on")


def _check_scale(document: dict[str, Any], problems: list[str]) -> None:
    scale = document.get("scale")
    if scale is None:
        return
    if not isinstance(scale, dict):
        problems.append("scale is not an object")
        return
    status = scale.get("status")
    if status not in SCALE_STATUSES:
        problems.append(f"scale.status {status!r} is not one of {list(SCALE_STATUSES)}")
    confidence = scale.get("confidence")
    if confidence is not None and not _in_unit_range(confidence):
        problems.append(f"scale.confidence {confidence!r} is outside [0, 1]")


# -- predicates --------------------------------------------------------------


def _is_absolute(path: str) -> bool:
    #: Asked without `pathlib`, because a POSIX reader must recognise a Windows
    #: absolute path as absolute and vice versa — `Path("C:/x").is_absolute()`
    #: is False on Linux, which would let exactly the unmovable package this
    #: check exists to catch straight through.
    if path.startswith(("/", "\\")):
        return True
    return len(path) > 1 and path[1] == ":"


def _is_vector3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(item, (int, float)) for item in value
    )


def _length(vector: Any) -> float:
    return sum(float(item) * float(item) for item in vector) ** 0.5


def _in_unit_range(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= number <= 1.0 and number == number  # NaN fails the comparison


def _is_orthonormal(matrix: Any) -> bool:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        return False
    rows: list[list[float]] = []
    for row in matrix:
        if not _is_vector3(row):
            return False
        rows.append([float(item) for item in row])

    for index, row in enumerate(rows):
        if abs(_length(row) - 1.0) > 1e-4:
            return False
        for other in rows[index + 1 :]:
            if abs(sum(a * b for a, b in zip(row, other))) > 1e-4:
                return False

    # A right-handed frame has determinant +1. Determinant -1 is a reflection,
    # which is orthonormal and still mirrors the scene.
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    return abs(determinant - 1.0) <= 1e-4


def _has_cycle(start: str, parents: dict[str, str | None]) -> bool:
    seen: set[str] = set()
    current: str | None = start
    while current:
        if current in seen:
            return True
        seen.add(current)
        current = parents.get(current)
    return False


def _revision_cycle(start: str, by_id: dict[str, Any]) -> bool:
    seen: set[str] = set()
    current: str | None = start
    while current:
        if current in seen:
            return True
        seen.add(current)
        item = by_id.get(current)
        derived = item.get("derived_from") if isinstance(item, dict) else None
        current = str(derived) if derived else None
    return False
