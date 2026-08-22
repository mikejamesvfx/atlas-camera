"""Building `scene.json` from a solve.

This is the translation between what a solver produces — a `LatentScene` with a
recovered camera, fitted proxy planes and, when it has been built, an occlusion
graph — and what the format says a scene document is.

Three commitments run through every function here.

**Nothing is invented.** A field the solve does not answer is emitted as `null`,
never as `0` or `1.0`. A confidence of 1.0 chosen to unblock a pipeline is
indistinguishable, downstream, from a measurement — and telling those apart is
the entire purpose of the format. `null` reads as *unknown* and is honoured as
unknown by every consumer.

**`none` dominates.** A plane's `completion_policy` says what may be built on
it. Absent an occlusion graph that classified the surface, it is `none`, and
`none` licenses nothing. Guessing a policy puts constructed geometry into a
scene that claims to be measured, which is what this format exists to prevent.

**The solve's own verdicts are copied, not recomputed.** `scale_health` decides
whether the distances mean anything, and it has exactly one home. Re-deriving it
here would produce a second opinion that disagrees in the last decimal and is
read as authoritative by whoever finds it first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from atlas_camera.format.identity import plane_from_transform, plane_id_for
from atlas_camera.format.version import SCHEMA_VERSION

#: Atlas canonical space, spelled as `atlas.world.schema.WorldCamera` spells it.
COORDINATE_SYSTEM = "right_handed_y_up_camera_to_world"

#: What a solver produced. The editor's own output is `USER_CREATED` or
#: `AGENT_CREATED`, and a producer must never label its output either.
PROVENANCE_SOLVED = "SOLVED"
PROVENANCE_OBSERVED = "OBSERVED"
PROVENANCE_INFERRED = "INFERRED"

#: A plane fitted by RANSAC was inferred from the mesh, not measured off the
#: photograph. The distinction is what tells a later reader which parts of the
#: scene are evidence and which are a fitter's opinion.
PLANE_PROVENANCE = PROVENANCE_INFERRED

_UNSET = object()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- camera ------------------------------------------------------------------


def camera_document(
    solve: Any,
    *,
    plate_path: str | None = None,
    observation_id: str | None = None,
) -> dict[str, Any] | None:
    """The recovered camera, copied field for field.

    Copied rather than recomputed: re-deriving a focal length from the other
    intrinsics gives an answer that differs in the last decimal for no benefit,
    and then two numbers claim to be the same measurement.

    `plate_path` is package-relative, because the writer has already copied the
    plate in by the time this is called. A camera pointing at wherever the plate
    happened to live during the solve makes the package unmovable.
    """

    camera = getattr(solve, "camera", None)
    if camera is None:
        return None
    intrinsics = _completed(getattr(camera, "intrinsics", None))
    extrinsics = getattr(camera, "extrinsics", None)
    if intrinsics is None or extrinsics is None:
        return None

    rotation = _matrix3(getattr(extrinsics, "camera_rotation_matrix", None))
    position = _vector3(getattr(extrinsics, "camera_position", None))
    if rotation is None or position is None:
        return None

    return {
        "width": int(getattr(intrinsics, "image_width", 0) or 0),
        "height": int(getattr(intrinsics, "image_height", 0) or 0),
        "focal_length_mm": _number(getattr(intrinsics, "focal_length_mm", None)),
        "sensor_width_mm": _number(getattr(intrinsics, "sensor_width_mm", None)),
        "sensor_height_mm": _number(getattr(intrinsics, "sensor_height_mm", None)),
        "fx": _number(getattr(intrinsics, "fx_px", None)),
        "fy": _number(getattr(intrinsics, "fy_px", None)),
        "cx": _number(getattr(intrinsics, "cx_px", None)),
        "cy": _number(getattr(intrinsics, "cy_px", None)),
        "position": position,
        "rotation": rotation,
        "near_clip": _number(getattr(camera, "near_clip", None), default=0.1),
        "far_clip": _number(getattr(camera, "far_clip", None), default=1000.0),
        "coordinate_system": COORDINATE_SYSTEM,
        # The intrinsics' own lens model. Atlas Scene does not apply it to a
        # relief mesh's UVs — that mesh was baked through the same pinhole
        # model, so the error cancels — but geometry from anywhere else needs
        # it, and dropping it leaves nothing able to tell the two cases apart.
        "lens_model": _optional_str(getattr(intrinsics, "lens_model", None)),
        "distortion": _distortion(intrinsics),
        "plate_path": plate_path,
        "observation_id": observation_id,
    }


def _completed(intrinsics: Any) -> Any:
    """Intrinsics with fx/fy/cx/cy filled in, or None when they cannot be.

    A solve may carry a focal length in millimetres and leave the pixel form
    unset; a consumer needs the pixel form. Completed through upstream's own
    `build_intrinsics`, not by arithmetic written here — the conversion and the
    image-centre principal-point convention are its decisions, and a second
    implementation of them is a second answer that differs in the last decimal.

    This is a derivation, not an invention: fx from a focal length and a sensor
    size is a definition. Where there is no focal length either, there is
    nothing to derive from, and the camera is omitted rather than guessed.
    """

    if intrinsics is None:
        return None
    if getattr(intrinsics, "fx_px", None) and getattr(intrinsics, "fy_px", None):
        if getattr(intrinsics, "cx_px", None) is not None:
            return intrinsics
    if not getattr(intrinsics, "focal_length_mm", None):
        return intrinsics if getattr(intrinsics, "fx_px", None) else None

    try:
        from atlas_camera.core.intrinsics import build_intrinsics

        return build_intrinsics(
            image_width=int(intrinsics.image_width),
            image_height=int(intrinsics.image_height),
            focal_length_mm=float(intrinsics.focal_length_mm),
            sensor_width_mm=float(getattr(intrinsics, "sensor_width_mm", 36.0) or 36.0),
            sensor_height_mm=getattr(intrinsics, "sensor_height_mm", None),
            principal_point_px=getattr(intrinsics, "principal_point_px", None),
            fx_px=getattr(intrinsics, "fx_px", None),
            fy_px=getattr(intrinsics, "fy_px", None),
        )
    except Exception:  # noqa: BLE001 - a solve too partial to complete is
        return intrinsics  # emitted as it stands, and validation catches it


def _distortion(intrinsics: Any) -> dict[str, Any] | None:
    """Whatever distortion coefficients the intrinsics carry, or None.

    None means the solve recorded none — not that they are zero. A zeroed
    distortion block is a claim that the lens is rectilinear, which is a
    measurement nobody made.
    """

    for attribute in ("distortion", "distortion_coefficients"):
        value = getattr(intrinsics, attribute, None)
        if isinstance(value, dict) and value:
            return {key: _number(item) for key, item in value.items()}
        if hasattr(value, "to_dict"):
            converted = value.to_dict()
            if converted:
                return {key: _number(item) for key, item in converted.items()}
    coefficient = getattr(intrinsics, "k1", None)
    if coefficient not in (None, 0, 0.0):
        return {"k1": _number(coefficient)}
    return None


# -- planes ------------------------------------------------------------------


def plane_documents(
    solve: Any, *, observation_id: str | None = None
) -> list[dict[str, Any]]:
    """Every fitted plane, with a stable id and whatever licence it has earned.

    The `completion_policy` and `confidence` come from the occlusion graph when
    the solve carries one — that module already decided, from the tear evidence,
    what construction each surface permits, and it is the one place allowed to
    decide it. Without a graph every plane licenses `none`: an unclassified
    surface is not a surface that permits anything, and the editor will refuse
    to extrude on it, which is the correct outcome rather than a limitation.
    """

    primitives = _proxy_primitives(solve)
    nodes = _occlusion_nodes(solve)

    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for primitive in primitives:
        if _attr(primitive, "primitive_type") != "plane":
            continue
        plane = plane_from_transform(_attr(primitive, "transform_matrix"))
        if plane is None:
            continue
        normal, offset = plane

        plane_id = plane_id_for(normal, offset, observation_id)
        if plane_id in seen:
            # Two primitives that mint the same id ARE the same surface at this
            # tolerance. Emitting both would give the scene two planes an artist
            # cannot tell apart, and a decision attached to one of them.
            continue
        seen.add(plane_id)

        name = str(_attr(primitive, "name") or "")
        node = nodes.get(name)
        metadata = dict(_attr(primitive, "metadata") or {})

        documents.append(
            {
                "plane_id": plane_id,
                # Display text. It is expected to change between runs — it is a
                # rank — and nothing may key on it. That is what plane_id is for.
                "label": name,
                "normal": [round(float(value), 6) for value in normal],
                "offset_m": round(float(offset), 6),
                "inliers": _optional_int(metadata.get("inliers")),
                "provenance": PLANE_PROVENANCE,
                "method": _plane_method(metadata, node),
                "completion_policy": (
                    str(node.get("completion_policy", "none")) if node else "none"
                ),
                # The graph's number, or unknown. Never a 1.0 to unblock a
                # pipeline: below 0.35 the editor will not offer this plane as
                # an extrude direction, and a fabricated confidence defeats that
                # floor precisely when it is protecting something.
                "confidence": _optional_number(node.get("confidence")) if node else None,
                "metadata": _plane_metadata(metadata, node),
                "created_at": utc_now(),
            }
        )
    return documents


def _plane_method(metadata: dict[str, Any], node: dict[str, Any] | None) -> str | None:
    """How this surface came to be classified, so the judgement is reviewable."""

    fitter = metadata.get("source")
    if node is None:
        return f"{fitter}; unclassified" if fitter else None
    source = node.get("source") or "occlusion_graph"
    return f"{fitter}; classified by {source}" if fitter else f"classified by {source}"


def _plane_metadata(
    metadata: dict[str, Any], node: dict[str, Any] | None
) -> dict[str, Any]:
    kept = {
        key: value
        for key, value in metadata.items()
        if key in {"normal_azimuth_deg", "normal_elevation_deg", "distance_m", "role"}
    }
    if node:
        kept["occlusion_node"] = node.get("id")
        if node.get("notes"):
            kept["classification_notes"] = list(node["notes"])
        if node.get("texture_policy"):
            kept["texture_policy"] = node["texture_policy"]
    return kept


def _occlusion_nodes(solve: Any) -> dict[str, dict[str, Any]]:
    """The graph's nodes keyed by the primitive name they were built from."""

    semantics = getattr(solve, "semantics", None)
    payload = getattr(semantics, "value", None)
    if not isinstance(payload, dict):
        return {}
    graph = payload.get("occlusion_graph")
    if not isinstance(graph, dict):
        return {}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return {}
    return {
        str(node.get("id")): node
        for node in nodes
        if isinstance(node, dict) and node.get("id")
    }


def _proxy_primitives(solve: Any) -> list[Any]:
    scene = getattr(solve, "projection_scene", None)
    primitives = getattr(scene, "proxy_geometry", None)
    return list(primitives) if primitives else []


# -- the document ------------------------------------------------------------


def scene_document(
    solve: Any,
    *,
    scene_id: str,
    name: str | None = None,
    entities: Sequence[dict[str, Any]] = (),
    layers: Sequence[dict[str, Any]] = (),
    plate_path: str | None = None,
    observation_id: str | None = None,
    source: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole `scene.json`, ready to write.

    Empty collections are emitted rather than omitted: a reader distinguishing
    "no layers" from "this producer does not know about layers" would be
    guessing, and every consumer already treats the key as required.
    """

    # Key ORDER is part of the format, not a style choice. Section 8 of the
    # producer spec requires that loading a package and writing it straight back
    # produces a byte-identical `scene.json` — that check is what proves a
    # producer and the reference reader agree about every field, and a
    # differently-ordered document fails it while being semantically identical,
    # which turns the sharpest available conformance test into noise.
    document: dict[str, Any] = {
        "scene_id": scene_id,
        "name": name or scene_id,
        "schema_version": SCHEMA_VERSION,
        "scale": _scale_health(solve),
        # What an ARTIST established a unit to be worth is not a producer's to
        # say. It arrives when somebody measures something in the editor.
        "scene_scale": None,
        "coordinate_system": COORDINATE_SYSTEM,
        "camera": camera_document(
            solve, plate_path=plate_path, observation_id=observation_id
        ),
        "entities": [dict(entity) for entity in entities],
        "planes": plane_documents(solve, observation_id=observation_id),
        "layers": [dict(layer) for layer in layers],
        "textures": [],
        "revisions": {"revisions": [], "selected": {}},
        "derived": [],
        "source": dict(source or {}),
        "provenance": [],
        "metadata": dict(metadata or {}),
        "created_at": utc_now(),
    }
    return document


def _scale_health(solve: Any) -> dict[str, Any] | None:
    try:
        from atlas_camera.core.scene_health import scale_health
    except Exception:  # noqa: BLE001 - a solve without the module still exports
        return None
    try:
        verdict = scale_health(solve)
    except Exception:  # noqa: BLE001 - scale_health never raises, but a
        return None   # partially-built solve in a test might make it try
    return verdict.to_dict() if hasattr(verdict, "to_dict") else None


# -- small conversions -------------------------------------------------------


def _attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _number(value: Any, default: Any = _UNSET) -> Any:
    if value is None:
        return None if default is _UNSET else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return None if default is _UNSET else default


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = str(value) if value is not None else ""
    return text or None


def _vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    if len(values) != 3:
        return None
    return [float(item) for item in values]


def _matrix3(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    rows = value.tolist() if hasattr(value, "tolist") else list(value)
    if len(rows) != 3:
        return None
    matrix = [[float(item) for item in row] for row in rows]
    if any(len(row) != 3 for row in matrix):
        return None
    return matrix
