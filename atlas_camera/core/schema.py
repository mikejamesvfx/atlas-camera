"""Portable Atlas latent-scene data schema.

Core convention:
- World coordinates are right-handed and Y-up by default.
- Image coordinates use origin top-left, x right, y down.
- DCC-specific conventions are converted at adapter boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from typing import Any, ClassVar

from atlas_camera.core.confidence import ConfidenceModel

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def identity_matrix3() -> Matrix3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def identity_matrix4() -> Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _matrix3_from(value: Any | None) -> Matrix3:
    if value is None:
        return identity_matrix3()
    rows = tuple(tuple(float(col) for col in row) for row in value)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Expected a 3x3 matrix.")
    return rows  # type: ignore[return-value]


def _matrix4_from(value: Any | None) -> Matrix4:
    if value is None:
        return identity_matrix4()
    rows = tuple(tuple(float(col) for col in row) for row in value)
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("Expected a 4x4 matrix.")
    return rows  # type: ignore[return-value]


@dataclass(slots=True)
class AtlasIntrinsics:
    image_width: int
    image_height: int
    focal_length_mm: float | None = None
    sensor_width_mm: float = 36.0
    sensor_height_mm: float | None = None
    principal_point_px: Point2D | None = None
    fx_px: float | None = None
    fy_px: float | None = None
    cx_px: float | None = None
    cy_px: float | None = None
    lens_model: str = "pinhole"
    distortion: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasIntrinsics":
        principal = data.get("principal_point_px")
        return cls(
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            focal_length_mm=data.get("focal_length_mm"),
            sensor_width_mm=float(data.get("sensor_width_mm", 36.0)),
            sensor_height_mm=data.get("sensor_height_mm"),
            principal_point_px=tuple(principal) if principal is not None else None,  # type: ignore[arg-type]
            fx_px=data.get("fx_px"),
            fy_px=data.get("fy_px"),
            cx_px=data.get("cx_px"),
            cy_px=data.get("cy_px"),
            lens_model=data.get("lens_model", "pinhole"),
            distortion=dict(data.get("distortion", {})),
        )


@dataclass(slots=True)
class AtlasExtrinsics:
    camera_position: Point3D = (0.0, 0.0, 0.0)
    camera_rotation_matrix: Matrix3 = field(default_factory=identity_matrix3)
    camera_world_matrix: Matrix4 = field(default_factory=identity_matrix4)
    camera_view_matrix: Matrix4 = field(default_factory=identity_matrix4)
    coordinate_system: str = "right_handed"
    up_axis: str = "Y"
    projection_convention: str = "Atlas pinhole camera, image origin top-left."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasExtrinsics":
        return cls(
            camera_position=tuple(float(v) for v in _as_tuple(data.get("camera_position", (0.0, 0.0, 0.0)))),  # type: ignore[arg-type]
            camera_rotation_matrix=_matrix3_from(data.get("camera_rotation_matrix")),
            camera_world_matrix=_matrix4_from(data.get("camera_world_matrix")),
            camera_view_matrix=_matrix4_from(data.get("camera_view_matrix")),
            coordinate_system=data.get("coordinate_system", "right_handed"),
            up_axis=data.get("up_axis", "Y"),
            projection_convention=data.get(
                "projection_convention",
                "Atlas pinhole camera, image origin top-left.",
            ),
        )


@dataclass(slots=True)
class LatentCamera:
    intrinsics: AtlasIntrinsics
    extrinsics: AtlasExtrinsics = field(default_factory=AtlasExtrinsics)
    name: str = "atlas_camera"
    confidence: ConfidenceModel = field(default_factory=ConfidenceModel.for_latent_camera)
    notes: list[str] = field(default_factory=list)
    focal_length_inferred: bool = False
    seed: int | None = None
    schema_version: str = "0.2"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentCamera":
        return cls(
            intrinsics=AtlasIntrinsics.from_dict(data["intrinsics"]),
            extrinsics=AtlasExtrinsics.from_dict(data.get("extrinsics", {})),
            name=data.get("name", "atlas_camera"),
            confidence=ConfidenceModel.from_dict(data.get("confidence")),
            notes=list(data.get("notes", [])),
            focal_length_inferred=bool(data.get("focal_length_inferred", False)),
            seed=data.get("seed"),
            schema_version=data.get("schema_version", "0.2"),
        )


@dataclass(slots=True)
class AtlasVanishingPoint:
    position_px: Point2D
    direction_label: str | None = None
    confidence: float = 0.0
    supporting_lines: list[tuple[Point2D, Point2D]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasVanishingPoint":
        lines = [
            (tuple(line[0]), tuple(line[1]))  # type: ignore[list-item]
            for line in data.get("supporting_lines", [])
        ]
        return cls(
            position_px=tuple(data["position_px"]),  # type: ignore[arg-type]
            direction_label=data.get("direction_label"),
            confidence=float(data.get("confidence", 0.0)),
            supporting_lines=lines,
        )


@dataclass(slots=True)
class AtlasHorizon:
    line_coefficients: tuple[float, float, float]
    endpoints_px: tuple[Point2D, Point2D] | None = None
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasHorizon":
        endpoints = data.get("endpoints_px")
        return cls(
            line_coefficients=tuple(float(v) for v in data["line_coefficients"]),  # type: ignore[arg-type]
            endpoints_px=(tuple(endpoints[0]), tuple(endpoints[1])) if endpoints else None,  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class AtlasShotCam:
    """Project-level render/output camera format — sensor + lens + target
    resolution, analogous to Nuke/Resolve project settings. Intrinsics-only
    (no position): it describes what the FINAL render/export should look
    like, decoupled from whatever sensor/lens/aspect any individual solved
    photo happened to imply. Never touches how a photo gets projected onto
    geometry — see AtlasMergeGeometry/AtlasBlockoutViewport for how this is
    consumed without disturbing per-source texture-sampling cameras.
    """
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 24.0
    focal_length_mm: float = 35.0
    resolution_long_edge_px: int = 1920

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "focal_length_mm": self.focal_length_mm,
            "resolution_long_edge_px": self.resolution_long_edge_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasShotCam":
        return cls(
            sensor_width_mm=float(data.get("sensor_width_mm", 36.0)),
            sensor_height_mm=float(data.get("sensor_height_mm", 24.0)),
            focal_length_mm=float(data.get("focal_length_mm", 35.0)),
            resolution_long_edge_px=int(data.get("resolution_long_edge_px", 1920)),
        )


@dataclass(slots=True)
class AtlasProxyPrimitive:
    name: str
    primitive_type: str
    transform_matrix: Matrix4 = field(default_factory=identity_matrix4)
    dimensions: Point3D = (1.0, 1.0, 1.0)
    material: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasProxyPrimitive":
        return cls(
            name=data["name"],
            primitive_type=data["primitive_type"],
            transform_matrix=_matrix4_from(data.get("transform_matrix")),
            dimensions=tuple(float(v) for v in data.get("dimensions", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
            material=data.get("material"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class AtlasProjectionScene:
    coordinate_system: str = "right_handed"
    up_axis: str = "Y"
    proxy_geometry: list[AtlasProxyPrimitive] = field(default_factory=list)
    landmarks: list[dict[str, Any]] = field(default_factory=list)
    debug_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasProjectionScene":
        return cls(
            coordinate_system=data.get("coordinate_system", "right_handed"),
            up_axis=data.get("up_axis", "Y"),
            proxy_geometry=[
                AtlasProxyPrimitive.from_dict(item)
                for item in data.get("proxy_geometry", [])
            ],
            landmarks=list(data.get("landmarks", [])),
            debug_metadata=dict(data.get("debug_metadata", {})),
        )


@dataclass(slots=True)
class ProjectionSource:
    """An extra camera + AI novel-view image + its own geometry, layered as a
    projection patch to texture areas the primary recovered camera could not see.

    Built by ``AtlasAddPatchView``: the ``camera`` is orbit-constructed around the
    scene pivot (``camera_math.orbit_camera``) so it shares the primary's world
    frame; ``image_b64`` is the multi-angle-LoRA novel view for that angle (a
    data-URI, kept JSON-safe like the relief mesh already is); ``proxy_geometry``
    is that view's own depth-derived geometry in the patch camera's frame.
    ``priority`` orders blending (higher wins; the primary is implicitly highest).
    """

    camera: LatentCamera
    name: str = "patch"
    image_b64: str | None = None
    proxy_geometry: list[AtlasProxyPrimitive] = field(default_factory=list)
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    distance_scale: float = 1.0
    priority: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectionSource":
        return cls(
            camera=LatentCamera.from_dict(data["camera"]),
            name=data.get("name", "patch"),
            image_b64=data.get("image_b64"),
            proxy_geometry=[
                AtlasProxyPrimitive.from_dict(item)
                for item in data.get("proxy_geometry", [])
            ],
            azimuth_deg=float(data.get("azimuth_deg", 0.0)),
            elevation_deg=float(data.get("elevation_deg", 0.0)),
            distance_scale=float(data.get("distance_scale", 1.0)),
            priority=float(data.get("priority", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class AtlasCameraKeyframe:
    """One waypoint of a ``AtlasCameraPath`` — an eye/target/up pose plus timing.

    Authored client-side in the blockout viewport's Camera Path mode (fly nav,
    not the clamped orbit control) and sampled server-side by
    ``camera_path.sample_camera_path`` via Catmull-Rom + easing into a full
    ``AtlasExtrinsics`` per output frame, reusing ``camera_math.look_at_view_matrix``
    so every sampled pose shares the same view/world matrix convention as the
    rest of Atlas.
    """

    frame_index: int
    position: Point3D
    target: Point3D
    up: Point3D = (0.0, 1.0, 0.0)
    fov_deg: float | None = None
    easing: str = "linear"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasCameraKeyframe":
        return cls(
            frame_index=int(data["frame_index"]),
            position=tuple(float(v) for v in data["position"]),  # type: ignore[arg-type]
            target=tuple(float(v) for v in data["target"]),  # type: ignore[arg-type]
            up=tuple(float(v) for v in data.get("up", (0.0, 1.0, 0.0))),  # type: ignore[arg-type]
            fov_deg=data.get("fov_deg"),
            easing=data.get("easing", "linear"),
        )


@dataclass(slots=True)
class AtlasCameraPath:
    """A keyframed camera move (orbit/dolly/pan) for testing projection under motion.

    ``keyframes`` must be sorted by ``frame_index`` (ascending, no duplicates)
    for ``camera_path.sample_camera_path`` to interpolate correctly.
    """

    keyframes: list[AtlasCameraKeyframe] = field(default_factory=list)
    fps: float = 24.0
    frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasCameraPath":
        return cls(
            keyframes=[
                AtlasCameraKeyframe.from_dict(item)
                for item in sorted(data.get("keyframes", []), key=lambda k: k["frame_index"])
            ],
            fps=float(data.get("fps", 24.0)),
            frame_count=int(data.get("frame_count", 0)),
        )


@dataclass(slots=True)
class LatentComponent:
    """Future scene component slot with explicit recovery metadata."""

    value: Any | None = None
    confidence: float = 0.0
    editable: bool = True
    exportable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any | None) -> "LatentComponent":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            return cls(value=data)
        return cls(
            value=data.get("value"),
            confidence=float(data.get("confidence", 0.0)),
            editable=bool(data.get("editable", True)),
            exportable=bool(data.get("exportable", False)),
            metadata=dict(data.get("metadata", {})),
            warnings=list(data.get("warnings", [])),
        )


@dataclass(slots=True)
class LatentScene:
    camera: LatentCamera
    image_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    vanishing_points: list[AtlasVanishingPoint] = field(default_factory=list)
    horizon_line: AtlasHorizon | None = None
    confidence: float = 0.0
    source_method: str = "manual"
    known_intrinsics_used: bool = False
    projection_scene: AtlasProjectionScene = field(default_factory=AtlasProjectionScene)
    projection_workspace: AtlasProjectionScene | None = None
    projection_sources: list[ProjectionSource] = field(default_factory=list)
    depth: LatentComponent = field(default_factory=LatentComponent)
    geometry: LatentComponent = field(default_factory=LatentComponent)
    lighting: LatentComponent = field(default_factory=LatentComponent)
    semantics: LatentComponent = field(default_factory=LatentComponent)
    landmarks: list[dict[str, Any]] = field(default_factory=list)
    debug_metadata: dict[str, Any] = field(default_factory=dict)
    shot_cam: AtlasShotCam | None = None
    schema_version: ClassVar[str] = "0.2"

    def __post_init__(self) -> None:
        if self.image_width is None:
            self.image_width = self.camera.intrinsics.image_width
        if self.image_height is None:
            self.image_height = self.camera.intrinsics.image_height
        if self.projection_workspace is None:
            self.projection_workspace = self.projection_scene

    def to_dict(self) -> dict[str, Any]:
        data = _json_ready(self)
        data["schema_version"] = self.schema_version
        data["scene_type"] = "latent_scene"
        data["confidence_detail"] = self.camera.confidence.to_dict()
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentScene":
        camera_data = dict(data["camera"])
        if "confidence" not in camera_data and data.get("confidence_detail"):
            camera_data["confidence"] = data["confidence_detail"]
        elif "confidence" not in camera_data and "confidence" in data:
            camera_data["confidence"] = {
                "global_score": data.get("confidence", 0.0),
                "individual_metrics": {},
            }
        camera = LatentCamera.from_dict(camera_data)
        projection_scene_data = data.get("projection_scene") or data.get("projection_workspace") or {}
        projection_workspace_data = data.get("projection_workspace") or projection_scene_data
        return cls(
            camera=camera,
            image_path=data.get("image_path"),
            image_width=data.get("image_width") or camera.intrinsics.image_width,
            image_height=data.get("image_height") or camera.intrinsics.image_height,
            vanishing_points=[
                AtlasVanishingPoint.from_dict(item)
                for item in data.get("vanishing_points", [])
            ],
            horizon_line=AtlasHorizon.from_dict(data["horizon_line"])
            if data.get("horizon_line")
            else None,
            confidence=float(data.get("confidence", 0.0)),
            source_method=data.get("source_method", "manual"),
            known_intrinsics_used=bool(data.get("known_intrinsics_used", False)),
            projection_scene=AtlasProjectionScene.from_dict(projection_scene_data),
            projection_workspace=AtlasProjectionScene.from_dict(projection_workspace_data),
            projection_sources=[
                ProjectionSource.from_dict(item)
                for item in data.get("projection_sources", [])
            ],
            depth=LatentComponent.from_dict(data.get("depth")),
            geometry=LatentComponent.from_dict(data.get("geometry")),
            lighting=LatentComponent.from_dict(data.get("lighting")),
            semantics=LatentComponent.from_dict(data.get("semantics")),
            landmarks=list(data.get("landmarks", [])),
            debug_metadata=dict(data.get("debug_metadata", {})),
            shot_cam=AtlasShotCam.from_dict(data["shot_cam"]) if data.get("shot_cam") else None,
        )

    @classmethod
    def from_json(cls, payload: str) -> "LatentScene":
        return cls.from_dict(json.loads(payload))


AtlasCamera = LatentCamera
AtlasSolve = LatentScene
