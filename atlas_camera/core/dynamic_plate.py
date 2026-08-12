"""Dynamic Plates: a dynamic region of a solved still, separable from the camera.

A `DynamicPlate` represents one time-varying region (ocean, clouds, smoke, ...)
of an Atlas-solved still image. Atlas owns the camera and geometry; a temporal
generator supplies temporal content only for the known region. The generated
sequence is projected through the FIXED crop camera onto known receiver
geometry, so the artist's render camera stays free to move — the camera move is
never baked into the generated video.

v0.1 scope: `water` is the supported case; the receiver is a horizontal plane
(large distant body of water, moderate camera move, small wave displacement
relative to scene scale — temporal appearance, not simulated water geometry).
Other semantic types exist as contracts/status placeholders.

Host-agnostic core: stdlib always; numpy only inside the matte helpers via
`_require_numpy` (install with ``pip install -e .[vision]``).
"""
from __future__ import annotations

import copy
import json
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from atlas_camera.core.camera_crop import CropTransform, RegionROI
from atlas_camera.core.schema import (
    AtlasProxyPrimitive,
    LatentCamera,
    _json_ready,
)

# ---------------------------------------------------------------------------
# Semantic types (spec §4). Plain string constants — the repo has no Enums.
# Lowercase to match every other Atlas status/provenance string. Append-only.
DYNAMIC_REGION_TYPES = (
    "water", "cloud", "smoke", "fire", "foliage", "cloth", "actor", "generic",
)

# Plate lifecycle statuses.
PLATE_STATUS_DRAFT = "draft"
PLATE_STATUS_READY = "ready"
PLATE_STATUS_GENERATED = "generated"
PLATE_STATUS_FAILED = "failed"

# Generator availability sentinel (spec §32).
GENERATOR_NOT_AVAILABLE = "not_available"

# Failure/status codes (spec §31), lowercase snake like scene_health codes.
REGION_INVALID = "region_invalid"
CAMERA_CROP_FAILURE = "camera_crop_failure"
RECEIVER_GEOMETRY_UNAVAILABLE = "receiver_geometry_unavailable"
GENERATOR_UNAVAILABLE = "generator_unavailable"
GENERATION_FAILURE = "generation_failure"
FRAME_SEQUENCE_INCOMPLETE = "frame_sequence_incomplete"
PROJECTION_SETUP_FAILURE = "projection_setup_failure"
EXPORT_FAILURE = "export_failure"

# Editable default prompt for water (spec §18: a preset, not a magic string).
WATER_PROMPT_DEFAULT = (
    "preserve coastline and large-scale composition; animate natural ocean "
    "wave motion, foam, surface variation, specular movement and subtle "
    "swell; no camera movement; no alteration to static land or architecture"
)

_DEFAULT_PROVENANCE = {
    "source_region": "observed",
    "crop_camera": "derived_from_solve",
    "receiver_geometry": "derived_from_solve",
    "generated_frames": "generated",
}


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Dynamic-plate matte utilities require numpy. Install with:\n"
            "    pip install -e .[vision]") from exc
    return np


# ---------------------------------------------------------------------------
# Matte utilities (numpy-gated)

def matte_bbox(matte: Any, *, threshold: float = 0.5) -> RegionROI | None:
    """Tight bounding ROI of the matte's active pixels, or None when empty.

    Accepts float mattes in [0, 1] or uint8 in [0, 255]; ``threshold`` is
    always expressed in the [0, 1] domain.
    """
    np = _require_numpy()
    m = np.asarray(matte)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == np.uint8:
        m = m.astype(np.float32) / 255.0
    active = m > float(threshold)
    if not bool(active.any()):
        return None
    rows = np.flatnonzero(active.any(axis=1))
    cols = np.flatnonzero(active.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    return RegionROI(x=x0, y=y0, width=x1 - x0 + 1, height=y1 - y0 + 1)


def validate_matte_dimensions(matte_shape: Any, image_width: int,
                              image_height: int) -> None:
    """Raise ValueError when the matte raster does not match the plate."""
    shape = tuple(int(v) for v in matte_shape)
    if len(shape) < 2:
        raise ValueError(f"Matte shape {shape} is not a 2D raster")
    h, w = shape[0], shape[1]
    if (w, h) != (int(image_width), int(image_height)):
        raise ValueError(
            f"Matte is {w}x{h} but the source plate is "
            f"{image_width}x{image_height}; they must match exactly")


def feather_matte(matte: Any, radius_px: float) -> Any:
    """Soften a matte edge: three separable box blurs ~ a gaussian.

    A soft matte is MULTIPLIED downstream, never thresholded; feathering here
    only widens support so the generator sees context past the hard boundary
    (spec §14). Returns float32 in [0, 1].
    """
    np = _require_numpy()
    m = np.asarray(matte)
    if m.dtype == np.uint8:
        m = m.astype(np.float32) / 255.0
    m = m.astype(np.float32, copy=True)
    radius = int(round(float(radius_px)))
    if radius <= 0:
        return m
    # box kernel width per pass so three passes approximate sigma ~ radius/2
    width = max(1, radius)

    def _box(arr, axis):
        # cumsum-based box filter (edge-padded), O(N) per axis — an 8K matte
        # is the normal case, so no per-row python loops.
        k = 2 * width + 1
        pad = [(0, 0), (0, 0)]
        pad[axis] = (width + 1, width)
        padded = np.pad(arr, pad, mode="edge")
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
        hi = np.take(csum, np.arange(k, k + arr.shape[axis]), axis=axis)
        lo = np.take(csum, np.arange(0, arr.shape[axis]), axis=axis)
        return ((hi - lo) / k).astype(np.float32)

    for _ in range(3):
        m = _box(m, 1)
        m = _box(m, 0)
    return np.clip(m, 0.0, 1.0)


def crop_image_region(image: Any, roi: RegionROI) -> Any:
    """Slice an HxW or HxWxC array to the ROI."""
    return image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]


@dataclass(slots=True)
class ReceiverGeometry:
    """The known scene geometry that receives the temporal projection.

    v0.1: a plane (`primitive` carries the Atlas Y-up world transform +
    dimensions). `path` is a package-relative exported mesh when written to
    disk. The receiver stays part of the Atlas scene — the generated sequence
    attaches to it, never to the artist camera.
    """

    kind: str = "plane"
    primitive: AtlasProxyPrimitive | None = None
    path: str | None = None
    coordinate_system: str = "right_handed"
    up_axis: str = "Y"
    provenance: str = "derived_from_solve"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReceiverGeometry | None":
        if not data:
            return None
        prim = data.get("primitive")
        return cls(
            kind=data.get("kind", "plane"),
            primitive=AtlasProxyPrimitive.from_dict(prim) if prim else None,
            path=data.get("path"),
            coordinate_system=data.get("coordinate_system", "right_handed"),
            up_axis=data.get("up_axis", "Y"),
            provenance=data.get("provenance", "derived_from_solve"),
        )


# ---------------------------------------------------------------------------
# Validation

@dataclass(slots=True)
class PlateValidationIssue:
    severity: str  # "fail" | "warn"
    code: str
    message: str


def _issue(code: str, message: str, *, severity: str = "fail",
           ) -> PlateValidationIssue:
    return PlateValidationIssue(severity=severity, code=code, message=message)


def frame_sequence_report(frame_paths: Any, *, expected_count: int,
                          expected_size: tuple[int, int] | None = None,
                          ) -> list[PlateValidationIssue]:
    """Check a generated frame sequence for completeness and consistency.

    Dimension checks need Pillow; when it is absent they degrade to a warning
    rather than blocking (the count/existence checks are always exact).
    """
    issues: list[PlateValidationIssue] = []
    paths = [Path(p) for p in frame_paths]
    if len(paths) != int(expected_count):
        issues.append(_issue(
            FRAME_SEQUENCE_INCOMPLETE,
            f"Sequence has {len(paths)} frame paths but metadata expects "
            f"{expected_count}"))
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        issues.append(_issue(
            FRAME_SEQUENCE_INCOMPLETE,
            f"{len(missing)} frame(s) missing on disk: {missing[:4]}"))
    if expected_size is not None and paths and not missing:
        try:
            from PIL import Image
        except ImportError:
            issues.append(_issue(
                "frame_dimensions_unverified",
                "Pillow unavailable; frame dimensions not verified "
                "(pip install -e .[image])", severity="warn"))
        else:
            bad = []
            for p in paths:
                with Image.open(p) as im:
                    if tuple(im.size) != tuple(expected_size):
                        bad.append(f"{p.name}={im.size[0]}x{im.size[1]}")
            if bad:
                issues.append(_issue(
                    "frame_dimensions_mismatch",
                    f"Frames differ from expected "
                    f"{expected_size[0]}x{expected_size[1]}: {bad[:4]}"))
    return issues


def validate_dynamic_plate(plate: "DynamicPlate", *,
                           package_dir: Any = None,
                           matte_shape: Any = None,
                           frame_paths: Any = None,
                           ) -> list[PlateValidationIssue]:
    """Validate a DynamicPlate (spec §30). Returns issues; empty = clean."""
    issues: list[PlateValidationIssue] = []

    if plate.semantic_type not in DYNAMIC_REGION_TYPES:
        issues.append(_issue(REGION_INVALID,
                             f"Unknown semantic type {plate.semantic_type!r}"))
    if plate.source_width <= 0 or plate.source_height <= 0:
        issues.append(_issue(REGION_INVALID, "Source image size missing"))
    if package_dir is not None and plate.source_image:
        pkg = Path(package_dir)
        recorded = plate.metadata.get("source_image_path")
        candidates = [pkg / plate.source_image, Path(plate.source_image),
                      pkg / "source" / "crop.png"]
        if recorded:
            candidates.insert(0, Path(recorded))
        if not any(p.exists() for p in candidates):
            issues.append(_issue(REGION_INVALID,
                                 f"Source image not found: {plate.source_image}"))
    roi = plate.source_roi
    if roi is None:
        issues.append(_issue(REGION_INVALID, "Plate has no source ROI"))
    else:
        if (roi.x < 0 or roi.y < 0
                or roi.x + roi.width > plate.source_width
                or roi.y + roi.height > plate.source_height):
            issues.append(_issue(
                REGION_INVALID,
                f"ROI {roi.to_dict()} lies outside the "
                f"{plate.source_width}x{plate.source_height} source"))
    if matte_shape is not None:
        try:
            validate_matte_dimensions(matte_shape, plate.source_width,
                                      plate.source_height)
        except ValueError as exc:
            issues.append(_issue(REGION_INVALID, str(exc)))

    cam = plate.crop_camera
    if cam is None:
        issues.append(_issue(CAMERA_CROP_FAILURE, "Plate has no crop camera"))
    else:
        intr = cam.intrinsics
        values = [intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px]
        if any(v is None or not math.isfinite(float(v)) for v in values):
            issues.append(_issue(CAMERA_CROP_FAILURE,
                                 "Crop camera intrinsics are not finite"))
        elif float(intr.fx_px) <= 0 or float(intr.fy_px) <= 0:
            issues.append(_issue(CAMERA_CROP_FAILURE,
                                 "Crop camera focal must be positive"))
        ct = plate.crop_transform
        expected_wh = None
        if ct is not None:
            expected_wh = (ct.output_width, ct.output_height)
        elif roi is not None:
            expected_wh = (roi.width, roi.height)
        if expected_wh is not None and (
                intr.image_width, intr.image_height) != expected_wh:
            issues.append(_issue(
                CAMERA_CROP_FAILURE,
                f"Crop camera raster {intr.image_width}x{intr.image_height} "
                f"does not match the crop output {expected_wh[0]}x{expected_wh[1]}"))
        extr_values = [v for row in cam.extrinsics.camera_view_matrix for v in row]
        if any(not math.isfinite(float(v)) for v in extr_values):
            issues.append(_issue(CAMERA_CROP_FAILURE,
                                 "Crop camera view matrix is not finite"))

    if plate.receiver is None or (plate.receiver.primitive is None
                                  and not plate.receiver.path):
        issues.append(_issue(RECEIVER_GEOMETRY_UNAVAILABLE,
                             "Plate has no receiver geometry"))
    elif package_dir is not None and plate.receiver.path:
        rec_path = Path(package_dir) / plate.receiver.path
        if not rec_path.exists():
            issues.append(_issue(RECEIVER_GEOMETRY_UNAVAILABLE,
                                 f"Receiver geometry file missing: "
                                 f"{plate.receiver.path}"))

    if plate.frame_rate <= 0:
        issues.append(_issue("frame_rate_invalid",
                             f"Frame rate {plate.frame_rate} must be positive"))
    if plate.frame_end < plate.frame_start:
        issues.append(_issue("frame_range_invalid",
                             f"frame_end {plate.frame_end} < frame_start "
                             f"{plate.frame_start}"))
    if not plate.status:
        issues.append(_issue("status_missing",
                             "Plate status must be explicit"))
    if not plate.color_metadata:
        issues.append(_issue("color_metadata_missing",
                             "Color-space metadata is required",
                             severity="warn"))
    if frame_paths is not None:
        issues.extend(frame_sequence_report(
            frame_paths, expected_count=plate.frame_count))
    return issues


def crop_intrinsics_for_plate(camera: LatentCamera, roi: RegionROI,
                              *, output_width: int | None = None,
                              output_height: int | None = None,
                              ) -> LatentCamera:
    """Derive the crop camera: cropped (and optionally resized) intrinsics,
    identical pose. The pose never changes — a crop is a window, not a move."""
    from atlas_camera.core.camera_crop import crop_intrinsics, scale_intrinsics

    intr = crop_intrinsics(camera.intrinsics, roi)
    if output_width is not None and output_height is not None and (
            int(output_width), int(output_height)) != (roi.width, roi.height):
        intr = scale_intrinsics(intr, int(output_width), int(output_height))
    return LatentCamera(
        intrinsics=intr,
        extrinsics=copy.deepcopy(camera.extrinsics),
        name=f"{camera.name}_crop",
        confidence=copy.deepcopy(camera.confidence),
        focal_length_inferred=camera.focal_length_inferred,
        seed=camera.seed,
    )


# ---------------------------------------------------------------------------
# Receiver geometry (pure stdlib — no numpy needed for a horizontal plane)

def pixel_ray_world(camera: LatentCamera, px: float, py: float,
                    ) -> tuple[tuple[float, float, float],
                               tuple[float, float, float]]:
    """World-space (origin, unit direction) of a pixel's viewing ray.

    Camera-frame direction is ``[(u-cx)/fx, -(v-cy)/fy, -1]`` (image origin
    top-left, camera looks down -Z); rotated to world by the cam->world 4x4
    (`camera_world_matrix` — always the full matrix, never the bare 3x3).
    """
    from atlas_camera.core.camera_spec import CameraSpec

    spec = CameraSpec.from_intrinsics(camera.intrinsics)
    if not spec.has_focal:
        raise ValueError("pixel_ray_world needs a solved focal length")
    d_cam = ((float(px) - spec.cx) / spec.fx,
             -(float(py) - spec.cy) / spec.fy,
             -1.0)
    world = camera.extrinsics.camera_world_matrix
    d_world = tuple(
        world[i][0] * d_cam[0] + world[i][1] * d_cam[1] + world[i][2] * d_cam[2]
        for i in range(3))
    length = math.sqrt(sum(v * v for v in d_world)) or 1.0
    direction = tuple(v / length for v in d_world)
    origin = (float(world[0][3]), float(world[1][3]), float(world[2][3]))
    return origin, direction  # type: ignore[return-value]


def build_receiver_plane(camera_or_solve: Any, roi: RegionROI, *,
                         plane_height: float = 0.0,
                         max_distance: float = 500.0,
                         margin: float = 1.1) -> ReceiverGeometry:
    """A horizontal receiver plane sized to catch every ROI viewing ray.

    v0.1 ocean model (spec §10): a plane at ``plane_height`` (world Y). ROI
    edge/corner rays are intersected with the plane; rays that never reach it
    (at/above the horizon) are clamped at ``max_distance`` along the ray and
    dropped onto the plane, so a sky-clipping ROI stays bounded. ``margin``
    grows the extents so projection has slack past the exact frustum edge.
    """
    camera = getattr(camera_or_solve, "camera", camera_or_solve)
    origin_y = float(camera.extrinsics.camera_world_matrix[1][3])
    if origin_y <= plane_height:
        raise ValueError(
            f"Camera height {origin_y:.3f} is at or below the receiver plane "
            f"y={plane_height:.3f}; a water plane must sit below the camera")

    xs = (roi.x, roi.x + roi.width / 2.0, roi.x + roi.width)
    ys = (roi.y, roi.y + roi.height / 2.0, roi.y + roi.height)
    hits: list[tuple[float, float]] = []
    for py in ys:
        for px in xs:
            origin, direction = pixel_ray_world(camera, px, py)
            dy = direction[1]
            if dy < -1e-9:
                t = (plane_height - origin[1]) / dy
                t = min(t, max_distance)
            else:
                t = max_distance
            hits.append((origin[0] + t * direction[0],
                         origin[2] + t * direction[2]))
    min_x = min(h[0] for h in hits)
    max_x = max(h[0] for h in hits)
    min_z = min(h[1] for h in hits)
    max_z = max(h[1] for h in hits)
    centre = ((min_x + max_x) / 2.0, plane_height, (min_z + max_z) / 2.0)
    ex = max(1e-3, (max_x - min_x)) * float(margin)
    ez = max(1e-3, (max_z - min_z)) * float(margin)
    # THREE.PlaneGeometry frame (depth_geometry.plane_transform convention):
    # local X=u=[1,0,0], local Y=v=[0,0,-1], normal n=[0,1,0].
    transform = (
        (1.0, 0.0, 0.0, centre[0]),
        (0.0, 0.0, 1.0, centre[1]),
        (0.0, -1.0, 0.0, centre[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    primitive = AtlasProxyPrimitive(
        name="dynamic_plate_receiver",
        primitive_type="plane",
        transform_matrix=transform,
        dimensions=(ex, ez, 0.0),
        material="atlas_dynamic_plate",
        metadata={
            "role": "dynamic_plate_receiver",
            "plane_height": float(plane_height),
            "max_distance": float(max_distance),
        },
    )
    return ReceiverGeometry(primitive=primitive)


def write_plane_obj(receiver: ReceiverGeometry, path: Any) -> Path:
    """Write the receiver plane as a single Y-up quad OBJ.

    UVs are a plain 0..1 sheet — projective UVs are the DCC's job (the
    projection camera is exported alongside; spec §12 keeps registration
    camera-based, never arbitrarily UV-fitted).
    """
    prim = receiver.primitive
    if prim is None or prim.primitive_type != "plane":
        raise ValueError("write_plane_obj needs a plane primitive")
    tf = prim.transform_matrix
    u = (tf[0][0], tf[1][0], tf[2][0])
    v = (tf[0][1], tf[1][1], tf[2][1])
    c = (tf[0][3], tf[1][3], tf[2][3])
    ex, ez, _ = prim.dimensions
    hu, hv = ex / 2.0, ez / 2.0
    corners = [
        tuple(c[i] - u[i] * hu - v[i] * hv for i in range(3)),
        tuple(c[i] + u[i] * hu - v[i] * hv for i in range(3)),
        tuple(c[i] + u[i] * hu + v[i] * hv for i in range(3)),
        tuple(c[i] - u[i] * hu + v[i] * hv for i in range(3)),
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Atlas dynamic-plate receiver plane (Y-up, metres)"]
    for p in corners:
        lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1",
              "vn 0 1 0",
              "f 1/1/1 2/2/1 3/3/1 4/4/1", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    receiver.path = out.name if receiver.path is None else receiver.path
    return out


@dataclass(slots=True)
class DynamicPlate:
    """One dynamic region of a solved still (spec §3).

    The temporal sequence is a time-varying texture projected through the
    fixed `crop_camera` onto `receiver` geometry; all frames share that camera
    and geometry while the artist camera moves independently (spec §11).
    """

    plate_id: str
    semantic_type: str
    source_image: str
    source_width: int
    source_height: int
    matte_path: str | None = None
    matte_bbox: RegionROI | None = None
    source_roi: RegionROI | None = None
    crop_transform: CropTransform | None = None
    source_camera: LatentCamera | None = None
    crop_camera: LatentCamera | None = None
    receiver: ReceiverGeometry | None = None
    frame_rate: float = 24.0
    frame_start: int = 0
    frame_end: int = 0
    generator: str = ""
    generator_config: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    seed: int | None = None
    projection_mode: str = "camera_projection"
    matte_feather_px: float = 0.0
    color_metadata: dict[str, Any] = field(default_factory=lambda: {
        "input_color_space": "sRGB",
        "generator_input_transform": "none",
        "generator_output_color_space": "sRGB",
        "atlas_working_color_space": "sRGB",
    })
    provenance: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_PROVENANCE))
    status: str = PLATE_STATUS_DRAFT
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = "0.1"

    def __post_init__(self) -> None:
        if self.semantic_type not in DYNAMIC_REGION_TYPES:
            raise ValueError(
                f"Unknown dynamic-region type {self.semantic_type!r}; "
                f"expected one of {DYNAMIC_REGION_TYPES}")

    @property
    def frame_count(self) -> int:
        return max(0, int(self.frame_end) - int(self.frame_start) + 1)

    def to_dict(self) -> dict[str, Any]:
        data = _json_ready(self)
        data["schema_version"] = self.schema_version
        data["plate_type"] = "dynamic_plate"
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicPlate":
        src_cam = data.get("source_camera")
        crop_cam = data.get("crop_camera")
        plate = cls(
            plate_id=str(data["plate_id"]),
            semantic_type=str(data["semantic_type"]),
            source_image=str(data.get("source_image", "")),
            source_width=int(data.get("source_width", 0)),
            source_height=int(data.get("source_height", 0)),
            matte_path=data.get("matte_path"),
            matte_bbox=RegionROI.from_dict(data.get("matte_bbox")),
            source_roi=RegionROI.from_dict(data.get("source_roi")),
            crop_transform=CropTransform.from_dict(data.get("crop_transform")),
            source_camera=LatentCamera.from_dict(src_cam) if src_cam else None,
            crop_camera=LatentCamera.from_dict(crop_cam) if crop_cam else None,
            receiver=ReceiverGeometry.from_dict(data.get("receiver")),
            frame_rate=float(data.get("frame_rate", 24.0)),
            frame_start=int(data.get("frame_start", 0)),
            frame_end=int(data.get("frame_end", 0)),
            generator=data.get("generator", ""),
            generator_config=dict(data.get("generator_config", {})),
            prompt=data.get("prompt", ""),
            seed=data.get("seed"),
            projection_mode=data.get("projection_mode", "camera_projection"),
            matte_feather_px=float(data.get("matte_feather_px", 0.0)),
            status=data.get("status", PLATE_STATUS_DRAFT),
            warnings=list(data.get("warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )
        if data.get("color_metadata"):
            plate.color_metadata = dict(data["color_metadata"])
        if data.get("provenance"):
            plate.provenance = dict(data["provenance"])
        return plate
