"""One photograph -> metric relief world. Public: ``atlas.recover_relief_world``.

The operation Atlas Showcase's web bake needed, which until 2026-09-13 it could only
perform by reaching into eleven Camera internals and re-deriving Camera's own
orchestration (proposal: ``docs/proposals/2026-09-13-relief-world-facade.md``).

:func:`recover_relief_world` recovers the camera from a single photograph with the
learned prior (honouring EXIF focal and sensor size when the file carries them),
estimates metric depth at the solve's resolution, fits ground scale and builds the
relief mesh. It returns a :class:`ReliefWorld` whose camera, scale evidence and mesh
statistics are a stable interface. :func:`export_relief_world_glb` writes the
self-contained GLB.

``ReliefWorld.mesh`` is deliberately opaque: pass it back to
:func:`export_relief_world_glb`; do not depend on its fields. The mesh type is an
implementation structure with many fields, and publishing it would freeze all of them.

The sequence is the one the Showcase bake ran, moved to its owner unchanged, so a bake
through this facade is byte-identical to the bake it replaces. Consolidating it with
the tier-2 depth cascade inside :func:`atlas_camera.core.solver.solve_still_image_learned`
is a Camera-internal follow-up; it would change results and is not part of this surface.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.core.scene_health import ScaleHealth
from atlas_camera.core.schema import AtlasSolve

#: Camera RAW formats, plus camera-processed JPEGs (EXIF focal/sensor evidence).
RAW_OR_JPEG_SUFFIXES = (".nef", ".cr2", ".cr3", ".raf", ".arw", ".dng", ".jpg", ".jpeg")
#: Every input :func:`recover_relief_world` accepts, lower-case.
RELIEF_WORLD_INPUT_SUFFIXES = RAW_OR_JPEG_SUFFIXES + (".png",)
#: Same model as ``inference.depth_estimator.DEFAULT_METRIC_OUTDOOR`` (pinned by a test);
#: repeated here so importing the facade does not import the depth stack.
DEFAULT_RELIEF_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
DEFAULT_RELIEF_GRID_LONG_EDGE = 192


@dataclass(frozen=True)
class ReliefWorldCamera:
    """The solved pinhole camera, in Atlas convention (Y-up, camera looks down -Z)."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    #: Row-major 4x4 world->camera matrix.
    view_matrix: tuple[tuple[float, float, float, float], ...]
    pitch_deg: float
    #: Image row of the solved horizon.
    horizon_y: float
    #: ``"exif"`` when a file focal length was used, ``"geocalib"`` when predicted.
    focal_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "image_width": self.image_width, "image_height": self.image_height,
            "view_matrix": [list(row) for row in self.view_matrix],
            "pitch_deg": self.pitch_deg,
            "horizon_y": self.horizon_y,
            "focal_source": self.focal_source,
        }


@dataclass(frozen=True)
class ReliefWorldScale:
    """Metric-scale evidence: the solve's scale health plus the measured ground fit."""

    health: ScaleHealth
    ground_scale: float
    ground_inliers: int
    measured_camera_height_m: float | None
    measured_height_confidence: float | None
    depth_model: str
    depth_is_metric: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.health.to_dict(),
            "ground_scale": self.ground_scale,
            "ground_inliers": self.ground_inliers,
            "measured_camera_height_m": self.measured_camera_height_m,
            "measured_height_confidence": self.measured_height_confidence,
            "depth_model": self.depth_model,
            "depth_is_metric": self.depth_is_metric,
        }


@dataclass(frozen=True)
class ReliefWorldMeshStats:
    grid_long_edge: int
    n_faces: int | None
    torn_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"grid_long_edge": self.grid_long_edge, "n_faces": self.n_faces,
                "torn_fraction": self.torn_fraction}


@dataclass(frozen=True)
class ReliefWorld:
    """A photograph recovered as a metric relief world."""

    solve: AtlasSolve
    camera: ReliefWorldCamera
    scale: ReliefWorldScale
    mesh_stats: ReliefWorldMeshStats
    #: EXIF capture evidence (make, model, lens, datetime, metadata source); empty for PNG.
    capture: dict[str, Any]
    #: The decoded RGB frame the solve used (PIL image).
    display_image: Any = field(repr=False, compare=False)
    #: Opaque. Pass to :func:`export_relief_world_glb`; do not depend on its fields.
    mesh: Any = field(repr=False, compare=False)


def _load_photograph(image_path: Path):
    """Return (PIL display image, solve hints, capture evidence)."""
    from PIL import Image

    if image_path.suffix.lower() in RAW_OR_JPEG_SUFFIXES:
        import numpy as np

        from atlas_camera.raw.pipeline import import_raw

        raw = import_raw(str(image_path), half_size=False)
        display = Image.fromarray(
            (np.clip(raw.display_srgb, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
        )
        hints = {
            "focal_length_mm_hint": raw.focal_length_mm,
            "sensor_width_mm": raw.sensor_width_mm or 36.0,
            "sensor_height_mm": raw.sensor_height_mm,
        }
        capture = {
            "camera_make": raw.camera_make,
            "camera_model": raw.camera_model,
            "lens_model": raw.lens_model,
            "capture_datetime": raw.capture_datetime,
            "metadata_source": raw.metadata_source,
            "undistort_status": raw.undistort_status,
        }
        return display, hints, capture
    display = Image.open(image_path).convert("RGB")
    return display, {"sensor_width_mm": 36.0}, {}


def recover_relief_world(
    image_path: str | Path,
    *,
    depth_model: str = DEFAULT_RELIEF_DEPTH_MODEL,
    device: str | None = None,
    grid_long_edge: int = DEFAULT_RELIEF_GRID_LONG_EDGE,
) -> ReliefWorld:
    """Recover one photograph as a metric relief world.

    Accepts the suffixes in :data:`RELIEF_WORLD_INPUT_SUFFIXES`; anything else raises
    ``ValueError`` before any model is loaded. Requires the ``[neural]`` extra
    (torch, GeoCalib and a depth backend).
    """
    image_path = Path(image_path)
    if image_path.suffix.lower() not in RELIEF_WORLD_INPUT_SUFFIXES:
        raise ValueError(
            f"recover_relief_world does not accept {image_path.suffix!r} inputs; "
            f"supported: {', '.join(RELIEF_WORLD_INPUT_SUFFIXES)}"
        )
    if grid_long_edge <= 0:
        raise ValueError("grid_long_edge must be positive")

    import numpy as np

    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
    from atlas_camera.core.scene_health import scale_health
    from atlas_camera.core.solver import (
        _resize_depth,
        estimate_ground_height_from_depth,
        solve_from_learned_prior,
    )
    from atlas_camera.inference import depth_estimator, learned_prior

    display, hints, capture = _load_photograph(image_path)

    # GeoCalib and the depth model read from a file path; a RAW display frame needs one.
    if image_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        solve_input = str(image_path)
        tmp = None
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        display.save(tmp.name, quality=95)
        solve_input = tmp.name

    try:
        prior = learned_prior.estimate_camera_prior(solve_input, device=device)
        solve = solve_from_learned_prior(
            prior,
            image_path=str(image_path),
            focal_length_mm_hint=hints.get("focal_length_mm_hint"),
            sensor_width_mm=hints.get("sensor_width_mm", 36.0),
            sensor_height_mm=hints.get("sensor_height_mm"),
        )
        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        width, height = int(intr.image_width), int(intr.image_height)
        fx = float(intr.fx_px)
        fy = float(intr.fy_px or fx)
        spec = CameraSpec.from_solve(solve)
        vm = np.asarray(extr.camera_view_matrix, dtype=np.float64)
        horizon_y = height / 2.0 + fy * math.tan(math.radians(prior.pitch_deg))

        depth_result = depth_estimator.estimate_depth(
            solve_input, model_id=depth_model, device=device, focal_px=prior.focal_px,
        )
        depth = depth_result.depth
        if depth.shape != (height, width):
            depth = _resize_depth(depth, width, height)

        scale, scale_info = estimate_ground_scale(
            depth, view_matrix=vm, fx=fx, fy=fy, cx=spec.cx, cy=spec.cy,
            horizon_y=horizon_y,
        )
        ground = estimate_ground_height_from_depth(
            depth, rotation=vm[:3, :3], fx=fx, fy=fy, cx=spec.cx, cy=spec.cy,
            horizon_y=horizon_y,
        )
        mesh = build_relief_mesh(
            depth, view_matrix=vm, fx=fx, fy=fy, cx=spec.cx, cy=spec.cy,
            grid_long_edge=grid_long_edge, scale=scale, horizon_y=horizon_y,
        )
        stats = getattr(mesh, "stats", {}) or {}

        return ReliefWorld(
            solve=solve,
            camera=ReliefWorldCamera(
                fx=fx, fy=fy, cx=spec.cx, cy=spec.cy,
                image_width=width, image_height=height,
                view_matrix=tuple(tuple(float(v) for v in row) for row in vm),
                pitch_deg=float(prior.pitch_deg),
                horizon_y=float(horizon_y),
                focal_source="exif" if hints.get("focal_length_mm_hint") else "geocalib",
            ),
            scale=ReliefWorldScale(
                health=scale_health(solve),
                ground_scale=float(scale),
                ground_inliers=int(scale_info.get("n_ground", scale_info.get("inliers", 0)) or 0),
                measured_camera_height_m=ground.get("camera_height"),
                measured_height_confidence=ground.get("confidence"),
                depth_model=depth_model,
                depth_is_metric=bool(depth_result.is_metric),
            ),
            mesh_stats=ReliefWorldMeshStats(
                grid_long_edge=grid_long_edge,
                n_faces=stats.get("n_faces"),
                torn_fraction=stats.get("torn_fraction"),
            ),
            capture=capture,
            display_image=display,
            mesh=mesh,
        )
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def export_relief_world_glb(
    world: ReliefWorld,
    output_dir: str | Path,
    *,
    texture: Any | None = None,
    name: str = "relief_world",
    texture_format: str = "PNG",
) -> Path:
    """Write ``{name}.glb`` for *world* and return its path.

    ``texture`` defaults to ``world.display_image``; pass a resized or otherwise
    prepared image to embed that instead. ``texture_format`` is ``"PNG"`` (default,
    lossless) or ``"JPEG"`` — see
    :func:`atlas_camera.exporters.relief_mesh_exporter.export_relief_mesh_glb`.
    """
    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh_glb

    if not isinstance(world, ReliefWorld):
        raise TypeError("export_relief_world_glb expects a ReliefWorld from recover_relief_world")
    image = world.display_image if texture is None else texture
    return Path(export_relief_mesh_glb(world.mesh, output_dir, texture=image, name=name,
                                       texture_format=texture_format)["glb"])
