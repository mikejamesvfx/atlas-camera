"""RAW import orchestrator: metadata -> decode -> undistort -> sensor resolve."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas_camera.raw.decode import decode_raw
from atlas_camera.raw.metadata import read_raw_metadata, resolve_sensor_size

RAW_EXTENSIONS = (".nef", ".cr2", ".cr3", ".raf", ".arw", ".dng")


@dataclass(slots=True)
class RawImportResult:
    linear_rgb: Any            # HxWx3 float32 scene-linear, sRGB/Rec.709 primaries
    display_srgb: Any          # HxWx3 float32 display-encoded (solve/preview)
    width: int
    height: int
    focal_length_mm: float | None
    sensor_width_mm: float | None
    sensor_height_mm: float | None
    sensor_source: str
    camera_make: str | None
    camera_model: str | None
    lens_model: str | None
    undistort_applied: bool
    undistort_status: str
    distortion: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_path: str = ""

    def intrinsics_hint(self) -> dict[str, Any]:
        """Exactly the dict ``solve_still_image(intrinsics_hint=...)`` consumes."""
        hint: dict[str, Any] = {}
        if self.focal_length_mm:
            hint["focal_length_mm"] = self.focal_length_mm
        if self.sensor_width_mm:
            hint["sensor_width_mm"] = self.sensor_width_mm
        if self.sensor_height_mm:
            hint["sensor_height_mm"] = self.sensor_height_mm
        return hint

    def summary_lines(self) -> list[str]:
        cam = " ".join(p for p in (self.camera_make, self.camera_model) if p) or "unknown camera"
        focal = f"{self.focal_length_mm:g} mm" if self.focal_length_mm else "focal unknown"
        if self.sensor_width_mm:
            h = f"x{self.sensor_height_mm:g}" if self.sensor_height_mm else ""
            sensor = f"sensor {self.sensor_width_mm:g}{h} mm ({self.sensor_source})"
        else:
            sensor = f"sensor unknown ({self.sensor_source})"
        lines = [f"{cam} · {focal} · {sensor}",
                 f"undistort: {self.undistort_status}"
                 + (f" ({self.lens_model})" if self.lens_model else "")]
        lines.extend(self.warnings)
        return lines


def import_raw(path: str, *, undistort: bool = True, half_size: bool = False,
               white_balance: str = "camera", exposure_ev: float = 0.0) -> RawImportResult:
    """Decode + meta-harvest a RAW file into everything the solve needs."""
    meta = read_raw_metadata(path)
    linear, display = decode_raw(path, half_size=half_size,
                                 white_balance=white_balance,
                                 exposure_ev=exposure_ev)
    height, width = linear.shape[:2]

    undistort_applied = False
    undistort_status = "disabled"
    distortion: dict[str, float] = {}
    if undistort:
        undistort_status, distortion, coords, profile = _try_build_undistort(
            meta, width, height)
        if profile:
            # Name the matched lensfun profile — a derived "24mm f/1.4"
            # descriptor can't distinguish same-spec lenses, so the artist
            # must be able to see (and judge) which calibration was used.
            meta.warnings.append(f"lensfun profile: {profile}")
        if coords is not None:
            import numpy as np
            from atlas_camera.raw.undistort import apply_undistort
            # ONE shared remap grid for both arrays — the EXR sidecar and the
            # solve tensor must stay geometrically identical. Lanczos overshoots
            # at hard edges (found live: -0.09 on a D810 frame), so clamp:
            # negatives are non-physical in both; display re-caps at 1.0,
            # linear keeps its >1.0 highlights.
            linear = np.clip(apply_undistort(linear, coords), 0.0, None)
            display = np.clip(apply_undistort(display, coords), 0.0, 1.0)
            undistort_applied = True

    # Sensor resolution uses the DECODED width — half_size halves pixels but
    # the FocalPlane EXIF describes the full-resolution sensor, so tier 2 must
    # see full-res dimensions: rawpy half_size halves both, compensate.
    meta_width = width * 2 if half_size else width
    meta_height = height * 2 if half_size else height
    sensor = resolve_sensor_size(meta, meta_width, meta_height)

    warnings = list(meta.warnings) + list(sensor.warnings)
    return RawImportResult(
        linear_rgb=linear,
        display_srgb=display,
        width=width,
        height=height,
        focal_length_mm=meta.focal_length_mm,
        sensor_width_mm=sensor.sensor_width_mm,
        sensor_height_mm=sensor.sensor_height_mm,
        sensor_source=sensor.source,
        camera_make=meta.camera_make,
        camera_model=meta.camera_model,
        lens_model=meta.lens_model,
        undistort_applied=undistort_applied,
        undistort_status=undistort_status,
        distortion=distortion,
        warnings=warnings,
        source_path=str(path),
    )


def _try_build_undistort(meta, width: int, height: int):
    """Build the lensfun remap grid, degrading to a status — never an error."""
    if not meta.lens_model and not meta.camera_model:
        return "no_lens_metadata", {}, None, None
    try:
        from atlas_camera.raw.undistort import build_undistort_map
    except ImportError:
        return "lensfunpy_missing", {}, None, None
    try:
        result = build_undistort_map(meta, width, height)
    except RuntimeError:
        return "lensfunpy_missing", {}, None, None
    profile = None
    if result.lens_name:
        profile = (f"{result.lens_name} on {result.cam_name}"
                   if result.cam_name else str(result.lens_name))
    return result.status, result.distortion, result.coords, profile
