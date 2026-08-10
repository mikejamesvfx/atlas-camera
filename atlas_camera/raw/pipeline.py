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
    # The headroom factor actually applied to linear_rgb. RECORDED, not
    # recomputed: the node's report and the EXR's atlas:headroom attribute both
    # read it from here, and a plate that has been scaled must be able to say
    # by how much or a downstream re-grade is guesswork.
    headroom: float = 6.0
    orientation: int | None = None
    body_serial_number: str | None = None
    lens_serial_number: str | None = None
    capture_datetime: str | None = None
    metadata_source: str | None = None

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
        if self.metadata_source == "embedded_jpeg":
            lines.append("metadata: embedded JPEG EXIF preview")
        # The scale is reported unconditionally. It changes exposure of the
        # delivered plate, so it can never be something the artist has to go
        # looking for — and at 1.0 the file is NOT ACES-referred, which is the
        # exact confusion this line exists to end.
        if float(self.headroom) == 1.0:
            lines.append(
                "headroom: 1.0x — clip-normalised, NOT ACES-referred; expect "
                "~2.6 stops dark under an ACES view transform. Set 6.0 to "
                "match rawtoaces.")
        else:
            lines.append(
                f"headroom: {self.headroom:g}x (rawtoaces default 6.0) · "
                f"diffuse white ~1.0, clip ~{self.headroom:g}")
        lines.extend(self.warnings)
        return lines


def import_raw(path: str, *, undistort: bool = True, half_size: bool = False,
               white_balance: str = "camera", exposure_ev: float = 0.0,
               headroom: float = 6.0) -> RawImportResult:
    """Decode + meta-harvest a RAW file into everything the solve needs.

    ``headroom`` scales the scene-linear master only (see ``decode_raw``); the
    display tensor the solver reads is unaffected.

    Camera-processed JPEGs are accepted too (added 2026-08-10): the camera's
    own engine already developed and lens-corrected the pixels, and JPEG EXIF
    carries the same body/lens/focal evidence the multi-view validator gates
    on. They route through :func:`_import_processed_jpeg` — a lower trust
    tier, stamped ``metadata_source="jpeg_exif"`` and
    ``undistort_status="camera_processed"`` so provenance stays honest.
    """
    if str(path).lower().endswith((".jpg", ".jpeg")):
        return _import_processed_jpeg(path, half_size=half_size)
    meta = read_raw_metadata(path)
    linear, display = decode_raw(path, half_size=half_size,
                                 white_balance=white_balance,
                                 exposure_ev=exposure_ev,
                                 headroom=headroom)
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
        orientation=getattr(meta, "orientation", None),
        body_serial_number=getattr(meta, "body_serial_number", None),
        lens_serial_number=getattr(meta, "lens_serial_number", None),
        capture_datetime=getattr(meta, "capture_datetime", None),
        metadata_source=getattr(meta, "metadata_source", None),
        headroom=float(headroom),
    )


def _import_processed_jpeg(path: str, *, half_size: bool = False) -> RawImportResult:
    """Import a camera-processed JPEG as trusted-EXIF capture evidence.

    The camera's JPEG engine already applied tone mapping and (on modern
    bodies) lens distortion correction, so no develop or lensfun pass runs
    here: pixels are taken verbatim (EXIF orientation applied), display is
    the file's sRGB, and linear is its exact sRGB-EOTF inversion.  headroom
    is 1.0 — there are no highlights above display white in an 8-bit source.
    """
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "JPEG capture import requires Pillow + numpy. "
            "Install with: pip install -e .[raw]") from exc

    meta = read_raw_metadata(path)
    if meta.metadata_source == "container":
        meta.metadata_source = "jpeg_exif"

    with Image.open(path) as handle:
        oriented = ImageOps.exif_transpose(handle.convert("RGB"))
        if half_size:
            oriented = oriented.resize(
                (max(1, oriented.width // 2), max(1, oriented.height // 2)),
                Image.Resampling.LANCZOS,
            )
        display = np.asarray(oriented, dtype=np.float32) / 255.0
    display = np.clip(display, 0.0, 1.0)
    linear = np.where(
        display <= 0.04045,
        display / 12.92,
        np.power((display + 0.055) / 1.055, 2.4),
    ).astype(np.float32)
    height, width = display.shape[:2]

    meta_width = width * 2 if half_size else width
    meta_height = height * 2 if half_size else height
    sensor = resolve_sensor_size(meta, meta_width, meta_height)

    warnings = list(meta.warnings) + list(sensor.warnings)
    warnings.append(
        "Camera-processed JPEG: pixels are the camera's own develop "
        "(lens correction assumed applied in-body); a RAW original is the "
        "higher trust tier."
    )
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
        undistort_applied=True,
        undistort_status="camera_processed",
        distortion={},
        warnings=warnings,
        source_path=str(path),
        # exif_transpose already rotated the pixels: orientation is now
        # normal, and reporting the ORIGINAL tag would trigger the
        # multi-view sensor-axis swap on already-upright pixels.
        orientation=1,
        body_serial_number=getattr(meta, "body_serial_number", None),
        lens_serial_number=getattr(meta, "lens_serial_number", None),
        capture_datetime=getattr(meta, "capture_datetime", None),
        metadata_source=meta.metadata_source,
        headroom=1.0,
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
