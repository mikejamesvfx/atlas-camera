"""EXIF metadata extraction for camera RAW files.

The parse core (`_metadata_from_tags`, `resolve_sensor_size`) is pure Python
over plain values so it unit-tests without exifread; only `read_raw_metadata`
touches the optional readers. exifread handles the TIFF-based formats natively
(NEF, CR2, ARW, and RAF's embedded EXIF); CR3 is ISO-BMFF, where exifread is
best-effort — a metadata miss degrades to warnings + manual widgets, never an
error (locked decision: no external binaries like ExifTool).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


@dataclass(slots=True)
class RawMetadata:
    camera_make: str | None = None
    camera_model: str | None = None
    lens_make: str | None = None
    lens_model: str | None = None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    aperture: float | None = None
    iso: int | None = None
    focal_plane_x_res: float | None = None
    focal_plane_y_res: float | None = None
    focal_plane_res_unit: int | None = None
    orientation: int | None = None
    raw_tags: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SensorResolution:
    sensor_width_mm: float
    sensor_height_mm: float | None
    source: str  # "camera_db" | "exif_focal_plane" | "exif_35mm_ratio" | "assumed_default"
    warnings: list[str] = field(default_factory=list)


# Sanity band for a computed sensor width: below ~4mm is smaller than any
# phone RAW sensor, above ~70mm is beyond medium format — either means the
# FocalPlane EXIF fields were garbage (common on adapted lenses / firmware
# quirks), so the computation is rejected rather than poisoning the solve.
_SENSOR_WIDTH_SANE_MM = (4.0, 70.0)

_FOCAL_PLANE_UNIT_TO_MM = {2: 25.4, 3: 10.0, 4: 1.0}


def _to_float(value: Any) -> float | None:
    """Coerce EXIF rational/string/number representations to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            denominator = float(den)
            if denominator == 0:
                return None
            return float(num) / denominator
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\x00").strip()
    return text or None


def _metadata_from_tags(tags: dict[str, Any]) -> RawMetadata:
    """Build RawMetadata from an exifread-style tag-name -> value mapping.

    Pure function over plain values (exifread IfdTag objects stringify
    cleanly, so callers may pass them directly or pre-stringified values).
    """
    def tag(*names: str) -> Any:
        for name in names:
            if name in tags and tags[name] is not None:
                return tags[name]
        return None

    meta = RawMetadata(
        camera_make=_to_str(tag("Image Make", "Make")),
        camera_model=_to_str(tag("Image Model", "Model")),
        lens_make=_to_str(tag("EXIF LensMake", "LensMake")),
        lens_model=_to_str(tag("EXIF LensModel", "LensModel")),
        focal_length_mm=_to_float(tag("EXIF FocalLength", "FocalLength")),
        focal_length_35mm=_to_float(
            tag("EXIF FocalLengthIn35mmFilm", "FocalLengthIn35mmFilm")),
        aperture=_to_float(tag("EXIF FNumber", "FNumber")),
        iso=_to_int(tag("EXIF ISOSpeedRatings", "ISOSpeedRatings", "EXIF PhotographicSensitivity")),
        focal_plane_x_res=_to_float(tag("EXIF FocalPlaneXResolution", "FocalPlaneXResolution")),
        focal_plane_y_res=_to_float(tag("EXIF FocalPlaneYResolution", "FocalPlaneYResolution")),
        focal_plane_res_unit=_to_int(
            tag("EXIF FocalPlaneResolutionUnit", "FocalPlaneResolutionUnit")),
        orientation=_to_int(tag("Image Orientation", "Orientation")),
        # MakerNote blobs (LensData etc.) can be huge — cap for the debug dump.
        raw_tags={key: str(value)[:120] for key, value in tags.items()},
    )
    # A 35mm-equivalent of 0 is EXIF's "unknown" sentinel, not a real value.
    if meta.focal_length_35mm is not None and meta.focal_length_35mm <= 0:
        meta.focal_length_35mm = None
    if meta.focal_length_mm is not None and meta.focal_length_mm <= 0:
        meta.focal_length_mm = None
    if meta.lens_model is None:
        # Many Nikon NEFs carry no lens NAME, only MakerNote focal/aperture
        # specs (found live on a D810 file: LensMinMaxFocalMaxAperture =
        # [24, 24, 7/5, 7/5] = a 24mm f/1.4 prime). Derive a descriptor so
        # lensfun's loose search has something to match — best-effort, and
        # flagged, since it can't distinguish same-spec lenses (e.g. Nikkor
        # vs Sigma Art 24/1.4); the undistort report names the profile used.
        derived = _lens_descriptor_from_makernote(
            tag("MakerNote LensMinMaxFocalMaxAperture", "LensMinMaxFocalMaxAperture",
                "MakerNote LensSpec", "LensSpec"))
        if derived:
            meta.lens_model = derived
            meta.warnings.append(
                f"Lens name absent from EXIF — derived '{derived}' from "
                "MakerNote specs; lensfun profile match is best-effort.")
    return meta


def _lens_descriptor_from_makernote(value: Any) -> str | None:
    """[min_focal, max_focal, max_ap_wide, max_ap_tele] -> "24mm f/1.4" /
    "24-70mm f/2.8" / "18-55mm f/3.5-5.6". Accepts exifread IfdTag values
    (list of Ratios) or their stringified "[24, 24, 7/5, 7/5]" form."""
    if value is None:
        return None
    parts: list[float] = []
    raw_values = getattr(value, "values", None)
    if raw_values is not None and not isinstance(raw_values, str):
        candidates = list(raw_values)
    else:
        text = str(value).strip().strip("[]")
        candidates = [p.strip() for p in text.split(",") if p.strip()]
    for item in candidates[:4]:
        number = _to_float(item)
        if number is None:
            return None
        parts.append(number)
    if len(parts) < 4 or parts[0] <= 0 or parts[2] <= 0:
        return None
    min_f, max_f, ap_wide, ap_tele = parts[:4]
    focal = (f"{min_f:g}mm" if abs(max_f - min_f) < 0.5
             else f"{min_f:g}-{max_f:g}mm")
    aperture = (f"f/{ap_wide:g}" if abs(ap_tele - ap_wide) < 0.05
                else f"f/{ap_wide:g}-{ap_tele:g}")
    return f"{focal} {aperture}"


def read_raw_metadata(path: str) -> RawMetadata:
    """Read EXIF from a RAW file — exifread first, Pillow TIFF fallback."""
    try:
        import exifread
    except ImportError as exc:
        raise RuntimeError(
            "Camera RAW metadata requires exifread. "
            "Install with: pip install -e .[raw]") from exc

    tags: dict[str, Any] = {}
    try:
        with open(path, "rb") as handle:
            # details=True: MakerNote parsing is required for lens metadata on
            # Nikon NEFs (no standard LensModel tag — found live on a D810).
            tags = dict(exifread.process_file(handle, details=True))
    except Exception as exc:  # noqa: BLE001 — any parse failure degrades soft
        meta = RawMetadata()
        meta.warnings.append(f"exifread failed on {path}: {exc}")
        tags = {}

    if not tags:
        # Pillow reads TIFF-container EXIF for some formats exifread chokes on.
        pil_tags = _pillow_exif_tags(path)
        if pil_tags:
            tags = pil_tags

    meta = _metadata_from_tags(tags) if tags else RawMetadata()
    if not tags:
        meta.warnings.append(
            "No EXIF metadata could be read (CR3 metadata is best-effort — "
            "set focal length / sensor size manually if needed).")
    return meta


def _pillow_exif_tags(path: str) -> dict[str, Any]:
    try:
        from PIL import ExifTags, Image
    except ImportError:
        return {}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return {}
            merged: dict[str, Any] = {}
            for tag_id, value in exif.items():
                merged[str(ExifTags.TAGS.get(tag_id, tag_id))] = value
            try:
                ifd = exif.get_ifd(ExifTags.IFD.Exif)
                for tag_id, value in ifd.items():
                    merged[str(ExifTags.TAGS.get(tag_id, tag_id))] = value
            except Exception:  # noqa: BLE001
                pass
            return merged
    except Exception:  # noqa: BLE001 — Pillow can't open most RAWs; that's fine
        return {}


def resolve_sensor_size(meta: RawMetadata, image_width_px: int,
                        image_height_px: int) -> SensorResolution:
    """Best-evidence-first sensor-size cascade, provenance recorded.

    Mirrors the solve's scale-source tiering: each tier is adopted only when
    its inputs are sane, otherwise the next tier is tried and the miss is
    surfaced as a warning — never silently.
    """
    warnings: list[str] = []

    # Tier 1: curated camera-body registry.
    from atlas_camera.reference_data.camera_registry import find_camera_body
    body = find_camera_body(meta.camera_make, meta.camera_model)
    if body is not None:
        return SensorResolution(body.sensor_width_mm, body.sensor_height_mm,
                                "camera_db", warnings)
    if meta.camera_model:
        warnings.append(
            f"Camera model '{meta.camera_model}' not in camera_bodies.json — "
            "falling back to EXIF-derived sensor size.")

    # Tier 2: FocalPlane resolution arithmetic.
    unit_mm = _FOCAL_PLANE_UNIT_TO_MM.get(meta.focal_plane_res_unit or 0)
    if meta.focal_plane_x_res and unit_mm and image_width_px > 0:
        width_mm = image_width_px / meta.focal_plane_x_res * unit_mm
        if _SENSOR_WIDTH_SANE_MM[0] <= width_mm <= _SENSOR_WIDTH_SANE_MM[1]:
            height_mm = None
            if meta.focal_plane_y_res and image_height_px > 0:
                height_mm = image_height_px / meta.focal_plane_y_res * unit_mm
            return SensorResolution(width_mm, height_mm, "exif_focal_plane", warnings)
        warnings.append(
            f"FocalPlane EXIF computed an implausible sensor width "
            f"({width_mm:.1f}mm) — ignored.")

    # Tier 3: 35mm-equivalent ratio (sensor_w = 36 * focal / focal35).
    if meta.focal_length_mm and meta.focal_length_35mm:
        width_mm = 36.0 * meta.focal_length_mm / meta.focal_length_35mm
        if _SENSOR_WIDTH_SANE_MM[0] <= width_mm <= _SENSOR_WIDTH_SANE_MM[1]:
            height_mm = None
            if image_width_px > 0 and image_height_px > 0:
                height_mm = width_mm * image_height_px / image_width_px
            return SensorResolution(width_mm, height_mm, "exif_35mm_ratio", warnings)
        warnings.append(
            f"35mm-equivalent ratio computed an implausible sensor width "
            f"({width_mm:.1f}mm) — ignored.")

    # Tier 4: assumed full frame, flagged.
    warnings.append(
        "Sensor size assumed 36.0mm full frame — no registry match and no "
        "usable EXIF sensor evidence.")
    return SensorResolution(36.0, None, "assumed_default", warnings)


# --------------------------------------------------------------------------
# RAW intrinsics-hint precedence + provenance stamping (phase 2 move from
# comfy/node_helpers.py). Pure dict/dataclass work; no ComfyUI import.
# --------------------------------------------------------------------------

def _resolve_raw_hints(focal_widget_mm, sensor_widget_mm, raw_meta):
    """Resolve (focal_hint, sensor_w, sensor_h) from widget values + an
    optionally wired ATLAS_RAW_META (AtlasLoadRAW's RawImportResult).

    Precedence: an explicit widget value (>0 focal / non-default sensor)
    always beats the wired metadata, so an artist override never fights
    the EXIF. sensor_height flows only from raw_meta (no widget exists).
    """
    focal_hint = float(focal_widget_mm) if focal_widget_mm and focal_widget_mm > 0 else None
    sensor_w = float(sensor_widget_mm)
    sensor_h = None
    if raw_meta is not None:
        if focal_hint is None and getattr(raw_meta, "focal_length_mm", None):
            focal_hint = float(raw_meta.focal_length_mm)
        if sensor_w == 36.0 and getattr(raw_meta, "sensor_width_mm", None):
            sensor_w = float(raw_meta.sensor_width_mm)
            if getattr(raw_meta, "sensor_height_mm", None):
                sensor_h = float(raw_meta.sensor_height_mm)
    return focal_hint, sensor_w, sensor_h
def _stamp_raw_provenance(solve, raw_meta):
    """Record where a RAW import's hints came from on the solve (in place)."""
    if raw_meta is None:
        return
    solve.debug_metadata["raw_import"] = {
        "source_path": getattr(raw_meta, "source_path", None),
        "camera_make": getattr(raw_meta, "camera_make", None),
        "camera_model": getattr(raw_meta, "camera_model", None),
        "lens_model": getattr(raw_meta, "lens_model", None),
        "focal_length_mm": getattr(raw_meta, "focal_length_mm", None),
        "sensor_width_mm": getattr(raw_meta, "sensor_width_mm", None),
        "sensor_height_mm": getattr(raw_meta, "sensor_height_mm", None),
        "sensor_source": getattr(raw_meta, "sensor_source", None),
        "undistort_status": getattr(raw_meta, "undistort_status", None),
    }
