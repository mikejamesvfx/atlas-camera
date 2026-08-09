"""RAW metadata extraction, including RAF embedded-preview fallback."""

import sys
from types import SimpleNamespace

import pytest

from atlas_camera.raw import metadata


def _install_exifread(monkeypatch, process_file):
    monkeypatch.setitem(sys.modules, "exifread", SimpleNamespace(process_file=process_file))


def test_raf_embedded_jpeg_metadata_is_preferred_and_closes_raw_handle(monkeypatch, tmp_path):
    """Removing the RAF-preview route would leave X-H2 metadata empty."""
    raw_handle = SimpleNamespace(closed=False)
    raw_handle.close = lambda: setattr(raw_handle, "closed", True)
    raw_handle.extract_thumb = lambda: SimpleNamespace(
        format="JPEG", data=b"jpeg-exif")
    rawpy = SimpleNamespace(
        ThumbFormat=SimpleNamespace(JPEG="JPEG"),
        imread=lambda path: raw_handle,
    )
    monkeypatch.setitem(sys.modules, "rawpy", rawpy)

    seen = {"calls": 0}

    def process_file(handle, *, details):
        seen["calls"] += 1
        seen["data"] = handle.read()
        seen["details"] = details
        return {
            "Image Make": "FUJIFILM",
            "Image Model": "X-H2",
            "EXIF FocalLength": "187/10",
            "Image Orientation": "8",
            "EXIF BodySerialNumber": " 4B000584\x00 ",
            "EXIF LensSerialNumber": " 06C02394\x00 ",
            "EXIF DateTimeOriginal": " 2026:08:09 12:34:56\x00 ",
        }

    _install_exifread(monkeypatch, process_file)
    monkeypatch.setattr(metadata, "_pillow_exif_tags", lambda path: pytest.fail("container route used"))

    path = tmp_path / "x-h2.RAF"
    path.write_bytes(b"RAF container must not be parsed")
    result = metadata.read_raw_metadata(str(path))

    assert seen == {"calls": 1, "data": b"jpeg-exif", "details": True}
    assert raw_handle.closed is True
    assert result.camera_make == "FUJIFILM"
    assert result.camera_model == "X-H2"
    assert result.focal_length_mm == pytest.approx(18.7)
    assert result.orientation == 8
    assert result.body_serial_number == "4B000584"
    assert result.lens_serial_number == "06C02394"
    assert result.capture_datetime == "2026:08:09 12:34:56"
    assert result.metadata_source == "embedded_jpeg"


def test_raf_thumbnail_failure_degrades_to_soft_no_metadata_warning(monkeypatch):
    """A broken preview must not turn RAW decoding's metadata stage fatal."""
    def fail_imread(path):
        raise RuntimeError("bad RAF preview")

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=fail_imread))
    _install_exifread(monkeypatch, lambda handle, *, details: {})
    monkeypatch.setattr(metadata, "_pillow_exif_tags", lambda path: {})

    result = metadata.read_raw_metadata("broken.raf")

    assert result.metadata_source == "none"
    assert result.camera_model is None
    assert any("RAF embedded JPEG metadata failed" in warning for warning in result.warnings)
    assert any("No EXIF metadata could be read" in warning for warning in result.warnings)


def test_non_raf_keeps_direct_exifread_container_path(monkeypatch):
    """Routing every RAW through rawpy would regress the established parser."""
    def process_file(handle, *, details):
        return {"Image Make": "NIKON CORPORATION", "Image Model": "NIKON D810"}

    _install_exifread(monkeypatch, process_file)
    monkeypatch.setattr(metadata, "_pillow_exif_tags", lambda path: pytest.fail("Pillow fallback used"))

    result = metadata.read_raw_metadata(__file__)

    assert result.camera_make == "NIKON CORPORATION"
    assert result.camera_model == "NIKON D810"
    assert result.metadata_source == "container"


def test_capture_datetime_falls_back_to_image_datetime():
    """Ignoring Image DateTime loses capture time on otherwise valid EXIF."""
    result = metadata._metadata_from_tags({"Image DateTime": "2026:08:09 12:34:56"})

    assert result.capture_datetime == "2026:08:09 12:34:56"


def test_raw_metadata_preserves_legacy_raw_tags_and_warnings_positions():
    """Inserting new fields before these legacy slots corrupts positional callers."""
    raw_tags = {"Image Make": "FUJIFILM"}
    warnings = ["legacy warning"]

    result = metadata.RawMetadata(
        "FUJIFILM", "X-H2", None, None, None, None, None, None,
        None, None, None, 8, raw_tags, warnings,
    )

    assert result.orientation == 8
    assert result.raw_tags == raw_tags
    assert result.warnings == warnings
    assert result.body_serial_number is None
    assert result.lens_serial_number is None
    assert result.capture_datetime is None
    assert result.metadata_source is None


def test_import_raw_propagates_embedded_metadata_fields(monkeypatch):
    """Dropping identity metadata at the pipeline boundary hides RAW provenance."""
    import numpy as np
    from atlas_camera.raw import pipeline

    raw_meta = metadata.RawMetadata(
        orientation=8,
        body_serial_number="4B000584",
        lens_serial_number="06C02394",
        capture_datetime="2026:08:09 12:34:56",
        metadata_source="embedded_jpeg",
    )
    pixels = np.zeros((4, 6, 3), dtype=np.float32)
    monkeypatch.setattr(pipeline, "read_raw_metadata", lambda path: raw_meta)
    monkeypatch.setattr(pipeline, "decode_raw", lambda *args, **kwargs: (pixels, pixels))
    monkeypatch.setattr(pipeline, "resolve_sensor_size", lambda *args: metadata.SensorResolution(36.0, 24.0, "camera_db"))

    result = pipeline.import_raw("x-h2.raf", undistort=False)

    assert result.orientation == 8
    assert result.body_serial_number == "4B000584"
    assert result.lens_serial_number == "06C02394"
    assert result.capture_datetime == "2026:08:09 12:34:56"
    assert result.metadata_source == "embedded_jpeg"
    assert "metadata: embedded JPEG EXIF preview" in result.summary_lines()
