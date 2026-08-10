"""Camera-processed JPEG capture import — trusted-EXIF evidence tier."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from atlas_camera.raw.pipeline import import_raw  # noqa: E402


def _write_jpeg(path, *, orientation: int = 1, width: int = 64, height: int = 48):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[271] = "Atlas"            # Make
    exif[272] = "JPEG-1"           # Model
    exif[274] = orientation        # Orientation
    # Photographic tags must live in the Exif sub-IFD or exifread reports
    # them under "Image ..." instead of the "EXIF ..." keys the parser reads.
    from PIL.TiffImagePlugin import IFDRational
    sub_ifd = exif.get_ifd(0x8769)
    sub_ifd[0x920A] = IFDRational(53, 1)  # FocalLength (plain tuples drop)
    sub_ifd[0xA405] = 53           # FocalLengthIn35mmFilm -> full-frame ratio
    sub_ifd[0xA434] = "Atlas Prime"  # LensModel
    image.save(path, "JPEG", exif=exif, quality=95)


def test_jpeg_import_reads_exif_and_marks_the_trust_tier(tmp_path):
    path = tmp_path / "capture.jpg"
    _write_jpeg(path)
    result = import_raw(str(path))

    assert result.camera_make == "Atlas"
    assert result.camera_model == "JPEG-1"
    assert result.lens_model == "Atlas Prime"
    assert result.focal_length_mm == pytest.approx(53.0)
    assert result.metadata_source == "jpeg_exif"
    assert result.undistort_status == "camera_processed"
    assert result.undistort_applied is True
    assert result.headroom == 1.0
    # 35mm-equivalent == real focal -> full-frame width recovered.
    assert result.sensor_width_mm == pytest.approx(36.0, abs=0.5)
    assert result.width == 64 and result.height == 48
    assert result.display_srgb.dtype == np.float32
    assert float(result.display_srgb.max()) <= 1.0
    # linear is the exact sRGB EOTF inversion of display.
    mid = result.display_srgb[24, 32, 0]
    lin = result.linear_rgb[24, 32, 0]
    expected = ((mid + 0.055) / 1.055) ** 2.4 if mid > 0.04045 else mid / 12.92
    assert lin == pytest.approx(expected, abs=1e-6)


def test_jpeg_import_applies_orientation_and_reports_upright(tmp_path):
    path = tmp_path / "portrait.jpg"
    _write_jpeg(path, orientation=8, width=64, height=48)
    result = import_raw(str(path))

    # Rotated 90 CCW by exif_transpose: dimensions swap, tag reads upright so
    # the multi-view sensor-axis swap does NOT fire on already-upright pixels.
    assert (result.width, result.height) == (48, 64)
    assert result.orientation == 1


def test_jpeg_import_honours_half_size(tmp_path):
    path = tmp_path / "half.jpg"
    _write_jpeg(path)
    result = import_raw(str(path), half_size=True)
    assert (result.width, result.height) == (32, 24)
    # Sensor resolution must be derived from FULL-resolution dimensions.
    assert result.sensor_width_mm == pytest.approx(36.0, abs=0.5)
