"""Camera-body registry + RAW sensor-size fallback chain (stdlib-only)."""

import pytest

from atlas_camera.raw.metadata import (
    RawMetadata,
    _metadata_from_tags,
    resolve_sensor_size,
)
from atlas_camera.reference_data.camera_registry import (
    find_camera_body,
    load_camera_bodies,
)


def test_d810_lookup_returns_manufacturer_sensor():
    body = find_camera_body("NIKON CORPORATION", "NIKON D810")
    assert body is not None
    assert body.id == "nikon_d810"
    assert body.sensor_width_mm == pytest.approx(35.9)
    assert body.sensor_height_mm == pytest.approx(24.0)


def test_lookup_normalizes_case_and_whitespace():
    assert find_camera_body("nikon", "  nikon   d810 ") is not None
    assert find_camera_body("SONY", "ilce-7m3").id == "sony_a7m3"
    # Model string that duplicates the make word matches make-stripped aliases.
    assert find_camera_body("Canon", "Canon EOS R5").id == "canon_r5"


def test_unknown_model_returns_none():
    assert find_camera_body("Nikon", "NIKON D40") is None
    assert find_camera_body(None, None) is None
    assert find_camera_body("Nikon", None) is None


def test_registry_covers_the_users_brands():
    makes = {body.make for body in load_camera_bodies()}
    assert {"Nikon", "Canon", "Fujifilm", "Sony"} <= makes


def test_registry_entries_are_sane():
    for body in load_camera_bodies():
        assert 4.0 <= body.sensor_width_mm <= 70.0, body.id
        assert 3.0 <= body.sensor_height_mm <= 60.0, body.id
        assert body.model_aliases, body.id


def test_metadata_from_exifread_style_tags():
    meta = _metadata_from_tags({
        "Image Make": "NIKON CORPORATION",
        "Image Model": "NIKON D810",
        "EXIF LensModel": "AF-S NIKKOR 20mm f/1.8G ED",
        "EXIF FocalLength": "20",
        "EXIF FocalLengthIn35mmFilm": "20",
        "EXIF FNumber": "28/10",
        "EXIF ISOSpeedRatings": "64",
        "Image Orientation": "1",
    })
    assert meta.camera_model == "NIKON D810"
    assert meta.focal_length_mm == pytest.approx(20.0)
    assert meta.aperture == pytest.approx(2.8)
    assert meta.iso == 64
    assert meta.lens_model.startswith("AF-S NIKKOR 20mm")


def test_metadata_rational_and_zero_sentinels():
    meta = _metadata_from_tags({
        "EXIF FocalLength": "700/10",
        "EXIF FocalLengthIn35mmFilm": "0",  # EXIF "unknown" sentinel
    })
    assert meta.focal_length_mm == pytest.approx(70.0)
    assert meta.focal_length_35mm is None


def test_sensor_tier1_camera_db():
    meta = RawMetadata(camera_make="Nikon", camera_model="NIKON D810")
    res = resolve_sensor_size(meta, 7360, 4912)
    assert res.source == "camera_db"
    assert res.sensor_width_mm == pytest.approx(35.9)
    assert res.sensor_height_mm == pytest.approx(24.0)


def test_sensor_tier2_focal_plane_arithmetic():
    # 7360 px / 2050.8 px-per-cm ≈ 35.9mm (unit 3 = cm).
    meta = RawMetadata(camera_make="X", camera_model="UnknownCam 9000",
                       focal_plane_x_res=2050.8, focal_plane_y_res=2050.8,
                       focal_plane_res_unit=3)
    res = resolve_sensor_size(meta, 7360, 4912)
    assert res.source == "exif_focal_plane"
    assert res.sensor_width_mm == pytest.approx(35.89, abs=0.02)
    assert res.sensor_height_mm == pytest.approx(23.95, abs=0.02)


def test_sensor_tier2_inch_unit():
    # 6000 px / 4233.9 px-per-inch ≈ 36.0mm (unit 2 = inch).
    meta = RawMetadata(camera_model="UnknownCam",
                       focal_plane_x_res=4233.9, focal_plane_res_unit=2)
    res = resolve_sensor_size(meta, 6000, 4000)
    assert res.source == "exif_focal_plane"
    assert res.sensor_width_mm == pytest.approx(36.0, abs=0.05)


def test_sensor_tier2_garbage_rejected_falls_through():
    # Computed width of ~1470mm is impossible — clamp rejects, tier 3 wins.
    meta = RawMetadata(camera_model="UnknownCam",
                       focal_plane_x_res=5.0, focal_plane_res_unit=4,
                       focal_length_mm=20.0, focal_length_35mm=30.0)
    res = resolve_sensor_size(meta, 7360, 4912)
    assert res.source == "exif_35mm_ratio"
    assert any("implausible" in w for w in res.warnings)


def test_sensor_tier3_35mm_ratio():
    # APS-C: 24mm real / 36mm-equivalent -> 36 * 24/36 = 24.0mm width.
    meta = RawMetadata(camera_model="UnknownCam",
                       focal_length_mm=24.0, focal_length_35mm=36.0)
    res = resolve_sensor_size(meta, 6000, 4000)
    assert res.source == "exif_35mm_ratio"
    assert res.sensor_width_mm == pytest.approx(24.0)
    assert res.sensor_height_mm == pytest.approx(16.0)


def test_sensor_tier4_assumed_default_flagged():
    res = resolve_sensor_size(RawMetadata(), 6000, 4000)
    assert res.source == "assumed_default"
    assert res.sensor_width_mm == pytest.approx(36.0)
    assert any("assumed" in w.casefold() for w in res.warnings)
