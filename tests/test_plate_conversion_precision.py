"""Colour conversion must not be performed at the file's storage precision.

``ImageBufAlgo.colorconvert`` works in the BUFFER's precision, and ``read_plate``
used to decode at the file's native type. A half EXR therefore had its colour
conversion performed in HALF -- and half is the common case, since
``AtlasLoadRAW`` writes half sidecars and ``write_exr`` defaults to half.

Measured on a real plate (2026-08-22), a Rec.709 -> ACES2065-1 -> Rec.709 round
trip:

    convert at NATIVE (half) : max abs 0.00292969
    convert at FLOAT         : max abs 0.00000107      <- 2700x

It was found from outside: a Photoshop round trip measured 2.9e-3 end to end,
and an Atlas-only control with Photoshop removed measured the same, so
Photoshop contributed 1.2e-7 and the noise floor was ours. It had been setting
the floor for every colour-managed handoff Atlas performs, including the DCC
exports.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("OpenImageIO")

from atlas_camera.plate.oiio_io import read_plate, write_exr  # noqa: E402


W, H = 48, 32


def _half_plate(tmp_path, name="plate.exr", space="Linear Rec.709 (sRGB)"):
    """Half float, with HDR values -- exactly what AtlasLoadRAW writes."""
    yy, xx = np.mgrid[0:H, 0:W]
    px = np.stack([xx / W * 2.5, yy / H * 1.5,
                   np.full((H, W), 0.3)], axis=-1).astype("float32")
    path = tmp_path / name
    write_exr(str(path), px, bit_depth="half", source_colorspace=space)
    return path


def test_colour_conversion_of_a_half_plate_is_done_at_float_precision(tmp_path):
    """ImageBufAlgo.colorconvert works in the BUFFER's precision, so a half
    EXR had its colour conversion performed in HALF.

    Measured on a real plate: a Rec.709 -> ACES2065-1 -> Rec.709 round trip
    came back with max abs error 0.00292969 at native precision versus
    0.00000107 forced to float -- a 2700x difference that silently set the
    noise floor for every colour-managed handoff. AtlasLoadRAW writes half
    sidecars and write_exr defaults to half, so this was the common case.
    """
    path = _half_plate(tmp_path)
    original = read_plate(str(path), raw_data=True).pixels.astype(np.float64)

    ap0 = read_plate(str(path), output_colorspace="ACES2065-1")
    mid = tmp_path / "ap0.exr"
    write_exr(str(mid), ap0.pixels, bit_depth="float",
              source_colorspace="ACES2065-1")
    back = read_plate(str(mid),
                      output_colorspace="Linear Rec.709 (sRGB)").pixels
    delta = np.abs(original - back.astype(np.float64)).max()

    assert delta < 1e-4, (
        f"round trip through ACES2065-1 lost {delta:.8f} — colour conversion "
        f"is running at the file's native (half) precision again")


def test_file_bit_depth_still_reports_the_ON_DISK_type(tmp_path):
    """Forcing the decode to float must not make a half file claim to be float.

    ``file_bit_depth`` is documented as the on-disk format and
    ``AtlasLoadPlate`` derives ``is_proxy`` from ``is_float``, so reporting the
    decode type instead of the storage type would be a provenance lie. Caught
    by tests/test_plate_oiio_io.py the first time.
    """
    half = _half_plate(tmp_path, name="h.exr")
    assert read_plate(str(half), raw_data=True).file_bit_depth == "half"

    px = np.full((H, W, 3), 0.5, dtype="float32")
    full = tmp_path / "f.exr"
    write_exr(str(full), px, bit_depth="float", source_colorspace="ACEScg")
    assert read_plate(str(full), raw_data=True).file_bit_depth == "float"


def test_conversion_precision_holds_for_a_wide_gamut_hop(tmp_path):
    """ACEScg is the working space a paint bridge actually round-trips through."""
    path = _half_plate(tmp_path)
    original = read_plate(str(path), raw_data=True).pixels.astype(np.float64)

    ap1 = read_plate(str(path), output_colorspace="ACEScg")
    mid = tmp_path / "ap1.exr"
    write_exr(str(mid), ap1.pixels, bit_depth="float",
              source_colorspace="ACEScg")
    back = read_plate(str(mid), output_colorspace="Linear Rec.709 (sRGB)")

    assert np.abs(original - back.pixels.astype(np.float64)).max() < 1e-4
