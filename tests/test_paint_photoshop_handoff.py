"""The manual Photoshop lane: colour-exact handoff, and hand-painted mattes.

Two measured facts define this contract, and both fail SILENTLY if ignored:

* Going out, Photoshop assumes an OCIO EXR is ACES2065-1 and ignores the
  file's colourspace tag.
* Coming back, Photoshop's TIFF carries NO colourspace tag, so reading it on
  'auto' makes OIIO guess `sRGB - Display` from the extension and mis-convert
  scene-linear data.

Neither raises. Both produce a plausible-looking picture that is wrong.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("OpenImageIO")
pytest.importorskip("PIL")

import OpenImageIO as oiio  # noqa: E402
from OpenImageIO import ImageBuf, ImageSpec, ROI  # noqa: E402

from atlas_camera.paint.photoshop import handoff  # noqa: E402
from atlas_camera.plate.oiio_io import read_plate, resolve_colorspace  # noqa: E402


W, H = 48, 32


def _plate(tmp_path, name="plate.exr", space="Linear Rec.709 (sRGB)",
           bit_depth="half"):
    """A half-float plate, because that is what AtlasLoadRAW actually writes."""
    from atlas_camera.plate.oiio_io import write_exr

    yy, xx = np.mgrid[0:H, 0:W]
    px = np.stack([xx / W * 2.5, yy / H * 1.5,
                   np.full((H, W), 0.3)], axis=-1).astype("float32")
    path = tmp_path / name
    write_exr(str(path), px, bit_depth=bit_depth, source_colorspace=space)
    return path


def _fake_photoshop_tiff(tmp_path, rgb, mattes=(), name="ps.tif"):
    """A TIFF shaped exactly like Photoshop's export: float, NO colourspace
    tag, RGB plus one empty transparency alpha plus the painted mattes."""
    n = 3 + 1 + len(mattes)
    spec = ImageSpec(W, H, n, "float")
    buf = ImageBuf(spec)
    data = np.zeros((H, W, n), "float32")
    data[..., :3] = rgb
    for i, m in enumerate(mattes):
        data[..., 4 + i] = m
    buf.set_pixels(ROI(), data)
    path = tmp_path / name
    buf.write(str(path))
    return path


# --- send -------------------------------------------------------------------

def test_send_converts_to_what_photoshop_will_assume(tmp_path):
    """Photoshop does not read the tag; it assumes ACES2065-1. So Atlas must
    put the pixels in that space, or the primaries are mis-read silently."""
    src = _plate(tmp_path)
    out = tmp_path / "handoff.exr"
    result = handoff.send(plate_path=src, out_path=out)

    assert result["sent_as"] == "ACES2065-1"
    written = read_plate(str(out), raw_data=True)
    assert (resolve_colorspace(written.input_colorspace)
            == resolve_colorspace("ACES2065-1"))

    expected = read_plate(str(src), output_colorspace="ACES2065-1").pixels
    assert np.abs(written.pixels - expected).max() < 1e-5


def test_send_writes_lossless_float(tmp_path):
    """half selects the lossy dwab codec, and this file is the input to a
    measured round trip."""
    out = tmp_path / "handoff.exr"
    handoff.send(plate_path=_plate(tmp_path), out_path=out)
    assert read_plate(str(out), raw_data=True).file_bit_depth == "float"


# --- receive ----------------------------------------------------------------

def test_receive_does_not_trust_the_untagged_tiff(tmp_path):
    """Photoshop's TIFF has no colourspace attribute. On 'auto' OIIO guesses
    'sRGB - Display' from the .tif extension, which would treat scene-linear
    ACEScg as display sRGB. receive must state what it is instead."""
    rgb = np.random.default_rng(0).random((H, W, 3)).astype("float32") * 2.0
    tif = _fake_photoshop_tiff(tmp_path, rgb)

    # The trap, demonstrated: auto infers a display space for this file.
    guessed = read_plate(str(tif))
    assert resolve_colorspace(guessed.input_colorspace) == resolve_colorspace(
        "sRGB - Display")

    out = tmp_path / "back.exr"
    result = handoff.receive(tiff_path=tif, out_path=out,
                             working_space="ACEScg", target_space="ACEScg")
    assert (resolve_colorspace(result["retagged_as"])
            == resolve_colorspace("ACEScg"))
    # No conversion was asked for, so the pixels must be untouched.
    assert np.abs(read_plate(str(out), raw_data=True).pixels - rgb).max() < 1e-6


def test_receive_converts_from_the_working_space_to_the_target(tmp_path):
    rgb = (np.random.default_rng(1).random((H, W, 3)) * 1.5).astype("float32")
    tif = _fake_photoshop_tiff(tmp_path, rgb)
    out = tmp_path / "back.exr"
    handoff.receive(tiff_path=tif, out_path=out, working_space="ACEScg",
                    target_space="Linear Rec.709 (sRGB)")

    got = read_plate(str(out), raw_data=True)
    assert (resolve_colorspace(got.input_colorspace)
            == resolve_colorspace("Linear Rec.709 (sRGB)"))
    # It must actually have converted, not merely re-labelled.
    assert np.abs(got.pixels - rgb).max() > 1e-3


# --- mattes -----------------------------------------------------------------

def test_painted_mattes_come_back_as_masks(tmp_path):
    boiler = np.zeros((H, W), "float32"); boiler[5:20, 5:25] = 1.0
    sky = np.zeros((H, W), "float32"); sky[0:6, :] = 1.0
    tif = _fake_photoshop_tiff(tmp_path, np.zeros((H, W, 3), "float32"),
                               mattes=(boiler, sky))

    result = handoff.receive(tiff_path=tif, out_path=tmp_path / "b.exr",
                             matte_dir=tmp_path / "mattes",
                             matte_names=["boiler", "sky"])
    kept = [m for m in result["mattes"] if not m.get("skipped")]
    assert [m["name"] for m in kept] == ["boiler", "sky"]
    assert kept[0]["coverage"] == pytest.approx(boiler.mean(), abs=0.01)
    assert kept[1]["coverage"] == pytest.approx(sky.mean(), abs=0.01)
    for m in kept:
        assert (tmp_path / "mattes" / f"{m['name']}.png").is_file()


def test_photoshops_empty_transparency_alpha_does_not_steal_a_label(tmp_path):
    """Photoshop injects its own (empty) alpha as the FIRST extra channel.
    Labelling by raw channel index made it consume the first name and shifted
    every matte name by one — the artist's 'boiler' landed on nothing and
    'sky' landed on the boiler."""
    boiler = np.zeros((H, W), "float32"); boiler[5:20, 5:25] = 1.0
    tif = _fake_photoshop_tiff(tmp_path, np.zeros((H, W, 3), "float32"),
                               mattes=(boiler,))

    result = handoff.receive(tiff_path=tif, out_path=tmp_path / "b.exr",
                             matte_dir=tmp_path / "m", matte_names=["boiler"])
    skipped = [m for m in result["mattes"] if m.get("skipped")]
    kept = [m for m in result["mattes"] if not m.get("skipped")]
    assert len(skipped) == 1, "the empty transparency alpha must be skipped"
    assert [m["name"] for m in kept] == ["boiler"]
    assert (tmp_path / "m" / "boiler.png").is_file()


def test_empty_mattes_can_be_kept_deliberately(tmp_path):
    tif = _fake_photoshop_tiff(tmp_path, np.zeros((H, W, 3), "float32"),
                               mattes=(np.zeros((H, W), "float32"),))
    result = handoff.receive(tiff_path=tif, out_path=tmp_path / "b.exr",
                             keep_empty_mattes=True)
    assert not any(m.get("skipped") for m in result["mattes"])


def test_unnamed_mattes_get_positional_labels(tmp_path):
    m0 = np.zeros((H, W), "float32"); m0[1:4, 1:4] = 1.0
    tif = _fake_photoshop_tiff(tmp_path, np.zeros((H, W, 3), "float32"),
                               mattes=(m0,))
    result = handoff.receive(tiff_path=tif, out_path=tmp_path / "b.exr")
    kept = [m for m in result["mattes"] if not m.get("skipped")]
    assert kept[0]["name"] == "matte_0"
