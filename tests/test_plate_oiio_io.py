"""Tests for the OpenImageIO-backed colour-managed plate I/O.

These pin the two properties that make Atlas a film tool rather than a
screenshot tool:

1. HDR survives. Scene-referred values above 1.0 must round-trip, because
   that is the entire reason for using EXR instead of PNG.
2. `raw_data` is genuinely raw. Depth, normals, mattes and UV passes are DATA;
   colour-converting them corrupts them silently, and silent corruption in a
   float pipeline is the worst possible failure.

Skips cleanly when the [oiio] extra is absent, matching how the other
optional-dependency suites behave.
"""

import numpy as np
import pytest

from atlas_camera.plate import (
    auto_colorspace_for_path,
    list_colorspaces,
    oiio_available,
    oiio_diagnostics,
    read_plate,
    write_exr,
)

pytestmark = pytest.mark.skipif(not oiio_available(),
                                reason="needs the [oiio] extra (OpenImageIO)")


def _ramp(h=8, w=12):
    a = np.full((h, w, 3), 0.18, dtype="float32")
    a[0, 0] = 8.0          # specular highlight, far above display white
    a[1, 1] = 0.0
    return a


def test_diagnostics_names_the_backend():
    assert "OpenImageIO" in oiio_diagnostics()


def test_builtin_aces_config_is_available_without_any_config_file():
    """OIIO's built-in ACES config is why this needs no `opencolorio` install
    and no $OCIO set up by the user."""
    spaces = list_colorspaces()
    for required in ("ACEScg", "ACES2065-1", "ACEScct"):
        assert required in spaces, f"{required} missing from {spaces}"


def test_auto_colorspace_by_extension():
    assert auto_colorspace_for_path("/x/plate.exr") == "ACEScg"
    assert auto_colorspace_for_path("/x/plate.dpx") == "ACEScg"
    assert auto_colorspace_for_path("/x/ref.jpg") == "sRGB - Display"
    assert auto_colorspace_for_path("/x/ref.PNG") == "sRGB - Display"


def test_hdr_survives_a_half_float_round_trip(tmp_path):
    """The whole point of EXR: values above 1.0 are not clipped."""
    src = _ramp()
    p = write_exr(str(tmp_path / "hdr.exr"), src, bit_depth="half",
                  source_colorspace="ACEScg")
    got = read_plate(p, raw_data=True)
    assert got.pixels[0, 0, 0] > 7.9, "highlight was clipped — not HDR"
    assert got.is_float and got.file_bit_depth == "half"
    assert (got.width, got.height) == (12, 8)


def test_raw_data_does_not_touch_the_values(tmp_path):
    """A data pass must come back bit-for-bit (modulo half quantisation)."""
    src = _ramp()
    p = write_exr(str(tmp_path / "data.exr"), src, bit_depth="float",
                  source_colorspace="ACEScg")
    got = read_plate(p, raw_data=True)
    assert np.allclose(got.pixels, src, atol=1e-6)
    assert got.output_colorspace == "", "raw read should record no conversion"


def test_colour_conversion_applies_the_transfer_function(tmp_path):
    """0.18 scene-linear through sRGB-Display is ~0.46. If this drifts, the
    colour pipeline is wrong in a way that looks plausible on screen."""
    p = write_exr(str(tmp_path / "grey.exr"), np.full((4, 4, 3), 0.18, "float32"),
                  source_colorspace="ACEScg")
    got = read_plate(p, output_colorspace="sRGB - Display")
    assert 0.44 < float(got.pixels[0, 0, 0]) < 0.48, got.pixels[0, 0, 0]
    assert got.output_colorspace == "sRGB - Display"


def test_written_file_self_describes_its_colorspace(tmp_path):
    """A tagged plate can be read back correctly with no out-of-band knowledge —
    the thing a DCC handoff needs and opencv's writer cannot do.

    NOTE: OIIO normalises names to the config's canonical form, so a file
    written as 'ACEScg' reports the equivalent 'lin_ap1_scene'. Aliases resolve,
    so conversion is unaffected; assert on behaviour, not on the literal string.
    """
    p = write_exr(str(tmp_path / "tagged.exr"), np.full((4, 4, 3), 0.5, "float32"),
                  source_colorspace="ACEScg")
    got = read_plate(p, input_colorspace="auto", raw_data=True)
    assert got.input_colorspace, "no colorspace recorded on the file"
    # Reading with a deliberately wrong hint must be OVERRIDDEN by the file's tag.
    hinted = read_plate(p, input_colorspace="sRGB - Display", raw_data=True)
    assert hinted.input_colorspace == got.input_colorspace


def test_alpha_is_returned_separately(tmp_path):
    rgba = np.zeros((6, 6, 4), dtype="float32")
    rgba[..., :3] = 0.4
    rgba[..., 3] = 0.75
    p = write_exr(str(tmp_path / "rgba.exr"), rgba, bit_depth="half")
    got = read_plate(p, raw_data=True)
    assert got.nchannels == 4
    assert got.alpha is not None and np.allclose(got.alpha, 0.75, atol=1e-2)
    assert got.pixels.shape[2] == 3, "RGB must be split from alpha"


def test_missing_file_raises_actionably(tmp_path):
    with pytest.raises(RuntimeError, match="Could not read"):
        read_plate(str(tmp_path / "nope.exr"))


def test_bad_colorspace_names_the_available_ones(tmp_path):
    p = write_exr(str(tmp_path / "x.exr"), np.zeros((4, 4, 3), "float32"))
    with pytest.raises(RuntimeError, match="Colour conversion"):
        read_plate(p, input_colorspace="ACEScg", output_colorspace="NotARealSpace")


def test_associated_alpha_is_handled_correctly(tmp_path):
    """Colour conversion must unpremultiply, convert, then re-premultiply.

    EXR alpha is ASSOCIATED (premultiplied). Applying a transfer function to
    premultiplied values directly is wrong, and wrong in a way that looks
    merely "a bit dark" rather than broken — the worst kind of colour bug.

    0.18 at alpha 0.5 must give srgb(0.18/0.5)*0.5 = 0.3171, not srgb(0.18) =
    0.4614. Pinned because this is a correctness property opencv's codec does
    not provide, and it is a large part of why this migration is worth doing.
    """
    def srgb(x):
        return 1.055 * x ** (1 / 2.4) - 0.055 if x > 0.0031308 else 12.92 * x

    for alpha in (1.0, 0.5):
        px = np.full((8, 8, 4), 0.18, dtype="float32")
        px[..., 3] = alpha
        p = write_exr(str(tmp_path / f"a{alpha}.exr"), px, bit_depth="float",
                      source_colorspace="ACEScg")
        got = float(read_plate(p, output_colorspace="sRGB - Display").pixels[1, 1, 0])
        assert abs(got - srgb(0.18 / alpha) * alpha) < 0.01, (alpha, got)
