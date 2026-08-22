"""Contracts for the scene-linear -> display preview encoder.

WHY THIS EXISTS. A roundtripped card carried ``mask_b64`` but ``image_b64:
null``, and the viewport textures a projection source from ``image_b64``
(``headless_evidence._decode_rgba``). A card with an alpha and no pixels is
invisible — which is exactly what the DSC_2552 verify graph showed.

The pixels live in an ACEScg EXR. Turning those into something a viewport can
display is a COLOUR TRANSFORM, not a cast: ACEScg primaries are not sRGB
primaries, so writing the linear values straight into an 8-bit PNG and calling
it a preview produces a visibly wrong image. These tests pin the transform.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from atlas_camera.core.preview_codec import (
    ACESCG_TO_SRGB,
    acescg_to_srgb_display,
    encode_preview_png,
)


def _decode(uri):
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64) / 255.0


# ------------------------------------------------------------------- colour


def test_mid_grey_survives_the_transform_as_grey():
    """A neutral in ACEScg must stay neutral in sRGB — the matrix rows each
    sum to one, and a transform that loses that tints every grey in the frame."""

    assert np.allclose(np.asarray(ACESCG_TO_SRGB).sum(axis=1), 1.0, atol=1e-3)

    out = acescg_to_srgb_display(np.full((2, 2, 3), 0.18))
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert np.allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_eighteen_percent_grey_lands_near_the_expected_display_value():
    """Scene-linear 0.18 through the sRGB EOTF is ~0.46, not 0.18. A preview
    that skips the transfer function is visibly, uniformly too dark."""

    out = acescg_to_srgb_display(np.full((1, 1, 3), 0.18))
    assert out[0, 0, 0] == pytest.approx(0.4613, abs=0.005)


def test_black_and_white_map_to_black_and_white():
    assert acescg_to_srgb_display(np.zeros((1, 1, 3)))[0, 0, 0] == pytest.approx(0.0)
    assert acescg_to_srgb_display(np.ones((1, 1, 3)))[0, 0, 0] == pytest.approx(1.0, abs=1e-6)


def test_saturated_acescg_is_not_the_same_as_saturated_srgb():
    """The load-bearing test. If the transform were a cast, pure ACEScg red
    would come back as pure sRGB red. It must not: AP1 red is outside the sRGB
    gamut, so the conversion pushes the other channels negative and they clip."""

    red = np.zeros((1, 1, 3))
    red[0, 0, 0] = 1.0
    out = acescg_to_srgb_display(red)

    assert out[0, 0, 0] == pytest.approx(1.0, abs=1e-6)   # clipped at the top
    assert out[0, 0, 1] == pytest.approx(0.0, abs=1e-6)   # driven negative, clipped
    assert out[0, 0, 2] == pytest.approx(0.0, abs=1e-6)
    # ...and a MID ACEScg red is measurably different from the same numbers
    # treated as sRGB, which is what makes the difference visible.
    mid = np.zeros((1, 1, 3))
    mid[0, 0, 0] = 0.5
    assert acescg_to_srgb_display(mid)[0, 0, 1] < 0.2


def test_values_above_one_clip_rather_than_wrapping():
    """Highlights clip: this is a display preview with no tonemap, and saying
    so is better than a rolloff nobody asked for. Wrapping would put black
    holes in every specular."""

    out = acescg_to_srgb_display(np.full((1, 1, 3), 40.0))
    assert np.all(out <= 1.0)
    assert out[0, 0, 0] == pytest.approx(1.0, abs=1e-6)


def test_non_finite_values_are_refused():
    with pytest.raises(ValueError, match="finite"):
        acescg_to_srgb_display(np.full((1, 1, 3), np.nan))


# -------------------------------------------------------------------- encode


def test_a_preview_round_trips_through_the_data_uri():
    rgb = np.zeros((8, 12, 3), dtype=np.float32)
    rgb[2:6, 3:9, 1] = 0.18

    decoded = _decode(encode_preview_png(rgb))

    assert decoded.shape == (8, 12, 3)
    assert decoded[0, 0, 1] == pytest.approx(0.0, abs=0.01)
    assert decoded[4, 5, 1] > 0.4


def test_an_rgba_input_drops_the_alpha_channel():
    """The alpha travels as ``mask_b64``; baking it into the preview would
    matte the card twice and darken its own edge."""

    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[..., :3] = 0.18
    rgba[..., 3] = 0.0

    decoded = _decode(encode_preview_png(rgba))
    assert decoded[0, 0, 0] > 0.4


def test_a_preview_is_downsampled_to_the_long_edge_cap():
    """A full-frame card preview is 7380x4928. Both viewport decoders resample
    to the target raster anyway, so shipping the full thing only bloats the
    solve JSON."""

    rgb = np.zeros((400, 1000, 3), dtype=np.float32)
    decoded = _decode(encode_preview_png(rgb, max_long_edge=250))

    assert max(decoded.shape[:2]) == 250
    assert decoded.shape[:2] == (100, 250)   # aspect ratio preserved


def test_an_image_below_the_cap_is_left_alone():
    rgb = np.zeros((10, 20, 3), dtype=np.float32)
    assert _decode(encode_preview_png(rgb, max_long_edge=2048)).shape[:2] == (10, 20)


def test_a_two_dimensional_array_is_refused():
    with pytest.raises(ValueError, match="HxWx3"):
        encode_preview_png(np.zeros((4, 4)))


def test_an_empty_image_is_refused():
    with pytest.raises(ValueError, match="empty"):
        encode_preview_png(np.zeros((0, 4, 3)))


def test_the_encoder_can_skip_the_colour_transform_for_data_already_display_referred():
    """Not every caller holds ACEScg. Applying the transform twice is a silent
    double-brightening, so it is opt-out rather than unconditional."""

    rgb = np.full((2, 2, 3), 0.5, dtype=np.float32)
    already = _decode(encode_preview_png(rgb, colorspace="srgb_display"))
    assert already[0, 0, 0] == pytest.approx(0.5, abs=0.01)


def test_an_unknown_colourspace_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="colorspace"):
        encode_preview_png(np.zeros((2, 2, 3)), colorspace="rec2020")


def test_an_in_gamut_colour_is_actually_matrixed_not_merely_gamma_encoded():
    """Mutation-found blind spot. Every earlier colour test used a neutral or a
    fully-saturated primary, and BOTH of those survive deleting the matrix —
    a neutral because the rows sum to one, a primary because it clips to the
    same corner either way. A non-neutral, non-clipping colour is the only
    thing that can tell the two apart.

    ACEScg (0.5, 0.3, 0.2) -> sRGB display (0.8263, 0.5611, 0.4612).
    Skipping the matrix would give (0.7354, 0.5838, 0.4845): a redder channel
    two points low and the other two points high — a wash nobody would notice
    by eye on a card, which is exactly why it needs a number.
    """

    src = np.zeros((1, 1, 3))
    src[0, 0] = (0.5, 0.3, 0.2)

    out = acescg_to_srgb_display(src)[0, 0]

    assert out[0] == pytest.approx(0.82628, abs=1e-4)
    assert out[1] == pytest.approx(0.56109, abs=1e-4)
    assert out[2] == pytest.approx(0.46124, abs=1e-4)
    # And state the direction: ACEScg -> sRGB widens saturation for in-gamut
    # colours, so the dominant channel moves UP and the others DOWN.
    naive = np.array([0.73536, 0.58383, 0.48453])
    assert out[0] > naive[0] and out[1] < naive[1] and out[2] < naive[2]
