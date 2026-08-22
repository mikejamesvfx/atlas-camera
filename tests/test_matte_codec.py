"""The strict codec and the comfy helper must stay one wire format.

`core.matte_codec` deliberately duplicates `comfy.node_helpers._mask_to_b64_png`
rather than sharing it: that helper fails soft to "" by design, which is right
for a node that would rather render un-matted than abort a graph and wrong for
an evidence path, and world-side code cannot import comfy at all. The
duplication is only safe while both ends produce and accept the same bytes.
"""

from __future__ import annotations

import pytest

from atlas_camera.core.matte_codec import decode_matte_png, encode_matte_png

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")


def _matte():
    matte = np.zeros((12, 16), dtype=np.float32)
    matte[3:8, 4:11] = 1.0
    matte[8, 4:11] = 0.5
    return matte


def test_the_comfy_helper_output_decodes_with_the_strict_decoder():
    from atlas_camera.comfy.node_helpers import _mask_to_b64_png

    encoded = _mask_to_b64_png(_matte())
    assert encoded.startswith("data:image/png;base64,")
    assert np.allclose(decode_matte_png(encoded), _matte(), atol=1.0 / 255.0)


def test_the_strict_output_decodes_with_the_comfy_decoder():
    from atlas_camera.comfy.node_helpers import _b64_png_to_mask

    decoded = _b64_png_to_mask(encode_matte_png(_matte()))
    assert decoded is not None
    assert np.array_equal(decoded, _matte() > 0.5)


def test_both_encoders_agree_byte_for_byte():
    from atlas_camera.comfy.node_helpers import _mask_to_b64_png

    assert encode_matte_png(_matte()) == _mask_to_b64_png(_matte())


def test_the_strict_codec_raises_where_the_soft_one_returns_empty():
    from atlas_camera.comfy.node_helpers import _mask_to_b64_png

    bad = np.zeros((2, 2, 3), dtype=np.float32)
    # The whole reason for two implementations: same input, opposite contract.
    assert _mask_to_b64_png(bad) == ""
    with pytest.raises(ValueError):
        encode_matte_png(bad)
