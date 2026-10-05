"""Frame-outpaint ring: smeared along the frame edge, real plate untouched."""

import time

import numpy as np

from atlas_camera.core.mask_ops import smear_outpaint_ring


def _ripple(rows):
    col = rows.mean(0)[:, 0]
    k = 61
    t = np.convolve(np.pad(col, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return float(((col - t) / np.maximum(t, 1e-6)).std())


def _padded(pad=200):
    rng = np.random.default_rng(0)
    plate = (0.4 + 0.2 * rng.random((300, 500, 3))).astype(np.float32)   # busy top edge
    return plate, np.pad(plate, ((pad, pad), (pad, pad), (0, 0)), mode="edge"), pad


def test_real_plate_is_untouched():
    plate, padded, pad = _padded()
    out = smear_outpaint_ring(padded, pad)
    np.testing.assert_array_equal(out[pad:-pad, pad:-pad], plate)


def test_ring_stripes_are_smoothed():
    _, padded, pad = _padded()
    out = smear_outpaint_ring(padded, pad)
    far = slice(0, pad // 2)                                   # well out into the ring
    assert _ripple(padded[far, pad:-pad]) > 0.05
    assert _ripple(out[far, pad:-pad]) < 0.2 * _ripple(padded[far, pad:-pad])


def test_continuous_at_the_frame_edge():
    _, padded, pad = _padded()
    out = smear_outpaint_ring(padded, pad)
    step = np.abs(out[pad - 1, pad:-pad] - out[pad, pad:-pad]).mean()
    assert step < 0.01                                         # first ring row ~ the edge


def test_noop_without_padding_and_fast_on_8k_ring():
    a = np.random.default_rng(1).random((40, 60, 3)).astype(np.float32)
    np.testing.assert_array_equal(smear_outpaint_ring(a, 0), a)
    big = np.zeros((4512 + 2048, 7680 + 2048, 3), np.float32)
    t0 = time.perf_counter()
    smear_outpaint_ring(big, 1024)
    assert time.perf_counter() - t0 < 60
