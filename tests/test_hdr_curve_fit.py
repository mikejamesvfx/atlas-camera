"""fit_sdr_to_hdr_curve: bounded sampling and a truthful refusal."""

import numpy as np
import pytest

from atlas_camera.core.hdr_transfer import (
    MIN_BIN_PIXELS,
    apply_curve,
    fit_sdr_to_hdr_curve,
    srgb_to_acescg,
)


def _model(x_acescg):
    return np.where(x_acescg < 0.4, x_acescg, 0.4 + (x_acescg - 0.4) * 6.0)


def _plate_pair(h=300, w=400, seed=0):
    sdr = np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)
    return sdr, _model(srgb_to_acescg(sdr)).astype(np.float32)


def test_subsampled_fit_matches_the_full_fit():
    """Above max_samples the fit draws a seeded sample; the curve must agree
    with the all-pixel fit to well under the one-curve residual budget."""
    sdr, hdr = _plate_pair()
    full = fit_sdr_to_hdr_curve(sdr, hdr)
    sub = fit_sdr_to_hdr_curve(sdr, hdr, max_samples=40_000)
    assert full["samples"] == 300 * 400
    assert sub["samples"] == 40_000
    # Compare where the plate's pixels actually are: the deep-shadow tail of a
    # uniform-sRGB plate holds a handful of pixels per log bin and is
    # interpolated either way, so a knot-grid max is a statement about that
    # tail, not about the sampler.
    a = np.log2(apply_curve(full, sdr.reshape(-1, 3)))
    b = np.log2(apply_curve(sub, sdr.reshape(-1, 3)))
    d = np.abs(a - b)
    assert np.median(d) < 0.01                       # stops
    assert np.percentile(d, 99) < 0.05
    assert sub["residual_stops"] < 0.02


def test_subsampled_fit_is_deterministic_per_seed():
    sdr, hdr = _plate_pair(120, 160)
    a = fit_sdr_to_hdr_curve(sdr, hdr, max_samples=5_000, seed=3)
    b = fit_sdr_to_hdr_curve(sdr, hdr, max_samples=5_000, seed=3)
    assert a == b


def test_sampling_never_builds_a_full_permutation(monkeypatch):
    """choice(replace=False) materialises a permutation of EVERY pixel (~265 MB
    on an 8K plate); the sampler must draw only the requested count."""
    real = np.random.default_rng
    drawn = []

    class Spy:
        def __init__(self, seed):
            self._rng = real(seed)

        def choice(self, *a, **k):  # pragma: no cover - the regression itself
            raise AssertionError("fit_sdr_to_hdr_curve must not use rng.choice")

        def integers(self, low, high, size):
            drawn.append((high, size))
            return self._rng.integers(low, high, size=size)

    sdr, hdr = _plate_pair(100, 100)
    monkeypatch.setattr(np.random, "default_rng", Spy)
    fit_sdr_to_hdr_curve(sdr, hdr, max_samples=2_000)
    assert drawn == [(10_000, 2_000)]


def test_small_plate_names_the_real_cause():
    """Every bin under MIN_BIN_PIXELS used to read 'is the HDR plate empty?' --
    wrong: the HDR plate is fine, there are just too few pixels per bin."""
    sdr, hdr = _plate_pair(3, 4)
    with pytest.raises(ValueError) as err:
        fit_sdr_to_hdr_curve(sdr, hdr)
    msg = str(err.value)
    assert "empty" not in msg
    assert "too small or too flat" in msg
    assert f">= {MIN_BIN_PIXELS} pixels" in msg
    assert "12 px over 64 bins" in msg


def test_small_flat_plate_names_the_real_cause():
    sdr = np.full((3, 3, 3), 0.5, np.float32)
    hdr = srgb_to_acescg(sdr).astype(np.float32)
    with pytest.raises(ValueError, match="too small or too flat"):
        fit_sdr_to_hdr_curve(sdr, hdr)


def test_fewer_bins_rescues_a_small_plate():
    """The message's advice must actually work."""
    sdr, hdr = _plate_pair(16, 16)
    with pytest.raises(ValueError, match="too small or too flat"):
        fit_sdr_to_hdr_curve(sdr, hdr, bins=256)
    curve = fit_sdr_to_hdr_curve(sdr, hdr, bins=4)
    assert len(curve["log2_x"]) == 4
