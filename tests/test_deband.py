"""plate.deband — gradient-gated model-free debanding (float-safe doctrine)."""

import numpy as np
import pytest

from atlas_camera.plate.deband import deband_plate


def _banded_gradient(h=64, w=256, levels=12):
    """Smooth horizontal ramp quantized to a handful of 8-bit-ish plateaus."""
    ramp = np.linspace(0.2, 0.4, w)[None, :] * np.ones((h, 1))
    steps = np.round(ramp * levels * (255.0 / levels)) / 255.0
    return np.stack([steps] * 3, axis=-1).astype(np.float32)


def _banding_energy(img):
    """Second-derivative spike energy along x — plateau steps light this up."""
    luma = img.mean(axis=-1) if img.ndim == 3 else img
    d2 = np.abs(np.diff(luma, n=2, axis=1))
    return float(d2.sum())


def test_banding_energy_reduced_on_quantized_gradient():
    img = _banded_gradient()
    out = deband_plate(img, strength=1.0, band_threshold_lsb=2.0, radius_px=24)
    assert _banding_energy(out) < 0.5 * _banding_energy(img)
    # And the overall ramp is preserved (no washout): endpoints stay put.
    assert float(np.abs(out[:, 5] - img[:, 5]).max()) < 4.0 / 255.0
    assert float(np.abs(out[:, -5] - img[:, -5]).max()) < 4.0 / 255.0


def test_sharp_edge_preserved_bit_exact():
    img = np.zeros((32, 64, 3), dtype=np.float32)
    img[:, 32:] = 0.8  # a real edge, far above any banding threshold
    out = deband_plate(img, strength=1.0, band_threshold_lsb=2.0,
                       radius_px=16, preserve_detail=1.0)
    # Edge columns gate to zero — bit-exact pass-through at the discontinuity.
    assert np.array_equal(out[:, 31:33], img[:, 31:33])
    # And nothing anywhere moved more than the clamped correction.
    assert float(np.abs(out - img).max()) <= 4.0 * (2.0 / 255.0) + 1e-6


def test_strength_zero_is_identity():
    rng = np.random.default_rng(0)
    img = rng.random((16, 16, 3)).astype(np.float32)
    out = deband_plate(img, strength=0.0)
    assert np.array_equal(out, img)


def test_hdr_values_pass_through_unclamped():
    img = np.full((16, 16, 3), 3.5, dtype=np.float32)  # flat HDR plate
    img[:, :8] = 3.49
    out = deband_plate(img, strength=1.0)
    assert float(out.max()) > 3.0  # never clamped to [0, 1]
    assert np.isfinite(out).all()


def test_grain_is_gated_and_deterministic():
    img = _banded_gradient(h=32, w=64)
    a = deband_plate(img, strength=1.0, grain=0.01, seed=0)
    b = deband_plate(img, strength=1.0, grain=0.01, seed=0)
    assert np.array_equal(a, b)
    c = deband_plate(img, strength=1.0, grain=0.0, seed=0)
    assert not np.array_equal(a, c)


def test_grayscale_2d_shape_round_trips():
    img = _banded_gradient()[..., 0]
    out = deband_plate(img, strength=1.0)
    assert out.shape == img.shape


def test_node_batch_and_report():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_solve import AtlasDeband
    img = torch.from_numpy(np.stack([_banded_gradient(32, 64)] * 2, axis=0))
    out, report = AtlasDeband().deband(img, strength=1.0)
    assert out.shape == img.shape
    assert "un-debanded" in report  # the P0 trust-path caveat is stated
