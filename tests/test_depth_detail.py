"""core.depth_detail — Frankot-Chellappa integration, high-pass, and the
scale-preserving blend, plus the AtlasDepthDetailEnhance/AtlasDepthCombine
node contracts (copy-not-mutate doctrine for the SHARED depth object).
"""

import numpy as np
import pytest

from atlas_camera.core.depth_detail import (
    blend_depth_detail,
    combine_depth_high_freq,
    highpass_detail,
    integrate_normals_frankot_chellappa,
)


def _bumpy_surface(h=64, w=64, freq=6.0, amp=0.5):
    """Smooth periodic surface with an analytic normal field."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    z = amp * np.sin(2 * np.pi * freq * x / w) * np.cos(2 * np.pi * freq * y / h)
    dzdx = amp * (2 * np.pi * freq / w) * np.cos(2 * np.pi * freq * x / w) * np.cos(2 * np.pi * freq * y / h)
    dzdy = -amp * (2 * np.pi * freq / h) * np.sin(2 * np.pi * freq * x / w) * np.sin(2 * np.pi * freq * y / h)
    n = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return z, n


# ---------------------------------------------------------------------------
# core math
# ---------------------------------------------------------------------------

def test_fc_round_trip_recovers_surface():
    z, n = _bumpy_surface()
    rec = integrate_normals_frankot_chellappa(n)
    # Both zero-mean; a pure-frequency surface should survive integration
    # nearly intact (boundary mirroring costs a little).
    z0 = z - z.mean()
    err = float(np.sqrt(np.mean((rec - z0) ** 2))) / float(np.std(z0))
    assert err < 0.15


def test_fc_flipped_z_convention_gives_same_magnitude():
    _, n = _bumpy_surface()
    rec_a = integrate_normals_frankot_chellappa(n)
    rec_b = integrate_normals_frankot_chellappa(n * np.array([1.0, 1.0, -1.0]))
    # z-toward vs z-away conventions may only flip the height sign.
    assert np.allclose(np.abs(rec_a), np.abs(rec_b), atol=1e-8)


def test_fc_rejects_bad_shape():
    with pytest.raises(ValueError):
        integrate_normals_frankot_chellappa(np.zeros((4, 4)))


def test_highpass_removes_coarse_keeps_fine():
    h = w = 128
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    # Mirror-even coarse wave (half cosine across the width, wavelength 256 px)
    # so the filter's mirror extension is seam-free and the measurement isolates
    # the rolloff itself rather than mirror-harmonic leakage.
    coarse = np.cos(np.pi * x / w)
    fine = 0.3 * np.sin(2 * np.pi * x / 8.0)        # wavelength 8 px
    out = highpass_detail(coarse + fine, cutoff_px=32.0)
    # Fine survives, coarse is mostly gone.
    assert float(np.std(out)) == pytest.approx(float(np.std(fine)), rel=0.25)
    out_coarse_only = highpass_detail(coarse, cutoff_px=32.0)
    assert float(np.std(out_coarse_only)) < 0.1 * float(np.std(coarse))
    assert abs(float(out.mean())) < 1e-9


def test_blend_preserves_median_and_positivity():
    rng = np.random.default_rng(7)
    depth = rng.uniform(2.0, 200.0, (64, 64))
    detail = highpass_detail(rng.normal(0, 1, (64, 64)), 16.0)
    out = blend_depth_detail(depth, detail, strength=1.0)
    assert np.nanmedian(out) == pytest.approx(np.nanmedian(depth), rel=0.005)
    assert (out > 0).all()
    assert not np.allclose(out, depth)  # it did do something


def test_blend_strength_zero_is_identity_and_nan_passthrough():
    depth = np.full((8, 8), 5.0)
    depth[0, 0] = np.nan
    out0 = blend_depth_detail(depth, np.zeros((8, 8)), strength=0.0)
    assert np.allclose(out0[1:], depth[1:]) and np.isnan(out0[0, 0])
    out1 = blend_depth_detail(depth, np.ones((8, 8)), strength=1.0)
    assert np.isnan(out1[0, 0])


def test_blend_exclude_mask_pins_pixels():
    rng = np.random.default_rng(3)
    depth = rng.uniform(1.0, 10.0, (32, 32))
    detail = rng.normal(0, 1, (32, 32))
    excl = np.zeros((32, 32), bool)
    excl[:16] = True
    out = blend_depth_detail(depth, detail, strength=1.0, exclude_mask=excl)
    assert np.allclose(out[:16], depth[:16])


def test_blend_proportional_at_distance():
    """The same detail wiggle displaces ~proportionally at 2 m and 200 m."""
    depth = np.concatenate([np.full((8, 8), 2.0), np.full((8, 8), 200.0)], axis=0)
    detail = np.tile(np.sin(np.linspace(0, 4 * np.pi, 8))[None, :], (16, 1))
    out = blend_depth_detail(depth, detail, strength=1.0, amplitude=0.02)
    rel_near = np.ptp(out[:8] / depth[:8])
    rel_far = np.ptp(out[8:] / depth[8:])
    assert rel_near == pytest.approx(rel_far, rel=1e-6)


def test_combine_high_freq_preserves_base_scale():
    rng = np.random.default_rng(11)
    base = np.full((64, 64), 50.0) + rng.normal(0, 0.5, (64, 64))
    src = 5.0 * (1.0 + 0.2 * np.sin(np.linspace(0, 40, 64))[None, :]
                 * np.ones((64, 1)))
    out = combine_depth_high_freq(base, src, strength=1.0, cutoff_px=16.0)
    assert np.nanmedian(out) == pytest.approx(np.nanmedian(base), rel=0.005)


# ---------------------------------------------------------------------------
# nodes (copy-not-mutate doctrine)
# ---------------------------------------------------------------------------

def _depth_result(depth_arr, normal=None, is_metric=True):
    from atlas_camera.inference.depth_estimator import DepthResult
    d = np.asarray(depth_arr, np.float32)
    return DepthResult(depth=d, is_metric=is_metric, model_id="test/model",
                       image_width=d.shape[1], image_height=d.shape[0],
                       near=float(np.nanmin(d)), far=float(np.nanmax(d)),
                       metadata={"origin": "fixture"}, normal=normal)


def test_enhance_node_copies_never_mutates():
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_depth import AtlasDepthDetailEnhance
    rng = np.random.default_rng(0)
    _, n = _bumpy_surface(48, 48)
    src = _depth_result(rng.uniform(2, 40, (48, 48)), normal=n.astype(np.float32))
    before = src.depth.copy()
    out, report = AtlasDepthDetailEnhance().enhance(src, strength=0.5)
    assert out is not src
    assert np.array_equal(src.depth, before)          # shared object untouched
    assert src.metadata == {"origin": "fixture"}
    assert out.metadata["detail_enhanced"] is True
    assert out.is_metric is True
    assert np.nanmedian(out.depth) == pytest.approx(np.nanmedian(before), rel=0.005)
    assert "scale unchanged" in report.lower() or "Metric scale unchanged" in report


def test_enhance_node_soft_passthrough_without_normals():
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_depth import AtlasDepthDetailEnhance
    src = _depth_result(np.full((16, 16), 3.0))
    out, report = AtlasDepthDetailEnhance().enhance(src)
    assert out is src
    assert "no normals" in report


def test_combine_node_min_max_exact():
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_depth import AtlasDepthCombine
    a = _depth_result(np.full((8, 8), 5.0))
    b = _depth_result(np.full((8, 8), 3.0))
    out_min, _ = AtlasDepthCombine().combine(a, b, mode="min")
    out_max, _ = AtlasDepthCombine().combine(a, b, mode="max")
    assert np.allclose(out_min.depth, 3.0)
    assert np.allclose(out_max.depth, 5.0)
    assert out_min is not a and np.allclose(a.depth, 5.0)
    assert out_min.metadata["combined_mode"] == "min"


def test_combine_node_metric_mismatch_warns_keeps_base_flag():
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_depth import AtlasDepthCombine
    a = _depth_result(np.full((8, 8), 5.0), is_metric=True)
    b = _depth_result(np.full((8, 8), 0.5), is_metric=False)
    out, report = AtlasDepthCombine().combine(a, b, mode="min")
    assert out.is_metric is True
    assert "WARNING" in report
