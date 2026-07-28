"""Depth calibration: recover a known distortion from paired depth.

PROTOTYPE STATUS. These prove the MACHINERY — that a known, deliberately
injected error is recovered — not that any particular coefficient is right for
any real camera. Real coefficients need real LiDAR captures, which do not exist
on this machine yet. Nothing here should be read as a validated calibration.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.depth_calibration import (
    DepthCorrection,
    apply_depth_correction,
    choose_correction,
    fit_depth_correction,
)

np = pytest.importorskip("numpy")


def _truth(h=64, w=64, near=1.5, far=30.0):
    """A depth ramp with enough far-field to expose disparity-space error."""
    return np.linspace(near, far, h * w).reshape(h, w)


# ------------------------------------------------------------- recovery


def test_recovers_a_pure_scale_error():
    truth = _truth()
    predicted = truth / 2.5
    c = fit_depth_correction(predicted, truth, model="scale")
    assert c.a == pytest.approx(2.5, rel=1e-6)
    assert np.allclose(apply_depth_correction(predicted, c), truth, rtol=1e-5)


def test_recovers_an_affine_depth_error():
    truth = _truth()
    predicted = (truth - 3.0) / 2.0            # truth = 2*pred + 3
    c = fit_depth_correction(predicted, truth, model="affine")
    assert c.a == pytest.approx(2.0, rel=1e-6)
    assert c.b == pytest.approx(3.0, abs=1e-6)


def test_recovers_an_affine_disparity_error():
    """The case that actually matters for monocular models.

    A network regressing disparity gets near surfaces roughly right and
    compresses the far field, which is affine in 1/d and NOT affine in d.
    """
    truth = _truth()
    disp = 1.0 / truth
    predicted = 1.0 / (0.7 * disp + 0.01)      # 1/truth = (1/pred - 0.01)/0.7
    c = fit_depth_correction(predicted, truth, model="affine_disparity")
    assert np.allclose(apply_depth_correction(predicted, c), truth, rtol=1e-4)


def test_fitting_in_the_wrong_space_leaves_far_field_error():
    """Why `affine_disparity` is the default, stated as a measurement.

    The same disparity-space distortion fitted in DEPTH space looks acceptable
    overall and stays clearly worse — most of the residual sitting in the far
    field, where a matte painting actually lives.
    """
    truth = _truth(near=1.5, far=60.0)
    predicted = 1.0 / (0.7 * (1.0 / truth) + 0.01)

    right = fit_depth_correction(predicted, truth, model="affine_disparity")
    wrong = fit_depth_correction(predicted, truth, model="affine")
    assert right.mae_after < wrong.mae_after * 0.2, (
        f"disparity fit {right.mae_after:.4f} vs depth fit {wrong.mae_after:.4f} — "
        "the default is no longer justified")

    far = truth > np.percentile(truth, 75)
    err_wrong = np.abs(apply_depth_correction(predicted, wrong) - truth)[far]
    err_right = np.abs(apply_depth_correction(predicted, right) - truth)[far]
    assert np.median(err_right) < np.median(err_wrong)


# ------------------------------------------------------------- guards


def test_refuses_to_fit_on_too_few_samples():
    truth = np.array([[2.0, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError, match="valid paired samples"):
        fit_depth_correction(truth / 2, truth, model="scale")


def test_mismatched_shapes_say_what_to_do():
    with pytest.raises(ValueError, match="resample"):
        fit_depth_correction(np.ones((64, 64)), np.ones((32, 32)))


def test_zero_and_nan_samples_are_excluded():
    """Record3D writes 0 for "no return"; 1/0 would poison a disparity fit."""
    truth = _truth()
    predicted = truth / 2.0
    predicted[:8, :] = 0.0
    truth_holed = truth.copy()
    truth_holed[-8:, :] = np.nan
    c = fit_depth_correction(predicted, truth_holed, model="scale")
    assert c.a == pytest.approx(2.0, rel=1e-6)
    assert c.n_samples < truth.size


def test_correction_never_returns_negative_depth():
    """A negative result is a sign flip, not a rescale — it puts geometry
    behind the camera, where every downstream stage misreads it."""
    d = np.linspace(1.0, 10.0, 4096).reshape(64, 64)
    c = DepthCorrection(model="affine", a=1.0, b=-50.0, n_samples=4096)
    out = apply_depth_correction(d, c)
    assert not (out[np.isfinite(out)] <= 0).any()


def test_mask_restricts_the_fit():
    truth = _truth()
    predicted = truth / 2.0
    predicted[:32, :] = truth[:32, :] / 9.0     # a wrong region
    mask = np.zeros_like(truth, dtype=bool)
    mask[32:, :] = True
    c = fit_depth_correction(predicted, truth, mask=mask, model="scale")
    assert c.a == pytest.approx(2.0, rel=1e-3)


# ------------------------------------------------------------- selection


def test_choose_reports_every_candidate_and_picks_the_best():
    truth = _truth()
    predicted = 1.0 / (0.7 * (1.0 / truth) + 0.01)
    c = choose_correction(predicted, truth)
    assert c.model == "affine_disparity"
    assert set(c.metadata["candidates"]) == {"affine_disparity", "affine", "scale"}
    assert c.improvement > 0.5


def test_choose_warns_when_nothing_helps():
    """An unrelated pair must not yield a confident-looking correction."""
    rng = np.random.default_rng(0)
    truth = _truth()
    predicted = rng.uniform(1.0, 30.0, truth.shape)
    c = choose_correction(predicted, truth)
    assert c.improvement <= 0.35, "noise should not be 'corrected' convincingly"


def test_round_trips_through_a_dict():
    """Corrections are meant to be persisted per model and scene type."""
    truth = _truth()
    c = fit_depth_correction(truth / 3.0, truth, model="scale")
    back = DepthCorrection.from_dict(c.to_dict())
    assert back.model == c.model and back.a == pytest.approx(c.a)
    d = np.linspace(1, 20, 4096).reshape(64, 64)
    assert np.allclose(apply_depth_correction(d, c), apply_depth_correction(d, back))


def test_improvement_is_reported_honestly():
    """mae_before AND mae_after both ride along.

    A fit that barely helps is worth seeing; it is invisible if only the final
    error is reported.
    """
    truth = _truth()
    c = fit_depth_correction(truth / 2.0, truth, model="scale")
    assert c.mae_before > 0
    assert c.mae_after < c.mae_before
    assert 0.0 < c.improvement <= 1.0
