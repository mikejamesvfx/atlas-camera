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
    corrected, _ = apply_depth_correction(predicted, c)
    assert np.allclose(corrected, truth, rtol=1e-5)


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
    corrected, _ = apply_depth_correction(predicted, c)
    assert np.allclose(corrected, truth, rtol=1e-4)


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
    err_wrong = np.abs(apply_depth_correction(predicted, wrong)[0] - truth)[far]
    err_right = np.abs(apply_depth_correction(predicted, right)[0] - truth)[far]
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
    out, _ = apply_depth_correction(d, c)
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
    assert np.allclose(apply_depth_correction(d, c)[0], apply_depth_correction(d, back)[0])


# ------------------------------------------- validity range / extrapolation


def test_a_fit_records_the_range_it_saw():
    truth = _truth(near=1.5, far=30.0)
    c = fit_depth_correction(truth / 2.0, truth, model="scale")
    assert c.predicted_range == pytest.approx((0.75, 15.0), rel=1e-6)
    assert c.measured_range == pytest.approx((1.5, 30.0), rel=1e-6)
    assert c.has_range and c.dynamic_range == pytest.approx(20.0, rel=1e-6)


def test_a_narrow_fit_looks_excellent_and_says_it_is_narrow():
    """The measured failure, as a regression.

    400 samples off one wall at 1.00-1.30 m reported 98.4% improvement and
    0.0074 m residual — then missed by 67% at 50-250 m. The count guard cannot
    see this: 400 clears MIN_SAMPLES easily. Only the RANGE shows it.
    """
    rng = np.random.default_rng(0)
    near = rng.uniform(1.0, 1.3, 400)
    pred = 1.0 / (0.7 * (1.0 / near) + 0.01) * rng.normal(1.0, 0.01, 400)

    c = fit_depth_correction(pred, near, model="affine_disparity")
    assert c.improvement > 0.9, "the fit really does look excellent in-range"
    assert c.dynamic_range < 2.0
    assert "narrow_fit" in c.metadata

    far = np.linspace(50.0, 250.0, 2000)
    far_pred = 1.0 / (0.7 * (1.0 / far) + 0.01)
    out, report = apply_depth_correction(far_pred, c)

    assert report.extrapolated_fraction == pytest.approx(1.0)
    # measured against the PREDICTED span (~1.39-1.85 m), which is what apply()
    # is handed — not the measured span the capture covered
    assert report.extrapolation_ratio > 20
    assert report.warnings, "a 50x extrapolation must not apply quietly"
    assert not report.ok
    # and the error really is as bad as the warning implies
    assert np.nanmedian(np.abs(out - far) / far) > 0.5


def test_in_range_application_is_clean():
    truth = _truth(near=1.5, far=30.0)
    c = fit_depth_correction(truth / 2.0, truth, model="scale")
    _, report = apply_depth_correction(truth / 2.0, c)
    assert report.extrapolated_fraction == 0.0
    assert report.lost_fraction == 0.0
    assert report.ok and not report.warnings


def test_nan_mode_voids_out_of_range_samples_instead_of_guessing():
    truth = _truth(near=2.0, far=10.0)
    c = fit_depth_correction(truth / 2.0, truth, model="scale")
    probe = np.array([[1.0, 3.0, 4.0, 400.0]])       # 1.0 and 400.0 are outside 1.0-5.0
    kept, _ = apply_depth_correction(probe, c, on_extrapolation="report")
    voided, report = apply_depth_correction(probe, c, on_extrapolation="nan")
    assert np.isfinite(kept).all(), "report mode still applies everywhere"
    assert np.isnan(voided[0, 3]), "400 m is far outside the fit and must be voided"
    assert np.isfinite(voided[0, 1]) and np.isfinite(voided[0, 2])
    assert report.lost_fraction > 0


def test_an_unranged_correction_says_it_cannot_be_checked():
    """A hand-built or pre-range coefficient must not read as verified."""
    c = DepthCorrection(model="scale", a=2.0, b=0.0, n_samples=9999)
    assert not c.has_range
    _, report = apply_depth_correction(np.linspace(1, 20, 4096), c)
    assert any("no fitted range" in w for w in report.warnings)


def test_ranges_survive_the_dict_round_trip():
    truth = _truth()
    c = fit_depth_correction(truth / 3.0, truth, model="scale")
    back = DepthCorrection.from_dict(c.to_dict())
    assert back.predicted_range == pytest.approx(c.predicted_range)
    assert back.measured_range == pytest.approx(c.measured_range)
    assert back.has_range


# ------------------------------------------------------- silent degradation


def test_a_correction_that_voids_the_frame_reports_it():
    """The second measured failure, as a regression.

    A plausible-looking correction turned 76% of a fully-valid frame into NaN
    and returned it as a bare array. An all-NaN region is indistinguishable
    downstream from a region the depth model had no opinion about.
    """
    depth = np.linspace(1.0, 80.0, 64 * 64).reshape(64, 64)
    bad = DepthCorrection(model="affine_disparity", a=1.0, b=-0.05,
                          n_samples=9999, predicted_range=(1.0, 80.0))
    out, report = apply_depth_correction(depth, bad)

    assert np.isnan(out).mean() > 0.5
    assert report.lost_fraction > 0.5
    assert report.n_output_valid < report.n_input_valid
    assert any("did not survive" in w for w in report.warnings)
    assert not report.ok


def test_on_extrapolation_rejects_an_unknown_mode():
    c = DepthCorrection(model="scale", a=1.0, b=0.0, n_samples=1,
                        predicted_range=(1.0, 2.0))
    with pytest.raises(ValueError, match="on_extrapolation"):
        apply_depth_correction(np.ones((4, 4)) * 1.5, c, on_extrapolation="clamp")


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
