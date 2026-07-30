"""Ground truth for the prior-guided depth-completion tier (METHOD_GUIDED).

Fills a hole by integrating a GENERATED relative-depth prior's gradients from
the measured rim. Content comes from the prior (a far stronger structural guess
than smoothness); placement comes entirely from measured pixels.

Unlike the tear scorer, this has a REAL objective rather than a proxy: an
analytic surface has exact gradients, so a correct implementation must recover
the hidden values to float precision. Every claim below is checked against that
rather than against "looks plausible".

The prior is deliberately given a wrong scale AND a wrong offset in most tests,
because that is the actual situation — a hallucinated depth map is defined only
up to an affine transform, and the whole design rests on differencing killing
the shift and one ring-band fit killing the scale.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.depth_completion import (  # noqa: E402
    METHOD_DIFFUSION,
    METHOD_GUIDED,
    METHOD_NAMES,
    METHOD_TANGENT,
    _METHOD_TRUST,
    complete_depth,
    integrate_prior_gradients,
    prior_gradient_scale,
)

H = W = 48
VIEW = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
CAM = dict(view_matrix=VIEW, fx=60.0, fy=60.0, cx=W / 2.0, cy=H / 2.0)


def _surface(kind="tilt"):
    """An analytic depth surface with exact, known gradients."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    if kind == "tilt":                       # planar ramp
        return 8.0 + 0.05 * x + 0.02 * y
    if kind == "bump":                       # smooth non-planar
        return 8.0 + 1.5 * np.exp(-(((x - W / 2) ** 2 + (y - H / 2) ** 2) / 90.0))
    if kind == "step":                       # structure a plane fit cannot express
        return 8.0 + 0.04 * x + np.where(y > H / 2, 1.2, 0.0)
    raise ValueError(kind)


def _hole(pad=14):
    m = np.zeros((H, W), dtype=bool)
    m[pad:H - pad, pad:W - pad] = True
    return m


def _affine(truth, scale=3.7, shift=-12.4):
    """A prior in the shape truth has, but arbitrary units — the real case."""
    return truth * scale + shift


class TestScaleRecovery:
    def test_the_projection_recovers_the_exact_scale(self):
        """s = sum(gm.gq)/sum(gq.gq) must invert the affine exactly."""
        truth = _surface("bump")
        prior = _affine(truth, scale=3.7, shift=-12.4)
        band = np.ones((H, W), dtype=bool)
        s, n, resid = prior_gradient_scale(np, prior, truth, band)
        assert n > 0
        assert s == pytest.approx(1.0 / 3.7, rel=1e-9)
        assert resid == pytest.approx(0.0, abs=1e-9)

    def test_shift_is_irrelevant_by_construction(self):
        truth = _surface("bump")
        band = np.ones((H, W), dtype=bool)
        a = prior_gradient_scale(np, _affine(truth, 2.0, 0.0), truth, band)[0]
        b = prior_gradient_scale(np, _affine(truth, 2.0, 999.0), truth, band)[0]
        assert a == pytest.approx(b, rel=1e-12), (
            "differencing must remove the shift entirely — if this drifts, the "
            "prior is being read as absolute depth somewhere")

    def test_too_few_samples_refuses_rather_than_fitting_noise(self):
        truth = _surface("tilt")
        band = np.zeros((H, W), dtype=bool)
        band[0, :3] = True                       # 3 samples
        s, n, resid = prior_gradient_scale(np, _affine(truth), truth, band)
        assert n < 24 and s == 0.0 and not np.isfinite(resid)

    def test_residual_exposes_a_prior_describing_different_structure(self):
        """The signal that a hallucination invented the wrong thing."""
        truth = _surface("bump")
        band = np.ones((H, W), dtype=bool)
        rng = np.random.default_rng(0)
        wrong = rng.normal(size=(H, W)) * 2.0        # unrelated structure
        _s, _n, resid_bad = prior_gradient_scale(np, wrong, truth, band)
        _s, _n, resid_ok = prior_gradient_scale(np, _affine(truth), truth, band)
        assert resid_ok < 1e-9 < resid_bad


class TestExactRecovery:
    def test_a_linear_surface_is_recovered_EXACTLY(self):
        """The hard objective, and it really is exact.

        Central differences are exact for a linear function and trapezoid
        integration of them is exact too, so a correct implementation has no
        error beyond float rounding. Measured 5.3e-15. This is the test that
        catches an integration bug; the curved cases below cannot, because
        genuine discretisation error would hide it.
        """
        truth = _surface("tilt")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        out = complete_depth(depth, holes=hole, prior=_affine(truth),
                             use_diffusion=False, **CAM)
        assert (out.method_map[hole] == METHOD_GUIDED).all(), \
            "every hole pixel must be filled by the guided tier"
        assert np.abs(out.depth[hole] - truth[hole]).max() < 1e-12

    @pytest.mark.parametrize("kind,tol", [("bump", 2e-2), ("step", 3e-1)])
    def test_curved_and_discontinuous_surfaces_carry_only_discretisation_error(
            self, kind, tol):
        """O(h^2) per step, accumulated along the path — physics, not a defect.

        The step case is the worst: np.gradient central-differences ACROSS the
        discontinuity, smearing it over two pixels, and integrating that smear
        misplaces the edge. Bounded, and still far better than smoothing it away
        entirely (see the comparison below).
        """
        truth = _surface(kind)
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        out = complete_depth(depth, holes=hole, prior=_affine(truth),
                             use_diffusion=False, **CAM)
        assert (out.method_map[hole] == METHOD_GUIDED).all()
        err = np.abs(out.depth[hole] - truth[hole]).max()
        assert err < tol, f"max error {err:.2e} on '{kind}'"

    def test_it_reconstructs_structure_a_plane_fit_cannot(self):
        """The point of the tier: a step is exactly what smoothness destroys."""
        truth = _surface("step")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan

        guided = complete_depth(depth, holes=hole, prior=_affine(truth),
                               use_diffusion=False, **CAM)
        smooth = complete_depth(depth, holes=hole, use_diffusion=True,
                                diffusion_iterations=256, **CAM)
        g_err = np.abs(guided.depth[hole] - truth[hole]).mean()
        s_err = np.abs(smooth.depth[hole] - truth[hole]).mean()
        # Measured 0.029 vs 0.079 — 2.7x better, not the 10x I first asserted
        # from nothing. The step is the tier's WORST case precisely because
        # np.gradient smears a discontinuity across two pixels, so a modest win
        # here is the honest result; the margin is far larger on smooth
        # structure. Pinned at 2x so a regression shows without the number being
        # invented.
        assert g_err < s_err * 0.5, (
            f"guided {g_err:.4f} vs diffusion {s_err:.4f} — the prior must beat "
            "smoothness on structure smoothness cannot represent")

    def test_measured_pixels_are_never_overwritten(self):
        truth = _surface("bump")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        out = complete_depth(depth, holes=hole, prior=_affine(truth),
                             use_diffusion=False, **CAM)
        keep = ~hole
        assert np.allclose(out.depth[keep], truth[keep])


class TestDegradation:
    def test_a_flat_prior_collapses_into_diffusion_not_nonsense(self):
        """With zero gradients every ray returns its nearest rim value, so the
        result is a distance-weighted average of the rim — harmonic-ish
        inpainting. The tier must degrade INTO the tier below it."""
        truth = _surface("tilt")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        flat = np.full((H, W), 5.0)

        known = ~hole
        filled, wsum, spread = integrate_prior_gradients(
            np, flat, np.where(known, truth, 0.0), known, hole, scale=1.0)
        inner = hole & (wsum > 0)
        assert inner.any()
        lo, hi = truth[known].min(), truth[known].max()
        assert (filled[inner] >= lo - 1e-9).all()
        assert (filled[inner] <= hi + 1e-9).all(), (
            "a zero-gradient integration must stay inside the rim's range")

    def test_spread_is_zero_for_a_perfectly_integrable_field(self):
        """A linear field is exactly curl-free, so all eight paths agree."""
        truth = _surface("tilt")
        hole = _hole()
        known = ~hole
        _f, _w, spread = integrate_prior_gradients(
            np, truth, np.where(known, truth, 0.0), known, hole, scale=1.0)
        assert spread[hole].max() < 1e-12

    def test_spread_TRACKS_error_which_is_the_whole_confidence_claim(self):
        """The load-bearing property, measured across three decades.

        `spread` is only useful if it rises when the fill is actually wrong.
        Measured on three surfaces of increasing difficulty:

            tilt  err 5.3e-15   spread 5.7e-15
            bump  err 1.5e-02   spread 3.4e-03
            step  err 2.8e-01   spread 2.3e-01

        Monotonic in both, which is what licenses reading spread as confidence
        WITHOUT ground truth — the situation every real plate is in.
        """
        hole = _hole()
        known = ~hole
        errs, spreads = [], []
        for kind in ("tilt", "bump", "step"):
            truth = _surface(kind)
            filled, w, spread = integrate_prior_gradients(
                np, truth, np.where(known, truth, 0.0), known, hole, scale=1.0)
            inner = hole & (w > 0)
            errs.append(float(np.abs(filled[inner] - truth[inner]).max()))
            spreads.append(float(spread[inner].max()))
        assert errs == sorted(errs), f"fixture difficulty not monotonic: {errs}"
        assert spreads == sorted(spreads), (
            f"spread failed to track error: errors {errs}, spreads {spreads} — "
            "if these decouple, spread cannot be used as confidence")

    def test_spread_rises_for_a_non_integrable_field(self):
        """THE confidence signal. A neural depth field is not curl-free; the
        disagreement between paths measures how much it is confabulating.
        Injecting curl deliberately must move the number."""
        truth = _surface("bump")
        hole = _hole()
        known = ~hole
        rng = np.random.default_rng(1)
        noisy = truth + rng.normal(scale=0.35, size=(H, W))
        _f, _w, spread = integrate_prior_gradients(
            np, noisy, np.where(known, truth, 0.0), known, hole, scale=1.0)
        assert spread[hole].max() > 1e-3

    def test_the_spread_map_reaches_the_caller(self):
        truth = _surface("bump")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        out = complete_depth(depth, holes=hole, prior=_affine(truth),
                             use_diffusion=False, **CAM)
        assert out.guided_spread is not None
        assert out.guided_spread.shape == (H, W)
        assert (out.guided_spread[~hole] == 0.0).all(), \
            "spread must be zero where nothing was guided"


class TestRefusal:
    def test_a_prior_with_unrelated_structure_is_declined(self):
        """Better to fall through to diffusion than to bulge the fill."""
        truth = _surface("bump")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        rng = np.random.default_rng(2)
        wrong = rng.normal(size=(H, W)) * 5.0

        out = complete_depth(depth, holes=hole, prior=wrong,
                             use_diffusion=True, **CAM)
        assert (out.method_map[hole] != METHOD_GUIDED).all()
        assert any("declined" in n for n in out.notes)
        assert any("DIFFERENT structure" in n for n in out.notes)

    def test_no_prior_leaves_the_existing_tiers_untouched(self):
        """Regression guard: the new tier must be inert when unused."""
        truth = _surface("tilt")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        out = complete_depth(depth, holes=hole, use_diffusion=True, **CAM)
        assert (out.method_map[hole] == METHOD_DIFFUSION).all()
        assert out.stats["n_guided"] == 0
        assert out.guided_spread is None


class TestProvenanceContract:
    def test_the_method_code_is_appended_not_inserted(self):
        """method_map is serialized, so codes are append-only even though this
        tier's TRUST sits between tangent and diffusion."""
        assert METHOD_GUIDED == 5
        assert METHOD_DIFFUSION == 4
        assert METHOD_NAMES[METHOD_GUIDED] == "guided"

    def test_trust_ranks_between_tangent_and_diffusion(self):
        assert _METHOD_TRUST[METHOD_DIFFUSION] < _METHOD_TRUST[METHOD_GUIDED] \
            < _METHOD_TRUST[METHOD_TANGENT]

    def test_confidence_reflects_the_new_tier(self):
        truth = _surface("bump")
        hole = _hole()
        depth = truth.copy()
        depth[hole] = np.nan
        guided = complete_depth(depth, holes=hole, prior=_affine(truth),
                                use_diffusion=False, **CAM)
        diffused = complete_depth(depth, holes=hole, use_diffusion=True, **CAM)
        assert guided.confidence() > diffused.confidence()
        assert "guided" in guided.method_histogram()
