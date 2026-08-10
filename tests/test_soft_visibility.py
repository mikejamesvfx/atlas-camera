"""Contract for soft-layering silhouettes (SLIDE, ICCV 2021).

The staircase this replaces is structural: tearing removes whole grid cells, so
a silhouette can only turn in cell-sized steps and every mitigation afterwards
is tidying a hole. Soft layering removes the hole instead — the mesh stays
CONTINUOUS across a depth cliff and the shader fades those fragments with a
per-pixel visibility

    A = exp(-beta * ||grad(disparity)||^2)

computed at PLATE resolution. No boundary to quantize means no staircase at any
grid, and the lattice survives, so the lattice-based repairs (planar hole patch,
CUDA grid repair) keep working — which `sub_quad_boundary` broke.

The geometric fact that makes it cheap here: the viewport recomputes each
vertex's UV by projecting its world position through the recovered camera
(`atlas_blockout.js` PROJECTION_VERTEX_SHADER -> vImagePx), so on a rubber-band
triangle the sampled UV sweeps between the foreground rim pixel and the
background pixel — which are ADJACENT plate pixels at an occlusion edge. The
whole smear therefore samples the few-texel band where ||grad D|| is largest,
which is exactly where A is smallest.

Fading needs something behind it to reveal (SLIDE composites A*I_fg +
(1-A)*I_bg over an inpainted background); that is the workflow's job, not this
module's. Here we pin the field and the untorn build.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.relief_mesh import SOFT_VISIBILITY_BETA, build_relief_mesh

H = W = 512
FX = FY = 450.0
GRID = 128
SLOPE, INTERCEPT = 0.6, 90.0
NEAR_M, FAR_M = 4.0, 12.0

BUILD = dict(view_matrix=np.eye(4), fx=FX, fy=FY, cx=W / 2.0, cy=H / 2.0,
             grid_long_edge=GRID, depth_edge_rel=0.5, max_edge_factor=12.0,
             floor_clamp=None, apply_sky_heuristic=False, quad_coherence=True)


@pytest.fixture(scope="module")
def cliff():
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.where(ys > (xs * SLOPE + INTERCEPT), NEAR_M, FAR_M).astype(np.float32)


@pytest.fixture(scope="module")
def smooth_ramp():
    """A steeply receding floor: large depth CHANGE, no discontinuity.

    The failure mode worth guarding is fading this — a ground plane running to
    the horizon must stay opaque, or soft layering dissolves every floor.
    """
    return np.linspace(40.0, 6.0, H)[:, None].repeat(W, 1).astype(np.float32)


def _distance_to_cliff():
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.abs(SLOPE * xs - ys + INTERCEPT) / np.hypot(SLOPE, 1.0)


def test_off_by_default_is_byte_identical(cliff):
    plain = build_relief_mesh(cliff, **BUILD)
    again = build_relief_mesh(cliff, soft_visibility=False, **BUILD)
    assert plain.silhouette_alpha is None
    assert np.array_equal(plain.faces, again.faces)
    assert "soft_visibility" not in plain.stats


def test_the_mesh_is_untorn_so_there_is_no_boundary_to_quantize(cliff):
    """No hole means no staircase — at ANY grid resolution.

    This is the whole argument for soft layering over every boundary-tidying
    pass: those make the hole neater, this removes it.
    """
    torn = build_relief_mesh(cliff, **BUILD)
    soft = build_relief_mesh(cliff, soft_visibility=True, **BUILD)
    assert torn.stats["torn_fraction"] > 0.0, "fixture must actually tear"
    assert soft.stats["torn_fraction"] == 0.0
    assert len(soft.faces) > len(torn.faces)


def test_the_honest_tear_number_survives(cliff):
    """torn_fraction is ~0 by construction now, so it stops being a quality
    signal. The tear tests still RAN with the caller's thresholds and their
    verdict has to travel, or `torn_excessive` reads a soft layer as flawless
    and the inpaint router loses the number telling it how much to fill."""
    torn = build_relief_mesh(cliff, **BUILD)
    soft = build_relief_mesh(cliff, soft_visibility=True, **BUILD)
    assert soft.stats["soft_visibility"]["torn_fraction_if_torn"] == pytest.approx(
        torn.stats["torn_fraction"], abs=1e-9)
    # The QA gate reads torn_fraction_whole_quad when present (see nodes_qa).
    assert soft.stats["torn_fraction_whole_quad"] == pytest.approx(
        torn.stats["torn_fraction"], abs=1e-9)


def test_visibility_collapses_at_the_cliff_and_holds_on_smooth_ground(cliff, smooth_ramp):
    """The two ends that matter, and the calibration of beta between them."""
    alpha = build_relief_mesh(cliff, soft_visibility=True, **BUILD).silhouette_alpha
    d = _distance_to_cliff()

    assert float(alpha[d < 1.5].min()) < 0.10, "the cliff must actually fade"
    assert float(alpha[d > 12.0].mean()) > 0.99, "surface away from the cliff must be opaque"

    # A 40 m -> 6 m sweep across 512 px is a deliberately extreme floor: 6.7x
    # disparity change over the frame, twice the per-pixel gradient of the same
    # ramp at 1024 px (which measures 0.987). It dims ~5% and no more — the
    # bound is the measurement, not a round number, because tightening beta
    # enough to make this exactly 1.0 stops the cliff reaching 0.1.
    ramp_alpha = build_relief_mesh(
        smooth_ramp, soft_visibility=True, **BUILD).silhouette_alpha
    assert float(ramp_alpha.min()) > 0.90, (
        "a steeply receding floor is not an occlusion — fading it would dissolve "
        "every ground plane")
    assert float(ramp_alpha.mean()) > 0.99, "the bulk of a floor must be untouched"


def test_the_fade_is_a_feather_not_a_line(cliff):
    """A 1-2px collapse would be a hard edge wearing a gradient's clothes.

    The gradient is widened before the exponential precisely so the smear —
    whose swept UV lands in a few-texel band at the cliff — is covered.
    """
    alpha = build_relief_mesh(cliff, soft_visibility=True, **BUILD).silhouette_alpha
    column = alpha[:, W // 2]
    faded = np.flatnonzero(column < 0.5)
    assert faded.size >= 3, "fade band too narrow to cover a smear"
    assert faded.size <= 24, "fade band so wide it is eating real surface"
    # Contiguous, and monotonic into the trough — no comb.
    assert faded.max() - faded.min() == faded.size - 1
    trough = int(column.argmin())
    assert np.all(np.diff(column[faded.min():trough + 1]) <= 1e-6)


def test_beta_is_tunable_and_monotonic(cliff):
    d = _distance_to_cliff()
    soft = [float(build_relief_mesh(cliff, soft_visibility=True,
                                    soft_visibility_beta=b, **BUILD)
                  .silhouette_alpha[d < 1.5].min())
            for b in (5.0, SOFT_VISIBILITY_BETA, 120.0)]
    assert soft[0] > soft[1] > soft[2], "larger beta must fade harder"


def test_excluded_pixels_still_cut_rather_than_fade(cliff):
    """A masked region has no pixels to show, so it is a CUT, not a fade —
    fading it would reveal whatever the smear happened to sample."""
    exclude = np.zeros((H, W), dtype=bool)
    exclude[:60, :] = True
    alpha = build_relief_mesh(cliff, soft_visibility=True, exclude_mask=exclude,
                              **BUILD).silhouette_alpha
    assert float(alpha[:60, :].max()) == 0.0


def test_soft_visibility_supersedes_the_sub_quad_cut(cliff):
    """Both aim at the same artifact and the cut is the one that breaks the
    lattice (off-lattice vertices disable planar hole patch). Asking for both
    must yield the soft build, not a hybrid."""
    both = build_relief_mesh(cliff, soft_visibility=True,
                             sub_quad_boundary=True, **BUILD)
    assert "sub_quad_cut" not in both.stats
    assert both.stats["torn_fraction"] == 0.0


def test_the_lattice_survives_so_lattice_repairs_still_work(cliff):
    """The regression sub_quad_boundary introduced, and the reason soft layering
    is the better answer: recover_lattice must still succeed."""
    from atlas_camera.core.hole_field import recover_lattice

    soft = build_relief_mesh(cliff, soft_visibility=True, **BUILD)
    lattice = recover_lattice(soft, W, H)
    assert set(lattice) >= {"vertices", "faces", "uvs", "rows", "cols"}


class TestScalesToLargePlates:
    """8K+ plates are routine here, so the field must be bounded AND its
    calibration must not drift with plate size."""

    @staticmethod
    def _scene(h, w):
        """The SAME scene at a different sampling density: one diagonal cliff
        at the same fractional position, same metric depths."""
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        return np.where(ys / h > (xs / w) * 0.4 + 0.2,
                        NEAR_M, FAR_M).astype(np.float32)

    def _alpha(self, h, w):
        return build_relief_mesh(
            self._scene(h, w), view_matrix=np.eye(4), fx=900.0, fy=900.0,
            cx=w / 2.0, cy=h / 2.0, grid_long_edge=128, floor_clamp=None,
            apply_sky_heuristic=False, soft_visibility=True).silhouette_alpha

    def test_the_matte_is_bounded_regardless_of_plate_size(self):
        from atlas_camera.core.relief_mesh import SOFT_VISIBILITY_MAX_EDGE

        for h, w in ((1024, 1024), (2160, 3840), (5464, 8192)):
            alpha = self._alpha(h, w)
            assert max(alpha.shape) <= SOFT_VISIBILITY_MAX_EDGE, (
                f"{w}x{h} produced a {alpha.shape} matte — an 8K float field is "
                f"~180MB in RAM and a quarter-GB on the GPU")

    def test_beta_means_the_same_thing_at_every_plate_size(self):
        """The gradient is divided by the decimation stride so it stays PER
        PLATE PIXEL. Without that a coarser stride reports a proportionally
        steeper cliff, and one beta would fade 4K correctly while dissolving 8K.
        """
        stats = []
        for h, w in ((1024, 1024), (2160, 3840), (5464, 8192)):
            alpha = self._alpha(h, w)
            stats.append((float(alpha.min()), float(np.median(alpha))))

        mins = [s[0] for s in stats]
        medians = [s[1] for s in stats]
        assert all(m < 0.15 for m in mins), f"cliff failed to fade somewhere: {mins}"
        assert all(m > 0.99 for m in medians), (
            f"open surface dimmed at some plate size: {medians}")

    def test_a_large_plate_stays_within_a_sane_time_budget(self):
        import time

        start = time.perf_counter()
        self._alpha(5464, 8192)
        elapsed = time.perf_counter() - start
        assert elapsed < 20.0, f"8K soft-visibility build took {elapsed:.1f}s"


def test_band_clipping_still_holes_and_dominates_the_tear_number(cliff):
    """A band layer owns a depth SLICE; outside it there is no data to fade.

    Measured on the live sh004 bands: torn_fraction stayed ~0.35 with soft
    visibility on, which reads as "it did nothing" until you separate the two
    causes. Cliff tears go to zero; band-clip holes are untouched and dominate
    the number. Anyone reading torn_fraction on a banded layer to judge soft
    visibility will conclude it is broken.
    """
    kw = dict(BUILD)
    kw["max_edge_factor"] = 12.0
    deep = np.where(cliff < 6.0, NEAR_M, 30.0).astype(np.float32)

    plain = build_relief_mesh(deep, **kw)
    soft = build_relief_mesh(deep, soft_visibility=True, **kw)
    assert plain.stats["torn_fraction"] > 0.0
    assert soft.stats["torn_fraction"] == 0.0, "cliff tears must go"

    clipped_plain = build_relief_mesh(deep, band_max_m=19.0, **kw)
    clipped_soft = build_relief_mesh(deep, band_max_m=19.0, soft_visibility=True, **kw)
    assert clipped_plain.stats["torn_fraction"] > 0.3, "fixture must actually clip"
    assert clipped_soft.stats["torn_fraction"] == pytest.approx(
        clipped_plain.stats["torn_fraction"], abs=1e-9), (
        "band-clip holes are missing DATA, not a silhouette — fading them would "
        "reveal whatever the smear happened to sample")
    assert clipped_soft.stats["soft_visibility"]["beta"] > 0.0
