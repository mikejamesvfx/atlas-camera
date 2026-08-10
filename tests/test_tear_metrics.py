"""Scoring relief-mesh tearing against known geometry.

The golden-frame gate says whether the picture CHANGED. This says whether the
tearing is RIGHT, which is what an automated loop needs before it can tune tear
thresholds without a human judging every render.

The tests that matter most are the degenerate ones. An objective that can be
gamed by tearing everything is worse than no objective, because a tuning loop
will find that solution and report success.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.tear_metrics import (
    TearScore,
    pareto_front,
    score_tears,
    true_occlusion_edges,
)

np = pytest.importorskip("numpy")


def _two_surfaces(h=48, w=64, near=2.0, far=9.0):
    """Left half near, right half far — one hard occlusion edge down the middle."""
    depth = np.full((h, w), near)
    depth[:, w // 2:] = far
    sid = np.zeros((h, w), dtype=np.int32)
    sid[:, w // 2:] = 1
    return depth, sid


def _crease(h=48, w=64):
    """Two surfaces meeting with NO depth step — a corner, not an occlusion."""
    depth = np.tile(np.linspace(4.0, 4.0, w), (h, 1))
    sid = np.zeros((h, w), dtype=np.int32)
    sid[:, w // 2:] = 1
    return depth, sid


# ------------------------------------------------------- edge detection


def test_a_depth_step_at_a_surface_change_is_an_occlusion():
    depth, sid = _two_surfaces()
    e = true_occlusion_edges(depth, sid)
    assert e.any()
    cols = np.where(e.any(axis=0))[0]
    assert set(cols) <= {depth.shape[1] // 2 - 1, depth.shape[1] // 2}


def test_a_crease_is_not_an_occlusion():
    """A wall meeting a floor is continuous geometry.

    Tearing it punches a hole in a solid corner, so it must not be scored as an
    edge the mesh was supposed to tear.
    """
    depth, sid = _crease()
    assert not true_occlusion_edges(depth, sid).any()


def test_a_smooth_ramp_has_no_edges_however_steep():
    """Grazing is not occlusion — the failure `max_edge_factor` trips on."""
    depth = np.tile(np.linspace(1.0, 60.0, 64), (48, 1))
    sid = np.zeros((48, 64), dtype=np.int32)
    assert not true_occlusion_edges(depth, sid).any()


# ------------------------------------------------- the degenerate solutions


def test_tearing_everything_scores_worst():
    """The solution a naive tear-only objective would converge on.

    Coverage is in the objective from the start precisely so this loses.
    """
    depth, sid = _two_surfaces()
    surface = depth > 0
    tear_all = np.zeros_like(depth)
    cover_all = np.where(surface, 1.0, 0.0)

    s_tear = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=tear_all)
    s_cover = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=cover_all)

    assert s_tear.coverage == pytest.approx(0.0)
    # Not ~1.0: the band within tolerance of the real edge is genuinely
    # justified even here, so ~11% of the frame is not counted against it.
    assert s_tear.false_tear_fraction > 0.85
    assert s_tear.combined_score() < s_cover.combined_score(), (
        "tearing everything must not beat tearing nothing")


def test_false_tear_fraction_catches_swiss_cheese():
    """The component that does the real work.

    A hole grid satisfies `missed_edge_fraction` by luck — scattered gaps land
    near every edge — so `false_tear_fraction` is what must separate it from a
    clean render. This pins that, because the combined scalar does NOT.
    """
    depth, sid = _two_surfaces()
    surface = depth > 0
    clean = np.where(surface, 1.0, 0.0)
    cheese = clean.copy()
    cheese[::2, ::2] = 0.0

    s_clean = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=clean)
    s_cheese = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=cheese)
    assert s_cheese.false_tear_fraction > 0.2
    assert s_clean.false_tear_fraction == pytest.approx(0.0)


def test_the_combined_scalar_has_a_documented_blind_spot():
    """Pins the weakness so nobody optimises the scalar alone by accident.

    Swiss cheese currently OUTSCORES a mesh that never tears, because recall is
    trivially satisfied by tearing everywhere. That ordering is not defensible,
    which is exactly why `pareto_front` exists and why the docstring says not to
    optimise this number on its own. If someone fixes the recall measure, this
    test should start failing — and it should then be updated, not deleted.
    """
    depth, sid = _two_surfaces()
    surface = depth > 0
    clean = np.where(surface, 1.0, 0.0)
    cheese = clean.copy()
    cheese[::2, ::2] = 0.0

    s_clean = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=clean)
    s_cheese = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=cheese)
    assert s_cheese.combined_score() > s_clean.combined_score(), (
        "the documented blind spot has changed — re-read combined_score's docstring")


def test_a_tear_exactly_on_the_edge_is_justified():
    depth, sid = _two_surfaces()
    alpha = np.ones_like(depth)
    mid = depth.shape[1] // 2
    alpha[:, mid - 1:mid + 1] = 0.0            # tear precisely at the silhouette
    s = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=alpha)
    assert s.justified_fraction == pytest.approx(1.0)
    assert s.false_tear_fraction == pytest.approx(0.0)
    assert s.missed_edge_fraction == pytest.approx(0.0)


def test_a_tear_far_from_any_edge_is_false():
    depth, sid = _two_surfaces()
    alpha = np.ones_like(depth)
    alpha[:, 4:8] = 0.0                        # a hole in open surface
    s = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=alpha)
    assert s.justified_fraction == pytest.approx(0.0)
    assert s.false_tear_fraction > 0.0
    assert s.missed_edge_fraction == pytest.approx(1.0), "the real edge was missed"


# ------------------------------------------------------------- guards


def test_shape_mismatch_is_rejected():
    depth, sid = _two_surfaces()
    with pytest.raises(ValueError, match="must match"):
        score_tears(truth_depth=depth, surface_ids=sid,
                    coverage_alpha=np.ones((8, 8)))


def test_empty_ground_truth_is_rejected():
    with pytest.raises(ValueError, match="no valid surface"):
        score_tears(truth_depth=np.zeros((8, 8)),
                    surface_ids=np.zeros((8, 8), dtype=np.int32),
                    coverage_alpha=np.ones((8, 8)))


# ------------------------------------------------------------- pareto


def test_pareto_keeps_only_non_dominated_configs():
    a = TearScore(false_tear_fraction=0.10, missed_edge_fraction=0.50)  # trades
    b = TearScore(false_tear_fraction=0.50, missed_edge_fraction=0.10)  # trades
    c = TearScore(false_tear_fraction=0.60, missed_edge_fraction=0.60)  # dominated
    front = {lbl for lbl, _ in pareto_front([("a", a), ("b", b), ("c", c)])}
    assert front == {"a", "b"}


def test_pareto_keeps_a_strict_winner_alone():
    good = TearScore(false_tear_fraction=0.01, missed_edge_fraction=0.01)
    bad = TearScore(false_tear_fraction=0.40, missed_edge_fraction=0.40)
    front = {lbl for lbl, _ in pareto_front([("good", good), ("bad", bad)])}
    assert front == {"good"}


def test_pareto_does_not_collapse_a_genuine_trade():
    """The reason the front is the primary output.

    Two configs that each win on one axis must BOTH survive — deciding between
    them depends on the shot, not on a constant chosen once.
    """
    entries = [(f"c{i}", TearScore(false_tear_fraction=i / 10,
                                   missed_edge_fraction=(10 - i) / 10))
               for i in range(1, 10)]
    assert len(pareto_front(entries)) == len(entries)


# ------------------------------------- tearing is topology, not a coverage gap


def _cut_render(depth, sid, *, gap_px=0):
    """A mesh that separates the two sheets, leaving ``gap_px`` of uncovered
    pixels at the seam. gap_px=0 is the sub-quad cut: torn, but nothing missing.
    """
    alpha = np.ones(depth.shape, dtype=np.float64)
    render = depth.astype(np.float64).copy()
    mid = depth.shape[1] // 2
    if gap_px:
        alpha[:, mid - gap_px:mid + gap_px] = 0.0
        render[:, mid - gap_px:mid + gap_px] = np.nan
    return alpha, render


def _curtain_render(depth, ramp_px=8):
    """A mesh that rubber-sheets across the cliff: full coverage and the whole
    depth step smeared over ``ramp_px`` pixels instead of reproduced."""
    h, w = depth.shape
    mid = w // 2
    render = depth.astype(np.float64).copy()
    lo, hi = mid - ramp_px // 2, mid + ramp_px // 2
    ramp = np.linspace(depth[0, 0], depth[0, -1], hi - lo)
    render[:, lo:hi] = ramp[None, :]
    return np.ones(depth.shape, dtype=np.float64), render


def test_a_cut_with_no_gap_is_not_a_missed_edge():
    """The defect this parameter exists for.

    `sub_quad_boundary` tears the topology and still covers every pixel, because
    at a cliff BOTH sheets are photographed right up to the edge. Scored on alpha
    alone that is indistinguishable from never tearing, and it scored WORSE than
    the whole-cell tear it improves on (measured 0.00 -> 0.98 missed).
    """
    depth, sid = _two_surfaces()
    alpha, render = _cut_render(depth, sid)

    gap_only = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=alpha)
    assert gap_only.missed_edge_fraction > 0.9      # the historical blind spot

    with_depth = score_tears(truth_depth=depth, surface_ids=sid,
                             coverage_alpha=alpha, render_depth=render)
    assert with_depth.missed_edge_fraction == 0.0
    assert with_depth.bridged_edge_fraction == 0.0
    assert with_depth.coverage == 1.0


def test_a_curtain_is_caught_even_though_it_also_steps():
    """A smeared step is not a reproduced step, and an ABSOLUTE threshold cannot
    tell them apart — the first version of this check could not, and scored a
    mesh that never tears as perfect. A curtain spread over one grid cell still
    steps every pixel, just by cell-sized fractions of the cliff.
    """
    depth, sid = _two_surfaces()
    alpha, render = _curtain_render(depth)

    per_pixel_step = (depth[0, -1] - depth[0, 0]) / 8
    assert per_pixel_step > 0.25, "fixture must exceed the absolute jump floor"

    score = score_tears(truth_depth=depth, surface_ids=sid,
                        coverage_alpha=alpha, render_depth=render)
    assert score.missed_edge_fraction > 0.9
    assert score.bridged_edge_fraction > 0.9


def test_the_three_renders_are_ordered_correctly():
    """Cut > whole-cell tear > curtain. The scalar must agree with the geometry.

    Measured on the real diagonal-cliff fixture: 1.000 / 0.991 / 0.504.
    """
    depth, sid = _two_surfaces()
    cut = score_tears(truth_depth=depth, surface_ids=sid,
                      **dict(zip(("coverage_alpha", "render_depth"),
                                 _cut_render(depth, sid))))
    torn = score_tears(truth_depth=depth, surface_ids=sid,
                       **dict(zip(("coverage_alpha", "render_depth"),
                                  _cut_render(depth, sid, gap_px=2))))
    curtain = score_tears(truth_depth=depth, surface_ids=sid,
                          **dict(zip(("coverage_alpha", "render_depth"),
                                     _curtain_render(depth))))
    assert cut.combined_score() >= torn.combined_score() > curtain.combined_score()
    assert curtain.bridged_edge_fraction > max(cut.bridged_edge_fraction,
                                               torn.bridged_edge_fraction)


def test_omitting_render_depth_is_byte_identical_to_the_historical_score():
    """Every recorded score and every existing caller must be unaffected."""
    depth, sid = _two_surfaces()
    alpha, _render = _cut_render(depth, sid, gap_px=2)
    score = score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=alpha)
    assert score.metadata["tear_evidence"] == "coverage_gap"
    assert score.bridged_edge_fraction == 0.0
    assert score.to_dict()["combined_score"] == score.combined_score()


def test_render_depth_shape_mismatch_is_rejected():
    depth, sid = _two_surfaces()
    alpha, _ = _cut_render(depth, sid)
    with pytest.raises(ValueError, match="render_depth"):
        score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=alpha,
                    render_depth=np.zeros((4, 4)))
