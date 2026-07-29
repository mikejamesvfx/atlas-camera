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
