"""Contracts for camera-space falsification against a real photograph.

The point of this module is that it can FAIL. Every test here is paired with a
way the metric could be silently useless: a sky violation that reads zero, a
containment score that ignores spill, a depth check that is really an absolute
metre comparison in disguise, a report published without the do-nothing
baseline beside it.

Synthetic arrays only: no scene, no GPU, no ComfyUI.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core import plate_falsification as pf
from atlas_camera.core.mask_ops import dilate
from atlas_camera.core.plate_falsification import (
    CHANCE_DEPTH_AGREEMENT,
    FalsificationReport,
    falsification_report,
    rasterize_candidate,
    score_geometry_against_plate,
)

H, W = 64, 96


def _rect(y0, y1, x0, x1, shape=(H, W)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _depth_from(mask, value):
    z = np.full((H, W), np.inf, dtype=np.float64)
    z[mask] = value
    return z


# --------------------------------------------------------------- sky violation


def test_geometry_standing_in_observed_sky_is_falsified():
    """A photograph's sky is hard evidence. Geometry there is wrong, full stop."""

    sky = _rect(0, 20, 0, W)
    alpha = _rect(10, 40, 30, 60)  # top half pokes into the sky

    report = score_geometry_against_plate(alpha=alpha, sky_mask=sky)

    sv = report["sky_violation"]
    assert sv["available"] is True
    expected = float((alpha & sky).sum()) / float(alpha.sum())
    assert sv["value"] == pytest.approx(expected)
    assert sv["value"] > 0.0
    assert sv["pass"] is False


def test_geometry_clear_of_the_sky_passes():
    sky = _rect(0, 20, 0, W)
    alpha = _rect(30, 50, 30, 60)

    sv = score_geometry_against_plate(alpha=alpha, sky_mask=sky)["sky_violation"]
    assert sv["value"] == pytest.approx(0.0)
    assert sv["pass"] is True


def test_a_missing_sky_mask_is_unavailable_not_zero():
    """The trap: an absent mask scoring 0.0 reads as 'no violation' and lets a
    candidate pass a check that was never run."""

    sv = score_geometry_against_plate(alpha=_rect(30, 50, 30, 60))["sky_violation"]
    assert sv["available"] is False
    assert sv["value"] is None
    assert sv["pass"] is None


# ------------------------------------------------------------------ containment


def test_spill_outside_the_authorised_region_is_counted_in_pixels():
    """The hole-splat failure, made a first-class number: 100% of the hole was
    closed while 143k px were painted outside it."""

    authorised = _rect(20, 40, 30, 60)
    alpha = _rect(20, 40, 30, 80)  # twice as wide as it was allowed to be

    c = score_geometry_against_plate(alpha=alpha, authorised_mask=authorised)["containment"]

    spill = int((alpha & ~authorised).sum())
    assert c["spill_px"] == spill
    assert spill > 0
    assert c["value"] == pytest.approx(1.0 - spill / float(alpha.sum()))
    assert c["pass"] is False


def test_geometry_confined_to_its_region_is_fully_contained():
    authorised = _rect(20, 40, 30, 60)
    alpha = _rect(24, 36, 34, 56)

    c = score_geometry_against_plate(alpha=alpha, authorised_mask=authorised)["containment"]
    assert c["spill_px"] == 0
    assert c["value"] == pytest.approx(1.0)
    assert c["pass"] is True


def test_closing_the_region_and_spilling_are_reported_separately():
    """Closure alone is the metric that called a degenerate fill a success."""

    authorised = _rect(20, 40, 30, 60)
    alpha = _rect(20, 40, 30, 90)

    c = score_geometry_against_plate(alpha=alpha, authorised_mask=authorised)["containment"]
    assert c["closure"] == pytest.approx(1.0)   # every authorised pixel covered
    assert c["value"] < 1.0                      # and yet it fails


# ----------------------------------------------------------------- silhouette


def test_silhouette_iou_rewards_agreement_with_the_observed_mask():
    observed = _rect(20, 40, 30, 60)
    exact = score_geometry_against_plate(
        alpha=observed.copy(), observed_mask=observed)["silhouette_iou"]
    assert exact["value"] == pytest.approx(1.0)

    half = _rect(20, 30, 30, 60)
    partial = score_geometry_against_plate(
        alpha=half, observed_mask=observed)["silhouette_iou"]
    assert partial["value"] == pytest.approx(0.5)


def test_silhouette_without_an_observed_mask_is_unavailable():
    s = score_geometry_against_plate(alpha=_rect(20, 40, 30, 60))["silhouette_iou"]
    assert s["available"] is False
    assert s["value"] is None


# ---------------------------------------------------------------- depth order


def _ramp_depth(near=2.0, far=20.0):
    xs = np.arange(W)[None, :].repeat(H, axis=0)
    return near + (far - near) * (xs / (W - 1))


def test_agreeing_depth_orderings_score_near_one():
    render = _ramp_depth()
    reference = _ramp_depth()

    d = score_geometry_against_plate(
        alpha=np.ones((H, W), dtype=bool),
        render_depth=render, reference_depth=reference, seed=3,
    )["depth_order_agreement"]

    assert d["available"] is True
    assert d["value"] > 0.99
    assert d["pass"] is True


def test_an_inverted_depth_ordering_scores_near_zero():
    render = _ramp_depth()
    reference = _ramp_depth()[:, ::-1].copy()  # same values, reversed order

    d = score_geometry_against_plate(
        alpha=np.ones((H, W), dtype=bool),
        render_depth=render, reference_depth=reference, seed=3,
    )["depth_order_agreement"]

    assert d["value"] < 0.05
    assert d["pass"] is False
    assert d["value"] < CHANCE_DEPTH_AGREEMENT


def test_depth_agreement_survives_an_arbitrary_scale_and_shift():
    """The load-bearing test. Monocular depth carries scale+shift error
    (design-rules.md:141), so a metric that compares METRES would call a
    perfectly ordered prediction a failure. This one must not."""

    render = _ramp_depth()
    reference = 7.0 * _ramp_depth() + 130.0

    d = score_geometry_against_plate(
        alpha=np.ones((H, W), dtype=bool),
        render_depth=render, reference_depth=reference, seed=3,
    )["depth_order_agreement"]

    assert d["value"] > 0.99


def test_depth_agreement_is_deterministic_for_a_given_seed():
    render = _ramp_depth()
    rng = np.random.default_rng(11)
    reference = _ramp_depth() + rng.normal(0.0, 3.0, size=(H, W))

    kwargs = dict(alpha=np.ones((H, W), dtype=bool),
                  render_depth=render, reference_depth=reference)
    a = score_geometry_against_plate(seed=5, **kwargs)["depth_order_agreement"]
    b = score_geometry_against_plate(seed=5, **kwargs)["depth_order_agreement"]
    c = score_geometry_against_plate(seed=6, **kwargs)["depth_order_agreement"]

    assert a["value"] == b["value"]
    assert a["n_pairs"] == b["n_pairs"]
    assert c["value"] != a["value"]


def test_depth_agreement_needs_both_buffers():
    d = score_geometry_against_plate(
        alpha=np.ones((H, W), dtype=bool), render_depth=_ramp_depth(),
    )["depth_order_agreement"]
    assert d["available"] is False
    assert d["value"] is None


def test_too_few_comparable_pixels_is_unavailable_not_a_lucky_score():
    """Two covered pixels can agree by chance and read 1.000. Refuse instead."""

    covered = _rect(0, 1, 0, 2)
    render = _depth_from(covered, 5.0)
    render[0, 1] = 6.0
    reference = np.zeros((H, W))
    reference[0, 0], reference[0, 1] = 1.0, 2.0

    d = score_geometry_against_plate(
        alpha=covered, render_depth=render, reference_depth=reference,
    )["depth_order_agreement"]
    assert d["available"] is False


# ---------------------------------------------------------------- seam gradient


def _flat_plate(value=0.4):
    return np.full((H, W, 3), value, dtype=np.float64)


def _textured_plate(seed=7):
    """A plate with a gradient scale to measure a rim against; a flat one has
    none, which is why its ratio is correctly infinite."""
    rng = np.random.default_rng(seed)
    return np.repeat(rng.random((H, W))[..., None], 3, axis=2) * 0.5 + 0.2


def test_a_hard_seam_raises_the_rim_gradient_above_the_plate():
    plate = _flat_plate()
    alpha = _rect(20, 40, 30, 60)
    composite = plate.copy()
    composite[alpha] = 0.95  # a bright patch pasted with a hard edge

    s = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=composite)["seam_gradient_ratio"]

    assert s["available"] is True
    assert s["value"] > 1.0


def test_a_seamless_composite_does_not_raise_the_rim_gradient():
    plate = _flat_plate()
    alpha = _rect(20, 40, 30, 60)
    composite = plate.copy()  # geometry that matches the plate exactly

    s = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=composite)["seam_gradient_ratio"]
    assert s["value"] == pytest.approx(1.0, abs=1e-6) or s["value"] == 0.0


def test_seam_metric_needs_both_plate_and_composite():
    s = score_geometry_against_plate(
        alpha=_rect(20, 40, 30, 60), plate=_flat_plate())["seam_gradient_ratio"]
    assert s["available"] is False


# ------------------------------------------------------------------- the report


def test_a_report_cannot_be_produced_without_a_baseline():
    """Doctrine, enforced structurally: a candidate is only ever reported
    beside the do-nothing render it has to beat."""

    with pytest.raises(TypeError):
        falsification_report(candidate={"alpha": _rect(20, 40, 30, 60)})  # type: ignore[call-arg]


def test_the_report_carries_candidate_baseline_and_the_deltas():
    sky = _rect(0, 20, 0, W)
    authorised = _rect(20, 40, 30, 60)

    report = falsification_report(
        candidate=dict(alpha=_rect(20, 40, 30, 60), sky_mask=sky,
                       authorised_mask=authorised),
        baseline=dict(alpha=_rect(10, 40, 30, 80), sky_mask=sky,
                      authorised_mask=authorised),
    )

    assert isinstance(report, FalsificationReport)
    assert report.candidate["containment"]["value"] == pytest.approx(1.0)
    assert report.baseline["containment"]["value"] < 1.0
    # Higher is better for containment, so beating the baseline is positive.
    assert report.deltas["containment"] > 0.0
    assert report.beats_baseline is True


def test_a_candidate_worse_than_doing_nothing_says_so():
    authorised = _rect(20, 40, 30, 60)
    report = falsification_report(
        candidate=dict(alpha=_rect(20, 40, 30, 90), authorised_mask=authorised),
        baseline=dict(alpha=_rect(22, 38, 32, 58), authorised_mask=authorised),
    )
    assert report.beats_baseline is False


def test_the_report_serializes_to_plain_json_types():
    import json

    report = falsification_report(
        candidate=dict(alpha=_rect(20, 40, 30, 60),
                       authorised_mask=_rect(20, 40, 30, 60)),
        baseline=dict(alpha=_rect(20, 40, 30, 80),
                      authorised_mask=_rect(20, 40, 30, 60)),
    )
    text = json.dumps(report.to_dict())
    assert "containment" in text


def test_an_empirical_gate_abstains_until_it_has_been_measured(monkeypatch):
    """An empirical threshold invented at the keyboard is as unfalsifiable as
    no threshold. With the constant unset the metric still REPORTS, but it
    refuses to render a verdict."""

    monkeypatch.setattr(pf, "MAX_SEAM_GRADIENT_RATIO", None)
    plate = _textured_plate()
    alpha = _rect(20, 40, 30, 60)
    composite = plate.copy()
    composite[alpha] = np.clip(plate[alpha] + 0.45, 0.0, 1.0)

    s = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=composite)["seam_gradient_ratio"]
    assert s["value"] is not None
    assert s["pass"] is None
    assert s["gate"] == "uncalibrated"


def test_the_calibrated_seam_gate_fires_on_a_measured_defect():
    """1.25 was measured to sit below the weakest defect on four real plates
    (flat smear, 1.426) and above a clean join (1.000)."""

    plate = _textured_plate()
    alpha = _rect(20, 40, 30, 60)

    clean = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=plate.copy())["seam_gradient_ratio"]
    assert clean["value"] == pytest.approx(1.0)
    assert clean["pass"] is True
    assert clean["gate"] == "empirical"

    offset = plate.copy()
    offset[alpha] = np.clip(plate[alpha] + 0.45, 0.0, 1.0)
    broken = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=offset)["seam_gradient_ratio"]
    assert broken["value"] > pf.MAX_SEAM_GRADIENT_RATIO
    assert broken["pass"] is False


def test_the_seam_ratio_is_referenced_to_the_rim_not_the_frame_average():
    """Measured blind spot: referenced to the FRAME average, a clean join on
    DSCF3916 scored 1.37 purely because silhouettes sit on busier content than
    the mean — which would fail honest geometry on a detailed photograph."""

    rng = np.random.default_rng(4)
    plate = np.repeat(rng.random((H, W))[..., None], 3, axis=2) * 0.1
    alpha = _rect(20, 40, 30, 60)
    busy = dilate(alpha, 2) & ~alpha
    plate[busy] = rng.random((int(busy.sum()), 3))  # loud content on the rim only

    s = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=plate.copy())["seam_gradient_ratio"]

    assert s["value"] == pytest.approx(1.0)
    assert s["plate_rim_gradient"] > s["plate_gradient"]


# ------------------------------------------------------------------- rasterize


def test_rasterize_candidate_puts_a_quad_where_the_camera_sees_it():
    """The convention check: a mirrored or transposed view matrix still
    produces a plausible-looking mask."""

    view = np.eye(4, dtype=np.float64)
    # A quad 5 m in front, spanning +x and +y, so it must land RIGHT of and
    # ABOVE the principal point (image y grows downward).
    verts = np.array([
        [0.5, 0.5, -5.0], [2.0, 0.5, -5.0], [2.0, 2.0, -5.0], [0.5, 2.0, -5.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    alpha, depth = rasterize_candidate(
        verts, faces, view_matrix=view, fx=80.0, fy=80.0,
        cx=W / 2.0, cy=H / 2.0, width=W, height=H)

    assert alpha.shape == (H, W)
    assert alpha.any()
    rows, cols = np.nonzero(alpha)
    assert cols.min() > W / 2.0
    assert rows.max() < H / 2.0
    assert np.allclose(depth[alpha], 5.0, atol=1e-6)
    assert np.isinf(depth[~alpha]).all()


def test_empty_alpha_is_refused():
    with pytest.raises(ValueError, match="empty"):
        score_geometry_against_plate(alpha=np.zeros((H, W), dtype=bool))


def test_mismatched_masks_are_refused():
    with pytest.raises(ValueError, match="match"):
        score_geometry_against_plate(
            alpha=_rect(20, 40, 30, 60),
            sky_mask=np.zeros((H + 1, W), dtype=bool))


def test_the_seam_metric_reads_the_rim_and_not_the_candidates_own_texture():
    """Mutation-found blind spot: a rim that still contains the candidate's
    interior measures how DETAILED the fill is, not whether it joins. Busy
    geometry with a perfect join would then be flagged as a seam."""

    plate = _flat_plate()
    alpha = _rect(20, 40, 30, 60)
    composite = plate.copy()
    # Loud interior, but the boundary pixels match the plate exactly.
    inner = _rect(23, 37, 33, 57)
    noise = np.indices((H, W)).sum(axis=0) % 2
    composite[inner] = np.where(noise[inner][:, None] > 0, 0.05, 0.95)

    s = score_geometry_against_plate(
        alpha=alpha, plate=plate, composite=composite)["seam_gradient_ratio"]

    assert s["rim_gradient"] == pytest.approx(0.0, abs=1e-9)


def test_tied_depths_carry_no_ordering_and_are_not_counted_as_agreement():
    """Mutation-found blind spot: sign(0) == sign(0) is True, so counting ties
    lets a flat reference buffer score 1.000 agreement with anything."""

    render = _ramp_depth()
    reference = np.full((H, W), 4.0)  # no ordering information at all

    d = score_geometry_against_plate(
        alpha=np.ones((H, W), dtype=bool),
        render_depth=render, reference_depth=reference, seed=3,
    )["depth_order_agreement"]

    assert d["available"] is False
    assert d["value"] is None


def test_no_comparable_metric_is_inconclusive_not_a_loss():
    """A bare False conflates "the candidate lost" with "nothing could judge
    it". Only one of those is a result."""

    report = falsification_report(
        candidate=dict(alpha=_rect(20, 40, 30, 60)),
        baseline=dict(alpha=_rect(20, 40, 30, 80)),
    )
    assert report.n_metrics_compared == 0
    assert report.verdict == "inconclusive"
    assert report.beats_baseline is False


def test_a_measured_loss_says_worse_and_counts_its_evidence():
    authorised = _rect(20, 40, 30, 60)
    report = falsification_report(
        candidate=dict(alpha=_rect(20, 40, 30, 90), authorised_mask=authorised),
        baseline=dict(alpha=_rect(22, 38, 32, 58), authorised_mask=authorised),
    )
    assert report.verdict == "worse"
    assert report.n_metrics_compared >= 1


def test_a_measured_win_says_better():
    authorised = _rect(20, 40, 30, 60)
    report = falsification_report(
        candidate=dict(alpha=_rect(22, 38, 32, 58), authorised_mask=authorised),
        baseline=dict(alpha=_rect(20, 40, 30, 90), authorised_mask=authorised),
    )
    assert report.verdict == "better"
    assert report.beats_baseline is True
