"""Contracts for the solve degradation ladder in the hole-splat driver.

WHY THIS EXISTS. Measured 2026-08-20 on the five atlas_raws/MultiShots bursts,
the strict profile solved NONE of them once the held-out frame entered the rig
(sh002 32 mutual matches, sh003 45 against balanced's floor of 48; sh001 and
sh004 ambiguous_motion_model). A tool that refuses on every capture handed to it
has not been strict, it has been useless — but a tool that quietly relaxes until
something comes back is worse. The ladder is the middle: relax one sanctioned
knob at a time, record every rung that refused and why, and carry the trust the
relaxation cost.

No RAW, no GPU, no solve: ``solve_multiview`` is stubbed so the ORDER of the
ladder and the bookkeeping are what is under test.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tools.hole_splat_experiment import Experiment, Stage


def _frame(label):
    return SimpleNamespace(label=label, image=None, raw_meta=None, metric_depth=None)


def _solve(n_sources=1):
    return SimpleNamespace(
        projection_sources=[SimpleNamespace(name=f"src{i}") for i in range(n_sources)],
        confidence=0.5, source_method="stub")


def _refusal(code, summary="refused"):
    return SimpleNamespace(
        solve=None,
        diagnostics=SimpleNamespace(outcome_code=code, summary=summary, warnings=[]))


def _success():
    return SimpleNamespace(
        solve=_solve(), diagnostics=SimpleNamespace(outcome_code="ok", warnings=[]))


def _args(**over):
    base = dict(holdout=2, match_quality="balanced", degrade=True, pose_holdout=True,
                camera_height=1.6, baseline_m=0.0, up_hint=False, depth_scale=False,
                depth_model="x", depth_max_side=512)
    base.update(over)
    return argparse.Namespace(**base)


def _experiment(args, labels=("A", "B", "C")):
    exp = Experiment(args)
    exp.state["frames"] = [_frame(x) for x in labels]
    return exp


def _install(monkeypatch, decide):
    """Stub the solver; `decide(frames, settings)` returns an outcome."""
    import atlas_camera.core.multiview_solver as solver

    calls = []

    def fake(frames, settings):
        calls.append({"labels": [f.label for f in frames],
                      "quality": settings.match_quality,
                      "mode": settings.capture_mode})
        return decide(frames, settings)

    monkeypatch.setattr(solver, "solve_multiview", fake)
    return calls


# ------------------------------------------------------------------ the ladder


def test_the_strictest_rung_wins_when_it_solves(monkeypatch):
    calls = _install(monkeypatch, lambda f, s: _success())
    exp = _experiment(_args())
    stage = Stage("solve")

    exp.stage_solve(stage)

    assert stage.data["solve_tier"] == "T1_balanced"
    assert stage.data["solve_trust"] == "measured"
    assert stage.data["tier_degraded"] is False
    assert len(calls) == 1  # nothing further was tried
    assert calls[0]["quality"] == "balanced"


def test_a_stricter_rung_without_the_holdout_beats_a_looser_one_with_it(monkeypatch):
    """Tier is OUTER to the frame set. A pose that is wrong cannot be redeemed
    by having more of them, so trust is spent before coverage."""

    def decide(frames, settings):
        if len(frames) == 3:
            return _refusal("insufficient_overlap", "45 of 48")
        return _success()

    _install(monkeypatch, decide)
    exp = _experiment(_args())
    stage = Stage("solve")

    exp.stage_solve(stage)

    assert stage.data["solve_tier"] == "T1_balanced"
    assert stage.data["holdout_posed_by_solve"] is False
    assert "no rung of the ladder solved a rig containing the holdout" in stage.data["holdout_leak"]


def test_forcing_the_translated_model_is_a_rung_not_a_default(monkeypatch):
    """sh001's evidence passes salvage's grid-cell floor; it refused only
    because `auto` would not choose between two marginal models."""

    def decide(frames, settings):
        if settings.capture_mode != "translated":
            return _refusal("ambiguous_motion_model", "neither model passed")
        return _success()

    _install(monkeypatch, decide)
    exp = _experiment(_args())
    stage = Stage("solve")

    exp.stage_solve(stage)

    assert stage.data["solve_tier"] == "T4_forced_translated"
    assert stage.data["solve_trust"] == "salvage_forced_model"
    assert stage.data["tier_degraded"] is True


def test_every_refused_rung_is_recorded_with_its_outcome_code(monkeypatch):
    """A ladder that hides the rungs it climbed is just a looser threshold."""

    def decide(frames, settings):
        if settings.match_quality != "salvage":
            return _refusal("insufficient_overlap", "32 of 48")
        return _success()

    _install(monkeypatch, decide)
    exp = _experiment(_args())
    stage = Stage("solve")

    exp.stage_solve(stage)

    attempts = stage.data["solve_attempts"]
    assert len(attempts) > 1
    assert attempts[-1]["outcome_code"] == "ok"
    refused = [a for a in attempts if a["outcome_code"] != "ok"]
    assert refused and all(a["outcome_code"] == "insufficient_overlap" for a in refused)
    assert all("tier" in a and "frames" in a for a in attempts)


def test_no_degrade_refuses_instead_of_relaxing(monkeypatch):
    calls = _install(monkeypatch, lambda f, s: _refusal("ambiguous_motion_model"))
    exp = _experiment(_args(degrade=False))

    with pytest.raises(RuntimeError, match="every rung"):
        exp.stage_solve(Stage("solve"))

    assert {c["quality"] for c in calls} == {"balanced"}


def test_when_every_rung_refuses_the_error_names_them(monkeypatch):
    _install(monkeypatch, lambda f, s: _refusal("ambiguous_motion_model", "detail"))
    exp = _experiment(_args())

    with pytest.raises(RuntimeError) as excinfo:
        exp.stage_solve(Stage("solve"))

    message = str(excinfo.value)
    assert "T1_balanced" in message and "T5_rotation_only" in message
    assert "detail" in message


def test_match_quality_sets_the_floor_of_the_ladder(monkeypatch):
    calls = _install(monkeypatch, lambda f, s: _refusal("insufficient_overlap"))
    exp = _experiment(_args(match_quality="salvage"))

    with pytest.raises(RuntimeError):
        exp.stage_solve(Stage("solve"))

    assert {c["quality"] for c in calls} == {"salvage"}


# --------------------------------------------------------------- bookkeeping


def test_the_holdout_is_dropped_by_LABEL_not_by_index(monkeypatch):
    """Regression. Filtering the USED frame list by the holdout's position in
    the ORIGINAL list drops a training frame whenever the holdout is not last —
    and the learned-depth-scale fallback replaces photo 1 with a copy, so the
    used list is not the original list either."""

    _install(monkeypatch, lambda f, s: _success())
    exp = _experiment(_args(holdout=0))
    stage = Stage("solve")

    exp.stage_solve(stage)

    assert stage.data["holdout_frame"] == "A"
    assert stage.data["train_frames"] == ["B", "C"]
    assert [f.label for f in exp.state["train_frames"]] == ["B", "C"]


def test_the_trust_label_reaches_the_state_the_later_stages_read(monkeypatch):
    """A salvage rig must not be able to produce a confident-looking claim."""

    def decide(frames, settings):
        return _success() if settings.match_quality == "salvage" else _refusal("x")

    _install(monkeypatch, decide)
    exp = _experiment(_args())
    exp.stage_solve(Stage("solve"))

    assert exp.state["solve_trust"].startswith("salvage")
    assert exp.state["solve_tier"].startswith("T3")


def test_a_holdout_index_outside_the_burst_is_refused(monkeypatch):
    _install(monkeypatch, lambda f, s: _success())
    exp = _experiment(_args(holdout=7))
    with pytest.raises(ValueError, match="outside"):
        exp.stage_solve(Stage("solve"))


def test_two_frames_leave_nothing_to_train_on_after_a_holdout(monkeypatch):
    _install(monkeypatch, lambda f, s: _success())
    exp = _experiment(_args(holdout=1), labels=("A", "B"))
    with pytest.raises(ValueError, match="need >= 2 frames"):
        exp.stage_solve(Stage("solve"))
