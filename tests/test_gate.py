"""Tests for `atlas_camera.comfy.gate` — the one Gate that every gated node
delegates to.

Four nodes used to hand-roll the same five steps (fingerprint, arming
comparison, re-arm sentence, ExecutionBlocker choice, ui/result envelope):
AtlasSolveGate, AtlasSceneHealthGate, AtlasAssessImage and
AtlasBlockoutViewport's patch branch. This module pins the decision table
itself, so a node only has to declare WHAT it gates.

The gate contract is a SAVED-WORKFLOW contract: the fingerprint values and the
`{"ui": {"text": [...], "fingerprint": [...]}, "result": (...)}` envelope are
read by three frontend extensions (atlas_solve_gate.js,
atlas_scene_health_gate.js, atlas_assess.js) and persist in every approved
workflow — hence the byte-level assertions here.
"""

import sys
import types

import pytest

from atlas_camera.comfy.gate import Gate, image_fingerprint, solve_fingerprint

FP = "abc123def4560000"
OTHER = "0000000000000000"


class FakeBlocker:
    def __init__(self, message):
        self.message = message


@pytest.fixture()
def comfy_runtime(monkeypatch):
    """A stand-in ComfyUI runtime so `_execution_blocker()` resolves."""
    mod = types.ModuleType("comfy_execution.graph")
    mod.ExecutionBlocker = FakeBlocker
    pkg = types.ModuleType("comfy_execution")
    pkg.graph = mod
    monkeypatch.setitem(sys.modules, "comfy_execution", pkg)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", mod)


# --------------------------------------------------------------------------
# The decision table
# --------------------------------------------------------------------------

def test_proceed_off_blocks():
    g = Gate(FP, proceed=False)
    assert g.fingerprint == FP
    assert not g.approved
    assert not g.passed
    assert not g.re_armed


def test_proceed_on_without_approval_is_the_manual_override():
    # An empty approved_for with proceed=True is the unconditional manual
    # toggle — the artist flipped the widget by hand, not via the button.
    g = Gate(FP, proceed=True, approved_for="")
    assert g.approved and g.passed and not g.re_armed


def test_matching_approval_passes():
    g = Gate(FP, proceed=True, approved_for=FP)
    assert g.approved and g.passed and not g.re_armed


def test_mismatched_approval_re_arms_and_blocks():
    g = Gate(FP, proceed=True, approved_for=OTHER)
    assert not g.approved
    assert not g.passed
    assert g.re_armed


def test_mismatched_approval_without_proceed_is_not_re_armed():
    # Nothing to re-arm: the gate is already closed.
    g = Gate(FP, proceed=False, approved_for=OTHER)
    assert not g.re_armed and not g.passed


def test_bypass_passes_without_approval():
    # `bypass` is the node-declared widening (AtlasAssessImage's
    # auto_continue, AtlasSceneHealthGate's pass_through_on_pass).
    g = Gate(FP, proceed=False, bypass=True)
    assert g.passed
    assert not g.approved          # the approval itself was never given


def test_bypass_does_not_hide_a_stale_approval():
    # AtlasSceneHealthGate shows the re-arm line even on a clean pass-through:
    # the user may OVERRIDE a warning but never LOSE it.
    g = Gate(FP, proceed=True, approved_for=OTHER, bypass=True)
    assert g.passed
    assert g.re_armed


def test_blank_approval_can_be_made_strict():
    # AtlasBlockoutViewport's patch branch: an extraction from before
    # fingerprints existed carries no fingerprint and must re-arm the pause,
    # never read as an unconditional approval.
    g = Gate(FP, proceed=True, approved_for="", blank_is_unconditional=False)
    assert not g.approved and not g.passed
    assert Gate(FP, proceed=True, approved_for=FP,
                blank_is_unconditional=False).passed


# --------------------------------------------------------------------------
# The re-arm sentence — each node keeps its own tail
# --------------------------------------------------------------------------

def test_re_arm_banner_wraps_the_callers_tail():
    g = Gate(FP, proceed=True, approved_for=OTHER)
    assert g.re_arm_banner("the tail.") == "*** GATE RE-ARMED: the tail. ***"


def test_annotate_prepends_only_when_re_armed():
    armed = Gate(FP, proceed=True, approved_for=OTHER)
    assert armed.annotate("body", "tail.") == "*** GATE RE-ARMED: tail. ***\nbody"
    assert armed.annotate("body", "tail.", sep="\n\n") == (
        "*** GATE RE-ARMED: tail. ***\n\nbody")
    clean = Gate(FP, proceed=True, approved_for=FP)
    assert clean.annotate("body", "tail.") == "body"


# --------------------------------------------------------------------------
# The ExecutionBlocker choice
# --------------------------------------------------------------------------

def test_route_blocks_inside_a_comfy_runtime(comfy_runtime):
    sentinel = object()
    blocked = Gate(FP).route(sentinel)
    assert isinstance(blocked, FakeBlocker)
    assert blocked.message is None                     # silent skip, not an error
    assert Gate(FP, proceed=True).route(sentinel) is sentinel


def test_route_degrades_to_passthrough_outside_comfy():
    sentinel = object()
    assert Gate(FP).route(sentinel) is sentinel


def test_route_each_blocks_every_slot(comfy_runtime):
    vals = ("a", "b", "c", "d", "e")
    out = Gate(FP).route_each(vals)
    assert len(out) == 5
    assert all(isinstance(v, FakeBlocker) for v in out)
    assert Gate(FP, proceed=True).route_each(vals) == vals


# --------------------------------------------------------------------------
# The ui/result envelope — read by three JS extensions
# --------------------------------------------------------------------------

def test_envelope_carries_report_and_fingerprint():
    g = Gate(FP, proceed=True, approved_for=FP)
    env = g.envelope("the report", ("solve", "the report"))
    assert env == {"ui": {"text": ["the report"], "fingerprint": [FP]},
                   "result": ("solve", "the report")}


def test_envelope_merges_extra_ui_keys_without_losing_the_contract():
    g = Gate(FP)
    env = g.envelope("r", ["a"], ui={"sam_prompts": ["sky"]})
    assert env["ui"]["text"] == ["r"]
    assert env["ui"]["fingerprint"] == [FP]
    assert env["ui"]["sam_prompts"] == ["sky"]
    assert env["result"] == ("a",)                     # always a tuple


# --------------------------------------------------------------------------
# Fingerprints — the identity the approvals are scoped to
# --------------------------------------------------------------------------

def test_fingerprint_helpers_are_the_shipped_ones():
    from atlas_camera.comfy.fingerprints import _image_fingerprint, _solve_fingerprint

    # Re-exports, not re-implementations: a byte of drift here silently
    # re-arms every gate in every saved workflow the user has approved.
    assert solve_fingerprint is _solve_fingerprint
    assert image_fingerprint is _image_fingerprint


def test_for_image_and_for_solve_build_the_same_identity():
    torch = pytest.importorskip("torch")
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.schema import (
        AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera,
    )

    view, world, rot3 = look_at_view_matrix((0.0, 1.6, 0.0), (0.0, 0.5, -10.0))
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(
            image_width=800, image_height=600, focal_length_mm=35.0,
            sensor_width_mm=36.0, fx_px=700.0, fy_px=700.0,
            cx_px=400.0, cy_px=300.0),
        extrinsics=AtlasExtrinsics(
            camera_position=(0.0, 1.6, 0.0), camera_rotation_matrix=rot3,
            camera_world_matrix=world, camera_view_matrix=view)))
    image = torch.rand(1, 600, 800, 3)

    assert Gate.for_solve(solve, image).fingerprint == solve_fingerprint(solve, image)
    assert Gate.for_image(image).fingerprint == image_fingerprint(image)
    # Widget state rides along.
    g = Gate.for_image(image, proceed=True, approved_for=image_fingerprint(image))
    assert g.passed
