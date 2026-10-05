"""Wrapper contract for AtlasAdherenceScore 📐.

The scoring itself is pinned in tests/test_adherence.py against known answers.
This file pins what only the node decides: that the control arm is a REQUIRED
socket rather than an optional courtesy, that the headline leaves as a FLOAT, and
that a degenerate run is refused at the graph boundary too.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.node_registry import NODE_CLASS_MAPPINGS  # noqa: E402
from atlas_camera.core.adherence import DegenerateArmError  # noqa: E402
from atlas_camera.core.conditioning import render_conditioning_sequence  # noqa: E402

SCORE = NODE_CLASS_MAPPINGS["AtlasAdherenceScore"]
W, H = 128, 96
FX = FY = 160.0
K = [[FX, 0.0, W / 2.0], [0.0, FY, H / 2.0], [0.0, 0.0, 1.0]]
WALL_Z = 10.0


def _view(x):
    vm = np.eye(4)
    vm[0, 3] = -x
    return vm


def _rect(z, x0, x1, y0, y1, label):
    verts = np.array([[x0, y0, -z], [x1, y0, -z], [x1, y1, -z], [x0, y1, -z]],
                     dtype=np.float64)
    return (label, verts, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            "primary", {})


def _checker(n=64, squares=8):
    yy, xx = np.mgrid[0:n, 0:n]
    c = (((xx * squares) // n + (yy * squares) // n) % 2).astype(np.float64)
    rgb = np.stack([c, c * 0.6 + 0.2, 1.0 - c], axis=-1) * 0.8 + 0.1
    return np.concatenate([rgb, np.ones((n, n, 1))], axis=-1)


@pytest.fixture
def bundle():
    """A wall with the occluded span REMOVED, so the move produces real GHOST —
    a solid wall behind an occluder reveals real geometry and is correctly all
    VALID, which would make this a coverage test rather than a camera test."""
    gap = 0.45 * WALL_Z / 3.0
    meshes = [_rect(WALL_Z, -40.0, -gap, -30.0, 30.0, "wall_left"),
              _rect(WALL_Z, gap, 40.0, -30.0, 30.0, "wall_right"),
              _rect(3.0, -0.45, 0.45, -0.6, 0.6, "blocker")]
    views = [_view(0.0), _view(0.25), _view(0.5), _view(0.75)]
    return render_conditioning_sequence(
        meshes, {"primary": _checker()}, views=views, intrinsics=[K] * 4,
        plate_view=views[0], plate_k=K, width=W, height=H)


def _tensor(frames):
    return torch.from_numpy(np.ascontiguousarray(frames)).float()


def _control(bundle):
    return _tensor(np.stack([np.roll(f, 9, axis=1) for f in bundle.rgb]))


def test_the_control_is_a_required_socket():
    """Enforced at graph level, because a report can only ask politely."""
    required = SCORE.INPUT_TYPES()["required"]
    assert "control" in required
    assert "control" not in SCORE.INPUT_TYPES().get("optional", {})
    assert "bundle" in required and "generated" in required


def test_socket_arity_and_types(bundle):
    out = SCORE().score(bundle, _tensor(bundle.rgb), _control(bundle))
    assert len(out) == len(SCORE.RETURN_TYPES) == len(SCORE.RETURN_NAMES)
    assert isinstance(out[0], float) and isinstance(out[1], float)
    assert out[2].ndim == 4          # IMAGE batch for the drift plot
    json.loads(out[3])


def test_a_perfect_arm_scores_one_with_a_positive_margin(bundle):
    adherence, margin, _plot, report = SCORE().score(
        bundle, _tensor(bundle.rgb), _control(bundle))
    payload = json.loads(report)
    assert adherence == pytest.approx(1.0, abs=1e-6)
    assert margin > 0.0
    assert payload["guard"]["passed"] is True
    assert payload["headline_measure"].startswith("gradient_zncc")


def test_a_frozen_arm_is_refused_at_the_node_boundary(bundle):
    frozen = _tensor(np.repeat(bundle.rgb[:1], bundle.frames, axis=0))
    with pytest.raises(DegenerateArmError, match="parallax_response"):
        SCORE().score(bundle, frozen, _control(bundle))


def test_the_refusal_can_be_disabled_only_deliberately(bundle):
    frozen = _tensor(np.repeat(bundle.rgb[:1], bundle.frames, axis=0))
    adherence, _margin, _plot, report = SCORE().score(
        bundle, frozen, _control(bundle), refuse_degenerate=False)
    payload = json.loads(report)
    assert adherence > 0.5, "the cheat does score well — that is the point"
    assert payload["guard"]["passed"] is False
    assert payload["guard"]["parallax_response"] <= 0.0


def test_a_missing_bundle_explains_itself(bundle):
    with pytest.raises(ValueError, match="no bundle"):
        SCORE().score(None, _tensor(bundle.rgb), _control(bundle))


def test_the_drift_plot_draws_every_arm(bundle):
    _a, _m, plot, report = SCORE().score(
        bundle, _tensor(bundle.rgb), _control(bundle))
    payload = json.loads(report)
    assert set(payload["arms"]) == {"atlas", "prompt_only", "static"}
    img = plot[0].cpu().numpy()
    assert img.shape[2] == 3
    # Three distinct series colours must actually appear on the canvas.
    colours = {tuple(np.round(c, 3)) for c in img.reshape(-1, 3)}
    assert len(colours) >= 4          # background, gridline, and the arms


def test_an_alpha_channel_on_the_input_is_dropped_not_averaged(bundle):
    """A 4-channel IMAGE must not fold alpha into the luminance the headline is
    computed from."""
    rgba = np.concatenate(
        [bundle.rgb, np.ones((*bundle.rgb.shape[:3], 1), dtype=np.float32)],
        axis=-1)
    adherence, _m, _p, _r = SCORE().score(
        bundle, _tensor(rgba), _control(bundle))
    assert adherence == pytest.approx(1.0, abs=1e-6)
