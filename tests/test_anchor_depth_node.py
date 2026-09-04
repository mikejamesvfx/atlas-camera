"""AtlasAnchorDepth — a per-frame estimate put on the render's scale.

The node takes the TRACKING from a monocular estimate and the SCALE from a
rendered pass, by fitting `z_render ~= s*z_est + t` on the one frame the two
share and carrying that onto every frame. Three properties are load-bearing and
pinned here:

  * the fit is carried onto EVERY frame, not just the anchored one — that is
    what puts a moving figure the render never contained at the right depth;
  * the trim removes disagreement from the FIT while leaving it in the OUTPUT,
    since those pixels (invented content, backplate cards) are the whole reason
    the estimate is present;
  * the intrinsics come from the ANCHOR, because a recovered focal is the other
    half of what the estimate gets wrong.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    AtlasAnchorDepth,
)

K_RENDER = np.array([[900.0, 0.0, 480.0],
                     [0.0, 900.0, 320.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)
K_EST = np.array([[640.0, 0.0, 480.0],
                  [0.0, 640.0, 320.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)


def _geom(depth, *, intrinsics=None, mask=None):
    g = {"depth": np.asarray(depth, dtype=np.float32)}
    if intrinsics is not None:
        g["intrinsics"] = np.asarray(intrinsics, dtype=np.float32)
    if mask is not None:
        g["mask"] = np.asarray(mask, dtype=bool)
    return g


def _scene(h=32, w=32):
    """A truthful render: depth receding from 5 m to 40 m."""
    return (np.linspace(5.0, 40.0, h)[:, None] * np.ones((1, w))).astype(np.float32)


def test_registered_and_output_names():
    assert NODE_CLASS_MAPPINGS["AtlasAnchorDepth"] is AtlasAnchorDepth
    assert AtlasAnchorDepth.RETURN_NAMES == ("moge_geometry", "report")


def test_recovers_the_scale_and_shift_the_estimate_was_out_by():
    render = _scene()
    # The estimate has the right STRUCTURE and the wrong scale/offset — exactly
    # the failure mode the node exists for.
    estimate = render / 2.5 - 1.0

    out, report = AtlasAnchorDepth().anchor(
        _geom(estimate), _geom(render, intrinsics=K_RENDER))

    got = out["depth"].numpy() if hasattr(out["depth"], "numpy") else np.asarray(out["depth"])
    assert got.shape[-2:] == render.shape
    np.testing.assert_allclose(got[0], render, rtol=1e-3, atol=1e-2)
    assert "z_render = 2.5" in report or "2.5000" in report


def test_the_fit_carries_onto_every_frame_of_the_estimate():
    """The render is a STILL repeated across the clip; the estimate is per-frame.
    A fit that only corrected the anchored frame would leave the rest at the
    estimate's own scale, which is the error nothing downstream detects."""
    render = _scene()
    # Three frames of estimate, all at the same wrong scale.
    estimate = np.stack([render / 3.0, render / 3.0 + 0.5, render / 3.0 - 0.25])

    out, _report = AtlasAnchorDepth().anchor(
        _geom(estimate), _geom(render, intrinsics=K_RENDER), fit_shift=True)

    got = np.asarray(out["depth"])
    assert got.shape[0] == 3
    # Frame 0 was the anchor; frames 1 and 2 must have moved to the same scale,
    # keeping their own per-frame differences.
    for i in range(3):
        assert abs(float(np.median(got[i])) - float(np.median(render))) < 2.0
    assert float(np.median(got[1])) > float(np.median(got[2]))


def test_intrinsics_come_from_the_anchor_never_the_estimate():
    """A recovered focal is the other half of what a monocular estimate gets
    wrong; the render's is measured."""
    render = _scene()
    out, report = AtlasAnchorDepth().anchor(
        _geom(render / 2.0, intrinsics=K_EST),
        _geom(render, intrinsics=K_RENDER))

    K = np.asarray(out["intrinsics"])
    assert K[..., 0, 0].ravel()[0] == pytest.approx(900.0)
    assert "intrinsics  taken from the anchor" in report


def test_disagreement_is_trimmed_from_the_fit_but_kept_in_the_output():
    """The two sources disagree exactly where the estimate is needed: content
    the render is missing. Trimming those pixels out of the FIT is the point;
    dropping them from the OUTPUT would throw away the reason for the node."""
    render = _scene()
    estimate = render / 2.0
    # A figure the render never contained: the estimate sees it much nearer.
    estimate[20:26, 10:16] = 0.4

    out, report = AtlasAnchorDepth().anchor(
        _geom(estimate), _geom(render, intrinsics=K_RENDER), trim_sigma=3.0)

    got = np.asarray(out["depth"])[0]
    # The fit was not dragged by the intruder: the rest still lands on render.
    clean = np.ones_like(render, bool)
    clean[20:26, 10:16] = False
    np.testing.assert_allclose(got[clean], render[clean], rtol=0.05, atol=0.5)
    # And the intruder SURVIVES, still much nearer than its surroundings.
    assert float(np.median(got[20:26, 10:16])) < float(np.median(got[clean]))
    assert "trimmed" in report


def test_fit_shift_off_forces_the_fit_through_the_origin():
    render = _scene()
    _out, report = AtlasAnchorDepth().anchor(
        _geom(render / 2.0), _geom(render, intrinsics=K_RENDER), fit_shift=False)
    assert "shift held at 0" in report


def test_disparity_space_is_selectable_for_a_relative_estimate():
    """depth is the measured default, but a relative estimator wants 1/z."""
    render = _scene()
    inv = 1.0 / render
    estimate = (1.0 / (inv * 2.0)).astype(np.float32)   # linear in DISPARITY

    out, report = AtlasAnchorDepth().anchor(
        _geom(estimate), _geom(render, intrinsics=K_RENDER),
        fit_space="disparity")

    got = np.asarray(out["depth"])[0]
    np.testing.assert_allclose(got, render, rtol=1e-2, atol=0.2)
    assert "disparity" in report


def test_mismatched_rasters_are_refused_with_both_sizes_named():
    with pytest.raises(ValueError, match="pixel by pixel"):
        AtlasAnchorDepth().anchor(
            _geom(np.ones((16, 16), np.float32) * 4.0),
            _geom(np.ones((32, 32), np.float32) * 4.0, intrinsics=K_RENDER))


def test_too_little_overlap_is_refused_rather_than_fitted():
    render = _scene()
    mask = np.zeros_like(render, bool)
    mask[:2, :4] = True                       # 8 valid pixels, far under 64
    with pytest.raises(ValueError, match="fewer than 64"):
        AtlasAnchorDepth().anchor(
            _geom(render / 2.0, mask=mask),
            _geom(render, intrinsics=K_RENDER))


def test_an_anchor_without_intrinsics_is_refused():
    render = _scene()
    with pytest.raises(ValueError, match="no intrinsics"):
        AtlasAnchorDepth().anchor(_geom(render / 2.0), _geom(render))


def test_the_report_names_how_far_out_the_estimate_alone_was():
    """`s` and `t` are reported, not hidden: a scale far from 1 is the honest
    measure of how wrong a run driven by the estimate alone would have been."""
    render = _scene()
    _out, report = AtlasAnchorDepth().anchor(
        _geom(render / 4.0), _geom(render, intrinsics=K_RENDER))
    assert "NOTE" in report and "off the render" in report
