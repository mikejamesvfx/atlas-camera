"""Depth outpainting: extend depth to match an already-widened plate.

The gap being closed: AtlasCleanPlateLayer can widen a layer past the frame
(frame_outpaint_px) and its tooltip calls the frame-edge reveal "the binding
constraint on wide scenes" — but the ring is edge-replicated smear and depth
cannot follow it (AtlasMogeNormals refuses to run when frame_outpaint_px != 0).
Colour with no surface under it.

The test that matters is `test_unanchored_stitch_steps_at_the_frame_edge`: it
demonstrates the failure the anchoring exists to prevent, so if anchoring ever
stops being load-bearing, that test says so instead of quietly passing.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.depth_outpaint import (
    outpaint_depth,
    ring_mask_for,
)

np = pytest.importorskip("numpy")


def _scene(h=64, w=96, pad=(16, 12, 16, 12)):
    """A depth ramp, plus the same ramp on a widened canvas at a DIFFERENT scale.

    The scale difference is the point: a monocular model run on a widened image
    returns a different scale than the same model on the original.
    """
    left, top, right, bottom = pad
    H, W = h + top + bottom, w + left + right
    full = np.tile(np.linspace(2.0, 20.0, W), (H, 1))
    original = full[top:top + h, left:left + w].copy()
    widened = full * 0.4 + 1.7          # the model's different scale on the wide frame
    return original, widened, pad


# --------------------------------------------------------------- anchoring


def test_anchor_recovers_the_widened_scale():
    original, widened, pad = _scene()
    r = outpaint_depth(original, widened, pad=pad)
    # widened = 0.4*full + 1.7  ->  full = (widened - 1.7)/0.4
    assert r.scale == pytest.approx(1.0 / 0.4, rel=1e-4)
    assert r.shift == pytest.approx(-1.7 / 0.4, rel=1e-4)
    assert r.anchor_residual < 1e-6
    assert r.metadata["anchored"] is True


def test_interior_keeps_the_original_measurement():
    """The widened pass saw invented pixels; it has no claim on what we knew."""
    original, widened, pad = _scene()
    r = outpaint_depth(original, widened, pad=pad)
    left, top = pad[0], pad[1]
    h, w = original.shape
    assert np.allclose(r.depth[top:top + h, left:left + w], original, atol=1e-5)


def test_ring_is_continuous_with_the_interior():
    """No step at the frame edge — the whole reason for anchoring."""
    original, widened, pad = _scene()
    r = outpaint_depth(original, widened, pad=pad)
    d = r.depth.astype(np.float64)
    dx = np.abs(np.diff(d, axis=1))
    # The underlying field is a uniform ramp, so every column step is equal.
    assert dx.max() < dx.mean() * 2.0, (
        f"a discontinuity survived: max step {dx.max():.5f} vs mean {dx.mean():.5f}")


def test_unanchored_stitch_steps_at_the_frame_edge():
    """Demonstrates the failure anchoring prevents.

    If this ever passes cleanly, the anchor step has stopped mattering and the
    extra fit could be dropped.
    """
    original, widened, pad = _scene()
    naive = outpaint_depth(original, widened, pad=pad, anchor=False)
    d = naive.depth.astype(np.float64)
    dx = np.abs(np.diff(d, axis=1))
    assert dx.max() > dx.mean() * 5.0, (
        "unanchored stitch produced no step — anchoring is no longer load-bearing")


def test_falls_back_gracefully_when_the_overlap_is_unusable():
    original, widened, pad = _scene()
    original = original.copy()
    original[:] = np.nan                      # nothing to anchor against
    r = outpaint_depth(original, widened, pad=pad)
    assert r.metadata["anchored"] is False
    assert (r.scale, r.shift) == (1.0, 0.0)


def test_flat_overlap_uses_offset_only():
    """A constant overlap makes the slope fit singular; an offset is still valid."""
    h, w, pad = 64, 64, (8, 8, 8, 8)
    original = np.full((h, w), 5.0)
    widened = np.full((h + 16, w + 16), 2.0)
    r = outpaint_depth(original, widened, pad=pad)
    assert r.scale == pytest.approx(1.0)
    assert r.shift == pytest.approx(3.0)


# ------------------------------------------------------------------ mask


def test_ring_mask_marks_only_invented_pixels():
    m = ring_mask_for(96, 64, (16, 12, 16, 12), np)
    assert m.shape == (64 + 24, 96 + 32)
    assert not m[12:12 + 64, 16:16 + 96].any(), "interior flagged as invented"
    assert m[0, 0] and m[-1, -1], "ring not flagged"


def test_ring_fraction_is_reported():
    original, widened, pad = _scene()
    r = outpaint_depth(original, widened, pad=pad)
    assert 0.0 < r.metadata["ring_fraction"] < 1.0
    assert r.ring_mask.sum() == int(round(r.metadata["ring_fraction"] * r.ring_mask.size))


# ------------------------------------------------------------- guards


def test_padding_mismatch_names_the_problem():
    original = np.ones((64, 96))
    with pytest.raises(ValueError, match="implies"):
        outpaint_depth(original, np.ones((70, 100)), pad=(16, 12, 16, 12))


def test_negative_padding_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        outpaint_depth(np.ones((16, 16)), np.ones((16, 16)), pad=(-1, 0, 0, 0))


def test_correction_cannot_push_depth_behind_the_camera():
    original = np.full((64, 64), 5.0)
    widened = np.full((80, 80), 1.0)
    widened[:4, :] = 0.001                       # would go negative under the fit
    r = outpaint_depth(original, widened, pad=(8, 8, 8, 8))
    finite = np.isfinite(r.depth)
    assert (r.depth[finite] > 0).all()


def test_zero_padding_is_a_no_op():
    d = np.linspace(1, 10, 4096).reshape(64, 64)
    r = outpaint_depth(d, d.copy(), pad=(0, 0, 0, 0))
    assert np.allclose(r.depth, d, atol=1e-5)
    assert not r.ring_mask.any()


# ------------------------------------------------------------ feathering


def test_feather_only_touches_edges_that_actually_grew():
    """Blending an edge with no ring beyond it would corrupt real data for nothing."""
    h, w = 64, 64
    original = np.tile(np.linspace(2.0, 8.0, w), (h, 1))
    pad = (0, 0, 16, 0)                       # grew on the RIGHT only
    widened = np.tile(np.linspace(2.0, 10.0, w + 16), (h, 1))
    r = outpaint_depth(original, widened, pad=pad, feather_px=8)
    # The left edge did not grow, so its original column must survive intact.
    assert np.allclose(r.depth[:, 0], original[:, 0], atol=1e-5)


def test_feather_reduces_a_residual_step():
    """When the fit is imperfect, feathering should spread the error, not keep it."""
    h, w = 64, 96
    pad = (16, 0, 16, 0)
    original = np.tile(np.linspace(3.0, 9.0, w), (h, 1))
    widened = np.tile(np.linspace(1.0, 13.0, w + 32), (h, 1)) * 0.9 + 0.4

    hard = outpaint_depth(original, widened, pad=pad, feather_px=0)
    soft = outpaint_depth(original, widened, pad=pad, feather_px=10)
    step_hard = np.abs(np.diff(hard.depth.astype(np.float64), axis=1)).max()
    step_soft = np.abs(np.diff(soft.depth.astype(np.float64), axis=1)).max()
    assert step_soft <= step_hard


# ------------------------------------------------------- AtlasOutpaintDepth

torch = pytest.importorskip("torch")


def _depth_result(h, w, near=2.0, far=20.0):
    from atlas_camera.inference.depth_estimator import DepthResult
    return DepthResult(depth=np.tile(np.linspace(near, far, w), (h, 1)).astype(np.float32),
                       is_metric=True, model_id="test", image_width=w, image_height=h,
                       near=near, far=far)


def _node(monkeypatch, wide_h, wide_w):
    """AtlasOutpaintDepth with the depth model stubbed out."""
    from atlas_camera.comfy import node_registry as reg
    from atlas_camera.inference import depth_estimator as de

    monkeypatch.setattr(de, "estimate_depth",
                        lambda *a, **k: _depth_result(wide_h, wide_w, 1.0, 40.0))
    return reg.NODE_CLASS_MAPPINGS["AtlasOutpaintDepth"]()


def test_node_derives_symmetric_padding_from_the_size_difference(monkeypatch):
    h, w, pad = 64, 96, 16
    node = _node(monkeypatch, h + 2 * pad, w + 2 * pad)
    wide_img = torch.zeros((1, h + 2 * pad, w + 2 * pad, 3))
    out, ring, report = node.outpaint(_depth_result(h, w), wide_img)

    assert out.image_width == w + 2 * pad and out.image_height == h + 2 * pad
    assert tuple(ring.shape) == (1, h + 2 * pad, w + 2 * pad)
    assert f"pad l{pad} t{pad} r{pad} b{pad}" in report


def test_node_puts_an_odd_remainder_on_right_and_bottom(monkeypatch):
    """The four paddings must reconstruct the widened size EXACTLY.

    Splitting an odd difference evenly would lose a pixel and the core would
    reject the mismatch — correctly, but confusingly.
    """
    h, w = 64, 96
    node = _node(monkeypatch, h + 7, w + 5)
    out, _ring, report = node.outpaint(_depth_result(h, w),
                                       torch.zeros((1, h + 7, w + 5, 3)))
    assert out.image_width == w + 5 and out.image_height == h + 7
    assert "pad l2 t3 r3 b4" in report


def test_node_rejects_an_unwidened_plate_with_a_usable_message(monkeypatch):
    node = _node(monkeypatch, 64, 96)
    with pytest.raises(ValueError, match="no ring to fill"):
        node.outpaint(_depth_result(64, 96), torch.zeros((1, 64, 96, 3)))


def test_node_rejects_a_smaller_plate(monkeypatch):
    node = _node(monkeypatch, 32, 48)
    with pytest.raises(ValueError, match="smaller than"):
        node.outpaint(_depth_result(64, 96), torch.zeros((1, 32, 48, 3)))


def test_node_honours_an_explicit_pad_override(monkeypatch):
    h, w = 64, 96
    node = _node(monkeypatch, h + 20, w + 30)
    _out, _ring, report = node.outpaint(
        _depth_result(h, w), torch.zeros((1, h + 20, w + 30, 3)),
        pad_override="10,5,20,15")
    assert "pad l10 t5 r20 b15" in report


def test_node_rejects_a_malformed_pad_override(monkeypatch):
    node = _node(monkeypatch, 84, 116)
    with pytest.raises(ValueError, match="left,top,right,bottom"):
        node.outpaint(_depth_result(64, 96), torch.zeros((1, 84, 116, 3)),
                      pad_override="10,5")


def test_node_drops_stale_normals_and_says_so(monkeypatch):
    """A normal map for the ORIGINAL frame is mis-registered against the widened
    plate — exactly the failure AtlasMogeNormals refuses frame_outpaint_px over.
    Silently keeping it would reintroduce that bug one node downstream."""
    h, w = 64, 96
    src = _depth_result(h, w)
    src.normal = np.zeros((h, w, 3), dtype=np.float32)
    node = _node(monkeypatch, h + 16, w + 16)
    out, _ring, report = node.outpaint(src, torch.zeros((1, h + 16, w + 16, 3)))
    assert out.normal is None
    assert "normals DROPPED" in report


def test_node_reports_the_ring_as_invented(monkeypatch):
    node = _node(monkeypatch, 96, 128)
    _out, _ring, report = node.outpaint(_depth_result(64, 96),
                                        torch.zeros((1, 96, 128, 3)))
    assert "INVENTED" in report
    assert "anchored" in report.lower()


def test_node_output_contract():
    from atlas_camera.comfy import node_registry as reg
    cls = reg.NODE_CLASS_MAPPINGS["AtlasOutpaintDepth"]
    assert cls.RETURN_TYPES == ("ATLAS_DEPTH_MAP", "MASK", "STRING")
    assert cls.RETURN_NAMES == ("depth", "ring_mask", "report")
