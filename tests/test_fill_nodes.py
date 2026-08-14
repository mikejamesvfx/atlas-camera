"""In-graph two-pass nodes: the gate's self-fallback and the composite stack.

The gate's contract is the load-bearing part: on FAILURE it outputs the
GUIDE, so a downstream texture pass re-touches nothing and the composite
degrades to a no-op — the in-graph equivalent of the CLI engine's fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_fill import (
    AtlasInterpassGate,
    AtlasMembraneComposite,
)


def _img(rgb):
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _plate(h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    lum = rng.integers(90, 140, size=(h, w)).astype(np.int16)
    lum[:, ::8] = lum[:, ::8] // 2 + 60
    return np.clip(np.stack([lum + 4, lum, lum - 4], -1), 0, 255).astype(np.uint8)


def _hole_mask(h=96, w=96):
    m = np.zeros((h, w), np.float32)
    m[32:64, 32:64] = 1.0
    return torch.from_numpy(m)


def test_gate_passes_a_real_fill_and_returns_it():
    guide = _plate()
    rng = np.random.default_rng(1)
    fill = guide.copy()
    fill[32:64, 32:64] = rng.integers(60, 200, size=(32, 32, 3))
    out, ok, report = AtlasInterpassGate().gate(_img(fill), _img(guide),
                                                _hole_mask())
    assert ok and "PASS" in report
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.abs(got.astype(int) - fill.astype(int)).mean() < 1.0


def test_gate_fails_a_smear_and_passes_the_guide_through():
    from atlas_camera.dynamic.fill_metrics import edge_extend

    guide = _plate()
    hole = np.zeros((96, 96), bool)
    hole[32:64, 32:64] = True
    smear = edge_extend(guide, hole)
    out, ok, report = AtlasInterpassGate().gate(_img(smear), _img(guide),
                                                _hole_mask())
    assert not ok and "FAIL" in report
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.abs(got.astype(int) - guide.astype(int)).mean() < 1.0
    assert "guide passed through" in report


def test_gate_resizes_a_low_raster_fill_before_scoring():
    """The WAN branch generates at 720p-class; the gate must score at the
    reference raster, not crash on the mismatch."""
    from PIL import Image as PILImage

    guide = _plate(128, 128, seed=2)
    rng = np.random.default_rng(3)
    fill_small = np.array(PILImage.fromarray(guide).resize((64, 64)))
    fill_small[16:32, 16:32] = rng.integers(60, 200, size=(16, 16, 3))
    m = np.zeros((128, 128), np.float32)
    m[32:64, 32:64] = 1.0
    out, ok, report = AtlasInterpassGate().gate(
        _img(fill_small), _img(guide), torch.from_numpy(m))
    assert out.shape[1:3] == (128, 128)


def test_gate_empty_hole_is_a_fail_with_guide_passthrough():
    guide = _plate()
    out, ok, report = AtlasInterpassGate().gate(
        _img(guide), _img(guide), torch.zeros(96, 96))
    assert not ok and "empty hole" in report


def test_membrane_composite_erases_an_offset_and_pastes_only_the_hole():
    ref = _plate(seed=4)
    hole = np.zeros((96, 96), bool)
    hole[32:64, 32:64] = True
    fill = ref.copy()
    fill[hole] = np.clip(ref[hole].astype(int) - 30, 0, 255).astype(np.uint8)
    out, report = AtlasMembraneComposite().composite(
        _img(fill), _img(ref), _hole_mask())
    got = (out[0].numpy() * 255).astype(np.uint8)
    # membrane recovers the offset inside the hole
    assert np.abs(got[hole].astype(int) - ref[hole].astype(int)).mean() < 4.0
    # outside the hole the reference is untouched
    assert np.array_equal(got[~hole], ref[~hole])
    assert "membrane applied" in report


def test_membrane_composite_empty_hole_returns_reference():
    ref = _plate(seed=5)
    out, report = AtlasMembraneComposite().composite(
        _img(np.zeros_like(ref)), _img(ref), torch.zeros(96, 96))
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.array_equal(got, ref)
    assert "empty hole" in report


def test_path_frame_index_computes_window_and_last():
    from atlas_camera.comfy.nodes_fill import AtlasPathFrameIndex
    from atlas_camera.core.camera_path import AtlasCameraPath

    # no path: solved-pose single frame
    count, last, start, report = AtlasPathFrameIndex().index(None)
    assert (count, last, start) == (1, 0, 0)
    assert "solved pose" in report

    # a real 30-frame path must agree with the guide's sampler
    from atlas_camera.core.camera_path import sample_camera_path
    path = AtlasCameraPath(frame_count=30, keyframes=[])
    n = len(sample_camera_path(path))
    count, last, start, report = AtlasPathFrameIndex().index(path, window=5)
    assert count == max(n, 1) or (n == 0 and count == 1)
    if n:
        assert last == n - 1 and start == max(0, n - 5)

    # non-4k+1 window is warned, not refused
    if n:
        *_ignore, report = AtlasPathFrameIndex().index(path, window=6)
        assert "4k+1" in report
