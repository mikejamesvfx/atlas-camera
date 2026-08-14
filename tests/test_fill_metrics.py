"""Falsifiable fill scoring — each gate must be shown passing AND failing.

The metric these replace (residual sentinel) could only fail if the generator
returned literal chroma green, and it read 0.0 on fills with a visible seam
and a global cast. Every gate here is exercised in both directions on
synthetic images so an arm report can be trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.dynamic.fill_metrics import (
    G2_MIN_EDGE_EXTEND_DIFF,
    G3A_MAX_RIM_GRADIENT_RATIO,
    edge_extend,
    encode_depth_guide,
    score_fill,
    unmasked_delta,
)


def _plate(h=96, w=96, seed=0):
    """Textured plate with CONTROLLED chroma: luminance noise + a constant
    channel offset, so a neutral fill's green excess matches its context by
    construction (random per-channel noise made G3b flaky at the 1.0/255
    threshold)."""
    rng = np.random.default_rng(seed)
    lum = rng.integers(90, 140, size=(h, w)).astype(np.int16)
    lum[:, ::8] = lum[:, ::8] // 2 + 60            # vertical texture lines
    base = np.stack([lum + 4, lum, lum - 4], axis=-1)   # fixed warm chroma
    return np.clip(base, 0, 255).astype(np.uint8)


def _hole(h=96, w=96):
    m = np.zeros((h, w), dtype=bool)
    m[32:64, 32:64] = True
    return m


# ------------------------------------------------------------- edge_extend

def test_edge_extend_fills_the_hole_from_outside():
    img = _plate()
    hole = _hole()
    img[hole] = (102, 255, 0)                      # sentinel in the hole
    out = edge_extend(img, hole)
    assert out.dtype == np.uint8
    # sentinel gone: nothing in the hole is chroma green any more
    filled = out[hole].astype(int)
    assert not np.any((filled[:, 1] > 200) & (filled[:, 0] < 150))
    # untouched outside
    assert np.array_equal(out[~hole], img[~hole])


# ---------------------------------------------------------------- G2 gate

def test_g2_fails_for_a_smear_and_passes_for_real_content():
    guide = _plate()
    hole = _hole()
    smear = edge_extend(guide, hole)
    s = score_fill(smear, guide, hole)
    assert not s["g2_pass"]                        # a no-op must fail G2
    rng = np.random.default_rng(1)
    invented = smear.copy()
    invented[hole] = rng.integers(0, 255, size=(int(hole.sum()), 3))
    s2 = score_fill(invented, guide, hole)
    assert s2["g2_pass"]
    assert s2["mean_abs_vs_edge_extend"] > G2_MIN_EDGE_EXTEND_DIFF


# --------------------------------------------------------------- G3a gate

def test_g3a_passes_for_a_seamless_fill_and_fails_for_a_hard_seam():
    guide = _plate()
    hole = _hole()
    seamless = edge_extend(guide, hole)            # continues the plate
    s = score_fill(seamless, guide, hole)
    assert s["g3a_pass"]
    assert s["rim_gradient_ratio"] < G3A_MAX_RIM_GRADIENT_RATIO
    seamy = guide.copy()
    seamy[hole] = 250                               # hard bright patch
    s2 = score_fill(seamy, guide, hole)
    assert not s2["g3a_pass"]
    assert s2["rim_gradient_ratio"] > s["rim_gradient_ratio"]


# --------------------------------------------------------------- G3b gate

def test_g3b_detects_a_green_cast_and_accepts_a_neutral_fill():
    guide = _plate()
    hole = _hole()
    neutral = edge_extend(guide, hole)
    assert score_fill(neutral, guide, hole)["g3b_pass"]
    cast = neutral.copy()
    px = cast[hole].astype(int)
    px[:, 1] = np.clip(px[:, 1] + 25, 0, 255)      # +25 green excess
    cast[hole] = px.astype(np.uint8)
    s = score_fill(cast, guide, hole)
    assert not s["g3b_pass"]
    assert s["green_excess_delta"] > 20


# --------------------------------------------------------- unmasked delta

def test_unmasked_delta_is_zero_when_context_is_untouched():
    guide = _plate()
    hole = _hole()
    fill = guide.copy()
    fill[hole] = 200
    assert unmasked_delta(fill, guide, hole) == 0.0


def test_unmasked_delta_sees_a_global_retone():
    guide = _plate()
    hole = _hole()
    retoned = np.clip(guide.astype(int) + 8, 0, 255).astype(np.uint8)
    d = unmasked_delta(retoned, guide, hole)
    assert 0.02 < d < 0.05                          # ~8/255 everywhere
    s = score_fill(retoned, guide, hole)
    assert s["unmasked_delta"] == pytest.approx(d)


# ------------------------------------------------------------- edge cases

def test_score_fill_rejects_mismatched_rasters_and_empty_holes():
    guide = _plate()
    with pytest.raises(ValueError, match="rasters differ"):
        score_fill(guide[:48], guide, _hole())
    with pytest.raises(ValueError, match="non-empty hole"):
        score_fill(guide, guide, np.zeros((96, 96), bool))


def test_score_fill_accepts_float_images():
    guide = _plate().astype(np.float32) / 255.0
    hole = _hole()
    fill = guide.copy()
    fill[hole] = 0.7
    s = score_fill(fill, guide, hole)
    assert s["unmasked_delta"] == 0.0
    assert np.isfinite(s["rim_gradient_ratio"])


# ------------------------------------------------------ depth guide encode

def test_depth_encode_spreads_structures_over_the_range():
    """The 2026-08-13 failure: full-range normalisation crushed the subject to
    0.07 brightness. Non-ground percentiles must spread the STRUCTURES."""
    depth = np.full((64, 64), 4.0)                  # near ground plane
    depth[:, 32:] = 40.0                            # far structure
    ground = np.zeros((64, 64), bool)
    ground[:, :32] = True
    full = encode_depth_guide(depth)
    scoped = encode_depth_guide(depth, ground_mask=ground)
    # full-range: the far structure sits at the very bottom of the range;
    # ground-scoped: the structure defines the range and is no longer crushed
    assert scoped[:, 32:, 0].mean() >= full[:, 32:, 0].mean()
    assert scoped.dtype == np.float32 and scoped.shape == (64, 64, 3)
    assert 0.0 <= scoped.min() and scoped.max() <= 1.0


def test_depth_encode_maps_holes_to_far():
    depth = np.full((32, 32), 10.0)
    depth[8:16, 8:16] = np.inf                      # nothing rasterized
    out = encode_depth_guide(depth)
    assert np.all(out[8:16, 8:16] == 0.0)


def test_depth_encode_rejects_empty_and_bad_shapes():
    with pytest.raises(ValueError, match="no finite samples"):
        encode_depth_guide(np.full((8, 8), np.inf))
    with pytest.raises(ValueError, match="HxW"):
        encode_depth_guide(np.zeros((8, 8, 3)))
    with pytest.raises(ValueError, match="does not match"):
        encode_depth_guide(np.ones((8, 8)), ground_mask=np.zeros((4, 4), bool))
