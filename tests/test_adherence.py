"""core.adherence: every assertion has a KNOWN answer, by construction.

The scorer's whole job is to be un-foolable, so the tests are adversarial
rather than illustrative: a graded clip must still score, a frozen clip must be
REFUSED, noise dumped in GHOST must not move the number, and a clip that starts
on the camera then stops must show negative drift. A golden-value pin would
pass while any of those broke.

numpy only: no torch, no ComfyUI.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.adherence import (
    ATLAS_ARM, CONTROL_ARM, DegenerateArmError, align_rasters, fit_grade,
    ghost_fill_coverage, gradient_zncc, masked_ssim, score_arms,
)
from atlas_camera.core.conditioning import render_conditioning_sequence
from atlas_camera.core.ghost_pixels import GhostClass

W, H = 128, 96
FX = FY = 160.0
CX, CY = W / 2.0, H / 2.0
K = [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]
WALL_Z = 10.0


def view_at(x: float):
    vm = np.eye(4)
    vm[0, 3] = -x
    return vm


def rect(z, x0, x1, y0, y1, label):
    verts = np.array([[x0, y0, -z], [x1, y0, -z], [x1, y1, -z], [x0, y1, -z]],
                     dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                   dtype=np.float64)
    return (label, verts, faces, uvs, "primary", {})


def quad(z, half_w, half_h, label):
    return rect(z, -half_w, half_w, -half_h, half_h, label)


def gapped_wall_scene(blocker_half=0.45, blocker_z=3.0):
    """A wall with a HOLE exactly where the blocker hides it, plus the blocker.

    This geometry is what makes GHOST possible at all. A solid wall behind an
    occluder is not a disocclusion test: moving the camera reveals real
    geometry, the rasteriser covers it, and every pixel is correctly VALID. A
    relief mesh built from a depth map has no surface behind its occluders --
    that is what a tear IS -- so the honest small-scene stand-in is a wall with
    the occluded span removed. Moving the camera then uncovers pixels that are
    inside the plate's frame and were NOT a hole in the plate (the blocker
    covered them), which is precisely the definition of GHOST.
    """
    gap = blocker_half * WALL_Z / blocker_z
    return [
        rect(WALL_Z, -40.0, -gap, -30.0, 30.0, "wall_left"),
        rect(WALL_Z, gap, 40.0, -30.0, 30.0, "wall_right"),
        quad(blocker_z, blocker_half, 0.6, "blocker"),
    ]


def checker(n=64, squares=8):
    """High-frequency texture on purpose: a gradient measure needs gradients,
    and a flat plate would let a wrong answer correlate perfectly with
    nothing."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (((xx * squares) // n + (yy * squares) // n) % 2).astype(np.float64)
    rgb = np.stack([c, c * 0.6 + 0.2, 1.0 - c], axis=-1) * 0.8 + 0.1
    return np.concatenate([rgb, np.ones((n, n, 1))], axis=-1)


@pytest.fixture
def bundle():
    """A wall with a nearer blocker, dollied laterally: produces real GHOST."""
    meshes = gapped_wall_scene()
    views = [view_at(0.0), view_at(0.25), view_at(0.5), view_at(0.75)]
    seq = render_conditioning_sequence(
        meshes, {"primary": checker()}, views=views, intrinsics=[K] * 4,
        plate_view=views[0], plate_k=K, width=W, height=H)
    assert (seq.class_map == int(GhostClass.GHOST)).any(), "fixture has no GHOST"
    return seq


def arms_from(bundle, atlas, control=None):
    """The control defaults to a rolled frame -- a plausible-looking clip that
    is simply not the requested camera, which is what a prompt-only arm is."""
    if control is None:
        control = np.stack([np.roll(f, 9, axis=1) for f in bundle.rgb])
    return {ATLAS_ARM: np.asarray(atlas, dtype=np.float64),
            CONTROL_ARM: np.asarray(control, dtype=np.float64)}


# --------------------------------------------------------------------------
# measures
# --------------------------------------------------------------------------

def test_identical_frames_score_exactly_one(bundle):
    m = bundle.class_map[1] == int(GhostClass.VALID)
    assert gradient_zncc(np, bundle.rgb[1], bundle.rgb[1], m) == pytest.approx(1.0)


def test_a_graded_clip_still_scores_but_raw_error_does_not(bundle):
    """THE measure-choice test. A gain+offset is exactly what a generator's
    tone shift looks like; the headline must survive it while MAE does not."""
    from atlas_camera.core.adherence import raw_error
    ref = bundle.rgb[1]
    graded = np.clip(ref * 0.55 + 0.22, 0.0, 1.0)
    m = bundle.class_map[1] == int(GhostClass.VALID)
    assert gradient_zncc(np, graded, ref, m) == pytest.approx(1.0, abs=1e-6)
    assert raw_error(np, graded, ref, m)["mae"] > 0.05


def test_the_fitted_grade_recovers_the_gain_and_offset(bundle):
    ref = bundle.rgb[1]
    graded = np.clip(ref * 0.55 + 0.22, 0.0, 1.0)
    m = bundle.class_map[1] == int(GhostClass.VALID)
    grade = fit_grade(np, graded, ref, m)
    assert np.allclose(grade["gain"], 0.55, atol=1e-6)
    assert np.allclose(grade["offset"], 0.22, atol=1e-6)
    assert grade["residual"] < 1e-6


def test_an_uncorrelated_frame_scores_far_below_one(bundle):
    rng = np.random.default_rng(3)
    noise = rng.random(bundle.rgb[1].shape)
    m = bundle.class_map[1] == int(GhostClass.VALID)
    assert gradient_zncc(np, noise, bundle.rgb[1], m) < 0.2


def test_a_flat_region_returns_nan_not_zero(bundle):
    """A region with no structure is a case with NO answer. Zero is a real
    answer meaning uncorrelated, so the two must not be conflated."""
    flat = np.full_like(bundle.rgb[1], 0.5)
    m = bundle.class_map[1] == int(GhostClass.VALID)
    assert np.isnan(gradient_zncc(np, flat, flat, m))


def test_masked_ssim_is_one_for_identical_and_lower_otherwise(bundle):
    m = bundle.class_map[1] == int(GhostClass.VALID)
    same = masked_ssim(np, bundle.rgb[1], bundle.rgb[1], m)
    rolled = masked_ssim(np, np.roll(bundle.rgb[1], 5, axis=1),
                         bundle.rgb[1], m)
    assert same == pytest.approx(1.0, abs=1e-6)
    assert rolled < same


# --------------------------------------------------------------------------
# ghost region
# --------------------------------------------------------------------------

def test_a_flat_grey_fill_scores_no_ghost_coverage(bundle):
    frame = bundle.rgb[2].copy()
    ghost = bundle.class_map[2] == int(GhostClass.GHOST)
    valid = bundle.class_map[2] == int(GhostClass.VALID)
    frame[ghost] = 0.5
    cov = ghost_fill_coverage(np, frame, valid, ghost)
    assert cov["coverage"] == pytest.approx(0.0, abs=0.05)


def test_textured_ghost_content_scores_coverage(bundle):
    rng = np.random.default_rng(11)
    frame = bundle.rgb[2].copy()
    ghost = bundle.class_map[2] == int(GhostClass.GHOST)
    valid = bundle.class_map[2] == int(GhostClass.VALID)
    frame[ghost] = rng.random((int(ghost.sum()), 3))
    assert ghost_fill_coverage(np, frame, valid, ghost)["coverage"] > 0.5


def test_noise_in_ghost_cannot_change_adherence(bundle):
    """Proves adherence is scored in VALID ONLY. If a generator's invented
    pixels moved the camera score, the metric would be measuring the fill."""
    rng = np.random.default_rng(5)
    clean = bundle.rgb.copy()
    dirty = bundle.rgb.copy()
    for i in range(bundle.frames):
        g = bundle.class_map[i] == int(GhostClass.GHOST)
        dirty[i][g] = rng.random((int(g.sum()), 3))
    a = score_arms(arms_from(bundle, clean), bundle=bundle)
    b = score_arms(arms_from(bundle, dirty), bundle=bundle)
    assert a["adherence"] == pytest.approx(b["adherence"], abs=1e-9)


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------

def test_a_frozen_clip_is_refused_despite_scoring_well(bundle):
    """THE test the whole design turns on. A repeat of frame 0 agrees with
    Atlas's reprojection wherever the move is small, so it scores HIGH -- and
    must still be refused, because it responded to no camera at all."""
    frozen = np.repeat(bundle.rgb[:1], bundle.frames, axis=0)
    naive = score_arms(arms_from(bundle, frozen), bundle=bundle,
                       refuse_degenerate=False)
    assert naive["adherence"] > 0.5, "the cheat really does score well"
    assert naive["guard"]["parallax_response"] <= 0.0
    assert naive["guard"]["passed"] is False
    with pytest.raises(DegenerateArmError, match="parallax_response"):
        score_arms(arms_from(bundle, frozen), bundle=bundle)


def test_an_honest_clip_passes_the_guard(bundle):
    result = score_arms(arms_from(bundle, bundle.rgb), bundle=bundle)
    assert result["guard"]["passed"] is True
    assert result["guard"]["parallax_response"] > 0.0
    assert result["adherence"] == pytest.approx(1.0, abs=1e-6)


def test_the_static_arm_is_synthesised_and_bounds_the_headline(bundle):
    """A caller cannot forget the cheat's ceiling, because the scorer builds
    it. The margin over static is what the number actually buys."""
    result = score_arms(arms_from(bundle, bundle.rgb), bundle=bundle)
    assert "static" in result["arms"]
    assert result["margin_over_static"] > 0.0
    assert result["guard"]["static_arm_adherence"] is not None


def test_a_zero_parallax_move_is_refused(bundle):
    """A static camera reveals nothing, so there is nothing to get right. The
    scorer must refuse rather than return a confident 1.0."""
    meshes = [quad(WALL_Z, 40.0, 30.0, "wall")]
    still = render_conditioning_sequence(
        meshes, {"primary": checker()}, views=[view_at(0.0)] * 3,
        intrinsics=[K] * 3, plate_view=view_at(0.0), plate_k=K,
        width=W, height=H)
    with pytest.raises(DegenerateArmError):
        score_arms(arms_from(still, still.rgb), bundle=still)


# --------------------------------------------------------------------------
# curves and the control arm
# --------------------------------------------------------------------------

def test_a_clip_that_stops_following_shows_negative_drift(bundle):
    """Match the camera, then freeze. The headline decays with frame index,
    which is the signature of a generator losing the camera."""
    frames = bundle.rgb.copy()
    frames[2:] = bundle.rgb[1]
    result = score_arms(arms_from(bundle, frames), bundle=bundle,
                        refuse_degenerate=False)
    assert result["arms"][ATLAS_ARM]["drift"]["slope"] < 0.0


def test_a_missing_control_arm_raises(bundle):
    with pytest.raises(ValueError, match="prompt_only"):
        score_arms({ATLAS_ARM: bundle.rgb}, bundle=bundle)


def test_a_missing_atlas_arm_raises(bundle):
    with pytest.raises(ValueError, match="atlas"):
        score_arms({CONTROL_ARM: bundle.rgb}, bundle=bundle)


def test_a_frame_count_mismatch_raises(bundle):
    with pytest.raises(ValueError, match="frames against the bundle"):
        score_arms(arms_from(bundle, bundle.rgb[:2]), bundle=bundle)


def test_the_comparison_reports_an_interval_and_a_sign_test(bundle):
    """Two means with no interval is not evidence."""
    result = score_arms(arms_from(bundle, bundle.rgb), bundle=bundle)
    cmp = result["comparison_vs_control"]
    assert cmp["delta"] > 0.0
    assert cmp["ci95"][0] <= cmp["delta"] <= cmp["ci95"][1]
    assert cmp["sign_test"]["n"] >= 1
    assert result["margin_over_control"] > 0.0


def test_the_bootstrap_is_seeded_and_reproducible(bundle):
    a = score_arms(arms_from(bundle, bundle.rgb), bundle=bundle,
                   bootstrap_seed=42)
    b = score_arms(arms_from(bundle, bundle.rgb), bundle=bundle,
                   bootstrap_seed=42)
    assert (a["comparison_vs_control"]["ci95"]
            == b["comparison_vs_control"]["ci95"])


# --------------------------------------------------------------------------
# raster alignment
# --------------------------------------------------------------------------

def test_rasters_align_to_the_smaller_and_the_class_map_is_eroded(bundle):
    """Upsampling a generated frame would invent detail the model never made.
    And the class map must be eroded: an un-eroded nearest remap bleeds GHOST
    into VALID, which would score invented pixels as known ones."""
    half = bundle.rgb[:, ::2, ::2, :]
    gen, ref, valid, ghost, meta = align_rasters(
        np, half, bundle.class_map, bundle.rgb)
    assert meta["raster"] == [W // 2, H // 2]
    assert meta["resampled"] is True
    assert gen.shape == ref.shape
    assert valid.shape[1:] == (H // 2, W // 2)

    from atlas_camera.core.adherence import _nearest_resize
    raw = np.stack([_nearest_resize(np, c, H // 2, W // 2)
                    == int(GhostClass.VALID) for c in bundle.class_map])
    assert valid.sum() < raw.sum(), "erosion must actually remove boundary px"
    assert not (valid & ghost).any(), "the classes must stay disjoint"


def test_an_unrendered_bundle_refuses(bundle):
    from atlas_camera.core.conditioning import ConditioningSequence
    with pytest.raises(ValueError, match="no rendered frames"):
        score_arms(arms_from(bundle, bundle.rgb),
                   bundle=ConditioningSequence())
