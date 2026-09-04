"""`solve_scale_from_primary` fits on points the primary CANNOT see.

Aligning two views' depth requires MUTUALLY VISIBLE points -- that is the whole
premise of overlap registration, and AtlasAddPatchView's docstring says so:
"Registration exploits the OVERLAP both cameras see". The solver never tests
visibility. It accepts any sample that lands in frame with a finite primary
depth, which includes every point the patch shows THROUGH an occluder.

For those points `sampled` is the OCCLUDER's depth, not the surface's. The
occluder is nearer by construction -- that is what makes it an occluder -- so
each hidden sample solves for a scale smaller than the truth, and the median is
dragged down in proportion to how much of the patch is hole.

That matters because the same scaled geometry is then fed to
`primary_camera_validity_mask`'s depth-shadow test, which asks "is this point
farther than what the primary stored here?". A patch pulled too near answers no,
is judged visible, and gets matted out of its own fill. Measured live on the
sea-cliff castle 2026-09-04: every one of five patches solved a scale below 1
(0.27-0.62), the depth ratio inside the holes ran 0.46-0.85 median with ROI 5 at
100% under 1.0 in a 0.435-0.496 band, and stripping the matte recovered 45% of
the residual.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.patch_registration import solve_scale_from_primary  # noqa: E402

W = H = 64
FX = FY = 60.0
CX = CY = W / 2.0
FAR_Z = -20.0          # the surface the patch depicts
OCCLUDER_Z = -5.0      # what hides it from the primary
TRUE_SCALE = 2.0       # the patch's relative depth is truth / TRUE_SCALE


def _cam(view, pos):
    return (dict(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                 width=W, height=H), np.asarray(pos, dtype=np.float64))


def _view_at(x):
    """World->camera for a camera at (x, 0, 0) looking down -Z."""
    v = np.eye(4)
    v[0, 3] = -x
    return v


def _scene():
    """Primary at origin, patch shifted right; a slab hides the middle of the
    far plane from the PRIMARY only."""
    primary, _ppos = _cam(_view_at(0.0), (0.0, 0.0, 0.0))
    patch, patch_pos = _cam(_view_at(6.0), (6.0, 0.0, 0.0))

    # The primary's metric depth: the far plane, except where the occluder is.
    xs = np.arange(W) + 0.5
    hidden_cols = (xs > CX - 10) & (xs < CX + 10)
    primary_metric = np.full((H, W), abs(FAR_Z), dtype=np.float64)
    primary_metric[:, hidden_cols] = abs(OCCLUDER_Z)

    # The patch sees the far plane everywhere. Its own depth is up to scale.
    ys, xs2 = np.mgrid[0:H, 0:W]
    Xc = (xs2 + 0.5 - CX) / FX
    Yc = -(ys + 0.5 - CY) / FY
    # forward depth from the patch camera to the plane z = FAR_Z
    depth_true = np.full((H, W), abs(FAR_Z), dtype=np.float64)
    patch_depth = depth_true / TRUE_SCALE
    return primary, patch, patch_pos, primary_metric, patch_depth, hidden_cols


#: Parallax between the two cameras at the patch's own (unscaled) depth. The
#: occluder is placed in PRIMARY pixels -- that is where an occluder lives --
#: and the exclude mask is then derived by PROJECTING each patch sample and
#: asking where it landed. Indexing one frame's mask in the other's is exactly
#: the mistake this file is about, and it bit while writing it.
PATCH_X = 6.0
SHIFT_PX = FX * PATCH_X / (abs(FAR_Z) / TRUE_SCALE)


def _solve(hidden_half_width, patch, patch_pos, primary, patch_depth,
           exclude=False):
    xs = np.arange(W) + 0.5
    hidden_primary = (xs > CX - hidden_half_width) & (xs < CX + hidden_half_width)
    pm = np.full((H, W), abs(FAR_Z), dtype=np.float64)
    pm[:, hidden_primary] = abs(OCCLUDER_Z)

    # Where each PATCH column lands in the primary, and whether that is occluded
    # or off the edge of the primary's frame (dropped by the solver either way).
    proj = np.round(xs + SHIFT_PX).astype(int)
    in_frame = (proj >= 0) & (proj < W)
    occluded = np.zeros(W, bool)
    occluded[in_frame] = hidden_primary[proj[in_frame]]

    mask = None
    if exclude:
        mask = np.zeros((H, W), bool)
        mask[:, occluded] = True

    s, info = solve_scale_from_primary(
        patch_depth, patch_camera=patch, patch_camera_position=patch_pos,
        primary_metric_map=pm, primary_camera=primary, exclude_mask=mask,
        min_samples=10)
    frac = float(occluded[in_frame].mean()) if in_frame.any() else 0.0
    return s, frac, info


def test_the_median_holds_until_the_hidden_samples_are_the_MAJORITY():
    """The solver takes a median, so it is exact while occluded samples are a
    minority -- and then fails completely, not gradually.

    Measured on this fixture: 1.00x the true scale at every hidden fraction up
    to and including 50%, then a cliff to 0.25x from 62.5% onward. That is the
    median's breakdown point behaving exactly as it should; the bug is that
    nothing stops the fit crossing it.
    """
    primary, patch, patch_pos, _pm, patch_depth, _h = _scene()

    minority = [_solve(hw, patch, patch_pos, primary, patch_depth)
                for hw in (4, 12, 16)]
    majority = [_solve(hw, patch, patch_pos, primary, patch_depth)
                for hw in (20, 24, 31)]

    for s, frac, _i in minority:
        assert frac <= 0.5
        assert s == pytest.approx(TRUE_SCALE, rel=0.02), (
            f"{frac:.0%} hidden should not move a median: got {s}")
    for s, frac, _i in majority:
        assert frac > 0.5
        assert s < TRUE_SCALE * 0.5, (
            f"{frac:.0%} hidden must break it: got {s}")


def test_excluding_the_hidden_region_recovers_the_true_scale():
    """The fix, expressed as a measurement: restrict the fit to mutually
    visible points and the same solver lands on the truth even at a hidden
    fraction that otherwise destroys it."""
    primary, patch, patch_pos, _pm, patch_depth, _h = _scene()

    broken, frac, _i = _solve(24, patch, patch_pos, primary, patch_depth)
    honest, frac2, info = _solve(24, patch, patch_pos, primary, patch_depth,
                                 exclude=True)

    assert frac == frac2 > 0.5
    assert broken < TRUE_SCALE * 0.5
    assert info["accepted"]
    assert honest == pytest.approx(TRUE_SCALE, rel=0.02), (
        f"visible-only fit should recover {TRUE_SCALE}, got {honest:.4f}")


def test_a_patch_that_is_mostly_hole_is_where_this_bites():
    """Why it matters for AtlasFillOccluded specifically: it crops TO a hole,
    so its patches are majority-hole by construction. The five live castle
    ROIs measured 31.5% / 45.8% / 54.0% / 63.8% / 67.7% hole -- three of them
    past the breakdown point, and the two lowest solved scales (0.2654,
    0.3763) were the 67.7% and 54.0% ones."""
    primary, patch, patch_pos, _pm, patch_depth, _h = _scene()

    below, f_below, _a = _solve(14, patch, patch_pos, primary, patch_depth)
    above, f_above, _b = _solve(22, patch, patch_pos, primary, patch_depth)

    assert f_below < 0.5 < f_above
    assert below > above * 2.0, (
        "crossing the breakdown point must be visible as a step, not noise: "
        f"{below:.3f} -> {above:.3f}")
