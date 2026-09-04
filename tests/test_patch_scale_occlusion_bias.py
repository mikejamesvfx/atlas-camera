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


# ------------------------------------------- the caller must supply the mask

def test_the_patch_node_excludes_its_own_hole_from_the_scale_fit(monkeypatch):
    """The fix, at the call site.

    The solver cannot discover which samples are occluded on its own -- deciding
    that from depth needs the scale, which is what is being solved. The caller
    HAS the answer independently: `patch_hole` is the region the patch was
    generated to fill, which is by construction what the primary cannot see.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from types import SimpleNamespace

    import atlas_camera.comfy.nodes_geometry as ng
    from atlas_camera.comfy.nodes import AtlasAddPatchView

    from test_add_patch_view import _patch_estimate_depth, _synthetic_primary

    _patch_estimate_depth(monkeypatch)
    seen = {}

    real = ng.solve_scale_from_primary

    def spy(*a, **kw):
        seen["exclude_mask"] = kw.get("exclude_mask")
        return real(*a, **kw)

    monkeypatch.setattr(ng, "solve_scale_from_primary", spy)

    solve, _p, _e = _synthetic_primary()
    ramp = np.linspace(30.0, 5.0, 512)[:, None] * np.ones((1, 512))
    depth = SimpleNamespace(depth=ramp.astype(np.float32), is_metric=True,
                            image_width=512, image_height=512, metadata={})
    hole = np.zeros((512, 512), np.float32)
    hole[100:400, 100:400] = 1.0

    AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        geometry_source="own_depth", relief_grid=48, primary_depth=depth,
        patch_mask=torch.from_numpy(hole))

    mask = seen.get("exclude_mask")
    assert mask is not None, "the hole must reach the scale fit"
    mask = np.asarray(mask, dtype=bool)
    assert mask[250, 250], "inside the hole must be excluded from the fit"
    assert not mask[10, 10], "outside the hole must still be sampled"


def test_a_patch_with_no_hole_mask_is_unchanged(monkeypatch):
    """A hand-placed novel view carries no patch_mask, so its registration must
    behave exactly as before -- this solver is shared with AtlasOcclusionMask
    and the artist path."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from types import SimpleNamespace

    import atlas_camera.comfy.nodes_geometry as ng
    from atlas_camera.comfy.nodes import AtlasAddPatchView

    from test_add_patch_view import _patch_estimate_depth, _synthetic_primary

    _patch_estimate_depth(monkeypatch)
    seen = {}
    real = ng.solve_scale_from_primary

    def spy(*a, **kw):
        seen["exclude_mask"] = kw.get("exclude_mask")
        return real(*a, **kw)

    monkeypatch.setattr(ng, "solve_scale_from_primary", spy)

    solve, _p, _e = _synthetic_primary()
    ramp = np.linspace(30.0, 5.0, 512)[:, None] * np.ones((1, 512))
    depth = SimpleNamespace(depth=ramp.astype(np.float32), is_metric=True,
                            image_width=512, image_height=512, metadata={})

    AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        geometry_source="own_depth", relief_grid=48, primary_depth=depth)

    assert seen.get("exclude_mask") is None


def test_the_solver_reports_how_well_its_samples_agreed():
    """A median hides its own disagreement.

    Two adjacent ground patches on the castle -- same 192x64 raster, same row,
    ~5 m unscaled depth both -- fitted 0.645 and 0.273. Nothing said the second
    was worse conditioned, because only the median came back, and the bad one
    put its ground 0.78 m in the air (the camera is at y=1.6, and scaling depth
    by k about it lands a ground point at 1.6*(1-k); 0.783 solves k=0.51,
    matching its 0.471 depth ratio).
    """
    primary, patch, patch_pos, _pm, patch_depth, _h = _scene()

    clean, _f, info = _solve(24, patch, patch_pos, primary, patch_depth,
                             exclude=True)
    assert info["scale_rel_iqr"] == pytest.approx(0.0, abs=1e-6), (
        "a fixture where every visible sample agrees must report no spread")
    assert info["scale_p25"] == pytest.approx(info["scale_p75"], rel=1e-6)

    # Contaminate the visible region and the spread has to show it, even though
    # the median may still land near the truth.
    xs = np.arange(W) + 0.5
    pm = np.full((H, W), abs(FAR_Z), dtype=np.float64)
    noisy_cols = (np.arange(W) % 3) == 0
    pm[:, noisy_cols] = abs(FAR_Z) * 0.4
    s2, info2 = solve_scale_from_primary(
        patch_depth, patch_camera=patch, patch_camera_position=patch_pos,
        primary_metric_map=pm, primary_camera=primary, min_samples=10)
    assert info2["scale_rel_iqr"] > 0.05, (
        f"disagreeing samples must raise the quartile spread: {info2}")
    # ...and rel_mad does NOT, which is why the IQR is the one to read: a
    # median of medians inherits the same 50% breakdown as the median it
    # describes.
    assert info2["scale_rel_mad"] == pytest.approx(0.0, abs=1e-6)


def test_the_patch_records_the_fit_quality(monkeypatch):
    """...and it has to reach the solve, or diagnosing the next one needs a
    source read and a rebuild again."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from types import SimpleNamespace

    from atlas_camera.comfy.nodes import AtlasAddPatchView

    from test_add_patch_view import _patch_estimate_depth, _synthetic_primary

    _patch_estimate_depth(monkeypatch)
    solve, _p, _e = _synthetic_primary()
    ramp = np.linspace(30.0, 5.0, 512)[:, None] * np.ones((1, 512))
    depth = SimpleNamespace(depth=ramp.astype(np.float32), is_metric=True,
                            image_width=512, image_height=512, metadata={})

    out, _r = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        geometry_source="own_depth", relief_grid=48, primary_depth=depth,
        # This fixture's fit is genuinely ill-conditioned (a random image over
        # a linear ramp), so the armed gate refuses it -- correctly. The gate
        # is pinned in test_add_patch_view_registration; what is under test
        # here is that the diagnostics get RECORDED, so read them off an
        # accepted fit.
        scale_max_rel_iqr=0.0)

    meta = out.projection_sources[-1].metadata
    assert meta.get("scale_source", "").startswith("primary_registration")
    for key in ("scale_n_samples", "scale_p25", "scale_p75", "scale_rel_mad",
                "scale_rel_iqr"):
        assert key in meta, key


def test_the_spread_separates_an_exact_fit_from_a_broken_one_and_where():
    """The number behind AtlasAddPatchView's scale_max_rel_iqr default of 1.0.

    Swept over this fixture's known truth, the quartile spread reads 0.000 up
    to 19% occluded, 0.750 from 26% to 41% while the median is still EXACT, and
    1.200 at 48% where the scale first goes wrong (37.5% out). So the gate has
    to sit above 0.75 and no higher than 1.2 -- 1.0 is the middle of the only
    interval that refuses nothing good and catches the first thing bad.
    """
    primary, patch, patch_pos, _pm, patch_depth, _h = _scene()
    exact = [_solve(hw, patch, patch_pos, primary, patch_depth)
             for hw in (12, 14, 16)]
    for s, _f, info in exact:
        assert s == pytest.approx(TRUE_SCALE)
        assert info["scale_rel_iqr"] <= 0.75

    broken, frac, info = _solve(18, patch, patch_pos, primary, patch_depth)
    assert abs(broken / TRUE_SCALE - 1.0) > 0.3, (broken, frac)
    assert info["scale_rel_iqr"] >= 1.2


def test_the_spread_goes_SILENT_once_the_occluded_samples_win_outright():
    """The gate's blind spot, pinned so nobody mistakes it for coverage.

    Past ~78% occluded the spread falls back to 0.000 while the scale is 75%
    wrong: one population has won outright and agrees with itself perfectly, so
    dispersion has nothing left to see. Catching that is the exclude mask's job
    -- the caller hands the fit the hole to drop -- and min_samples'. Raising
    the gate's sensitivity cannot reach it, and lowering the threshold to try
    would only start refusing exact fits.
    """
    primary, patch, patch_pos, _pm, patch_depth, _h = _scene()
    s, frac, info = _solve(28, patch, patch_pos, primary, patch_depth)
    assert frac > 0.75
    assert abs(s / TRUE_SCALE - 1.0) > 0.5      # badly wrong
    assert info["scale_rel_iqr"] == pytest.approx(0.0)   # and says nothing
