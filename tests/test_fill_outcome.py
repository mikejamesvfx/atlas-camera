"""Does a registered fill actually CLOSE the hole? Nothing else asserts this.

Every other fill test asserts graph SHAPE -- which node is wired to which
socket. That is worth having and it caught nothing this feature got wrong. The
things that went wrong were outcomes: a patch mesh that tore away 40% of itself
and left the interior of its own ROI open, an 11 m stretched triangle shipped as
DCC geometry, and a headline metric that counted sky and read 72% success as
29%. All three were found by hand, and every number that settled them lives in a
commit message where nothing can regress against it.

So this scores the geometry half of the loop, deterministically and with no GPU:
a synthetic occluder scene, a move that opens a real disocclusion, one patch
built through the ordinary AtlasAddPatchView path, and the hole measured before
and after. The fill CONTENT is the model's business; the geometry is ours and is
what regresses in silence.

Two rules this file exists to enforce, both learned the hard way:

* Score against the FILLABLE hole, never the raw coverage figure the nodes
  report. `peak hole` comes off the render before `survey_hole_rois` subtracts
  the exclude mask and the move-revealed test, so it counts sky and off-plate --
  measured on a real plate, 64.8% of the hole before a fill and 86.1% after. A
  test scored on it would be mostly measuring sky.
* A hole closed by a metres-long shard is not closed. Coverage and the worst
  triangle have to be asserted together, or "improve coverage" licenses exactly
  the geometry the shard bound exists to stop.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import AtlasAddPatchView  # noqa: E402
from atlas_camera.comfy.nodes_fill import (AtlasCameraMovePreset,  # noqa: E402
                                           AtlasCropROI)

from test_fill_nodes import _img, _wall_solve_with_occluder  # noqa: E402

RESOLUTION = 256          # the scene is three quads; this is plenty and fast
WALL_DEPTH_M = 10.0       # the wall sits at z = -10, camera near z = 0


def _scene_solve():
    """The shared occluder fixture, with the wall lifted ABOVE the ground.

    `build_relief_mesh` clamps a patch mesh at `floor_clamp=-0.25`, and the
    shared fixture's wall spans y -2..6 — so its lower half is unreachable by
    any patch and the test would be scoring the clamp, not the fill. Lifting
    the geometry is the fixture's business, not a reason to relax the clamp.
    """
    from atlas_camera.core.schema import AtlasProxyPrimitive

    def quad(name, x0, x1, z):
        return AtlasProxyPrimitive(
            name=name, primitive_type="mesh",
            metadata={"vertices": [x0, 0.2, z, x1, 0.2, z,
                                   x1, 6.0, z, x0, 6.0, z],
                      "faces": [0, 1, 2, 0, 2, 3],
                      "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]})

    solve = _wall_solve_with_occluder()
    solve.projection_scene.proxy_geometry = [
        quad("wall_left", -6.0, -2.0, -10.0),
        quad("wall_right", 2.0, 6.0, -10.0),
        quad("pillar", -0.8, 0.8, -4.0),
    ]
    return solve


def _no_sky_heuristic(shape=(256, 256)):
    """An all-False exclude mask.

    `AtlasAddPatchView` keys `apply_sky_heuristic` on whether an exclude mask
    was given, so passing an empty one turns the heuristic OFF without
    excluding anything. That heuristic invalidates vertices by horizon row and
    is a SEPARATE mechanism with its own pinned test; leaving it on here would
    tear half the patch and this file would be measuring it instead.
    """
    return torch.zeros(1, *shape, dtype=torch.float32)


def _end_view(solve, angle_deg=35.0):
    from atlas_camera.core.camera_path import sample_camera_path

    path, _exact, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                     angle_deg=angle_deg)
    return sample_camera_path(path)[-1].camera_view_matrix, path


def _fillable_hole(solve, plate, view, *, base_solve):
    """(fillable_mask, raw_mask) at the survey raster.

    FILLABLE = uncovered AND not sky/off-plate, applying the same
    move-revealed test `survey_hole_rois` uses to decide what may be targeted.
    `base_solve` supplies the plate-hole survey, so the sky/off-plate verdict is
    the scene's own and does not move when a patch is added.
    """
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.dynamic.occlusion_fill import (not_disocclusion_mask,
                                                     plate_hole_survey,
                                                     render_disocclusion_sequence)

    _guide, mask, _cov = render_disocclusion_sequence(
        solve, plate, [view], resolution=RESOLUTION, hole_dilate_px=0)[0]
    raw = mask > 127
    h, w = raw.shape

    intr = base_solve.camera.intrinsics
    spec = CameraSpec.from_intrinsics(intr)
    s = RESOLUTION / max(int(intr.image_width), int(intr.image_height))
    drop = not_disocclusion_mask(
        plate_hole_survey(base_solve, plate, resolution=RESOLUTION),
        view=view, fx=spec.fx * s, fy=spec.fy * s, cx=spec.cx * s,
        cy=spec.cy * s, width=w, height=h)
    return (raw & ~drop), raw


def _wall_depth_map(solve, crop, gen_w, gen_h, view, plane_z=-10.0):
    """The depth a CORRECT fill would have: the wall, seen from the crop camera.

    A constant depth is not a stand-in for this. It is a fronto-parallel sheet
    that lands wherever the ground fit and the floor clamp put it, so the test
    ends up scoring those instead of the fill. Ray-casting the fixture's own
    wall plane gives the honest baseline -- "the model guessed the geometry
    right" -- and any failure after that is the node's.

    Exact rather than approximate because the patch camera is knowable: with
    `exact_view_override` wired (which is how AtlasFillOccluded wires it) the
    patch camera IS the move's end camera, so the same `view` that measures the
    hole also casts the rays.
    """
    from atlas_camera.core.camera_crop import (RegionROI, crop_intrinsics,
                                               scale_intrinsics)
    from atlas_camera.core.camera_spec import CameraSpec

    roi = RegionROI(x=crop["x"], y=crop["y"],
                    width=crop["width"], height=crop["height"])
    ci = scale_intrinsics(crop_intrinsics(solve.camera.intrinsics, roi),
                          int(gen_w), int(gen_h))
    spec = CameraSpec.from_intrinsics(ci)

    V = np.asarray(view, dtype=np.float64)
    world = np.linalg.inv(V)
    R_cw, eye = world[:3, :3], world[:3, 3]

    u = np.arange(int(gen_w), dtype=np.float64) + 0.5
    v = np.arange(int(gen_h), dtype=np.float64) + 0.5
    X = (u[None, :] - spec.cx) / spec.fx
    Y = -(v[:, None] - spec.cy) / spec.fy
    cam = np.stack([np.broadcast_to(X, (len(v), len(u))),
                    np.broadcast_to(Y, (len(v), len(u))),
                    -np.ones((len(v), len(u)))], axis=-1)
    dirs = cam @ R_cw.T                       # camera -> world directions

    with np.errstate(divide="ignore", invalid="ignore"):
        d = (plane_z - eye[2]) / dirs[..., 2]
    # Behind the camera or parallel to the plane: park it at the wall distance
    # rather than emitting a NaN the mesher would tear around.
    d = np.where(np.isfinite(d) & (d > 0.1), d, abs(plane_z - eye[2]))
    return d.astype(np.float32)


def _mock_depth(monkeypatch, depth):
    from dataclasses import dataclass

    @dataclass
    class _D:
        depth: object
        is_metric: bool = True
        model_id: str = "fake"

    def fake(image_path, *, model_id=None, device=None, focal_px=None):
        return _D(depth=depth)

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)


def _patched(solve, plate_t, view, monkeypatch, *, depth=None, **patch_kw):
    """Crop the largest move-revealed cluster and register it as a patch, the
    way AtlasFillOccluded's expansion does."""
    _v, path = _end_view(solve)
    from atlas_camera.comfy.view_prompts import _format_exact_view  # noqa: F401

    guide, mask, gw, gh, crop, report, nw, nh = AtlasCropROI().crop(
        solve, plate_t, camera_path=path, roi_slot=1,
        roi_source="auto_largest", min_area_px=16, snap=16, pad_frac=0.10)
    assert not crop.get("empty"), f"fixture opened no cluster: {report}"

    _mock_depth(monkeypatch,
                _wall_depth_map(solve, crop, gw, gh, view)
                if depth is None else depth(solve, crop, gw, gh, view))

    _path, exact, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                     angle_deg=35.0)
    out, _r2 = AtlasAddPatchView().add_patch(
        solve, guide, crop=crop, patch_mask=mask, geometry_source="own_depth",
        mask_unseen_only=False, relief_grid=48, name="fill_test_roi1",
        exact_view_override=exact,
        exclude_mask=_no_sky_heuristic((int(gh), int(gw))), **patch_kw)
    return out, crop


def _worst_edge_m(solve):
    from atlas_camera.core.projection_render import gather_scene_meshes

    worst = 0.0
    for label, verts, tris, uvs, _tex, _meta in gather_scene_meshes(
            solve, with_uvs=True):
        if not label.startswith("fill_test"):
            continue
        P = np.asarray(verts, dtype=np.float64)[np.asarray(tris, dtype=np.int64)]
        for a, b in ((0, 1), (1, 2), (0, 2)):
            worst = max(worst, float(
                np.linalg.norm(P[:, b] - P[:, a], axis=-1).max()))
    return worst


@pytest.fixture
def scene():
    solve = _scene_solve()
    plate = np.full((96, 128, 3), 120, np.uint8)
    view, _path = _end_view(solve)
    return solve, plate, _img(plate), view


def test_the_fixture_opens_a_real_disocclusion(scene):
    """Guard the guard: if the move stops opening a hole, every assertion below
    passes by measuring nothing."""
    solve, plate, _plate_t, view = scene

    fillable, raw = _fillable_hole(solve, plate, view, base_solve=solve)

    assert int(fillable.sum()) > 200, "no fillable disocclusion to close"
    # And the metric is doing its job: some of the raw hole is NOT fillable
    # (the surround the camera never looked at), which is the distinction the
    # whole file rests on.
    assert int(raw.sum()) > int(fillable.sum())


def test_a_registered_patch_closes_the_hole_it_was_built_for(scene, monkeypatch):
    solve, plate, plate_t, view = scene
    before, _raw = _fillable_hole(solve, plate, view, base_solve=solve)
    out, _crop = _patched(solve, plate_t, view, monkeypatch)
    after, _raw2 = _fillable_hole(out, plate, view, base_solve=solve)

    closed = int((before & ~after).sum())
    opened = int((after & ~before).sum())

    assert opened == 0, f"the patch OPENED {opened} px it should not have"
    assert closed / max(int(before.sum()), 1) > 0.60, (
        f"one patch closed {closed} of {int(before.sum())} fillable px — the "
        "geometry is not covering the hole it was generated for")


def test_the_patch_does_not_close_the_hole_with_a_shard(scene, monkeypatch):
    """Coverage alone would license the 11 m triangles that shipped on
    2026-09-04. A hole closed by a metres-long smear is not closed; it is a
    lie that exports to the DCC as real geometry."""
    solve, plate, plate_t, view = scene
    out, _crop = _patched(solve, plate_t, view, monkeypatch)

    worst = _worst_edge_m(out)
    assert worst > 0.0, "no patch geometry was built"
    # The fixture's wall is 12 m wide and 8 m tall; a triangle approaching that
    # is spanning the scene, not describing it.
    assert worst < 3.0, f"worst patch triangle edge {worst:.2f} m"


def test_disabling_the_edge_guard_is_what_produces_a_shard(scene, monkeypatch):
    """The bound is load-bearing, not decoration.

    Note what the guard is and is not. Its budget is `max_edge_factor` x the
    expected local sample SPACING, which scales with the triangle's own depth --
    so it bounds a shard relative to how densely that region is sampled, never
    to an absolute metre count. A far-field hallucination therefore gets a
    generous budget by construction, and it takes a violent one to reach the
    shipped 40 (measured on this fixture: a 2x cliff never reaches it, 6x does
    not, 60x does). That is not a weakness -- the real castle fill produced
    edges at 597x the local spacing -- but it does mean "40" is not a promise
    about metres, and the earlier claim that it "bounds shards at ~2.4 m" was
    true of that scene's depths and not in general.
    """
    solve, plate, plate_t, view = scene

    def cliff(sv, crop, gw, gh, vw):
        """The wall, with half the fill hallucinating a far field -- exactly
        the "one far-depth corner" relief_mesh warns about."""
        d = _wall_depth_map(sv, crop, gw, gh, vw)
        d[:, d.shape[1] // 2:] *= 60.0
        return d

    bounded, _c1 = _patched(solve, plate_t, view, monkeypatch, depth=cliff,
                            depth_edge_rel=1e9, max_edge_factor=40.0)
    unbounded, _c2 = _patched(solve, plate_t, view, monkeypatch, depth=cliff,
                              depth_edge_rel=1e9, max_edge_factor=0.0)

    wb, wu = _worst_edge_m(bounded), _worst_edge_m(unbounded)
    assert wu > wb * 2.0, (
        f"the edge guard is not doing anything on a fixture built to need it: "
        f"bounded {wb:.2f} m, unbounded {wu:.2f} m")
    # It tore something to achieve that -- otherwise the two are the same mesh
    # and the metres above moved for some other reason.
    faces = lambda s: s.projection_sources[-1].proxy_geometry[0].metadata["n_faces"]
    assert faces(bounded) < faces(unbounded)


def test_tearing_at_the_relief_defaults_costs_real_coverage(scene, monkeypatch):
    """The tearing fix, scored in HOLE rather than in torn_fraction.

    `test_fill_patch_tearing.py` pins the mechanism -- that the silhouette
    thresholds are settable and that a fill patch sets them. This pins what it
    was FOR: a patch built at the relief defaults tears itself apart over a
    fill's own depth and leaves the hole open. Nothing else in the suite
    connects the two, which is why the defect was found by eye on a live run
    and not here.

    Blocky depth on purpose. A monocular estimate over inpainted content is
    cliffy at cell scale -- rock, water, distance -- and one smooth step is not
    a stand-in: measured on this fixture a single 1.6x step costs only 2.5
    points of coverage, where the blocky field costs 31 and reaches
    torn_fraction 0.335, close to the 0.404 the real castle fill produced.
    """
    solve, plate, plate_t, view = scene

    def blocky(sv, crop, gw, gh, vw):
        d = _wall_depth_map(sv, crop, gw, gh, vw)
        rng = np.random.default_rng(3)
        h, w = d.shape
        blocks = rng.uniform(0, 1, size=(8, 8)) < 0.45
        f = np.where(blocks, 1.0, 1.9)
        f = np.repeat(np.repeat(f, -(-h // 8), 0), -(-w // 8), 1)[:h, :w]
        return (d * f).astype(np.float32)

    before, _raw = _fillable_hole(solve, plate, view, base_solve=solve)

    def closed_with(**kw):
        out, _c = _patched(solve, plate_t, view, monkeypatch, depth=blocky, **kw)
        after, _r = _fillable_hole(out, plate, view, base_solve=solve)
        torn = out.projection_sources[-1].proxy_geometry[0].metadata["torn_fraction"]
        return int((before & ~after).sum()), torn

    kept, torn_kept = closed_with(depth_edge_rel=1e9, max_edge_factor=40.0)
    torn_off, torn_frac = closed_with(depth_edge_rel=0.5, max_edge_factor=12.0)

    assert torn_frac > 0.20, (
        f"the fixture must actually tear at the relief defaults (got "
        f"{torn_frac:.3f}) or this compares two identical meshes")
    assert torn_kept < 0.10
    assert kept > torn_off * 1.4, (
        f"not tearing closed {kept} px vs {torn_off} — the fix is not paying "
        "for itself on a fixture built to show it")


def test_the_metric_is_not_the_one_the_node_prints(scene, monkeypatch):
    """`peak hole` counts sky and off-plate, so scoring on it understates the
    result. Pinned because every earlier note in this repo quotes it as a score
    and reads a success as a near-failure."""
    solve, plate, plate_t, view = scene
    before_fillable, before_raw = _fillable_hole(solve, plate, view,
                                                 base_solve=solve)
    out, _crop = _patched(solve, plate_t, view, monkeypatch)
    after_fillable, after_raw = _fillable_hole(out, plate, view,
                                               base_solve=solve)

    fillable_gain = 1 - after_fillable.sum() / before_fillable.sum()
    raw_gain = 1 - after_raw.sum() / before_raw.sum()

    assert fillable_gain > raw_gain, (
        f"fillable {fillable_gain:.1%} vs raw {raw_gain:.1%} — if these agree "
        "the fixture has no unfillable hole and the distinction is untested")
