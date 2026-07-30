"""Equirect -> perspective splitting: geometry, not just "it ran".

The failure mode this guards is a silent convention slip — a sign flip in yaw,
latitude, or the camera's -Z facing produces crops that look plausible and place
geometry in the wrong direction, which only shows up much later as a solve that
does not line up. So every test here asserts a DIRECTION or an ANGLE, not a
shape.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from atlas_camera.core.equirect import (
    direction_to_equirect_uv,
    equirect_to_perspective,
    intrinsics_for_view,
    perspective_view_angles,
    split_equirect,
    view_matrix_for_angles,
)


def _marker_equirect(h=64, w=128):
    """An equirect encoding its own direction CONTINUOUSLY.

    Longitude rides as (sin, cos) rather than a 0->1 ramp. A ramp is
    discontinuous at the 360 deg seam, so bilinear sampling across the wrap
    averages ~1.0 and ~0.0 into ~0.5 — which makes the probe report garbage at
    exactly the place the seam test is trying to inspect. sin/cos are smooth
    across the wrap, so any discontinuity in a crop is the CODE's, not the
    marker's. Latitude does not wrap, so a linear ramp is fine there.
    """
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    lon = ((u + 0.5) / w - 0.5) * 2.0 * math.pi
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[..., 0] = np.sin(lon)
    img[..., 1] = (v + 0.5) / h          # latitude coordinate (no wrap)
    img[..., 2] = np.cos(lon)
    return img


def _lon_of(sample):
    """Recover longitude in [0,1) from a sampled (sin, _, cos) triple."""
    return (math.atan2(sample[0], sample[2]) / (2.0 * math.pi) + 0.5) % 1.0


def test_view_angles_cover_the_full_circle_without_duplicates():
    angles = perspective_view_angles(12)
    yaws = [a for a, _ in angles]
    assert len(angles) == 12
    assert yaws[0] == 0.0                      # view 0 faces -Z, Atlas's default
    assert all(abs((yaws[i + 1] - yaws[i]) - 30.0) < 1e-9 for i in range(11))
    assert yaws[-1] == 330.0                   # 360 would duplicate view 0

    # The offset rotates the whole ring — for moving a seam off a subject.
    assert perspective_view_angles(4, yaw_offset_deg=45.0)[0][0] == 45.0
    with pytest.raises(ValueError):
        perspective_view_angles(0)


def test_intrinsics_match_the_requested_fov():
    size, fov = 512, 90.0
    fx, fy, cx, cy = intrinsics_for_view(size, fov)
    assert fx == pytest.approx(fy)                       # square crop
    assert (cx, cy) == (size / 2.0, size / 2.0)
    # At 90 deg the half-width equals the focal length; recovering the angle
    # from the intrinsics must give back what was asked for.
    assert 2.0 * math.degrees(math.atan((size / 2.0) / fx)) == pytest.approx(fov)
    assert intrinsics_for_view(256, 60.0)[0] == pytest.approx(
        (256 / 2.0) / math.tan(math.radians(30.0)))
    for bad in (0.0, 180.0, -10.0):
        with pytest.raises(ValueError):
            intrinsics_for_view(256, bad)


def test_view_zero_looks_along_negative_z():
    """Atlas canonicalises the recovered camera to face -Z. View 0 must match,
    or a split panorama's primary view disagrees with an ordinary solve."""
    _view, _world, rot3 = view_matrix_for_angles(0.0, 0.0)
    r = np.asarray(rot3, dtype=float)
    forward = r @ np.array([0.0, 0.0, -1.0])     # cam -Z into world
    assert forward == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)

    # Yaw 90 must turn to +X (right), not -X. This is the sign that silently
    # mirrors a whole panorama if it is wrong.
    r90 = np.asarray(view_matrix_for_angles(90.0, 0.0)[2], dtype=float)
    assert r90 @ np.array([0.0, 0.0, -1.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)

    # Positive pitch looks UP.
    r_up = np.asarray(view_matrix_for_angles(0.0, 30.0)[2], dtype=float)
    assert (r_up @ np.array([0.0, 0.0, -1.0]))[1] == pytest.approx(0.5, abs=1e-9)


def test_crop_centre_samples_the_direction_it_was_asked_for():
    """Round-trip: the centre pixel of a crop at yaw must carry the equirect
    coordinate that yaw maps to. Catches any lon/lat sign or offset error."""
    img = _marker_equirect()
    for yaw in (0.0, 45.0, 90.0, 180.0, 270.0, 330.0):
        crop, _ = equirect_to_perspective(img, yaw_deg=yaw, fov_deg=90.0, size=33)
        centre = crop[16, 16]
        d = (math.sin(math.radians(yaw)), 0.0, -math.cos(math.radians(yaw)))
        want_u, want_v = direction_to_equirect_uv(*d)
        # Longitude is circular, so compare on the circle rather than linearly.
        du = abs(((_lon_of(centre) - want_u + 0.5) % 1.0) - 0.5)
        assert du < 0.01, f"yaw {yaw}: longitude {_lon_of(centre)} != {want_u}"
        assert centre[1] == pytest.approx(want_v, abs=0.01)


def test_latitude_polarity_top_of_crop_is_up():
    """The +90 deg latitude row is the TOP of an equirect. If this inverts, every
    split panorama is upside down while still looking like a valid image."""
    img = _marker_equirect()
    crop, _ = equirect_to_perspective(img, yaw_deg=0.0, fov_deg=90.0, size=33)
    assert crop[0, 16][1] < crop[32, 16][1], "v must increase downward"
    # Looking up must sample nearer the top row (smaller v) than looking level.
    up, _ = equirect_to_perspective(img, yaw_deg=0.0, pitch_deg=45.0,
                                    fov_deg=90.0, size=33)
    assert up[16, 16][1] < crop[16, 16][1]


def test_seam_wraps_instead_of_clamping():
    """A view straddling longitude 180 must sample continuously across the
    wrap. Clamping there would put a hard edge down the middle of one crop —
    the classic equirect bug, and invisible unless you look at that one view."""
    img = _marker_equirect()
    crop, _ = equirect_to_perspective(img, yaw_deg=180.0, fov_deg=90.0, size=65)
    row = crop[32]
    # With a continuous marker, a correct wrap is SMOOTH across the seam. A
    # clamp would repeat the edge column and flatten one side to zero slope.
    lons = np.array([_lon_of(px) for px in row])
    unwrapped = np.unwrap(lons * 2.0 * math.pi) / (2.0 * math.pi)
    steps = np.diff(unwrapped)
    assert np.all(steps > 0), "longitude must advance monotonically across the seam"
    assert steps.max() / steps.min() < 3.0, "a clamp would show as a flat run"
    # And it really did straddle the seam: raw (wrapped) values hit both ends.
    assert lons.min() < 0.05 and lons.max() > 0.95


def test_split_returns_parallel_crops_angles_and_shared_intrinsics():
    img = _marker_equirect()
    crops, angles, intr = split_equirect(img, n_views=6, fov_deg=90.0, size=16)
    assert len(crops) == len(angles) == 6
    assert all(c.shape == (16, 16, 3) for c in crops)
    assert intr == intrinsics_for_view(16, 90.0)
    # Distinct directions must give distinct imagery — a bug that ignored yaw
    # would return six identical crops and still pass every shape assertion.
    centres = [tuple(np.round(c[8, 8], 4)) for c in crops]
    assert len(set(centres)) == 6


def test_greyscale_and_dtype_survive_the_round_trip():
    grey = (_marker_equirect()[..., 0] * 255).astype(np.uint8)
    crop, _ = equirect_to_perspective(grey, yaw_deg=0.0, fov_deg=90.0, size=8)
    assert crop.shape == (8, 8) and crop.dtype == np.uint8

    with pytest.raises(ValueError):
        equirect_to_perspective(np.zeros((1, 1, 3)), yaw_deg=0.0)


# --------------------------------------------------------------------------
# Node layer — the wrapper, not the maths (see test_node_layer_contracts.py).
# --------------------------------------------------------------------------

def test_split_equirect_node_emits_a_wirable_exact_view():
    """`exact_view` must be in the exact format AtlasAddPatchView parses.

    That string is the whole point of this node: equirect angles are MEASURED,
    not estimated, so they bypass the named-view combos ('front-right quarter
    view') and go in through exact_view_override. A format drift here fails
    silently — the patch node falls back to its combo defaults and the view
    lands in the wrong direction.
    """
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import node_registry as reg

    cls = reg.NODE_CLASS_MAPPINGS["AtlasSplitEquirect"]
    pano = torch.from_numpy(_marker_equirect(64, 128).astype(np.float32))[None]

    view, exact, focal_mm, all_views, report = getattr(cls(), cls.FUNCTION)(
        pano, n_views=12, fov_deg=90.0, size=32, view_index=3)

    assert tuple(view.shape) == (1, 32, 32, 3)
    assert tuple(all_views.shape) == (12, 32, 32, 3)
    assert view.dtype == torch.float32

    # Same grammar AtlasBlockoutViewport's patch_exact emits.
    assert re.fullmatch(
        r"azimuth_deg=-?\d+\.\d+ elevation_deg=-?\d+\.\d+ distance_scale=\d+\.\d+",
        exact), exact
    assert "azimuth_deg=90.0000" in exact, "view 3 of 12 is 90 deg"
    # A panorama has ONE optical centre, so the camera never dollies.
    assert "distance_scale=1.0000" in exact
    assert "12 view(s)" in report


def test_split_equirect_node_warns_on_a_non_2to1_panorama():
    """A cropped pano still samples correctly but has no data outside its band.
    Artists routinely feed one believing it is full 360 — say so rather than
    silently returning clamped edge rows."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import node_registry as reg

    cls = reg.NODE_CLASS_MAPPINGS["AtlasSplitEquirect"]
    square = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    _v, _e, _f, _a, report = getattr(cls(), cls.FUNCTION)(square, n_views=4, size=16)
    assert "WARNING" in report and "2:1" in report

    wide = torch.zeros(1, 64, 128, 3, dtype=torch.float32)
    _v, _e, _f, _a, ok = getattr(cls(), cls.FUNCTION)(wide, n_views=4, size=16)
    assert "WARNING" not in ok


def test_split_equirect_node_clamps_the_view_index():
    """view_index past the end must clamp, not raise — n_views is editable and
    an artist lowering it with a high index set is an ordinary sequence."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import node_registry as reg

    cls = reg.NODE_CLASS_MAPPINGS["AtlasSplitEquirect"]
    pano = torch.from_numpy(_marker_equirect(32, 64).astype(np.float32))[None]
    _v, exact, _f, _a, _r = getattr(cls(), cls.FUNCTION)(
        pano, n_views=4, size=16, view_index=99)
    assert "azimuth_deg=270.0000" in exact, "clamped to the last view, not view 0"


def test_node_emits_the_exact_focal_rather_than_leaving_it_to_be_guessed():
    """`focal_mm` must be CONSTRUCTED from the requested FOV, not estimated.

    Measured live on an 8K parish-road panorama: four views recovered their FOV
    to within 1.6, 10.2, 9.1 and 3.8 degrees when the solver guessed, and to
    0.000 degrees on all four when told. One also fell back to
    scale_source=assumed_default while guessing and reached depth_ground_plane
    once the focal was known — a wrong focal poisons the ground-plane fit that
    metric scale depends on. So this output is not a convenience, it is the
    difference between a measured and an assumed solve.
    """
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import node_registry as reg

    cls = reg.NODE_CLASS_MAPPINGS["AtlasSplitEquirect"]
    assert cls.RETURN_NAMES == ("view", "exact_view", "focal_mm", "all_views", "report")
    pano = torch.zeros(1, 64, 128, 3, dtype=torch.float32)

    # 90 deg on a 36mm reference: fx == size/2, so focal == 18mm at any size.
    for size in (256, 1024, 2048):
        out = getattr(cls(), cls.FUNCTION)(pano, n_views=12, fov_deg=90.0, size=size)
        assert out[2] == pytest.approx(18.0), f"size {size}"

    # And it tracks FOV, not just the 90 deg case.
    for fov in (60.0, 75.0, 120.0):
        out = getattr(cls(), cls.FUNCTION)(pano, n_views=8, fov_deg=fov, size=512)
        want = ((512 / 2.0) / math.tan(math.radians(fov) / 2.0)) * 36.0 / 512.0
        assert out[2] == pytest.approx(want), f"fov {fov}"
        # Round-trips: the emitted focal must reproduce the requested FOV.
        fx = out[2] * 512.0 / 36.0
        assert 2.0 * math.degrees(math.atan(256.0 / fx)) == pytest.approx(fov)


def test_load_plate_resolves_a_bare_filename_against_comfyui_input(tmp_path, monkeypatch):
    """A shipped workflow can only reference a plate by BARE filename.

    Absolute paths are banned outright (test_shipping_workflow_paths — baking an
    authoring machine's path broke a reviewer's clone), so a bare name is the
    only portable form. It used to raise "no such file", which meant the shipped
    equirect workflow could not run until the artist repointed it by hand.
    """
    pytest.importorskip("OpenImageIO")
    pytest.importorskip("torch")     # AtlasLoadPlate returns a torch tensor
    import sys, types as _types
    from atlas_camera.comfy import node_registry as reg

    inp = tmp_path / "input"
    inp.mkdir()
    plate = inp / "pano.exr"

    import numpy as _np
    import OpenImageIO as oiio
    spec = oiio.ImageSpec(16, 8, 3, "float")
    buf = oiio.ImageBuf(spec)
    buf.set_pixels(oiio.ROI(0, 16, 0, 8, 0, 1, 0, 3),
                   _np.zeros((8, 16, 3), dtype=_np.float32))
    assert buf.write(str(plate)), buf.geterror()

    monkeypatch.setitem(sys.modules, "folder_paths",
                        _types.SimpleNamespace(get_input_directory=lambda: str(inp)))

    cls = reg.NODE_CLASS_MAPPINGS["AtlasLoadPlate"]
    image, _alpha, _ref, report = getattr(cls(), cls.FUNCTION)("pano.exr")
    assert tuple(image.shape)[1:3] == (8, 16)
    assert "pano.exr" in report

    # A name that is nowhere must still fail loudly rather than silently pass.
    with pytest.raises(RuntimeError, match="no such file"):
        getattr(cls(), cls.FUNCTION)("definitely_not_here.exr")


# --------------------------------------------------------------------------
# AtlasEquirectMultiView — the whole ring in one node.
# --------------------------------------------------------------------------

def _stub_multiview(monkeypatch, heights):
    """Patch the solve + depth backends so the geometry path runs modelless.

    `heights` is the per-sampled-view camera height the fake solver returns, so a
    test can plant a deliberate outlier and check how it is consolidated.
    """
    import atlas_camera.comfy.nodes_solve as ns
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import (AtlasCamera, AtlasExtrinsics, AtlasSolve)
    from atlas_camera.inference.depth_estimator import DepthResult

    calls = {"solves": [], "depths": 0}
    seq = list(heights)

    # NO **kw on purpose: a permissive stub is what let `focal_length_mm`
    # through to a real function that wanted `focal_length_mm_hint`, so every
    # unit test passed while the node could not run at all. The stub must be
    # as strict as the thing it replaces.
    def fake_solve(path, *, camera_height="auto", depth_model=None,
                   focal_length_mm_hint=None, sensor_width_mm=36.0):
        calls["solves"].append(camera_height)
        h = (seq.pop(0) if (camera_height == "auto" and seq)
             else (1.6 if camera_height == "auto" else float(camera_height)))
        w = h_px = 64
        return AtlasSolve(
            camera=AtlasCamera(
                intrinsics=build_intrinsics(image_width=w, image_height=h_px,
                                            focal_length_mm=focal_length_mm_hint or 18.0,
                                            sensor_width_mm=sensor_width_mm),
                extrinsics=AtlasExtrinsics(camera_position=(0.0, h, 0.0))),
            image_width=w, image_height=h_px)

    def fake_depth(path, *, model_id=None, device=None, focal_px=None, max_side=0):
        calls["depths"] += 1
        d = np.full((64, 64), 10.0, dtype=np.float32)
        d[32:, :] = 4.0                      # a ground-ish lower half
        return DepthResult(depth=d, is_metric=True, model_id=model_id or "stub",
                           image_width=64, image_height=64, near=4.0, far=10.0)

    monkeypatch.setattr(ns, "solve_still_image_learned", fake_solve, raising=False)
    import atlas_camera.core.solver as solver
    monkeypatch.setattr(solver, "solve_still_image_learned", fake_solve)
    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake_depth)
    return calls


def _run_multiview(monkeypatch, heights, **kw):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import node_registry as reg
    _stub_multiview(monkeypatch, heights)
    cls = reg.NODE_CLASS_MAPPINGS["AtlasEquirectMultiView"]
    pano = torch.from_numpy(_marker_equirect(64, 128).astype(np.float32))[None]
    params = dict(n_views=4, size=64, height_samples=len(heights), relief_grid=32)
    params.update(kw)
    return getattr(cls(), cls.FUNCTION)(pano, **params)


def test_multiview_every_camera_shares_one_optical_centre(monkeypatch):
    """THE defining property, and the whole reason this node exists.

    AtlasAddPatchView builds patch cameras with orbit_camera, which MOVES the
    eye — it rotates the camera's offset from a ground pivot and re-aims,
    displacing it by ~2*r*sin(delta/2), metres at a typical pivot distance.
    Panorama views share ONE optical centre. If anyone ever routes this through
    the orbit path, this is the test that catches it.
    """
    solve, _views, _report = _run_multiview(monkeypatch, [1.4, 1.4, 1.4, 1.4])
    srcs = solve.projection_sources
    assert len(srcs) == 4
    eye = tuple(solve.camera.extrinsics.camera_position)
    for s in srcs:
        assert tuple(s.camera.extrinsics.camera_position) == pytest.approx(eye, abs=1e-9)
        # A panorama camera never dollies, so this is exact, not approximate.
        assert s.distance_scale == 1.0


def test_multiview_azimuths_walk_the_ring_and_rotations_differ(monkeypatch):
    solve, _v, _r = _run_multiview(monkeypatch, [1.4] * 4)
    srcs = solve.projection_sources
    assert [s.azimuth_deg for s in srcs] == [0.0, 90.0, 180.0, 270.0]
    assert all(s.elevation_deg == 0.0 for s in srcs)
    # Same eye, DIFFERENT rotation — otherwise every view would be the primary.
    mats = {tuple(np.asarray(s.camera.extrinsics.camera_view_matrix).ravel().round(6))
            for s in srcs}
    assert len(mats) == 4


def test_multiview_consolidates_height_by_MEDIAN_not_mean(monkeypatch):
    """The specific choice made here, so it needs a test that fails if someone
    'simplifies' it to an average.

    Planted sample: 1.40 / 1.42 / 1.44 / 6.40. Median 1.43, mean 2.665 — a real
    panorama produced exactly this shape (one view at 6.37 against a 5.91-6.08
    cluster), and averaging would have baked the outlier into every view.
    """
    solve, _v, report = _run_multiview(monkeypatch, [1.40, 1.42, 1.44, 6.40])
    assert solve.camera.extrinsics.camera_position[1] == pytest.approx(1.43, abs=1e-6)
    assert solve.camera.extrinsics.camera_position[1] < 2.0, "the mean would be 2.665"
    # And it is genuinely ONE height, not per-view estimates left to disagree.
    assert {round(s.metadata["shared_height_m"], 9)
            for s in solve.projection_sources} == {round(1.43, 9)}


def test_multiview_report_exposes_the_spread_and_the_worst_sample(monkeypatch):
    """Auditability is the CONDITION on automating the consolidation at all — a
    median you cannot inspect is how a wrong scale gets baked in unnoticed."""
    _s, _v, report = _run_multiview(monkeypatch, [1.40, 1.42, 1.44, 6.40])
    assert "MEDIAN" in report
    assert "spread 5.0000 m" in report, report
    assert "furthest from the median: view 3" in report, report
    assert "1.4000" in report and "6.4000" in report      # every sample listed
    assert "EXACT" in report                              # the focal provenance
    assert "ONE eye shared" in report


def test_multiview_uses_one_shared_ground_scale(monkeypatch):
    """All views share an eye and a focal, so a per-view ground fit would only
    re-derive the same number with more noise. One scale is what makes the ring
    a single consistent metric world."""
    solve, _v, _r = _run_multiview(monkeypatch, [1.4] * 4)
    scales = {round(s.metadata["shared_ground_scale"], 9)
              for s in solve.projection_sources}
    assert len(scales) == 1, f"expected one shared scale, got {scales}"


def test_multiview_emits_geometry_and_imagery_per_view(monkeypatch):
    solve, views, _r = _run_multiview(monkeypatch, [1.4] * 4)
    assert tuple(views.shape)[0] == 4
    for s in solve.projection_sources:
        assert s.proxy_geometry, f"{s.name} carries no geometry"
        assert s.image_b64 and s.image_b64.startswith("data:image")
        assert s.metadata["equirect_view_index"] in (0, 1, 2, 3)


def test_multiview_calls_the_real_backend_signatures(monkeypatch):
    """Guard against a stub hiding an interface mismatch.

    The first live run of this node failed with "solve_still_image_learned() got
    an unexpected keyword argument 'focal_length_mm'" — the real parameter is
    `focal_length_mm_hint`. Every unit test passed, because they all ran against
    a hand-written stub that accepted the invented name. A stub can only prove
    the node's logic; it cannot prove the node can talk to the thing it stubs.
    So bind the node's actual keyword arguments against the REAL signatures.
    """
    import inspect
    from atlas_camera.core.solver import solve_still_image_learned
    from atlas_camera.inference.depth_estimator import estimate_depth

    solve_sig = inspect.signature(solve_still_image_learned)
    # exactly what the node passes
    solve_sig.bind("path.png", camera_height="auto",
                   depth_model="Ruicheng/moge-2-vitl-normal",
                   focal_length_mm_hint=18.0, sensor_width_mm=36.0)
    solve_sig.bind("path.png", camera_height=1.43,
                   depth_model="Ruicheng/moge-2-vitl-normal",
                   focal_length_mm_hint=18.0, sensor_width_mm=36.0)

    depth_sig = inspect.signature(estimate_depth)
    depth_sig.bind("path.png", model_id="Ruicheng/moge-2-vitl-normal",
                   device=None, focal_px=512.0, max_side=0)

    from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
    bs = inspect.signature(build_relief_mesh)
    bs.bind(np.zeros((4, 4)), view_matrix=np.eye(4), fx=1.0, fy=1.0, cx=1.0, cy=1.0,
            grid_long_edge=32, scale=1.0)
    inspect.signature(estimate_ground_scale).bind(
        np.zeros((4, 4)), view_matrix=np.eye(4), fx=1.0, fy=1.0, cx=1.0, cy=1.0,
        horizon_y=None)


def test_multiview_returns_a_SELF_CONTAINED_solve(monkeypatch):
    """The primary's geometry must reach the SCENE, not only its ProjectionSource.

    Found live, not in a unit test: the first working run produced 12 sources
    with correct cameras and geometry, and AtlasMoveBudget still refused it —
    "this solve has proxy primitives but no relief mesh". The node was returning
    something that looked complete and could not actually be measured. Anything
    reading solve.projection_scene rather than projection_sources (move budget,
    occlusion graph, the exporters) needs the primary geometry there.
    """
    solve, _views, _report = _run_multiview(monkeypatch, [1.4] * 4)
    prims = solve.projection_scene.proxy_geometry
    assert prims, "primary relief mesh missing from projection_scene"
    meta = solve.projection_scene.debug_metadata["proxy_derivation"]
    assert meta["derive_node"] == "AtlasEquirectMultiView"
    mv = meta["equirect_multiview"]
    assert mv["n_views"] == 4
    # The consolidation inputs travel WITH the solve, so the audit trail
    # survives past the report string into anything that reads the solve.
    assert mv["shared_height_m"] == pytest.approx(1.4)
    assert "height_samples" in mv and "height_spread_m" in mv


def test_multiview_default_n_views_is_the_measured_sweet_spot():
    """4, not 12 — and the number is measured, so pin it.

    Going 2 -> 4 views closes the +-90 deg gap that makes a sideways dolly
    disocclude: safe z dolly jumps 4.2x (0.233 -> 0.983 m). Past 4 the budget
    PLATEAUS and drifts slightly down, while torn frame keeps falling (11.7% at
    4, 7.4% at 8, 5.5% at 12). So views 5-12 buy projection coverage rather than
    camera freedom, and each costs a full depth pass — an expensive default for
    a benefit many shots do not need.

    AtlasSplitEquirect stays at 12 deliberately: it only crops, so extra views
    there are free.
    """
    from atlas_camera.comfy import node_registry as reg

    mv = reg.NODE_CLASS_MAPPINGS["AtlasEquirectMultiView"].INPUT_TYPES()
    assert mv["optional"]["n_views"][1]["default"] == 4
    # The reasoning must travel with the widget, or the next person "tidies" it
    # back to a round number.
    tip = mv["optional"]["n_views"][1]["tooltip"]
    assert "sweet spot" in tip and "0.983" in tip and "coverage" in tip

    split = reg.NODE_CLASS_MAPPINGS["AtlasSplitEquirect"].INPUT_TYPES()
    assert split["optional"]["n_views"][1]["default"] == 12, (
        "the splitter has no per-view cost — do not shrink it in sympathy")


# --------------------------------------------------------------------------
# The shipped multiview workflow — pinning the couplings that are implicit in
# the graph and would otherwise only surface as breakage.
# --------------------------------------------------------------------------

def _multiview_workflow():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "examples" / "atlas_equirect_360_multiview_workflow.json")
        .read_text(encoding="utf-8"))


def test_multiview_workflow_feeds_the_viewport_from_all_views_slot():
    """`all_views` is an N-frame BATCH wired into a single-image input.

    That is correct, not sloppy: the viewport's `_image_tensor_to_pil` takes
    frame [0], and index 0 is the primary view by construction —
    `perspective_view_angles` puts yaw 0 first and the node solves that view as
    the primary. But the correctness depends on view ORDER, which nothing else
    forces, so pin both halves here: the wiring, and the ordering it relies on.
    """
    wf = _multiview_workflow()
    by_id = {n["id"]: n for n in wf["nodes"]}
    mv = next(n for n in wf["nodes"] if n["type"] == "AtlasEquirectMultiView")
    vp = next(n for n in wf["nodes"] if n["type"] == "AtlasBlockoutViewport")

    # slot 1 of the multiview node is `all_views`
    assert mv["outputs"][1]["name"] == "all_views"
    src_slot = {l[0]: (l[1], l[2]) for l in wf["links"]}
    link = vp["inputs"][1]["link"]
    assert vp["inputs"][1]["name"] == "source_image"
    assert src_slot[link] == (mv["id"], 1), "viewport must be fed from all_views"

    # ...and the ordering that makes frame 0 the primary.
    assert perspective_view_angles(4)[0] == (0.0, 0.0)


def test_multiview_workflow_keeps_move_budget_as_a_regression_guard():
    """AtlasMoveBudget is in this graph on purpose.

    It refused the solve during development — "has proxy primitives but no
    relief mesh" — because the primary geometry reached its ProjectionSource but
    not projection_scene. No unit test caught that; only a live run did. Keeping
    it wired in a SHIPPED graph is what stops the regression returning quietly.
    """
    wf = _multiview_workflow()
    mv = next(n for n in wf["nodes"] if n["type"] == "AtlasEquirectMultiView")
    mb = next((n for n in wf["nodes"] if n["type"] == "AtlasMoveBudget"), None)
    assert mb is not None, "the move-budget guard was removed from the shipped graph"
    src_slot = {l[0]: (l[1], l[2]) for l in wf["links"]}
    assert src_slot[mb["inputs"][0]["link"]] == (mv["id"], 0),         "move budget must measure the multiview solve, not something else"


def test_multiview_workflow_does_not_drift_from_the_measured_default():
    """The saved graph's n_views must equal the node's own default.

    The builder reads defaults off the live class precisely so the example cannot
    quietly disagree with the measurement the default encodes. If someone raises
    the node default, regenerating the workflow is part of that change.
    """
    from atlas_camera.comfy import node_registry as reg

    wf = _multiview_workflow()
    mv = next(n for n in wf["nodes"] if n["type"] == "AtlasEquirectMultiView")
    live = reg.NODE_CLASS_MAPPINGS["AtlasEquirectMultiView"].INPUT_TYPES()
    assert mv["widgets_values"][0] == live["optional"]["n_views"][1]["default"] == 4
