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
