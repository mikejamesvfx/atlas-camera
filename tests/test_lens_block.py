"""The ``lens`` block for matrixZone (schema 1.2) and the extended remap grids
under it.

Everything here runs on a synthetic radial lens so it is hermetic; the same
properties are checked live against a Burning Palms X-H2 RAF when
``ATLAS_LIVE_RAF`` points at one (see the last test).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from atlas_camera.raw import lens_block as lb
from atlas_camera.raw import undistort as und
from atlas_camera.raw.redistort import (
    build_redistort_stmap,
    build_undistort_stmap,
    invert_remap,
)


# --- a synthetic lens that can be evaluated anywhere, like lensfun's region call

class _RadialLens:
    """A radial correction about the PLATE centre, normalised by the plate
    half-dimensions, evaluable over any region — the shape of
    ``lensfunpy.Modifier.apply_geometry_distortion(xu, yu, width, height)``.

    Sign convention, stated because it is easy to get backwards: ``coords`` is
    where an UNDISTORTED pixel samples the DISTORTED plate. ``k < 0`` samples
    an INSET (the rectilinear frame shows less than the plate holds, which is
    what lensfun's correction of the X-H2 / XF16-55 at 16 mm does: x=133.8 at
    the left edge of a 7752 plate), so the plate's edge pixels are only reached
    by undistorted pixels OUTSIDE the frame and overscan is owed. ``k > 0``
    samples beyond the plate: the rectilinear frame already covers everything
    and nothing is owed. Keep ``|k|`` small: every polynomial folds at some
    radius, and ``r·(1 + k·r²)`` folds at ``r = sqrt(1/(3|k|))``, which must
    stay well past the plate corner (``r = 1.41``) for the probe to be sane.
    """

    def __init__(self, plate_w, plate_h, k=-0.04):
        self.w, self.h, self.k = plate_w, plate_h, k

    def apply_geometry_distortion(self, xu=0.0, yu=0.0, width=-1, height=-1):
        width = self.w if width < 0 else width
        height = self.h if height < 0 else height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx = xx + float(xu)
        yy = yy + float(yu)
        cx, cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
        nx = (xx - cx) / cx
        ny = (yy - cy) / cy
        f = 1.0 + self.k * (nx * nx + ny * ny)
        return np.stack([cx + nx * f * cx, cy + ny * f * cy],
                        axis=-1).astype(np.float32)


def _fake_resolver(lens):
    def _resolve(meta, width, height):
        return "applied", None, None, lens
    return _resolve


@pytest.fixture
def barrel(monkeypatch):
    """The X-H2-shaped case: an inset correction that owes overscan."""
    lens = _RadialLens(96, 64, k=-0.04)
    monkeypatch.setattr(und, "_resolve_modifier", _fake_resolver(lens))
    return lens


# --- extended grids ----------------------------------------------------------

def test_extend_splices_the_plate_grid_bit_for_bit(barrel):
    coords = barrel.apply_geometry_distortion()
    # Perturb the plate grid so the splice is observable: the interior of the
    # extended grid must be THIS array, not a re-evaluation.
    coords = coords + np.float32(0.25)
    ext = und.extend_undistort_map(None, 96, 64, coords, (10, 6, 12, 8))
    assert ext.shape == (64 + 6 + 8, 96 + 10 + 12, 2)
    np.testing.assert_array_equal(ext[6:6 + 64, 10:10 + 96], coords)
    # ...and the band outside is the lens continued, not padding.
    assert np.isfinite(ext).all()
    assert ext[0, 0, 0] < coords[0, 0, 0] - 0.25  # further out than the corner


def test_extend_refuses_negative_overscan(barrel):
    with pytest.raises(ValueError):
        und.extend_undistort_map(None, 96, 64,
                                 barrel.apply_geometry_distortion(), (-1, 0, 0, 0))


def test_measure_excursion_is_positive_for_inset_and_zero_for_overreach(monkeypatch):
    monkeypatch.setattr(und, "_resolve_modifier",
                        _fake_resolver(_RadialLens(96, 64, k=-0.04)))
    exc = und.measure_excursion(None, 96, 64)
    assert set(exc) == {"left", "top", "right", "bottom"}
    assert exc["left"] > 0 and exc["right"] > 0
    assert exc["top"] > 0 and exc["bottom"] > 0
    # radial and centred: symmetric to a pixel
    assert abs(exc["left"] - exc["right"]) <= 1
    assert abs(exc["top"] - exc["bottom"]) <= 1

    monkeypatch.setattr(und, "_resolve_modifier",
                        _fake_resolver(_RadialLens(96, 64, k=0.12)))
    exc = und.measure_excursion(None, 96, 64)
    assert exc == {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}


def test_measure_excursion_grows_the_probe_rather_than_clipping(monkeypatch):
    """A first probe of 2% is too small for this correction; the answer must
    still be the true extent, not the probe's edge."""
    lens = _RadialLens(96, 64, k=-0.06)
    monkeypatch.setattr(und, "_resolve_modifier", _fake_resolver(lens))
    exc = und.measure_excursion(None, 96, 64, probe=0.02)
    exc_big = und.measure_excursion(None, 96, 64, probe=0.2)
    assert exc == exc_big
    assert exc["left"] > 0.02 * 96


def test_measure_excursion_refuses_a_lens_that_never_reaches_its_corners(monkeypatch):
    """Fold before the corner: the distorted corners are unreachable, so the
    honest answer is a refusal, not a number."""
    lens = _RadialLens(96, 64, k=-0.2)
    monkeypatch.setattr(und, "_resolve_modifier", _fake_resolver(lens))
    with pytest.raises(RuntimeError, match="refusing to guess"):
        und.measure_excursion(None, 96, 64)


# --- the inverse over an extended grid --------------------------------------

def test_redistort_over_extended_grid_covers_every_plate_pixel(barrel):
    coords = barrel.apply_geometry_distortion()
    exc = und.measure_excursion(None, 96, 64)
    ov = tuple(int(np.ceil(exc[k])) + 1 for k in ("left", "top", "right", "bottom"))
    grid = und.extend_undistort_map(None, 96, 64, coords, ov)
    stmap, info = build_redistort_stmap(grid, plate_size=(96, 64),
                                        plate_origin=(ov[0], ov[1]),
                                        iterations=30)
    assert stmap.shape == (64, 96, 4)                  # domain: the plate
    assert info["domain"] == "plate" and info["samples"] == "render"
    assert info["samples_size"] == [96 + ov[0] + ov[2], 64 + ov[1] + ov[3]]
    assert info["converged"]
    assert info["outside_fraction"] == 0.0             # the overscan was enough
    assert np.all(stmap[..., 3] == 1.0)
    # UVs are normalised over the RENDER, so the plate corner sits inside (0,1)
    assert 0.0 < stmap[0, 0, 0] < 0.5 and 0.5 < stmap[0, 0, 1] < 1.0
    # excursion measured on the converged inverse agrees with the forward probe
    for k in ("left", "top", "right", "bottom"):
        assert abs(info["excursion_px"][k] - exc[k]) < 1.5, (k, info["excursion_px"], exc)


def test_plate_sized_redistort_is_unchanged_by_default(barrel):
    """The shipping Nuke export calls with no plate_size; its output must be
    what it was — plate domain, plate samples, corners flagged."""
    coords = barrel.apply_geometry_distortion()
    stmap, info = build_redistort_stmap(coords)
    assert stmap.shape == (64, 96, 4)
    assert info["domain"] == "plate" and info["samples"] == "plate"
    assert info["plate_origin"] == [0, 0]
    assert info["outside_fraction"] > 0.0             # barrel corners have no source
    assert "excursion_px" in info


def test_invert_remap_rejects_a_plate_that_does_not_fit():
    with pytest.raises(ValueError):
        invert_remap(np.zeros((10, 10, 2), np.float32), plate_size=(12, 8))


# --- the forward map ---------------------------------------------------------

def test_undistort_stmap_conventions(barrel):
    coords = barrel.apply_geometry_distortion()
    stmap, info = build_undistort_stmap(coords)
    assert info["direction"] == "undistort"
    assert info["domain"] == "plate" and info["samples"] == "plate"
    u, v, zero, alpha = (stmap[..., i] for i in range(4))
    assert np.allclose(zero, 0.0)
    assert v[0, 48] > v[-1, 48]                        # V flipped for Nuke
    # barrel samples an inset: every plate pixel has a source
    assert alpha.min() == 1.0
    np.testing.assert_allclose(u, coords[..., 0] / 95.0, atol=1e-6)


def test_undistort_stmap_on_extended_grid_lives_on_the_render(barrel):
    coords = barrel.apply_geometry_distortion()
    grid = und.extend_undistort_map(None, 96, 64, coords, (10, 6, 10, 6))
    stmap, info = build_undistort_stmap(grid, plate_size=(96, 64))
    assert stmap.shape == (76, 116, 4)
    assert info["domain"] == "render" and info["samples"] == "plate"
    assert info["samples_size"] == [96, 64]
    # the band the lens never saw is flagged; the band it did see is not
    assert stmap[0, 0, 3] == 0.0
    assert stmap[6 + 32, 10 + 48, 3] == 1.0


# --- sizing the render -------------------------------------------------------

def test_plan_render_is_128_clean_and_keeps_its_invariants():
    exc = {"left": 218.3, "top": 145.1, "right": 218.3, "bottom": 145.1}
    r = lb.plan_render(7752, 5178, exc)
    assert r["width"] % 128 == 0 and r["height"] % 128 == 0
    L, T, R, B = r["overscan"]
    assert r["plateOrigin"] == [L, T]
    assert r["width"] == 7752 + L + R and r["height"] == 5178 + T + B
    for edge, need in zip((L, T, R, B), (219, 146, 219, 146)):
        assert edge >= need
    # padding is split, odd pixel to the far edge
    assert R - L in (0, 1) and B - T in (0, 1)


def test_plan_render_takes_the_larger_of_lens_and_margin():
    r = lb.plan_render(7680, 4320, None, margin=(0, 16, 0, 16))
    assert r["width"] == 7680 and r["height"] == 4352
    assert r["overscan"] == [0, 16, 0, 16]
    r2 = lb.plan_render(7680, 4320, {"left": 300, "right": 300, "top": 0, "bottom": 0},
                        margin=(0, 16, 0, 16))
    assert r2["overscan"][0] >= 300 and r2["overscan"][2] >= 300


# --- the block ---------------------------------------------------------------

class _Result:
    def __init__(self, status, applied=False, **kw):
        self.undistort_status = status
        self.undistort_applied = applied
        self.width, self.height = 96, 64
        self.distortion = kw.get("distortion", {})
        self.source_path = kw.get("source_path", "")


def test_block_for_camera_processed_is_rectilinear_with_no_maps(tmp_path):
    block, _ = lb.build_lens_block(_Result("camera_processed", True), str(tmp_path))
    assert block["source"] == "atlas-camera"
    assert block["space"] == "rectilinear"
    assert block["maps"] == []
    assert block["excursionPx"] == {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", ["disabled", "no_lens_metadata",
                                    "no_profile_camera", "no_profile_lens",
                                    "lensfunpy_missing"])
def test_block_for_uncorrected_plates_says_so_and_carries_nothing(status, tmp_path):
    block, _ = lb.build_lens_block(_Result(status), str(tmp_path))
    assert block["space"] == "distorted-uncorrected"
    assert block["maps"] == [] and "excursionPx" not in block


def test_block_rejects_an_unknown_status():
    with pytest.raises(ValueError):
        lb.build_lens_block(_Result("something_else"), None)


def test_block_for_applied_plate_with_render_window(tmp_path, monkeypatch):
    lens = _RadialLens(96, 64, k=-0.04)
    monkeypatch.setattr(und, "_resolve_modifier", _fake_resolver(lens))
    written = {}

    def _fake_write(stmap, path, *, content="redistort_stmap"):
        written[os.path.basename(path)] = (stmap.shape, content)
        return path

    from atlas_camera.raw import redistort, metadata
    monkeypatch.setattr(redistort, "write_stmap_exr", _fake_write)
    monkeypatch.setattr(metadata, "read_raw_metadata", lambda p: object())

    class _Und:
        status = "applied"
        cam_name = "Cam"
        lens_name = "Lens 16mm"
        coords = lens.apply_geometry_distortion()
        distortion = {"lensfun_crop_factor": 1.5, "lensfun_focal_mm": 16.0}
    monkeypatch.setattr(und, "build_undistort_map", lambda meta, w, h: _Und())

    exc = und.measure_excursion(None, 96, 64)
    render = lb.plan_render(96, 64, exc, clean=8)
    res = _Result("applied", True, source_path="fake.raf",
                  distortion=_Und.distortion)
    block, info = lb.build_lens_block(res, str(tmp_path), render=render,
                                      iterations=30)

    assert block["undistortStatus"] == "applied" and block["space"] == "rectilinear"
    assert block["profile"] == "Lens 16mm on Cam"
    assert block["distortion"] == _Und.distortion
    assert block["excursionPx"] == {k: float(exc[k]) for k in exc}
    dirs = {m["direction"]: m for m in block["maps"]}
    assert set(dirs) == {"redistort", "undistort"}
    rd, ud = dirs["redistort"], dirs["undistort"]
    assert rd["domain"] == "plate" and rd["samples"] == "render"
    assert rd["size"] == [96, 64] and rd["outsideFraction"] == 0.0 and rd["converged"]
    assert ud["domain"] == "render" and ud["samples"] == "plate"
    assert ud["size"] == [render["width"], render["height"]]
    assert written["redistort_stmap.exr"] == ((64, 96, 4), "redistort_stmap")
    assert written["undistort_stmap.exr"][0] == (render["height"], render["width"], 4)
    assert written["undistort_stmap.exr"][1] == "undistort_stmap"
    for m in block["maps"]:
        assert m["origin"] == "bottom-left" and m["alphaMeans"] == "source-inside"


def test_block_refuses_a_render_smaller_than_the_lens_excursion(tmp_path, monkeypatch):
    lens = _RadialLens(96, 64, k=-0.04)
    monkeypatch.setattr(und, "_resolve_modifier", _fake_resolver(lens))
    from atlas_camera.raw import metadata
    monkeypatch.setattr(metadata, "read_raw_metadata", lambda p: object())

    class _Und:
        status = "applied"; cam_name = "Cam"; lens_name = "L"
        coords = lens.apply_geometry_distortion(); distortion = {}
    monkeypatch.setattr(und, "build_undistort_map", lambda meta, w, h: _Und())
    res = _Result("applied", True, source_path="fake.raf")
    with pytest.raises(ValueError, match="smaller than the lens excursion"):
        lb.build_lens_block(res, str(tmp_path),
                            render={"width": 96, "height": 64, "plateOrigin": [0, 0]})


# --- delivery: the block comes out of the SHIPPING export ---------------------

def test_export_nuke_writes_the_lens_block_for_an_uncorrected_plate(tmp_path):
    """Same rule as the ST map: a record that only a research script writes
    is not a record. An uncorrected plate has no maps and still gets the
    block, because 'never corrected' is what the comp needs to read."""
    import json
    from atlas_camera.comfy.node_registry import NODE_CLASS_MAPPINGS as M
    from atlas_camera.core.schema import (
        AtlasCamera, AtlasExtrinsics, AtlasIntrinsics, AtlasSolve)

    class _FakeRawImport:
        undistort_applied = False
        undistort_status = "no_profile_lens"
        source_path = ""
        width = height = 64

    solve = AtlasSolve(camera=AtlasCamera(
        intrinsics=AtlasIntrinsics(image_width=64, image_height=64, fx_px=64.0,
                                   fy_px=64.0, cx_px=32.0, cy_px=32.0),
        extrinsics=AtlasExtrinsics()))
    out = tmp_path / "nuke"
    M["AtlasExportNuke"]().export(solve, str(out), raw_meta=_FakeRawImport(),
                                  write_redistort_stmap=True)
    doc = json.loads((out / "lens_block.json").read_text())
    assert doc["schema"] == lb.LENS_BLOCK_SCHEMA
    assert doc["lens"]["undistortStatus"] == "no_profile_lens"
    assert doc["lens"]["space"] == "distorted-uncorrected"
    assert doc["lens"]["maps"] == []
    assert not (out / "redistort_stmap.exr").exists()


# --- live, when a RAF is at hand ---------------------------------------------

@pytest.mark.skipif(not os.environ.get("ATLAS_LIVE_RAF"),
                    reason="set ATLAS_LIVE_RAF to a RAW with a lensfun profile")
def test_live_raf_block_round_trips(tmp_path):
    pytest.importorskip("lensfunpy")
    pytest.importorskip("rawpy")
    pytest.importorskip("OpenImageIO")
    from atlas_camera.raw.pipeline import import_raw
    res = import_raw(os.environ["ATLAS_LIVE_RAF"], undistort=True, half_size=True)
    assert res.undistort_status == "applied", res.undistort_status
    from atlas_camera.raw.metadata import read_raw_metadata
    exc = und.measure_excursion(read_raw_metadata(res.source_path),
                                res.width, res.height)
    render = lb.plan_render(res.width, res.height, exc)
    block, info = lb.build_lens_block(res, str(tmp_path), render=render)
    rd = next(m for m in block["maps"] if m["direction"] == "redistort")
    assert rd["outsideFraction"] == 0.0, "overscan did not cover the lens"
    assert rd["converged"]
    assert (tmp_path / "redistort_stmap.exr").exists()
    assert (tmp_path / "undistort_stmap.exr").exists()
