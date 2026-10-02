"""matrixZone for a still plate: bridge-parity plan, exact split/stitch, radiance anchor."""

import numpy as np
import pytest

from atlas_camera.core.matrixzone import (
    crop_zones,
    frames_8k1,
    lowpass,
    pad_to_render,
    plan_still,
    seam_metrics,
    stitch,
    to_log2,
    zones_as_frames,
)


def test_plan_matches_bridge_uhd_2x2():
    # atlas-unreal Docs/ATLAS_MATRIXZONE.md rev 3, "UHD 8K, camera-corrected plate".
    p = plan_still(7680, 4320, (2, 2))
    assert (p["render"]["width"], p["render"]["height"]) == (7680, 4352)
    assert p["render"]["plateOrigin"] == [0, 16]
    assert p["zone_size"] == [3904, 2240]
    assert p["overlap"]["px"] == [128, 128]
    assert p["global"]["size"] == [1920, 1088]
    assert [z["plateRect"] for z in p["zones"]] == [
        [0, 16, 3840, 2160], [3840, 16, 3840, 2160],
        [0, 2176, 3840, 2160], [3840, 2176, 3840, 2160]]


def test_plan_4x4_is_the_1080p_tier():
    p = plan_still(7680, 4320, (4, 4))
    zw, zh = p["zone_size"]
    assert zw % 64 == 0 and zh % 64 == 0 and zw <= 2048 and zh <= 1152
    assert min(p["overlap"]["px"]) >= 64


def test_render_and_zones_are_clean_and_tile():
    p = plan_still(7680, 4512, (3, 2))
    rw, rh = p["render"]["width"], p["render"]["height"]
    assert rw % 128 == 0 and rh % 128 == 0
    cover = np.zeros((rh, rw), bool)
    for z in p["zones"]:
        x, y, w, h = z["renderRect"]
        assert w % 64 == 0 and h % 64 == 0 and x >= 0 and y >= 0
        assert x + w <= rw and y + h <= rh
        cover[y:y + h, x:x + w] = True
    assert cover.all()


def test_serpentine_scan_keeps_neighbours_adjacent():
    p = plan_still(7680, 4320, (3, 2))
    assert p["scan"] == [0, 1, 2, 5, 4, 3]


def test_zones_as_frames_is_8k_plus_1_and_maps_back():
    p = plan_still(1024, 576, (2, 2))
    zones = [np.full((4, 4, 3), i, np.float32) for i in range(4)]
    frames, frame_of_zone = zones_as_frames(zones, p)
    assert len(frames) == 9 == frames_8k1(4)
    for zi, fi in enumerate(frame_of_zone):
        assert frames[fi][0, 0, 0] == zi


def _ramp_plate(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([0.05 + xx / w, 0.05 + yy / h, 0.05 + 0.5 * (xx + yy) / (w + h)], -1)


def test_split_then_stitch_is_identity_without_a_model():
    # Gate 1: zones straight back -> the plate, exactly (to float precision).
    plate = _ramp_plate(1100, 620)
    p = plan_still(1100, 620, (2, 2))
    zones = crop_zones(pad_to_render(plate, p), p)
    out, rep = stitch(zones, p)
    assert out.shape == plate.shape
    assert np.max(np.abs(out - plate) / plate) < 1e-5
    assert all(s["p95_stops"] < 1e-5 for s in rep["seams_before"])


def test_anchor_removes_a_per_zone_exposure_disagreement():
    plate = _ramp_plate(1100, 620)
    p = plan_still(1100, 620, (2, 2))
    zones = crop_zones(pad_to_render(plate, p), p)
    gains = [1.0, 1.6, 0.7, 1.3]                       # each zone guessed its own exposure
    off = [z * g for z, g in zip(zones, gains)]
    glob = pad_to_render(plate, p)[::4, ::4]            # a low-tier whole-frame pass
    raw, rep_raw = stitch(off, p)
    fixed, rep = stitch(off, p, global_hdr=glob)
    before = max(s["median_stops"] for s in rep["seams_before"])
    after = max(s["median_stops"] for s in rep["seams_after_anchor"])
    assert before > 0.4 and after < 0.05
    err_raw = np.median(np.abs(np.log2(raw / plate)))
    err_fixed = np.median(np.abs(np.log2(fixed / plate)))
    assert err_fixed < 0.05 < err_raw


def test_anchor_keeps_zone_detail():
    rng = np.random.default_rng(0)
    plate = _ramp_plate(1100, 620) * (1 + 0.2 * rng.random((620, 1100, 1))).astype(np.float32)
    p = plan_still(1100, 620, (2, 2))
    zones = crop_zones(pad_to_render(plate, p), p)
    glob = pad_to_render(plate, p)[::8, ::8]            # too coarse to carry the texture
    out, _ = stitch(zones, p, global_hdr=glob, split_px=64)
    hi_true = to_log2(plate) - lowpass(to_log2(plate), 16)
    hi_out = to_log2(out) - lowpass(to_log2(out), 16)
    assert np.corrcoef(hi_true.ravel(), hi_out.ravel())[0, 1] > 0.95


def test_seam_metric_reports_disagreement():
    p = plan_still(1024, 576, (2, 1))
    zones = crop_zones(pad_to_render(np.ones((576, 1024, 3), np.float32), p), p)
    zones[1] = zones[1] * 2.0
    (s,) = seam_metrics([to_log2(z) for z in zones], p)
    assert s["median_stops"] == pytest.approx(1.0, abs=1e-4)


def test_wrong_zone_count_raises():
    p = plan_still(1024, 576, (2, 2))
    with pytest.raises(ValueError, match="zone results"):
        stitch([np.ones((4, 4, 3))], p)


def _striped_conversion(seed=0):
    """An SDR plate with its OWN faint stripes, and an 'HDR conversion' that
    adds different, stronger ones plus a real highlight expansion."""
    rng = np.random.default_rng(seed)
    h, w = 400, 900
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    own = 1 + 0.008 * np.sin(2 * np.pi * xx / 48)                 # the plate's own banding
    sdr = (0.2 + 0.5 * xx / w)[..., None] * own[..., None] * np.ones(3, np.float32)
    sdr[50:90, 600:700] = 0.97                                      # a clipped highlight
    added = np.exp2(0.03 * np.sin(2 * np.pi * xx / 90 + rng.random()))  # model stripes
    hdr = sdr * added[..., None]
    hdr[50:90, 600:700] = 6.0                                       # expansion the model made
    return sdr, hdr, added


def test_destripe_removes_added_stripes_and_keeps_highlights():
    from atlas_camera.core.matrixzone import destripe_columns
    sdr, hdr, added = _striped_conversion()
    out, rep = destripe_columns(hdr, sdr)
    assert rep["ripple_before_stops"] > 0.015 and rep["ripple_after_stops"] < 0.003
    # the plate's own 48 px banding survives (it is not the conversion's)
    np.testing.assert_allclose(out[200:300, 100:500], sdr[200:300, 100:500], rtol=0.01)
    # the reconstructed highlight stays reconstructed
    assert out[60:80, 620:680].mean() > 5.0


def test_destripe_per_band_interpolates_without_a_row_edge():
    from atlas_camera.core.matrixzone import destripe_columns
    sdr, hdr, _ = _striped_conversion()
    out, rep = destripe_columns(hdr, sdr, bands=[(0, 200), (200, 400)])
    assert rep["bands"] == 2
    row_jump = np.abs(np.log2(out[200, 50:550, 0]) - np.log2(out[199, 50:550, 0])).max()
    assert row_jump < 0.01
