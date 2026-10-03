"""AtlasMatrixZoneSplit -> (identity 'model') -> AtlasMatrixZoneStitch round trip."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_matrixzone import (  # noqa: E402
    AtlasMatrixZoneSplit,
    AtlasMatrixZoneStitch,
)

H, W = 620, 1100


def _plate():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rgb = np.stack([0.05 + 0.9 * xx / W, 0.05 + 0.9 * yy / H, 0.3 + 0.0 * xx], -1)
    return torch.from_numpy(rgb)[None]


def _run(mode, monkeypatch, tmp_path, gains=None):
    from atlas_camera.comfy import nodes_matrixzone
    monkeypatch.setattr(nodes_matrixzone, "output_paths", lambda p: (tmp_path, "hdr_00001"))
    clips, handle, rep = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, mode, 9)
    # Stand-in for the LTX chain: an "HDR" that is the input, optionally with a
    # per-zone exposure guess (what a real model may do).
    out = [clips[0]]
    for i, c in enumerate(clips[1:]):
        g = 1.0 if gains is None else gains[i]
        out.append(c * g)
    res = AtlasMatrixZoneStitch().stitch(out, [handle], anchor=[True], split_px=[0],
                                         colorspace=["ACEScg"], filename_prefix=["x"])
    return clips, handle, res["result"]


@pytest.mark.parametrize("mode", ["per_zone_clip", "zones_as_frames"])
def test_round_trip_reproduces_the_plate(mode, monkeypatch, tmp_path):
    clips, handle, (preview, exr_path, report) = _run(mode, monkeypatch, tmp_path)
    assert len(clips) == (5 if mode == "per_zone_clip" else 2)
    assert all(len(c) % 8 == 1 for c in clips)
    assert preview.shape == (1, H, W, 3)
    assert "MODEL RECONSTRUCTION" in report
    # The EXR half: an explicit skip without OIIO (an `if exr_path:` guard
    # passed vacuously whenever the write failed).
    oiio = pytest.importorskip("OpenImageIO", reason="OIIO not installed: EXR not checked")
    assert exr_path, report
    px = oiio.ImageBuf(exr_path).get_pixels(oiio.FLOAT)
    assert px.shape[:2] == (H, W)
    assert np.median(np.abs(np.log2(px / _plate()[0].numpy()))) < 0.01


def test_anchor_fixes_zone_exposure_guesses(monkeypatch, tmp_path):
    _, _, (_, _, report) = _run("per_zone_clip", monkeypatch, tmp_path,
                                gains=[1.0, 1.7, 0.6, 1.2])
    before = report.split("seams before")[1].split("\n")[0]
    after = report.split("seams after radiance anchor")[1].split("\n")[0]
    import re
    worst = lambda s: max(float(m) for m in re.findall(r"z\d\d\|z\d\d (\d+\.\d+)/", s))  # noqa: E731
    assert worst(before) > 0.5 and worst(after) < 0.1


def test_stitch_refuses_a_mismatched_list(monkeypatch, tmp_path):
    clips, handle, _ = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "per_zone_clip", 9)
    with pytest.raises(ValueError, match="clip"):
        AtlasMatrixZoneStitch().stitch(clips[:3], [handle])


def test_stitch_report_carries_the_seam_step_gate(monkeypatch, tmp_path):
    _, _, (_, _, report) = _run("per_zone_clip", monkeypatch, tmp_path)
    assert "48 px strips, worst of ~256 px windows, NOT SDR-controlled" in report
    assert ("seam step test (delivered EXR" in report) or ("seam step test (in-memory plate" in report)
    assert "pass <= 1.5x random-line p90" in report


@pytest.mark.parametrize("size,grid", [((620, 1100), (2, 2)), ((577, 1001), (3, 2)),
                                       ((333, 517), (1, 3)), ((720, 1280), (4, 4)),
                                       ((401, 999), (2, 1))])
def test_split_torch_pad_and_crop_match_core(size, grid):
    # T10/R2: the node pads/crops in torch; core.pad_to_render/crop_zones is the
    # reference the bridge parity pins test. Pin the two paths equal, exactly.
    from atlas_camera.core.matrixzone import crop_zones, pad_to_render
    h, w = size
    rgb = np.random.default_rng(h * w).random((h, w, 3)).astype(np.float32)
    clips, handle, _ = AtlasMatrixZoneSplit().split(torch.from_numpy(rgb)[None], grid[0], grid[1],
                                                    64, "per_zone_clip", 1)
    plan = handle["plan"]
    render = pad_to_render(rgb, plan)
    assert render.shape[:2] == (plan["render"]["height"], plan["render"]["width"])
    ref = crop_zones(render, plan)
    assert len(clips) == 1 + len(ref)
    for clip, z in zip(clips[1:], ref):
        assert clip.shape[0] == 1
        np.testing.assert_array_equal(clip[0].numpy(), z)


def test_split_uses_core_frames_8k1():
    from atlas_camera.comfy import nodes_matrixzone
    from atlas_camera.core.matrixzone import frames_8k1
    assert not hasattr(nodes_matrixzone, "_frames_8k1")
    for n in (1, 2, 8, 9, 10, 17, 97):
        clips, handle, _ = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "per_zone_clip", n)
        assert handle["clip_frames"] == frames_8k1(n) == len(clips[0])


def test_exr_write_failure_is_report_line_one(monkeypatch, tmp_path):
    from atlas_camera.plate import oiio_io

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(oiio_io, "write_exr", boom)
    _, _, (_, exr_path, report) = _run("per_zone_clip", monkeypatch, tmp_path)
    assert exr_path == ""
    assert report.splitlines()[0] == "EXR NOT WRITTEN: OSError: disk full"


def test_zones_as_frames_refuses_a_short_sequence(monkeypatch, tmp_path):
    clips, handle, _ = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "zones_as_frames", 9)
    short = [clips[0], clips[1][:2]]                     # 4 zones need frames 0..3
    with pytest.raises(ValueError, match=r"2 frame\(s\); the split's 4 zones need at least 4"):
        AtlasMatrixZoneStitch().stitch(short, [handle])


def test_sdr_resample_is_reported_with_sizes(monkeypatch, tmp_path):
    from atlas_camera.comfy import nodes_matrixzone
    monkeypatch.setattr(nodes_matrixzone, "output_paths", lambda p: (tmp_path, "hdr_00001"))
    clips, handle, _ = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "per_zone_clip", 9)
    small = torch.nn.functional.interpolate(_plate().permute(0, 3, 1, 2), size=(310, 550),
                                            mode="area").permute(0, 2, 3, 1)
    res = AtlasMatrixZoneStitch().stitch(clips, [handle], sdr_plate=[small])
    report = res["result"][2]
    assert "warning: sdr_plate is 550x310, the stitched plate 1100x620: SDR resampled" in report


def test_split_warns_when_batch_frames_are_dropped():
    batch = torch.cat([_plate(), _plate() * 0.5, _plate() * 0.25])
    _, _, report = AtlasMatrixZoneSplit().split(batch, 2, 2, 64, "per_zone_clip", 9)
    assert "warning: image batch has 3 frames; only the first was split, 2 dropped" in report
    _, _, single = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "per_zone_clip", 9)
    assert "dropped" not in single


def test_per_zone_clips_are_zero_copy_expanded_views():
    clips, _, _ = AtlasMatrixZoneSplit().split(_plate(), 2, 2, 64, "per_zone_clip", 17)
    for c in clips:
        assert c.shape[0] == 17 and c.stride(0) == 0            # one frame's storage
        assert torch.equal(c[0], c[-1])
        # what downstream consumers do (VAE encode: movedim, *2-1, .to()) works
        x = (c.movedim(-1, 1) * 2.0 - 1.0).to(torch.float16)
        assert x.stride(0) != 0 and x.shape[0] == 17          # materialised, own storage


def test_exr_carries_provenance_attributes(monkeypatch, tmp_path):
    oiio = pytest.importorskip("OpenImageIO", reason="OIIO not installed: no EXR to read back")
    import json

    import atlas_camera
    _, _, (_, exr_path, report) = _run("per_zone_clip", monkeypatch, tmp_path)
    assert exr_path, report
    inp = oiio.ImageInput.open(exr_path)
    try:
        spec = inp.spec()
        params = json.loads(spec.getattribute("atlas:matrixzone_params"))
        worst = json.loads(spec.getattribute("atlas:seam_worst"))
        version = spec.getattribute("atlas:version")
        content = spec.getattribute("atlas:content")
    finally:
        inp.close()
    assert params["grid"] == [2, 2] and params["mode"] == "per_zone_clip"
    assert params["zone_tier"] == "1080p" and len(params["zone_size"]) == 2
    assert params["anchor"] is True and params["split_px"] > 0 and params["clip_frames"] == 9
    assert params["sdr_plate"] is False and params["destripe"] is False
    assert params["destripe_widget"] is True and params["detail_from_sdr"] is False
    assert params["guided_radius_px"] == 64 and params["guided_eps"] == pytest.approx(0.01)
    assert params["sdr_clip_window"] == [0.85, 0.97] and params["overlap_px"]
    assert worst["seam"].startswith("z") and worst["ratio"] >= 0
    assert "in-memory" in worst["scored_on"]
    assert version == atlas_camera.__version__
    assert content == "matrixZone SDR->HDR model reconstruction"


def test_stitch_with_sdr_plate_runs_cleanup_and_keeps_clipped_hdr(monkeypatch, tmp_path):
    # T-1(b): sdr_plate wired -> destripe, local destripe and detail-from-SDR
    # all run and report; a highlight the SDR clipped keeps the model's HDR.
    from atlas_camera.comfy import nodes_matrixzone
    from atlas_camera.core.generated_mesh import srgb_to_linear
    from atlas_camera.core.matrixzone import rec709_linear_to_acescg
    monkeypatch.setattr(nodes_matrixzone, "output_paths", lambda p: (tmp_path, "hdr_00002"))
    sdr = _plate().clone()
    sdr[0, 100:220, 150:400] = 1.0                       # a clipped highlight
    clips, handle, _ = AtlasMatrixZoneSplit().split(sdr, 2, 2, 64, "per_zone_clip", 9)

    def fake_model(c):                                   # SDR -> ACEScg, clipped -> 8.0
        a = c.numpy()
        hdr = rec709_linear_to_acescg(srgb_to_linear(a)).astype(np.float32)
        hdr[a.min(-1) >= 0.999] = 8.0
        return torch.from_numpy(hdr)

    out = [fake_model(c) for c in clips]
    res = AtlasMatrixZoneStitch().stitch(out, [handle], sdr_plate=[sdr])
    preview, exr_path, report = res["result"]
    assert "destripe: vertical ripple the conversion added" in report
    assert "local destripe (flat regions" in report
    assert "detail from SDR (guided filter r64)" in report
    assert "skipped" not in report
    assert "SDR-controlled" in report and "NOT SDR-controlled" not in report
    # tonemapped preview of 8.0 is ~0.95; the SDR's own 1.0 would be ~0.74
    assert float(preview[0, 130:190, 200:350].min()) > 0.9
    oiio = pytest.importorskip("OpenImageIO", reason="OIIO not installed: EXR not checked")
    assert exr_path, report
    px = oiio.ImageBuf(exr_path).get_pixels(oiio.FLOAT)
    assert np.median(px[130:190, 200:350]) == pytest.approx(8.0, rel=0.02)


def test_range_line_counts_non_finite_pixels():
    from atlas_camera.comfy.nodes_matrixzone import _stitch_report
    plate = np.full((8, 8, 3), 0.5, np.float32)
    plate[0, 0, 0], plate[1, 1, 1] = np.nan, np.inf
    rep = {"seams_before": [], "anchored": False}
    step = {"seams": [], "worst": None, "flagged": [], "pass": None}
    text = _stitch_report(np, plate, rep, step, n_zones=1, mode="per_zone_clip",
                          colorspace="ACEScg", exr_path="", exr_note="", delivered=None,
                          diff_note="", stripe_rep=None, local_rep=None, xfer_rep=None,
                          destripe=False, detail_from_sdr=False)
    line = [ln for ln in text.splitlines() if ln.startswith("range:")][0]
    assert "2 NON-FINITE pixel value(s)" in line and "max 0.50" in line
