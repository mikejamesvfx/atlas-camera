"""The matrixZone HDR zone ablation: the dump hook, the regions, the variants.

Design: ``docs/designs/matrixzone-hdr-zone-ablation.md``. These are the six
tests it asks for, synthetic and numpy-only except the two that genuinely need
the node (torch-gated, the pattern ``tests/test_matrixzone_nodes.py`` uses).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from atlas_camera.core import matrixzone as mz

REPO = Path(__file__).resolve().parents[1]

H, W = 620, 1100


def _ablation():
    spec = importlib.util.spec_from_file_location(
        "matrixzone_zone_ablation", REPO / "tools" / "matrixzone_zone_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sdr(h=H, w=W, blob=True):
    """A display plate with structure, and a clipped blob so C is not empty."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([0.04 + 0.80 * xx / w,
                    0.04 + 0.80 * yy / h,
                    0.20 + 0.40 * np.sin(xx / 37.0) ** 2], -1).astype(np.float32)
    if blob:
        cy, cx, r = int(h * 0.30), int(w * 0.72), 46
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        rgb[m] = 1.0
    return np.clip(rgb, 0.0, 1.0)


def _global_sdr_numpy(sdr, plan):
    """The split's global clip, block-meaned (a fixture, not the torch path)."""
    r = mz.pad_to_render(sdr, plan)
    gw, gh = plan["global"]["size"]
    ky, kx = r.shape[0] // gh, r.shape[1] // gw
    return r[:gh * ky, :gw * kx].reshape(gh, ky, gw, kx, 3).mean((1, 3)).astype(np.float32)


def _fake_dump(tmp_path, *, zones_from_global=False, perturb=None):
    """A dump directory built with core calls only (no ComfyUI, no torch).

    ``zones_from_global`` makes every zone the crop of the upsampled global --
    exactly what V2 feeds -- so V1 and V2 must agree bit for bit.
    ``perturb`` is ``(zone_index, slice_y, slice_x, gain)``.
    """
    sdr = _sdr()
    plan = mz.plan_still(W, H, (2, 2))
    rw, rh = plan["render"]["width"], plan["render"]["height"]
    render_lin = mz.srgb_to_linear_f32(mz.pad_to_render(sdr, plan))
    # Stand-in for the LTX conversion: a smooth tone expansion, plus a
    # per-zone exposure wobble so the zones are not already identical.
    gl = mz.resize_bilinear(render_lin, plan["global"]["size"][1], plan["global"]["size"][0])
    global_hdr = (gl * 1.9 + gl ** 2 * 6.0).astype(np.float32)
    if zones_from_global:
        up = mz.resize_bilinear(global_hdr, rh, rw)
        zones = [up[y:y + h, x:x + w].copy()
                 for (x, y, w, h) in (z["renderRect"] for z in plan["zones"])]
    else:
        zones = []
        for i, z in enumerate(plan["zones"]):
            x, y, w, h = z["renderRect"]
            a = render_lin[y:y + h, x:x + w]
            zones.append(((a * 1.9 + a ** 2 * 6.0) * (1.0 + 0.04 * i)).astype(np.float32))
    if perturb is not None:
        zi, sy, sx, gain = perturb
        zones[zi] = zones[zi].copy()
        zones[zi][sy, sx] *= gain
    d = tmp_path / "dump"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "global.npy", global_hdr)
    for i, z in enumerate(zones):
        np.save(d / f"zone_{i:02d}.npy", z)
    np.save(d / "sdr.npy", sdr)
    np.save(d / "sdr_global.npy", _global_sdr_numpy(sdr, plan))
    (d / "plan.json").write_text(json.dumps(
        {"kind": "atlas_matrixzone", "plan": plan, "mode": "per_zone_clip",
         "clip_frames": 9, "frame_of_zone": None, "n_global": 1}), encoding="utf-8")
    return d, plan, sdr


# ---------------------------------------------------------------------------
# the `keep` refactor
# ---------------------------------------------------------------------------

def test_keep_mask_is_the_shipped_formula_byte_for_byte():
    """The factored-out mask IS ``sdr_detail_transfer``'s, not a lookalike."""
    sd = _sdr(220, 260)
    # the four lines the refactor replaced, verbatim
    c = mz._box_mean(np, np.clip(sd, 0.0, 1.0).max(-1), 4)
    lo, hi = mz.SDR_TRANSFER_CLIP
    ref = np.clip((c - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    ref = ref * ref * (3 - 2 * ref)
    got = mz.sdr_transfer_keep_mask(sd)
    assert got.dtype == ref.dtype
    assert np.array_equal(got, ref)


def test_sdr_detail_transfer_uses_that_same_mask():
    """``kept_hdr_fraction`` is ``keep > 0.5`` of the public mask, exactly."""
    sd = _sdr(220, 260)
    hdr = mz.rec709_linear_to_acescg(mz.srgb_to_linear_f32(sd)) * 3.0
    _out, rep = mz.sdr_detail_transfer(hdr, sd, radius=16)
    assert rep["kept_hdr_fraction"] == float((mz.sdr_transfer_keep_mask(sd) > 0.5).mean())


def test_keep_mask_rejects_nothing_on_a_dark_plate_and_everything_on_a_white_one():
    assert float(mz.sdr_transfer_keep_mask(np.zeros((80, 90, 3), np.float32)).max()) == 0.0
    assert float(mz.sdr_transfer_keep_mask(np.ones((80, 90, 3), np.float32)).min()) == 1.0


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

def test_regions_are_disjoint_and_cover_the_plate():
    abl = _ablation()
    C, T, U = abl.region_masks(_sdr())
    assert not (C & T).any() and not (C & U).any() and not (T & U).any()
    assert (C | T | U).all()
    assert C.any() and T.any() and U.any()


def test_transition_is_the_guided_filter_support_around_any_influence():
    """T is 128 px of dilation around ``keep > 0``, not around ``keep > 0.5``."""
    abl = _ablation()
    sdr = _sdr()
    keep = mz.sdr_transfer_keep_mask(sdr)
    C, T, _U = abl.region_masks(sdr, transition_px=abl.TRANSITION_PX)
    influenced = keep > 0
    assert (influenced & ~C).any(), "the fixture must leave feather outside C"
    assert (C | T)[influenced].all(), "every influenced pixel is in C or T"
    assert abl.TRANSITION_PX == 2 * mz.SDR_TRANSFER_RADIUS


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

def test_zones_equal_to_the_global_crop_make_v2_identical_to_v1(tmp_path):
    abl = _ablation()
    d, _plan, sdr = _fake_dump(tmp_path, zones_from_global=True)
    data = abl.load_dump(str(d))
    masks = abl.region_masks(sdr)
    v1, _ = abl.run_chain(abl.zone_inputs("V1", data, masks), data)
    v2, _ = abl.run_chain(abl.zone_inputs("V2", data, masks), data)
    assert np.array_equal(v1, v2)


def test_v2_zone_inputs_are_the_upsampled_global_crops(tmp_path):
    abl = _ablation()
    d, plan, sdr = _fake_dump(tmp_path)
    data = abl.load_dump(str(d))
    got = abl.zone_inputs("V2", data, abl.region_masks(sdr))
    up = mz.resize_bilinear(data["global_hdr"], plan["render"]["height"],
                            plan["render"]["width"])
    for z, a in zip(plan["zones"], got):
        x, y, w, h = z["renderRect"]
        assert np.array_equal(a, up[y:y + h, x:x + w])


def test_v3_keeps_the_zone_pixels_inside_the_clipped_core(tmp_path):
    """V3 is zone pixels in dilate(C), the upsampled global outside."""
    abl = _ablation()
    d, plan, sdr = _fake_dump(tmp_path)
    data = abl.load_dump(str(d))
    C, T, U = abl.region_masks(sdr)
    v3 = abl.zone_inputs("V3", data, (C, T, U))
    v2 = abl.zone_inputs("V2", data, (C, T, U))
    # the zone that holds the blob keeps its own pixels at the blob's centre
    core = abl._box_any(np, abl._render_mask(np, C, plan), 1)
    hit = None
    for i, z in enumerate(plan["zones"]):
        x, y, w, h = z["renderRect"]
        if core[y:y + h, x:x + w].mean() > 0.001:
            hit = (i, core[y:y + h, x:x + w])
            break
    assert hit is not None, "the fixture must clip inside at least one zone"
    i, m = hit
    assert np.allclose(v3[i][m], data["zones"][i][m], rtol=1e-5)
    assert not np.allclose(v3[i][m], v2[i][m], rtol=1e-3)
    # and far from C it is the global, not the zone
    far = ~abl._box_any(np, abl._render_mask(np, C, plan), 3 * abl.TRANSITION_PX)
    x, y, w, h = plan["zones"][i]["renderRect"]
    fm = far[y:y + h, x:x + w]
    if fm.any():
        assert np.allclose(v3[i][fm], v2[i][fm], rtol=1e-5)


def test_v4_needs_the_sdr_and_fits_a_monotone_curve(tmp_path):
    abl = _ablation()
    d, _plan, sdr = _fake_dump(tmp_path)
    data = abl.load_dump(str(d))
    masks = abl.region_masks(sdr)
    out = abl.zone_inputs("V4", data, masks)
    assert len(out) == len(data["zones"])
    assert data["reports"]["V4_curve"]["residual_stops"] >= 0.0
    data2 = dict(data, sdr_global=None)
    with pytest.raises(ValueError, match="sdr_global"):
        abl.zone_inputs("V4", data2, masks)


def test_a_perturbation_in_u_is_reported_in_u_not_c_or_t(tmp_path):
    """Region attribution: a gain injected far from the clipped core shows up
    in U alone.

    The band destripe is off here on purpose: its column profile spans a whole
    band, so it would carry a local gain across the plate and the test would be
    measuring the destripe, not the regions.
    """
    abl = _ablation()
    base_d, plan, sdr = _fake_dump(tmp_path / "a")
    masks = abl.region_masks(sdr)
    C, T, U = masks
    # a patch deep inside U: the zone that does NOT hold the blob, and > 2x the
    # guided filter's reach away from any influenced pixel
    far = ~abl._box_any(np, abl._render_mask(np, C | T, plan), 3 * abl.TRANSITION_PX)
    zi = 2
    x, y, w, h = plan["zones"][zi]["renderRect"]
    fm = far[y:y + h, x:x + w]
    ys, xs = np.nonzero(fm)
    assert len(ys) > 10_000, "the fixture must leave a deep-U patch to perturb"
    cy, cx = int(np.median(ys)), int(np.median(xs))
    sy, sx = slice(cy - 30, cy + 30), slice(cx - 30, cx + 30)
    pert_d, _, _ = _fake_dump(tmp_path / "b", perturb=(zi, sy, sx, 1.8))

    base = abl.load_dump(str(base_d))
    pert = abl.load_dump(str(pert_d))
    p0, _ = abl.run_chain(abl.zone_inputs("V1", base, masks), base, destripe=False)
    p1, _ = abl.run_chain(abl.zone_inputs("V1", pert, masks), pert, destripe=False)
    st = abl.region_stops(p1, p0, masks)
    assert st["U"]["max"] > 0.05, st
    assert st["C"]["p99"] < 1e-3, st
    assert st["T"]["p99"] < 1e-3, st


# ---------------------------------------------------------------------------
# cover + decision rule
# ---------------------------------------------------------------------------

def test_crop_cover_covers_the_dilated_core(tmp_path):
    abl = _ablation()
    _d, plan, sdr = _fake_dump(tmp_path)
    C, _T, _U = abl.region_masks(sdr)
    cover = abl.crop_cover(C, plan)
    assert cover["covered"] and cover["crops"] >= 1
    for x, y, w, h in cover["crop_rects"]:
        assert x % 64 == 0 and y % 64 == 0
        assert 0 <= x and x + w <= plan["render"]["width"]
        assert 0 <= y and y + h <= plan["render"]["height"]


def test_decision_rule_is_cheapest_first_and_always_yields_one_outcome():
    abl = _ablation()
    good = {"stops": {"U": {"p99": 0.01}, "T": {"p99": 0.02}}, "seam_worst": 0.5}
    bad = {"stops": {"U": {"p99": 0.9}, "T": {"p99": 0.9}}, "seam_worst": 3.0}
    assert abl.decide({"V4": good, "V2": good, "V3": good})["outcome"].startswith("1")
    assert abl.decide({"V4": bad, "V2": good, "V3": good})["outcome"].startswith("2")
    assert abl.decide({"V4": bad, "V2": bad, "V3": good})["outcome"].startswith("3")
    assert abl.decide({"V4": bad, "V2": bad, "V3": bad})["outcome"].startswith("4")
    assert abl.decide({})["outcome"].startswith("4")


def test_the_pass_bar_is_the_designs_provisional_one():
    abl = _ablation()
    assert (abl.PASS_U_P99, abl.PASS_T_P99, abl.PASS_SEAM_WORST) == (0.05, 0.10, 1.0)


# ---------------------------------------------------------------------------
# the dump hook (needs the node)
# ---------------------------------------------------------------------------

def _node_run(monkeypatch, tmp_path, dump_dir):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_matrixzone
    from atlas_camera.comfy.nodes_matrixzone import (
        AtlasMatrixZoneSplit,
        AtlasMatrixZoneStitch,
    )
    monkeypatch.setattr(nodes_matrixzone, "output_paths", lambda p: (tmp_path, "hdr_00001"))
    if dump_dir is None:
        monkeypatch.delenv(nodes_matrixzone.DUMP_DIR_ENV, raising=False)
    else:
        monkeypatch.setenv(nodes_matrixzone.DUMP_DIR_ENV, str(dump_dir))
    sdr = _sdr()
    plate = torch.from_numpy(sdr)[None]
    clips, handle, _ = AtlasMatrixZoneSplit().split(plate, 2, 2, 64, "per_zone_clip", 9)
    # Stand-in for the LTX chain: a smooth tone expansion with a per-zone wobble.
    out = []
    for i, c in enumerate(clips):
        a = c.numpy()
        g = 1.0 if i == 0 else 1.0 + 0.04 * (i - 1)
        lin = mz.srgb_to_linear_f32(a)
        out.append(torch.from_numpy(((lin * 1.9 + lin ** 2 * 6.0) * g).astype(np.float32)))
    grabbed = {}

    def spy(np_, plate_, colorspace, folder, stem, *, params=None, step_mem=None):
        grabbed["plate"] = np.array(plate_, copy=True)
        return "", ""

    monkeypatch.setattr(nodes_matrixzone, "_write_plate_exr", spy)
    res = AtlasMatrixZoneStitch().stitch(out, [handle], anchor=[True], split_px=[0],
                                         colorspace=["ACEScg"], filename_prefix=["x"],
                                         sdr_plate=[plate])
    return grabbed["plate"], res["result"][2]


def test_with_the_env_var_unset_the_stitch_writes_no_dump(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _plate, report = _node_run(monkeypatch, out, None)
    stray = [p.name for p in tmp_path.rglob("*") if p.suffix in (".npy", ".json")]
    assert stray == [], stray
    assert "dump" not in report.lower()


def test_v1_replayed_from_the_dump_matches_the_live_stitch(monkeypatch, tmp_path):
    """The whole point of the dump: the offline replay IS the delivered plate."""
    abl = _ablation()
    dump = tmp_path / "dump"
    live, report = _node_run(monkeypatch, tmp_path / "out", dump)
    assert "dump:" in report and "FAILED" not in report
    for name in ("global.npy", "zone_00.npy", "zone_03.npy", "sdr.npy",
                 "sdr_global.npy", "plan.json", "params.json"):
        assert (dump / name).exists(), name
    data = abl.load_dump(str(dump))
    masks = abl.region_masks(data["sdr"])
    replay, _ = abl.run_chain(abl.zone_inputs("V1", data, masks), data)
    assert replay.shape == live.shape
    rel = np.abs(replay - live) / np.maximum(np.abs(live), 1e-6)
    assert float(rel.max()) < 1e-5, float(rel.max())


def test_the_dumped_plan_handle_round_trips_through_json(monkeypatch, tmp_path):
    dump = tmp_path / "dump"
    _node_run(monkeypatch, tmp_path / "out", dump)
    raw = json.loads((dump / "plan.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(raw)) == raw
    params = json.loads((dump / "params.json").read_text(encoding="utf-8"))
    for key in ("guided_radius_px", "guided_eps", "sdr_clip_window", "destripe",
                "destripe_local", "detail_from_sdr", "anchor", "split_px"):
        assert key in params["params"], key
    assert params["widgets"]["anchor"] is True


def test_a_dump_failure_never_costs_the_run(monkeypatch, tmp_path):
    """A scratch directory that cannot be written must not lose the plate."""
    from atlas_camera.comfy import nodes_matrixzone
    pytest.importorskip("torch")

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(nodes_matrixzone, "_dump_zone_arrays", boom)
    plate, report = _node_run(monkeypatch, tmp_path / "out", tmp_path / "dump")
    assert plate is not None and plate.shape[:2] == (H, W)
    assert "dump FAILED (zones): OSError: nope" in report


def test_blind_ab_pack_hides_the_labels_and_records_the_key(tmp_path):
    """Outcome 4's protocol: the harness randomises, the key is written apart."""
    abl = _ablation()
    out = tmp_path / "out"
    out.mkdir()
    for v in ("V1", "V2"):
        (out / f"{v}.exr").write_bytes(v.encode())
        for s in (-3, 0, 3):
            (out / f"sweep_{v}_{s:+d}.png").write_bytes(f"{v}{s}".encode())
    key = abl.ab_pack(str(out), ("V1", "V2"), seed=7)
    ab = out / "ab"
    assert sorted(key["A"] + key["B"]) == sorted("V1V2")
    assert key["A"] != key["B"]
    for label in "AB":
        assert (ab / f"{label}.exr").read_bytes() == key[label].encode()
        for s in (-3, 0, 3):
            assert (ab / f"sweep_{label}_{s:+d}.png").exists()
    # nothing in the pack names a variant except the key
    assert not any("V1" in p.name or "V2" in p.name for p in ab.iterdir())
    assert json.loads((ab / "key.json").read_text(encoding="utf-8"))["A"] == key["A"]
