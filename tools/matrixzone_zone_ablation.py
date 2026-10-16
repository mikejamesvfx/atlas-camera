"""What do the matrixZone zones buy, outside the SDR-clipped highlights?

The design is ``docs/designs/matrixzone-hdr-zone-ablation.md`` (APPROVED
2026-10-03). A 2x2 matrixZone SDR->HDR run is one global LTX clip plus four
zone clips, 14 min 27 s cold on a 5090. The delivered plate's last step,
``core.matrixzone.sdr_detail_transfer``, rebuilds every UNclipped pixel as
``sdr * 2**ratio_smooth`` through a guided filter at radius 64 px, so outside
the clipped highlights the zones survive only through that filter's slope and
offset. This harness measures how much of the delivered EXR therefore depends
on the zone clips at all, by replaying the SAME chain from a dumped live run
with the zone inputs replaced:

* **V1 full** -- the dumped zones and global pass, exactly as live (reference).
* **V2 global-ratio** -- every zone slot is the crop of the upsampled global
  HDR at that zone's renderRect. No zone clip contributes anything.
* **V3 global + clipped crops** -- zone pixels inside ``dilate(C, 128)``, the
  upsampled global elsewhere, blended in log2 space with a 128 px feather.
  An UPPER BOUND on approach B: these crop pixels come from the FULL run, and
  fresh smaller crops would see less scene context.
* **V4 global curve** -- a monotone SDR->HDR curve fitted on the global pass
  (``core.hdr_transfer.fit_sdr_to_hdr_curve``) applied to the full-resolution
  SDR outside ``dilate(C, 128)``, zone pixels inside, same feather.

Specification resolved here (the design's reviewer concern R2-1 left it open):
every variant replaces only the ZONE INPUTS and keeps the dumped global pass in
the global slot, so the whole chain downstream -- anchor, log-space feather,
band destripe, local destripe, ``sdr_detail_transfer``, seam step test -- is
byte-for-byte the live one. V4 keeps the global pass because approach C fits
its curve FROM that pass, so a shipped C still runs it.

What the seam score can and cannot detect, per variant: V1 measures real tonal
disagreement between independently converted zones. V2 is seam-free by
construction (one continuous field cropped four ways), so its score only
re-measures the content baseline and must not be read as a quality pass. V3 and
V4 can only show a seam where a seam line crosses ``dilate(C, 128)``; outside
it they are V2-like, and V4's fill is a global function of the SDR, which
cannot produce a seam at all.

Layering: ``core`` + numpy (+ PIL for the sweep PNGs, OpenImageIO for the
optional EXRs). Nothing here imports ``comfy``.

Usage::

    python tools/matrixzone_zone_ablation.py <dump_dir> [--out DIR]
        [--variants V1,V2,V3,V4] [--write-exr] [--sweep-max-px 2560]
        [--per-clip-seconds 170] [--load-seconds 60]

``<dump_dir>`` is what ``AtlasMatrixZoneStitch`` writes with
``ATLAS_MATRIXZONE_DUMP_DIR`` set: ``global.npy``, ``zone_NN.npy``, ``sdr.npy``,
``sdr_global.npy``, ``plan.json`` and ``params.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from atlas_camera.core import matrixzone as mz  # noqa: E402
from atlas_camera.core.hdr_transfer import (  # noqa: E402
    apply_curve,
    fit_sdr_to_hdr_curve,
)

#: The guided filter's 2r support at radius 64: how far a zone pixel can still
#: reach into the delivered plate through ``sdr_detail_transfer``.
TRANSITION_PX = 128

#: Provisional pass bars (the design says so, and says to recalibrate after the
#: first live re-score). Stops of |log2 luminance| against V1.
PASS_U_P99 = 0.05
PASS_T_P99 = 0.10
PASS_SEAM_WORST = 1.0

VARIANTS = ("V1", "V2", "V3", "V4")


# ---------------------------------------------------------------------------
# dump I/O
# ---------------------------------------------------------------------------

def load_dump(folder: str) -> dict[str, Any]:
    """The dumped live run: plan handle, params, global HDR, zone HDRs, SDR."""
    import numpy as np

    def _opt(name):
        p = os.path.join(folder, name)
        return np.load(p) if os.path.exists(p) else None

    with open(os.path.join(folder, "plan.json"), encoding="utf-8") as fh:
        handle = json.load(fh)
    params = None
    pp = os.path.join(folder, "params.json")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as fh:
            params = json.load(fh)
    plan = handle["plan"]
    zones = []
    for i in range(len(plan["zones"])):
        p = os.path.join(folder, f"zone_{i:02d}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: the dump has fewer zones than the plan")
        zones.append(np.load(p))
    return {"handle": handle, "plan": plan, "params": params,
            "global_hdr": np.load(os.path.join(folder, "global.npy")),
            "zones": zones, "sdr": _opt("sdr.npy"), "sdr_global": _opt("sdr_global.npy")}


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

def _box_any(np, mask, r: int):
    """Square dilation by radius ``r``: any True in the (2r+1)^2 window.

    Exact (int64 cumulative sums), and far cheaper on an 8K plate than a
    general morphology call.
    """
    a = mask.astype(np.int64)
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="constant"), axis=axis)
        n = a.shape[axis]
        hi, lo = [slice(None)] * a.ndim, [slice(None)] * a.ndim
        hi[axis], lo[axis] = slice(k, k + n), slice(0, n)
        a = c[tuple(hi)] - c[tuple(lo)]
    return a > 0


def region_masks(sdr_display, *, transition_px: int = TRANSITION_PX):
    """``(C, T, U)`` at plate resolution: clipped core, transition, unclipped.

    ``C`` is ``sdr_transfer_keep_mask > 0.5`` -- the same feathered mask
    ``sdr_detail_transfer`` uses to decide where the HDR's own pixels are kept,
    so C is exactly "the pixels the conversion still decides". ``T`` is
    everything within ``transition_px`` of ANY feathered influence
    (``keep > 0``) that is not C: the band the guided filter can still carry a
    zone's slope and offset into. ``U`` is the rest. Disjoint, and they cover
    the plate.
    """
    import numpy as np
    keep = mz.sdr_transfer_keep_mask(sdr_display)
    C = keep > 0.5
    T = _box_any(np, keep > 0, int(transition_px)) & ~C
    U = ~(C | T)
    return C, T, U


def _box_mean(np, a, r: int):
    """Separable box mean of radius ``r`` (edge-padded), via cumulative sums."""
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = a.shape[axis]
        hi, lo = [slice(None)] * a.ndim, [slice(None)] * a.ndim
        hi[axis], lo[axis] = slice(k, k + n), slice(0, n)
        a = ((c[tuple(hi)] - c[tuple(lo)]) / k).astype(np.float32)
    return a


def _feather_weight(np, mask, r: int):
    """0..1 ramp: 1 well inside ``mask``, 0 well outside, over about ``2r`` px."""
    return np.clip(_box_mean(np, mask.astype(np.float32), int(r)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

def _global_at_render(np, global_hdr, plan):
    rw, rh = plan["render"]["width"], plan["render"]["height"]
    return mz.resize_bilinear(np.asarray(global_hdr, dtype=np.float32), rh, rw)


def _render_mask(np, plate_mask, plan):
    """A plate-space mask placed into the render raster (padding edge-extended)."""
    rw, rh = plan["render"]["width"], plan["render"]["height"]
    L, T = plan["render"]["plateOrigin"]
    ph, pw = plate_mask.shape[:2]
    out = np.zeros((rh, rw), dtype=bool)
    out[T:T + ph, L:L + pw] = plate_mask
    if T:
        out[:T] = out[T:T + 1]
    if T + ph < rh:
        out[T + ph:] = out[T + ph - 1:T + ph]
    if L:
        out[:, :L] = out[:, L:L + 1]
    if L + pw < rw:
        out[:, L + pw:] = out[:, L + pw - 1:L + pw]
    return out


def _log_blend(np, inside, outside, w):
    """``inside`` where ``w`` is 1, ``outside`` where 0, blended in log2."""
    a = mz.to_log2(np.asarray(inside, dtype=np.float32))
    b = mz.to_log2(np.asarray(outside, dtype=np.float32))
    return np.exp2(a * w[..., None] + b * (1.0 - w[..., None]))


def zone_inputs(variant: str, data: dict[str, Any], masks) -> list[Any]:
    """The four arrays fed into the stitch's zone slots for ``variant``."""
    import numpy as np
    plan, zones = data["plan"], data["zones"]
    if variant == "V1":
        return list(zones)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    C, _T, _U = masks
    if variant == "V2":
        grender = _global_at_render(np, data["global_hdr"], plan)
        return [grender[y:y + h, x:x + w].copy()
                for (x, y, w, h) in (z["renderRect"] for z in plan["zones"])]

    # V3 / V4: keep the LTX zone pixels inside dilate(C, 128), fill outside.
    keepmask = _box_any(np, _render_mask(np, C, plan), TRANSITION_PX)
    w_full = _feather_weight(np, keepmask, TRANSITION_PX // 2)
    del keepmask
    if variant == "V3":
        fill = _global_at_render(np, data["global_hdr"], plan)
    else:
        if data["sdr"] is None or data["sdr_global"] is None:
            raise ValueError("V4 needs sdr.npy and sdr_global.npy in the dump")
        curve = fit_sdr_to_hdr_curve(data["sdr_global"], data["global_hdr"])
        data.setdefault("reports", {})["V4_curve"] = {
            "residual_stops": curve["residual_stops"], "samples": curve["samples"],
            "bins": curve["bins"]}
        sdr_render = mz.pad_to_render(np.asarray(data["sdr"], dtype=np.float32), plan)
        # In row bands: apply_curve interpolates in float64, so a whole 8K
        # render at once costs a couple of GB of temporaries for nothing.
        fill = np.empty(sdr_render.shape[:2] + (3,), dtype=np.float32)
        for y0 in range(0, fill.shape[0], 512):
            y1 = min(y0 + 512, fill.shape[0])
            fill[y0:y1] = apply_curve(curve, sdr_render[y0:y1, :, :3])
        del sdr_render
    out = []
    for z, img in zip(plan["zones"], zones):
        x, y, w, h = z["renderRect"]
        out.append(_log_blend(np, img, fill[y:y + h, x:x + w],
                              w_full[y:y + h, x:x + w]))
    return out


# ---------------------------------------------------------------------------
# the chain (identical to AtlasMatrixZoneStitch's, core calls only)
# ---------------------------------------------------------------------------

def run_chain(zone_hdr, data: dict[str, Any], *, anchor=None, split_px=None,
              destripe=None, detail_from_sdr=None):
    """Stitch + band destripe + local destripe + SDR detail transfer + seam test.

    Defaults come from the dump's ``params.json`` so the replay uses the values
    the live run used, not this file's.
    """
    import numpy as np
    plan = data["plan"]
    w = (data.get("params") or {}).get("widgets") or {}
    p = (data.get("params") or {}).get("params") or {}
    anchor = w.get("anchor", True) if anchor is None else anchor
    split_px = w.get("split_px", 0) if split_px is None else split_px
    destripe = p.get("destripe", True) if destripe is None else destripe
    detail_from_sdr = p.get("detail_from_sdr", True) if detail_from_sdr is None else detail_from_sdr

    plate, rep = mz.stitch(zone_hdr, plan,
                           global_hdr=np.asarray(data["global_hdr"], dtype=np.float32)[..., :3]
                           if anchor else None,
                           split_px=int(split_px) or None)
    s_disp = data["sdr"]
    sdr_lin = None if s_disp is None else mz.srgb_to_linear_f32(
        np.asarray(s_disp, dtype=np.float32)[..., :3])
    reports = {"stitch": rep}
    if destripe and sdr_lin is not None:
        plate, reports["destripe"] = mz.destripe_columns(
            plate, sdr_lin, bands=mz.zone_row_bands(plan))
        plate, reports["destripe_local"] = mz.destripe_local(plate, sdr_lin)
    if detail_from_sdr and s_disp is not None:
        plate, reports["transfer"] = mz.sdr_detail_transfer(plate, s_disp)
    reports["seam"] = mz.seam_step_test(plate, plan, sdr_linear=sdr_lin)
    return plate, reports


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _log2_lum(np, a):
    lum = np.tensordot(np.asarray(a, dtype=np.float32)[..., :3],
                       np.asarray(mz.LUMA_AP1, dtype=np.float32), axes=([-1], [0]))
    return np.log2(np.maximum(lum, mz.LOG_EPS))


def region_stops(plate, reference, masks) -> dict[str, Any]:
    """|log2 luminance| difference against the reference, per region."""
    import numpy as np
    d = np.abs(_log2_lum(np, plate) - _log2_lum(np, reference))
    C, T, U = masks
    out = {}
    for name, m in (("C", C), ("T", T), ("U", U), ("all", None)):
        v = d if m is None else d[m]
        if v.size == 0:
            out[name] = {"px": 0}
            continue
        p50, p95, p99 = (float(x) for x in np.percentile(v, [50, 95, 99]))
        out[name] = {"px": int(v.size), "p50": p50, "p95": p95, "p99": p99,
                     "max": float(v.max()),
                     # How much of the region actually moved: a percentile says
                     # how big the differences are, not how widespread. 0.05 is
                     # the pass bar, 0.1 stops is about a grade nudge.
                     "frac_over_0.05": float((v > 0.05).mean()),
                     "frac_over_0.10": float((v > 0.10).mean())}
    return out


def seam_worst(seam_report) -> float | None:
    """The worst windowed seam ratio in a ``seam_step_test`` report."""
    if not isinstance(seam_report, dict):
        return None
    worst = seam_report.get("worst")
    if isinstance(worst, dict) and worst.get("ratio") is not None:
        return float(worst["ratio"])
    vals = [s.get("ratio") for s in (seam_report.get("seams") or [])
            if isinstance(s, dict) and s.get("ratio") is not None]
    return max(float(v) for v in vals) if vals else None


# ---------------------------------------------------------------------------
# crop cover accounting (approach B, accounting only -- a shipped B would add
# the mode to core.matrixzone.plan_still and pin parity there)
# ---------------------------------------------------------------------------

def crop_cover(C, plan, *, dilate_px: int = TRANSITION_PX, clean: int = 64,
               overlap: int = TRANSITION_PX) -> dict[str, Any]:
    """Greedy cover of ``dilate(C)`` with zone-tier crops on 64-clean origins."""
    import numpy as np
    rw, rh = plan["render"]["width"], plan["render"]["height"]
    zw, zh = plan["zone_size"]
    need = _box_any(np, _render_mask(np, C, plan), int(dilate_px))
    todo = need.copy()
    crops = []
    guard = 0
    while todo.any() and guard < 512:
        guard += 1
        ys, xs = np.nonzero(todo)
        # first uncovered pixel in scan order; centre a crop on it
        y0, x0 = int(ys[0]), int(xs[0])
        x = min(max(0, x0 - zw // 2), max(0, rw - zw))
        y = min(max(0, y0 - zh // 2), max(0, rh - zh))
        x -= x % clean
        y -= y % clean
        if y + zh <= y0:                      # the snap must still cover the pixel
            y = min(max(0, y0 - zh + clean), max(0, rh - zh))
        if x + zw <= x0:
            x = min(max(0, x0 - zw + clean), max(0, rw - zw))
        crops.append([x, y, zw, zh])
        todo[y:y + zh, x:x + zw] = False
    area = sum(c[2] * c[3] for c in crops)
    return {"crops": len(crops), "crop_rects": crops, "zone_size": [zw, zh],
            "overlap_px": int(overlap),
            "need_fraction_of_render": float(need.mean()),
            "crop_area_vs_render": float(area / max(rw * rh, 1)),
            "covered": bool(not todo.any())}


def runtime_model(crops: int, *, load_s: float, global_s: float,
                  per_clip_s: float, zones: int) -> dict[str, Any]:
    """Live time vs projected approach-B time. Not scaled by crop area."""
    live = load_s + global_s + zones * per_clip_s
    proj = load_s + global_s + crops * per_clip_s
    return {"live_s": live, "projected_s": proj,
            "fraction_of_live": (proj / live) if live else None,
            "crops": crops, "per_clip_s": per_clip_s, "load_s": load_s,
            "global_s": global_s}


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _srgb_encode(np, lin):
    x = np.clip(lin, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055)


def write_sweep(plate, out_dir: str, tag: str, *, stops=(-3, 0, 3), max_px: int = 2560):
    """Exposure sweep PNGs for the eyes-on half of the pass bar."""
    import numpy as np
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    h, w = plate.shape[:2]
    k = max(1, int(max(h, w) // max_px))
    small = plate[::k, ::k] if k > 1 else plate
    paths = []
    for s in stops:
        img = (_srgb_encode(np, np.asarray(small, dtype=np.float32) * (2.0 ** s))
               * 255.0 + 0.5).astype(np.uint8)
        p = os.path.join(out_dir, f"sweep_{tag}_{s:+d}.png")
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


def write_exr(plate, out_dir: str, tag: str):
    try:
        from atlas_camera.plate.oiio_io import write_exr as _w
    except Exception as exc:                                    # noqa: BLE001
        return f"EXR not written ({type(exc).__name__}: {exc})"
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{tag}.exr")
    try:
        _w(p, plate, bit_depth="half", extra_attribs={
            "oiio:ColorSpace": "ACEScg",
            "atlas:content": "MODEL RECONSTRUCTION of highlight radiance from a "
                             f"display-referred plate (zone ablation {tag})"})
        return p
    except Exception as exc:                                    # noqa: BLE001
        return f"EXR not written ({type(exc).__name__}: {exc})"


def ab_pack(out_dir: str, pair=("V1", "V2"), *, seed: int | None = None) -> dict[str, Any]:
    """A blind A/B pack for the design's outcome 4 (review R2-9's protocol).

    Copies two variants' EXRs and sweeps to ``ab/A*`` and ``ab/B*`` with the
    assignment drawn here, and writes ``ab/key.json``. The harness randomises
    and hides the labels so the same person can judge: look at A and B, write
    the verdict down, THEN read the key.
    """
    import random
    import shutil
    rnd = random.Random(seed)
    order = list(pair)
    rnd.shuffle(order)
    ab = os.path.join(out_dir, "ab")
    os.makedirs(ab, exist_ok=True)
    key = {"A": order[0], "B": order[1], "seed": seed, "files": {}}
    for label, variant in zip("AB", order):
        for src, dst in [(f"{variant}.exr", f"{label}.exr")] + [
                (f"sweep_{variant}_{s:+d}.png", f"sweep_{label}_{s:+d}.png")
                for s in (-3, 0, 3)]:
            s_path = os.path.join(out_dir, src)
            if os.path.exists(s_path):
                shutil.copyfile(s_path, os.path.join(ab, dst))
                key["files"].setdefault(label, []).append(dst)
    with open(os.path.join(ab, "key.json"), "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)
    return key


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------

def passes(stats: dict[str, Any], worst: float | None) -> tuple[bool, list[str]]:
    """The design's PROVISIONAL numeric bar. The eyes-on half is not here."""
    why = []
    u = stats["U"].get("p99")
    t = stats["T"].get("p99")
    if u is None or u > PASS_U_P99:
        why.append(f"U p99 {u} > {PASS_U_P99}")
    if t is None or t > PASS_T_P99:
        why.append(f"T p99 {t} > {PASS_T_P99}")
    if worst is not None and worst > PASS_SEAM_WORST:
        why.append(f"seam worst {worst:.2f}x > {PASS_SEAM_WORST}x")
    return (not why), why


def decide(results: dict[str, Any]) -> dict[str, Any]:
    """Cheapest first. Numeric half only -- the human verdict is recorded next
    to these numbers in the proposal, not computed here (review R2-10)."""
    out = {"numeric": {}}
    for v in ("V4", "V2", "V3"):
        r = results.get(v)
        if not r:
            continue
        ok, why = passes(r["stops"], r.get("seam_worst"))
        out["numeric"][v] = {"pass": ok, "why_not": why}
    if out["numeric"].get("V4", {}).get("pass"):
        out["outcome"] = "1: V4 passes -> build approach C (curve outside C)"
    elif out["numeric"].get("V2", {}).get("pass"):
        out["outcome"] = "2: V2 passes -> build the global-ratio + clipped-crops mode"
    elif out["numeric"].get("V3", {}).get("pass"):
        out["outcome"] = ("3: V3 passes -> build B IF the projected time bar and one live "
                          "scheduled-crop check hold")
    else:
        out["outcome"] = ("4: no variant passes -> the zones change the output materially; "
                          "blind A/B of V1 vs V2 in Nuke before concluding")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default=None, help="output directory (default: <dump_dir>/ablation)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--write-exr", action="store_true")
    ap.add_argument("--sweep-max-px", type=int, default=2560)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--ab", default=None,
                    help="two variants, e.g. V1,V2: also write a blind A/B pack")
    ap.add_argument("--load-seconds", type=float, default=0.0)
    ap.add_argument("--global-seconds", type=float, default=0.0)
    ap.add_argument("--per-clip-seconds", type=float, default=0.0)
    args = ap.parse_args(argv)

    import numpy as np
    out_dir = args.out or os.path.join(args.dump_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    data = load_dump(args.dump_dir)
    plan = data["plan"]
    if data["sdr"] is None:
        print("REFUSED: sdr.npy is missing from the dump; every region mask and every "
              "clean-up pass needs the SDR plate the split was given.")
        return 2
    masks = region_masks(data["sdr"])
    C, T, U = masks
    overlap = int((C & T).sum() + (C & U).sum() + (T & U).sum())
    print(f"plate {plan['plate']['width']}x{plan['plate']['height']}  render "
          f"{plan['render']['width']}x{plan['render']['height']}  zones {len(plan['zones'])} "
          f"{plan['zone_size']}  global {plan['global']['size']}  mode {data['handle']['mode']}")
    print(f"regions: C {C.mean():.4%}  T {T.mean():.4%}  U {U.mean():.4%}  "
          f"overlap {overlap} px  covered {bool((C | T | U).all())}")

    wanted = [v for v in args.variants.split(",") if v.strip()]
    if "V1" in wanted:                       # the reference goes first and is kept
        wanted = ["V1"] + [v for v in wanted if v != "V1"]
    results: dict[str, Any] = {}
    ref = None
    # One variant plate at a time beside the reference: four 8K float32 plates
    # are 1.7 GB of nothing useful once their numbers are out.
    for v in wanted:
        tv = time.time()
        plate, reports = run_chain(zone_inputs(v, data, masks), data)
        results[v] = {"seconds": round(time.time() - tv, 1),
                      "seam_worst": seam_worst(reports["seam"]),
                      "kept_hdr_fraction": (reports.get("transfer") or {}).get(
                          "kept_hdr_fraction"),
                      "transfer_change_p99_stops": (reports.get("transfer") or {}).get(
                          "change_p99_stops"),
                      "seam_report": reports["seam"]}
        if v == "V1":
            ref = plate
        results[v]["stops"] = {} if ref is None else region_stops(plate, ref, masks)
        print(f"  {v}: {results[v]['seconds']}s  seam worst "
              f"{results[v]['seam_worst']}  kept_hdr {results[v]['kept_hdr_fraction']}")
        if not args.no_sweep:
            write_sweep(plate, out_dir, v, max_px=args.sweep_max_px)
        if args.write_exr:
            results[v]["exr"] = write_exr(plate, out_dir, v)
        if v != "V1":
            del plate
    if ref is None:
        print("no V1 reference in --variants: nothing was compared")

    ab = None
    if args.ab:
        ab = ab_pack(out_dir, tuple(v.strip() for v in args.ab.split(",")[:2]))
        print(f"blind A/B pack in {os.path.join(out_dir, 'ab')} "
              f"({len(ab['files'].get('A', []))} file(s) per side; key.json holds the "
              "assignment -- judge first, read it after)")
    cover = crop_cover(C, plan)
    rt = None
    if args.per_clip_seconds:
        rt = runtime_model(cover["crops"], load_s=args.load_seconds,
                           global_s=args.global_seconds or args.per_clip_seconds,
                           per_clip_s=args.per_clip_seconds, zones=len(plan["zones"]))
    dec = decide({v: results[v] for v in wanted if v != "V1"})

    print("\n|log2 luminance| vs V1, stops (p50 / p95 / p99 / max)")
    print("| variant | C | T | U |")
    print("|---|---|---|---|")
    for v in wanted:
        if v == "V1":
            continue
        s = results[v]["stops"]
        cells = []
        for r in ("C", "T", "U"):
            q = s.get(r, {})
            cells.append("n/a" if not q.get("px") else
                         f"{q['p50']:.4f} / {q['p95']:.4f} / {q['p99']:.4f} / {q['max']:.3f}")
        print(f"| {v} | " + " | ".join(cells) + " |")
    print("\nshare of each region moved by more than 0.05 / 0.10 stops")
    print("| variant | C | T | U |")
    print("|---|---|---|---|")
    for v in wanted:
        if v == "V1":
            continue
        s = results[v]["stops"]
        cells = []
        for r in ("C", "T", "U"):
            q = s.get(r, {})
            cells.append("n/a" if not q.get("px") else
                         f"{q['frac_over_0.05']:.1%} / {q['frac_over_0.10']:.1%}")
        print(f"| {v} | " + " | ".join(cells) + " |")
    print(f"\ncrop cover of dilate(C,{TRANSITION_PX}): {cover['crops']} crop(s) at "
          f"{cover['zone_size']}, need {cover['need_fraction_of_render']:.2%} of the render, "
          f"crop area {cover['crop_area_vs_render']:.2f}x the render")
    if rt:
        print(f"runtime model: live {rt['live_s']:.0f}s -> projected {rt['projected_s']:.0f}s "
              f"({rt['fraction_of_live']:.2f}x of live)")
    for v, d in dec["numeric"].items():
        print(f"bar {v}: {'PASS' if d['pass'] else 'FAIL'}"
              + ("" if d["pass"] else "  (" + "; ".join(d["why_not"]) + ")"))
    print(f"\nOUTCOME (numeric half) -> {dec['outcome']}")
    print(f"thresholds are PROVISIONAL: U p99 <= {PASS_U_P99}, T p99 <= {PASS_T_P99}, "
          f"seam worst <= {PASS_SEAM_WORST}x")

    payload = {"dump_dir": os.path.abspath(args.dump_dir), "plan": plan,
               "params": data.get("params"),
               "regions": {"C": float(C.mean()), "T": float(T.mean()), "U": float(U.mean()),
                           "disjoint": overlap == 0, "covers": bool((C | T | U).all()),
                           "transition_px": TRANSITION_PX},
               "variants": results, "cover": cover, "runtime": rt, "decision": dec,
               "ab_pack": ab,
               "extra": data.get("reports", {}),
               "harness_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {os.path.join(out_dir, 'ablation.json')} "
          f"({payload['harness_seconds']}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
