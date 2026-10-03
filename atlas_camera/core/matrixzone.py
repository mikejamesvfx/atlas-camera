"""matrixZone for a STILL plate: exact crops, a radiance anchor, a log-space stitch.

The design of record is ``atlas-unreal/Docs/ATLAS_MATRIXZONE.md`` (rev 3,
schema ``matrixZone/1.2``) and its planner ``atlas_bridge.matrixzone.plan()``.
The pixel arithmetic here is a port of that planner's -- padded render
(128-clean), one common 64-clean zone size, exterior edges flush to the render
edge, interior slack becoming overlap -- and ``tests/test_matrixzone.py`` pins it
to the bridge's worked numbers so the two repos cannot drift. The USD aperture
half of the bridge planner is not needed for a still and is not ported.

What is NEW here is the use: an SDR->HDR CONVERSION (LTX-2.5 IC-LoRA) per
zone. A conversion invents no objects, so zones cannot disagree about what is
in the frame -- but each zone's highlight reconstruction is the model's guess
about how BRIGHT it is, and neighbours can disagree. So global-first becomes a
radiance anchor: a whole-frame low-tier conversion supplies every zone's low
frequencies (in log2 radiance), the zones supply only detail, and the zones are
feathered together in log space.

Conventions: arrays are (H, W, C) float, rects are ``[x, y, w, h]`` in RENDER
pixels, ``plateOrigin`` locates the plate inside the render. Nothing is
resampled by a non-integer factor except the anchor's deliberately low-pass
global upsample.

Layering: ``core`` only -- numpy, no torch, no ComfyUI.
"""

from __future__ import annotations

import math
from typing import Any

CLEAN = 128          # render axes: quarter-linear global stays 32-clean
ZONE_CLEAN = 64      # zone sizes
LOG_EPS = 1e-6       # radiance floor before log2

#: Luminance weights for ACEScg (AP1) linear -- the space LTX-2.5 ``hdr_linear``
#: and every stitched plate here are in. Rec.709 weights on AP1 data mis-weight
#: saturated colours (F-5, 2026-10-03 review).
LUMA_AP1 = (0.2722287168, 0.6740817658, 0.0536895174)


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matrixZone requires numpy.") from exc


def _rec709_to_acescg_matrix(np):
    # The same Bradford D65->D60 matrix core.hdr_transfer fits its curve in,
    # so the SDR is compared with the HDR in ONE space everywhere.
    from atlas_camera.core.hdr_transfer import REC709_TO_ACESCG
    return np.asarray(REC709_TO_ACESCG, dtype=np.float32)


def rec709_linear_to_acescg(a: Any) -> Any:
    """Linear Rec.709/sRGB (D65) RGB -> ACEScg (AP1, D60) linear, float32."""
    np = _require_numpy()
    return np.asarray(a, dtype=np.float32)[..., :3] @ _rec709_to_acescg_matrix(np).T


def _luma_ap1(np):
    return np.asarray(LUMA_AP1, dtype=np.float32)


def _luma_ap1_of_rec709(np):
    """Weights giving the AP1 luminance of a Rec.709-linear pixel directly
    (``LUMA_AP1 . M``), so luminance-only passes need no full-plate convert."""
    return (_rec709_to_acescg_matrix(np).T @ _luma_ap1(np)).astype(np.float32)


# ---------------------------------------------------------------------------
# Plan (port of atlas_bridge.matrixzone.plan_render / _cells / _axis)
# ---------------------------------------------------------------------------

def plan_render(plate_width: int, plate_height: int,
                margin: tuple[int, int, int, int] = (0, 0, 0, 0),
                clean: int = CLEAN) -> dict[str, Any]:
    """Plate grown by ``margin`` (L, T, R, B) and padded to ``clean`` on both
    axes, split as evenly as possible. ``plateOrigin == overscan[:2]``."""
    left, top, right, bottom = (int(m) for m in margin)
    width = plate_width + left + right
    height = plate_height + top + bottom
    padw = (-width) % clean
    padh = (-height) % clean
    left += padw // 2
    right += padw - padw // 2
    top += padh // 2
    bottom += padh - padh // 2
    return {"width": plate_width + left + right, "height": plate_height + top + bottom,
            "plateOrigin": [left, top], "overscan": [left, top, right, bottom]}


def _cells(total: int, n: int) -> list[tuple[int, int]]:
    base = total // n
    return [(i * base, base if i < n - 1 else total - i * base) for i in range(n)]


def _axis(cells, origin, ext_before, ext_after, overlap_min):
    n = len(cells)
    half = int(math.ceil(overlap_min / 2.0))
    need = 0
    for i, (_s, size) in enumerate(cells):
        a = ext_before if i == 0 else half
        b = ext_after if i == n - 1 else half
        need = max(need, a + size + b)
    zsize = int(math.ceil(need / ZONE_CLEAN)) * ZONE_CLEAN
    total = ext_before + sum(c[1] for c in cells) + ext_after
    out = []
    for i, (start, size) in enumerate(cells):
        if i == 0:
            zstart = 0
        elif i == n - 1:
            zstart = total - zsize
        else:
            # Centred on its cell, then clamped into the render: on a tall/narrow
            # grid the 64-clean zone can be much larger than its cell, and the
            # centred start then fell off the canvas (3840x2160, 1x8, overlap
            # 512: z10 at y=-3, z60 ending at 2179 > 2176). Clamping keeps the
            # cell covered -- the zone only slides away from the edge it crossed.
            # Not mirrored in atlas_bridge yet (TODOS.md).
            zstart = origin + start - (zsize - size) // 2
            zstart = min(max(zstart, 0), max(total - zsize, 0))
        out.append((zstart, zsize))
    return out, zsize


def plan_still(plate_width: int, plate_height: int, grid: tuple[int, int], *,
               overlap_min: tuple[int, int] = (64, 64),
               margin: tuple[int, int, int, int] = (0, 0, 0, 0)) -> dict[str, Any]:
    """A matrixZone plan for one still plate.

    ``zones`` are row-major (``z<row><col>``), each with its ``renderRect`` (the
    crop the model converts) and ``plateRect`` (the cell it owns). ``global`` is
    the whole render at a quarter linear (32-clean by construction).
    ``scan`` is the serpentine order -- consecutive entries are spatial
    neighbours -- used when the zones are fed to the model as frames.
    """
    cols, rows = int(grid[0]), int(grid[1])
    if cols < 1 or rows < 1:
        raise ValueError("grid must be at least 1x1")
    render = plan_render(plate_width, plate_height, margin)
    rw, rh = render["width"], render["height"]
    L, T, R, B = render["overscan"]
    xcells, ycells = _cells(plate_width, cols), _cells(plate_height, rows)
    xz, zw = _axis(xcells, L, L, R, overlap_min[0])
    yz, zh = _axis(ycells, T, T, B, overlap_min[1])
    if zw > rw or zh > rh:
        raise ValueError(f"zone {zw}x{zh} exceeds the render {rw}x{rh}: use a smaller grid")
    zones = []
    for row in range(rows):
        for col in range(cols):
            x, w = xz[col]
            y, h = yz[row]
            px, pw = xcells[col]
            py, ph = ycells[row]
            if x < 0 or y < 0 or x + w > rw or y + h > rh:   # planner invariant
                raise RuntimeError(f"matrixZone planner bug: zone z{row}{col} renderRect "
                                   f"{[x, y, w, h]} leaves the render {rw}x{rh}")
            zones.append({"id": f"z{row}{col}", "index": [col, row],
                          "plateRect": [L + px, T + py, pw, ph],
                          "renderRect": [x, y, w, h]})
    scan = []
    for row in range(rows):
        cols_order = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        scan.extend(row * cols + c for c in cols_order)
    seams_x = [xz[i][0] + xz[i][1] - xz[i + 1][0] for i in range(cols - 1)]
    seams_y = [yz[i][0] + yz[i][1] - yz[i + 1][0] for i in range(rows - 1)]
    return {
        "schema": "atlasMatrixZoneStill/1",
        "grid": [cols, rows],
        "plate": {"width": int(plate_width), "height": int(plate_height)},
        "render": render,
        "zone_size": [zw, zh],
        "global": {"size": [rw // 4, rh // 4]},
        "overlap": {"px": [min(seams_x) if seams_x else 0, min(seams_y) if seams_y else 0],
                    "edges": "interior", "blend": "feather-log2"},
        "zones": zones,
        "scan": scan,
    }


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------

def pad_to_render(plate: Any, plan: dict[str, Any]) -> Any:
    """Edge-extend the plate into the planned render canvas (a still has no
    renderer overscan to crop from; the ring is invented, and it is cropped
    away again after the stitch)."""
    np = _require_numpy()
    a = np.asarray(plate)
    L, T, R, B = plan["render"]["overscan"]
    pad = [(T, B), (L, R)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, mode="edge")


def crop_zones(render_img: Any, plan: dict[str, Any]) -> list[Any]:
    """Each zone's renderRect, in plan (row-major) order."""
    out = []
    for z in plan["zones"]:
        x, y, w, h = z["renderRect"]
        out.append(render_img[y:y + h, x:x + w])
    return out


def frames_8k1(n: int) -> int:
    """Smallest frame count >= n with count % 8 == 1."""
    return n if n % 8 == 1 else n + (1 - n % 8) % 8


def zones_as_frames(zone_imgs: list[Any], plan: dict[str, Any]) -> tuple[Any, list[int]]:
    """Zones stacked in serpentine order, padded to 8k+1 frames by ping-pong.
    Returns ``(frames (F,H,W,C), frame_of_zone)``."""
    np = _require_numpy()
    order = list(plan["scan"])
    # Ping-pong along the scan so every consecutive pair stays a spatial
    # neighbour even in the padding: 0 1 2 3 2 1 0 1 2 ...
    cycle = order + order[-2:0:-1] if len(order) > 2 else order
    seq = [cycle[k % len(cycle)] for k in range(frames_8k1(len(order)))]
    frame_of_zone = [order.index(i) for i in range(len(order))]
    return np.stack([np.asarray(zone_imgs[i]) for i in seq]), frame_of_zone


# --- resampling helpers (numpy, separable bilinear) --------------------------

def _resize_axis(np, a, n, axis):
    m = a.shape[axis]
    if m == n:
        return a
    pos = (np.arange(n) + 0.5) * (m / n) - 0.5
    pos = np.clip(pos, 0, m - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, m - 1)
    t = pos - i0
    shape = [1] * a.ndim
    shape[axis] = n
    t = t.reshape(shape)
    return np.take(a, i0, axis=axis) * (1 - t) + np.take(a, i1, axis=axis) * t


def resize_bilinear(a: Any, height: int, width: int) -> Any:
    np = _require_numpy()
    a = np.asarray(a, dtype=np.float32)
    return _resize_axis(np, _resize_axis(np, a, int(height), 0), int(width), 1)


def lowpass(a: Any, k: int) -> Any:
    """Block-mean downsample by ``k`` then bilinear back: frequencies below
    ~1/k cycles per pixel."""
    np = _require_numpy()
    a = np.asarray(a, dtype=np.float32)
    k = max(1, int(k))
    h, w = a.shape[:2]
    hp, wp = -h % k, -w % k
    p = np.pad(a, [(0, hp), (0, wp)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    small = p.reshape(p.shape[0] // k, k, p.shape[1] // k, k, *p.shape[2:]).mean(axis=(1, 3))
    return resize_bilinear(small, h + hp, w + wp)[:h, :w]


def to_log2(a: Any) -> Any:
    np = _require_numpy()
    return np.log2(np.maximum(np.asarray(a, dtype=np.float32), LOG_EPS))


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------

def _feather(np, length: int, before: int, after: int) -> Any:
    """Weight along one axis: ramp up over ``before`` px, down over ``after``."""
    w = np.ones(length, dtype=np.float32)
    if before > 0:
        w[:before] = (np.arange(before, dtype=np.float32) + 0.5) / before
    if after > 0:
        w[length - after:] = np.minimum(w[length - after:],
                                        (np.arange(after, 0, -1, dtype=np.float32) - 0.5) / after)
    return w


def _overlaps(plan):
    """Per zone: interior overlap (px) shared with the left/top/right/bottom
    neighbour; 0 on exterior edges."""
    cols, rows = plan["grid"]
    rects = {tuple(z["index"]): z["renderRect"] for z in plan["zones"]}
    out = []
    for z in plan["zones"]:
        c, r = z["index"]
        x, y, w, h = z["renderRect"]
        left = rects[(c - 1, r)][0] + rects[(c - 1, r)][2] - x if c > 0 else 0
        right = x + w - rects[(c + 1, r)][0] if c < cols - 1 else 0
        top = rects[(c, r - 1)][1] + rects[(c, r - 1)][3] - y if r > 0 else 0
        bottom = y + h - rects[(c, r + 1)][1] if r < rows - 1 else 0
        out.append((left, top, right, bottom))
    return out


def anchor_zone(zone_log: Any, global_log_render: Any, rect, split_px: int) -> Any:
    """Replace the zone's low frequencies (log2) with the global pass's."""
    x, y, w, h = rect
    g = global_log_render[y:y + h, x:x + w]
    return zone_log + (lowpass(g, split_px) - lowpass(zone_log, split_px))


def stitch(zone_hdr: list[Any], plan: dict[str, Any], *, global_hdr: Any = None,
           split_px: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Place every zone's HDR result back, feathered in log2 space.

    ``zone_hdr`` is in plan order, each (h, w, C) linear radiance at its
    renderRect size. ``global_hdr`` (any resolution, whole render) enables the
    radiance anchor. Returns ``(plate (H, W, C) linear, report)`` with the
    padding already cropped away.
    """
    np = _require_numpy()
    rw, rh = plan["render"]["width"], plan["render"]["height"]
    if len(zone_hdr) != len(plan["zones"]):
        raise ValueError(f"{len(zone_hdr)} zone results for {len(plan['zones'])} zones")
    c = np.asarray(zone_hdr[0]).shape[-1]
    split = int(split_px or max(plan["overlap"]["px"]) or 64)
    glog = None
    if global_hdr is not None:
        glog = to_log2(resize_bilinear(np.asarray(global_hdr, dtype=np.float32), rh, rw))

    for z, img in zip(plan["zones"], zone_hdr):
        a = np.asarray(img)
        if a.size == 0 or a.ndim < 2 or 0 in a.shape[:2]:
            raise ValueError(f"zone {z['id']} came back empty {tuple(a.shape)}: refusing to "
                             "stitch around a hole -- re-run that zone")
        bad = int(a.size - np.isfinite(a).sum())
        if bad:
            raise ValueError(f"zone {z['id']} has {bad} non-finite value(s): refusing to "
                             "stitch around a hole -- re-run that zone")

    acc = np.zeros((rh, rw, c), dtype=np.float64)
    wsum = np.zeros((rh, rw, 1), dtype=np.float64)
    logs = []
    for z, img, (ol, ot, orr, ob) in zip(plan["zones"], zone_hdr, _overlaps(plan)):
        x, y, w, h = z["renderRect"]
        a = np.asarray(img, dtype=np.float32)
        if a.shape[:2] != (h, w):
            a = resize_bilinear(a, h, w)   # tolerated, and reported
        lg = to_log2(a)
        logs.append(lg)
        if glog is not None:
            lg = anchor_zone(lg, glog, z["renderRect"], split)
        wy = _feather(np, h, ot, ob)[:, None, None]
        wx = _feather(np, w, ol, orr)[None, :, None]
        wt = wy * wx
        acc[y:y + h, x:x + w] += lg * wt
        wsum[y:y + h, x:x + w] += wt
    out = np.exp2(acc / np.maximum(wsum, 1e-12)).astype(np.float32)
    L, T = plan["render"]["plateOrigin"]
    pw, ph = plan["plate"]["width"], plan["plate"]["height"]
    plate = out[T:T + ph, L:L + pw]
    report = {"seams_before": seam_metrics(logs, plan),
              "anchored": glog is not None, "split_px": split,
              "resized_zones": sum(1 for z, i in zip(plan["zones"], zone_hdr)
                                   if np.asarray(i).shape[:2] != tuple(z["renderRect"][3:1:-1]))}
    if glog is not None:
        anchored = [anchor_zone(lg, glog, z["renderRect"], split)
                    for lg, z in zip(logs, plan["zones"])]
        report["seams_after_anchor"] = seam_metrics(anchored, plan)
    return plate, report


#: Highlights the SDR plate clipped are where the conversion SHOULD change the
#: picture; they are excluded when measuring stripes the model added.
DESTRIPE_SDR_CLIP = 0.9


def _column_highpass(np, profile, window: int):
    k = max(3, int(window) | 1)
    pad = np.pad(profile, k // 2, mode="edge")
    trend = np.convolve(pad, np.ones(k) / k, mode="valid")
    return profile - trend


def destripe_columns(hdr: Any, sdr_linear: Any, *, bands: list[tuple[int, int]] | None = None,
                     window: int = 257, clip: float = DESTRIPE_SDR_CLIP,
                     iterations: int = 2) -> tuple[Any, dict[str, Any]]:
    """Remove VERTICAL stripes the conversion added, measured against its input.

    ``hdr`` is ACEScg linear (LTX ``hdr_linear``); ``sdr_linear`` is the
    linearised SDR plate (Rec.709 primaries); pixel-aligned. Both are compared
    as AP1 luminance -- the SDR moved to ACEScg first -- so a saturated colour
    is not mis-weighted (F-5). Per
    horizontal band (default: the whole frame; pass the zone rows so each
    model run is measured on its own), the column profile of
    ``log2(hdr / sdr)`` -- a median over the band's unclipped rows -- is
    high-passed horizontally (anything wider than ``window`` px -- default 257, ~3x the
    64-111 px stripe periods measured on LTX-2.5 output -- is real tone
    change and kept) and divided out. Band profiles are interpolated between
    band centres so the correction has no horizontal edge. The SDR plate's
    OWN stripes are untouched: only what the conversion added is removed.
    """
    np = _require_numpy()
    h_img = np.asarray(hdr, dtype=np.float32)
    s_img = np.asarray(sdr_linear, dtype=np.float32)
    if h_img.shape[:2] != s_img.shape[:2]:
        raise ValueError(f"hdr {h_img.shape[:2]} and sdr {s_img.shape[:2]} differ")
    H, W = h_img.shape[:2]
    lum_w = _luma_ap1(np)                       # on the ACEScg HDR
    hl = (np.maximum(h_img[..., :3], LOG_EPS) * lum_w).sum(-1)
    sl = (np.maximum(s_img[..., :3], LOG_EPS) * _luma_ap1_of_rec709(np)).sum(-1)
    ratio = np.log2(np.maximum(hl, LOG_EPS)) - np.log2(np.maximum(sl, LOG_EPS))
    usable = s_img[..., :3].max(-1) < clip
    bands = bands or [(0, H)]
    centres, profiles = [], []
    for y0, y1 in bands:
        r = np.where(usable[y0:y1], ratio[y0:y1], np.nan)
        prof = np.nanmedian(r, axis=0)
        prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof) if np.isfinite(prof).any() else 0)
        profiles.append(_column_highpass(np, prof, window))
        centres.append((y0 + y1) / 2.0)
    order = np.argsort(centres)
    centres = np.asarray(centres)[order]
    profiles = np.stack([profiles[i] for i in order])
    if len(centres) == 1:
        field = np.broadcast_to(profiles[0][None, :], (H, W))
    else:
        rows = np.arange(H, dtype=np.float32)
        idx = np.clip(np.searchsorted(centres, rows) - 1, 0, len(centres) - 2)
        t = np.clip((rows - centres[idx]) / (centres[idx + 1] - centres[idx]), 0, 1)[:, None]
        field = profiles[idx] * (1 - t) + profiles[idx + 1] * t
    out = h_img * np.exp2(-field)[..., None].astype(np.float32)
    before = float(np.nanstd([_column_highpass(np, np.nanmedian(np.where(usable, ratio, np.nan)[y0:y1], 0), window)
                              for y0, y1 in bands]))
    r2 = np.log2(np.maximum((np.maximum(out[..., :3], LOG_EPS) * lum_w).sum(-1), LOG_EPS)) - \
        np.log2(np.maximum(sl, LOG_EPS))
    after = float(np.nanstd([_column_highpass(np, np.nanmedian(np.where(usable, r2, np.nan)[y0:y1], 0), window)
                             for y0, y1 in bands]))
    rep = {"ripple_before_stops": before, "ripple_after_stops": after,
           "bands": len(bands), "window_px": int(window), "iterations": 1}
    if iterations > 1:
        # Band profiles are interpolated between band centres, so one pass only
        # approximates each band's own profile; a second pass closes most of the
        # remainder (measured on an 8K LTX plate: 0.084 -> 0.028 -> 0.016 stops),
        # a third buys little because what is left is real content.
        out, nxt = destripe_columns(out, sdr_linear, bands=bands, window=window, clip=clip,
                                    iterations=iterations - 1)
        rep.update(ripple_after_stops=nxt["ripple_after_stops"],
                   iterations=1 + nxt["iterations"])
    return out, rep


#: Local destripe: SDR log2-luminance gradient (stops/px, smoothed) above which
#: a pixel is "structure" and the correction fades out. Sky reads ~0.01-0.03.
LOCAL_DESTRIPE_FLAT_GRAD = 0.04


def destripe_local(hdr: Any, sdr_linear: Any, *, rows: int = 256, window: int = 257,
                   flat_grad: float = LOCAL_DESTRIPE_FLAT_GRAD,
                   min_flat_rows: int = 64) -> tuple[Any, dict[str, Any]]:
    """Second destripe pass for stripes that are LOCAL and sit on flat regions.

    The band pass (``destripe_columns``) takes one column profile per band from
    UNCLIPPED pixels, so it cannot see stripes that live in part of a band or in
    the highlights the SDR clipped -- found live 2026-10-02 in Nuke: streaks in
    the bright sky above the machine and thin lines in the lower sky survived
    the band pass. Here the profile of ``log2(hdr / sdr)`` is taken per
    overlapping ``rows`` window (half-step, linearly blended), from FLAT pixels
    only (smoothed SDR gradient below ``flat_grad``; clipped highlights are
    flat in the SDR, so they count), high-passed at ``window`` px, and applied
    weighted by flatness. Structure (the object, rocks, grass) carries real
    vertical detail in that ratio and gets a weight near 0; a stripe shows on
    flat sky and texture hides it anyway. Measured on the 8K 2x2 plate: |field|
    p99 0.10 stops, p99.9 0.25; without the flat weighting the same field
    reached 2.6 stops on the machine.
    """
    np = _require_numpy()
    h_img = np.asarray(hdr, dtype=np.float32)
    s_img = np.asarray(sdr_linear, dtype=np.float32)
    if h_img.shape[:2] != s_img.shape[:2]:
        raise ValueError(f"hdr {h_img.shape[:2]} and sdr {s_img.shape[:2]} differ")
    H, W = h_img.shape[:2]
    # AP1 luminance on both: the HDR is ACEScg, the Rec.709 SDR is weighted as
    # if moved to ACEScg (F-5).
    ls = np.log2(np.maximum((np.maximum(s_img[..., :3], 0) * _luma_ap1_of_rec709(np)).sum(-1), 1e-4))
    ratio = np.log2(np.maximum((np.maximum(h_img[..., :3], 0) * _luma_ap1(np)).sum(-1), 1e-4)) - ls
    gy, gx = np.gradient(ls)
    grad = lowpass(np.sqrt(gx * gx + gy * gy), 16)
    flat = np.exp(-(grad / float(flat_grad)) ** 2).astype(np.float32)
    rows = max(2, min(int(rows), H))
    step = max(1, rows // 2)
    centres, profs = [], []
    for y0 in range(0, max(1, H - step), step):
        y1 = min(H, y0 + rows)
        f = flat[y0:y1] > 0.5
        enough = f.sum(0) >= min(int(min_flat_rows), y1 - y0)
        if not enough.any():
            profs.append(np.zeros(W, dtype=np.float64))
        else:
            r = np.where(f, ratio[y0:y1], np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns: filled below
                prof = np.nanmedian(np.where(enough[None, :], r, np.nan), 0)
            base = float(np.nanmedian(prof[enough]))
            prof = np.where(enough & np.isfinite(prof), prof, base)
            profs.append(np.where(enough, _column_highpass(np, prof, window), 0.0))
        centres.append((y0 + y1) / 2.0)
    if len(centres) == 1:
        field = np.broadcast_to(profs[0][None, :], (H, W))
    else:
        c = np.asarray(centres)
        P = np.stack(profs)
        rr = np.arange(H, dtype=np.float32)
        idx = np.clip(np.searchsorted(c, rr) - 1, 0, len(c) - 2)
        t = np.clip((rr - c[idx]) / (c[idx + 1] - c[idx]), 0, 1)[:, None]
        field = P[idx] * (1 - t) + P[idx + 1] * t
    field = (field * flat).astype(np.float32)
    out = h_img * np.exp2(-field)[..., None]
    a = np.abs(field)
    rep = {"flat_fraction": float((flat > 0.5).mean()), "rows": rows, "window_px": int(window),
           "field_p99_stops": float(np.percentile(a, 99)),
           "field_max_stops": float(a.max())}
    return out, rep


#: SDR detail transfer: guided-filter radius (px) and regulariser, and the SDR
#: display level (max channel) over which the HDR's own pixels are kept.
SDR_TRANSFER_RADIUS = 64
SDR_TRANSFER_EPS = 0.01
SDR_TRANSFER_CLIP = (0.85, 0.97)


def _box_mean(np, a, r: int):
    """Separable box mean of radius ``r`` (edge-padded), via cumulative sums."""
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = a.shape[axis]
        a = ((np.take(c, np.arange(k, k + n), axis=axis)
              - np.take(c, np.arange(n), axis=axis)) / k).astype(np.float32)
    return a


def sdr_detail_transfer(hdr: Any, sdr_display: Any, *, radius: int = SDR_TRANSFER_RADIUS,
                        eps: float = SDR_TRANSFER_EPS,
                        clip: tuple[float, float] = SDR_TRANSFER_CLIP) -> tuple[Any, dict[str, Any]]:
    """Keep the conversion's RADIANCE, take the picture's structure from the SDR.

    The model's stripes are thin to ~100 px wide and as fine as real texture,
    so no column filter separates them everywhere (found live 2026-10-02 in
    Nuke: three destripe passes left visible lines). The SDR has the same
    content and none of the stripes, so it is the guide: per channel, the log
    ratio ``log2(hdr) - log2(sdr)`` -- what the conversion did -- is smoothed
    with a guided filter (He et al.) on the SDR's log luminance. Anything in
    the ratio the SDR cannot explain linearly within ``radius`` (the stripes,
    at any width up to the window) is dropped; real edges and the local tone
    slope pass through, so there is no halo at the object's silhouette. The
    result is ``sdr * 2**ratio_smooth``. Where the SDR clipped (max channel
    over ``clip``, feathered) it has no structure to give and the HDR's own
    pixels are kept.

    ``sdr_display`` is the display-referred plate (0..1, sRGB) the split got;
    ``hdr`` and the result are ACEScg linear.
    Cost: fine detail the model reconstructed in UNclipped areas (it smooths
    JPEG blocking) is replaced by the SDR's.
    """
    np = _require_numpy()
    from atlas_camera.core.generated_mesh import srgb_to_linear

    h = np.asarray(hdr, dtype=np.float32)[..., :3]
    sd = np.clip(np.asarray(sdr_display, dtype=np.float32)[..., :3], 0.0, 1.0)
    if h.shape[:2] != sd.shape[:2]:
        raise ValueError(f"hdr {h.shape[:2]} and sdr {sd.shape[:2]} differ")
    floor = 1e-4
    # Per-channel ratios need ONE colour space: the SDR is linearised and moved
    # to ACEScg (the HDR's space) before log2(hdr) - log2(sdr), else a pure
    # colour picks up a hue shift from the primaries mismatch (F-5).
    ls = np.log2(np.maximum(rec709_linear_to_acescg(srgb_to_linear(sd)), floor))
    lh = np.log2(np.maximum(h, floor))
    guide = (ls * _luma_ap1(np)).sum(-1)
    r = max(1, int(radius))
    mg = _box_mean(np, guide, r)
    vg = _box_mean(np, guide * guide, r) - mg * mg
    c = _box_mean(np, sd.max(-1), 4)
    lo, hi = float(clip[0]), float(clip[1])
    keep = np.clip((c - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    keep = keep * keep * (3 - 2 * keep)
    out = np.empty_like(h)
    for ch in range(3):
        ratio = lh[..., ch] - ls[..., ch]
        mr = _box_mean(np, ratio, r)
        a = (_box_mean(np, guide * ratio, r) - mg * mr) / (vg + float(eps))
        b = mr - a * mg
        smooth = _box_mean(np, a, r) * guide + _box_mean(np, b, r)
        out[..., ch] = np.exp2((ls[..., ch] + smooth) * (1 - keep) + lh[..., ch] * keep)
    out = np.where(h < 0, h, out)          # the few negatives stay as the model made them
    d = np.abs(np.log2(np.maximum(out, floor)) - lh)
    rep = {"radius_px": r, "eps": float(eps), "kept_hdr_fraction": float((keep > 0.5).mean()),
           "change_p50_stops": float(np.percentile(d, 50)),
           "change_p99_stops": float(np.percentile(d, 99))}
    return out, rep


#: Destripe bands are at most this fraction of the plate height. The stripes
#: drift down a tall zone: on a 2x2 8K plate (2256-row zones) one band per zone
#: row left 0.085 stops at quarter-height granularity; quarter-height bands took
#: it to 0.016, the 4x4 figure. Eighth-height bands get noisier (too few rows
#: per column median).
DESTRIPE_MAX_BAND_FRACTION = 0.25


def zone_row_bands(plan: dict[str, Any], *,
                   max_fraction: float = DESTRIPE_MAX_BAND_FRACTION) -> list[tuple[int, int]]:
    """Each zone row's plate-space row span (for per-run destriping), split
    evenly so no band is taller than ``max_fraction`` of the plate."""
    import math
    L, T = plan["render"]["plateOrigin"]
    ph = plan["plate"]["height"]
    spans = sorted({(z["plateRect"][1] - T, z["plateRect"][1] - T + z["plateRect"][3])
                    for z in plan["zones"]})
    cap = max(1, int(math.ceil(ph * float(max_fraction)))) if max_fraction else ph
    out = []
    for a, b in spans:
        a, b = max(0, a), min(ph, b)
        n = max(1, int(math.ceil((b - a) / cap)))
        edges = [a + round(i * (b - a) / n) for i in range(n + 1)]
        out.extend(zip(edges[:-1], edges[1:]))
    return out


def seam_metrics(zone_logs: list[Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Per interior seam: |log2 luminance| disagreement between the two zones
    over their shared overlap (median, p95). 0 = the neighbours agree."""
    np = _require_numpy()
    rects = [z["renderRect"] for z in plan["zones"]]
    idx = {tuple(z["index"]): i for i, z in enumerate(plan["zones"])}
    cols, rows = plan["grid"]
    lum_w = _luma_ap1(np)                       # zone results are ACEScg
    out = []

    def lum(lg):
        lin = np.exp2(lg[..., :3])
        return np.log2(np.maximum((lin * lum_w).sum(-1), LOG_EPS))

    for (c, r), i in idx.items():
        for dc, dr in ((1, 0), (0, 1)):
            j = idx.get((c + dc, r + dr))
            if j is None:
                continue
            ax, ay, aw, ah = rects[i]
            bx, by, bw, bh = rects[j]
            x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
            y0, y1 = max(ay, by), min(ay + ah, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            a = lum(zone_logs[i][y0 - ay:y1 - ay, x0 - ax:x1 - ax])
            b = lum(zone_logs[j][y0 - by:y1 - by, x0 - bx:x1 - bx])
            d = np.abs(a - b)
            out.append({"seam": f"{plan['zones'][i]['id']}|{plan['zones'][j]['id']}",
                        "median_stops": float(np.median(d)),
                        "p95_stops": float(np.percentile(d, 95))})
    return out


#: Stitched-plate step test: strip width either side of a seam line, and the
#: flag threshold as a multiple of the random-line baseline (its p90).
SEAM_STRIP_PX = 48
SEAM_STEP_RATIO_MAX = 1.5
SEAM_BASELINE_PERCENTILE = 90
#: Seam segments are scored in windows of about this many px along the line, so
#: a seam bright on one half and dark on the other cannot cancel to zero.
SEAM_WINDOW_PX = 256


def _windows(s0: int, s1: int, window: int = SEAM_WINDOW_PX) -> list[tuple[int, int]]:
    """``[s0, s1)`` split into ``round(len / window)`` (>= 1) near-equal windows."""
    n = max(1, int(round((s1 - s0) / float(max(1, window)))))
    edges = [s0 + int(round(i * (s1 - s0) / n)) for i in range(n + 1)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _log2_lum(np, a, weights=None):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] >= 3:
        a = (a[..., :3] * (_luma_ap1(np) if weights is None else weights)).sum(-1)
    return np.log2(np.maximum(a.reshape(a.shape[:2]), LOG_EPS))


def seam_step_test(plate: Any, plan: dict[str, Any], *, sdr_linear: Any = None,
                   strip_px: int = SEAM_STRIP_PX, ratio_max: float = SEAM_STEP_RATIO_MAX,
                   n_random: int = 64, seed: int = 0,
                   window_px: int = SEAM_WINDOW_PX) -> dict[str, Any]:
    """The gate a viewer actually sees, on the STITCHED plate.

    Per interior seam segment: the per-row (per-column) log2-luminance step
    between the means of the ``strip_px`` strips either side. With
    ``sdr_linear`` (the linearised plate the split was given, same size, Rec.709
    primaries; ``plate`` is ACEScg, luminance is AP1 on both) the
    SDR's own step on the same line is subtracted first: the SDR has the same
    structure and no seams, so structure cancels and a tonal seam (an offset)
    remains. Without it structure on the line scores too; the report says so.

    The segment is scored in WINDOWS of ~``window_px`` along the line, never
    as one median: a seam +2 stops on its upper half and -2 on its lower half
    has a whole-line median of 0 and used to PASS (found by the 2026-10-03
    outside review). Per window the score is ``|median(step_hdr - step_sdr)|``
    (SDR-controlled) or ``median |step_hdr|`` (plain); the segment's score is
    its WORST window, and the window p95 is reported beside it::

        zone A | zone B            one vertical seam segment, s0..s1
               |
        -------+------- s0   w0 :  |median(d[w0])|   <- d = step_hdr - step_sdr
         48|48 |                   per row, strips of strip_px either side
        -------+-------      w1 :  |median(d[w1])|
               |               ...                    windows ~window_px long
        -------+------- s1   wN :  |median(d[wN])|
                             score = max_w, also p95_w

    The baseline is the p90 of the SAME score -- same length, same windows,
    worst window -- on random lines of that orientation away from the seams,
    so a long seam is not flagged just for having more windows to be unlucky
    in. Against a MEDIAN baseline the plain step flagged 11 of 24 segments of
    the 8K machine plate (4x4) -- structure, not seams -- hence the p90 (those
    numbers predate windowing). A ratio above ``ratio_max`` FLAGS a seam for a
    look -- never refused, never blended away.
    """
    np = _require_numpy()
    lg = _log2_lum(np, plate)
    ls = None
    if sdr_linear is not None:
        ls = _log2_lum(np, sdr_linear, _luma_ap1_of_rec709(np))   # Rec.709 SDR as AP1 Y
        if ls.shape != lg.shape:
            ls = None
    ph, pw = lg.shape
    k = int(strip_px)
    L, T = plan["render"]["plateOrigin"]
    idx = {tuple(z["index"]): z for z in plan["zones"]}

    def diff(a, orient, pos, s0, s1):
        if orient == "v":
            return a[s0:s1, pos - k:pos].mean(1) - a[s0:s1, pos:pos + k].mean(1)
        return a[pos - k:pos, s0:s1].mean(0) - a[pos:pos + k, s0:s1].mean(0)

    def score(orient, pos, s0, s1):
        """``(worst window, window p95, worst window span)`` or None."""
        n = pw if orient == "v" else ph
        if pos - k < 0 or pos + k > n or s1 <= s0:
            return None
        d = diff(lg, orient, pos, s0, s1)
        if ls is not None:
            d = d - diff(ls, orient, pos, s0, s1)
        vals, spans = [], []
        for a, b in _windows(s0, s1, window_px):
            w = d[a - s0:b - s0]
            vals.append(float(abs(np.median(w))) if ls is not None
                        else float(np.median(np.abs(w))))
            spans.append((a, b))
        i = int(np.argmax(vals))
        return vals[i], float(np.percentile(vals, 95)), spans[i]

    segs = []
    for (c, r), z in idx.items():
        px, py, zw, zh = z["plateRect"]
        px, py = px - L, py - T
        right, below = idx.get((c + 1, r)), idx.get((c, r + 1))
        if right is not None:
            segs.append(("v", f"{z['id']}|{right['id']}", px + zw, py, py + zh))
        if below is not None:
            segs.append(("h", f"{z['id']}|{below['id']}", py + zh, px, px + zw))
    avoid = {"v": {s[2] for s in segs if s[0] == "v"}, "h": {s[2] for s in segs if s[0] == "h"}}
    rng = np.random.default_rng(seed)

    def baseline(orient, length):
        n, extent = (pw, ph) if orient == "v" else (ph, pw)
        vals = []
        for _ in range(n_random * 8):
            if len(vals) >= n_random or n <= 2 * k + 1:
                break
            pos = int(rng.integers(k, n - k))
            if any(abs(pos - q) < 2 * k for q in avoid[orient]):
                continue
            s0 = int(rng.integers(0, max(1, extent - length + 1)))
            v = score(orient, pos, s0, min(extent, s0 + length))
            if v is not None:
                vals.append(v[0])
        return float(np.percentile(vals, SEAM_BASELINE_PERCENTILE)) if vals else None

    seams, base_cache = [], {}
    for orient, name, pos, s0, s1 in segs:
        sc = score(orient, pos, s0, s1)
        step, p95, span = (None, None, None) if sc is None else sc
        key = (orient, s1 - s0)
        if key not in base_cache:
            base_cache[key] = baseline(orient, s1 - s0)
        base = base_cache[key]
        ratio = None if step is None or base is None else step / max(base, 1e-4)
        seams.append({"seam": name, "orientation": orient, "at_px": int(pos),
                      "step_stops": step, "window_p95_stops": p95,
                      "windows": len(_windows(s0, s1, window_px)),
                      "worst_window": None if span is None else [int(span[0]), int(span[1])],
                      "baseline_stops": base, "ratio": ratio,
                      "flagged": ratio is not None and ratio > ratio_max})
    scored = [s for s in seams if s["ratio"] is not None]
    worst = max(scored, key=lambda s: s["ratio"]) if scored else None
    return {"strip_px": k, "ratio_max": float(ratio_max), "sdr_controlled": ls is not None,
            "window_px": int(window_px),
            "baseline_percentile": SEAM_BASELINE_PERCENTILE, "seams": seams, "worst": worst,
            "flagged": [s["seam"] for s in seams if s["flagged"]],
            "pass": (not any(s["flagged"] for s in seams)) if scored else None}
