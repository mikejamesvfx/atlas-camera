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


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matrixZone requires numpy.") from exc


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
    out = []
    for i, (start, size) in enumerate(cells):
        if i == 0:
            zstart = 0
        elif i == n - 1:
            zstart = ext_before + sum(c[1] for c in cells) + ext_after - zsize
        else:
            zstart = origin + start - (zsize - size) // 2
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


def seam_metrics(zone_logs: list[Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Per interior seam: |log2 luminance| disagreement between the two zones
    over their shared overlap (median, p95). 0 = the neighbours agree."""
    np = _require_numpy()
    rects = [z["renderRect"] for z in plan["zones"]]
    idx = {tuple(z["index"]): i for i, z in enumerate(plan["zones"])}
    cols, rows = plan["grid"]
    lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
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
