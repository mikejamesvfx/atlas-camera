"""Falsifiable scoring for generated occlusion fills.

Every metric here exists because a metric that could not fail once reported
success on broken output: residual-sentinel read 0.0000% on fills with a
visible seam and a global colour cast, because after the outward-feather fix
every hole pixel is overwritten and only a literal chroma-green return could
trip it. These are the replacements, each calibrated against measurements from
2026-08-13 on the DSC_2289 street plate:

- fill vs a trivial edge-extend smear: 17.6/255 (a no-op scores ~0)
- rim gradient ratio: 2.2-2.3x on a seamy fill vs ~1.0 for the smear baseline
- green-excess delta: 5.8 on a cast fill, 0.2 after plate-referenced correction
- unmasked delta: measurably non-zero whenever the generator re-tones the
  whole crop (the region-limited-denoise experiment drives this to ~0)

Layering: dynamic/ may import core, never comfy. Needs numpy
(``pip install -e .[vision]``).
"""
from __future__ import annotations


from atlas_camera.dynamic.occlusion_fill import _dilate, _require_deps

#: G2 — a fill closer than this to the smear baseline is doing nothing worth
#: a diffusion run (measured: real fill 17.6, smear 0 by construction).
G2_MIN_EDGE_EXTEND_DIFF = 5.0 / 255.0

#: G3a — the rim's gradient energy may exceed the plate's own statistics by at
#: most this factor (measured failures: 2.2-2.3x; smear baseline ~0.94x).
G3A_MAX_RIM_GRADIENT_RATIO = 1.5

#: G3b — |green_excess(fill) - green_excess(context)| in 8-bit units
#: (measured: 5.8 uncorrected, 0.2 with the plate-referenced cast fix).
G3B_MAX_GREEN_EXCESS_DELTA = 1.0


def _as_rgb_f64(np, img):
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3 or a.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 image, got shape {a.shape}")
    return a[..., :3]


def _as_hole(np, hole):
    m = np.asarray(hole)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == np.bool_:
        return m
    if m.dtype == np.uint8:
        return m > 127
    return m > 0.5


def edge_extend(img, hole, *, iters: int = 200):
    """Nearest-neighbour smear of ``img`` into ``hole`` — the null hypothesis.

    Deterministic, content-free hole filling: each pass copies every hole
    pixel from an already-filled 4-neighbour. Any generator that cannot beat
    this by a margin is not earning its GPU time (G2).
    """
    np, _ = _require_deps()
    out = _as_rgb_f64(np, img).copy()
    remaining = _as_hole(np, hole).copy()
    for _ in range(int(iters)):
        if not bool(remaining.any()):
            break
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            src = np.roll(out, shift, axis=axis)
            src_known = np.roll(~remaining, shift, axis=axis)
            take = remaining & src_known
            out[take] = src[take]
            remaining = remaining & ~take
    source = np.asarray(img)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(source.dtype)


def _gradient_mag(np, img):
    grey = _as_rgb_f64(np, img).mean(axis=2)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


def _green_excess(np, pixels) -> float:
    px = np.asarray(pixels, dtype=np.float64)
    return float((px[:, 1] - (px[:, 0] + px[:, 2]) / 2.0).mean())


def unmasked_delta(fill, guide, hole) -> float:
    """Mean |fill - guide| OUTSIDE the hole, in [0, 1].

    The generator was only ever asked to invent the masked pixels; everything
    else should come back untouched. In practice a full-crop resynthesis
    re-tones the whole raster — this measures that directly, and the
    region-limited-denoise experiment (E1) succeeds exactly when this is ~0.
    """
    np, _ = _require_deps()
    f = _as_rgb_f64(np, fill)
    g = _as_rgb_f64(np, guide)
    if f.shape != g.shape:
        raise ValueError(f"fill {f.shape} and guide {g.shape} rasters differ")
    scale = 255.0 if np.asarray(fill).dtype != np.float32 and \
        np.asarray(fill).dtype != np.float64 else 1.0
    keep = ~_as_hole(np, hole)
    if not bool(keep.any()):
        return 0.0
    return float(np.abs(f[keep] - g[keep]).mean() / scale)


def score_fill(fill, guide, hole, *, rim_px: int = 2,
               edge_extend_iters: int = 200) -> dict:
    """Score a composited fill against the plate render behind it.

    ``fill`` is the crop AFTER any colour correction, at the ROI raster;
    ``guide`` is the plate render of the same window (hole pixels may carry
    the sentinel — they are never read as reference); ``hole`` is the
    generation mask. Returns raw numbers plus per-gate booleans so an arm
    report is self-judging.
    """
    np, _ = _require_deps()
    f = _as_rgb_f64(np, fill)
    g = _as_rgb_f64(np, guide)
    if f.shape != g.shape:
        raise ValueError(f"fill {f.shape} and guide {g.shape} rasters differ")
    mask = _as_hole(np, hole)
    if not bool(mask.any()):
        raise ValueError("score_fill needs a non-empty hole mask")
    scale = 255.0 if np.asarray(fill).dtype not in (np.float32, np.float64) \
        else 1.0

    # G2 — distance from the deterministic smear, hole pixels only.
    smear = _as_rgb_f64(np, edge_extend(guide, mask, iters=edge_extend_iters))
    g2 = float(np.abs(f[mask] - smear[mask]).mean() / scale)

    # G3a — gradient statistics at the rim vs the plate's own. The composite
    # under test is guide-with-fill-pasted; the reference is the plate away
    # from both hole and rim.
    comp = g.copy()
    comp[mask] = f[mask]
    rim = _dilate(np, mask, int(rim_px)) & ~mask
    plate_ref = ~mask & ~rim
    grad = _gradient_mag(np, comp)
    plate_grad = float(_gradient_mag(np, g)[plate_ref].mean())
    rim_grad = float(grad[rim].mean()) if bool(rim.any()) else 0.0
    g3a = (rim_grad / plate_grad) if plate_grad > 1e-9 else float("inf")

    # G3b — colour balance of the fill vs the real context around it.
    g3b = abs(_green_excess(np, f[mask] * (255.0 / scale) if scale != 255.0
                            else f[mask]) -
              _green_excess(np, g[plate_ref] * (255.0 / scale)
                            if scale != 255.0 else g[plate_ref]))

    # Detail proxy (not a gate — reported for arm comparison).
    fill_energy = float(grad[mask].mean())

    return {
        "hole_px": int(mask.sum()),
        "mean_abs_vs_edge_extend": g2,
        "rim_gradient": rim_grad,
        "plate_gradient": plate_grad,
        "rim_gradient_ratio": g3a,
        "green_excess_delta": g3b,
        "fill_gradient_energy": fill_energy,
        "unmasked_delta": unmasked_delta(fill, guide, mask),
        "g2_pass": bool(g2 > G2_MIN_EDGE_EXTEND_DIFF),
        "g3a_pass": bool(g3a <= G3A_MAX_RIM_GRADIENT_RATIO),
        "g3b_pass": bool(g3b < G3B_MAX_GREEN_EXCESS_DELTA),
    }


def encode_depth_guide(depth, *, ground_mask=None, lo_pct: float = 5.0,
                       hi_pct: float = 95.0, log_encode: bool = False):
    """Metric z-buffer -> disparity guide image in [0, 1] (HxWx3 float32).

    The 2026-08-13 failure was the ENCODE, not the plumbing: disparity
    normalised over the scene's full 3.61-70.90 m range put the subject at
    0.07 brightness against a ground plane at 0.71, and the model read an
    almost-black control map. Percentiles are therefore taken over NON-GROUND
    pixels (``ground_mask`` true where ground), so the structures the fill
    must respect occupy the encode's dynamic range.

    +inf (nothing rasterized) encodes to 0 — far, exactly how a depth-control
    net reads emptiness. ``log_encode`` compresses the near field when a
    scene's structure spans decades of depth.
    """
    np, _ = _require_deps()
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"expected an HxW depth buffer, got shape {z.shape}")
    finite = np.isfinite(z) & (z > 0)
    if not bool(finite.any()):
        raise ValueError("depth buffer has no finite samples")
    disp = np.zeros_like(z)
    disp[finite] = 1.0 / z[finite]
    if log_encode:
        disp[finite] = np.log1p(disp[finite])

    sample = finite.copy()
    if ground_mask is not None:
        g = _as_hole(np, ground_mask)
        if g.shape != z.shape:
            raise ValueError(
                f"ground_mask {g.shape} does not match depth {z.shape}")
        non_ground = sample & ~g
        # Fall back to all finite pixels rather than normalising over nothing.
        if bool(non_ground.any()):
            sample = non_ground
    lo, hi = np.percentile(disp[sample], [float(lo_pct), float(hi_pct)])
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    out = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return np.repeat(out[..., None], 3, axis=2).astype(np.float32)
