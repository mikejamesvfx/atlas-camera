"""Carry a plate's SDR->HDR conversion onto colours the model never saw.

An SDR->HDR model (LTX-2.5 IC-LoRA) converts IMAGES. A generated object's
hidden side is per-vertex colour (core.generated_mesh), not an image, so the
model cannot see it -- and laying it out in UVs would hand the model a texture
atlas with no scene context to judge highlights by. Instead, the model's own
mapping is measured where it DID run: the SDR plate and its HDR conversion are
pixel-aligned pairs, so a per-channel curve ``sdr -> hdr`` (binned in log
radiance, forced monotone) is fitted from them and applied to the vertex
colours. The hidden side then shares the plate's tone, exactly as its colours
were already graded onto the plate in SDR.

Colour: the SDR plate is display sRGB (Rec.709); LTX-2.5's ``hdr_linear`` is
ACEScg. SDR is linearised and moved to ACEScg (Bradford D65 -> D60) BEFORE the
fit, so the curve is a pure tone curve in one space, and the output vertex
colours are ACEScg linear (may exceed 1).

Layering: ``core`` only -- numpy.
"""

from __future__ import annotations

from typing import Any

#: Linear Rec.709 / sRGB (D65) -> ACEScg (AP1, D60), Bradford adaptation.
REC709_TO_ACESCG = (
    (0.6130974, 0.3395231, 0.0473795),
    (0.0701937, 0.9163539, 0.0134524),
    (0.0206156, 0.1095698, 0.8698147),
)

_LOG_FLOOR = 2.0 ** -14


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("HDR transfer requires numpy.") from exc


def srgb_to_acescg(srgb: Any) -> Any:
    """Display sRGB (0..1) -> ACEScg linear."""
    np = _require_numpy()
    from atlas_camera.core.generated_mesh import srgb_to_linear
    lin = srgb_to_linear(np.asarray(srgb, dtype=np.float32)[..., :3])
    return lin @ np.asarray(REC709_TO_ACESCG, dtype=np.float32).T


def fit_sdr_to_hdr_curve(sdr_srgb: Any, hdr_acescg: Any, *, bins: int = 64,
                         max_samples: int = 2_000_000, seed: int = 0) -> dict[str, Any]:
    """Per-channel monotone tone curve from pixel-aligned SDR/HDR plates.

    Knots are log2 radiance bins of the SDR (in ACEScg); each knot's value is
    the median HDR over that bin, forced non-decreasing. Bins with too few
    samples are interpolated from their neighbours. ``residual_stops`` is the
    median |log2(curve(sdr) / hdr)| on the fitted samples -- how well ONE global
    curve explains the conversion (local tone the model applied differently in
    different places shows up here).
    """
    np = _require_numpy()
    s = np.asarray(sdr_srgb, dtype=np.float32)[..., :3].reshape(-1, 3)
    h = np.asarray(hdr_acescg, dtype=np.float32)[..., :3].reshape(-1, 3)
    if s.shape != h.shape:
        raise ValueError(f"SDR {s.shape} and HDR {h.shape} must be pixel-aligned")
    if len(s) > max_samples:
        idx = np.random.default_rng(seed).choice(len(s), size=max_samples, replace=False)
        s, h = s[idx], h[idx]
    x = np.maximum(srgb_to_acescg(s), _LOG_FLOOR)
    h = np.maximum(h, _LOG_FLOOR)
    lx = np.log2(x)
    lo, hi = float(np.percentile(lx, 0.1)), float(np.percentile(lx, 99.99))
    edges = np.linspace(lo, hi, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    curves, counts = [], []
    for c in range(3):
        b = np.clip(np.digitize(lx[:, c], edges) - 1, 0, bins - 1)
        ys = np.full(bins, np.nan)
        n = np.bincount(b, minlength=bins)
        for k in range(bins):
            if n[k] >= 16:
                ys[k] = np.median(np.log2(h[b == k, c]))
        ok = np.isfinite(ys)
        if not ok.any():
            raise ValueError("no populated bins: is the HDR plate empty?")
        ys = np.interp(centres, centres[ok], ys[ok])
        curves.append(np.maximum.accumulate(ys))
        counts.append(n)
    curves = np.stack(curves)
    pred = np.stack([np.interp(lx[:, c], centres, curves[c]) for c in range(3)], -1)
    resid = float(np.median(np.abs(pred - np.log2(h))))
    return {"log2_x": centres.tolist(), "log2_y": curves.tolist(),
            "residual_stops": resid, "samples": int(len(s)), "bins": int(bins),
            "space": "ACEScg"}


def apply_curve(curve: dict[str, Any], srgb: Any) -> Any:
    """sRGB colours (..., 3) -> ACEScg linear through the fitted curve.

    Beyond the fitted range the curve is extended at the end slope (in log), so
    a vertex brighter than anything on the plate still brightens monotonically.
    """
    np = _require_numpy()
    lx = np.log2(np.maximum(srgb_to_acescg(srgb), _LOG_FLOOR))
    xs = np.asarray(curve["log2_x"], dtype=np.float64)
    out = np.empty_like(lx)
    for c in range(3):
        ys = np.asarray(curve["log2_y"][c], dtype=np.float64)
        v = np.interp(lx[..., c], xs, ys)
        slope_hi = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
        slope_lo = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9)
        v = np.where(lx[..., c] > xs[-1], ys[-1] + max(slope_hi, 0.0) * (lx[..., c] - xs[-1]), v)
        v = np.where(lx[..., c] < xs[0], ys[0] + max(slope_lo, 0.0) * (lx[..., c] - xs[0]), v)
        out[..., c] = v
    return np.exp2(out).astype(np.float32)
