"""Normal-integration depth detail (pure numpy, host-agnostic).

Monocular depth maps (V2/DA3/MoGe) are metrically sound but low-frequency —
brick courses, window reveals, rock striations flatten out. Predicted surface
normals carry exactly that high-frequency shape. This module integrates a
normal map into a relative height field (Frankot-Chellappa), strips everything
below a cutoff wavelength, and blends ONLY the surviving high-frequency detail
onto a metric depth map — multiplicatively in log-depth, median-renormalized —
so the metric base (ground fit, band edges, camera height) can never drift.

Scale preservation is structural, not incidental:
  * ``highpass_detail`` removes the low-frequency content that could tilt or
    re-scale the surface (the integrated height's absolute scale is arbitrary
    anyway — normals only constrain slope, not depth).
  * ``blend_depth_detail`` renormalizes the blended map back to the input's
    median, so the net scale change is exactly zero by construction.

No torch, no scipy — FFT-based throughout, unit-testable without ComfyUI.
"""
from __future__ import annotations

from typing import Any


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "depth detail enhancement requires numpy. Install with:\n"
            "    pip install -e .[vision]"
        ) from exc
    return np


# Gradients steeper than this (|dz/dx| in height units per pixel) are clamped —
# near-silhouette normals (nz -> 0) would otherwise dominate the integration
# with slopes the surface can't actually have.
_SLOPE_CLAMP = 10.0

# Detail is normalized to robust units and clipped here before blending, so a
# single wild integration artifact cannot spike the output depth.
_DETAIL_CLIP_SIGMA = 3.0


def integrate_normals_frankot_chellappa(normals: Any) -> Any:
    """Integrate a HxWx3 normal map into a relative height field.

    Classic Frankot-Chellappa: project the (p, q) gradient field onto the
    integrable subspace in the Fourier domain and invert. Mirror-extension to
    2H x 2W enforces Neumann-ish boundaries (even height, odd gradients), which
    kills the wrap-around seam a plain FFT would produce.

    The normals' camera-frame handedness only flips the height sign, and the
    absolute scale is arbitrary — both are irrelevant downstream because the
    consumer high-passes and re-normalizes the result. Output is float64 HxW,
    zero-mean.
    """
    np = _require_numpy()
    n = np.asarray(normals, dtype=np.float64)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError(f"normals must be HxWx3, got {n.shape}")

    nz = n[..., 2]
    finite = np.isfinite(nz) & (np.abs(nz) > 1e-6)
    # Make the dominant nz sign positive so p/q signs are consistent across
    # model conventions (z toward vs away from camera).
    sign = 1.0
    if finite.any() and float(np.median(nz[finite])) < 0:
        sign = -1.0
    nz_s = np.where(np.isfinite(nz), nz * sign, 1.0)
    nz_s = np.where(np.abs(nz_s) < 1e-3, np.copysign(1e-3, nz_s + 1e-30), nz_s)
    p = np.clip(-np.nan_to_num(n[..., 0]) * sign / nz_s, -_SLOPE_CLAMP, _SLOPE_CLAMP)
    q = np.clip(-np.nan_to_num(n[..., 1]) * sign / nz_s, -_SLOPE_CLAMP, _SLOPE_CLAMP)

    # Mirror extension: height even in both axes => p odd in x / even in y,
    # q even in x / odd in y.
    p2 = np.block([[p, -p[:, ::-1]], [p[::-1, :], -p[::-1, ::-1]]])
    q2 = np.block([[q, q[:, ::-1]], [-q[::-1, :], -q[::-1, ::-1]]])

    h2, w2 = p2.shape
    wx = 2.0 * np.pi * np.fft.fftfreq(w2)[None, :]
    wy = 2.0 * np.pi * np.fft.fftfreq(h2)[:, None]
    denom = wx * wx + wy * wy
    denom[0, 0] = 1.0  # DC is unconstrained by gradients; forced to 0 below

    fp = np.fft.fft2(p2)
    fq = np.fft.fft2(q2)
    fz = (-1j * wx * fp - 1j * wy * fq) / denom
    fz[0, 0] = 0.0
    height = np.real(np.fft.ifft2(fz))[: p.shape[0], : p.shape[1]]
    return height - float(height.mean())


def highpass_detail(height: Any, cutoff_px: float) -> Any:
    """Suppress wavelengths longer than ``cutoff_px``; keep the fine detail.

    Gaussian high-pass in the frequency domain on the mirror-extended field
    (seam-free). The output is zero-mean by construction (DC removed), so it
    carries shape only — no offset, no tilt-scale energy that could move a
    metric surface.
    """
    np = _require_numpy()
    z = np.asarray(height, dtype=np.float64)
    if cutoff_px <= 0:
        return z - float(z.mean())
    h, w = z.shape
    z2 = np.block([[z, z[:, ::-1]], [z[::-1, :], z[::-1, ::-1]]])
    fx = np.fft.fftfreq(2 * w)[None, :]
    fy = np.fft.fftfreq(2 * h)[:, None]
    f2 = fx * fx + fy * fy
    # Order-2 Gaussian low-pass rolling off around frequency 1/(2*cutoff)
    # cycles/px: wavelengths comfortably longer than cutoff pass essentially
    # untouched, shorter ones survive as detail. A plain first-order Gaussian
    # is too soft here — it leaks ~15% of a 4x-cutoff wavelength into the
    # "detail", which is exactly the tilt/scale energy this filter exists to
    # keep out of a metric surface.
    sigma_f = 0.5 / max(float(cutoff_px), 1e-6)
    ratio = f2 / (sigma_f * sigma_f)
    lowpass = np.exp(-0.5 * ratio * ratio)
    fz = np.fft.fft2(z2) * (1.0 - lowpass)
    detail = np.real(np.fft.ifft2(fz))[:h, :w]
    return detail - float(detail.mean())


def blend_depth_detail(
    depth_m: Any,
    detail: Any,
    strength: float,
    *,
    amplitude: float = 0.02,
    exclude_mask: Any = None,
) -> Any:
    """Emboss zero-mean ``detail`` onto ``depth_m``; scale strictly preserved.

    Multiplicative in log-depth — ``depth * exp(s * a * detail_units)`` — so a
    given detail wiggle displaces proportionally at 2 m and at 200 m, instead
    of an additive offset that would be gigantic up close and invisible far
    away. ``detail`` is normalized to robust (median-abs-dev) units and clipped
    at +/-3 sigma first; ``amplitude`` is the log-depth swing per detail sigma
    at strength 1 (default 2%).

    Invariants (tested): the valid-pixel median is renormalized to the input's
    exactly; NaNs and excluded pixels pass through untouched; output stays
    strictly positive wherever the input was.
    """
    np = _require_numpy()
    d = np.asarray(depth_m, dtype=np.float64)
    det = np.asarray(detail, dtype=np.float64)
    if d.shape != det.shape:
        raise ValueError(f"depth {d.shape} vs detail {det.shape} shape mismatch")
    s = float(np.clip(strength, 0.0, 1.0))
    if s == 0.0:
        return d.astype(np.float32)

    valid = np.isfinite(d) & (d > 0)
    keep = valid.copy()
    if exclude_mask is not None:
        keep &= ~np.asarray(exclude_mask, dtype=bool)
    if not keep.any():
        return d.astype(np.float32)

    mad = float(np.median(np.abs(det[keep] - np.median(det[keep]))))
    det_units = np.clip(
        (det - float(np.median(det[keep]))) / (mad * 1.4826 or 1.0),
        -_DETAIL_CLIP_SIGMA, _DETAIL_CLIP_SIGMA)

    out = d.copy()
    out[keep] = d[keep] * np.exp(s * float(amplitude) * det_units[keep])
    # Renormalize: net scale change is exactly zero by construction.
    med_in = float(np.median(d[keep]))
    med_out = float(np.median(out[keep]))
    if med_out > 0 and med_in > 0:
        out[keep] *= med_in / med_out
    return out.astype(np.float32)


def combine_depth_high_freq(
    base_m: Any,
    detail_src_m: Any,
    strength: float,
    *,
    cutoff_px: float = 64.0,
) -> Any:
    """Graft ``detail_src_m``'s high-frequency structure onto ``base_m``.

    The "MoGe detail on V2 far-field" combo: high-pass the SOURCE's log-depth
    (log so the extraction is scale-invariant), then blend it onto the base via
    the same scale-preserving log-domain emboss. Arrays must already share a
    shape (the node layer resizes).
    """
    np = _require_numpy()
    src = np.asarray(detail_src_m, dtype=np.float64)
    logsrc = np.log(np.clip(np.nan_to_num(src, nan=1.0), 1e-6, None))
    detail = highpass_detail(logsrc, cutoff_px)
    # log-domain detail is already in log units — amplitude 1 passes it through.
    return blend_depth_detail(base_m, detail, strength, amplitude=1.0)
