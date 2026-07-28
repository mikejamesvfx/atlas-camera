"""Fit a correction from a monocular depth estimate onto MEASURED depth.

Atlas's biggest single weakness is scale. Given one photograph it assumes a
1.6 m camera height, which on wide exterior plates has measured out roughly ten
times too small. A LiDAR iPhone changes the arithmetic: every capture is an
(RGB, metric depth) PAIR, i.e. labelled data, and a handful of them is enough to
learn how a given model misreads a given kind of scene.

This does NOT train a depth model. Training something competitive with MoGe or
Depth-Anything is millions of images and months of GPU. What it does is far
cheaper and targets the actual failure: fit two or three coefficients that map a
model's output onto measured truth, per model and per scene type, then apply them.

WHICH SPACE TO FIT IN — the decision that matters
-------------------------------------------------
Monocular networks regress something disparity-like (1/d): near surfaces get most
of the representational budget and the far field is compressed. So their error is
typically affine in DISPARITY, not in depth. Fitting ``a*d + b`` in depth space
can look excellent on a room interior and still be badly wrong at 200 m, because
the far tail is where the mismatch lives and where the fit has fewest samples.

``affine_disparity`` is therefore the default. ``affine`` and ``scale`` are kept
because a genuinely metric sensor (ARKit, a stereo rig) really is affine in
depth, and forcing it through a reciprocal only adds noise.

STATUS: prototype. The machinery and its tests are real; the COEFFICIENTS are
not — fitting them needs real captures, and the synthetic tests here only prove
that a known distortion is recovered. Do not ship fitted numbers taken from
ray-cast fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Below this many valid paired samples a fit is noise. ARKit depth is 256x192,
#: so even one frame clears it easily — the guard is for heavily masked pairs.
MIN_SAMPLES = 256

#: Depths at or below this (metres) are treated as invalid rather than near.
#: Zero is Record3D's "no return" value, and 1/0 would poison a disparity fit.
MIN_VALID_DEPTH = 1e-3

MODELS = ("affine_disparity", "affine", "scale")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("depth calibration requires numpy") from exc
    return np


@dataclass
class DepthCorrection:
    """A fitted correction, plus enough provenance to distrust it later."""

    model: str
    a: float
    b: float
    n_samples: int
    #: Median absolute error in metres BEFORE and AFTER, on the fitted data.
    #: Both are reported because a fit that barely improves the error is worth
    #: knowing about — and it is invisible if only the final number is shown.
    mae_before: float = 0.0
    mae_after: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Fraction of the original error removed. Negative means it made it worse."""
        if self.mae_before <= 0:
            return 0.0
        return (self.mae_before - self.mae_after) / self.mae_before

    def to_dict(self) -> dict:
        return {"model": self.model, "a": self.a, "b": self.b,
                "n_samples": self.n_samples, "mae_before": self.mae_before,
                "mae_after": self.mae_after, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: dict) -> "DepthCorrection":
        return cls(model=str(d["model"]), a=float(d["a"]), b=float(d["b"]),
                   n_samples=int(d.get("n_samples", 0)),
                   mae_before=float(d.get("mae_before", 0.0)),
                   mae_after=float(d.get("mae_after", 0.0)),
                   metadata=dict(d.get("metadata") or {}))


def _valid_pair(predicted, measured, mask, np):
    p = np.asarray(predicted, dtype=np.float64).ravel()
    m = np.asarray(measured, dtype=np.float64).ravel()
    if p.shape != m.shape:
        raise ValueError(
            f"predicted {p.shape} and measured {m.shape} must be the same shape — "
            "resample the LiDAR depth to the estimate's resolution first")
    ok = (np.isfinite(p) & np.isfinite(m)
          & (p > MIN_VALID_DEPTH) & (m > MIN_VALID_DEPTH))
    if mask is not None:
        ok &= np.asarray(mask).ravel().astype(bool)
    return p[ok], m[ok]


def apply_depth_correction(depth, correction: "DepthCorrection"):
    """Apply a fitted correction. Invalid samples stay invalid."""
    np = _require_numpy()
    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d > MIN_VALID_DEPTH)
    out = np.full(d.shape, np.nan, dtype=np.float64)

    if correction.model == "scale":
        out[ok] = d[ok] * correction.a
    elif correction.model == "affine":
        out[ok] = d[ok] * correction.a + correction.b
    elif correction.model == "affine_disparity":
        # Fitted as  1/measured = a * (1/pred) + b, so invert to get depth back.
        disp = correction.a / d[ok] + correction.b
        good = disp > 1e-9
        idx = np.where(ok)
        vals = np.full(disp.shape, np.nan)
        vals[good] = 1.0 / disp[good]
        out[idx] = vals
    else:
        raise ValueError(f"unknown correction model {correction.model!r}")

    # A correction must never turn depth negative — that is a sign flip, not a
    # rescale, and it would put geometry behind the camera.
    out[np.isfinite(out) & (out <= 0)] = np.nan
    return out.astype(np.float32)


def fit_depth_correction(predicted, measured, *, mask=None,
                         model: str = "affine_disparity") -> DepthCorrection:
    """Fit ``predicted -> measured``. See the module docstring on model choice."""
    np = _require_numpy()
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    p, m = _valid_pair(predicted, measured, mask, np)
    n = int(p.size)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"only {n} valid paired samples (need {MIN_SAMPLES}). A fit on this "
            "little data would encode noise as a correction.")

    mae_before = float(np.median(np.abs(p - m)))

    if model == "scale":
        # Least squares through the origin.
        a, b = float((p @ m) / (p @ p)), 0.0
    elif model == "affine":
        a, b = (float(v) for v in np.polyfit(p, m, 1))
    else:  # affine_disparity
        a, b = (float(v) for v in np.polyfit(1.0 / p, 1.0 / m, 1))

    corr = DepthCorrection(model=model, a=a, b=b, n_samples=n,
                           mae_before=mae_before)
    fitted = apply_depth_correction(p, corr)
    ok = np.isfinite(fitted)
    corr.mae_after = float(np.median(np.abs(fitted[ok] - m[ok]))) if ok.any() else float("inf")
    return corr


def choose_correction(predicted, measured, *, mask=None) -> DepthCorrection:
    """Fit every model and return whichever actually helps most.

    Deliberately not "always use disparity": which space fits best depends on
    the sensor and the scene, and asserting a preference the data does not
    support is how a plausible-looking correction makes things worse. The
    chosen model and both error figures ride along so the decision is auditable.
    """
    np = _require_numpy()
    results = []
    for m in MODELS:
        try:
            results.append(fit_depth_correction(predicted, measured, mask=mask, model=m))
        except Exception:  # noqa: BLE001 - a model that cannot fit is not a failure
            continue
    if not results:
        raise ValueError("no correction model could be fitted to this pair")
    best = min(results, key=lambda c: c.mae_after)
    best.metadata["candidates"] = {
        c.model: round(c.mae_after, 6) for c in results}
    if best.improvement <= 0:
        best.metadata["warning"] = (
            "no candidate reduced the error — the estimate is probably not "
            "affine-related to the measurement, and applying this would be "
            "cosmetic at best")
    return best
