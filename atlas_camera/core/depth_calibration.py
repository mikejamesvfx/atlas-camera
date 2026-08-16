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

A FIT IS ONLY VALID OVER THE RANGE IT SAW
-----------------------------------------
The argument above cuts both ways and the sample count does not capture it.
``MIN_SAMPLES`` counts PIXELS, and 400 adjacent pixels of one wall are one
surface at one depth, not 400 independent constraints. Measured on a synthetic
pair (2026-08-17): 400 samples spanning 1.00–1.30 m, fitted with 1% noise,
reported ``improvement=98.4%`` and ``mae_after=0.0074`` — an excellent-looking
correction. Applied at 50–250 m the same coefficients were out by a median of
**100.1 m, 67% of true depth**.

So every ``DepthCorrection`` records ``predicted_range``, and every application
reports how much of the incoming depth falls outside it. Extrapolation is not
forbidden — a mild overshoot is usually fine and refusing it outright would
make the correction useless at frame edges — but it is never silent, and
``on_extrapolation="nan"`` voids the out-of-range samples when the caller wants
the hard guarantee.

STATUS: prototype, DELIBERATELY UNWIRED. The machinery and its tests are real;
the COEFFICIENTS are not — fitting them needs real captures, and the synthetic
tests here only prove that a known distortion is recovered. Do not ship fitted
numbers taken from ray-cast fixtures.

No product code path calls this module, on purpose: it landed 2026-07-29, a
week before the Record3D/LiDAR capture nodes that would feed it, and the last
mile (a store for fitted coefficients keyed by model and scene type, then
selection and application in the depth chain) is not built. That last mile,
the capture set it needs, and the reason a LiDAR capture is NOT the case that
justifies it, are written up under "Deferred engineering backlog" in
docs/ROADMAP.md. Read that before wiring, deleting, or re-deriving any of it.
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
    #: (min, max) of the PREDICTED depths the fit saw, in metres. This is the
    #: range `apply_depth_correction` is entitled to; anything beyond it is
    #: extrapolation. It is the predicted side rather than the measured side
    #: because apply() is handed a model estimate, not ground truth.
    #: (0.0, 0.0) means UNKNOWN — a correction deserialized from before ranges
    #: were recorded — and is reported as such rather than treated as valid.
    predicted_range: tuple = (0.0, 0.0)
    #: (min, max) of the MEASURED depths, for provenance only. Answers "what
    #: kind of capture was this fitted on" when reading a stored coefficient.
    measured_range: tuple = (0.0, 0.0)
    metadata: dict = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Fraction of the original error removed. Negative means it made it worse."""
        if self.mae_before <= 0:
            return 0.0
        return (self.mae_before - self.mae_after) / self.mae_before

    @property
    def has_range(self) -> bool:
        lo, hi = self.predicted_range
        return hi > lo > 0

    @property
    def dynamic_range(self) -> float:
        """max/min of the fitted predictions. 1.0 = every sample at one depth.

        Reported rather than gated on: there is no threshold that separates a
        safe fit from an unsafe one without knowing where it will be applied,
        which is exactly what `apply_depth_correction` does know.
        """
        lo, hi = self.predicted_range
        return (hi / lo) if lo > 0 else 0.0

    def to_dict(self) -> dict:
        return {"model": self.model, "a": self.a, "b": self.b,
                "n_samples": self.n_samples, "mae_before": self.mae_before,
                "mae_after": self.mae_after,
                "predicted_range": list(self.predicted_range),
                "measured_range": list(self.measured_range),
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: dict) -> "DepthCorrection":
        def _range(key):
            v = d.get(key)
            return (float(v[0]), float(v[1])) if v else (0.0, 0.0)

        return cls(model=str(d["model"]), a=float(d["a"]), b=float(d["b"]),
                   n_samples=int(d.get("n_samples", 0)),
                   mae_before=float(d.get("mae_before", 0.0)),
                   mae_after=float(d.get("mae_after", 0.0)),
                   predicted_range=_range("predicted_range"),
                   measured_range=_range("measured_range"),
                   metadata=dict(d.get("metadata") or {}))


@dataclass
class CorrectionReport:
    """What applying a correction actually did to a depth map.

    `apply_depth_correction` used to return a bare array. Measured on a
    synthetic pair (2026-08-17), a plausible-looking correction turned 76% of a
    fully-valid frame into NaN and said nothing — and an all-NaN region is
    indistinguishable downstream from a region the depth model simply had no
    opinion about. Every node in this repo returns a report for the same
    reason; a core helper that degrades silently is the same defect one layer
    down.
    """

    n_input_valid: int = 0
    n_output_valid: int = 0
    #: Fraction of valid input samples the correction destroyed.
    lost_fraction: float = 0.0
    #: Fraction of valid input samples outside the fit's `predicted_range`.
    extrapolated_fraction: float = 0.0
    #: How far outside, as a multiple of the fitted bound. 1.0 = at the edge.
    extrapolation_ratio: float = 1.0
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        return {"n_input_valid": self.n_input_valid,
                "n_output_valid": self.n_output_valid,
                "lost_fraction": self.lost_fraction,
                "extrapolated_fraction": self.extrapolated_fraction,
                "extrapolation_ratio": self.extrapolation_ratio,
                "warnings": list(self.warnings)}

    def __str__(self) -> str:
        head = (f"{self.n_output_valid}/{self.n_input_valid} samples corrected"
                f" ({self.lost_fraction:.1%} lost,"
                f" {self.extrapolated_fraction:.1%} extrapolated)")
        return head + ("" if self.ok else "\n  ! " + "\n  ! ".join(self.warnings))


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


#: A correction is entitled to its fitted range. Beyond this multiple of it,
#: say so loudly: the measured failure was 192x outside and 67% wrong.
EXTRAPOLATION_WARN_RATIO = 2.0

#: Losing more than this fraction of valid samples is a defect, not a detail.
LOST_WARN_FRACTION = 0.05


def apply_depth_correction(depth, correction: "DepthCorrection", *,
                           on_extrapolation: str = "report"):
    """Apply a fitted correction. Returns ``(corrected, CorrectionReport)``.

    Invalid samples stay invalid. ``on_extrapolation`` is ``"report"`` (default
    — apply everywhere, say what happened) or ``"nan"`` (void samples outside
    the fit's range). It is never a silent clamp: clamping would return
    confident depth values the fit has no evidence for, which is the failure
    this reporting exists to surface.
    """
    np = _require_numpy()
    if on_extrapolation not in ("report", "nan"):
        raise ValueError(
            f"on_extrapolation must be 'report' or 'nan', got {on_extrapolation!r}")

    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d > MIN_VALID_DEPTH)
    out = np.full(d.shape, np.nan, dtype=np.float64)
    report = CorrectionReport(n_input_valid=int(ok.sum()))

    # --- extrapolation, measured before anything is applied ------------------
    lo, hi = correction.predicted_range
    outside = np.zeros(d.shape, dtype=bool)
    if correction.has_range:
        outside = ok & ((d < lo) | (d > hi))
        if report.n_input_valid:
            report.extrapolated_fraction = float(outside.sum()) / report.n_input_valid
        if outside.any():
            vals = d[outside]
            report.extrapolation_ratio = float(max(
                (vals.max() / hi) if vals.max() > hi else 1.0,
                (lo / vals.min()) if 0 < vals.min() < lo else 1.0))
        if report.extrapolation_ratio >= EXTRAPOLATION_WARN_RATIO:
            report.warnings.append(
                f"{report.extrapolated_fraction:.0%} of samples are outside the "
                f"fitted range {lo:.2f}–{hi:.2f} m, by up to "
                f"{report.extrapolation_ratio:.0f}x. The coefficients carry no "
                f"evidence there; a low mae_after says nothing about it.")
    elif report.n_input_valid:
        report.warnings.append(
            "this correction records no fitted range, so extrapolation cannot "
            "be checked — refit it, or treat the result as unverified")

    if on_extrapolation == "nan":
        ok = ok & ~outside

    if correction.model == "scale":
        out[ok] = d[ok] * correction.a
    elif correction.model == "affine":
        out[ok] = d[ok] * correction.a + correction.b
    elif correction.model == "affine_disparity":
        # Fitted as  1/measured = a * (1/pred) + b, so invert to get depth back.
        disp = correction.a / d[ok] + correction.b
        vals = np.full(disp.shape, np.nan)
        good = disp > 1e-9
        vals[good] = 1.0 / disp[good]
        out[ok] = vals
    else:
        raise ValueError(f"unknown correction model {correction.model!r}")

    # A correction must never turn depth negative — that is a sign flip, not a
    # rescale, and it would put geometry behind the camera.
    out[np.isfinite(out) & (out <= 0)] = np.nan

    report.n_output_valid = int(np.isfinite(out).sum())
    if report.n_input_valid:
        report.lost_fraction = 1.0 - report.n_output_valid / report.n_input_valid
    if report.lost_fraction > LOST_WARN_FRACTION:
        report.warnings.append(
            f"{report.lost_fraction:.0%} of valid samples did not survive the "
            f"correction ({report.n_input_valid} in, {report.n_output_valid} "
            f"out). An all-NaN region reads downstream as 'no depth here', not "
            f"as 'the correction failed here'.")

    return out.astype(np.float32), report


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

    corr = DepthCorrection(
        model=model, a=a, b=b, n_samples=n, mae_before=mae_before,
        # Recorded BEFORE the self-application below, so the fit is not
        # reported as extrapolating against its own data.
        predicted_range=(float(p.min()), float(p.max())),
        measured_range=(float(m.min()), float(m.max())),
    )
    fitted, _ = apply_depth_correction(p, corr)
    ok = np.isfinite(fitted)
    corr.mae_after = float(np.median(np.abs(fitted[ok] - m[ok]))) if ok.any() else float("inf")

    # A narrow fit is not refused — there is no threshold that separates safe
    # from unsafe without knowing where it will be applied, and apply() is what
    # knows that. It is recorded so a stored coefficient carries the caveat.
    if corr.dynamic_range < 2.0:
        corr.metadata["narrow_fit"] = (
            f"predictions span only {corr.predicted_range[0]:.2f}–"
            f"{corr.predicted_range[1]:.2f} m ({corr.dynamic_range:.2f}x). "
            "Two coefficients fitted over one depth are barely constrained; "
            "expect this to extrapolate badly.")
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
