"""Camera-space falsification of geometry against the original photograph.

WHY THIS MODULE EXISTS. Atlas could render geometry into a solved camera
(:mod:`atlas_camera.core.projection_render`, :mod:`atlas_camera.core.move_budget`)
and could score a torn mesh against SYNTHETIC truth
(:mod:`atlas_camera.core.tear_metrics`, which requires ``truth_depth`` and
``surface_ids``), but there was nothing that scored geometry against a REAL
plate. The cost of that gap was measured: the 2026-08-20 hole-splat run on
``sh001`` reported "closed 100.0% of hole" while painting 143,465 pixels
outside it, because the only number in the report was hole closure and the
error was computed against the frames the optimiser had already been shown.

So the doctrine here is the same as :mod:`atlas_camera.dynamic.fill_metrics`:

* every metric must be able to FAIL, and a test proves it does;
* an unavailable input reports ``available=False``, never a convenient ``0.0``
  that reads as "no violation" for a check that was never run;
* a candidate is only ever reported beside the do-nothing baseline it must
  beat — :func:`falsification_report` takes both or raises.

Gates come in two kinds, and the difference is stated per metric:

``definitional``
    The threshold follows from what the photograph means. Geometry standing in
    observed sky is wrong; alpha outside the region a candidate was authorised
    to occupy is spill; a depth ordering worse than a coin flip is not a
    prediction. These gate now.

``uncalibrated``
    An empirical threshold. ``pass`` is ``None`` until the perturbation sweep
    (``tools/calibrate_falsification.py``) measures what broken geometry
    actually reads, exactly as ``fill_metrics`` documents 17.6 / 2.2 / 5.8 from
    the DSC_2289 plate. A number invented at the keyboard is as unfalsifiable
    as no number.

One honest limit, stated once: reprojection can only falsify where the
photograph carries evidence. The occluded volume carries none. These metrics
discipline the silhouette and the visible seam; they say nothing about what is
behind an occluder, and must never be quoted as if they did.

Layering: ``core`` only — numpy, no torch, no ComfyUI, no ``dynamic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas_camera.core.mask_ops import dilate

#: A depth ordering must at minimum beat a coin flip. Definitional, not tuned.
CHANCE_DEPTH_AGREEMENT = 0.5

#: Antialiasing at a silhouette puts a thin fringe of alpha wherever the
#: geometry's edge falls, so neither definitional gate is set at a hard zero.
MAX_SKY_VIOLATION = 0.005
MAX_SPILL_FRACTION = 0.01

#: Below these, a "score" is arithmetic on too little evidence: two covered
#: pixels can agree by chance and read 1.000.
MIN_DEPTH_PIXELS = 64
MIN_DEPTH_PAIRS = 64

#: Both measured by ``tools/calibrate_falsification.py`` on 2026-08-20, on an
#: analytic two-box scene plus four X-H2 plates from atlas_raws/MultiShots.
#:
#: Seam ratio, self-referenced rim (clean / flat-smear / wrong-content /
#: exposure-offset), per plate:
#:     DSCF3912  1.000 / 1.426 / 1.751 / 4.729
#:     DSCF3916  1.000 / 1.475 / 1.885 / 4.713
#:     DSCF3921  1.000 / 1.890 / 2.548 / 9.808
#:     DSCF3928  1.000 / 1.705 / 2.123 / 6.526
#: 1.25 sits below the weakest defect measured anywhere (1.426) and above the
#: clean join. HONEST LIMIT: the clean side was measured on a composite that
#: IS the plate, so it carries no antialiasing at the rim; a correct real
#: composite will read somewhat above 1.0 and this gate may need raising once
#: one has actually been measured. It corroborates the independently measured
#: 1.5 in ``dynamic/fill_metrics.py`` rather than being derived from it.
MAX_SEAM_GRADIENT_RATIO: float | None = 1.25

#: Silhouette IoU against the truth render of the same analytic scene:
#:     truth 1.000 | yaw 5deg 0.964 | translate 0.1 m 0.960 | scale 10% 0.892
#:     translate 0.5 m 0.821 | scale 30% 0.727 | translate 1.0 m 0.648
#:     depth swap 0.397 | ground plane only 0.227
#: 0.90 separates a tolerable pose error from geometry that is in the wrong
#: place. It is a threshold on THIS fixture's occluder sizes; a scene whose
#: entities subtend far fewer pixels will need it re-measured.
MIN_SILHOUETTE_IOU: float | None = 0.90

_EPS = 1e-9

#: Sign per metric: +1 when a larger value is better, -1 when smaller is.
_METRIC_DIRECTION = {
    "sky_violation": -1,
    "containment": +1,
    "silhouette_iou": +1,
    "depth_order_agreement": +1,
    "seam_gradient_ratio": -1,
}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.plate_falsification requires numpy. Install "
            "with: pip install -e .[vision]") from exc
    return np


def _as_bool(np: Any, mask: Any, name: str, shape: tuple[int, int]) -> Any:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == np.bool_:
        out = m
    elif m.dtype == np.uint8:
        out = m > 127
    else:
        out = m > 0.5
    if out.shape != shape:
        raise ValueError(f"{name} {out.shape} must match alpha {shape}")
    return out


def _as_rgb(np: Any, img: Any, name: str, shape: tuple[int, int]) -> Any:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3 or a.shape[-1] < 3:
        raise ValueError(f"{name}: expected HxWx3, got shape {a.shape}")
    if a.shape[:2] != shape:
        raise ValueError(f"{name} {a.shape[:2]} must match alpha {shape}")
    return a[..., :3]


def _as_field(np: Any, buf: Any, name: str, shape: tuple[int, int]) -> Any:
    a = np.asarray(buf, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} {a.shape} must match alpha {shape}")
    return a


def _unavailable(reason: str, **extra: Any) -> dict:
    return {"value": None, "available": False, "pass": None,
            "gate": "unavailable", "reason": reason, **extra}


def _gradient_mag(np: Any, img: Any) -> Any:
    grey = np.asarray(img, dtype=np.float64).mean(axis=2)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


# ---------------------------------------------------------------------------
# Rasterization helper
# ---------------------------------------------------------------------------

def rasterize_candidate(vertices: Any, faces: Any, *, view_matrix: Any,
                        fx: float, fy: float, cx: float, cy: float,
                        width: int, height: int, backend: str = "auto"):
    """Rasterize candidate geometry into a solved camera.

    Thin pass-through to :func:`atlas_camera.core.move_budget.rasterize_coverage`
    so callers of this module do not each rediscover which of the several
    rasterizers in the tree returns a true metric z-buffer. Returns
    ``(alpha HxW bool, depth HxW float64 metres, inf where uncovered)``.

    ``view_matrix`` is the full 4x4 ``camera_view_matrix``. Never pass a 3x3
    rotation: the transpose ambiguity produces a mirrored mask that still looks
    plausible.
    """
    np = _require_numpy()
    vm = np.asarray(view_matrix, dtype=np.float64)
    if vm.shape != (4, 4):
        raise ValueError(f"view_matrix must be 4x4, got {vm.shape}")
    from atlas_camera.core.move_budget import rasterize_coverage

    return rasterize_coverage(
        vertices, faces, view_matrix=vm, fx=fx, fy=fy, cx=cx, cy=cy,
        width=int(width), height=int(height), backend=backend,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sky_violation(np: Any, alpha: Any, sky_mask: Any, shape) -> dict:
    if sky_mask is None:
        return _unavailable("no observed sky mask supplied")
    sky = _as_bool(np, sky_mask, "sky_mask", shape)
    violating = int((alpha & sky).sum())
    value = violating / float(alpha.sum())
    return {
        "value": value,
        "available": True,
        "pass": bool(value <= MAX_SKY_VIOLATION),
        "gate": "definitional",
        "threshold": MAX_SKY_VIOLATION,
        "violating_px": violating,
        "sky_px": int(sky.sum()),
    }


def _containment(np: Any, alpha: Any, authorised_mask: Any, shape) -> dict:
    if authorised_mask is None:
        return _unavailable("no authorised region supplied")
    authorised = _as_bool(np, authorised_mask, "authorised_mask", shape)
    if not bool(authorised.any()):
        return _unavailable("authorised region is empty")
    spill = int((alpha & ~authorised).sum())
    alpha_px = float(alpha.sum())
    value = 1.0 - spill / alpha_px
    # Closure is reported BESIDE containment, never instead of it: closure
    # alone is the metric that called a degenerate splat fill a success.
    closure = float((alpha & authorised).sum()) / float(authorised.sum())
    return {
        "value": value,
        "available": True,
        "pass": bool(spill / alpha_px <= MAX_SPILL_FRACTION),
        "gate": "definitional",
        "threshold": 1.0 - MAX_SPILL_FRACTION,
        "spill_px": spill,
        "closure": closure,
        "alpha_px": int(alpha_px),
    }


def _silhouette_iou(np: Any, alpha: Any, observed_mask: Any, shape) -> dict:
    if observed_mask is None:
        return _unavailable("no observed silhouette mask supplied")
    observed = _as_bool(np, observed_mask, "observed_mask", shape)
    union = int((alpha | observed).sum())
    if union == 0:
        return _unavailable("silhouette union is empty")
    value = float((alpha & observed).sum()) / float(union)
    gated = MIN_SILHOUETTE_IOU is not None
    return {
        "value": value,
        "available": True,
        "pass": bool(value >= MIN_SILHOUETTE_IOU) if gated else None,
        "gate": "empirical" if gated else "uncalibrated",
        "threshold": MIN_SILHOUETTE_IOU,
    }


def _depth_order_agreement(np: Any, alpha: Any, render_depth: Any,
                           reference_depth: Any, shape, *, pairs: int,
                           seed: int) -> dict:
    if render_depth is None or reference_depth is None:
        return _unavailable("needs both a render depth buffer and a reference")
    rd = _as_field(np, render_depth, "render_depth", shape)
    ref = _as_field(np, reference_depth, "reference_depth", shape)

    valid = alpha & np.isfinite(rd) & np.isfinite(ref)
    n_valid = int(valid.sum())
    if n_valid < MIN_DEPTH_PIXELS:
        return _unavailable(
            f"only {n_valid} comparable pixels (need {MIN_DEPTH_PIXELS})",
            n_valid_px=n_valid)

    idx = np.flatnonzero(valid.reshape(-1))
    rng = np.random.default_rng(int(seed))
    n = int(pairs)
    i = rng.integers(0, idx.size, size=n)
    j = rng.integers(0, idx.size, size=n)
    a, b = idx[i], idx[j]

    rd_flat, ref_flat = rd.reshape(-1), ref.reshape(-1)
    d_render = rd_flat[a] - rd_flat[b]
    d_ref = ref_flat[a] - ref_flat[b]

    # ORDER, not metres. Monocular depth carries scale+shift error
    # (design-rules.md:141), so any comparison of absolute values would call a
    # perfectly ordered prediction a failure. Ties in either buffer carry no
    # ordering information and are dropped rather than counted as agreement.
    keep = (np.abs(d_render) > _EPS) & (np.abs(d_ref) > _EPS)
    n_pairs = int(keep.sum())
    if n_pairs < MIN_DEPTH_PAIRS:
        return _unavailable(
            f"only {n_pairs} non-tied pairs (need {MIN_DEPTH_PAIRS})",
            n_pairs=n_pairs, n_valid_px=n_valid)

    agree = np.sign(d_render[keep]) == np.sign(d_ref[keep])
    value = float(agree.mean())
    return {
        "value": value,
        "available": True,
        "pass": bool(value > CHANCE_DEPTH_AGREEMENT),
        "gate": "definitional",
        "threshold": CHANCE_DEPTH_AGREEMENT,
        "n_pairs": n_pairs,
        "n_valid_px": n_valid,
    }


def _seam_gradient_ratio(np: Any, alpha: Any, plate: Any, composite: Any,
                         shape, *, rim_px: int) -> dict:
    if plate is None or composite is None:
        return _unavailable("needs both the plate and the composited render")
    p = _as_rgb(np, plate, "plate", shape)
    c = _as_rgb(np, composite, "composite", shape)

    rim = dilate(alpha, max(1, int(rim_px))) & ~alpha
    plate_ref = ~alpha & ~rim
    if not bool(rim.any()) or not bool(plate_ref.any()):
        return _unavailable("no rim or no plate reference region")

    plate_grad_map = _gradient_mag(np, p)
    rim_grad = float(_gradient_mag(np, c)[rim].mean())
    # Reference the rim against ITSELF on the plate, not against the frame
    # average. Measured on DSCF3916: a perfectly clean join scored 1.37 under
    # the frame-average reference, purely because object silhouettes sit on
    # busier content than the mean — an absolute gate would then be
    # plate-dependent and would fail honest geometry on a detailed photograph.
    # Self-referenced, a clean join is 1.0 on every plate by construction.
    plate_rim_grad = float(plate_grad_map[rim].mean())
    plate_grad = float(plate_grad_map[plate_ref].mean())
    if plate_rim_grad <= _EPS:
        # A featureless rim has no scale to measure against. A composite rim
        # that is equally featureless is not a seam; anything else is one, and
        # the ratio is genuinely unbounded rather than conveniently large.
        value = 1.0 if rim_grad <= _EPS else float("inf")
    else:
        value = rim_grad / plate_rim_grad

    gated = MAX_SEAM_GRADIENT_RATIO is not None
    return {
        "value": value,
        "available": True,
        "pass": bool(value <= MAX_SEAM_GRADIENT_RATIO) if gated else None,
        "gate": "empirical" if gated else "uncalibrated",
        "threshold": MAX_SEAM_GRADIENT_RATIO,
        "rim_gradient": rim_grad,
        "plate_rim_gradient": plate_rim_grad,
        "plate_gradient": plate_grad,
    }


def score_geometry_against_plate(
    *,
    alpha: Any,
    render_depth: Any = None,
    plate: Any = None,
    composite: Any = None,
    sky_mask: Any = None,
    authorised_mask: Any = None,
    observed_mask: Any = None,
    reference_depth: Any = None,
    rim_px: int = 2,
    depth_pairs: int = 20000,
    seed: int = 0,
) -> dict:
    """Score one candidate's reprojection against the photograph.

    ``alpha`` is the candidate's coverage in the plate raster (bool, uint8 or
    float). Everything else is optional evidence; each metric that lacks its
    input reports ``available=False`` rather than a passing score.

    ``reference_depth`` must be oriented FARTHER = LARGER. Pass metric depth or
    any monotone function of it; the metric compares ordering only, so scale
    and shift are irrelevant, but an inverse-depth (disparity) buffer must be
    inverted by the caller or every ordering reads backwards.

    ``composite`` is the candidate's render pasted over whatever sits behind
    it, at the plate raster — the seam metric needs the assembled image, not
    the candidate in isolation.
    """
    np = _require_numpy()
    a = np.asarray(alpha)
    if a.ndim == 3:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"alpha must be HxW, got shape {a.shape}")
    shape = (int(a.shape[0]), int(a.shape[1]))
    alpha_b = _as_bool(np, a, "alpha", shape)
    if not bool(alpha_b.any()):
        raise ValueError("alpha is empty: nothing was rasterized to score")

    return {
        "alpha_px": int(alpha_b.sum()),
        "raster": {"height": shape[0], "width": shape[1]},
        "sky_violation": _sky_violation(np, alpha_b, sky_mask, shape),
        "containment": _containment(np, alpha_b, authorised_mask, shape),
        "silhouette_iou": _silhouette_iou(np, alpha_b, observed_mask, shape),
        "depth_order_agreement": _depth_order_agreement(
            np, alpha_b, render_depth, reference_depth, shape,
            pairs=depth_pairs, seed=seed),
        "seam_gradient_ratio": _seam_gradient_ratio(
            np, alpha_b, plate, composite, shape, rim_px=rim_px),
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FalsificationReport:
    """A candidate scored beside the do-nothing render it has to beat.

    There is no constructor that omits the baseline. The hole-splat run that
    prompted this module reported a candidate alone, and a candidate alone
    always sounds like progress.
    """

    candidate: dict
    baseline: dict
    deltas: dict = field(default_factory=dict)
    beats_baseline: bool = False
    #: "better" / "worse" / "inconclusive". A bare False conflates "the
    #: candidate lost" with "no metric had the evidence to judge it", and
    #: keeping those apart is the whole reason this module exists.
    verdict: str = "inconclusive"
    n_metrics_compared: int = 0

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "deltas": self.deltas,
            "beats_baseline": self.beats_baseline,
            "verdict": self.verdict,
            "n_metrics_compared": self.n_metrics_compared,
        }


def falsification_report(*, candidate: dict, baseline: dict) -> FalsificationReport:
    """Score ``candidate`` and ``baseline`` identically and compare them.

    Both arguments are keyword dicts forwarded verbatim to
    :func:`score_geometry_against_plate`, so the two runs cannot silently
    differ in which evidence they were given.
    """
    cand = score_geometry_against_plate(**candidate)
    base = score_geometry_against_plate(**baseline)

    deltas: dict[str, float] = {}
    for name, direction in _METRIC_DIRECTION.items():
        c, b = cand[name], base[name]
        if not (c["available"] and b["available"]):
            continue
        cv, bv = c["value"], b["value"]
        if name == "seam_gradient_ratio":
            # Better means CLOSER TO THE PLATE'S OWN STATISTICS, not smaller:
            # a rim flatter than the plate is its own defect (a smear).
            cv, bv = abs(cv - 1.0), abs(bv - 1.0)
        delta = direction * (cv - bv) if direction > 0 else (bv - cv)
        if delta != delta:  # NaN
            continue
        deltas[name] = float(delta)

    beats = bool(deltas) and all(v >= 0.0 for v in deltas.values()) \
        and any(v > 0.0 for v in deltas.values())
    if not deltas:
        verdict = "inconclusive"          # no metric had the evidence to judge
    elif beats:
        verdict = "better"
    elif any(v < 0.0 for v in deltas.values()):
        verdict = "worse"
    else:
        verdict = "inconclusive"          # every comparable metric tied
    return FalsificationReport(candidate=cand, baseline=base, deltas=deltas,
                               beats_baseline=beats, verdict=verdict,
                               n_metrics_compared=len(deltas))
