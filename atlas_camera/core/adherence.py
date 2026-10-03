"""Did the generator obey the camera? Scored where Atlas knows the answer.

THE MEASUREMENT PROBLEM THIS SOLVES. The obvious way to ask whether a generated
move obeyed a requested camera is to solve the generated frames and compare the
recovered trajectory. That route is measured shut: ``comfy.nodes_fill`` records
that the real plate registers against the primary at 1022 SIFT inliers while
EVERY frame of an LTX CrossView move -- including one at almost zero parallax --
collapses to 12-20 and is refused. Video diffusion re-synthesises exactly the
fine texture feature matching depends on. So no amount of care makes SfM on a
generated frame into evidence.

THE WAY THROUGH is that Atlas does not need to recover the camera, because it
already knows it. ``core.ghost_pixels`` partitions each target frame into VALID
(the rasteriser covered this pixel, so Atlas knows its appearance by
reprojection) and GHOST (genuinely newly-visible, the only class a generator
should be inventing). That gives two disjoint questions with two different
answers:

- **camera adherence**, scored in VALID only, against Atlas's own deterministic
  reprojection. Needs no registration, no features, no second photograph.
- **generation quality**, judged in GHOST only, where there is nothing to
  compare against and coverage is the honest measure.

WHY GRADIENT ZNCC IS THE HEADLINE. Generated frames come back sRGB and often
graded -- ``dynamic.ltx_comfy`` stamps its results ``unverified_i2v`` with an
sRGB output colour space precisely because the generator's tone is not under
Atlas's control. A raw-intensity headline would therefore be measuring the grade
rather than the camera. Taking the gradient removes the DC offset (an exposure
lift) and normalising the correlation removes the gain, so what survives is
structure: is the edge where we said it would be. The fitted gain/offset is
reported alongside as a measured number instead of being left as a confound.

THE DEGENERACY THAT MAKES NAIVE VERSIONS OF THIS METRIC WORTHLESS. A generator
that ignores the prompt and returns frame 0 unchanged scores PERFECT adherence
in VALID, because a static plate agrees with Atlas's reprojection wherever the
move is small. Any headline number published without testing for this is
meaningless. ``parallax_response`` is the test: adherence against frame i minus
adherence against frame 0. A frozen clip scores <= 0 there by construction, and
this module REFUSES to publish rather than failing soft.

Host-agnostic: numpy only, no torch, no ComfyUI.
"""
from __future__ import annotations

from typing import Any

from atlas_camera.core.conditioning import PARALLAX_FLOOR_PX
from atlas_camera.core.ghost_pixels import GhostClass

#: The arm with no geometric conditioning at all. Required, not optional: two
#: numbers with nothing to compare them against is not evidence, so the API
#: refuses rather than letting a caller compute a lone adherence figure.
CONTROL_ARM = "prompt_only"

#: The arm under test.
ATLAS_ARM = "atlas"

#: Synthesised internally: the arm-under-test's own first frame, repeated. Its
#: adherence is the ceiling a generator earns by ignoring the camera entirely,
#: which is why the headline is always reported as a margin over it.
STATIC_ARM = "static"

#: Window radius for the local-statistics measures.
SSIM_RADIUS = 3

#: A window must be at least this fraction VALID to be scored at all. Below it
#: the window straddles a class boundary and its statistics mix a known surface
#: with pixels Atlas has no opinion about.
SSIM_MIN_VALID_FRAC = 0.9

#: A variance below this is indistinguishable from zero. `E[x^2] - E[x]^2`
#: cancels catastrophically on a flat window and leaves float noise around
#: 1e-17, so a floor derived from genuinely flat plate content (a checkerboard's
#: interior, a clear sky) lands at exactly 0.0 and that noise then clears it: a
#: completely flat grey fill measured 0.08 "coverage" made entirely of rounding
#: error. The floor is lifted to this value so zero variance reads as zero.
VARIANCE_EPS = 1e-12

#: SSIM stabilisers, for data in [0, 1].
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.adherence requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def gradient_magnitude(np: Any, img: Any) -> Any:
    """Greyscale gradient magnitude.

    Three lines, duplicated rather than imported: the two existing copies
    (``dynamic.fill_metrics._gradient_mag``, ``core.plate_falsification.
    _gradient_mag``) are both private, and reaching into a sibling module's
    private name is a worse dependency than restating ``np.gradient``. If the
    recipe ever changes, all three change.
    """
    grey = np.asarray(img, dtype=np.float64)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


def _box_sum(np: Any, field: Any, radius: int) -> Any:
    """Sum over a (2r+1)^2 window, edges CLAMPED (not wrapped, not zeroed).

    Summed-area table rather than a blur: the masked statistics need the raw
    sum and the sample COUNT separately, and a normalised blur has already
    thrown the count away.
    """
    pad = radius
    padded = np.pad(np.asarray(field, dtype=np.float64), pad, mode="edge")
    csum = padded.cumsum(axis=0).cumsum(axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    size = 2 * radius + 1
    h, w = np.asarray(field).shape[:2]
    return (csum[size:size + h, size:size + w] - csum[0:h, size:size + w]
            - csum[size:size + h, 0:w] + csum[0:h, 0:w])


def zncc(np: Any, a: Any, b: Any) -> float:
    """Zero-mean normalised cross-correlation of two 1-D samples.

    Returns NaN for a degenerate sample (fewer than two pixels, or either side
    constant). NaN rather than 0.0: a flat region is a case with NO answer, and
    0.0 is a legitimate answer meaning "uncorrelated".
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size != a.size:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def gradient_zncc(np: Any, generated: Any, reference: Any, mask: Any) -> float:
    """THE HEADLINE. Gradient ZNCC over ``mask`` only.

    Gradients are taken on the FULL frame and only then masked, because a
    gradient computed inside a cropped region invents an edge at the crop
    boundary that no surface has.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    return zncc(np, gradient_magnitude(np, generated)[m],
                gradient_magnitude(np, reference)[m])


def masked_ssim(np: Any, generated: Any, reference: Any, mask: Any, *,
                radius: int = SSIM_RADIUS,
                min_valid_frac: float = SSIM_MIN_VALID_FRAC) -> float:
    """Local-statistics agreement, DIAGNOSTIC ONLY -- never the headline.

    Reported as a second opinion, deliberately not promoted: two correlated
    headline numbers invite picking whichever one reads better after the fact.

    Windows are scored only where at least ``min_valid_frac`` of the window is
    inside the mask, so a window straddling a class boundary never mixes a
    surface Atlas knows with pixels it has no opinion about.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    a = np.asarray(generated, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)

    w = m.astype(np.float64)
    n = _box_sum(np, w, radius)
    size = (2 * radius + 1) ** 2
    scored = (n / float(size)) >= min_valid_frac
    if not scored.any():
        return float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(n > 0, n, 1.0)
        mu_a = _box_sum(np, a * w, radius) / safe
        mu_b = _box_sum(np, b * w, radius) / safe
        va = _box_sum(np, a * a * w, radius) / safe - mu_a ** 2
        vb = _box_sum(np, b * b * w, radius) / safe - mu_b ** 2
        cov = _box_sum(np, a * b * w, radius) / safe - mu_a * mu_b
        va = np.maximum(va, 0.0)
        vb = np.maximum(vb, 0.0)
        ssim = (((2 * mu_a * mu_b + _C1) * (2 * cov + _C2))
                / ((mu_a ** 2 + mu_b ** 2 + _C1) * (va + vb + _C2)))
    return float(np.nanmean(ssim[scored]))


def fit_grade(np: Any, generated: Any, reference: Any, mask: Any) -> dict:
    """Per-channel least-squares gain+offset taking reference -> generated.

    The generator's tone shift, measured instead of assumed away. ``residual``
    is the mean absolute error AFTER correction, so it scores structure the way
    the headline does while the gain/offset say how much grading had to be
    removed to get there.
    """
    m = np.asarray(mask, dtype=bool)
    a = np.asarray(generated, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    if not m.any():
        return {"gain": None, "offset": None, "residual": None}

    gains, offsets, residuals = [], [], []
    for c in range(a.shape[2]):
        y, x = a[..., c][m], b[..., c][m]
        if x.size < 2 or float(x.std()) < 1e-12:
            gains.append(1.0)
            offsets.append(0.0)
            residuals.append(float(np.abs(y - x).mean()))
            continue
        gain, offset = np.polyfit(x, y, 1)
        gains.append(float(gain))
        offsets.append(float(offset))
        residuals.append(float(np.abs(y - (gain * x + offset)).mean()))
    return {"gain": gains, "offset": offsets,
            "residual": float(np.mean(residuals))}


def raw_error(np: Any, generated: Any, reference: Any, mask: Any) -> dict:
    """MAE and PSNR. REPORTED, NOT THE HEADLINE -- they measure the grade.

    Kept because a reader will ask for them, and because a large MAE beside a
    high gradient ZNCC is the positive signature of a re-graded but
    geometrically obedient frame.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"mae": None, "psnr_db": None,
                "note": "no scored pixels in this frame"}
    a = np.asarray(generated, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    sel = m if a.ndim == 2 else np.repeat(m[..., None], a.shape[2], axis=2)
    diff = (a - b)[sel]
    mse = float((diff ** 2).mean())
    psnr = float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))
    return {"mae": float(np.abs(diff).mean()), "psnr_db": psnr,
            "note": "grade-sensitive; gradient_zncc is the headline"}


def _erode(np: Any, mask: Any, iterations: int = 1) -> Any:
    """8-connected erosion (``core.mask_ops.erode``)."""
    from atlas_camera.core.mask_ops import erode
    return erode(mask, iterations, connectivity=8)


def _nearest_resize(np: Any, arr: Any, height: int, width: int) -> Any:
    """Nearest-neighbour index remap to (height, width), leading axes kept."""
    src_h, src_w = arr.shape[-2:] if arr.ndim == 2 else arr.shape[:2][-2:]
    a = np.asarray(arr)
    src_h, src_w = a.shape[0], a.shape[1]
    if (src_h, src_w) == (height, width):
        return a
    rows = np.clip((np.arange(height) * src_h) // height, 0, src_h - 1)
    cols = np.clip((np.arange(width) * src_w) // width, 0, src_w - 1)
    return a[np.ix_(rows, cols)] if a.ndim == 2 else a[rows][:, cols]


def align_rasters(np: Any, frames: Any, class_map: Any, reference: Any) -> tuple:
    """Bring generated frames, class map and reference onto ONE raster.

    Resamples to the SMALLER of the two rasters -- upsampling a generated frame
    would invent detail the model never produced and flatter the measure.

    THE CLASS MAP IS RESAMPLED NEAREST AND THEN ERODED 1 px. Nearest because an
    interpolated class value is a class no pixel has; eroded because even a
    nearest remap lands class boundaries a pixel either side of the truth, and
    an un-eroded boundary bleeds GHOST into VALID -- which would score invented
    pixels as though Atlas knew their appearance, the one error this whole
    module exists to avoid.
    """
    gen = np.asarray(frames, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    cls = np.asarray(class_map)

    gh, gw = gen.shape[1], gen.shape[2]
    rh, rw = ref.shape[1], ref.shape[2]
    height, width = min(gh, rh), min(gw, rw)

    gen_out = np.stack([_nearest_resize(np, f, height, width) for f in gen])
    ref_out = np.stack([_nearest_resize(np, f, height, width) for f in ref])
    cls_out = np.stack([_erode(np, _nearest_resize(np, c, height, width) ==
                               int(GhostClass.VALID))
                        for c in cls])
    ghost_out = np.stack([_erode(np, _nearest_resize(np, c, height, width) ==
                                 int(GhostClass.GHOST))
                          for c in cls])
    resampled = (height, width) != (gh, gw) or (height, width) != (rh, rw)
    return gen_out, ref_out, cls_out, ghost_out, {
        "raster": [width, height],
        "generated_raster": [gw, gh],
        "reference_raster": [rw, rh],
        "resampled": bool(resampled),
        "class_map": "nearest + 1px erosion",
    }


def ghost_fill_coverage(np: Any, frame: Any, valid: Any, ghost: Any, *,
                        radius: int = SSIM_RADIUS,
                        percentile: float = 10.0,
                        min_pure_frac: float = 1.0) -> dict:
    """Did the generator put real texture in the pixels it was asked to invent?

    The variance floor is MEASURED from this same frame's VALID local-variance
    distribution rather than being a constant: what counts as "textured" is a
    property of the plate and the grade, not a number that transfers between
    scenes. A flat grey fill lands below the floor; plausible detail lands
    above it.

    ONLY WINDOWS ENTIRELY INSIDE GHOST ARE SCORED. Without that rule the
    measure is dominated by boundary bleed: a disocclusion is often a thin
    sliver, so a window centred a few pixels inside it still sees the textured
    plate next door, and a completely flat grey fill measured 0.50 coverage in
    exactly that way. Relaxing the purity to the 0.9 that ``masked_ssim`` uses
    still left 0.08, because a 7x7 window at 90% admits four contaminated
    pixels and four textured pixels are enough to clear a p10 floor -- so this
    measure takes the strict threshold. ``masked_ssim`` can afford 0.9 because
    it needs enough windows to average; a texture floor does not.

    The count of scorable pixels is reported so a caller can see when a sliver
    is thinner than the window and gets no verdict at all, rather than reading
    a confident number off two windows.
    """
    g = np.asarray(ghost, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    if not g.any():
        return {"coverage": None, "reason": "no ghost pixels in this frame"}
    if not v.any():
        return {"coverage": None, "reason": "no valid pixels to set a floor"}
    a = np.asarray(frame, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)

    size = float((2 * radius + 1) ** 2)
    mu = _box_sum(np, a, radius) / size
    var = np.maximum(_box_sum(np, a * a, radius) / size - mu ** 2, 0.0)

    floor = max(float(np.percentile(var[v], percentile)), VARIANCE_EPS)
    pure = (_box_sum(np, g.astype(np.float64), radius) / size) >= min_pure_frac
    scorable = g & pure
    coverage = (float((var[scorable] > floor).mean())
                if scorable.any() else None)
    return {"coverage": coverage,
            "variance_floor": floor,
            "floor_source": (f"p{percentile:g} of VALID local variance, "
                             f"floored at {VARIANCE_EPS:g}"),
            "ghost_px": int(g.sum()),
            "scorable_px": int(scorable.sum()),
            "window_radius": int(radius),
            "reason": (None if scorable.any() else
                       "every ghost window straddles the boundary; the region "
                       "is thinner than the measurement window"),
            "ghost_vs_valid_texture_ratio": (
                float(np.mean(var[scorable]) / np.mean(var[v]))
                if scorable.any() and np.mean(var[v]) > 0 else None)}


def _slope(np: Any, values: list) -> dict:
    """Least-squares slope of a per-frame series against frame index.

    A negative slope on the headline is the signature of a generator that
    started on the camera and drifted off it.
    """
    y = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                   dtype=np.float64)
    x = np.asarray([i for i, v in enumerate(values)
                    if v is not None and np.isfinite(v)], dtype=np.float64)
    if y.size < 3:
        return {"slope": None, "r2": None, "first_vs_last": None,
                "reason": "fewer than 3 scored frames"}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(slope),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "first_vs_last": float(y[-1] - y[0])}


def _paired_stats(np: Any, a: list, b: list, *, seed: int,
                  iterations: int = 2000) -> dict:
    """Paired bootstrap CI and sign test on (a - b), frame by frame.

    Two means with no interval is not evidence, and frames along one move are
    not independent samples of anything -- so the bootstrap is PAIRED over
    frames and the sign test is reported beside it as the assumption-free
    fallback.
    """
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return {"delta": None, "ci95": None, "sign_test": None,
                "n": len(pairs), "reason": "fewer than 3 paired frames"}
    d = np.asarray([x - y for x, y in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, d.size, size=(iterations, d.size))
    means = d[draws].mean(axis=1)
    wins = int((d > 0).sum())
    return {"delta": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)),
                     float(np.percentile(means, 97.5))],
            "sign_test": {"frames_favouring_first": wins,
                          "n": int(d.size),
                          "fraction": float(wins) / float(d.size)},
            "bootstrap_seed": int(seed), "bootstrap_iterations": iterations}


class DegenerateArmError(RuntimeError):
    """The arm scored well without responding to the camera at all.

    Raised rather than reported, because a headline adherence number computed
    for a frozen or near-frozen clip is not a weak result -- it is a
    meaningless one, and publishing it with a caveat invites the caveat being
    dropped downstream.
    """


def score_arms(arms: dict, *, bundle, min_parallax_px: float = PARALLAX_FLOOR_PX,
               bootstrap_seed: int = 0, bootstrap_iterations: int = 2000,
               refuse_degenerate: bool = True) -> dict:
    """Score every arm against the bundle's deterministic reprojection.

    ``arms`` maps a label to an (F, H, W, 3) float array in [0, 1] and MUST
    contain both ``"atlas"`` and ``"prompt_only"``; a ``"static"`` arm is
    synthesised from the atlas arm's own first frame. Returns a dict carrying
    per-arm per-frame measures, the drift curve, the parallax response, the
    paired comparison against the control, and the headline expressed as a
    margin over the static arm.
    """
    np = _require_numpy()
    if bundle.rgb is None or bundle.class_map is None:
        raise ValueError(
            "the bundle carries no rendered frames to score against; call "
            "render_conditioning_sequence first")
    for required in (ATLAS_ARM, CONTROL_ARM):
        if required not in arms:
            raise ValueError(
                f"score_arms needs a {required!r} arm. The control arm is "
                "required, not optional: an adherence figure with nothing to "
                "compare it against cannot support a claim either way.")

    reference = np.asarray(bundle.rgb, dtype=np.float64)
    work = dict(arms)
    atlas_frames = np.asarray(work[ATLAS_ARM], dtype=np.float64)
    # The cheat's ceiling, synthesised so a caller cannot forget it.
    work[STATIC_ARM] = np.repeat(atlas_frames[:1], atlas_frames.shape[0], axis=0)

    parallax = bundle.meta.get("parallax_px") or bundle.parallax_px()
    scored_frames = [i for i, p in enumerate(parallax)
                     if p >= float(min_parallax_px)]

    results: dict = {}
    for label, frames in work.items():
        frames = np.asarray(frames, dtype=np.float64)
        if frames.shape[0] != reference.shape[0]:
            raise ValueError(
                f"arm {label!r} has {frames.shape[0]} frames against the "
                f"bundle's {reference.shape[0]}; a scorer cannot guess which "
                "generated frame corresponds to which camera")
        gen, ref, valid, ghost, raster = align_rasters(
            np, frames, bundle.class_map, reference)
        ref0 = ref[0]

        per_frame = []
        for i in range(gen.shape[0]):
            head = gradient_zncc(np, gen[i], ref[i], valid[i])
            against_first = gradient_zncc(np, gen[i], ref0, valid[i])
            per_frame.append({
                "frame": i,
                "parallax_px": float(parallax[i]) if i < len(parallax) else None,
                "scored": i in scored_frames,
                "gradient_zncc": head,
                "gradient_zncc_vs_frame0": against_first,
                "parallax_response": (head - against_first
                                      if np.isfinite(head)
                                      and np.isfinite(against_first)
                                      else None),
                "masked_ssim": masked_ssim(np, gen[i], ref[i], valid[i]),
                "grade": fit_grade(np, gen[i], ref[i], valid[i]),
                "raw": raw_error(np, gen[i], ref[i], valid[i]),
                "ghost": ghost_fill_coverage(np, gen[i], valid[i], ghost[i]),
                "valid_px": int(valid[i].sum()),
                "ghost_px": int(ghost[i].sum()),
            })

        def _agg(key, only_scored=True):
            vals = [f[key] for f in per_frame
                    if (f["scored"] or not only_scored)
                    and f[key] is not None and np.isfinite(f[key])]
            return float(np.mean(vals)) if vals else None

        responses = [f["parallax_response"] for f in per_frame
                     if f["scored"] and f["parallax_response"] is not None]
        results[label] = {
            "per_frame": per_frame,
            "raster": raster,
            "aggregate": {
                "gradient_zncc": _agg("gradient_zncc"),
                "masked_ssim": _agg("masked_ssim"),
                "parallax_response": (float(np.median(responses))
                                      if responses else None),
            },
            "drift": _slope(np, [f["gradient_zncc"] if f["scored"] else None
                                 for f in per_frame]),
        }

    atlas = results[ATLAS_ARM]
    control = results[CONTROL_ARM]
    static = results[STATIC_ARM]
    response = atlas["aggregate"]["parallax_response"]

    guard = {
        "parallax_response": response,
        "frames_scored": scored_frames,
        "frames_without_parallax": [i for i, p in enumerate(parallax)
                                    if p < float(min_parallax_px)],
        "min_parallax_px": float(min_parallax_px),
        "static_arm_adherence": static["aggregate"]["gradient_zncc"],
        "passed": bool(response is not None and response > 0.0
                       and scored_frames),
    }
    if not guard["passed"] and refuse_degenerate:
        raise DegenerateArmError(
            "refusing to publish an adherence headline: parallax_response is "
            f"{response!r} over {len(scored_frames)} scored frame(s). A clip "
            "that repeats its first frame scores near-perfect adherence in "
            "VALID, so a non-positive response means the number measures "
            "nothing about the camera. Check that the move actually has "
            "parallax (a zoom has none) and that the generator moved at all.")

    head = atlas["aggregate"]["gradient_zncc"]
    static_head = static["aggregate"]["gradient_zncc"]
    return {
        "headline_measure": "gradient_zncc over VALID pixels",
        "adherence": head,
        "margin_over_control": (
            head - control["aggregate"]["gradient_zncc"]
            if head is not None
            and control["aggregate"]["gradient_zncc"] is not None else None),
        "margin_over_static": (head - static_head
                               if head is not None
                               and static_head is not None else None),
        "guard": guard,
        "comparison_vs_control": _paired_stats(
            np,
            [f["gradient_zncc"] for f in atlas["per_frame"] if f["scored"]],
            [f["gradient_zncc"] for f in control["per_frame"] if f["scored"]],
            seed=bootstrap_seed, iterations=bootstrap_iterations),
        "arms": results,
        "notes": [
            "Adherence is scored ONLY in VALID pixels, where Atlas knows the "
            "true appearance by reprojection. It needs no registration, which "
            "is the point: generated frames do not register.",
            "GHOST coverage is a separate question with a separate answer and "
            "is never folded into adherence.",
            "masked_ssim and raw MAE/PSNR are diagnostics. The headline is "
            "gradient_zncc, chosen because it survives the generator's grade.",
        ],
    }
