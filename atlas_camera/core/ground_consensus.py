"""Gravity-locked ground height as a one-dimensional robust consensus.

EXPERIMENTAL -- a measurement instrument, not a shipping path. Nothing in the
package imports this module; it exists so the gravity-locked hypothesis can be
measured against ``solver.estimate_ground_height_from_depth`` on the same
inputs. See ``docs/development/gravity-locked-ground-experiment.md``.

What this is NOT: a new way to orient the ground. Atlas already locks the
ground normal to world +Y in every single-image path (``proxy_geometry``'s
``_build_ground_primitive``, ``plane_extraction``, ``room_layout`` all emit
``n_g = [0, 1, 0]``); ``ground_normal_min`` only *selects* candidate pixels and
the selected normals are never fitted into a plane. The one true ground RANSAC
in the repo is ``multiview_solver._fit_ground_plane``, on the multi-view path.
So the orientation half of the hypothesis is already shipped.

That claim is asserted everywhere else in the codebase and measured nowhere, so
this module measures it: ``probe_ground_normal`` fits a normal from the depth
map itself, two independent ways, and reports its angle from +Y.

What must never happen is ROTATING THE WORLD onto that normal. World +Y *is*
the solve's gravity (``solver._rotation_from_up_vector``), so a fitted normal
that disagrees is a second, independent gravity estimate; turning the world
onto it would lean every facade the camera solve stands on.

Tilting the GROUND PRIMITIVE is a different matter, and is already expressible:
``AtlasProxyPrimitive`` carries its own ``transform_matrix``, and
``depth_geometry.plane_transform`` / ``arbitrary_plane_axes`` build a plane
basis for an arbitrary normal -- the shipping ground emitters simply hand them
a hard-coded ``[0, 1, 0]``. A ground plane that sits at a measured tilt while
the world stays gravity-aligned needs no new math, and it exports to a DCC as
one rotated object rather than a rotated scene.

Either way the angle is decisive evidence about the failure: small, and
orientation is provably innocent so the offset is the culprit; large, and the
real failure is wrong gravity or a non-planar road, which no scalar estimator
can fix.

The other half is where the shipping code is weak. With the normal fixed the
plane has exactly one free parameter, and each candidate pixel votes for it:

    h_i = dot(up, C - P_i)

The shipping estimator reduces those votes with an unweighted 48-bin histogram
mode over the 1-99 percentile range, then an unweighted median refine. Three
measurable weaknesses this module exists to test:

  1. no weighting at all, so distant road pixels win on sheer count;
  2. ``plane_tolerance = max(0.15, 0.03 * span)`` is set by the FAR-field
     spread, so a deep street widens the acceptance band to metres;
  3. no exclusion mask reaches it, so cars and clutter vote.

Every knob is a keyword argument on purpose -- the point is to measure which of
them matters, not to assert an answer. All six estimators are always computed
side by side into ``GroundConsensus.estimators``; the ``estimator`` argument
only chooses which one is reported as ``camera_height``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .depth_geometry import back_project_normals, detect_sky_mask
from .solver import _median_filter_3x3

WEIGHTINGS = ("uniform", "inverse_depth", "inverse_depth_sq", "image_y", "stratified")
ESTIMATORS = ("median", "trimmed", "mad_median", "mode", "ransac1d", "huber")

# 1-D RANSAC is seeded and swept deterministically -- a measurement instrument
# that returns a different number on a re-run is not a measurement.
RANSAC1D_SEED = 20260818
RANSAC1D_ITERS = 512


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ground_consensus requires numpy. Install with: pip install -e .[dev]"
        ) from exc
    return np


@dataclass
class GroundConsensus:
    """One gravity-locked ground estimate, with the evidence behind it."""

    camera_height: float | None       # None = explicit failure. Never a guess.
    confidence: float                 # bottom-band coverage, defined exactly as
                                      # solver.estimate_ground_height_from_depth
                                      # defines it, so the two are comparable
    plane_y: float | None
    ground_mask: Any
    candidate_mask: Any
    accepted: int
    candidates: int
    tolerance: float
    distribution: dict = field(default_factory=dict)
    estimators: dict = field(default_factory=dict)
    band_support: dict = field(default_factory=dict)
    rejections: dict = field(default_factory=dict)
    confidences: dict = field(default_factory=dict)
    normal_probe: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------
# weighted 1-D statistics
# --------------------------------------------------------------------------

def _wquantile(np: Any, x: Any, w: Any, q: float) -> float:
    """Weighted quantile. ``x`` need not be sorted; ``w`` must be non-negative."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ws = w[order]
    total = float(ws.sum())
    if total <= 0.0:
        return float(np.median(xs)) if xs.size else float("nan")
    cum = np.cumsum(ws) - 0.5 * ws
    return float(np.interp(q * total, cum, xs))


def _wmedian(np: Any, x: Any, w: Any) -> float:
    return _wquantile(np, x, w, 0.5)


def _wmean(np: Any, x: Any, w: Any) -> float:
    total = float(w.sum())
    if total <= 0.0:
        return float(x.mean()) if x.size else float("nan")
    return float((x * w).sum() / total)


def _wmad(np: Any, x: Any, w: Any, centre: float) -> float:
    """Weighted median absolute deviation, unscaled."""
    return _wmedian(np, np.abs(x - centre), w)


def _est_median(np: Any, h: Any, w: Any, **_: Any) -> float:
    return _wmedian(np, h, w)


def _est_trimmed(np: Any, h: Any, w: Any, trim: float = 0.1, **_: Any) -> float:
    """Weighted mean of the central (1 - 2*trim) mass."""
    lo = _wquantile(np, h, w, trim)
    hi = _wquantile(np, h, w, 1.0 - trim)
    keep = (h >= lo) & (h <= hi)
    if not keep.any():
        return _wmedian(np, h, w)
    return _wmean(np, h[keep], w[keep])


def _est_mad_median(np: Any, h: Any, w: Any, k: float = 3.0, **_: Any) -> float:
    """MAD-filtered weighted median, two passes.

    The scale comes from the data instead of from ``0.03 * span``, which is the
    specific thing that makes the shipping tolerance far-field-driven.
    """
    m = _wmedian(np, h, w)
    for _ in range(2):
        mad = _wmad(np, h, w, m) * 1.4826
        if not np.isfinite(mad) or mad <= 1e-9:
            break
        keep = np.abs(h - m) < k * mad
        if int(keep.sum()) < 8:
            break
        m = _wmedian(np, h[keep], w[keep])
    return float(m)


def _est_mode(np: Any, h: Any, w: Any, bins: int = 48, **_: Any) -> float:
    """Weighted histogram mode, then a weighted median inside the peak.

    Deliberately mirrors the shipping reduction so a comparison against it
    isolates the WEIGHTING rather than the shape of the reduction.
    """
    lo, hi = np.percentile(h, [1, 99])
    span = float(hi - lo)
    if not np.isfinite(span) or span < 1e-3:
        return _wmedian(np, h, w)
    hist, edges = np.histogram(h, bins=bins, range=(lo, hi), weights=w)
    peak = int(np.argmax(hist))
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    width = float(edges[1] - edges[0])
    keep = np.abs(h - centre) < max(width, 1e-6)
    if int(keep.sum()) >= 8:
        return _wmedian(np, h[keep], w[keep])
    return float(centre)


def _est_ransac1d(np: Any, h: Any, w: Any, **_: Any) -> float:
    """1-D RANSAC on the height votes; band width from the data's own MAD.

    Deterministic: fixed seed, fixed iteration count, so re-running the
    experiment reproduces the number exactly.
    """
    m0 = _wmedian(np, h, w)
    mad = _wmad(np, h, w, m0) * 1.4826
    tol = float(mad) if np.isfinite(mad) and mad > 1e-6 else 0.05
    rng = np.random.RandomState(RANSAC1D_SEED)
    n = int(h.size)
    iters = min(RANSAC1D_ITERS, max(16, n))
    picks = rng.randint(0, n, size=iters)
    best_score = -1.0
    best = m0
    for idx in picks:
        c = float(h[idx])
        inl = np.abs(h - c) <= tol
        score = float(w[inl].sum())
        if score > best_score:
            best_score = score
            best = _wmedian(np, h[inl], w[inl])
    return float(best)


def _est_huber(np: Any, h: Any, w: Any, k: float = 1.345, **_: Any) -> float:
    """Huber location by iteratively reweighted least squares."""
    m = _wmedian(np, h, w)
    mad = _wmad(np, h, w, m) * 1.4826
    if np.isfinite(mad) and mad > 1e-9:
        sigma = float(mad)
    else:
        sigma = float(np.std(h)) or 1.0
    for _ in range(24):
        r = (h - m) / sigma
        psi = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-9))
        m_new = _wmean(np, h, w * psi)
        if abs(m_new - m) < 1e-9:
            m = m_new
            break
        m = m_new
    return float(m)


_ESTIMATOR_FNS = {
    "median": _est_median,
    "trimmed": _est_trimmed,
    "mad_median": _est_mad_median,
    "mode": _est_mode,
    "ransac1d": _est_ransac1d,
    "huber": _est_huber,
}


# --------------------------------------------------------------------------
# weighting
# --------------------------------------------------------------------------

def _build_weights(np: Any, mode: str, *, depth: Any, rows: Any,
                   horizon_y: float, height: int) -> Any:
    """Per-sample authority. The whole point is that far samples must not win
    on count alone -- see the module docstring."""
    if mode == "uniform":
        return np.ones_like(depth)
    if mode == "inverse_depth":
        return 1.0 / np.maximum(depth, 1e-6)
    if mode == "inverse_depth_sq":
        return 1.0 / np.maximum(depth, 1e-6) ** 2
    if mode == "image_y":
        denom = max(float(height) - float(horizon_y), 1e-6)
        return np.clip((rows - horizon_y) / denom, 0.0, 1.0)
    if mode == "stratified":
        # Equal TOTAL weight per near/mid/far depth tercile, so a stratum cannot
        # dominate by being more numerous. This is the brief's "near/mid/far
        # stratified median" expressed as a weight, which lets it compose with
        # every estimator instead of being its own branch.
        w = np.ones_like(depth)
        cuts = np.percentile(depth, [33.3333, 66.6667])
        strata = (depth > cuts[0]).astype(int) + (depth > cuts[1]).astype(int)
        for s in (0, 1, 2):
            sel = strata == s
            n = int(sel.sum())
            if n:
                w[sel] = 1.0 / float(n)
        return w
    raise ValueError(f"unknown weighting {mode!r}; expected one of {WEIGHTINGS}")


# --------------------------------------------------------------------------
# normal probe -- diagnostic only, NEVER adopted
# --------------------------------------------------------------------------

def probe_ground_normal(np: Any, *, normals: Any, pts_world: Any,
                        candidate: Any, depth: Any, near_frac: float = 0.5) -> dict:
    """Fit a ground normal from the depth map and report its angle from +Y.

    Two independent fits, because they fail differently:

    * ``median_normal`` -- component-wise median of the candidate pixels' own
      surface normals, sign-aligned to +Y. Local, and insensitive to a road
      that is planar in patches but bends over its length.
    * ``svd_normal`` -- smallest singular vector of the NEAR-field candidate
      points about their centroid. Global over the region that matters, and it
      is the fit a plane-RANSAC would converge to.

    A large disagreement between the two is itself the signal that the road is
    not planar, which is a different failure from a wrong gravity vector.

    This function never applies anything. If the measurement says the tilt is
    real, the place to spend it is the ground PRIMITIVE's own
    ``transform_matrix`` (via ``plane_transform`` with this normal instead of
    the hard-coded ``[0, 1, 0]``), so the plane arrives in the DCC already
    rotated. The world stays on the solve's gravity either way -- see the
    module docstring.
    """
    up = np.array([0.0, 1.0, 0.0])
    out: dict = {
        "applied": False,
        "reference": "world +Y (solve gravity)",
        "safe_to_apply_to": "ground primitive transform only, never the world",
    }

    n_cand = int(candidate.sum())
    if n_cand < 16:
        out["error"] = f"too few candidates ({n_cand})"
        return out

    nrm = normals[candidate]
    nrm = np.where(nrm[:, 1:2] < 0.0, -nrm, nrm)      # sign-align to +Y
    med = np.median(nrm, axis=0)
    norm = float(np.linalg.norm(med))
    if norm > 1e-9:
        med = med / norm
        out["median_normal"] = [float(v) for v in med]
        out["median_angle_deg"] = float(
            np.degrees(np.arccos(np.clip(float(np.dot(med, up)), -1.0, 1.0))))

    d = depth[candidate]
    p = pts_world[candidate]
    cut = float(np.percentile(d, 100.0 * near_frac))
    near = d <= cut
    if int(near.sum()) >= 16:
        pts = p[near]
        centred = pts - pts.mean(axis=0)
        try:
            _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            out["svd_error"] = str(exc)
            return out
        n_svd = vh[-1]
        if n_svd[1] < 0.0:
            n_svd = -n_svd
        out["svd_normal"] = [float(v) for v in n_svd]
        out["svd_angle_deg"] = float(
            np.degrees(np.arccos(np.clip(float(np.dot(n_svd, up)), -1.0, 1.0))))
        out["svd_near_points"] = int(near.sum())
        out["svd_near_depth_cut"] = cut
        # Singular-value ratio: how planar the near field actually is. A road
        # that bends shows up here before it shows up in the angle.
        out["svd_planarity"] = float(_s[-1] / max(float(_s[0]), 1e-12))
        if "median_normal" in out:
            dot = float(np.clip(np.dot(np.asarray(out["median_normal"]), n_svd),
                                -1.0, 1.0))
            out["fits_disagree_deg"] = float(np.degrees(np.arccos(dot)))
    return out


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def estimate_ground_height_consensus(
    depth: Any,
    *,
    rotation: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    horizon_y: float | None = None,
    exclude_mask: Any = None,
    apply_sky_heuristic: bool = True,
    roi: tuple[float, float, float, float] | None = None,
    roi_top_width_frac: float | None = None,
    weighting: str = "uniform",
    estimator: str = "median",
    depth_edge_rel: float = 0.05,
    ground_normal_min: float = 0.90,
    height_prior: tuple[float, float] | None = None,
    min_candidates: int = 200,
    min_accepted: int = 50,
    max_pixels: int = 2_000_000,
) -> GroundConsensus:
    """Estimate camera height above a gravity-locked ground plane.

    ``rotation`` is the world->cam 3x3, matching
    ``solver.estimate_ground_height_from_depth``; the camera sits at the world
    origin, so ``h_i = -world_y_i``. ``depth`` is a HxW forward-distance map.

    ``exclude_mask`` is True-where-excluded. Unlike the shipping estimator,
    which accepts no mask at all, an explicit mask here REPLACES the sky
    heuristic rather than OR-ing with it -- the same semantics as
    ``node_helpers._metric_depth_and_validity``, so the two agree about what
    "excluded" means. Need both? OR them before calling.

    ``roi`` is ``(y0, y1, x0, x1)`` as fractions of the image, intersected with
    the below-horizon test. ``roi_top_width_frac`` narrows the ROI's top edge
    into a trapezoid, since under perspective the road is narrower further away.

    ``height_prior`` is ``(lo, hi)`` metres. It is recorded in ``notes`` and
    never clamps the answer: Atlas explicitly supports elevated and drone
    cameras, so penalising an "unusual" height would misfire on legitimate
    shots -- the same reasoning behind the plausibility penalty ``solver.py``
    considered and rejected.
    """
    np = _require_numpy()

    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}; expected one of {WEIGHTINGS}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")

    depth = np.asarray(depth)
    height, width = depth.shape
    notes: list = []

    # Same >2MP stride guard as the shipping estimator, and for the same reason:
    # plane fitting needs thousands of votes, not megapixels. Kept here so the
    # two are comparable on a 36MP plate, where the shipping path fits on a
    # ~1476x986 map and a full-resolution candidate would measure a different
    # thing. Rays stay exact: (u_s - cx/s)/(fx/s) == (u_s*s - cx)/fx.
    if height * width > max_pixels:
        s = int(np.ceil(np.sqrt(height * width / float(max_pixels))))
        sub_exclude = None
        if exclude_mask is not None:
            sub_exclude = np.asarray(exclude_mask, dtype=bool)[::s, ::s]
        res = estimate_ground_height_consensus(
            depth[::s, ::s], rotation=rotation, fx=fx / s, fy=fy / s,
            cx=cx / s, cy=cy / s,
            horizon_y=None if horizon_y is None else horizon_y / s,
            exclude_mask=sub_exclude, apply_sky_heuristic=apply_sky_heuristic,
            roi=roi, roi_top_width_frac=roi_top_width_frac,
            weighting=weighting, estimator=estimator,
            depth_edge_rel=depth_edge_rel, ground_normal_min=ground_normal_min,
            height_prior=height_prior, min_candidates=min_candidates,
            min_accepted=min_accepted, max_pixels=max_pixels,
        )
        for name in ("ground_mask", "candidate_mask"):
            m = np.asarray(getattr(res, name))
            up = np.repeat(np.repeat(m, s, axis=0), s, axis=1)[:height, :width]
            setattr(res, name, up)
        res.notes.append(f"strided by {s} (input over {max_pixels} px)")
        return res

    depth = np.asarray(depth, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)

    valid_depth = np.isfinite(depth) & (depth > 1e-4)
    depth = _median_filter_3x3(depth, valid_depth)

    # Camera at the world origin: a view matrix that is pure rotation. This is
    # exactly solver.estimate_ground_height_from_depth's frame, which is what
    # makes the two directly comparable -- and it lets us reuse the audited
    # back-projection + normal + discontinuity code instead of a third copy.
    view_matrix = np.eye(4, dtype=np.float64)
    view_matrix[:3, :3] = rotation
    bp = back_project_normals(depth, view_matrix=view_matrix, fx=fx, fy=fy,
                              cx=cx, cy=cy, depth_edge_rel=depth_edge_rel)

    world_y = bp.pts_world[..., 1]
    cam_y = float(bp.cam_pos[1])
    heights = cam_y - world_y          # h_i = dot(up, C - P_i), up = +Y

    if horizon_y is None:
        horizon_y = height * 0.45
    below = bp.vv > horizon_y
    horizontal = np.abs(bp.normals[..., 1]) > ground_normal_min

    # Exclusion. An explicit mask REPLACES the heuristic (the drift rule); with
    # no mask we fall back to the sky heuristic the shipping estimator never got.
    excluded = np.zeros((height, width), dtype=bool)
    exclude_source = "none"
    if exclude_mask is not None:
        excluded = np.asarray(exclude_mask, dtype=bool)
        exclude_source = "explicit"
    elif apply_sky_heuristic:
        excluded = detect_sky_mask(depth, horizon_y=horizon_y)
        exclude_source = "detect_sky_mask"

    region = np.ones((height, width), dtype=bool)
    if roi is not None:
        y0f, y1f, x0f, x1f = roi
        rows = bp.vv
        cols = np.broadcast_to(np.arange(width, dtype=np.float64), (height, width))
        y0, y1 = y0f * height, y1f * height
        x0, x1 = x0f * width, x1f * width
        if roi_top_width_frac is None:
            region = (rows >= y0) & (rows <= y1) & (cols >= x0) & (cols <= x1)
        else:
            # Trapezoid: full ROI width at the bottom edge, narrowed at the top.
            t = np.clip((rows - y0) / max(y1 - y0, 1e-6), 0.0, 1.0)
            half_bot = 0.5 * (x1 - x0)
            half_top = half_bot * float(roi_top_width_frac)
            half = half_top + (half_bot - half_top) * t
            centre = 0.5 * (x0 + x1)
            region = (rows >= y0) & (rows <= y1) & (np.abs(cols - centre) <= half)

    candidate = (bp.valid_normal & below & horizontal & region & ~excluded
                 & np.isfinite(heights) & (depth > 0))

    rejections = {
        "total_px": int(height * width),
        "invalid_depth": int((~bp.valid_depth).sum()),
        "above_horizon": int((~below).sum()),
        "not_horizontal": int((~horizontal).sum()),
        "near_discontinuity": int((bp.valid_depth & ~bp.valid_normal).sum()),
        "excluded": int(excluded.sum()),
        "exclude_source": exclude_source,
        "outside_roi": int((~region).sum()),
        "candidates": int(candidate.sum()),
    }

    empty_mask = np.zeros((height, width), dtype=bool)
    n_cand = int(candidate.sum())

    # The normal probe runs on the candidate set whatever happens downstream --
    # if the height estimate fails, knowing whether the normal was sane is
    # exactly what tells us why.
    normal_probe = probe_ground_normal(
        np, normals=bp.normals, pts_world=bp.pts_world,
        candidate=candidate, depth=depth)

    if n_cand < min_candidates:
        notes.append(f"insufficient ground candidates: {n_cand} < {min_candidates}")
        return GroundConsensus(
            camera_height=None, confidence=0.0, plane_y=None,
            ground_mask=empty_mask, candidate_mask=candidate,
            accepted=0, candidates=n_cand, tolerance=float("nan"),
            rejections=rejections, normal_probe=normal_probe, notes=notes,
        )

    h = heights[candidate]
    d = depth[candidate]
    rows_sel = bp.vv[candidate]
    w = _build_weights(np, weighting, depth=d, rows=rows_sel,
                       horizon_y=float(horizon_y), height=height)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(w.sum()) <= 0.0:
        notes.append("all sample weights collapsed to zero")
        return GroundConsensus(
            camera_height=None, confidence=0.0, plane_y=None,
            ground_mask=empty_mask, candidate_mask=candidate,
            accepted=0, candidates=n_cand, tolerance=float("nan"),
            rejections=rejections, normal_probe=normal_probe, notes=notes,
        )

    # Every estimator, every time. A comparison cell in the report is one call
    # per (weighting, ROI, mask), not one call per estimator as well.
    estimates = {}
    for name, fn in _ESTIMATOR_FNS.items():
        try:
            estimates[name] = float(fn(np, h, w))
        except Exception as exc:  # pragma: no cover - diagnostic path
            estimates[name] = float("nan")
            notes.append(f"estimator {name} failed: {exc}")

    h_est = estimates[estimator]

    # Tolerance from the data's own dispersion, NOT from the far-field span.
    # That is defect 2 in the module docstring, isolated so it can be measured.
    centre_mad = _wmad(np, h, w, h_est) * 1.4826
    tolerance = float(max(0.05, min(0.5, 3.0 * centre_mad)))
    if not np.isfinite(tolerance):
        tolerance = 0.15

    accepted_sel = np.abs(h - h_est) < tolerance
    n_acc = int(accepted_sel.sum())
    ground_mask = np.zeros((height, width), dtype=bool)
    ground_mask[candidate] = accepted_sel

    q = np.percentile(h, [1, 5, 25, 50, 75, 95, 99])
    med = _wmedian(np, h, w)
    distribution = {
        "count": int(h.size),
        "mean": float(h.mean()),
        "weighted_mean": _wmean(np, h, w),
        "median": float(np.median(h)),
        "weighted_median": med,
        "trimmed_mean": _est_trimmed(np, h, w),
        "mad": float(np.median(np.abs(h - np.median(h)))),
        "weighted_mad": _wmad(np, h, w, med),
        "iqr": float(q[4] - q[2]),
        "span_1_99": float(q[6] - q[0]),
        "percentiles": {"p1": float(q[0]), "p5": float(q[1]), "p25": float(q[2]),
                        "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
                        "p99": float(q[6])},
        "depth_min": float(d.min()), "depth_max": float(d.max()),
        "depth_median": float(np.median(d)),
    }

    # Where does the accepted support actually live? The hypothesis says clean
    # near-field ground forms a tight dominant cluster; this is what confirms or
    # falsifies that on a real plate.
    cuts = np.percentile(d, [33.3333, 66.6667])
    bands: dict = {}
    for label, sel in (("near", d <= cuts[0]),
                       ("mid", (d > cuts[0]) & (d <= cuts[1])),
                       ("far", d > cuts[1])):
        n_band = int(sel.sum())
        acc_band = int((sel & accepted_sel).sum())
        bands[label] = {
            "candidates": n_band,
            "accepted": acc_band,
            "accept_rate": float(acc_band) / float(max(n_band, 1)),
            "share_of_accepted": float(acc_band) / float(max(n_acc, 1)),
            "median_h": float(np.median(h[sel])) if n_band else float("nan"),
            "weight_share": float(w[sel].sum()) / float(max(float(w.sum()), 1e-12)),
        }
    bands["depth_cuts"] = [float(cuts[0]), float(cuts[1])]

    # Bottom-band coverage, defined exactly as solver.py defines it, so the two
    # confidences are comparable numbers rather than two different scales.
    band_top = int(height * 0.80)
    band = np.zeros((height, width), dtype=bool)
    band[band_top:, :] = True
    band_valid = int((band & (depth > 0)).sum())
    conf_bottom = float((ground_mask & band).sum()) / float(max(band_valid, 1))
    confidences = {
        "bottom_band": conf_bottom,
        "accept_rate": float(n_acc) / float(max(n_cand, 1)),
        "dispersion_ratio": float(abs(h_est) / max(tolerance, 1e-6)),
    }

    if height_prior is not None:
        lo, hi = height_prior
        if not (lo <= h_est <= hi):
            notes.append(
                f"height {h_est:.3f} m outside prior [{lo}, {hi}] "
                "-- reported, NOT clamped (elevated/drone shots are legitimate)")

    if n_acc < min_accepted:
        notes.append(f"only {n_acc} accepted samples < {min_accepted}")
        return GroundConsensus(
            camera_height=None, confidence=conf_bottom, plane_y=None,
            ground_mask=ground_mask, candidate_mask=candidate,
            accepted=n_acc, candidates=n_cand, tolerance=tolerance,
            distribution=distribution, estimators=estimates,
            band_support=bands, rejections=rejections,
            confidences=confidences, normal_probe=normal_probe, notes=notes,
        )

    if not np.isfinite(h_est) or h_est <= 0.0:
        notes.append(f"camera at or below the fitted ground (h={h_est})")
        return GroundConsensus(
            camera_height=None, confidence=conf_bottom, plane_y=None,
            ground_mask=ground_mask, candidate_mask=candidate,
            accepted=n_acc, candidates=n_cand, tolerance=tolerance,
            distribution=distribution, estimators=estimates,
            band_support=bands, rejections=rejections,
            confidences=confidences, normal_probe=normal_probe, notes=notes,
        )

    return GroundConsensus(
        camera_height=float(h_est),
        confidence=conf_bottom,
        plane_y=float(cam_y - h_est),
        ground_mask=ground_mask,
        candidate_mask=candidate,
        accepted=n_acc,
        candidates=n_cand,
        tolerance=tolerance,
        distribution=distribution,
        estimators=estimates,
        band_support=bands,
        rejections=rejections,
        confidences=confidences,
        normal_probe=normal_probe,
        notes=notes,
    )
