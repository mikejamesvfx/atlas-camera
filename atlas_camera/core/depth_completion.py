"""Hidden-surface depth, computed rather than predicted.

A silhouette tear is a region where the camera ray hit an occluder and the
surface behind it was never photographed. Layered-ray models (LaRI and
relatives) answer that by predicting several ray intersections per pixel; Atlas
consumes such a stack in :mod:`atlas_camera.core.hidden_geometry`. This module
answers the same question **deterministically**, for the cases where the answer
is already implied by geometry Atlas has fitted.

The insight is that the solve's own back-projection

    x = (u - cx) / fx * depth        y = -(v - cy) / fy * depth

is a ray parametrisation: the pixel fixes a ray, depth picks a point on it.
Hidden geometry is picking a *different* point on the same ray. Where the
occlusion graph says a fitted plane lies behind the occluder, that point is the
exact ray-plane intersection — not an estimate, not an interpolation:

    t = (D - n·cam) / (n·dir)        P = cam + t·dir

That covers background continuation, which is the case that actually produces
visible tearing under a modest camera move. Two weaker tiers handle the rest:
first-order tangential extension using surface normals, then isotropic
diffusion. Every pixel records WHICH tier produced it, because a ray-plane
result and a diffusion guess deserve very different trust.

**Position in the pipeline.** This runs BEFORE the relief mesh is built. Repair
the depth map and a mesh with no holes follows by construction; repair the mesh
and you are reconstructing information that was discarded a step earlier.
Tearing then becomes purely the deliberate ``depth_edge_rel`` decision it is
supposed to be, rather than a consequence of missing data.

**Nothing here overwrites measured depth.** Completion only ever writes where
the input was invalid or explicitly marked as a tear, and the returned
``synthesized_mask`` / ``method_map`` make every invented pixel identifiable
downstream — the same discipline as ``ReliefMesh.filled_mask``.

Output deliberately matches :func:`hidden_geometry.select_hidden_surface`'s
shape — ``(hidden_depth, hidden_valid, stats)`` in the pipeline's depth units —
so this module and a layered model are interchangeable producers for the same
consumers, and a licensable layered model could be added later without
reworking anything downstream.

Numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-pixel provenance. Ordered best-to-worst; the numbering is stored in
# `method_map`, so these are effectively APPEND-ONLY once serialized.
METHOD_NONE = 0
METHOD_MEASURED = 1
METHOD_RAY_PLANE = 2
METHOD_TANGENT = 3
METHOD_DIFFUSION = 4
#: Appended, NOT inserted between TANGENT and DIFFUSION, even though that is
#: where its TRUST sits — the numbering is serialized into `method_map`, so the
#: codes are append-only while the trust table is free to order them properly.
METHOD_GUIDED = 5

METHOD_NAMES = {
    METHOD_NONE: "none",
    METHOD_MEASURED: "measured",
    METHOD_RAY_PLANE: "ray_plane",
    METHOD_TANGENT: "tangent",
    METHOD_DIFFUSION: "diffusion",
    METHOD_GUIDED: "guided",
}

# How much each tier is trusted, used to weight the reported confidence.
# Ray-plane is exact given the plane fit, so it inherits the fit's own
# confidence rather than being discounted further.
_METHOD_TRUST = {
    METHOD_MEASURED: 1.0,
    METHOD_RAY_PLANE: 0.9,
    METHOD_TANGENT: 0.5,
    # Between tangent and diffusion. A generated prior supplies the SHAPE while
    # measured rim values supply the placement, so it beats a smoothness guess
    # and loses to a first-order continuation of a surface actually observed.
    METHOD_GUIDED: 0.4,
    METHOD_DIFFUSION: 0.2,
}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Depth completion requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class DepthCompletion:
    """A repaired depth map plus a full account of what was invented."""

    depth: Any                       # (H, W) float64, pipeline units
    synthesized_mask: Any            # (H, W) bool — True where invented
    method_map: Any                  # (H, W) uint8 — one of METHOD_*
    stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: (H, W) float64 — per-pixel disagreement between the eight integration
    #: paths, non-zero only where METHOD_GUIDED wrote. A generated gradient field
    #: is not curl-free, so path disagreement measures directly how much the
    #: prior is confabulating rather than describing. Kept per-pixel, not
    #: summarised: the useful question is WHICH part of a fill to distrust.
    guided_spread: Any = None

    @property
    def synthesized_fraction(self) -> float:
        return float(self.synthesized_mask.mean()) if self.synthesized_mask.size else 0.0

    def method_histogram(self) -> dict[str, int]:
        np = _require_numpy()
        counts = np.bincount(self.method_map.ravel(), minlength=len(METHOD_NAMES))
        return {name: int(counts[code]) for code, name in METHOD_NAMES.items()
                if code < len(counts) and counts[code]}

    def confidence(self) -> float:
        """Trust-weighted mean over the frame.

        A frame that is 90% measured and 10% ray-plane reads near 1.0; one
        rescued mostly by diffusion reads low, which is the honest answer and
        the number ``scene_health`` should degrade a verdict against.
        """
        np = _require_numpy()
        weights = np.zeros(self.method_map.shape, dtype=np.float64)
        for code, trust in _METHOD_TRUST.items():
            weights[self.method_map == code] = trust
        return float(weights.mean()) if weights.size else 0.0

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in self.method_histogram().items()]
        lines = [
            f"Depth completion: {self.synthesized_fraction:.1%} of frame synthesized, "
            f"confidence {self.confidence():.2f}",
            f"  pixels by method: {', '.join(parts) if parts else 'none'}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def as_hidden_surface(self) -> tuple[Any, Any, dict[str, Any]]:
        """``(hidden_depth, hidden_valid, stats)`` — the layered-model contract.

        Lets a caller treat this module and
        :func:`hidden_geometry.select_hidden_surface` as interchangeable
        producers of hidden-surface depth.
        """
        np = _require_numpy()
        hidden = np.where(self.synthesized_mask, self.depth, 0.0)
        stats = dict(self.stats)
        stats.update(coverage=self.synthesized_fraction,
                     n_hidden_pixels=int(self.synthesized_mask.sum()),
                     confidence=self.confidence(),
                     source="depth_completion")
        return hidden, self.synthesized_mask.copy(), stats


def pixel_rays(np: Any, height: int, width: int, *, view_matrix: Any,
               fx: float, fy: float, cx: float, cy: float) -> tuple[Any, Any]:
    """World-space ``(camera_position, directions)`` for every pixel.

    Directions are the exact rays the solve's back-projection uses, so a point
    at distance ``t`` along ray ``(v, u)`` reprojects to pixel ``(v, u)`` by
    construction. Uses the full 4x4 view matrix and its inverse — never the 3x3
    rotation, whose transpose ambiguity is the standing trap in this codebase.
    """
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    R_cw = cam_to_world[:3, :3]
    cam_pos = cam_to_world[:3, 3]

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    dirs_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy,
                         -np.ones_like(uu)], axis=-1)
    dirs_world = dirs_cam @ R_cw.T
    return cam_pos, dirs_world


def ray_plane_depth(np: Any, cam_pos: Any, dirs: Any, normal: Any,
                    d: float, *, min_depth: float = 1e-3) -> tuple[Any, Any]:
    """Forward depth where each ray meets the plane ``n·x = d``.

    Returns ``(depth, valid)``. A ray is invalid when it is parallel to the
    plane, or meets it behind the camera — both mean this plane genuinely does
    not explain that pixel, and inventing a value would be worse than leaving
    the hole for a later tier.
    """
    normal = np.asarray(normal, dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        shape = dirs.shape[:2]
        return np.zeros(shape), np.zeros(shape, dtype=bool)
    normal = normal / nrm
    d = float(d) / nrm

    denom = dirs @ normal
    parallel = np.abs(denom) < 1e-9
    numer = d - float(np.dot(normal, cam_pos))
    t = np.divide(numer, np.where(parallel, 1.0, denom))

    # `dirs` has -1 in camera z, so |t| scaled by the ray's forward component
    # gives forward distance; t > 0 means in front of the camera.
    forward = t
    valid = (~parallel) & np.isfinite(forward) & (forward > min_depth)
    return np.where(valid, forward, 0.0), valid


def complete_depth(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    holes: Any = None,
    planes: list[dict[str, Any]] | None = None,
    normals: Any = None,
    prior: Any = None,
    prior_band_px: int = 3,
    prior_max_residual_rel: float = 0.75,
    use_diffusion: bool = True,
    diffusion_iterations: int = 64,
    max_plane_depth_ratio: float = 50.0,
    grazing_min_cos: float = 0.087,   # ~85 degrees incidence
) -> DepthCompletion:
    """Fill every hole in ``depth``, best tier first, recording which was used.

    ``holes`` (H, W) bool, optional — pixels to complete. Defaults to wherever
    ``depth`` is non-finite or non-positive.

    ``planes`` is a list of ``{"normal": [x,y,z], "d": float}`` in world space,
    ordered NEAREST-FIRST in the sense that earlier entries win. The occlusion
    graph produces exactly this from its occludee nodes, so the caller normally
    passes ``[n.plane for n in graph.nodes if n.plane]``.

    ``normals`` (H, W, 3), optional — enables the tangential tier, a first-order
    continuation of the surface at the hole's rim. Normals carry no information
    about what is *behind* a surface (they are the visible field's derivative),
    but they do say which way it was heading, which beats extending it flat.

    Tiers are applied in order and never overwrite an earlier, better one.
    """
    np = _require_numpy()
    depth = np.array(depth, dtype=np.float64, copy=True)
    if depth.ndim != 2:
        raise ValueError("complete_depth expects a 2D (H, W) depth map")
    height, width = depth.shape

    measured = np.isfinite(depth) & (depth > 1e-6)
    if holes is None:
        holes = ~measured
    else:
        holes = np.asarray(holes, dtype=bool) | ~measured

    method_map = np.where(measured & ~holes, METHOD_MEASURED,
                          METHOD_NONE).astype(np.uint8)
    out = np.where(measured, depth, 0.0)
    remaining = holes.copy()
    notes: list[str] = []
    stats: dict[str, Any] = {"n_holes_in": int(holes.sum())}

    if not remaining.any():
        return DepthCompletion(depth=out, synthesized_mask=np.zeros_like(holes),
                               method_map=method_map, stats=stats,
                               notes=["nothing to complete — depth had no holes."])

    cam_pos, dirs = pixel_rays(np, height, width, view_matrix=view_matrix,
                               fx=fx, fy=fy, cx=cx, cy=cy)

    # --- tier 1: exact ray-plane -------------------------------------------
    #
    # Every candidate plane is evaluated and the NEAREST valid intersection
    # wins per pixel. That is the physical answer — the closest surface behind
    # the occluder is the one you would see — and it needs no plane ordering.
    #
    # Ordering the planes globally and taking first-wins does not work, by any
    # ordering: a ground plane's perpendicular distance from the camera is just
    # the camera height, so "nearest plane" ranks it ahead of a wall it is
    # nowhere near, and it then claims near-horizon pixels at absurd distances
    # where its ray intersection is technically valid and practically useless.
    n_plane = 0
    if planes:
        scale_ref = float(np.median(depth[measured])) if measured.any() else 1.0
        limit = max_plane_depth_ratio * max(scale_ref, 1e-6)
        best = np.full(depth.shape, np.inf, dtype=np.float64)
        for plane in planes:
            normal = plane.get("normal")
            if normal is None or plane.get("d") is None:
                continue
            candidate, ok = ray_plane_depth(np, cam_pos, dirs, normal,
                                            float(plane["d"]))
            # A near-grazing hit is geometrically valid and physically
            # meaningless — the ray skims the surface, so a millimetre of fit
            # error moves the intersection by metres.
            cos_incidence = np.abs(dirs @ (np.asarray(normal, dtype=np.float64)
                                           / max(float(np.linalg.norm(normal)), 1e-12)))
            ok &= (candidate < limit) & (cos_incidence > grazing_min_cos)
            best = np.where(ok & (candidate < best), candidate, best)
        take = remaining & np.isfinite(best)
        if take.any():
            out[take] = best[take]
            method_map[take] = METHOD_RAY_PLANE
            remaining &= ~take
            n_plane = int(take.sum())
    stats["n_ray_plane"] = n_plane

    # --- tier 2: first-order tangential extension ---------------------------
    n_tangent = 0
    if normals is not None and remaining.any():
        filled, took = _tangent_extend(np, out, remaining, measured, normals,
                                       cam_pos, dirs)
        if took.any():
            out[took] = filled[took]
            method_map[took] = METHOD_TANGENT
            remaining &= ~took
            n_tangent = int(took.sum())
    stats["n_tangent"] = n_tangent

    guided_spread = None

    # --- tier 3: prior-guided gradient integration --------------------------
    #
    # Takes SHAPE from a generated relative-depth prior and PLACEMENT from the
    # measured rim. The prior's own absolute values are never used: differencing
    # removes its shift, and one least-squares scale over the ring band removes
    # the rest. That is the whole reason a hallucinated depth map is usable here
    # at all — its structure is a far stronger prior than smoothness, while its
    # placement is worthless.
    n_guided = 0
    if prior is not None and remaining.any():
        known = method_map > METHOD_NONE
        band = known & _dilate(np, remaining, prior_band_px) & ~remaining
        # NaN, not `out`. `out` carries 0.0 inside holes, and np.gradient's
        # central difference then reads a cliff into those zeros at exactly the
        # rim pixels being fitted — measured live, it returned s=-5.00 with
        # residual 19.5 (true s=0.27, residual 0.0) and the tier declined itself
        # out of every valid case. NaN propagates instead, and the fit's own
        # isfinite filter drops the contaminated pixels.
        grad_src = np.where(known, out, np.nan)
        scale, n_samples, resid_rel = prior_gradient_scale(
            np, prior, grad_src, band)
        stats["prior_scale"] = float(scale)
        stats["prior_scale_samples"] = int(n_samples)
        stats["prior_residual_rel"] = float(resid_rel)
        if n_samples < _MIN_SCALE_SAMPLES or not np.isfinite(resid_rel):
            notes.append(
                f"prior tier declined: only {n_samples} ring-band samples "
                f"(need {_MIN_SCALE_SAMPLES}) — fitting one scale to that few "
                "gradients invents a plausible wrong answer.")
        elif resid_rel > prior_max_residual_rel:
            notes.append(
                f"prior tier declined: gradient residual {resid_rel:.2f} exceeds "
                f"{prior_max_residual_rel:.2f} — the prior describes DIFFERENT "
                "structure from the measured rim, so integrating it would bulge "
                "the fill rather than shape it.")
        else:
            filled, wsum, spread = integrate_prior_gradients(
                np, prior, out, known, remaining, scale=scale)
            took = remaining & (wsum > 0) & np.isfinite(filled)
            if took.any():
                out[took] = filled[took]
                method_map[took] = METHOD_GUIDED
                remaining &= ~took
                n_guided = int(took.sum())
                guided_spread = np.where(took, spread, 0.0)
                stats["prior_spread_median"] = float(np.median(spread[took]))
                stats["prior_spread_max"] = float(spread[took].max())
    stats["n_guided"] = n_guided

    # --- tier 4: isotropic diffusion ---------------------------------------
    n_diffusion = 0
    if use_diffusion and remaining.any():
        filled, took = _diffuse(np, out, remaining, method_map > METHOD_NONE,
                                iterations=diffusion_iterations)
        if took.any():
            out[took] = filled[took]
            method_map[took] = METHOD_DIFFUSION
            remaining &= ~took
            n_diffusion = int(took.sum())
    stats["n_diffusion"] = n_diffusion

    if remaining.any():
        notes.append(
            f"{int(remaining.sum())} pixels could not be completed by any tier "
            "and remain holes — no plane explained them and they touch no "
            "measured depth to diffuse from."
        )

    synthesized = holes & ~remaining
    return DepthCompletion(depth=out, synthesized_mask=synthesized,
                           method_map=method_map, stats=stats, notes=notes,
                           guided_spread=guided_spread)


#: Compass steps (dy, dx). Eight, because these land EXACTLY on grid pixels —
#: 16 would need interpolation and smear the gradient field being integrated.
_RAYS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

#: Below this |grad| a prior pixel carries no usable direction, so it is left out
#: of the scale fit rather than contributing a near-0/0 ratio.
_PRIOR_GRAD_FLOOR = 1e-6

#: Minimum ring-band samples before the scale fit is trusted. Fitting one scalar
#: to a handful of noisy gradients is how a plausible-looking wrong answer gets
#: made, so below this the tier declines instead.
_MIN_SCALE_SAMPLES = 24


def _dilate(np: Any, mask: Any, radius: int) -> Any:
    """Grow a boolean mask by `radius`, without wrapping at the frame edge."""
    if radius <= 0:
        return mask
    h, w = mask.shape
    pad = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    pad[radius:radius + h, radius:radius + w] = mask
    out = np.zeros_like(pad)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= np.roll(np.roll(pad, dy, axis=0), dx, axis=1)
    return out[radius:radius + h, radius:radius + w]


def prior_gradient_scale(np: Any, prior: Any, depth: Any, band: Any) -> tuple:
    """Least-squares scale ``s`` taking the prior's gradients into metres.

    ``s = sum(grad_m . grad_q) / sum(grad_q . grad_q)`` over the ring band — the
    projection of the measured gradient field onto the prior's. A relative depth
    map is defined only up to scale AND shift; differentiating kills the shift
    outright, leaving exactly this one scalar, which is why the prior never needs
    to be metric.

    Returns ``(scale, n_samples, residual_rel)``. ``residual_rel`` is the median
    |s*grad_q - grad_m| over the median |grad_m| — small means the prior really
    is describing the same surface, large means it invented different structure
    and the caller should distrust the fill.
    """
    gy_q, gx_q = np.gradient(np.asarray(prior, dtype=np.float64))
    gy_m, gx_m = np.gradient(np.asarray(depth, dtype=np.float64))
    mag_q = np.hypot(gx_q, gy_q)
    use = band & np.isfinite(mag_q) & (mag_q > _PRIOR_GRAD_FLOOR) \
        & np.isfinite(gx_m) & np.isfinite(gy_m)
    n = int(use.sum())
    if n < _MIN_SCALE_SAMPLES:
        return 0.0, n, float("inf")
    num = float((gx_m[use] * gx_q[use] + gy_m[use] * gy_q[use]).sum())
    den = float((gx_q[use] ** 2 + gy_q[use] ** 2).sum())
    if den <= 0.0:
        return 0.0, n, float("inf")
    scale = num / den
    resid = np.hypot(scale * gx_q[use] - gx_m[use],
                     scale * gy_q[use] - gy_m[use])
    denom = float(np.median(np.hypot(gx_m[use], gy_m[use]))) or 1.0
    return scale, n, float(np.median(resid) / denom)


def integrate_prior_gradients(np: Any, prior: Any, depth: Any, known: Any,
                              remaining: Any, *, scale: float) -> tuple:
    """Fill ``remaining`` by integrating ``scale * grad(prior)`` from ``known``.

    For each of the eight compass directions a running sum marches the image:
    a pixel takes its predecessor's value plus the prior's directional
    derivative over that step (``grad . u``, times the step length). Paths START
    at measured pixels, so the rim is continuous by construction rather than by
    blending.

    Returns ``(filled, weight_sum, spread)``.

    SPREAD IS THE POINT, not a diagnostic afterthought. A neural depth field is
    not curl-free, so the eight paths reaching a pixel disagree, and by how much
    is a free per-pixel measure of whether the prior is describing structure or
    confabulating it. Averaging the paths is also the only honest response to a
    non-integrable field — there is no single "correct" integral to find.

    Degenerate case worth knowing: with ``grad(prior) == 0`` every ray returns
    its nearest measured value, so the result is a distance-weighted average of
    the eight nearest rim pixels. That is ordinary harmonic-ish inpainting, i.e.
    this tier collapses into the diffusion tier rather than into nonsense.
    """
    gy_q, gx_q = np.gradient(np.asarray(prior, dtype=np.float64))
    gy_q = np.nan_to_num(gy_q)
    gx_q = np.nan_to_num(gx_q)
    height, width = depth.shape

    acc = np.zeros((height, width), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)
    per_ray: list[Any] = []
    per_weight: list[Any] = []

    for dy, dx in _RAYS:
        step = float(np.hypot(dy, dx))
        val = np.full((height, width), np.nan, dtype=np.float64)
        length = np.full((height, width), np.inf, dtype=np.float64)
        val[known] = depth[known]
        length[known] = 0.0
        # March so a pixel is always visited AFTER its predecessor p-(dy,dx).
        # The predecessor is at y-dy, so for dy<0 it lies at a HIGHER row and the
        # scan must run descending. Getting this backwards does not error — it
        # silently fills only the pixels whose predecessor happened to be ready
        # (76 of 400, measured), which reads as a weak prior rather than a bug.
        rows = list(range(height) if dy >= 0 else range(height - 1, -1, -1))
        cols = list(range(width) if dx >= 0 else range(width - 1, -1, -1))
        for y in rows:
            py = y - dy
            if not (0 <= py < height):
                continue
            for x in cols:
                px = x - dx
                if not (0 <= px < width) or not remaining[y, x]:
                    continue
                prev = val[py, px]
                if not np.isfinite(prev):
                    continue
                # Directional derivative at the midpoint of the step.
                deriv = 0.5 * ((gx_q[y, x] + gx_q[py, px]) * dx
                               + (gy_q[y, x] + gy_q[py, px]) * dy)
                val[y, x] = prev + scale * deriv
                length[y, x] = length[py, px] + step
        w = np.where(np.isfinite(val) & np.isfinite(length),
                     1.0 / (1.0 + length), 0.0)
        contrib = np.where(np.isfinite(val), val, 0.0)
        acc += w * contrib
        wsum += w
        per_ray.append(val)
        per_weight.append(w)

    filled = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
    var = np.zeros_like(filled)
    for val, w in zip(per_ray, per_weight):
        d = np.where(np.isfinite(val), val - filled, 0.0)
        var += w * d * d
    spread = np.sqrt(np.where(wsum > 0, var / np.maximum(wsum, 1e-12), 0.0))
    return filled, wsum, spread


def _tangent_extend(np: Any, depth: Any, remaining: Any, measured: Any,
                    normals: Any, cam_pos: Any, dirs: Any) -> tuple[Any, Any]:
    """Extend the rim surface along its own tangent plane into the hole.

    Each hole pixel adjacent to measured depth adopts the local plane of its
    nearest measured neighbour — that neighbour's 3D point and normal define a
    plane, and this pixel's ray is intersected with it. That is a first-order
    continuation: flat fill assumes the surface stops, this assumes it keeps
    going the way it was going.

    One dilation ring per call keeps it local; the caller's diffusion tier
    handles anything deeper, where a first-order guess would not be credible
    anyway.
    """
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape[:2] != depth.shape or normals.shape[-1] != 3:
        return depth, np.zeros_like(remaining)

    pts = cam_pos + dirs * depth[..., None]
    took = np.zeros_like(remaining)
    filled = depth.copy()

    for shift_axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        src = np.roll(measured, shift, axis=shift_axis)
        src_pts = np.roll(pts, shift, axis=shift_axis)
        src_nrm = np.roll(normals, shift, axis=shift_axis)
        cand = remaining & src & ~took
        if not cand.any():
            continue
        n = src_nrm
        d = np.einsum("ijk,ijk->ij", n, src_pts)
        denom = np.einsum("ijk,ijk->ij", dirs, n)
        good = cand & (np.abs(denom) > 1e-9)
        t = np.divide(d - np.einsum("ijk,k->ij", n, cam_pos),
                      np.where(np.abs(denom) > 1e-9, denom, 1.0))
        good &= np.isfinite(t) & (t > 1e-3)
        if not good.any():
            continue
        filled = np.where(good, t, filled)
        took |= good
    return filled, took


def _diffuse(np: Any, depth: Any, remaining: Any, known: Any, *,
             iterations: int) -> tuple[Any, Any]:
    """Jacobi diffusion of known depth into the remaining holes.

    The last resort, and labelled as such: it produces a smooth interpolation
    of the surrounding surface, which is plausible-looking and carries no
    evidence whatsoever. Only pixels actually reached by the diffusion front
    are reported as filled, so an enclosed region with no known neighbour stays
    an honest hole.
    """
    work = np.where(known, depth, 0.0)
    reached = known.copy()
    for _ in range(max(int(iterations), 0)):
        if not (remaining & ~reached).any():
            break
        acc = np.zeros_like(work)
        cnt = np.zeros(work.shape, dtype=np.float64)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb_v = np.roll(reached, shift, axis=axis)
            nb = np.roll(work, shift, axis=axis)
            # Rolling wraps; a wrapped edge row must not seed the far side.
            nb_v = _clear_wrapped(np, nb_v, axis, shift)
            acc += np.where(nb_v, nb, 0.0)
            cnt += nb_v
        has = (cnt > 0) & remaining
        work = np.where(has & ~reached, acc / np.maximum(cnt, 1.0), work)
        reached |= has
    took = remaining & reached
    return work, took


def _clear_wrapped(np: Any, arr: Any, axis: int, shift: int) -> Any:
    out = arr.copy()
    if axis == 0:
        if shift > 0:
            out[0, :] = False
        else:
            out[-1, :] = False
    else:
        if shift > 0:
            out[:, 0] = False
        else:
            out[:, -1] = False
    return out


def complete_depth_from_graph(
    solve: Any,
    depth: Any,
    graph: Any,
    *,
    holes: Any = None,
    normals: Any = None,
    **kwargs: Any,
) -> DepthCompletion:
    """:func:`complete_depth` driven by an occlusion graph's fitted planes.

    Only nodes whose ``completion_policy`` actually licenses building are used.
    A node the graph marked ``none`` — an unclassifiable tear — contributes no
    plane, so the hole it guards stays open. That refusal is the whole point of
    the graph, and it has to survive into the thing that does the filling.
    """
    from atlas_camera.core.occlusion_graph import POLICY_NONE
    from atlas_camera.core.relief_mesh import ReliefMeshCameraSpec

    spec = ReliefMeshCameraSpec.from_solve(solve)

    # The backdrop is excluded for the same reason the move budget excludes it:
    # a cyclorama spans the whole frustum, so it "explains" every pixel and
    # would win tears belonging to a real surface metres in front of it.
    # Filling from it looks successful and puts the geometry at the far plane.
    # No ordering is needed beyond that — complete_depth takes the nearest
    # valid intersection per pixel.
    planes = [n.plane for n in getattr(graph, "nodes", [])
              if n.plane and n.completion_policy != POLICY_NONE
              and n.kind != "backdrop"]

    result = complete_depth(
        depth, view_matrix=spec.view_matrix, fx=spec.fx, fy=spec.fy,
        cx=spec.cx, cy=spec.cy, holes=holes, planes=planes, normals=normals,
        **kwargs,
    )
    blocked = [n.id for n in getattr(graph, "nodes", [])
               if n.completion_policy == POLICY_NONE and n.plane]
    if blocked:
        result.notes.append(
            "left unfilled by graph policy (unclassifiable tears): "
            + ", ".join(blocked)
        )
    return result
