"""Register an AI novel view's CAMERA to the measured world (generated -> measured).

WHY. `AtlasAddPatchView` places a Qwen Multiple-Angles patch with a camera that
is CONSTRUCTED from the declared named-view difference (`orbit_camera`), and
DESIGN_RULES calls "the requested angle actually corresponds to the orbit" the
load-bearing unverified assumption — `flip_azimuth` is calibrated by eye. This
module MEASURES that camera instead:

    patch pixels --MoGe--> metric pointmap in the PATCH camera frame
    primary pixels --Atlas depth--> metric points in the WORLD
    SIFT matches primary<->patch --> 3D<->3D correspondences
    RANSAC + Umeyama similarity  --> (s, R, t):  world = s*R*patch_cam + t

`(R, t)` IS the patch camera's cam->world pose; `s` is the residual scale of
the patch's monocular metric estimate against the measured world. Sparse robust
correspondences are enough for a POSE — unlike per-pixel scale on hallucinated
depth (documented insufficient, DESIGN_RULES 207) which needs every pixel right.

FIREWALL. This is one-directional. Generated pixels register TO the photographed
world; nothing here can move the primary camera or the photographed rig, and the
patch keeps `evidence_type=generated`. Rejection is a third outcome: below the
gates the caller falls back to the declared orbit and reports the numbers.

Host-agnostic: numpy (+cv2 through `multiview_features`) only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from atlas_camera.core.depth_geometry import opencv_points_to_atlas_cam
from atlas_camera.core.multiview_features import extract_features, match_features
from atlas_camera.core.multiview_types import QUALITY_PROFILES


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "atlas_camera.core.patch_camera_registration requires numpy. Install "
            "with: pip install -e .[vision]") from exc
    return np


# ---------------------------------------------------------------------------
# Similarity solve

@dataclass(slots=True)
class SimilarityResult:
    scale: float
    rotation: Any            # (3,3) — dst ≈ scale * rotation @ src + translation
    translation: Any         # (3,)
    inlier_mask: Any         # (N,) bool
    n_inliers: int
    n_candidates: int
    rms_m: float             # RMS residual over inliers, in dst units


def umeyama_similarity(src: Any, dst: Any, *, with_scale: bool = True) -> tuple[float, Any, Any]:
    """Closed-form least-squares similarity ``dst ≈ s R src + t`` (Umeyama 1991).

    ``R`` is a proper rotation (det +1 enforced via the reflection fix). Needs
    at least 3 non-degenerate points. `core.normals.procrustes_rotation` is the
    rotation-only cousin and is deliberately not reused: this one carries the
    translation and the scale that a camera pose needs.
    """
    np = _require_numpy()
    a = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(a) < 3 or len(a) != len(b):
        raise ValueError(f"need >=3 paired points; got {len(a)} and {len(b)}")
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    a0 = a - mu_a
    b0 = b - mu_b
    var_a = float((a0 ** 2).sum() / len(a))
    cov = b0.T @ a0 / len(a)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_a) if (with_scale and var_a > 1e-18) else 1.0
    t = mu_b - s * (R @ mu_a)
    return s, R, t


def ransac_similarity(src: Any, dst: Any, *, threshold_m: float, iters: int = 600,
                      min_inliers: int = 12, seed: int = 0,
                      with_scale: bool = True) -> SimilarityResult | None:
    """3-point RANSAC over `umeyama_similarity`, refit on the consensus set.

    Deterministic for a fixed ``seed``. Returns None when no hypothesis reaches
    ``min_inliers`` (or fewer than 3 candidates exist).
    """
    np = _require_numpy()
    a = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = len(a)
    if n < 3 or n != len(b):
        return None
    rng = np.random.default_rng(int(seed))
    best_mask = None
    best_count = 0
    best_rms = float("inf")
    thr2 = float(threshold_m) ** 2
    for _ in range(int(iters)):
        idx = rng.choice(n, size=3, replace=False)
        tri = a[idx]
        # Degenerate (collinear) sample: skip.
        if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 1e-12:
            continue
        # COPLANAR consensus sets are NOT degenerate for scale — 3D<->3D
        # similarity fixes it from in-plane distances, verified on a synthetic
        # facade (200 points on z=0, true scale recovered to 4 decimals). The
        # planar degeneracy is a MIRROR: a patch pointmap reflected about the
        # facade plane fits with the same scale, the same RMS, the same inlier
        # count, and a PROPER rotation (det +1), so Umeyama's reflection
        # handling cannot flag it and nothing in this function will.
        #
        # What catches it is `RegistrationConfig.max_deviation_deg`: a camera
        # on the wrong side of the facade lands far from every declared orbit
        # pose. That protection is INCIDENTAL — the gate is aimed at a
        # generator ignoring the requested angle, not at a mirrored pointmap —
        # so relaxing it removes a guard nobody wrote on purpose. No physical
        # trigger is known: MoGe regresses metric points, and the
        # concave/convex inversion that would cause this belongs to
        # shape-from-shading, not a pointmap regressor.
        try:
            s, R, t = umeyama_similarity(tri, b[idx], with_scale=with_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        pred = s * (a @ R.T) + t
        d2 = ((pred - b) ** 2).sum(axis=1)
        mask = d2 <= thr2
        count = int(mask.sum())
        if count > best_count or (count == best_count and count and
                                  float(np.sqrt(d2[mask].mean())) < best_rms):
            best_count = count
            best_mask = mask
            best_rms = float(np.sqrt(d2[mask].mean())) if count else float("inf")
    if best_mask is None or best_count < max(3, int(min_inliers)):
        return None
    # Refit on all inliers, then re-evaluate the consensus once (cheap, stabler).
    s, R, t = umeyama_similarity(a[best_mask], b[best_mask], with_scale=with_scale)
    pred = s * (a @ R.T) + t
    d2 = ((pred - b) ** 2).sum(axis=1)
    mask = d2 <= thr2
    if int(mask.sum()) >= 3:
        s, R, t = umeyama_similarity(a[mask], b[mask], with_scale=with_scale)
        pred = s * (a @ R.T) + t
        d2 = ((pred - b) ** 2).sum(axis=1)
        mask = d2 <= thr2
    else:
        mask = best_mask
    n_in = int(mask.sum())
    rms = float(np.sqrt(d2[mask].mean())) if n_in else float("inf")
    return SimilarityResult(scale=float(s), rotation=R, translation=t,
                            inlier_mask=mask, n_inliers=n_in, n_candidates=n, rms_m=rms)


# ---------------------------------------------------------------------------
# Patch-camera registration

@dataclass(slots=True)
class RegistrationConfig:
    match_quality: str = "permissive"     # QUALITY_PROFILES key; AI views match thinly
    min_inliers: int = 40
    max_residual_m: float = 0.35
    # Also the only thing standing between a mirrored planar pointmap and an
    # accepted camera on the wrong side of the facade — see the MIRROR note in
    # `ransac_similarity`. Read that before loosening this.
    max_deviation_deg: float = 25.0
    far_depth_factor: float = 3.0         # drop patch points beyond factor*median depth
    ransac_iters: int = 800
    seed: int = 0
    auto_flip: bool = True


@dataclass(slots=True)
class PatchCameraRegistration:
    accepted: bool
    reason: str
    view_matrix: Any = None               # (4,4) world->cam, row-major
    camera_position: Any = None           # (3,)
    scale: float = 1.0
    n_matches: int = 0
    n_candidates: int = 0
    n_inliers: int = 0
    rms_m: float = float("inf")
    reproj_rms_px: float = float("inf")
    deviation_deg: float = float("inf")   # vs the closest declared pose
    deviation_m: float = float("inf")
    flip_resolved: bool | None = None     # which declared pose was closer
    inlier_points_world: Any = None       # (K,3) accepted correspondences
    inlier_points_patch_px: Any = None    # (K,2)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Flat scalars only — safe for primitive/source metadata."""
        return {
            "registration_accepted": bool(self.accepted),
            "registration_reason": str(self.reason),
            "registration_scale": round(float(self.scale), 5),
            "registration_n_matches": int(self.n_matches),
            "registration_n_candidates": int(self.n_candidates),
            "registration_n_inliers": int(self.n_inliers),
            "registration_rms_m": (round(float(self.rms_m), 4)
                                   if math.isfinite(self.rms_m) else -1.0),
            "registration_reproj_rms_px": (round(float(self.reproj_rms_px), 3)
                                           if math.isfinite(self.reproj_rms_px) else -1.0),
            "registration_deviation_deg": (round(float(self.deviation_deg), 3)
                                           if math.isfinite(self.deviation_deg) else -1.0),
            "registration_deviation_m": (round(float(self.deviation_m), 3)
                                         if math.isfinite(self.deviation_m) else -1.0),
            "flip_azimuth_resolved": (bool(self.flip_resolved)
                                      if self.flip_resolved is not None else False),
        }


def _sample_points(points: Any, xy: Any) -> Any:
    """Nearest-pixel lookup of an (H,W,3) map at (N,2) float pixel coords."""
    np = _require_numpy()
    h, w = points.shape[:2]
    u = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
    return points[v, u]


def _pose_from_view(view_matrix: Any) -> tuple[Any, Any]:
    """(camera_position, forward_unit) in world from a world->cam 4x4."""
    np = _require_numpy()
    c2w = np.linalg.inv(np.asarray(view_matrix, dtype=np.float64).reshape(4, 4))
    fwd = -c2w[:3, 2]
    return c2w[:3, 3], fwd / max(np.linalg.norm(fwd), 1e-12)


def _view_from_pose(R_cw: Any, position: Any) -> Any:
    np = _require_numpy()
    c2w = np.eye(4)
    c2w[:3, :3] = np.asarray(R_cw, dtype=np.float64)
    c2w[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return np.linalg.inv(c2w)


def register_patch_camera(
    *,
    patch_image: Any,
    primary_image: Any,
    patch_points_cam: Any,
    primary_points_world: Any,
    patch_intrinsics: dict[str, float],
    declared_view_matrices: dict[str, Any] | None = None,
    config: RegistrationConfig | None = None,
) -> PatchCameraRegistration:
    """Measure the patch camera against the primary's metric world.

    ``patch_points_cam``: (Hp,Wp,3) metric pointmap of the PATCH in OpenCV camera
    axes (MoGe's ``points`` — `DepthResult.points`), NaN where invalid.
    ``primary_points_world``: (H,W,3) world points of the PRIMARY (Atlas depth
    back-projected through the recovered camera), NaN where invalid.
    ``patch_intrinsics``: ``{"fx","fy","cx","cy"}`` in patch pixels — used only
    for the reprojection diagnostic; the pose is solved in 3D.
    ``declared_view_matrices``: e.g. ``{"noflip": vm, "flip": vm}`` — the
    constructed orbit poses; the closest one defines ``deviation_*`` and
    ``flip_resolved`` (key == "flip").
    """
    np = _require_numpy()
    cfg = config or RegistrationConfig()
    profile = QUALITY_PROFILES.get(cfg.match_quality, QUALITY_PROFILES["permissive"])

    fa = extract_features(primary_image, profile)
    fb = extract_features(patch_image, profile)
    pm = match_features(fa, fb, profile, 0, 1)
    n_matches = int(len(pm.points_a))
    if n_matches < 3:
        return PatchCameraRegistration(False, f"only {n_matches} feature matches", n_matches=n_matches)

    ppw = np.asarray(primary_points_world, dtype=np.float64)
    ppc = np.asarray(patch_points_cam, dtype=np.float64)
    dst = _sample_points(ppw, np.asarray(pm.points_a, dtype=np.float64))
    src_cv = _sample_points(ppc, np.asarray(pm.points_b, dtype=np.float64))
    ok = np.isfinite(dst).all(axis=1) & np.isfinite(src_cv).all(axis=1)
    if ok.any():
        depth_p = src_cv[:, 2]
        med = float(np.median(depth_p[ok]))
        if med > 0:
            ok &= depth_p <= cfg.far_depth_factor * med   # MoGe far-field runaway
    n_cand = int(ok.sum())
    if n_cand < 3:
        return PatchCameraRegistration(False, f"{n_cand} usable 3D correspondences "
                                       f"of {n_matches} matches", n_matches=n_matches,
                                       n_candidates=n_cand)
    src = opencv_points_to_atlas_cam(src_cv[ok])
    dst = dst[ok]
    px_b = np.asarray(pm.points_b, dtype=np.float64)[ok]

    fit = ransac_similarity(src, dst, threshold_m=float(cfg.max_residual_m),
                            iters=int(cfg.ransac_iters),
                            min_inliers=min(int(cfg.min_inliers), 3), seed=int(cfg.seed))
    if fit is None:
        return PatchCameraRegistration(False, "RANSAC found no consensus",
                                       n_matches=n_matches, n_candidates=n_cand)

    R_cw = fit.rotation
    pos = fit.translation
    view = _view_from_pose(R_cw, pos)

    # Reprojection diagnostic in patch pixels: world inliers -> patch camera.
    K = patch_intrinsics
    cam = (dst[fit.inlier_mask] - pos) @ R_cw          # world -> patch cam (Atlas axes)
    z = -cam[:, 2]
    with np.errstate(all="ignore"):
        u = K["cx"] + K["fx"] * cam[:, 0] / z
        v = K["cy"] - K["fy"] * cam[:, 1] / z
    good = np.isfinite(u) & np.isfinite(v) & (z > 1e-6)
    pxs = px_b[fit.inlier_mask]
    reproj = (float(np.sqrt(((u[good] - pxs[good, 0]) ** 2 +
                             (v[good] - pxs[good, 1]) ** 2).mean()))
              if good.any() else float("inf"))

    # Deviation from the declared orbit(s).
    dev_deg = float("inf")
    dev_m = float("inf")
    flip_resolved = None
    _, fwd = _pose_from_view(view)
    for key, vm in (declared_view_matrices or {}).items():
        try:
            dpos, dfwd = _pose_from_view(vm)
        except Exception:  # noqa: BLE001
            continue
        ang = math.degrees(math.acos(float(np.clip(np.dot(fwd, dfwd), -1.0, 1.0))))
        dist = float(np.linalg.norm(pos - dpos))
        if ang < dev_deg:
            dev_deg, dev_m, flip_resolved = ang, dist, (key == "flip")

    reasons = []
    warnings = []
    if fit.n_inliers < int(cfg.min_inliers):
        reasons.append(f"{fit.n_inliers} inliers < {int(cfg.min_inliers)}")
    if fit.rms_m > float(cfg.max_residual_m):
        reasons.append(f"rms {fit.rms_m:.3f} m > {float(cfg.max_residual_m):.3f}")
    if declared_view_matrices and math.isfinite(dev_deg) and dev_deg > float(cfg.max_deviation_deg):
        # A STRONG measurement that disagrees with the declaration means the
        # generator ignored the requested angle — the pixels are where they
        # are, so the measurement wins and the disagreement is reported. Only
        # a weak match (below 2x the inlier floor) lets deviation refuse:
        # then "far from declared" is more likely a bad fit than a bad prompt.
        # Found live 2026-08-16: Qwen re-framed a castle 29 deg off the
        # requested quarter view with 125 inliers @ 0.13 m; refusing placed the
        # patch where the prompt CLAIMED instead of where the pixels are.
        strong = fit.n_inliers >= 2 * int(cfg.min_inliers) and fit.rms_m <= 0.5 * float(cfg.max_residual_m)
        msg = (f"deviates {dev_deg:.1f}° from the declared orbit "
               f"(> {float(cfg.max_deviation_deg):.1f}°)")
        if strong:
            warnings.append(msg + " — strong match, measurement kept: the generator ignored the requested angle")
        else:
            reasons.append(msg)
    if not cfg.auto_flip and flip_resolved:
        # The caller asked to trust its own flip setting; a closer FLIPPED pose
        # is then a disagreement, not a resolution.
        reasons.append("closest declared pose is the FLIPPED one but auto_flip is off")
    accepted = not reasons
    return PatchCameraRegistration(
        accepted=accepted,
        reason="; ".join(reasons) if reasons else ("registered; " + "; ".join(warnings) if warnings else "registered"),
        view_matrix=view, camera_position=pos, scale=float(fit.scale),
        n_matches=n_matches, n_candidates=n_cand, n_inliers=int(fit.n_inliers),
        rms_m=float(fit.rms_m), reproj_rms_px=reproj,
        deviation_deg=dev_deg, deviation_m=dev_m, flip_resolved=flip_resolved,
        inlier_points_world=dst[fit.inlier_mask], inlier_points_patch_px=pxs,
        diagnostics={"match_quality": cfg.match_quality,
                     "far_depth_factor": float(cfg.far_depth_factor),
                     "ransac_iters": int(cfg.ransac_iters), "seed": int(cfg.seed)},
    )
