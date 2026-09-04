"""Register a patch view against the primary camera's measured depth.

WHY THIS MODULE EXISTS. This was ~118 lines inside `AtlasAddPatchView.add`, in
`comfy/`. Nothing in it mentions torch or a ComfyUI type — it is a pinhole
projection, a forward splat, and a closed-form scale solve — but its only
interface was a node class, so the whole derivation was reachable in tests only
by constructing a node and reading one metadata string off the result.

THE SCALE MODEL. A patch view's monocular depth is relative: a point the primary
camera measured at metric distance `m` appears in the patch at some depth whose
world position, once the patch depth is multiplied by an unknown scale `s`,
must land back on the primary's measurement. Along the ray from the patch
camera the primary-frame depth is affine in `s`:

    z(s) = z_cam + s * (z_p - z_cam)

with `z_cam` the patch camera's own depth in the primary frame and `z_p` the
unscaled point's. Setting z(s) = m and inverting gives one estimate per pixel:

    s = (m - z_cam) / (z_p - z_cam)

The per-pixel estimates are then reduced by a MEDIAN, not a mean — a handful of
pixels where the monocular depth disagrees with the plate would drag a mean
anywhere — and the whole answer is refused below a support floor, because a
median of forty samples is not a registration.

Host-agnostic: numpy only, no torch, no ComfyUI.
"""
from __future__ import annotations

from typing import Any

from atlas_camera.core.mask_ops import dilate


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.patch_registration requires numpy. Install "
            "with: pip install -e .[vision]") from exc
    return np


def _project(points_world: Any, camera: dict[str, Any]) -> tuple[Any, Any, Any]:
    """World points -> (u, v, forward_depth) in a camera. NaN where behind."""
    np = _require_numpy()
    vm = np.asarray(camera["view_matrix"], dtype=np.float64).reshape(4, 4)
    rotation, translation = vm[:3, :3], vm[:3, 3]
    cam_pts = points_world @ rotation.T + translation
    depth = -cam_pts[..., 2]
    in_front = depth > 1e-6
    with np.errstate(all="ignore"):
        safe = np.where(in_front, depth, np.nan)
        u = camera["cx"] + camera["fx"] * cam_pts[..., 0] / safe
        v = camera["cy"] - camera["fy"] * cam_pts[..., 1] / safe
    return u, v, depth


def splat_coverage(points_world: Any, *, camera: dict[str, Any],
                   close_px: int = 0) -> Any:
    """Which pixels of `camera` receive a forward-splatted world point.

    Coverage means "the primary has trusted data that lands on this pixel" —
    no invented patch depth is involved. The closing step exists because a
    sparse splat leaves holes between projected samples, and an UNDERCOUNTED
    coverage is the dangerous direction: it marks real pixels as unseen and
    lets a generated patch overwrite them.
    """
    np = _require_numpy()
    width, height = int(camera["width"]), int(camera["height"])
    coverage = np.zeros((height, width), dtype=bool)
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if not len(pts):
        return coverage
    u, v, depth = _project(pts, camera)
    hit = (
        (depth > 1e-6) & np.isfinite(u) & np.isfinite(v)
        & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    )
    coverage[v[hit].astype(np.int64), u[hit].astype(np.int64)] = True
    return dilate(coverage, int(close_px))


def solve_scale_from_primary(
    patch_depth: Any,
    *,
    patch_camera: dict[str, Any],
    patch_camera_position: Any,
    primary_metric_map: Any,
    primary_camera: dict[str, Any],
    exclude_mask: Any = None,
    min_samples: int = 500,
) -> tuple[float | None, dict[str, Any]]:
    """Metric scale for a patch view's relative depth, or None.

    Returns `(scale, info)`. `scale` is None when the registration lacked
    support; `info` always reports `n_samples` and `accepted` so a caller can
    say WHY it fell back rather than silently choosing another source.
    """
    np = _require_numpy()
    from atlas_camera.core.depth_geometry import back_project_normals

    metric = np.asarray(primary_metric_map, dtype=np.float64)
    back = back_project_normals(
        np.asarray(patch_depth, dtype=np.float64),
        view_matrix=patch_camera["view_matrix"],
        fx=patch_camera["fx"], fy=patch_camera["fy"],
        cx=patch_camera["cx"], cy=patch_camera["cy"],
    )
    u, v, z_p = _project(back.pts_world, primary_camera)

    vm = np.asarray(primary_camera["view_matrix"], dtype=np.float64).reshape(4, 4)
    rotation, translation = vm[:3, :3], vm[:3, 3]
    position = np.asarray([float(x) for x in patch_camera_position],
                          dtype=np.float64)
    z_cam = float(-(rotation @ position + translation)[2])

    p_w, p_h = int(primary_camera["width"]), int(primary_camera["height"])
    in_frame = (
        np.isfinite(u) & np.isfinite(v)
        & (u >= 0) & (u < p_w) & (v >= 0) & (v < p_h)
    )
    sx = np.clip(np.where(in_frame, u, 0.0), 0, metric.shape[1] - 1).astype(np.int64)
    sy = np.clip(np.where(in_frame, v, 0.0), 0, metric.shape[0] - 1).astype(np.int64)
    sampled = metric[sy, sx]

    denom = z_p - z_cam
    ok = (
        back.valid_depth & in_frame & (z_p > 1e-6)
        & np.isfinite(sampled) & (sampled > 1e-4) & (np.abs(denom) > 1e-3)
    )
    if exclude_mask is not None:
        # Sky pixels are noise for registration.
        ok &= ~np.asarray(exclude_mask, dtype=bool)
    with np.errstate(all="ignore"):
        samples = (sampled - z_cam) / denom
    ok &= np.isfinite(samples) & (samples > 1e-3) & (samples < 1e3)

    count = int(ok.sum())
    info: dict[str, Any] = {"n_samples": count, "min_samples": int(min_samples)}
    if count < int(min_samples):
        info["accepted"] = False
        return None, info
    info["accepted"] = True
    # The MEDIAN is only as good as the agreement behind it. Two adjacent
    # ground patches on the castle fitted 0.645 and 0.273 -- a 2.4x
    # disagreement on the same scene at the same distance -- and nothing said
    # the second fit was worse conditioned than the first, because only the
    # median came back. Report the spread so a caller can tell a converged fit
    # from a coin toss: `scale_rel_mad` is the median absolute deviation over
    # the median, i.e. 0 for perfect agreement and ~1 for none.
    chosen = samples[ok]
    med = float(np.median(chosen))
    mad = float(np.median(np.abs(chosen - med)))
    p25 = float(np.percentile(chosen, 25))
    p75 = float(np.percentile(chosen, 75))
    info["scale_p25"] = p25
    info["scale_p75"] = p75
    info["scale_rel_mad"] = float(mad / med) if med > 1e-9 else float("inf")
    # The IQR is the one to read. `rel_mad` is a median of medians and so
    # inherits the same 50% breakdown as the estimate it is describing -- on a
    # third of samples disagreeing it still reports 0. The quartile spread moves
    # at a quarter, which is where a fit starts being worth doubting.
    info["scale_rel_iqr"] = float((p75 - p25) / med) if med > 1e-9 else float("inf")
    return med, info
