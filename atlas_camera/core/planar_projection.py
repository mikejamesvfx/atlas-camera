"""Planar unwarp/rewarp math (pure numpy, host-agnostic).

Matte-paint workflow: flatten a solved plane (the ground, a wall facade, any
proxy plane) into an orthographic texture via the homography the recovered
camera implies, let the artist edit/inpaint the flat image, then warp the edit
back into the plate — perspective-correct by construction, because both
directions run through the same solve.

Conventions (identical to the rest of Atlas — see headless_evidence._project):
  * the 4x4 ``camera_view_matrix`` is THE world-math convention; the camera
    looks down -Z, so forward depth is ``-z_cam``;
  * pixels: ``u = cx + fx * x / w``, ``v = cy - fy * y / w`` with ``w = -z``
    (image origin top-left, y down);
  * plane frames are the THREE.PlaneGeometry frame plane_transform() encodes:
    columns u, v, n, c — in-plane X, in-plane Y, normal, center.

A ``WarpSpec`` is passed BY REFERENCE between the unwarp and rewarp nodes
(the ATLAS_DEPTH_MAP pattern — never serialized into workflow JSON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "planar projection requires numpy. Install with:\n"
            "    pip install -e .[vision]"
        ) from exc
    return np


# Rays grazing the plane flatter than this cosine-of-incidence produce texels
# stretched beyond any editing use — masked out of the flat image.
_GRAZING_COS_MIN = 0.05

# Behind-camera / near-plane guard on the homography's w (forward depth).
_W_EPS = 1e-9


@dataclass(slots=True)
class PlaneBasis:
    """A solved plane as (u, v, n, c) world vectors (each a length-3 tuple)."""

    u: tuple
    v: tuple
    n: tuple
    c: tuple
    name: str = "ground"


@dataclass(slots=True)
class WarpSpec:
    """Everything the rewarp needs to invert an unwarp, by reference."""

    homography: Any            # 3x3: flat PIXEL (x, y, 1) -> plate pixel (u*w, v*w, w)
    plate_width: int
    plate_height: int
    flat_width: int
    flat_height: int
    flat_alpha: Any            # HxW float in flat space: 1 = real plate pixel landed here
    plane: PlaneBasis = None   # type: ignore[assignment]
    px_per_meter: float = 0.0
    rect_origin_m: tuple = (0.0, 0.0)   # plane-space (u, v) of flat pixel (0, 0)
    metadata: dict = field(default_factory=dict)


def ground_plane_basis() -> PlaneBasis:
    """The solved world's ground plane (Y=0) — always exists in an Atlas solve.

    u = world +Z, v = world +X (right-handed with n = +Y, matching
    depth_geometry.arbitrary_plane_axes for a vertical normal).
    """
    return PlaneBasis(u=(0.0, 0.0, 1.0), v=(1.0, 0.0, 0.0),
                      n=(0.0, 1.0, 0.0), c=(0.0, 0.0, 0.0), name="ground")


def plane_basis_from_primitive(prim: Any) -> PlaneBasis:
    """Extract (u, v, n, c) from an AtlasProxyPrimitive's transform columns."""
    np = _require_numpy()
    m = np.asarray(prim.transform_matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"primitive '{prim.name}' has no 4x4 transform")
    return PlaneBasis(
        u=tuple(float(x) for x in m[:3, 0]),
        v=tuple(float(x) for x in m[:3, 1]),
        n=tuple(float(x) for x in m[:3, 2]),
        c=tuple(float(x) for x in m[:3, 3]),
        name=str(getattr(prim, "name", "plane")),
    )


def _camera_pieces(view_matrix: Any):
    """(R 3x3, t 3, eye 3) from the 4x4 view matrix — never the bare 3x3."""
    np = _require_numpy()
    view = np.asarray(view_matrix, dtype=np.float64)
    if view.shape != (4, 4):
        raise ValueError("camera_view_matrix must be 4x4 (the world-math rule)")
    rot = view[:3, :3]
    t = view[:3, 3]
    eye = np.linalg.inv(view)[:3, 3]
    return rot, t, eye


def homography_plane_to_image(
    view_matrix: Any, fx: float, fy: float, cx: float, cy: float,
    basis: PlaneBasis,
) -> Any:
    """3x3 H mapping plane-space METERS (a, b, 1) -> plate pixels (u*w, v*w, w).

    A plane point is ``X = c + a*u + b*v``; through the view matrix and the
    Atlas pixel convention above this collapses to a single homography.
    ``w`` is forward depth: <= 0 means behind the camera.
    """
    np = _require_numpy()
    rot, t, _ = _camera_pieces(view_matrix)
    a1 = rot @ np.asarray(basis.u, dtype=np.float64)
    a2 = rot @ np.asarray(basis.v, dtype=np.float64)
    a3 = rot @ np.asarray(basis.c, dtype=np.float64) + t
    cam_cols = np.stack([a1, a2, a3], axis=1)          # 3x3: cam = cols @ (a, b, 1)
    w_row = -cam_cols[2]                                # forward depth
    h = np.stack([
        fx * cam_cols[0] + cx * w_row,
        -fy * cam_cols[1] + cy * w_row,
        w_row,
    ], axis=0)
    return h


def flat_pixel_to_plane(px_per_meter: float, rect_origin_m: tuple) -> Any:
    """3x3 mapping flat PIXELS (x, y, 1) -> plane METERS (a, b, 1).

    Flat +x = plane +u; flat +y (down, image convention) = plane -v, so the
    flat image reads "up is up" for a wall and "away is up" for the ground.
    """
    np = _require_numpy()
    u0, v0 = float(rect_origin_m[0]), float(rect_origin_m[1])
    inv = 1.0 / float(px_per_meter)
    return np.array([[inv, 0.0, u0], [0.0, -inv, v0], [0.0, 0.0, 1.0]])


def warp_by_homography(image: Any, h: Any, out_width: int, out_height: int,
                       *, fill: float = 0.0) -> tuple:
    """Sample ``image`` at ``H @ (x_out, y_out, 1)`` for every output pixel.

    Bilinear; returns ``(rgb, alpha)`` where alpha=0 for behind-camera
    (w <= 0) or out-of-frame samples. Works for both directions — unwarp uses
    the flat->plate map, rewarp the plate->flat inverse.
    """
    np = _require_numpy()
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = img[..., None]
    ih, iw = img.shape[:2]

    xs, ys = np.meshgrid(np.arange(out_width, dtype=np.float64) + 0.5,
                         np.arange(out_height, dtype=np.float64) + 0.5)
    h = np.asarray(h, dtype=np.float64)
    denom = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
    valid = denom > _W_EPS
    safe = np.where(valid, denom, 1.0)
    su = (h[0, 0] * xs + h[0, 1] * ys + h[0, 2]) / safe - 0.5
    sv = (h[1, 0] * xs + h[1, 1] * ys + h[1, 2]) / safe - 0.5

    inb = valid & (su >= 0) & (su <= iw - 1) & (sv >= 0) & (sv <= ih - 1)
    x0 = np.clip(np.floor(su), 0, iw - 2).astype(np.int64)
    y0 = np.clip(np.floor(sv), 0, ih - 2).astype(np.int64)
    fxs = np.clip(su - x0, 0.0, 1.0)[..., None]
    fys = np.clip(sv - y0, 0.0, 1.0)[..., None]
    p00 = img[y0, x0]
    p01 = img[y0, x0 + 1]
    p10 = img[y0 + 1, x0]
    p11 = img[y0 + 1, x0 + 1]
    out = (p00 * (1 - fxs) * (1 - fys) + p01 * fxs * (1 - fys)
           + p10 * (1 - fxs) * fys + p11 * fxs * fys)
    alpha = inb.astype(np.float64)
    out = np.where(inb[..., None], out, fill)
    return out.astype(np.float32), alpha.astype(np.float32)


def _plane_coords_of_plate_rays(view_matrix, fx, fy, cx, cy, basis,
                                plate_width, plate_height, grid=64):
    """Plane-space (a, b) hit by a grid of plate-pixel rays; NaN where the ray
    misses the plane forward or grazes it below the incidence cutoff."""
    np = _require_numpy()
    rot, _, eye = _camera_pieces(view_matrix)
    n = np.asarray(basis.n, dtype=np.float64)
    c = np.asarray(basis.c, dtype=np.float64)
    us = np.linspace(0.5, plate_width - 0.5, grid)
    vs = np.linspace(0.5, plate_height - 0.5, grid)
    uu, vv = np.meshgrid(us, vs)
    d_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], axis=-1)
    d_world = d_cam @ rot                    # rot.T applied row-wise
    denom = d_world @ n
    tnum = float(np.dot(c - eye, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = tnum / denom
    d_norm = d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)
    grazing = np.abs(d_norm @ n) < _GRAZING_COS_MIN
    ok = np.isfinite(tval) & (tval > 1e-6) & ~grazing
    pts = eye[None, None, :] + tval[..., None] * d_world
    rel = pts - c[None, None, :]
    a = rel @ np.asarray(basis.u, dtype=np.float64)
    b = rel @ np.asarray(basis.v, dtype=np.float64)
    a = np.where(ok, a, np.nan)
    b = np.where(ok, b, np.nan)
    return a, b


def fit_visible_rect(view_matrix, fx, fy, cx, cy, basis,
                     plate_width, plate_height,
                     *, percentile: float = 2.0) -> tuple | None:
    """Plane-space rect (u_min, v_min, u_max, v_max) covering the visible
    footprint of the plane, percentile-trimmed so the near-horizon tail of a
    ground plane cannot blow the rect out to kilometres. None when the plane
    is not visible at all."""
    np = _require_numpy()
    a, b = _plane_coords_of_plate_rays(
        view_matrix, fx, fy, cx, cy, basis, plate_width, plate_height)
    if not np.isfinite(a).any():
        return None
    p = float(percentile)
    return (float(np.nanpercentile(a, p)), float(np.nanpercentile(b, p)),
            float(np.nanpercentile(a, 100 - p)), float(np.nanpercentile(b, 100 - p)))


def auto_px_per_meter(view_matrix, fx, fy, cx, cy, basis, rect) -> float:
    """Pixel density that roughly matches the plate's own sampling at the rect
    centre: project two plane points 1 m apart, measure their pixel distance."""
    np = _require_numpy()
    h = homography_plane_to_image(view_matrix, fx, fy, cx, cy, basis)
    a_mid = 0.5 * (rect[0] + rect[2])
    b_mid = 0.5 * (rect[1] + rect[3])

    def px(a, b):
        vec = h @ np.array([a, b, 1.0])
        w = vec[2]
        if w <= _W_EPS:
            return None
        return vec[:2] / w

    p0 = px(a_mid, b_mid)
    p1 = px(a_mid + 0.5, b_mid)
    p2 = px(a_mid, b_mid + 0.5)
    if p0 is None:
        return 50.0
    d = 0.0
    if p1 is not None:
        d = max(d, 2.0 * float(np.hypot(*(p1 - p0))))
    if p2 is not None:
        d = max(d, 2.0 * float(np.hypot(*(p2 - p0))))
    return max(d, 1.0)


def build_warp_spec(
    view_matrix, fx, fy, cx, cy, basis: PlaneBasis,
    plate_width: int, plate_height: int,
    *,
    px_per_meter: float = 0.0,
    rect: tuple | None = None,
    max_resolution: int = 4096,
) -> WarpSpec | None:
    """Resolve rect/resolution and compose the full flat-pixel -> plate-pixel
    homography. None when the plane is entirely out of view."""
    np = _require_numpy()
    if rect is None:
        rect = fit_visible_rect(view_matrix, fx, fy, cx, cy, basis,
                                plate_width, plate_height)
        if rect is None:
            return None
    u_min, v_min, u_max, v_max = (float(x) for x in rect)
    if not (u_max > u_min and v_max > v_min):
        return None
    ppm = float(px_per_meter)
    if ppm <= 0:
        ppm = auto_px_per_meter(view_matrix, fx, fy, cx, cy, basis, rect)
    flat_w = int(round((u_max - u_min) * ppm))
    flat_h = int(round((v_max - v_min) * ppm))
    longest = max(flat_w, flat_h, 1)
    if longest > int(max_resolution):
        scale = float(max_resolution) / float(longest)
        ppm *= scale
        flat_w = max(1, int(round((u_max - u_min) * ppm)))
        flat_h = max(1, int(round((v_max - v_min) * ppm)))
    flat_w = max(flat_w, 8)
    flat_h = max(flat_h, 8)

    # Flat pixel (0, 0) is the rect's TOP-left: +y down = -v, so the origin
    # sits at v_max.
    h_plane = homography_plane_to_image(view_matrix, fx, fy, cx, cy, basis)
    h_full = np.asarray(h_plane) @ flat_pixel_to_plane(ppm, (u_min, v_max))
    return WarpSpec(
        homography=h_full,
        plate_width=int(plate_width), plate_height=int(plate_height),
        flat_width=flat_w, flat_height=flat_h,
        flat_alpha=None, plane=basis, px_per_meter=ppm,
        rect_origin_m=(u_min, v_max),
        metadata={"rect": (u_min, v_min, u_max, v_max)},
    )


def unwarp_plate(image: Any, spec: WarpSpec) -> tuple:
    """Flatten the plate onto the plane rect. Returns (flat_rgb, flat_alpha)
    and records the alpha on the spec for the rewarp's coverage mask."""
    flat, alpha = warp_by_homography(
        image, spec.homography, spec.flat_width, spec.flat_height)
    spec.flat_alpha = alpha
    return flat, alpha


def rewarp_into_plate(
    edited_flat: Any, original_image: Any, spec: WarpSpec,
    *, edit_mask: Any = None, feather_px: int = 4,
) -> tuple:
    """Warp an edited flat image back and composite over the original plate.

    Only pixels that map into the flat rect (and, when given, into
    ``edit_mask``) are touched; ``feather_px`` softens the composite edge in
    PLATE space so the seam never shows as a hard cut. Returns
    ``(composited, coverage_mask)``.
    """
    np = _require_numpy()
    h_inv = np.linalg.inv(np.asarray(spec.homography, dtype=np.float64))
    warped, cov = warp_by_homography(
        edited_flat, h_inv, spec.plate_width, spec.plate_height)

    weight = cov.astype(np.float64)
    if spec.flat_alpha is not None:
        # Only where the unwarp had real plate content — keeps the composite
        # off regions the flat image invented beyond the plate's footprint.
        alpha_back, _ = warp_by_homography(
            spec.flat_alpha, h_inv, spec.plate_width, spec.plate_height)
        weight *= np.clip(np.asarray(alpha_back, dtype=np.float64)[..., 0], 0.0, 1.0)
    if edit_mask is not None:
        m_back, _ = warp_by_homography(
            np.asarray(edit_mask, dtype=np.float64),
            h_inv, spec.plate_width, spec.plate_height)
        weight *= np.clip(np.asarray(m_back, dtype=np.float64)[..., 0], 0.0, 1.0)

    k = int(feather_px)
    if k > 0:
        # Erode-by-min then average = a cheap separable feather inward.
        w = weight
        for _ in range(k):
            wp = np.pad(w, 1, mode="edge")
            w = np.minimum.reduce([wp[1:-1, 1:-1], wp[:-2, 1:-1], wp[2:, 1:-1],
                                   wp[1:-1, :-2], wp[1:-1, 2:]])
        blur = weight
        for _ in range(max(1, k // 2)):
            bp = np.pad(w, 1, mode="edge")
            w = 0.2 * (bp[1:-1, 1:-1] + bp[:-2, 1:-1] + bp[2:, 1:-1]
                       + bp[1:-1, :-2] + bp[1:-1, 2:])
        weight = np.minimum(blur, w)

    orig = np.asarray(original_image, dtype=np.float64)
    if orig.ndim == 2:
        orig = orig[..., None]
    out = orig * (1.0 - weight[..., None]) + np.asarray(warped, np.float64) * weight[..., None]
    return out.astype(np.float32), weight.astype(np.float32)
