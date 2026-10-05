"""Place a pixel-aligned generated object mesh into a solved Atlas scene.

A pixel-aligned image-to-3D model (Pixal3D) returns a mesh in its own unit
frame: the object inside ``[-0.5, 0.5]^3``, Y up, +Z toward a centred pinhole
camera at ``(0, 0, d)`` with ``d = 0.5 / tan(fov/2)`` (V135
``comfy_extras/nodes_trellis2.py`` Pixal3DConditioning). Fed a crop from
:func:`atlas_camera.core.object_crop.virtual_object_camera`, that pinhole IS
the virtual object camera, so the chain back to Atlas world is exact up to ONE
unknown -- scale along the camera rays, which the model cannot know::

    p_v = v - (0, 0, d)            # Pixal frame -> virtual camera frame
    p_s = R_v^T p_v                # virtual camera -> source camera (rotation)
    X   = inv(view) [s p_s, 1]     # source camera -> world, s unknown

Uniform scale about the camera centre preserves every pixel, so ``s`` is
measured, never assumed: the mesh's own forward z-buffer at ``s = 1`` against
the shared metric depth over the object's visible pixels, one median ratio
(:func:`atlas_camera.core.hidden_geometry.register_layers_to_depth`, the same
estimator ``plate_depth`` uses). :func:`ground_contact_scale` is an independent
second opinion for objects that stand on the ground.

The colour doctrine: the photo paints wherever the solved camera SAW the
surface; the model's vertex colour paints only what it did not.
:func:`photo_visibility_weights` decides which is which per vertex. The hidden
side is a hypothesis -- no reprojection test can confirm it, and the report
says so.

Layering: ``core`` only -- numpy, no torch, no ComfyUI.
"""

from __future__ import annotations

import math
from typing import Any

from atlas_camera.core.mask_ops import erode
from atlas_camera.core.scene_health import (
    generated_object_grade,
)
# Re-exported: callers and tests import the curve from here.
from atlas_camera.core.srgb import linear_to_srgb, srgb_to_linear  # noqa: F401

#: ``metadata["source"]`` of every generated object mesh (Pixal3D import).
#: The import node writes it; scene_health, the USD exporter and the viewport
#: payload select generated meshes by it. One value, here.
GENERATED_SOURCE = "pixal3d"

#: Per-vertex ``photo_weight`` above this paints from the photo, below from
#: the model's vertex colour. Mirrored in atlas_blockout.js
#: (``PHOTO_WEIGHT_SPLIT``) and pinned by tests/test_frontend_mirrors.py.
PHOTO_WEIGHT_SPLIT = 0.5

#: A facing below this (|n . to_camera|) is grazing: the photo smears there.
MIN_PHOTO_FACING = 0.2

#: Relative slack on the mesh's own z-buffer test (decimation + rounding).
SELF_DEPTH_BIAS_REL = 0.02

#: Relative slack on the Atlas depth test. The mesh is registered by a
#: median, so its surface sits within rel_mad of the depth map; only content
#: clearly IN FRONT (a pole across the object) should take the photo away.
SCENE_DEPTH_BIAS_REL = 0.15

#: Vertex-colour gain is clamped; a gain outside this is a mismatch the
#: report should show, not a correction to apply.
COLOUR_GAIN_RANGE = (0.25, 4.0)
# Was (0.5, 2.0) until found live 2026-10-05: a black SUV in a dusk RAW wanted
# a gain below 0.5 (the model textures it as if lit at a normal exposure), so
# the clamp left the hidden side ~2x too bright. Two stops each way still
# flags a genuinely wrong match.

MIN_REGISTRATION_PX = 100


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Generated-mesh placement requires numpy. Install with: pip install -e .[vision]"
        ) from exc


# ---------------------------------------------------------------------------
# Frame chain
# ---------------------------------------------------------------------------

def pixal_camera_distance(fov_deg: float) -> float:
    """Pixal3D's camera distance to the unit-cube centre for a given FOV."""
    return 0.5 / math.tan(math.radians(float(fov_deg)) / 2.0)


def pixal_to_source_camera(vertices: Any, *, rotation: Any, fov_deg: float) -> Any:
    """Pixal-frame vertices -> SOURCE camera frame, at scale 1.

    ``rotation`` is the crop's ``R_v`` (source -> virtual). Row-vector form of
    ``R_v^T p_v`` is ``p_v @ R_v``.
    """
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    p_v = v - np.array([0.0, 0.0, pixal_camera_distance(fov_deg)])
    return p_v @ np.asarray(rotation, dtype=np.float64)


def source_camera_to_world(points_cam: Any, *, view_matrix: Any, scale: float = 1.0) -> Any:
    """Camera-frame points scaled about the camera centre, then to world.

    Uses the full 4x4 (``inv(view)``), never a 3x3 -- transpose ambiguity.
    """
    np = _require_numpy()
    c2w = np.linalg.inv(np.asarray(view_matrix, dtype=np.float64))
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3) * float(scale)
    return p @ c2w[:3, :3].T + c2w[:3, 3]


def world_to_source_camera(points_world: Any, *, view_matrix: Any) -> Any:
    np = _require_numpy()
    vm = np.asarray(view_matrix, dtype=np.float64)
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    return p @ vm[:3, :3].T + vm[:3, 3]


def cluster_decimate(vertices: Any, faces: Any, *, max_faces: int,
                     colours: Any = None, max_rounds: int = 12,
                     return_stats: bool = False) -> tuple[Any, ...]:
    """Vertex-clustering decimation to at most ``max_faces`` triangles.

    The mesh rides the solve as JSON, so its size is a payload budget, not a
    quality knob; model-side DecimateMesh is the better tool and should run
    first. This is the backstop: quantize to a grid, merge each cell to its
    mean (colours averaged with it), drop collapsed and duplicate faces, and
    coarsen the grid until under budget. Returns ``(v, f, colours|None)``;
    with ``return_stats`` a fourth item ``{"met_budget", "rounds",
    "faces_in", "faces_out"}`` -- ``met_budget`` is False when ``max_rounds``
    ran out with the mesh still over budget (the caller must SAY so).
    """
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    c = None if colours is None else np.asarray(colours, dtype=np.float64)
    n_in = int(len(f))

    def done(nv, nf, nc, rounds):
        if not return_stats:
            return nv, nf, nc
        return nv, nf, nc, {"met_budget": bool(len(nf) <= int(max_faces)),
                            "rounds": int(rounds), "faces_in": n_in,
                            "faces_out": int(len(nf))}

    if len(f) <= int(max_faces):
        return done(v, f, c, 0)
    extent = float(np.max(v.max(axis=0) - v.min(axis=0))) or 1.0
    cells = max(8, int(math.sqrt(max_faces / 2.0) * 1.5))
    nv, nf, nc = v, f, c
    rounds = 0
    for _ in range(int(max_rounds)):
        rounds += 1
        q = np.floor((v - v.min(axis=0)) / (extent / cells)).astype(np.int64)
        _, inv = np.unique(q, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        n = int(inv.max()) + 1
        cnt = np.bincount(inv, minlength=n).astype(np.float64)[:, None]
        nv = np.stack([np.bincount(inv, weights=v[:, k], minlength=n) for k in range(3)],
                      axis=1) / cnt
        nc = None
        if c is not None:
            nc = np.stack([np.bincount(inv, weights=c[:, k], minlength=n)
                           for k in range(c.shape[1])], axis=1) / cnt
        nf = inv[f]
        keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[keep]
        if len(nf):
            nf = np.unique(nf, axis=0)
        if len(nf) <= int(max_faces):
            return done(nv, nf, nc, rounds)
        cells = max(4, int(cells * 0.8))
    return done(nv, nf, nc, rounds)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

def _erode(np: Any, mask: Any, px: int) -> Any:
    """4-connected erosion by ``px`` steps (``px <= 0``: unchanged)."""
    return erode(mask, int(px), connectivity=4)


def register_object_scale(
    points_cam: Any,
    faces: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    metric_depth: Any,
    object_mask: Any,
    depth_valid: Any = None,
    erode_px: int = 3,
    backend: str = "auto",
) -> dict[str, Any]:
    """Measure the ray scale ``s`` against the shared METRIC depth.

    Rasterizes the mesh at ``s = 1`` into the solve camera and takes one
    median ratio ``metric_depth / mesh_z`` over coverage AND the eroded object
    mask (rims are where monocular depth and the mesh disagree most) AND valid
    depth. Returns ``scale``, ``rel_mad``, pixel counts, and the unit-scale
    coverage + z-buffer so callers can score the placement without
    re-rasterizing (at scale ``s`` the z-buffer is exactly ``s * z``).
    """
    np = _require_numpy()
    from atlas_camera.core.hidden_geometry import register_layers_to_depth
    from atlas_camera.core.plate_falsification import rasterize_candidate

    world_unit = source_camera_to_world(points_cam, view_matrix=view_matrix, scale=1.0)
    alpha, z = rasterize_candidate(
        world_unit, np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
        width=int(width), height=int(height), backend=backend)
    # inf (uncovered) must become NaN BEFORE the ratio: inf passes a
    # "> valid_min" test and turns into a ratio of 0.
    z = np.where(np.isfinite(z), z, np.nan)

    depth = np.asarray(metric_depth, dtype=np.float64)
    if depth.shape != (int(height), int(width)):
        raise ValueError(f"metric depth {depth.shape} does not match the solve "
                         f"raster {(int(height), int(width))}")
    region = alpha & _erode(np, object_mask, erode_px) & np.isfinite(depth) & (depth > 0)
    if depth_valid is not None:
        region &= np.asarray(depth_valid, dtype=bool)
    n_region = int(region.sum())
    result: dict[str, Any] = {
        "coverage_px": int(alpha.sum()),
        "registration_px": n_region,
        "unit_alpha": alpha,
        "unit_depth": z,
    }
    if n_region < MIN_REGISTRATION_PX:
        result.update(scale=None, rel_mad=float("inf"),
                      reason=f"only {n_region} pixels where the mesh, the eroded "
                             f"object mask and valid depth overlap "
                             f"(need {MIN_REGISTRATION_PX})")
        return result
    zl = np.where(region, z, np.nan)[..., None]
    dv = np.where(region, depth, np.nan)
    scale, rel_mad, _ = register_layers_to_depth(zl, dv)
    result.update(scale=float(scale), rel_mad=float(rel_mad), reason="")
    return result


def ground_contact_scale(points_cam: Any, *, view_matrix: Any, ground_y: float = 0.0) -> float | None:
    """The ray scale at which the object's lowest point touches ``Y = ground_y``.

    ``Y_v(s) = c_y + s * a_v`` along each vertex ray, so the lowest point
    reaches the ground at ``s = (ground_y - c_y) / min(a_v)``. ``None`` when
    the object never descends toward the ground (camera below it, or every
    ray rises) -- the check is then not applicable, not passed.
    """
    np = _require_numpy()
    c2w = np.linalg.inv(np.asarray(view_matrix, dtype=np.float64))
    a = (np.asarray(points_cam, dtype=np.float64).reshape(-1, 3) @ c2w[:3, :3].T)[:, 1]
    c_y = float(c2w[1, 3])
    a_min = float(a.min()) if a.size else 0.0
    if a_min >= -1e-12 or c_y <= ground_y:
        return None
    s = (float(ground_y) - c_y) / a_min
    return s if s > 0 else None


def scale_verdict(*, rel_mad: float, depth_scale: float | None,
                  ground_scale: float | None) -> dict[str, Any]:
    """Grade the scale registration alone (thin wrapper, kept for callers).

    The verdict is :func:`atlas_camera.core.scene_health.generated_object_grade`
    with no plate scores. Thresholds are UNCALIBRATED.
    """
    grade, issues = generated_object_grade(depth_scale, rel_mad, None,
                                           ground_scale=ground_scale)
    return {"grade": grade, "issues": issues, "calibrated": False}


# ---------------------------------------------------------------------------
# Colour: photo where seen, vertex colour where not
# ---------------------------------------------------------------------------

def vertex_normals(vertices: Any, faces: Any) -> Any:
    """Area-weighted vertex normals (unit; zero where undefined)."""
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norm > 1e-12, n / np.maximum(norm, 1e-12), 0.0)


#: Width of the photo / generated-colour ramp, as a fraction of the object's
#: bounding-box diagonal. A FIXED iteration count made the ramp shrink with
#: mesh density (found live 2026-10-05: at 198k faces the seam was one triangle
#: row wide and jagged), so the iterations are derived from this.
PHOTO_FEATHER_FRAC = 0.01

#: Reach of the LOCAL colour match into the hidden side (fraction of the bbox
#: diagonal); past it the gain fades to the single global one.
LOCAL_GAIN_REACH_FRAC = 0.06

#: Smoothing iterations are capped (each is two bincounts over 6F entries).
MAX_SMOOTH_ITERATIONS = 400


def weld_index(vertices: Any, *, rel_tol: float = 1e-6) -> tuple[Any, int]:
    """``(inverse, n_unique)``: vertices at the same position share an index.

    A UV-unwrapped mesh SPLITS its vertices along every chart seam, so its face
    adjacency never crosses a seam: smoothing or normals computed on it are
    per-chart (jaggies along chart boundaries). Analysis runs on the welded
    graph; the mesh itself (and its UVs) is never changed.
    """
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if not len(v):
        return np.zeros(0, dtype=np.int64), 0
    span = float(np.ptp(v, axis=0).max()) or 1.0
    key = np.round(v / (span * rel_tol)).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return inverse, int(inverse.max()) + 1


def _welded(np: Any, values: Any, inverse: Any, n_unique: int) -> Any:
    """Per-vertex values averaged onto welded vertices."""
    vals = np.asarray(values, dtype=np.float64)
    flat = vals.reshape(len(vals), -1)
    cnt = np.bincount(inverse, minlength=n_unique).astype(np.float64)
    out = np.stack([np.bincount(inverse, weights=flat[:, k], minlength=n_unique)
                    for k in range(flat.shape[1])], axis=1) / np.maximum(cnt, 1.0)[:, None]
    return out.reshape((n_unique,) + vals.shape[1:])


def smoothing_iterations_for(vertices: Any, faces: Any, width_frac: float) -> int:
    """Adjacency-smoothing iterations whose spread is ``width_frac`` of the
    bbox diagonal: each ``0.5 * self + 0.5 * mean(neighbours)`` step spreads
    by ~ edge / sqrt(2), so ``k ~ 2 * (width / mean_edge) ** 2``."""
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return 0
    diag = float(np.linalg.norm(np.ptp(v, axis=0)))
    e = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
    edge = float(np.median(e[e > 0])) if (e > 0).any() else 0.0
    if diag <= 0 or edge <= 0:
        return 3
    return int(np.clip(2.0 * (width_frac * diag / edge) ** 2, 3, MAX_SMOOTH_ITERATIONS))


def _vertex_adjacency_smooth(np: Any, values: Any, faces: Any, iterations: int) -> Any:
    out = np.asarray(values, dtype=np.float64).copy()
    if iterations <= 0 or faces.size == 0:
        return out
    n = out.shape[0]
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                        faces[:, 1], faces[:, 2], faces[:, 0]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                        faces[:, 0], faces[:, 1], faces[:, 2]])
    deg = np.bincount(a, minlength=n).astype(np.float64)
    two_d = out.ndim == 2
    cols = out if two_d else out[:, None]
    for _ in range(int(iterations)):
        nb = np.empty_like(cols)
        for k in range(cols.shape[1]):
            acc = np.bincount(a, weights=cols[b, k], minlength=n)
            nb[:, k] = np.where(deg > 0, acc / np.maximum(deg, 1.0), cols[:, k])
        cols = 0.5 * cols + 0.5 * nb
    return cols if two_d else cols[:, 0]


def _welded_faces(np: Any, faces: Any, inverse: Any) -> Any:
    f = inverse[np.asarray(faces, dtype=np.int64).reshape(-1, 3)]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return f[keep]


def photo_visibility_weights(
    world_vertices: Any,
    faces: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    object_mask: Any,
    metric_depth: Any = None,
    mesh_depth: Any = None,
    erode_px: int = 2,
    smooth_iterations: int | None = None,
    backend: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    """Per-vertex ``photo_weight`` in [0, 1]: did the solved camera SEE it?

    1 only where the vertex is in frame, is the mesh's own front surface
    (self z-buffer; sign-free so inconsistent winding cannot flip it), is not
    behind nearer scene content (Atlas metric depth), falls inside the eroded
    object mask, and faces the camera better than :data:`MIN_PHOTO_FACING`.
    Smoothed over the WELDED vertex adjacency (:func:`weld_index`: a UV
    unwrap's split seams would otherwise smooth each chart on its own) so the
    photo/vertex-colour seam is a ramp, not a stair. ``smooth_iterations``
    None = a ramp :data:`PHOTO_FEATHER_FRAC` of the object wide, whatever the
    mesh density. ``mesh_depth`` (the placed mesh's z-buffer) is rasterized
    here when not supplied.
    """
    np = _require_numpy()
    v = np.asarray(world_vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam = v @ vm[:3, :3].T + vm[:3, 3]
    fwd = -cam[:, 2]
    in_front = fwd > 1e-9
    safe = np.where(in_front, fwd, 1.0)
    px = cx + fx * cam[:, 0] / safe
    py = cy - fy * cam[:, 1] / safe
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    in_frame = in_front & (ix >= 0) & (ix < int(width)) & (iy >= 0) & (iy < int(height))
    ixc = np.clip(ix, 0, int(width) - 1)
    iyc = np.clip(iy, 0, int(height) - 1)

    if mesh_depth is None:
        from atlas_camera.core.plate_falsification import rasterize_candidate
        _, mesh_depth = rasterize_candidate(
            v, f, view_matrix=vm, fx=fx, fy=fy, cx=cx, cy=cy,
            width=int(width), height=int(height), backend=backend)
    zbuf = np.asarray(mesh_depth, dtype=np.float64)
    z_at = zbuf[iyc, ixc]
    self_front = np.isfinite(z_at) & (fwd <= z_at * (1.0 + SELF_DEPTH_BIAS_REL))

    if metric_depth is not None:
        d = np.asarray(metric_depth, dtype=np.float64)
        d_at = d[iyc, ixc]
        occluded = np.isfinite(d_at) & (d_at > 0) & (fwd > d_at * (1.0 + SCENE_DEPTH_BIAS_REL))
    else:
        occluded = np.zeros(len(v), dtype=bool)

    mask_e = _erode(np, object_mask, erode_px)
    in_mask = mask_e[iyc, ixc]

    inverse, n_u = weld_index(v)
    fw = _welded_faces(np, f, inverse)
    vw = _welded(np, v, inverse, n_u)
    c2w = np.linalg.inv(vm)
    to_cam = c2w[:3, 3][None, :] - v
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-12)
    # welded normals: a split UV seam would otherwise give each chart its own
    normals = vertex_normals(vw, fw)[inverse]
    facing = np.abs(np.sum(normals * to_cam, axis=1))
    facing_ok = facing > MIN_PHOTO_FACING

    hard = in_frame & self_front & ~occluded & in_mask & facing_ok
    iters = (smoothing_iterations_for(vw, fw, PHOTO_FEATHER_FRAC)
             if smooth_iterations is None else int(smooth_iterations))
    hard_w = _welded(np, hard.astype(np.float64), inverse, n_u)
    weight = np.clip(_vertex_adjacency_smooth(np, hard_w, fw, iters), 0.0, 1.0)[inverse]
    stats = {
        "n_vertices": int(len(v)),
        "photo_fraction": float((weight >= PHOTO_WEIGHT_SPLIT).mean()) if len(v) else 0.0,
        "out_of_frame": int((~in_frame).sum()),
        "self_hidden": int((in_frame & ~self_front).sum()),
        "scene_occluded": int((in_frame & occluded).sum()),
        "outside_mask": int((in_frame & ~in_mask).sum()),
        "grazing": int((in_frame & ~facing_ok).sum()),
        "smooth_iterations": int(iters),
        "welded_vertices": int(n_u),
    }
    return weight, stats


def local_colour_gains(
    vertices: Any,
    faces: Any,
    vertex_colours_srgb: Any,
    photo_weight: Any,
    plate_samples_srgb: Any,
    *,
    global_gain: Any = (1.0, 1.0, 1.0),
    reach_frac: float = LOCAL_GAIN_REACH_FRAC,
    seen_min: float = 0.9,
) -> tuple[Any, dict[str, Any]]:
    """Per-vertex LINEAR gain matching the model's colour to the photo LOCALLY.

    One global gain per channel leaves a seam wherever the model's hue or
    exposure drifts locally (found live 2026-10-05: teal-rust photo against a
    pinker generated texture). Here the log ratio photo / model is measured on
    vertices the camera saw, low-passed, and carried into the hidden side by
    normalised diffusion over the WELDED adjacency: its confidence decays with
    distance from the seen region, and the gain blends to ``global_gain`` as
    it does. Ratios are clamped to :data:`COLOUR_GAIN_RANGE`.

    Returns ``(gains (N, 3), stats)``.
    """
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    vc = srgb_to_linear(np.asarray(vertex_colours_srgb, dtype=np.float64)[:, :3])
    plate = srgb_to_linear(np.asarray(plate_samples_srgb, dtype=np.float64)[:, :3])
    w = np.asarray(photo_weight, dtype=np.float64).reshape(-1)
    lo, hi = COLOUR_GAIN_RANGE
    g0 = np.log2(np.clip(np.asarray(global_gain, dtype=np.float64), lo, hi))
    inverse, n_u = weld_index(v)
    fw = _welded_faces(np, f, inverse)
    vw = _welded(np, v, inverse, n_u)
    seen = (w >= seen_min) & (vc.min(axis=1) > 0.01) & (plate.min(axis=1) > 0.0)
    ratio = np.zeros_like(vc)
    ratio[seen] = np.log2(np.clip(plate[seen] / np.maximum(vc[seen], 1e-6), lo, hi))
    num = _welded(np, ratio * seen[:, None], inverse, n_u)
    den = _welded(np, seen.astype(np.float64), inverse, n_u)
    iters = smoothing_iterations_for(vw, fw, reach_frac)
    num = _vertex_adjacency_smooth(np, num, fw, iters)
    den = _vertex_adjacency_smooth(np, den, fw, iters)
    local = num / np.maximum(den, 1e-9)[:, None]
    # den = smoothed seen-fraction (0..1): 1 inside the seen region, ~0.5 on
    # its edge, decaying with distance -> confidence 1 up to the edge
    conf = np.clip(2.0 * den, 0.0, 1.0)[:, None]
    log_gain = conf * local + (1.0 - conf) * g0[None, :]
    gains = np.exp2(np.clip(log_gain, np.log2(lo), np.log2(hi)))[inverse]
    return gains, {"seen_vertices": int(seen.sum()), "iterations": int(iters),
                   "gain_p05": np.percentile(gains, 5, axis=0).round(3).tolist(),
                   "gain_p95": np.percentile(gains, 95, axis=0).round(3).tolist()}


def match_vertex_colours(
    vertex_colours_srgb: Any,
    photo_weight: Any,
    plate_samples_srgb: Any,
    *,
    min_vertices: int = 32,
) -> tuple[Any, dict[str, Any]]:
    """Grade the model's vertex colours onto the photo, per channel.

    The model re-synthesises the visible side too, usually a little off in
    exposure and white balance; where the photo and the vertex colours meet,
    that shows as a seam. One median gain per channel in LINEAR light, fitted
    on vertices the camera saw (``photo_weight >= 0.9``), clamped to
    :data:`COLOUR_GAIN_RANGE` and reported. Too few seen vertices -> no gain.
    """
    np = _require_numpy()
    vc = srgb_to_linear(np.asarray(vertex_colours_srgb, dtype=np.float64)[:, :3])
    plate = srgb_to_linear(np.asarray(plate_samples_srgb, dtype=np.float64)[:, :3])
    w = np.asarray(photo_weight, dtype=np.float64)
    seen = w >= 0.9
    gains = [1.0, 1.0, 1.0]
    clamped = [False, False, False]
    applied = False
    if int(seen.sum()) >= int(min_vertices):
        for ch in range(3):
            ok = seen & (vc[:, ch] > 0.02) & (plate[:, ch] > 0.0)
            if int(ok.sum()) < int(min_vertices):
                continue
            g = float(np.median(plate[ok, ch] / vc[ok, ch]))
            lo, hi = COLOUR_GAIN_RANGE
            clamped[ch] = not (lo <= g <= hi)
            gains[ch] = float(min(max(g, lo), hi))
            applied = True
    out = linear_to_srgb(vc * np.asarray(gains)[None, :])
    return out, {"gain": gains, "clamped": clamped, "applied": applied,
                 "seen_vertices": int(seen.sum())}
