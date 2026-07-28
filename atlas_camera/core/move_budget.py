"""Camera move budget — how far can the camera move before invented geometry shows?

Atlas tears the relief mesh at silhouettes deliberately: a tear is the honest
statement that nothing was photographed behind that occluder. The tear only
becomes a *problem* once the camera moves far enough to look into it. This
module measures exactly when that happens, and answers it two ways:

* **envelope** — the per-DOF safe limits (dolly x/y/z, pan, tilt), the abstract
  budget the occlusion graph and the completion pass consume as a stopping
  condition.
* **path check** — per-frame disocclusion along an authored
  :class:`AtlasCameraPath`, the answer an artist actually wants ("does *this*
  move tear, and on which frame").

The measurement is a forward question — *from a candidate camera, how much of
frame has no primary geometry at all* — which is the opposite of
:func:`depth_geometry.primary_camera_validity_mask` (a reverse lookup: given a
target camera's own points, is the primary valid there). So it needs a real
rasterizer, and it gets one: the relief mesh is rasterized into the candidate
camera with a z-buffer, and the uncovered fraction of frame IS the disocclusion.
Because the mesh is already torn correctly at silhouettes, no footprint radius
heuristic and no sky/backdrop fudge enter the answer — what this measures is
what the viewport would show.

Two backends, identical semantics:

``numpy``
    Reference implementation and correctness oracle. Runs anywhere, needed by
    the analytic test in ``tests/test_move_budget.py``, and slow enough that it
    is not the production path at full resolution.
``torch``
    Production path. Same algorithm, device ``cuda`` -> ``mps`` -> ``cpu``,
    matching the tier ``relief_mesh.repair_relief_grid_cuda`` already
    established. The z-buffer is a native ``scatter_reduce_(reduce="amin")`` —
    no custom kernel and no new dependency, since torch is present in every
    ComfyUI venv.

The two are kept as independent implementations rather than a shared abstraction
so that the backend-parametrized tests in ``tests/test_move_budget.py``
genuinely cross-check them instead of exercising one shared code path twice.

Measurement is quantized to whole pixels, and the rounding is deliberately
biased toward counting a boundary pixel as a hole — the reported budget is
never larger than the true one. For a tool whose whole job is telling an artist
how far they can safely go, erring toward "less room than you think" is the
only defensible direction.

Convention (critical, same as everywhere in ``core``): the full 4x4
``extrinsics.camera_view_matrix`` (row-major, world->cam, column-vector points,
translation in column 3), camera looks down -Z. Never the 3x3 rotation.

Known limitation: triangles with any vertex behind the candidate camera are
dropped rather than near-plane clipped. Budget probes are small offsets from a
camera that already framed the scene, so this only bites on probes large enough
to be far outside any usable budget — where the answer is "way past the limit"
either way. Recorded in the sample as ``clipped_faces``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_EPS = 1e-9
# Faces whose screen bbox exceeds this on either axis are rasterized one at a
# time instead of in a padded block, so one degenerate face cannot allocate a
# block grid proportional to the whole frame.
_MAX_BLOCK = 64


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The move budget requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


def _torch_or_none() -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch absent is a supported state
        return None
    return torch


def coverage_backends() -> tuple[str, ...]:
    """Backends available in this environment, cheapest-to-verify first.

    ``numpy`` is always present; ``torch`` appears only when importable. Tests
    parametrize over this so the suite covers whatever the machine can run
    without ever failing on a machine that lacks a GPU stack.
    """
    backends = ["numpy"]
    if _torch_or_none() is not None:
        backends.append("torch")
    return tuple(backends)


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "torch" if _torch_or_none() is not None else "numpy"
    if backend == "torch" and _torch_or_none() is None:
        raise RuntimeError(
            "backend='torch' requested but torch is not importable. Use "
            "backend='numpy' (slower, reference implementation) or install torch."
        )
    if backend not in ("numpy", "torch"):
        raise ValueError(f"Unknown coverage backend {backend!r}; expected numpy/torch/auto.")
    return backend


def _torch_device(torch: Any) -> Any:
    """cuda -> mps -> cpu, mirroring relief_mesh.repair_relief_grid_cuda."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------

def rasterize_coverage(
    vertices: Any,
    faces: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    backend: str = "auto",
) -> tuple[Any, Any]:
    """Rasterize a mesh into a candidate camera; return ``(coverage, zbuffer)``.

    ``coverage`` is ``(H, W)`` bool — True where at least one triangle covered
    the pixel. ``zbuffer`` is ``(H, W)`` float64 forward distance in metres,
    ``inf`` where uncovered. Depth is interpolated perspective-correctly (linear
    in 1/z), so the buffer is usable as a real depth buffer and not only as a
    coverage stencil.
    """
    coverage, zbuf, _ = rasterize_coverage_stats(
        vertices, faces, view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height, backend=backend,
    )
    return coverage, zbuf


def rasterize_coverage_stats(
    vertices: Any,
    faces: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    backend: str = "auto",
) -> tuple[Any, Any, int]:
    """As :func:`rasterize_coverage`, also returning the near-plane clipped
    face count so callers can record it rather than silently discard it.
    """
    np = _require_numpy()
    resolved = _resolve_backend(backend)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        return (np.zeros((height, width), dtype=bool),
                np.full((height, width), np.inf, dtype=np.float64), 0)

    if resolved == "torch":
        return _rasterize_torch(vertices, faces, view_matrix=view_matrix,
                                fx=fx, fy=fy, cx=cx, cy=cy,
                                width=width, height=height)
    return _rasterize_numpy(vertices, faces, view_matrix=view_matrix,
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            width=width, height=height)


def _project_numpy(np: Any, vertices: Any, view_matrix: Any,
                   fx: float, fy: float, cx: float, cy: float):
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam = vertices @ vm[:3, :3].T + vm[:3, 3]
    fwd = -cam[:, 2]                      # -Z forward convention
    in_front = fwd > _EPS
    safe = np.where(in_front, fwd, 1.0)
    sx = cx + fx * cam[:, 0] / safe
    sy = cy - fy * cam[:, 1] / safe
    return sx, sy, fwd, in_front


def _rasterize_numpy(vertices, faces, *, view_matrix, fx, fy, cx, cy, width, height):
    """Reference rasterizer. Bucketed by screen-space bbox size so triangles of
    a given footprint are evaluated as one vectorized block — a per-face Python
    loop over a full-resolution mesh would be unusably slow even for an oracle.
    """
    np = _require_numpy()
    sx, sy, fwd, in_front = _project_numpy(np, vertices, view_matrix, fx, fy, cx, cy)

    tri_ok = in_front[faces].all(axis=1)
    clipped = int((~tri_ok).sum())
    faces = faces[tri_ok]
    zbuf = np.full(height * width, np.inf, dtype=np.float64)
    if faces.size == 0:
        return (np.zeros((height, width), dtype=bool),
                zbuf.reshape(height, width), clipped)

    x = sx[faces]                          # (F, 3)
    y = sy[faces]
    inv_z = 1.0 / fwd[faces]

    x0 = np.ceil(x.min(axis=1)).astype(np.int64)
    x1 = np.floor(x.max(axis=1)).astype(np.int64)
    y0 = np.ceil(y.min(axis=1)).astype(np.int64)
    y1 = np.floor(y.max(axis=1)).astype(np.int64)
    np.clip(x0, 0, width - 1, out=x0)
    np.clip(x1, 0, width - 1, out=x1)
    np.clip(y0, 0, height - 1, out=y0)
    np.clip(y1, 0, height - 1, out=y1)

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    live = (bw > 0) & (bh > 0)
    block = np.maximum(bw, bh)

    for size in np.unique(block[live]):
        sel = live & (block == size)
        if not sel.any():
            continue
        step = int(size)
        if step > _MAX_BLOCK:
            for fi in np.flatnonzero(sel):
                _raster_block_numpy(np, zbuf, x[fi:fi + 1], y[fi:fi + 1],
                                    inv_z[fi:fi + 1], x0[fi:fi + 1], y0[fi:fi + 1],
                                    x1[fi:fi + 1], y1[fi:fi + 1], step, width)
            continue
        _raster_block_numpy(np, zbuf, x[sel], y[sel], inv_z[sel],
                            x0[sel], y0[sel], x1[sel], y1[sel], step, width)

    zbuf = zbuf.reshape(height, width)
    return np.isfinite(zbuf), zbuf, clipped


def _raster_block_numpy(np, zbuf, x, y, inv_z, x0, y0, x1, y1, step, width):
    """Evaluate a (step x step) candidate block per triangle and scatter-min."""
    off = np.arange(step, dtype=np.int64)
    px = x0[:, None, None] + off[None, None, :]        # (F, 1, step)
    py = y0[:, None, None] + off[None, :, None]        # (F, step, 1)
    inside_box = (px <= x1[:, None, None]) & (py <= y1[:, None, None])

    pxf = px.astype(np.float64)
    pyf = py.astype(np.float64)
    ax, bx, ccx = x[:, 0, None, None], x[:, 1, None, None], x[:, 2, None, None]
    ay, by, cyy = y[:, 0, None, None], y[:, 1, None, None], y[:, 2, None, None]

    w0 = (ccx - bx) * (pyf - by) - (cyy - by) * (pxf - bx)
    w1 = (ax - ccx) * (pyf - cyy) - (ay - cyy) * (pxf - ccx)
    w2 = (bx - ax) * (pyf - ay) - (by - ay) * (pxf - ax)
    area = w0 + w1 + w2

    sign = np.where(area >= 0.0, 1.0, -1.0)
    inside = inside_box & (np.abs(area) > _EPS) & \
        ((w0 * sign) >= 0.0) & ((w1 * sign) >= 0.0) & ((w2 * sign) >= 0.0)
    if not inside.any():
        return

    safe_area = np.where(np.abs(area) > _EPS, area, 1.0)
    # Perspective-correct: 1/z is what varies linearly in screen space.
    interp_inv_z = (w0 * inv_z[:, 0, None, None]
                    + w1 * inv_z[:, 1, None, None]
                    + w2 * inv_z[:, 2, None, None]) / safe_area
    depth = np.where(interp_inv_z > _EPS, 1.0 / np.maximum(interp_inv_z, _EPS), np.inf)

    flat_idx = (py * width + px)
    flat_idx = np.broadcast_to(flat_idx, inside.shape)
    np.minimum.at(zbuf, flat_idx[inside], depth[inside])


def _rasterize_torch(vertices, faces, *, view_matrix, fx, fy, cx, cy, width, height):
    """Production rasterizer. Same algorithm as the numpy reference, with the
    z-buffer expressed as torch's native atomic ``scatter_reduce_`` amin.
    """
    np = _require_numpy()
    torch = _torch_or_none()
    device = _torch_device(torch)

    verts = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    tris = torch.as_tensor(faces, dtype=torch.long, device=device)
    vm = torch.as_tensor(np.asarray(view_matrix, dtype=np.float64),
                         dtype=torch.float32, device=device)

    cam = verts @ vm[:3, :3].T + vm[:3, 3]
    fwd = -cam[:, 2]
    in_front = fwd > _EPS
    safe = torch.where(in_front, fwd, torch.ones_like(fwd))
    sx = cx + fx * cam[:, 0] / safe
    sy = cy - fy * cam[:, 1] / safe

    tri_ok = in_front[tris].all(dim=1)
    clipped = int((~tri_ok).sum().item())
    tris = tris[tri_ok]
    zbuf = torch.full((height * width,), float("inf"),
                      dtype=torch.float32, device=device)
    if tris.numel() == 0:
        out = zbuf.reshape(height, width).double().cpu().numpy()
        return np.zeros((height, width), dtype=bool), out, clipped

    x = sx[tris]
    y = sy[tris]
    inv_z = 1.0 / fwd[tris]

    x0 = torch.ceil(x.min(dim=1).values).long().clamp(0, width - 1)
    x1 = torch.floor(x.max(dim=1).values).long().clamp(0, width - 1)
    y0 = torch.ceil(y.min(dim=1).values).long().clamp(0, height - 1)
    y1 = torch.floor(y.max(dim=1).values).long().clamp(0, height - 1)

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    live = (bw > 0) & (bh > 0)
    block = torch.maximum(bw, bh).clamp(max=_MAX_BLOCK)

    for size in torch.unique(block[live]).tolist():
        sel = live & (block == size)
        if not bool(sel.any()):
            continue
        _raster_block_torch(torch, zbuf, x[sel], y[sel], inv_z[sel],
                            x0[sel], y0[sel], x1[sel], y1[sel], int(size), width)

    zbuf_np = zbuf.reshape(height, width).double().cpu().numpy()
    return np.isfinite(zbuf_np), zbuf_np, clipped


def _raster_block_torch(torch, zbuf, x, y, inv_z, x0, y0, x1, y1, step, width):
    off = torch.arange(step, device=x.device, dtype=torch.long)
    px = x0[:, None, None] + off[None, None, :]
    py = y0[:, None, None] + off[None, :, None]
    inside_box = (px <= x1[:, None, None]) & (py <= y1[:, None, None])

    pxf = px.to(x.dtype)
    pyf = py.to(x.dtype)
    ax, bx, ccx = x[:, 0, None, None], x[:, 1, None, None], x[:, 2, None, None]
    ay, by, cyy = y[:, 0, None, None], y[:, 1, None, None], y[:, 2, None, None]

    w0 = (ccx - bx) * (pyf - by) - (cyy - by) * (pxf - bx)
    w1 = (ax - ccx) * (pyf - cyy) - (ay - cyy) * (pxf - ccx)
    w2 = (bx - ax) * (pyf - ay) - (by - ay) * (pxf - ax)
    area = w0 + w1 + w2

    sign = torch.where(area >= 0, torch.ones_like(area), -torch.ones_like(area))
    inside = inside_box & (area.abs() > _EPS) & \
        ((w0 * sign) >= 0) & ((w1 * sign) >= 0) & ((w2 * sign) >= 0)
    if not bool(inside.any()):
        return

    safe_area = torch.where(area.abs() > _EPS, area, torch.ones_like(area))
    interp_inv_z = (w0 * inv_z[:, 0, None, None]
                    + w1 * inv_z[:, 1, None, None]
                    + w2 * inv_z[:, 2, None, None]) / safe_area
    depth = torch.where(interp_inv_z > _EPS,
                        1.0 / interp_inv_z.clamp(min=_EPS),
                        torch.full_like(interp_inv_z, float("inf")))

    flat_idx = (py * width + px).expand_as(inside)
    zbuf.scatter_reduce_(0, flat_idx[inside], depth[inside],
                         reduce="amin", include_self=True)


def disocclusion_fraction(
    vertices: Any,
    faces: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    ignore_mask: Any = None,
    backend: str = "auto",
) -> float:
    """Fraction of the candidate frame with no primary geometry behind it.

    ``ignore_mask`` (H, W) bool, optional — pixels excluded from BOTH numerator
    and denominator. Its intended use is a sky/backdrop region the artist has
    already accepted will be handled by a matte rather than by geometry;
    without it, sky reads as a permanent hole and every budget collapses to
    zero. Left None the measurement is raw and pessimistic, which is the honest
    default.
    """
    np = _require_numpy()
    coverage, _ = rasterize_coverage(
        vertices, faces, view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height, backend=backend,
    )
    holes = ~coverage
    if ignore_mask is not None:
        keep = ~np.asarray(ignore_mask, dtype=bool)
        total = int(keep.sum())
        if total == 0:
            return 0.0
        return float((holes & keep).sum()) / float(total)
    return float(holes.sum()) / float(holes.size)


def tear_disocclusion_fraction(
    covered: tuple[Any, Any],
    sealed: tuple[Any, Any],
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    ignore_mask: Any = None,
    backend: str = "auto",
) -> tuple[float, Any, int]:
    """Fraction of frame where a silhouette tear is open. ``(fraction, mask, clipped)``.

    ``covered`` is every surface that actually renders — the torn relief mesh,
    plus ground/object proxies, plus whatever the completion pass has filled in.
    ``sealed`` is the same relief surface WITHOUT its tears: the envelope of
    everything the primary camera's depth map described.

    A pixel is disoccluded when the sealed surface covers it and the real
    geometry does not — precisely "the photographed surface continues here, but
    the tear took it away, and you are now looking through the gap".

    Why not simply count uncovered pixels: a real Atlas scene always carries a
    far backdrop cyclorama, so *every* pixel is covered by something and the
    naive measure reports zero disocclusion no matter how badly the scene tears.
    The backdrop showing through a tear is an artifact — wrong depth, wrong
    parallax, wrong texture — not a fill, and this measure is blind to it by
    construction because the backdrop is not part of ``sealed``.

    The same construction gives the completion pass its success criterion for
    free: geometry that fills a tear joins ``covered``, the disoccluded set
    shrinks, and the budget grows.
    """
    np = _require_numpy()
    kwargs = dict(view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
                  width=width, height=height, backend=backend)
    covered_mask, _, clipped = rasterize_coverage_stats(covered[0], covered[1], **kwargs)
    sealed_mask, _, _ = rasterize_coverage_stats(sealed[0], sealed[1], **kwargs)

    torn = sealed_mask & ~covered_mask
    if ignore_mask is not None:
        keep = ~np.asarray(ignore_mask, dtype=bool)
        torn = torn & keep
        total = int(keep.sum())
    else:
        total = int(torn.size)
    fraction = float(torn.sum()) / float(total) if total else 0.0
    return fraction, torn, clipped


# --------------------------------------------------------------------------
# Candidate cameras
# --------------------------------------------------------------------------

def _rodrigues(np: Any, axis: Any, angle_rad: float) -> Any:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), _EPS)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def offset_view_matrix(
    view_matrix: Any,
    *,
    dolly_x: float = 0.0,
    dolly_y: float = 0.0,
    dolly_z: float = 0.0,
    pan_deg: float = 0.0,
    tilt_deg: float = 0.0,
) -> Any:
    """A candidate camera offset from ``view_matrix`` in its own frame.

    Dollies translate along the camera's right / up / forward axes (``dolly_z``
    positive moves the camera FORWARD, i.e. along -Z in camera space).

    Pan is a world-side yaw about world +Y — a tripod pan, which is what a VFX
    artist means by the word, and which keeps the horizon level. Tilt is a
    camera-frame pitch. Per the standing transpose rule this is expressed as a
    LEFT-multiply for the world-side rotation and a RIGHT-multiply for the
    camera-frame one, on the cam_to_world block: ``R' = Ry(pan) @ R_cw @ Rx(tilt)``.
    Getting these backwards silently inverts pitch, which is exactly the class
    of bug the rule exists to prevent.
    """
    np = _require_numpy()
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    R_cw = cam_to_world[:3, :3]
    eye = cam_to_world[:3, 3].copy()

    right, up, back = R_cw[:, 0], R_cw[:, 1], R_cw[:, 2]
    eye = eye + right * dolly_x + up * dolly_y - back * dolly_z

    R_new = R_cw
    if abs(pan_deg) > _EPS:
        R_new = _rodrigues(np, (0.0, 1.0, 0.0), math.radians(pan_deg)) @ R_new
    if abs(tilt_deg) > _EPS:
        R_new = R_new @ _rodrigues(np, (1.0, 0.0, 0.0), math.radians(tilt_deg))

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_new.T
    out[:3, 3] = -R_new.T @ eye
    return out


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------

_AXES = ("dolly_x", "dolly_y", "dolly_z", "pan", "tilt")
_ANGULAR_AXES = ("pan", "tilt")
# The cyclorama emitted by depth_geometry.build_backdrop_primitive. It shares
# PROXY_ROLE with every other derived primitive, so it can only be identified by
# name — and it must be identified, because a surface that covers the entire
# frustum would otherwise report every tear as filled.
_BACKDROP_NAMES = ("projection_backdrop",)


@dataclass(slots=True)
class MoveBudgetSample:
    """One probe: an offset along one axis and the disocclusion it produced."""

    axis: str
    offset: float          # metres for dollies, degrees for pan/tilt
    fraction: float
    clipped_faces: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "offset": round(float(self.offset), 6),
                "fraction": round(float(self.fraction), 6),
                "clipped_faces": int(self.clipped_faces)}


@dataclass(slots=True)
class PathFrameSample:
    """One frame of an authored camera path and its disocclusion."""

    frame_index: int
    fraction: float
    within_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": int(self.frame_index),
                "fraction": round(float(self.fraction), 6),
                "within_budget": bool(self.within_budget)}


@dataclass(slots=True)
class AtlasMoveBudget:
    """The safe camera envelope, plus an optional verdict on an authored move.

    Every limit is the largest offset along that axis whose disocclusion stays
    at or below ``threshold``. Symmetric axes report the SMALLER of the two
    directions — a budget an artist can move either way without checking which.
    ``saturated`` lists axes that never crossed the threshold within the search
    cap, where the true limit is larger than reported.
    """

    dolly_x_m: float = 0.0
    dolly_y_m: float = 0.0
    dolly_z_m: float = 0.0
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    threshold: float = 0.02
    method: str = "relief_raster"
    backend: str = "numpy"
    samples: list[MoveBudgetSample] = field(default_factory=list)
    saturated: list[str] = field(default_factory=list)
    geometry_sources: list[str] = field(default_factory=list)
    # Disocclusion already visible from the recovered camera. Not a
    # move-budget problem, but a real projection-coverage signal.
    baseline_fraction: float = 0.0
    path_frames: list[PathFrameSample] = field(default_factory=list)
    path_worst_fraction: float | None = None
    path_worst_frame: int | None = None
    path_within_budget: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dolly_x_m": round(float(self.dolly_x_m), 5),
            "dolly_y_m": round(float(self.dolly_y_m), 5),
            "dolly_z_m": round(float(self.dolly_z_m), 5),
            "pan_deg": round(float(self.pan_deg), 4),
            "tilt_deg": round(float(self.tilt_deg), 4),
            "threshold": float(self.threshold),
            "method": self.method,
            "backend": self.backend,
            "samples": [s.to_dict() for s in self.samples],
            "saturated": list(self.saturated),
            "geometry_sources": list(self.geometry_sources),
            "baseline_fraction": round(float(self.baseline_fraction), 6),
            "notes": list(self.notes),
        }
        if self.path_frames:
            out["path"] = {
                "frames": [f.to_dict() for f in self.path_frames],
                "worst_fraction": round(float(self.path_worst_fraction or 0.0), 6),
                "worst_frame": self.path_worst_frame,
                "within_budget": self.path_within_budget,
            }
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "AtlasMoveBudget":
        if not isinstance(data, dict):
            return cls()
        path = data.get("path") or {}
        return cls(
            dolly_x_m=float(data.get("dolly_x_m", 0.0)),
            dolly_y_m=float(data.get("dolly_y_m", 0.0)),
            dolly_z_m=float(data.get("dolly_z_m", 0.0)),
            pan_deg=float(data.get("pan_deg", 0.0)),
            tilt_deg=float(data.get("tilt_deg", 0.0)),
            threshold=float(data.get("threshold", 0.02)),
            method=str(data.get("method", "relief_raster")),
            backend=str(data.get("backend", "numpy")),
            samples=[MoveBudgetSample(**s) for s in data.get("samples", [])],
            saturated=list(data.get("saturated", [])),
            geometry_sources=list(data.get("geometry_sources", [])),
            baseline_fraction=float(data.get("baseline_fraction", 0.0)),
            path_frames=[PathFrameSample(**f) for f in path.get("frames", [])],
            path_worst_fraction=path.get("worst_fraction"),
            path_worst_frame=path.get("worst_frame"),
            path_within_budget=path.get("within_budget"),
        )

    def describe(self) -> str:
        """Artist-facing one-screen summary."""
        lines = [
            f"Safe camera envelope (disocclusion <= {self.threshold:.1%}):",
            f"  dolly  x +/-{self.dolly_x_m:.3f} m   y +/-{self.dolly_y_m:.3f} m"
            f"   z +/-{self.dolly_z_m:.3f} m",
            f"  rotate pan +/-{self.pan_deg:.1f} deg   tilt +/-{self.tilt_deg:.1f} deg",
        ]
        if self.geometry_sources:
            lines.append(f"  measured against: {', '.join(self.geometry_sources)}")
        if self.baseline_fraction > 1e-4:
            lines.append(
                f"  {self.baseline_fraction:.1%} of frame already tears at the recovered "
                "camera (projection coverage, not a move limit) — excluded from the budget.")
        if self.saturated:
            lines.append(f"  (unbounded within search cap: {', '.join(self.saturated)})")
        if self.path_within_budget is not None:
            verdict = "WITHIN budget" if self.path_within_budget else "EXCEEDS budget"
            lines.append(
                f"Authored path: {verdict} — worst frame {self.path_worst_frame} "
                f"at {float(self.path_worst_fraction or 0.0):.2%} disocclusion."
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _axis_kwargs(axis: str, offset: float) -> dict[str, float]:
    return {
        "dolly_x": {"dolly_x": offset},
        "dolly_y": {"dolly_y": offset},
        "dolly_z": {"dolly_z": offset},
        "pan": {"pan_deg": offset},
        "tilt": {"tilt_deg": offset},
    }[axis]


def _search_limit(probe, *, threshold: float, cap: float, bisect_steps: int,
                  seed: float) -> tuple[float, bool]:
    """Largest offset whose disocclusion stays <= threshold. Returns
    ``(limit, saturated)``; saturated means the cap was reached without ever
    exceeding the threshold, so the true limit is larger than reported."""
    if probe(cap) <= threshold:
        return cap, True
    lo, hi = 0.0, min(seed, cap)
    while hi < cap and probe(hi) <= threshold:
        lo, hi = hi, min(hi * 2.0, cap)
    for _ in range(bisect_steps):
        mid = 0.5 * (lo + hi)
        if probe(mid) <= threshold:
            lo = mid
        else:
            hi = mid
    return lo, False


def _unsealed_warning(torn: Any, sealed: Any, *, image_width: int,
                      image_height: int) -> str | None:
    """Detect a sealing pass that silently did nothing on a mesh that HAS tears.

    ``repair_relief_mesh_grid_cuda`` recovers its sampling lattice from the mesh
    UVs and returns ``(0, 0)`` when it cannot — indistinguishable from outside
    from a mesh that had nothing to close. The two cases matter very
    differently: an unsealed mesh makes ``sealed`` equal ``covered``, every
    probe reads zero disocclusion, and the budget comes back unbounded. That is
    the one failure direction that must never be silent.

    The test is specifically for INTERIOR boundary. A relief mesh built from a
    frame-filling depth map is a complete lattice whose only open boundary is
    the plate perimeter — nothing to seal, and warning there would fire on
    essentially every real scene and train the artist to ignore the message.
    Same distinction ``mesh_repair`` draws, and the same helper draws it.
    """
    np = _require_numpy()
    if len(getattr(sealed, "vertices", ())) != len(getattr(torn, "vertices", ())):
        return None                     # sealing added geometry — it worked
    try:
        from atlas_camera.core.mesh_repair import (
            _perimeter_loops, boundary_edges, walk_loops,
        )
        faces = np.asarray(torn.faces, dtype=np.int64)
        edges = boundary_edges(faces)
        if len(edges) == 0:
            return None                 # watertight; nothing to seal
        loops = walk_loops(edges, faces)
        perimeter = _perimeter_loops(loops, getattr(torn, "uvs", None),
                                     image_width, image_height)
        interior = [i for i in range(len(loops)) if i not in perimeter]
    except Exception:
        return None
    if not interior:
        return None                     # only the plate perimeter is open
    return (
        f"sealing closed nothing despite {len(interior)} interior boundary "
        "loop(s) — the UV lattice could not be recovered, so disocclusion is "
        "being measured against the torn surface itself and this budget is NOT "
        "trustworthy. Pass sealed_mesh= explicitly."
    )


def seal_relief_mesh(mesh: Any, solve: Any) -> Any:
    """A copy of ``mesh`` with its silhouette tears closed.

    Reuses ``mesh_repair.repair_relief_mesh_grid_cuda``, which already recovers
    the regular sampling lattice from the mesh's own UVs and materializes the
    missing cells by back-projecting along each cell's own camera ray — exactly
    the surface the depth map described before tearing removed it. Run with the
    thresholds wide open, since here the goal is the sealed envelope rather
    than a conservative repair.

    The input mesh is never mutated: the live projection mesh keeps its
    deliberate tears, per the standing rule that only AtlasRetopologizeLayer
    touches live projection geometry.
    """
    import copy

    from atlas_camera.core.mesh_repair import repair_relief_mesh_grid_cuda
    from atlas_camera.core.relief_mesh import ReliefMeshCameraSpec

    sealed = copy.deepcopy(mesh)
    spec = ReliefMeshCameraSpec.from_solve(solve)
    repair_relief_mesh_grid_cuda(
        sealed,
        view_matrix=spec.view_matrix, fx=spec.fx, fy=spec.fy,
        cx=spec.cx, cy=spec.cy,
        image_width=int(solve.camera.intrinsics.image_width or 1024),
        image_height=int(solve.camera.intrinsics.image_height or 1024),
        fill_holes=True, fill_sawteeth=True, cap_enclosed=True,
        depth_edge_rel=1e9, max_edge_factor=1e9,
    )
    return sealed


def estimate_move_budget(
    solve: Any,
    *,
    threshold: float = 0.02,
    backend: str = "auto",
    ignore_mask: Any = None,
    bisect_steps: int = 8,
    max_dolly_m: float | None = None,
    max_angle_deg: float = 30.0,
    camera_path: Any = None,
    mesh: Any = None,
    sealed_mesh: Any = None,
    axes: tuple[str, ...] = _AXES,
) -> AtlasMoveBudget:
    """Measure the safe camera envelope for ``solve``'s relief mesh.

    ``max_dolly_m`` defaults to a quarter of the median scene distance — past
    that a single-image projection has stopped being a projection regardless of
    what the disocclusion number says, so searching further wastes probes.

    ``camera_path`` (an :class:`AtlasCameraPath`), when given, is additionally
    evaluated frame by frame and its verdict recorded. That is the question an
    artist actually asks; the envelope is what the completion pass consumes as
    a stopping condition.

    Disocclusion is measured as sealed-minus-covered (see
    :func:`tear_disocclusion_fraction`), so the answer is unaffected by the
    backdrop and grows as the completion pass fills tears. ``mesh`` and
    ``sealed_mesh`` override the two geometry sets directly, which is how the
    analytic tests pin the measurement without depending on scene assembly.
    """
    np = _require_numpy()
    from atlas_camera.core.relief_mesh import ReliefMeshCameraSpec

    from atlas_camera.core.primitive_mesh import collect_scene_triangles

    seal_warning: str | None = None
    if mesh is not None:
        covered = (np.asarray(mesh.vertices, dtype=np.float64),
                   np.asarray(mesh.faces, dtype=np.int64))
        geometry_sources = ["relief_mesh (explicit)"]
        torn_mesh = mesh
    else:
        # The backdrop is deliberately excluded: it covers everything, so
        # including it would make every tear read as filled.
        cv, cf, geometry_sources = collect_scene_triangles(
            solve, exclude_names=_BACKDROP_NAMES)
        covered = (cv, cf)
        from atlas_camera.core.relief_mesh import _relief_mesh_from_solve
        torn_mesh = _relief_mesh_from_solve(solve)

    if len(covered[1]) == 0:
        raise ValueError(
            "estimate_move_budget found no geometry on this solve. Run "
            "AtlasDeriveReliefMesh / AtlasDeriveProjectionGeometry (or "
            "AtlasInput) first, or pass mesh= explicitly."
        )

    if sealed_mesh is not None:
        sealed = (np.asarray(sealed_mesh.vertices, dtype=np.float64),
                  np.asarray(sealed_mesh.faces, dtype=np.int64))
        geometry_sources.append("sealed (explicit)")
    elif torn_mesh is not None:
        sealed_built = seal_relief_mesh(torn_mesh, solve)
        sealed = (np.asarray(sealed_built.vertices, dtype=np.float64),
                  np.asarray(sealed_built.faces, dtype=np.int64))
        geometry_sources.append("sealed (grid hole-fill)")
        seal_warning = _unsealed_warning(
            torn_mesh, sealed_built,
            image_width=int(solve.camera.intrinsics.image_width or 1024),
            image_height=int(solve.camera.intrinsics.image_height or 1024))
    else:
        raise ValueError(
            "estimate_move_budget needs a relief mesh to seal — this solve has "
            "proxy primitives but no relief mesh. Run AtlasDeriveReliefMesh, or "
            "pass sealed_mesh= explicitly."
        )

    spec = ReliefMeshCameraSpec.from_solve(solve)
    width = int(solve.camera.intrinsics.image_width or 1024)
    height = int(solve.camera.intrinsics.image_height or 1024)
    base_view = np.asarray(spec.view_matrix, dtype=np.float64)
    resolved = _resolve_backend(backend)

    if max_dolly_m is None:
        cam = covered[0] @ base_view[:3, :3].T + base_view[:3, 3]
        fwd = -cam[:, 2]
        fwd = fwd[np.isfinite(fwd) & (fwd > _EPS)]
        median_depth = float(np.median(fwd)) if fwd.size else 10.0
        max_dolly_m = max(0.05, 0.25 * median_depth)

    samples: list[MoveBudgetSample] = []

    def raw(view: Any) -> tuple[float, int]:
        fraction, _, clipped = tear_disocclusion_fraction(
            covered, sealed, view_matrix=view, fx=spec.fx, fy=spec.fy,
            cx=spec.cx, cy=spec.cy, width=width, height=height,
            ignore_mask=ignore_mask, backend=resolved,
        )
        return fraction, clipped

    # The sealed envelope always covers more than the torn mesh — that is what
    # makes it a sealed envelope. So a naive reading reports disocclusion even
    # at the RECOVERED camera, where by definition there is none: you are
    # looking at the photograph. On a real 4K plate that baseline measured 6%
    # against a 2% threshold, which collapsed every axis of the budget to zero.
    #
    # The budget's question is what the MOVE opens, so the source view is the
    # zero point and every probe is reported relative to it. Tearing already
    # visible from the recovered camera is a projection-coverage problem, not a
    # camera-move problem, and it is reported separately as `baseline_fraction`
    # rather than silently folded in.
    baseline, _ = raw(base_view)

    def probe(axis: str, offset: float) -> float:
        view = offset_view_matrix(base_view, **_axis_kwargs(axis, offset))
        fraction, clipped = raw(view)
        opened = max(0.0, fraction - baseline)
        samples.append(MoveBudgetSample(axis=axis, offset=offset, fraction=opened,
                                        clipped_faces=clipped))
        return opened

    budget = AtlasMoveBudget(threshold=float(threshold), backend=resolved,
                             geometry_sources=geometry_sources,
                             baseline_fraction=baseline)
    for axis in axes:
        angular = axis in _ANGULAR_AXES
        cap = float(max_angle_deg if angular else max_dolly_m)
        seed = cap * 0.125
        pos, sat_pos = _search_limit(lambda o: probe(axis, o), threshold=threshold,
                                     cap=cap, bisect_steps=bisect_steps, seed=seed)
        neg, sat_neg = _search_limit(lambda o: probe(axis, -o), threshold=threshold,
                                     cap=cap, bisect_steps=bisect_steps, seed=seed)
        limit = min(pos, neg)
        if sat_pos and sat_neg:
            budget.saturated.append(axis)
        setattr(budget, f"{axis}_deg" if angular else f"{axis}_m", limit)

    budget.samples = samples
    if seal_warning:
        budget.notes.append(seal_warning)
    if any(s.clipped_faces for s in samples):
        worst = max(s.clipped_faces for s in samples)
        budget.notes.append(
            f"{worst} faces fell behind the candidate camera on the widest probe "
            "and were dropped rather than near-plane clipped."
        )

    if camera_path is not None:
        _evaluate_path(budget, camera_path, covered=covered, sealed=sealed, spec=spec,
                       width=width, height=height, ignore_mask=ignore_mask,
                       backend=resolved, threshold=threshold)
    return budget


def _evaluate_path(budget: AtlasMoveBudget, camera_path: Any, *, covered, sealed,
                   spec, width, height, ignore_mask, backend, threshold) -> None:
    np = _require_numpy()
    from atlas_camera.core.camera_path import sample_camera_path

    extrinsics = sample_camera_path(camera_path)
    frames: list[PathFrameSample] = []
    for index, extr in enumerate(extrinsics):
        fraction, _, _ = tear_disocclusion_fraction(
            covered, sealed,
            view_matrix=np.asarray(extr.camera_view_matrix, dtype=np.float64),
            fx=spec.fx, fy=spec.fy, cx=spec.cx, cy=spec.cy,
            width=width, height=height, ignore_mask=ignore_mask, backend=backend,
        )
        frames.append(PathFrameSample(frame_index=index, fraction=fraction,
                                      within_budget=fraction <= threshold))
    budget.path_frames = frames
    if frames:
        worst = max(frames, key=lambda f: f.fraction)
        budget.path_worst_fraction = worst.fraction
        budget.path_worst_frame = worst.frame_index
        budget.path_within_budget = all(f.within_budget for f in frames)
