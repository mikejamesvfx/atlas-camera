"""A small differentiable Gaussian-splat rasterizer, and the hole training loop.

WHY THIS EXISTS RATHER THAN gsplat. gsplat ships a pure-Python wheel and
JIT-compiles its CUDA kernels on first use; on this stack that compile fails in
a torch header, because Windows' RPC headers ``#define small char`` and torch's
``StreamSegmentSize(cudaStream_t, bool small, size_t)`` then expands to
``bool char``. That is a header collision, not a Blackwell or CUDA-version
problem (nvcc targeted ``sm_120`` correctly), but fixing it means editing
site-packages. Phase 1 only ever rasterizes ONE hole — a small ROI and a few
thousand gaussians — so a plain-torch implementation is fast enough, adds no
dependency, and can be read.

IT HAS NO ORACLE BEHIND IT. Everything downstream of this file inherits its
correctness, so the projection Jacobian and the compositing order are written
out explicitly below rather than folded into a clever tensor expression, and
``tests/test_hole_splat_train.py`` checks them against hand-reasoned geometry.
Finite differences alone would NOT catch a wrong projection — autograd and the
difference quotient agree on whatever function was written — so the geometry
tests carry correctness and the FD tests carry graph integrity.

CONVENTION. Identical to ``core.depth_geometry``: the full 4x4 world->camera
``camera_view_matrix``, +X right, +Y up, **-Z forward**, image origin top-left,
so forward distance is ``d = -z_cam`` and

    u = fx * x / d + cx        v = -fy * y / d + cy

Layering: torch lives here, never in ``core``. Contracts, seeds and metrics are
``core.hole_splat``; this module consumes a ``HoleSeed`` and returns arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas_camera.core.hole_splat import SPLAT_SOURCE, HoleSeed

#: Isotropic blur added to the 2D covariance diagonal, in px^2. Without it a
#: gaussian smaller than a pixel has a near-singular 2D covariance and its
#: gradient explodes; this is the standard EWA low-pass.
DILATION_PX2 = 0.3

#: Opacity below this contributes nothing visible and is skipped when pruning.
MIN_OPACITY = 1.0 / 255.0


def _require_torch() -> Any:
    """Import torch lazily with an informative error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Hole-splat training requires torch. Install with:\n"
            "    pip install -e .[splat]") from exc
    return torch


@dataclass(frozen=True, slots=True)
class TrainView:
    """One supervision view: an image, where its loss applies, and its camera.

    ``image`` is float RGB in [0, 1] — read it from the float plate via
    ``plate.oiio_io.read_plate``, never from a base64 preview, or training
    supervises on display-referred 8-bit-origin pixels.
    """

    image: Any
    loss_mask: Any
    view_matrix: Any
    fx: float
    fy: float
    cx: float
    cy: float
    weight: float = 1.0
    name: str = "view"


@dataclass(frozen=True, slots=True)
class SplatRender:
    rgb: Any
    alpha: Any
    depth: Any


@dataclass(frozen=True, slots=True)
class TrainedSplats:
    means: Any
    quats: Any
    scales: Any
    opacities: Any
    colors: Any
    report: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(len(self.means))


def _quat_to_rotation(torch: Any, quats: Any) -> Any:
    """(N,4) wxyz -> (N,3,3). Normalised here so the optimiser may drift."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def render_gaussians(
    means: Any,
    quats: Any,
    scales: Any,
    opacities: Any,
    colors: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near_m: float = 1e-3,
    pixel_chunk: int = 65536,
) -> SplatRender:
    """Alpha-composite anisotropic 3D gaussians into one camera.

    Differentiable in ``means``, ``quats``, ``scales``, ``opacities`` and
    ``colors``. Returns ``(rgb HxWx3, alpha HxW, depth HxW)`` where ``depth`` is
    the alpha-weighted expected forward distance in metres — that is what
    ``core.tear_metrics.score_tears`` wants for ``render_depth``, and without it
    a closed coverage gap reads as a missed edge.
    """

    torch = _require_torch()
    device = means.device
    view = view_matrix.to(device=device, dtype=means.dtype)
    rot_wc = view[:3, :3]
    trans_wc = view[:3, 3]

    # --- world -> camera -------------------------------------------------
    pts_cam = means @ rot_wc.T + trans_wc
    depth = -pts_cam[:, 2]
    visible = depth > float(near_m)
    if not bool(visible.any()):
        zeros = torch.zeros((height, width), device=device, dtype=means.dtype)
        return SplatRender(
            rgb=torch.zeros((height, width, 3), device=device, dtype=means.dtype),
            alpha=zeros, depth=zeros.clone())

    pts_cam = pts_cam[visible]
    depth = depth[visible]
    opac = opacities[visible]
    col = colors[visible]

    x, y = pts_cam[:, 0], pts_cam[:, 1]
    u = fx * x / depth + cx
    v = -fy * y / depth + cy

    # --- 3D covariance, then its image-plane projection ------------------
    rot = _quat_to_rotation(torch, quats[visible])
    s = scales[visible]
    # Sigma = R diag(s^2) R^T, built as (R S)(R S)^T so autograd sees one product.
    rs = rot * s[:, None, :]
    cov3d = rs @ rs.transpose(1, 2)
    cov_cam = rot_wc @ cov3d @ rot_wc.T

    # J = d(u, v)/d(x, y, z) at the mean, for d = -z:
    #     du/dx = fx/d      du/dz = fx*x/d^2
    #     dv/dy = -fy/d     dv/dz = -fy*y/d^2
    zero = torch.zeros_like(depth)
    inv_d = 1.0 / depth
    inv_d2 = inv_d * inv_d
    row_u = torch.stack([fx * inv_d, zero, fx * x * inv_d2], dim=-1)
    row_v = torch.stack([zero, -fy * inv_d, -fy * y * inv_d2], dim=-1)
    jac = torch.stack([row_u, row_v], dim=-2)

    cov2d = jac @ cov_cam @ jac.transpose(1, 2)
    eye2 = torch.eye(2, device=device, dtype=means.dtype)
    cov2d = cov2d + DILATION_PX2 * eye2

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = det.clamp_min(1e-12)
    # Explicit 2x2 inverse — torch.inverse on a tiny batch is slower and its
    # backward is noisier than the closed form.
    inv00 = cov2d[:, 1, 1] / det
    inv11 = cov2d[:, 0, 0] / det
    inv01 = -cov2d[:, 0, 1] / det

    # --- front-to-back compositing --------------------------------------
    order = torch.argsort(depth)
    u, v, depth = u[order], v[order], depth[order]
    opac, col = opac[order], col[order]
    inv00, inv11, inv01 = inv00[order], inv11[order], inv01[order]

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=means.dtype),
        torch.arange(width, device=device, dtype=means.dtype),
        indexing="ij",
    )
    px = xs.reshape(-1)
    py = ys.reshape(-1)

    rgb_flat = torch.zeros((height * width, 3), device=device, dtype=means.dtype)
    alpha_flat = torch.zeros(height * width, device=device, dtype=means.dtype)
    depth_flat = torch.zeros(height * width, device=device, dtype=means.dtype)

    chunk = max(1, int(pixel_chunk))
    for start in range(0, height * width, chunk):
        stop = min(start + chunk, height * width)
        dx = px[start:stop, None] - u[None, :]
        dy = py[start:stop, None] - v[None, :]
        power = -0.5 * (inv00[None, :] * dx * dx
                        + inv11[None, :] * dy * dy
                        + 2.0 * inv01[None, :] * dx * dy)
        # Clamp before exp: a far-tail gaussian otherwise underflows to a hard
        # zero and takes its gradient with it.
        gauss = torch.exp(power.clamp(min=-30.0, max=0.0))
        a = (opac[None, :] * gauss).clamp(0.0, 0.999)

        # Exclusive cumulative transmittance along the depth-sorted axis.
        one_minus = 1.0 - a
        trans = torch.cumprod(one_minus, dim=1) / one_minus.clamp_min(1e-12)
        weight = a * trans

        rgb_flat[start:stop] = weight @ col
        alpha_flat[start:stop] = weight.sum(dim=1)
        depth_flat[start:stop] = weight @ depth

    total = alpha_flat.clamp_min(1e-8)
    return SplatRender(
        rgb=rgb_flat.reshape(height, width, 3),
        alpha=alpha_flat.reshape(height, width),
        # Expected depth is a weighted MEAN, so normalise by coverage rather
        # than reporting a darker pixel as a nearer one.
        depth=(depth_flat / total).reshape(height, width),
    )


def train_hole_splats(
    seed: HoleSeed,
    views: list[TrainView],
    *,
    iters: int = 300,
    lr: float = 0.01,
    device: str | None = None,
    isotropic_init_px: float | None = None,
    seed_value: int = 0,
    log_every: int = 0,
    pixel_chunk: int = 16384,
) -> TrainedSplats:
    """Fit gaussians to the masked hole region across the supervision views.

    Only ``TrainView.loss_mask`` pixels contribute gradient — nothing outside
    the hole is optimised, which is what keeps the mesh's measured pixels out
    of the objective entirely rather than merely weighted down.

    MEMORY. This rasterizer is dense: it evaluates every gaussian against every
    pixel of a chunk, so peak memory grows as ``pixel_chunk x n_gaussians`` and
    the autograd graph retains one such block per chunk for the backward pass.
    A 46k-gaussian seed at full frame will exhaust a 32 GB card. Keep the seed
    sparse (``pixel_stride``/``layers``) and the raster small — a bounded hole
    is the regime this was written for. Tile binning is the fix if phase 2
    needs whole scenes; it is deliberately not here.
    """

    torch = _require_torch()
    if not views:
        raise ValueError("training needs at least one view")
    if int(iters) < 1:
        raise ValueError("iters must be >= 1")

    from atlas_camera.inference._common import resolve_device

    dev = resolve_device(device, torch)
    torch.manual_seed(int(seed_value))
    dtype = torch.float32

    pts = torch.as_tensor(seed.points_world, dtype=dtype, device=dev)
    n = pts.shape[0]
    if not n:
        raise ValueError("seed has no points")

    # Isotropic init sized to the slab step: a gaussian should span roughly the
    # gap between depth layers, so the slab is covered rather than dotted.
    init_scale = float(isotropic_init_px) if isotropic_init_px else max(
        float(seed.scale_m) * 0.5, 1e-4)

    means = pts.clone().requires_grad_(True)
    quats = torch.zeros((n, 4), dtype=dtype, device=dev)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    log_scales = torch.full((n, 3), float(torch.log(torch.tensor(init_scale))),
                            dtype=dtype, device=dev, requires_grad=True)
    logit_opacity = torch.full((n,), 0.0, dtype=dtype, device=dev,
                               requires_grad=True)
    colors = torch.as_tensor(seed.colors, dtype=dtype, device=dev).clone()
    colors.requires_grad_(True)

    params = [means, quats, log_scales, logit_opacity, colors]
    opt = torch.optim.Adam(params, lr=float(lr))

    prepared = []
    for view in views:
        image = torch.as_tensor(view.image, dtype=dtype, device=dev)
        mask = torch.as_tensor(view.loss_mask, dtype=torch.bool, device=dev)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"view {view.name!r}: image {tuple(image.shape[:2])} and "
                f"loss_mask {tuple(mask.shape)} disagree")
        if not bool(mask.any()):
            raise ValueError(f"view {view.name!r}: loss_mask is empty")
        prepared.append((
            view,
            image,
            mask,
            torch.as_tensor(view.view_matrix, dtype=dtype, device=dev),
        ))

    history: list[float] = []
    for step in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=dtype, device=dev)
        for view, image, mask, view_matrix in prepared:
            height, width = mask.shape
            out = render_gaussians(
                means, quats, torch.exp(log_scales),
                torch.sigmoid(logit_opacity), colors,
                view_matrix=view_matrix, fx=view.fx, fy=view.fy,
                cx=view.cx, cy=view.cy, width=int(width), height=int(height),
                pixel_chunk=int(pixel_chunk),
            )
            residual = (out.rgb - image[..., :3]).abs().mean(dim=-1)
            total = total + float(view.weight) * residual[mask].mean()
        total.backward()
        opt.step()
        history.append(float(total.detach()))
        if log_every and step % int(log_every) == 0:
            print(f"  [{step:4d}] loss {history[-1]:.6f}", flush=True)

    with torch.no_grad():
        final_scales = torch.exp(log_scales)
        final_opacity = torch.sigmoid(logit_opacity)
        kept = final_opacity > MIN_OPACITY

    return TrainedSplats(
        means=means.detach()[kept].cpu().numpy(),
        quats=quats.detach()[kept].cpu().numpy(),
        scales=final_scales[kept].cpu().numpy(),
        opacities=final_opacity[kept].cpu().numpy(),
        colors=colors.detach()[kept].cpu().numpy(),
        report={
            "source": SPLAT_SOURCE,
            "device": str(dev),
            "iters": int(iters),
            "lr": float(lr),
            "seed": int(seed_value),
            "n_seeded": int(n),
            "n_kept": int(kept.sum()),
            "init_scale_m": init_scale,
            "loss_first": history[0] if history else None,
            "loss_last": history[-1] if history else None,
            "near_m": float(seed.near_m),
            "far_m": float(seed.far_m),
            "views": [view.name for view in views],
        },
    )
