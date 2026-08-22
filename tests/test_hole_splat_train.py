"""The rasterizer has no oracle behind it, so verify it rather than trust it.

gsplat's CUDA JIT does not build on this stack (a Windows ``#define small char``
collision inside a torch header), so there is no reference implementation to
cross-check against — everything downstream inherits this file's correctness.

Two classes of test therefore matter, and they cover DIFFERENT failures —
measured by mutating the source and seeing which tests noticed:

* **Geometry reasoned about by hand** is what catches wrong math. A gaussian on
  the optical axis lands in the middle; +X is right; +Y is up while rows go
  down; a near opaque gaussian hides a far one; reported depth is metres.
* **Finite differences** catch a broken GRAPH, not wrong geometry. Autograd
  differentiates whatever function was written, so if the projection is wrong
  autograd and the difference quotient agree on the wrong thing and both pass.
  Flipping the image-Y sign was measured to pass all six FD checks and fail
  exactly one geometry test. What FD does earn: it proves the compositing
  cumprod, the clamps and the depth-sort permutation did not silently sever
  the gradient.

The subtlest hole found this way: the projection Jacobian's ``dv/dz`` term
scales with ``y`` and only bends an ANISOTROPIC footprint, so every near-axis
isotropic test here was blind to it — flipping its sign passed all 22 tests.
``test_a_ray_elongated_gaussian_projects_radially_from_the_principal_point``
exists solely to close that gap.

Runs on CPU. GPU-only paths are not exercised here on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.core.hole_splat import HoleSeed  # noqa: E402
from atlas_camera.inference.hole_splat_train import (  # noqa: E402
    TrainView,
    render_gaussians,
    train_hole_splats,
)

W, H = 32, 24
FX = FY = 40.0
CX, CY = W / 2.0, H / 2.0
VIEW = torch.eye(4, dtype=torch.float64)


def _one_gaussian(*, xyz=(0.0, 0.0, -5.0), scale=0.15, opacity=0.9,
                  color=(1.0, 0.0, 0.0), requires_grad=False):
    means = torch.tensor([list(xyz)], dtype=torch.float64,
                         requires_grad=requires_grad)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    scales = torch.full((1, 3), float(scale), dtype=torch.float64,
                        requires_grad=requires_grad)
    opacities = torch.tensor([float(opacity)], dtype=torch.float64,
                             requires_grad=requires_grad)
    colors = torch.tensor([list(color)], dtype=torch.float64,
                          requires_grad=requires_grad)
    return means, quats, scales, opacities, colors


def _render(*args, **kwargs):
    return render_gaussians(*args, view_matrix=VIEW, fx=FX, fy=FY, cx=CX, cy=CY,
                            width=W, height=H, **kwargs)


# ------------------------------------------------------------------ geometry


def test_a_gaussian_on_the_optical_axis_lands_in_the_middle():
    out = _render(*_one_gaussian())

    peak = int(out.alpha.argmax())
    row, col = divmod(peak, W)
    assert abs(row - CY) <= 1.0
    assert abs(col - CX) <= 1.0
    assert float(out.alpha.max()) > 0.5


def test_moving_a_gaussian_right_moves_the_splat_right():
    """+X is right and the image origin is top-left; a sign error here would
    survive every symmetric test."""

    left = _render(*_one_gaussian(xyz=(-0.5, 0.0, -5.0)))
    right = _render(*_one_gaussian(xyz=(0.5, 0.0, -5.0)))

    col_left = int(left.alpha.argmax()) % W
    col_right = int(right.alpha.argmax()) % W
    assert col_right > col_left


def test_raising_a_gaussian_moves_the_splat_up_the_image():
    """+Y is UP in world, and image rows increase DOWNWARD."""

    low = _render(*_one_gaussian(xyz=(0.0, -0.5, -5.0)))
    high = _render(*_one_gaussian(xyz=(0.0, 0.5, -5.0)))

    row_low = int(low.alpha.argmax()) // W
    row_high = int(high.alpha.argmax()) // W
    assert row_high < row_low


def test_a_gaussian_behind_the_camera_renders_nothing():
    out = _render(*_one_gaussian(xyz=(0.0, 0.0, 5.0)))  # +Z is behind
    assert float(out.alpha.max()) == 0.0


def test_reported_depth_is_the_forward_distance_in_metres():
    out = _render(*_one_gaussian(xyz=(0.0, 0.0, -7.0), opacity=0.99))

    peak = int(out.alpha.argmax())
    row, col = divmod(peak, W)
    assert float(out.depth[row, col]) == pytest.approx(7.0, abs=0.05)


def test_a_nearer_opaque_gaussian_occludes_a_farther_one():
    """Front-to-back compositing, checked by colour rather than by depth."""

    means = torch.tensor([[0.0, 0.0, -4.0], [0.0, 0.0, -9.0]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64)
    scales = torch.full((2, 3), 0.2, dtype=torch.float64)
    opacities = torch.tensor([0.99, 0.99], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    out = _render(means, quats, scales, opacities, colors)
    centre = out.rgb[int(CY), int(CX)]
    assert float(centre[0]) > 0.5      # near red dominates
    assert float(centre[1]) < 0.2      # far green is hidden


def test_a_transparent_near_gaussian_lets_the_far_one_through():
    means = torch.tensor([[0.0, 0.0, -4.0], [0.0, 0.0, -9.0]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64)
    scales = torch.full((2, 3), 0.2, dtype=torch.float64)
    opacities = torch.tensor([0.05, 0.99], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    out = _render(means, quats, scales, opacities, colors)
    assert float(out.rgb[int(CY), int(CX)][1]) > 0.3


def test_a_bigger_scale_covers_more_pixels():
    small = _render(*_one_gaussian(scale=0.08))
    large = _render(*_one_gaussian(scale=0.30))

    assert float((large.alpha > 0.01).sum()) > float((small.alpha > 0.01).sum())


def test_chunking_does_not_change_the_image():
    """pixel_chunk is a memory knob and must never be a numerical one."""

    args = _one_gaussian(scale=0.25)
    whole = _render(*args, pixel_chunk=10 ** 9)
    chunked = _render(*args, pixel_chunk=37)  # deliberately not a row multiple

    assert torch.allclose(whole.rgb, chunked.rgb, atol=1e-12)
    assert torch.allclose(whole.alpha, chunked.alpha, atol=1e-12)


# --------------------------------------------------------------- gradients


def _finite_difference(param, index, render_fn, *, eps=1e-5):
    """Central difference of the scalar loss w.r.t. one parameter entry."""
    with torch.no_grad():
        original = param.view(-1)[index].item()
        param.view(-1)[index] = original + eps
        plus = float(render_fn())
        param.view(-1)[index] = original - eps
        minus = float(render_fn())
        param.view(-1)[index] = original
    return (plus - minus) / (2 * eps)


@pytest.mark.parametrize("which,index", [
    ("means", 0),        # x
    ("means", 1),        # y
    ("means", 2),        # z (depth — enters through the Jacobian too)
    ("scales", 0),
    ("opacities", 0),
    ("colors", 0),
])
def test_analytic_gradients_match_finite_differences(which, index):
    """Autograd differentiates whatever was written — including the wrong
    function. This is the check that the math is the intended math."""

    means, quats, scales, opacities, colors = _one_gaussian(
        xyz=(0.12, -0.08, -5.0), scale=0.35, opacity=0.7, requires_grad=True)
    tensors = {"means": means, "scales": scales,
               "opacities": opacities, "colors": colors}
    target = torch.zeros((H, W, 3), dtype=torch.float64)

    def loss_value():
        out = _render(means, quats, scales, opacities, colors)
        return (out.rgb - target).square().mean()

    analytic_loss = loss_value()
    analytic_loss.backward()
    param = tensors[which]
    analytic = float(param.grad.view(-1)[index])

    numeric = _finite_difference(param, index, loss_value)

    assert analytic == pytest.approx(numeric, rel=2e-3, abs=1e-9), (
        f"{which}[{index}]: analytic {analytic:.3e} vs numeric {numeric:.3e}")


def test_gradients_are_finite_for_a_sub_pixel_gaussian():
    """The EWA dilation exists so a gaussian smaller than a pixel does not
    produce a near-singular covariance and an exploding gradient."""

    means, quats, scales, opacities, colors = _one_gaussian(
        scale=1e-4, requires_grad=True)
    out = _render(means, quats, scales, opacities, colors)
    out.rgb.square().mean().backward()

    assert bool(torch.isfinite(means.grad).all())
    assert bool(torch.isfinite(scales.grad).all())


# ------------------------------------------------------------------ training


def _seed_and_views(n=48):
    rng = np.random.default_rng(0)
    pts = np.stack([
        rng.uniform(-0.4, 0.4, n),
        rng.uniform(-0.3, 0.3, n),
        rng.uniform(-6.0, -4.0, n),
    ], axis=-1)
    seed = HoleSeed(points_world=pts, colors=np.full((n, 3), 0.5, np.float32),
                    near_m=4.0, far_m=6.0, scale_m=0.2)

    target = np.zeros((H, W, 3), dtype=np.float32)
    target[:, :, 2] = 1.0                       # want blue
    mask = np.zeros((H, W), dtype=bool)
    mask[6:18, 8:24] = True
    view = TrainView(image=target, loss_mask=mask,
                     view_matrix=np.eye(4), fx=FX, fy=FY, cx=CX, cy=CY,
                     name="primary")
    return seed, [view]


def test_training_reduces_the_masked_loss():
    seed, views = _seed_and_views()
    trained = train_hole_splats(seed, views, iters=25, lr=0.05, device="cpu")

    assert trained.report["loss_last"] < trained.report["loss_first"]
    assert trained.count > 0
    assert trained.report["device"] == "cpu"
    assert trained.report["views"] == ["primary"]


def test_training_moves_colour_toward_the_target():
    seed, views = _seed_and_views()
    trained = train_hole_splats(seed, views, iters=40, lr=0.1, device="cpu")

    # Target is pure blue; the seed started neutral grey.
    assert trained.colors[:, 2].mean() > trained.colors[:, 0].mean()


def test_an_empty_loss_mask_is_refused():
    seed, views = _seed_and_views()
    empty = TrainView(image=views[0].image,
                      loss_mask=np.zeros((H, W), dtype=bool),
                      view_matrix=np.eye(4), fx=FX, fy=FY, cx=CX, cy=CY,
                      name="empty")
    with pytest.raises(ValueError, match="loss_mask is empty"):
        train_hole_splats(seed, [empty], iters=2, device="cpu")


def test_mismatched_image_and_mask_are_refused():
    seed, views = _seed_and_views()
    bad = TrainView(image=np.zeros((H + 2, W, 3), np.float32),
                    loss_mask=np.ones((H, W), dtype=bool),
                    view_matrix=np.eye(4), fx=FX, fy=FY, cx=CX, cy=CY,
                    name="bad")
    with pytest.raises(ValueError, match="disagree"):
        train_hole_splats(seed, [bad], iters=2, device="cpu")


def test_training_is_deterministic_for_a_given_seed():
    seed, views = _seed_and_views()
    a = train_hole_splats(seed, views, iters=10, lr=0.05, device="cpu",
                          seed_value=3)
    b = train_hole_splats(seed, views, iters=10, lr=0.05, device="cpu",
                          seed_value=3)

    assert np.allclose(a.means, b.means)
    assert a.report["loss_last"] == pytest.approx(b.report["loss_last"])


def test_no_views_is_refused():
    seed, _views = _seed_and_views()
    with pytest.raises(ValueError, match="at least one view"):
        train_hole_splats(seed, [], iters=2, device="cpu")


def _alpha_orientation(alpha):
    """Principal axis of the rendered footprint, as a unit (col, row) vector."""
    a = alpha.detach().numpy().astype(np.float64)
    total = a.sum()
    assert total > 1e-6, "footprint is empty"
    rows, cols = np.nonzero(np.ones_like(a))
    w = a.reshape(-1)
    cy_ = (w * rows).sum() / total
    cx_ = (w * cols).sum() / total
    dr = rows - cy_
    dc = cols - cx_
    cov = np.array([
        [(w * dc * dc).sum() / total, (w * dc * dr).sum() / total],
        [(w * dc * dr).sum() / total, (w * dr * dr).sum() / total],
    ])
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, int(np.argmax(vals))], (cx_, cy_)


def test_a_ray_elongated_gaussian_projects_radially_from_the_principal_point():
    """Catches a wrong sign in the projection Jacobian's dv/dz term.

    That term scales with y and only bends an ANISOTROPIC footprint, so every
    near-axis isotropic test in this file is blind to it — flipping its sign
    was measured to pass all of them. A gaussian stretched along the view ray
    must project to an ellipse whose long axis points along the image-space
    radial direction from the principal point; a flipped dv/dz mirrors that
    tilt about the horizontal axis.
    """

    # Well off-axis in BOTH image axes, so the radial direction is diagonal
    # and a mirrored tilt cannot coincide with the correct one.
    means = torch.tensor([[0.9, 0.7, -5.0]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    # Thin across the ray, long ALONG it (world -Z is the view direction here).
    scales = torch.tensor([[0.02, 0.02, 0.9]], dtype=torch.float64)
    opacities = torch.tensor([0.9], dtype=torch.float64)
    colors = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

    out = _render(means, quats, scales, opacities, colors)
    axis, (cx_, cy_) = _alpha_orientation(out.alpha)

    radial = np.array([cx_ - CX, cy_ - CY], dtype=np.float64)
    radial /= np.linalg.norm(radial)
    alignment = abs(float(np.dot(axis, radial)))

    assert alignment > 0.9, (
        f"footprint major axis {axis} is not radial from the principal point "
        f"(radial {radial}, |cos| {alignment:.3f}) — check the Jacobian")
