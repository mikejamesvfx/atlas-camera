"""AtlasScaleOverride — the artist's manual metric-scale dial.

Single-image scale is ambiguous; this node rescales a solve by a factor (or to
an absolute camera height) by scaling the camera position + both matrices'
translation columns. The node itself is dependency-free (pure Python); only the
'flows to metric' test needs numpy for estimate_ground_scale.
"""

import pytest

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, AtlasScaleOverride
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera

W = H = 256
FX = FY = 250.0
CX = CY = 128.0


def _view_matrix(h):
    """Level camera at (0, h, 0), identity rotation — world->cam translation only."""
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -h), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _solve(h=1.6):
    intr = AtlasIntrinsics(image_width=W, image_height=H, fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY)
    extr = AtlasExtrinsics(camera_view_matrix=_view_matrix(h))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def test_registered():
    node = NODE_CLASS_MAPPINGS["AtlasScaleOverride"]
    assert node.RETURN_NAMES == ("solve", "report")


def test_scale_multiplier_scales_camera_height():
    out, report = AtlasScaleOverride().override(_solve(1.6), scale=10.0)
    vm = out.camera.extrinsics.camera_view_matrix
    assert vm[1][3] == pytest.approx(-16.0)                          # translation ×10
    assert vm[0][0] == pytest.approx(1.0) and vm[1][1] == pytest.approx(1.0)  # rotation untouched
    assert out.camera.extrinsics.camera_position[1] == pytest.approx(16.0)
    assert out.debug_metadata["scale_override"] == pytest.approx(10.0)
    assert out.debug_metadata["scale_source"] == "manual_override"
    assert "16.00 m" in report


def test_input_solve_is_untouched():
    src = _solve(1.6)
    AtlasScaleOverride().override(src, scale=10.0)
    assert src.camera.extrinsics.camera_view_matrix[1][3] == pytest.approx(-1.6)  # deep-copied


def test_absolute_camera_height_computes_factor():
    out, _ = AtlasScaleOverride().override(_solve(1.6), camera_height_m=16.0)
    assert out.camera.extrinsics.camera_position[1] == pytest.approx(16.0)
    assert out.debug_metadata["scale_override"] == pytest.approx(10.0)
    # absolute wins over the multiplier when set
    out2, _ = AtlasScaleOverride().override(_solve(1.6), scale=3.0, camera_height_m=8.0)
    assert out2.camera.extrinsics.camera_position[1] == pytest.approx(8.0)


def test_default_is_a_noop():
    out, _ = AtlasScaleOverride().override(_solve(1.6))   # scale=1.0
    assert out.camera.extrinsics.camera_view_matrix[1][3] == pytest.approx(-1.6)
    assert out.debug_metadata["scale_override"] == pytest.approx(1.0)


def test_scale_flows_into_metric_ground_scale():
    """The real proof: estimate_ground_scale on the rescaled solve is factor× the
    original, so every downstream metric distance (band cutoffs, exports) scales."""
    np = pytest.importorskip("numpy")
    from atlas_camera.core.relief_mesh import estimate_ground_scale

    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    depth = np.full((H, W), 50.0)
    looking_down = dy < -1e-6
    depth[looking_down] = -1.6 / dy[looking_down]        # ground plane at the h=1.6 scale

    solve = _solve(1.6)
    s0, _ = estimate_ground_scale(depth, view_matrix=solve.camera.extrinsics.camera_view_matrix,
                                  fx=FX, fy=FY, cx=CX, cy=CY)
    out, _ = AtlasScaleOverride().override(solve, scale=10.0)
    s1, _ = estimate_ground_scale(depth, view_matrix=out.camera.extrinsics.camera_view_matrix,
                                  fx=FX, fy=FY, cx=CX, cy=CY)
    assert s1 == pytest.approx(s0 * 10.0, rel=1e-6)


def test_override_sets_scale_confidence_metric():
    out, _ = AtlasScaleOverride().override(_solve(1.6), scale=10.0)
    assert out.camera.confidence.individual_metrics["scale"] == pytest.approx(1.0)
