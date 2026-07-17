"""AtlasCleanPlateStack — up to four artist plates + alphas as one node.

Contracts under test: farthest-highest priorities by slot order, the seam
doctrine (behind slots smear, the nearest used slot keeps a clean cut),
mask-membership geometry (matte grown → exclude), fail-soft slot skipping,
and pure pass-through when nothing is wired.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, AtlasCleanPlateStack
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 128
FX = FY = 125.0
CX = CY = 64.0


def _solve(h=1.6):
    intr = AtlasIntrinsics(image_width=W, image_height=H, focal_length_mm=35.0,
                           sensor_width_mm=36.0, fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY)
    extr = AtlasExtrinsics(camera_view_matrix=(
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _depth():
    d = np.full((H, W), 8.0, dtype=np.float64)
    d[:, : W // 2] = 3.0          # near half
    return DepthResult(depth=d, is_metric=True, model_id="fake",
                       image_width=W, image_height=H, near=3.0, far=8.0)


def _plate(v=0.5):
    return torch.full((1, H, W, 3), v, dtype=torch.float32)


def _matte(x0, x1):
    m = torch.zeros(1, H, W, dtype=torch.float32)
    m[:, 20:100, x0:x1] = 1.0
    return m


def test_registered():
    assert NODE_CLASS_MAPPINGS["AtlasCleanPlateStack"] is AtlasCleanPlateStack


def test_two_slots_farthest_highest_and_seam_doctrine():
    out, report = AtlasCleanPlateStack().stack(
        _solve(), _depth(),
        plate_1=_plate(0.3), matte_1=_matte(70, 120),   # far stratum
        plate_4=_plate(0.8), matte_4=_matte(10, 60),    # near stratum
        name_1="mountains", name_4="road", geometry_1="card", geometry_4="ground",
        relief_grid=64,
    )
    srcs = out.projection_sources
    assert [s.name for s in srcs] == ["mountains", "road"]
    assert [s.priority for s in srcs] == [15.0, 0.0]          # farthest-highest
    assert all(s.mask_b64 for s in srcs)                       # mattes embedded
    assert "'road' added" in report and "clean cut" in report  # nearest = clean
    assert "edge_extend=24" in report                          # behind slot smears
    assert "edge_extend=0" in report


def test_incomplete_and_empty_slots_skip_soft():
    out, report = AtlasCleanPlateStack().stack(
        _solve(), _depth(),
        plate_2=_plate(),                                  # matte missing
        plate_3=_plate(), matte_3=torch.zeros(1, H, W),    # matte empty
        plate_4=_plate(), matte_4=_matte(10, 60), relief_grid=64,
    )
    assert len(out.projection_sources) == 1                # only the complete slot
    assert "slot 2: SKIPPED" in report and "slot 3: SKIPPED" in report
    # sole used slot is also the nearest -> clean cut
    assert "edge_extend=0" in report


def test_no_slots_passes_through():
    solve = _solve()
    out, report = AtlasCleanPlateStack().stack(solve, _depth())
    assert out is not solve                                # deep copy, never mutate
    assert out.projection_sources == []
    assert "passes through" in report


def test_transparency_convention_inverts():
    # LoadImage MASK marks TRANSPARENT pixels: an all-opaque plate gives a
    # zero mask, which must become a FULL matte when the toggle is on.
    out, _ = AtlasCleanPlateStack().stack(
        _solve(), _depth(),
        plate_4=_plate(), matte_4=torch.zeros(1, H, W),
        mattes_are_transparency=True, relief_grid=64,
    )
    assert len(out.projection_sources) == 1
