"""Tests for skyline wall handling: distance-mode splitting within an azimuth
peak (ProxyDerivationConfig.wall_distance_modes) and exclude_mask scoping on
the primitive derive nodes.

The failure this guards: a street-grid skyline has ~2 facing directions but
MANY depths, and the classic azimuth clustering fit ONE plane per direction at
the median distance — thirty facades collapsed into one slab (found on a real
6K city plate). Fixture: TWO fronto-parallel walls (same azimuth) at different
depths, side by side in image space.
"""

import numpy as np
import pytest

from atlas_camera.comfy.nodes import AtlasDeriveTowersSpires, AtlasDeriveWalls
from atlas_camera.core.proxy_geometry import (
    ProxyDerivationConfig,
    derive_projection_proxies,
)
from atlas_camera.core.schema import (
    AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera,
)
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 60.0
CAM_HEIGHT = 1.6
NEAR_Z, FAR_Z = 6.0, 14.0


def _view_matrix(h):
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _two_wall_depth(h=CAM_HEIGHT):
    """Ground plane + two fronto-parallel walls (SAME facing direction):
    left image half -> wall at z=-NEAR_Z (h 4m), right half -> z=-FAR_Z (h 6m)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY

    depth = np.full((H, W), SKY)
    t_ground = np.full((H, W), np.inf)
    down = dy < -1e-6
    t_ground[down] = -h / dy[down]

    t_wall = np.full((H, W), np.inf)
    for cols, t, wall_h in ((uu < CX, NEAR_Z, 4.0), (uu >= CX, FAR_Z, 6.0)):
        y_at = h + dy * t
        vis = cols & (y_at >= 0.0) & (y_at <= wall_h)
        t_wall[vis] = t

    stacked = np.stack([
        depth,
        np.where(np.isfinite(t_ground), t_ground, SKY),
        np.where(np.isfinite(t_wall), t_wall, SKY),
    ])
    return stacked.min(axis=0).astype(np.float32)


def _solve(h=CAM_HEIGHT):
    intr = AtlasIntrinsics(
        image_width=W, image_height=H, focal_length_mm=35.0, sensor_width_mm=36.0,
        fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
    )
    extr = AtlasExtrinsics(camera_view_matrix=_view_matrix(h))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _depth_result(depth_map):
    return DepthResult(
        depth=depth_map, is_metric=True, model_id="fake",
        image_width=W, image_height=H,
        near=float(depth_map.min()), far=float(depth_map.max()),
    )


def _walls(out_solve):
    return [p for p in out_solve.projection_scene.proxy_geometry
            if p.name.startswith("projection_wall_")]


def test_classic_single_mode_collapses_same_azimuth_depths():
    # The documented limitation distance_modes exists to fix: one plane per
    # facing direction, everything else discarded.
    (out,) = AtlasDeriveWalls().derive(
        _solve(), _depth_result(_two_wall_depth()), max_walls=16)
    assert len(_walls(out)) == 1


def test_distance_modes_split_same_azimuth_into_depth_rows():
    (out,) = AtlasDeriveWalls().derive(
        _solve(), _depth_result(_two_wall_depth()), max_walls=16,
        distance_modes=4)
    walls = _walls(out)
    assert len(walls) == 2
    # Plane offset d = p·n with the normal flipped toward the camera, so a
    # frontal wall at world z=-Z carries d = -Z: compare magnitudes.
    dists = sorted(abs(p.metadata["distance_m"]) for p in walls)
    assert abs(dists[0] - NEAR_Z) < 0.5
    assert abs(dists[1] - FAR_Z) < 0.5
    assert out.projection_scene.debug_metadata["proxy_derivation"][
        "distance_modes"] == 4


def test_distance_modes_work_on_towers_spires_too():
    (out,) = AtlasDeriveTowersSpires().derive(
        _solve(), _depth_result(_two_wall_depth()), max_walls=16,
        distance_modes=4)
    assert len(_walls(out)) == 2


def test_max_walls_budget_still_caps_modes():
    (out,) = AtlasDeriveWalls().derive(
        _solve(), _depth_result(_two_wall_depth()), max_walls=1,
        distance_modes=4)
    assert len(_walls(out)) == 1


def test_exclude_mask_scopes_walls_core():
    # Core-level (numpy mask): exclude the LEFT half -> only the far wall fits.
    depth = _two_wall_depth().astype(np.float64)
    excl = np.zeros((H, W), dtype=bool)
    excl[:, : W // 2] = True
    cfg = ProxyDerivationConfig(wall_distance_modes=4)
    prims, stats = derive_projection_proxies(
        depth, view_matrix=_view_matrix(CAM_HEIGHT), fx=FX, fy=FY, cx=CX, cy=CY,
        max_walls=16, config=cfg, exclude_mask=excl)
    walls = [p for p in prims if p.name.startswith("projection_wall_")]
    assert len(walls) == 1
    assert abs(abs(walls[0].metadata["distance_m"]) - FAR_Z) < 0.5


def test_exclude_mask_never_changes_ground_scale():
    # The whole point of walls-and-objects-only masking: masked branches must
    # land in the SAME metric world so AtlasMergeGeometry can combine them.
    depth = _two_wall_depth().astype(np.float64)
    excl = np.zeros((H, W), dtype=bool)
    excl[:, : W // 2] = True
    _, stats_full = derive_projection_proxies(
        depth, view_matrix=_view_matrix(CAM_HEIGHT), fx=FX, fy=FY, cx=CX, cy=CY,
        max_walls=16, config=ProxyDerivationConfig())
    _, stats_masked = derive_projection_proxies(
        depth, view_matrix=_view_matrix(CAM_HEIGHT), fx=FX, fy=FY, cx=CX, cy=CY,
        max_walls=16, config=ProxyDerivationConfig(), exclude_mask=excl)
    assert stats_masked["ground_scale"] == pytest.approx(
        stats_full["ground_scale"], rel=1e-9)


def test_exclude_mask_node_input_via_torch_tensor():
    torch = pytest.importorskip("torch")
    excl = torch.zeros(1, H, W)
    excl[:, :, : W // 2] = 1.0
    (out,) = AtlasDeriveWalls().derive(
        _solve(), _depth_result(_two_wall_depth()), max_walls=16,
        distance_modes=4, exclude_mask=excl)
    walls = _walls(out)
    assert len(walls) == 1
    assert abs(abs(walls[0].metadata["distance_m"]) - FAR_Z) < 0.5
