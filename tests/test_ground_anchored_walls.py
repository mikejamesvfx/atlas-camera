"""Tests for ground-anchored extrusion + roofline segmentation.

The premise (user-proposed, 2026-07-10): monocular depth "bananas" up tall
structures but the ray DIRECTION through a base pixel is exact — so where a
building visibly meets the analytic Y=0 ground plane, its footprint distance
is pure geometry. Depth is demoted to grouping pixels. Guards:
- banana-bias recovery (the headline claim),
- the occlusion poison-gate (hidden bases must NOT anchor — a 1.5m occluder
  overshoots ray-ground intersection by orders of magnitude),
- roofline segmentation (one plane per silhouette step),
- defaults-off regression (flag off == previous behavior).
"""

import numpy as np
import pytest

from atlas_camera.comfy.nodes import AtlasDeriveTowersSpires, AtlasDeriveWalls
from atlas_camera.core.schema import (
    AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera,
)
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 60.0
CAM_H = 1.6
WALL_Z = 8.0


def _view_matrix(h):
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -h),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _solve(h=CAM_H):
    intr = AtlasIntrinsics(
        image_width=W, image_height=H, focal_length_mm=35.0, sensor_width_mm=36.0,
        fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY)
    extr = AtlasExtrinsics(camera_view_matrix=_view_matrix(h))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _depth_result(depth_map):
    return DepthResult(depth=depth_map, is_metric=True, model_id="fake",
                       image_width=W, image_height=H,
                       near=float(depth_map.min()), far=float(depth_map.max()))


def _scene(banana=0.0, wall_h=6.0, base_lift_m=0.0, two_rooflines=False,
           h=CAM_H, wall_z=WALL_Z):
    """Ground plane + one fronto-parallel wall at z=-wall_z whose DEPTH is
    optionally banana-warped: pixels higher up the wall report depth
    inflated by up to ``banana`` (relative), mimicking DepthAnything's
    low-frequency drift. ``base_lift_m`` hides the lowest metres of the wall
    behind ground (occluded contact). ``two_rooflines``: left half height
    wall_h, right half 2*wall_h.
    """
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY

    depth = np.full((H, W), SKY)
    t_ground = np.full((H, W), np.inf)
    down = dy < -1e-6
    t_ground[down] = -h / dy[down]

    t_wall = np.full((H, W), np.inf)
    y_at = h + dy * wall_z
    h_col = np.where(uu < CX, wall_h, 2.0 * wall_h) if two_rooflines \
        else np.full((H, W), wall_h)
    vis = (y_at >= base_lift_m) & (y_at <= h_col)
    # banana: depth error grows linearly with height on the wall — the
    # LOW-FREQUENCY drift that shifts the whole offset distribution (a
    # quadratic profile packs most pixels near zero warp and the classic
    # fit's ±2% inlier slab self-cleans it; real DepthAnything drift moves
    # the median, which is exactly what the anchor must beat).
    frac = np.clip(y_at / np.maximum(h_col, 1e-6), 0.0, 1.0)
    t_wall[vis] = (wall_z * (1.0 + banana * frac[vis]))

    stacked = np.stack([
        depth,
        np.where(np.isfinite(t_ground), t_ground, SKY),
        np.where(np.isfinite(t_wall), t_wall, SKY),
    ])
    return stacked.min(axis=0).astype(np.float32)


def _walls(out):
    return [p for p in out.projection_scene.proxy_geometry
            if p.name.startswith("projection_wall_")]


def test_banana_bias_recovered_by_ground_anchor():
    # 15% linear drift: strong enough to shift the fit's median, mild enough
    # that the tilted normals stay under wall_normal_max (a >~19% drift on
    # this wall leans the synthetic surface past the vertical-candidate
    # filter and no wall is found at all — itself a real banana failure
    # mode, but not the one under test here).
    depth = _scene(banana=0.15)
    biased, = AtlasDeriveWalls().derive(
        _solve(), _depth_result(depth), max_walls=4)
    anchored, = AtlasDeriveWalls().derive(
        _solve(), _depth_result(depth), max_walls=4, ground_anchor=True)
    d_biased = abs(_walls(biased)[0].metadata["distance_m"])
    d_anchor = abs(_walls(anchored)[0].metadata["distance_m"])
    assert _walls(anchored)[0].metadata["ground_anchored"] is True
    # anchored distance must be much closer to the true 8m than the biased fit
    assert abs(d_anchor - WALL_Z) < 0.4
    assert abs(d_anchor - WALL_Z) < abs(d_biased - WALL_Z)


def test_flag_off_is_previous_behavior():
    depth = _scene(banana=0.15)
    off, = AtlasDeriveWalls().derive(_solve(), _depth_result(depth), max_walls=4)
    w = _walls(off)[0]
    assert w.metadata["ground_anchored"] is False
    assert w.metadata["anchor_weight"] is None


def test_occluded_base_never_anchors():
    # Contact hidden: wall only visible from 1.2m up — rays through the
    # lowest VISIBLE pixels land far behind the building. The poison-gate
    # must refuse the anchor and keep the depth-median distance.
    depth = _scene(banana=0.0, base_lift_m=1.2)
    out, = AtlasDeriveWalls().derive(
        _solve(), _depth_result(depth), max_walls=4, ground_anchor=True)
    w = _walls(out)[0]
    assert w.metadata["ground_anchored"] is False
    assert abs(abs(w.metadata["distance_m"]) - WALL_Z) < 0.5


def test_roofline_split_emits_one_wall_per_step():
    depth = _scene(two_rooflines=True, wall_h=3.0)
    single, = AtlasDeriveTowersSpires().derive(
        _solve(), _depth_result(depth), max_walls=8)
    split, = AtlasDeriveTowersSpires().derive(
        _solve(), _depth_result(depth), max_walls=8, roofline_split=True)
    assert len(_walls(single)) == 1
    seg_walls = _walls(split)
    assert len(seg_walls) == 2
    hs = sorted(p.dimensions[1] for p in seg_walls)
    assert hs[0] == pytest.approx(3.0, abs=0.8)
    assert hs[1] == pytest.approx(6.0, abs=0.8)
    assert all(p.metadata["roofline_segment"] for p in seg_walls)


def test_roofline_split_with_anchor_per_segment():
    depth = _scene(two_rooflines=True, wall_h=3.0, banana=0.10)
    out, = AtlasDeriveTowersSpires().derive(
        _solve(), _depth_result(depth), max_walls=8,
        roofline_split=True, ground_anchor=True)
    seg_walls = _walls(out)
    assert len(seg_walls) == 2
    for p in seg_walls:
        assert p.metadata["ground_anchored"] is True
        assert abs(abs(p.metadata["distance_m"]) - WALL_Z) < 0.5


def test_foreign_base_contamination_gated():
    """A near clutter band (car row) sharing the wall's azimuth must NOT
    teleport the anchor to its own footprint (found live on a real street
    photo: 12m facades 'anchored' at the 2m car row)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    depth = np.full((H, W), SKY)
    t_ground = np.full((H, W), np.inf)
    down = dy < -1e-6
    t_ground[down] = -CAM_H / dy[down]
    # main wall at 8m (full width, 0..6m tall)...
    t_wall = np.full((H, W), np.inf)
    y8 = CAM_H + dy * WALL_Z
    vis8 = (y8 >= 0.0) & (y8 <= 6.0)
    t_wall[vis8] = WALL_Z
    # ...plus a same-facing clutter band at 2m (0..0.9m tall, center columns)
    y2 = CAM_H + dy * 2.0
    clutter = (y2 >= 0.0) & (y2 <= 0.9) & (uu > 64) & (uu < 192)
    t_wall[clutter] = 2.0
    stacked = np.stack([depth,
                        np.where(np.isfinite(t_ground), t_ground, SKY),
                        np.where(np.isfinite(t_wall), t_wall, SKY)])
    dm = stacked.min(axis=0).astype(np.float32)

    out, = AtlasDeriveWalls().derive(
        _solve(), _depth_result(dm), max_walls=4, ground_anchor=True)
    for p in _walls(out):
        # every surviving wall must sit near the real 8m facade OR the real
        # 2m clutter — never in between and never teleported
        d = abs(p.metadata["distance_m"])
        assert abs(d - WALL_Z) < 1.2 or abs(d - 2.0) < 0.6, d
