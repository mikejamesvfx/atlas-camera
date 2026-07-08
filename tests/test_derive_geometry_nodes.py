"""Tests for the composable geometry-derivation nodes (AtlasDeriveReliefMesh,
AtlasDeriveWalls, AtlasDeriveTowersSpires, AtlasDeriveRoofsFacades,
AtlasDeriveInteriorRoom) — each a thin wrapper around an already-tested core
extraction function, taking a pre-computed ATLAS_DEPTH_MAP instead of running
its own depth estimation (see AtlasDepthMap). Uses a self-contained analytic
ground+wall depth map (same convention as test_proxy_geometry.py: level
camera at (0, h, 0), identity rotation) rather than a real photo, so these
run with only numpy — no [neural] extra or model download needed.
"""

import numpy as np
import pytest

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasDeriveInteriorRoom,
    AtlasDeriveReliefMesh,
    AtlasDeriveRoofsFacades,
    AtlasDeriveTowersSpires,
    AtlasDeriveWalls,
)
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 60.0
CAM_HEIGHT = 1.6


def _view_matrix(h):
    """Level camera at (0, h, 0), identity rotation — world->cam translation only."""
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _room_depth(h=CAM_HEIGHT, wall_z=-8.0, wall_h=3.0):
    """Ground plane (Y=0) + one fronto-parallel wall at world z=wall_z, height wall_h."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dx = (uu - CX) / FX
    dy = -(vv - CY) / FY  # cam y-up; image v grows downward

    depth = np.full((H, W), SKY)

    # Ground: t where h + dy*t == 0 (looking downward only).
    t_ground = np.full((H, W), np.inf)
    looking_down = dy < -1e-6
    t_ground[looking_down] = -h / dy[looking_down]

    # Wall: world z = -t == wall_z -> t = -wall_z; visible where 0 <= y <= wall_h.
    t_wall = np.full((H, W), np.inf)
    t = -wall_z
    y_at = h + dy * t
    visible = (y_at >= 0.0) & (y_at <= wall_h)
    t_wall[visible] = t

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
        image_width=W, image_height=H, near=float(depth_map.min()), far=float(depth_map.max()),
    )


def _proxy_names(out_solve):
    return [p.name for p in out_solve.projection_scene.proxy_geometry]


def _all_tagged(out_solve):
    return all((p.metadata or {}).get("role") == PROXY_ROLE
               for p in out_solve.projection_scene.proxy_geometry)


def test_all_five_nodes_registered():
    # AtlasDeriveReliefMesh additionally exposes the relief mesh's own
    # hole_mask (see relief_mesh.ReliefMesh.hole_mask); the other four are
    # primitive-only derivers and have no mesh to have holes in.
    for name in ["AtlasDeriveWalls", "AtlasDeriveTowersSpires",
                 "AtlasDeriveRoofsFacades", "AtlasDeriveInteriorRoom"]:
        assert name in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS[name].RETURN_TYPES == ("ATLAS_SOLVE",)
    assert "AtlasDeriveReliefMesh" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["AtlasDeriveReliefMesh"].RETURN_TYPES == ("ATLAS_SOLVE", "MASK")


def test_relief_mesh_produces_mesh_and_backdrop():
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, hole_mask = AtlasDeriveReliefMesh().derive(solve, depth, relief_grid=32)

    names = _proxy_names(out)
    assert "projection_relief_mesh" in names
    assert "projection_backdrop" in names
    assert _all_tagged(out)
    assert tuple(hole_mask.shape) == (1, H, W)


def test_relief_quality_overrides_relief_grid():
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _hole_mask = AtlasDeriveReliefMesh().derive(solve, depth, relief_grid=32, relief_quality="low")
    assert out.projection_scene.debug_metadata["proxy_derivation"]["relief_grid"] == 64


def test_walls_node_finds_ground_and_wall():
    solve = _solve()
    depth = _depth_result(_room_depth())
    (out,) = AtlasDeriveWalls().derive(solve, depth, max_walls=4, max_objects=0)

    names = _proxy_names(out)
    assert "projection_ground" in names
    assert any(n.startswith("projection_wall_") for n in names)
    assert "projection_backdrop" in names
    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "azimuth_walls"
    assert _all_tagged(out)


def test_towers_spires_node_finds_ground_and_wall():
    solve = _solve()
    depth = _depth_result(_room_depth())
    (out,) = AtlasDeriveTowersSpires().derive(solve, depth, max_walls=4, max_objects=0)

    names = _proxy_names(out)
    assert "projection_ground" in names
    assert any(n.startswith("projection_wall_") for n in names)
    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "vertical_extrusion"


def test_no_focal_length_returns_solve_unchanged():
    intr = AtlasIntrinsics(image_width=W, image_height=H)  # no fx_px
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=AtlasExtrinsics()))
    depth = _depth_result(_room_depth())

    (out,) = AtlasDeriveWalls().derive(solve, depth)
    assert out is solve  # returned unchanged, per the fx<=0 guard


def test_roofs_facades_node_runs_and_tags_output():
    solve = _solve()
    depth = _depth_result(_room_depth())
    (out,) = AtlasDeriveRoofsFacades().derive(solve, depth, max_planes=8)

    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "ransac_planes"
    assert _all_tagged(out)


def test_interior_room_node_runs_and_tags_output():
    solve = _solve()
    depth = _depth_result(_room_depth())
    (out,) = AtlasDeriveInteriorRoom().derive(solve, depth)

    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "room_cuboid"
    assert _all_tagged(out)


def test_derive_node_does_not_mutate_input_solve():
    solve = _solve()
    depth = _depth_result(_room_depth())
    before = len(solve.projection_scene.proxy_geometry)

    AtlasDeriveWalls().derive(solve, depth)

    assert len(solve.projection_scene.proxy_geometry) == before
