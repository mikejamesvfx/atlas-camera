"""Camera-path tear-island selection regression coverage."""

from __future__ import annotations

import numpy as np

from atlas_camera.core.path_hole_repair import (
    PathHoleRepairConfig,
    build_path_hole_repair,
)
from atlas_camera.core.relief_mesh import build_relief_mesh
from atlas_camera.core.schema import (
    AtlasCameraKeyframe,
    AtlasCameraPath,
    AtlasExtrinsics,
    AtlasIntrinsics,
    LatentCamera,
)


W = H = 65
FX = FY = 80.0
CX = CY = 32.0
VIEW = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _fixture():
    depth = np.full((H, W), 5.0, dtype=np.float32)
    exclusion = np.zeros((H, W), dtype=bool)
    exclusion[27:38, 27:38] = True
    mesh = build_relief_mesh(
        depth,
        view_matrix=VIEW,
        fx=FX, fy=FY, cx=CX, cy=CY,
        grid_long_edge=W,
        depth_edge_rel=5.0,
        max_edge_factor=0.0,
        floor_clamp=None,
        exclude_mask=exclusion,
        apply_sky_heuristic=False,
        quad_coherence=True,
    )
    camera = LatentCamera(
        intrinsics=AtlasIntrinsics(
            image_width=W, image_height=H,
            fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
        ),
        extrinsics=AtlasExtrinsics(
            camera_position=(0.0, 0.0, 0.0),
            camera_view_matrix=VIEW,
            camera_world_matrix=VIEW,
        ),
    )
    path = AtlasCameraPath(
        keyframes=[
            AtlasCameraKeyframe(
                frame_index=0,
                position=(0.0, 0.0, 0.0),
                target=(0.0, 0.0, -5.0),
            ),
            AtlasCameraKeyframe(
                frame_index=9,
                position=(0.75, 0.0, 0.0),
                target=(0.0, 0.0, -5.0),
            ),
        ],
        fps=24.0,
        frame_count=10,
        lens_scale=0.8,
    )
    return mesh, camera, path


def test_path_repair_selects_visible_candidate_as_exact_source_mask():
    mesh, camera, path = _fixture()
    result = build_path_hole_repair(
        mesh,
        mesh.hole_mask,
        source_camera=camera,
        camera_path=path,
        config=PathHoleRepairConfig(
            frame_offset_from_end=0,
            resolution=128,
            normal_tolerance_deg=15.0,
            max_plane_error_m=0.02,
            max_hole_fraction=0.20,
        ),
    )

    assert result["frame_index"] == 9
    assert result["lens_scale"] == 0.8
    assert result["visible_ids"]
    assert result["selected_ids"] == result["visible_ids"]
    assert result["view_id_map"].shape == (128, 128)
    assert result["view_id_map"].max() > 0
    assert result["repair_mask"].shape == mesh.hole_mask.shape
    assert result["repair_mask"].any()
    assert np.logical_and(result["repair_mask"], ~mesh.hole_mask).sum() == 0


def test_path_repair_paint_selects_stable_island_id():
    mesh, camera, path = _fixture()
    automatic = build_path_hole_repair(
        mesh,
        mesh.hole_mask,
        source_camera=camera,
        camera_path=path,
        config=PathHoleRepairConfig(
            resolution=128,
            normal_tolerance_deg=15.0,
            max_plane_error_m=0.02,
            max_hole_fraction=0.20,
        ),
    )
    paint = automatic["view_id_map"] > 0
    painted = build_path_hole_repair(
        mesh,
        mesh.hole_mask,
        source_camera=camera,
        camera_path=path,
        paint_mask=paint,
        config=PathHoleRepairConfig(
            resolution=128,
            selection_mode="paint_overlap",
            normal_tolerance_deg=15.0,
            max_plane_error_m=0.02,
            max_hole_fraction=0.20,
        ),
    )

    assert painted["selected_ids"] == automatic["visible_ids"]
    assert np.array_equal(painted["repair_mask"], automatic["repair_mask"])


def test_path_repair_exclusion_removes_background_connected_holes():
    mesh, camera, path = _fixture()
    result = build_path_hole_repair(
        mesh,
        mesh.hole_mask,
        source_camera=camera,
        camera_path=path,
        exclude_mask=mesh.hole_mask,
        config=PathHoleRepairConfig(
            resolution=128,
            normal_tolerance_deg=15.0,
            max_plane_error_m=0.02,
            max_hole_fraction=0.20,
        ),
    )

    assert not result["repair_mask"].any()
    assert not result["visible_ids"]
    assert "exclude mask removed" in result["report"]


def test_path_repair_empty_path_returns_observable_noop():
    mesh, camera, _path = _fixture()
    result = build_path_hole_repair(
        mesh,
        mesh.hole_mask,
        source_camera=camera,
        camera_path=AtlasCameraPath(),
    )

    assert result["frame_index"] == -1
    assert not result["repair_mask"].any()
    assert "no sampled frames" in result["report"]
