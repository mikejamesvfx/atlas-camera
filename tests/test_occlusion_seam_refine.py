import numpy as np

from atlas_camera.core.mesh_voxel import render_depth_grid
from atlas_camera.core.occlusion_seam_refine import (
    OcclusionSeamConfig,
    refine_occlusion_seams,
)
from atlas_camera.core.relief_mesh import build_relief_mesh


W = H = 65
FX = FY = 72.0
CX = CY = 32.0
VIEW = np.eye(4, dtype=np.float64)


def _two_layer_diagonal_relief():
    yy, xx = np.mgrid[:H, :W]
    split = 23.0 + 0.42 * yy
    depth = np.where(xx < split, 3.0, 10.0).astype(np.float32)
    mesh = build_relief_mesh(
        depth,
        view_matrix=VIEW,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        grid_long_edge=32,
        depth_edge_rel=0.20,
        smooth_iterations=0,
        floor_clamp=None,
        apply_sky_heuristic=False,
    )
    return mesh


def _forward_depth(vertices):
    homogeneous = np.concatenate(
        [np.asarray(vertices, dtype=np.float64),
         np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    return -(homogeneous @ VIEW.T)[:, 2]


def test_dual_sheet_strips_reduce_camera_holes_without_crossing_depth_layers():
    """A wrong implementation either leaves coverage unchanged or creates a
    near-to-far curtain face. The seam pass must append disconnected strips
    belonging wholly to the 3 m or 10 m sheet."""
    mesh = _two_layer_diagonal_relief()
    vertices_before = np.asarray(mesh.vertices).copy()
    uvs_before = np.asarray(mesh.uvs).copy()
    faces_before = np.asarray(mesh.faces).copy()
    mask = np.asarray(mesh.hole_mask, dtype=np.float32)
    covered_before = np.isfinite(render_depth_grid(
        mesh.vertices, mesh.faces, VIEW, FX, FY, CX, CY, W, H))

    refined, remaining, created, report = refine_occlusion_seams(
        mesh,
        mask,
        view_matrix=VIEW,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        image_width=W,
        image_height=H,
        config=OcclusionSeamConfig(
            seam_width_cells=2.0,
            smooth_iterations=8,
            max_chains=64,
            max_layer_depth_rel=0.08,
        ),
    )

    assert report["chains_refined"] >= 2
    assert report["faces_added"] > 0
    assert report["vertices_added"] > 0
    assert report["camera_mask_pixels_covered"] > 0
    assert np.count_nonzero(created) > 0
    assert np.count_nonzero(remaining) < np.count_nonzero(mask)

    # Existing geometry and its projective registration are immutable.
    np.testing.assert_array_equal(
        refined.vertices[: len(vertices_before)], vertices_before)
    np.testing.assert_array_equal(refined.uvs[: len(uvs_before)], uvs_before)
    np.testing.assert_array_equal(refined.faces[: len(faces_before)], faces_before)

    # Every appended face stays on one side of the discontinuity. Connecting
    # 3 m to 10 m would have a relative range of 7/3 and fail dramatically.
    added_faces = refined.faces[-report["faces_added"] :]
    depth = _forward_depth(refined.vertices)
    face_depth = depth[added_faces]
    relative_span = np.ptp(face_depth, axis=1) / np.maximum(
        np.min(face_depth, axis=1), 1e-6)
    assert float(relative_span.max()) <= 0.08 + 1e-6
    assert np.any(np.median(face_depth, axis=1) < 5.0)
    assert np.any(np.median(face_depth, axis=1) > 8.0)

    covered_after = np.isfinite(render_depth_grid(
        refined.vertices, refined.faces, VIEW, FX, FY, CX, CY, W, H))
    assert np.count_nonzero(covered_after & ~covered_before) > 0

    # Newly-created UVs are still the exact projection of their vertices.
    new_vertices = refined.vertices[len(vertices_before) :]
    new_uvs = refined.uvs[len(uvs_before) :]
    z = -new_vertices[:, 2]
    expected_u = (FX * new_vertices[:, 0] / z + CX) / (W - 1)
    expected_v = 1.0 - (-FY * new_vertices[:, 1] / z + CY) / (H - 1)
    np.testing.assert_allclose(new_uvs[:, 0], expected_u, atol=2e-5)
    np.testing.assert_allclose(new_uvs[:, 1], expected_v, atol=2e-5)


def test_frame_boundary_is_never_extended_into_a_synthetic_shell():
    """Dropping the frame exclusion would turn the image perimeter into a
    camera-frustum skirt—the Snowglobe failure under another name."""
    mesh = _two_layer_diagonal_relief()
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :8] = 1.0

    refined, remaining, created, report = refine_occlusion_seams(
        mesh,
        mask,
        view_matrix=VIEW,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        image_width=W,
        image_height=H,
        config=OcclusionSeamConfig(seam_width_cells=2.0),
    )

    np.testing.assert_array_equal(refined.vertices, mesh.vertices)
    np.testing.assert_array_equal(refined.faces, mesh.faces)
    np.testing.assert_array_equal(refined.uvs, mesh.uvs)
    np.testing.assert_array_equal(remaining, mask.astype(bool))
    assert not np.any(created)
    assert report["faces_added"] == 0


def test_outer_contour_recedes_along_camera_minus_z_instead_of_staying_y_up():
    """Copying the source depth unchanged makes the new strip a lateral X/Y
    ribbon.  Every outer vertex must sit behind its paired source-rim vertex
    in camera forward depth while retaining its new projected position."""
    mesh = _two_layer_diagonal_relief()
    refined, _, _, report = refine_occlusion_seams(
        mesh,
        np.asarray(mesh.hole_mask, dtype=np.float32),
        view_matrix=VIEW,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        image_width=W,
        image_height=H,
        config=OcclusionSeamConfig(
            seam_width_cells=2.0,
            smooth_iterations=8,
            max_chains=64,
            max_layer_depth_rel=0.08,
        ),
    )

    added = refined.faces[-report["faces_added"] :]
    # The first triangle of each zipper pair is [next_source, source, outer].
    source_outer_pairs = added[::2, 1:3]
    forward = _forward_depth(refined.vertices)
    recession = (
        forward[source_outer_pairs[:, 1]]
        - forward[source_outer_pairs[:, 0]]
    )
    assert float(recession.min()) > 0.0


def test_global_away_from_camera_direction_has_no_camera_xy_displacement():
    """Per-boundary screen normals create the observed Y-up shelves.  Global
    camera-forward mode must displace every outer vertex only along camera -Z."""
    mesh = _two_layer_diagonal_relief()
    refined, _, _, report = refine_occlusion_seams(
        mesh,
        np.asarray(mesh.hole_mask, dtype=np.float32),
        view_matrix=VIEW,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        image_width=W,
        image_height=H,
        config=OcclusionSeamConfig(
            seam_width_cells=2.0,
            smooth_iterations=8,
            max_chains=64,
            max_layer_depth_rel=0.08,
            global_direction="away_from_camera",
        ),
    )

    added = refined.faces[-report["faces_added"] :]
    pairs = added[::2, 1:3]
    displacement = (
        np.asarray(refined.vertices)[pairs[:, 1]]
        - np.asarray(refined.vertices)[pairs[:, 0]]
    )
    np.testing.assert_allclose(displacement[:, :2], 0.0, atol=2e-6)
    assert float(displacement[:, 2].max()) < 0.0
