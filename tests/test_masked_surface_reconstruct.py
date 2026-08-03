import numpy as np

from atlas_camera.core.masked_surface_reconstruct import (
    MaskedSurfaceReconstructConfig,
    reconstruct_masked_surface,
)
from atlas_camera.core.mesh_repair import boundary_edges, walk_loops
from atlas_camera.core.relief_mesh import build_relief_mesh


def _relief_fixture(size: int = 65):
    yy, xx = np.mgrid[:size, :size]
    depth = (
        6.0
        + 0.003 * (xx.astype(np.float32) - size / 2.0) ** 2
        + 0.002 * (yy.astype(np.float32) - size / 2.0) ** 2
    ).astype(np.float32)
    mesh = build_relief_mesh(
        depth,
        view_matrix=np.eye(4, dtype=np.float32),
        fx=70.0,
        fy=72.0,
        cx=(size - 1) / 2.0,
        cy=(size - 1) / 2.0,
        grid_long_edge=32,
        smooth_iterations=0,
        far_clip_percentile=100.0,
        depth_edge_rel=10.0,
        max_edge_factor=0.0,
        floor_clamp=None,
        apply_sky_heuristic=False,
    )
    return mesh, depth


def _interior_boundary_loops(mesh):
    loops = walk_loops(boundary_edges(mesh.faces))
    interior = []
    for loop in loops:
        uv = mesh.uvs[np.asarray(loop, dtype=np.int64)]
        on_frame = (
            np.isclose(uv[:, 0], 0.0, atol=1e-5)
            | np.isclose(uv[:, 0], 1.0, atol=1e-5)
            | np.isclose(uv[:, 1], 0.0, atol=1e-5)
            | np.isclose(uv[:, 1], 1.0, atol=1e-5)
        )
        if not np.all(on_frame):
            interior.append(loop)
    return interior


def test_mask_can_manufacture_a_rim_in_an_intact_mesh_and_reconstruct_it():
    mesh, depth = _relief_fixture()
    mask = np.zeros(depth.shape, dtype=np.float32)
    mask[25:40, 24:41] = 1.0

    rebuilt, remaining, created, report = reconstruct_masked_surface(
        mesh,
        mask,
        view_matrix=np.eye(4, dtype=np.float32),
        fx=70.0,
        fy=72.0,
        cx=32.0,
        cy=32.0,
        image_width=65,
        image_height=65,
        config=MaskedSurfaceReconstructConfig(
            rim_cells=1,
            max_hole_fraction=0.20,
            smooth_iterations=128,
        ),
    )

    assert report["components_reconstructed"] == 1
    assert report["faces_removed"] > 0
    assert report["faces_added"] > 0
    assert report["vertices_added"] > 0
    assert np.count_nonzero(created) > np.count_nonzero(mask)
    assert np.count_nonzero(remaining[25:40, 24:41]) == 0

    # Existing geometry is immutable. The reconstructed interior is appended.
    np.testing.assert_array_equal(rebuilt.vertices[: len(mesh.vertices)], mesh.vertices)
    np.testing.assert_array_equal(rebuilt.uvs[: len(mesh.uvs)], mesh.uvs)
    assert not _interior_boundary_loops(rebuilt)

    # New vertices remain projective and their depth cannot overshoot the rim.
    new_vertices = rebuilt.vertices[len(mesh.vertices) :]
    new_uvs = rebuilt.uvs[len(mesh.uvs) :]
    z_forward = -new_vertices[:, 2]
    expected_u = (70.0 * (new_vertices[:, 0] / z_forward) + 32.0) / 64.0
    expected_v = 1.0 - (-72.0 * (new_vertices[:, 1] / z_forward) + 32.0) / 64.0
    np.testing.assert_allclose(new_uvs[:, 0], expected_u, atol=2e-5)
    np.testing.assert_allclose(new_uvs[:, 1], expected_v, atol=2e-5)
    assert z_forward.min() >= report["support_depth_min"] - 1e-5
    assert z_forward.max() <= report["support_depth_max"] + 1e-5


def test_frame_touching_mask_is_rejected_without_mutating_the_mesh():
    mesh, depth = _relief_fixture()
    mask = np.zeros(depth.shape, dtype=np.float32)
    mask[:12, 20:44] = 1.0

    rebuilt, remaining, created, report = reconstruct_masked_surface(
        mesh,
        mask,
        view_matrix=np.eye(4, dtype=np.float32),
        fx=70.0,
        fy=72.0,
        cx=32.0,
        cy=32.0,
        image_width=65,
        image_height=65,
        config=MaskedSurfaceReconstructConfig(rim_cells=1),
    )

    np.testing.assert_array_equal(rebuilt.vertices, mesh.vertices)
    np.testing.assert_array_equal(rebuilt.faces, mesh.faces)
    np.testing.assert_array_equal(rebuilt.uvs, mesh.uvs)
    np.testing.assert_array_equal(remaining, mask)
    assert not np.any(created)
    assert report["components_reconstructed"] == 0
    assert report["components_rejected"] == 1
    assert report["component_records"][0]["reason"] == "touches_frame"


def test_comfy_node_is_explicit_mask_driven_and_has_no_blender_dependency():
    from atlas_camera.comfy.nodes import (
        AtlasMaskedSurfaceReconstruct,
        EXPERIMENTAL_NODE_CLASS_MAPPINGS,
    )

    inputs = AtlasMaskedSurfaceReconstruct.INPUT_TYPES()
    assert tuple(inputs["required"]) == ("solve", "hole_mask")
    assert "rim_cells" in inputs["optional"]
    assert "smooth_iterations" in inputs["optional"]
    assert "blender_path" not in inputs["optional"]
    assert AtlasMaskedSurfaceReconstruct.RETURN_NAMES == (
        "solve", "remaining_holes", "created_region", "report"
    )
    assert (
        EXPERIMENTAL_NODE_CLASS_MAPPINGS["AtlasMaskedSurfaceReconstruct"]
        is AtlasMaskedSurfaceReconstruct
    )
