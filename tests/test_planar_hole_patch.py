"""Normal-guided relief-hole patch regression coverage."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from atlas_camera.core.planar_hole_patch import (
    PlanarHolePatchConfig,
    _component_depth_diagnostic,
    patch_planar_holes,
)
from atlas_camera.core.proxy_geometry import relief_mesh_primitive
from atlas_camera.core.relief_mesh import build_relief_mesh
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasSolve,
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
PLANE_NORMAL = np.asarray((0.20, 0.10, 0.974679434), dtype=np.float64)
PLANE_OFFSET = -5.0


def _tilted_depth():
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    rays = np.stack([
        (uu - CX) / FX,
        -(vv - CY) / FY,
        -np.ones_like(uu),
    ], axis=-1)
    return (PLANE_OFFSET / (rays @ PLANE_NORMAL)).astype(np.float32)


def _mesh_with_hole(*, touches_frame=False):
    exclusion = np.zeros((H, W), dtype=bool)
    if touches_frame:
        exclusion[0:12, 25:40] = True
    else:
        exclusion[27:38, 27:38] = True
    mesh = build_relief_mesh(
        _tilted_depth(),
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
    return mesh


def _patch(mesh, mask, **overrides):
    config = PlanarHolePatchConfig(
        ring_cells=overrides.pop("ring_cells", 2),
        max_components=overrides.pop("max_components", 8),
        normal_tolerance_deg=overrides.pop("normal_tolerance_deg", 15.0),
        max_plane_error_m=overrides.pop("max_plane_error_m", 0.02),
        max_hole_fraction=overrides.pop("max_hole_fraction", 0.20),
        enclosed_only=overrides.pop("enclosed_only", True),
        **overrides,
    )
    return patch_planar_holes(
        mesh,
        mask,
        view_matrix=VIEW,
        fx=FX, fy=FY, cx=CX, cy=CY,
        image_width=W,
        image_height=H,
        config=config,
    )


def test_patch_fills_enclosed_hole_on_boundary_plane_with_exact_uvs():
    mesh = _mesh_with_hole()
    n_vertices_before = len(mesh.vertices)
    n_holes_before = int(mesh.hole_mask.sum())

    patched, remaining, report = _patch(mesh, mesh.hole_mask)

    assert report["components_filled"] == 1
    assert report["components_rejected"] == 0
    assert report["vertices_added"] > 0
    assert report["faces_added"] > report["faces_removed"]
    assert int(remaining.sum()) < n_holes_before

    added = np.asarray(patched.vertices[n_vertices_before:], dtype=np.float64)
    assert len(added) == report["vertices_added"]
    residual = np.abs(added @ PLANE_NORMAL - PLANE_OFFSET)
    # The patch deliberately uses the average of agreeing relief normals
    # rather than an SVD refit, so retain centimetre-level agreement with the
    # analytic plane while pinning the requested orientation behavior.
    assert float(residual.max()) < 2e-2

    added_uv = np.asarray(patched.uvs[n_vertices_before:], dtype=np.float64)
    depth = -added[:, 2]
    projected_u = (FX * added[:, 0] / depth + CX) / (W - 1)
    projected_v = 1.0 - (-FY * added[:, 1] / depth + CY) / (H - 1)
    assert np.allclose(added_uv[:, 0], projected_u, atol=1e-6)
    assert np.allclose(added_uv[:, 1], projected_v, atol=1e-6)

    new_faces = patched.faces[-report["faces_added"]:]
    assert np.any(new_faces < n_vertices_before)  # shared perimeter indices
    assert np.any(new_faces >= n_vertices_before)
    assert len(patched.edge_risk) == len(patched.vertices)
    assert np.all(patched.edge_risk[np.unique(new_faces)] == 0.0)
    assert report["filled"][0]["patch_edge_factor"] < 20.0
    assert report["filled"][0]["local_support_world_per_pixel"] > 0.0
    assert report["filled"][0]["worst_edge_pixel_span"] > 0.0


def test_patch_edge_factor_rejects_unsafe_generated_geometry_atomically():
    mesh = _mesh_with_hole()

    patched, remaining, report = _patch(
        mesh, mesh.hole_mask, max_patch_edge_factor=1.0)

    assert report["components_filled"] == 0
    assert report["vertices_added"] == 0
    assert report["faces_added"] == 0
    assert np.array_equal(patched.vertices, mesh.vertices)
    assert np.array_equal(patched.faces, mesh.faces)
    assert np.array_equal(remaining, mesh.hole_mask)
    rejection = next(
        item for item in report["rejected"]
        if item["reason"] == "generated patch edge exceeds local scale"
    )
    assert rejection["patch_edge_factor"] > 1.0
    assert rejection["patch_edge_max_m"] > 0.0
    assert rejection["local_support_edge_m"] > 0.0


def test_patch_depth_factor_catches_forward_and_backward_z_needles():
    support = np.asarray([
        (-1.0, -1.0, -5.0),
        (1.0, -1.0, -5.2),
        (-1.0, 1.0, -4.8),
        (1.0, 1.0, -5.0),
    ])
    generated = np.asarray([
        (0.0, 0.0, -0.25),
        (0.5, 0.0, -20.0),
    ])

    diagnostic = _component_depth_diagnostic(generated, support, VIEW)
    scaled = _component_depth_diagnostic(
        generated * 10.0, support * 10.0, VIEW)

    assert diagnostic["patch_depth_factor"] > 10.0
    assert scaled["patch_depth_factor"] == pytest.approx(
        diagnostic["patch_depth_factor"])
    assert diagnostic["generated_depth_min_m"] == pytest.approx(0.25)
    assert diagnostic["generated_depth_max_m"] == pytest.approx(20.0)


def test_patch_edge_factor_groups_source_pixels_with_world_camera_scale():
    mesh = _mesh_with_hole()
    _base, _base_remaining, base_report = _patch(mesh, mesh.hole_mask)

    scale_px = 8
    hi_width = (W - 1) * scale_px + 1
    hi_height = (H - 1) * scale_px + 1
    source_x = np.rint(
        np.arange(hi_width, dtype=np.float64) / scale_px).astype(np.int64)
    source_y = np.rint(
        np.arange(hi_height, dtype=np.float64) / scale_px).astype(np.int64)
    hi_mask = mesh.hole_mask[np.ix_(source_y, source_x)]
    _hi, _hi_remaining, hi_report = patch_planar_holes(
        mesh,
        hi_mask,
        view_matrix=VIEW,
        fx=FX * scale_px,
        fy=FY * scale_px,
        cx=CX * scale_px,
        cy=CY * scale_px,
        image_width=hi_width,
        image_height=hi_height,
        config=PlanarHolePatchConfig(
            ring_cells=2,
            max_components=8,
            normal_tolerance_deg=15.0,
            max_plane_error_m=0.02,
            max_hole_fraction=0.20,
            enclosed_only=True,
        ),
    )

    world_scale = 10.0
    scaled_mesh = copy.deepcopy(mesh)
    scaled_mesh.vertices = np.asarray(
        scaled_mesh.vertices, dtype=np.float32) * world_scale
    _scaled, _scaled_remaining, scaled_report = _patch(
        scaled_mesh,
        scaled_mesh.hole_mask,
        max_plane_error_m=0.02 * world_scale,
    )

    base_factor = base_report["filled"][0]["patch_edge_factor"]
    assert hi_report["components_filled"] == 1
    assert scaled_report["components_filled"] == 1
    assert hi_report["filled"][0]["patch_edge_factor"] == pytest.approx(
        base_factor, rel=1e-5)
    assert scaled_report["filled"][0]["patch_edge_factor"] == pytest.approx(
        base_factor, rel=1e-5)
    assert hi_report["filled"][0]["worst_edge_pixel_span"] == pytest.approx(
        base_report["filled"][0]["worst_edge_pixel_span"] * scale_px)
    assert scaled_report["filled"][0]["local_support_world_per_pixel"] == (
        pytest.approx(
            base_report["filled"][0]["local_support_world_per_pixel"]
            * world_scale,
            rel=1e-5,
        )
    )


def test_patch_rejects_open_frame_gap_by_default():
    mesh = _mesh_with_hole(touches_frame=True)
    patched, remaining, report = _patch(mesh, mesh.hole_mask)

    assert report["components_filled"] == 0
    assert report["components_rejected"] >= 1
    assert any(item["reason"] == "touches image frame"
               for item in report["rejected"])
    assert np.array_equal(remaining, mesh.hole_mask)
    assert np.array_equal(patched.faces, mesh.faces)


def test_parallel_surfaces_choose_the_densest_plane_position_band():
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    rays = np.stack([
        (uu - CX) / FX,
        -(vv - CY) / FY,
        -np.ones_like(uu),
    ], axis=-1)
    offsets = np.where(uu < CX, -4.0, -6.0)
    depth = offsets / rays[..., 2]
    exclusion = np.zeros((H, W), dtype=bool)
    exclusion[27:38, 27:38] = True
    mesh = build_relief_mesh(
        depth.astype(np.float32),
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

    patched, remaining, report = _patch(
        mesh,
        mesh.hole_mask,
        max_plane_error_m=0.05,
        max_patch_edge_factor=32.0,
    )

    assert report["components_filled"] == 1
    assert report["components_rejected"] == 0
    assert 0.40 < report["filled"][0]["plane_support_fraction"] < 0.70
    added = np.asarray(
        patched.vertices[len(mesh.vertices):], dtype=np.float64)
    assert np.allclose(np.median(added[:, 2]), -6.0, atol=0.05)
    assert int(remaining.sum()) < int(mesh.hole_mask.sum())


def test_one_sided_planar_support_is_artist_controllable():
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    rays = np.stack([
        (uu - CX) / FX,
        -(vv - CY) / FY,
        -np.ones_like(uu),
    ], axis=-1)
    other_normal = np.asarray((-0.45, 0.05, 0.8916), dtype=np.float64)
    other_normal /= np.linalg.norm(other_normal)
    left_depth = PLANE_OFFSET / (rays @ PLANE_NORMAL)
    right_depth = -6.0 / (rays @ other_normal)
    depth = np.where(uu < CX, left_depth, right_depth)
    exclusion = np.zeros((H, W), dtype=bool)
    exclusion[27:38, 27:38] = True
    mesh = build_relief_mesh(
        depth.astype(np.float32),
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

    _strict, strict_remaining, strict_report = _patch(
        mesh,
        mesh.hole_mask,
        min_normal_support_fraction=0.60,
        max_plane_error_m=0.03,
    )
    patched, remaining, report = _patch(
        mesh,
        mesh.hole_mask,
        min_normal_support_fraction=0.30,
        max_plane_error_m=0.03,
    )

    assert strict_report["components_filled"] == 0
    assert strict_report["rejected"][0]["reason"] == (
        "normal consensus below threshold")
    assert np.array_equal(strict_remaining, mesh.hole_mask)
    assert report["components_filled"] == 1
    assert report["filled"][0]["normal_support_fraction"] < 0.60
    assert report["filled"][0]["plane_support_fraction"] > 0.0
    assert int(remaining.sum()) < int(mesh.hole_mask.sum())
    assert len(patched.faces) > len(mesh.faces)


def test_frame_components_do_not_consume_the_fit_budget():
    exclusion = np.zeros((H, W), dtype=bool)
    exclusion[0:12, 25:40] = True
    exclusion[27:38, 27:38] = True
    mesh = build_relief_mesh(
        _tilted_depth(),
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

    _patched, remaining, report = _patch(
        mesh, mesh.hole_mask, max_components=1)

    assert report["components_found"] == 2
    assert report["components_eligible"] == 1
    assert report["components_attempted"] == 1
    assert report["components_budget_skipped"] == 0
    assert report["components_filled"] == 1
    assert any(item["reason"] == "touches image frame"
               for item in report["rejected"])
    assert int(remaining.sum()) < int(mesh.hole_mask.sum())


def test_fit_budget_selects_the_smallest_eligible_island_first():
    exclusion = np.zeros((H, W), dtype=bool)
    exclusion[9:14, 9:14] = True
    exclusion[27:39, 27:39] = True
    mesh = build_relief_mesh(
        _tilted_depth(),
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

    _patched, _remaining, report = _patch(
        mesh, mesh.hole_mask, max_components=1)

    budget_rejections = [
        item for item in report["rejected"]
        if item["reason"] == "component budget exceeded"
    ]
    assert report["components_eligible"] == 2
    assert report["components_attempted"] == 1
    assert report["components_budget_skipped"] == 1
    assert report["components_filled"] == 1
    assert len(budget_rejections) == 1
    assert report["filled"][0]["cells"] < budget_rejections[0]["cells"]


def test_filled_islands_are_reported_smallest_to_largest():
    exclusion = np.zeros((H, W), dtype=bool)
    exclusion[7:12, 7:12] = True
    exclusion[24:32, 24:32] = True
    exclusion[43:55, 43:55] = True
    mesh = build_relief_mesh(
        _tilted_depth(),
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

    _patched, _remaining, report = _patch(
        mesh, mesh.hole_mask, max_components=3, max_plane_error_m=0.03)

    filled_sizes = [item["cells"] for item in report["filled"]]
    assert report["components_eligible"] == 3
    assert report["components_attempted"] == 3
    assert report["components_filled"] == 3
    assert filled_sizes == sorted(filled_sizes)
    assert len(set(filled_sizes)) == 3


def test_comfy_node_stitches_patch_into_one_retopologizable_relief_mesh():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import (
        AtlasPlanarHolePatch,
        AtlasRetopologizeLayer,
        NODE_CLASS_MAPPINGS,
    )

    mesh = _mesh_with_hole()
    intr = AtlasIntrinsics(
        image_width=W, image_height=H,
        fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
        focal_length_mm=35.0, sensor_width_mm=36.0,
    )
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=intr,
        extrinsics=AtlasExtrinsics(camera_view_matrix=VIEW),
    ))
    solve.projection_scene.proxy_geometry = [relief_mesh_primitive(mesh)]
    mask = torch.from_numpy(mesh.hole_mask.astype(np.float32)).unsqueeze(0)

    out, remaining, report, created = AtlasPlanarHolePatch().patch(
        solve,
        mask,
        normal_tolerance_deg=15.0,
        max_plane_error_m=0.02,
        max_hole_fraction=0.20,
    )
    relief = [
        prim for prim in out.projection_scene.proxy_geometry
        if (prim.metadata or {}).get("source") == "depth_relief_mesh"
    ]
    assert NODE_CLASS_MAPPINGS["AtlasPlanarHolePatch"] is AtlasPlanarHolePatch
    patch_inputs = AtlasPlanarHolePatch.INPUT_TYPES()["optional"]
    assert patch_inputs["max_components"][1]["default"] == 64
    assert patch_inputs["max_components"][1]["max"] == 1024
    assert patch_inputs["min_normal_support_fraction"][1]["default"] == 0.30
    assert patch_inputs["max_patch_edge_factor"][1]["default"] == 20.0
    assert patch_inputs["max_patch_depth_factor"][1]["default"] == 2.0
    assert len(relief) == 1
    assert relief[0].metadata["planar_hole_patch"]["components_filled"] == 1
    assert int(remaining.sum()) < int(mask.sum())
    assert int(created.sum()) == int(mask.sum()) - int(remaining.sum())
    assert torch.count_nonzero(created * remaining) == 0
    assert AtlasPlanarHolePatch.RETURN_NAMES[-1] == "created_islands"
    assert "filled 1/1" in report
    assert "source pixels × local m/px" in report
    assert "local support depth band" in report

    _retopo_out, retopo_report = AtlasRetopologizeLayer().retopo(
        out, method="off")
    # Retopo now labels each primitive it touched (scope/name), so a drawn
    # plane in the union is visible rather than folded into "primary".
    assert "primary/projection_relief_mesh: unchanged" in retopo_report


def test_wall_and_tower_nodes_default_to_no_foreground_cubes():
    from atlas_camera.comfy.nodes import AtlasDeriveTowersSpires, AtlasDeriveWalls

    assert AtlasDeriveWalls.INPUT_TYPES()["optional"]["max_objects"][1]["default"] == 0
    assert AtlasDeriveTowersSpires.INPUT_TYPES()["optional"]["max_objects"][1]["default"] == 0


# ---------------------------------------------------------------------------
# whole-solve sweep — the AtlasLiveMeshRepair replacement
# ---------------------------------------------------------------------------

def _multi_layer_solve():
    """Primary relief mesh + two named ProjectionSources, each with a hole."""
    from atlas_camera.core.schema import ProjectionSource

    intr = AtlasIntrinsics(
        image_width=W, image_height=H, fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
        focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = LatentCamera(intrinsics=intr,
                       extrinsics=AtlasExtrinsics(camera_view_matrix=VIEW))
    solve = AtlasSolve(camera=cam)
    solve.projection_scene.proxy_geometry = [relief_mesh_primitive(_mesh_with_hole())]
    for name in ("machine", "foreground"):
        solve.projection_sources.append(ProjectionSource(
            camera=copy.deepcopy(cam), name=name,
            proxy_geometry=[relief_mesh_primitive(_mesh_with_hole())]))
    return solve


def _filled_counts(solve):
    out = {}
    for prim in solve.projection_scene.proxy_geometry:
        rep = (prim.metadata or {}).get("planar_hole_patch")
        if rep:
            out["primary"] = rep["components_filled"]
    for src in solve.projection_sources:
        for prim in src.proxy_geometry:
            rep = (prim.metadata or {}).get("planar_hole_patch")
            if rep:
                out[src.name] = rep["components_filled"]
    return out


def test_layer_star_sweeps_every_relief_mesh():
    """One node replaces one-patch-per-layer: '*' walks primary + all sources.

    AtlasLiveMeshRepair swept the whole solve; AtlasPlanarHolePatch targets one
    layer. Without '*' a faithful migration of a 3-layer graph needs three
    patch nodes.
    """
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasPlanarHolePatch

    solve = _multi_layer_solve()
    out, remaining, report, created = AtlasPlanarHolePatch().patch(
        solve, None, layer="*",
        normal_tolerance_deg=15.0, max_plane_error_m=0.02, max_hole_fraction=0.20)

    filled = _filled_counts(out)
    assert set(filled) == {"primary", "machine", "foreground"}
    assert all(v == 1 for v in filled.values()), filled
    assert "swept 3 layer(s)" in report
    for label in ("primary", "machine", "foreground"):
        assert f"[{label}]" in report
    assert int(created.sum()) > 0


def test_layer_star_mask_algebra_is_intersection_and_union():
    """A hole filled by one layer is still open in another's `remaining`, so
    the sweep's remaining is the INTERSECTION and created the UNION."""
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasPlanarHolePatch

    solve = _multi_layer_solve()
    _out, remaining, _report, created = AtlasPlanarHolePatch().patch(
        solve, None, layer="*",
        normal_tolerance_deg=15.0, max_plane_error_m=0.02, max_hole_fraction=0.20)
    import torch
    assert torch.count_nonzero(created * remaining) == 0


def test_unwired_mask_treats_every_hole_as_a_candidate():
    """Omitting hole_mask must sweep, not no-op.

    Before this, an absent mask resolved to all-zeros, so the node silently
    filled nothing — the worst possible default for a whole-solve sweep.
    """
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasPlanarHolePatch

    mesh = _mesh_with_hole()
    intr = AtlasIntrinsics(image_width=W, image_height=H, fx_px=FX, fy_px=FY,
                           cx_px=CX, cy_px=CY, focal_length_mm=35.0,
                           sensor_width_mm=36.0)
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=intr, extrinsics=AtlasExtrinsics(camera_view_matrix=VIEW)))
    solve.projection_scene.proxy_geometry = [relief_mesh_primitive(mesh)]

    out, _remaining, report, created = AtlasPlanarHolePatch().patch(
        solve, None,
        normal_tolerance_deg=15.0, max_plane_error_m=0.02, max_hole_fraction=0.20)
    relief = [p for p in out.projection_scene.proxy_geometry
              if (p.metadata or {}).get("source") == "depth_relief_mesh"]
    assert relief[0].metadata["planar_hole_patch"]["components_filled"] == 1
    assert int(created.sum()) > 0
    assert "filled 1/1" in report


def test_hole_mask_is_optional_in_input_types():
    from atlas_camera.comfy.nodes import AtlasPlanarHolePatch

    it = AtlasPlanarHolePatch.INPUT_TYPES()
    assert list(it["required"]) == ["solve"]
    assert "hole_mask" in it["optional"]


def test_layer_star_combines_masks_across_mismatched_layer_resolutions():
    """Layers do NOT share a plate size.

    A frame-outpainted clean plate is padded, so its camera is larger than the
    primary's — measured live at 4640x7808 beside a 4512x7680 primary. Each
    per-layer pass returns its mask at ITS OWN resolution, so combining them
    raw raised "operands could not be broadcast together". The original
    fixture gave every layer an identical camera and never caught it.
    """
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasPlanarHolePatch
    from atlas_camera.core.schema import ProjectionSource

    intr = AtlasIntrinsics(image_width=W, image_height=H, fx_px=FX, fy_px=FY,
                           cx_px=CX, cy_px=CY, focal_length_mm=35.0,
                           sensor_width_mm=36.0)
    cam = LatentCamera(intrinsics=intr,
                       extrinsics=AtlasExtrinsics(camera_view_matrix=VIEW))
    solve = AtlasSolve(camera=cam)
    solve.projection_scene.proxy_geometry = [relief_mesh_primitive(_mesh_with_hole())]

    # A padded layer: bigger plate, shifted principal point — exactly what
    # frame_outpaint_px produces on a clean-plate layer.
    pad = 16
    big = AtlasIntrinsics(image_width=W + 2 * pad, image_height=H + 2 * pad,
                          fx_px=FX, fy_px=FY, cx_px=CX + pad, cy_px=CY + pad,
                          focal_length_mm=35.0, sensor_width_mm=36.0)
    solve.projection_sources.append(ProjectionSource(
        camera=LatentCamera(intrinsics=big,
                            extrinsics=AtlasExtrinsics(camera_view_matrix=VIEW)),
        name="outpainted",
        proxy_geometry=[relief_mesh_primitive(_mesh_with_hole())]))

    out, remaining, report, created = AtlasPlanarHolePatch().patch(
        solve, None, layer="*",
        normal_tolerance_deg=15.0, max_plane_error_m=0.02, max_hole_fraction=0.20)

    # Combined masks come back in the SOLVE's frame, not a layer's.
    assert tuple(remaining.shape) == (1, H, W)
    assert tuple(created.shape) == (1, H, W)
    assert "swept 2 layer(s)" in report
