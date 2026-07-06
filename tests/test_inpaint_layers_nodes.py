"""Tests for the inpaint-layers nodes (AtlasDepthLayerMask, AtlasCleanPlateLayer)
and the depth-band clip they share with `build_relief_mesh`
(`atlas_camera/core/relief_mesh.py::band_min_m/band_max_m`, covered separately in
tests/test_relief_mesh.py).

Uses the same self-contained analytic ground+wall depth-map fixture pattern as
tests/test_derive_geometry_nodes.py (level camera at (0, h, 0), identity
rotation), extended with a second, nearer occluder box so occlusion-mask
behavior is meaningfully testable. Runs with only numpy/torch — no [neural]
extra or model download needed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasCleanPlateLayer,
    AtlasDepthLayerMask,
    _extract_blockout_camera,
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


def _occluder_depth(h=CAM_HEIGHT, near_wall_z=-3.0, near_wall_h=2.0,
                     near_col_lo=96, near_col_hi=160,
                     far_wall_z=-10.0, far_wall_h=3.0):
    """Ground plane (Y=0) + a FAR wall (background layer, far_wall_z) + a NEAR
    occluder (near_wall_z) restricted to a finite column band — a finite-width
    foreground object, like a real photo, rather than an infinite
    fronto-parallel wall spanning the whole frame. This leaves columns outside
    the occluder's footprint showing the ground/far-wall band through, while
    columns inside it are genuinely occluded — giving occlusion_mask something
    real (and spatially bounded) to detect."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY

    depth = np.full((H, W), SKY)

    t_ground = np.full((H, W), np.inf)
    looking_down = dy < -1e-6
    t_ground[looking_down] = -h / dy[looking_down]

    def _wall_visibility(wall_z, wall_h):
        t = -wall_z
        y_at = h + dy * t
        visible = (y_at >= 0.0) & (y_at <= wall_h)
        return t, visible

    t_far, vis_far = _wall_visibility(far_wall_z, far_wall_h)
    t_near, vis_near = _wall_visibility(near_wall_z, near_wall_h)
    vis_near = vis_near & (uu >= near_col_lo) & (uu <= near_col_hi)

    far_full = np.where(vis_far, t_far, SKY)
    near_full = np.where(vis_near, t_near, SKY)

    stacked = np.stack([
        depth,
        np.where(np.isfinite(t_ground), t_ground, SKY),
        far_full,
        near_full,
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


def _plate_image():
    return torch.rand(1, H, W, 3, dtype=torch.float32)


# --- registration -----------------------------------------------------------

def test_nodes_registered():
    assert NODE_CLASS_MAPPINGS["AtlasDepthLayerMask"].RETURN_TYPES == ("MASK", "MASK")
    assert NODE_CLASS_MAPPINGS["AtlasCleanPlateLayer"].RETURN_TYPES == ("ATLAS_SOLVE",)


# --- AtlasDepthLayerMask -----------------------------------------------------

def test_layer_mask_selects_background_band_only():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    # Background band: far wall (~10m) but not the near occluder (~3m).
    # feather_px=0 to test the raw (unfeathered) band logic in isolation —
    # feathering deliberately grows occlusion_mask past the band edge, so it's
    # tested separately below.
    layer_mask, occlusion_mask = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=0)

    layer_np = layer_mask[0].numpy()
    occ_np = occlusion_mask[0].numpy()
    assert layer_np.shape == (H, W)
    assert layer_np.sum() > 0
    # The near occluder (~3m, nearer than near_m=5.0) must show up as occlusion.
    assert occ_np.sum() > 0
    # A pixel can't be both in-band and occluding (near < near, far edges disjoint).
    assert not np.any((layer_np > 0) & (occ_np > 0))


def test_feather_px_grows_occlusion_mask():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    _layer0, occ0 = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0, feather_px=0)
    _layer4, occ4 = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0, feather_px=4)
    assert float(occ4.sum()) >= float(occ0.sum())


def test_occlusion_mask_only_marks_nearer_than_near_edge():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    _layer, occlusion_mask = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=0)
    occ_np = occlusion_mask[0].numpy()

    # Bottom-center pixel sits inside the near occluder's column footprint
    # (cols 96-160) — nearer than the 5m near edge, so it must be occlusion.
    assert occ_np[H - 1, W // 2] > 0.5
    # Same row range but OUTSIDE the occluder's column footprint (col 20):
    # ground shows through there at ~7.7m, inside [5,12] — not occlusion.
    assert occ_np[180, 20] == 0.0


def test_percentile_fallback_matches_manual_percentile():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)

    from atlas_camera.core.depth_geometry import detect_sky_mask
    from atlas_camera.core.relief_mesh import estimate_ground_scale

    extr = solve.camera.extrinsics
    scale, _ = estimate_ground_scale(
        depth_map, view_matrix=extr.camera_view_matrix, fx=FX, fy=FY, cx=CX, cy=CY)
    metric = depth_map.astype(np.float64) * scale
    valid = (np.isfinite(depth_map) & (depth_map > 1e-4)
             & ~detect_sky_mask(depth_map, horizon_y=H * 0.45))
    expected_far = float(np.percentile(metric[valid], 50.0))

    layer_mask, _occ = AtlasDepthLayerMask().generate(
        solve, depth, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5)
    layer_np = layer_mask[0].numpy()
    selected = metric[valid & (layer_np.astype(bool))]
    if selected.size:
        assert float(selected.max()) <= expected_far + 1e-6


def test_feather_does_not_wrap_around_image_borders():
    # Occlusion touching the bottom edge (the common case — a near-camera
    # foreground object usually extends to the bottom of frame) must not
    # bleed onto the opposite (top) edge via a wrapping dilation.
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    _layer, occlusion_mask = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=4)
    occ_np = occlusion_mask[0].numpy()

    assert occ_np[H - 1, :].any()          # bottom edge genuinely occluded
    assert not occ_np[0, :].any()          # top edge must NOT have wrapped in


def test_no_focal_length_returns_zero_masks():
    intr = AtlasIntrinsics(image_width=W, image_height=H)  # no fx_px
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    depth = _depth_result(_occluder_depth())

    layer_mask, occlusion_mask = AtlasDepthLayerMask().generate(solve, depth)
    assert layer_mask.shape == (1, H, W)
    assert occlusion_mask.shape == (1, H, W)
    assert float(layer_mask.sum()) == 0.0
    assert float(occlusion_mask.sum()) == 0.0


# --- AtlasCleanPlateLayer -----------------------------------------------------

def test_clean_plate_layer_appends_one_source_with_unchanged_camera():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    (out,) = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, name="bg", priority=0.0, relief_grid=32)

    assert len(solve.projection_sources) == 0        # input untouched
    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    assert src.name == "bg"
    assert src.metadata.get("projection_mode") == "clean_plate"
    assert src.proxy_geometry  # non-empty
    assert all((p.metadata or {}).get("role") == PROXY_ROLE for p in src.proxy_geometry)

    # Camera is the PRIMARY's, unchanged — no orbit applied.
    assert src.camera.extrinsics.camera_view_matrix == solve.camera.extrinsics.camera_view_matrix
    assert src.camera.intrinsics.fx_px == solve.camera.intrinsics.fx_px
    assert src.azimuth_deg == 0.0
    assert src.elevation_deg == 0.0
    assert src.distance_scale == 1.0


def test_clean_plate_layer_round_trips_through_solve_dict():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    (out,) = AtlasCleanPlateLayer().add_layer(solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)
    reloaded = AtlasSolve.from_dict(out.to_dict())

    assert len(reloaded.projection_sources) == 1
    assert reloaded.projection_sources[0].metadata.get("projection_mode") == "clean_plate"


def test_clean_plate_layer_passes_through_when_primary_has_no_focal():
    intr = AtlasIntrinsics(image_width=W, image_height=H)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    (out,) = AtlasCleanPlateLayer().add_layer(solve, depth, plate)
    assert out is solve
    assert len(out.projection_sources) == 0


def test_mask_band_and_clean_plate_mesh_stay_in_lockstep():
    """AtlasDepthLayerMask and AtlasCleanPlateLayer share _resolve_depth_band —
    calling both with identical band params on the same scene must produce a
    mesh whose vertex distances are consistent with the mask's own band."""
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    layer_mask, _occ = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0)
    (out,) = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)

    src = out.projection_sources[0]
    mesh_prim = next(p for p in src.proxy_geometry if p.primitive_type == "mesh")
    verts = np.array(mesh_prim.metadata["vertices"]).reshape(-1, 3)
    cam = np.array([0.0, CAM_HEIGHT, 0.0])
    dist = np.linalg.norm(verts - cam, axis=1)

    assert layer_mask.sum() > 0
    assert len(verts) > 0
    # Mesh vertices should fall within the requested band (with tear/edge slack).
    assert float(dist.min()) >= 5.0 - 1.0
    assert float(dist.max()) <= 12.0 + 1.0


# --- serialization (_extract_blockout_camera) --------------------------------

def test_serialization_includes_projection_mode_for_clean_plate():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()
    (out,) = AtlasCleanPlateLayer().add_layer(solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)

    image = torch.rand(1, H, W, 3, dtype=torch.float32)
    payload = _extract_blockout_camera(out, image, target_width=W, target_height=H)

    assert len(payload["projection_sources"]) == 1
    assert payload["projection_sources"][0]["projection_mode"] == "clean_plate"


def test_serialization_projection_mode_is_none_for_ordinary_patch():
    from atlas_camera.core.schema import ProjectionSource

    solve = _solve()
    solve.projection_sources.append(ProjectionSource(camera=solve.camera, name="ordinary_patch"))
    image = torch.rand(1, H, W, 3, dtype=torch.float32)
    payload = _extract_blockout_camera(solve, image, target_width=W, target_height=H)

    assert payload["projection_sources"][0]["projection_mode"] is None
