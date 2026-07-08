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
    AtlasSkyDomeLayer,
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
    assert NODE_CLASS_MAPPINGS["AtlasDepthLayerMask"].RETURN_TYPES == ("MASK", "MASK", "MASK")
    assert NODE_CLASS_MAPPINGS["AtlasCleanPlateLayer"].RETURN_TYPES == ("ATLAS_SOLVE", "MASK", "MASK")
    assert NODE_CLASS_MAPPINGS["AtlasSkyDomeLayer"].RETURN_TYPES == ("ATLAS_SOLVE", "MASK", "MASK")


# --- AtlasDepthLayerMask -----------------------------------------------------

def test_layer_mask_selects_background_band_only():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    # Background band: far wall (~10m) but not the near occluder (~3m).
    # feather_px=0 to test the raw (unfeathered) band logic in isolation —
    # feathering deliberately grows occlusion_mask past the band edge, so it's
    # tested separately below.
    layer_mask, occlusion_mask, _hole = AtlasDepthLayerMask().generate(
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
    _layer0, occ0, _hole0 = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0, feather_px=0)
    _layer4, occ4, _hole4 = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0, feather_px=4)
    assert float(occ4.sum()) >= float(occ0.sum())


def test_occlusion_mask_only_marks_nearer_than_near_edge():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    _layer, occlusion_mask, _hole = AtlasDepthLayerMask().generate(
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

    layer_mask, _occ, _hole = AtlasDepthLayerMask().generate(
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
    _layer, occlusion_mask, _hole = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=4)
    occ_np = occlusion_mask[0].numpy()

    assert occ_np[H - 1, :].any()          # bottom edge genuinely occluded
    assert not occ_np[0, :].any()          # top edge must NOT have wrapped in


def test_no_focal_length_returns_zero_masks():
    intr = AtlasIntrinsics(image_width=W, image_height=H)  # no fx_px
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    depth = _depth_result(_occluder_depth())

    layer_mask, occlusion_mask, hole_mask = AtlasDepthLayerMask().generate(solve, depth)
    assert layer_mask.shape == (1, H, W)
    assert occlusion_mask.shape == (1, H, W)
    assert hole_mask.shape == (1, H, W)
    assert float(layer_mask.sum()) == 0.0
    assert float(occlusion_mask.sum()) == 0.0
    assert float(hole_mask.sum()) == 0.0


def test_exclude_mask_removes_pixel_from_layer_and_marks_hole():
    # An external exclude_mask (e.g. a real SAM/RMBG sky segmentation) must
    # remove a pixel from layer_mask/occlusion_mask (can't belong to any band
    # once excluded) AND show up in hole_mask when compute_hole_mask=True.
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    exclude = torch.zeros(1, H, W, dtype=torch.float32)
    exclude[0, 180, 20] = 1.0  # normally inside the [5,12]m band (far ground)

    layer_mask, occlusion_mask, hole_mask = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=0,
        compute_hole_mask=True, relief_grid=64, exclude_mask=exclude)

    assert float(layer_mask[0, 180, 20]) == 0.0
    assert float(occlusion_mask[0, 180, 20]) == 0.0
    assert float(hole_mask[0, 180, 20]) == 1.0


def test_exclude_mask_none_is_backward_compatible():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    baseline = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0, feather_px=0)
    explicit_none = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=0, exclude_mask=None)
    for a, b in zip(baseline, explicit_none):
        assert torch.equal(a, b)


# --- AtlasCleanPlateLayer -----------------------------------------------------

def test_clean_plate_layer_appends_one_source_with_unchanged_camera():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    out, _hole_mask, _ext = AtlasCleanPlateLayer().add_layer(
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

    out, _hole_mask, _ext = AtlasCleanPlateLayer().add_layer(solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)
    reloaded = AtlasSolve.from_dict(out.to_dict())

    assert len(reloaded.projection_sources) == 1
    assert reloaded.projection_sources[0].metadata.get("projection_mode") == "clean_plate"


def test_clean_plate_layer_passes_through_when_primary_has_no_focal():
    intr = AtlasIntrinsics(image_width=W, image_height=H)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    out, hole_mask, _ext = AtlasCleanPlateLayer().add_layer(solve, depth, plate)
    assert out is solve
    assert len(out.projection_sources) == 0
    assert float(hole_mask.sum()) == 0.0


def test_mask_band_and_clean_plate_mesh_stay_in_lockstep():
    """AtlasDepthLayerMask and AtlasCleanPlateLayer share _resolve_depth_band —
    calling both with identical band params on the same scene must produce a
    mesh whose vertex distances are consistent with the mask's own band."""
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    layer_mask, _occ, _hole = AtlasDepthLayerMask().generate(solve, depth, near_m=5.0, far_m=12.0)
    out, _hole_mask, _ext = AtlasCleanPlateLayer().add_layer(
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


def test_fill_occluded_covers_the_occluder_footprint():
    """fill_occluded=True must produce band-mesh geometry across the near
    occluder's pixel columns — the disocclusion fix: the inpainted plate's
    content there needs geometry to land on."""
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    out_base, hole_base, _ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)
    out_fill, hole_fill, _ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32,
        fill_occluded=True)

    base_mesh = next(p for p in out_base.projection_sources[0].proxy_geometry
                     if p.primitive_type == "mesh")
    fill_mesh = next(p for p in out_fill.projection_sources[0].proxy_geometry
                     if p.primitive_type == "mesh")
    n_base = len(base_mesh.metadata["vertices"]) // 3
    n_fill = len(fill_mesh.metadata["vertices"]) // 3
    assert n_fill > n_base  # filled mesh has more geometry
    assert out_fill.projection_sources[0].metadata["n_filled_cells"] > 0
    assert out_base.projection_sources[0].metadata["n_filled_cells"] == 0
    # hole_mask shrinks accordingly
    assert float(hole_fill.sum()) < float(hole_base.sum())


def test_fill_occluded_keeps_mask_node_in_lockstep():
    """AtlasDepthLayerMask(compute_hole_mask=True, fill_occluded=True) must
    agree with AtlasCleanPlateLayer's fill for the same band."""
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    _l, _o, hole_mask_node = AtlasDepthLayerMask().generate(
        solve, depth, near_m=5.0, far_m=12.0, feather_px=0,
        compute_hole_mask=True, relief_grid=32, fill_occluded=True)
    _out, hole_layer_node, _ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32,
        fill_occluded=True)
    assert torch.equal(hole_mask_node, hole_layer_node)


# --- AtlasSkyDomeLayer --------------------------------------------------------

def _sky_mask_tensor(depth_map):
    # _occluder_depth's un-hit pixels (nothing intersected) sit at the flat
    # SKY constant — a legitimate stand-in for a real sky segmentation here.
    sky = depth_map >= SKY - 1e-6
    return torch.from_numpy(sky.astype(np.float32)).unsqueeze(0)


def test_sky_dome_layer_appends_source_with_unchanged_camera():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky_mask = _sky_mask_tensor(depth_map)
    plate = _plate_image()

    out, hole_mask, _ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, radius_m=300.0, relief_grid=32, name="sky")

    assert len(solve.projection_sources) == 0        # input untouched
    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    assert src.name == "sky"
    assert src.metadata.get("projection_mode") == "clean_plate"
    assert src.metadata.get("source") == "sky_dome"
    assert src.proxy_geometry
    assert all((p.metadata or {}).get("role") == PROXY_ROLE for p in src.proxy_geometry)

    # Camera is the PRIMARY's, unchanged — no orbit applied.
    assert src.camera.extrinsics.camera_view_matrix == solve.camera.extrinsics.camera_view_matrix
    assert tuple(hole_mask.shape) == (1, H, W)


def test_sky_dome_layer_mesh_is_flat_card_at_radius():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky_mask = _sky_mask_tensor(depth_map)
    plate = _plate_image()

    out, _hole_mask, _ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, radius_m=300.0, relief_grid=32)

    src = out.projection_sources[0]
    mesh_prim = next(p for p in src.proxy_geometry if p.primitive_type == "mesh")
    verts = np.array(mesh_prim.metadata["vertices"]).reshape(-1, 3)
    assert len(verts) > 0
    # Forward-Z (world Z under this fixture's identity rotation) sits at
    # -radius_m — a flat card, not a literal sphere (see build_sky_dome_mesh).
    assert np.allclose(verts[:, 2], -300.0, atol=5.0)


def test_sky_dome_layer_empty_mask_passes_through_unchanged():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    empty_mask = torch.zeros(1, H, W, dtype=torch.float32)
    plate = _plate_image()

    out, hole_mask, _ext = AtlasSkyDomeLayer().add_layer(solve, depth, empty_mask, plate)
    assert out is solve
    assert len(out.projection_sources) == 0
    assert float(hole_mask.sum()) == 0.0


def test_sky_dome_layer_passes_through_when_primary_has_no_focal():
    intr = AtlasIntrinsics(image_width=W, image_height=H)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky_mask = _sky_mask_tensor(depth_map)
    plate = _plate_image()

    out, hole_mask, _ext = AtlasSkyDomeLayer().add_layer(solve, depth, sky_mask, plate)
    assert out is solve
    assert len(out.projection_sources) == 0


def test_sky_dome_edge_extend_smears_plate_and_dilates_matte():
    """edge_extend_px must push sky colors past the silhouette into the plate
    AND grow the embedded matte to expose them — the classic edge-extend, so
    disocclusion reveals gradient sky instead of black slivers."""
    import base64
    import io

    from PIL import Image

    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky = depth_map >= SKY - 1e-6
    sky_mask = torch.from_numpy(sky.astype(np.float32)).unsqueeze(0)
    # Plate: sky region blue, everything else red.
    plate_np = np.zeros((H, W, 3), dtype=np.float32)
    plate_np[sky] = (0.2, 0.4, 1.0)
    plate_np[~sky] = (1.0, 0.1, 0.1)
    plate = torch.from_numpy(plate_np).unsqueeze(0)

    def _decode(b64):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[1]))))

    out_off, _, _ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, relief_grid=32, edge_extend_px=0)
    out_on, _, _ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, relief_grid=32, edge_extend_px=32)

    matte_off = _decode(out_off.projection_sources[0].mask_b64)
    matte_on = _decode(out_on.projection_sources[0].mask_b64)
    assert (matte_on > 128).sum() > (matte_off > 128).sum()  # matte dilated

    # A pixel ~16px below the sky boundary: red in the raw plate, sky-blue in
    # the extended one. Find a boundary column and probe below it.
    col = W // 2
    boundary_row = int(np.argmax(~sky[:, col]))  # first non-sky row
    probe = (boundary_row + 16, col)
    plate_on = _decode(out_on.projection_sources[0].image_b64)
    r, g, b = plate_on[probe][:3].astype(int)
    assert b > r, f"extension should smear sky blue past the edge, got rgb=({r},{g},{b})"
    assert matte_on[probe] > 128  # and the matte exposes it

    # Mesh overhang grew with the extension.
    n_off = out_off.projection_sources[0].metadata["n_vertices"]
    n_on = out_on.projection_sources[0].metadata["n_vertices"]
    assert n_on > n_off


def test_sky_dome_frame_outpaint_widens_the_source_camera():
    """frame_outpaint_px pads the plate canvas and widens THIS source's own
    intrinsics (cx/cy shifted, W/H grown) so a small orbit doesn't hit the
    frame edge — the primary solve's camera must stay untouched."""
    import base64
    import io

    from PIL import Image

    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky_mask = _sky_mask_tensor(depth_map)
    plate = _plate_image()
    PAD = 32

    out, hole, _ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, relief_grid=32,
        edge_extend_px=0, frame_outpaint_px=PAD)
    src = out.projection_sources[0]

    # Source camera widened; primary untouched.
    assert src.camera.intrinsics.image_width == W + 2 * PAD
    assert src.camera.intrinsics.image_height == H + 2 * PAD
    assert src.camera.intrinsics.cx_px == pytest.approx(CX + PAD)
    assert src.camera.intrinsics.fx_px == pytest.approx(FX)
    assert out.camera.intrinsics.image_width == W  # primary unchanged
    assert src.metadata["frame_outpaint_px"] == PAD

    # Plate + matte canvases are padded to match.
    plate_img = Image.open(io.BytesIO(base64.b64decode(src.image_b64.split(",", 1)[1])))
    assert plate_img.size == (W + 2 * PAD, H + 2 * PAD)
    matte_img = Image.open(io.BytesIO(base64.b64decode(src.mask_b64.split(",", 1)[1])))
    assert matte_img.size == (W + 2 * PAD, H + 2 * PAD)

    # Mesh extends past the original frustum: some vertex projects outside
    # the ORIGINAL frame through the primary camera.
    mesh_prim = next(p for p in src.proxy_geometry if p.primitive_type == "mesh")
    verts = np.array(mesh_prim.metadata["vertices"]).reshape(-1, 3)
    cam_pts = verts  # identity rotation fixture: world == camera + height offset
    z = -(verts[:, 2])
    u = CX + FX * verts[:, 0] / np.maximum(z, 1e-6)
    assert (u < 0).any() or (u > W).any()

    # hole_mask output cropped back to the ORIGINAL frame.
    assert tuple(hole.shape) == (1, H, W)


# --- per-pixel edge mattes (ProjectionSource.mask_b64) ------------------------

def test_sky_dome_embeds_its_segmentation_as_edge_matte():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky_mask = _sky_mask_tensor(depth_map)
    plate = _plate_image()

    out, _, _ext = AtlasSkyDomeLayer().add_layer(solve, depth, sky_mask, plate, relief_grid=32)
    src = out.projection_sources[0]
    assert src.mask_b64 and src.mask_b64.startswith("data:image/png;base64,")
    # Round-trips through solve JSON.
    reloaded = AtlasSolve.from_dict(out.to_dict())
    assert reloaded.projection_sources[0].mask_b64 == src.mask_b64


def test_clean_plate_embed_matte_is_opt_in():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    out_off, _, _ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)
    assert out_off.projection_sources[0].mask_b64 is None

    out_on, _, _ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32, embed_matte=True)
    matte_b64 = out_on.projection_sources[0].mask_b64
    assert matte_b64 and matte_b64.startswith("data:image/png;base64,")

    # Decoded matte matches the band: in-band pixel white, occluder pixel black.
    import base64
    import io

    from PIL import Image
    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(matte_b64.split(",", 1)[1]))))
    assert arr[180, 20] > 128       # far ground in [5,12] band
    assert arr[H - 1, W // 2] < 128  # near occluder, outside band (no fill)


def test_clean_plate_matte_includes_filled_footprint_when_fill_occluded():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()

    out, _, _ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32,
        embed_matte=True, fill_occluded=True)
    import base64
    import io

    from PIL import Image
    matte_b64 = out.projection_sources[0].mask_b64
    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(matte_b64.split(",", 1)[1]))))
    # With disocclusion fill, the occluder footprint carries inpainted plate
    # content on synthesized geometry — the matte must NOT cut it away.
    assert arr[H - 1, W // 2] > 128


def test_blockout_payload_carries_mask_b64():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    out, _, _ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, _sky_mask_tensor(depth_map), _plate_image(), relief_grid=32)

    image = torch.rand(1, H, W, 3, dtype=torch.float32)
    payload = _extract_blockout_camera(out, image, target_width=W, target_height=H)
    assert payload["projection_sources"][0]["mask_b64"].startswith("data:image/png;base64,")


# --- serialization (_extract_blockout_camera) --------------------------------

def test_serialization_includes_projection_mode_for_clean_plate():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()
    out, _hole_mask, _ext = AtlasCleanPlateLayer().add_layer(solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32)

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


def test_clean_plate_edge_extend_smears_colors_and_reports_mask():
    """edge_extend_px on a band layer: plate colors pushed past the matte
    edge, embedded matte dilated to expose them, and the invented region
    reported both as the extend_mask output and ProjectionSource.extend_mask_b64."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    solve = _solve()
    depth = _depth_result(_occluder_depth())
    # Plate: solid red so smeared pixels are unmistakable.
    plate = torch.zeros(1, H, W, 3, dtype=torch.float32)
    plate[..., 0] = 1.0

    out, _hole, ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32,
        edge_extend_px=24)

    src = out.projection_sources[0]
    assert src.mask_b64 and src.extend_mask_b64
    assert src.metadata["edge_extend_px"] == 24

    def decode(b64):
        return np.asarray(Image.open(io.BytesIO(
            base64.b64decode(b64.split(",", 1)[1]))).convert("L"),
            dtype=np.float32) / 255.0

    matte = decode(src.mask_b64)
    extend = decode(src.extend_mask_b64)
    ext_np = ext[0].numpy()
    # Extension exists, sits INSIDE the dilated matte, OUTSIDE nothing else.
    assert extend.sum() > 0
    assert ((extend > 0.5) & ~(matte > 0.5)).sum() == 0
    # Output mask matches the embedded one.
    assert np.allclose(ext_np > 0.5, extend > 0.5)
    # The re-encoded plate carries red into (some of) the extended region.
    from atlas_camera.exporters._layers import decode_plate_b64
    pil = decode_plate_b64(src.image_b64)
    rgb = np.asarray(pil, dtype=np.float32)
    ys, xs = np.nonzero(extend > 0.5)
    reds = rgb[ys, xs, 0]
    assert reds.mean() > 128  # smeared red, not black


def test_clean_plate_edge_extend_off_by_default():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)
    out, _hole, ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32,
        embed_matte=True)
    src = out.projection_sources[0]
    assert src.extend_mask_b64 is None
    assert float(ext.sum()) == 0.0


def test_sky_dome_extend_mask_covers_extension_and_outpaint_ring():
    import base64
    import io

    import numpy as np
    from PIL import Image

    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)
    sky = torch.zeros(1, H, W, dtype=torch.float32)
    sky[0, :H // 3, :] = 1.0

    out, _hole, ext = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky, plate, relief_grid=32,
        edge_extend_px=16, frame_outpaint_px=32)
    src = out.projection_sources[0]
    assert src.extend_mask_b64
    extend = np.asarray(Image.open(io.BytesIO(base64.b64decode(
        src.extend_mask_b64.split(",", 1)[1]))).convert("L"), dtype=np.float32) / 255.0
    # Padded plate frame: W+2*32 x H+2*32.
    assert extend.shape == (H + 64, W + 64)
    assert extend.sum() > 0
    # The outpaint ring above the horizon (sky region, edge-replicated pad)
    # is invented: the top-left pad corner pixel must be flagged.
    assert extend[0, 0] > 0.5
    # extend_mask output is in the same padded plate frame.
    assert tuple(ext.shape[1:]) == (H + 64, W + 64)


def test_layers_collector_writes_extend_matte(tmp_path):
    from atlas_camera.exporters._layers import collect_projection_layers

    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)
    sky = torch.zeros(1, H, W, dtype=torch.float32)
    sky[0, :H // 3, :] = 1.0
    out, _h, _e = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky, plate, relief_grid=32, name="sky",
        edge_extend_px=16, frame_outpaint_px=0)
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        out, depth, plate, near_m=5.0, far_m=12.0, relief_grid=32, name="bg",
        edge_extend_px=24)

    layers, skipped = collect_projection_layers(out, tmp_path)
    assert not skipped
    by_name = {l["name"]: l for l in layers}
    for lname in ("sky", "bg"):
        assert by_name[lname]["extend_matte_path"]
        assert (tmp_path / f"{lname}_extend_matte.png").exists()

    from atlas_camera.exporters.nuke_exporter import write_nuke_layers_script
    result = write_nuke_layers_script(out, tmp_path)
    nk = (tmp_path / "nuke_layers.nk").read_text(encoding="utf-8")
    assert "ExtendMatte_sky" in nk and "ExtendMatte_bg" in nk
    assert "regrain" in nk


def test_clean_plate_frame_outpaint_widens_camera_and_mesh():
    """frame_outpaint_px on a band layer: this source gets its OWN widened
    camera (cx/cy+P, W/H+2P; primary untouched), the mesh extends past the
    original frustum, the ring lands in extend_mask, and the hole output is
    cropped back to the source frame."""
    import numpy as np

    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)
    P = 32

    out, hole, ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=40.0, relief_grid=32,
        frame_outpaint_px=P)

    src = out.projection_sources[0]
    intr = src.camera.intrinsics
    assert (intr.image_width, intr.image_height) == (W + 2 * P, H + 2 * P)
    assert intr.cx_px == pytest.approx(CX + P)
    assert intr.cy_px == pytest.approx(CY + P)
    # Primary solve camera untouched.
    assert out.camera.intrinsics.image_width == W
    assert src.metadata["frame_outpaint_px"] == P
    assert src.mask_b64  # embed_matte implied

    # Mesh extends past the ORIGINAL frustum: project vertices with the
    # original intrinsics; some must land outside [0, W) x [0, H).
    mesh_prim = next(p for p in src.proxy_geometry if p.primitive_type == "mesh")
    verts = np.asarray(mesh_prim.metadata["vertices"], dtype=np.float64).reshape(-1, 3)
    vm = np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
    cam = verts @ vm[:3, :3].T + vm[:3, 3]
    z = -cam[:, 2]
    ok = z > 1e-6
    px = CX + FX * cam[ok, 0] / z[ok]
    py = CY - FY * cam[ok, 1] / z[ok]
    outside = (px < 0) | (px >= W) | (py < 0) | (py >= H)
    assert outside.mean() > 0.05  # a real ring of geometry beyond the frame

    # extend_mask is in the PADDED plate frame and flags the ring; hole
    # output is cropped back to the source frame.
    assert tuple(ext.shape[1:]) == (H + 2 * P, W + 2 * P)
    assert float(ext.sum()) > 0
    assert tuple(hole.shape[1:]) == (H, W)
    assert src.extend_mask_b64


def test_clean_plate_frame_outpaint_composes_with_edge_extend():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)
    out, _hole, ext = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=5.0, far_m=40.0, relief_grid=32,
        frame_outpaint_px=32, edge_extend_px=16)
    src = out.projection_sources[0]
    assert src.metadata["frame_outpaint_px"] == 32
    assert src.metadata["edge_extend_px"] == 16
    # Ring + smears both flagged as invented.
    assert float(ext.sum()) > 0
    assert src.extend_mask_b64


def test_border_flood_heals_segmenter_fade():
    """_flood_mask_to_frame_borders: a mask faded at the frame border floods
    to it (sky at row 40 means sky at rows 0-39); content genuinely cut by
    the frame (no mask within the margin) is untouched."""
    import numpy as np

    from atlas_camera.comfy.nodes import _flood_mask_to_frame_borders

    m = np.zeros((200, 200), dtype=bool)
    m[40:100, :100] = True        # sky region, faded above row 40 (left half)
    # right half: no mask within the top margin at all (a "spire" to the top)
    out = _flood_mask_to_frame_borders(m, margin_px=64)
    assert out[0:40, :100].all()          # flooded to the top border
    assert not out[0:40, 100:].any()      # spire columns untouched
    assert (out[40:100, :100] == m[40:100, :100]).all()  # interior unchanged
    # empty mask passes through
    assert not _flood_mask_to_frame_borders(np.zeros((50, 50), bool), 16).any()
