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
    AtlasBoundedBand,
    AtlasCleanPlateLayer,
    AtlasDepthLayerMask,
    AtlasSkyDomeLayer,
    _BOUNDED_BAND_NOOP_M,
    _apply_band_split,
    _extract_blockout_camera,
    _metric_depth_and_validity,
    _resolve_exclude_mask,
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


def test_band_positions_are_log_depth_not_pixel_percentile():
    """near_pct/far_pct interpolate the scene's LOG-depth range: on a skewed
    distribution (90% of pixels at ~2m, 10% at ~80m — the typical ground-
    dominated photo) t=0.5 must land near the geometric mean of the depth
    range (~13m), NOT at the pixel median (~2m), and the useful bg split no
    longer hides at 0.9+."""
    import math

    import numpy as np

    from atlas_camera.comfy.nodes import _resolve_depth_band

    rng = np.random.default_rng(7)
    metric = np.concatenate([
        rng.uniform(1.8, 2.2, 9000),    # dominant near ground
        rng.uniform(70.0, 90.0, 1000),  # distant scene
    ]).reshape(100, 100)
    valid = np.ones_like(metric, dtype=bool)

    near, far = _resolve_depth_band(metric, valid, 0.0, 0.0, 0.5, 0.0)
    d_lo = float(np.percentile(metric, 1.0))
    d_hi = float(np.percentile(metric, 99.0))
    geo_mean = math.exp((math.log(d_lo) + math.log(d_hi)) / 2)
    assert near == pytest.approx(geo_mean, rel=0.05)
    assert near > 8.0                       # far from the ~2m pixel median
    assert far == float("inf")

    # far_pct at/above ~1.0 = no cap; explicit metres still win outright.
    _, far_cap = _resolve_depth_band(metric, valid, 0.0, 0.0, 0.0, 1.0)
    assert far_cap == float("inf")
    near_m, far_m = _resolve_depth_band(metric, valid, 5.0, 40.0, 0.5, 0.5)
    assert (near_m, far_m) == (5.0, 40.0)


def test_band_split_partitions_fg_bg_exactly():
    """One AtlasDepthBandSplit wired into both band nodes: fg = [0, split),
    bg = [split, +inf) — the boundary resolves through the same log mapping
    on both sides, so the partition is exact and the nodes' own near/far
    widgets are ignored."""
    from atlas_camera.comfy.nodes import (
        NODE_CLASS_MAPPINGS,
        AtlasDepthBandSplit,
        _apply_band_split,
    )
    import numpy as np

    assert NODE_CLASS_MAPPINGS["AtlasDepthBandSplit"] is AtlasDepthBandSplit
    (split,) = AtlasDepthBandSplit().define(split=0.6)

    rng = np.random.default_rng(3)
    metric = rng.uniform(2.0, 60.0, (64, 64))
    valid = np.ones_like(metric, dtype=bool)

    # Deliberately conflicting per-node widget values — must be ignored.
    fg = _apply_band_split(split, "foreground", metric, valid, 0.0, 0.0, 0.9, 0.1)
    bg = _apply_band_split(split, "background", metric, valid, 3.0, 7.0, 0.2, 0.3)
    assert fg[0] == 0.0
    assert bg[1] == float("inf")
    assert fg[1] == pytest.approx(bg[0])   # exact shared boundary
    assert 2.0 < fg[1] < 60.0

    # split_m overrides; manual falls through to the node's own widgets.
    (split_m,) = AtlasDepthBandSplit().define(split=0.6, split_m=25.0)
    fg_m = _apply_band_split(split_m, "foreground", metric, valid, 0, 0, 0, 0)
    assert fg_m[1] == 25.0
    manual = _apply_band_split(split, "manual", metric, valid, 5.0, 40.0, 0, 0)
    assert manual == (5.0, 40.0)

    # End-to-end through both nodes: same solve, one split, two sides.
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    plate = _plate_image()
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, name="bg", relief_grid=32,
        band_side="background", band_split=split)
    out, _h2, _e2 = AtlasCleanPlateLayer().add_layer(
        out, depth, plate, name="fg", relief_grid=32,
        band_side="foreground", band_split=split)
    bgs = next(s for s in out.projection_sources if s.name == "bg")
    fgs = next(s for s in out.projection_sources if s.name == "fg")
    assert bgs.metadata["far_m"] is None                 # +inf
    assert fgs.metadata["far_m"] == pytest.approx(bgs.metadata["near_m"])


# --- band_geometry: card / ground flat modes (VLM-recommendable, 2026-07-11) --

def _layer_verts(out):
    src = out.projection_sources[-1]
    v = np.array(src.proxy_geometry[0].metadata["vertices"], dtype=float).reshape(-1, 3)
    return v, src.metadata


def test_band_geometry_card_is_fronto_parallel_plane():
    """card = ONE flat constant-forward-Z plane at the band's median depth
    (the hangar-far-wall case). Above-ground vertices must share EXACTLY one
    forward depth — zero tearing/noise by construction. (Below-ground card
    extent is floor-clamped along the ray, same rule as relief meshes.)"""
    solve, depth, plate = _solve(), _depth_result(_occluder_depth()), _plate_image()
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=8.0, far_m=12.0, band_geometry="card")
    v, meta = _layer_verts(out)
    assert meta["band_geometry"] == "card"
    above = v[v[:, 1] > 0.01]
    assert len(above) > 100
    assert above[:, 2].std() < 1e-6           # exactly planar
    assert -12.0 <= above[:, 2].mean() <= -8.0  # at the band's depth


def test_band_geometry_ground_lands_on_y0_plane():
    """ground = the exact analytic Y=0 plane (the desert-floor case): every
    vertex on the ground plane regardless of depth noise, capped at the
    band's far edge so wall-like pixels don't run out to the horizon."""
    solve, depth, plate = _solve(), _depth_result(_occluder_depth()), _plate_image()
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=0.0, far_m=5.0, band_geometry="ground")
    v, meta = _layer_verts(out)
    assert meta["band_geometry"] == "ground"
    assert len(v) > 100
    assert np.abs(v[:, 1]).max() < 0.05       # on the plane (mm rounding)
    assert v[:, 2].min() >= -5.0 - 1e-6       # capped at the band far edge


def test_band_geometry_relief_default_unchanged():
    solve, depth, plate = _solve(), _depth_result(_occluder_depth()), _plate_image()
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=8.0, far_m=12.0)
    v, meta = _layer_verts(out)
    assert meta["band_geometry"] == "relief"
    assert v[:, 2].std() > 0.01               # a real depth-following mesh


def test_geometry_override_wins_and_garbage_errors():
    solve, depth, plate = _solve(), _depth_result(_occluder_depth()), _plate_image()
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=8.0, far_m=12.0,
        band_geometry="relief", geometry_override="card")
    assert out.projection_sources[-1].metadata["band_geometry"] == "card"
    # "" falls through to the combo; unknown values error loudly
    out2, _h2, _e2 = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=8.0, far_m=12.0,
        band_geometry="ground", geometry_override="")
    assert out2.projection_sources[-1].metadata["band_geometry"] == "ground"
    with pytest.raises(ValueError, match="band geometry"):
        AtlasCleanPlateLayer().add_layer(solve, depth, plate, geometry_override="flat")


# --- sky card: distance_m vs radius_m-as-size (2026-07-11) -------------------

def test_sky_card_distance_m_places_card_and_radius_becomes_size():
    solve = _solve()
    depth_map = _occluder_depth()
    depth = _depth_result(depth_map)
    sky_mask = _sky_mask_tensor(depth_map)
    plate = _plate_image()

    # Legacy: distance_m=0 -> radius_m IS the distance (unchanged behavior).
    out, _h, _e = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, radius_m=300.0, frame_outpaint_px=0)
    src = out.projection_sources[-1]
    v = np.array(src.proxy_geometry[0].metadata["vertices"], dtype=float).reshape(-1, 3)
    assert np.allclose(v[:, 2], -300.0, atol=5.0)
    assert src.metadata["distance_m"] == 300.0
    assert src.metadata["size_pad_px"] == 0

    # distance_m places the card; radius_m = minimum half-extent (SIZE).
    # Natural half-width at 50m on this fixture: 50 * 128/250 = 25.6m — a
    # 40m radius needs extra outpaint, which must actually deliver >= 40m.
    out2, _h2, _e2 = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, radius_m=40.0, distance_m=50.0,
        frame_outpaint_px=0)
    src2 = out2.projection_sources[-1]
    v2 = np.array(src2.proxy_geometry[0].metadata["vertices"], dtype=float).reshape(-1, 3)
    assert np.allclose(v2[:, 2], -50.0, atol=2.0)           # sits AT distance_m
    assert src2.metadata["size_pad_px"] > 0
    assert np.abs(v2[:, 0]).max() >= 40.0 * 0.95            # reaches the size
    # The grown ring is invented pixels — declared, and the source camera
    # widened to match (per-source camera, primary untouched).
    assert src2.metadata["frame_outpaint_px"] == src2.metadata["size_pad_px"]
    assert src2.camera.intrinsics.image_width > solve.camera.intrinsics.image_width

    # A size the frustum already covers adds no padding (never shrinks).
    out3, _h3, _e3 = AtlasSkyDomeLayer().add_layer(
        solve, depth, sky_mask, plate, radius_m=10.0, distance_m=50.0,
        frame_outpaint_px=0)
    assert out3.projection_sources[-1].metadata["size_pad_px"] == 0


# --- AtlasScopeMask: self-disarming band scoping (2026-07-11) ----------------

def _scope():
    from atlas_camera.comfy.nodes import AtlasScopeMask
    return AtlasScopeMask()


class _DynPromptStub:
    """Minimal DynPrompt: reports the named inputs as graph links (lists),
    mirroring how ComfyUI records a wired input as [source_id, slot]. Lets the
    lazy-status wiring guard be exercised outside a live executor."""
    def __init__(self, *wired):
        self._wired = wired

    def get_node(self, _uid):
        return {"inputs": {name: ["src", 0] for name in self._wired}}


def test_scope_mask_empty_prompt_is_band_only_and_lazy():
    sky = torch.zeros(1, H, W); sky[:, :60] = 1.0
    excl, status = _scope().build(sky, prompt="")
    assert torch.equal(excl, sky)                       # exactly the sky mask
    assert "band-only" in status
    # Empty prompt: the segment branch is never pulled (wiring irrelevant).
    assert _scope().check_lazy_status(sky, prompt="") == []
    # Prompt set + segment WIRED: pull it.
    assert _scope().check_lazy_status(
        sky, prompt="rocks",
        dynprompt=_DynPromptStub("segment_mask"), unique_id="1") == ["segment_mask"]
    # Prompt set + segment UNWIRED: band-only, never request the input (the
    # 2026-07-12 crash guard — asking for an unconnected lazy input aborts the
    # queue).
    assert _scope().check_lazy_status(
        sky, prompt="rocks", dynprompt=_DynPromptStub(), unique_id="1") == []
    # A no-match segment pulls the (lazy) fallback_mask next when it's WIRED.
    assert _scope().check_lazy_status(
        sky, prompt="rocks", segment_mask=torch.zeros(1, H, W),
        dynprompt=_DynPromptStub("fallback_mask"), unique_id="1") == ["fallback_mask"]
    # No-match segment + fallback UNWIRED: band-only.
    assert _scope().check_lazy_status(
        sky, prompt="rocks", segment_mask=torch.zeros(1, H, W),
        dynprompt=_DynPromptStub(), unique_id="1") == []
    # A real (matching) segment pulls neither.
    good = torch.zeros(1, H, W); good[:, 100:180, 80:240] = 1.0
    assert _scope().check_lazy_status(
        sky, prompt="rocks", segment_mask=good,
        dynprompt=_DynPromptStub("fallback_mask"), unique_id="1") == []


def test_scope_mask_semantic_fallback_scopes_on_no_match():
    """Item 10 (CV audit): a no-match SAM segment tries the geometry-prior
    fallback (AtlasSemanticMask) BEFORE degrading to band-only."""
    sky = torch.zeros(1, H, W); sky[:, :60] = 1.0
    empty_seg = torch.zeros(1, H, W)
    fb = torch.zeros(1, H, W); fb[:, 100:180, 80:240] = 1.0
    excl, status = _scope().build(sky, prompt="desert floor and boulder",
                                  segment_mask=empty_seg, fallback_mask=fb,
                                  grow_px=8)
    assert "semantic FALLBACK" in status and "no-matched" in status
    assert float(excl[0, 140, 160]) == 0.0              # fallback interior kept
    assert float(excl[0, 140, 100 - 30]) == 1.0         # outside excluded
    # An ALSO-empty fallback still degrades to band-only, never exclude-all.
    excl2, status2 = _scope().build(sky, prompt="x", segment_mask=empty_seg,
                                    fallback_mask=torch.zeros(1, H, W))
    assert torch.equal(excl2, sky)
    assert "band-only FALLBACK" in status2


def test_scope_mask_no_match_segment_falls_back():
    """The live failure: SAM3 scored 0.0% for 'desert floor and boulder' and
    the old Grow->Invert->Composite row turned that into exclude-EVERYTHING,
    silently emptying the layer to zero mesh. The node must fall back."""
    sky = torch.zeros(1, H, W); sky[:, :60] = 1.0
    empty_seg = torch.zeros(1, H, W)
    excl, status = _scope().build(sky, prompt="desert floor and boulder",
                                  segment_mask=empty_seg)
    assert torch.equal(excl, sky)
    assert "FALLBACK" in status and "no-match" in status


def test_scope_mask_real_segment_scopes_band():
    sky = torch.zeros(1, H, W); sky[:, :60] = 1.0
    seg = torch.zeros(1, H, W); seg[:, 100:180, 80:240] = 1.0
    excl, status = _scope().build(sky, prompt="rock formations",
                                  segment_mask=seg, grow_px=8)
    assert "scoped to 'rock formations'" in status
    # Inside the grown segment: NOT excluded; far outside: excluded.
    assert float(excl[0, 140, 160]) == 0.0              # segment interior kept
    assert float(excl[0, 140, 100 - 30]) == 1.0         # 30px left of segment+grow
    assert float(excl[0, 105, 160]) == 0.0              # grow ring (8px) kept
    assert float(excl[0, 10, 160]) == 1.0               # sky stays excluded


# --- band_ref_mask: shared band-edge population (2026-07-11 debug finding) ---

def test_band_ref_mask_removes_scope_induced_band_drift():
    """Per-layer scoped excludes change each layer's depth population, so the
    same percentages resolve to different metres per layer (debug-report
    finding: real metric GAPs between adjacent bands). With band_ref_mask
    (the plain sky mask) wired, two layers with different scoped excludes
    must resolve IDENTICAL band edges."""
    # A vertical depth RAMP (1..30m) so excludes genuinely bias the robust
    # percentile bounds (the analytic wall fixture pins them at [3,10] no
    # matter what is excluded).
    vv = np.linspace(1.0, 30.0, H, dtype=np.float32)[:, None]
    depth_map = np.repeat(vv, W, axis=1)
    solve, depth = _solve(), _depth_result(depth_map)
    sky = torch.zeros(1, H, W)
    # a scoped exclude removes the far half of the ramp -> different d_hi
    scoped = sky.clone()
    scoped[:, H // 2:, :] = 1.0

    def far_edge(exclude, band_ref):
        # far_pct deliberately NOT 0.5: 0.5 is the geometric mean, which this
        # analytic fixture keeps coincidentally invariant across populations.
        out, _h, _e = AtlasCleanPlateLayer().add_layer(
            solve, depth, _plate_image(), near_pct=0.0, far_pct=0.3,
            exclude_mask=exclude, band_ref_mask=band_ref, name="t")
        return out.projection_sources[-1].metadata["far_m"]

    # Legacy: different exclude populations -> the same far_pct drifts apart.
    assert far_edge(sky, None) != pytest.approx(far_edge(scoped, None), rel=0.01)
    # band_ref_mask: identical edges regardless of each layer's scoping.
    assert far_edge(sky, sky) == pytest.approx(far_edge(scoped, sky), rel=1e-9)


# --- AtlasLayerPreview: cut-out layer preview in 🎨 legend colors ------------

def test_layer_preview_cutout_and_palette():
    from atlas_camera.comfy.nodes import (
        _LAYER_DEBUG_PALETTE_HEX,
        _LAYER_DEBUG_PRIMARY_HEX,
        AtlasLayerPreview,
    )
    img = torch.rand(1, H, W, 3)
    m = torch.zeros(1, H, W); m[:, 40:120, 60:200] = 1.0
    (out,) = AtlasLayerPreview().preview(img, m, layer_index=1)  # 3d8bff blue
    assert out.shape == (1, H, W, 3)
    assert torch.allclose(out[:, 40:120, 60:200, :], img[:, 40:120, 60:200, :])  # cutout
    expect = torch.tensor([0x3d / 255, 0x8b / 255, 0xff / 255])
    assert torch.allclose(out[0, 0, 0, :], expect, atol=1e-6)                    # surround
    # -1 = primary teal; explicit hex override wins; malformed hex = magenta.
    (teal,) = AtlasLayerPreview().preview(img, m, layer_index=-1)
    assert torch.allclose(teal[0, 0, 0, :],
                          torch.tensor([0x2f / 255, 0xd6 / 255, 0xc3 / 255]), atol=1e-6)
    (ovr,) = AtlasLayerPreview().preview(img, m, layer_index=1, color_hex="#ff0000")
    assert torch.allclose(ovr[0, 0, 0, :], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    (bad,) = AtlasLayerPreview().preview(img, m, color_hex="zz")
    assert torch.allclose(bad[0, 0, 0, :], torch.tensor([1.0, 0.0, 1.0]), atol=1e-6)
    # palette mirrors atlas_blockout.js (LAYER_DEBUG_PALETTE)
    assert _LAYER_DEBUG_PALETTE_HEX[0] == "ff6a3d" and _LAYER_DEBUG_PRIMARY_HEX == "2fd6c3"


# --- band_override: the VLM band-boundary channel (2026-07-11) ---------------

def test_band_override_wins_and_stays_watertight():
    from atlas_camera.comfy.nodes import _parse_band_override

    solve, depth, plate = _solve(), _depth_result(_occluder_depth()), _plate_image()

    def edges(**kw):
        out, _h, _e = AtlasCleanPlateLayer().add_layer(
            solve, depth, plate, near_m=99.0, far_m=100.0, name="t", **kw)
        m = out.projection_sources[-1].metadata
        return m["near_m"], m["far_m"]

    # Override beats the node's own (absurd) metre widgets.
    n1, f1 = edges(band_override="near_pct=0.000 far_pct=0.400")
    n2, f2 = edges(band_override="near_pct=0.400 far_pct=1.000")
    assert (n1 or 0) < f1
    # Adjacent overrides share the edge EXACTLY (watertight by construction).
    assert f1 == pytest.approx(n2, rel=1e-9)
    assert f2 is None  # far_pct=1.0 -> open-ended (+inf)

    # "" is a no-op; garbage errors loudly.
    assert _parse_band_override("") is None
    with pytest.raises(ValueError, match="band override"):
        _parse_band_override("bands: 0.3 to 0.6")
    with pytest.raises(ValueError, match="out of range"):
        _parse_band_override("near_pct=0.8 far_pct=0.2")

    # AtlasDepthLayerMask takes the same string -> same band as the layer.
    lm, occ, _h = AtlasDepthLayerMask().generate(
        solve, depth, near_m=99.0, far_m=100.0,
        band_override="near_pct=0.000 far_pct=0.400")
    assert float(lm.sum()) > 0


# --- AtlasBoundedBand --------------------------------------------------------

def _foreground_lower_half_mask():
    """A mask over the lower frame (ground + walls) — a region with a REAL
    metric depth spread, unlike a single fronto-parallel wall (W≈0)."""
    m = torch.zeros(1, H, W, dtype=torch.float32)
    m[:, int(CY) + 8:, :] = 1.0
    return m


def test_bounded_band_registered():
    node = NODE_CLASS_MAPPINGS["AtlasBoundedBand"]
    assert node.RETURN_TYPES == ("ATLAS_BAND_SPLIT", "FLOAT", "STRING")
    assert node.RETURN_NAMES == ("band_split", "cutoff_m", "report")


def test_bounded_band_cutoff_is_near_plus_multiplier_times_width():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    mask = _foreground_lower_half_mask()

    band_split, cutoff, report = AtlasBoundedBand().measure(
        solve, depth, mask, extrude_multiplier=2.0, near_pct=5.0, far_pct=95.0)

    # Recompute the expected cutoff from the IDENTICAL metric setup the node used.
    setup = _metric_depth_and_validity(solve, depth)
    fg = _resolve_exclude_mask(mask, setup.height, setup.width)
    valid = setup.valid & np.isfinite(setup.metric) & fg.astype(bool)
    vals = setup.metric[valid]
    near = float(np.percentile(vals, 5.0))
    far = float(np.percentile(vals, 95.0))
    assert far - near > 0.1                      # sanity: the region has real spread
    expected = near + 2.0 * (far - near)

    assert cutoff == pytest.approx(expected, rel=1e-6)
    assert band_split["split_m"] == pytest.approx(cutoff)
    assert f"{cutoff:.2f}m" in report


def test_bounded_band_multiplier_scales_the_extent():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    mask = _foreground_lower_half_mask()

    _bs1, cutoff1, _ = AtlasBoundedBand().measure(solve, depth, mask, extrude_multiplier=1.0)
    _bs3, cutoff3, _ = AtlasBoundedBand().measure(solve, depth, mask, extrude_multiplier=3.0)

    setup = _metric_depth_and_validity(solve, depth)
    fg = _resolve_exclude_mask(mask, setup.height, setup.width)
    valid = setup.valid & np.isfinite(setup.metric) & fg.astype(bool)
    vals = setup.metric[valid]
    near = float(np.percentile(vals, 5.0))
    width = float(np.percentile(vals, 95.0)) - near
    # cutoff3 - cutoff1 == (3-1) * W ; cutoff1 == near + W
    assert (cutoff3 - cutoff1) == pytest.approx(2.0 * width, rel=1e-6)
    assert cutoff1 == pytest.approx(near + width, rel=1e-6)


def test_bounded_band_split_partitions_both_layers_at_the_cutoff():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    mask = _foreground_lower_half_mask()

    band_split, cutoff, _ = AtlasBoundedBand().measure(solve, depth, mask)

    setup = _metric_depth_and_validity(solve, depth)
    # The ONE split, fed to both sides, partitions exactly at the cutoff.
    near_fg, far_fg = _apply_band_split(band_split, "foreground", setup.metric, setup.valid, 0, 0, 0, 0)
    near_bg, far_bg = _apply_band_split(band_split, "background", setup.metric, setup.valid, 0, 0, 0, 0)
    assert near_fg == 0.0
    assert far_fg == pytest.approx(cutoff)       # foreground relief clipped to [0, cutoff]
    assert near_bg == pytest.approx(cutoff)      # background card median beyond the cutoff
    assert far_bg == float("inf")


def test_bounded_band_empty_mask_emits_unclipped_sentinel():
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    empty = torch.zeros(1, H, W, dtype=torch.float32)

    band_split, cutoff, report = AtlasBoundedBand().measure(solve, depth, empty)

    # A partition can't no-op both sides; the sentinel keeps the FOREGROUND
    # unclipped ([0, 1e6m]) rather than collapsing it to [0, 0].
    assert cutoff == _BOUNDED_BAND_NOOP_M
    assert band_split["split_m"] == _BOUNDED_BAND_NOOP_M
    assert "sentinel" in report.lower()
    setup = _metric_depth_and_validity(solve, depth)
    near_fg, far_fg = _apply_band_split(band_split, "foreground", setup.metric, setup.valid, 0, 0, 0, 0)
    assert near_fg == 0.0 and far_fg == pytest.approx(_BOUNDED_BAND_NOOP_M)


def test_bounded_band_no_focal_passes_through_as_sentinel():
    intr = AtlasIntrinsics(image_width=W, image_height=H)  # no fx_px -> no metric depth
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    depth = _depth_result(_occluder_depth())
    mask = _foreground_lower_half_mask()

    band_split, cutoff, report = AtlasBoundedBand().measure(solve, depth, mask)
    assert cutoff == _BOUNDED_BAND_NOOP_M
    assert band_split["split_m"] == _BOUNDED_BAND_NOOP_M


# --- MoGe predicted-normal relight map ---------------------------------------

def test_clean_plate_layer_embeds_aligned_normal_map_when_present():
    """A depth result carrying predicted normals (MoGe *-normal) → the layer's
    ProjectionSource gets a world-aligned normal-map data URI for the relight."""
    pytest.importorskip("PIL")
    solve = _solve()
    depth = _depth_result(_occluder_depth())
    rng = np.random.default_rng(0)
    nrm = rng.normal(size=(H, W, 3)).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    depth.normal = nrm                                   # simulate MoGe's per-pixel normals
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, _plate_image(), near_m=5.0, far_m=12.0, relief_grid=32)
    src = out.projection_sources[-1]
    assert src.normal_map_b64 and src.normal_map_b64.startswith("data:image/png;base64,")


def test_clean_plate_layer_no_normal_map_without_predicted_normals():
    solve = _solve()
    depth = _depth_result(_occluder_depth())      # DepthResult.normal defaults to None
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, _plate_image(), near_m=5.0, far_m=12.0, relief_grid=32)
    assert out.projection_sources[-1].normal_map_b64 is None
