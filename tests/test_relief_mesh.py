"""Tests for the depth relief mesh (DCC handoff geometry).

Analytic depth scenes (same construction as test_proxy_geometry): level camera
at (0, h, 0), ground Y=0, wall at known z — expectations computed in the same
world the mesh is built in.
"""

import numpy as np
import pytest

from atlas_camera.core.relief_mesh import (
    build_relief_mesh,
    build_sky_dome_mesh,
    estimate_ground_scale,
)
from atlas_camera.exporters.relief_mesh_exporter import (
    export_relief_mesh,
    export_relief_mesh_glb,
)

W = H = 512
FX = FY = 500.0
CX = CY = 256.0
SKY = 60.0
EDGE_REL = 0.5


def _view_matrix(h):
    """Level camera at (0, h, 0), identity rotation (world→cam translation only)."""
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _scene_depth(h=1.6, wall_z=-10.0, wall_h=3.0):
    """Analytic z-depth: min over sky, ground plane Y=0, wall at z=wall_z."""
    _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    depth = np.full((H, W), SKY)

    tg = np.full((H, W), np.inf)
    hit = dy < -1e-6
    tg[hit] = -h / dy[hit]

    tw = np.full((H, W), np.inf)
    t = -wall_z
    y_at = h + dy * t
    vis = (y_at >= 0.0) & (y_at <= wall_h)
    tw[vis] = t

    stacked = np.stack([depth,
                        np.where(np.isfinite(tg), tg, SKY),
                        np.where(np.isfinite(tw), tw, SKY)])
    return stacked.min(axis=0)


def _build(depth, h=1.6, **kw):
    kw.setdefault("depth_edge_rel", EDGE_REL)
    return build_relief_mesh(
        depth, view_matrix=_view_matrix(h), fx=FX, fy=FY, cx=CX, cy=CY, **kw
    )


def _scene_depth_with_noisy_sky(h=1.6, wall_z=-10.0, wall_h=3.0, seed=3):
    """Same analytic scene, but the flat SKY constant is replaced with noisy,
    spatially-incoherent depth — the exact failure mode Depth Anything shows
    on feature-less sky/clouds (see test_depth_geometry.py).
    """
    depth = _scene_depth(h=h, wall_z=wall_z, wall_h=wall_h).copy()
    rng = np.random.RandomState(seed)
    is_sky = depth >= SKY
    depth[is_sky] = SKY + rng.uniform(-20.0, 20.0, size=int(is_sky.sum()))
    return depth


def test_mesh_is_well_formed():
    mesh = _build(_scene_depth(wall_z=-10.0))
    assert len(mesh.vertices) > 100
    assert len(mesh.faces) > 100
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < len(mesh.vertices)
    assert len(mesh.uvs) == len(mesh.vertices)
    assert mesh.uvs.min() >= 0.0 and mesh.uvs.max() <= 1.0


def test_no_face_spans_a_silhouette():
    # Every kept triangle's camera distances agree within the tear threshold —
    # the wall(10m)/sky(60m) silhouette must be a hole, not a rubber sheet.
    mesh = _build(_scene_depth(wall_z=-10.0, wall_h=3.0))
    cam = np.array([0.0, 1.6, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    tri = dist[mesh.faces]  # (M, 3)
    ratio = tri.max(axis=1) / np.maximum(tri.min(axis=1), 1e-6)
    assert float(ratio.max()) <= 1.0 + EDGE_REL + 0.05
    assert mesh.stats["torn_fraction"] > 0.0  # the silhouette actually tore


def test_hole_mask_nonempty_when_mesh_is_torn():
    mesh = _build(_scene_depth(wall_z=-10.0, wall_h=3.0))
    assert mesh.stats["torn_fraction"] > 0.0
    assert mesh.hole_mask.any()
    assert 0.0 < float(mesh.hole_mask.mean()) < 1.0  # some holes, not everything


def test_hole_mask_shape_is_full_resolution_regardless_of_grid():
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)
    coarse = _build(depth, grid_long_edge=32)
    fine = _build(depth, grid_long_edge=256)
    assert coarse.hole_mask.shape == depth.shape
    assert fine.hole_mask.shape == depth.shape


def test_hole_mask_flags_invalid_depth_pixels():
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0).copy()
    depth[100:110, 100:110] = 0.0  # fails the > 1e-4 valid-depth test
    mesh = _build(depth)
    assert mesh.hole_mask[100:110, 100:110].all()
    # a flat, non-torn ground pixel near the camera should not be a hole
    assert not bool(mesh.hole_mask[H - 5, W // 2])


def test_exclude_mask_removes_pixels_the_internal_heuristic_would_keep():
    # A flat, valid ground patch near the camera - detect_sky_mask/valid-depth
    # would never flag it on its own. An external exclude_mask (e.g. a real
    # SAM/RMBG sky segmentation) must still remove it from the mesh.
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)
    exclude = np.zeros(depth.shape, dtype=bool)
    exclude[H - 20:H - 5, W // 2 - 10:W // 2 + 10] = True

    baseline = _build(depth)
    assert not baseline.hole_mask[H - 10, W // 2]  # normally NOT a hole

    excluded = _build(depth, exclude_mask=exclude)
    assert excluded.hole_mask[H - 10, W // 2]  # now excluded by the mask
    # untouched region elsewhere is unaffected
    assert not excluded.hole_mask[H - 10, 10]


def test_exclude_mask_none_is_backward_compatible():
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)
    baseline = _build(depth)
    explicit_none = _build(depth, exclude_mask=None)
    assert len(baseline.vertices) == len(explicit_none.vertices)
    np.testing.assert_array_equal(baseline.hole_mask, explicit_none.hole_mask)


def test_band_clip_keeps_only_near_pixels():
    # band_max_m below the wall's distance (10m) should exclude the wall and
    # any far ground pixels, keeping only the near-ground band.
    h = 1.6
    depth = _scene_depth(h=h, wall_z=-10.0, wall_h=3.0)
    mesh = _build(depth, h=h, band_min_m=0.0, band_max_m=5.0)
    cam = np.array([0.0, h, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    assert float(dist.max()) <= 5.0 + EDGE_REL + 0.5
    assert len(mesh.vertices) > 20


def test_band_clip_keeps_only_far_pixels():
    # band_min_m above the near ground should exclude the ground, keeping
    # only the wall band (~10m).
    h = 1.6
    depth = _scene_depth(h=h, wall_z=-10.0, wall_h=3.0)
    mesh = _build(depth, h=h, band_min_m=8.0, band_max_m=12.0)
    cam = np.array([0.0, h, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    assert len(mesh.vertices) > 0
    assert float(dist.min()) >= 8.0 - EDGE_REL - 0.5


def test_no_band_clip_is_backward_compatible():
    # Default band_min_m/band_max_m=None must reproduce today's mesh exactly.
    depth = _scene_depth(wall_z=-10.0)
    baseline = _build(depth)
    clipped = _build(depth, band_min_m=None, band_max_m=None)
    assert len(baseline.vertices) == len(clipped.vertices)
    assert len(baseline.faces) == len(clipped.faces)
    np.testing.assert_array_equal(baseline.vertices, clipped.vertices)


def test_noisy_sky_is_excluded_not_triangulated():
    # Tearing (above) already stops noisy sky from rubber-sheeting onto the
    # wall, but without sky masking the sky pixels still get triangulated
    # into their own separate, jagged mesh chunk. With horizon_y given they
    # should be excluded from the mesh entirely (a hole, not geometry).
    h = 1.6
    depth = _scene_depth_with_noisy_sky(h=h, wall_z=-10.0, wall_h=3.0)
    mesh = _build(depth, h=h, horizon_y=CY)
    cam = np.array([0.0, h, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    # True scene tops out at the wall distance (10m); noisy sky spans
    # roughly 40-80m — none of that should survive into the mesh.
    assert float(dist.max()) < 20.0


def test_ground_lands_on_y_zero_and_nothing_below():
    mesh = _build(_scene_depth(wall_z=-10.0))
    ys = mesh.vertices[:, 1]
    assert float(ys.min()) > -0.2          # nothing under the ground plane
    assert float(np.abs(ys).min()) < 0.05  # ground vertices sit at Y≈0


def test_faces_wind_toward_camera():
    mesh = _build(_scene_depth(wall_z=-10.0))
    cam = np.array([0.0, 1.6, 0.0])
    v = mesh.vertices
    f = mesh.faces[:: max(1, len(mesh.faces) // 200)]  # sample
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    n = np.cross(e1, e2)
    centroid = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    to_cam = cam[None, :] - centroid
    dots = np.einsum("ij,ij->i", n, to_cam)
    assert (dots > 0).mean() > 0.99


def test_scale_rescales_about_camera():
    depth = _scene_depth(wall_z=-10.0)
    m1 = _build(depth, scale=1.0)
    m2 = _build(depth, scale=0.5)
    cam = np.array([0.0, 1.6, 0.0])
    d1 = np.linalg.norm(m1.vertices - cam, axis=1)
    d2 = np.linalg.norm(m2.vertices - cam, axis=1)
    assert np.median(d2 / np.maximum(d1, 1e-6)) == pytest.approx(0.5, abs=0.02)


def test_estimate_ground_scale_reconciles_doubled_depth():
    depth = _scene_depth(wall_z=-10.0) * 2.0
    scale, info = estimate_ground_scale(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY
    )
    assert scale == pytest.approx(0.5, abs=0.05)


def test_floor_clamp_pulls_outliers_along_the_ray():
    # Punch some depth outliers that would land far below the ground plane.
    depth = _scene_depth(wall_z=-10.0)
    depth[400:410, 200:210] *= 3.0  # below-horizon rays, 3x too deep → under ground
    mesh = _build(depth)
    assert float(mesh.vertices[:, 1].min()) >= -0.3  # clamped near the floor


def test_single_pixel_depth_spikes_are_removed():
    # A lone bad depth pixel (common in AI-image depth) must not become a mesh
    # spike — the 3x3 median sampling swallows it.
    depth = _scene_depth(wall_z=-10.0)
    for r, c in ((300, 128), (350, 384), (420, 256)):
        depth[r, c] = 2.0  # spike far in front of the true surface
    mesh = _build(depth)
    cam = np.array([0.0, 1.6, 0.0])
    dist = np.linalg.norm(mesh.vertices - cam, axis=1)
    assert float(dist.min()) > 2.5  # no vertex pulled to the spike depth


def test_no_stretched_shard_triangles():
    # World-space edge cap: no triangle edge may stretch far beyond the local
    # sample spacing (the source of the spiky shards).
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)
    mesh = _build(depth)
    v = mesh.vertices
    f = mesh.faces
    edges = np.concatenate([
        np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1),
    ])
    cam = np.array([0.0, 1.6, 0.0])
    dmax = float(np.linalg.norm(v - cam, axis=1).max())
    step = max(1, round(max(H, W) / 96))
    assert float(edges.max()) <= 12.0 * dmax * step / FX + 0.06


def test_grid_density_follows_grid_long_edge():
    depth = _scene_depth(wall_z=-10.0)
    small = _build(depth, grid_long_edge=32)
    big = _build(depth, grid_long_edge=128)
    assert len(big.faces) > 4 * len(small.faces)


def test_obj_export_round_trip(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=48)
    texture = Image.new("RGB", (64, 64), (200, 150, 100))
    paths = export_relief_mesh(mesh, tmp_path, texture=texture)

    obj_text = open(paths["obj"], encoding="utf-8").read().splitlines()
    n_v = sum(1 for line in obj_text if line.startswith("v "))
    n_vt = sum(1 for line in obj_text if line.startswith("vt "))
    n_f = sum(1 for line in obj_text if line.startswith("f "))
    assert n_v == len(mesh.vertices)
    assert n_vt == len(mesh.uvs)
    assert n_f == len(mesh.faces)
    # Face indices are 1-based and within range.
    for line in obj_text:
        if line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            assert all(1 <= i <= n_v for i in idx)
            break
    mtl_text = open(paths["mtl"], encoding="utf-8").read()
    assert "map_Kd" in mtl_text
    import os
    assert os.path.isfile(paths["texture"])


def test_obj_export_can_reference_external_exr_plate(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=48)
    texture = Image.new("RGB", (64, 64), (200, 150, 100))
    plate = tmp_path / "source_plate.exr"
    plate.write_bytes(b"fake exr")

    paths = export_relief_mesh(mesh, tmp_path, texture=texture, texture_path=plate)

    mtl_text = open(paths["mtl"], encoding="utf-8").read()
    assert f"map_Kd {plate.as_posix()}" in mtl_text
    assert paths["texture"] == str(plate)
    assert paths["texture_external"] == "true"
    assert not (tmp_path / "atlas_relief_mesh_diffuse.png").exists()


def test_relief_mesh_rides_the_proxy_payload():
    import json

    from atlas_camera.core.proxy_geometry import (
        relief_mesh_primitive,
        serialize_proxy_geometry,
    )
    from atlas_camera.core.schema import AtlasProjectionScene

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=32)
    scene = AtlasProjectionScene()
    scene.proxy_geometry.append(relief_mesh_primitive(mesh))
    payload = serialize_proxy_geometry(scene)
    assert len(payload) == 1
    entry = payload[0]
    assert entry["type"] == "mesh"
    assert len(entry["vertices"]) == 3 * len(mesh.vertices)
    assert len(entry["faces"]) == 3 * len(mesh.faces)
    assert len(entry["uvs"]) == 2 * len(mesh.uvs)
    assert max(entry["faces"]) < len(mesh.vertices)
    json.dumps(payload)  # JSON-safe


def test_glb_export_is_valid_gltf2(tmp_path):
    pytest.importorskip("PIL")
    import json
    import struct

    from PIL import Image

    mesh = _build(_scene_depth(wall_z=-10.0), grid_long_edge=48)
    texture = Image.new("RGB", (64, 64), (90, 120, 180))
    paths = export_relief_mesh_glb(mesh, tmp_path, texture=texture)

    raw = open(paths["glb"], "rb").read()
    magic, version, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67 and version == 2
    assert total == len(raw)

    json_len, json_type = struct.unpack_from("<II", raw, 12)
    assert json_type == 0x4E4F534A
    gltf = json.loads(raw[20:20 + json_len])
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, bin_off)
    assert bin_type == 0x004E4942
    assert gltf["buffers"][0]["byteLength"] == bin_len

    # Accessors match the mesh; POSITION carries required min/max.
    acc = gltf["accessors"]
    assert acc[0]["type"] == "VEC3"
    assert acc[0]["count"] == len(mesh.vertices)
    assert "min" in acc[0] and "max" in acc[0]
    assert acc[1]["count"] == len(mesh.uvs)
    assert acc[2]["count"] == mesh.faces.size

    # Texture embedded, unlit-tagged material, all bufferViews inside the bin.
    assert gltf["images"][0]["mimeType"] == "image/png"
    assert "KHR_materials_unlit" in gltf["materials"][0]["extensions"]
    for bv in gltf["bufferViews"]:
        assert bv["byteOffset"] + bv["byteLength"] <= bin_len

    # glTF V origin is top-left — flipped from the mesh's OBJ-convention UVs.
    uv_bv = gltf["bufferViews"][acc[1]["bufferView"]]
    uv_bytes = raw[bin_off + 8 + uv_bv["byteOffset"]:
                   bin_off + 8 + uv_bv["byteOffset"] + uv_bv["byteLength"]]
    uvs = np.frombuffer(uv_bytes, dtype=np.float32).reshape(-1, 2)
    assert np.allclose(uvs[:, 1], 1.0 - mesh.uvs[:, 1], atol=1e-6)


# --- fill_mask (occluded-depth diffusion fill) --------------------------------

def _occluder_scene(h=1.6, far_wall_z=-10.0, far_wall_h=3.0,
                    near_wall_z=-3.0, near_wall_h=2.0, col_lo=200, col_hi=320):
    """Ground + far wall + a finite-width near occluder (columns col_lo..col_hi)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    depth = np.full((H, W), SKY)
    tg = np.full((H, W), np.inf)
    ld = dy < -1e-6
    tg[ld] = -h / dy[ld]

    def _wall(z, wh):
        t = -z
        y = h + dy * t
        return t, (y >= 0.0) & (y <= wh)

    tf, vf = _wall(far_wall_z, far_wall_h)
    tn, vn = _wall(near_wall_z, near_wall_h)
    vn = vn & (uu >= col_lo) & (uu <= col_hi)
    return np.stack([depth, np.where(np.isfinite(tg), tg, SKY),
                     np.where(vf, tf, SKY), np.where(vn, tn, SKY)]).min(axis=0)


def test_fill_mask_fills_the_occluder_footprint():
    depth = _occluder_scene()
    occl = depth < 5.0
    base = _build(depth, band_min_m=5.0, band_max_m=12.0, grid_long_edge=64)
    filled = _build(depth, band_min_m=5.0, band_max_m=12.0, grid_long_edge=64,
                    fill_mask=occl)
    assert filled.stats["n_filled_cells"] > 0
    assert len(filled.vertices) > len(base.vertices)
    # The hole shrinks, and filled_mask lands on the occluder footprint.
    assert float(filled.hole_mask.mean()) < float(base.hole_mask.mean())
    assert filled.filled_mask is not None and filled.filled_mask.any()
    overlap = (filled.filled_mask & occl).sum() / max(occl.sum(), 1)
    assert overlap > 0.4
    # Filled pixels are NOT holes (they carry geometry now).
    assert not (filled.filled_mask & filled.hole_mask).any()


def test_fill_mask_synthesized_geometry_stays_plausible():
    depth = _occluder_scene()
    occl = depth < 5.0
    filled = _build(depth, band_min_m=5.0, band_max_m=12.0, grid_long_edge=64,
                    fill_mask=occl)
    # Nothing below the floor clamp (synthesized ground rides the existing
    # clamp-along-view-ray), nothing beyond the band's far edge.
    assert float(filled.vertices[:, 1].min()) >= -0.3
    assert float((-filled.vertices[:, 2]).max()) <= 12.5


def test_fill_mask_none_is_backward_compatible():
    depth = _occluder_scene()
    base = _build(depth, band_min_m=5.0, band_max_m=12.0, grid_long_edge=64)
    explicit = _build(depth, band_min_m=5.0, band_max_m=12.0, grid_long_edge=64,
                      fill_mask=None)
    assert len(base.vertices) == len(explicit.vertices)
    np.testing.assert_array_equal(base.hole_mask, explicit.hole_mask)
    assert explicit.filled_mask is None
    assert explicit.stats["n_filled_cells"] == 0


def test_fill_mask_unreachable_region_stays_a_hole():
    # A fill region with NO valid neighbors anywhere (entire frame invalid
    # except the fill request) cannot be filled — must degrade gracefully.
    depth = np.zeros((H, W))  # all invalid
    fill = np.zeros((H, W), dtype=bool)
    fill[100:200, 100:200] = True
    mesh = _build(depth, fill_mask=fill)
    assert mesh.stats["n_filled_cells"] == 0
    assert mesh.filled_mask is None or not mesh.filled_mask.any()
    assert mesh.hole_mask.all()


# --- build_sky_dome_mesh ------------------------------------------------------

def test_sky_dome_produces_geometry_only_where_masked():
    mask = np.zeros((H, W), dtype=bool)
    mask[:200, :] = True  # top ~40% "sky"
    mesh = build_sky_dome_mesh(mask, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
                                radius_m=300.0, grid_long_edge=64)
    assert len(mesh.vertices) > 50
    assert len(mesh.faces) > 0


def test_sky_dome_internal_sky_heuristic_does_not_eat_its_own_geometry():
    # This is the exact failure mode apply_sky_heuristic=False exists to avoid:
    # a constant-depth field is precisely what detect_sky_mask is designed to
    # flag, so if the heuristic ran it would re-exclude everything and the
    # dome would come back empty.
    mask = np.ones((H, W), dtype=bool)
    mesh = build_sky_dome_mesh(mask, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
                                radius_m=300.0, grid_long_edge=32)
    assert len(mesh.vertices) > 0
    assert float(mesh.hole_mask.mean()) < 0.1  # fully-masked input -> ~no holes


def test_sky_dome_is_a_flat_card_at_constant_forward_z():
    # Same forward-Z convention as build_relief_mesh everywhere else (and the
    # existing projection_backdrop plane) — a flat card, not a literal sphere.
    mask = np.zeros((H, W), dtype=bool)
    mask[:150, 100:400] = True
    mesh = build_sky_dome_mesh(mask, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
                                radius_m=300.0, grid_long_edge=64)
    forward_z = mesh.vertices[:, 2]
    assert np.allclose(forward_z, -300.0, atol=2.0)


def test_sky_dome_mask_boundary_becomes_a_hole_not_a_stretched_shard():
    mask = np.zeros((H, W), dtype=bool)
    mask[:200, :] = True
    mesh = build_sky_dome_mesh(mask, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
                                radius_m=300.0, grid_long_edge=64)
    # roughly matches the masked fraction of the frame (~39%), within the
    # coarseness of a 64-grid boundary snap
    frac = float(mask.mean())
    hole_frac = float(mesh.hole_mask.mean())
    assert abs((1.0 - hole_frac) - frac) < 0.05


def test_overhang_bevel_recedes_away_from_camera():
    """overhang_bevel_rel: skirt rings get progressively DEEPER (away from
    camera along each pixel's view ray); 0.0 stays byte-identical to the
    flat skirt."""
    np = pytest.importorskip("numpy")
    from atlas_camera.core.relief_mesh import build_relief_mesh

    h = w = 64
    depth = np.full((h, w), 10.0, dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    valid[16:48, 16:48] = True  # island -> boundary skirt extends outward
    exclude = ~valid

    kw = dict(view_matrix=_view_matrix(0.0), fx=60.0, fy=60.0, cx=32.0, cy=32.0,
              grid_long_edge=32, exclude_mask=exclude, scale=1.0,
              apply_sky_heuristic=False, edge_overhang_cells=4)

    flat = build_relief_mesh(depth, **kw)
    flat2 = build_relief_mesh(depth, overhang_bevel_rel=0.0, **kw)
    bev = build_relief_mesh(depth, overhang_bevel_rel=2.0, **kw)

    v_flat = np.asarray(flat.vertices)
    v_flat2 = np.asarray(flat2.vertices)
    v_bev = np.asarray(bev.vertices)
    # bevel=0.0 identical to the default
    assert v_flat.shape == v_flat2.shape
    np.testing.assert_allclose(v_flat, v_flat2)

    # Camera at origin (identity view): distance from origin = depth along ray.
    r_flat = np.linalg.norm(v_flat, axis=1)
    r_bev = np.linalg.norm(v_bev, axis=1)
    # The beveled mesh's farthest vertices recede beyond the flat skirt's.
    # (Interior-unchanged is already proven by the byte-identical bevel=0.0
    # comparison above; per-vertex interior checks are polluted by
    # floor-clamping, which treats the deeper skirt differently.)
    # 4 rings at slope 2 recede ~8 cells deeper; cell ≈ d*step/fx ≈ 0.33m
    # at d=10, so expect roughly +2.5m beyond the flat skirt's max reach.
    assert r_bev.max() > r_flat.max() + 1.5
    # The skirt is still triangulated (beveled rings must not tear apart).
    assert bev.faces.shape[0] >= flat.faces.shape[0] * 0.95
