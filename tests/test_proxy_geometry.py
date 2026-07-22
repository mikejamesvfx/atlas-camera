"""Tests for depth-derived proxy geometry (camera-projection blockout).

Analytic depth maps with known geometry (style of test_depth_height.py): a level
camera at (0, h, 0) looking along -Z, ground plane Y=0, and known walls/objects.
Expectations are computed in the same world the viewport reconstructs, which
locks the view-matrix convention end-to-end.
"""

import json

import numpy as np
import pytest

from atlas_camera.core.proxy_geometry import (
    derive_projection_proxies,
    derive_vertical_extrusion_proxies,
    serialize_proxy_geometry,
)
from atlas_camera.core.schema import AtlasProjectionScene


W = H = 512
FX = FY = 500.0
CX = CY = 256.0
SKY = 60.0


def _view_matrix(h):
    """Level camera at (0, h, 0), identity rotation (world→cam translation only)."""
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rays():
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dx = (uu - CX) / FX
    dy = -(vv - CY) / FY  # cam y-up; image v grows downward
    return dx, dy


def _ground_depth(h, dx, dy):
    """z-depth of the ground plane Y=0 for a camera at height h (inf if none)."""
    t = np.full(dx.shape, np.inf)
    hit = dy < -1e-6  # looking downward
    t[hit] = -h / dy[hit]
    return t


def _scene_depth(h=1.6, wall_z=None, wall_h=3.0, box=None, cyl=None):
    """Compose an analytic z-depth map: min over sky, ground, wall, box, cylinder.

    box: (x_center, z_front, half_width, height) — fronto-parallel front face.
    cyl: (x_center, z_center, radius, height) — vertical cylinder (analytic hit).
    """
    dx, dy = _rays()
    depth = np.full((H, W), SKY)
    tg = _ground_depth(h, dx, dy)

    candidates = [np.where(np.isfinite(tg), tg, np.inf)]

    if wall_z is not None:
        tw = np.full((H, W), np.inf)
        t = -wall_z  # z_world = -t → t = -wall_z
        y_at = h + dy * t
        vis = (y_at >= 0.0) & (y_at <= wall_h)
        tw[vis] = t
        candidates.append(tw)

    if box is not None:
        bx, bz, hw, bh = box
        tb = np.full((H, W), np.inf)
        t = -bz
        x_at = dx * t
        y_at = h + dy * t
        vis = (np.abs(x_at - bx) <= hw) & (y_at >= 0.0) & (y_at <= bh)
        tb[vis] = t
        candidates.append(tb)

    if cyl is not None:
        ccx, ccz, r, ch = cyl
        # Ray in XZ (z-depth param t): point = (dx*t, -t). Solve
        # (dx t − ccx)² + (−t − ccz)² = r².
        A = dx ** 2 + 1.0
        B = -2.0 * (dx * ccx - ccz)
        C = ccx ** 2 + ccz ** 2 - r ** 2
        disc = B ** 2 - 4 * A * C
        tc = np.full((H, W), np.inf)
        ok = disc >= 0
        t_hit = (-B[ok] - np.sqrt(disc[ok])) / (2 * A[ok])
        y_at = h + dy[ok] * t_hit
        good = (t_hit > 0.1) & (y_at >= 0.0) & (y_at <= ch)
        vals = np.full(t_hit.shape, np.inf)
        vals[good] = t_hit[good]
        tc[ok] = vals
        candidates.append(tc)

    stacked = np.stack([depth] + [np.where(np.isfinite(c), c, SKY) for c in candidates])
    return stacked.min(axis=0)


def _derive(depth, h=1.6, **kw):
    return derive_projection_proxies(
        depth, view_matrix=_view_matrix(h), fx=FX, fy=FY, cx=CX, cy=CY, **kw
    )


def _translation(prim):
    m = prim.transform_matrix
    return (m[0][3], m[1][3], m[2][3])


def _local_z(prim):
    m = prim.transform_matrix
    return np.array([m[0][2], m[1][2], m[2][2]])


def _local_y(prim):
    m = prim.transform_matrix
    return np.array([m[0][1], m[1][1], m[2][1]])


def _by_prefix(prims, prefix):
    return [p for p in prims if p.name.startswith(prefix)]


def test_derives_wall_position_and_orientation():
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)
    prims, stats = _derive(depth)
    walls = _by_prefix(prims, "projection_wall")
    assert len(walls) == 1
    t = _translation(walls[0])
    assert t[2] == pytest.approx(-10.0, abs=0.3)
    assert t[0] == pytest.approx(0.0, abs=0.5)
    assert t[1] == pytest.approx(1.5, abs=0.3)
    # Plane normal (local Z) faces the camera: +Z world.
    assert float(np.dot(_local_z(walls[0]), [0, 0, 1])) > 0.98
    assert walls[0].dimensions[1] == pytest.approx(3.0, abs=0.4)


def test_derives_ground_extent():
    depth = _scene_depth(wall_z=-10.0)
    prims, _ = _derive(depth)
    grounds = _by_prefix(prims, "projection_ground")
    assert len(grounds) == 1
    g = grounds[0]
    assert _translation(g)[1] == pytest.approx(0.0, abs=0.05)
    assert float(np.dot(_local_z(g), [0, 1, 0])) > 0.99  # normal is world up
    # Ground reaches toward the wall at z=-10 (center + half extent).
    z_far = _translation(g)[2] - g.dimensions[1] / 2.0
    assert z_far == pytest.approx(-10.0, abs=1.5)


def test_backdrop_always_present_and_faces_camera():
    # All-sky depth: no ground, no walls — backdrop only, no crash.
    depth = np.full((H, W), SKY)
    prims, _ = _derive(depth)
    assert not _by_prefix(prims, "projection_ground")
    assert not _by_prefix(prims, "projection_wall")
    backs = _by_prefix(prims, "projection_backdrop")
    assert len(backs) == 1
    assert float(np.dot(_local_z(backs[0]), [0, 0, 1])) > 0.99  # faces camera
    assert -_translation(backs[0])[2] >= 50.0  # at/beyond far depth


def test_depth_scale_reconciliation():
    # 2× the analytic depth, same camera height: rescale (s=0.5) must land the
    # ground on Y=0 and the wall back at z≈-10.
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0) * 2.0
    prims, stats = _derive(depth)
    assert stats["ground_scale"] == pytest.approx(0.5, abs=0.05)
    grounds = _by_prefix(prims, "projection_ground")
    walls = _by_prefix(prims, "projection_wall")
    assert grounds and walls
    assert _translation(grounds[0])[1] == pytest.approx(0.0, abs=0.05)
    assert _translation(walls[0])[2] == pytest.approx(-10.0, abs=0.5)


def test_derives_box_for_compact_object():
    # 1 m cube at z=-6 (front face) — too narrow to be a wall, becomes a box.
    depth = _scene_depth(box=(0.0, -6.0, 0.5, 1.0))
    prims, stats = _derive(depth)
    assert not _by_prefix(prims, "projection_wall")  # narrow fit is not a wall
    boxes = _by_prefix(prims, "projection_box")
    assert len(boxes) == 1
    t = _translation(boxes[0])
    assert t[2] == pytest.approx(-6.0, abs=0.5)
    assert t[0] == pytest.approx(0.0, abs=0.4)
    assert boxes[0].dimensions[1] == pytest.approx(1.0, abs=0.3)  # height
    # Footprint spans roughly the visible 1 m face.
    assert max(boxes[0].dimensions[0], boxes[0].dimensions[2]) == pytest.approx(1.0, abs=0.4)


def test_object_height_capped_at_plausibility_ceiling():
    # Regression test for a bug found via live browser verification (not
    # unit tests): a real photo's foreground-object clustering (XZ occupancy
    # grid, no height awareness) merged a genuine foreground object's
    # footprint with an unrelated tall background structure that wasn't
    # caught by wall-fitting, producing an implausible ~14m "foreground" box
    # that filled the entire viewport. A 10m box here exceeds
    # ProxyDerivationConfig's default object_max_height_m=6.0 — the fit
    # should clip to the plausible ceiling (refitting from only the points
    # below it) rather than passing an unbounded height through.
    depth = _scene_depth(box=(0.0, -6.0, 0.5, 10.0))
    prims, stats = _derive(depth)
    boxes = _by_prefix(prims, "projection_box")
    assert len(boxes) == 1
    from atlas_camera.core.proxy_geometry import ProxyDerivationConfig
    assert boxes[0].dimensions[1] <= ProxyDerivationConfig().object_max_height_m + 1e-6
    assert boxes[0].dimensions[1] > 0.2  # still produced a real object, not degenerate


def test_derives_cylinder_for_curved_object():
    depth = _scene_depth(cyl=(0.0, -8.0, 1.0, 3.0))
    prims, stats = _derive(depth)
    cyls = _by_prefix(prims, "projection_cylinder")
    assert len(cyls) == 1
    c = cyls[0]
    t = _translation(c)
    assert t[0] == pytest.approx(0.0, abs=0.3)
    assert t[2] == pytest.approx(-8.0, abs=0.5)
    radius = c.dimensions[0] / 2.0
    assert radius == pytest.approx(1.0, rel=0.25)
    assert c.dimensions[1] == pytest.approx(3.0, abs=0.5)


def test_payload_is_json_safe():
    depth = _scene_depth(wall_z=-10.0, box=(2.0, -5.0, 0.5, 1.0))
    prims, _ = _derive(depth)
    scene = AtlasProjectionScene()
    scene.proxy_geometry.extend(prims)
    payload = serialize_proxy_geometry(scene)
    assert payload  # at least ground/backdrop
    text = json.dumps(payload)
    assert text
    for entry in payload:
        assert len(entry["transform"]) == 16
        assert len(entry["dimensions"]) == 3
        assert all(isinstance(v, (str, int, float, bool)) or v is None
                   for v in entry["metadata"].values())


# ---------------------------------------------------------------------------
# vertical_extrusion — Photo-Pop-up-style silhouette height (see the
# "Sky-Aware Depth + Vertical-Silhouette Extrusion" design notes in CLAUDE.md)
# ---------------------------------------------------------------------------

def _scene_depth_with_noisy_sky_and_gable(h=1.6, wall_z=-10.0, wall_h=3.0, seed=7):
    """A wall (azimuth_walls' own documented ceiling) with a wide, smoothly
    sloped 'gable' above it — real geometry a sloped-roof silhouette, not
    noise — plus realistic noisy sky everywhere else, matching what Depth
    Anything actually produces. Returns (depth, true_gable_peak_y).
    """
    depth = _scene_depth(h=h, wall_z=wall_z, wall_h=wall_h)
    rng = np.random.RandomState(seed)
    is_sky = depth >= SKY
    depth = depth.copy()
    depth[is_sky] += rng.uniform(-8.0, 8.0, size=int(is_sky.sum()))

    rows = np.arange(H)
    cols = np.arange(W)
    row_mask = (rows >= 150) & (rows < 186)   # immediately above the wall's own top edge
    col_mask = (cols >= 180) & (cols < 330)   # wide — a realistic gable width, not a thin spike
    gable_rows, gable_cols = np.where(row_mask[:, None] & col_mask[None, :])
    t_gable = 10.0 + 0.3 * (186 - gable_rows)  # recedes smoothly as it rises — a real slope
    depth[gable_rows, gable_cols] = t_gable

    _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    true_peak_y = float((h + dy[gable_rows, gable_cols] * t_gable).max())
    return depth, true_peak_y


def test_vertical_extrusion_captures_gable_height_azimuth_walls_misses():
    # This is the exact failure mode reported on exterior_10_church:
    # azimuth_walls' wall_normal_max filter excludes the sloped gable's
    # points entirely, so its height reflects only the plain wall below.
    depth, true_peak_y = _scene_depth_with_noisy_sky_and_gable()

    walls_old = _by_prefix(_derive(depth)[0], "projection_wall")
    prims_new, _ = derive_vertical_extrusion_proxies(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
    )
    walls_new = _by_prefix(prims_new, "projection_wall")

    assert len(walls_old) == 1 and len(walls_new) == 1
    height_old = walls_old[0].dimensions[1]
    height_new = walls_new[0].dimensions[1]

    assert height_old == pytest.approx(2.9, abs=0.3)          # stuck at the plain wall
    assert height_new > height_old + 1.5                      # reaches well past it
    assert height_new == pytest.approx(true_peak_y, abs=1.0)  # close to the real gable peak


def test_vertical_extrusion_matches_azimuth_walls_on_a_simple_box_wall():
    # No sloped/complex feature above the wall — both methods should agree
    # closely on a plain box building (vertical_extrusion isn't a worse fit
    # for the common case just because it computes height differently).
    depth = _scene_depth(wall_z=-10.0, wall_h=3.0)

    walls_old = _by_prefix(_derive(depth)[0], "projection_wall")
    prims_new, _ = derive_vertical_extrusion_proxies(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
    )
    walls_new = _by_prefix(prims_new, "projection_wall")

    assert len(walls_old) == 1 and len(walls_new) == 1
    assert walls_new[0].dimensions[1] == pytest.approx(walls_old[0].dimensions[1], abs=0.5)


def test_vertical_extrusion_payload_is_json_safe():
    depth, _ = _scene_depth_with_noisy_sky_and_gable()
    prims, _ = derive_vertical_extrusion_proxies(
        depth, view_matrix=_view_matrix(1.6), fx=FX, fy=FY, cx=CX, cy=CY,
    )
    scene = AtlasProjectionScene()
    scene.proxy_geometry.extend(prims)
    payload = serialize_proxy_geometry(scene)
    assert payload
    assert json.dumps(payload)


def test_derive_projection_proxies_parity_with_depth_geometry():
    """Parity test verifying derive_projection_proxies matches core/depth_geometry.py shared functions."""
    from atlas_camera.core.depth_geometry import back_project_normals, fit_ground_and_scale, build_backdrop_primitive
    depth = _scene_depth(h=1.6, wall_z=-10.0, wall_h=3.0, box=(2.0, -8.0, 1.0, 2.0))
    vm = _view_matrix(1.6)

    prims, stats = derive_projection_proxies(
        depth, view_matrix=vm, fx=FX, fy=FY, cx=CX, cy=CY, horizon_y=H * 0.45,
    )

    bp = back_project_normals(depth, view_matrix=vm, fx=FX, fy=FY, cx=CX, cy=CY, depth_edge_rel=0.5)
    gf = fit_ground_and_scale(bp, horizon_y=H * 0.45)
    scaled_depth = depth * gf.scale
    backdrop = build_backdrop_primitive(
        bp=bp, scaled_depth=scaled_depth, valid_depth=bp.valid_depth,
        fx=FX, fy=FY, cx=CX, cy=cy_val if (cy_val := CY) else CY, width=W, height=H, scale=gf.scale,
    )

    assert stats["ground_scale"] == pytest.approx(gf.scale, abs=1e-12)
    assert stats["ground_inliers"] == gf.inliers

    backdrop_prim = next(p for p in prims if p.name == "projection_backdrop")
    assert backdrop_prim.dimensions == pytest.approx(backdrop.dimensions, abs=1e-9)
    assert np.array(backdrop_prim.transform_matrix) == pytest.approx(np.array(backdrop.transform_matrix), abs=1e-9)


