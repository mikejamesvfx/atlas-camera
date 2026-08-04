"""Tests for viewport-drawn N-gons: plane establishment + world-space packaging.

An occluded hole has no geometry to raycast against, so the plane is
established from the clicks that DO hit, and later clicks are ray x plane
intersections across the hole. These are the rules behind that, kept in core
(numpy only, no ComfyUI, no browser) and mirrored in atlas_blockout.js.
"""

import numpy as np
import pytest

from atlas_camera.core.polygon_planes import (
    establish_plane_from_hits,
    intersect_ray_with_plane,
    polygon_from_world_points,
)

UP = (0.0, 1.0, 0.0)


def _on_plane(point, plane, tol=1e-9):
    normal, offset = plane
    return abs(float(np.dot(np.asarray(normal), np.asarray(point))) - offset) < tol


# --- plane establishment ----------------------------------------------------

def test_two_hits_raise_a_vertical_plane_through_both():
    """The boundary case: only two edges of the hole are visible to click."""
    a = (1.0, 0.5, -4.0)
    b = (3.0, 2.5, -4.0)

    plane = establish_plane_from_hits([a, b])

    assert plane is not None
    normal, _offset = plane
    assert _on_plane(a, plane)
    assert _on_plane(b, plane)
    # Vertical means the plane CONTAINS world-up, i.e. its normal is horizontal.
    assert abs(float(np.dot(np.asarray(normal), np.asarray(UP)))) < 1e-9


def test_two_hits_at_different_heights_still_give_a_vertical_plane():
    plane = establish_plane_from_hits([(0.0, 0.0, -5.0), (2.0, 9.0, -5.0)])

    normal, _ = plane
    assert abs(float(np.dot(np.asarray(normal), np.asarray(UP)))) < 1e-9


def test_three_hits_best_fit_a_plane_through_all_of_them():
    pts = [(0.0, 0.0, -6.0), (2.0, 0.0, -6.0), (0.0, 3.0, -8.0)]

    plane = establish_plane_from_hits(pts)

    assert plane is not None
    for p in pts:
        assert _on_plane(p, plane, tol=1e-6)


def test_a_sloped_roof_from_four_hits_is_not_forced_vertical():
    # Four coplanar points on a tilted surface.
    pts = [(0.0, 2.0, -5.0), (4.0, 2.0, -5.0), (4.0, 4.0, -9.0), (0.0, 4.0, -9.0)]

    normal, _offset = establish_plane_from_hits(pts)

    assert abs(float(np.dot(np.asarray(normal), np.asarray(UP)))) > 0.1


def test_three_collinear_hits_fall_back_to_the_vertical_rule():
    """Collinear points have no unique best-fit plane; vertical is the sane one."""
    pts = [(0.0, 0.0, -5.0), (1.0, 1.0, -5.0), (2.0, 2.0, -5.0)]

    plane = establish_plane_from_hits(pts)

    assert plane is not None
    normal, _ = plane
    assert abs(float(np.dot(np.asarray(normal), np.asarray(UP)))) < 1e-9
    for p in pts:
        assert _on_plane(p, plane, tol=1e-6)


def test_one_hit_cannot_establish_a_plane():
    assert establish_plane_from_hits([(0.0, 0.0, -5.0)]) is None
    assert establish_plane_from_hits([]) is None


def test_two_hits_on_the_same_vertical_line_cannot_establish_a_plane():
    """Both points differ only in height: every vertical plane contains them."""
    assert establish_plane_from_hits([(1.0, 0.0, -5.0), (1.0, 4.0, -5.0)]) is None


# --- drawing across the hole ------------------------------------------------

def test_a_ray_over_a_hole_lands_on_the_established_plane():
    plane = establish_plane_from_hits([(1.0, 0.5, -4.0), (3.0, 2.5, -4.0)])
    origin = np.array([0.0, 1.6, 0.0])
    direction = np.array([0.2, 0.1, -1.0])

    hit = intersect_ray_with_plane(origin, direction, plane)

    assert hit is not None
    assert _on_plane(hit, plane, tol=1e-6)


def test_a_ray_pointing_away_from_the_plane_does_not_land():
    plane = establish_plane_from_hits([(1.0, 0.5, -4.0), (3.0, 2.5, -4.0)])

    assert intersect_ray_with_plane(
        np.array([0.0, 1.6, 0.0]), np.array([0.0, 0.0, 1.0]), plane) is None


def test_a_ray_parallel_to_the_plane_does_not_land():
    plane = ((0.0, 0.0, 1.0), -4.0)

    assert intersect_ray_with_plane(
        np.array([0.0, 1.6, 0.0]), np.array([1.0, 0.0, 0.0]), plane) is None


# --- packaging into a mesh primitive ---------------------------------------

def test_world_points_package_into_a_triangulated_polygon():
    quad = [(0.0, 0.0, -5.0), (4.0, 0.0, -5.0), (4.0, 3.0, -5.0), (0.0, 3.0, -5.0)]

    packed = polygon_from_world_points(quad, normal=(0.0, 0.0, 1.0))

    assert len(packed.vertices) == 12          # 4 corners x xyz
    assert len(packed.faces) == 6              # 2 triangles
    assert len(packed.uvs) == 8
    assert min(packed.uvs) >= 0.0 and max(packed.uvs) <= 1.0


def test_a_concave_outline_packages_without_inverted_triangles():
    ell = [(0.0, 0.0, -5.0), (6.0, 0.0, -5.0), (6.0, 2.0, -5.0),
           (2.0, 2.0, -5.0), (2.0, 6.0, -5.0), (0.0, 6.0, -5.0)]

    packed = polygon_from_world_points(ell, normal=(0.0, 0.0, 1.0))

    assert len(packed.faces) == (len(ell) - 2) * 3
    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    normal = np.asarray(packed.normal, dtype=float)
    for i in range(0, len(packed.faces), 3):
        a, b, c = (verts[packed.faces[i + k]] for k in range(3))
        assert float(np.dot(np.cross(b - a, c - a), normal)) > 0.0


def test_packaging_faces_the_polygon_normal_toward_a_given_camera():
    quad = [(0.0, 0.0, -5.0), (4.0, 0.0, -5.0), (4.0, 3.0, -5.0), (0.0, 3.0, -5.0)]

    packed = polygon_from_world_points(
        quad, normal=(0.0, 0.0, -1.0), camera_position=(0.0, 1.6, 0.0))

    centre = np.asarray(packed.vertices, dtype=float).reshape(-1, 3).mean(axis=0)
    to_cam = np.array([0.0, 1.6, 0.0]) - centre
    assert float(np.dot(np.asarray(packed.normal), to_cam)) > 0.0


def test_a_slightly_non_planar_outline_still_packages():
    """Edge-snapping lets a dragged point leave the plane deliberately.

    Points snap onto the mesh so the patch MEETS the geometry at a torn hole's
    rim; the rim is not perfectly flat, so the outline no longer is either.
    Triangulation projects into the plane basis, and the emitted mesh keeps the
    real world points.
    """
    quad = [(0.0, 0.0, -5.0), (4.0, 0.0, -5.0),
            (4.0, 3.0, -5.35), (0.0, 3.0, -4.7)]   # corners pulled off-plane

    packed = polygon_from_world_points(quad, normal=(0.0, 0.0, 1.0))

    assert len(packed.faces) == 6
    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    # The snapped depths survive: the mesh follows the geometry it snapped to.
    assert verts[2][2] == pytest.approx(-5.35)
    assert verts[3][2] == pytest.approx(-4.7)


def test_too_few_points_refuses_to_package():
    with pytest.raises(ValueError):
        polygon_from_world_points([(0.0, 0.0, -5.0), (1.0, 0.0, -5.0)],
                                  normal=(0.0, 0.0, 1.0))


def test_a_self_intersecting_outline_refuses_to_package():
    bowtie = [(0.0, 0.0, -5.0), (4.0, 4.0, -5.0), (4.0, 0.0, -5.0), (0.0, 4.0, -5.0)]

    with pytest.raises(ValueError):
        polygon_from_world_points(bowtie, normal=(0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# AtlasBlockoutViewport: drawn polygons -> solve / mask / report
# ---------------------------------------------------------------------------

import json


def _viewport_solve():
    from atlas_camera.core.schema import (
        AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera)
    intr = AtlasIntrinsics(
        image_width=256, image_height=256, focal_length_mm=35.0,
        sensor_width_mm=36.0, fx_px=250.0, fy_px=250.0, cx_px=128.0, cy_px=128.0)
    extr = AtlasExtrinsics(camera_view_matrix=(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -1.6),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0)))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _drawn_payload(fingerprint, points=None, enabled=True):
    quad = points or [[-2.0, 0.0, -6.0], [2.0, 0.0, -6.0],
                      [2.0, 3.0, -6.0], [-2.0, 3.0, -6.0]]
    return json.dumps({"drawn_polygons": [{
        "id": "d1", "label": "back building", "enabled": enabled,
        "points_world": quad,
        "plane": {"normal": [0.0, 0.0, 1.0], "offset": -6.0},
        "established_from": {"hits": 2, "rule": "vertical_through_two_hits"},
        "fingerprint": fingerprint,
    }]})


def _run_viewport(client_data, solve=None, **kwargs):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasBlockoutViewport
    solve = solve if solve is not None else _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    out = AtlasBlockoutViewport().render(
        solve, image, 256, client_data, **kwargs)
    return out["result"], solve, image


def _fingerprint_for(solve, image):
    from atlas_camera.comfy.fingerprints import _solve_fingerprint
    return _solve_fingerprint(solve, image)


def _drawn_prims(solve):
    return [p for p in solve.projection_scene.proxy_geometry
            if (p.metadata or {}).get("source") == "viewport_polygon"]


def test_viewport_appends_three_outputs_leaving_saved_slots_intact():
    """Saved workflows address outputs positionally — 0-11 must not move."""
    from atlas_camera.comfy.nodes import AtlasBlockoutViewport as V

    assert V.RETURN_NAMES[:12] == (
        "shaded", "depth", "normal", "mask", "path_frames", "camera_path",
        "patch_azimuth_view", "patch_elevation_view", "patch_distance",
        "patch_prompt", "patch_exact", "patch_render_mask")
    assert V.RETURN_NAMES[12:] == ("solve", "drawn_mask", "draw_report")
    assert V.RETURN_TYPES[12:] == ("ATLAS_SOLVE", "MASK", "STRING")
    assert len(V.RETURN_TYPES) == len(V.RETURN_NAMES) == 15


def test_nothing_drawn_still_passes_the_solve_through():
    """Unlike the patch outputs, this must never block a downstream export."""
    (result, solve, _img) = _run_viewport("")

    assert result[12] is solve or result[12] == solve
    assert result[12].projection_scene.proxy_geometry == []
    assert "no polygons" in result[14].lower()


def test_a_drawn_polygon_becomes_a_mesh_primitive_on_the_solve():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    payload = _drawn_payload(_fingerprint_for(solve, image))

    (result, _s, _i) = _run_viewport(payload, solve=solve)

    out_solve = result[12]
    prims = _drawn_prims(out_solve)
    assert len(prims) == 1
    prim = prims[0]
    assert prim.primitive_type == "mesh"
    assert prim.metadata["role"] == "projection_proxy"
    assert len(prim.metadata["vertices"]) == 12
    assert len(prim.metadata["faces"]) == 6
    # World points ride through untouched — Python re-fits nothing.
    assert prim.metadata["vertices"][:3] == [-2.0, 0.0, -6.0]
    assert "back building" in result[14]


def test_drawn_polygons_append_without_clobbering_existing_geometry():
    torch = pytest.importorskip("torch")
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import AtlasProxyPrimitive
    solve = _viewport_solve()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="projection_relief_mesh", primitive_type="mesh",
        metadata={"role": PROXY_ROLE, "source": "depth_relief_mesh"}))
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _drawn_payload(_fingerprint_for(solve, image)), solve=solve)

    names = [p.name for p in result[12].projection_scene.proxy_geometry]
    assert "projection_relief_mesh" in names
    assert len(_drawn_prims(result[12])) == 1


def test_a_stale_fingerprint_does_not_apply_and_says_so():
    """Outlines drawn against another solve/image must not silently re-apply."""
    (result, _s, _i) = _run_viewport(_drawn_payload("not-this-solve"))

    assert _drawn_prims(result[12]) == []
    assert "stale" in result[14].lower()


def test_a_disabled_polygon_is_skipped():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _drawn_payload(_fingerprint_for(solve, image), enabled=False), solve=solve)

    assert _drawn_prims(result[12]) == []


def test_drawn_mask_marks_where_the_polygon_projects():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _drawn_payload(_fingerprint_for(solve, image)), solve=solve)

    drawn_mask = result[13]
    assert tuple(drawn_mask.shape) == (1, 256, 256)
    assert float(drawn_mask.max()) > 0.5, "the quad is in front of the camera"
    assert float(drawn_mask.float().mean()) < 1.0, "it does not cover the frame"


def test_a_degenerate_polygon_is_reported_not_raised():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    bowtie = [[-2.0, 0.0, -6.0], [2.0, 3.0, -6.0],
              [2.0, 0.0, -6.0], [-2.0, 3.0, -6.0]]

    (result, _s, _i) = _run_viewport(
        _drawn_payload(_fingerprint_for(solve, image), points=bowtie), solve=solve)

    assert _drawn_prims(result[12]) == []
    assert "skipped" in result[14].lower()


# ---------------------------------------------------------------------------
# AtlasRetopologizeLayer: drawn N-gons join the relief mesh in the union
# ---------------------------------------------------------------------------

def _mesh_prim(name, source, *, z=-6.0):
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import AtlasProxyPrimitive
    quad = [[-2.0, 0.0, z], [2.0, 0.0, z], [2.0, 3.0, z], [-2.0, 3.0, z]]
    return AtlasProxyPrimitive(
        name=name, primitive_type="mesh",
        metadata={
            "role": PROXY_ROLE, "source": source,
            "vertices": [c for p in quad for c in p],
            "faces": [0, 1, 2, 0, 2, 3],
            "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        })


def test_retopo_no_longer_skips_viewport_drawn_planes():
    """The merge that makes one exportable mesh: drawn N-gons must be seen."""
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer
    solve = _viewport_solve()
    solve.projection_scene.proxy_geometry.append(
        _mesh_prim("drawn_plane_01", "viewport_polygon"))

    _out, report = AtlasRetopologizeLayer().retopo(solve, layer="*")

    assert "drawn_plane_01" in report, (
        "a drawn plane must appear in the retopo report, not be silently dropped")


def test_retopo_collects_relief_mesh_and_drawn_planes_together():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer
    solve = _viewport_solve()
    solve.projection_scene.proxy_geometry.extend([
        _mesh_prim("projection_relief_mesh", "depth_relief_mesh"),
        _mesh_prim("drawn_plane_01", "viewport_polygon", z=-9.0),
    ])

    _out, report = AtlasRetopologizeLayer().retopo(solve, layer="*")

    assert "projection_relief_mesh" in report
    assert "drawn_plane_01" in report


def test_retopo_still_ignores_non_mesh_primitives():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import AtlasProxyPrimitive
    solve = _viewport_solve()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="projection_backdrop", primitive_type="plane",
        metadata={"role": PROXY_ROLE, "source": "depth_derivation"}))

    _out, report = AtlasRetopologizeLayer().retopo(solve, layer="*")

    assert "projection_backdrop" not in report
