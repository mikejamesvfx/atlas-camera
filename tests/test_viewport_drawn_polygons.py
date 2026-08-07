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


# ---------------------------------------------------------------------------
# Blockout boxes: 8 corners -> a closed mesh
# ---------------------------------------------------------------------------

from atlas_camera.core.polygon_planes import box_mesh_from_corners


def _unit_box(h=3.0):
    """Bottom ring CCW seen from above, then the matching top ring."""
    return [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, -4.0), (0.0, 0.0, -4.0),
            (0.0, h, 0.0), (2.0, h, 0.0), (2.0, h, -4.0), (0.0, h, -4.0)]


def _faces_as_tris(packed):
    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    return [(verts[packed.faces[i]], verts[packed.faces[i + 1]], verts[packed.faces[i + 2]])
            for i in range(0, len(packed.faces), 3)]


def test_a_box_is_eight_corners_and_twelve_triangles():
    packed = box_mesh_from_corners(_unit_box())

    assert len(packed.vertices) == 24        # 8 corners x xyz
    assert len(packed.faces) == 36           # 6 quads x 2 tris x 3 indices
    assert len(packed.uvs) == 16


def test_every_box_face_points_outward():
    """A closed blockout box is only useful if it is not inside-out."""
    packed = box_mesh_from_corners(_unit_box())
    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    centroid = verts.mean(axis=0)

    for a, b, c in _faces_as_tris(packed):
        normal = np.cross(b - a, c - a)
        outward = ((a + b + c) / 3.0) - centroid
        assert float(np.dot(normal, outward)) > 0.0


def test_faces_stay_outward_after_a_corner_is_dragged():
    """Edit moves individual corners, so winding cannot assume a neat cuboid."""
    skewed = _unit_box()
    skewed[6] = (3.4, 4.2, -5.1)      # drag one top corner well out
    skewed[1] = (2.6, -0.3, 0.4)      # and one bottom corner

    packed = box_mesh_from_corners(skewed)
    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    centroid = verts.mean(axis=0)

    assert len(packed.faces) == 36
    for a, b, c in _faces_as_tris(packed):
        normal = np.cross(b - a, c - a)
        outward = ((a + b + c) / 3.0) - centroid
        assert float(np.dot(normal, outward)) > 0.0


def test_box_corners_ride_through_untouched():
    corners = _unit_box()

    packed = box_mesh_from_corners(corners)

    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    for given, got in zip(corners, verts):
        assert got == pytest.approx(np.asarray(given))


def test_a_box_needs_exactly_eight_corners():
    with pytest.raises(ValueError):
        box_mesh_from_corners(_unit_box()[:6])


def test_a_flat_box_is_rejected():
    """Zero height is a degenerate solid, not a blockout."""
    flat = _unit_box(h=0.0)

    with pytest.raises(ValueError):
        box_mesh_from_corners(flat)


def _box_payload(fingerprint, corners=None, kind="box"):
    return json.dumps({"drawn_polygons": [{
        "id": "b1", "label": "back building mass", "enabled": True,
        "kind": kind,
        "points_world": corners or _unit_box(),
        "fingerprint": fingerprint,
    }]})


def test_a_drawn_box_becomes_a_closed_mesh_primitive():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _box_payload(_fingerprint_for(solve, image)), solve=solve)

    prims = [p for p in result[12].projection_scene.proxy_geometry
             if (p.metadata or {}).get("source") == "viewport_box"]
    assert len(prims) == 1
    assert len(prims[0].metadata["vertices"]) == 24
    assert len(prims[0].metadata["faces"]) == 36
    assert "back building mass" in result[14]


def test_a_degenerate_box_is_reported_not_raised():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _box_payload(_fingerprint_for(solve, image), corners=_unit_box(h=0.0)),
        solve=solve)

    assert [p for p in result[12].projection_scene.proxy_geometry
            if (p.metadata or {}).get("source") == "viewport_box"] == []
    assert "skipped" in result[14].lower()


def test_retopo_collects_drawn_boxes_too():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer
    solve = _viewport_solve()
    solve.projection_scene.proxy_geometry.append(
        _mesh_prim("drawn_box_01", "viewport_box"))

    _out, report = AtlasRetopologizeLayer().retopo(solve, layer="*")

    assert "drawn_box_01" in report


# ---------------------------------------------------------------------------
# Blockout spheres: two control points -> a closed mesh
# ---------------------------------------------------------------------------

from atlas_camera.core.polygon_planes import (
    SPHERE_RINGS,
    SPHERE_SEGMENTS,
    sphere_mesh_from_control_points,
)


def test_a_sphere_is_built_from_a_centre_and_a_surface_point():
    """Only TWO handles, so Edit drags a sphere with no sphere-specific code."""
    packed = sphere_mesh_from_control_points((1.0, 2.0, -5.0), (4.0, 2.0, -5.0))

    n_verts = (SPHERE_RINGS + 1) * (SPHERE_SEGMENTS + 1)
    assert len(packed.vertices) == n_verts * 3
    assert len(packed.uvs) == n_verts * 2
    # Two triangles per cell, minus the degenerate one in each polar row.
    assert len(packed.faces) == (2 * SPHERE_RINGS * SPHERE_SEGMENTS
                                 - 2 * SPHERE_SEGMENTS) * 3


def test_every_sphere_vertex_sits_on_the_radius():
    centre = np.array([1.0, 2.0, -5.0])
    packed = sphere_mesh_from_control_points(centre, (4.0, 2.0, -5.0))

    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)
    radii = np.linalg.norm(verts - centre, axis=1)
    # Vertices are rounded to 4 dp on the way out (the same compaction
    # relief_mesh_primitive uses to keep solve JSON small), so 1e-4 is the
    # tightest honest tolerance here.
    assert radii.min() == pytest.approx(3.0, abs=1e-4)
    assert radii.max() == pytest.approx(3.0, abs=1e-4)


def test_sphere_faces_point_outward():
    centre = np.array([0.0, 0.0, 0.0])
    packed = sphere_mesh_from_control_points(centre, (2.0, 0.0, 0.0))
    verts = np.asarray(packed.vertices, dtype=float).reshape(-1, 3)

    for i in range(0, len(packed.faces), 3):
        a, b, c = (verts[packed.faces[i + k]] for k in range(3))
        normal = np.cross(b - a, c - a)
        outward = (a + b + c) / 3.0 - centre
        assert float(np.dot(normal, outward)) > 0.0


def test_the_radius_is_the_distance_between_the_control_points():
    """Dragging the surface handle in any direction only changes the radius."""
    a = sphere_mesh_from_control_points((0.0, 0.0, 0.0), (0.0, 5.0, 0.0))
    b = sphere_mesh_from_control_points((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))

    va = np.asarray(a.vertices, dtype=float).reshape(-1, 3)
    vb = np.asarray(b.vertices, dtype=float).reshape(-1, 3)
    assert np.allclose(np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1))


def test_a_zero_radius_sphere_is_rejected():
    with pytest.raises(ValueError):
        sphere_mesh_from_control_points((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def test_a_sphere_needs_exactly_two_control_points():
    with pytest.raises(ValueError):
        sphere_mesh_from_control_points((0.0, 0.0, 0.0), None)


def test_an_unknown_shape_kind_is_named_not_guessed_at():
    """A viewport newer than the running Python must not fail silently.

    Live on 2026-08-04: a ComfyUI started before box support received
    kind="box" records, treated them as outlines, and reported them as
    self-intersecting polygons — true of the projected corners, but useless as
    a diagnosis.
    """
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _box_payload(_fingerprint_for(solve, image), kind="dodecahedron"),
        solve=solve)

    assert result[12].projection_scene.proxy_geometry == []
    assert "unknown shape kind" in result[14]
    assert "restart ComfyUI" in result[14]


def _sphere_payload(fingerprint, points=None):
    return json.dumps({"drawn_polygons": [{
        "id": "s1", "label": "dome mass", "enabled": True, "kind": "sphere",
        "points_world": points or [[0.0, 3.0, -8.0], [3.0, 3.0, -8.0]],
        "fingerprint": fingerprint,
    }]})


def test_a_drawn_sphere_becomes_a_closed_mesh_primitive():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _sphere_payload(_fingerprint_for(solve, image)), solve=solve)

    prims = [p for p in result[12].projection_scene.proxy_geometry
             if (p.metadata or {}).get("source") == "viewport_sphere"]
    assert len(prims) == 1
    n_verts = (SPHERE_RINGS + 1) * (SPHERE_SEGMENTS + 1)
    assert len(prims[0].metadata["vertices"]) == n_verts * 3
    assert "dome mass" in result[14]


def test_a_sphere_with_coincident_control_points_is_reported():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _sphere_payload(_fingerprint_for(solve, image),
                        points=[[0.0, 3.0, -8.0], [0.0, 3.0, -8.0]]),
        solve=solve)

    assert [p for p in result[12].projection_scene.proxy_geometry
            if (p.metadata or {}).get("source") == "viewport_sphere"] == []
    assert "skipped" in result[14].lower()


def test_a_sphere_needs_two_control_points_to_be_built():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    (result, _s, _i) = _run_viewport(
        _sphere_payload(_fingerprint_for(solve, image), points=[[0.0, 3.0, -8.0]]),
        solve=solve)

    assert _drawn_prims(result[12]) == []
    assert "skipped" in result[14].lower()


def test_retopo_collects_drawn_spheres_too():
    from atlas_camera.comfy.nodes import AtlasRetopologizeLayer
    solve = _viewport_solve()
    solve.projection_scene.proxy_geometry.append(
        _mesh_prim("drawn_sphere_01", "viewport_sphere"))

    _out, report = AtlasRetopologizeLayer().retopo(solve, layer="*")

    assert "drawn_sphere_01" in report


# ---------------------------------------------------------------------------
# Default pixel fill: the drawn footprint smeared in from its surroundings
# ---------------------------------------------------------------------------

def _payload_for(unique_id):
    from atlas_camera.comfy.node_helpers import _ATLAS_BLOCKOUT_CACHE
    return _ATLAS_BLOCKOUT_CACHE[unique_id]


def test_a_drawn_surface_gets_a_plate_smeared_in_from_its_surroundings():
    """The camera never photographed behind an occluder, so projecting the
    untouched plate onto a drawn surface paints it with the OCCLUDER. The
    default is the same deterministic edge-extend the hole-fill path uses."""
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.rand((1, 256, 256, 3), dtype=torch.float32)
    from atlas_camera.comfy.nodes import AtlasBlockoutViewport
    AtlasBlockoutViewport().render(
        solve, image, 256, _drawn_payload(_fingerprint_for(solve, image)),
        unique_id="fill-on")

    payload = _payload_for("fill-on")
    assert payload.get("drawn_plate_b64", "").startswith("data:image/")


def test_a_connected_clean_plate_feeds_the_background_not_the_fills():
    """clean_plate is the BACKGROUND layer (backdrop plane + merged solve_b
    geometry): it is published as clean_plate_b64, while drawn/wand fills —
    which patch the PRIMARY layer's own tears — keep the source-plate smear.
    An earlier build routed the clean plate onto the fills too and machine
    tears came back filled with background (found live, layered solve)."""
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.rand((1, 256, 256, 3), dtype=torch.float32)
    clean = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    clean[..., 1] = 1.0     # flat green — trivially distinct from the plate
    from atlas_camera.comfy.nodes import AtlasBlockoutViewport
    AtlasBlockoutViewport().render(
        solve, image, 256, _drawn_payload(_fingerprint_for(solve, image)),
        clean_plate=clean, unique_id="clean-plate")

    payload = _payload_for("clean-plate")
    clean_b64 = payload.get("clean_plate_b64", "")
    drawn_b64 = payload.get("drawn_plate_b64", "")
    assert clean_b64.startswith("data:image/")
    assert drawn_b64.startswith("data:image/")
    assert drawn_b64 != clean_b64        # fills keep the smear
    # The published clean layer IS the clean plate: flat green dominates.
    import base64 as _b64
    from io import BytesIO
    from PIL import Image as PILImage
    raw = _b64.b64decode(clean_b64.split(",", 1)[1])
    arr = PILImage.open(BytesIO(raw)).convert("RGB")
    import numpy as _np
    px = _np.asarray(arr, dtype=_np.float32)
    assert px[..., 1].mean() > 200 and px[..., 0].mean() < 60


def test_incremental_smear_matches_a_fresh_full_compute():
    """The per-node smear cache re-smears only a crop around the mask diff;
    the result must be byte-identical to a fresh full-frame pass, including
    when the new fill lands close enough to an old one that the crop has to
    grow to keep the old fill's smear neighborhood intact."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from atlas_camera.comfy.nodes_viewport import (
        _drawn_fill_plate_b64, _DRAWN_FILL_CACHE)

    torch.manual_seed(7)
    image = torch.rand((1, 128, 128, 3), dtype=torch.float32)
    mask_a = np.zeros((128, 128), dtype=bool)
    mask_a[20:40, 20:40] = True
    mask_ab = mask_a.copy()
    mask_ab[44:60, 30:50] = True      # near A: forces the crop-growth path
    mask_ab[90:110, 80:100] = True    # far from A: plain incremental region

    _DRAWN_FILL_CACHE.clear()
    _drawn_fill_plate_b64(image, mask_a, 16, cache_key="inc")      # prime
    incremental = _drawn_fill_plate_b64(image, mask_ab, 16, cache_key="inc")
    fresh = _drawn_fill_plate_b64(image, mask_ab, 16)              # no cache
    assert incremental == fresh

    # An unchanged mask short-circuits to the cached encoding.
    again = _drawn_fill_plate_b64(image, mask_ab.copy(), 16, cache_key="inc")
    assert again == incremental

    # px=0 / empty mask drops the entry rather than serving a stale plate.
    _drawn_fill_plate_b64(image, np.zeros((128, 128), dtype=bool), 16,
                          cache_key="inc")
    assert "inc" not in _DRAWN_FILL_CACHE


def test_no_smear_plate_when_the_fill_is_switched_off():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.rand((1, 256, 256, 3), dtype=torch.float32)
    from atlas_camera.comfy.nodes import AtlasBlockoutViewport
    AtlasBlockoutViewport().render(
        solve, image, 256, _drawn_payload(_fingerprint_for(solve, image)),
        drawn_fill_px=0, unique_id="fill-off")

    assert _payload_for("fill-off").get("drawn_plate_b64", "") == ""


def test_no_smear_plate_when_nothing_was_drawn():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasBlockoutViewport
    AtlasBlockoutViewport().render(
        _viewport_solve(), torch.zeros((1, 256, 256, 3)), 256, "",
        unique_id="fill-none")

    assert _payload_for("fill-none").get("drawn_plate_b64", "") == ""


def test_the_smear_only_repaints_inside_the_drawn_footprint():
    """Pixels outside the footprint are real photography and must not move."""
    np_ = pytest.importorskip("numpy")
    from atlas_camera.plate.ops import _extend_edge_colors

    rgb = np_.zeros((64, 64, 3), dtype="float32")
    rgb[:] = 200.0
    hole = np_.zeros((64, 64), dtype=bool)
    hole[20:40, 20:40] = True
    rgb[hole] = 0.0

    filled, _grown = _extend_edge_colors(rgb, ~hole, 64)

    assert filled[hole].max() > 0.0, "the footprint should be filled in"
    assert np_.allclose(filled[~hole], 200.0), "real pixels must be untouched"


# --- tool-rail collapse toggle (atlas_blockout.js, 2026-08-07) ---------------
#
# Text pins, because the rail is browser-only DOM with no Python side. What
# these protect is not the styling but three decisions that are easy to undo by
# accident and expensive to notice: the tools default to VISIBLE, the toggle is
# NOT inside the container it hides, and collapsing is presentation-only.

import os as _os

_WEB_DIR = _os.path.join(_os.path.dirname(__file__), "..",
                         "atlas_camera", "comfy", "web")


def _blockout_src():
    return open(_os.path.join(_WEB_DIR, "atlas_blockout.js"),
                encoding="utf-8").read()


def test_rail_tools_are_visible_by_default():
    assert "let railToolsVisible = true;" in _blockout_src()


def test_the_toggle_is_not_inside_the_container_it_hides():
    """syncRailCollapsed hides `railTools`, so the toggle must be a sibling.
    Appending it to railTools instead would hide the button along with the
    tools and leave no way to bring them back."""
    src = _blockout_src()
    assert "drawRail.append(railToggleBtn, railTools);" in src
    assert 'railTools.style.display = railToolsVisible ? "flex" : "none";' in src
    # the toggle is appended to the rail, never to the collapsing container
    assert "railTools.append(railToggleBtn" not in src


def test_collapsing_the_rail_does_not_touch_draw_state():
    """Folding the rail away mid-draw must not discard shapes or flip tools —
    it is a view control, not an escape hatch."""
    src = _blockout_src()
    start = src.index("railToggleBtn.onclick")
    body = src[start:start + 240]
    for forbidden in ("drawnPolygons", "drawDirty", "drawOn", "editOn", "editSnap"):
        assert forbidden not in body, f"collapse handler must not touch {forbidden}"


# --- the drawn fill renders in the window it was drawn in --------------------
#
# Reported live 2026-08-07: an artist wand-filled holes, pressed Apply, and had
# to wire a SECOND viewport off this node's `solve` output to see the result —
# where the fills then rendered BLACK.
#
# One cause behind both halves. The browser payload was extracted from the
# INPUT solve, before _apply_drawn_polygons ran, so the baked meshes existed
# only on the `solve` OUTPUT and this viewport could show them nowhere. A
# downstream viewport does inherit the geometry (it rides the solve) but NOT
# the smear: `drawn_plate_b64` is rebuilt per node from that node's own
# client_data, so a viewport that drew nothing yields an empty plate — and the
# frontend reads an empty plate as "no drawn surfaces" and strips
# _projMaterial from every atlasDrawn mesh, which is why they came out black
# rather than merely mistextured.


def _payload_after(client_data, solve=None):
    """Render once and return the payload the browser would fetch."""
    from atlas_camera.comfy.node_helpers import _ATLAS_BLOCKOUT_CACHE
    _ATLAS_BLOCKOUT_CACHE.clear()
    result, solve_used, image = _run_viewport(client_data, solve=solve)
    assert _ATLAS_BLOCKOUT_CACHE, "render must publish a payload for the browser"
    return list(_ATLAS_BLOCKOUT_CACHE.values())[-1], result


def _payload_drawn_meshes(payload):
    """Proxies the frontend will flag atlasDrawn (DRAWN_PROXY_SOURCES)."""
    drawn = {"viewport_polygon", "viewport_box", "viewport_sphere"}
    return [p for p in (payload.get("proxy_geometry") or [])
            if ((p.get("metadata") or {}).get("source")) in drawn]


def test_the_payload_carries_the_geometry_this_viewport_just_baked():
    """The fix: after Apply, THIS viewport's own payload contains the drawn
    mesh, so the fill appears in the window it was drawn in."""
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    fingerprint = _fingerprint_for(solve, image)

    payload, result = _payload_after(_drawn_payload(fingerprint), solve=solve)

    assert _drawn_prims(result[12]), "precondition: the polygon was baked"
    assert _payload_drawn_meshes(payload), (
        "the render payload must carry the drawn mesh — without it the fill is "
        "invisible in this viewport and only a second downstream viewport can "
        "show it")


def test_the_drawn_smear_travels_with_the_geometry_it_belongs_to():
    """Geometry and its texture must appear in the SAME payload. Split across
    two nodes, the frontend deletes the material and the fill renders black."""
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    fingerprint = _fingerprint_for(solve, image)

    payload, _ = _payload_after(_drawn_payload(fingerprint), solve=solve)

    assert _payload_drawn_meshes(payload)
    assert payload.get("drawn_plate_b64"), (
        "a payload carrying atlasDrawn meshes must also carry the smear plate "
        "they project — an empty plate makes buildDrawnMat strip their "
        "material and they render black")


def test_nothing_drawn_leaves_the_payload_untouched():
    """The re-extraction is conditional: no polygons means no extra work and
    no change to what the browser receives."""
    payload, result = _payload_after("")
    assert not _payload_drawn_meshes(payload)
    assert not payload.get("drawn_plate_b64")


def test_a_skipped_outline_is_reported_to_the_browser():
    """A dropped fill must be explainable from the viewport alone.

    draw_report was an output-only STRING, so an outline skipped for a stale
    fingerprint / unknown kind / bad geometry simply failed to appear and the
    artist had nothing to read — three drawn, two filled, no reason given.
    The gate doctrine forbids exactly that silence, so the report now rides the
    payload too and the frontend raises it in the draw HUD.
    """
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    # A deliberately STALE outline: drawn against a different solve/image.
    payload, result = _payload_after(_drawn_payload("not-the-current-fingerprint"),
                                     solve=solve)
    assert "skipped(" in (payload.get("draw_report") or ""), (
        "the payload must carry the per-outline report so the viewport can "
        "explain a dropped fill")
    assert "skipped(" in result[14], "the STRING output keeps its contract too"


def test_a_clean_apply_reports_no_skips():
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    payload, _ = _payload_after(_drawn_payload(_fingerprint_for(solve, image)),
                                solve=solve)
    assert "skipped(" not in (payload.get("draw_report") or "")


def test_re_extraction_keeps_the_input_solve_fingerprint():
    """The payload's fingerprint must stay the INPUT solve's.

    Re-extracting the payload from the APPLIED solve is one small step away
    from also re-deriving the fingerprint off that solve — and that would be
    quietly catastrophic: the browser stamps newly drawn outlines with the
    fingerprint it was handed, Python still gates against the input solve's,
    and every outline drawn after the first Apply would be dropped as
    'stale (drawn against another solve/image)'. Exactly the symptom this
    whole investigation started from, and it would look like the fills
    randomly stopped working.
    """
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    expected = _fingerprint_for(solve, image)

    payload, _ = _payload_after(_drawn_payload(expected), solve=solve)

    assert payload.get("solve_fingerprint") == expected, (
        "the payload must keep the INPUT solve's fingerprint — deriving it "
        "from the applied solve makes every later outline read as stale")


# --- naming the self-intersection (2026-08-07) -------------------------------
#
# Two wand fills came back "skipped(polygon is self-intersecting)" and that was
# the end of the trail: the message named the category, not the defect, so there
# was no way to tell a rim that PINCHES (walks back through a vertex it already
# used — a torn relief mesh where two tears meet at a shared vertex) from one
# whose edges genuinely CROSS. Those want different fixes, and one of them —
# a rim that merely touches itself without crossing — is a perfectly fillable
# shape we are currently refusing.

from atlas_camera.core.polygon_planes import triangulate_polygon


def _reason(points):
    with pytest.raises(ValueError) as excinfo:
        triangulate_polygon(points)
    return str(excinfo.value)


def test_a_repeated_vertex_is_named_as_such():
    pinch = [(0, 0), (3, 0), (3, 1), (1.5, 1), (3, 2), (3, 3), (0, 3), (1.5, 1)]
    reason = _reason(pinch)
    assert "self-intersecting" in reason
    assert "vertex 7" in reason and "vertex 3" in reason, reason


def test_a_touching_loop_is_reported_as_a_repeat_not_a_crossing():
    """It shares a point without any edge crossing — calling that a crossing
    would send the fix in the wrong direction."""
    touch = [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 1)]
    reason = _reason(touch)
    assert "repeat" in reason.lower(), reason
    assert "cross" not in reason.lower(), reason


def test_genuinely_crossing_edges_name_the_edge_pair():
    bowtie = [(0, 0), (4, 4), (4, 0), (0, 4)]
    reason = _reason(bowtie)
    assert "cross" in reason.lower(), reason
    # the two offending edges are identified by index, not just "somewhere"
    assert "edge" in reason.lower(), reason


def test_the_reason_reaches_the_viewport_hud():
    """End to end: the detail must survive into the payload the HUD reads."""
    torch = pytest.importorskip("torch")
    solve = _viewport_solve()
    image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    fingerprint = _fingerprint_for(solve, image)
    bowtie = [[0.0, 0.0, -5.0], [4.0, 4.0, -5.0], [4.0, 0.0, -5.0], [0.0, 4.0, -5.0]]
    payload, _ = _payload_after(_drawn_payload(fingerprint, points=bowtie),
                                solve=solve)
    report = payload.get("draw_report") or ""
    assert "skipped(" in report
    assert "edge" in report.lower(), (
        "the HUD must receive the specific defect, not just the category:\n"
        + report)
