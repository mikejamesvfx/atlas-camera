"""Tests for pre-mesh depth completion.

The headline test is again ANALYTIC. A hole punched into a known plane must be
refilled with the plane's own depth to floating-point accuracy, because the
ray-plane tier is exact rather than interpolative — that distinction is the
entire justification for preferring it over a learned layered model for
background continuation, so it is worth pinning to 1e-9 rather than a tolerance.

The rest is about provenance and refusal: measured depth must never be
overwritten, every invented pixel must be identifiable and attributed to a
tier, and a hole no tier can justify must stay a hole.
"""

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.depth_completion import (
    METHOD_DIFFUSION,
    METHOD_MEASURED,
    METHOD_RAY_PLANE,
    METHOD_TANGENT,
    complete_depth,
    pixel_rays,
    ray_plane_depth,
)

H, W, F = 64, 96, 96.0
CX, CY = W / 2.0, H / 2.0


def _view(eye=(0.0, 0.0, 0.0)):
    view, _, _ = look_at_view_matrix(eye, (eye[0], eye[1], eye[2] - 1.0), (0.0, 1.0, 0.0))
    return np.asarray(view, dtype=np.float64)


def _fronto_plane_depth(distance=8.0):
    """A plane at z=-distance seen head-on: every pixel's forward depth equals
    the distance, since forward depth is measured along -Z, not along the ray."""
    return np.full((H, W), float(distance)), {"normal": [0.0, 0.0, 1.0],
                                              "d": -float(distance)}


def _tilted_plane_depth(normal=(0.0, 0.6, 0.8), d=-6.0):
    """Analytic depth of an arbitrary plane, from the rays themselves."""
    cam, dirs = pixel_rays(np, H, W, view_matrix=_view(), fx=F, fy=F, cx=CX, cy=CY)
    depth, valid = ray_plane_depth(np, cam, dirs, np.asarray(normal), d)
    return depth, valid, {"normal": list(normal), "d": d}


def _complete(depth, holes, planes=None, **kw):
    return complete_depth(depth, view_matrix=_view(), fx=F, fy=F, cx=CX, cy=CY,
                          holes=holes, planes=planes, **kw)


# --- the analytic oracle ---

def test_ray_plane_refills_a_punched_hole_exactly():
    depth, plane = _fronto_plane_depth(8.0)
    truth = depth.copy()
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True
    punched = np.where(holes, np.nan, depth)

    out = _complete(punched, holes, planes=[plane])

    assert np.allclose(out.depth, truth, atol=1e-9), "ray-plane is exact, not approximate"
    assert (out.method_map[holes] == METHOD_RAY_PLANE).all()
    assert out.synthesized_mask[holes].all()


def test_ray_plane_is_exact_on_a_tilted_plane_too():
    """A fronto-parallel plane would pass even with a constant-fill bug."""
    truth, valid, plane = _tilted_plane_depth()
    holes = np.zeros((H, W), dtype=bool)
    holes[10:30, 20:50] = True
    holes &= valid
    punched = np.where(holes, np.nan, truth)

    out = _complete(punched, holes, planes=[plane])

    assert np.allclose(out.depth[holes], truth[holes], atol=1e-9)
    # The fill must actually vary across the hole, not be a single value.
    assert out.depth[holes].std() > 1e-3


def test_completed_depth_reprojects_to_the_pixel_it_came_from():
    """The completion lives on the pixel's own ray, which is what makes the
    repaired map safe to feed straight back into the mesh builder."""
    depth, plane = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[30:35, 40:45] = True
    out = _complete(np.where(holes, np.nan, depth), holes, planes=[plane])

    cam, dirs = pixel_rays(np, H, W, view_matrix=_view(), fx=F, fy=F, cx=CX, cy=CY)
    pts = cam + dirs * out.depth[..., None]
    u = CX + F * pts[..., 0] / -pts[..., 2]
    v = CY - F * pts[..., 1] / -pts[..., 2]
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    assert np.allclose(u[holes], uu[holes], atol=1e-6)
    assert np.allclose(v[holes], vv[holes], atol=1e-6)


# --- provenance ---

def test_measured_depth_is_never_overwritten():
    depth, plane = _fronto_plane_depth(8.0)
    depth[5, 5] = 3.21                        # a real measurement off the plane
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True

    out = _complete(np.where(holes, np.nan, depth), holes, planes=[plane])

    assert out.depth[5, 5] == pytest.approx(3.21)
    assert out.method_map[5, 5] == METHOD_MEASURED
    assert not out.synthesized_mask[5, 5]


def test_every_invented_pixel_is_attributed_to_a_tier():
    depth, plane = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True
    out = _complete(np.where(holes, np.nan, depth), holes, planes=[plane])

    assert (out.method_map[out.synthesized_mask] != 0).all()
    assert set(out.method_histogram()) <= {"measured", "ray_plane", "tangent", "diffusion"}
    assert out.synthesized_fraction == pytest.approx(holes.mean())


def test_confidence_falls_as_weaker_tiers_do_more_of_the_work():
    depth, plane = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True
    punched = np.where(holes, np.nan, depth)

    exact = _complete(punched, holes, planes=[plane])
    guessed = _complete(punched, holes, planes=None)

    assert exact.confidence() > guessed.confidence()
    assert (guessed.method_map[holes] == METHOD_DIFFUSION).all()


def test_diffusion_can_be_refused_leaving_an_honest_hole():
    depth, _ = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True

    out = _complete(np.where(holes, np.nan, depth), holes,
                    planes=None, use_diffusion=False)

    assert not out.synthesized_mask.any()
    assert any("could not be completed" in n for n in out.notes)


def test_nearest_valid_intersection_wins_not_the_first_plane_listed():
    """Found in a live smoke test: a ground plane claimed a wall's tear.

    Several fitted planes can validly intersect one hole's rays. The physical
    answer is the NEAREST intersection — the closest surface behind the
    occluder is the one you would see. First-wins in list order picked whatever
    the fitter emitted first, and no global plane ordering fixes it: a ground
    plane's perpendicular distance from the camera is just the camera height,
    so "nearest plane" ranks it ahead of a wall it is nowhere near.
    """
    near = {"normal": [0.0, 0.0, 1.0], "d": -5.0}
    far = {"normal": [0.0, 0.0, 1.0], "d": -40.0}
    depth, _ = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[25:35, 40:55] = True
    punched = np.where(holes, np.nan, depth)

    for order in ([near, far], [far, near]):
        out = _complete(punched, holes, planes=order, use_diffusion=False)
        assert np.allclose(out.depth[holes], 5.0, atol=1e-6), \
            "the nearer plane must win regardless of list order"


def test_grazing_intersections_are_refused():
    """A ray skimming a plane is geometrically valid and physically useless.

    Near-horizon rays meet a ground plane hundreds of metres away, where a
    millimetre of fit error moves the intersection by metres. The live smoke
    test surfaced this as a pillar's tear being filled at 200 m.
    """
    ground = {"normal": [0.0, 1.0, 0.0], "d": 0.0}   # camera is ON this plane
    depth, _ = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[CY_ROW := int(CY), 10:30] = True           # the horizon row
    punched = np.where(holes, np.nan, depth)

    out = _complete(punched, holes, planes=[ground], use_diffusion=False)
    assert not (out.method_map == METHOD_RAY_PLANE).any()


def test_a_plane_that_does_not_explain_a_pixel_is_not_used_for_it():
    """A plane behind the camera must not be pressed into service."""
    depth, _ = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True
    behind_camera = {"normal": [0.0, 0.0, 1.0], "d": 5.0}

    out = _complete(np.where(holes, np.nan, depth), holes,
                    planes=[behind_camera], use_diffusion=False)

    assert not (out.method_map == METHOD_RAY_PLANE).any()
    assert not out.synthesized_mask.any()


def test_tangent_tier_used_when_no_plane_is_available():
    depth, _ = _fronto_plane_depth(8.0)
    normals = np.zeros((H, W, 3))
    normals[..., 2] = 1.0
    holes = np.zeros((H, W), dtype=bool)
    holes[30:32, 40:42] = True

    out = _complete(np.where(holes, np.nan, depth), holes,
                    planes=None, normals=normals, use_diffusion=False)

    assert (out.method_map[holes] == METHOD_TANGENT).any()
    assert np.allclose(out.depth[holes], 8.0, atol=1e-6)


def test_no_holes_is_a_no_op_that_says_so():
    depth, plane = _fronto_plane_depth(8.0)
    out = _complete(depth, None, planes=[plane])
    assert not out.synthesized_mask.any()
    assert any("no holes" in n for n in out.notes)


# --- interchangeability with the layered-model path ---

def test_output_matches_the_hidden_surface_contract():
    """Shaped like hidden_geometry.select_hidden_surface so a layered model and
    this module are interchangeable producers for the same consumers."""
    depth, plane = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True
    out = _complete(np.where(holes, np.nan, depth), holes, planes=[plane])

    hidden, valid, stats = out.as_hidden_surface()
    assert hidden.shape == depth.shape
    assert valid.dtype == bool
    assert valid.sum() == holes.sum()
    assert (hidden[valid] > 0).all()
    assert stats["source"] == "depth_completion"
    assert 0.0 <= stats["confidence"] <= 1.0


def test_graph_policy_none_blocks_the_fill_it_guards():
    """The graph's refusal must survive into the thing that does the filling."""
    from atlas_camera.core.occlusion_graph import (
        POLICY_EXTEND_PLANE, POLICY_NONE, AtlasOcclusionGraph, OcclusionNode,
    )
    from atlas_camera.core.depth_completion import complete_depth_from_graph
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import (
        AtlasCamera, AtlasExtrinsics, AtlasSolve,
    )

    intr = build_intrinsics(image_width=W, image_height=H, focal_length_mm=35.0)
    intr.fx_px = intr.fy_px = F
    intr.cx_px, intr.cy_px = CX, CY
    solve = AtlasSolve(camera=AtlasCamera(
        intrinsics=intr,
        extrinsics=AtlasExtrinsics(camera_view_matrix=_view())),
        image_width=W, image_height=H)

    depth, plane = _fronto_plane_depth(8.0)
    holes = np.zeros((H, W), dtype=bool)
    holes[20:40, 30:60] = True
    punched = np.where(holes, np.nan, depth)

    allowed = AtlasOcclusionGraph(nodes=[OcclusionNode(
        id="wall", kind="surface", plane=plane,
        completion_policy=POLICY_EXTEND_PLANE)])
    blocked = AtlasOcclusionGraph(nodes=[OcclusionNode(
        id="wall", kind="surface", plane=plane,
        completion_policy=POLICY_NONE)])

    ok = complete_depth_from_graph(solve, punched, allowed, holes=holes,
                                   use_diffusion=False)
    no = complete_depth_from_graph(solve, punched, blocked, holes=holes,
                                   use_diffusion=False)

    assert (ok.method_map[holes] == METHOD_RAY_PLANE).all()
    assert not no.synthesized_mask.any()
    assert any("unclassifiable tears" in n for n in no.notes)
