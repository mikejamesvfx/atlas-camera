"""Placing a Pixal3D-frame mesh into a solved scene: frame chain, scale, colour."""

from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.generated_mesh import (
    PHOTO_WEIGHT_SPLIT,
    ground_contact_scale,
    match_vertex_colours,
    photo_visibility_weights,
    pixal_camera_distance,
    pixal_to_source_camera,
    register_object_scale,
    scale_verdict,
    source_camera_to_world,
    world_to_source_camera,
)
from atlas_camera.core.move_budget import rasterize_coverage
from atlas_camera.core.object_crop import virtual_object_camera
from test_object_crop import CX, CY, FX, FY, H, W, _view_matrix, box_mesh


def _subdivided_box(center, size, n=6):
    """Box with a vertex lattice per face, so per-vertex tests have resolution."""
    verts, faces = [], []
    c = np.asarray(center, dtype=float)
    h = np.asarray(size, dtype=float) / 2.0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            base = len(verts)
            for i in range(n + 1):
                for j in range(n + 1):
                    p = np.zeros(3)
                    p[axis] = sign * h[axis]
                    p[u_ax] = (-1 + 2 * i / n) * h[u_ax]
                    p[v_ax] = (-1 + 2 * j / n) * h[v_ax]
                    verts.append(c + p)
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b, d, e = a + 1, a + (n + 1), a + (n + 1) + 1
                    faces += [[a, d, e], [a, e, b]]
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def _depth(view, verts, faces):
    _, z = rasterize_coverage(verts, faces, view_matrix=view, fx=FX, fy=FY, cx=CX,
                              cy=CY, width=W, height=H, backend="numpy")
    return z


@pytest.fixture()
def placed():
    view = _view_matrix([0.0, 1.6, 0.0])
    verts, faces = _subdivided_box((2.6, 0.6, -6.0), (1.2, 1.2, 1.0))
    depth = _depth(view, verts, faces)
    mask = np.isfinite(depth)
    crop = virtual_object_camera(view_matrix=view, fx=FX, fy=FY, cx=CX, cy=CY,
                                 image_width=W, image_height=H, mask=mask)
    # Express the truth in Pixal3D's frame at an arbitrary model scale s0.
    s0 = 7.3
    p_s = world_to_source_camera(verts, view_matrix=view)
    p_v = (p_s / s0) @ crop.rotation.T
    pixal = p_v + np.array([0.0, 0.0, pixal_camera_distance(crop.fov_deg)])
    return dict(view=view, verts=verts, faces=faces, depth=depth, mask=mask,
                crop=crop, s0=s0, pixal=pixal)


def test_pixal_frame_round_trips_to_world(placed):
    p = placed
    cam = pixal_to_source_camera(p["pixal"], rotation=p["crop"].rotation,
                                 fov_deg=p["crop"].fov_deg)
    world = source_camera_to_world(cam, view_matrix=p["view"], scale=p["s0"])
    assert np.max(np.abs(world - p["verts"])) < 1e-6


def test_pixal_frame_object_sits_in_unit_cube(placed):
    # Sanity on the fixture: a padded crop puts the object near the cube.
    assert np.abs(placed["pixal"][:, :2]).max() < 0.75


def test_scale_recovered_from_metric_depth(placed):
    p = placed
    cam = pixal_to_source_camera(p["pixal"], rotation=p["crop"].rotation,
                                 fov_deg=p["crop"].fov_deg)
    reg = register_object_scale(cam, p["faces"], view_matrix=p["view"], fx=FX, fy=FY,
                                cx=CX, cy=CY, width=W, height=H,
                                metric_depth=np.where(np.isfinite(p["depth"]), p["depth"], np.nan),
                                object_mask=p["mask"], backend="numpy")
    assert reg["scale"] == pytest.approx(p["s0"], rel=1e-3)
    assert reg["rel_mad"] < 0.01
    assert scale_verdict(rel_mad=reg["rel_mad"], depth_scale=reg["scale"],
                         ground_scale=None)["grade"] == "ok"


def test_wrong_depth_scale_structure_is_refused(placed):
    p = placed
    cam = pixal_to_source_camera(p["pixal"], rotation=p["crop"].rotation,
                                 fov_deg=p["crop"].fov_deg)
    rng = np.random.default_rng(0)
    noisy = np.where(np.isfinite(p["depth"]), p["depth"], np.nan) * \
        rng.uniform(0.3, 2.5, size=p["depth"].shape)
    reg = register_object_scale(cam, p["faces"], view_matrix=p["view"], fx=FX, fy=FY,
                                cx=CX, cy=CY, width=W, height=H, metric_depth=noisy,
                                object_mask=p["mask"], backend="numpy")
    v = scale_verdict(rel_mad=reg["rel_mad"], depth_scale=reg["scale"], ground_scale=None)
    assert v["grade"] == "refuse"
    assert v["calibrated"] is False


def test_empty_overlap_is_explained_not_raised(placed):
    p = placed
    cam = pixal_to_source_camera(p["pixal"], rotation=p["crop"].rotation,
                                 fov_deg=p["crop"].fov_deg)
    reg = register_object_scale(cam, p["faces"], view_matrix=p["view"], fx=FX, fy=FY,
                                cx=CX, cy=CY, width=W, height=H,
                                metric_depth=np.full((H, W), np.nan),
                                object_mask=p["mask"], backend="numpy")
    assert reg["scale"] is None and "pixels" in reg["reason"]
    assert scale_verdict(rel_mad=reg["rel_mad"], depth_scale=None,
                         ground_scale=None)["grade"] == "refuse"


def test_ground_contact_scale_lands_base_on_ground():
    view = _view_matrix([0.0, 1.6, 0.0])
    verts, _ = box_mesh((2.0, 0.5, -6.0), (1.0, 1.0, 1.0))   # base at Y=0
    cam = world_to_source_camera(verts, view_matrix=view) / 4.0
    s = ground_contact_scale(cam, view_matrix=view)
    assert s == pytest.approx(4.0, rel=1e-9)


def test_ground_disagreement_is_inspect():
    v = scale_verdict(rel_mad=0.02, depth_scale=4.0, ground_scale=5.0)
    assert v["grade"] == "inspect"


def test_photo_weight_front_back_and_occluder(placed):
    p = placed
    weight, stats = photo_visibility_weights(
        p["verts"], p["faces"], view_matrix=p["view"], fx=FX, fy=FY, cx=CX, cy=CY,
        width=W, height=H, object_mask=p["mask"],
        metric_depth=np.where(np.isfinite(p["depth"]), p["depth"], np.nan),
        erode_px=0, smooth_iterations=0, backend="numpy")
    # Interior lattice points of the camera-facing +Z face are seen ...
    c = np.array([2.6, 0.6, -6.0])
    front = (np.abs(p["verts"][:, 2] - (c[2] + 0.5)) < 1e-9) & \
        (np.abs(p["verts"][:, 0] - c[0]) < 0.45) & (np.abs(p["verts"][:, 1] - c[1]) < 0.45)
    # Back-face rim vertices lie on the top/left silhouette the camera DOES
    # see (it looks down at a box to its right), so only the interior counts.
    back = (np.abs(p["verts"][:, 2] - (c[2] - 0.5)) < 1e-9) & \
        (np.abs(p["verts"][:, 0] - c[0]) < 0.45) & (np.abs(p["verts"][:, 1] - c[1]) < 0.45)
    assert front.sum() > 4 and back.sum() > 4
    assert (weight[front] >= PHOTO_WEIGHT_SPLIT).all()
    # ... the far face is not.
    assert (weight[back] < PHOTO_WEIGHT_SPLIT).all()
    assert 0.0 < stats["photo_fraction"] < 1.0

    # A pole in front of the object takes the photo away behind it.
    occ_depth = np.where(np.isfinite(p["depth"]), p["depth"], np.nan).copy()
    occ_depth[:, :] = np.minimum(np.nan_to_num(occ_depth, nan=1e9), 2.0)
    weight_occ, stats_occ = photo_visibility_weights(
        p["verts"], p["faces"], view_matrix=p["view"], fx=FX, fy=FY, cx=CX, cy=CY,
        width=W, height=H, object_mask=p["mask"], metric_depth=occ_depth,
        erode_px=0, smooth_iterations=0, backend="numpy")
    assert (weight_occ < PHOTO_WEIGHT_SPLIT).all()
    assert stats_occ["scene_occluded"] > 0


def test_colour_gain_recovers_exposure_offset():
    rng = np.random.default_rng(1)
    plate = rng.uniform(0.2, 0.8, size=(400, 3))
    from atlas_camera.core.generated_mesh import linear_to_srgb, srgb_to_linear
    model = linear_to_srgb(srgb_to_linear(plate) * np.array([0.8, 1.0, 1.25]))
    w = np.ones(400)
    out, rep = match_vertex_colours(model, w, plate)
    assert rep["applied"]
    assert rep["gain"] == pytest.approx([1.25, 1.0, 0.8], rel=0.02)
    assert np.allclose(out, plate, atol=0.01)


def test_colour_gain_needs_seen_vertices():
    out, rep = match_vertex_colours(np.full((50, 3), 0.5), np.zeros(50), np.full((50, 3), 0.2))
    assert not rep["applied"] and rep["gain"] == [1.0, 1.0, 1.0]
