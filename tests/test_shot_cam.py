"""Tests for AtlasShotCam — the project-level render/output camera format
(sensor + lens + resolution), conformed at AtlasMergeGeometry (attach) and
AtlasBlockoutViewport (render). The critical invariant under test: the
render/viewing camera's format (render_fy/render_image_height, target_width/
target_height) may be shot-cam-conformed, but fx/fy/cx/cy/image_width/
image_height — also read by the frontend for the PRIMARY source's own
texture-sampling — must stay exactly the solve's own values, never the shot
cam's, regardless of whether a shot_cam is present.
"""

import pytest

from atlas_camera.comfy.nodes import (
    _ATLAS_BLOCKOUT_CACHE,
    AtlasBlockoutViewport,
    AtlasDefineShotCam,
    AtlasMergeGeometry,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _extract_blockout_camera,
)
from atlas_camera.core.intrinsics import intrinsics_from_shot_cam
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasShotCam,
    AtlasSolve,
    LatentCamera,
)


def _solve(width=800, height=600, fx=700.0, fy=700.0):
    intr = AtlasIntrinsics(image_width=width, image_height=height, fx_px=fx, fy_px=fy,
                            cx_px=width / 2.0, cy_px=height / 2.0, focal_length_mm=35.0)
    extr = AtlasExtrinsics(camera_position=(0.0, 1.6, 0.0))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


# --- intrinsics_from_shot_cam --------------------------------------------

def test_intrinsics_from_shot_cam_preserves_sensor_aspect():
    shot_cam = AtlasShotCam(sensor_width_mm=36.0, sensor_height_mm=20.25,
                             focal_length_mm=35.0, resolution_long_edge_px=1920)
    intr = intrinsics_from_shot_cam(shot_cam)

    assert intr.image_width == 1920
    assert abs(intr.image_width / intr.image_height - 36.0 / 20.25) < 0.01
    # fx/fy from the standard pinhole formula (focal_mm * px / sensor_mm)
    assert intr.fx_px == pytest.approx(35.0 * 1920 / 36.0)
    assert intr.fy_px == pytest.approx(35.0 * intr.image_height / 20.25)


def test_intrinsics_from_shot_cam_square_sensor_gives_square_output():
    shot_cam = AtlasShotCam(sensor_width_mm=24.0, sensor_height_mm=24.0,
                             focal_length_mm=50.0, resolution_long_edge_px=1024)
    intr = intrinsics_from_shot_cam(shot_cam)
    assert intr.image_width == intr.image_height == 1024


# --- AtlasDefineShotCam ----------------------------------------------------

def test_node_registered():
    assert NODE_CLASS_MAPPINGS["AtlasDefineShotCam"] is AtlasDefineShotCam
    assert "AtlasDefineShotCam" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasDefineShotCam.RETURN_TYPES == ("ATLAS_SHOT_CAM",)


def test_define_returns_shot_cam_with_defaults():
    (sc,) = AtlasDefineShotCam().define()
    assert sc == AtlasShotCam(sensor_width_mm=36.0, sensor_height_mm=24.0,
                               focal_length_mm=35.0, resolution_long_edge_px=1920)


def test_define_passes_through_custom_values():
    (sc,) = AtlasDefineShotCam().define(sensor_width_mm=24.0, sensor_height_mm=13.5,
                                         focal_length_mm=50.0, resolution=3840)
    assert sc.sensor_width_mm == 24.0
    assert sc.sensor_height_mm == 13.5
    assert sc.focal_length_mm == 50.0
    assert sc.resolution_long_edge_px == 3840


# --- AtlasMergeGeometry: attach, never mutate camera -----------------------

def test_merge_attaches_shot_cam_without_touching_camera():
    solve_a = _solve(fx=700.0, fy=700.0)
    solve_b = _solve(fx=900.0, fy=900.0)
    shot_cam = AtlasShotCam(resolution_long_edge_px=2560)

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b, shot_cam=shot_cam)

    assert out.shot_cam == shot_cam
    # solve_a's own camera intrinsics must survive completely untouched —
    # this is what any projection source uses for texture-sampling.
    assert out.camera.intrinsics.fx_px == 700.0
    assert out.camera.intrinsics.fy_px == 700.0


def test_merge_without_shot_cam_leaves_it_none():
    solve_a = _solve()
    solve_b = _solve()
    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)
    assert out.shot_cam is None


# --- AtlasBlockoutViewport: the critical safety invariant ------------------

def test_extract_blockout_camera_without_shot_cam_is_unchanged():
    torch = pytest.importorskip("torch")
    solve = _solve(width=800, height=600, fx=700.0, fy=700.0)
    image = torch.rand(1, 600, 800, 3, dtype=torch.float32)

    payload = _extract_blockout_camera(solve, image, target_width=400, target_height=300)

    assert payload["fx"] == 700.0
    assert payload["fy"] == 700.0
    assert payload["image_width"] == 800
    assert payload["image_height"] == 600
    assert payload["render_fy"] == payload["fy"]
    assert payload["render_image_height"] == payload["image_height"]


def test_extract_blockout_camera_with_shot_cam_conforms_render_camera_only():
    torch = pytest.importorskip("torch")
    solve = _solve(width=800, height=600, fx=700.0, fy=700.0)
    image = torch.rand(1, 600, 800, 3, dtype=torch.float32)
    shot_cam = AtlasShotCam(sensor_width_mm=36.0, sensor_height_mm=20.25,
                             focal_length_mm=50.0, resolution_long_edge_px=1920)
    shot_intr = intrinsics_from_shot_cam(shot_cam)

    payload = _extract_blockout_camera(
        solve, image, target_width=shot_intr.image_width, target_height=shot_intr.image_height,
        shot_intrinsics=shot_intr)

    # The render/viewing camera's FOV inputs now reflect the shot cam.
    assert payload["render_fy"] == pytest.approx(shot_intr.fy_px)
    assert payload["render_image_height"] == shot_intr.image_height
    assert payload["render_fy"] != payload["fy"]  # genuinely different from the solve's own

    # CRITICAL: fx/fy/cx/cy/image_width/image_height — read by the frontend
    # for the PRIMARY source's own texture-sampling uniforms — must stay
    # exactly the solve's own values, untouched by the shot cam.
    assert payload["fx"] == 700.0
    assert payload["fy"] == 700.0
    assert payload["image_width"] == 800
    assert payload["image_height"] == 600


def test_render_precedence_direct_input_wins_over_solve_attached(monkeypatch):
    torch = pytest.importorskip("torch")
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve(width=800, height=600, fx=700.0, fy=700.0)
    solve.shot_cam = AtlasShotCam(resolution_long_edge_px=1000)  # attached, e.g. by a merge
    direct_shot_cam = AtlasShotCam(resolution_long_edge_px=2000)  # wired directly
    image = torch.rand(1, 600, 800, 3, dtype=torch.float32)

    AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data="", shot_cam=direct_shot_cam, unique_id="test_precedence")

    payload = _ATLAS_BLOCKOUT_CACHE["test_precedence"]
    assert payload["target_width"] == intrinsics_from_shot_cam(direct_shot_cam).image_width


def test_render_falls_back_to_solve_attached_shot_cam(monkeypatch):
    torch = pytest.importorskip("torch")
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve(width=800, height=600, fx=700.0, fy=700.0)
    solve.shot_cam = AtlasShotCam(resolution_long_edge_px=1000)
    image = torch.rand(1, 600, 800, 3, dtype=torch.float32)

    AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data="", unique_id="test_fallback")

    payload = _ATLAS_BLOCKOUT_CACHE["test_fallback"]
    assert payload["target_width"] == intrinsics_from_shot_cam(solve.shot_cam).image_width


def test_render_without_any_shot_cam_follows_source_image_aspect():
    torch = pytest.importorskip("torch")
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve(width=800, height=600, fx=700.0, fy=700.0)
    image = torch.rand(1, 600, 800, 3, dtype=torch.float32)  # 4:3 aspect

    AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data="", unique_id="test_no_shot_cam")

    payload = _ATLAS_BLOCKOUT_CACHE["test_no_shot_cam"]
    assert payload["target_width"] == 768  # long edge follows the resolution widget
    assert payload["target_height"] == pytest.approx(768 * 600 / 800, abs=8)
