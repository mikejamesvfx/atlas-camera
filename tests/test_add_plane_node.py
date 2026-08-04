"""Tests for AtlasAddPlanePolygon — hand-clicked polygon planes.

Unlike every AtlasDerive* node, this one APPENDS: a hand-authored surface is an
addition (like AtlasAddPatchView), not a re-derivation of the whole geometry
set, so chaining several of them must not clobber. That promise gets an
explicit test here.

Analytic depth only (numpy), so no [neural] extra and no torch are needed.
"""

import json

import numpy as np

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, AtlasAddPlanePolygon
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
)
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 200.0
CAM_HEIGHT = 1.6


def _view_matrix(h=CAM_HEIGHT):
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _solve():
    intr = AtlasIntrinsics(
        image_width=W, image_height=H, focal_length_mm=35.0, sensor_width_mm=36.0,
        fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
    )
    return AtlasSolve(camera=LatentCamera(
        intrinsics=intr, extrinsics=AtlasExtrinsics(camera_view_matrix=_view_matrix())))


def _wall_scene(wall_z=-8.0, h=CAM_HEIGHT):
    """Ground (Y=0) + a fronto-parallel wall; returns (depth, wall visibility)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    a = (uu - CX) / FX
    b = -(vv - CY) / FY

    ground = np.full((H, W), np.inf)
    down = b < -1e-6
    ground[down] = -h / b[down]

    wall = np.full((H, W), np.inf)
    t = -wall_z
    y_at = h + b * t
    wall[(y_at >= 0.0) & (y_at <= 3.0)] = t

    depth = np.minimum(ground, wall)
    visible = np.isfinite(wall) & (wall <= ground)
    return np.where(np.isfinite(depth), depth, SKY), visible


def _depth_result(depth_map):
    return DepthResult(
        depth=depth_map.astype(np.float32), is_metric=True, model_id="fake",
        image_width=W, image_height=H,
        near=float(depth_map.min()), far=float(depth_map.max()),
    )


def _wall_polygon_json(visible, label="facade", inset=12):
    rows, cols = np.where(visible)
    y0, y1 = rows.min() + inset, rows.max() - inset
    x0, x1 = cols.min() + inset, cols.max() - inset
    pts = [(x0 / W, y0 / H), (x1 / W, y0 / H), (x1 / W, y1 / H), (x0 / W, y1 / H)]
    return json.dumps({"version": 1, "polygons": [
        {"id": "p1", "label": label, "points": pts,
         "fit_mode": "inherit", "enabled": True}]})


def _add(**kwargs):
    """Unwrap the node's {ui, result} return — the widget needs `ui` statuses."""
    return AtlasAddPlanePolygon().add(**kwargs)["result"]


def _hand_planes(solve):
    return [p for p in solve.projection_scene.proxy_geometry
            if (p.metadata or {}).get("source") == "hand_polygon"]


def test_node_is_registered_with_the_derive_node_return_shape():
    assert "AtlasAddPlanePolygon" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["AtlasAddPlanePolygon"].RETURN_TYPES == ("ATLAS_SOLVE", "STRING")
    assert NODE_CLASS_MAPPINGS["AtlasAddPlanePolygon"].RETURN_NAMES == ("solve", "report")


def test_emits_a_mesh_primitive_for_the_clicked_polygon():
    depth, visible = _wall_scene()
    out, report = _add(
        solve=_solve(), depth=_depth_result(depth), polygons=_wall_polygon_json(visible))

    planes = _hand_planes(out)
    assert len(planes) == 1
    plane = planes[0]
    assert plane.name == "hand_plane_01"
    assert plane.primitive_type == "mesh"
    assert plane.metadata["role"] == PROXY_ROLE
    assert len(plane.metadata["vertices"]) == 12       # 4 corners x xyz
    assert len(plane.metadata["faces"]) == 6           # 2 triangles
    assert len(plane.metadata["uvs"]) == 8
    assert plane.metadata["method"] == "depth_ransac"
    assert "depth_ransac" in report


def test_appends_without_clobbering_prior_proxy_geometry():
    """The compositional promise: chain these, no AtlasMergeGeometry needed."""
    depth, visible = _wall_scene()
    solve = _solve()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="projection_relief_mesh", primitive_type="mesh",
        metadata={"role": PROXY_ROLE, "source": "depth_relief_mesh"}))

    once, _ = _add(
        solve=solve, depth=_depth_result(depth), polygons=_wall_polygon_json(visible))
    twice, _ = _add(
        solve=once, depth=_depth_result(depth),
        polygons=_wall_polygon_json(visible), name_prefix="second")

    names = [p.name for p in twice.projection_scene.proxy_geometry]
    assert "projection_relief_mesh" in names
    assert "hand_plane_01" in names
    assert "second_01" in names


def test_input_solve_is_not_mutated():
    depth, visible = _wall_scene()
    solve = _solve()
    out, _ = _add(
        solve=solve, depth=_depth_result(depth), polygons=_wall_polygon_json(visible))

    assert solve.projection_scene.proxy_geometry == []
    assert len(out.projection_scene.proxy_geometry) == 1


def test_missing_depth_uses_the_rectangle_solve_and_says_so():
    _depth, visible = _wall_scene()
    out, report = _add(
        solve=_solve(), depth=None, polygons=_wall_polygon_json(visible))

    planes = _hand_planes(out)
    assert len(planes) == 1
    assert planes[0].metadata["method"] == "rectangle_homography"
    assert "rectangle_homography" in report


def test_disabled_polygons_are_ignored():
    depth, visible = _wall_scene()
    blob = json.loads(_wall_polygon_json(visible))
    blob["polygons"][0]["enabled"] = False

    out, _report = _add(
        solve=_solve(), depth=_depth_result(depth), polygons=json.dumps(blob))

    assert _hand_planes(out) == []


def test_a_failing_polygon_is_reported_and_leaves_the_solve_intact():
    depth, _visible = _wall_scene()
    bowtie = json.dumps({"version": 1, "polygons": [
        {"id": "bad", "label": "bowtie",
         "points": [[0.1, 0.1], [0.4, 0.4], [0.4, 0.1], [0.1, 0.4]]}]})

    out, report = _add(
        solve=_solve(), depth=_depth_result(depth), polygons=bowtie)

    assert _hand_planes(out) == []
    assert "skipped" in report and "self_intersecting" in report


def test_malformed_polygon_json_reports_instead_of_raising():
    depth, _visible = _wall_scene()

    out, report = _add(
        solve=_solve(), depth=_depth_result(depth), polygons="{not json")

    assert out.projection_scene.proxy_geometry == []
    assert "could not be read" in report


def test_empty_polygons_passes_the_solve_through():
    depth, _visible = _wall_scene()

    out, report = _add(
        solve=_solve(), depth=_depth_result(depth), polygons="")

    assert out.projection_scene.proxy_geometry == []
    assert "no polygons" in report


def test_plate_is_published_for_the_canvas_widget():
    """The widget has nothing to click on until the node ships it the plate."""
    import pytest
    torch = pytest.importorskip("torch")
    depth, _visible = _wall_scene()
    image = torch.zeros((1, H, W, 3), dtype=torch.float32)

    result = AtlasAddPlanePolygon().add(
        solve=_solve(), depth=_depth_result(depth), image=image, polygons="")

    plate = result["ui"]["atlas_plate"][0]
    assert plate.startswith("data:image/png;base64,")


def test_no_plate_published_when_no_image_is_wired():
    depth, _visible = _wall_scene()

    result = AtlasAddPlanePolygon().add(
        solve=_solve(), depth=_depth_result(depth), polygons="")

    assert result["ui"].get("atlas_plate", []) == []


def test_fingerprint_tracks_the_polygon_blob():
    """Without this ComfyUI serves a cached solve after an outline is edited."""
    first = AtlasAddPlanePolygon.IS_CHANGED(polygons='{"a": 1}', fit_mode="auto")
    same = AtlasAddPlanePolygon.IS_CHANGED(polygons='{"a": 1}', fit_mode="auto")
    edited = AtlasAddPlanePolygon.IS_CHANGED(polygons='{"a": 2}', fit_mode="auto")
    remoded = AtlasAddPlanePolygon.IS_CHANGED(polygons='{"a": 1}', fit_mode="rectangle")

    assert first == same
    assert first != edited
    assert first != remoded
