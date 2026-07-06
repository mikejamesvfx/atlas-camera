"""Tests for the upgraded USDExporter.

usd-core is an optional dependency so every test that exercises the real pxr
API uses pytest.importorskip("pxr") to skip cleanly when it is absent.
"""

import pytest

from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath
from atlas_camera.exporters.usd_exporter import USDExporter


# ---------------------------------------------------------------------------
# export_camera
# ---------------------------------------------------------------------------

def test_usd_export_camera_writes_file(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    solve = make_atlas_solve()
    path = USDExporter().export_camera(solve, tmp_path / "camera.usda")
    assert path.is_file()
    assert path.suffix == ".usda"


def test_usd_export_camera_sets_focal_and_aperture(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve(focal=50.0, sensor_w=36.0, image_width=1920, image_height=1080)
    path = USDExporter().export_camera(solve, tmp_path / "camera.usda")
    stage = Usd.Stage.Open(str(path))
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/AtlasCamera/Camera"))
    assert cam.GetFocalLengthAttr().Get() == pytest.approx(50.0)
    assert cam.GetHorizontalApertureAttr().Get() == pytest.approx(36.0)


def test_usd_export_camera_sets_vertical_aperture_from_aspect(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve(focal=35.0, sensor_w=36.0, image_width=1920, image_height=1080)
    path = USDExporter().export_camera(solve, tmp_path / "camera.usda")
    stage = Usd.Stage.Open(str(path))
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/AtlasCamera/Camera"))
    expected_sensor_h = 36.0 * 1080 / 1920
    assert cam.GetVerticalApertureAttr().Get() == pytest.approx(expected_sensor_h, rel=1e-4)


def test_usd_export_camera_sets_principal_point_offsets(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    # cx is 10px right of centre on a 1920-wide frame
    solve = make_atlas_solve(
        image_width=1920, image_height=1080,
        sensor_w=36.0,
        principal_point_px=(970.0, 540.0),
    )
    path = USDExporter().export_camera(solve, tmp_path / "camera.usda")
    stage = Usd.Stage.Open(str(path))
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/AtlasCamera/Camera"))
    # cx offset = (970 - 960) / 1920 * 36 = 0.1875 mm
    assert cam.GetHorizontalApertureOffsetAttr().Get() == pytest.approx(0.1875, rel=1e-4)
    # cy offset = -(540 - 540) / 1080 * sensor_h = 0
    assert cam.GetVerticalApertureOffsetAttr().Get() == pytest.approx(0.0, abs=1e-6)


def test_usd_export_camera_writes_world_transform(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve(position=(1.0, 2.0, 3.0))
    path = USDExporter().export_camera(solve, tmp_path / "camera.usda")
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/AtlasCamera/Camera")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    assert len(ops) == 1  # one transform op (the world matrix)


# ---------------------------------------------------------------------------
# export_proxy_scene
# ---------------------------------------------------------------------------

def test_usd_export_proxy_scene_writes_file(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    solve = make_atlas_solve()
    path = USDExporter().export_proxy_scene(solve, tmp_path / "proxy.usda")
    assert path.is_file()


def test_usd_export_proxy_scene_has_ground_plane(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve()
    path = USDExporter().export_proxy_scene(solve, tmp_path / "proxy.usda")
    stage = Usd.Stage.Open(str(path))
    plane_prim = stage.GetPrimAtPath("/AtlasProjectionScene/atlas_projection_plane")
    assert plane_prim.IsValid()
    mesh = UsdGeom.Mesh(plane_prim)
    points = mesh.GetPointsAttr().Get()
    assert len(points) == 4  # quad


def test_usd_export_proxy_scene_ground_plane_is_40m(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve()
    path = USDExporter().export_proxy_scene(solve, tmp_path / "proxy.usda")
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/AtlasProjectionScene/atlas_projection_plane"))
    points = mesh.GetPointsAttr().Get()
    xs = [p[0] for p in points]
    assert min(xs) == pytest.approx(-20.0)
    assert max(xs) == pytest.approx(20.0)


def test_usd_export_proxy_scene_ground_plane_has_st_primvar(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve()
    path = USDExporter().export_proxy_scene(solve, tmp_path / "proxy.usda")
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/AtlasProjectionScene/atlas_projection_plane")
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    st = primvars_api.GetPrimvar("st")
    assert st.IsDefined()
    uv_values = st.Get()
    assert len(uv_values) == 4  # one UV per vertex


# ---------------------------------------------------------------------------
# export_projection_scene
# ---------------------------------------------------------------------------

def test_usd_export_projection_scene_writes_file(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    solve = make_atlas_solve()
    path = USDExporter().export_projection_scene(solve, tmp_path / "projection.usda")
    assert path.is_file()


def test_usd_export_projection_scene_has_camera(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve(focal=35.0)
    path = USDExporter().export_projection_scene(solve, tmp_path / "projection.usda")
    stage = Usd.Stage.Open(str(path))
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/AtlasProjection/Camera"))
    assert cam.GetFocalLengthAttr().Get() == pytest.approx(35.0)


def test_usd_export_projection_scene_has_ground_plane(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve()
    path = USDExporter().export_projection_scene(solve, tmp_path / "projection.usda")
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/AtlasProjection/ProjectionPlane")
    assert prim.IsValid()
    mesh = UsdGeom.Mesh(prim)
    assert len(mesh.GetPointsAttr().Get()) == 4


def test_usd_export_projection_scene_has_material(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade
    solve = make_atlas_solve()
    path = USDExporter().export_projection_scene(solve, tmp_path / "projection.usda")
    stage = Usd.Stage.Open(str(path))
    mat_prim = stage.GetPrimAtPath("/AtlasProjection/Materials/ProjectionMat")
    assert mat_prim.IsValid()
    # PbrShader child should exist and declare a surface output
    pbr_prim = stage.GetPrimAtPath("/AtlasProjection/Materials/ProjectionMat/PbrShader")
    assert pbr_prim.IsValid()
    pbr = UsdShade.Shader(pbr_prim)
    assert pbr.GetIdAttr().Get() == "UsdPreviewSurface"


def test_usd_export_projection_scene_material_references_source_image(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade
    solve = make_atlas_solve()
    path = USDExporter().export_projection_scene(
        solve, tmp_path / "projection.usda", source_image_name="source_image.png"
    )
    stage = Usd.Stage.Open(str(path))
    tex_prim = stage.GetPrimAtPath("/AtlasProjection/Materials/ProjectionMat/SourceTexture")
    assert tex_prim.IsValid()
    tex_shader = UsdShade.Shader(tex_prim)
    file_input = tex_shader.GetInput("file")
    assert "source_image.png" in str(file_input.Get())


def test_usd_export_projection_scene_plane_is_bound_to_material(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade
    solve = make_atlas_solve()
    path = USDExporter().export_projection_scene(solve, tmp_path / "projection.usda")
    stage = Usd.Stage.Open(str(path))
    plane_prim = stage.GetPrimAtPath("/AtlasProjection/ProjectionPlane")
    binding_api = UsdShade.MaterialBindingAPI(plane_prim)
    bound_mat, _ = binding_api.ComputeBoundMaterial()
    assert bound_mat.GetPath() == "/AtlasProjection/Materials/ProjectionMat"


# ---------------------------------------------------------------------------
# export_camera_animation
# ---------------------------------------------------------------------------

def _two_keyframe_path(frame_count=11):
    return AtlasCameraPath(
        keyframes=[
            AtlasCameraKeyframe(frame_index=0, position=(0.0, 5.0, 10.0), target=(0.0, 0.0, 0.0)),
            AtlasCameraKeyframe(frame_index=frame_count - 1, position=(10.0, 5.0, 0.0), target=(0.0, 0.0, 0.0)),
        ],
        fps=24.0,
        frame_count=frame_count,
    )


def test_usd_export_camera_animation_writes_file(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    solve = make_atlas_solve()
    path = USDExporter().export_camera_animation(
        _two_keyframe_path(), solve.camera.intrinsics, tmp_path / "camera_path.usda"
    )
    assert path.is_file()


def test_usd_export_camera_animation_sets_time_codes(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd
    solve = make_atlas_solve()
    frame_count = 11
    path = USDExporter().export_camera_animation(
        _two_keyframe_path(frame_count), solve.camera.intrinsics, tmp_path / "camera_path.usda"
    )
    stage = Usd.Stage.Open(str(path))
    assert stage.GetStartTimeCode() == pytest.approx(0.0)
    assert stage.GetEndTimeCode() == pytest.approx(float(frame_count - 1))
    assert stage.GetFramesPerSecond() == pytest.approx(24.0)


def test_usd_export_camera_animation_has_time_sampled_transform(tmp_path, make_atlas_solve):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    solve = make_atlas_solve()
    frame_count = 11
    path = USDExporter().export_camera_animation(
        _two_keyframe_path(frame_count), solve.camera.intrinsics, tmp_path / "camera_path.usda"
    )
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/AtlasCamera/Camera")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    assert len(ops) == 1
    time_samples = ops[0].GetTimeSamples()
    assert len(time_samples) == frame_count

    first_matrix = ops[0].Get(Usd.TimeCode(0))
    last_matrix = ops[0].Get(Usd.TimeCode(frame_count - 1))
    assert first_matrix.ExtractTranslation() == pytest.approx((0.0, 5.0, 10.0))
    assert last_matrix.ExtractTranslation() == pytest.approx((10.0, 5.0, 0.0))
