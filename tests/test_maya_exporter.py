import ast

import pytest

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
from atlas_camera.core.solver import solve_from_vanishing_points
from atlas_camera.exporters.maya_exporter import (
    NODE_CAMERA,
    NODE_DEBUG_GRP,
    NODE_GEOMETRY_GRP,
    NODE_PROJECTION_GRP,
    NODE_PROJECTION_PLANE,
    NODE_REFERENCE_GRP,
    write_maya_scene_script,
)


def test_maya_exporter_script_is_valid_python(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            )
        ),
        image_width=1920,
        image_height=1080,
    )
    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")
    ast.parse(script)


def test_maya_exporter_writes_script_file(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            )
        ),
        image_width=1920,
        image_height=1080,
    )

    script_path = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py")
    script = script_path.read_text(encoding="utf-8")

    assert "cmds.camera" in script
    assert "Y-up" in script
    assert "cmds.upAxis" in script
    assert NODE_CAMERA in script


def test_maya_exporter_uses_frozen_node_names(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            )
        ),
        image_width=1920,
        image_height=1080,
    )

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    for name in (NODE_CAMERA, NODE_PROJECTION_GRP, NODE_GEOMETRY_GRP, NODE_DEBUG_GRP, NODE_REFERENCE_GRP):
        assert name in script


def test_maya_exporter_converts_aperture_and_film_offset(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=50.0,
                sensor_width_mm=36.0,
                principal_point_px=(1010.0, 500.0),
            )
        ),
        image_width=1920,
        image_height=1080,
    )

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    assert "horizontalFilmAperture\", 1.4173228346456694" in script
    assert "horizontalFilmOffset\", 0.026041666666666668" in script
    assert "verticalFilmOffset\", -0.037037037037037035" in script


def test_maya_exporter_preserves_y_up_camera_translation(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            ),
            extrinsics=AtlasExtrinsics(camera_position=(1.0, 2.0, 3.0)),
        ),
        image_width=1920,
        image_height=1080,
    )

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    assert 'cmds.setAttr(camera_transform + ".translateX", 1.0)' in script
    assert 'cmds.setAttr(camera_transform + ".translateY", 2.0)' in script
    assert 'cmds.setAttr(camera_transform + ".translateZ", 3.0)' in script


def test_maya_exporter_writes_projection_plane_and_shader(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            )
        ),
        image_width=1920,
        image_height=1080,
    )

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    assert NODE_PROJECTION_PLANE in script
    assert "polyPlane" in script
    assert "width=40" in script
    assert "subdivisionsX=64" in script
    assert "projection" in script
    assert "projType" in script
    assert "place3dTexture" in script
    assert "atlas_proj_file" in script
    assert "atlas_proj_SG" in script


def test_maya_exporter_rejects_missing_focal_length(tmp_path):
    solve = AtlasSolve(
        camera=AtlasCamera(intrinsics=build_intrinsics(image_width=1920, image_height=1080)),
        image_width=1920,
        image_height=1080,
    )

    with pytest.raises(ValueError, match="focal_length_mm"):
        write_maya_scene_script(solve, tmp_path / "maya_open_scene.py")


def test_maya_exporter_warns_for_inferred_focal(tmp_path):
    pytest.importorskip("numpy")
    solve = solve_from_vanishing_points(
        (0.0, 48.0),
        (40.0, 20.0),
        image_width=160,
        image_height=96,
    )

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    assert "cmds.warning" in script
    assert solve.camera.focal_length_inferred is True
