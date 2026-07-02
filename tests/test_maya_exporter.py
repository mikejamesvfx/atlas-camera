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
    write_maya_mel_launcher,
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


def test_maya_exporter_applies_world_matrix(tmp_path):
    # World matrix with identity rotation and position (1, 2, 3).
    # Maya row-vector convention: 3x3 rotation transposed, translation in last row.
    # Identity rotation is self-transposing, so flat Maya matrix is:
    # [1,0,0,0, 0,1,0,0, 0,0,1,0, 1,2,3,1]
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            ),
            extrinsics=AtlasExtrinsics(
                camera_position=(1.0, 2.0, 3.0),
                camera_world_matrix=(
                    (1.0, 0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0, 2.0),
                    (0.0, 0.0, 1.0, 3.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
        image_width=1920,
        image_height=1080,
    )

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    assert "cmds.xform(camera_transform, worldSpace=True, matrix=" in script
    assert "1.0, 0.0, 0.0, 0.0" in script   # first row: camera right (identity)
    assert "1.0, 2.0, 3.0, 1.0" in script   # last row: translation + homogeneous


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


def test_maya_mel_launcher_writes_file(tmp_path):
    mel_path = write_maya_mel_launcher(tmp_path, review_name="atlas_review_001")
    assert mel_path == tmp_path / "open_atlas_review_001.mel"
    assert mel_path.is_file()


def test_maya_mel_launcher_uses_forward_slashes(tmp_path):
    mel = write_maya_mel_launcher(tmp_path, review_name="atlas_review_001").read_text(encoding="utf-8")
    # Path inside MEL must use forward slashes only
    assert "\\\\" not in mel
    assert 'string $packageDir = "' in mel
    pkg_line = next(line for line in mel.splitlines() if "string $packageDir" in line)
    assert "\\" not in pkg_line


def test_maya_mel_launcher_proc_name_and_call(tmp_path):
    mel = write_maya_mel_launcher(tmp_path, review_name="atlas_review_001").read_text(encoding="utf-8")
    assert "global proc atlas_open_atlas_review_001()" in mel
    assert "atlas_open_atlas_review_001();" in mel


def test_maya_mel_launcher_contains_build_scene_call(tmp_path):
    mel = write_maya_mel_launcher(tmp_path, review_name="atlas_review_001").read_text(encoding="utf-8")
    assert "maya_open_scene.build_scene(package_dir)" in mel
    assert "importlib.reload(maya_open_scene)" in mel


def test_maya_mel_launcher_sanitises_proc_name_for_special_chars(tmp_path):
    mel = write_maya_mel_launcher(tmp_path, review_name="my-review 01!").read_text(encoding="utf-8")
    assert "global proc atlas_open_my_review_01_(" in mel


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
