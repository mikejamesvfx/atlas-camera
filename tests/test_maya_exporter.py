import ast

import pytest

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasProjectionScene,
    AtlasProxyPrimitive,
    AtlasSolve,
)
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


def _solve_with_proxies(proxies):
    return AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1920,
                image_height=1080,
                focal_length_mm=35.0,
            )
        ),
        image_width=1920,
        image_height=1080,
        projection_scene=AtlasProjectionScene(proxy_geometry=proxies),
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


def test_maya_exporter_exports_box_proxy_with_dimensions_and_transform(tmp_path):
    box = AtlasProxyPrimitive(
        name="atlas_box_01",
        primitive_type="box",
        transform_matrix=(
            (1.0, 0.0, 0.0, 5.0),
            (0.0, 1.0, 0.0, 0.5),
            (0.0, 0.0, 1.0, -2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dimensions=(2.0, 3.0, 4.0),
        metadata={"role": PROXY_ROLE},
    )
    solve = _solve_with_proxies([box])

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")
    ast.parse(script)

    assert "'name': 'atlas_box_01'" in script
    assert "'type': 'box'" in script
    assert "'dimensions': [2.0, 3.0, 4.0]" in script
    assert "cmds.polyCube(name=spec[\"name\"], width=dx, height=dy, depth=dz)" in script
    assert "5.0, 0.5, -2.0, 1.0" in script  # translation row of the converted matrix


def test_maya_exporter_exports_cylinder_proxy(tmp_path):
    cylinder = AtlasProxyPrimitive(
        name="atlas_cyl_01",
        primitive_type="cylinder",
        dimensions=(1.0, 2.0, 1.0),
        metadata={"role": PROXY_ROLE},
    )
    solve = _solve_with_proxies([cylinder])

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")
    ast.parse(script)

    assert "'name': 'atlas_cyl_01'" in script
    assert "'type': 'cylinder'" in script
    assert "cmds.polyCylinder(name=spec[\"name\"], radius=dx / 2.0, height=dy)" in script


def test_maya_exporter_exports_plane_proxy(tmp_path):
    plane = AtlasProxyPrimitive(
        name="projection_backdrop",
        primitive_type="plane",
        dimensions=(40.0, 20.0, 0.0),
        metadata={"role": PROXY_ROLE},
    )
    solve = _solve_with_proxies([plane])

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")
    ast.parse(script)

    assert "'name': 'projection_backdrop'" in script
    assert "'type': 'plane'" in script
    assert "cmds.polyPlane(name=spec[\"name\"], width=dx, height=dz)" in script


def test_maya_exporter_excludes_non_projection_proxy_role(tmp_path):
    other = AtlasProxyPrimitive(
        name="some_debug_helper",
        primitive_type="box",
        dimensions=(1.0, 1.0, 1.0),
        metadata={"role": "not_a_projection_proxy"},
    )
    solve = _solve_with_proxies([other])

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")

    assert "some_debug_helper" not in script


def test_maya_exporter_imports_relief_mesh_obj_when_path_given(tmp_path):
    mesh = AtlasProxyPrimitive(
        name="projection_relief_mesh",
        primitive_type="mesh",
        metadata={"role": PROXY_ROLE, "vertices": [0.0, 0.0, 0.0], "faces": [0, 0, 0]},
    )
    solve = _solve_with_proxies([mesh])
    obj_path = tmp_path / "atlas_relief_mesh.obj"
    obj_path.write_text("# fake obj\n", encoding="utf-8")

    script = write_maya_scene_script(
        solve, tmp_path / "maya_open_scene.py", relief_mesh_obj_path=obj_path,
    ).read_text(encoding="utf-8")
    ast.parse(script)

    assert "cmds.file(" in script
    # Compare against the repr'd form actually embedded in the generated
    # script (repr() escapes backslashes on Windows paths, so comparing
    # against the raw str(obj_path) would spuriously fail there).
    assert repr(str(obj_path)) in script
    assert 'type="OBJ"' in script
    # The mesh entry itself must never appear in the placeholder box/cylinder/plane loop.
    assert "'type': 'mesh'" not in script


def test_maya_exporter_skips_relief_mesh_import_when_no_path(tmp_path):
    mesh = AtlasProxyPrimitive(
        name="projection_relief_mesh",
        primitive_type="mesh",
        metadata={"role": PROXY_ROLE, "vertices": [0.0, 0.0, 0.0], "faces": [0, 0, 0]},
    )
    solve = _solve_with_proxies([mesh])

    script = write_maya_scene_script(solve, tmp_path / "maya_open_scene.py").read_text(encoding="utf-8")
    ast.parse(script)

    assert "relief_mesh_obj_path = None" in script
