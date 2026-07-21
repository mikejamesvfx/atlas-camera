import ast

from atlas_camera.core.schema import AtlasProxyPrimitive
from atlas_camera.exporters.blender_exporter import write_blender_scene_script


def test_blender_exporter_script_is_valid_python(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_blender_scene_script(solve, tmp_path / "blender_open_scene.py").read_text(encoding="utf-8")
    ast.parse(script)


def test_blender_exporter_writes_script_file(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    path = write_blender_scene_script(solve, tmp_path / "blender_open_scene.py")
    assert path.is_file()
    script = path.read_text(encoding="utf-8")
    assert "import bpy" in script
    assert "import mathutils" in script
    assert "build_scene" in script


def test_blender_exporter_sets_focal_and_sensor(tmp_path, make_atlas_solve):
    solve = make_atlas_solve(focal=50.0, sensor_w=36.0)
    script = write_blender_scene_script(solve, tmp_path / "blender.py").read_text(encoding="utf-8")
    assert "camera_data.lens = 50.0" in script
    assert "camera_data.sensor_width = 36.0" in script


def test_blender_exporter_ground_plane_is_40m(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_blender_scene_script(solve, tmp_path / "blender.py").read_text(encoding="utf-8")
    assert "primitive_plane_add(size=40" in script
    assert "atlas_ground_plane_z_up" in script


def test_blender_exporter_projection_material_nodes_present(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_blender_scene_script(solve, tmp_path / "blender.py").read_text(encoding="utf-8")
    assert "atlas_projection_mat" in script
    assert "ShaderNodeTexCoord" in script
    assert "ShaderNodeSeparateXYZ" in script
    assert "ShaderNodeCombineXYZ" in script
    assert "ShaderNodeTexImage" in script
    assert "ShaderNodeBsdfDiffuse" in script
    assert "ShaderNodeOutputMaterial" in script
    assert "Camera" in script  # TexCoord Camera output


def test_blender_exporter_bakes_scale_factors(tmp_path, make_atlas_solve):
    solve = make_atlas_solve(focal=50.0, sensor_w=36.0)
    script = write_blender_scene_script(solve, tmp_path / "blender.py").read_text(encoding="utf-8")
    # scale_u = 50 / 36 ≈ 1.3889
    scale_u = 50.0 / 36.0
    assert str(round(scale_u, 4))[:5] in script


def test_blender_exporter_applies_world_matrix_z_up(tmp_path, make_atlas_solve):
    # Atlas Y-up position (1, 2, 3) with identity rotation → Blender Z-up matrix_world.
    # T @ M_atlas: Row0 = [1,0,0,1], Row1 = -[0,0,1,3] = [0,0,-1,-3], Row2 = [0,1,0,2]
    solve = make_atlas_solve(position=(1.0, 2.0, 3.0))
    script = write_blender_scene_script(solve, tmp_path / "blender.py").read_text(encoding="utf-8")
    assert "camera.matrix_world = mathutils.Matrix(" in script
    # Translation column in Blender Z-up: X=1.0 (unchanged), Y=-3.0 (-Atlas Z), Z=2.0 (Atlas Y)
    assert "1.0" in script
    assert "-3.0" in script
    assert "2.0" in script


def test_blender_exporter_embeds_retopologized_relief_mesh(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="projection_relief_mesh",
        primitive_type="mesh",
        metadata={
            "vertices": [0, 0, 1, 1, 0, 1, 0, 1, 1],
            "faces": [0, 1, 2],
            "uvs": [0, 0, 1, 0, 0, 1],
        },
    ))
    script = write_blender_scene_script(solve, tmp_path / "blender.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'relief_data.from_pydata' in script
    assert 'atlas_retopologized_relief' in script
    # Atlas (x, y, z) -> Blender (x, -z, y).
    assert '(0.0, -1.0, 0.0)' in script
    assert 'AtlasProjectionUV' in script
