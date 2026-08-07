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
    # Emission since 2026-08-07, not Diffuse — the scene creates no light, so a
    # BSDF rendered black. See test_blender_projection_material_is_unlit.
    assert "ShaderNodeEmission" in script
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


# --- source plate resolution (found live 2026-08-07) --------------------------
#
# The Blender export came out of the fan-out with geometry but NO texture while
# Maya from the same solve was fine. Cause: this exporter was the only one of
# the three not using the shared primary_plate_path helper — it read
# solve.source_plate.image_path directly, so with no AtlasRegisterPlate in the
# graph it got "" and fell back to <package_dir>/source_image.png, a file that
# only exists inside a review PACKAGE. The Image Texture node was created and
# wired but never assigned an image, which renders as untextured rather than as
# an error.


def test_blender_falls_back_to_the_solve_image_path(tmp_path, make_atlas_solve):
    """No registered plate_ref must still texture, exactly as Nuke and Maya do
    from the same solve."""
    plate = tmp_path / "plate.png"
    plate.write_bytes(b"\x89PNG\r\n\x1a\n")
    solve = make_atlas_solve()
    solve.source_plate = None
    solve.image_path = str(plate)
    script = write_blender_scene_script(
        solve, tmp_path / "scene.py").read_text(encoding="utf-8")
    assert "plate.png" in script


def test_blender_prefers_a_registered_plate_over_the_solve_path(tmp_path,
                                                                make_atlas_solve):
    from atlas_camera.core.schema import AtlasPlateRef
    solve = make_atlas_solve()
    solve.image_path = str(tmp_path / "auto.png")
    solve.source_plate = AtlasPlateRef(image_path="/mnt/show/final_plate.exr",
                                       is_proxy=False)
    script = write_blender_scene_script(
        solve, tmp_path / "scene.py").read_text(encoding="utf-8")
    assert "final_plate.exr" in script
    assert "auto.png" not in script


def test_blender_drops_a_dangling_solve_image_path(tmp_path, make_atlas_solve):
    """Every tensor solve records a NamedTemporaryFile the solve node has
    already unlinked. Baking that dead path in makes a script that silently
    textures nothing; dropping it lets the package fallback take over."""
    solve = make_atlas_solve()
    solve.source_plate = None
    solve.image_path = str(tmp_path / "already_unlinked.png")
    script = write_blender_scene_script(
        solve, tmp_path / "scene.py").read_text(encoding="utf-8")
    assert "already_unlinked.png" not in script


def test_blender_ignores_a_proxy_plate_ref(tmp_path, make_atlas_solve):
    """A proxy plate_ref is a preview, not a deliverable — same rule the other
    exporters follow through primary_plate_path."""
    from atlas_camera.core.schema import AtlasPlateRef
    plate = tmp_path / "real.png"
    plate.write_bytes(b"\x89PNG\r\n\x1a\n")
    solve = make_atlas_solve()
    solve.image_path = str(plate)
    solve.source_plate = AtlasPlateRef(image_path="/tmp/proxy_preview.png",
                                       is_proxy=True)
    script = write_blender_scene_script(
        solve, tmp_path / "scene.py").read_text(encoding="utf-8")
    assert "proxy_preview.png" not in script
    assert "real.png" in script


def test_blender_projection_material_is_unlit(tmp_path, make_atlas_solve):
    """Second half of the same live report. The script creates no light, so a
    Diffuse BSDF renders BLACK in Rendered view however good the texture is —
    which reads as "no texture" and was indistinguishable from the path bug.

    Emission is also the correct model on its own merits: a projected plate is
    already-lit photography, and shading it a second time is double-lighting.
    Same doctrine as the viewport's unlit projection shader, and it means the
    review scene is readable with no lights in it.
    """
    script = write_blender_scene_script(
        make_atlas_solve(), tmp_path / "scene.py").read_text(encoding="utf-8")
    assert "ShaderNodeEmission" in script
    assert "ShaderNodeBsdfDiffuse" not in script
