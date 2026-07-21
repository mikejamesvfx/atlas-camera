"""Blender handoff script writer.

Atlas core is Y-up right-handed. Blender is Z-up right-handed.
The coordinate conversion T: (x,y,z) -> (x,-z,y) is applied to the full
4×4 world matrix so camera position AND rotation both land correctly.
"""

from __future__ import annotations

from pathlib import Path

from atlas_camera.core.camera_math import derive_sensor_height_mm
from atlas_camera.core.schema import AtlasSolve


def _embedded_relief_mesh(solve: AtlasSolve) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[float, float]]] | None:
    """Return the solve's relief mesh converted from Atlas Y-up to Blender Z-up.

    AtlasExportReliefMesh places the *post-retopology* mesh on the preview
    solve.  Embedding that payload here keeps the Blender handoff identical to
    the viewport/Maya/Nuke result instead of silently exporting only a camera.
    """
    scene = getattr(solve, "projection_scene", None)
    primitives = getattr(scene, "proxy_geometry", None) or []
    meshes = [p for p in primitives if getattr(p, "primitive_type", "") == "mesh"]
    if not meshes:
        return None
    primitive = next((p for p in meshes if p.name == "projection_relief_mesh"), meshes[0])
    metadata = primitive.metadata or {}
    vertices_raw = metadata.get("vertices") or []
    faces_raw = metadata.get("faces") or []
    uvs_raw = metadata.get("uvs") or []
    if len(vertices_raw) < 9 or len(faces_raw) < 3:
        return None
    vertices = [
        (float(vertices_raw[i]), -float(vertices_raw[i + 2]), float(vertices_raw[i + 1]))
        for i in range(0, len(vertices_raw) - 2, 3)
    ]
    faces = [
        (int(faces_raw[i]), int(faces_raw[i + 1]), int(faces_raw[i + 2]))
        for i in range(0, len(faces_raw) - 2, 3)
    ]
    uvs = [
        (float(uvs_raw[i]), float(uvs_raw[i + 1]))
        for i in range(0, len(uvs_raw) - 1, 2)
    ]
    return vertices, faces, uvs


def write_blender_scene_script(solve: AtlasSolve, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Convert Atlas Y-up world matrix to Blender Z-up: M_blender = T @ M_atlas
    # T maps (x,y,z) -> (x,-z,y), i.e. new_Y = -old_Z, new_Z = old_Y.
    # Row 0 = Atlas row 0 (X unchanged)
    # Row 1 = negated Atlas row 2 (Blender Y = -Atlas Z)
    # Row 2 = Atlas row 1 (Blender Z = Atlas Y)
    wm = solve.camera.extrinsics.camera_world_matrix
    blender_world = [
        [ wm[0][0],  wm[0][1],  wm[0][2],  wm[0][3]],
        [-wm[2][0], -wm[2][1], -wm[2][2], -wm[2][3]],
        [ wm[1][0],  wm[1][1],  wm[1][2],  wm[1][3]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    intrinsics = solve.camera.intrinsics
    focal = intrinsics.focal_length_mm or 35.0
    sensor_w = intrinsics.sensor_width_mm or 36.0
    image_w = intrinsics.image_width
    image_h = intrinsics.image_height
    sensor_h = derive_sensor_height_mm(intrinsics)

    # Projection scale factors: converts camera-space X/Y (divided by depth) to 0-1 UV.
    # u = offset_u + scale_u * (cam_x / -cam_z)
    # v = offset_v - scale_v * (cam_y / -cam_z)   [image Y is down, camera Y is up]
    scale_u = focal / sensor_w
    scale_v = focal / sensor_h
    offset_u = (intrinsics.cx_px / image_w) if intrinsics.cx_px is not None else 0.5
    offset_v = (intrinsics.cy_px / image_h) if intrinsics.cy_px is not None else 0.5

    source_plate = getattr(solve, "source_plate", None)
    source_image_path = str(getattr(source_plate, "image_path", "") or "")
    relief_mesh = _embedded_relief_mesh(solve)
    relief_block = ""
    if relief_mesh is not None:
        vertices, faces, uvs = relief_mesh
        relief_block = f'''
    # Retopologized relief carried by the solve (already Atlas Y-up -> Blender Z-up).
    relief_vertices = {vertices!r}
    relief_faces = {faces!r}
    relief_uvs = {uvs!r}
    relief_data = bpy.data.meshes.new("atlas_retopologized_relief")
    relief_data.from_pydata(relief_vertices, [], relief_faces)
    relief_data.update()
    relief = bpy.data.objects.new("atlas_retopologized_relief", relief_data)
    bpy.context.collection.objects.link(relief)
    if relief_uvs and len(relief_uvs) == len(relief_vertices):
        uv_layer = relief_data.uv_layers.new(name="AtlasProjectionUV")
        for polygon in relief_data.polygons:
            for loop_index in polygon.loop_indices:
                vertex_index = relief_data.loops[loop_index].vertex_index
                uv_layer.data[loop_index].uv = relief_uvs[vertex_index]
    projection_targets.append(relief)
'''

    script = f'''"""Atlas Camera Blender review scene.

Atlas core is Y-up right-handed. This script applies the solved world matrix
(position + rotation) converted to Blender Z-up via matrix_world.

Projection material: TexCoord(Camera) -> perspective division -> ImageTexture.
Scale factors are baked from the solved focal length and sensor dimensions.
"""

import os
import bpy
import mathutils


def build_scene(package_dir=None):
    package_dir = package_dir or os.path.dirname(os.path.abspath(__file__))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Camera
    camera_data = bpy.data.cameras.new({solve.camera.name!r})
    camera_data.lens = {focal!r}
    camera_data.sensor_width = {sensor_w!r}
    camera = bpy.data.objects.new({solve.camera.name!r}, camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.matrix_world = mathutils.Matrix({blender_world!r})
    bpy.context.view_layer.update()

    # Ground plane (40 x 40 m, Blender Z-up so it lies in the XY plane at Z=0)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "atlas_ground_plane_z_up"
    projection_targets = [ground]
{relief_block}

    # --- Camera-projection material ---
    # u = {offset_u!r} + {scale_u!r} * cam_x / depth   (depth = -cam_z)
    # v = {offset_v!r} - {scale_v!r} * cam_y / depth
    mat = bpy.data.materials.new("atlas_projection_mat")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    coord = nodes.new("ShaderNodeTexCoord")

    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Camera"], sep.inputs["Vector"])

    # depth = -cam_z  (camera looks at -Z, so points in front have cam_z < 0)
    neg_z = nodes.new("ShaderNodeMath")
    neg_z.operation = "MULTIPLY"
    neg_z.inputs[1].default_value = -1.0
    links.new(sep.outputs["Z"], neg_z.inputs[0])

    div_x = nodes.new("ShaderNodeMath")
    div_x.operation = "DIVIDE"
    links.new(sep.outputs["X"], div_x.inputs[0])
    links.new(neg_z.outputs["Value"], div_x.inputs[1])

    mul_u = nodes.new("ShaderNodeMath")
    mul_u.operation = "MULTIPLY"
    mul_u.inputs[1].default_value = {scale_u!r}
    links.new(div_x.outputs["Value"], mul_u.inputs[0])

    add_u = nodes.new("ShaderNodeMath")
    add_u.operation = "ADD"
    add_u.inputs[1].default_value = {offset_u!r}
    links.new(mul_u.outputs["Value"], add_u.inputs[0])

    div_y = nodes.new("ShaderNodeMath")
    div_y.operation = "DIVIDE"
    links.new(sep.outputs["Y"], div_y.inputs[0])
    links.new(neg_z.outputs["Value"], div_y.inputs[1])

    mul_v = nodes.new("ShaderNodeMath")
    mul_v.operation = "MULTIPLY"
    mul_v.inputs[1].default_value = {-scale_v!r}  # negate: image Y down, camera Y up
    links.new(div_y.outputs["Value"], mul_v.inputs[0])

    add_v = nodes.new("ShaderNodeMath")
    add_v.operation = "ADD"
    add_v.inputs[1].default_value = {offset_v!r}
    links.new(mul_v.outputs["Value"], add_v.inputs[0])

    combine = nodes.new("ShaderNodeCombineXYZ")
    combine.inputs["Z"].default_value = 0.0
    links.new(add_u.outputs["Value"], combine.inputs["X"])
    links.new(add_v.outputs["Value"], combine.inputs["Y"])

    img_tex = nodes.new("ShaderNodeTexImage")
    img_tex.extension = "CLIP"
    img_path = {source_image_path!r}
    if not img_path:
        img_path = os.path.join(package_dir, "source_image.png")
    elif not os.path.isabs(img_path):
        img_path = os.path.join(package_dir, img_path)
    if os.path.exists(img_path):
        img_tex.image = bpy.data.images.load(img_path, check_existing=True)
    links.new(combine.outputs["Vector"], img_tex.inputs["Vector"])

    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    links.new(img_tex.outputs["Color"], diffuse.inputs["Color"])

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])

    for target in projection_targets:
        target.data.materials.append(mat)

    bpy.context.scene.render.resolution_x = {image_w}
    bpy.context.scene.render.resolution_y = {image_h}

    return camera


if __name__ == "__main__":
    build_scene()
'''
    destination.write_text(script, encoding="utf-8")
    return destination


class BlenderExporter:
    def write_scene(self, solve: AtlasSolve, output_path: str | Path) -> Path:
        return write_blender_scene_script(solve, output_path)
