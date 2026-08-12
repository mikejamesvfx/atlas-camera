"""Blender handoff for a DynamicPlate (spec §29).

Fundamental split, and the reason this writer exists at all:

    atlas_projection_camera  — the FIXED crop camera the temporal sequence is
                               projected through. Never the scene camera.
    atlas_render_camera      — the artist's camera (starts at the source solve
                               pose). THIS is scene.camera; the artist moves it
                               freely while the projection stays registered.

The projection material mirrors `blender_exporter.py`'s unlit
perspective-division node chain, with two deliberate differences:

* Texture coordinates come from ``ShaderNodeTexCoord.object =
  projection_camera`` via the **Object** output. The exporter's original
  ``Camera`` output tracks the ACTIVE render camera — exactly the coupling a
  dynamic plate must break.
* The image is an image SEQUENCE (``generated/frame_0000.png ...``) with
  frame mapping from the plate's frame range. With no generated frames yet it
  falls back to the still crop so the scene is still inspectable.

Receiver geometry is rebuilt from the plate's plane primitive (Atlas Y-up ->
Blender Z-up via `dcc_transform.blender_point_from_atlas`, the one sanctioned
conversion seam).
"""
from __future__ import annotations

from pathlib import Path

from atlas_camera.core.dynamic_plate import DynamicPlate
from atlas_camera.exporters.dcc_transform import (
    blender_matrix_from_atlas,
    blender_point_from_atlas,
)


def _receiver_corners(plate: DynamicPlate):
    prim = plate.receiver.primitive if plate.receiver else None
    if prim is None or prim.primitive_type != "plane":
        raise ValueError("DynamicPlate Blender export needs a plane receiver")
    tf = prim.transform_matrix
    u = (tf[0][0], tf[1][0], tf[2][0])
    v = (tf[0][1], tf[1][1], tf[2][1])
    c = (tf[0][3], tf[1][3], tf[2][3])
    ex, ez, _ = prim.dimensions
    hu, hv = ex / 2.0, ez / 2.0
    atlas_corners = [
        tuple(c[i] - u[i] * hu - v[i] * hv for i in range(3)),
        tuple(c[i] + u[i] * hu - v[i] * hv for i in range(3)),
        tuple(c[i] + u[i] * hu + v[i] * hv for i in range(3)),
        tuple(c[i] - u[i] * hu + v[i] * hv for i in range(3)),
    ]
    return [blender_point_from_atlas(*p) for p in atlas_corners]


def write_dynamic_plate_blender_script(plate: DynamicPlate,
                                       package_dir: str | Path,
                                       output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    package_dir = Path(package_dir)

    if plate.crop_camera is None or plate.source_camera is None:
        raise ValueError("DynamicPlate Blender export needs source and crop "
                         "cameras")
    crop_intr = plate.crop_camera.intrinsics
    if crop_intr.fx_px is None or crop_intr.cx_px is None:
        raise ValueError("Crop camera intrinsics are incomplete")

    proj_world = blender_matrix_from_atlas(
        plate.crop_camera.extrinsics.camera_world_matrix)
    render_world = blender_matrix_from_atlas(
        plate.source_camera.extrinsics.camera_world_matrix)

    # Pixel-space projection factors for the CROP camera raster:
    #   u = cx'/W' + (fx'/W') * cam_x / depth
    #   V = 1 - v_img/H' = (1 - cy'/H') + (fy'/H') * cam_y / depth
    # Blender's image V origin is BOTTOM; through the TexCoord OBJECT output
    # (camera-local space, y up) the top-down image row must be flipped into
    # texture V explicitly — verified live in Blender 5.2 on the 4K seacliff
    # plate (the un-flipped form projects the plate upside down).
    w, h = crop_intr.image_width, crop_intr.image_height
    scale_u = float(crop_intr.fx_px) / w
    scale_v = float(crop_intr.fy_px or crop_intr.fx_px) / h
    offset_u = float(crop_intr.cx_px) / w
    offset_v = 1.0 - float(crop_intr.cy_px) / h

    corners = _receiver_corners(plate)
    frame_count = plate.frame_count
    first_frame_rel = "generated/frame_0000.png"
    still_rel = "source/crop.png"

    proj_focal = crop_intr.focal_length_mm or 35.0
    proj_sensor_w = crop_intr.sensor_width_mm or 36.0
    src_intr = plate.source_camera.intrinsics
    src_focal = src_intr.focal_length_mm or 35.0
    src_sensor_w = src_intr.sensor_width_mm or 36.0

    script = f'''"""Atlas DynamicPlate Blender scene — {plate.plate_id}.

Two cameras, two jobs (do not merge them):
  * atlas_projection_camera: fixed; the temporal sequence projects through it.
  * atlas_render_camera: the artist camera; move it freely.
"""

import os
import bpy
import mathutils


def build_scene(package_dir=None):
    package_dir = package_dir or os.path.dirname(os.path.abspath(__file__))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # --- Projection camera (FIXED — the crop camera of the solve) ---
    projection_data = bpy.data.cameras.new("atlas_projection_camera")
    projection_data.lens = {proj_focal!r}
    projection_data.sensor_width = {proj_sensor_w!r}
    projection_camera = bpy.data.objects.new("atlas_projection_camera",
                                             projection_data)
    bpy.context.collection.objects.link(projection_camera)
    projection_camera.matrix_world = mathutils.Matrix({proj_world!r})
    projection_camera.hide_render = True

    # --- Artist render camera (moves freely; starts at the source pose) ---
    render_data = bpy.data.cameras.new("atlas_render_camera")
    render_data.lens = {src_focal!r}
    render_data.sensor_width = {src_sensor_w!r}
    render_camera = bpy.data.objects.new("atlas_render_camera", render_data)
    bpy.context.collection.objects.link(render_camera)
    render_camera.matrix_world = mathutils.Matrix({render_world!r})
    bpy.context.scene.camera = render_camera
    bpy.context.view_layer.update()

    # --- Receiver geometry (Atlas plane, already converted to Z-up) ---
    receiver_vertices = {corners!r}
    receiver_faces = [(0, 1, 2, 3)]
    receiver_data = bpy.data.meshes.new("atlas_dynamic_plate_receiver")
    receiver_data.from_pydata(receiver_vertices, [], receiver_faces)
    receiver_data.update()
    receiver = bpy.data.objects.new("atlas_dynamic_plate_receiver",
                                    receiver_data)
    bpy.context.collection.objects.link(receiver)

    # --- Temporal projection material ---
    # u = {offset_u!r} + {scale_u!r} * cam_x / depth   (crop-camera pixels)
    # V = {offset_v!r} + {scale_v!r} * cam_y / depth   (bottom-origin texture V)
    mat = bpy.data.materials.new("atlas_dynamic_plate_mat")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    coord = nodes.new("ShaderNodeTexCoord")
    # Object-space of the PROJECTION camera, not the active render camera —
    # this is what keeps the plate registered while the artist camera moves.
    coord.object = projection_camera

    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Object"], sep.inputs["Vector"])

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
    mul_v.inputs[1].default_value = {scale_v!r}  # V flip baked into offset_v
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
    sequence_path = os.path.join(package_dir, {first_frame_rel!r})
    still_path = os.path.join(package_dir, {still_rel!r})
    if os.path.exists(sequence_path):
        img = bpy.data.images.load(sequence_path, check_existing=True)
        img.source = "SEQUENCE"
        img_tex.image = img
        img_tex.image_user.frame_start = {plate.frame_start + 1}
        img_tex.image_user.frame_offset = 0
        img_tex.image_user.frame_duration = {frame_count}
        img_tex.image_user.use_auto_refresh = True
        img_tex.image_user.use_cyclic = True
    elif os.path.exists(still_path):
        # No generated frames yet — project the still crop so the scene is
        # inspectable; regenerate after the temporal pass.
        img_tex.image = bpy.data.images.load(still_path, check_existing=True)
    links.new(combine.outputs["Vector"], img_tex.inputs["Vector"])

    # Unlit on purpose: a projected plate is already-lit photography (same
    # doctrine as blender_exporter.py / the Atlas viewport shader).
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    links.new(img_tex.outputs["Color"], emission.inputs["Color"])

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    receiver.data.materials.append(mat)

    scene = bpy.context.scene
    scene.frame_start = {plate.frame_start}
    scene.frame_end = {plate.frame_end}
    scene.render.fps = {int(round(plate.frame_rate))}
    scene.render.resolution_x = {plate.source_width}
    scene.render.resolution_y = {plate.source_height}

    return projection_camera, render_camera


if __name__ == "__main__":
    build_scene()
'''
    destination.write_text(script, encoding="utf-8")
    return destination
