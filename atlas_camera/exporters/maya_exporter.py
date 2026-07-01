"""Maya handoff exporter.

This writes a Maya Python scene-builder script instead of raw .ma. The core
schema remains Y-up and DCC-agnostic; Maya-specific commands live here.
"""

from __future__ import annotations

from pathlib import Path

from atlas_camera.core.camera_math import derive_sensor_height_mm, mm_to_inches, pixel_offset_to_normalized_film_offset
from atlas_camera.core.schema import AtlasSolve

NODE_CAMERA = "atlas_CAMERA"
NODE_PROJECTION_GRP = "atlas_PROJECTION_GRP"
NODE_GEOMETRY_GRP = "atlas_GEOMETRY_GRP"
NODE_DEBUG_GRP = "atlas_DEBUG_GRP"
NODE_REFERENCE_GRP = "atlas_REFERENCE_GRP"
NODE_PROJECTION_PLANE = "atlas_PROJECTION_PLANE"


def write_maya_scene_script(
    solve: AtlasSolve,
    output_path: str | Path,
    *,
    source_image_name: str = "source_image.png",
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = solve.camera.intrinsics
    if intrinsics.focal_length_mm is None:
        raise ValueError(
            "Cannot export Maya scene without focal_length_mm. "
            "Run a camera solve or provide an explicit focal-length hint."
        )
    focal = intrinsics.focal_length_mm
    sensor_height_mm = derive_sensor_height_mm(intrinsics)
    horizontal_aperture_in = mm_to_inches(intrinsics.sensor_width_mm)
    vertical_aperture_in = mm_to_inches(sensor_height_mm)
    cx = intrinsics.cx_px if intrinsics.cx_px is not None else intrinsics.image_width / 2.0
    cy = intrinsics.cy_px if intrinsics.cy_px is not None else intrinsics.image_height / 2.0
    horizontal_offset = pixel_offset_to_normalized_film_offset(
        cx - (intrinsics.image_width / 2.0),
        aperture_mm=intrinsics.sensor_width_mm,
        image_size_px=intrinsics.image_width,
    )
    vertical_offset = pixel_offset_to_normalized_film_offset(
        cy - (intrinsics.image_height / 2.0),
        aperture_mm=sensor_height_mm,
        image_size_px=intrinsics.image_height,
    )
    proxy_names = [primitive.name for primitive in solve.projection_scene.proxy_geometry]
    focal_warning = ""
    if solve.camera.focal_length_inferred:
        focal_warning = (
            "\n    cmds.warning(\"Atlas focal_length_mm was inferred from a fallback assumption; "
            "review atlas_solve.json before final handoff.\")"
        )

    script = f'''"""Open an Atlas Camera review scene in Maya.

Generated from an Atlas core solve. Atlas core convention is right-handed Y-up.
"""

from __future__ import annotations

import os
import maya.cmds as cmds


def build_scene(package_dir=None):
    package_dir = package_dir or os.path.dirname(os.path.abspath(__file__))
    cmds.file(new=True, force=True)
    cmds.upAxis(axis="y", rotateView=True)

    projection_group = cmds.group(empty=True, name="{NODE_PROJECTION_GRP}")
    geometry_group = cmds.group(empty=True, name="{NODE_GEOMETRY_GRP}")
    debug_group = cmds.group(empty=True, name="{NODE_DEBUG_GRP}")
    reference_group = cmds.group(empty=True, name="{NODE_REFERENCE_GRP}")

    camera_transform, camera_shape = cmds.camera(name="{NODE_CAMERA}")
    cmds.parent(camera_transform, projection_group)
    cmds.setAttr(camera_shape + ".focalLength", {focal!r})
    cmds.setAttr(camera_shape + ".horizontalFilmAperture", {horizontal_aperture_in!r})
    cmds.setAttr(camera_shape + ".verticalFilmAperture", {vertical_aperture_in!r})
    cmds.setAttr(camera_shape + ".horizontalFilmOffset", {horizontal_offset!r})
    cmds.setAttr(camera_shape + ".verticalFilmOffset", {vertical_offset!r})
    cmds.setAttr(camera_transform + ".translateX", {solve.camera.extrinsics.camera_position[0]!r})
    cmds.setAttr(camera_transform + ".translateY", {solve.camera.extrinsics.camera_position[1]!r})
    cmds.setAttr(camera_transform + ".translateZ", {solve.camera.extrinsics.camera_position[2]!r})
    {focal_warning}

    image_path = os.path.join(package_dir, "{source_image_name}")
    if os.path.exists(image_path):
        image_plane = cmds.imagePlane(camera=camera_shape, fileName=image_path)[0]
        cmds.setAttr(image_plane + ".displayMode", 3)
        cmds.parent(image_plane, reference_group)

    cmds.grid(size=12, spacing=1)

    # --- Camera-projection shader on 40 x 40 m ground plane ---
    # place3dTexture parented to camera so its worldInverseMatrix = camera view matrix,
    # which is what projection.pm expects for a perspective projection from the camera.
    proj_place3d = cmds.shadingNode("place3dTexture", asUtility=True, name="atlas_proj_place3d")
    cmds.parent(proj_place3d, camera_transform)
    cmds.setAttr(proj_place3d + ".translate", 0, 0, 0, type="double3")
    cmds.setAttr(proj_place3d + ".rotate", 0, 0, 0, type="double3")

    proj_file = cmds.shadingNode("file", asTexture=True, isColorManaged=True, name="atlas_proj_file")
    cmds.setAttr(proj_file + ".fileTextureName", image_path, type="string")
    cmds.setAttr(proj_file + ".ignoreColorSpaceFileRules", True)

    proj_tex = cmds.shadingNode("projection", asTexture=True, name="atlas_proj_texture")
    cmds.setAttr(proj_tex + ".projType", 8)  # 8 = perspective projection
    cmds.setAttr(proj_tex + ".focalLength", {focal!r})
    cmds.setAttr(proj_tex + ".horizontalFilmAperture", {horizontal_aperture_in!r})
    cmds.setAttr(proj_tex + ".verticalFilmAperture", {vertical_aperture_in!r})
    cmds.setAttr(proj_tex + ".fitType", 2)  # best fit
    cmds.connectAttr(proj_file + ".outColor", proj_tex + ".image", force=True)
    cmds.connectAttr(proj_place3d + ".worldInverseMatrix[0]", proj_tex + ".pm", force=True)

    proj_mat = cmds.shadingNode("lambert", asShader=True, name="atlas_proj_mat")
    cmds.connectAttr(proj_tex + ".outColor", proj_mat + ".color", force=True)
    proj_sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="atlas_proj_SG")
    cmds.connectAttr(proj_mat + ".outColor", proj_sg + ".surfaceShader", force=True)

    projection_plane = cmds.polyPlane(
        name="{NODE_PROJECTION_PLANE}",
        width=40, height=40, subdivisionsX=64, subdivisionsY=64,
    )[0]
    cmds.sets(projection_plane, edit=True, forceElement=proj_sg)
    cmds.parent(projection_plane, geometry_group)
    # -----------------------------------------------------------------

    for axis, color, translate, scale in (
        ("x", 13, (2, 0, 0), (4, 0.04, 0.04)),
        ("y", 14, (0, 2, 0), (0.04, 4, 0.04)),
        ("z", 6, (0, 0, 2), (0.04, 0.04, 4)),
    ):
        cube = cmds.polyCube(name=f"atlas_{{axis}}_axis_guide")[0]
        cmds.setAttr(cube + ".translate", *translate, type="double3")
        cmds.setAttr(cube + ".scale", *scale, type="double3")
        shader = cmds.shadingNode("lambert", asShader=True, name=f"atlas_{{axis}}_axis_mat")
        cmds.setAttr(shader + ".color", *(1, 0, 0) if axis == "x" else (0, 1, 0) if axis == "y" else (0, 0.2, 1), type="double3")
        cmds.select(cube)
        cmds.hyperShade(assign=shader)
        cmds.parent(cube, debug_group)

    for proxy_name in {proxy_names!r}:
        if proxy_name == "ground_plane":
            continue
        proxy = cmds.polyCube(name=proxy_name)[0]
        cmds.parent(proxy, geometry_group)

    cmds.lookThru(camera_transform)
    return camera_transform


if __name__ == "__main__":
    build_scene()
'''
    destination.write_text(script, encoding="utf-8")
    return destination


class MayaExporter:
    def write_scene(self, solve: AtlasSolve, output_path: str | Path) -> Path:
        return write_maya_scene_script(solve, output_path)
