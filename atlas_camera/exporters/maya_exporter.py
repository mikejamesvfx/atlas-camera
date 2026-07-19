"""Maya handoff exporter.

This writes a Maya Python scene-builder script instead of raw .ma. The core
schema remains Y-up and DCC-agnostic; Maya-specific commands live here.
"""

from __future__ import annotations

from pathlib import Path
import re

from atlas_camera.core.camera_math import derive_sensor_height_mm, mm_to_inches, pixel_offset_to_normalized_film_offset
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasSolve, Matrix4
from atlas_camera.exporters._plate import primary_plate_colorspace, primary_plate_path

NODE_CAMERA = "atlas_CAMERA"
NODE_PROJECTION_GRP = "atlas_PROJECTION_GRP"
NODE_GEOMETRY_GRP = "atlas_GEOMETRY_GRP"
NODE_DEBUG_GRP = "atlas_DEBUG_GRP"
NODE_REFERENCE_GRP = "atlas_REFERENCE_GRP"
NODE_PROJECTION_PLANE = "atlas_PROJECTION_PLANE"


def _maya_matrix_from_atlas(matrix: Matrix4) -> list[float]:
    """Atlas column-vector row-major 4x4 -> Maya row-vector convention.

    Transpose the 3x3 rotation block and put translation in the last row.
    Both systems are Y-up right-handed so no coordinate-axis swap is needed.
    Shared by the camera and every proxy primitive transform below.
    """
    return [
        matrix[0][0], matrix[1][0], matrix[2][0], 0.0,
        matrix[0][1], matrix[1][1], matrix[2][1], 0.0,
        matrix[0][2], matrix[1][2], matrix[2][2], 0.0,
        matrix[0][3], matrix[1][3], matrix[2][3], 1.0,
    ]



def write_maya_scene_script(
    solve: AtlasSolve,
    output_path: str | Path,
    *,
    source_image_name: str = "source_image.png",
    relief_mesh_obj_path: str | Path | None = None,
    use_package_source: bool = False,
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
    # Only projection-derived proxies (AtlasDeriveProjectionGeometry output) —
    # excludes debug/reference helpers that live elsewhere in proxy_geometry.
    proxies = [p for p in solve.projection_scene.proxy_geometry
               if (p.metadata or {}).get("role") == PROXY_ROLE]
    # box/cylinder/plane get real dimensions + transform via cmds.polyCube/
    # polyCylinder/polyPlane + cmds.xform. The relief mesh ("mesh" type) is
    # NOT reconstructed here — its vertices/faces live in the primitive's
    # metadata and are already exported as a textured OBJ by
    # AtlasExportReliefMesh; importing that proven file is far more robust
    # than re-deriving mesh construction in generated MEL/Python. Skipped
    # here regardless of whether relief_mesh_obj_path was supplied to the
    # caller (that import happens in a separate script block below).
    proxy_specs = [
        {
            "name": p.name,
            "type": p.primitive_type,
            "dimensions": [float(v) for v in p.dimensions],
            "matrix": _maya_matrix_from_atlas(p.transform_matrix),
        }
        for p in proxies if p.primitive_type != "mesh"
    ]
    relief_mesh_obj_path_str = str(relief_mesh_obj_path) if relief_mesh_obj_path else None
    source_plate_path = None if use_package_source else primary_plate_path(solve)
    source_colorspace = primary_plate_colorspace(solve)
    output_profile = getattr(solve, "output_profile", None)
    ocio_summary = (
        output_profile.to_dict() if output_profile and hasattr(output_profile, "to_dict") else None
    )

    focal_warning = ""
    if solve.camera.focal_length_inferred:
        focal_warning = (
            "\n    cmds.warning(\"Atlas focal_length_mm was inferred from a fallback assumption; "
            "review atlas_solve.json before final handoff.\")"
        )

    maya_matrix = _maya_matrix_from_atlas(solve.camera.extrinsics.camera_world_matrix)

    script = f'''"""Open an Atlas Camera review scene in Maya.

Generated from an Atlas core solve. Atlas core convention is right-handed Y-up.
Camera position and rotation are applied via cmds.xform worldSpace matrix.
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
    cmds.xform(camera_transform, worldSpace=True, matrix={maya_matrix!r})
    {focal_warning}

    image_path = {source_plate_path!r} or os.path.join(package_dir, {source_image_name!r})
    source_colorspace = {source_colorspace!r}
    ocio_summary = {str(ocio_summary)!r}
    if os.path.exists(image_path):
        image_plane = cmds.imagePlane(camera=camera_shape, fileName=image_path)[0]
        cmds.setAttr(image_plane + ".displayMode", 3)
        cmds.parent(image_plane, reference_group)
    else:
        cmds.warning("Atlas source plate not found: " + image_path)

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
    if source_colorspace and cmds.attributeQuery("colorSpace", node=proj_file, exists=True):
        try:
            cmds.setAttr(proj_file + ".colorSpace", source_colorspace, type="string")
        except Exception:
            cmds.warning("Atlas could not set Maya file colorSpace to: " + source_colorspace)
    cmds.addAttr(proj_file, longName="atlasSourceColorspace", dataType="string")
    cmds.setAttr(proj_file + ".atlasSourceColorspace", source_colorspace or "", type="string")
    cmds.addAttr(proj_file, longName="atlasOutputProfile", dataType="string")
    cmds.setAttr(proj_file + ".atlasOutputProfile", ocio_summary, type="string")

    # Perspective projection takes its whole frustum from linkedCamera — the
    # projection node has NO focalLength/aperture attrs (confirmed live in
    # Maya 2027 while verifying the layers exporter; the previous attrs-based
    # setup errored on open and had never been run in a real Maya).
    proj_tex = cmds.shadingNode("projection", asTexture=True, name="atlas_proj_texture")
    cmds.setAttr(proj_tex + ".projType", 8)  # 8 = perspective projection
    cmds.connectAttr(proj_file + ".outColor", proj_tex + ".image", force=True)
    cmds.connectAttr(camera_shape + ".message", proj_tex + ".linkedCamera", force=True)

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

    for spec in {proxy_specs!r}:
        dx, dy, dz = spec["dimensions"]
        ptype = spec["type"]
        if ptype == "cylinder":
            proxy = cmds.polyCylinder(name=spec["name"], radius=dx / 2.0, height=dy)[0]
        elif ptype == "plane":
            proxy = cmds.polyPlane(name=spec["name"], width=dx, height=dz)[0]
        else:
            proxy = cmds.polyCube(name=spec["name"], width=dx, height=dy, depth=dz)[0]
        cmds.xform(proxy, worldSpace=True, matrix=spec["matrix"])
        cmds.parent(proxy, geometry_group)

    # Relief mesh: vertices are already world-space (identity transform), so
    # importing the OBJ that AtlasExportReliefMesh already wrote needs no
    # extra positioning.
    relief_mesh_obj_path = {relief_mesh_obj_path_str!r}
    if relief_mesh_obj_path and os.path.exists(relief_mesh_obj_path):
        if not cmds.pluginInfo("objExport", query=True, loaded=True):
            cmds.loadPlugin("objExport")
        imported_nodes = cmds.file(
            relief_mesh_obj_path, i=True, type="OBJ", returnNewNodes=True,
            groupReference=True, groupName="atlas_relief_mesh_grp",
            mergeNamespacesOnClash=False, namespace="atlas_relief",
        )
        for node in imported_nodes or []:
            if node.endswith("atlas_relief_mesh_grp") and cmds.nodeType(node) == "transform":
                cmds.parent(node, geometry_group)
                break
    elif relief_mesh_obj_path:
        cmds.warning("Atlas relief mesh OBJ not found at: " + relief_mesh_obj_path)

    cmds.lookThru(camera_transform)
    return camera_transform


if __name__ == "__main__":
    build_scene()
'''
    destination.write_text(script, encoding="utf-8")
    return destination


class MayaExporter:
    def write_scene(
        self,
        solve: AtlasSolve,
        output_path: str | Path,
        *,
        source_image_name: str = "source_image.png",
        relief_mesh_obj_path: str | Path | None = None,
        use_package_source: bool = False,
    ) -> Path:
        return write_maya_scene_script(
            solve,
            output_path,
            source_image_name=source_image_name,
            relief_mesh_obj_path=relief_mesh_obj_path,
            use_package_source=use_package_source,
        )


def _mel_safe_path(path: Path) -> str:
    return path.resolve().as_posix()


def _mel_safe_proc_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = "atlas_" + safe
    return safe


def write_maya_mel_launcher(
    review_package_dir: Path | str,
    review_name: str = "atlas_review_001",
) -> Path:
    """Write a MEL launcher beside maya_open_scene.py in the review package directory.

    Passes the absolute package path to build_scene() so artists can source or
    drag/drop the .mel file without __file__ being defined in Maya's Script Editor.
    """
    review_package_dir = Path(review_package_dir)
    package_dir_mel = _mel_safe_path(review_package_dir)
    proc_name = f"atlas_open_{_mel_safe_proc_name(review_name)}"
    mel_path = review_package_dir / f"open_{review_name}.mel"

    mel = f'''// -----------------------------------------------------------------------------
// Atlas Camera Maya Review Launcher — {review_name}
// Generated by Atlas Camera.
// Source or drag/drop this file in Maya to open the solved review scene.
// -----------------------------------------------------------------------------

global proc {proc_name}()
{{
    string $packageDir = "{package_dir_mel}";

    print("\\n[Atlas] Opening Maya review package...\\n");
    print("[Atlas] Package: " + $packageDir + "\\n");

    string $py =
        "import os, sys, importlib\\n"
        + "package_dir = r'" + $packageDir + "'\\n"
        + "script_path = os.path.join(package_dir, 'maya_open_scene.py')\\n"
        + "if not os.path.exists(script_path):\\n"
        + "    import maya.cmds as cmds\\n"
        + "    cmds.error('Atlas: maya_open_scene.py not found at: ' + script_path)\\n"
        + "if package_dir not in sys.path:\\n"
        + "    sys.path.insert(0, package_dir)\\n"
        + "import maya_open_scene\\n"
        + "importlib.reload(maya_open_scene)\\n"
        + "maya_open_scene.build_scene(package_dir)\\n"
        + "print('[Atlas] Review scene opened from: ' + package_dir)\\n";

    python($py);
}}

{proc_name}();
'''
    mel_path.write_text(mel, encoding="utf-8")
    return mel_path


# ---------------------------------------------------------------------------
# All-in-one layered projection scene (.ma) — the Maya twin of
# nuke_exporter.write_nuke_layers_script.
# ---------------------------------------------------------------------------

def _matrix_to_maya_trs(world_matrix):
    """Atlas column-vector world matrix -> Maya (translate, rotate XYZ degrees).

    Native .ma transform nodes take translate/rotate attributes, not a
    matrix, so the rotation block is decomposed into Maya's default rotate
    order "xyz" (column-vector composition C = Rz @ Ry @ Rx). Standard
    extraction: rx = atan2(C21, C22), ry = atan2(-C20, hypot(C21, C22)),
    rz = atan2(C10, C00). Both frames are right-handed Y-up, so the 3x3
    passes through untransposed (Atlas stores column-vector matrices).
    Verified by a recomposition round-trip in tests/test_maya_layers_export.py.
    """
    import math

    m = world_matrix
    t = (float(m[0][3]), float(m[1][3]), float(m[2][3]))
    rx = math.degrees(math.atan2(m[2][1], m[2][2]))
    ry = math.degrees(math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2])))
    rz = math.degrees(math.atan2(m[1][0], m[0][0]))
    return t, (rx, ry, rz)


def _ma_camera_blocks(node_name: str, camera, parent: str):
    """Native .ma transform+camera blocks for one camera. Long attribute
    names are legal MEL (a .ma file IS MEL statements), mirroring exactly
    the attributes the verified cmds-based exporter above sets."""
    from atlas_camera.exporters._layers import layer_focal_mm

    intr = camera.intrinsics
    focal = layer_focal_mm(intr)
    sensor_h_mm = derive_sensor_height_mm(intr)
    hfa = mm_to_inches(intr.sensor_width_mm or 36.0)
    vfa = mm_to_inches(sensor_h_mm)
    w = intr.image_width or 1
    h = intr.image_height or 1
    cx = intr.cx_px if intr.cx_px is not None else w / 2.0
    cy = intr.cy_px if intr.cy_px is not None else h / 2.0
    hfo = pixel_offset_to_normalized_film_offset(
        cx - w / 2.0, aperture_mm=intr.sensor_width_mm or 36.0, image_size_px=w)
    vfo = pixel_offset_to_normalized_film_offset(
        cy - h / 2.0, aperture_mm=sensor_h_mm, image_size_px=h)
    t, r = _matrix_to_maya_trs(camera.extrinsics.camera_world_matrix)
    block = (
        f'createNode transform -n "{node_name}" -p "{parent}";\n'
        f'\tsetAttr ".translate" -type "double3" {t[0]!r} {t[1]!r} {t[2]!r} ;\n'
        f'\tsetAttr ".rotate" -type "double3" {r[0]!r} {r[1]!r} {r[2]!r} ;\n'
        f'createNode camera -n "{node_name}Shape" -p "{node_name}";\n'
        f'\tsetAttr ".focalLength" {focal!r};\n'
        f'\tsetAttr ".horizontalFilmAperture" {hfa!r};\n'
        f'\tsetAttr ".verticalFilmAperture" {vfa!r};\n'
        f'\tsetAttr ".horizontalFilmOffset" {hfo!r};\n'
        f'\tsetAttr ".verticalFilmOffset" {vfo!r};\n'
        f'\tsetAttr ".locatorScale" 0.3;\n'
    )
    return block, {"focal": focal, "hfa": hfa, "vfa": vfa}


def _maya_layers_on_open_script(script_layers) -> str:
    """The on-open scriptNode payload: OBJ imports + the verified projection
    shading network per layer (place3dTexture parented to that layer's
    camera -> projection.pm, projType 8 — identical to
    write_maya_scene_script's single-projection setup). Idempotent: skips
    layers whose geo group already exists; warns instead of erroring on
    missing assets so a moved package degrades gracefully."""
    return f'''import os
import maya.cmds as cmds


def _atlas_load_layers():
    layers = {script_layers!r}
    if not cmds.pluginInfo("objExport", query=True, loaded=True):
        try:
            cmds.loadPlugin("objExport")
        except Exception:
            cmds.warning("Atlas: objExport plugin unavailable - layer meshes not imported")
            return
    for L in layers:
        grp = "atlas_" + L["name"] + "_geo_grp"
        if cmds.objExists(grp):
            continue  # already built on a previous open
        if not cmds.objExists(L["cam"]):
            cmds.warning("Atlas: missing layer camera " + L["cam"])
            continue
        if not os.path.exists(L["obj"]):
            cmds.warning("Atlas: missing layer mesh " + L["obj"])
            continue
        imported = cmds.file(
            L["obj"], i=True, type="OBJ", returnNewNodes=True,
            groupReference=True, groupName=grp,
            mergeNamespacesOnClash=False, namespace="atlas_" + L["name"],
        )
        f = cmds.shadingNode("file", asTexture=True, isColorManaged=True,
                             name="atlas_" + L["name"] + "_file")
        cmds.setAttr(f + ".fileTextureName", L["plate"], type="string")
        cmds.setAttr(f + ".ignoreColorSpaceFileRules", True)
        if L["colorspace"]:
            try:
                cmds.setAttr(f + ".colorSpace", L["colorspace"], type="string")
            except Exception:
                cmds.warning("Atlas: could not set colorspace for " + L["name"])
        # Perspective projection takes its whole frustum from linkedCamera —
        # the projection node has NO focalLength/aperture attrs (confirmed
        # live in Maya 2027: setAttr on them errors and aborts the build).
        proj = cmds.shadingNode("projection", asTexture=True,
                                name="atlas_" + L["name"] + "_proj")
        cmds.setAttr(proj + ".projType", 8)
        cmds.connectAttr(f + ".outColor", proj + ".image", force=True)
        cmds.connectAttr(L["cam"] + "Shape.message", proj + ".linkedCamera", force=True)
        mat = cmds.shadingNode("lambert", asShader=True,
                               name="atlas_" + L["name"] + "_mat")
        cmds.connectAttr(proj + ".outColor", mat + ".color", force=True)
        if L["has_matte"]:
            cmds.connectAttr(f + ".outTransparency", mat + ".transparency", force=True)
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                       name="atlas_" + L["name"] + "_SG")
        cmds.connectAttr(mat + ".outColor", sg + ".surfaceShader", force=True)
        for node in imported or []:
            if node.endswith(grp) and cmds.nodeType(node) == "transform":
                # Maya's OBJ importer lands raw values as internal CM no
                # matter the scene unit; the scene declares meters, so the
                # imported geometry needs a x100 compensation to line up
                # with the native cameras (confirmed live in Maya 2027: an
                # unscaled 300m sky card measured 3m).
                #
                # The x100 MUST scale about the world origin, not the import
                # group's own pivot. groupReference lands the pivot at the
                # cm-imported geometry CENTRE, so scaling about it grows the
                # geometry's size x100 but leaves its centre where the cm
                # import put it (~1m from origin) -> every band collapses onto
                # the camera and the projection tiles/garbles (verified live
                # in Maya 2027: a wall that belongs 9m ahead landed ~0.5m
                # away, engulfing the camera). Zero the scale/rotate pivots
                # first so the x100 carries POSITION too (band_1 -> true
                # -8..-10m; render lines up exactly).
                for _piv in (".scalePivot", ".scalePivotTranslate",
                             ".rotatePivot", ".rotatePivotTranslate"):
                    cmds.setAttr(node + _piv, 0, 0, 0, type="double3")
                cmds.setAttr(node + ".scale", 100, 100, 100, type="double3")
                cmds.sets(node, edit=True, forceElement=sg)
                try:
                    cmds.parent(node, "atlas_LAYERS_GEO_GRP")
                except Exception:
                    pass
                break


_atlas_load_layers()
'''


def write_maya_layers_scene(
    solve: AtlasSolve,
    output_dir: str | Path,
    *,
    name: str = "maya_layers",
    retopo_method: str = "off",
    retopo_target_vertex_count: int = 2000,
    retopo_smooth_iterations: int = 0,
    retopo_crease_angle: float = 30.0,
    retopo_pure_quad: bool = False,
) -> dict:
    """Export EVERY projection layer on a solve — each ``ProjectionSource``
    (sky dome, clean-plate bands, multi-angle patches) — as ONE Maya ASCII
    scene: per-layer projector cameras as NATIVE .ma nodes (statically
    inspectable, referenceable) plus an embedded on-open scriptNode that
    imports each layer's OBJ and builds the projection shading network.

    The Maya twin of ``nuke_exporter.write_nuke_layers_script``: the same
    shared collection (``exporters._layers.collect_projection_layers``)
    materializes identical on-disk assets (plates with edge mattes in ALPHA,
    standalone mattes, OBJ+MTL meshes), and the same "native nodes + one
    on-open script for what static text can't wire" split the .nk writer
    settled on with its Root onScriptLoad callback.

    Edge mattes drive ``lambert.transparency`` from the plate's alpha via
    the mesh's own baked UVs — which match the plate frame by construction,
    so the per-pixel silhouette cut carries into Maya renders. Scene units
    are declared meters (Atlas is metric; imported OBJ values land in the
    same unit). NOTE: Maya's script-security preference can block
    scriptNodes in untrusted scenes — the assets sit next to the .ma, so a
    blocked scene just needs a manual OBJ import.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ma_path = out / f"{name}.ma"

    from atlas_camera.exporters._layers import collect_projection_layers

    layers, skipped = collect_projection_layers(
        solve, out,
        retopo_method=retopo_method,
        retopo_target_vertex_count=retopo_target_vertex_count,
        retopo_smooth_iterations=retopo_smooth_iterations,
        retopo_crease_angle=retopo_crease_angle,
        retopo_pure_quad=retopo_pure_quad,
    )
    if not layers:
        raise ValueError(
            "No exportable projection layers on this solve — add layers with "
            "AtlasSkyDomeLayer / AtlasCleanPlateLayer / AtlasAddPatchView first "
            f"(skipped: {skipped or 'none'})."
        )

    blocks = [
        "//Maya ASCII 2020 scene",
        f"//Name: {ma_path.name}",
        "//Atlas Camera layered projection scene (generated).",
        'requires maya "2020";',
        "currentUnit -l meter -a degree -t film;",
        'createNode transform -n "atlas_LAYERS_GRP";',
        'createNode transform -n "atlas_LAYER_CAMS_GRP" -p "atlas_LAYERS_GRP";',
        'createNode transform -n "atlas_LAYERS_GEO_GRP" -p "atlas_LAYERS_GRP";',
    ]
    render_block, _ = _ma_camera_blocks("atlas_RenderCam", solve.camera, "atlas_LAYER_CAMS_GRP")
    blocks.append(render_block)

    script_layers = []
    for layer in layers:
        cam_name = f"atlas_{layer['name']}_ProjCam"
        cam_block, _cam_meta = _ma_camera_blocks(cam_name, layer["camera"], "atlas_LAYER_CAMS_GRP")
        blocks.append(cam_block)
        script_layers.append({
            "name": layer["name"],
            "cam": cam_name,
            "obj": layer["obj_path"],
            "plate": layer["plate_path"],
            "colorspace": layer["colorspace"] or "",
            "has_matte": bool(layer["has_matte"]),
        })

    # .ma string attributes are C-style: escape backslashes, quotes, newlines.
    py = _maya_layers_on_open_script(script_layers)
    escaped = py.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    blocks.append(
        'createNode script -n "atlasLayersOnOpen";\n'
        f'\tsetAttr ".before" -type "string" "{escaped}";\n'
        '\tsetAttr ".scriptType" 1;\n'
        '\tsetAttr ".sourceType" 1;\n'
    )

    ma_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return {
        "ma_path": str(ma_path),
        "layers": [l["name"] for l in layers],
        "skipped": skipped,
    }
