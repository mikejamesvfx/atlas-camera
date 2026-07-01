"""USD export boundary for Atlas Camera.

Produces three USD assets:
  camera.usda            — UsdGeom.Camera with full intrinsics + world transform
  proxy_scene.usda       — projection ground plane + proxy geometry with transforms
  projection_scene.usda  — combined stage: camera + plane + UsdPreviewSurface material
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_camera.core.camera_math import derive_sensor_height_mm
from atlas_camera.core.schema import AtlasSolve


def _import_pxr() -> tuple[Any, Any]:
    """Minimal import for legacy callers and monkeypatching in tests."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "USD export requires the optional usd-core package. "
            "Install with: pip install -e .[usd]"
        ) from exc
    return Usd, UsdGeom


def _import_pxr_full() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Full pxr import used by all exporter methods."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    except ImportError as exc:
        raise RuntimeError(
            "USD export requires the optional usd-core package. "
            "Install with: pip install -e .[usd]"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def _gf_mat4(world_mat: Any, Gf: Any) -> Any:
    """Build a Gf.Matrix4d from a 4×4 row-major tuple."""
    return Gf.Matrix4d(
        world_mat[0][0], world_mat[0][1], world_mat[0][2], world_mat[0][3],
        world_mat[1][0], world_mat[1][1], world_mat[1][2], world_mat[1][3],
        world_mat[2][0], world_mat[2][1], world_mat[2][2], world_mat[2][3],
        world_mat[3][0], world_mat[3][1], world_mat[3][2], world_mat[3][3],
    )


def _define_ground_plane(stage: Any, path: str, Gf: Any, Sdf: Any, UsdGeom: Any, Vt: Any) -> Any:
    """Define a 40×40 m quad mesh at Y=0 in the XZ plane with flat `st` UV primvar.

    Camera-projected UVs are generated in the consuming DCC; the `st` primvar
    provides a fallback flat mapping for DCC applications that read it directly.
    """
    plane = UsdGeom.Mesh.Define(stage, path)
    plane.CreatePointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-20.0, 0.0, -20.0),
        Gf.Vec3f( 20.0, 0.0, -20.0),
        Gf.Vec3f( 20.0, 0.0,  20.0),
        Gf.Vec3f(-20.0, 0.0,  20.0),
    ]))
    plane.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    plane.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    plane.CreateExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-20.0, 0.0, -20.0),
        Gf.Vec3f( 20.0, 0.0,  20.0),
    ]))
    plane.CreateSubdivisionSchemeAttr().Set("catmullClark")
    primvars_api = UsdGeom.PrimvarsAPI(plane.GetPrim())
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    st.Set(Vt.Vec2fArray([
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ]))
    return plane


def _define_projection_material(
    stage: Any,
    mat_path: str,
    source_image_name: str,
    Sdf: Any,
    UsdShade: Any,
) -> Any:
    """Define a UsdPreviewSurface material that references the source image.

    The UsdUVTexture reads the `st` primvar.  Camera-projected UVs can be
    baked onto the mesh in the consuming DCC to replace the flat fallback.
    """
    material = UsdShade.Material.Define(stage, mat_path)

    pbr = UsdShade.Shader.Define(stage, f"{mat_path}/PbrShader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    tex = UsdShade.Shader.Define(stage, f"{mat_path}/SourceTexture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(source_image_name)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set((0.18, 0.18, 0.18, 1.0))
    rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    uv_reader = UsdShade.Shader.Define(stage, f"{mat_path}/UVReader")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_out = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    material.CreateSurfaceOutput().ConnectToSource(
        pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return material


def _set_camera_intrinsics(camera: Any, intrinsics: Any, Gf: Any) -> None:
    """Write focal length, apertures, clipping range, and principal-point offsets."""
    focal = float(intrinsics.focal_length_mm or 35.0)
    sensor_w = float(intrinsics.sensor_width_mm)
    sensor_h = derive_sensor_height_mm(intrinsics)
    camera.GetFocalLengthAttr().Set(focal)
    camera.GetHorizontalApertureAttr().Set(sensor_w)
    camera.GetVerticalApertureAttr().Set(sensor_h)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

    if intrinsics.cx_px is not None:
        cx_offset_mm = (
            (intrinsics.cx_px - intrinsics.image_width / 2.0)
            * (sensor_w / intrinsics.image_width)
        )
        camera.GetHorizontalApertureOffsetAttr().Set(float(cx_offset_mm))
    if intrinsics.cy_px is not None:
        cy_offset_mm = (
            -(intrinsics.cy_px - intrinsics.image_height / 2.0)
            * (sensor_h / intrinsics.image_height)
        )
        camera.GetVerticalApertureOffsetAttr().Set(float(cy_offset_mm))


class USDExporter:
    def export_camera(self, solve: AtlasSolve, output_path: str | Path) -> Path:
        """Camera-only USD asset with full intrinsics and world transform."""
        Gf, Sdf, Usd, UsdGeom, UsdShade, Vt = _import_pxr_full()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        stage = Usd.Stage.CreateNew(str(destination))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        camera = UsdGeom.Camera.Define(stage, "/AtlasCamera/Camera")
        _set_camera_intrinsics(camera, solve.camera.intrinsics, Gf)
        camera.AddTransformOp().Set(_gf_mat4(solve.camera.extrinsics.camera_world_matrix, Gf))

        stage.GetRootLayer().Save()
        return destination

    def export_proxy_scene(self, solve: AtlasSolve, output_path: str | Path) -> Path:
        """Projection ground plane + proxy geometry with world transforms."""
        Gf, Sdf, Usd, UsdGeom, UsdShade, Vt = _import_pxr_full()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        stage = Usd.Stage.CreateNew(str(destination))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        root = UsdGeom.Xform.Define(stage, "/AtlasProjectionScene")
        root.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)

        _define_ground_plane(stage, "/AtlasProjectionScene/atlas_projection_plane", Gf, Sdf, UsdGeom, Vt)

        for index, primitive in enumerate(solve.projection_scene.proxy_geometry):
            prim_name = (primitive.name or f"proxy_{index}").replace(" ", "_").replace("-", "_")
            prim_path = f"/AtlasProjectionScene/{prim_name}"
            if primitive.primitive_type == "plane":
                prim = _define_ground_plane(stage, prim_path, Gf, Sdf, UsdGeom, Vt)
            else:
                prim = UsdGeom.Cube.Define(stage, prim_path)
                dx, dy, dz = primitive.dimensions
                prim.GetSizeAttr().Set(float(max(dx, dy, dz)))
            prim.AddTransformOp().Set(_gf_mat4(primitive.transform_matrix, Gf))

        stage.GetRootLayer().Save()
        return destination

    def export_projection_scene(
        self,
        solve: AtlasSolve,
        output_path: str | Path,
        *,
        source_image_name: str = "source_image.png",
    ) -> Path:
        """Combined stage: camera + 40×40 m projection plane + UsdPreviewSurface material.

        The UsdPreviewSurface reads the `st` UV primvar on the ground plane.
        Camera-projected UVs can be baked in the consuming DCC to replace the
        flat fallback UVs that are stored in the file.
        """
        Gf, Sdf, Usd, UsdGeom, UsdShade, Vt = _import_pxr_full()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        stage = Usd.Stage.CreateNew(str(destination))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.Xform.Define(stage, "/AtlasProjection")

        # Camera
        camera = UsdGeom.Camera.Define(stage, "/AtlasProjection/Camera")
        _set_camera_intrinsics(camera, solve.camera.intrinsics, Gf)
        camera.AddTransformOp().Set(_gf_mat4(solve.camera.extrinsics.camera_world_matrix, Gf))

        # Projection plane with UV primvar
        plane = _define_ground_plane(
            stage, "/AtlasProjection/ProjectionPlane", Gf, Sdf, UsdGeom, Vt
        )

        # UsdPreviewSurface material bound to the plane
        material = _define_projection_material(
            stage, "/AtlasProjection/Materials/ProjectionMat",
            source_image_name, Sdf, UsdShade,
        )
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material)

        stage.GetRootLayer().Save()
        return destination
