"""Helpers for building simple projection proxy scenes."""

from __future__ import annotations

from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive, identity_matrix4


def create_default_projection_scene() -> AtlasProjectionScene:
    """An empty projection scene with just the coordinate-convention metadata.

    Used to previously seed a placeholder "ground_plane" primitive
    (role="ground") here, but it had no downstream consumer anywhere in the
    codebase — serialize_proxy_geometry only ever sends role=="projection_proxy"
    (PROXY_ROLE) primitives to the viewport, so it was never rendered, never
    exported, and up_axis/coordinate_system are already exposed as their own
    fields on AtlasProjectionScene, making its metadata redundant too. Its one
    real effect was confusion: a real, rendered "projection_ground" primitive
    (PROXY_ROLE, from AtlasDeriveWalls/derive_projection_proxies) has an
    almost-identical name, and AtlasMergeGeometry originally duplicated this
    placeholder across merged branches before that bug was found and fixed
    (see AtlasMergeGeometry's docstring) — removed rather than renamed, since
    nothing needs it.
    """
    return AtlasProjectionScene(
        coordinate_system="right_handed",
        up_axis="Y",
        debug_metadata={
            "convention": "Atlas core uses right-handed Y-up coordinates.",
        },
    )


def proxy_box(
    name: str,
    *,
    dimensions: tuple[float, float, float] = (1.0, 1.0, 1.0),
    material: str = "atlas_proxy",
) -> AtlasProxyPrimitive:
    return AtlasProxyPrimitive(
        name=name,
        primitive_type="box",
        transform_matrix=identity_matrix4(),
        dimensions=dimensions,
        material=material,
        metadata={"role": "projection_proxy"},
    )


def add_axis_guides(scene: AtlasProjectionScene) -> AtlasProjectionScene:
    for axis, material in (("X", "atlas_axis_x"), ("Y", "atlas_axis_y"), ("Z", "atlas_axis_z")):
        scene.proxy_geometry.append(
            AtlasProxyPrimitive(
                name=f"{axis.lower()}_axis_guide",
                primitive_type="axis_guide",
                dimensions=(1.0, 1.0, 1.0),
                material=material,
                metadata={"axis": axis},
            )
        )
    return scene

