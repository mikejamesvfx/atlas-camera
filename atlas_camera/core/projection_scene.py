"""Helpers for building simple projection proxy scenes."""

from __future__ import annotations

from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive, identity_matrix4


def create_default_projection_scene() -> AtlasProjectionScene:
    scene = AtlasProjectionScene(
        coordinate_system="right_handed",
        up_axis="Y",
        debug_metadata={
            "convention": "Atlas core uses right-handed Y-up coordinates.",
        },
    )
    scene.proxy_geometry.append(
        AtlasProxyPrimitive(
            name="ground_plane",
            primitive_type="plane",
            dimensions=(10.0, 0.0, 10.0),
            material="atlas_ground",
            metadata={"role": "ground", "up_axis": "Y"},
        )
    )
    return scene


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

