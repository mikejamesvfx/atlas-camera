"""Extrinsics and coordinate-convention helpers."""

from __future__ import annotations

from atlas_camera.core.schema import AtlasExtrinsics, identity_matrix3, identity_matrix4


def default_extrinsics(*, up_axis: str = "Y") -> AtlasExtrinsics:
    if up_axis not in {"X", "Y", "Z"}:
        raise ValueError("up_axis must be X, Y, or Z.")
    return AtlasExtrinsics(
        camera_rotation_matrix=identity_matrix3(),
        camera_world_matrix=identity_matrix4(),
        camera_view_matrix=identity_matrix4(),
        up_axis=up_axis,
    )


def atlas_y_up_to_blender_z_up(position: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an Atlas Y-up position to Blender-style Z-up position."""

    x, y, z = position
    return (x, -z, y)


def atlas_y_up_to_maya_y_up(position: tuple[float, float, float]) -> tuple[float, float, float]:
    """Maya is also Y-up by default, so positions pass through unchanged."""

    return position

