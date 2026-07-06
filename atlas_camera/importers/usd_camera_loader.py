"""USD camera loader boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, Matrix4


def _world_matrix_from_usd_prim(prim: Any, Usd: Any, UsdGeom: Any) -> Matrix4:
    """Read a USD prim's world-space transform and convert to Atlas's 4x4
    row-major, column-vector convention (translation in the last column).

    USD's Gf.Matrix4d is row-vector (``p' = p @ M``, translation in the last
    ROW) — the exact transpose of Atlas's convention. This mirrors
    usd_exporter.py's ``_gf_mat4``, which transposes Atlas's world matrix
    into USD's convention on export; this is the same transpose applied on
    the way back in.
    """
    xformable = UsdGeom.Xformable(prim)
    usd_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(
        tuple(float(usd_matrix[j][i]) for j in range(4))
        for i in range(4)
    )  # type: ignore[return-value]


def _extrinsics_from_world_matrix(world_matrix: Matrix4) -> AtlasExtrinsics:
    """Derive full AtlasExtrinsics (position, rotation, world+view matrices)
    from a cam->world matrix already in Atlas's convention.

    Inverts camera_math.look_at_view_matrix's own construction:
    ``camera_rotation_matrix`` is just the world matrix's 3x3 block (columns
    = camera axes in world space); the view matrix is the rigid-transform
    inverse (rotation transposed, translation = -R^T @ position) — a plain
    rotation matrix's inverse is its transpose, so no general 4x4 inversion
    is needed.
    """
    r = [[world_matrix[i][j] for j in range(3)] for i in range(3)]
    position = (float(world_matrix[0][3]), float(world_matrix[1][3]), float(world_matrix[2][3]))

    r_t = [[r[j][i] for j in range(3)] for i in range(3)]
    t_view = [-sum(r_t[i][k] * position[k] for k in range(3)) for i in range(3)]

    view_matrix: Matrix4 = (
        (r_t[0][0], r_t[0][1], r_t[0][2], t_view[0]),
        (r_t[1][0], r_t[1][1], r_t[1][2], t_view[1]),
        (r_t[2][0], r_t[2][1], r_t[2][2], t_view[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    rotation3 = tuple(tuple(row) for row in r)

    return AtlasExtrinsics(
        camera_position=position,
        camera_rotation_matrix=rotation3,  # type: ignore[arg-type]
        camera_world_matrix=world_matrix,
        camera_view_matrix=view_matrix,
    )


class USDCameraLoader:
    def load(self, path: str | Path, *, image_size: tuple[int, int] = (1920, 1080)) -> AtlasCamera:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise RuntimeError(
                "USD camera loading requires the optional usd-core package. "
                "Install with: pip install -e .[usd]"
            ) from exc

        stage = Usd.Stage.Open(str(source))
        if stage is None:
            raise RuntimeError(f"Unable to open USD stage: {source}")

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                camera = UsdGeom.Camera(prim)
                focal = camera.GetFocalLengthAttr().Get() or 35.0
                sensor_width = camera.GetHorizontalApertureAttr().Get() or 36.0
                sensor_height = camera.GetVerticalApertureAttr().Get()
                world_matrix = _world_matrix_from_usd_prim(prim, Usd, UsdGeom)
                return AtlasCamera(
                    name=prim.GetName() or "usd_camera",
                    intrinsics=build_intrinsics(
                        image_width=image_size[0],
                        image_height=image_size[1],
                        focal_length_mm=float(focal),
                        sensor_width_mm=float(sensor_width),
                        sensor_height_mm=float(sensor_height) if sensor_height else None,
                    ),
                    extrinsics=_extrinsics_from_world_matrix(world_matrix),
                )
        raise RuntimeError(f"No USD camera found in stage: {source}")

