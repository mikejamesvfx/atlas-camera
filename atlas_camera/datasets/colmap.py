"""COLMAP text-format readers used by ETH3D adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, Matrix3, Matrix4, Point3D


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def fx_px(self) -> float:
        return self.params[0]

    @property
    def fy_px(self) -> float:
        if self.model == "SIMPLE_PINHOLE":
            return self.params[0]
        return self.params[1]

    @property
    def cx_px(self) -> float:
        if self.model == "SIMPLE_PINHOLE":
            return self.params[1]
        return self.params[2]

    @property
    def cy_px(self) -> float:
        if self.model == "SIMPLE_PINHOLE":
            return self.params[2]
        return self.params[3]

    def to_atlas_intrinsics(self) -> AtlasIntrinsics:
        intrinsics = build_intrinsics(
            image_width=self.width,
            image_height=self.height,
            principal_point_px=(self.cx_px, self.cy_px),
            fx_px=self.fx_px,
            fy_px=self.fy_px,
        )
        intrinsics.lens_model = self.model.lower()
        if self.model == "THIN_PRISM_FISHEYE":
            names = ("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
            intrinsics.distortion = {
                name: value
                for name, value in zip(names, self.params[4:])
            }
        return intrinsics

    def intrinsics_hint(self) -> dict[str, object]:
        return {
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "principal_point_px": (self.cx_px, self.cy_px),
        }


@dataclass(frozen=True, slots=True)
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str

    @property
    def world_to_camera_rotation(self) -> Matrix3:
        return quaternion_to_rotation_matrix(self.qw, self.qx, self.qy, self.qz)

    @property
    def camera_center_world(self) -> Point3D:
        rotation = self.world_to_camera_rotation
        translation = (self.tx, self.ty, self.tz)
        return tuple(
            -sum(rotation[row][col] * translation[row] for row in range(3))
            for col in range(3)
        )  # type: ignore[return-value]

    @property
    def camera_to_world_rotation(self) -> Matrix3:
        rotation = self.world_to_camera_rotation
        return tuple(
            tuple(rotation[row][col] for row in range(3))
            for col in range(3)
        )  # type: ignore[return-value]

    def to_atlas_extrinsics(self) -> AtlasExtrinsics:
        camera_rotation = self.camera_to_world_rotation
        camera_position = self.camera_center_world
        return AtlasExtrinsics(
            camera_position=camera_position,
            camera_rotation_matrix=camera_rotation,
            camera_world_matrix=matrix4_from_rotation_translation(camera_rotation, camera_position),
            camera_view_matrix=matrix4_from_rotation_translation(
                self.world_to_camera_rotation,
                (self.tx, self.ty, self.tz),
            ),
            coordinate_system="right_handed",
            up_axis="dataset",
            projection_convention=(
                "COLMAP world-to-camera pose converted to Atlas extrinsics. "
                "COLMAP camera frame is x right, y down, z forward."
            ),
        )


def read_colmap_cameras(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid COLMAP camera line: {line}")
        camera = ColmapCamera(
            camera_id=int(parts[0]),
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(value) for value in parts[4:]),
        )
        cameras[camera.camera_id] = camera
    return cameras


def read_colmap_images(path: str | Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    data_lines = _non_comment_lines(path)
    index = 0
    while index < len(data_lines):
        if not data_lines[index]:
            index += 1
            continue
        parts = data_lines[index].split()
        if len(parts) < 10:
            raise ValueError(f"Invalid COLMAP image pose line: {data_lines[index]}")
        image = ColmapImage(
            image_id=int(parts[0]),
            qw=float(parts[1]),
            qx=float(parts[2]),
            qy=float(parts[3]),
            qz=float(parts[4]),
            tx=float(parts[5]),
            ty=float(parts[6]),
            tz=float(parts[7]),
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
        )
        images[image.image_id] = image
        index += 2
    return images


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> Matrix3:
    norm = sqrt((qw * qw) + (qx * qx) + (qy * qy) + (qz * qz))
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive.")
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return (
        (
            1.0 - (2.0 * ((qy * qy) + (qz * qz))),
            2.0 * ((qx * qy) - (qz * qw)),
            2.0 * ((qx * qz) + (qy * qw)),
        ),
        (
            2.0 * ((qx * qy) + (qz * qw)),
            1.0 - (2.0 * ((qx * qx) + (qz * qz))),
            2.0 * ((qy * qz) - (qx * qw)),
        ),
        (
            2.0 * ((qx * qz) - (qy * qw)),
            2.0 * ((qy * qz) + (qx * qw)),
            1.0 - (2.0 * ((qx * qx) + (qy * qy))),
        ),
    )


def matrix4_from_rotation_translation(rotation: Matrix3, translation: Point3D) -> Matrix4:
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _data_lines(path: str | Path) -> list[str]:
    return [
        line
        for line in _non_comment_lines(path)
        if line
    ]


def _non_comment_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if not line.lstrip().startswith("#")
        ]
