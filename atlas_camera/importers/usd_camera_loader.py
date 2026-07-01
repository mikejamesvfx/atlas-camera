"""USD camera loader boundary."""

from __future__ import annotations

from pathlib import Path

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics


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
                return AtlasCamera(
                    name=prim.GetName() or "usd_camera",
                    intrinsics=build_intrinsics(
                        image_width=image_size[0],
                        image_height=image_size[1],
                        focal_length_mm=float(focal),
                        sensor_width_mm=float(sensor_width),
                        sensor_height_mm=float(sensor_height) if sensor_height else None,
                    ),
                    extrinsics=AtlasExtrinsics(),
                )
        raise RuntimeError(f"No USD camera found in stage: {source}")

