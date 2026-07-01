import pytest

from atlas_camera.exporters import usd_exporter
from atlas_camera.exporters.usd_exporter import USDExporter
from atlas_camera.importers.usd_camera_loader import USDCameraLoader
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasSolve


def test_usd_camera_loader_missing_file_fails_before_dependency_import(tmp_path):
    with pytest.raises(FileNotFoundError):
        USDCameraLoader().load(tmp_path / "missing.usda")


def test_usd_exporter_fails_gracefully_when_usd_dependency_missing(monkeypatch, tmp_path):
    def missing_pxr():
        raise RuntimeError("USD export requires the optional usd-core package.")

    monkeypatch.setattr(usd_exporter, "_import_pxr_full", missing_pxr)
    solve = AtlasSolve(
        camera=AtlasCamera(intrinsics=build_intrinsics(image_width=100, image_height=100))
    )

    with pytest.raises(RuntimeError, match="usd-core"):
        USDExporter().export_camera(solve, tmp_path / "camera.usda")

