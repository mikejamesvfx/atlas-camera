import pytest

from atlas_camera.exporters import usd_exporter
from atlas_camera.exporters.usd_exporter import USDExporter
from atlas_camera.importers.usd_camera_loader import USDCameraLoader
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve


def test_usd_camera_loader_missing_file_fails_before_dependency_import(tmp_path):
    with pytest.raises(FileNotFoundError):
        USDCameraLoader().load(tmp_path / "missing.usda")


def test_usd_camera_loader_round_trips_a_placed_camera(tmp_path):
    # Regression test: the loader used to always return AtlasExtrinsics()
    # (identity, camera at the origin) regardless of the USD file's actual
    # camera placement — a silent extrinsics loss that would corrupt any
    # DCC round-trip through camera.usda. This exports a camera placed away
    # from the origin and pitched/yawed (via camera_math.look_at_view_matrix,
    # not an identity pose) and confirms the reloaded camera's position and
    # rotation match the original within floating-point tolerance.
    pytest.importorskip("pxr")
    from atlas_camera.core.camera_math import look_at_view_matrix

    eye = (3.0, 1.6, 5.0)
    target = (0.0, 0.3, -6.0)
    view, world, rotation3 = look_at_view_matrix(eye, target)

    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(image_width=1920, image_height=1080, focal_length_mm=35.0),
            extrinsics=AtlasExtrinsics(
                camera_position=eye,
                camera_rotation_matrix=rotation3,
                camera_world_matrix=world,
                camera_view_matrix=view,
            ),
        )
    )

    usd_path = USDExporter().export_camera(solve, tmp_path / "camera.usda")
    loaded = USDCameraLoader().load(usd_path)

    assert loaded.extrinsics.camera_position != (0.0, 0.0, 0.0)
    for got, want in zip(loaded.extrinsics.camera_position, eye):
        assert got == pytest.approx(want, abs=1e-4)
    for got_row, want_row in zip(loaded.extrinsics.camera_rotation_matrix, rotation3):
        for got, want in zip(got_row, want_row):
            assert got == pytest.approx(want, abs=1e-4)
    for got_row, want_row in zip(loaded.extrinsics.camera_view_matrix, view):
        for got, want in zip(got_row, want_row):
            assert got == pytest.approx(want, abs=1e-4)


def test_usd_exporter_fails_gracefully_when_usd_dependency_missing(monkeypatch, tmp_path):
    def missing_pxr():
        raise RuntimeError("USD export requires the optional usd-core package.")

    monkeypatch.setattr(usd_exporter, "_import_pxr_full", missing_pxr)
    solve = AtlasSolve(
        camera=AtlasCamera(intrinsics=build_intrinsics(image_width=100, image_height=100))
    )

    with pytest.raises(RuntimeError, match="usd-core"):
        USDExporter().export_camera(solve, tmp_path / "camera.usda")

