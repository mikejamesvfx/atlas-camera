from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasSolve
from atlas_camera.exporters.review_package import build_review_package


def test_review_package_folder_can_be_created(tmp_path):
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"not a real png, only copied")
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=build_intrinsics(
                image_width=1280,
                image_height=720,
                focal_length_mm=35.0,
            )
        ),
        image_path=str(image_path),
        image_width=1280,
        image_height=720,
        source_method="test",
    )

    result = build_review_package(solve, tmp_path, include_usd=True)

    assert (result.package_dir / "atlas_solve.json").is_file()
    assert (result.package_dir / "maya_open_scene.py").is_file()
    assert (result.package_dir / "open_atlas_review_001.mel").is_file()
    assert (result.package_dir / "report.md").is_file()
    assert (result.package_dir / "source_image.png").is_file()
    if result.warnings:
        assert "usd-core" in result.warnings[0]
    else:
        assert (result.package_dir / "camera.usda").is_file()
        assert (result.package_dir / "proxy_scene.usda").is_file()
        assert (result.package_dir / "projection_scene.usda").is_file()

