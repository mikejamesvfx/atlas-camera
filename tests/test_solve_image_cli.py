import subprocess
import sys


def test_solve_image_cli_builds_review_package(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    output_dir = tmp_path / "packages"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/solve_image.py",
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--package-name",
            "atlas_cli_review",
            "--no-usd",
            "--focal-length-mm",
            "35",
            "--canny-low",
            "30",
            "--canny-high",
            "100",
            "--hough-threshold",
            "14",
            "--min-line-length",
            "24",
            "--max-line-gap",
            "6",
            "--ransac-iterations",
            "800",
            "--ransac-threshold",
            "5.0",
            "--random-seed",
            "7",
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    package_dir = output_dir / "atlas_cli_review"
    report = (package_dir / "report.md").read_text(encoding="utf-8")

    assert "atlas_cli_review" in completed.stdout
    assert "vanishing_points: " in completed.stdout
    assert (package_dir / "source_image.png").is_file()
    assert (package_dir / "debug_overlay.png").is_file()
    assert (package_dir / "atlas_solve.json").is_file()
    assert (package_dir / "maya_open_scene.py").is_file()
    assert (package_dir / "report.md").is_file()
    assert "Vanishing points:" in report
    assert "Horizon angle:" in report
    assert "Focal source: known_focal_length_hint" in report
