import json
import subprocess
import sys


def test_solve_constraints_cli_builds_guided_review_package(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    output_dir = tmp_path / "packages"
    constraints_path = tmp_path / "constraints.json"
    constraints = {
        "line_groups": {
            "left": [
                [[0.0, 36.0], [160.0, 12.0]],
                [[0.0, 40.6667], [160.0, 26.0]],
                [[0.0, 44.6667], [160.0, 38.0]],
            ],
            "right": [
                [[0.0, 12.0], [160.0, 36.0]],
                [[0.0, 26.0], [160.0, 40.6667]],
                [[0.0, 38.0], [160.0, 44.6667]],
            ],
        },
        "camera_height": 1.7,
        "scale_constraints": [
            {
                "reference_id": "person_175cm",
                "image_points": [[80.0, 70.0], [80.0, 20.0]],
            }
        ],
    }
    constraints_path.write_text(json.dumps(constraints), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/solve_constraints.py",
            "--image",
            str(image_path),
            "--constraints",
            str(constraints_path),
            "--output-dir",
            str(output_dir),
            "--package-name",
            "atlas_guided_cli_review",
            "--focal-length-mm",
            "35",
            "--sensor-width-mm",
            "36",
            "--no-usd",
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    package_dir = output_dir / "atlas_guided_cli_review"
    report = (package_dir / "report.md").read_text(encoding="utf-8")
    solve_json = (package_dir / "atlas_solve.json").read_text(encoding="utf-8")

    assert "solve: artist_guided_constraints" in completed.stdout
    assert "left_lines: 3" in completed.stdout
    assert "right_lines: 3" in completed.stdout
    assert (package_dir / "source_image.png").is_file()
    assert (package_dir / "debug_overlay.png").is_file()
    assert (package_dir / "atlas_solve.json").is_file()
    assert (package_dir / "maya_open_scene.py").is_file()
    assert "Source method: artist_guided_constraints" in report
    assert "Focal source: known_focal_length_hint" in report
    assert "Scale references: 1" in report
    assert '"source_method": "artist_guided_constraints"' in solve_json
    assert '"person_175cm_height_guide"' in solve_json
    assert '"reference_id": "person_175cm"' in solve_json
    assert '"metric_depth_solved": false' in solve_json
