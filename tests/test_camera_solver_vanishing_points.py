from atlas_camera.core.solver import solve_still_image
from atlas_camera.core.vanishing_points import VanishingPointDetector, draw_debug_overlay


DETECTION_OPTIONS = {
    "canny_low": 30,
    "canny_high": 100,
    "hough_threshold": 14,
    "min_line_length": 24,
    "max_line_gap": 6,
    "ransac_iterations": 800,
    "ransac_threshold": 5.0,
    "random_seed": 7,
}


def test_detector_finds_two_vps_from_synthetic_fixture(synthetic_perspective_image):
    _, image = synthetic_perspective_image

    result = VanishingPointDetector.detect_vanishing_points(image, **DETECTION_OPTIONS)

    assert result["num_lines_total"] >= 4
    assert result["vp1"] is not None
    assert result["vp2"] is not None


def test_solve_still_image_writes_debug_overlay(synthetic_perspective_image, tmp_path):
    path, image = synthetic_perspective_image
    overlay_path = tmp_path / "debug_overlay.png"

    solve = solve_still_image(
        path,
        detect_vanishing_points=True,
        debug_overlay_path=overlay_path,
        detection_options=DETECTION_OPTIONS,
    )

    assert solve.source_method == "automatic_still_image_vanishing_points"
    assert solve.horizon_line is not None
    assert len(solve.vanishing_points) >= 2
    assert overlay_path.is_file()
    assert "camera_estimation" in solve.debug_metadata
    assert solve.to_json()

    overlay = draw_debug_overlay(image, {"lines": [], "num_lines_total": 0})
    assert overlay.shape == image.shape

