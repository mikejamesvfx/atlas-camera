from atlas_camera.core.vanishing_points import (
    horizon_from_vanishing_points,
    intersect_lines,
    line_from_points,
    vanishing_point_from_line_pair,
)


def test_line_intersection_finds_vanishing_point():
    first = line_from_points((0, 0), (10, 10))
    second = line_from_points((0, 10), (10, 0))

    assert intersect_lines(first, second) == (5.0, 5.0)


def test_horizon_from_two_vanishing_points():
    left = vanishing_point_from_line_pair(
        ((0, 0), (10, 10)),
        ((0, 10), (10, 0)),
        direction_label="x",
    )
    right = vanishing_point_from_line_pair(
        ((20, 0), (30, 10)),
        ((20, 10), (30, 0)),
        direction_label="z",
    )

    horizon = horizon_from_vanishing_points(left, right, image_width=40)

    assert horizon.endpoints_px is not None
    assert horizon.line_coefficients[1] != 0

