import pytest

from atlas_camera.core.solver import solve_from_constraints
from atlas_camera.core.vanishing_points import fit_vanishing_point_from_lines


def _synthetic_constraint_lines():
    return {
        "left": [
            ((0.0, 36.0), (160.0, 12.0)),
            ((0.0, 40.6667), (160.0, 26.0)),
            ((0.0, 44.6667), (160.0, 38.0)),
        ],
        "right": [
            ((0.0, 12.0), (160.0, 36.0)),
            ((0.0, 26.0), (160.0, 40.6667)),
            ((0.0, 38.0), (160.0, 44.6667)),
        ],
    }


def test_fit_vanishing_point_from_artist_lines():
    pytest.importorskip("numpy")

    left_vp = fit_vanishing_point_from_lines(
        _synthetic_constraint_lines()["left"],
        direction_label="left",
    )

    assert round(left_vp.position_px[0]) == -80
    assert round(left_vp.position_px[1]) == 48
    assert len(left_vp.supporting_lines) == 3


def test_solve_from_artist_guided_line_constraints():
    pytest.importorskip("numpy")
    constraints = {
        "image_width": 160,
        "image_height": 96,
        "line_groups": _synthetic_constraint_lines(),
        "camera_height": 1.7,
        "focal_length_mm": 35.0,
        "sensor_width_mm": 36.0,
        "scale_constraints": [
            {
                "reference_id": "door_210cm",
                "image_points": [[80.0, 70.0], [80.0, 20.0]],
            }
        ],
    }

    solve = solve_from_constraints("concept.png", constraints)

    assert solve.source_method == "artist_guided_constraints"
    assert solve.known_intrinsics_used is True
    assert solve.confidence == 0.85
    assert solve.horizon_line is not None
    assert len(solve.vanishing_points) == 2
    assert solve.debug_metadata["constraint_summary"]["left_line_count"] == 3
    assert solve.debug_metadata["camera_estimation"]["focal_source"] == "known_focal_length_hint"
    assert solve.debug_metadata["scale_constraints"]["count"] == 1
    assert solve.debug_metadata["scale_constraints"]["metric_depth_solved"] is False
    assert solve.debug_metadata["scale_constraints"]["reference_ids"] == ["door_210cm"]
    assert solve.landmarks[0]["known_height"] == 2.1
    assert solve.landmarks[0]["metadata"]["reference_id"] == "door_210cm"
    assert solve.projection_scene.landmarks[0]["name"] == "door_210cm"
    assert any(
        primitive.name == "door_210cm_height_guide"
        for primitive in solve.projection_scene.proxy_geometry
    )
    assert solve.to_json()


def test_solve_from_explicit_artist_vanishing_points():
    pytest.importorskip("numpy")
    constraints = {
        "image_size": (160, 96),
        "vanishing_points": {
            "left": (-80.0, 48.0),
            "right": (240.0, 48.0),
        },
    }

    solve = solve_from_constraints("concept.png", constraints)

    assert solve.source_method == "artist_guided_constraints"
    assert solve.debug_metadata["constraint_summary"]["explicit_vanishing_points_used"] is True
    assert solve.horizon_line.endpoints_px == ((0.0, 48.0), (160.0, 48.0))


def test_fallback_focal_is_marked_and_warned():
    pytest.importorskip("numpy")

    solve = solve_from_constraints(
        "concept.png",
        {
            "image_size": (160, 96),
            "vanishing_points": {
                "left": (0.0, 48.0),
                "right": (40.0, 20.0),
            },
        },
        seed=17,
    )

    assert solve.camera.focal_length_inferred is True
    assert solve.camera.confidence.individual_metrics["focal"] < 0.75
    assert solve.camera.seed == 17
    assert solve.debug_metadata["seed"] == 17
    assert solve.debug_metadata["warnings"]
