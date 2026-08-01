"""The camera seam: one CameraSpec, one principal-point fallback.

The fallback `cx_px if cx_px is not None else width / 2.0` was written by hand at
27 sites across 20 files before this module existed, while the one bundle that
did it correctly sat inside `relief_mesh.py` where nothing outside relief-mesh
code discovered it. These tests pin the fallback ladder so the 27 sites can be
routed through a single interface without changing what any of them computed.
"""
from types import SimpleNamespace

import pytest

from atlas_camera.core.camera_spec import CameraSpec
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasHorizon,
    AtlasIntrinsics,
    AtlasSolve,
)


def _solve(**intrinsics_kwargs):
    intr_kwargs = {"image_width": 1920, "image_height": 1080, "fx_px": 1600.0}
    intr_kwargs.update(intrinsics_kwargs)
    view = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -5.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return AtlasSolve(
        camera=AtlasCamera(
            intrinsics=AtlasIntrinsics(**intr_kwargs),
            extrinsics=AtlasExtrinsics(camera_view_matrix=view),
        ),
        image_width=intr_kwargs["image_width"],
        image_height=intr_kwargs["image_height"],
    )


def test_recorded_pixel_centre_wins():
    spec = CameraSpec.from_solve(_solve(cx_px=900.0, cy_px=500.0))
    assert (spec.cx, spec.cy) == (900.0, 500.0)


def test_principal_point_is_used_when_the_pixel_centre_is_missing():
    """A solve loaded from JSON can carry principal_point_px without cx_px.

    Falling straight to the image centre there discards a measured value, which
    is why AtlasRetopologizeLayer already consulted principal_point_px by hand.
    """
    spec = CameraSpec.from_solve(_solve(principal_point_px=(880.0, 470.0)))
    assert (spec.cx, spec.cy) == (880.0, 470.0)


def test_image_centre_is_the_last_resort():
    spec = CameraSpec.from_solve(_solve())
    assert (spec.cx, spec.cy) == (960.0, 540.0)


def test_fy_falls_back_to_fx():
    spec = CameraSpec.from_solve(_solve(fx_px=1600.0, fy_px=None))
    assert spec.fy == 1600.0


def test_view_matrix_and_image_size_come_along():
    spec = CameraSpec.from_solve(_solve())
    assert spec.width == 1920 and spec.height == 1080
    assert spec.view_matrix[1][3] == -5.0


def test_has_focal_is_false_without_a_usable_focal_length():
    assert CameraSpec.from_solve(_solve(fx_px=None)).has_focal is False
    assert CameraSpec.from_solve(_solve(fx_px=0.0)).has_focal is False
    assert CameraSpec.from_solve(_solve(fx_px=1600.0)).has_focal is True


def test_fallback_dimensions_fill_in_for_intrinsics_that_carry_none():
    """The ATLAS_DEPTH_MAP path: intrinsics may have no image size, and the
    depth estimate's own resolution stands in."""
    spec = CameraSpec.from_solve(
        _solve(image_width=0, image_height=0), width=640, height=480
    )
    assert (spec.width, spec.height) == (640, 480)
    assert (spec.cx, spec.cy) == (320.0, 240.0)


def test_horizon_row_is_read_off_the_solve():
    solve = _solve()
    solve.horizon_line = AtlasHorizon(
        line_coefficients=(0.0, 1.0, -600.0),
        endpoints_px=((0.0, 600.0), (1920.0, 620.0)),
    )
    assert CameraSpec.from_solve(solve).horizon_y == 610.0


def test_horizon_row_is_none_without_a_horizon_line():
    assert CameraSpec.from_solve(_solve()).horizon_y is None


def test_as_params_matches_the_hand_rolled_tuple_it_replaces():
    """`_solve_camera_params` is the 15-site idiom in comfy/. The spec must
    produce the same tuple in the same order, or routing those sites through it
    would silently reorder fx/fy/cx/cy."""
    from atlas_camera.core.depth_geometry import _solve_camera_params

    solve = _solve(cx_px=900.0, cy_px=500.0, fy_px=1580.0)
    depth = SimpleNamespace(image_width=640, image_height=480)
    spec = CameraSpec.from_solve(
        solve, width=depth.image_width, height=depth.image_height
    )
    assert spec.as_params() == _solve_camera_params(solve, depth)


def test_scale_defaults_to_one_and_is_carried():
    assert CameraSpec.from_solve(_solve()).scale == 1.0
    assert CameraSpec.from_solve(_solve(), scale=2.5).scale == 2.5


def test_relief_mesh_keeps_its_name_for_the_same_bundle():
    """`ReliefMeshCameraSpec` is imported by depth_completion, move_budget and
    a test; it must keep resolving, to the same class."""
    from atlas_camera.core.relief_mesh import ReliefMeshCameraSpec

    assert ReliefMeshCameraSpec is CameraSpec


def test_the_bundle_is_still_constructible_by_keyword_without_image_size():
    """test_relief_mesh.py builds one with no width/height. Adding those fields
    must not make them required."""
    spec = CameraSpec(
        view_matrix=None, fx=200.0, fy=200.0, cx=20.0, cy=20.0,
        scale=1.0, horizon_y=18.0,
    )
    assert spec.fx == 200.0 and spec.horizon_y == 18.0


def test_for_image_makes_the_given_size_authoritative():
    """Some callers hold an array whose resolution differs from the recorded
    image, and fall back to the centre of THAT array — not of the intrinsics'
    image. `from_solve` fills in a missing size; `for_image` overrides it.
    """
    intr = AtlasIntrinsics(image_width=1920, image_height=1080, fx_px=1600.0)
    spec = CameraSpec.for_image(intr, width=640, height=480)
    assert (spec.width, spec.height) == (640, 480)
    assert (spec.cx, spec.cy) == (320.0, 240.0)


def test_for_image_still_prefers_a_recorded_pixel_centre():
    intr = AtlasIntrinsics(image_width=1920, image_height=1080,
                           fx_px=1600.0, cx_px=900.0, cy_px=500.0)
    spec = CameraSpec.for_image(intr, width=640, height=480)
    assert (spec.cx, spec.cy) == (900.0, 500.0)


def test_as_params_refuses_when_the_image_size_is_unknown():
    spec = CameraSpec(view_matrix=None, fx=200.0, fy=200.0, cx=20.0, cy=20.0)
    with pytest.raises(ValueError, match="image size"):
        spec.as_params()
