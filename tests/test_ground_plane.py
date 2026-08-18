"""The artist-placed ground plane: frame, placement, and the tags it carries.

Two things here are load-bearing beyond "the maths is right".

The local frame is the THREE.PlaneGeometry one (local X=u, Y=v, Z=normal), NOT
an XZ quad, and it must stay bit-identical to what every derived ground already
emits (``proxy_geometry._build_ground_primitive``). Getting it wrong stands the
plane on its edge and still looks like plausible geometry.

And the world must never rotate. Tilt and roll turn the PRIMITIVE only; world
+Y is the solve's gravity, so a ground that rotated the world would lean every
facade in the scene.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas_camera.core.ground_plane import (
    DEFAULT_NAME,
    GROUND_PLANE_SOURCE,
    PROXY_ROLE,
    build_ground_plane_primitive,
    ground_plane_axes,
    solve_ground_centre,
)
from atlas_camera.core.primitive_mesh import tessellate_primitive


def _axes(tilt=0.0, roll=0.0):
    return [np.asarray(a, dtype=float) for a in ground_plane_axes(tilt, roll)]


class TestTheFrameMatchesEveryOtherGround:

    def test_untilted_frame_is_the_derived_ground_convention(self):
        # proxy_geometry._build_ground_primitive:522-524 verbatim.
        u, v, n = _axes()
        assert u == pytest.approx([1.0, 0.0, 0.0])
        assert v == pytest.approx([0.0, 0.0, -1.0])
        assert n == pytest.approx([0.0, 1.0, 0.0])

    @pytest.mark.parametrize("tilt,roll", [(0, 0), (10, 0), (0, 10), (6, -4),
                                           (-25, 33), (89, 89)])
    def test_frame_stays_orthonormal_and_right_handed(self, tilt, roll):
        u, v, n = _axes(tilt, roll)
        for a in (u, v, n):
            assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-12)
        assert u @ v == pytest.approx(0.0, abs=1e-12)
        assert v @ n == pytest.approx(0.0, abs=1e-12)
        assert n @ u == pytest.approx(0.0, abs=1e-12)
        # u x v = n is what makes the plane face the right way.
        assert np.cross(u, v) == pytest.approx(n, abs=1e-12)

    def test_tilt_turns_the_normal_toward_z_by_exactly_that_angle(self):
        _, _, n = _axes(tilt=12.0)
        assert math.degrees(math.acos(np.clip(n @ [0, 1, 0], -1, 1)) ) \
            == pytest.approx(12.0, abs=1e-9)
        assert n[2] > 0.0 and n[0] == pytest.approx(0.0, abs=1e-12)

    def test_roll_turns_the_normal_toward_x_by_exactly_that_angle(self):
        _, _, n = _axes(roll=12.0)
        assert math.degrees(math.acos(np.clip(n @ [0, 1, 0], -1, 1))) \
            == pytest.approx(12.0, abs=1e-9)
        assert n[0] < 0.0 and n[2] == pytest.approx(0.0, abs=1e-12)

    def test_zero_tilt_and_roll_is_exactly_level(self):
        _, _, n = _axes(0.0, 0.0)
        assert n @ [0, 1, 0] == pytest.approx(1.0, abs=1e-15)


class TestItLandsWhereTheWidgetsSay:

    def test_size_and_centre_survive_tessellation(self):
        prim = build_ground_plane_primitive(
            width_m=20.0, depth_m=30.0, centre=(1.0, 0.0, -2.0))
        verts, faces = tessellate_primitive(prim)
        assert faces.shape == (2, 3)
        assert verts[:, 0].max() - verts[:, 0].min() == pytest.approx(20.0)
        assert verts[:, 2].max() - verts[:, 2].min() == pytest.approx(30.0)
        assert verts.mean(axis=0) == pytest.approx([1.0, 0.0, -2.0])
        assert verts[:, 1] == pytest.approx([0.0] * 4)

    def test_offsets_move_it_including_height(self):
        prim = build_ground_plane_primitive(
            width_m=10.0, depth_m=10.0, centre=(0.0, 0.0, 0.0),
            offset_x=3.0, offset_y=1.5, offset_z=-4.0)
        verts, _ = tessellate_primitive(prim)
        assert verts.mean(axis=0) == pytest.approx([3.0, 1.5, -4.0])

    def test_a_tilted_plane_actually_tilts(self):
        prim = build_ground_plane_primitive(
            width_m=10.0, depth_m=10.0, tilt_deg=15.0)
        verts, _ = tessellate_primitive(prim)
        # Far and near edges must sit at different heights, and the quad must
        # still be planar.
        assert verts[:, 1].max() - verts[:, 1].min() > 1.0
        e1, e2 = verts[1] - verts[0], verts[3] - verts[0]
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        assert abs(n @ (verts[2] - verts[0])) == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_size_is_clamped_not_nan(self):
        prim = build_ground_plane_primitive(width_m=0.0, depth_m=-5.0)
        verts, _ = tessellate_primitive(prim)
        assert np.isfinite(verts).all()
        assert prim.dimensions[0] > 0.0 and prim.dimensions[1] > 0.0

    def test_dimensions_third_component_is_zero_like_every_other_ground(self):
        prim = build_ground_plane_primitive(width_m=4.0, depth_m=6.0)
        assert prim.dimensions == (4.0, 6.0, 0.0)
        assert prim.primitive_type == "plane"


class TestDefaultPlacement:

    def test_centre_defaults_to_under_the_camera_on_y_zero(self, make_atlas_solve):
        solve = make_atlas_solve()          # conftest: camera at (0, 5, 10)
        assert solve_ground_centre(solve) == pytest.approx([0.0, 0.0, 10.0])

    def test_missing_position_falls_back_to_the_origin(self):
        class _NoPos:
            camera = None
        assert solve_ground_centre(_NoPos()) == (0.0, 0.0, 0.0)

    def test_non_finite_position_falls_back_to_the_origin(self, make_atlas_solve):
        solve = make_atlas_solve()
        solve.camera.extrinsics.camera_position = (float("nan"), 0.0, 1.0)
        assert solve_ground_centre(solve) == (0.0, 0.0, 0.0)


class TestItAdmitsWhatItIs:

    def test_carries_the_proxy_role_so_downstream_picks_it_up(self):
        prim = build_ground_plane_primitive(width_m=1.0, depth_m=1.0)
        assert prim.metadata["role"] == PROXY_ROLE
        assert prim.metadata["source"] == GROUND_PLANE_SOURCE
        assert prim.material == "atlas_projection_proxy"
        assert prim.name == DEFAULT_NAME

    def test_is_tagged_inferred_never_measured(self):
        # The whole point: a placed plane must not be promoted to evidence.
        prim = build_ground_plane_primitive(width_m=1.0, depth_m=1.0)
        assert prim.metadata["provenance"] == "artist_placed"
        assert prim.metadata["trust"] == "placeholder"

    def test_records_the_widget_values_it_was_built_from(self):
        prim = build_ground_plane_primitive(
            width_m=12.0, depth_m=8.0, tilt_deg=3.0, roll_deg=-2.0)
        assert prim.metadata["tilt_deg"] == 3.0
        assert prim.metadata["roll_deg"] == -2.0
        assert prim.metadata["width_m"] == 12.0
        assert prim.metadata["depth_m"] == 8.0

    def test_the_solve_is_never_rotated_only_the_primitive(self, make_atlas_solve):
        solve = make_atlas_solve()
        before = tuple(map(tuple, solve.camera.extrinsics.camera_world_matrix))
        prim = build_ground_plane_primitive(
            width_m=5.0, depth_m=5.0, tilt_deg=20.0, roll_deg=15.0,
            centre=solve_ground_centre(solve))
        after = tuple(map(tuple, solve.camera.extrinsics.camera_world_matrix))
        assert after == before
        _, _, n = np.asarray(ground_plane_axes(20.0, 15.0), dtype=float)
        assert n @ [0.0, 1.0, 0.0] < 0.95     # the PRIMITIVE really is tilted
        assert prim.transform_matrix[3] == (0.0, 0.0, 0.0, 1.0)

    def test_primitive_is_json_safe(self):
        import json

        prim = build_ground_plane_primitive(width_m=2.0, depth_m=3.0,
                                            tilt_deg=1.0, roll_deg=1.0)
        json.dumps({"transform": prim.transform_matrix,
                    "dimensions": prim.dimensions,
                    "metadata": prim.metadata})
