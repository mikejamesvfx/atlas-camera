"""AtlasGroundPlane: the node, and the pipeline it has to survive.

The core maths lives in tests/test_ground_plane.py. What is defended here is
the part that silently breaks instead of failing loudly: the plane has to APPEND
rather than clobber, it has to survive `AtlasMergeGeometry` as the B side (which
drops anything not tagged `role=PROXY_ROLE`), and it has to reach the viewport
payload and the exporters through the paths they already walk. A node that
builds a perfect quad and then vanishes at the merge looks like a geometry bug
for a long time before anyone suspects the tag.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.comfy.nodes_geometry import AtlasGroundPlane, AtlasMergeGeometry
from atlas_camera.core.ground_plane import GROUND_PLANE_SOURCE, PROXY_ROLE
from atlas_camera.core.primitive_mesh import tessellate_primitive


def _place(solve, **kw):
    return AtlasGroundPlane().place(solve, **kw)


def _grounds(solve):
    return [p for p in solve.projection_scene.proxy_geometry
            if (p.metadata or {}).get("source") == GROUND_PLANE_SOURCE]


class TestItAppendsWithoutDisturbingAnything:

    def test_emits_exactly_one_ground(self, make_atlas_solve):
        out, report = _place(make_atlas_solve())
        assert len(_grounds(out)) == 1
        assert "ground plane" in report

    def test_the_input_solve_is_not_mutated(self, make_atlas_solve):
        solve = make_atlas_solve()
        before = len(solve.projection_scene.proxy_geometry)
        _place(solve)
        assert len(solve.projection_scene.proxy_geometry) == before

    def test_existing_geometry_is_preserved(self, make_atlas_solve):
        from atlas_camera.core.schema import AtlasProxyPrimitive

        solve = make_atlas_solve()
        solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
            name="measured_wall_01", primitive_type="plane",
            metadata={"role": PROXY_ROLE, "source": "depth_derivation"}))
        out, _ = _place(solve)
        names = [p.name for p in out.projection_scene.proxy_geometry]
        assert "measured_wall_01" in names
        assert len(names) == 2

    def test_repeated_placement_auto_suffixes_the_name(self, make_atlas_solve):
        # AtlasMergeGeometry de-duplicates only `projection_backdrop`, so two
        # grounds sharing a name would be indistinguishable downstream.
        out, _ = _place(make_atlas_solve())
        out, _ = _place(out)
        out, _ = _place(out)
        assert [p.name for p in _grounds(out)] == [
            "artist_ground", "artist_ground_02", "artist_ground_03"]

    def test_a_blank_name_falls_back_rather_than_producing_an_unnamed_prim(
            self, make_atlas_solve):
        out, _ = _place(make_atlas_solve(), name="   ")
        assert _grounds(out)[0].name == "artist_ground"


class TestPlacement:

    def test_default_anchor_puts_it_under_the_camera_on_y_zero(self, make_atlas_solve):
        solve = make_atlas_solve()                 # conftest camera at (0, 5, 10)
        out, _ = _place(solve, width_m=10.0, depth_m=10.0)
        verts, _ = tessellate_primitive(_grounds(out)[0])
        assert verts.mean(axis=0) == pytest.approx([0.0, 0.0, 10.0])

    def test_world_origin_anchor_ignores_the_camera(self, make_atlas_solve):
        out, _ = _place(make_atlas_solve(), anchor="world_origin",
                        width_m=10.0, depth_m=10.0)
        verts, _ = tessellate_primitive(_grounds(out)[0])
        assert verts.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])

    def test_widgets_reach_the_geometry(self, make_atlas_solve):
        out, _ = _place(make_atlas_solve(), anchor="world_origin",
                        width_m=25.0, depth_m=8.0,
                        offset_x=2.0, offset_y=-0.75, offset_z=5.0)
        verts, _ = tessellate_primitive(_grounds(out)[0])
        assert verts[:, 0].max() - verts[:, 0].min() == pytest.approx(25.0)
        assert verts[:, 2].max() - verts[:, 2].min() == pytest.approx(8.0)
        assert verts.mean(axis=0) == pytest.approx([2.0, -0.75, 5.0])

    def test_tilt_and_roll_turn_the_primitive_and_not_the_camera(
            self, make_atlas_solve):
        solve = make_atlas_solve()
        before = tuple(map(tuple, solve.camera.extrinsics.camera_world_matrix))
        out, report = _place(solve, tilt_deg=12.0, roll_deg=-8.0)
        after = tuple(map(tuple, out.camera.extrinsics.camera_world_matrix))
        assert after == before, "the world must never rotate to match a ground"

        verts, _ = tessellate_primitive(_grounds(out)[0])
        e1, e2 = verts[1] - verts[0], verts[3] - verts[0]
        n = np.cross(e1, e2)
        n = n / np.linalg.norm(n)
        assert abs(n @ [0.0, 1.0, 0.0]) < 0.99      # genuinely tilted
        assert "world gravity untouched" in report


class TestItSurvivesTheRestOfThePipeline:

    def test_survives_merge_as_the_b_side(self, make_atlas_solve):
        # Merge drops any solve_b primitive whose role != PROXY_ROLE. This is
        # the test that catches a missing role tag, which otherwise presents as
        # "my ground disappeared" three nodes later.
        ground_solve, _ = _place(make_atlas_solve())
        merged, = AtlasMergeGeometry().merge(make_atlas_solve(), ground_solve)
        grounds = _grounds(merged)
        assert len(grounds) == 1
        assert grounds[0].metadata["merged_from"] == "solve_b"

    def test_survives_merge_as_the_a_side(self, make_atlas_solve):
        ground_solve, _ = _place(make_atlas_solve())
        merged, = AtlasMergeGeometry().merge(ground_solve, make_atlas_solve())
        assert len(_grounds(merged)) == 1

    def test_reaches_the_viewport_payload(self, make_atlas_solve):
        from atlas_camera.core.proxy_geometry import serialize_proxy_geometry

        out, _ = _place(make_atlas_solve(), width_m=12.0, depth_m=7.0)
        entries = serialize_proxy_geometry(out.projection_scene)
        mine = [e for e in entries
                if e["metadata"].get("source") == GROUND_PLANE_SOURCE]
        assert len(mine) == 1
        assert mine[0]["type"] == "plane"
        assert len(mine[0]["transform"]) == 16       # flat 4x4 for Matrix4.set
        assert mine[0]["dimensions"] == [12.0, 7.0, 0.0]
        assert mine[0]["metadata"]["provenance"] == "artist_placed"

    def test_tessellates_for_export(self, make_atlas_solve):
        out, _ = _place(make_atlas_solve(), tilt_deg=5.0)
        verts, faces = tessellate_primitive(_grounds(out)[0])
        assert np.isfinite(verts).all()
        assert faces.shape == (2, 3)

    def test_the_solve_still_serializes(self, make_atlas_solve):
        # A manifest failure must never be able to fail an export.
        out, _ = _place(make_atlas_solve(), tilt_deg=3.0, roll_deg=2.0)
        assert "artist_ground" in out.to_json()


class TestTheNodeSurface:

    def test_declared_defaults_match_the_signature(self):
        # Also pinned globally by test_comfy_node_registry, kept local so a
        # drift here fails next to the node it belongs to.
        import inspect

        declared = AtlasGroundPlane.INPUT_TYPES()["optional"]
        sig = inspect.signature(AtlasGroundPlane.place).parameters
        for name, spec in declared.items():
            assert sig[name].default == spec[1]["default"], name

    def test_solve_is_the_only_required_input(self):
        assert set(AtlasGroundPlane.INPUT_TYPES()["required"]) == {"solve"}

    def test_anchor_combo_values_are_the_registered_pair(self):
        # Combo values serialise into saved workflows: append-only, never
        # renamed or reordered.
        assert AtlasGroundPlane.INPUT_TYPES()["optional"]["anchor"][0] == [
            "solve_ground_centre", "world_origin"]

    def test_returns_a_solve_and_a_report(self):
        assert AtlasGroundPlane.RETURN_TYPES == ("ATLAS_SOLVE", "STRING")
        assert AtlasGroundPlane.RETURN_NAMES == ("solve", "report")
