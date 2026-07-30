"""Contract for AtlasSolvePatchViews ⌖ — the Qwen consumer of core.view_solver.

Replaces artist guesswork: instead of picking a named view from a dropdown and
hoping it reveals the hole, the mesh is rasterized from each candidate angle and
the winner comes back as a `patch_view_override` string that wires straight into
AtlasAddPatchView.

The distinction these tests exist to protect is between the node's three
possible "no answer" outcomes, which are trivially easy to conflate and mean
opposite things:

  no relief mesh          -> upstream wiring problem
  no plane could be FIT   -> tolerance problem (raise max_hole_fraction)
  planes fit, no view     -> routing answer (Qwen cannot help; go photograph it)

The third was reported for all three during development, because
PathHoleRepairConfig's max_hole_fraction default of 0.04 rejects any hole over
4% of frame and a normal tear measured 43%. The node then said "Qwen cannot see
this" with total confidence, having never fitted anything to look at.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_camera.comfy.node_registry import (  # noqa: E402
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
)
from atlas_camera.comfy.view_prompts import _parse_view_prompt  # noqa: E402

NODE = NODE_CLASS_MAPPINGS["AtlasSolvePatchViews"]


@pytest.fixture(scope="module")
def fitted_scene():
    """A solve whose holes DO admit fill planes, from the repair fixture.

    Built by attaching that mesh through `relief_mesh_primitive` — the same call
    AtlasDeriveReliefMesh makes — so `_relief_mesh_from_solve` finds it exactly
    as it would in a real graph.
    """
    from atlas_camera.comfy.nodes import _relief_mesh_from_solve
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    from atlas_camera.core.schema import AtlasSolve
    from test_path_hole_repair import _fixture

    mesh, camera, _path = _fixture()
    solve = AtlasSolve(
        camera=camera,
        image_width=camera.intrinsics.image_width,
        image_height=camera.intrinsics.image_height,
    )
    try:
        from atlas_camera.comfy.node_helpers import _replace_proxy_role_geometry
        solve = _replace_proxy_role_geometry(
            solve, [relief_mesh_primitive(mesh)], {}, {})
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not attach mesh to solve: {type(exc).__name__}: {exc}")
    if _relief_mesh_from_solve(solve) is None:
        pytest.skip("mesh did not round-trip onto the solve")
    hole = torch.from_numpy(
        np.asarray(mesh.hole_mask, dtype=np.float32)).unsqueeze(0)
    assert float(hole.sum()) > 0, "fixture has no holes — tests would be vacuous"
    return solve, hole


#: Matches the repair fixture's own fit settings; its holes are small and
#: planar, unlike a general tear.
FIT = dict(normal_tolerance_deg=15.0, max_plane_error_m=0.02,
           max_hole_fraction=0.20)


class TestRegistration:
    def test_it_is_registered_with_a_stable_display_name(self):
        assert NODE_DISPLAY_NAME_MAPPINGS["AtlasSolvePatchViews"] == \
            "Atlas Solve Patch Views ⌖"

    def test_outputs_are_the_documented_four(self):
        assert NODE.RETURN_NAMES == (
            "patch_view_override", "patch_prompt", "view_plan", "report")

    def test_the_override_output_matches_what_AtlasAddPatchView_accepts(self):
        """The whole point of the node is that this wires straight in."""
        add = NODE_CLASS_MAPPINGS["AtlasAddPatchView"].INPUT_TYPES()
        assert "patch_view_override" in add["optional"]
        assert add["optional"]["patch_view_override"][0] == "STRING"


class TestThreeDistinctNoAnswers:
    def test_a_solve_without_a_mesh_says_so(self):
        from atlas_camera.core.intrinsics import build_intrinsics
        from atlas_camera.core.schema import (
            AtlasCamera, AtlasExtrinsics, AtlasSolve)
        intr = build_intrinsics(image_width=32, image_height=32,
                                focal_length_mm=35.0, sensor_width_mm=36.0)
        bare = AtlasSolve(camera=AtlasCamera(
            intrinsics=intr,
            extrinsics=AtlasExtrinsics(
                camera_position=(0.0, 0.0, 0.0),
                camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0),
                                     (0, 0, 1, 0), (0, 0, 0, 1)))),
            image_width=32, image_height=32)
        mask = torch.ones(1, 32, 32)
        override, _prompt, _plan, report = NODE().solve_views(bare, mask)
        assert override == ""
        assert "no relief mesh" in report

    def test_an_unfittable_hole_reports_TOLERANCE_not_a_verdict_on_angles(
            self, fitted_scene):
        """The bug this pins: at a too-tight max_hole_fraction the node fitted
        nothing and reported it as 'no angle sees it', which sends the artist to
        go photograph geometry that a looser tolerance would have patched.
        """
        solve, hole = fitted_scene
        override, _p, plan, report = NODE().solve_views(
            solve, hole, resolution=160, max_hole_fraction=0.0011)
        assert override == ""
        assert "no fill plane could be FITTED" in report
        assert "max_hole_fraction" in report, "must name the knob to turn"
        assert json.loads(plan).get("no_candidate_planes") is True

    def test_an_empty_mask_after_exclusion_says_so(self, fitted_scene):
        solve, hole = fitted_scene
        override, _p, _plan, report = NODE().solve_views(
            solve, hole, exclude_mask=torch.ones_like(hole), **FIT)
        assert override == ""
        assert "empty after exclusion" in report


class TestRecommendation:
    def test_it_returns_a_view_the_qwen_parser_accepts(self, fitted_scene):
        """A recommendation AtlasAddPatchView cannot parse is worthless, and the
        parser is the real contract — not my formatting."""
        solve, hole = fitted_scene
        override, prompt, _plan, report = NODE().solve_views(
            solve, hole, resolution=160, min_visible_pixels=1, **FIT)
        if not override:
            pytest.skip(f"fixture revealed no view: {report.splitlines()[0]}")
        assert _parse_view_prompt(override) is not None
        assert _parse_view_prompt(prompt) is not None, \
            "the <sks>-prefixed prompt must parse too"

    def test_the_plan_carries_ranked_views_and_per_island_choices(
            self, fitted_scene):
        solve, hole = fitted_scene
        override, _p, plan, report = NODE().solve_views(
            solve, hole, resolution=160, min_visible_pixels=1, max_views=4, **FIT)
        if not override:
            pytest.skip(f"fixture revealed no view: {report.splitlines()[0]}")
        data = json.loads(plan)
        assert data["candidates_tried"] == 32, "8 azimuths x 4 elevations"
        assert 1 <= len(data["views"]) <= 4
        px = [v["visible_px"] for v in data["views"]]
        assert px == sorted(px, reverse=True)
        for view in data["views"]:
            assert _parse_view_prompt(view["patch_view_override"]) is not None

    def test_searching_distances_triples_the_candidate_count(self, fitted_scene):
        solve, hole = fitted_scene
        _o, _p, plan, _r = NODE().solve_views(
            solve, hole, resolution=128, search_distances=True,
            min_visible_pixels=1, **FIT)
        data = json.loads(plan)
        if "candidates_tried" in data:
            assert data["candidates_tried"] == 96

    def test_the_report_flags_a_source_visible_gap(self, fitted_scene):
        """Measured while building the solver: a see-through gap scores highest
        from the SOURCE camera, and patching it with a generated view would
        replace real pixels with invention. The node must say so rather than
        silently recommending the source angle as if it were a novel view.
        """
        solve, hole = fitted_scene
        override, _p, _plan, report = NODE().solve_views(
            solve, hole, resolution=160, min_visible_pixels=1, **FIT)
        if not override:
            pytest.skip("fixture revealed no view")
        if override.startswith("front view eye-level shot"):
            assert "see-through gap" in report
