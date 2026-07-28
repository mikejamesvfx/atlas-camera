"""Node-level tests for AtlasLayerPlan — the manifest node and its VLM handoff.

The live finding that motivates the override inputs: on a real plate the
fitter-id placeholder concepts collapsed three distinct objects into the one
word "box", which SAM3 cannot use to separate anything. The VLM supplies
words, SAM3 supplies pixels — so the node must accept named concepts from
AtlasAssessImage's sam_prompt outputs and must record which kind it emitted.
"""

import numpy as np

from atlas_camera.comfy.nodes import AtlasLayerPlan, AtlasOcclusionGraph
from atlas_camera.core.occlusion_graph import (
    POLICY_EXTEND_PLANE,
    AtlasOcclusionGraph as GraphData,
    OcclusionEdge,
    OcclusionNode,
    attach_occlusion_graph,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve


def _solve_with_graph():
    intr = build_intrinsics(image_width=128, image_height=96, focal_length_mm=35.0)
    solve = AtlasSolve(camera=AtlasCamera(intrinsics=intr,
                                          extrinsics=AtlasExtrinsics()),
                       image_width=128, image_height=96)
    graph = GraphData(
        nodes=[
            OcclusionNode(id="projection_box_01", kind="object",
                          completion_policy=POLICY_EXTEND_PLANE,
                          depth_range_m=(3.0, 4.0)),
            OcclusionNode(id="projection_wall_01", kind="surface",
                          completion_policy=POLICY_EXTEND_PLANE,
                          depth_range_m=(8.0, 9.0)),
        ],
        edges=[OcclusionEdge(occluder="projection_box_01",
                             occludee="projection_wall_01")],
    )
    attach_occlusion_graph(solve, graph)
    return solve


def test_placeholder_concepts_are_flagged_as_such():
    solve, report, fg, bg = AtlasLayerPlan().plan(_solve_with_graph())
    assert fg == "box" and bg == "wall"
    assert "placeholders" in report
    meta = solve.semantics.value["layer_plan_concepts"]
    assert meta["source"] == "fitter_id_placeholder"


def test_vlm_override_replaces_placeholders_and_is_recorded():
    solve, report, fg, bg = AtlasLayerPlan().plan(
        _solve_with_graph(),
        foreground_concepts_override="red phone box, bollard",
        background_concepts_override="brick terrace facade",
    )
    assert fg == "red phone box, bollard"
    assert bg == "brick terrace facade"
    assert "VLM-named" in report
    meta = solve.semantics.value["layer_plan_concepts"]
    assert meta["source"] == "vlm_override"
    assert meta["foreground"] == "red phone box, bollard"


def test_no_graph_reports_rather_than_planning_nothing_silently():
    intr = build_intrinsics(image_width=64, image_height=48, focal_length_mm=35.0)
    solve = AtlasSolve(camera=AtlasCamera(intrinsics=intr,
                                          extrinsics=AtlasExtrinsics()),
                       image_width=64, image_height=48)
    _, report, fg, bg = AtlasLayerPlan().plan(solve)
    assert "No occlusion graph" in report
    assert fg == "" and bg == ""


def test_override_inputs_are_appended_last_and_are_link_sockets():
    """Saved-workflow contract: new widgets append; overrides are forceInput
    STRING sockets (the sanctioned pattern, since STRING->combo links are
    rejected by the backend but STRING->STRING sockets are fine)."""
    opt = AtlasLayerPlan.INPUT_TYPES()["optional"]
    keys = list(opt.keys())
    assert keys[-2:] == ["foreground_concepts_override",
                         "background_concepts_override"]
    for k in keys[-2:]:
        assert opt[k][1].get("forceInput") is True
