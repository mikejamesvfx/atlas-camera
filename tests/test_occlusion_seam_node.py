"""ComfyUI contract for the experimental occlusion-seam refinement node."""

import json
from pathlib import Path

from atlas_camera.comfy.nodes import AtlasRefineOcclusionSeams


def test_occlusion_seam_node_contract_is_append_only_and_blender_free():
    required = AtlasRefineOcclusionSeams.INPUT_TYPES()["required"]
    optional = AtlasRefineOcclusionSeams.INPUT_TYPES()["optional"]

    assert tuple(required) == ("solve", "hole_mask")
    assert tuple(optional) == (
        "layer",
        "seam_width_cells",
        "smooth_iterations",
        "smooth_strength",
        "max_chains",
        "max_layer_depth_rel",
        "min_chain_edges",
        "global_direction",
    )
    assert AtlasRefineOcclusionSeams.RETURN_TYPES == (
        "ATLAS_SOLVE", "MASK", "MASK", "STRING")
    assert AtlasRefineOcclusionSeams.RETURN_NAMES == (
        "solve", "remaining_holes", "created_region", "report")
    assert AtlasRefineOcclusionSeams.FUNCTION == "refine"


def test_occlusion_seam_node_source_does_not_invoke_blender():
    import inspect

    source = inspect.getsource(AtlasRefineOcclusionSeams).lower()
    assert "blender" not in source
    assert "subprocess" not in source


def test_holefill_lab_compares_raw_reconstruction_with_refined_final_solve():
    workflow = Path(__file__).resolve().parents[1] / (
        "examples/local/atlas_holefill_boundary_fill_comparison.json")
    graph = json.loads(workflow.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    refine = next(node for node in nodes.values()
                  if node["type"] == "AtlasRefineOcclusionSeams")
    reconstruct = next(node for node in nodes.values()
                       if node["type"] == "AtlasMaskedSurfaceReconstruct")
    viewport = next(node for node in nodes.values()
                    if node.get("title") ==
                    "[H_seams] viewport — smoothed dual-sheet underlap")

    links = {link[0]: link for link in graph["links"]}
    refine_inputs = {item["name"]: item for item in refine["inputs"]}
    viewport_inputs = {item["name"]: item for item in viewport["inputs"]}
    assert links[refine_inputs["solve"]["link"]][1] == reconstruct["id"]
    assert links[refine_inputs["solve"]["link"]][2] == 0
    assert links[refine_inputs["hole_mask"]["link"]][1] == reconstruct["id"]
    assert links[refine_inputs["hole_mask"]["link"]][2] == 1
    assert links[viewport_inputs["solve"]["link"]][1] == refine["id"]
    assert links[viewport_inputs["patch_mask"]["link"]][1] == refine["id"]
    assert refine["widgets_values"][-1] == "away_from_camera"
