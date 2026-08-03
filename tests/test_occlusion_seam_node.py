"""ComfyUI contract for the experimental occlusion-seam refinement node."""

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
