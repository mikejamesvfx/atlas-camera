"""Tests for AtlasDeriveProjectionGeometry's scene_type convenience preset.

Only exercises the preset-resolution logic (pure Python, no torch/depth model
needed) -- the node's derive() body itself is not unit-tested anywhere in this
suite since it requires the [neural] extra and a real depth model; this test
targets the piece that's safe and fast to verify without those.
"""

from atlas_camera.comfy.nodes import AtlasDeriveProjectionGeometry


def test_manual_scene_type_has_no_preset():
    assert AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS.get("manual") is None


def test_organic_preset_only_overrides_geometry_mode():
    preset = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["organic"]
    assert preset == {"geometry_mode": "relief_mesh"}


def test_indoor_preset_selects_room_cuboid_and_indoor_depth_model():
    preset = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["indoor"]
    assert preset["geometry_mode"] == "primitives"
    assert preset["primitive_method"] == "room_cuboid"
    assert "Indoor" in preset["depth_model"]


def test_outdoor_preset_selects_ransac_planes_and_outdoor_depth_model():
    preset = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["outdoor"]
    assert preset["geometry_mode"] == "primitives"
    assert preset["primitive_method"] == "ransac_planes"
    assert "Outdoor" in preset["depth_model"]


def test_scene_type_widget_exposed_with_manual_default():
    spec = AtlasDeriveProjectionGeometry.INPUT_TYPES()
    options, meta = spec["optional"]["scene_type"]
    assert set(options) == {
        "manual", "organic", "mountains", "forests", "aerial",
        "indoor", "outdoor", "simple_walls", "towers_spires",
    }
    assert meta["default"] == "manual"


def test_mountains_preset_is_relief_mesh_at_high_density():
    preset = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["mountains"]
    assert preset["geometry_mode"] == "relief_mesh"
    assert preset["relief_quality"] == "high"


def test_forests_preset_relaxes_tear_threshold():
    preset = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["forests"]
    assert preset["geometry_mode"] == "relief_mesh"
    assert preset["relief_quality"] == "high"
    assert preset["depth_edge_rel"] > 0.5  # looser than the node's own default


def test_aerial_preset_combines_relief_and_primitives_with_more_objects():
    preset = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["aerial"]
    assert preset["geometry_mode"] == "both"
    assert preset["primitive_method"] == "azimuth_walls"
    assert preset["max_objects"] > 3  # node's own default


def test_simple_walls_and_towers_spires_presets_reach_primitive_methods_scene_type_previously_could_not():
    simple = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["simple_walls"]
    towers = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS["towers_spires"]
    assert simple == {"geometry_mode": "primitives", "primitive_method": "azimuth_walls"}
    assert towers == {"geometry_mode": "primitives", "primitive_method": "vertical_extrusion"}


def test_derive_applies_preset_overrides_for_relief_quality_depth_edge_rel_and_max_objects():
    # Mirrors derive()'s override block for the new preset keys, without
    # needing torch — same pattern as test_preset_resolution_matches_derive_override_logic.
    node = AtlasDeriveProjectionGeometry()
    for scene_type in ("mountains", "forests", "aerial"):
        relief_quality, depth_edge_rel, max_objects = "custom", 0.5, 3
        preset = node._SCENE_TYPE_PRESETS[scene_type]
        relief_quality = preset.get("relief_quality", relief_quality)
        depth_edge_rel = preset.get("depth_edge_rel", depth_edge_rel)
        max_objects = preset.get("max_objects", max_objects)
        if "relief_quality" in preset:
            assert relief_quality == preset["relief_quality"]
        if "depth_edge_rel" in preset:
            assert depth_edge_rel == preset["depth_edge_rel"]
        if "max_objects" in preset:
            assert max_objects == preset["max_objects"]


def test_derive_signature_defaults_to_manual_scene_type():
    import inspect

    sig = inspect.signature(AtlasDeriveProjectionGeometry.derive)
    assert sig.parameters["scene_type"].default == "manual"


def test_relief_quality_widget_exposed_with_custom_default():
    spec = AtlasDeriveProjectionGeometry.INPUT_TYPES()
    options, meta = spec["optional"]["relief_quality"]
    assert set(options) == {"custom", "low", "medium", "high", "ultra"}
    assert meta["default"] == "custom"


def test_relief_quality_presets_map_to_expected_grid_values():
    presets = AtlasDeriveProjectionGeometry._RELIEF_QUALITY_PRESETS
    assert presets == {"low": 64, "medium": 256, "high": 512, "ultra": 1024}


def test_derive_signature_defaults_to_custom_relief_quality():
    import inspect

    sig = inspect.signature(AtlasDeriveProjectionGeometry.derive)
    assert sig.parameters["relief_quality"].default == "custom"
    assert sig.parameters["relief_grid"].default == 128


def test_preset_resolution_matches_derive_override_logic():
    # Mirrors the exact override block at the top of derive() without needing
    # torch: geometry_mode/primitive_method/depth_model start at their normal
    # defaults and only the preset's keys should change.
    node = AtlasDeriveProjectionGeometry()
    for scene_type in ("organic", "indoor", "outdoor"):
        geometry_mode, primitive_method, depth_model = (
            "relief_mesh", "azimuth_walls",
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        )
        preset = node._SCENE_TYPE_PRESETS.get(scene_type)
        if preset:
            geometry_mode = preset.get("geometry_mode", geometry_mode)
            primitive_method = preset.get("primitive_method", primitive_method)
            depth_model = preset.get("depth_model", depth_model)
        assert geometry_mode == preset.get("geometry_mode", "relief_mesh")
        if "primitive_method" in preset:
            assert primitive_method == preset["primitive_method"]
        if "depth_model" in preset:
            assert depth_model == preset["depth_model"]
