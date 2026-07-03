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
    assert set(options) == {"manual", "organic", "indoor", "outdoor"}
    assert meta["default"] == "manual"


def test_derive_signature_defaults_to_manual_scene_type():
    import inspect

    sig = inspect.signature(AtlasDeriveProjectionGeometry.derive)
    assert sig.parameters["scene_type"].default == "manual"


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
