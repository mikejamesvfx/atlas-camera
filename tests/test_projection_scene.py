from atlas_camera.core.projection_scene import add_axis_guides, create_default_projection_scene, proxy_box


def test_projection_scene_defaults_to_y_up():
    scene = create_default_projection_scene()

    assert scene.up_axis == "Y"
    assert scene.coordinate_system == "right_handed"
    assert scene.proxy_geometry[0].name == "ground_plane"
    assert scene.proxy_geometry[0].metadata["up_axis"] == "Y"


def test_projection_scene_can_add_proxy_geometry_and_guides():
    scene = create_default_projection_scene()
    scene.proxy_geometry.append(proxy_box("building_block"))
    add_axis_guides(scene)

    names = {primitive.name for primitive in scene.proxy_geometry}
    assert "building_block" in names
    assert "y_axis_guide" in names

