"""Tests for AtlasViewportControls — the detached toolbar/panel companion to
AtlasBlockoutViewport. It carries no real computation; the graph link to
AtlasBlockoutViewport's `controls` input exists only so the two nodes'
frontend JS extensions can find each other (see atlas_blockout.js). These
tests only cover what's checkable from Python: registration and the
placeholder output/input shapes.
"""

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AtlasBlockoutViewport,
    AtlasViewportControls,
)


def test_node_registered_and_return_types():
    assert NODE_CLASS_MAPPINGS["AtlasViewportControls"] is AtlasViewportControls
    assert "AtlasViewportControls" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasViewportControls.RETURN_TYPES == ("ATLAS_VIEWPORT_LINK",)


def test_no_required_inputs():
    assert AtlasViewportControls.INPUT_TYPES() == {"required": {}}


def test_noop_returns_placeholder():
    assert AtlasViewportControls().noop() == ("",)


def test_viewport_has_optional_controls_input():
    spec = AtlasBlockoutViewport.INPUT_TYPES()
    assert spec["optional"]["controls"][0] == "ATLAS_VIEWPORT_LINK"
