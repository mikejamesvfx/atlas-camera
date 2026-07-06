"""Tests for AtlasViewportControls / Atlas Output Desk.

The first output remains the legacy detached-controls link for
AtlasBlockoutViewport. The second output carries OCIO-style output metadata
for exporters and browser preview labels.
"""

import pytest

from atlas_camera.core.schema import AtlasOutputProfile
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasSolve
from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AtlasAttachSourcePlate,
    AtlasBlockoutViewport,
    AtlasRegisterPlate,
    AtlasViewportControls,
)


def test_node_registered_and_return_types():
    assert NODE_CLASS_MAPPINGS["AtlasViewportControls"] is AtlasViewportControls
    assert "AtlasViewportControls" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasViewportControls.RETURN_TYPES == (
        "ATLAS_VIEWPORT_LINK",
        "ATLAS_OUTPUT_PROFILE",
    )
    assert AtlasViewportControls.RETURN_NAMES == ("controls", "output_profile")


def test_output_desk_has_no_required_inputs():
    spec = AtlasViewportControls.INPUT_TYPES()
    assert spec["required"] == {}
    assert "optional" in spec
    assert spec["optional"]["working_colorspace"][0] == "STRING"
    assert spec["optional"]["output_colorspace"][0] == "STRING"


def test_profile_returns_controls_link_and_output_profile():
    controls, profile = AtlasViewportControls().profile()
    assert controls == ""
    assert isinstance(profile, AtlasOutputProfile)
    assert profile.preview_only is True
    assert profile.working_colorspace == "ACEScg"
    assert profile.output_colorspace == "ACES - ACEScg"


def test_viewport_has_optional_controls_input():
    spec = AtlasBlockoutViewport.INPUT_TYPES()
    assert spec["optional"]["controls"][0] == "ATLAS_VIEWPORT_LINK"
    assert spec["optional"]["output_profile"][0] == "ATLAS_OUTPUT_PROFILE"


def test_register_plate_and_attach_source_plate(tmp_path):
    torch = pytest.importorskip("torch")
    image = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
    plate_path = tmp_path / "hero.exr"
    plate_path.write_bytes(b"fake exr header")

    passthrough, plate_ref = AtlasRegisterPlate().register(
        image,
        plate_path=str(plate_path),
        colorspace="ACEScg",
        role="source",
    )

    assert passthrough is image
    assert plate_ref.image_path == str(plate_path)
    assert plate_ref.is_proxy is False
    assert plate_ref.bit_depth == "16f/32f"
    assert plate_ref.metadata["path_exists"] is True
    assert plate_ref.preview_b64.startswith("data:image/jpeg;base64,")

    solve = AtlasSolve(camera=AtlasCamera(intrinsics=build_intrinsics(image_width=8, image_height=8)))
    (attached,) = AtlasAttachSourcePlate().attach(solve, plate_ref)

    assert attached is not solve
    assert attached.source_plate.image_path == str(plate_path)
    assert attached.image_path == str(plate_path)
