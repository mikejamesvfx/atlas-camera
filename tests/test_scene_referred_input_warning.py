"""Scene-referred IMAGE input (linear EXR / HDR / ACEScct) is named, not silently clipped."""

import logging

import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.node_helpers import (  # noqa: E402
    _image_tensor_to_pil,
    _scene_referred_input_warning,
)
from atlas_camera.comfy.nodes_solve import _with_input_warning  # noqa: E402


def test_display_referred_image_is_quiet():
    assert _scene_referred_input_warning(torch.rand(1, 8, 8, 3)) == ""
    # Resampling overshoot just past 1.0 is not a scene-referred plate.
    assert _scene_referred_input_warning(torch.full((1, 4, 4, 3), 1.005)) == ""


def test_linear_hdr_image_warns():
    img = torch.rand(1, 8, 8, 3)
    img[0, 0, 0] = 6.5          # a linear EXR highlight
    msg = _scene_referred_input_warning(img)
    assert "scene-referred" in msg and "6.500" in msg


def test_negative_values_warn():
    img = torch.rand(1, 8, 8, 3)
    img[0, 1, 1] = -0.2         # out-of-gamut linear / log
    assert "WARNING" in _scene_referred_input_warning(img)


def test_conversion_chokepoint_logs(caplog):
    img = torch.rand(1, 8, 8, 3) * 4.0
    with caplog.at_level(logging.WARNING, logger="atlas_camera"):
        _image_tensor_to_pil(img)
    assert any("scene-referred" in r.message for r in caplog.records)


def test_solve_report_puts_warning_first():
    img = torch.rand(1, 8, 8, 3) * 3.0
    out = _with_input_warning(img, "solve summary")
    assert out.startswith("WARNING") and out.endswith("solve summary")
    assert _with_input_warning(torch.rand(1, 8, 8, 3), "solve summary") == "solve summary"
