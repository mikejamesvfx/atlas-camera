"""Tests for AtlasExportCameraPathUSD's guard against an un-baked camera_path.

AtlasBlockoutViewport's camera_path output is None whenever the Camera Path
mode hasn't been baked yet (fresh client_data, e.g. right after loading a
workflow) — an OUTPUT_NODE like AtlasExportCameraPathUSD still executes on
every queue regardless, so it must fail with an actionable message instead of
an AttributeError from sample_camera_path indexing into a None path.
"""

import pytest

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AtlasExportCameraPathUSD,
)
from atlas_camera.core.schema import AtlasCameraPath


def test_node_registered_and_return_types():
    assert NODE_CLASS_MAPPINGS["AtlasExportCameraPathUSD"] is AtlasExportCameraPathUSD
    assert "AtlasExportCameraPathUSD" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasExportCameraPathUSD.RETURN_TYPES == ("STRING",)


def test_export_raises_clear_error_when_camera_path_is_none(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    with pytest.raises(ValueError, match="Bake Path"):
        AtlasExportCameraPathUSD().export(solve, None, str(tmp_path))


def test_export_raises_clear_error_when_camera_path_has_no_keyframes(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    empty_path = AtlasCameraPath(keyframes=[], fps=24.0, frame_count=0)
    with pytest.raises(ValueError, match="Bake Path"):
        AtlasExportCameraPathUSD().export(solve, empty_path, str(tmp_path))
