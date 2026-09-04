"""Tests for AtlasExportCameraPathUSD's handling of an un-baked camera_path.

AtlasBlockoutViewport's camera_path output is None whenever the Camera Path
mode hasn't been baked yet (fresh client_data, e.g. right after loading a
workflow) — and an OUTPUT_NODE like AtlasExportCameraPathUSD still executes on
every queue regardless.

This USED to raise. It no longer does, deliberately: the path is authored in
the BROWSER, so on the queue that opens the viewport there is nothing to export
yet — and with `auto_preset` that is EVERY first queue, because Python runs
before Three.js has baked. Raising aborted the whole prompt (ComfyUI stops at
the first failing node), taking the video and the approval stills with it, on a
run that was only ever going to produce a path for the NEXT one. The bake
re-queues automatically, so the second pass writes the USD.

What is still pinned is the SHAPE of that no-op, because both halves are
load-bearing: an EMPTY string, never the explanation, since this output is a
file path and downstream nodes read it as one — a sentence here would be a lie
in the shape of a location. The reason goes to the log instead, so a run that
exported nothing still says why.
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


@pytest.mark.parametrize("path", [
    pytest.param(None, id="never_baked"),
    pytest.param(AtlasCameraPath(keyframes=[], fps=24.0, frame_count=0),
                 id="baked_empty"),
])
def test_unbaked_path_is_a_no_op_not_a_failure(tmp_path, make_atlas_solve, capsys,
                                               path):
    solve = make_atlas_solve()

    (out,) = AtlasExportCameraPathUSD().export(solve, path, str(tmp_path))

    # An empty PATH, not a message: downstream reads this as a location.
    assert out == ""
    # Nothing written, and no directory conjured for a file that never came.
    assert not list(tmp_path.iterdir())
    # The reason is not swallowed -- it goes to the log.
    logged = capsys.readouterr().out
    assert "no camera path yet" in logged
    assert "re-queues" in logged


def test_a_real_path_still_writes_the_usd(tmp_path, make_atlas_solve):
    """The no-op must not have swallowed the actual job."""
    pytest.importorskip("pxr")
    from atlas_camera.core.schema import AtlasCameraKeyframe

    solve = make_atlas_solve()
    keys = [AtlasCameraKeyframe(frame_index=i,
                                position=(float(i), 1.6, 0.0),
                                target=(float(i), 1.6, -10.0),
                                up=(0.0, 1.0, 0.0))
            for i in range(3)]
    baked = AtlasCameraPath(keyframes=keys, fps=24.0, frame_count=3)

    (out,) = AtlasExportCameraPathUSD().export(solve, baked, str(tmp_path))

    assert out.endswith("camera_path.usda")
    assert (tmp_path / "camera_path.usda").is_file()
