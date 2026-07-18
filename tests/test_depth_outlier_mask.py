from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import AtlasDepthOutlierMask


def test_depth_outlier_mask_isolates_spike_and_dilates():
    depth = np.ones((9, 9), dtype=np.float32)
    depth[4, 4] = 8.0
    payload = SimpleNamespace(depth=depth, image_height=9, image_width=9)

    mask, report = AtlasDepthOutlierMask().detect(
        payload, relative_threshold=0.35, mad_threshold=6.0, dilate_px=1
    )

    assert tuple(mask.shape) == (1, 9, 9)
    assert float(mask[0, 4, 4]) == 1.0
    assert int((mask > 0.5).sum()) == 5
    assert "depth outlier mask: 5 px" in report


def test_depth_outlier_mask_does_not_flag_smooth_gradient():
    depth = np.linspace(1.0, 3.0, 81, dtype=np.float32).reshape(9, 9)
    payload = SimpleNamespace(depth=depth, image_height=9, image_width=9)

    mask, report = AtlasDepthOutlierMask().detect(
        payload, relative_threshold=0.35, mad_threshold=6.0, dilate_px=0
    )

    assert int((mask > 0.5).sum()) == 0
    assert report.startswith("depth outlier mask: 0 px")
