"""AtlasSAM3Mask 🪄 — native SAM3 concept mask node.

Mirrors tests/test_moge_normals_node.py's pattern: the real SAM3 inference
call (sam3_concept_mask) is mocked here, pinning the node's report
formatting and its gated-repo-vs-raised-error boundary. Live inference
needs real weights + transformers>=5.5.4 + HF auth, so it isn't exercised
here (same "pure logic tested, inference exercised live" split as
test_semantic_segmenter.py / test_sam3_segmenter.py).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_inpaint import AtlasSAM3Mask
from atlas_camera.inference.sam3_segmenter import Sam3GatedRepoError


def _image(h=8, w=8):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def test_segment_reports_matched_concepts_and_coverage(monkeypatch):
    mask_np = np.zeros((8, 8), dtype=bool)
    mask_np[0, :] = True
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask",
        lambda *a, **k: (mask_np, ["sky"], float(mask_np.mean())))

    mask, report = AtlasSAM3Mask().segment(_image(), concepts="sky")

    assert mask.shape == (1, 8, 8)
    assert bool(mask[0, 0, :].all())
    assert "sky" in report and "facebook/sam3" in report


def test_segment_reports_no_match(monkeypatch):
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask",
        lambda *a, **k: (np.zeros((8, 8), dtype=bool), [], 0.0))
    mask, report = AtlasSAM3Mask().segment(_image(), concepts="ghost")
    assert "NO MATCH" in report
    assert not bool(mask.any())


def test_segment_catches_gated_repo_error_into_report(monkeypatch):
    def _boom(*a, **k):
        raise Sam3GatedRepoError("request access at https://huggingface.co/facebook/sam3")
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask", _boom)

    mask, report = AtlasSAM3Mask().segment(_image(), concepts="sky")

    assert not bool(mask.any())                 # empty mask, never raises
    assert "huggingface.co/facebook/sam3" in report


def test_segment_lets_non_gated_errors_propagate(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("transformers>=5.5.4 required, found 4.40.0")
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask", _boom)

    with pytest.raises(RuntimeError, match="transformers"):
        AtlasSAM3Mask().segment(_image(), concepts="sky")
