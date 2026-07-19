"""AtlasMogeNormals — the decoupled predicted-normal pass.

Runs a MoGe `*-normal` model purely for normals and attaches them onto an
existing depth map (any depth model), so V2/DA3 depth + MoGe normals can be
combined. MoGe itself is mocked here — the tests pin the orchestration:
resize-to-input-resolution, renormalization, non-mutation of the input, and the
no-normals pass-through.
"""

import numpy as np
import pytest

from atlas_camera.inference.depth_estimator import DepthResult


def _depth(h, w, normal=None):
    return DepthResult(
        depth=np.ones((h, w), dtype=np.float32),
        is_metric=True, model_id="test", image_width=w, image_height=h,
        normal=normal)


def _patch_moge(monkeypatch, moge_result):
    # AtlasMogeNormals lives in nodes_depth after modularization; its helper
    # lookups (_save_image_tensor_to_tmp, os) resolve there, so patch that module.
    import atlas_camera.comfy.nodes_depth as nodes
    monkeypatch.setattr(nodes, "_save_image_tensor_to_tmp", lambda img: "dummy.png")
    monkeypatch.setattr(nodes.os, "unlink", lambda p: None)
    monkeypatch.setattr(
        "atlas_camera.inference.depth_estimator.estimate_depth",
        lambda *a, **k: moge_result)
    return nodes


def test_attach_resizes_to_input_depth_and_renormalizes(monkeypatch):
    # input depth 64x48 (no normals); MoGe returns normals at a DIFFERENT res
    in_depth = _depth(64, 48)
    moge_n = np.zeros((32, 24, 3), dtype=np.float32)
    moge_n[..., 2] = 1.0                                   # +Z, unit
    nodes = _patch_moge(monkeypatch, _depth(32, 24, normal=moge_n))

    out, report = nodes.AtlasMogeNormals().attach(in_depth, image=object())

    assert out.normal.shape == (64, 48, 3)                 # resized to the input depth
    lens = np.linalg.norm(out.normal, axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-5)               # still unit vectors
    assert in_depth.normal is None                         # input NOT mutated
    assert out.depth is in_depth.depth                     # depth passed through (shared)
    assert "attached" in report.lower()


def test_matching_resolution_is_a_noop_resize(monkeypatch):
    in_depth = _depth(40, 40)
    moge_n = np.zeros((40, 40, 3), dtype=np.float32)
    moge_n[..., 1] = 1.0
    nodes = _patch_moge(monkeypatch, _depth(40, 40, normal=moge_n))
    out, _ = nodes.AtlasMogeNormals().attach(in_depth, image=object())
    assert out.normal.shape == (40, 40, 3)
    assert np.allclose(out.normal[..., 1], 1.0, atol=1e-6)


def test_pass_through_when_model_returns_no_normals(monkeypatch):
    in_depth = _depth(16, 16)
    nodes = _patch_moge(monkeypatch, _depth(16, 16, normal=None))
    out, report = nodes.AtlasMogeNormals().attach(in_depth, image=object())
    assert out is in_depth                                  # unchanged pass-through
    assert "no normals" in report.lower()


def test_resize_normal_field_direct():
    from atlas_camera.comfy.nodes import _resize_normal_field
    n = np.zeros((8, 8, 3), dtype=np.float32)
    n[..., 0] = 1.0
    out = _resize_normal_field(n, (20, 12))
    assert out.shape == (20, 12, 3)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-5)
