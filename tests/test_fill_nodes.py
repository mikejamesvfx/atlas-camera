"""In-graph two-pass nodes: the gate's self-fallback and the composite stack.

The gate's contract is the load-bearing part: on FAILURE it outputs the
GUIDE, so a downstream texture pass re-touches nothing and the composite
degrades to a no-op — the in-graph equivalent of the CLI engine's fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_fill import (
    AtlasInterpassGate,
    AtlasMembraneComposite,
)


def _img(rgb):
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _plate(h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    lum = rng.integers(90, 140, size=(h, w)).astype(np.int16)
    lum[:, ::8] = lum[:, ::8] // 2 + 60
    return np.clip(np.stack([lum + 4, lum, lum - 4], -1), 0, 255).astype(np.uint8)


def _hole_mask(h=96, w=96):
    m = np.zeros((h, w), np.float32)
    m[32:64, 32:64] = 1.0
    return torch.from_numpy(m)


def test_gate_passes_a_real_fill_and_returns_it():
    guide = _plate()
    rng = np.random.default_rng(1)
    fill = guide.copy()
    fill[32:64, 32:64] = rng.integers(60, 200, size=(32, 32, 3))
    out, ok, report = AtlasInterpassGate().gate(_img(fill), _img(guide),
                                                _hole_mask())
    assert ok and "PASS" in report
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.abs(got.astype(int) - fill.astype(int)).mean() < 1.0


def test_gate_fails_a_smear_and_passes_the_guide_through():
    from atlas_camera.dynamic.fill_metrics import edge_extend

    guide = _plate()
    hole = np.zeros((96, 96), bool)
    hole[32:64, 32:64] = True
    smear = edge_extend(guide, hole)
    out, ok, report = AtlasInterpassGate().gate(_img(smear), _img(guide),
                                                _hole_mask())
    assert not ok and "FAIL" in report
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.abs(got.astype(int) - guide.astype(int)).mean() < 1.0
    assert "guide passed through" in report


def test_gate_resizes_a_low_raster_fill_before_scoring():
    """The WAN branch generates at 720p-class; the gate must score at the
    reference raster, not crash on the mismatch."""
    from PIL import Image as PILImage

    guide = _plate(128, 128, seed=2)
    rng = np.random.default_rng(3)
    fill_small = np.array(PILImage.fromarray(guide).resize((64, 64)))
    fill_small[16:32, 16:32] = rng.integers(60, 200, size=(16, 16, 3))
    m = np.zeros((128, 128), np.float32)
    m[32:64, 32:64] = 1.0
    out, ok, report = AtlasInterpassGate().gate(
        _img(fill_small), _img(guide), torch.from_numpy(m))
    assert out.shape[1:3] == (128, 128)


def test_gate_empty_hole_is_a_fail_with_guide_passthrough():
    guide = _plate()
    out, ok, report = AtlasInterpassGate().gate(
        _img(guide), _img(guide), torch.zeros(96, 96))
    assert not ok and "empty hole" in report


def test_membrane_composite_erases_an_offset_and_pastes_only_the_hole():
    ref = _plate(seed=4)
    hole = np.zeros((96, 96), bool)
    hole[32:64, 32:64] = True
    fill = ref.copy()
    fill[hole] = np.clip(ref[hole].astype(int) - 30, 0, 255).astype(np.uint8)
    out, report = AtlasMembraneComposite().composite(
        _img(fill), _img(ref), _hole_mask())
    got = (out[0].numpy() * 255).astype(np.uint8)
    # membrane recovers the offset inside the hole
    assert np.abs(got[hole].astype(int) - ref[hole].astype(int)).mean() < 4.0
    # outside the hole the reference is untouched
    assert np.array_equal(got[~hole], ref[~hole])
    assert "membrane applied" in report


def test_membrane_composite_empty_hole_returns_reference():
    ref = _plate(seed=5)
    out, report = AtlasMembraneComposite().composite(
        _img(np.zeros_like(ref)), _img(ref), torch.zeros(96, 96))
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.array_equal(got, ref)
    assert "empty hole" in report


def test_path_frame_index_computes_window_and_last():
    from atlas_camera.comfy.nodes_fill import AtlasPathFrameIndex
    from atlas_camera.core.camera_path import AtlasCameraPath

    # no path: solved-pose single frame
    count, last, start, report = AtlasPathFrameIndex().index(None)
    assert (count, last, start) == (1, 0, 0)
    assert "solved pose" in report

    # a real 30-frame path must agree with the guide's sampler
    from atlas_camera.core.camera_path import sample_camera_path
    path = AtlasCameraPath(frame_count=30, keyframes=[])
    n = len(sample_camera_path(path))
    count, last, start, report = AtlasPathFrameIndex().index(path, window=5)
    assert count == max(n, 1) or (n == 0 and count == 1)
    if n:
        assert last == n - 1 and start == max(0, n - 5)

    # non-4k+1 window is warned, not refused
    if n:
        *_ignore, report = AtlasPathFrameIndex().index(path, window=6)
        assert "4k+1" in report


# ------------------------------------------------------ crop-out / paste-back

def _wall_solve_with_region():
    """Camera at origin looking -Z; textured wall at z=-10; one artist
    ▦ Fill ROI region stored the way AtlasBlockoutViewport writes it."""
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import (
        AtlasExtrinsics,
        AtlasProjectionScene,
        AtlasProxyPrimitive,
        AtlasSolve,
        LatentCamera,
    )

    view, world, rot3 = look_at_view_matrix((0.0, 1.6, 0.0),
                                            (0.0, 1.6, -10.0))
    cam = LatentCamera(
        intrinsics=build_intrinsics(image_width=128, image_height=96,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 1.6, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view))
    verts = [-6.0, -2.0, -10.0, 6.0, -2.0, -10.0, 6.0, 6.0, -10.0,
             -6.0, 6.0, -10.0]
    wall = AtlasProxyPrimitive(
        name="wall", primitive_type="mesh",
        metadata={"vertices": verts, "faces": [0, 1, 2, 0, 2, 3],
                  "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]})
    region = {"label": "fill 1", "points_world": [
        [-2.0, 0.0, -10.0], [2.0, 0.0, -10.0],
        [2.0, 3.0, -10.0], [-2.0, 3.0, -10.0]]}
    return AtlasSolve(camera=cam, image_width=128, image_height=96,
                      projection_scene=AtlasProjectionScene(
                          proxy_geometry=[wall],
                          debug_metadata={"fill_rois": {"budget": 3,
                                                        "regions": [region]}}))


def test_crop_roi_renders_the_artist_region_at_native_raster():
    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.zeros((96, 128, 3), np.uint8)
    source[..., 0] = np.arange(128, dtype=np.uint8)[None, :]
    guide, mask, gw, gh, crop, report = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        hole_dilate_px=0, max_gen_long_edge=4096)
    assert not crop["empty"]
    assert (gw, gh) == (crop["width"], crop["height"])   # native, uncapped
    assert guide.shape[1:3] == (gh, gw)
    assert mask.shape[-2:] == (gh, gw)
    # the crop is the same window of the full-frame render (crop-camera pin)
    from atlas_camera.core.camera_crop import RegionROI
    from atlas_camera.dynamic.occlusion_fill import render_crop_sequence
    roi = RegionROI(x=crop["x"], y=crop["y"],
                    width=crop["width"], height=crop["height"])
    direct = render_crop_sequence(
        solve, source, [solve.camera.extrinsics.camera_view_matrix], roi,
        hole_dilate_px=0)[0][0]
    got = (guide[0].numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(got, direct)
    assert "1:1 native" in report


def test_crop_roi_caps_the_generation_raster_and_keeps_aspect():
    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 100, np.uint8)
    guide, mask, gw, gh, crop, report = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        max_gen_long_edge=32)
    assert max(gw, gh) <= 48            # 32 cap, /16 snapped
    assert gw % 16 == 0 and gh % 16 == 0
    assert (crop["width"], crop["height"]) != (gw, gh)
    assert "capped" in report


def test_crop_roi_unused_slot_is_an_empty_no_op():
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop, AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 100, np.uint8)
    guide, mask, gw, gh, crop, report = AtlasCropROI().crop(
        solve, _img(source), roi_slot=3)
    assert crop["empty"] and "no-op" in report
    assert float(mask.sum()) == 0.0
    frame = _plate()
    out, preport = AtlasCompositeCrop().paste(_img(frame), guide, crop)
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.array_equal(got, frame)
    assert "unchanged" in preport


def test_composite_crop_pastes_exactly_and_resizes_a_capped_fill():
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop

    frame = _plate()
    crop = {"empty": False, "x": 16, "y": 8, "width": 32, "height": 24,
            "gen_w": 16, "gen_h": 16}
    fill = np.full((24, 32, 3), 250, np.uint8)
    out, report = AtlasCompositeCrop().paste(_img(frame), _img(fill), crop)
    got = (out[0].numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(got[8:32, 16:48], fill)
    # outside the rect: untouched
    untouched = got.copy(); untouched[8:32, 16:48] = frame[8:32, 16:48]
    assert np.array_equal(untouched, frame)
    # a low-raster (capped) fill is resized to the native rect
    small = np.full((16, 16, 3), 250, np.uint8)
    out2, _ = AtlasCompositeCrop().paste(_img(frame), _img(small), crop)
    got2 = (out2[0].numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(got2[8:32, 16:48], fill)


def test_composite_crop_refuses_a_rect_off_the_frame():
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop

    frame = _plate()
    crop = {"empty": False, "x": 90, "y": 90, "width": 32, "height": 24,
            "gen_w": 32, "gen_h": 24}
    out, report = AtlasCompositeCrop().paste(
        _img(frame), _img(np.zeros((24, 32, 3), np.uint8)), crop)
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.array_equal(got, frame)
    assert "does not fit" in report
