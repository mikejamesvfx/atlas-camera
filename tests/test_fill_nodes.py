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


def _tilted_street_solve():
    """The pivot case: a camera at eye height tilted slightly DOWN, geometry at
    ~10 m — the ordinary street shot. Its ground ray meets Y=0 tens of metres
    out, far past the scene, which is exactly why the two pivots diverge and
    why the ground one made the arcs swing wide."""
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.schema import AtlasExtrinsics

    solve = _wall_solve_with_region()
    view, world, rot3 = look_at_view_matrix((0.0, 1.6, 0.0),
                                            (0.0, 1.24, -9.78))
    solve.camera.extrinsics = AtlasExtrinsics(
        camera_position=(0.0, 1.6, 0.0), camera_rotation_matrix=rot3,
        camera_world_matrix=world, camera_view_matrix=view)
    return solve


def _wall_solve_with_occluder():
    """`_wall_solve_with_region` + a near pillar, so the scene has REAL
    disocclusion: wall the plate camera cannot see because the pillar stands in
    front of it, revealed by a move.

    The plain wall fixture has none — its only holes are the surround, which
    the camera never looked at (outpainting, not disocclusion). Auto ROI
    selection is precisely the thing that must tell those apart, so it needs a
    fixture that contains both.
    """
    from atlas_camera.core.schema import AtlasProxyPrimitive

    def quad(name, x0, x1, z):
        return AtlasProxyPrimitive(
            name=name, primitive_type="mesh",
            metadata={"vertices": [x0, -2.0, z, x1, -2.0, z,
                                   x1, 6.0, z, x0, 6.0, z],
                      "faces": [0, 1, 2, 0, 2, 3],
                      "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]})

    solve = _wall_solve_with_region()
    # The wall is TORN where the pillar hides it — a relief mesh has no
    # geometry behind an occluder, which is what makes a hole a disocclusion
    # rather than a coverage gap. The pillar (x ±0.8 at z=-4, eye at z=0)
    # shadows exactly x ±2 on the wall at z=-10, so from the plate camera the
    # tear is invisible and the frame has no hole there; a move opens it.
    scene = solve.projection_scene
    scene.proxy_geometry = [quad("wall_left", -6.0, -2.0, -10.0),
                            quad("wall_right", 2.0, 6.0, -10.0),
                            quad("pillar", -0.8, 0.8, -4.0)]
    return solve


def test_crop_roi_renders_the_artist_region_at_native_raster():
    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.zeros((96, 128, 3), np.uint8)
    source[..., 0] = np.arange(128, dtype=np.uint8)[None, :]
    guide, mask, gw, gh, crop, report, nw, nh = AtlasCropROI().crop(
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
    guide, mask, gw, gh, crop, report, nw, nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        max_gen_long_edge=32)
    assert max(gw, gh) <= 48            # 32 cap, /16 snapped
    assert gw % 16 == 0 and gh % 16 == 0
    assert (crop["width"], crop["height"]) != (gw, gh)
    assert "capped" in report


def test_crop_roi_raises_a_small_crop_to_the_generation_floor():
    """max_gen_long_edge only ever scales DOWN, so a small auto cluster reached
    the model at its native size — far outside a diffusion model's training
    band, where it hallucinates rather than continues (FLUX Fill answered two
    256x256 sea-cliff ROIs with a sand beach, 2026-09-04)."""
    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 100, np.uint8)
    guide, mask, gw, gh, crop, report, nw, nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        min_gen_long_edge=256)

    assert max(gw, gh) >= 240                  # raised, then /16 snapped
    assert gw % 16 == 0 and gh % 16 == 0
    assert (gw, gh) == tuple(guide.shape[2:0:-1])
    assert tuple(mask.shape[1:]) == (gh, gw)
    # The NATIVE rect is untouched — the floor moves the working raster only.
    assert (nw, nh) == (crop["width"], crop["height"])
    assert max(nw, nh) < max(gw, gh)
    # Aspect preserved within one snap step.
    assert abs(gw / gh - nw / nh) < 0.25
    assert "RAISED" in report


def test_generation_cap_beats_the_floor_because_it_is_a_vram_bound():
    """A floor is a quality preference; a cap is a memory limit. A preference
    must never be able to OOM a run, so the cap wins when they conflict."""
    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 100, np.uint8)
    _g, _m, gw, gh, _crop, report, _nw, _nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        min_gen_long_edge=2048, max_gen_long_edge=256)

    assert max(gw, gh) <= 256 + 16
    assert "capped" not in report or "RAISED" not in report


def test_crop_roi_floor_off_by_default_leaves_the_raster_alone():
    """0 is the shipped behaviour: every saved AtlasCropROI keeps its raster."""
    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 100, np.uint8)
    a = AtlasCropROI().crop(solve, _img(source), roi_slot=1, snap=16,
                            pad_frac=0.0)
    b = AtlasCropROI().crop(solve, _img(source), roi_slot=1, snap=16,
                            pad_frac=0.0, min_gen_long_edge=0)
    assert a[2:4] == b[2:4] == (a[6], a[7]), "1:1 native with the floor off"
    assert "1:1 native" in a[5]


def test_crop_roi_unused_slot_is_an_empty_no_op():
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop, AtlasCropROI

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 100, np.uint8)
    guide, mask, gw, gh, crop, report, nw, nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=3)
    assert crop["empty"] and "no-op" in report
    assert float(mask.sum()) == 0.0
    frame = _plate()
    out, preport, _pm = AtlasCompositeCrop().paste(_img(frame), guide, crop)
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.array_equal(got, frame)
    assert "unchanged" in preport


def test_composite_crop_pastes_exactly_and_resizes_a_capped_fill():
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop

    frame = _plate()
    crop = {"empty": False, "x": 16, "y": 8, "width": 32, "height": 24,
            "gen_w": 16, "gen_h": 16}
    fill = np.full((24, 32, 3), 250, np.uint8)
    out, report, pm = AtlasCompositeCrop().paste(_img(frame), _img(fill), crop)
    got = (out[0].numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(got[8:32, 16:48], fill)
    # outside the rect: untouched
    untouched = got.copy(); untouched[8:32, 16:48] = frame[8:32, 16:48]
    assert np.array_equal(untouched, frame)
    # a low-raster (capped) fill is resized to the native rect
    small = np.full((16, 16, 3), 250, np.uint8)
    out2, *_rest2 = AtlasCompositeCrop().paste(_img(frame), _img(small), crop)
    got2 = (out2[0].numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(got2[8:32, 16:48], fill)


def test_composite_crop_refuses_a_rect_off_the_frame():
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop

    frame = _plate()
    crop = {"empty": False, "x": 90, "y": 90, "width": 32, "height": 24,
            "gen_w": 32, "gen_h": 24}
    out, report, pm = AtlasCompositeCrop().paste(
        _img(frame), _img(np.zeros((24, 32, 3), np.uint8)), crop)
    assert float(pm.sum()) == 0.0
    got = (out[0].numpy() * 255).astype(np.uint8)
    assert np.array_equal(got, frame)
    assert "does not fit" in report


# ------------------------------------------------------ move preset + auto ROI

def test_move_preset_exact_view_round_trips_through_the_patch_parser():
    """The whole point of the node: its exact_view string, fed through the
    SAME parser + orbit reconstruction AtlasAddPatchView uses, must land on
    the path's final pose exactly — no baked frame required."""
    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset
    from atlas_camera.comfy.view_prompts import _parse_exact_pivot, _parse_exact_view
    from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera
    from atlas_camera.core.camera_path import sample_camera_path

    solve = _tilted_street_solve()
    path, exact, report = AtlasCameraMovePreset().build(solve, "arc_left")
    delta = _parse_exact_view(exact)
    pivot = _parse_exact_pivot(exact)
    assert delta is not None
    # The pivot is CARRIED, and it is the scene's, not the ground ray's — a
    # near-level camera's ground pivot sits far past the subject and swings the
    # arc several times too wide (the 2026-08-15 field report).
    assert pivot is not None
    extr = solve.camera.extrinsics
    ground = ground_lookat_pivot(extr)
    assert max(abs(a - b) for a, b in zip(pivot, ground)) > 1.0
    end = sample_camera_path(path)[-1].camera_position
    rec = orbit_camera(extr, pivot, d_azimuth_deg=delta[0],
                       d_elevation_deg=delta[1],
                       distance_scale=delta[2]).camera_position
    assert max(abs(a - b) for a, b in zip(end, rec)) < 1e-4   # 4-decimal string
    assert len(path.keyframes) == 3 and path.frame_count == 100
    assert "Scene pivot" in report


def test_move_preset_pivot_travels_into_the_patch_view_reconstruction():
    """End to end on the contract: the pose AtlasAddPatchView reconstructs from
    the preset's exact_view IS the path's end pose. Reconstructing about the
    default ground pivot instead — what shipped first — lands metres away."""
    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset
    from atlas_camera.comfy.view_prompts import _parse_exact_pivot, _parse_exact_view
    from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera
    from atlas_camera.core.camera_path import sample_camera_path

    solve = _tilted_street_solve()
    path, exact, _report = AtlasCameraMovePreset().build(solve, "arc_right",
                                                         angle_deg=12.0)
    extr = solve.camera.extrinsics
    delta = _parse_exact_view(exact)
    end = sample_camera_path(path)[-1].camera_position
    carried = orbit_camera(extr, _parse_exact_pivot(exact), d_azimuth_deg=delta[0],
                           d_elevation_deg=delta[1],
                           distance_scale=delta[2]).camera_position
    assumed = orbit_camera(extr, ground_lookat_pivot(extr), d_azimuth_deg=delta[0],
                           d_elevation_deg=delta[1],
                           distance_scale=delta[2]).camera_position
    assert max(abs(a - b) for a, b in zip(end, carried)) < 1e-4
    assert max(abs(a - b) for a, b in zip(end, assumed)) > 1.0


def test_move_preset_pan_emits_the_zero_delta_and_warns():
    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

    solve = _wall_solve_with_region()
    path, exact, report = AtlasCameraMovePreset().build(solve, "pan_left")
    assert exact.startswith("azimuth_deg=0.0000 elevation_deg=0.0000 "
                            "distance_scale=1.0000")
    assert "pan swivels in place" in report
    assert len(path.keyframes) == 2


def test_crop_roi_auto_largest_ranks_holes_by_area():
    """Orbit the wall solve so real disocclusion opens, then let auto mode
    pick clusters: rank 1 exists, an absurd rank no-ops, and artist mode on
    the same inputs still requires drawn regions (regression pin)."""
    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset, AtlasCropROI

    solve = _wall_solve_with_occluder()
    source = np.full((96, 128, 3), 120, np.uint8)
    path, _exact, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                     angle_deg=35.0)
    guide, mask, gw, gh, crop, report, nw, nh = AtlasCropROI().crop(
        solve, _img(source), camera_path=path, roi_slot=1, snap=16,
        roi_source="auto_largest", min_area_px=16)
    assert not crop["empty"], report
    assert "auto rank 1/" in report
    assert float(mask.sum()) > 0.0
    *_rest, crop9, report9, _nw9, _nh9 = AtlasCropROI().crop(
        solve, _img(source), camera_path=path, roi_slot=3, snap=16,
        roi_source="auto_largest", min_area_px=1 << 19)
    assert crop9["empty"] and "no-op" in report9
    # artist default unchanged: slot beyond drawn regions still no-ops
    *_a, crop_a, report_a, _nwa, _nha = AtlasCropROI().crop(
        solve, _img(source), camera_path=path, roi_slot=2)
    assert crop_a["empty"] and "no artist region" in report_a


def test_auto_mode_never_ranks_sky_class_holes():
    """THE sky failsafe. The wall fixture's surround (no geometry — the sky
    class) is a hole from the SOLVED pose too, so move-revealed subtraction
    must keep it out of the ranking entirely: without the failsafe it would
    dwarf every real disocclusion cluster and rank first (the measured G5
    failure). What survives must be revealed by the move."""
    import numpy as np2

    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset
    from atlas_camera.dynamic.occlusion_fill import survey_hole_rois
    from atlas_camera.core.camera_path import sample_camera_path

    solve = _wall_solve_with_occluder()
    source = np2.full((96, 128, 3), 120, np2.uint8)
    path, _e, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                 angle_deg=35.0)
    view = sample_camera_path(path)[-1].camera_view_matrix

    naive, *_n = survey_hole_rois(solve, source, [view],
                                  survey_resolution=128, min_area_px=16,
                                  snap=8)
    safe, *_s = survey_hole_rois(solve, source, [view],
                                 survey_resolution=128, min_area_px=16,
                                 snap=8, move_revealed_only=True)
    assert naive, "fixture must produce clusters at all"
    # the never-covered surround dominates the naive ranking
    assert naive[0].area_px > sum(r.area_px for r in safe),         "naive rank 1 should be the huge never-covered region"
    # everything the failsafe kept is genuinely move-revealed: none of it was
    # a hole at the solved pose
    from atlas_camera.dynamic.occlusion_fill import (
        render_disocclusion_sequence,
    )
    base_mask = render_disocclusion_sequence(
        solve, source, [solve.camera.extrinsics.camera_view_matrix],
        resolution=128, hole_dilate_px=0)[0][1] > 127
    end_mask = render_disocclusion_sequence(
        solve, source, [view], resolution=128, hole_dilate_px=0)[0][1] > 127
    revealed = end_mask & ~base_mask
    for roi in safe:
        # survey res == plate res here (128), so coords map 1:1
        window = revealed[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
        assert window.any(), "kept cluster contains no move-revealed pixels"


def test_auto_mode_exclude_mask_removes_a_region_from_ranking():
    import numpy as np2

    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset, AtlasCropROI

    solve = _wall_solve_with_occluder()
    source = np2.full((96, 128, 3), 120, np2.uint8)
    path, _e, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                 angle_deg=35.0)
    # find rank 1 normally, then exclude exactly that rect — rank 1 must move
    g, m, gw, gh, crop1, rep1, _nw1, _nh1 = AtlasCropROI().crop(
        solve, _img(source), camera_path=path, roi_slot=1, snap=16,
        roi_source="auto_largest", min_area_px=16)
    assert not crop1["empty"], rep1
    ex = np2.zeros((96, 128), np2.float32)
    ex[crop1["y"]:crop1["y"] + crop1["height"],
       crop1["x"]:crop1["x"] + crop1["width"]] = 1.0
    *_o, crop2, rep2, _nw2, _nh2 = AtlasCropROI().crop(
        solve, _img(source), camera_path=path, roi_slot=1, snap=16,
        roi_source="auto_largest", min_area_px=16,
        exclude_mask=torch.from_numpy(ex))
    assert "exclude_mask applied" in rep2
    if not crop2["empty"]:
        assert (crop2["x"], crop2["y"], crop2["width"], crop2["height"]) !=             (crop1["x"], crop1["y"], crop1["width"], crop1["height"])


def test_composite_crop_pasted_mask_accumulates_hole_pixels_only():
    """Patch re-entry contract: pasted_mask carries EXACTLY the filled hole
    pixels at plate coordinates, unioned across a chain of pastes."""
    from atlas_camera.comfy.nodes_fill import AtlasCompositeCrop

    frame = _plate()
    crop = {"empty": False, "x": 16, "y": 8, "width": 32, "height": 24,
            "gen_w": 32, "gen_h": 24}
    fill = np.full((24, 32, 3), 250, np.uint8)
    hole = np.zeros((24, 32), np.float32)
    hole[4:12, 6:20] = 1.0
    out, report, pm = AtlasCompositeCrop().paste(
        _img(frame), _img(fill), crop, mask=torch.from_numpy(hole))
    got = pm[0].numpy()
    assert got.shape == (96, 96)
    expected = np.zeros((96, 96), np.float32)
    expected[8 + 4:8 + 12, 16 + 6:16 + 20] = 1.0
    assert np.array_equal(got, expected)
    assert "hole pixels only" in report
    # chain: a second paste unions with prior_mask
    crop2 = {"empty": False, "x": 48, "y": 48, "width": 16, "height": 16,
             "gen_w": 16, "gen_h": 16}
    _o2, _r2, pm2 = AtlasCompositeCrop().paste(
        _img(frame), _img(np.full((16, 16, 3), 9, np.uint8)), crop2,
        prior_mask=pm)
    got2 = pm2[0].numpy()
    assert got2[8 + 4, 16 + 6] == 1.0          # prior survives
    assert got2[48:64, 48:64].all()            # whole new rect (no mask given)
    # empty crop passes prior through
    _o3, _r3, pm3 = AtlasCompositeCrop().paste(
        _img(frame), _img(frame), {"empty": True}, prior_mask=pm2)
    assert np.array_equal(pm3[0].numpy(), got2)


def test_auto_crop_mask_never_asks_the_generator_for_sky():
    """SELECTION and the CROP MASK must apply the same test. The ROI is a
    RECT: a legitimate cluster's padded rect can contain sky the ranking
    already rejected, and the emitted mask used to hand that sky straight to
    the generator — which pasted it, which rode pasted_mask into
    AtlasAddPatchView as geometry to build (measured live 2026-08-15: a
    sky-textured sheet above the roofline). Auto mode only."""
    import numpy as np2

    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset, AtlasCropROI
    from atlas_camera.core.camera_crop import crop_intrinsics
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.camera_path import sample_camera_path
    from atlas_camera.dynamic.occlusion_fill import (
        not_disocclusion_mask,
        plate_hole_survey,
    )
    from atlas_camera.core.camera_crop import RegionROI

    solve = _wall_solve_with_occluder()
    source = np2.full((96, 128, 3), 120, np2.uint8)
    path, _e, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                 angle_deg=35.0)
    view = sample_camera_path(path)[-1].camera_view_matrix

    *_o, crop, report, _nw, _nh = AtlasCropROI().crop(
        solve, _img(source), camera_path=path, roi_slot=1, snap=16,
        roi_source="auto_largest", min_area_px=16)
    assert not crop["empty"], report
    mask = _o[1][0].numpy()                       # _o = (guide, mask, w, h)

    roi = RegionROI(x=crop["x"], y=crop["y"], width=crop["width"],
                    height=crop["height"])
    spec = CameraSpec.from_intrinsics(crop_intrinsics(solve.camera.intrinsics,
                                                      roi))
    plate = plate_hole_survey(solve, source, resolution=1024)
    drop = not_disocclusion_mask(plate, view=view, fx=spec.fx, fy=spec.fy,
                                 cx=spec.cx, cy=spec.cy,
                                 width=roi.width, height=roi.height)
    if mask.shape != drop.shape:                 # generation raster was capped
        import numpy as _np
        yi = (_np.arange(mask.shape[0]) * (drop.shape[0] / mask.shape[0])).astype(int)
        xi = (_np.arange(mask.shape[1]) * (drop.shape[1] / mask.shape[1])).astype(int)
        drop = drop[yi.clip(0, drop.shape[0] - 1)][:, xi.clip(0, drop.shape[1] - 1)]
    assert not (mask > 0.5)[drop].any(), \
        "auto crop mask still asks the generator to fill non-disocclusion"
    assert mask.sum() > 0, "the whole mask was dropped — fixture has no fill"


def test_artist_crop_mask_is_left_alone():
    """Artist-wins: a drawn region is a judgement, not a ranking, so the
    not-disocclusion trim must not touch it."""
    import numpy as np2

    from atlas_camera.comfy.nodes_fill import AtlasCropROI

    solve = _wall_solve_with_region()
    source = np2.full((96, 128, 3), 120, np2.uint8)
    _g, _m, _w, _h, crop, report, _nw, _nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16)
    assert not crop["empty"], report
    assert "dropped as not-disocclusion" not in report


# ---------------------------------------------------------------------------
# AtlasCropSourcePhoto — the pristine photo crop for the Qwen ROI loop (2026-08-16)

def test_crop_source_photo_is_the_untouched_plate_window():
    from atlas_camera.comfy.nodes_fill import AtlasCropROI, AtlasCropSourcePhoto

    solve = _wall_solve_with_region()
    source = np.zeros((96, 128, 3), np.uint8)
    source[..., 0] = np.arange(128, dtype=np.uint8)[None, :]
    source[..., 1] = np.arange(96, dtype=np.uint8)[:, None]
    _g, _m, gw, gh, crop, _r, _nw, _nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        hole_dilate_px=0, max_gen_long_edge=4096)
    photo, w, h, report = AtlasCropSourcePhoto().crop_photo(_img(source), crop)
    assert (w, h) == (gw, gh)
    got = (photo[0].numpy() * 255).round().astype(np.uint8)
    x, y = crop["x"], crop["y"]
    assert np.array_equal(got, source[y:y + crop["height"], x:x + crop["width"]])
    assert "register_to_primary" in report


def test_crop_source_photo_pads_and_squares_within_the_plate():
    from atlas_camera.comfy.nodes_fill import AtlasCropROI, AtlasCropSourcePhoto

    solve = _wall_solve_with_region()
    source = np.full((96, 128, 3), 77, np.uint8)
    _g, _m, gw, gh, crop, _r, _nw, _nh = AtlasCropROI().crop(
        solve, _img(source), roi_slot=1, snap=16, pad_frac=0.0,
        hole_dilate_px=0, max_gen_long_edge=4096)
    photo, w, h, report = AtlasCropSourcePhoto().crop_photo(
        _img(source), crop, pad_frac=0.5, square=True)
    assert w % 16 == 0 and h % 16 == 0
    assert "square" in report and "pad 0.50" in report
    assert photo.shape[1:3] == (h, w)


def test_crop_source_photo_empty_handle_is_a_noop():
    from atlas_camera.comfy.nodes_fill import AtlasCropSourcePhoto
    photo, w, h, report = AtlasCropSourcePhoto().crop_photo(
        _img(np.zeros((32, 32, 3), np.uint8)), {"empty": True})
    assert (w, h) == (64, 64) and "no-op" in report


# ---------------------------------------------------------------------------
# AtlasFillOccluded — the expanded per-cluster loop
#
# The node's whole value is that a fill becomes GEOMETRY rather than pixels, so
# these tests pin the expansion's SHAPE (one branch per cluster, chained, every
# patch measured) and the three refusals that must never silently produce a
# half-built graph.
# ---------------------------------------------------------------------------
_QWEN_PACK = ("QwenEditConfigPreparer",
              "TextEncodeQwenImageEditPlusCustom_lrzjason",
              "QwenEditOutputExtractor", "CropWithPadInfo")
_FLUX_PACK = ("InpaintModelConditioning", "FluxGuidance")
#: both, so a test picks its fill_model rather than its registry
_FILL_PACK = _FLUX_PACK + _QWEN_PACK


class _FakeDepth:
    """Stand-in for ATLAS_DEPTH_MAP: the node only ever passes it through."""


def _fill_node(monkeypatch, registry=_FILL_PACK):
    """AtlasFillOccluded with the third-party pack's presence under test
    control — CI has no Qwen nodes installed."""
    from atlas_camera.comfy import node_helpers, nodes_fill

    monkeypatch.setattr(node_helpers, "_comfy_registry",
                        lambda: {name: object() for name in registry})
    return nodes_fill.AtlasFillOccluded()


def _fill_scene():
    """The occluder fixture orbited far enough to open real disocclusion."""
    from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

    solve = _wall_solve_with_occluder()
    source = _img(np.full((96, 128, 3), 120, np.uint8))
    path, exact, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                    angle_deg=35.0)
    return solve, source, path, exact


def _expanded(out):
    """(nodes-by-class-type, result tuple) from an expansion dict."""
    assert isinstance(out, dict), f"expected an expansion, got {type(out)}"
    graph = out["expand"]
    by_type = {}
    for node in graph.values():
        by_type.setdefault(node["class_type"], []).append(node)
    return by_type, out["result"]



def _wall_solve_with_two_occluders():
    """Two pillars, so a move opens TWO separate disocclusion clusters.

    The single-pillar fixture only ever yields one, which cannot exercise the
    chaining contract (patch i building on patch i-1). Measured: two pillars
    at a 35 deg arc give exactly 2 clusters; 25 deg merges them and 45 deg
    swings past the wall segments.
    """
    from atlas_camera.core.schema import AtlasProxyPrimitive

    def quad(name, x0, x1, z):
        return AtlasProxyPrimitive(
            name=name, primitive_type="mesh",
            metadata={"vertices": [x0, -2.0, z, x1, -2.0, z,
                                   x1, 6.0, z, x0, 6.0, z],
                      "faces": [0, 1, 2, 0, 2, 3],
                      "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]})

    solve = _wall_solve_with_occluder()
    solve.projection_scene.proxy_geometry = [
        quad("wall_left", -8.0, -4.0, -10.0),
        quad("wall_mid", -2.0, 2.0, -10.0),
        quad("wall_right", 4.0, 8.0, -10.0),
        quad("pillar_a", -3.4, -2.6, -4.0),
        quad("pillar_b", 2.6, 3.4, -4.0),
    ]
    return solve


def test_fill_occluded_expands_one_registered_branch_per_cluster():
    """The shape of the loop: N clusters -> N crops, N fills, N patches, and
    every patch MEASURED against the primary rather than trusted."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        solve, source, path, exact = _fill_scene()
        out = node.fill(solve, source, model="MODEL", clip="CLIP", vae="VAE",
                        camera_path=path, primary_depth=_FakeDepth(),
                        exact_view_override=exact, min_area_px=16, snap=16,
                        max_rois=8)
    finally:
        mp.undo()

    by_type, result = _expanded(out)
    crops = by_type["AtlasCropROI"]
    patches = by_type["AtlasAddPatchView"]
    assert crops, "no cluster was found — fixture no longer opens a hole"
    assert len(patches) == len(crops) == len(by_type["AtlasCompositeCrop"])
    # one FLUX Fill chain per crop (the default model)
    for name in _FLUX_PACK:
        assert len(by_type[name]) == len(crops), name
    assert len(by_type["KSampler"]) == len(crops)
    # noise_mask is the whole reason FLUX Fill is the default: it masks the
    # LATENT, so the sampler cannot repaint outside the hole
    assert all(n["inputs"]["noise_mask"] is True
               for n in by_type["InpaintModelConditioning"])
    assert "QwenEditConfigPreparer" not in by_type

    # slots are 1..N, auto mode, and share the survey's own parameters
    assert [c["inputs"]["roi_slot"] for c in crops] == list(
        range(1, len(crops) + 1))
    assert all(c["inputs"]["roi_source"] == "auto_largest" for c in crops)
    assert all(c["inputs"]["min_area_px"] == 16 for c in crops)

    # every patch takes the move's CONSTRUCTED pose, is handed THE CROP paired
    # with its own ATLAS_CROP handle (2026-09-04 -- the handle makes
    # crop_intrinsics the patch camera, so the crop no longer has to be pasted
    # into a full frame to be describable), and derives its own geometry
    # bounded by the crop's hole mask. Masks travel in the IMAGE's frame: the
    # crop's raster, not plate coordinates.
    comp_ids = {nid for nid, n in out["expand"].items()
                if n["class_type"] == "AtlasCompositeCrop"}
    crop_ids = {nid for nid, n in out["expand"].items()
                if n["class_type"] == "AtlasCropROI"}
    # The fill comes back through ImageScale, which returns it to the crop's
    # NATIVE rect -- min_gen_long_edge raises the WORKING raster and that must
    # not leak past the generation.
    scale_ids = {nid for nid, n in out["expand"].items()
                 if n["class_type"] == "ImageScale"}
    decode_ids = {nid for nid, n in out["expand"].items()
                  if n["class_type"] in ("VAEDecode", "CropWithPadInfo")}
    for nid, n in out["expand"].items():
        if n["class_type"] != "ImageScale":
            continue
        assert n["inputs"]["image"][0] in decode_ids
        # width/height are the crop's native_width/native_height outputs (6, 7)
        assert n["inputs"]["width"][1] == 6 and n["inputs"]["height"][1] == 7
    for p in patches:
        assert p["inputs"]["camera_source"] == "declared_orbit"
        assert p["inputs"]["geometry_source"] == "own_depth"
        assert p["inputs"]["exact_view_override"] == exact
        img = p["inputs"]["patch_image"]
        assert isinstance(img, list) and img[0] in scale_ids and img[1] == 0, (
            "patch_image must be the FILL resampled to the crop's native "
            "rect, not the composited full frame")
        # The handle is what makes that legal.
        assert p["inputs"]["crop"][0] in crop_ids and p["inputs"]["crop"][1] == 4
        # The hole mask in the crop's OWN frame (AtlasCropROI output 1), not
        # the plate-space pasted mask.
        assert p["inputs"]["patch_mask"][0] in crop_ids
        assert p["inputs"]["patch_mask"][1] == 1
        assert p["inputs"]["patch_mask"][0] not in comp_ids

    assert result[0] is not solve, "the solve must chain through the patches"
    assert "expanded graph" in result[2]


def test_fill_occluded_chains_solve_and_image_through_every_branch():
    """Each ROI must build on the last: patch i takes patch i-1's solve and
    composite i takes composite i-1's image. A fan-out would silently keep
    only the final branch's geometry."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

        node = _fill_node(mp)
        solve = _wall_solve_with_two_occluders()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        path, _exact, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                         angle_deg=35.0)
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        min_area_px=16, snap=16, max_rois=8)
    finally:
        mp.undo()

    graph = out["expand"]
    patches = [(nid, n) for nid, n in graph.items()
               if n["class_type"] == "AtlasAddPatchView"]
    comps = [(nid, n) for nid, n in graph.items()
             if n["class_type"] == "AtlasCompositeCrop"]
    assert len(patches) >= 2, ("the two-occluder fixture must open at least "
                               "two clusters or this pins nothing")

    patch_ids = {nid for nid, _ in patches}
    comp_ids = {nid for nid, _ in comps}
    upstream_patch = [n["inputs"]["solve"] for _, n in patches
                      if isinstance(n["inputs"]["solve"], list)]
    upstream_comp = [n["inputs"]["image"] for _, n in comps
                     if isinstance(n["inputs"]["image"], list)]
    # every branch but the first draws its solve/image from a sibling
    assert len(upstream_patch) == len(patches) - 1
    assert len(upstream_comp) == len(comps) - 1
    assert all(ref[0] in patch_ids for ref in upstream_patch)
    assert all(ref[0] in comp_ids for ref in upstream_comp)


def test_fill_occluded_honours_the_roi_budget():
    """max_rois caps the generations and SAYS how many it left behind — the
    long tail of speckle is not worth a sampler pass each."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        solve, source, path, _exact = _fill_scene()
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        min_area_px=16, snap=16, max_rois=1)
    finally:
        mp.undo()

    by_type, result = _expanded(out)
    assert len(by_type["AtlasCropROI"]) == 1
    assert len(by_type["AtlasAddPatchView"]) == 1
    assert "filling the largest 1" in result[2]


def test_fill_occluded_passes_through_when_the_chosen_stack_is_missing():
    """A missing pack is REPORTED and skipped, never raised — the AtlasInput
    doctrine — and the probe follows fill_model, so having one stack installed
    does not excuse the other."""
    import pytest as _pytest

    solve, source, path, _exact = _fill_scene()

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp, registry=("KSampler",))
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        min_area_px=16, snap=16)
    finally:
        mp.undo()
    assert isinstance(out, tuple), "must not expand a graph it cannot run"
    assert out[0] is solve and out[1] is source
    assert "InpaintModelConditioning" in out[2] and "not installed" in out[2]

    # qwen_edit chosen while only the FLUX stack is present
    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp, registry=_FLUX_PACK)
        out2 = node.fill(solve, source, model="M", clip="C", vae="V",
                         camera_path=path, primary_depth=_FakeDepth(),
                         fill_model="qwen_edit", min_area_px=16, snap=16)
    finally:
        mp.undo()
    assert isinstance(out2, tuple)
    assert "QwenEditConfigPreparer" in out2[2]


def test_fill_occluded_qwen_edit_still_wires_its_own_chain():
    """The measured loser stays reachable: fill_model=qwen_edit emits the
    config-preparer chain and its pad/crop round trip, not FLUX's."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        solve, source, path, _exact = _fill_scene()
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        fill_model="qwen_edit", min_area_px=16, snap=16,
                        max_rois=1)
    finally:
        mp.undo()

    by_type, _result = _expanded(out)
    for name in _QWEN_PACK:
        assert by_type[name], name
    assert "InpaintModelConditioning" not in by_type
    assert "FluxGuidance" not in by_type


def test_fill_occluded_refuses_to_run_unmeasured_without_primary_depth():
    """The primary_depth guard belongs to MEASUREMENT only.

    In register_to_primary mode without metric primary depth every ROI would
    fail its gates, so the node refuses up front rather than expanding a graph
    whose every patch is destined to be dropped. In the default declared_orbit
    mode there is nothing to measure, so depth is irrelevant and the loop must
    expand regardless — pinning that the guard did not become a blanket
    requirement."""
    import pytest as _pytest

    solve, source, path, _exact = _fill_scene()

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=None,
                        camera_source="register_to_primary",
                        min_area_px=16, snap=16)
    finally:
        mp.undo()
    assert isinstance(out, tuple)
    assert out[0] is solve
    assert "primary_depth" in out[2]

    # the artist fallback stays reachable on request
    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        out2 = node.fill(solve, source, model="M", clip="C", vae="V",
                         camera_path=path, primary_depth=None,
                         camera_source="register_to_primary",
                         min_area_px=16, snap=16,
                         on_registration_failure="declared_orbit")
    finally:
        mp.undo()
    by_type, _result = _expanded(out2)
    assert by_type["AtlasAddPatchView"]

    # ...and the DEFAULT mode never needed depth at all
    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        out3 = node.fill(solve, source, model="M", clip="C", vae="V",
                         camera_path=path, primary_depth=None,
                         min_area_px=16, snap=16)
    finally:
        mp.undo()
    by_type3, result3 = _expanded(out3)
    assert by_type3["AtlasAddPatchView"]
    assert all(p["inputs"]["camera_source"] == "declared_orbit"
               for p in by_type3["AtlasAddPatchView"])
    assert "constructed" in result3[2].lower()


def test_a_second_move_appends_under_its_own_name(monkeypatch=None):
    """Chaining a second fill for a DIFFERENT move must not collide.

    APPEND, not clobber, and the survey is why: it renders
    `gather_scene_meshes(solve)`, which includes prior projection_sources, so a
    second run SEES the first move's geometry and targets only what is still
    open. That is the node's advertised story -- "every LATER move reuses it
    instead of re-inventing it" -- and clobbering would delete exactly the
    geometry that makes it true. A second run of the SAME move is self-limiting
    for the same reason: nothing is left to find, so it passes through.

    What was broken is identity, not policy: both runs named their patches
    `fill_roi1..N`, so two moves produced duplicate names on one solve.
    """
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

        node = _fill_node(mp)
        solve = _wall_solve_with_two_occluders()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        left, _e1, _r1 = AtlasCameraMovePreset().build(solve, "arc_left",
                                                       angle_deg=35.0)
        right, _e2, _r2 = AtlasCameraMovePreset().build(solve, "arc_right",
                                                        angle_deg=35.0)
        kw = dict(model="M", clip="C", vae="V", primary_depth=_FakeDepth(),
                  min_area_px=16, snap=16, max_rois=4)
        a = node.fill(solve, source, camera_path=left, **kw)
        b = node.fill(solve, source, camera_path=right, **kw)
    finally:
        mp.undo()

    def names(out):
        return sorted(n["inputs"]["name"] for n in out["expand"].values()
                      if n["class_type"] == "AtlasAddPatchView")

    na, nb = names(a), names(b)
    assert na and nb
    assert not set(na) & set(nb), (
        f"two moves must not produce colliding patch names: {na} vs {nb}")


def test_the_same_move_names_its_patches_identically_every_time():
    """The tag is the MOVE, so a re-run is stable rather than accumulating a
    new set of names for geometry that is already there."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

        node = _fill_node(mp)
        solve = _wall_solve_with_two_occluders()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        path, _e, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                     angle_deg=35.0)
        kw = dict(model="M", clip="C", vae="V", primary_depth=_FakeDepth(),
                  min_area_px=16, snap=16, max_rois=4)
        one = node.fill(solve, source, camera_path=path, **kw)
        two = node.fill(solve, source, camera_path=path, **kw)
    finally:
        mp.undo()

    def names(out):
        return sorted(n["inputs"]["name"] for n in out["expand"].values()
                      if n["class_type"] == "AtlasAddPatchView")

    assert names(one) == names(two)


def test_patch_names_carry_the_move_and_the_roi():
    """A name an artist reads in the viewport layer list has to say which move
    and which cluster it came from, or a two-move solve is unreadable."""
    import re

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

        node = _fill_node(mp)
        solve = _wall_solve_with_two_occluders()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        path, _e, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                     angle_deg=35.0)
        out = node.fill(solve, source, camera_path=path, model="M", clip="C",
                        vae="V", primary_depth=_FakeDepth(), min_area_px=16,
                        snap=16, max_rois=4)
    finally:
        mp.undo()

    names = [n["inputs"]["name"] for n in out["expand"].values()
             if n["class_type"] == "AtlasAddPatchView"]
    assert names
    for n in names:
        assert re.fullmatch(r"fill_[0-9a-f]{6}_roi\d+", n), n


def test_the_docstring_states_the_bounds_it_actually_ships_with():
    """The spec must not promise more than the defaults deliver.

    It used to open with "Fill EVERY move-revealed hole" and "every accepted
    fill is appended as a MEASURED ProjectionSource". Neither survived contact
    with the shipped defaults: `min_area_px` is a floor, `max_rois` a budget
    (5 of 18 clusters on the castle), and `camera_source=declared_orbit` means
    nothing is measured and nothing is gated. An operator who reads "every" and
    sees residual hole concludes the node is broken; one who reads "measured"
    believes a gate exists that does not run.

    So the prose has to name the bounds it ships with, and this fails if either
    the prose or the default moves without the other.
    """
    from atlas_camera.comfy.nodes_fill import AtlasFillOccluded

    doc = AtlasFillOccluded.__doc__
    opt = AtlasFillOccluded.INPUT_TYPES()["optional"]

    assert opt["min_area_px"][1]["default"] == 1024
    assert opt["max_rois"][1]["default"] == 8
    # No explicit default on the combo, so ComfyUI takes the first entry.
    assert opt["camera_source"][0][0] == "declared_orbit"

    assert "min_area_px" in doc and "max_rois" in doc
    assert 'NOT "every"' in doc, "the scope bound must be stated, not implied"
    assert "declared_orbit" in doc and 'NOR "measured"' in doc
    assert "measured ``ProjectionSource``" not in doc, "the retracted claim is back"


def test_the_docstring_points_at_the_fillable_metric_not_the_reported_one():
    """`peak hole` counts sky -- 86% of it after a fill -- so a reader who
    scores the node on the number it prints will read 72% success as 29%."""
    from atlas_camera.comfy.nodes_fill import AtlasFillOccluded

    doc = AtlasFillOccluded.__doc__
    assert "FILLABLE" in doc
    assert "peak hole" in doc
    assert "sky" in doc


def test_fill_occluded_reports_when_the_move_opens_nothing():
    """No move, no disocclusion: a pass-through with a reason, not an empty
    expansion."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        solve = _wall_solve_with_occluder()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=None, primary_depth=_FakeDepth(),
                        min_area_px=1 << 19, snap=16)
    finally:
        mp.undo()

    assert isinstance(out, tuple)
    assert out[0] is solve
    assert "nothing to fill" in out[2]


def test_fill_occluded_survey_agrees_with_crop_roi_slot_for_slot():
    """The node counts clusters itself to know how many branches to emit, and
    AtlasCropROI resolves each slot independently. If those two ever disagreed
    the graph would fill the wrong regions, so pin that slot i is the same
    rect on both sides."""
    from atlas_camera.comfy.nodes_fill import AtlasCropROI, AtlasFillOccluded

    solve, source, path, _exact = _fill_scene()
    rois, _peak, _view = AtlasFillOccluded._survey(
        solve, source, path, None, pad_frac=0.10, min_area_px=16, snap=16)
    assert rois, "fixture no longer opens a cluster"

    *_out, crop, report, _nw, _nh = AtlasCropROI().crop(
        solve, source, camera_path=path, roi_slot=1, snap=16, pad_frac=0.10,
        roi_source="auto_largest", min_area_px=16)
    assert not crop["empty"], report
    assert f"auto rank 1/{len(rois)}" in report
    for key in ("x", "y", "width", "height"):
        assert crop[key] == getattr(rois[0], key), key


def test_fill_occluded_report_carries_the_per_roi_registration_verdicts():
    """The report must answer "what actually registered", not just "what did
    it try". It is built at EXPANSION time, before any patch has run, so the
    per-ROI verdicts can only reach the output by CHAINING each
    AtlasAddPatchView's own report into it."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp, registry=_FILL_PACK + ("StringConcatenate",))
        solve, source, path, _exact = _fill_scene()
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        min_area_px=16, snap=16, max_rois=8)
    finally:
        mp.undo()

    by_type, result = _expanded(out)
    # the report output is a LINK to the join, not a bare string
    assert isinstance(result[2], list), "report should chain the patch reports"
    joins = by_type["StringConcatenate"]
    assert joins, "no join emitted"
    patch_ids = {nid for nid, n in out["expand"].items()
                 if n["class_type"] == "AtlasAddPatchView"}
    fed_by_patches = [j for j in joins
                      if isinstance(j["inputs"].get("string_b"), list)
                      and j["inputs"]["string_b"][0] in patch_ids]
    assert fed_by_patches, "no patch report reaches the join"
    # slot 1 of AtlasAddPatchView is its report
    assert all(j["inputs"]["string_b"][1] == 1 for j in fed_by_patches)


def test_fill_occluded_report_degrades_without_the_join_node():
    """No StringConcatenate: the plan still comes back as a plain string, and
    says why the verdicts are missing rather than pretending to have them."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)   # pack only, no StringConcatenate
        solve, source, path, _exact = _fill_scene()
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        min_area_px=16, snap=16, max_rois=8)
    finally:
        mp.undo()

    _by_type, result = _expanded(out)
    assert isinstance(result[2], str)
    assert "StringConcatenate unavailable" in result[2]


def test_fill_occluded_neutralises_the_green_sentinel_before_the_edit_model():
    """render_crop_sequence paints holes LTX_INPAINT_GREEN, which the LTX
    inpaint pipeline was TRAINED on and an edit model was not — Qwen returns
    an out-of-gamut block untouched, exactly as it returned magenta. Pin that
    the guide reaching QwenEditConfigPreparer is the repainted one, never the
    raw crop."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        node = _fill_node(mp)
        solve, source, path, _exact = _fill_scene()
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FakeDepth(),
                        min_area_px=16, snap=16, max_rois=1)
    finally:
        mp.undo()

    graph = out["expand"]
    by_type, _result = _expanded(out)
    crop_ids = {nid for nid, n in graph.items()
                if n["class_type"] == "AtlasCropROI"}
    neutral_ids = {nid for nid, n in graph.items()
                   if n["class_type"] == "ImageCompositeMasked"}
    assert neutral_ids, "no sentinel repaint emitted"

    prep = by_type["InpaintModelConditioning"][0]
    assert prep["inputs"]["pixels"][0] in neutral_ids, (
        "the fill model must see the repainted guide, not the green crop")
    assert prep["inputs"]["mask"][0] in crop_ids, "the mask still comes raw"

    # grey, and sized from the crop's own generation raster
    empty = by_type["EmptyImage"][0]
    assert empty["inputs"]["color"] == 0x808080
    assert empty["inputs"]["width"][0] in crop_ids
    assert empty["inputs"]["height"][0] in crop_ids

