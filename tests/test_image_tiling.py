"""Host-agnostic image tiling maths (``atlas_camera.core.image_tiling``).

The tiling/blending algorithm is pure array arithmetic — it knows nothing about
depth models, torch, or ComfyUI — so it lives in ``core`` and is testable with
numpy alone. These tests exercise the four things that silently go wrong:
tile layout (coverage, overlap, no sliver tiles), the feather ramp (symmetry,
frame edges left alone, partition of unity once normalised), the affine anchor
(recovering a known scale + offset, declining degenerate fits), and assembly
(reconstructing a known image without seams).
"""
from __future__ import annotations

import pytest

from atlas_camera.core.image_tiling import (
    _feather_weights,
    assemble_tiles,
    fit_affine_to_reference,
    tile_boxes,
)

np = pytest.importorskip("numpy")


# ------------------------------------------------------------------- layout


def test_tiles_cover_every_pixel():
    w, h, side = 900, 640, 256
    hit = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in tile_boxes(w, h, side, 0.25):
        hit[y0:y1, x0:x1] = True
    assert hit.all(), "tiling left pixels uncovered"


def test_tiles_stay_inside_the_frame():
    w, h = 900, 640
    for x0, y0, x1, y1 in tile_boxes(w, h, 256, 0.25):
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h


def test_no_sliver_tile_from_a_ragged_remainder():
    """Tiles are spread evenly, so no thin strip gets its own inferred scale."""
    boxes = tile_boxes(1000, 700, 256, 0.25)
    widths = {x1 - x0 for x0, _, x1, _ in boxes}
    heights = {y1 - y0 for _, y0, _, y1 in boxes}
    assert min(widths) > 128, f"sliver tile widths {sorted(widths)}"
    assert min(heights) > 128, f"sliver tile heights {sorted(heights)}"


def test_single_tile_when_the_image_already_fits():
    assert tile_boxes(200, 150, 256, 0.25) == [(0, 0, 200, 150)]


def test_consecutive_tiles_actually_overlap():
    side = 256
    boxes = tile_boxes(1000, 300, side, 0.25)
    xs = sorted({b[0] for b in boxes})
    assert len(xs) > 1
    for a, b in zip(xs, xs[1:]):
        assert b - a < side, "consecutive tiles must share pixels to blend across"


def test_more_overlap_means_more_tiles():
    assert len(tile_boxes(2000, 1000, 256, 0.5)) > len(
        tile_boxes(2000, 1000, 256, 0.1))


# ------------------------------------------------------------------- feather


def _box_weights(box, width, height, ramp):
    x0, y0, x1, y1 = box
    return _feather_weights(y1 - y0, x1 - x0, box, width, height, ramp, np)


def test_feather_is_symmetric_for_an_interior_tile():
    w = _box_weights((100, 100, 356, 356), 1000, 1000, 32)
    assert np.allclose(w, w[:, ::-1])
    assert np.allclose(w, w[::-1, :])


def test_feather_is_bounded_and_peaks_at_one_in_the_middle():
    w = _box_weights((100, 100, 356, 356), 1000, 1000, 32)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert w[128, 128] == pytest.approx(1.0)
    assert w[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_feather_leaves_frame_edges_at_full_weight():
    """Fading an edge with no neighbour would leave a dark rim on the plate."""
    w = _box_weights((0, 0, 256, 256), 1000, 1000, 32)
    # Left and top meet the frame, so nothing along them is faded: each row/column
    # already carries the largest weight it can.
    assert np.allclose(w[:, 0], w.max(axis=1))
    assert np.allclose(w[0, :], w.max(axis=0))
    assert w[0, 0] == pytest.approx(1.0)
    # The right and bottom edges face a neighbouring tile, so they do feather.
    assert w[0, -1] == pytest.approx(0.0, abs=1e-12)
    assert w[-1, 0] == pytest.approx(0.0, abs=1e-12)


def test_a_tile_that_is_the_whole_frame_is_unfeathered():
    w = _box_weights((0, 0, 64, 48), 64, 48, 16)
    assert np.allclose(w, 1.0)


def test_feather_weights_form_a_partition_of_unity_once_normalised():
    """Normalised overlap weights must sum to 1 at every pixel.

    That is exactly what keeps overlapping regions from reading brighter than
    their neighbours; the assembly step relies on it.
    """
    width, height, side, ramp = 700, 500, 256, 32
    total = np.zeros((height, width), dtype=np.float64)
    for box in tile_boxes(width, height, side, 0.25):
        x0, y0, x1, y1 = box
        total[y0:y1, x0:x1] += _box_weights(box, width, height, ramp)
    assert (total > 1e-9).all(), "some pixel carries zero total weight"


def test_zero_ramp_disables_feathering():
    w = _box_weights((100, 100, 356, 356), 1000, 1000, 0)
    assert np.allclose(w, 1.0)


# ------------------------------------------------------------------ anchoring


def test_affine_recovers_a_known_scale_and_offset():
    reference = np.linspace(1.0, 10.0, 64 * 64).reshape(64, 64)
    tile = (reference - 2.0) / 3.0            # reference = 3*tile + 2
    a, b = fit_affine_to_reference(tile, reference, np)
    assert a == pytest.approx(3.0, abs=1e-6)
    assert b == pytest.approx(2.0, abs=1e-6)


def test_affine_is_identity_for_an_already_matching_tile():
    reference = np.linspace(1.0, 5.0, 1024).reshape(32, 32)
    a, b = fit_affine_to_reference(reference, reference, np)
    assert a == pytest.approx(1.0, abs=1e-9)
    assert b == pytest.approx(0.0, abs=1e-9)


def test_affine_ignores_non_finite_and_non_positive_samples():
    reference = np.linspace(1.0, 10.0, 4096).reshape(64, 64)
    tile = (reference - 2.0) / 3.0
    poisoned = tile.copy()
    poisoned[0, :8] = np.nan
    poisoned[1, :8] = -5.0                    # non-positive samples are dropped
    a, b = fit_affine_to_reference(poisoned, reference, np)
    assert a == pytest.approx(3.0, abs=1e-6)
    assert b == pytest.approx(2.0, abs=1e-6)


def test_affine_declines_to_fit_on_too_few_samples():
    reference = np.full((8, 8), np.nan)
    reference[0, :3] = [1.0, 2.0, 3.0]
    tile = np.full((8, 8), np.nan)
    tile[0, :3] = [2.0, 4.0, 6.0]
    assert fit_affine_to_reference(tile, reference, np) == (1.0, 0.0)


def test_affine_min_samples_threshold_is_honoured():
    reference = np.linspace(1.0, 10.0, 100).reshape(10, 10)
    tile = reference / 2.0
    assert fit_affine_to_reference(tile, reference, np, min_samples=200) == (1.0, 0.0)
    a, _ = fit_affine_to_reference(tile, reference, np, min_samples=10)
    assert a == pytest.approx(2.0, abs=1e-6)


def test_affine_falls_back_to_a_pure_offset_for_a_flat_tile():
    """A constant tile makes the normal equations singular."""
    reference = np.full((32, 32), 5.0)
    tile = np.full((32, 32), 2.0)
    a, b = fit_affine_to_reference(tile, reference, np)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(3.0)


def test_affine_rejects_a_negative_scale():
    """Depth cannot be flipped; a negative slope means the fit is garbage."""
    reference = np.linspace(1.0, 5.0, 1024).reshape(32, 32)
    assert fit_affine_to_reference(-reference, reference, np) == (1.0, 0.0)


# ------------------------------------------------------------------ assembly


def _crop_tiles(source, boxes, scale=None):
    out = []
    for k, b in enumerate(boxes):
        crop = source[b[1]:b[3], b[0]:b[2]].astype(np.float64).copy()
        if scale is not None:
            crop = crop * scale(k)
        out.append((b, crop))
    return out


def test_assembly_reconstructs_a_known_image_exactly():
    w, h, side = 640, 480, 256
    yy, xx = np.mgrid[0:h, 0:w]
    truth = 2.0 + 0.01 * xx + 0.003 * yy + np.sin(xx / 37.0)
    boxes = tile_boxes(w, h, side, 0.25)
    out = assemble_tiles(_crop_tiles(truth, boxes), w, h, 32, np)
    assert out.dtype == np.float32
    assert out.shape == (h, w)
    assert np.abs(out - truth).max() < 1e-4


def test_assembly_of_a_constant_field_is_flat():
    """The seam test: weights must be normalised or overlaps read brighter."""
    w, h = 600, 400
    boxes = tile_boxes(w, h, 256, 0.25)
    out = assemble_tiles(_crop_tiles(np.full((h, w), 7.0), boxes), w, h, 32, np)
    assert np.isfinite(out).all()
    assert np.allclose(out, 7.0, atol=1e-6)


def test_assembly_leaves_no_seam_spike_in_a_gradient():
    w, h = 640, 320
    truth = np.tile(np.linspace(1.0, 9.0, w), (h, 1))
    boxes = tile_boxes(w, h, 256, 0.25)
    out = assemble_tiles(_crop_tiles(truth, boxes), w, h, 32, np).astype(np.float64)
    dx = np.abs(np.diff(out, axis=1))
    assert dx.max() < dx.mean() * 1.5 + 1e-6


def test_assembly_frame_edges_are_not_faded():
    w, h = 600, 400
    boxes = tile_boxes(w, h, 256, 0.25)
    out = assemble_tiles(_crop_tiles(np.full((h, w), 4.0), boxes), w, h, 48, np)
    for edge in (out[0, :], out[-1, :], out[:, 0], out[:, -1]):
        assert np.allclose(edge, 4.0, atol=1e-6)


def test_assembly_drops_invalid_samples_instead_of_poisoning_the_blend():
    w, h = 600, 400
    boxes = tile_boxes(w, h, 256, 0.25)
    tiles = []
    for k, b in enumerate(boxes):
        tile = np.full((b[3] - b[1], b[2] - b[0]), 3.0)
        if k == 0:
            tile[:] = np.nan
        tiles.append((b, tile))
    out = assemble_tiles(tiles, w, h, 32, np)
    covered = np.isfinite(out)
    assert covered.any()
    assert np.allclose(out[covered], 3.0, atol=1e-6)


def test_uncovered_pixels_come_back_as_nan():
    """Region no tile touches stays NaN — tiling never invents coverage.

    Feathering is off here, because a partial cover's inner edge legitimately
    ramps to zero weight and would read as uncovered too.
    """
    out = assemble_tiles([((0, 0, 10, 10), np.full((10, 10), 2.0))], 20, 10, 0, np)
    assert np.allclose(out[:, :10], 2.0)
    assert np.isnan(out[:, 10:]).all()


def test_unanchored_tiles_seam_which_is_why_the_affine_fit_exists():
    w, h = 512, 256
    truth = np.tile(np.linspace(2.0, 6.0, w), (h, 1))
    boxes = tile_boxes(w, h, 256, 0.25)
    tiles = _crop_tiles(truth, boxes, scale=lambda k: 1.8 if k % 2 else 1.0)
    naive = assemble_tiles(tiles, w, h, 32, np)
    assert np.abs(naive - truth).max() > 0.5, (
        "unanchored tiles blended cleanly — the anchoring step is no longer needed")

    fixed = []
    for box, tile in tiles:
        a, b = fit_affine_to_reference(tile, truth[box[1]:box[3], box[0]:box[2]], np)
        fixed.append((box, tile * a + b))
    assert np.abs(assemble_tiles(fixed, w, h, 32, np) - truth).max() < 1e-4


# ------------------------------------------------------------------- contract


def test_numpy_may_be_omitted_and_is_resolved_internally():
    """``core`` is host-agnostic: numpy is imported lazily, not at module scope."""
    reference = np.linspace(1.0, 5.0, 1024).reshape(32, 32)
    assert fit_affine_to_reference(reference / 2.0, reference)[0] == pytest.approx(2.0)
    out = assemble_tiles([((0, 0, 8, 8), np.full((8, 8), 1.5))], 8, 8, 2)
    assert np.allclose(out, 1.5)


def test_module_imports_no_host_dependency_at_module_scope():
    import ast
    from pathlib import Path

    import atlas_camera.core.image_tiling as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    banned = {"torch", "transformers", "comfy", "folder_paths", "numpy", "cv2"}
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert not banned.intersection(names), f"module-scope import of {names}"


def test_depth_estimator_still_re_exports_the_tiling_names():
    """Existing callers import these from the inference module; keep that working."""
    from atlas_camera.core import image_tiling
    from atlas_camera.inference import depth_estimator as de

    for name in ("tile_boxes", "_feather_weights", "fit_affine_to_reference",
                 "assemble_tiles"):
        assert getattr(de, name) is getattr(image_tiling, name)
