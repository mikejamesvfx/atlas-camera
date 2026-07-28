"""Native-resolution tiled depth inference.

A monocular model resamples its input to a fixed token budget, so a 36 MP plate
is effectively inferred at a fraction of its resolution and the fine structure is
gone before anything downstream can use it. Tiling runs the model over crops at
source scale, which raises effective resolution with no training and no new
model.

THE PART THAT IS NOT OBVIOUS: you cannot simply paste the tiles together.
Monocular depth is scale- and shift-ambiguous per input, so a tile of sky and a
tile of pavement come back on different scales even from a metric model. Pasting
them puts a step at every seam, and feathering only turns a hard step into a soft
one. Every tile is therefore fitted onto ONE global low-resolution pass first, so
they share a frame of reference before blending.

These tests use a stub model: the arithmetic is what silently goes wrong, and it
needs no GPU to check.
"""
from __future__ import annotations

import pytest

from atlas_camera.inference.depth_estimator import (
    assemble_tiles,
    fit_affine_to_reference,
    tile_boxes,
)

np = pytest.importorskip("numpy")


# ------------------------------------------------------------------- layout


def test_tiles_cover_every_pixel():
    w, h, side = 900, 640, 256
    boxes = tile_boxes(w, h, side, 0.25)
    hit = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        hit[y0:y1, x0:x1] = True
    assert hit.all(), f"{(~hit).sum()} pixels uncovered by {len(boxes)} tiles"


def test_tiles_are_full_size_not_a_ragged_remainder():
    """No thin final strip.

    A sliver tile would get its own inferred scale from almost no context — the
    worst possible input to a monocular model — and shows up as a bright or dark
    band down one edge. Tiles are spread evenly instead.
    """
    boxes = tile_boxes(1000, 700, 256, 0.25)
    for x0, y0, x1, y1 in boxes:
        assert (x1 - x0) == 256 or x1 == 1000
        assert (y1 - y0) == 256 or y1 == 700
    widths = {x1 - x0 for x0, _, x1, _ in boxes}
    assert min(widths) > 256 * 0.5, f"a sliver tile got through: widths {sorted(widths)}"


def test_single_tile_when_the_image_already_fits():
    assert tile_boxes(200, 150, 256, 0.25) == [(0, 0, 200, 150)]


def test_overlap_actually_overlaps():
    boxes = tile_boxes(1000, 300, 256, 0.25)
    xs = sorted({b[0] for b in boxes})
    assert len(xs) > 1
    assert xs[1] - xs[0] < 256, "consecutive tiles must share pixels to blend across"


# ------------------------------------------------------------------ anchoring


def test_affine_recovers_a_known_scale_and_shift():
    ref = np.linspace(1.0, 10.0, 64 * 64).reshape(64, 64)
    tile = ref / 3.0 - 2.0 / 3.0          # i.e. ref = 3*tile + 2
    a, b = fit_affine_to_reference(tile, ref, np)
    assert a == pytest.approx(3.0, abs=1e-6)
    assert b == pytest.approx(2.0, abs=1e-6)


def test_affine_declines_to_fit_on_too_few_samples():
    """Better an unadjusted tile than one warped by a fit on ten pixels."""
    ref = np.full((8, 8), np.nan)
    ref[0, :3] = [1.0, 2.0, 3.0]
    tile = np.full((8, 8), np.nan)
    tile[0, :3] = [2.0, 4.0, 6.0]
    assert fit_affine_to_reference(tile, ref, np) == (1.0, 0.0)


def test_affine_handles_a_flat_tile_without_exploding():
    """A degenerate (constant) tile makes the normal equations singular.

    Fitting anyway yields an enormous slope that would blow the tile out of
    range; falling back to a pure offset keeps it sane.
    """
    ref = np.full((32, 32), 5.0)
    tile = np.full((32, 32), 2.0)
    a, b = fit_affine_to_reference(tile, ref, np)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(3.0)
    assert abs(a) < 10.0


def test_affine_rejects_a_negative_scale():
    """A depth map cannot be flipped; a negative slope means the fit is garbage."""
    ref = np.linspace(1.0, 5.0, 1024).reshape(32, 32)
    a, b = fit_affine_to_reference(-ref, ref, np)
    assert (a, b) == (1.0, 0.0)


# ------------------------------------------------------------------ blending


def test_assembly_of_a_constant_field_is_flat():
    """The seam test. Weights must be normalised, or overlaps read brighter."""
    w, h, side = 600, 400, 256
    boxes = tile_boxes(w, h, side, 0.25)
    tiles = [(b, np.full((b[3] - b[1], b[2] - b[0]), 7.0)) for b in boxes]
    out = assemble_tiles(tiles, w, h, ramp=32, np=np)
    assert np.isfinite(out).all()
    assert np.allclose(out, 7.0, atol=1e-6), (
        f"constant field came back with spread {out.min():.4f}..{out.max():.4f} — "
        "the overlap weights are not normalised")


def test_assembly_preserves_a_gradient_without_seams():
    """A smooth ramp must survive tiling. Steps at seams show up as gradient spikes."""
    w, h, side = 640, 320, 256
    ramp_full = np.tile(np.linspace(1.0, 9.0, w), (h, 1))
    boxes = tile_boxes(w, h, side, 0.25)
    tiles = [(b, ramp_full[b[1]:b[3], b[0]:b[2]].copy()) for b in boxes]
    out = assemble_tiles(tiles, w, h, ramp=32, np=np)

    assert np.allclose(out, ramp_full, atol=1e-5)
    # No local discontinuity anywhere: the true step between columns is uniform.
    dx = np.abs(np.diff(out.astype(np.float64), axis=1))
    assert dx.max() < dx.mean() * 1.5 + 1e-6, "a seam introduced a gradient spike"


def test_unanchored_tiles_would_seam_which_is_why_anchoring_exists():
    """Demonstrates the failure the anchoring step prevents.

    Two tiles carrying the SAME scene on different scales are blended without
    being fitted first. The result must be visibly wrong — if this ever passes
    cleanly, the anchoring step has stopped being load-bearing and the extra
    inference pass could be dropped.
    """
    w, h, side = 512, 256, 256
    truth = np.tile(np.linspace(2.0, 6.0, w), (h, 1))
    boxes = tile_boxes(w, h, side, 0.25)
    # Give every other tile a 1.8x scale error, as a real model would.
    tiles = []
    for k, b in enumerate(boxes):
        crop = truth[b[1]:b[3], b[0]:b[2]].copy()
        tiles.append((b, crop * (1.8 if k % 2 else 1.0)))
    naive = assemble_tiles(tiles, w, h, ramp=32, np=np)
    assert np.abs(naive - truth).max() > 0.5, (
        "unanchored tiles blended cleanly — the anchoring step is no longer needed")

    # Now anchor each tile to the reference first, and it comes back correct.
    fixed = []
    for b, tile in tiles:
        a, off = fit_affine_to_reference(tile, truth[b[1]:b[3], b[0]:b[2]], np)
        fixed.append((b, tile * a + off))
    out = assemble_tiles(fixed, w, h, ramp=32, np=np)
    assert np.abs(out - truth).max() < 1e-4


def test_frame_edges_are_not_feathered_to_zero():
    """Feathering an edge with no neighbour would leave a dark rim on the plate."""
    w, h, side = 600, 400, 256
    boxes = tile_boxes(w, h, side, 0.25)
    tiles = [(b, np.full((b[3] - b[1], b[2] - b[0]), 4.0)) for b in boxes]
    out = assemble_tiles(tiles, w, h, ramp=48, np=np)
    for edge in (out[0, :], out[-1, :], out[:, 0], out[:, -1]):
        assert np.allclose(edge, 4.0, atol=1e-6), "frame edge was faded by the feather"


def test_invalid_tile_regions_do_not_poison_the_blend():
    w, h, side = 600, 400, 256
    boxes = tile_boxes(w, h, side, 0.25)
    tiles = []
    for k, b in enumerate(boxes):
        t = np.full((b[3] - b[1], b[2] - b[0]), 3.0)
        if k == 0:
            t[:] = np.nan          # a wholly invalid tile
        tiles.append((b, t))
    out = assemble_tiles(tiles, w, h, ramp=32, np=np)
    covered = np.isfinite(out)
    assert np.allclose(out[covered], 3.0, atol=1e-6)


# ---------------------------------------------------------------- contract


def test_tiling_is_inert_by_default():
    """Existing graphs must be bit-identical until someone opts in."""
    import inspect

    from atlas_camera.inference.depth_estimator import estimate_depth

    params = inspect.signature(estimate_depth).parameters
    assert params["tile_side"].default == 0
    assert params["tile_overlap"].default == 0.25


def test_tiling_knobs_fragment_the_result_cache():
    """Two different tile settings must not return the same cached depth.

    The same bug was already fixed once for resolution_level/max_side: a cache
    keyed only on image+model returns the FIRST result forever, so a second run
    at different settings silently reports the first one's numbers.
    """
    import inspect

    from atlas_camera.inference import depth_estimator as de

    src = inspect.getsource(de.estimate_depth)

    # The tiling knobs ride in `moge_key`, which is folded into `cache_key`.
    # Assert BOTH links, so removing either is caught: a fragment nobody
    # references is exactly as broken as no fragment at all.
    frag_start = src.index("moge_key")
    frag = src[frag_start:src.index("cache_key")]
    assert "tile_side" in frag, "tile_side missing from the moge cache fragment"
    assert "tile_overlap" in frag, "tile_overlap missing from the moge cache fragment"

    key_line = src[src.index("cache_key = ("):]
    key_line = key_line[:key_line.index("\n")]
    assert "moge_key" in key_line, (
        f"the moge fragment is not part of the cache key: {key_line}")
