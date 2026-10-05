"""Photo / generated-colour seam on a UV-unwrapped generated object.

Found live 2026-10-05 (Pixal3D machine, 198k faces, 2048 UV texture): the seam
was hard and jagged. Three causes, each pinned here:

* a UV unwrap SPLITS vertices along chart seams, so face-adjacency smoothing
  never crossed a seam -> per-chart weights (jaggies on chart borders);
* a fixed 3-iteration smooth shrinks with mesh density -> one triangle row;
* one global gain leaves the model's local hue drift showing at the seam.
"""

from __future__ import annotations

import numpy as np

from atlas_camera.core.generated_mesh import (
    local_colour_gains,
    smoothing_iterations_for,
    weld_index,
)


def _grid(n, split_col=None):
    """Unit plane in XY with n x n quads; ``split_col`` duplicates that column
    of vertices so the two halves share NO vertex (a UV-seam split)."""
    xs = np.linspace(0, 1, n + 1)
    verts = [(x, y, 0.0) for y in xs for x in xs]
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    faces = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]
            faces += [(a, b, c), (a, c, d)]
    verts = np.asarray(verts, float)
    faces = np.asarray(faces)
    if split_col is not None:
        dup = idx[:, split_col]
        new = np.arange(len(verts), len(verts) + len(dup))
        verts = np.vstack([verts, verts[dup]])
        remap = dict(zip(dup.tolist(), new.tolist()))
        cx = verts[faces].mean(axis=1)[:, 0]
        right = cx > xs[split_col]
        faces = faces.copy()
        for fi in np.flatnonzero(right):
            faces[fi] = [remap.get(int(v), int(v)) for v in faces[fi]]
    return verts, faces


def test_weld_index_merges_a_split_uv_seam():
    v, f = _grid(8, split_col=4)
    inv, n = weld_index(v)
    assert n == 81 and len(v) == 90
    dup = np.arange(81, 90)
    assert len(set(inv[dup])) == 9 and set(inv[dup]) <= set(inv[:81])


def test_smoothing_scales_with_mesh_density_not_a_fixed_count():
    coarse = smoothing_iterations_for(*_grid(10), 0.2)
    fine = smoothing_iterations_for(*_grid(40), 0.2)
    assert coarse > 3                  # above the floor, so the ratio means something
    assert 10 * coarse < fine <= 400   # ~ (4x density)^2 = 16x


def _local(v, f, *, seen_frac=0.5):
    n = len(v)
    vc = np.full((n, 3), 0.5)
    plate = vc.copy()
    seen = v[:, 0] < seen_frac
    lin = ((0.5 + 0.055) / 1.055) ** 2.4
    plate[seen, 0] = ((lin * 1.5) ** (1 / 2.4)) * 1.055 - 0.055     # photo 1.5x redder
    w = seen.astype(float)
    return local_colour_gains(v, f, vc, w, plate, global_gain=(1.0, 1.0, 1.0),
                              reach_frac=0.15)


def test_local_gain_matches_at_the_seam_and_fades_to_global_far_away():
    v, f = _grid(40)
    g, st = _local(v, f)
    at_seam = np.abs(v[:, 0] - 0.55) < 0.02
    far = v[:, 0] > 0.97
    assert st["seen_vertices"] > 0
    assert np.median(g[at_seam, 0]) > 1.3              # the local red shift carries over
    assert np.median(g[far, 0]) < np.median(g[at_seam, 0])
    assert np.allclose(g[:, 1], 1.0, atol=1e-6)        # green untouched


def test_local_gain_is_continuous_across_a_split_seam():
    v, f = _grid(40, split_col=30)
    g, _ = _local(v, f)
    inv, _ = weld_index(v)
    # every duplicated seam vertex gets exactly its twin's gain
    for k in range(len(v)):
        twins = np.flatnonzero(inv == inv[k])
        assert np.allclose(g[twins], g[k])


def test_colour_gain_range_allows_a_dusk_plate_but_still_flags_a_wild_one():
    """Found live 2026-10-05 (black SUV, dusk RAW): the needed gain was < 0.5
    and the old (0.5, 2.0) clamp left the hidden side ~2x too bright."""
    from atlas_camera.core.generated_mesh import match_vertex_colours
    from atlas_camera.core.srgb import linear_to_srgb, srgb_to_linear

    n = 64
    vc = np.full((n, 3), 0.6)
    w = np.ones(n)
    for gain, clamped in ((0.4, False), (0.1, True)):
        plate = linear_to_srgb(srgb_to_linear(vc) * gain)
        _, rep = match_vertex_colours(vc, w, plate)
        assert rep["clamped"] == [clamped] * 3
        if not clamped:
            assert np.allclose(rep["gain"], gain, atol=1e-3)
