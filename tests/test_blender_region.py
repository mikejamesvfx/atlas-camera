"""Contract for torn-collar selection. Pure numpy; no Blender.

This is where the policy lives — Blender only does geometry. The two properties
that matter are that the plate PERIMETER is never selected (measured live: a
whole-mesh voxel remesh closed the perimeter, 974 boundary edges -> 0, turning a
matte painting into a watertight solid) and that the outermost ring is held back
as a weld anchor.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.blender.region import compact, select_torn_collar  # noqa: E402
from atlas_camera.core.mesh_repair import boundary_edges, walk_loops  # noqa: E402

N = 16  # grid vertices per side


def _grid(hole=None):
    """A flat N x N triangulated grid with UVs, optionally with a punched hole.

    UVs are normalized pixel coordinates, which is what `_perimeter_loops` reads
    to tell the plate edge from a tear.
    """
    ys, xs = np.mgrid[0:N, 0:N]
    verts = np.stack([xs.ravel().astype(float), ys.ravel().astype(float),
                      np.zeros(N * N)], axis=1)
    uvs = np.stack([xs.ravel() / (N - 1.0), ys.ravel() / (N - 1.0)], axis=1)
    faces = []
    for y in range(N - 1):
        for x in range(N - 1):
            a, b = y * N + x, y * N + x + 1
            c, d = (y + 1) * N + x, (y + 1) * N + x + 1
            if hole and hole[0] <= x < hole[1] and hole[0] <= y < hole[1]:
                continue
            faces.extend([[a, b, c], [b, d, c]])
    return verts, np.asarray(faces, dtype=np.int64), uvs


def _interior_loop_count(faces):
    loops = walk_loops(boundary_edges(np.asarray(faces)), faces=np.asarray(faces))
    return max(0, len(loops) - 1)   # largest is the perimeter


class TestPerimeterIsNeverSelected:
    def test_an_untorn_grid_selects_nothing(self):
        """The plate silhouette is a boundary loop too. Treating it as a tear
        is how you end up welding the frame edge shut."""
        v, f, uv = _grid()
        got = select_torn_collar(v, f, uv, image_width=N, image_height=N)
        assert got["tear_loops"] == []
        assert len(got["patch_faces"]) == 0
        assert got["skipped_perimeter"] >= 1, "the perimeter must be recognised"

    def test_a_punched_hole_is_found(self):
        v, f, uv = _grid(hole=(6, 10))
        assert _interior_loop_count(f) == 1
        got = select_torn_collar(v, f, uv, image_width=N, image_height=N)
        assert len(got["tear_loops"]) == 1
        assert len(got["patch_faces"]) > 0

    def test_without_uvs_it_degrades_but_does_not_crash(self):
        """`_perimeter_loops` returns an empty set with no UVs, so the caller
        must still behave — it just cannot distinguish the perimeter."""
        v, f, _uv = _grid(hole=(6, 10))
        got = select_torn_collar(v, f, None)
        assert got["skipped_perimeter"] == 0
        assert isinstance(got["patch_faces"], np.ndarray)


class TestTheAnchorRingIsHeldBack:
    def test_patch_and_anchor_do_not_overlap(self):
        """The outer ring stays in the untouched mesh; without a preserved
        anchor the returned rim has nothing to weld onto and the seam reappears
        as a new tear."""
        v, f, uv = _grid(hole=(6, 10))
        got = select_torn_collar(v, f, uv, rings=3, image_width=N, image_height=N)
        patch_verts = set(np.unique(f[got["patch_faces"]]).tolist())
        anchor = set(got["anchor_vertices"].tolist())
        assert anchor, "no anchor ring was reserved"
        assert not (patch_verts & anchor)

    def test_more_rings_selects_more(self):
        v, f, uv = _grid(hole=(6, 10))
        sizes = [len(select_torn_collar(v, f, uv, rings=r, image_width=N,
                                        image_height=N)["patch_faces"])
                 for r in (2, 3, 4)]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_the_target_is_measured_surface_OUTSIDE_the_patch(self):
        """A patch can only shrinkwrap onto geometry it was given, and snapping
        to itself is meaningless."""
        v, f, uv = _grid(hole=(6, 10))
        got = select_torn_collar(v, f, uv, rings=3, image_width=N, image_height=N)
        assert len(got["target_faces"]) > 0
        assert not (set(got["patch_faces"].tolist())
                    & set(got["target_faces"].tolist()))


class TestSizeGate:
    def test_a_tear_larger_than_max_hole_edges_is_skipped_and_counted(self):
        v, f, uv = _grid(hole=(4, 12))
        got = select_torn_collar(v, f, uv, max_hole_edges=4,
                                 image_width=N, image_height=N)
        assert got["tear_loops"] == []
        assert got["skipped_too_large"] == 1, (
            "a skipped tear must be COUNTED — silently selecting nothing reads "
            "as 'no tears found', which is a different answer")

    def test_a_generous_limit_admits_it(self):
        v, f, uv = _grid(hole=(4, 12))
        got = select_torn_collar(v, f, uv, max_hole_edges=999,
                                 image_width=N, image_height=N)
        assert len(got["tear_loops"]) == 1
        assert got["skipped_too_large"] == 0


class TestCompaction:
    def test_it_reindexes_from_zero_and_maps_back(self):
        v, f, uv = _grid(hole=(6, 10))
        got = select_torn_collar(v, f, uv, image_width=N, image_height=N)
        sub_v, sub_f, index_map = compact(v, f, got["patch_faces"])
        assert sub_f.min() == 0
        assert sub_f.max() == len(sub_v) - 1
        # index_map is what lets the result weld back onto vertices that never
        # left Atlas.
        np.testing.assert_allclose(sub_v, v[index_map])

    def test_an_empty_selection_compacts_to_empty(self):
        v, f, uv = _grid()
        sub_v, sub_f, idx = compact(v, f, np.zeros(0, dtype=np.int64))
        assert len(sub_v) == len(sub_f) == len(idx) == 0

    def test_compaction_preserves_face_count(self):
        v, f, uv = _grid(hole=(6, 10))
        got = select_torn_collar(v, f, uv, image_width=N, image_height=N)
        _sv, sub_f, _idx = compact(v, f, got["patch_faces"])
        assert len(sub_f) == len(got["patch_faces"])
