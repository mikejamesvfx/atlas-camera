"""Contract for core.view_solver — "which angles actually see this hole".

Built by extracting the visibility computation that path-guided repair had
hardwired to one camera-path frame, so the Qwen multi-angle patch, the iPhone
shoot list and clean-plate selection can all ask the same question instead of
each guessing separately.

The fixture is deliberately the SAME occluded-slab geometry
tests/test_path_hole_repair.py uses: if the two modules ever disagree about what
an island is, that is the drift `build_island_candidates` exists to prevent, and
these tests should be the ones that notice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_camera.core.view_solver import (  # noqa: E402
    CandidateView,
    IslandVisibility,
    ViewScore,
    best_view_per_island,
    rank_views,
)


#: The repair module's own fit settings. The PathHoleRepairConfig defaults
#: (max_hole_fraction=0.04) reject this fixture's holes outright, which would
#: leave every test scoring zero and passing vacuously.
FIT = None


@pytest.fixture(scope="module")
def scene():
    """Mesh + source camera + hole mask, from the repair module's own fixture."""
    global FIT
    from atlas_camera.core.path_hole_repair import PathHoleRepairConfig
    from test_path_hole_repair import _fixture
    mesh, camera, _path = _fixture()
    FIT = PathHoleRepairConfig(
        normal_tolerance_deg=15.0, max_plane_error_m=0.02,
        max_hole_fraction=0.20,
    )
    hole = np.asarray(mesh.hole_mask, dtype=bool)
    assert hole.any(), "fixture produced no holes — every test would be vacuous"
    return mesh, camera, hole


def _views(*triples):
    return [CandidateView(a, e, d, label=f"az{a:+g} el{e:+g} x{d:g}")
            for a, e, d in triples]


class TestRankingContract:
    def test_every_candidate_is_returned_including_the_useless_ones(self, scene):
        """A view that sees nothing is an ANSWER, not an omission.

        "No candidate angle reveals this island" is what should route a hole to a
        real capture or a clean plate instead of to Qwen. Dropping zero-scoring
        views would make that indistinguishable from "not evaluated".
        """
        mesh, camera, hole = scene
        cands = _views((0, 0, 1.0), (45, 0, 1.0), (-45, 0, 1.0), (180, 0, 1.0))
        scores = rank_views(mesh, hole, source_camera=camera, candidates=cands,
                            resolution=192, config=FIT)
        assert len(scores) == len(cands)
        assert {type(s) for s in scores} == {ViewScore}

    def test_results_are_sorted_best_first(self, scene):
        mesh, camera, hole = scene
        scores = rank_views(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            candidates=_views((0, 0, 1.0), (30, 0, 1.0), (60, 0, 1.0),
                              (-30, 0, 1.0), (180, 0, 1.0)))
        px = [s.visible_px for s in scores]
        assert px == sorted(px, reverse=True)

    def test_an_empty_candidate_list_is_an_empty_ranking(self, scene):
        mesh, camera, hole = scene
        assert rank_views(mesh, hole, source_camera=camera, candidates=[]) == []

    def test_a_mismatched_hole_mask_is_rejected_loudly(self, scene):
        """Silently resizing would score a hole that is not where the caller
        thinks it is."""
        mesh, camera, _hole = scene
        with pytest.raises(ValueError, match="does not match source camera"):
            rank_views(mesh, np.zeros((7, 9), dtype=bool),
                       source_camera=camera, candidates=_views((0, 0, 1.0)))


class TestOcclusionIsRespected:
    def test_the_support_mesh_occludes_candidate_planes(self, scene):
        """The load-bearing property. If the support mesh were not rasterized
        first, every island would score as fully visible from every angle and the
        ranking would be meaningless — which is exactly the failure that made
        artist-picked angles unreliable.
        """
        mesh, camera, hole = scene
        scores = rank_views(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            candidates=_views((0, 0, 1.0), (50, 0, 1.0), (-50, 0, 1.0)))
        totals = {s.view.label: s.visible_px for s in scores}
        assert len(set(totals.values())) > 1, (
            f"all views scored identically ({totals}) — occluder is not being "
            "rasterized, so visibility is not actually being measured")

    def test_a_source_visible_gap_scores_BEST_from_the_source_camera(self, scene):
        """Measured, and it corrects an intuition worth writing down.

        I first asserted the opposite — "a hole is by definition not visible from
        the camera that made it" — and it failed 502 > 900. Wrong premise: these
        holes are see-through GAPS in the mesh, so the candidate plane filling one
        faces the source camera squarely and is fully visible from it. Orbiting
        away only makes that plane grazing and partly occluded.

        The consequence is a routing rule, not a curiosity: a source-visible gap
        does not need a patch ANGLE at all — the source plate already sees it, and
        sending it to Qwen for a novel view would be pure invention where real
        pixels exist. Candidate angles earn their keep on geometry hidden BEHIND
        an occluder, where the source count is what drops to zero.
        """
        mesh, camera, hole = scene
        scores = {s.view.label: s for s in rank_views(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            candidates=_views((0, 0, 1.0), (55, 0, 1.0), (-55, 0, 1.0)))}
        on_axis = scores["az+0 el+0 x1"].visible_px
        assert on_axis > 0
        for label in ("az+55 el+0 x1", "az-55 el+0 x1"):
            assert scores[label].visible_px < on_axis, (
                f"{label} beat the source view on a see-through gap — either the "
                "fixture's holes are now genuinely occluded, or the occluder is "
                "not being rasterized")

    def test_grazing_angles_score_lower_than_moderate_ones(self, scene):
        """Monotonic falloff, which is what makes the ranking meaningful rather
        than an arbitrary ordering of near-equal numbers."""
        mesh, camera, hole = scene
        scores = {s.view.label: s.visible_px for s in rank_views(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            candidates=_views((20, 0, 1.0), (55, 0, 1.0), (80, 0, 1.0)))}
        assert scores["az+20 el+0 x1"] > scores["az+80 el+0 x1"]


class TestPerIslandReporting:
    def test_islands_are_reported_individually_not_just_a_total(self, scene):
        """A multi-angle patch needs to know WHICH island each view covers; a
        single total cannot answer that."""
        mesh, camera, hole = scene
        scores = rank_views(mesh, hole, source_camera=camera, resolution=192, config=FIT,
                            candidates=_views((55, 0, 1.0), (-55, 0, 1.0)))
        best = scores[0]
        assert best.islands_seen == len(best.islands)
        assert sum(i.visible_px for i in best.islands) == best.visible_px
        assert all(isinstance(i, IslandVisibility) for i in best.islands)

    def test_island_cells_distinguishes_small_seen_from_large_clipped(self, scene):
        """Raw pixel counts cannot: a tiny island fully revealed and a huge one
        barely clipped can report the same visible_px."""
        mesh, camera, hole = scene
        scores = rank_views(mesh, hole, source_camera=camera, resolution=192, config=FIT,
                            candidates=_views((55, 0, 1.0)))
        for item in scores[0].islands:
            assert item.island_cells > 0

    def test_sees_reports_zero_for_an_island_this_view_misses(self, scene):
        mesh, camera, hole = scene
        score = rank_views(mesh, hole, source_camera=camera, resolution=192, config=FIT,
                           candidates=_views((55, 0, 1.0)))[0]
        assert score.sees(10_000) == 0

    def test_min_visible_pixels_filters_slivers(self, scene):
        """A handful of pixels at a grazing angle is not a usable patch view."""
        mesh, camera, hole = scene
        common = dict(mesh=mesh, hole_mask=hole, source_camera=camera,
                      resolution=192, config=FIT,
                      candidates=_views((55, 0, 1.0)))
        loose = rank_views(**{**common, "min_visible_pixels": 1})[0]
        strict = rank_views(**{**common, "min_visible_pixels": 5_000})[0]
        assert strict.islands_seen <= loose.islands_seen


class TestBestViewPerIsland:
    def test_it_picks_the_highest_scoring_view_for_each_island(self, scene):
        mesh, camera, hole = scene
        scores = rank_views(mesh, hole, source_camera=camera, resolution=192, config=FIT,
                            candidates=_views((55, 0, 1.0), (-55, 0, 1.0),
                                              (0, 40, 1.0)))
        best = best_view_per_island(scores)
        for island_id, score in best.items():
            mine = score.sees(island_id)
            assert mine > 0
            assert all(s.sees(island_id) <= mine for s in scores)

    def test_an_island_no_view_sees_is_absent_not_zero(self, scene):
        """Absence is the routing signal: send it to a real capture or a clean
        plate. A zero entry would read as "Qwen can do it, badly"."""
        mesh, camera, hole = scene
        scores = rank_views(mesh, hole, source_camera=camera, resolution=192, config=FIT,
                            candidates=_views((55, 0, 1.0)))
        best = best_view_per_island(scores)
        assert all(px > 0 for px in
                   (s.sees(i) for i, s in best.items()))

    def test_no_scores_yields_no_recommendations(self):
        assert best_view_per_island([]) == {}


class TestLayering:
    def test_core_does_not_import_comfy(self):
        """`comfy/` may import anything; nothing outside it may import `comfy/`.

        The solver is vocabulary-agnostic for this reason AND a better one: Qwen
        wants 96 discrete named views, a phone shoot wants a continuous sphere,
        a clean plate wants "the same camera". Hard-coding one would serve one
        consumer and get worked around by the other two.
        """
        source = (Path(__file__).resolve().parents[1]
                  / "atlas_camera" / "core" / "view_solver.py").read_text(
                      encoding="utf-8")
        assert "atlas_camera.comfy" not in source
        assert "from atlas_camera.core.camera_math import orbit_camera" in source, (
            "placement must go through the SAME helper AtlasAddPatchView uses, "
            "or a ranked view is not the view that gets rendered")
