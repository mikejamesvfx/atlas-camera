"""Contract for core.fill_policy — routing a hole to the right fill machinery.

Two families that must stay disjoint: planar scenes get a fitted construction
licensed per occlusion node; organic scenes get no fit at all, just wider bands,
mesh hole-fill and retopology. Organic geometry is forgiving, so an approximate
fill reads as foliage; the same approximation on a wall reads as broken, and a
plane forced through a hillside is confidently, structurally false.

The VLM picks the ROUTE and never a value. Every test below is really asking one
of two questions: does an inferred signal stay inside its authority, and can the
two families ever contaminate each other.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.fill_policy import (
    ROUTE_NONE,
    ROUTE_ORGANIC,
    ROUTE_PLANAR,
    _SCENE_ROUTES,
    FillPlan,
    plans_for_layers,
    policy_from_assessment,
)
from atlas_camera.core.occlusion_graph import (
    POLICY_BACKDROP,
    POLICY_EXTEND_PLANE,
    POLICY_EXTRUDE_PROFILE,
    POLICY_NONE,
    POLICY_ROOM_ENVELOPE,
)

PLANAR = ("indoor", "simple_walls", "outdoor", "aerial", "towers_spires")
ORGANIC = ("organic", "mountains", "forests")


def _payload(scene_type="organic", layers=None):
    return {
        "recommended_settings": {"scene_type": scene_type},
        "layers": layers or [],
    }


class TestTheTwoFamiliesAreDisjoint:
    """The property the whole module exists to guarantee."""

    @pytest.mark.parametrize("scene_type", ORGANIC)
    def test_an_organic_scene_never_gets_a_plane_policy(self, scene_type):
        plan = policy_from_assessment(_payload(scene_type))
        assert plan.route == ROUTE_ORGANIC
        assert plan.policy == POLICY_NONE, (
            "a fitted plane through organic geometry is the failure this "
            "routing exists to prevent")

    @pytest.mark.parametrize("scene_type", PLANAR)
    def test_a_planar_scene_never_gets_mesh_fill_settings(self, scene_type):
        plan = policy_from_assessment(_payload(scene_type))
        assert plan.route == ROUTE_PLANAR
        assert plan.policy != POLICY_NONE
        assert not plan.live_fill_holes
        assert plan.retopo_method == "off"
        assert plan.band_widening == 1.0, (
            "band widening is an organic-route trick; widening bands on "
            "architecture merges surfaces that should stay separate")

    def test_every_known_scene_type_lands_in_exactly_one_family(self):
        for scene_type, (route, policy) in _SCENE_ROUTES.items():
            assert route in (ROUTE_PLANAR, ROUTE_ORGANIC)
            if route == ROUTE_ORGANIC:
                assert policy == POLICY_NONE
            else:
                assert policy != POLICY_NONE

    def test_the_vocabulary_matches_the_REAL_scene_type_table(self):
        """Cross-module pin against the authority, not against myself.

        The first version of this test compared _SCENE_ROUTES to constants
        defined in this file FROM that same mapping — tautological, and it would
        have passed while the two vocabularies silently diverged. The authority
        is AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS, which is what the
        derive nodes actually branch on and what the assessor prompt lists.

        A new scene type added there and not here would fall through to the
        organic route, which is safe but silent; this makes it loud instead.
        """
        from atlas_camera.comfy.nodes_geometry import (
            AtlasDeriveProjectionGeometry as Derive)
        assert set(_SCENE_ROUTES) == set(Derive._SCENE_TYPE_PRESETS), (
            "fill_policy and the derive nodes disagree about the scene-type "
            "vocabulary")

    def test_the_manual_escape_hatch_is_not_treated_as_a_scene(self):
        """`manual` appears in the node's combo but not in the presets — it
        means 'use my widgets', and is never a VLM output. It must fall through
        to the safe route rather than matching something by accident."""
        assert policy_from_assessment(_payload("manual")).route == ROUTE_ORGANIC


class TestPlanarMapping:
    @pytest.mark.parametrize("scene_type,expected", [
        ("indoor", POLICY_ROOM_ENVELOPE),
        ("simple_walls", POLICY_EXTEND_PLANE),
        ("outdoor", POLICY_EXTEND_PLANE),
        ("aerial", POLICY_EXTEND_PLANE),
        ("towers_spires", POLICY_EXTRUDE_PROFILE),
    ])
    def test_each_planar_type_maps_to_its_derive_nodes_construction(
            self, scene_type, expected):
        assert policy_from_assessment(_payload(scene_type)).policy == expected


class TestOrganicTuning:
    def test_forests_are_the_most_forgiving_and_get_the_loosest_budget(self):
        """Noisiest depth, most forgiving geometry — the ordering should hold
        across all three organic types rather than being one hand-set number."""
        plans = [policy_from_assessment(_payload(s)) for s in
                 ("organic", "mountains", "forests")]
        edges = [p.live_fill_max_hole_edges for p in plans]
        widening = [p.band_widening for p in plans]
        smoothing = [p.boundary_smooth_iterations for p in plans]
        assert edges == sorted(edges)
        assert widening == sorted(widening)
        assert smoothing == sorted(smoothing)

    def test_organic_enables_the_whole_mesh_route(self, ):
        plan = policy_from_assessment(_payload("mountains"))
        assert plan.live_fill_holes
        assert plan.live_fill_edge_sawteeth, "organic tears are ragged"
        assert plan.retopo_method != "off"
        assert plan.band_widening > 1.0


class TestInferredSignalsCanOnlyReduce:
    def test_fill_occluded_false_is_an_absolute_veto(self):
        """Strongest precedence, and deliberately one-directional: a wrong veto
        costs a tear, a wrong licence costs invented geometry that looks
        deliberate."""
        for scene_type in PLANAR + ORGANIC:
            plan = policy_from_assessment(
                _payload(scene_type, [{"name": "L", "fill_occluded": False}]),
                layer="L")
            assert plan.route == ROUTE_NONE
            assert plan.policy == POLICY_NONE
            assert not plan.builds_geometry
            assert not plan.live_fill_holes

    def test_fill_occluded_true_does_not_upgrade_anything(self):
        """It must not turn an organic scene planar, or vice versa."""
        a = policy_from_assessment(_payload("forests"))
        b = policy_from_assessment(
            _payload("forests", [{"name": "L", "fill_occluded": True}]),
            layer="L")
        assert a.route == b.route == ROUTE_ORGANIC
        assert a.policy == b.policy == POLICY_NONE

    def test_a_missing_fill_occluded_is_not_read_as_a_veto(self):
        plan = policy_from_assessment(
            _payload("indoor", [{"name": "L"}]), layer="L")
        assert plan.route == ROUTE_PLANAR


class TestSkyAndUnknowns:
    def test_sky_is_a_backdrop_not_a_fitted_surface(self):
        plan = policy_from_assessment(
            _payload("outdoor", [{"name": "sky", "role": "sky"}]), layer="sky")
        assert plan.policy == POLICY_BACKDROP

    def test_the_veto_outranks_sky(self):
        plan = policy_from_assessment(
            _payload("outdoor",
                     [{"name": "sky", "role": "sky", "fill_occluded": False}]),
            layer="sky")
        assert plan.route == ROUTE_NONE

    @pytest.mark.parametrize("scene_type", ["", "whatever", "Indoor Scene"])
    def test_an_unknown_scene_type_takes_the_ORGANIC_route(self, scene_type):
        """Both guesses are wrong sometimes; this one is recoverable. A soft
        fill on architecture reads as sloppy; a plane forced through foliage is
        confidently, structurally false."""
        plan = policy_from_assessment(_payload(scene_type))
        assert plan.route == ROUTE_ORGANIC
        assert plan.policy == POLICY_NONE
        assert any("not in the known vocabulary" in r for r in plan.reasons)

    def test_an_empty_payload_does_not_explode(self):
        for bad in ({}, {"recommended_settings": None}, {"layers": None}):
            assert policy_from_assessment(bad).route == ROUTE_ORGANIC


class TestExplainability:
    def test_every_plan_carries_its_reasoning(self):
        for scene_type in PLANAR + ORGANIC:
            plan = policy_from_assessment(_payload(scene_type))
            assert plan.reasons, f"{scene_type} produced no reason"
            assert plan.describe()

    def test_the_planar_reason_names_the_remaining_gate(self):
        """The VLM licenses; it does not decide. The plan must say so, or a
        reader will take a policy as a guarantee geometry got built."""
        plan = policy_from_assessment(_payload("indoor"))
        assert any("fit gate" in r for r in plan.reasons)


class TestPerLayerPlans:
    def test_layers_are_planned_independently(self):
        payload = _payload("outdoor", [
            {"name": "sky", "role": "sky"},
            {"name": "hero", "fill_occluded": False},
            {"name": "mid"},
        ])
        plans = plans_for_layers(payload)
        assert set(plans) == {"", "sky", "hero", "mid"}
        assert plans["sky"].policy == POLICY_BACKDROP
        assert plans["hero"].route == ROUTE_NONE
        assert plans["mid"].policy == POLICY_EXTEND_PLANE

    def test_the_scene_wide_plan_is_keyed_empty(self):
        plans = plans_for_layers(_payload("forests"))
        assert plans[""].route == ROUTE_ORGANIC
        assert isinstance(plans[""], FillPlan)
