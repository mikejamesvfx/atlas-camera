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


class TestTheVetoCanReachThePrimary:
    """The regression this class exists for, found live 2026-07-30.

    The layer lookup used to read `== layer and layer`, so an entry could never
    match the PRIMARY — the solve's own relief mesh, whose layer key is "".
    A `fill_occluded: false` aimed at it did nothing, and said nothing while
    doing it, which is the worst combination: the operator sees a veto in the
    assessment and geometry gets built anyway.

    Two routes to a primary veto, because the primary is not one of the VLM's
    named projection sources: a top-level flag for the whole scene, and an
    explicit empty-name entry for the primary alone.
    """

    def test_a_scene_wide_veto_stops_the_primary(self):
        payload = _payload("forests")
        payload["fill_occluded"] = False
        plan = policy_from_assessment(payload)
        assert plan.route == ROUTE_NONE
        assert not plan.builds_geometry
        assert any("SCENE" in r for r in plan.reasons)

    def test_a_scene_wide_veto_stops_every_NAMED_layer_too(self):
        payload = _payload("outdoor", [{"name": "sky", "role": "sky"},
                                       {"name": "mid"}])
        payload["fill_occluded"] = False
        plans = plans_for_layers(payload)
        assert {p.route for p in plans.values()} == {ROUTE_NONE}, (
            "sky has its own early return; a scene veto must outrank it")

    def test_an_explicit_empty_name_entry_vetoes_the_PRIMARY_alone(self):
        payload = _payload("forests", [{"name": "", "fill_occluded": False},
                                       {"name": "mid"}])
        plans = plans_for_layers(payload)
        assert plans[""].route == ROUTE_NONE
        assert plans["mid"].route == ROUTE_ORGANIC, (
            "vetoing the primary must not veto the named layers")

    def test_the_old_guard_would_fail_this(self):
        """Pins the exact shape the bug hid behind: an entry keyed "" reaching
        the scene-level call. Under `== layer and layer` this returned ORGANIC.
        """
        plan = policy_from_assessment(
            _payload("forests", [{"name": "", "fill_occluded": False}]))
        assert plan.route == ROUTE_NONE

    def test_a_NAMELESS_entry_does_not_become_the_primary_by_accident(self):
        """A dropped name is malformed input, not a primary veto. Matching on
        the value alone would let junk silently govern the whole scene."""
        plan = policy_from_assessment(
            _payload("forests", [{"fill_occluded": False}]))
        assert plan.route == ROUTE_ORGANIC

    def test_a_true_at_scene_level_does_not_upgrade_anything(self):
        payload = _payload("forests")
        payload["fill_occluded"] = True
        assert policy_from_assessment(payload).route == ROUTE_ORGANIC


class TestAMissingLayerIsReportedNotAssumed:
    """The same class of trap as the veto gap: asking for a layer that is not in
    the assessment used to fall through to the scene plan in silence. That is a
    wiring mistake producing a plan nobody chose for that layer."""

    def test_an_unknown_layer_name_says_so(self):
        plan = policy_from_assessment(
            _payload("forests", [{"name": "mid"}]), layer="typo")
        assert any("not in the assessment" in r for r in plan.reasons)
        assert any("mid" in r for r in plan.reasons), (
            "the reason must list what WAS available, or the operator cannot "
            "find the typo")

    def test_it_still_returns_a_usable_plan(self):
        """Reporting, not raising — a 20-minute graph must not die on a typo."""
        plan = policy_from_assessment(
            _payload("forests", [{"name": "mid"}]), layer="typo")
        assert plan.route == ROUTE_ORGANIC

    def test_a_matched_layer_is_not_accused_of_being_missing(self):
        plan = policy_from_assessment(
            _payload("forests", [{"name": "mid"}]), layer="mid")
        assert not any("not in the assessment" in r for r in plan.reasons)

    def test_no_layers_at_all_is_not_a_typo(self):
        """An assessment with no layer breakdown is normal; only a name that
        misses an existing list is a mistake worth naming."""
        plan = policy_from_assessment(_payload("forests"), layer="primary")
        assert not any("not in the assessment" in r for r in plan.reasons)


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
