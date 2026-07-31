"""Metric scale from a known HORIZONTAL distance on the ground.

Atlas's original scale reference is a vertical object of assumed height. Some of
the best real-world constants are spans instead: railway standard gauge is
exactly 1435 mm, a specification rather than an average. A span cannot be filed
as a height — doing so would solve it as an object standing 1.435 m tall and
return a plausible, wrong camera height with nothing raised — so it needs its
own estimator and its own registry field.

The claim these tests defend is narrow and worth stating precisely: the
ground-span *geometry* is NOT better conditioned than the vertical one. It wins
only when its constant is better known. Several tests below exist specifically to
stop that being overstated later.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.solver import (  # noqa: E402
    GROUND_SPAN_MIN_PX,
    metric_height_from_ground_span,
    metric_height_from_reference,
    resolve_reference_scale,
)
from atlas_camera.reference_data import get_scale_reference  # noqa: E402

FX = FY = 3200.0
CX, CY = 1920.0, 1080.0
WIDTH = 3840


def _rig(camera_height=4.2, pitch_deg=-12.0):
    """A synthetic camera: rotation, and a projector for world points."""
    p = np.radians(pitch_deg)
    cam_to_world = np.array([[1, 0, 0],
                             [0, np.cos(p), -np.sin(p)],
                             [0, np.sin(p), np.cos(p)]])
    rotation = cam_to_world.T

    def project(P):
        Pc = rotation @ np.asarray(P, dtype=np.float64)
        z = -Pc[2]
        return (FX * Pc[0] / z + CX, -FY * Pc[1] / z + CY)

    return rotation, project, camera_height


def _span_points(project, h, span_m, distance):
    """The two endpoints of a ground span, laid across the view at `distance`."""
    a = project([-span_m / 2.0, -h, -distance])
    b = project([span_m / 2.0, -h, -distance])
    return a, b


def _solve_span(a, b, span_m, rotation, **kw):
    return metric_height_from_ground_span(
        a, b, span_m, rotation=rotation, fx=FX, fy=FY, cx=CX, cy=CY,
        image_width=WIDTH, **kw)


class TestTheFormIsExact:
    """No least squares — the algebra closes, so the answer should be exact to
    floating point, not merely close."""

    def test_it_recovers_the_camera_height(self):
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 9.0)
        got = _solve_span(a, b, 1.435, rotation)
        assert got["camera_height"] == pytest.approx(h, abs=1e-9)

    @pytest.mark.parametrize("pitch", [-3.0, -12.0, -30.0, -55.0])
    @pytest.mark.parametrize("height", [1.6, 4.2, 18.0])
    @pytest.mark.parametrize("distance", [4.0, 9.0, 40.0])
    def test_it_holds_across_the_whole_rig(self, pitch, height, distance):
        rotation, project, h = _rig(height, pitch)
        a, b = _span_points(project, h, 1.435, distance)
        got = _solve_span(a, b, 1.435, rotation)
        if got["camera_height"] is None:      # refused (short span far away)
            assert got["reason"]
            return
        assert got["camera_height"] == pytest.approx(h, rel=1e-9)

    def test_the_recovered_ground_points_lie_on_the_ground(self):
        """A sanity check the closed form cannot fake: both solved points must
        sit at y = -h, or the plane assumption was not actually applied."""
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 9.0)
        got = _solve_span(a, b, 1.435, rotation)
        assert got["ground_a"][1] == pytest.approx(-h, abs=1e-9)
        assert got["ground_b"][1] == pytest.approx(-h, abs=1e-9)

    def test_the_recovered_points_are_the_stated_distance_apart(self):
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 9.0)
        got = _solve_span(a, b, 1.435, rotation)
        d = np.linalg.norm(np.array(got["ground_a"]) - np.array(got["ground_b"]))
        assert float(d) == pytest.approx(1.435, abs=1e-9)

    def test_endpoint_order_does_not_change_the_answer(self):
        """Unlike a vertical reference, a span's endpoints are peers."""
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 9.0)
        assert (_solve_span(a, b, 1.435, rotation)["camera_height"]
                == pytest.approx(_solve_span(b, a, 1.435, rotation)["camera_height"]))


class TestItRefusesRatherThanGuessing:
    """Each refusal has its own reason. A silently-wrong scale propagates into
    every downstream metre in the scene, so silence is the expensive failure."""

    def test_an_endpoint_above_the_horizon_is_refused(self):
        rotation, _project, _h = _rig()
        got = _solve_span((100.0, 50.0), (900.0, 50.0), 1.435, rotation)
        assert got["camera_height"] is None
        assert "horizon" in got["reason"]

    def test_a_short_span_is_refused_and_says_how_short(self):
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 200.0)   # far away = few pixels
        got = _solve_span(a, b, 1.435, rotation)
        assert got["camera_height"] is None
        assert "px" in got["reason"]
        assert f"{GROUND_SPAN_MIN_PX:.0f}px" in got["reason"]

    def test_identical_endpoints_are_refused(self):
        rotation, project, h = _rig()
        a, _b = _span_points(project, h, 1.435, 9.0)
        got = _solve_span(a, a, 1.435, rotation, min_span_px=0.0)
        assert got["camera_height"] is None

    def test_a_non_positive_span_is_refused(self):
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 9.0)
        assert _solve_span(a, b, 0.0, rotation)["camera_height"] is None

    def test_nothing_raises(self):
        """A 20-minute graph must not die on a badly marked reference."""
        rotation, _project, _h = _rig()
        for a, b, d in [((0, 0), (10, 10), 1.435), ((100, 50), (900, 50), 1.435),
                        ((1, 2000), (2, 2001), -5.0)]:
            assert _solve_span(a, b, d, rotation)["camera_height"] is None


class TestConsistencyMeansConditioning:
    """The vertical solver's consistency is a residual. This one has no residual
    — the form is exact — so reporting 1.0 would tell a caller nothing at all."""

    def test_a_wide_near_span_scores_higher_than_a_narrow_far_one(self):
        rotation, project, h = _rig()
        near = _solve_span(*_span_points(project, h, 1.435, 4.0), 1.435, rotation)
        far = _solve_span(*_span_points(project, h, 1.435, 20.0), 1.435, rotation)
        assert near["consistency"] > far["consistency"]

    def test_it_never_reports_certainty_just_because_the_algebra_closed(self):
        rotation, project, h = _rig()
        got = _solve_span(*_span_points(project, h, 1.435, 20.0), 1.435, rotation)
        assert got["camera_height"] == pytest.approx(h, rel=1e-9)
        assert got["consistency"] < 1.0, (
            "an exact answer from a poorly conditioned measurement must not "
            "advertise itself as certain")

    def test_residual_is_zero_because_there_is_nothing_to_fit(self):
        rotation, project, h = _rig()
        assert _solve_span(*_span_points(project, h, 1.435, 9.0), 1.435,
                           rotation)["residual"] == 0.0


class TestTheHonestErrorComparison:
    """The claim the whole track rests on, pinned with numbers.

    Total error = pixel-marking error + how well the real size is known. The
    second term dominates, because camera height scales LINEARLY with the
    reference's real size: an 8%-uncertain adult height is an 8%-uncertain scale
    however cleanly it was marked.
    """

    @staticmethod
    def _span_error(distance, size_sigma, trials=400, seed=7):
        rng = np.random.default_rng(seed)
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, distance)
        errs = []
        for _ in range(trials):
            aa = (a[0] + rng.normal(0, 2), a[1] + rng.normal(0, 2))
            bb = (b[0] + rng.normal(0, 2), b[1] + rng.normal(0, 2))
            d = 1.435 * (1.0 + rng.normal(0, size_sigma))
            got = _solve_span(aa, bb, d, rotation)
            if got["camera_height"]:
                errs.append(abs(got["camera_height"] - h) / h)
        return float(np.median(errs))

    @staticmethod
    def _vertical_error(real_h, size_sigma, distance=9.0, trials=400, seed=7):
        rng = np.random.default_rng(seed)
        rotation, project, h = _rig()
        base = project([3.0, -h, -distance])
        top = project([3.0, -h + real_h, -distance])
        errs = []
        for _ in range(trials):
            bb = (base[0] + rng.normal(0, 2), base[1] + rng.normal(0, 2))
            tt = (top[0] + rng.normal(0, 2), top[1] + rng.normal(0, 2))
            got = metric_height_from_reference(
                bb, tt, real_h * (1.0 + rng.normal(0, size_sigma)),
                rotation=rotation, fx=FX, fy=FY, cx=CX, cy=CY)
            if got["camera_height"]:
                errs.append(abs(got["camera_height"] - h) / h)
        return float(np.median(errs))

    def test_gauge_beats_a_door_beats_a_person(self):
        """The ordering, not the exact numbers — sizes known to 0.1% / 3% / 8%."""
        gauge = self._span_error(4.0, 0.001)
        door = self._vertical_error(2.1, 0.03)
        person = self._vertical_error(1.75, 0.075)
        assert gauge < door < person, (
            f"gauge={gauge:.4%} door={door:.4%} person={person:.4%}")
        assert gauge < 0.01, "an exact constant should land inside 1%"
        assert person > 0.02, "an 8%-uncertain size cannot give a 2% scale"

    def test_the_advantage_is_the_CONSTANT_not_the_geometry(self):
        """Guards against the plausible misreading that ground spans are
        inherently more accurate. Given an equally exact size, the VERTICAL
        estimator is the better-conditioned of the two at matched distance —
        a taller object spans more pixels. Anyone tempted to route everything
        through spans should see this fail first.
        """
        span_exact = self._span_error(9.0, 0.0)
        vertical_exact = self._vertical_error(2.1, 0.0)
        assert vertical_exact < span_exact, (
            "if this reverses, the comment in metric_height_from_ground_span "
            f"is wrong: span={span_exact:.4%} vertical={vertical_exact:.4%}")

    def test_error_grows_as_the_span_narrows(self):
        near, mid, far = (self._span_error(d, 0.001) for d in (4.0, 9.0, 25.0))
        assert near < mid < far


class TestRegistryEntries:
    def test_standard_gauge_is_a_span_with_no_height(self):
        ref = get_scale_reference("rail_gauge_standard")
        assert ref.ground_span_m == 1.435
        assert ref.height is None, (
            "filing gauge as a height would solve it as a 1.435m tall object")

    def test_the_gauge_note_says_which_edges_to_mark(self):
        """Rail centre to rail centre is ~1.5m — marking the wrong pair is a ~4%
        error that looks entirely reasonable."""
        notes = get_scale_reference("rail_gauge_standard").notes or ""
        assert "INNER" in notes

    def test_the_other_gauges_are_separate_ids(self):
        """1.435 / 1.067 / 1.000 must never silently substitute — picking wrong
        is a silent 30-44% scale error."""
        spans = {get_scale_reference(i).ground_span_m for i in
                 ("rail_gauge_standard", "rail_gauge_cape", "rail_gauge_metre")}
        assert len(spans) == 3

    def test_typical_values_are_not_dressed_up_as_specifications(self):
        assert get_scale_reference("rail_sleeper_pitch").confidence == "heuristic"
        assert get_scale_reference("rail_gauge_standard").confidence == "standard"

    def test_the_regional_reference_says_it_is_regional(self):
        notes = get_scale_reference("platform_height_uk").notes or ""
        assert "REGIONAL" in notes

    def test_platform_height_is_a_HEIGHT_not_a_span(self):
        ref = get_scale_reference("platform_height_uk")
        assert ref.height == 0.915 and ref.ground_span_m is None

    def test_an_entry_with_neither_fails_loudly_at_load(self):
        """Silence here surfaces much later as a 'no real height' skip that names
        the spec rather than the malformed registry entry."""
        from atlas_camera.reference_data.registry import ScaleReference
        with pytest.raises(ValueError, match="broken_ref"):
            ScaleReference.from_dict({"id": "broken_ref", "label": "x"})

    def test_the_whole_registry_still_loads(self):
        from atlas_camera.reference_data import load_scale_references
        refs = load_scale_references()
        assert all(r.height is not None or r.ground_span_m is not None for r in refs)


class TestThroughTheAggregator:
    """resolve_reference_scale is the shared entry AtlasApplyScaleReferences uses,
    so a span must flow through it with no node-side change at all."""

    RAIL = 0.172

    def _rig_and_gauge(self, distance=6.0):
        """Gauge marked where it actually is: on the RAILHEADS, standing a rail's
        height above the ballast. Marking it on the ground plane would make the
        fixture physically impossible and hide the datum correction."""
        rotation, project, h = _rig()
        a, b = _span_points(project, h - self.RAIL, 1.435, distance)
        return rotation, project, h, {"reference_id": "rail_gauge_standard",
                                      "point_a_px": list(a), "point_b_px": list(b)}

    def _resolve(self, rotation, *specs):
        return resolve_reference_scale(list(specs), rotation=rotation,
                                       fx=FX, fy=FY, cx=CX, cy=CY)

    def test_a_registry_id_alone_resolves_the_span(self):
        rotation, _p, h, gauge = self._rig_and_gauge()
        got = self._resolve(rotation, gauge)
        assert got["camera_height"] == pytest.approx(h, rel=1e-9)
        assert got["references"][0]["kind"] == "ground_span"

    def test_an_explicit_ground_span_m_needs_no_registry_entry(self):
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 2.5, 6.0)
        got = self._resolve(rotation, {"ground_span_m": 2.5,
                                       "point_a_px": list(a), "point_b_px": list(b)})
        assert got["camera_height"] == pytest.approx(h, rel=1e-9)

    def test_spans_and_heights_aggregate_together(self):
        rotation, project, h, gauge = self._rig_and_gauge()
        door = {"label": "door", "height_m": 2.1,
                "base_px": list(project([3.0, -h, -6.0])),
                "top_px": list(project([3.0, -h + 2.1, -6.0]))}
        got = self._resolve(rotation, gauge, door)
        assert got["camera_height"] == pytest.approx(h, rel=1e-6)
        assert [r["kind"] for r in got["references"]] == ["ground_span", "vertical"]

    def test_an_EXACT_reference_outvotes_a_GUESSED_one(self):
        """Found live 2026-07-31, and the reason _SIZE_PRIOR exists.

        metric_height_from_reference reports consistency 1.0 for a *wrong*
        height — a person marked across 3.5m of world still fits their own two
        rays perfectly, residual 0. Before size priors that gave the guess more
        weight than the exact constant, and the aggregate took the wrong answer
        while only the confidence number hinted at trouble.
        """
        rotation, project, h, gauge = self._rig_and_gauge()
        wrong_person = {"reference_id": "person_175cm",
                        "base_px": list(project([3.0, -h, -6.0])),
                        "top_px": list(project([3.0, -h + 3.5, -6.0]))}
        alone = self._resolve(rotation, wrong_person)
        assert alone["camera_height"] != pytest.approx(h, rel=0.1), (
            "precondition: this reference really is wrong on its own")
        assert alone["references"][0]["consistency"] == pytest.approx(1.0, abs=1e-6), (
            "precondition: and it is wrong while reporting perfect consistency")

        together = self._resolve(rotation, gauge, wrong_person)
        assert together["camera_height"] == pytest.approx(h, rel=1e-6), (
            "the exact constant must win")

    def test_disagreement_lowers_confidence(self):
        rotation, project, h, gauge = self._rig_and_gauge()
        wrong = {"reference_id": "person_175cm",
                 "base_px": list(project([3.0, -h, -6.0])),
                 "top_px": list(project([3.0, -h + 3.5, -6.0]))}
        agree = self._resolve(rotation, gauge)["confidence"]
        disagree = self._resolve(rotation, gauge, wrong)["confidence"]
        assert disagree < agree

    def test_a_bare_height_is_taken_at_face_value(self):
        """No reference_id means no provenance to rank by — the artist asserted
        the number directly, so it is not silently demoted."""
        rotation, project, h = _rig()
        spec = {"height_m": 2.1,
                "base_px": list(project([3.0, -h, -6.0])),
                "top_px": list(project([3.0, -h + 2.1, -6.0]))}
        got = self._resolve(rotation, spec)
        assert got["references"][0]["size_prior"] == 1.0

    def test_a_refused_span_is_reported_not_dropped(self):
        rotation, _project, _h = _rig()
        got = self._resolve(rotation, {"ground_span_m": 1.435,
                                       "point_a_px": [100, 50],
                                       "point_b_px": [900, 50]})
        assert got["camera_height"] is None
        assert got["references"][0]["status"] == "rejected"
        assert "horizon" in got["references"][0]["reason"]

    def test_a_span_spec_without_pixels_is_skipped_with_a_reason(self):
        rotation, _project, _h = _rig()
        got = self._resolve(rotation, {"reference_id": "rail_gauge_standard"})
        assert got["references"][0]["status"] == "skipped"

    def test_a_span_bbox_is_read_along_its_NEAR_edge(self):
        """_reference_segment collapses a bbox to its vertical centre line, which
        would destroy a horizontal span. The span path takes the bottom edge —
        the ground closest to camera, and the best conditioned."""
        rotation, project, h = _rig()
        a, b = _span_points(project, h, 1.435, 6.0)
        y = max(a[1], b[1])
        got = self._resolve(rotation, {"ground_span_m": 1.435,
                                       "bbox_px": [min(a[0], b[0]), y - 4.0,
                                                   max(a[0], b[0]), y]})
        assert got["camera_height"] == pytest.approx(h, rel=1e-6)


class TestTheRailheadDatum:
    """Measured 2026-07-31, after the first version of this track shipped
    claiming 0.20% accuracy for rail gauge.

    Gauge is measured across the RAILHEADS, which stand a rail's height above
    the ballast. Both estimators return the camera height above the plane the
    MARKED points lie in, so an uncorrected gauge answers a question nobody
    asked: how high is the camera above the rails. The error is e/h and it is
    therefore worst exactly where these plates are commonest — at eye level it
    is 10.75%, which is worse than the assumed-height references gauge was
    brought in to beat. The correction is a straight addition and must be
    applied, and reported.
    """

    RAIL = 0.172   # 60E1/UIC60; profiles span 0.159-0.186

    def _marked_on_rails(self, h_ground, distance=6.0):
        """A camera h_ground above the ballast, gauge marked on the railheads."""
        rotation, project, _h = _rig(h_ground)
        y = -(h_ground - self.RAIL)
        a, b = project([-1.435 / 2, y, -distance]), project([1.435 / 2, y, -distance])
        return rotation, {"reference_id": "rail_gauge_standard",
                          "point_a_px": list(a), "point_b_px": list(b)}

    def _resolve(self, rotation, spec):
        return resolve_reference_scale([spec], rotation=rotation,
                                       fx=FX, fy=FY, cx=CX, cy=CY)

    @pytest.mark.parametrize("h_ground", [1.6, 4.2, 12.0])
    def test_the_reported_height_is_above_the_GROUND_not_the_rails(self, h_ground):
        rotation, spec = self._marked_on_rails(h_ground)
        got = self._resolve(rotation, spec)
        assert got["camera_height"] == pytest.approx(h_ground, rel=1e-9)

    def test_the_raw_uncorrected_value_is_kept_and_named(self):
        """A reader who wants height above rail level must not have to subtract
        it back out and guess whether the correction was applied."""
        rotation, spec = self._marked_on_rails(4.2)
        ref = self._resolve(rotation, spec)["references"][0]
        assert ref["camera_height_above_marked_plane"] == pytest.approx(4.2 - self.RAIL,
                                                                       rel=1e-9)
        assert ref["datum_offset_m"] == pytest.approx(self.RAIL)

    def test_the_correction_is_never_silent(self):
        rotation, spec = self._marked_on_rails(4.2)
        note = self._resolve(rotation, spec)["references"][0]["datum_note"]
        assert "0.172" in note and "ground" in note

    def test_the_offset_propagates_with_NO_distortion(self):
        """The load-bearing measurement: this is a pure datum shift, not a scale
        error. If it were multiplicative the fix would have to be a rescale, not
        an addition, so the property is worth asserting rather than assuming."""
        for h_ground in (1.2, 4.2, 25.0):
            for pitch in (-3.0, -12.0, -45.0):
                rotation, project, _ = _rig(h_ground, pitch)
                y = -(h_ground - self.RAIL)
                a, b = _span_points(project, h_ground - self.RAIL, 1.435, 6.0)
                raw = _solve_span(a, b, 1.435, rotation)["camera_height"]
                assert raw == pytest.approx(h_ground - self.RAIL, rel=1e-9), (
                    f"h={h_ground} pitch={pitch}: offset distorted the solve")

    @pytest.mark.parametrize("h_ground,worst", [(1.6, 0.09), (4.2, 0.03), (12.0, 0.01)])
    def test_uncorrected_error_is_e_over_h_and_worst_when_lowest(
            self, h_ground, worst):
        """Pins the direction of the problem: it is RELATIVE, so a low camera
        suffers most. The first version of this track quoted the 4.2m figure as
        though it were the general case."""
        rotation, project, _ = _rig(h_ground)
        a, b = _span_points(project, h_ground - self.RAIL, 1.435, 6.0)
        raw = _solve_span(a, b, 1.435, rotation)["camera_height"]
        err = abs(raw - h_ground) / h_ground
        assert err == pytest.approx(self.RAIL / h_ground, rel=1e-6)
        assert err > worst, "the error must not be dismissible at this height"

    def test_uncorrected_gauge_is_WORSE_than_an_assumed_door_at_eye_level(self):
        """The finding that makes the correction mandatory rather than a
        refinement. A door assumed to +/-3% gives about 2.06% total; gauge
        uncorrected at 1.6m gives 10.75%. Preferring gauge on the strength of
        its exact constant, while ignoring its datum, is a net loss."""
        rotation, project, _ = _rig(1.6)
        a, b = _span_points(project, 1.6 - self.RAIL, 1.435, 6.0)
        raw = _solve_span(a, b, 1.435, rotation)["camera_height"]
        assert abs(raw - 1.6) / 1.6 > 0.0206

    def test_a_spec_can_override_the_offset_for_buried_track(self):
        """Abandoned track — this project's actual use case — can have ballast
        or vegetation up to the railhead, where the registry default is wrong."""
        rotation, project, _ = _rig(4.2)
        a, b = _span_points(project, 4.2, 1.435, 6.0)      # rails AT ground level
        spec = {"reference_id": "rail_gauge_standard", "datum_offset_m": 0.0,
                "point_a_px": list(a), "point_b_px": list(b)}
        got = self._resolve(rotation, spec)
        assert got["camera_height"] == pytest.approx(4.2, rel=1e-9)

    def test_applying_the_offset_wrongly_costs_THE_SAME_as_omitting_it(self):
        """The correction is two-sided. It is not a safe default to sprinkle on;
        on buried track the registry value overshoots by exactly as much as
        omitting it undershoots on clean track."""
        rotation, project, _ = _rig(4.2)
        a, b = _span_points(project, 4.2, 1.435, 6.0)      # rails AT ground level
        got = self._resolve(rotation, {"reference_id": "rail_gauge_standard",
                                       "point_a_px": list(a), "point_b_px": list(b)})
        assert got["camera_height"] == pytest.approx(4.2 + self.RAIL, rel=1e-9)
        assert abs(got["camera_height"] - 4.2) / 4.2 == pytest.approx(
            self.RAIL / 4.2, rel=1e-6)

    def test_a_reference_with_no_offset_is_untouched(self):
        rotation, project, h = _rig()
        spec = {"reference_id": "person_175cm",
                "base_px": list(project([3.0, -h, -6.0])),
                "top_px": list(project([3.0, -h + 1.75, -6.0]))}
        got = self._resolve(rotation, spec)
        assert "datum_note" not in got["references"][0]
        assert got["camera_height"] == pytest.approx(h, rel=1e-9)


class TestTheRailwayDatumsAreDeclared:
    def test_every_gauge_carries_the_railhead_offset(self):
        for gid in ("rail_gauge_standard", "rail_gauge_cape", "rail_gauge_metre"):
            assert get_scale_reference(gid).datum_offset_m == pytest.approx(0.172)

    def test_sleeper_pitch_declares_ZERO_rather_than_omitting_it(self):
        """An omitted offset and a considered zero look identical to a solver but
        not to a reader; sleeper tops really are at ballast crown."""
        assert get_scale_reference("rail_sleeper_pitch").datum_offset_m == 0.0

    def test_platform_height_shares_the_rail_datum(self):
        """0.915m is specified FROM RAIL LEVEL, so its base is the railhead too.
        Marking base-to-surface off the ballast measures ~1.087m instead, and
        mixing the two datums is the easiest error available here."""
        ref = get_scale_reference("platform_height_uk")
        assert ref.datum_offset_m == pytest.approx(0.172)
        assert "RAIL LEVEL" in (ref.notes or "")

    def test_the_gauge_note_states_the_measured_consequence(self):
        notes = get_scale_reference("rail_gauge_standard").notes or ""
        assert "10.75%" in notes, "the eye-level figure is the one that matters"
        assert "8.35" in notes, "and the height below which gauge stops winning"


class TestTheConstraintsPathExplainsItself:
    def test_a_span_reference_in_the_vertical_path_names_the_problem(self):
        """The artist gave a valid reference_id, so "requires reference_id" would
        be a baffling thing to tell them."""
        from atlas_camera.core.solver import _constraint_scale_measurements
        with pytest.raises(ValueError, match="ground span"):
            _constraint_scale_measurements({"scale_constraints": [
                {"reference_id": "rail_gauge_standard",
                 "image_segment": [[0, 100], [200, 100]]}]})
