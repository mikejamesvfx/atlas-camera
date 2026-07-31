"""Placeholder building mass for ground that was never photographed.

The module's whole claim is that it invents as little as possible: only *that a
building exists there*. Every number comes off the plate. These tests are mostly
asking one question — can an invented value get in.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.block_massing import (  # noqa: E402
    MIN_GRID_COHERENCE,
    GridFit,
    estimate_grid_azimuth,
    grid_basis,
    massing_report,
    place_massing,
    sample_heights,
)


def _segments(azimuth_deg, n=40, length=20.0, jitter=0.0, seed=0,
              both_families=True):
    """Ground segments lying on a grid at `azimuth_deg`."""
    rng = np.random.default_rng(seed)
    U, V = grid_basis(azimuth_deg)
    out = []
    for i in range(n):
        d = U if (not both_families or i % 2 == 0) else V
        if jitter:
            a = np.radians(rng.normal(0, jitter))
            d = np.array([d[0] * np.cos(a) - d[2] * np.sin(a), 0.0,
                          d[0] * np.sin(a) + d[2] * np.cos(a)])
        p0 = np.array([rng.uniform(-100, 100), -50.0, rng.uniform(-200, -50)])
        out.append([p0, p0 + d * length])
    return np.array(out)


class TestTheGridFit:
    @pytest.mark.parametrize("az", [0.0, 12.5, 45.0, 87.6, 89.9])
    def test_it_recovers_the_azimuth(self, az):
        fit = estimate_grid_azimuth(_segments(az))
        assert fit.usable
        # mod-90 circular distance, so 89.9 and 0.0 are 0.1 apart not 89.9
        err = abs(((fit.azimuth_deg - az + 45.0) % 90.0) - 45.0)
        assert err < 0.5, f"got {fit.azimuth_deg}"

    def test_both_street_families_reinforce_one_answer(self):
        """A city grid has two perpendicular families. Fitting mod 180 would
        average them into something between — the classic wrong answer that
        still looks like a number."""
        one = estimate_grid_azimuth(_segments(30.0, both_families=False))
        two = estimate_grid_azimuth(_segments(30.0, both_families=True))
        err = abs(((two.azimuth_deg - one.azimuth_deg + 45.0) % 90.0) - 45.0)
        assert err < 0.5

    def test_the_wraparound_case_that_a_plain_mean_gets_wrong(self):
        """Angles straddling 0/90: an arithmetic mean of 89.5 and 0.5 returns
        45, which is perpendicular to the truth and entirely plausible."""
        segs = np.concatenate([_segments(89.5, n=20, seed=1),
                               _segments(0.5, n=20, seed=2)])
        fit = estimate_grid_azimuth(segs)
        err = min(fit.azimuth_deg, 90.0 - fit.azimuth_deg)
        assert err < 1.5, f"circular mean failed at the wrap: {fit.azimuth_deg}"

    def test_coherence_is_weighted_by_LENGTH_not_count(self):
        """One long kerb is better evidence than twenty short fragments, and a
        count-weighted fit says the opposite."""
        good = _segments(80.0, n=2, length=200.0, seed=3)
        noise = _segments(10.0, n=30, length=1.2, seed=4)
        fit = estimate_grid_azimuth(np.concatenate([good, noise]))
        err = abs(((fit.azimuth_deg - 80.0 + 45.0) % 90.0) - 45.0)
        assert err < 3.0, "short noise outvoted the long lines"

    def test_a_scene_with_no_grid_is_REFUSED_with_a_reason(self):
        rng = np.random.default_rng(5)
        segs = np.array([[[rng.uniform(-99, 99), -50.0, rng.uniform(-200, -50)],
                          [rng.uniform(-99, 99), -50.0, rng.uniform(-200, -50)]]
                         for _ in range(60)])
        fit = estimate_grid_azimuth(segs)
        assert not fit.usable
        assert "no usable street grid" in fit.reason
        assert f"{100 * MIN_GRID_COHERENCE:.0f}%" in fit.reason

    def test_no_segments_at_all_is_refused_not_zero(self):
        fit = estimate_grid_azimuth(np.zeros((0, 2, 3)))
        assert not fit.usable and fit.reason

    def test_segments_too_short_to_carry_a_direction_are_refused(self):
        fit = estimate_grid_azimuth(_segments(30.0, n=40, length=0.2))
        assert not fit.usable
        assert "too short" in fit.reason

    def test_a_marginal_fit_still_reports_its_azimuth(self):
        """Refusal must not discard the measurement — an operator needs to see
        how close it came."""
        fit = estimate_grid_azimuth(_segments(30.0, n=40, length=0.2))
        assert isinstance(fit.azimuth_deg, float)


class TestHeightsCannotBeInvented:
    def test_every_sampled_height_was_actually_observed(self):
        obs = [12.0, 18.5, 31.0]
        got = sample_heights(obs, 200, seed=1)
        assert set(np.unique(got)).issubset(set(obs)), (
            "resampling must never produce a height nobody saw")

    def test_it_never_extrapolates_beyond_the_observed_range(self):
        obs = [12.0, 18.5, 31.0]
        got = sample_heights(obs, 500, seed=2)
        assert got.min() >= min(obs) and got.max() <= max(obs)

    def test_no_observations_is_an_ERROR_not_a_default(self):
        """A default height would be the module inventing the one thing it
        exists to avoid inventing."""
        with pytest.raises(ValueError, match="nothing to take it from"):
            sample_heights([], 5)

    def test_non_finite_and_negative_observations_are_dropped(self):
        got = sample_heights([np.nan, -3.0, 0.0, 20.0], 50, seed=3)
        assert np.all(got == 20.0)

    def test_it_is_reproducible(self):
        a = sample_heights([5.0, 9.0, 14.0], 40, seed=7)
        b = sample_heights([5.0, 9.0, 14.0], 40, seed=7)
        assert np.array_equal(a, b)


class TestPlacement:
    BASE = dict(azimuth_deg=87.6, ground_y=-52.6,
                observed_heights_m=[15.0, 21.0, 28.0, 34.0],
                region_uv=(-150.0, 150.0, -260.0, -90.0))

    def test_it_places_masses_and_they_sit_on_the_ground(self):
        boxes = place_massing(**self.BASE, seed=1)
        assert boxes
        for b in boxes:
            base_y = [c[1] for c in b.corners[:4]]
            assert all(abs(y - (-52.6)) < 1e-9 for y in base_y)

    def test_the_top_is_exactly_the_height_above_the_base(self):
        b = place_massing(**self.BASE, seed=1)[0]
        for i in range(4):
            assert b.corners[i + 4][1] - b.corners[i][1] == pytest.approx(b.height_m)

    def test_every_height_came_from_the_observed_set(self):
        obs = set(self.BASE["observed_heights_m"])
        assert {b.height_m for b in place_massing(**self.BASE, seed=2)} <= obs

    def test_boxes_are_aligned_to_the_GRID_not_to_world_axes(self):
        """A world-axis-aligned box in a rotated city reads wrong instantly."""
        b = place_massing(**self.BASE, seed=1)[0]
        edge = np.array(b.corners[1]) - np.array(b.corners[0])
        U, _ = grid_basis(87.6)
        cos = abs(float(np.dot(edge / np.linalg.norm(edge), U)))
        assert cos > 0.999

    def test_measured_footprints_are_never_overlapped(self):
        """A placeholder driven through a photographed building is the one
        failure that cannot be excused as 'it is only a blockout'."""
        occupied = [(-40.0, 40.0, -200.0, -140.0)]
        boxes = place_massing(**self.BASE, occupied_uv=occupied, seed=3)
        ou0, ou1, ov0, ov1 = occupied[0]
        for b in boxes:
            assert (b.u1 <= ou0 or ou1 <= b.u0 or b.v1 <= ov0 or ov1 <= b.v0), (
                f"box {b.u0:.0f},{b.v0:.0f} overlaps a measured footprint")

    def test_street_bands_are_kept_clear(self):
        bands = [(-200.0, -170.0)]
        boxes = place_massing(**self.BASE, street_bands_v=bands, seed=4)
        assert boxes
        for b in boxes:
            assert b.v1 <= -200.0 or b.v0 >= -170.0

    def test_an_empty_region_places_nothing_rather_than_raising(self):
        assert place_massing(**{**self.BASE, "region_uv": (0.0, 0.0, 0.0, 0.0)}) == []

    def test_a_fully_occupied_region_places_nothing(self):
        boxes = place_massing(**self.BASE, seed=5,
                              occupied_uv=[(-1e4, 1e4, -1e4, 1e4)])
        assert boxes == []

    def test_every_box_is_tagged_placeholder(self):
        """Nothing downstream may mistake this for measurement."""
        assert all(b.provenance == "placeholder"
                   for b in place_massing(**self.BASE, seed=6))

    def test_masses_do_not_overlap_each_other(self):
        boxes = place_massing(**self.BASE, seed=8)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert (a.u1 <= b.u0 or b.u1 <= a.u0
                        or a.v1 <= b.v0 or b.v1 <= a.v0), "masses interpenetrate"

    def test_placement_is_reproducible(self):
        a = place_massing(**self.BASE, seed=9)
        b = place_massing(**self.BASE, seed=9)
        assert [x.corners for x in a] == [x.corners for x in b]

    def test_the_zone_label_rides_along(self):
        boxes = place_massing(**self.BASE, seed=10, zone="behind the tower")
        assert all(b.zone == "behind the tower" for b in boxes)


class TestTheReport:
    def test_it_states_the_grid_and_that_nothing_is_measured(self):
        fit = estimate_grid_azimuth(_segments(87.6))
        boxes = place_massing(azimuth_deg=fit.azimuth_deg, ground_y=-52.6,
                              observed_heights_m=[15.0, 30.0],
                              region_uv=(-100.0, 100.0, -200.0, -100.0), seed=1)
        text = massing_report(boxes, fit)
        assert "placeholder" in text
        assert "resampled from observed roofs" in text
        assert "deg grid" in text

    def test_an_empty_result_says_so_plainly(self):
        assert "no placeholder masses" in massing_report([], GridFit(0, 0, 0))


class TestTheNode:
    """AtlasBlockoutMassing end to end, with a synthetic plate and solve."""

    @staticmethod
    def _fixture(az_deg=30.0):
        torch = pytest.importorskip("torch")
        pytest.importorskip("cv2")
        from atlas_camera.core.solver import solve_from_learned_prior
        from atlas_camera.inference.learned_prior import CameraPrior
        W, H = 640, 480
        prior = CameraPrior(
            focal_px=500.0, fov_h_deg=65.0, fov_v_deg=51.0, roll_deg=0.0,
            pitch_deg=-35.0, up_cam=(0.0, 0.819, 0.574),
            principal_point_px=(W / 2, H / 2), image_width=W, image_height=H)
        solve = solve_from_learned_prior(prior, camera_height=20.0)
        # a plate with strong straight lines low in frame
        img = np.zeros((H, W, 3), dtype=np.float32)
        for y in range(300, 470, 14):
            img[y:y + 3, 40:600] = 1.0
        return solve, torch.from_numpy(img)[None]

    def test_it_refuses_without_observed_heights(self):
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        solve, img = self._fixture()
        out, rep = AtlasBlockoutMassing().blockout(solve, img, "")
        assert "SKIPPED" in rep and "resample" in rep
        assert not (out.projection_scene.proxy_geometry or [])

    def test_a_real_run_appends_placeholder_boxes(self):
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        solve, img = self._fixture()
        out, rep = AtlasBlockoutMassing().blockout(solve, img, "12,18,25", seed=3)
        prims = out.projection_scene.proxy_geometry or []
        if not prims:
            pytest.skip(f"no grid fitted on the synthetic plate: {rep}")
        assert all(p.primitive_type == "box" for p in prims)
        assert all(p.metadata["provenance"] == "placeholder" for p in prims)
        assert all(p.metadata["height_m"] in (12.0, 18.0, 25.0) for p in prims)

    def test_boxes_sit_ON_the_ground_plane(self):
        """The datum error that cost real time on 2026-07-31: geometry built at
        y=0 when y=0 was the CAMERA's height, not the ground. Here the solve puts
        the ground at y=0 and the camera above it, so every box base must be 0."""
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        solve, img = self._fixture()
        out, rep = AtlasBlockoutMassing().blockout(solve, img, "12,18,25", seed=3)
        prims = out.projection_scene.proxy_geometry or []
        if not prims:
            pytest.skip(f"no grid fitted: {rep}")
        for p in prims:
            base_y = p.transform_matrix[1][3] - p.dimensions[1] / 2.0
            assert abs(base_y) < 1e-6, f"{p.name} floats at {base_y:.3f}"

    def test_existing_geometry_is_never_clobbered(self):
        """This is NOT a derive node. Derive nodes deliberately replace all
        PROXY_ROLE geometry; doing that here would delete the relief mesh the
        placeholders exist to surround."""
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        from atlas_camera.core.proxy_geometry import PROXY_ROLE
        from atlas_camera.core.schema import AtlasProxyPrimitive
        solve, img = self._fixture()
        keep = AtlasProxyPrimitive(name="projection_relief_mesh",
                                   primitive_type="mesh",
                                   metadata={"role": PROXY_ROLE})
        solve.projection_scene.proxy_geometry = [keep]
        out, _ = AtlasBlockoutMassing().blockout(solve, img, "12,18,25", seed=3)
        names = [p.name for p in out.projection_scene.proxy_geometry or []]
        assert "projection_relief_mesh" in names

    def test_the_input_solve_is_not_mutated(self):
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        solve, img = self._fixture()
        before = len(solve.projection_scene.proxy_geometry or [])
        AtlasBlockoutMassing().blockout(solve, img, "12,18,25", seed=3)
        assert len(solve.projection_scene.proxy_geometry or []) == before

    def test_a_camera_below_the_ground_is_refused(self):
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        solve, img = self._fixture()
        e = solve.camera.extrinsics
        e.camera_position = (0.0, -1.0, 0.0)
        out, rep = AtlasBlockoutMassing().blockout(solve, img, "12,18,25")
        assert "SKIPPED" in rep and "below the ground" in rep

    def test_the_report_names_the_missing_ground_mask(self):
        """Without it, facade edges pollute the fit — measured live: 2.87deg
        instead of 87.72deg on the NYC plate, a 5.1deg error that still returns
        a confident grid."""
        from atlas_camera.comfy.nodes_geometry import AtlasBlockoutMassing
        solve, img = self._fixture()
        _out, rep = AtlasBlockoutMassing().blockout(solve, img, "12,18,25")
        assert "no ground_mask" in rep


class TestOrientationIsScaleFree:
    """h cancels out of the direction, and that is load-bearing here.

    From the ground-span derivation, a ray meets the plane at P_i = h*q_i with
    q_i = r_i/(-r_i.y). A horizontal edge between two such points has direction

        (P_1 - P_0)/|P_1 - P_0| = (q_1 - q_0)/|q_1 - q_0|

    in which h has vanished. So the street grid — and with it the orientation of
    every occluded facade, since faces are assumed vertical and perpendicular —
    is recoverable BEFORE metric scale is known, and stays correct even if the
    scale is later revised.

    That matters practically: the camera height used here moved twice in one
    session (assumed 1.6 m, then 52.6 m from references, then confirmed by
    depth). Had azimuth depended on it, every box would have needed replacing.
    """

    def test_scaling_the_whole_scene_does_not_move_the_azimuth(self):
        segs = _segments(37.25, n=40, seed=11)
        base = estimate_grid_azimuth(segs).azimuth_deg
        for factor in (0.1, 0.5, 2.0, 33.0):
            got = estimate_grid_azimuth(segs * factor).azimuth_deg
            assert abs(((got - base + 45.0) % 90.0) - 45.0) < 1e-9, (
                f"azimuth moved under a x{factor} rescale")

    def test_coherence_is_also_scale_free(self):
        segs = _segments(37.25, n=40, seed=12)
        a = estimate_grid_azimuth(segs).coherence
        b = estimate_grid_azimuth(segs * 7.0).coherence
        assert a == pytest.approx(b)

    def test_the_perpendicular_axis_follows_from_the_grid_and_world_up(self):
        """Z_b = X_b x Y. The receding axis must lie IN the ground plane — a
        derivation that leaves a vertical component in it has mixed the optical
        axis into a horizontal direction."""
        U, V = grid_basis(37.25)
        cross = np.cross(U, np.array([0.0, 1.0, 0.0]))
        cross /= np.linalg.norm(cross)
        assert abs(cross[1]) < 1e-12, "the receding axis must be horizontal"
        assert abs(abs(float(np.dot(cross, V))) - 1.0) < 1e-9
