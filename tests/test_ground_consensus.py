"""Gravity-locked ground consensus: the maths, defended on known geometry.

The claim under test is narrow and worth stating plainly, because the shipping
estimator already does half of it. Atlas locks the ground normal to world +Y in
every single-image path, and ``solver.estimate_ground_height_from_depth``
already reduces the ground to one scalar. What it does NOT do is weight its
votes, bound its tolerance by the data's own dispersion, or accept an exclusion
mask -- so a deep street lets far pixels win on count while cars and kerbs vote
freely.

These tests build scenes where the true camera height is known by construction,
then contaminate them the specific ways a street plate contaminates a real one.
A test that only proved "recovers 1.6 on clean synthetic ground" would pass for
the shipping estimator too and prove nothing.

tests/ is not a package -- no cross-file imports, so the rig is local.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.ground_consensus import (
    ESTIMATORS,
    WEIGHTINGS,
    estimate_ground_height_consensus,
)

W = H = 256
FX = FY = 250.0
CX, CY = 128.0, 128.0
CAM_H = 1.6
SKY = 400.0


def _ground_depth(cam_h=CAM_H, *, sky=SKY):
    """Forward depth of a level camera looking down -Z at the plane Y = -cam_h.

    Atlas camera frame: x right, y up, z back; the unnormalized ray is
    ((u-cx)/fx, -(v-cy)/fy, -1), so the ray parameter IS forward depth and a
    pixel below the horizon hits the plane at d = cam_h / (-dy).
    """
    _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(dy < -1e-6, cam_h / np.maximum(-dy, 1e-9), sky)
    return np.clip(np.nan_to_num(d, nan=sky, posinf=sky), 0.0, sky)


def _height_map_depth(hmap, *, sky=SKY):
    """Depth of a ground whose height below the camera varies PER PIXEL.

    This is the honest way to contaminate a ground test. Multiplying depth in a
    region instead tilts that region, and ``|n_y| > 0.90`` then rejects it
    outright -- the contamination deletes itself and every estimator scores a
    perfect 1.6, which measures nothing. Painting a height map keeps each
    region exactly horizontal (so it stays a legitimate candidate and really
    does compete for the vote) and puts the error where a depth model actually
    puts it: in the height, not the tilt.
    """
    _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(dy < -1e-6, np.asarray(hmap) / np.maximum(-dy, 1e-9), sky)
    return np.clip(np.nan_to_num(d, nan=sky, posinf=sky), 0.0, sky)


def _run(depth, **kw):
    kw.setdefault("apply_sky_heuristic", False)
    return estimate_ground_height_consensus(
        depth, rotation=np.eye(3), fx=FX, fy=FY, cx=CX, cy=CY, **kw)


class TestTheFormIsExact:
    """On geometry with no error in it, the answer has no error in it."""

    def test_recovers_known_height(self):
        res = _run(_ground_depth())
        assert res.camera_height == pytest.approx(CAM_H, abs=1e-6)
        assert res.plane_y == pytest.approx(-CAM_H, abs=1e-6)

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_every_estimator_agrees_on_clean_ground(self, estimator):
        res = _run(_ground_depth(), estimator=estimator)
        assert res.camera_height == pytest.approx(CAM_H, abs=1e-3)

    @pytest.mark.parametrize("weighting", WEIGHTINGS)
    def test_every_weighting_agrees_on_clean_ground(self, weighting):
        # Weighting only decides WHOSE vote counts more. When every voter is
        # right it must not change the answer -- otherwise the weighting itself
        # is introducing bias and any later improvement is unreadable.
        res = _run(_ground_depth(), weighting=weighting)
        assert res.camera_height == pytest.approx(CAM_H, abs=1e-3)

    @pytest.mark.parametrize("cam_h", [0.6, 1.6, 4.2, 52.6])
    def test_holds_across_camera_heights(self, cam_h):
        # 52.6 m is the newyork_Birdseye solve height: an elevated camera is a
        # legitimate Atlas input, not an outlier to be penalised.
        res = _run(_ground_depth(cam_h, sky=cam_h * 250.0))
        assert res.camera_height == pytest.approx(cam_h, rel=1e-3)


class TestContaminationDoesNotMoveIt:
    """Street plates are not clean ground, and this is how they are dirty."""

    def test_foreground_boxes_do_not_capture_the_estimate(self):
        # Car roofs: flat horizontal patches 0.7 m above the road, so |n_y| is
        # ~1 and the normal filter CANNOT reject them. They are exactly what an
        # unweighted vote can latch onto.
        hmap = np.full((H, W), CAM_H)
        hmap[150:190, 20:90] = CAM_H - 0.7
        hmap[140:175, 150:230] = CAM_H - 0.7
        res = _run(_height_map_depth(hmap), estimator="mad_median")
        assert res.camera_height == pytest.approx(CAM_H, abs=0.20)

    def test_far_field_drift_does_not_move_a_mode_like_reduction(self):
        # A NEGATIVE result, kept because it is the most useful thing these
        # tests found, and because deleting it would let the story drift back
        # to the intuitive-but-wrong version.
        #
        # The intuitive claim is that distant road pixels swamp the vote by
        # sheer count. They do not. Under perspective the NEAR ground occupies
        # far more of the frame (measured here: near 8890 / mid 8890 / far 8636
        # candidates), and the horizon-compressed rows are edge-rejected on top
        # of that. Meanwhile the near ground still forms the dominant cluster,
        # so a mode- or median-like reduction simply ignores the far tail --
        # even when the road bends 3 m away over its length.
        #
        # What the far field really has is LEVERAGE, not votes: a small depth
        # error out there is metres of world height. That shows up in the SPREAD
        # (see the tolerance test below), not in the centre.
        d0 = _ground_depth()
        drift = np.clip((d0 - 5.0) / 25.0, 0.0, 1.0)
        d = _height_map_depth(CAM_H + 3.0 * drift)

        uniform = _run(d, weighting="uniform")
        near = _run(d, weighting="inverse_depth_sq")

        for res in (uniform, near):
            for name in ("median", "mad_median", "mode", "ransac1d"):
                assert res.estimators[name] == pytest.approx(CAM_H, abs=0.05), name

        # Mean-like reductions are the ones with something to gain, and
        # near-weighting must never make any reduction worse.
        assert (abs(near.estimators["trimmed"] - CAM_H)
                < abs(uniform.estimators["trimmed"] - CAM_H))
        for name in ("median", "trimmed", "mad_median", "mode", "ransac1d"):
            assert (abs(near.estimators[name] - CAM_H)
                    <= abs(uniform.estimators[name] - CAM_H) + 1e-9), name

    def test_far_field_drift_inflates_the_shipping_style_tolerance(self):
        # Defect 2, isolated. plane_tolerance = max(0.15, 0.03 * span) with
        # span the 1-99 percentile spread of ALL candidates: the far tail sets
        # the acceptance band for the near ground. This module derives its
        # tolerance from the weighted MAD instead, so the band tracks the
        # cluster that is actually being measured.
        d0 = _ground_depth()
        drift = np.clip((d0 - 5.0) / 25.0, 0.0, 1.0)
        d = _height_map_depth(CAM_H + 1.2 * drift)

        res = _run(d, weighting="inverse_depth_sq", estimator="mad_median")
        shipping_style = max(0.15, 0.03 * res.distribution["span_1_99"])
        assert res.tolerance < shipping_style

    def test_stratified_weighting_denies_the_far_band_its_majority(self):
        d = _ground_depth()
        res = _run(d, weighting="stratified")
        shares = [res.band_support[b]["weight_share"] for b in ("near", "mid", "far")]
        # Equal total weight per tercile is the entire point of the mode.
        assert max(shares) - min(shares) < 0.02

    def test_occlusion_gap_does_not_break_the_scalar(self):
        # A hole in the observed road: the plane is inferred through it, and a
        # missing region must not be mistaken for a different ground level.
        d = _ground_depth()
        d[170:210, 60:200] = np.nan
        res = _run(d)
        assert res.camera_height == pytest.approx(CAM_H, abs=1e-3)

    def test_exclude_mask_silences_the_contaminated_region(self):
        # A big flat sheet at the wrong height -- a lorry roof, a raised plaza.
        # Horizontal, so it survives every filter the shipping estimator has,
        # and it is large enough to move the vote. The mask is the only thing
        # that can remove it, and the shipping estimator accepts no mask.
        bad = np.zeros((H, W), dtype=bool)
        bad[150:210, 20:236] = True
        hmap = np.full((H, W), CAM_H)
        hmap[bad] = CAM_H - 0.9
        d = _height_map_depth(hmap)

        unmasked = _run(d, estimator="median")
        masked = _run(d, exclude_mask=bad, estimator="median")

        assert masked.camera_height == pytest.approx(CAM_H, abs=1e-3)
        assert abs(masked.camera_height - CAM_H) < abs(unmasked.camera_height - CAM_H)
        assert masked.rejections["exclude_source"] == "explicit"

    def test_near_field_roi_excludes_the_far_half_by_construction(self):
        d = _ground_depth()
        d[d > 6.0] *= 1.6
        res = _run(d, roi=(0.55, 1.0, 0.15, 0.85), estimator="median")
        assert res.camera_height == pytest.approx(CAM_H, abs=0.05)
        assert res.rejections["outside_roi"] > 0

    def test_trapezoid_roi_narrows_toward_the_horizon(self):
        d = _ground_depth()
        wide = _run(d, roi=(0.55, 1.0, 0.1, 0.9))
        narrow = _run(d, roi=(0.55, 1.0, 0.1, 0.9), roi_top_width_frac=0.35)
        assert narrow.candidates < wide.candidates
        assert narrow.camera_height == pytest.approx(CAM_H, abs=1e-3)


class TestTheNormalIsMeasuredNotAssumed:
    """The premise that orientation is innocent has to be evidence, not faith."""

    def test_level_ground_probes_as_gravity_aligned(self):
        res = _run(_ground_depth())
        assert res.normal_probe["median_angle_deg"] == pytest.approx(0.0, abs=0.5)
        assert res.normal_probe["svd_angle_deg"] == pytest.approx(0.0, abs=0.5)
        assert res.normal_probe["applied"] is False

    def test_noise_does_not_move_the_locked_normal(self):
        # The returned plane is gravity-locked by construction: whatever the
        # probe reports, plane_y stays the offset along +Y and nothing rotates.
        rng = np.random.RandomState(7)
        d = _ground_depth() * (1.0 + rng.normal(0.0, 0.01, size=(H, W)))
        res = _run(d, estimator="mad_median")
        assert res.camera_height == pytest.approx(CAM_H, abs=0.1)
        assert res.plane_y == pytest.approx(-res.camera_height, abs=1e-9)
        assert res.normal_probe["applied"] is False

    def test_a_real_tilt_is_detected_rather_than_absorbed(self):
        # A ground plane genuinely tilted about X. The height estimate is still
        # a +Y offset -- that is the lock -- but the probe must SEE the tilt,
        # because that angle is what would go on the ground primitive's own
        # transform if the tilt turns out to be real on a plate.
        tilt_deg = 6.0
        t = np.radians(tilt_deg)
        n = np.array([0.0, np.cos(t), -np.sin(t)])
        uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        dirs = np.stack([(uu - CX) / FX, -(vv - CY) / FY, -np.ones_like(uu)], axis=-1)
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(denom < -1e-6, -CAM_H / denom, SKY)
        d = np.clip(np.nan_to_num(d, nan=SKY, posinf=SKY), 0.0, SKY)

        res = _run(d)
        assert res.normal_probe["svd_angle_deg"] == pytest.approx(tilt_deg, abs=1.0)
        assert res.normal_probe["median_angle_deg"] == pytest.approx(tilt_deg, abs=1.5)
        assert res.normal_probe["applied"] is False


class TestItFailsOutLoud:
    """An invented plane is worse than no plane."""

    def test_no_ground_returns_none_not_a_guess(self):
        d = np.full((H, W), SKY, dtype=float)      # sky everywhere
        res = _run(d)
        assert res.camera_height is None
        assert res.confidence == 0.0
        assert res.notes

    def test_camera_below_the_ground_is_refused(self):
        # Ground above the camera: h comes out negative, which is not a plane
        # the caller can use, so it must not be handed one.
        d = _ground_depth()
        _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        dy = -(vv - CY) / FY
        flipped = np.where(dy > 1e-6, CAM_H / np.maximum(dy, 1e-9), SKY)
        flipped = np.clip(np.nan_to_num(flipped, nan=SKY, posinf=SKY), 0.0, SKY)
        res = _run(flipped, horizon_y=-1.0)
        assert res.camera_height is None
        assert res.notes

    def test_height_prior_reports_but_never_clamps(self):
        # Atlas supports elevated and drone cameras; a prior that silently
        # pulled a 52 m camera down to 1.6 would be a correctness bug, not a
        # safety net.
        cam_h = 52.6
        res = _run(_ground_depth(cam_h, sky=cam_h * 250.0), height_prior=(1.0, 2.2))
        assert res.camera_height == pytest.approx(cam_h, rel=1e-3)
        assert any("outside prior" in n for n in res.notes)

    def test_unknown_knobs_raise(self):
        with pytest.raises(ValueError):
            _run(_ground_depth(), weighting="nope")
        with pytest.raises(ValueError):
            _run(_ground_depth(), estimator="nope")


class TestStrideParity:
    """The plate is 36MP, so the strided path is the one that actually runs."""

    def test_strided_matches_unstrided(self):
        d = _ground_depth()
        full = _run(d, max_pixels=10_000_000)
        strided = _run(d, max_pixels=20_000)
        assert strided.camera_height == pytest.approx(full.camera_height, rel=1e-3)
        assert any("strided" in n for n in strided.notes)
        assert strided.ground_mask.shape == (H, W)
