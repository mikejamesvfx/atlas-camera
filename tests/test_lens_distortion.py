"""Lens distortion: carried on the solve, and applied where it changes numbers.

Every back-projection in Atlas assumes a pinhole ray through ``(u - cx) / fx``.
When the plate has real distortion that assumption bends straight world lines,
and nothing downstream can notice — the solve still returns confident values,
they are just measured off curved evidence.

Found live 2026-07-31 on a screen-grabbed plate with no EXIF: GeoCalib's
``distorted`` weights estimate a k1 (-0.006633 here), ``CameraPrior`` had no
field for it, and ``solve_from_learned_prior`` never populated the
``lens_model`` / ``distortion`` slots that ``AtlasIntrinsics`` already had. So
asking for the distorted model returned a differently-solved camera whose
distortion term went nowhere. With no EXIF, lensfun cannot help and this
estimate is the only one available.
"""
from __future__ import annotations

import math

import pytest

from atlas_camera.core.intrinsics import distort_pixel, undistort_pixel
from atlas_camera.core.schema import AtlasIntrinsics

K1 = -0.006633            # measured by GeoCalib on the real plate
W, H = 2070, 1381
FX = 1445.65


def _intr(k1=K1):
    return AtlasIntrinsics(
        image_width=W, image_height=H, fx_px=FX, fy_px=FX,
        cx_px=W / 2.0, cy_px=H / 2.0,
        lens_model="simple_radial" if k1 else "pinhole",
        distortion={"k1": k1} if k1 else {})


class TestTheModelInverts:
    @pytest.mark.parametrize("u,v", [(1035, 690.5), (1435, 690.5), (2069, 1380),
                                     (0, 0), (200, 1200), (2069, 0)])
    def test_distort_undoes_undistort_exactly(self, u, v):
        """A forward/inverse pair that does not round-trip would put a slow
        drift into every projective UV regenerated through it."""
        uu, vv = undistort_pixel(u, v, _intr())
        ru, rv = distort_pixel(uu, vv, _intr())
        assert math.hypot(ru - u, rv - v) < 1e-9

    def test_the_principal_point_never_moves(self):
        assert undistort_pixel(W / 2, H / 2, _intr()) == pytest.approx((W / 2, H / 2))

    def test_the_correction_grows_with_radius(self):
        """Radial by construction — if this ever went flat the coefficient is
        being applied in the wrong units."""
        shifts = []
        for r in (200, 400, 800, 1240):
            u, v = W / 2 + r, H / 2
            uu, vv = undistort_pixel(u, v, _intr())
            shifts.append(math.hypot(uu - u, vv - v))
        assert shifts == sorted(shifts)
        assert shifts[0] < 0.1 < shifts[-1]

    def test_the_measured_magnitudes(self):
        """Pins the numbers the design decisions were made on: sub-pixel over
        the central half, ~6 px at the corner. If these move, the conclusion
        that distortion is negligible for centre-frame work moves with them."""
        def shift(r):
            u, v = W / 2 + r, H / 2
            uu, vv = undistort_pixel(u, v, _intr())
            return math.hypot(uu - u, vv - v)
        assert shift(400) == pytest.approx(0.20, abs=0.03)
        assert shift(800) == pytest.approx(1.63, abs=0.10)
        assert shift(1244) == pytest.approx(6.11, abs=0.30)


class TestItRefusesRatherThanCorrupting:
    def test_a_pinhole_intrinsics_is_an_exact_no_op(self):
        """Callers apply this unconditionally, so the no-distortion path must be
        bit-identical rather than merely close."""
        for u, v in ((0, 0), (500, 300), (2069, 1380)):
            assert undistort_pixel(u, v, _intr(0)) == (float(u), float(v))
            assert distort_pixel(u, v, _intr(0)) == (float(u), float(v))

    def test_a_focal_less_intrinsics_leaves_the_pixel_alone(self):
        """No focal means no normalised radius, so there is no correction to
        make — and dividing by it would be worse than doing nothing."""
        bad = AtlasIntrinsics(image_width=10, image_height=10,
                              distortion={"k1": -0.1})
        assert undistort_pixel(500, 300, bad) == (500.0, 300.0)

    def test_k1_of_zero_is_treated_as_pinhole(self):
        assert undistort_pixel(2069, 1380, _intr(0.0)) == (2069.0, 1380.0)


class TestItReachesTheSolve:
    def _prior(self, k1):
        from atlas_camera.inference.learned_prior import CameraPrior
        return CameraPrior(
            focal_px=FX, fov_h_deg=71.3, fov_v_deg=51.2,
            roll_deg=0.2, pitch_deg=-32.9,
            up_cam=(0.0, 0.84, 0.55), principal_point_px=(W / 2, H / 2),
            image_width=W, image_height=H, k1=k1,
            source_model="geocalib:distorted")

    def test_the_prior_carries_k1_onto_the_intrinsics(self):
        from atlas_camera.core.solver import solve_from_learned_prior
        solve = solve_from_learned_prior(self._prior(K1))
        K = solve.camera.intrinsics
        assert K.distortion.get("k1") == pytest.approx(K1)
        assert K.lens_model == "simple_radial"

    def test_a_pinhole_prior_leaves_the_solve_pinhole(self):
        from atlas_camera.core.solver import solve_from_learned_prior
        K = solve_from_learned_prior(self._prior(None)).camera.intrinsics
        assert K.lens_model == "pinhole"
        assert not K.distortion

    def test_k1_is_NOT_rescaled_with_the_focal(self):
        """k1 multiplies r**2 in focal-length units, so it is already
        resolution-independent. Scaling it alongside fx would apply the
        correction twice — and the bug would only show on plates solved at a
        size other than the prior's own."""
        from atlas_camera.core.solver import solve_from_learned_prior
        big = solve_from_learned_prior(self._prior(K1), image_size=(W * 2, H * 2))
        assert big.camera.intrinsics.distortion["k1"] == pytest.approx(K1)
        assert big.camera.intrinsics.fx_px == pytest.approx(FX * 2, rel=1e-6)


class TestTheWeightsAndTheCameraModelAreChosenTogether:
    """The bug the rest of this file missed, found only by running a real plate.

    `GeoCalib(weights="distorted")` loads a network trained to see distortion,
    but `calibrate()` defaults to `camera_model="pinhole"` — which has no k1 to
    fit. Atlas set the weights and not the model, so the distorted path returned
    a differently-solved camera with NO distortion term, reading exactly like
    "this lens is clean" rather than "you never asked for a coefficient".

    Every other test here builds a CameraPrior by hand and so proved nothing
    about extraction. This one stubs the model and watches the call.
    """

    @staticmethod
    def _run(monkeypatch, weights):
        """Drive estimate_camera_prior with GeoCalib stubbed; return the call."""
        torch = pytest.importorskip("torch")
        import atlas_camera.inference.learned_prior as lp
        seen: dict = {}

        class _Cam:
            size = torch.tensor([[float(W), float(H)]])
            vfov = torch.tensor([math.radians(51.2)])
            k1 = torch.tensor([K1])

        class _Grav:
            rp = torch.tensor([math.radians(0.2), math.radians(-32.9)])
            vec3d = torch.tensor([0.0, 0.84, 0.55])

        class _Model:
            def load_image(self, path):
                return torch.zeros(1)

            def calibrate(self, img, camera_model="pinhole", **kw):
                seen["camera_model"] = camera_model
                return {"camera": _Cam(), "gravity": _Grav()}

        monkeypatch.setattr(lp, "_require_geocalib", lambda: (torch, None))
        monkeypatch.setattr(lp, "_get_model", lambda w, d: _Model())
        monkeypatch.setattr(lp, "resolve_device", lambda d, t: "cpu")
        seen["prior"] = lp.estimate_camera_prior("x.png", weights=weights)
        return seen

    def test_distorted_weights_fit_a_simple_radial_camera(self, monkeypatch):
        assert self._run(monkeypatch, "distorted")["camera_model"] == "simple_radial"

    def test_pinhole_weights_stay_pinhole(self, monkeypatch):
        assert self._run(monkeypatch, "pinhole")["camera_model"] == "pinhole"

    def test_an_unknown_weight_set_falls_back_to_pinhole(self, monkeypatch):
        """GeoCalib also ships `radial` and `simple_divisional`. undistort_pixel
        implements simple_radial only, so fitting one of those would produce a
        coefficient nothing in Atlas could apply — safer to stay pinhole."""
        assert self._run(monkeypatch, "some_future_model")["camera_model"] == "pinhole"

    def test_the_coefficient_actually_lands_on_the_prior(self, monkeypatch):
        """The other half of the same bug: reading k1 off the camera object."""
        assert self._run(monkeypatch, "distorted")["prior"].k1 == pytest.approx(K1)

    def test_the_mapping_is_a_table_not_a_string_match(self):
        from atlas_camera.inference.learned_prior import _CAMERA_MODEL_FOR_WEIGHTS
        assert _CAMERA_MODEL_FOR_WEIGHTS["distorted"] == "simple_radial"
        assert _CAMERA_MODEL_FOR_WEIGHTS["pinhole"] == "pinhole"


class TestItChangesMeasuredScale:
    """The point of carrying it at all. A correction nothing consumes is just
    metadata."""

    @staticmethod
    def _rig():
        import numpy as np
        p = np.radians(-32.9)
        c2w = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        return c2w.T

    def _resolve(self, k1, a, b):
        from atlas_camera.core.solver import resolve_reference_scale
        return resolve_reference_scale(
            [{"ground_span_m": 4.6, "point_a_px": list(a), "point_b_px": list(b)}],
            rotation=self._rig(), fx=FX, fy=FX, cx=W / 2, cy=H / 2, k1=k1)

    def test_a_reference_near_the_frame_edge_moves(self):
        """Where the correction is biggest, it must actually reach the answer."""
        a, b = (120.0, 1290.0), (250.0, 1300.0)
        off = self._resolve(None, a, b)["camera_height"]
        on = self._resolve(K1, a, b)["camera_height"]
        assert off and on
        assert abs(on - off) / off > 1e-4, (
            "an edge reference must not give the same scale with and without k1")

    def test_a_centre_frame_reference_barely_moves(self):
        """And where it is sub-pixel, it must not swing the answer around —
        this is what licenses calling it negligible for centre-frame work."""
        a, b = (960.0, 830.0), (1100.0, 838.0)
        off = self._resolve(None, a, b)["camera_height"]
        on = self._resolve(K1, a, b)["camera_height"]
        assert abs(on - off) / off < 0.01

    def test_k1_none_matches_k1_zero_exactly(self):
        a, b = (120.0, 1290.0), (250.0, 1300.0)
        assert (self._resolve(None, a, b)["camera_height"]
                == self._resolve(0.0, a, b)["camera_height"])

    def test_apply_reference_scale_reads_k1_off_the_solve(self):
        """The caller must not have to know about distortion; the solve already
        does."""
        import numpy as np

        from atlas_camera.core.solver import apply_reference_scale, solve_from_learned_prior
        from atlas_camera.inference.learned_prior import CameraPrior
        prior = CameraPrior(
            focal_px=FX, fov_h_deg=71.3, fov_v_deg=51.2, roll_deg=0.0,
            pitch_deg=-32.9, up_cam=(0.0, 0.84, 0.55),
            principal_point_px=(W / 2, H / 2), image_width=W, image_height=H,
            k1=K1, source_model="geocalib:distorted")
        solve = solve_from_learned_prior(prior)
        assert solve.camera.intrinsics.distortion["k1"] == pytest.approx(K1)
        apply_reference_scale(
            solve, [{"ground_span_m": 4.6, "point_a_px": [120.0, 1290.0],
                     "point_b_px": [250.0, 1300.0]}], adopt=False)
        rs = solve.debug_metadata.get("reference_scale", {})
        assert rs, "reference scale must be recorded even when not adopted"
        assert np.isfinite(rs.get("camera_height_m") or np.nan)
