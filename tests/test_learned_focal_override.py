"""EXIF focal override on the learned-prior solve (pure numpy, no torch)."""

from dataclasses import dataclass

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.solver import solve_from_learned_prior


@dataclass
class _FakePrior:
    """CameraPrior-shaped stand-in (solve_from_learned_prior duck-types it)."""
    image_width: int = 6000
    image_height: int = 4000
    focal_px: float = 4000.0          # ~24mm-equivalent on a 36mm sensor
    up_cam: tuple = (0.0, -0.994521895368273, -0.10452846326765347)  # ~6deg pitch
    pitch_deg: float = 6.0
    roll_deg: float = 0.0
    focal_uncertainty_px: float | None = None
    source_model: str = "fake"

    def to_dict(self):
        return {"source_model": self.source_model}


def test_hint_replaces_predicted_focal():
    prior = _FakePrior()
    solve = solve_from_learned_prior(
        prior, focal_length_mm_hint=20.0, sensor_width_mm=35.9,
        sensor_height_mm=24.0)
    intr = solve.camera.intrinsics
    assert intr.fx_px == pytest.approx(20.0 / 35.9 * 6000)
    assert intr.fy_px == pytest.approx(20.0 / 24.0 * 4000)
    assert intr.focal_length_mm == pytest.approx(20.0)
    assert solve.known_intrinsics_used is True
    assert solve.debug_metadata["focal_source"] == "known_focal_length_hint"
    assert solve.debug_metadata["learned_focal_px_predicted"] == pytest.approx(4000.0)


def test_no_hint_is_geocalib_focal_and_stamped():
    solve = solve_from_learned_prior(_FakePrior())
    assert solve.camera.intrinsics.fx_px == pytest.approx(4000.0)
    assert solve.known_intrinsics_used is False
    assert solve.debug_metadata["focal_source"] == "learned_geocalib"
    assert "learned_focal_px_predicted" not in solve.debug_metadata


def test_zero_or_negative_hint_means_no_override():
    for bogus in (0.0, -5.0):
        solve = solve_from_learned_prior(_FakePrior(), focal_length_mm_hint=bogus)
        assert solve.debug_metadata["focal_source"] == "learned_geocalib"


def test_rotation_unchanged_by_hint():
    prior = _FakePrior()
    with_hint = solve_from_learned_prior(prior, focal_length_mm_hint=20.0)
    without = solve_from_learned_prior(prior)
    assert np.allclose(with_hint.camera.extrinsics.camera_rotation_matrix,
                       without.camera.extrinsics.camera_rotation_matrix)


def test_horizon_uses_overridden_fy():
    prior = _FakePrior()
    solve = solve_from_learned_prior(
        prior, focal_length_mm_hint=20.0, sensor_width_mm=36.0,
        sensor_height_mm=24.0)
    fy = solve.camera.intrinsics.fy_px
    expected_y = 4000 / 2.0 + fy * np.tan(np.radians(prior.pitch_deg))
    (_, y0), (_, y1) = solve.horizon_line.endpoints_px
    assert y0 == pytest.approx(expected_y)
    assert y1 == pytest.approx(expected_y)


def test_large_disagreement_warns():
    # Predicted 4000px vs hint 100mm/36mm*6000 = 16667px — huge disagreement.
    solve = solve_from_learned_prior(_FakePrior(), focal_length_mm_hint=100.0)
    assert solve.debug_metadata["focal_disagreement"] > 0.25
    assert any("differs" in w for w in solve.debug_metadata["warnings"])


def test_close_agreement_no_warning():
    # Predicted 4000px ~= 24mm on 36mm sensor; hint 24.0 -> exact agreement.
    solve = solve_from_learned_prior(_FakePrior(), focal_length_mm_hint=24.0)
    assert solve.debug_metadata["focal_disagreement"] == pytest.approx(0.0)
    assert solve.debug_metadata["warnings"] == []
