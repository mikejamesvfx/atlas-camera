import random

import pytest

from atlas_camera.core.confidence import ConfidenceModel, LATENT_CAMERA_CONFIDENCE_KEYS


def test_confidence_model_clamps_global_and_metrics():
    confidence = ConfidenceModel(
        global_score=1.7,
        individual_metrics={"focal": -3.0, "vp1": 99.0},
    )

    assert confidence.global_score == 1.0
    assert confidence.individual_metrics["focal"] == 0.0
    assert confidence.individual_metrics["vp1"] == 1.0


def test_latent_camera_confidence_uses_fixed_key_set():
    confidence = ConfidenceModel.for_latent_camera(global_score=0.4, defaults=0.2)

    assert set(confidence.individual_metrics) == set(LATENT_CAMERA_CONFIDENCE_KEYS)
    assert confidence.metric_semantics == "relative_heuristic"


def test_random_confidence_inputs_stay_in_range():
    rng = random.Random(0)
    for _ in range(200):
        confidence = ConfidenceModel(
            global_score=rng.uniform(-100.0, 100.0),
            individual_metrics={
                "horizon": rng.uniform(-100.0, 100.0),
                "focal": rng.uniform(-100.0, 100.0),
            },
        )
        assert 0.0 <= confidence.global_score <= 1.0
        assert all(0.0 <= value <= 1.0 for value in confidence.individual_metrics.values())


def test_scale_and_depth_keys_appended():
    """P0 trust tier (2026-07-18): the key tuple grew append-only."""
    assert LATENT_CAMERA_CONFIDENCE_KEYS[-2:] == ("scale", "depth")
    seeded = ConfidenceModel.for_latent_camera()
    assert "scale" in seeded.individual_metrics
    assert "depth" in seeded.individual_metrics


def test_old_confidence_dict_without_new_keys_loads():
    old = {"global_score": 0.7,
           "individual_metrics": {"horizon": 0.8, "focal": 0.75},
           "metric_semantics": "relative_heuristic"}
    model = ConfidenceModel.from_dict(old)
    assert model.individual_metrics.get("scale") is None  # absent, not an error
    assert model.global_score == 0.7


def test_learned_solve_populates_scale_and_depth(monkeypatch):
    pytest.importorskip("numpy")
    import atlas_camera.core.solver as solver
    import atlas_camera.inference.learned_prior as lp

    class FakePrior:
        image_width, image_height = 640, 480
        focal_px = 500.0
        up_cam = (0.0, -1.0, 0.0)
        pitch_deg = 0.0
        roll_deg = 0.0
        focal_uncertainty_px = None
        source_model = "fake"

        def to_dict(self):
            return {}

    monkeypatch.setattr(lp, "estimate_camera_prior", lambda *a, **k: FakePrior())

    # Assumed tier: scale is a flagged guess, no depth ran.
    s = solver.solve_still_image_learned("unused.png", camera_height=1.6)
    m = s.camera.confidence.individual_metrics
    assert m["scale"] == pytest.approx(0.15)
    assert m["depth"] == 0.0
    assert s.debug_metadata["scale_source"] == "assumed_default"

    # Reference tier: the winning tier's consistency becomes the metric.
    monkeypatch.setattr(solver, "resolve_reference_scale",
                        lambda *a, **k: {"camera_height": 10.0,
                                         "confidence": 0.7, "references": []})
    s2 = solver.solve_still_image_learned(
        "unused.png", camera_height=1.6,
        scale_references=[{"reference_id": "door", "top_px": (1, 1),
                           "base_px": (1, 100)}])
    m2 = s2.camera.confidence.individual_metrics
    assert m2["scale"] == pytest.approx(0.7)
    assert s2.debug_metadata["scale_source"] == "reference_object"
