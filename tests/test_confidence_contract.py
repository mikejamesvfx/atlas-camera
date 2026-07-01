import random

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
