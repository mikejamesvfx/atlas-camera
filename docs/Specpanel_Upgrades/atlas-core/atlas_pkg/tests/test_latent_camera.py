import unittest

from atlas.core.confidence import ConfidenceModel
from atlas.core.latent_camera import LatentCamera
from atlas.core.base import SCHEMA_VERSION

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def make_camera(**overrides) -> LatentCamera:
    defaults = dict(
        image_width=1920,
        image_height=1080,
        sensor_width_mm=36.0,
        sensor_height_mm=20.25,
        principal_point_px=(960.2, 541.8),
        film_offset=(0.0021, -0.0083),
        world_matrix=IDENTITY,
        view_matrix=IDENTITY,
        projection_matrix=IDENTITY,
        confidence=ConfidenceModel(
            global_score=0.84,
            individual_metrics={
                "horizon": 0.91, "vp1": 0.87, "vp2": 0.79,
                "vp3": 0.63, "focal": 0.82, "extrinsics": 0.78,
                "sensor": 0.95,
            },
        ),
        focal_length_mm=47.3,
        rotation_euler=(1.2, -3.4, 0.05),
        translation=(12.347, -4.721, 248.612),
        horizon_line=(0.0, 1.0, -0.043),
        vanishing_points=[(-1.217, 0.003), (2.143, -0.002), (0.002, -7.592)],
        notes=[],
    )
    defaults.update(overrides)
    return LatentCamera(**defaults)


# Required test 1: serialization / deserialization round trip
class TestSerializationRoundTrip(unittest.TestCase):

    def test_to_dict_from_dict_round_trip(self):
        cam = make_camera()
        restored = LatentCamera.from_dict(cam.to_dict())
        self.assertEqual(restored.to_dict(), cam.to_dict())

    def test_to_json_from_json_round_trip(self):
        cam = make_camera()
        restored = LatentCamera.from_json(cam.to_json())
        self.assertEqual(restored.to_dict(), cam.to_dict())

    def test_schema_version_present(self):
        cam = make_camera()
        d = cam.to_dict()
        self.assertIn("schema_version", d)
        self.assertEqual(d["schema_version"], SCHEMA_VERSION)


# Required test 2: matrix shape validation
class TestMatrixShapeValidation(unittest.TestCase):

    def test_accepts_valid_4x4(self):
        make_camera()  # should not raise

    def test_rejects_wrong_row_count(self):
        bad = IDENTITY[:3]
        with self.assertRaises(ValueError):
            make_camera(world_matrix=bad)

    def test_rejects_wrong_col_count(self):
        bad = [row[:3] for row in IDENTITY]
        with self.assertRaises(ValueError):
            make_camera(view_matrix=bad)


# Required test 5: missing focal length fallback
class TestFocalLengthFallback(unittest.TestCase):

    def test_fallback_flags_inferred_and_lowers_confidence(self):
        base_conf = ConfidenceModel(global_score=0.8, individual_metrics={"focal": 0.9})
        cam = LatentCamera.with_estimated_focal(
            fov_deg=60.0,
            sensor_width_mm=None,
            image_width=1920, image_height=1080,
            principal_point_px=(960.0, 540.0),
            film_offset=(0.0, 0.0),
            world_matrix=IDENTITY, view_matrix=IDENTITY, projection_matrix=IDENTITY,
            confidence=base_conf,
        )
        self.assertTrue(cam.focal_inferred)
        self.assertLess(cam.confidence.get_metric("focal"), 0.9)
        self.assertTrue(any("inferred" in n.lower() for n in cam.notes))

    def test_no_fallback_when_sensor_known(self):
        base_conf = ConfidenceModel(global_score=0.8, individual_metrics={"focal": 0.9})
        cam = LatentCamera.with_estimated_focal(
            fov_deg=60.0,
            sensor_width_mm=23.5,
            image_width=1920, image_height=1080,
            principal_point_px=(960.0, 540.0),
            film_offset=(0.0, 0.0),
            world_matrix=IDENTITY, view_matrix=IDENTITY, projection_matrix=IDENTITY,
            confidence=base_conf,
        )
        self.assertFalse(cam.focal_inferred)
        self.assertEqual(cam.confidence.get_metric("focal"), 0.9)
        self.assertEqual(cam.notes, [])


# Required test 6 + panel property test: confidence always clamped [0,1]
class TestConfidenceClamping(unittest.TestCase):

    def test_global_score_clamped_on_construction(self):
        cm = ConfidenceModel(global_score=1.4, individual_metrics={})
        self.assertEqual(cm.global_score, 1.0)
        cm2 = ConfidenceModel(global_score=-0.3, individual_metrics={})
        self.assertEqual(cm2.global_score, 0.0)

    def test_individual_metrics_clamped_on_construction(self):
        cm = ConfidenceModel(global_score=0.5, individual_metrics={"focal": 5.0, "vp1": -2.0})
        self.assertEqual(cm.individual_metrics["focal"], 1.0)
        self.assertEqual(cm.individual_metrics["vp1"], 0.0)

    def test_set_metric_always_clamped(self):
        cm = ConfidenceModel(global_score=0.5, individual_metrics={})
        cm.set_metric("focal", 99.0)
        self.assertEqual(cm.get_metric("focal"), 1.0)
        cm.set_metric("focal", -99.0)
        self.assertEqual(cm.get_metric("focal"), 0.0)

    def test_property_random_inputs_always_in_range(self):
        # Panel-requested property test: regardless of what garbage the
        # inference pipeline hands back, confidence stays in [0, 1].
        import random
        rng = random.Random(0)
        for _ in range(200):
            g = rng.uniform(-50, 50)
            metrics = {k: rng.uniform(-50, 50) for k in ("horizon", "vp1", "focal")}
            cm = ConfidenceModel(global_score=g, individual_metrics=metrics)
            self.assertGreaterEqual(cm.global_score, 0.0)
            self.assertLessEqual(cm.global_score, 1.0)
            for v in cm.individual_metrics.values():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
