import math
import unittest

from atlas.core import camera_math


class TestUnitConversions(unittest.TestCase):

    def test_mm_inches_round_trip(self):
        for mm in (1.0, 36.0, 24.892, 0.0):
            self.assertAlmostEqual(
                camera_math.inches_to_mm(camera_math.mm_to_inches(mm)),
                mm, places=9,
            )

    def test_mm_to_inches_known_value(self):
        # 36mm full-frame width is the textbook reference value.
        self.assertAlmostEqual(camera_math.mm_to_inches(36.0), 1.41732, places=4)

    def test_px_normalized_offset_round_trip(self):
        aperture_mm, img_dim = 36.0, 1920
        for px_offset in (-200.3, 0.0, 5.5, 480.0):
            norm = camera_math.px_to_normalized_offset(px_offset, aperture_mm, img_dim, 50.0)
            back_px = camera_math.normalized_offset_to_px(norm, aperture_mm, img_dim)
            self.assertAlmostEqual(back_px, px_offset, places=6)

    def test_fov_focal_round_trip(self):
        sensor = 36.0
        for fov in (10.0, 45.0, 90.0, 120.0):
            focal = camera_math.fov_to_focal_length(fov, sensor)
            back_fov = camera_math.focal_length_to_fov(focal, sensor)
            self.assertAlmostEqual(back_fov, fov, places=6)

    def test_fov_focal_known_value(self):
        # 90 degree horizontal FOV on a 36mm sensor -> 18mm focal length.
        self.assertAlmostEqual(
            camera_math.fov_to_focal_length(90.0, 36.0), 18.0, places=4,
        )

    def test_estimate_focal_with_fallback_flags_inference(self):
        focal, used_fallback, penalty = camera_math.estimate_focal_with_fallback(
            fov_deg=60.0, sensor_width_mm=None,
        )
        self.assertTrue(used_fallback)
        self.assertGreater(penalty, 0.0)
        expected = camera_math.fov_to_focal_length(60.0, camera_math.FALLBACK_SENSOR_WIDTH_MM)
        self.assertAlmostEqual(focal, expected, places=6)

    def test_estimate_focal_no_fallback_when_sensor_known(self):
        focal, used_fallback, penalty = camera_math.estimate_focal_with_fallback(
            fov_deg=60.0, sensor_width_mm=23.5,
        )
        self.assertFalse(used_fallback)
        self.assertEqual(penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
