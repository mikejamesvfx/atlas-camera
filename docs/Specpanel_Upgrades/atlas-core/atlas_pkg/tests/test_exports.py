import math
import re
import tempfile
import unittest
from pathlib import Path

from atlas.core import camera_math
from atlas.core.confidence import ConfidenceModel
from atlas.core.latent_camera import LatentCamera
from atlas.export import maya as maya_export
from atlas.export import json_export

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
        principal_point_px=(1010.0, 500.0),   # offset from center: +50, -40
        film_offset=(0.0, 0.0),
        world_matrix=IDENTITY,
        view_matrix=IDENTITY,
        projection_matrix=IDENTITY,
        confidence=ConfidenceModel(global_score=0.8, individual_metrics={"focal": 0.9}),
        focal_length_mm=50.0,
        translation=(10.0, 5.0, 100.0),
        rotation_euler=(2.0, -3.0, 0.0),
        notes=[],
    )
    defaults.update(overrides)
    return LatentCamera(**defaults)


def _grab_float(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise AssertionError(f"pattern not found in Maya output: {pattern}")
    return float(m.group(1))


# Required test 3: Maya export value conversion (mm->inches, px->normalized)
class TestMayaExportConversion(unittest.TestCase):

    def setUp(self):
        self.cam = make_camera()
        self.ma_text = self.cam.to_maya()

    def test_aperture_converted_to_inches(self):
        expected = camera_math.mm_to_inches(36.0)
        actual = _grab_float(r'horizontalFilmAperture"\s+([\-0-9.]+)', self.ma_text)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_film_offset_is_normalized_not_pixels(self):
        offset_x = _grab_float(r'horizontalFilmOffset"\s+([\-0-9.]+)', self.ma_text)
        # principal point is 50px right of center on a 1920px / 36mm
        # sensor — normalized offset must be small (fraction of aperture),
        # not the raw 50.0 pixel value.
        self.assertLess(abs(offset_x), 0.05)
        self.assertNotAlmostEqual(offset_x, 50.0, places=1)

    def test_frozen_node_names_present(self):
        for name in (
            maya_export.NODE_CAMERA, maya_export.NODE_PROJECTION_GRP,
            maya_export.NODE_GEOMETRY_GRP, maya_export.NODE_DEBUG_GRP,
            maya_export.NODE_REFERENCE_GRP,
        ):
            self.assertIn(name, self.ma_text)

    def test_inferred_focal_emits_warning_comment(self):
        cam = LatentCamera.with_estimated_focal(
            fov_deg=50.0, sensor_width_mm=None,
            image_width=1920, image_height=1080,
            principal_point_px=(960.0, 540.0), film_offset=(0.0, 0.0),
            world_matrix=IDENTITY, view_matrix=IDENTITY, projection_matrix=IDENTITY,
            confidence=ConfidenceModel(global_score=0.8, individual_metrics={"focal": 0.9}),
        )
        text = cam.to_maya()
        self.assertIn("WARNING", text)

    def test_export_fails_without_focal_length(self):
        cam = make_camera(focal_length_mm=None)
        with self.assertRaises(ValueError):
            cam.to_maya()


# Required test 4: JSON export and import
class TestJSONExportImport(unittest.TestCase):

    def test_write_and_read_json_file(self):
        cam = make_camera()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "camera.json"
            json_export.write_json(cam, path)
            self.assertTrue(path.exists())
            restored = json_export.read_json(path)
            self.assertEqual(restored.to_dict(), cam.to_dict())


# Panel-requested: golden-camera round trip. A synthetic camera with a
# known pose exports to Maya and the values read back within tolerance.
# This is the test that fails loudly if someone flips a coordinate-
# convention sign (DECISIONS.md §1).
class TestGoldenCameraRoundTrip(unittest.TestCase):

    def test_maya_translation_matches_known_remap(self):
        # OpenCV translation (+Y down, +Z into scene) -> Maya (+Y up,
        # -Z into scene) flips Y and Z, leaves X unchanged.
        cam = make_camera(translation=(10.0, 5.0, 100.0))
        text = cam.to_maya()
        tx = _grab_float(r'translate"\s+-type\s+"double3"\s+([\-0-9.]+)', text)
        # second and third floats on the same line
        m = re.search(
            r'translate"\s+-type\s+"double3"\s+([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)',
            text,
        )
        self.assertIsNotNone(m)
        x, y, z = (float(g) for g in m.groups())
        self.assertAlmostEqual(x, 10.0, places=5)
        self.assertAlmostEqual(y, -5.0, places=5)
        self.assertAlmostEqual(z, -100.0, places=5)

    def test_aperture_round_trip_within_tolerance(self):
        cam = make_camera(sensor_width_mm=24.0, sensor_height_mm=13.5)
        text = cam.to_maya()
        w_in = _grab_float(r'horizontalFilmAperture"\s+([\-0-9.]+)', text)
        h_in = _grab_float(r'verticalFilmAperture"\s+([\-0-9.]+)', text)
        self.assertAlmostEqual(camera_math.inches_to_mm(w_in), 24.0, places=4)
        self.assertAlmostEqual(camera_math.inches_to_mm(h_in), 13.5, places=4)

    def test_focal_length_passes_through_unchanged(self):
        cam = make_camera(focal_length_mm=85.0)
        text = cam.to_maya()
        focal = _grab_float(r'focalLength"\s+([\-0-9.]+)', text)
        self.assertAlmostEqual(focal, 85.0, places=5)


if __name__ == "__main__":
    unittest.main()
