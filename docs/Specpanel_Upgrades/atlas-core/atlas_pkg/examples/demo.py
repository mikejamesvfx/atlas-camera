"""
Demo: construct the same LatentCamera shown in the Recovery Chamber UI,
export it to Maya and JSON, and print both. Run with:

    python3 examples/demo.py

This bypasses atlas.recover() (inference isn't implemented yet — see
atlas/inference/camera_estimator.py) and builds the LatentCamera directly,
the way the inference pipeline eventually will internally.
"""

from atlas.core.confidence import ConfidenceModel
from atlas.core.latent_camera import LatentCamera
from atlas.core.scene import LatentScene

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def build_demo_camera() -> LatentCamera:
    confidence = ConfidenceModel(
        global_score=0.84,
        individual_metrics={
            "horizon": 0.91, "vp1": 0.87, "vp2": 0.79,
            "vp3": 0.63, "focal": 0.82, "extrinsics": 0.78, "sensor": 0.95,
        },
    )
    # focal_length_mm derived from FOV with no recovered sensor size —
    # exercises the §3 fallback path, same as the UI's "~EST" tag.
    return LatentCamera.with_estimated_focal(
        fov_deg=41.8,
        sensor_width_mm=None,
        image_width=1920,
        image_height=1080,
        principal_point_px=(960.2, 541.8),
        film_offset=(0.0021, -0.0083),
        world_matrix=IDENTITY,
        view_matrix=IDENTITY,
        projection_matrix=IDENTITY,
        confidence=confidence,
        translation=(12.347, -4.721, 248.612),
        rotation_euler=(1.2, -3.4, 0.05),
        horizon_line=(0.0, 1.0, -0.043),
        vanishing_points=[(-1.217, 0.003), (2.143, -0.002), (0.002, -7.592)],
        seed=42,
    )


if __name__ == "__main__":
    camera = build_demo_camera()
    scene = LatentScene(camera=camera)

    print("=" * 70)
    print("LatentCamera — focal_inferred:", camera.focal_inferred)
    print("notes:", camera.notes)
    print("=" * 70)
    print(scene.export.maya())
    print("=" * 70)
    print(scene.export.json())
