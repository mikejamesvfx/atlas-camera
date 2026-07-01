"""
Atlas — recover the latent world.

Simple API (vision-doc surface):

    import atlas
    scene = atlas.recover("castle.png")
    scene.camera.to_maya()
    scene.export.json()

Advanced API:

    from atlas.core import LatentCamera
    camera = LatentCamera.with_estimated_focal(...)
    camera.to_json()
    camera.to_maya()
"""

from __future__ import annotations

from atlas.core.latent_camera import LatentCamera
from atlas.core.scene import LatentScene
from atlas.core.confidence import ConfidenceModel
from atlas.inference.camera_estimator import recover_camera

__version__ = "0.1.0"

# Backwards-compatibility aliases (Codex Instruction doc naming).
EstimatedCamera = LatentCamera
CameraResult = LatentCamera

__all__ = [
    "LatentCamera", "LatentScene", "ConfidenceModel",
    "recover", "EstimatedCamera", "CameraResult", "__version__",
]


def recover(image_path: str, *, seed: int = 0) -> LatentScene:
    """Recover the LatentScene (currently: just a LatentCamera) hidden
    inside a single image."""
    camera = recover_camera(image_path, seed=seed)
    return LatentScene(camera=camera)
