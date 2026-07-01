"""
camera_estimator — recovers a LatentCamera from a single image.

NOT IMPLEMENTED in this pass. This module exists to document the contract
the eventual CV pipeline (perspective lines -> vanishing points -> horizon
-> LatentCamera, per the locked v0.1 scope) must satisfy, so that core,
export, and tests can all be built against a stable signature now.

Determinism contract (DECISIONS.md §5): "deterministic under a fixed,
surfaced seed" — not unconditional bit-reproducibility. RANSAC (vanishing
point fitting) and any GPU-evaluated neural step are seeded, the seed is
always returned to the caller, and re-running with the same image + seed
reproduces the same LatentCamera.
"""

from __future__ import annotations

from atlas.core.latent_camera import LatentCamera


def recover_camera(image_path: str, *, seed: int = 0) -> LatentCamera:
    """Recover a LatentCamera from a single image.

    Args:
        image_path: path to the source image.
        seed: RNG seed for RANSAC vanishing-point fitting. Defaults to 0.
              The seed used is always written onto the returned
              LatentCamera.seed — callers can reproduce a result exactly
              by reusing it.

    Raises:
        NotImplementedError: the perspective/VP/horizon inference
        pipeline is not yet implemented. This signature is locked so
        core/export/tests can be built against it now.
    """
    raise NotImplementedError(
        "camera_estimator.recover_camera is not yet implemented — "
        "core, export, and tests are built against this signature so "
        "the inference pipeline can land without touching them."
    )
