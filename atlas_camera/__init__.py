"""Atlas public package interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasHorizon,
    AtlasIntrinsics,
    AtlasProjectionScene,
    AtlasProxyPrimitive,
    AtlasShotCam,
    AtlasSolve,
    AtlasVanishingPoint,
    LatentCamera,
    LatentComponent,
    LatentScene,
    ProjectionSource,
)
from atlas_camera.core.confidence import ConfidenceModel
from atlas_camera.core.solver import solve_still_image


def recover(
    image_path: str | Path,
    *,
    method: str = "vanishing_points",
    image_size: tuple[int, int] | None = None,
    intrinsics_hint: dict[str, Any] | None = None,
    detect_vanishing_points: bool = False,
    debug_overlay_path: str | Path | None = None,
    camera_height: float = 1.6,
    detection_options: dict[str, Any] | None = None,
    weights: str = "pinhole",
    device: str | None = None,
    seed: int = 0,
) -> LatentScene:
    """Recover the latent scene representation currently supported by Atlas.

    ``method`` selects the camera engine:

    - ``"vanishing_points"`` (default) — deterministic geometric solve. Fast and
      dependency-light, but fragile on AI-generated images whose perspective is
      only locally consistent.
    - ``"learned"`` — single-image neural prior (GeoCalib) predicting focal length
      and gravity. Far more robust on AI renders; requires the ``[neural]`` extra
      (torch + geocalib). See :func:`atlas_camera.core.solver.solve_still_image_learned`.

    Returns an :class:`AtlasSolve`/``LatentScene`` with the recovered camera,
    horizon, confidence, and debug metadata. Future releases will expand the same
    object toward depth, geometry, lighting, and semantic scene components.
    """

    if method == "learned":
        from atlas_camera.core.solver import solve_still_image_learned

        return solve_still_image_learned(
            image_path,
            image_size=image_size,
            camera_height=camera_height,
            weights=weights,
            device=device,
            seed=seed,
        )
    if method not in ("vanishing_points", "vp", "geometric"):
        raise ValueError(
            f"Unknown recover method {method!r}. Use 'vanishing_points' or 'learned'."
        )

    return solve_still_image(
        image_path,
        image_size=image_size,
        intrinsics_hint=intrinsics_hint,
        detect_vanishing_points=detect_vanishing_points,
        debug_overlay_path=debug_overlay_path,
        camera_height=camera_height,
        detection_options=detection_options,
        seed=seed,
    )

__all__ = [
    "AtlasCamera",
    "AtlasExtrinsics",
    "AtlasHorizon",
    "AtlasIntrinsics",
    "AtlasProjectionScene",
    "AtlasProxyPrimitive",
    "AtlasShotCam",
    "AtlasSolve",
    "AtlasVanishingPoint",
    "ConfidenceModel",
    "LatentCamera",
    "LatentComponent",
    "LatentScene",
    "ProjectionSource",
    "recover",
]

__version__ = "0.4.0"
