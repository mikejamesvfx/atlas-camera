"""Experimental 3DGS interfaces.

No 3D Gaussian Splat solving is implemented in this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GaussianScenePrior:
    source_path: str | None = None
    metadata: dict[str, Any] | None = None


class GaussianPoseEstimator:
    def estimate_pose(self, image, scene_prior: GaussianScenePrior, intrinsics_hint=None):
        raise NotImplementedError(
            "3DGS scene-prior pose estimation is planned but not implemented."
        )

