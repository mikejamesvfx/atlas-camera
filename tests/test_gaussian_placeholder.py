import pytest

from atlas_camera.gaussian import GaussianPoseEstimator, GaussianScenePrior


def test_gaussian_placeholder_raises_not_implemented():
    estimator = GaussianPoseEstimator()

    with pytest.raises(NotImplementedError):
        estimator.estimate_pose("image.png", GaussianScenePrior(), intrinsics_hint=None)

