# Gaussian Splats

3DGS mode is not implemented yet.

The first target is a clean interface for matching a new image to an existing
splat, point-cloud, or COLMAP scene prior.

Target workflow:

```text
target image
+ existing 3D Gaussian Splat / point cloud / COLMAP reconstruction
+ known or guessed intrinsics
-> estimate camera pose in the scene prior
-> render-compare refine
-> export solved camera
```

Current interfaces:

- `GaussianScenePrior`
- `GaussianPoseEstimator.estimate_pose(...)`

`estimate_pose` raises `NotImplementedError` until there is a real implementation
and validation dataset.

## Cautions

- Do not present 3DGS camera registration as a solved feature.
- Do not add fake pose outputs.
- Do not require 3DGS dependencies for the Atlas core package.

