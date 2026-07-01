# Atlas Camera 3DGS Hooks

This module is intentionally experimental.

Atlas Camera does not currently estimate camera poses from 3D Gaussian Splats,
point clouds, or COLMAP reconstructions. The first target is a clean interface:

```text
target image
+ existing 3D Gaussian Splat / point cloud / COLMAP reconstruction
+ known or guessed intrinsics
-> estimate camera pose in the scene prior
-> render-compare refine
-> export solved camera
```

Current contents:

- `GaussianScenePrior`: lightweight container for a future scene prior.
- `GaussianPoseEstimator`: interface that raises `NotImplementedError`.

