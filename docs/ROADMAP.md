# Roadmap

> **Vintage note (2026-07-09):** this roadmap predates the beta-0.2 work —
> the DMP layer stack, Output Desk/OCIO handoff, verified Nuke/Maya layer
> exports, DA3 depth, and the experimental hidden-geometry track all shipped
> since. Treat it as the original plan of record; CHANGELOG.md is the record
> of what actually happened.

## Deferred engineering backlog

- **Fixer render repair (2026-08-14).** `atlas_camera/inference/fixer_render_fix.py`
  shells out to NVIDIA Fixer in a Docker container (`docker/fixer/Dockerfile`)
  to repair projected-render artifacts, and `tools/generate_fixer_training_pairs.py`
  builds fine-tune pairs from baked orbits. The `AtlasRenderFix` node that drove
  it was deregistered 2026-08-03 and its wrapper deleted 2026-08-14, so the
  backend now has **no product consumer** — only its own test imports it. Kept
  deliberately: the container recipe and the pair-generation tool are the
  research trail, and INSTALL.md already tells users the node is gone.
  Do **not** treat this as dead code to sweep. If taken up again, the open
  question is not the model but the surface: it was Docker-only (the
  cosmos/transformer_engine stack has no native Windows build), which is why
  the node was removed from a pack whose selling point is that nothing needs
  Docker, a cloned research repo, or Blender. Re-landing it means either
  solving that, or accepting it as an explicitly opt-in extra.

- **Measured depth calibration (2026-07-29).**
  `atlas_camera/core/depth_calibration.py` fits a two-or-three coefficient
  correction from a monocular depth estimate onto MEASURED depth
  (`fit_depth_correction`, `choose_correction`, `apply_depth_correction`,
  `DepthCorrection` with dict round-trip). It is **implemented, tested and
  deliberately unwired** — no product code path calls it, and that is the
  current intent, not an oversight.
  It landed a week before its data source: the Record3D/LiDAR capture nodes
  arrived 2026-08-05, so nothing ever looped back to connect them. Its own
  docstring carries the blocker — the machinery is real but the COEFFICIENTS
  are not, and fitting them needs real captures. **Do not ship numbers fitted
  from the ray-cast fixtures**; those tests only prove a known distortion is
  recovered.
  Do **not** pursue it as "make Record3D captures more accurate" — a LiDAR
  capture already carries measured metric depth, so there is nothing to
  correct. The case that justifies it is **transfer**: fit on captures where
  truth is available, then apply to an ordinary photograph where it is not.
  That is the assumed-1.6 m failure on wide exteriors and AI-generated
  cityscapes, which measure out roughly ten times too small. If taken up,
  sequence it: (1) shoot a capture set spanning the scene types that fail,
  (2) persist fitted coefficients keyed by `(model_id, scene_type)` — the
  serialization already exists, the store does not, (3) select and apply them
  in the depth chain, behind the same confirm-to-adopt discipline as the other
  scale tiers, (4) only then consider a node surface.
  Fitting in disparity rather than depth is already decided and pinned by
  `test_fitting_in_the_wrong_space_leaves_far_field_error`; do not relitigate it.
  **Two acceptance criteria landed 2026-08-17, before step (3) can be built**,
  both from measured failures rather than review opinion. A fit now records the
  `predicted_range` it saw and `apply_depth_correction` returns a
  `CorrectionReport` — because a 400-sample fit off one wall at 1.00–1.30 m
  reported 98.4% improvement and 0.0074 m residual, then missed by 67% at
  50–250 m, and `MIN_SAMPLES` cannot see that (400 clears it easily; only the
  RANGE shows it); and because a plausible correction voided 76% of a valid
  frame and returned it as a bare array, which reads downstream as "no depth
  here" rather than "the correction failed here". Both are pinned by
  regression tests. **Still open from the same review**, and cheaper to fix
  before there are callers: `choose_correction` selects on in-sample error with
  no significance margin (measured picking `affine_disparity` over the true
  `scale` model on a 1.1% margin, and the in-sample ranking did not survive a
  holdout split); its blanket `except Exception` converts a caller's shape
  error into "no correction model could be fitted"; mask shape is unvalidated,
  so a `(32,128)` mask against a `(64,64)` depth is silently accepted; and the
  serialized form has no `schema_version`, which step (2) will want on day one.

- **ONNX Runtime depth backend (2026-07-13).** `tools/export_depth_v2_onnx.py`
  already exports Depth Anything V2 to ONNX with a torch-vs-onnxruntime parity
  gate (fp32; fp16 is only a downstream TensorRT/OpenVINO suggestion in the
  tool's help). Wiring it into `atlas_camera.inference.depth_estimator` at
  runtime is deferred, not dropped. Do **not** pursue it as a "make depth faster
  on CUDA" item — depth inference is not the pipeline bottleneck (mesh build +
  viewport serialization dominate). The one case that justifies it is
  **broadening hardware reach**:
  ONNX Runtime with DirectML (Windows AMD/Intel GPUs) or CoreML (Apple Silicon)
  would give GPU-accelerated depth to non-CUDA users. If taken up, sequence it:
  (1) target the non-CUDA GPU path specifically, (2) export the SegFormer
  semantic model to ONNX too (not just V2 depth), (3) add a **metric-accuracy**
  parity gate (derived camera height / ground scale, not just raw-depth
  deviation) before any fp16 path is allowed to feed metric geometry.


## Version 0.1: LatentCamera MVP

- Recover a practical still-image camera from metadata, artist constraints, or
  vanishing-point detection.
- Store horizon, vanishing points, projection scene helpers, landmarks,
  confidence, and debug metadata in a portable scene object.
- Provide `atlas.recover(...)`, `LatentCamera`, and `LatentScene` API names
  alongside the stable `atlas_camera`/`Atlas*` names.
- Build review packages with JSON, debug overlays, Maya scripts, placeholder
  DCC scripts, reports, and optional USD files.
- Provide the optional local React/FastAPI workbench with artist guide drawing,
  solve review, local guidance hooks, and a Three.js 3D lineup viewport for
  image plates, camera frustums, guides, and editable proxy objects.
- Keep ComfyUI wrappers thin and optional.

## Version 0.5: Interchange and Projection Helpers

- Improve camera optimization and confidence scoring.
- Expand JSON interchange and schema validation.
- Harden USD export and loader behavior.
- Add richer projection cards, ground planes, and scene bounding guides.
- Improve artist-guided line, horizon, scale-reference, and 3D proxy editing.
- Use viewport proxy objects as candidates for future explicit geometry
  constraints without letting UI-only state silently affect deterministic
  camera solves.

## Version 1.0: Production LatentCamera

- Stabilize the `LatentCamera` API.
- Complete Maya camera and helper creation:
  `atlas_CAMERA`, `atlas_PROJECTION_GRP`, `atlas_GEOMETRY_GRP`,
  `atlas_DEBUG_GRP`, and `atlas_REFERENCE_GRP`.
- Ship a documented CLI, plugin SDK, test suite, and repeatable validation
  harness.
- Promote Blender, Nuke, Houdini, USD, OpenCV, and JSON exporters from
  placeholders to production-ready adapters as their behavior matures.

## Version 2.0: LatentScene Expansion

- Add `LatentDepth`, proxy geometry, plane extraction, lighting estimation, and
  semantic object anchors.
- Record uncertainty and confidence maps for recovered components.
- Keep model-assisted suggestions advisory until confirmed by artists or
  pipeline rules.

## Version 3.0: Interactive Reconstruction

- Build a full inspection workspace for camera, depth, geometry, projection,
  lighting, confidence, and export.
- Support scene editing, projection workspaces, multi-image fusion, point-cloud
  registration, and Gaussian splat scene priors.
