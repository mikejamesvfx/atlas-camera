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

- **Measured depth calibration (2026-07-29, WIRED 2026-08-17).**
  `atlas_camera/core/depth_calibration.py` fits a two-or-three coefficient
  correction from a monocular depth estimate onto MEASURED depth, and
  `core/depth_calibration_store.py` persists one per `(model_id, scene_type)`.
  Two nodes reach it: `AtlasFitDepthCalibration` 📐 (measured + predicted depth
  → fitted correction, `save` off by default) and `AtlasApplyDepthCalibration`
  📐 (store lookup → corrected depth).
  **Nothing auto-applies.** `AtlasDepthMap` is untouched, so every existing
  graph behaves exactly as before; a calibration reaches the depth chain only
  because an artist wired the apply node. That is deliberate —
  `ATLAS_DEPTH_MAP` feeds nine node modules, and a stored coefficient silently
  rescaling shared depth would move geometry, bands, exports and the viewport
  at once, from a file the graph never names. Pinned by
  `test_atlas_depth_map_did_not_grow_a_calibration_input`.
  The remaining blocker is step (1) and it is a SHOOTING task, not an
  engineering one: the machinery is real but the COEFFICIENTS are not, and an
  empty store is the correct state of a fresh clone. **Do not ship numbers
  fitted from the ray-cast fixtures**; those tests only prove a known
  distortion is recovered.
  Do **not** pursue it as "make Record3D captures more accurate" — a LiDAR
  capture already carries measured metric depth, so there is nothing to
  correct. The case that justifies it is **transfer**: fit on captures where
  truth is available, then apply to an ordinary photograph where it is not.
  That is the assumed-1.6 m failure on wide exteriors and AI-generated
  cityscapes, which measure out roughly ten times too small. The sequence was:
  (1) shoot a capture set spanning the scene types that fail — **STILL OPEN,
  and the only remaining blocker**; (2) persist fitted coefficients keyed by
  `(model_id, scene_type)` — **done**, `core/depth_calibration_store.py`, exact
  lookup with no fallback, because a near-miss is how an interior coefficient
  rescales an exterior; (3) select and apply in the depth chain behind
  confirm-to-adopt — **done as an EXPLICIT node** rather than an automatic
  step, which is the confirm-to-adopt discipline expressed as wiring: the
  artist adds the node or no calibration is applied; (4) a node surface —
  **done**, and taken with (3) rather than after it, since without a fit node
  there is no way to produce a coefficient in the first place and step (1)
  would have nothing to write into.
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
  regression tests. The remaining four from the same review landed the same
  day, all while the module still had no callers to migrate: selection now runs
  on a **held-out** stride-2 split with a 5% margin and a simplest-first
  tie-break, then refits the winner on every sample (it was picking
  `affine_disparity` over the true `scale` model on a 1.1% in-sample margin —
  smaller than the noise that produced it); `choose_correction` validates the
  pair once up front so a caller's shape error keeps its "resample the LiDAR
  depth first" message instead of becoming "no correction model could be
  fitted"; a mask must now match `predicted`'s SHAPE, not merely its size, so
  the `(32,128)`-against-`(64,64)` case that silently selected the wrong pixels
  is refused; and the serialized form carries `SCHEMA_VERSION`, rejects a
  future version rather than reading fields with the wrong meanings, and
  refuses an unknown `model` at load instead of at apply. Step (2) — the store
  keyed by `(model_id, scene_type)` — can therefore be built against a
  versioned format from its first write.

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
