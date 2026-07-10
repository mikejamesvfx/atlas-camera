# Changelog

User-facing release notes for Atlas Camera. Dates are branch-cut dates; the
full engineering narrative lives in CLAUDE.md's design rules and `docs/dev/`.

## Unreleased

### Output Desk slimmed: look/LUT/exposure/gamma widgets removed

- `AtlasViewportControls` dropped its `look`, `lut_path`, `exposure`, and
  `gamma` widgets (artist-requested: redundant — exposure duplicated the
  viewport's own ☀ control, gamma was a crude CSS-filter preview, look/LUT
  were inert metadata — and they crowded the Output Desk). `display_trim`
  stays. The `ATLAS_OUTPUT_PROFILE` schema still carries all four fields at
  neutral defaults, and `profile()` still accepts them as kwargs, so
  downstream exporters and old API prompts are unaffected. Every shipped
  example carrying the node was re-saved in the same commit (widgets_values
  is positional — see CLAUDE.md's append-only rule; this is the one
  sanctioned removal, done with a coordinated workflow repair).

### Staged master workflow v3

- `examples/atlas_camera_staged_master_workflow.json` updated to v3: the
  rgthree Fast Groups Bypasser now targets only stage groups
  (`matchTitle: '^[1-6] ·'`), SAM3 sky segmentation moved into the always-on
  SHARED group (so starting at any band stage still gets the sky mask rail),
  and the solve gate ships closed. Verified: a bg-band-only session (all
  other stages bypassed) runs end-to-end.

### Two distributions: `main` (working) vs `experimental` (🔬 enabled)

- Experimental nodes (`AtlasRenderFix`, `AtlasPredictHiddenGeometry`) now
  register behind an `ATLAS_EXPERIMENTAL` gate. `main` hides them by default
  (a stock ComfyUI gets a universal node menu with zero Docker/CUDA/research
  -license requirements); the `experimental` branch flips the one-line
  default. `ATLAS_EXPERIMENTAL=1`/`=0` overrides on any branch.

### Experimental Fixer render repair (Docker) 🔬

- New `AtlasRenderFix` node: repairs projected-render artifacts (torn
  silhouettes, stretched texels, hard tear-holes) in an IMAGE batch with
  NVIDIA **Fixer** (the Difix3D+ successor, single-step diffusion) —
  typically between the viewport's baked `path_frames` and a Video Combine
  node. Spike-verified on this repo's own baked orbits: ~1/3 of hard black
  tear pixels filled on a bare relief mesh, no temporal flicker added,
  ~0.3–0.45 s/frame on an RTX 5090. Known limits stated on the node: mild
  overall softening; large frame-edge reveals are not outpainted.
- Friendly licensing for this tier: Fixer repo Apache-2.0, weights under the
  NVIDIA Open Model License (commercial use permitted). Same
  user-clones-upstream pattern as the hidden-geometry track
  (`fixer_path` / `ATLAS_FIXER_PATH`).
- Inference runs in a Docker container (the cosmos/transformer_engine stack
  has no native Windows build). One-time image build from
  `docker/fixer/Dockerfile`; setup in INSTALL.md ("Experimental: Fixer
  Render Repair").

### Exact-angle patches — the render-conditioned patch loop 🔬

- `AtlasBlockoutViewport` gained a 5th appended patch output, `patch_exact`:
  📐 Extract Angle's RAW measured orbit floats
  (`azimuth_deg=… elevation_deg=… distance_scale=…`), before named-view
  snapping. Same pause/fingerprint gating as the other patch outputs.
- `AtlasAddPatchView` and `AtlasOcclusionMask` gained `exact_view_override`:
  wires from `patch_exact` and wins over the named-view controls, placing
  the patch/target camera at the artist's exact orbit (the 45° named-view
  grid would misregister a projected frame). `flip_azimuth` is ignored for
  exact overrides.
- Together with `AtlasRenderFix` this enables the training-free loop —
  orbit → render passes → 📐 → Fixer-repair the projected view → project it
  back from the identical pose onto the same geometry (`reuse_scene`,
  unseen-masked). Example: `atlas_camera_render_fix_v2_loop_workflow.json`.

### Recovered camera now faces DCC-default forward (−Z)

- Both solve paths canonicalize the (unobservable) yaw so the recovered
  camera looks down world −Z — Maya/Nuke default forward. Imported Atlas
  scenes no longer need the manual −180° Y rotation a real Maya lineup
  exposed. Gravity/pitch are provably untouched by the flip.

### Staged master workflow

- `atlas_camera_staged_master_workflow.json`: the five-stage layered DMP
  master (solve-confirm gate -> sky -> four depth bands at 80-100/60-80/
  30-60/0-30% with real LaMa clean plates -> assemble + Nuke/Maya layer
  exports), organized as titled groups driven by rgthree's Fast Groups
  Bypasser so artists work one layer at a time with the solve chain intact.
  v2: KJ Get/Set rails replace the shared-signal wiring, every stage has
  its own preview viewport (the stack up to that layer), and the shipped
  gate is closed. Also requires ComfyUI-KJNodes.
- Viewports now restore from the server payload cache on creation — no
  more empty grid after a page reload or a fully-cached re-queue.

### Solve-confirm gate

- `AtlasSolveGate` ✅: pause the expensive graph until the artist approves
  the camera solve. First Queue costs the solve + a cheap ungated preview;
  the node renders a solve summary (focal/FOV/height/pitch/confidence);
  ✅ Approve Solve re-queues with a fingerprint-scoped approval — a new
  photo or a re-solve re-arms the gate.

### Ground-anchored extrusion + roofline segmentation

- `ground_anchor` on both wall-derive nodes: building footprints from
  ray-through-base-pixel x the analytic ground plane — pure geometry, immune
  to monocular depth's "banana" warp; anchored buildings sit on the ground
  and get banana-immune heights (ray x anchored plane). Four safety gates
  (wide base pool, contact band, occlusion poison-gate, and a contamination
  gate so the anchor refines but never teleports to street clutter).
  Assumes visible ground contact — inpaint occluders off the ground line
  for best accuracy. Measured on a real street photo: rooftop heights
  130-140m -> 27-30m on anchored far facades.
- `roofline_split` (Towers & Spires): one plane per silhouette step — a row
  of buildings stops sharing a single rectangle that spans sky above its
  shorter members; each segment re-anchors on its own base.

### Skyline walls: distance modes + mask-scoped derives

- `AtlasDeriveWalls` / `AtlasDeriveTowersSpires` gained `distance_modes`
  (split each facing direction into one wall per depth mode — a city grid
  stops collapsing into two slabs; measured 2 → 7 walls on a 6K skyline
  plate) and `exclude_mask` (scope wall/object fitting to a SAM segment per
  branch and merge; ground fit stays full-frame so branches share one
  metric world). Wall/object caps raised to 64/32.

### Fixer fine-tune groundwork

- `tools/generate_fixer_training_pairs.py` packages
  {degraded render, ground truth} pairs into Fixer's training JSON
  (aligned 576×1024 letterboxing, interleaved train/test split);
  `docs/dev/fixer_finetune_data_plan.md` holds the multi-view data recipe
  and pre-registered success criteria. No training run yet.

## 0.3.0 — `release/beta-0.2` (2026-07-08 → 2026-07-09)

### Depth Anything 3 becomes the default depth model

- New `depth-anything/DA3*` backend in `inference/depth_estimator.py`;
  `DA3METRIC-LARGE` is now the default in every node combo and the
  indoor/outdoor scene presets. DA3's canonical depth is converted to metres
  using the **solved** focal length, so metric scale inherits the camera
  solve's accuracy. Measured: ~3× fewer relief-mesh tears on 2 of 4 test
  scenes; a usable mesh on a pitched shot where V2 produced zero faces.
- V2 remains selectable everywhere; core-library defaults stay V2 so
  `[neural]`-only installs keep working. Install notes: INSTALL.md
  ("Optional Depth Anything 3 Backend" — `--no-deps` required).

### Experimental hidden-geometry track (research-only) 🔬

- New `AtlasPredictHiddenGeometry` node: layered-ray models predict the
  surfaces behind foreground occluders; the node registers the stack to the
  pipeline depth, selects per-pixel hidden surfaces with a scene-adaptive
  clearing margin, and outputs a patched depth map + provenance masks +
  a per-run registration-quality report (rel MAD, gate at 0.2).
- Two swappable backends, one contract: **LaRI** (regression, ~0.2 s;
  upstream has no license — user-cloned only) and **World Tracing** r69l
  (diffusion, ~20–34 s; HF-gated, CC BY-NC-ND 4.0). Backend choice is
  per-scene — the shipped workflows encode the measured winner per scene.
- v2 layer architecture after 6 measured calibration rounds: mask-membership
  X-ray layers (depth bands lost 76–97% of near-field predictions), a
  coherence pass (`fill_gaps` diffusion + gaussian `smooth_px`), and a
  separate paint matte so geometry stays continuous while see-through gaps
  discard. Final hole-in-paint: hangar 0.07, canyon 0.19, jungle 0.26.
- Six calibrated hero workflows in `examples/`
  (`atlas_camera_hidden_geometry_*_workflow.json`) — cathedral, space
  hangar, jungle temple, canyon, steep ridge, wide valley — each with the
  full five-layer stack (base + feathered clean-plate composite, matted
  foreground, X-ray, sky dome on outdoor scenes, SAM foreground mattes
  where band edges step) and a dolly-in video bake. Seeds ship pinned.

### Viewport

- **🎨 Layers** debug overlay: opaque per-layer identity tints + legend.
- **🩻 X-ray** overlay: tints invented-geometry pixels (red = LaRI,
  blue = World Tracing).
- Orbit pivot now sits at the median sampled vertex depth on the camera's
  central view ray (was a tail-dominated bounding-box center).
- 📷 Camera View reset works again after the fingerprint-guard regression;
  navigation survives same-solve re-executions.
- Removed the dead Box/Plane/Cylinder/Person/Woman/Sedan toolbar buttons.

### Docs

- README, INSTALL, CLAUDE.md refreshed; new `docs/dev/da3_backend_test_plan.md`
  and `docs/dev/hidden_geometry_training_free_research.md`; DCC_EXPORTS,
  USER_GUIDE, ECOSYSTEM_GUIDE updated 2026-07-09; new CHANGELOG and
  THIRD_PARTY notices; three published companion guides (build-up, examples
  catalog, technical measurements).

## 0.2.0-beta — `release/beta-0.1` (2026-07-06 → 2026-07-08)

- **Complete DMP pipeline**: mesh `hole_mask` outputs, SAM-driven
  `AtlasSkyDomeLayer` with deterministic edge-extend + frame outpaint,
  per-pixel edge mattes with boundary overhang, depth-shadow occlusion mode,
  disocclusion `fill_occluded`, the `AtlasAssessImage` VLM pre-flight gate,
  📐 Extract Angle with ExecutionBlocker branch pauses, and all-in-one
  layer exports: `AtlasExportNukeLayers` (.nk) + `AtlasExportMayaLayers`
  (native .ma, verified live in Maya 2027).
- **MVP pivot**: v1 ships without diffusion patches — 🧭 Safe Zone measures
  the scene's hole-free camera envelope and clamps the orbit to it; patches
  became texture projectors onto existing geometry (`reuse_scene`);
  per-layer edge extend with invented-pixels mattes; beveled occlusion
  skirts; `frame_outpaint_px` for band layers.
- **Nuke export verified in a real Nuke** (16.1): corrected projection
  topology, new native drag-and-drop `.nk` output, real relief mesh via
  ReadGeo2.
- **Output Desk**: `AtlasRegisterPlate`/`AtlasAttachSourcePlate` float-safe
  plate refs, OCIO ACEScg full-float handoff example, output profiles.
- Vendored three.js r185 bundle (replaces the silently-broken CDN chain);
  movable point lights; projection shader sRGB encode fix; 2.5D
  inpaint-layer nodes (`AtlasDepthLayerMask`/`AtlasCleanPlateLayer`).

## 0.1.x — pre-beta (2026-06 → 2026-07-05)

- Core solve (vanishing-point + learned GeoCalib), tiered metric scale,
  geometry derivation strategies, the Three.js blockout viewport with 📽
  camera projection, camera-path authoring/baking, multi-angle patch views,
  composable derive/merge nodes, ShotCam project formats, and the
  Maya/Blender/Nuke/USD/review exporters.
