# Atlas Viewport Opportunities Research - 2026-07-04

## Executive Summary

Atlas should not replace its ComfyUI viewport wholesale with an existing custom node. The current viewport does product-specific work that generic 3D viewers do not: recovered-camera projection shading, source-photo reassembly, patch projection sources, diagnostic overlays, and pass round-tripping back through `client_data`.

The best opportunity is a staged migration toward ComfyUI's first-party Load3D frontend infrastructure for the boring viewer substrate: camera manager, controls, scene manager, model loaders, pass capture, serialization, and export menu patterns. Then keep Atlas's custom projection material, solve payload, derived-proxy construction, patch layering, and render-pass outputs as Atlas-specific modules on top.

The most concrete immediate fix remains Atlas-owned: `ProjectionSource.priority` is represented and documented, but the current viewport does not appear to apply it for deterministic patch ordering/blending.

## Atlas Viewport Requirements

Atlas's viewport needs more than a mesh preview:

- Consume solve payloads directly from `GET /atlas/camera_data/{node_id}`.
- Apply a recovered camera pose/intrinsics exactly.
- Build derived proxy geometry from Atlas JSON, not only load OBJ/GLB files.
- Use source-photo projection materials with depth/normal/mask render passes.
- Layer multiple patch projection sources, ideally by priority and facing ratio.
- Keep ComfyUI node/widget state round-trippable through `client_data`.
- Preserve diagnostic overlays: VP/horizon/ground diagram, camera HUD, exposure for grey preview.

Local relevant files:

- `C:/Users/miike/Desktop/AtlasCamera_Codex/atlas_camera/comfy/web/atlas_blockout.js`
- `C:/Users/miike/Desktop/AtlasCamera_Codex/atlas_camera/comfy/nodes.py`
- `C:/Users/miike/Desktop/AtlasCamera_Codex/atlas_camera/core/schema.py`

## Candidate Options

### 1. ComfyUI First-Party Load3D

Source: `Comfy-Org/ComfyUI_frontend`, `src/extensions/core/load3d`.

Observed capabilities:

- `Viewport3d`, `Load3d`, `SceneManager`, `CameraManager`, `ControlsManager`, `LoaderManager`, `ModelExporter`, `RecordingManager`, all with tests.
- Supported mesh formats include `stl`, `fbx`, `obj`, `gltf`, `glb`; point cloud and splat adapters handle `ply`, `spz`, `splat`, `ksplat`.
- Has `setCameraFromMatrices(extrinsics, intrinsics)`.
- Has `captureScene(width, height)` returning scene, mask, and normal.
- Has serialization patterns for `camera_info` and `model_3d_info`.
- Official extension path uses Vue/component widgets rather than ad hoc iframe-only viewers.

Fit for Atlas:

- Best long-term substrate.
- Not a drop-in replacement because Atlas requires projection shader logic and solve-driven proxy geometry.
- The right direction is to wrap/adapt first-party managers or mirror their API shape in Atlas, then migrate when custom-node access to those internals is stable enough.

### 2. MrForExample/ComfyUI-3D-Pack

Source: https://github.com/MrForExample/ComfyUI-3D-Pack

Observed capabilities:

- Very active and popular 3D node suite.
- README says it previews 3DGS and 3D meshes inside ComfyUI using `gsplat.js` and `three.js`.
- Web implementation embeds an iframe and loads viewer pages such as `threeVisualizer`.
- `threeVisualizer.js` loads OBJ, GLB, and PLY by filepath via a `/viewfile` route.

Fit for Atlas:

- Good reference for low-friction iframe preview and file serving.
- Not a replacement for Atlas's viewport because it is file-preview oriented and does not handle solve payloads, recovered-camera projection materials, patch sources, or pass round-tripping.
- Its dependency/install surface is heavy because the broader pack includes many 3D generation systems and compiled/JIT components.

### 3. kijai/ComfyUI-Hunyuan3DWrapper

Source: https://github.com/kijai/ComfyUI-Hunyuan3DWrapper

Observed capabilities:

- Active Hunyuan3D wrapper with `web/js/jsnodes.js`.
- README notes a very new ComfyUI version is required for the `Preview3D` node.

Fit for Atlas:

- It mostly confirms the direction of travel: modern 3D custom nodes are leaning on first-party Preview3D/Load3D where possible.
- Not a viewport replacement; it is model-generation workflow glue.

### 4. jtydhr88/ComfyUI-qwenmultiangle

Source: https://github.com/jtydhr88/ComfyUI-qwenmultiangle

Observed capabilities:

- Vue 3 + TypeScript + Vite ComfyUI node.
- Interactive Three.js angle widget.
- Bidirectional sync between sliders, dropdown presets, and 3D handles.
- Outputs LoRA-compatible angle prompt strings.
- Backend returns `camera_info` describing the preview camera.

Fit for Atlas:

- Strong opportunity for the `AtlasAddPatchView` UX, not for the main projection viewport.
- Atlas already uses named absolute source/patch views. This node could inspire or be integrated as a front-end angle picker for patch/source view selection, reducing the chance of azimuth/elevation mismatch.

### 5. CarlMarkswx/comfyui_GaussianViewer

Source: https://github.com/CarlMarkswx/comfyui_GaussianViewer

Observed capabilities:

- Interactive Gaussian Splatting PLY preview and high-quality image output.
- Supports optional extrinsics/intrinsics and optional reference image overlay.
- Caches camera state and outputs rendered images.
- Has iframe message passing, camera persistence, overlay, and render-result relay patterns.

Fit for Atlas:

- Useful pattern source for camera-state capture, reference-overlay UX, and "Set Camera" semantics.
- Not a replacement: it is splat-specific and GPL-licensed, which is likely incompatible with directly copying into an MIT-style project unless the whole distribution strategy accepts GPL obligations.

### 6. VRM / Multi-Model Editors

Sources inspected include `ketle-man/comfyui-vrm-pose-editor` and `xuxiao305/ComfyUI-MultiModel3D`.

Observed capabilities:

- Strong DOM-widget interaction isolation patterns, capture buttons, camera resets, model loading, background image controls, and sub-model visibility/focus controls.

Fit for Atlas:

- Good UX references for event isolation and sub-object controls.
- Not replacements for the projection viewport.

## Recommended Improvements

### P0 - Fix Patch Source Priority

`ProjectionSource.priority` should drive deterministic layering. Current Atlas JS sees patch sources but does not appear to sort or order rendering by priority. Implement either:

- sort patch source groups by ascending priority and assign `renderOrder`, or
- create a real blending/selection pass where priority resolves overlap after facing threshold.

This is the highest-confidence "missed opportunity" because it is already in the schema and UI contract.

### P1 - Adopt First-Party Load3D Concepts Incrementally

Do not replace Atlas's viewport with Load3D yet. Instead:

- Mirror Load3D's `camera_info`/`model_3d_info` style state object for Atlas viewport state.
- Steal the good shape of `captureScene(width,height)` and manager boundaries.
- Move Atlas's current monolithic JS toward modules: `AtlasCameraManager`, `AtlasSceneManager`, `AtlasProjectionManager`, `AtlasPatchSourceManager`, `AtlasPassCapture`.
- Keep the Atlas projection shader and solve JSON renderer custom.

Longer term, evaluate whether a custom Atlas node can instantiate/reuse `createLoad3d` or the same Vue/component widget mechanism without depending on private frontend internals.

### P1 - Make Patch Angle Selection Visual

Use `ComfyUI-qwenmultiangle` as the design reference:

- Replace or augment text dropdowns in `AtlasAddPatchView` with a compact angle widget.
- Show source view and patch view as two markers around the same subject-relative orbit.
- Output both named LoRA prompt terms and numeric orbit deltas.
- Preserve `flip_azimuth` as a visible handedness toggle with immediate preview feedback.

This improves actual usability more than another generic 3D viewer would.

### P1 - Add Viewport Visual Regression

Atlas's viewport is now central enough to test like product code:

- Start ComfyUI or a local fixture page.
- Feed a small synthetic solve payload with relief mesh + one patch source.
- Use Playwright screenshot/canvas checks for camera view, orbit view, projection mode, patch ordering, pass capture, and diagram overlay.

### P2 - Add Projection Debug Modes

Add artist-facing toggles:

- Source coverage heatmap: primary vs patch source ID.
- Facing-ratio mask preview.
- Out-of-frame/behind-camera discard preview.
- Priority order overlay.
- Dilation conflict warning when `preview_expand > 1` and Project is active.

These directly address the current "why did it go black?" class of issues.

### P2 - Consider GLB as Internal Preview Interchange

Atlas can export GLB relief meshes already. A useful workflow bridge would be:

- Export a GLB preview artifact for first-party Preview3D/Load3D.
- Let users inspect the raw mesh in ComfyUI's standard viewer.
- Keep Atlas Viewport as the projection-specific viewer.

This splits "is the mesh sane?" from "does projection reassemble the source photo?"

## Bottom Line

No existing ComfyUI custom node is a better drop-in replacement for Atlas Blockout Viewport. The strongest strategic path is:

1. Fix Atlas-specific projection/patch behavior now.
2. Borrow first-party Load3D architecture and state conventions.
3. Use qwenmultiangle-style UX for patch angle selection.
4. Add browser-level regression tests so viewport changes are safe.

This keeps Atlas's unique value while reducing the maintenance burden of owning every part of a 3D viewer.

## Sources

- ComfyUI JavaScript extension docs: https://docs.comfy.org/custom-nodes/js/javascript_overview
- ComfyUI extension hooks docs: https://docs.comfy.org/custom-nodes/js/javascript_hooks
- ComfyUI first-party Load3D frontend: https://github.com/Comfy-Org/ComfyUI_frontend/tree/main/src/extensions/core/load3d
- ComfyUI-3D-Pack: https://github.com/MrForExample/ComfyUI-3D-Pack
- Hunyuan3DWrapper: https://github.com/kijai/ComfyUI-Hunyuan3DWrapper
- ComfyUI-qwenmultiangle: https://github.com/jtydhr88/ComfyUI-qwenmultiangle
- comfyui_GaussianViewer: https://github.com/CarlMarkswx/comfyui_GaussianViewer
- comfyui-vrm-pose-editor: https://github.com/ketle-man/comfyui-vrm-pose-editor
