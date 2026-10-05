# TODOS

## feat/pixal3d-object-hdr follow-ups (CEO review 2026-10-03)

- [ ] **Land the research workflows in a follow-up PR.** These four files were deferred out of
  the code merge so the code diff (~6.6k lines) fits /ultrareview's 8,000-line limit:
  the Pixal3D object, Pixal3D/TRELLIS.2 image-to-model, HDR still (matrixZone) and HDR clip
  research workflows (14,011 lines in all), kept on branch `feat/research-workflows`
  (their Pixal3D graph already carries the appended SAM3 override widget). They need ComfyUI V135+, are not shipping examples, and no test pins them.
  CHANGELOG, README, INSTALL, NODE_CATALOG and USER_GUIDE mention `research/`, so the code PR must
  forward-reference or drop those lines. Decision: CEO review D16.

- [x] **Consolidate duplicated helpers (P3, human: M / CC: S).** DONE 2026-10-03 where the copies were truly equal. Each moved copy was checked byte for byte (values, dtype, shape) against the old code on random inputs, and every test still passes. Two pairs are left as they are, on purpose (see below).
  - DONE: the sRGB curve now lives once, in the new `core/srgb.py`. It has a float64 `srgb_to_linear`/`linear_to_srgb` (copies, clipped) and a float32 `srgb_to_linear_f32` (built in place). `generated_mesh` and `matrixzone` re-export the old names. `preview_codec` and `raw/pipeline` (the JPEG path) call it now. The float32 inline form in `raw/pipeline` matches `srgb_to_linear_f32` bit for bit on numpy 1.26 and 2.4.
  - KEPT: `raw/decode.srgb_encode` works in the input's dtype and returns float32. Routing it through the float64 curve would change the last bit. `comfy/nodes_matrixzone._tonemap_preview` applies a Reinhard tonemap first and does not clip, so it is a different function.
  - DONE: `mask_ops.erode(mask, iterations, connectivity=, wrap=)`. `generated_mesh._erode` (4-connected) and `adherence._erode` (8-connected) are now thin wrappers that pass their own connectivity.
  - KEPT: `matrixzone._box_mean` and `plate/deband._box_blur_2d` differ. `_box_mean` casts to float32 between the two passes and returns float32, which is the memory fix for 8K plates. `_box_blur_2d` stays float64 throughout and returns a copy when `r <= 0`. One shared version would change one caller's numbers or memory use.
  - DONE: the GLB framing (`pad4` plus the header and chunk headers) now lives in the new `exporters/_glb.py`, and both writers use it. `scene_glb._pad4` stays as an alias. Each writer still builds its own JSON and buffers: one builds them in memory, the other streams them and deletes a partial file on failure. GLB sha256 hashes match before and after for all 24 deterministic GLBs the five GLB test files write: all 14 from `export_relief_mesh_glb` and 10 from `write_scene_glb`. The other 6 are `AtlasSceneTo3D` node outputs, which differ from run to run even without this change.
  - DONE: `GENERATED_SOURCE` now lives in `core/generated_mesh.py`. `nodes_object_mesh` (which re-exports it), `scene_health` (imports it inside the function, to avoid an import cycle) and `usd_exporter` all use it. `nodes_inpaint._SAM3_HF_BACKEND` is now an alias of `sam3_core_backend.HF_BACKEND`.
  - Already fixed before this pass: the Rec.709 luma tuple in `matrixzone.py`. F-5 (b819044) replaced it with the single constant `LUMA_AP1`.
  Why: copies drift; a fix lands in one and not the others. Decision: CEO review D25.

- [x] **Mirror the F-3 planner clamp in atlas_bridge (P2, human: S / CC: S).** DONE 2026-10-03: the bridge had the same bug, with identical numbers. Fixed on `atlas-unreal` branch `fix/matrixzone-interior-clamp`, off `main`, with a sweep test. It is NOT merged into atlas-unreal's main or into `raf-anchored-camera-solves`; merging is your call.
  - What: if `atlas_bridge.matrixzone.plan()` has the same unclamped interior zone start, apply the same clamp and sweep test there.
  - Why: `core/matrixzone.py` is a parity-pinned port. Fixing only Atlas lets the two planners drift on non-default grids.
  - Context: F-3 clamps Atlas's `_axis`. The parity pins cover only the UHD worked cases, which the bug doesn't affect.
  - Depends on: F-3 landing. Decision: CEO review D36.

- [ ] **Decide a repo-wide export write-path policy (P3, human: M / CC: S).**
  - What: decide where Atlas exporters may write: ComfyUI's output directory, the AtlasProject delivery tree, and absolute paths only with an explicit opt-in.
  - Why: `AtlasExport` (`nodes_export.py:140`), the review package and `atlas_solve.json` write to any path a workflow names. That's by design for DCC delivery, but it means a workflow shared by someone else can write anywhere the ComfyUI user can.
  - Pros: one consistent rule instead of per-node behaviour, with F-1's helper as the template.
  - Cons: DCC pipelines that write straight into show folders would need an opt-in, and several shipped nodes change (widgets append-only).
  - Context: F-1 (CEO D20) fixes only the new SceneTo3D/matrixZone helper.
  - Depends on: F-1 landing. Decision: eng review D6.
  - Same class, found by the pre-landing review on 2026-10-03:
    - `AtlasHDRVertexTransfer` reads any existing absolute `hdr_exr_path`. Only relative paths are confined to output/input.
    - `AtlasProject.project_root` accepts any absolute path, so the new project-routed GLB/EXR writes from `AtlasSceneTo3D` and `AtlasMatrixZoneStitch` can land anywhere.
    - Decide both together with the exporters.

- [x] **Reserve export names atomically across processes (P3, human: S / CC: S).** RESOLVED 2026-10-03: `node_helpers._claim_name` takes each `<stem>_NNNNN` with an `O_EXCL` placeholder, and the writers release it in a `finally` block.
  - What: `node_helpers._next_counter` picks `<stem>_NNNNN` by scanning the folder, with no reservation.
  - Why: two ComfyUI processes exporting the same shot and prefix at once can choose the same number and overwrite each other. A budget refusal could also unlink a GLB the other job wrote.
  - Fix idea: create the file with `O_EXCL`, retrying on collision. Only delete files this invocation created.
  - Context: one ComfyUI process runs prompts one at a time, so this needs multi-process or shared-folder setups. Found by the pre-landing review on 2026-10-03 (Codex).

- [x] **Report narrowed overlap when the planner clamps a zone (P3, human: S / CC: S).** RESOLVED 2026-10-03: a 181,280-plan sweep showed the clamp can only WIDEN an overlap (edge zones are pinned; a clamped interior zone slides toward its edge neighbour). `plan_still` now records `overlap_actual` and the split warns if any zone ever falls below the request, as a safety net.
  - What: `core.matrixzone._axis` clamps interior zones into the canvas (F-3). The clamp keeps every cell covered, but the overlap facing the next zone can fall below `overlap_min/2`, so the feather band is narrower, and nothing reports it.
  - Fix idea: add a per-zone `overlap_px` actual-vs-requested field to the plan, and a report line in the split node.
  - Depends on: pairs with the atlas_bridge clamp mirror TODO, since the two repos must agree on clamped grids. Found by the pre-landing review on 2026-10-03.

- [ ] **Run the node-level regression tests in CI (P2, human: M / CC: S).**
  - What: CI installs no torch and no OpenImageIO, so most new node tests skip there. That covers matrixZone nodes, object-mesh refusals, the AtlasInput SAM3 contract, project delivery and scene export.
  - Fix idea: move torch-free logic (report builders, provenance params, refusals on numpy input) into numpy-only tests, or add a CI job with `torch` CPU wheels and OIIO.
  - Why: today these regressions are only proven on a dev machine with ComfyUI's environment. Found by the pre-landing review on 2026-10-03.
