# TODOS

## feat/pixal3d-object-hdr follow-ups (CEO review 2026-10-03)

- [ ] **Land the research workflows in a follow-up PR.** These four files were deferred out of
  the code merge so the code diff (~6.6k lines) fits /ultrareview's 8,000-line limit:
  `research/atlas_pixal3d_object_workflow.json`, `research/3d_pixal3d_trellis2_image_to_model.json`,
  `research/atlas_hdr_still_matrixzone_workflow.json` and `research/atlas_hdr_clip_workflow.json`
  (14,011 lines in all). They need ComfyUI V135+, are not shipping examples, and no test pins them.
  CHANGELOG, README, INSTALL, NODE_CATALOG and USER_GUIDE mention `research/`, so the code PR must
  forward-reference or drop those lines. Decision: CEO review D16.

- [ ] **Consolidate duplicated helpers (P3, human: M / CC: S).** The same code exists in several places:
  - `srgb_to_linear`/`linear_to_srgb` in 4 places: `core/generated_mesh.py`, `raw/pipeline.py`, `core/preview_codec.py`, `comfy/nodes_matrixzone.py`.
  - `matrixzone._box_mean` duplicates `plate/deband._box_blur_2d`.
  - `generated_mesh._erode` duplicates `core/adherence._erode`; it should move to `core/mask_ops`.
  - The GLB chunk writer and `_pad4` exist in both `exporters/scene_glb.py` and `relief_mesh_exporter.export_relief_mesh_glb`.
  - String constants: `"pixal3d"` in 3 places (move `GENERATED_SOURCE` into core), the SAM3 HF backend string in 2, and the Rec.709 luma tuple 5 times in `matrixzone.py`.
  Why: copies drift; a fix lands in one and not the others. Kept out of the feature merge to avoid touching older modules. Decision: CEO review D25.

- [ ] **Mirror the F-3 planner clamp in atlas_bridge (P2, human: S / CC: S).**
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

- [ ] **Reserve export names atomically across processes (P3, human: S / CC: S).**
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
