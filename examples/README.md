# Example workflows

Drag any `.json` here onto the ComfyUI canvas.

Filenames are a **public contract** — saved graphs and the docs reference them,
so nothing is renamed to fit this index. The grouping below is the hierarchy;
the directory stays flat on purpose.

---

## Hero — start here

The three workflows that explain what Atlas Camera is. Each completes one
meaningful artist task end to end rather than demonstrating every node.

| Workflow | The question it answers | Status |
|---|---|---|
| `atlas_hero_02_photo_to_editable_scene_workflow.json` | **One photograph becomes a camera-aware editable 3D scene, handed to Blender.** | Hero 02 — validated end to end 2026-08-17 |
| *Hero 01 — professional RAW → DCC/Nuke round trip* | Camera RAW through the colour-managed path, out to a DCC, and back into original lens space via a redistort STMap. | **not yet built** — needs a real RAW capture |
| *Hero 03 — advanced occlusion / scene recovery* | What becomes possible past the deterministic path, with measured / inferred / generated kept visibly distinct. | **not yet built** |

Hero 01 and Hero 03 are named here because the gap is deliberate, not an
oversight. Neither is blocked on missing Atlas capability — Hero 01 needs
capture material, Hero 03 needs a decision about how provenance is surfaced.

## Reference — the stages, opened up

Start from a hero, then reach for these when you need to take over a stage.

| Workflow | What it shows |
|---|---|
| `atlas_input_quickstart_workflow.json` | The front door. One node: plate in, camera and geometry out. |
| `atlas_quickstart_solve_project_export_workflow.json` | The same result with every solve stage exposed and overridable. |
| `atlas_export_fanout_workflow.json` | One solve into Nuke, Maya, Blender, USD, relief mesh and a review package, routed by `AtlasProject`. |
| `atlas_layered_projection_workflow.json` | The 2.5D layer stack a matte painter actually works in — depth bands, clean plates, sky dome. |

## Advanced — specialist paths

Real capability, narrower audience. Several need a capture set, a local model
or an external DCC.

| Workflow | What it shows | Needs |
|---|---|---|
| `atlas_multiview_raw_qwen_workflow.json` | Camera RAW ×3 → calibrated multi-view rig → generated patch. | `[raw]`, RAW files |
| `atlas_burst_multiview_solve_workflow.json` | A handheld burst registered into a metric rig. | a burst |
| `atlas_burst_photographed_hole_patch_workflow.json` | Holes repaired from **photographed** flanking frames, not invented ones. | a burst |
| `atlas_qwen_multiangle_hole_patch_workflow.json` | Occlusion repair from generated novel views. | Qwen models |
| `atlas_qwen_roi_registered_patch_workflow.json` | ROI-cropped generation at native resolution, registered back. | Qwen models |
| `atlas_cleanplate_depth_layer_workflow.json` | A same-camera clean plate contributing depth, without a second solve. | `[sam3]` |
| `atlas_blender_measured_primitives_workflow.json` | The Blender massing bridge and the in-graph agent handoff. | Blender ≥ 4.2, `ATLAS_EXPERIMENTAL=1` |

---

## What ships, and what does not

**No plates ship with most workflows.** A single RAW would be larger than the
whole tracked tree. Every graph starts on ComfyUI's bundled `example.png` or on
one of the three small plates in `examples/images/`, so a fresh clone runs
without downloading anything — but for a real result, use your own photograph.

**These files are generated, not hand-edited.** `tools/build_v1_shipping_workflows.py`
builds the hero and reference set from a live ComfyUI's `/object_info`, because
the UI format carries redundant link arrays and positional `widgets_values`,
and both fail silently on load when edited by hand. Change the builder, not the
JSON.

**Pinned by tests.** `tests/test_example_workflows.py` and
`tests/test_shipping_workflow_paths.py` hold the file list, forbid absolute
machine paths, and require workflow ids to be real UUIDs. Adding a workflow
means adding it to the pin.
