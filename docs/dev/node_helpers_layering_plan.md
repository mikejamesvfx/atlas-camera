# Plan — phase 2 of the `nodes.py` split: fix the core/adapter layering

Status: PLAN. Nothing executed.
Goal chosen: **correct layering (core vs adapter)**, full three-bucket refactor,
folding moved math into **existing** core modules.

## Background

`stale_code_report_response.md` deferred the `nodes.py` domain split as "a
regression-risk refactor that deserves its own planned task". That split landed
2026-07-19: the 9,110-line monolith became six group modules plus a ~185-line
compatibility façade. **That part is done.**

What it did not do is decide where the *helpers* belong. `node_helpers.py` was
created as the shared leaf and has since accumulated **1,591 lines / 58
top-level items across ~10 unrelated concerns**. This plan is phase 2.

## The finding — a layering violation, not a size problem

`CLAUDE.md` states the architecture as:

    atlas_camera.core   <- DCC-agnostic schema, solver, math (no host deps)
    atlas_camera.comfy  <- ComfyUI node library

Measured against that, **43 of 58 items (853 lines) in `node_helpers.py` touch
no torch, no PIL, no ComfyUI and no base64.** They are host-agnostic math and
logic living in the adapter layer. Moving them is not tidying — it restores the
stated architecture and makes that math testable with no ComfyUI import at all.

### Inventory

| Bucket | Group | Lines | Items |
|---|---|---:|---:|
| **A** (move to `core/`) | depth/band math | 127 | 4 |
| | ground/scale math | 172 | 6 |
| | image array ops | 96 | 3 |
| | solve utils | 97 | 5 |
| **B** (stay in `comfy/`, own modules) | view-prompt parsing | 67 | 3 |
| | reports / manifest | 104 | 6 |
| | gate / fingerprint | 27 | 2 |
| **C** (stay — genuinely coupled) | **viewport payload** | **231** | **1** |
| | tensor / b64 | 68 | 7 |
| | guards / registry | 97 | 8 |
| — | needs triage during the work | 174 | 13 |

The standout is `_extract_blockout_camera` at **231 lines** — 57% of bucket C by
itself. It is the viewport wire protocol and has grown organically (the
occlusion depth-packing block was added to it on 2026-07-20). It is its own
module regardless of what else happens here.

### Why this is low risk

- **37 of 58 items have no internal dependencies** — freely movable.
- Deepest entanglement is 8 (`_metric_depth_and_validity`,
  `_extract_blockout_camera`). The call graph is shallow, not a web.
- The compatibility mechanism already exists and is proven: `nodes.py`
  re-exports **75 symbols** from `node_helpers`, and the 2026-07-19 split used
  exactly this pattern.
- Internal importers pull only 10–19 symbols each; no module is deeply bound.

## Target layout

Per the "fold into existing" decision:

| Moves | Destination | Rationale |
|---|---|---|
| `_resolve_depth_band`, `_metric_depth_and_validity`, `_band_resolution_validity`, `_depth_map_for_solve`, `_apply_band_split`, `_MetricDepthSetup` | `core/depth_geometry.py` | already the shared depth/geometry math module |
| `_ground_depth_compute`, `_analytic_ground_forward_depth`, `_solve_camera_params`, `_horizon_y_from_solve`, `_recompute_horizon_line` | `core/depth_geometry.py` | same family; all consume the view matrix + intrinsics |
| `_ground_scale_cached` | **stays in `comfy/`** | it is a memoisation of `relief_mesh.estimate_ground_scale`; per-execution caching is an ADAPTER concern, not math |
| `_clone_solve_with_metadata`, `_solve_with_relief_mesh`, `_relief_mesh_from_solve` | `core/schema.py` or `core/relief_mesh.py` | solve/mesh construction |
| `_resolve_raw_hints`, `_stamp_raw_provenance` | `atlas_camera/raw/metadata.py` | RAW domain already exists |
| `_resize_normal_field` | `core/normals.py` | see resolution below |
| `_extend_edge_colors`, `_flood_mask_to_frame_borders` | `atlas_camera/plate/ops.py` | see resolution below |

### Resolved (2026-07-20) — no new `core/` module needed

The first draft called these three "no natural home" and proposed a new
`core/image_ops.py`. Checking the actual call sites dissolved that:

- **`_resize_normal_field` was never ambiguous.** `core/normals.py` already
  exists and already does this work (`world_normals_from_depth`,
  `align_predicted_normals_to_world`, `encode_normal_map_b64`) — and already
  imports PIL locally at line 117, so the "PIL means it must stay in comfy"
  assumption behind the original classification was simply wrong. Its only
  caller is `nodes_depth.py`. → `core/normals.py`.

- **The other two have exactly ONE consumer module** (`nodes_inpaint.py`). They
  were never shared utilities; they sat in `node_helpers` because it was the
  default dumping ground. That tempts a move into `nodes_inpaint.py` itself,
  which would slim the leaf with no new module — but both are pure numpy,
  host-agnostic algorithms (quarter-res colour propagation; mask flood), so
  leaving them in `comfy/` preserves exactly the layering violation this
  refactor exists to fix. They belong in a library layer.

  Not `core/image_ops.py` though: that name says nothing, and it would be a
  module invented for two functions. They are *plate* operations, and
  `atlas_camera/plate/` now means precisely that — I/O, colour, and pixel ops
  on plates — sitting beside `plate/oiio_io.py`.

**Sequencing note:** `atlas_camera/plate/` arrives with PR #24 (the OIIO work).
Phase 2 is several phases out so #24 should land first; if it has not, the
fallback is the original `core/image_ops.py`. Phases 0 and 1 do not depend on
this at all.

New `comfy/` modules for bucket B and the payload:

    comfy/viewport_payload.py   <- _extract_blockout_camera + its 8 helpers
    comfy/view_prompts.py       <- named-view tables + _parse_view_prompt/_parse_exact_view
    comfy/node_reports.py       <- manifest writing + report/summary formatting
    comfy/node_helpers.py       <- SHRINKS to the true leaf: guards, tensor/b64,
                                   registry probes, graph builder, caches

## Sequencing

Each phase is one commit, suite green before the next. Ordered
lowest-risk-first so an early stop still leaves the tree better than it started.

**Phase 0 — safety net.** Record the baseline: full suite count, registry
surface, and the exact 75 symbols `nodes.py` re-exports. Add a test asserting
that re-export list is unchanged, so any accidental drop fails loudly. *This
test is the whole safety net; write it first.*

**Phase 1 — extract the viewport payload.** `_extract_blockout_camera` and its
helpers into `comfy/viewport_payload.py`. Pure code motion, one consumer
(`nodes_viewport.py`), biggest single readability win. No layer change.

**Phase 2 — the core math moves (THE GOAL).** Bucket A into the existing core
modules above. Do it in the dependency order already computed (leaves first).
After this phase the layering violation is gone.

**Phase 3 — bucket B into its own comfy modules.** View prompts, reports,
manifests.

**Phase 4 — `node_helpers.py` becomes a true leaf.** Whatever remains should be
only: guarded imports, tensor/b64 conversion, registry probes, graph builder,
caches. Re-audit `__all__` against reality.

**Phase 5 — docs.** Update the "Module layout" section of `CLAUDE.md` to
describe the new map. It currently documents the 2026-07-19 layout only.

## Risk controls

- **Keep every façade re-export working.** `from atlas_camera.comfy.nodes import X`
  is a saved-workflow-adjacent contract. Moves re-export from the new home.
- **The monkeypatch trap.** Three tests patch helpers on *the class's own
  module* (`_comfy_registry` for `AtlasInput`/`AtlasSDXLInpaint`,
  `_save_image_tensor_to_tmp` for `AtlasMogeNormals`). Patch targets must follow
  moved code — this bit the 2026-07-19 split and is called out in `CLAUDE.md`.
- **Pinned surfaces already exist:** `tests/test_comfy_node_registry.py` pins
  every node key and display name; `tests/test_node_usage_audit.py` pins counts.
- **Verify per phase:** full suite green, registry counts unchanged (69 standard
  / 5 experimental / 74 total), and the node pack still imports the way ComfyUI
  loads it (`tests/test_node_pack_entrypoint.py`).
- **No behaviour changes.** This is code motion only. Any bug found en route
  gets its own commit, before or after — never folded into a move.

## Explicitly NOT in scope

- Splitting the *node group* modules (`nodes_geometry.py` 2,228 lines,
  `nodes_solve.py` 1,668). They are large but each is cohesive by domain. Judge
  them after this pass; the helper layer is the tangled part.
- Any change to node classes, registry keys or display names — those are a
  saved-workflow contract.
- Runtime performance. Nothing measured suggests module layout costs anything at
  runtime; if that becomes the goal, profile first.
- Import/startup time. Plausible small win, but unmeasured — do not claim it.

## Status

- **Phase 0 — DONE** (`refactor/phase0-facade-pin`). `tests/test_facade_surface.py`
  pins all 155 façade names (79 public + 76 underscore helpers) and was verified
  to FAIL when a symbol is dropped, not merely to pass today.
- Destination question resolved (see above) — no new `core/` module is needed.
- **Phase 1 — next.** Depends on nothing outstanding.
