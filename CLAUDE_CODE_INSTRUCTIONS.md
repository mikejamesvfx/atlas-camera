# Claude Code handoff: portal repair and node-surface cleanup

This repository contains the Atlas Camera ComfyUI node pack. The current
branch (`agent/portal-hole-repair`) is the result of a cleanup and portal-hole
repair pass. Preserve the decisions below when making follow-up changes.

## Why this work was done

The interior portal workflow exposed three separate problems that had been
mixed together:

1. Several old controls were redundant. The live-fill controls duplicated the
   newer export/repair contracts and made saved workflows harder to understand.
2. The Planar Unwrap/Rewrap nodes were not part of the intended Atlas surface
   and duplicated functionality that belongs in the geometry/repair chain.
3. The experimental node menu contained research/experimental entries that
   were not part of the supported Atlas workflow. They should not appear in a
   normal node search.

The recommended repair strategy is now explicit: derive the portal geometry,
patch ordinary planar holes, use `AtlasMaskedSurfaceReconstruct` when a mask
identifies a missing region without a usable mesh boundary, then use
`AtlasRefineOcclusionSeams` for narrow staircase/occlusion underlaps.

## Nodes removed from registration

These nodes no longer appear in the ComfyUI registry or experimental menu:

- `AtlasPlanarUnwarp` (displayed as **Atlas Planar Unwrap**)
- `AtlasPlanarRewarp` (displayed as **Atlas Planar Rewarp**)
- `AtlasPredictHiddenGeometry`
- `AtlasBlenderOrganicFill`
- `AtlasRenderFix`

The Planar Unwrap/Rewrap implementation module was deleted. The three
experimental implementations above remain in source history where needed by
legacy tests/documentation, but they are intentionally not registered. Do not
re-register them unless the user explicitly asks to restore those workflows.

`AtlasBlenderBoundaryFill` was **deleted outright on 2026-08-03**, not merely
de-registered — it was never in any registry mapping in any commit, so no saved
workflow can reference it. Removed with it: `atlas_camera/blender/boundary_fill.py`,
its `recipes/boundary_fill.py` driver, `tests/test_blender_boundary_fill.py`, and
`tools/build_holefill_boundary_fill_comparison_workflow.py` (the native-vs-Fill-Mesh
lab builder). Its design rule is retained in `docs/DESIGN_RULES.md` and marked
REMOVED, because the doctrine it records now governs the NumPy pair. Blender
remains an optional path only for the pre-existing organic-fill/shrinkwrap
recipes; do not add new Blender-dependent nodes before beta 0.8.

The old live-fill widgets were also removed from the current derive/repair
node contracts, including `live_fill_holes`, `live_fill_distance_m`,
`live_fill_max_hole_edges`, and `live_fill_edge_sawteeth`. Do not insert new
widgets into existing saved workflow positions; append widgets only.

## Nodes added or retained for the new workflow

- `AtlasMaskedSurfaceReconstruct` (experimental, NumPy): mask-authoritative
  local reconstruction for intact meshes with no usable topology boundary. It
  manufactures a local rim, solves forward depth harmonically, creates exact
  camera-ray vertices/projective UVs, and rejects frame-touching or invalid
  components.
- `AtlasRefineOcclusionSeams` (experimental, NumPy): narrow zipper-style
  underlap for staircase/sawtooth occlusion seams. It keeps near/far sheets
  separate and rejects cross-depth and frame edges.
- `AtlasGravityCompass` (standard): direct-manipulation orientation control
  backed by `AtlasGravityOverride`; `USE SOLVE` is a true pass-through and
  heading override is appended for positional-workflow compatibility.
The Blender boundary-fill comparison path is gone entirely (see above); the two
NumPy nodes are the whole repair surface.

All experimental nodes remain behind `ATLAS_EXPERIMENTAL=1`.

## Portal workflow recommendation

The checked-in best workflows are:

- `examples/local/2026-07-16_atlas_retopo_portal_best.json`
- `examples/local/2026-07-16_atlas_retopo_portal_best_api.json`

The tuned chain increases the interior-hole budgets, disables the overly
restrictive normal-edge gate, and uses a wider seam pass. The API workflow
adds `AtlasMaskedSurfaceReconstruct` between planar repair and seam refinement:

```text
derive projection geometry
  -> planar hole patch
  -> masked surface reconstruct
  -> refine occlusion seams
```

Use `created_region`/`remaining_holes` for QA and clean-plate decisions. The
masked node creates smooth, evidence-bounded geometry; it does not recover
hidden semantic structure or texture.

## Platform expectations

The core and the two new reconstruction nodes are CPU/NumPy friendly. Device
selection supports CUDA, MPS, and CPU, with an MPS fallback in the SAM3 path.
Blender discovery supports Windows, `/Applications/Blender*.app`, `PATH`, and
`ATLAS_BLENDER_PATH`. Apple Silicon should be smoke-tested with a native
arm64 Python/PyTorch environment; neural backends may need CPU fallback for
unsupported MPS operators. Launch ComfyUI with `ATLAS_EXPERIMENTAL=1` when
loading the portal workflow.

## Verification and maintenance

The focused regression suite passed with 159 tests and one expected numerical
warning. Before changing node contracts, run the relevant registry, facade,
geometry, experimental-gate, masked-reconstruction, seam, planar, and repair
tests. After code changes, run `graphify update .` and the project test suite.

Keep registry, display-name mappings, frontend mirrors, MCP snippets, node
catalog/audit reports, and tests synchronized. Do not use `git add -A` in a
dirty worktree; stage only the files belonging to the requested change.

There are still historical references to removed research nodes in some long
form design/user documents. Treat the registry and current node catalog as the
source of truth; update those historical references only when specifically
doing documentation cleanup.
