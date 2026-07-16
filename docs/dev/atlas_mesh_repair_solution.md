# Interior Hole-Fill on Exported Relief Meshes (Mesh-Topology, No Experimental Branch)

**Status:** Complete & verified (`tests/test_mesh_repair.py` — 12/12 green).
**Branch:** `claude/atlas-pytest-pil-error-f87721` (LOCAL experiment copy — not pushed).
**Date:** 2026-07-15.

## Motivation

Before this work, mesh hole-filling in AtlasCamera required the **experimental**
LaRI / World-Tracing hidden-geometry branch (`AtlasPredictHiddenGeometry`).
There was no way to cap the small interior tear holes in a *main* (non-experimental)
relief mesh — the kind of mesh every `AtlasExportReliefMesh` / `AtlasCleanPlateLayer` /
`AtlasDeriveReliefMesh` / `AtlasDeriveProjectionGeometry` already produces.

Atlas relief meshes are deliberately **torn at depth discontinuities** (silhouettes,
sky, band clips) so the live 📽 projection never rubber-sheets background onto
foreground. That doctrine is load-bearing and must not be weakened. But the
**exported** OBJ/GLB handed to a DCC benefits from being closer to watertight
*inside the subject*: small interior tear holes (depth-model noise, fine
structure, band-clip seams) become stray open boundaries that block retopo,
boolean ops, and 3D-print prep in Maya / ZBrush / Blender.

The goal: fill **only interior enclosed boundary loops** — never the outer
silhouette / frame boundary — and do it **export-only** so the live viewport
projection mesh and its deliberate tears are never touched.

## Scope decision (user-stated, governing)

> "we only want to fill holes within the inside bounds of the constructed mesh
> from the atlas solve"

The fill is **export-only**, applied to the resolved `ReliefMesh` *after* it is
built/recovered and *before* the OBJ/GLB writers run — it never mutates the live
projection mesh or the solve's own `proxy_geometry`. The outer frame/silhouette
boundary is left open by construction (see the two scoping mechanisms below).

## Band-box spatial scope (user proposal, implemented)

> "i could use the bound band box that constrains the projection mesh to define
> the areas to perform hole filling?"

Yes — `AtlasBoundedBand` measures the foreground's metric depth extent
`W = P(far_pct) − P(near_pct)` and emits a `cutoff_m = near + extrude_multiplier·W`
(default 2×). That cutoff is a **forward distance in metres along the recovered
camera's forward axis** (the camera faces −Z, so view-space z = −cutoff and
forward depth = −view-space z — the same axis the band box's cutoff plane lives
on). The hole-fill transcribes that window directly:

- `fill_depth_near_m` ← the band's near depth
- `fill_depth_far_m` ← `AtlasBoundedBand.cutoff_m`

A loop fills only if **all** its boundary vertices' forward depth lies within
`[near, far]`. The outer frame spans the full depth range (its far corners sit
at background depth, beyond the cutoff) so it is excluded **automatically** —
and background/sky holes outside the window stay open too, which is the
DMP-correct behavior. This is the cleaner realization of "fill holes within the
inside bounds": the band box **is** the inside bound.

## Algorithm (pure numpy, no external deps)

Implemented in `atlas_camera/core/mesh_repair.py`. Stays in `atlas_camera.core`
(no Three.js / numpy-only-at-boundary violations — numpy is the one dep `core`
already uses everywhere).

1. **`boundary_edges(faces)`** — edges appearing in exactly one triangle = the
   open boundary. Each face's three edge-pairs are sorted to canonical `(lo, hi)`,
   `np.unique(..., return_counts)` keeps the count==1 set. Vectorized.
2. **`walk_loops(bedges)`** — each boundary vertex has degree 2 on the boundary
   graph, so from any unvisited start we follow the non-previous neighbour until
   we return to the start (a closed loop). Vertices are marked done only after a
   loop closes, so the start-vertex cycle-re-entry check stays correct; a
   mid-path revisit before closing (non-simple cycle) bails rather than spins.
3. **`fill_interior_holes(...)`** — fan-triangulates each qualifying loop
   **from its existing vertices** (`for i in 1..n−2: [loop[0], loop[i], loop[i+1]]`).
   No new vertices → the 1:1 vertex-UV mapping the OBJ/GLB writers depend on
   stays exact, so projection-baked UVs remain valid on the filled faces.
   Two composable scope gates decide which loops fill (below).
4. **`apply_interior_hole_fill(mesh, ...)`** — node-facing in-place wrapper;
   returns `(n_loops_filled, filled_edge_counts)` for the node report. No-op
   `(0, [])` when `max_hole_edges <= 0` or the mesh has no faces / no boundary
   (already watertight).

### The two scoping gates

```python
depth_filter = (
    view_matrix is not None
    and vertices is not None
    and depth_near_m > 0.0      # BOTH bounds > 0 required to activate
    and depth_far_m > 0.0
)
outer = max(range(len(loops)), key=lambda i: len(loops[i]))  # largest = frame
for k, loop in enumerate(loops):
    n = len(loop)
    if depth_filter:
        d = depths[k]
        if not (np.all(d >= depth_near_m) and np.all(d <= depth_far_m)):
            continue            # window IS the scope — no largest-loop guard
    else:
        if k == outer:          # threshold-only mode: always leave frame open
            continue
    if n >= max_hole_edges:
        continue
    for i in range(1, n - 1):
        new_faces.append([loop[0], loop[i], loop[i + 1]])
    filled.append(n)
```

- **Edge-count threshold** (`max_hole_edges`, default 64): the outer frame
  perimeter is ~the grid perimeter (e.g. ~512 edges at grid 128) while interior
  tear loops are ~4–30, so a threshold around 64 separates them by construction.
  Belt-and-braces: in **threshold-only** mode (no depth window) the **single
  largest loop is always left open** even if the threshold is raised past it.
- **Band-box depth window** (`depth_near_m`/`depth_far_m` + `view_matrix`):
  when active, the window **is** the scope — the largest-loop guard is bypassed
  (the `else` branch containing the outer-skip only runs when `depth_filter` is
  False). The outer frame is then excluded *by depth*, not by being largest.
  Both bounds `0` → filter disabled → threshold-only mode (guard back on).

### Load-bearing subtlety (caught by tests)

`depth_near_m = 0.0` **disables** the depth filter (not "near = 0"). This is
deliberate: a zero bound means "window not set", so the node falls back to
threshold-only mode with the largest-loop guard. Tests that exercise window
mode must pass **both** bounds strictly positive.

## Node wiring (`AtlasExportReliefMesh`)

Four optional widgets appended to `atlas_camera/comfy/nodes.py`'s
`AtlasExportReliefMesh.INPUT_TYPES` (all default to off / disabled, so every
saved workflow keeps working unchanged):

| Widget | Type | Default | Meaning |
|---|---|---|---|
| `fill_interior_holes` | BOOLEAN | `False` | Master switch. Off → no change. |
| `max_hole_edges` | INT (3–4096) | `64` | Edge-count threshold; a loop fills only if `edges < this`. `0` = disabled. |
| `fill_depth_near_m` | FLOAT (step 0.1) | `0.0` | Band-box near bound (m). `0` = off. |
| `fill_depth_far_m` | FLOAT (step 0.1) | `0.0` | Band-box far bound = `AtlasBoundedBand.cutoff_m`. `0` = off. |

The `export()` signature gained the same four kwargs (appended last, positional
`widget_values` order preserved). The fill is applied between mesh resolution
and the texture/OBJ/GLB writers:

```python
if fill_interior_holes:
    from atlas_camera.core.mesh_repair import apply_interior_hole_fill
    apply_interior_hole_fill(
        mesh,
        max_hole_edges=int(max_hole_edges),
        view_matrix=extr.camera_view_matrix,     # 4x4 row-major view matrix
        depth_near_m=float(fill_depth_near_m),
        depth_far_m=float(fill_depth_far_m),
    )
texture = _image_tensor_to_pil(image)
```

`extr = solve.camera.extrinsics`; `extr.camera_view_matrix` is the 4×4 row-major
view matrix (`cam_to_world = inv(view_matrix)`). The fill runs **after** the
mesh is resolved — covering both the `_relief_mesh_from_solve(solve)` re-use
path and the re-derive path — so it always caps the final exported geometry.

## Why both OBJ and GLB writers are unchanged

The fan-fill appends faces to `mesh.faces` using **existing** vertex indices
and adds **no** vertices or UVs. `relief_mesh_exporter.py` was re-read and
confirmed unchanged:

- OBJ writer uses 1:1 vertex-UV (`f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}`) —
  the filled faces reference verts whose `vt` already exists. ✓
- GLB writer does `uvs[:, 1] = 1.0 - uvs[:, 1]` (OBJ bottom-left → glTF
  top-left) over the full uv array — new faces reuse existing uv entries. ✓

No writer edit was needed or made.

## Tests (`tests/test_mesh_repair.py`, 12 tests)

Fixture `_grid_mesh(z=-10.0, z_outer=None)`: a 4×4 flat grid, 2 tris/quad, with
the center quad `(1,1)` removed → a 4-edge **interior hole loop**; the grid
perimeter → a 12-edge **outer frame loop**. `z_outer` places perimeter verts at
a different depth (mirrors a real relief mesh whose frame spans near-to-far
while an interior tear sits at one depth). Vertices are 1:1 with uvs
(`uv == xy`). `_identity_view()` = `np.eye(4)`.

| Test | Pins |
|---|---|
| `test_boundary_loops_found` | 16 boundary edges → loops `[4, 12]` |
| `test_fill_interior_keeps_outer_frame` | threshold-only fills `[4]`, +2 faces, 12-edge frame stays open |
| `test_threshold_blocks_hole` | `max_hole_edges=3` → `[]` (4 ≥ 3) |
| `test_largest_loop_guard_even_above_threshold` | `max_hole_edges=4096` → still `[4]` (frame skipped as largest) |
| `test_depth_window_includes_hole_excludes_frame` | z=10 / z_outer=30, window `[5,15]` → `[4]`, frame stays open |
| `test_depth_window_can_admit_frame_when_set_to_it` | window `[25,35]` → `[12]` (no largest guard in window mode) |
| `test_depth_window_excludes_both_fills_nothing` | window `[50,60]` → `[]` |
| `test_depth_window_zero_falls_back_to_threshold_only` | `(0, 0)` → `[4]` (largest guard re-engages) |
| `test_apply_to_mesh_in_place` | fills 1 loop `[4]`, +2 faces, uvs identity-preserved |
| `test_apply_noop_when_disabled` | `max_hole_edges=0` → `(0, [])` + mesh unchanged; fill then 2nd pass = idempotent `(0, [])` |
| `test_watertight_mesh_is_noop` | tetrahedron → `[]` |
| `test_trimesh_validates_fill` | hole-only fill → χ=1 disk; `[5,15]` window fill of both loops → χ=2 sphere |

`trimesh` (4.12.2, MIT) is used **only** for the optional round-trip validation
test (`pytest.importorskip`), never in the runtime path.

## Test-logic bugs found & fixed during verification (NOT implementation bugs)

Both failures were in the *tests*, not `mesh_repair.py` — the implementation
behaved per its documented design throughout.

1. **`test_apply_noop_when_disabled`** asserted a second `max_hole_edges=64`
   call returned `(0, [])` — but the first call was the *disabled*
   (`max_hole_edges=0`) path, which correctly left the hole present. The second
   call then correctly *filled* it → `(1, [4])`. Fix: restructure so the
   disabled call asserts `(0, [])` + unchanged faces, then a real fill, then a
   *third* call asserts the idempotency noop.
2. **`test_trimesh_validates_fill`** passed `depth_near_m=0.0, depth_far_m=20.0`
   expecting a sphere (χ=2), but `depth_near_m > 0.0` is required to activate
   the depth filter — `0.0` falls back to threshold-only mode → largest-loop
   guard on → only the hole fills (χ=1 disk). Fix: use `[5.0, 15.0]` so the
   filter activates; the flat grid (all verts at depth 10) admits both loops →
   χ=2 sphere.

## Relationship to the experimental branch

This is the **main-branch** (non-experimental) answer to "fill mesh holes
without LaRI / World-Tracing". It is purely topological (no learned model, no
extra deps, no Docker) — it caps existing tear holes in the existing mesh.
It does **not** *predict* hidden geometry behind occluders the way LaRI/WT do;
for predicted-geometry reveals you still need the experimental
`AtlasPredictHiddenGeometry`. The two are complementary: this repairs what's
already there; that invents what isn't.

## Pre-existing unrelated suite failures

The full suite (`python -m pytest -q`) shows **5 failed, 572 passed, 26
skipped**. The 5 failures are all `ExecutionBlocker` object-comparison tests
(`test_assess_image`, `test_exact_patch_view`, `test_extract_angle`,
`test_solve_gate`) that expect a passthrough value *outside* ComfyUI's runtime
but get a live `ExecutionBlocker` object on this fresh portable install (where
`comfy_execution.graph_utils` happens to be importable, so the guard doesn't
trigger). None touch `mesh_repair.py` or `AtlasExportReliefMesh`; they pre-date
this feature and are environment artifacts, not regressions.

## Files changed / created

**New files:**
- `atlas_camera/core/mesh_repair.py` — pure-numpy core module (boundary-edge
  discovery, loop walking, selective fan-fill with the two scope gates,
  in-place node-facing wrapper).
- `tests/test_mesh_repair.py` — 12 tests pinning the selective-fill behavior.
- `docs/dev/atlas_mesh_repair_solution.md` — this document.

**Edited files:**
- `atlas_camera/comfy/nodes.py` — `AtlasExportReliefMesh`:
  - `INPUT_TYPES` optional block gained 4 widgets
    (`fill_interior_holes`, `max_hole_edges`, `fill_depth_near_m`,
    `fill_depth_far_m`).
  - `export()` signature gained the 4 matching kwargs (appended last).
  - Fill applied to the resolved `mesh` between mesh resolution and the
    texture/OBJ/GLB writers, gated behind `if fill_interior_holes`.

**Unchanged (verified by re-read, no edit needed):**
- `atlas_camera/exporters/relief_mesh_exporter.py` — both OBJ and GLB writers
  work unchanged because the fan-fill reuses existing vertex indices and adds
  no vertices/UVs.