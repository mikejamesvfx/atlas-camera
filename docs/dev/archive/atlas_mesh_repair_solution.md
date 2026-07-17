> ARCHIVED 2026-07-18 — shipped as core/mesh_repair.py + AtlasExportReliefMesh fill_interior_holes

# Interior Hole-Fill on Exported Relief Meshes (Mesh-Topology, No Experimental Branch)

**Status:** Complete & verified (`tests/test_mesh_repair.py` — 19/19 green; full
suite 609 passed / 3 skipped / 0 failed).
**Branch:** `claude/atlas-mesh-repair-review-580254` (off `main`, for beta 0.5).
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

**The governing contract: a fill may leave a hole open, but must never make the
exported mesh worse than not filling at all.** Every design choice below follows
from it — the whole point is clean DCC geometry, and a back-facing face, a
zero-area sliver, or a non-manifold edge breaks a retopo/boolean just as surely
as the hole did.

1. **`boundary_edges(faces)`** — edges appearing in exactly one triangle = the
   open boundary. Each face's three edge-pairs are sorted to canonical `(lo, hi)`,
   `np.unique(..., return_counts)` keeps the count==1 set. Vectorized.
2. **`walk_loops(bedges, faces=None)`** — with `faces`, a **pivot walk** over the
   face fan: a directed edge `(a,b)` is a boundary half-edge when `(b,a)` doesn't
   exist, and its successor is found by rotating around `b` across *interior*
   edges only. Successor is a bijection over boundary half-edges, so the walk
   decomposes them into disjoint cycles and drops nothing. Without `faces` it
   falls back to the old undirected degree-2 walk (kept for API compatibility;
   it silently drops pinched loops — see below).
3. **`_split_at_repeats(loop)`** — a pinch (two tears meeting at one grid vertex,
   boundary degree 4) is **not** two holes sharing a vertex: the surface's
   boundary is ONE curve that passes through that vertex twice. The pivot walk
   recovers that figure-eight intact; splitting at the repeat recovers the two
   simple loops an artist would call holes.
4. **`_triangulate_loop(...)`** — **ear clipping** in the loop's best-fit plane,
   using only the loop's existing vertices (no new verts → 1:1 vertex-UV stays
   exact, so projection-baked UVs remain valid on the filled faces), then wound
   to match the mesh. Three subtleties, each found by measurement (below).
5. **`fill_interior_holes(...)`** / **`apply_interior_hole_fill(...)`** — the two
   scope gates + the node-facing in-place wrapper, returning
   `(n_loops_filled, filled_edge_counts)`.

### The two scoping gates

Which loops are eligible at all (orthogonal to *how* an eligible loop is
triangulated):

```python
depth_filter = (
    view_matrix is not None
    and depth_near_m > 0.0      # BOTH bounds > 0 required to activate
    and depth_far_m > 0.0
)
outer = max(range(len(loops)), key=lambda i: len(loops[i]))  # largest = frame
for k, loop in enumerate(loops):
    if depth_filter:
        d = depths[k]
        if not (np.all(d >= depth_near_m) and np.all(d <= depth_far_m)):
            continue            # window IS the scope — no largest-loop guard
    else:
        if k == outer:          # threshold-only mode: always leave frame open
            continue
    if len(loop) >= max_hole_edges:
        continue
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

**Load-bearing subtlety (caught by tests):** `depth_near_m = 0.0` **disables**
the depth filter (not "near = 0"). A zero bound means "window not set", so the
node falls back to threshold-only mode with the largest-loop guard. Tests that
exercise window mode must pass **both** bounds strictly positive.

### Why ear clipping, not a fan (the original bug)

A fan from `loop[0]` is valid only when the loop is **star-shaped from that
vertex** — i.e. essentially only for convex loops. Tear footprints on a decimated
grid are routinely non-convex (L / staircase), and both the walk's start vertex
and its direction are arbitrary, so the anchor lands on a reflex vertex often.
The fan then emits triangles that spill **outside the hole**, over real geometry,
with inverted winding. Measured on a plain L-shaped tear: 4 of the 6 possible
anchors produce `sum|area| = 4.0` against a true hole area of `3.0` — a third of
the fill lying outside. Across 12 synthetic torn relief meshes, **9 of 30 fills
(30%)** were affected. Ear clipping handles any simple polygon.

### The three load-bearing subtleties

*   **Winding is a per-edge rule, not a geometric one.** A boundary edge belongs
    to exactly one face, so the fill triangle sharing it must traverse it the
    *opposite* way; ear clipping preserves the polygon's own direction along its
    boundary edges, so one lookup settles every triangle. Comparing the loop's
    best-fit-plane normal against an adjacent face normal *looks* equivalent and
    passes on a flat fixture — but real tear holes are not planar, and the fitted
    normal can sit at any angle to the face. That version shipped green against
    the flat unit tests and was caught only by the end-to-end control
    (`is_winding_consistent: True → False` on a real export).
    `test_fill_winding_consistent_on_a_real_relief_mesh` pins it.
*   **Collinear vertices are skipped, never dropped.** Grid-aligned tears have
    collinear runs, and triangulating across one yields a zero-area sliver. But
    *deleting* the vertex is worse: the cap would then span boundary edges that
    still exist on the mesh — a T-junction, no more watertight than the hole was.
    So a collinear vertex is skipped as an ear candidate and picked up once its
    neighbours change; the two-ears theorem guarantees a strictly-convex ear
    exists meanwhile.
*   **Never reuse an existing edge.** If a loop's diagonal already exists in the
    mesh, using it puts that edge in three faces — non-manifold. Ear selection
    refuses such diagonals. Greedy clipping is scan-order dependent and can corner
    itself, so **every rotation of the polygon is tried** before giving up; if
    none succeeds the hole is simply left open. (Measured: refusing outright cost
    4 of 42 fills and greedy-with-relaxation left 8 non-manifold edges; the
    rotation retry gets 38 fills with **zero** non-manifold edges.)

### `vertices` is now required

`fill_interior_holes` fills nothing without `vertices`. A correct triangulation
— non-convex, correctly wound, sliver-free — is simply not decidable from
connectivity alone. `apply_interior_hole_fill` always has them (off the
`ReliefMesh`), so the node path is unaffected.

### Measured, fan vs. ear clipping

Across 12 synthetic torn relief meshes (`build_relief_mesh`, grid 128, noise +
fine structure), and end-to-end through `AtlasExportReliefMesh` on a real export:

| | fan (original) | ear clip (now) |
|---|---|---|
| holes filled | 30 | **38** |
| fills with triangles outside the hole | 9 (30%) | **0** |
| coverage vs. hole area (in-plane) | — | **exact** |
| degenerate faces added | 2 | **0** |
| non-manifold edges added | — | **0** |
| boundary loops silently dropped | 4% of boundary verts | **0** |
| `is_winding_consistent` OFF → ON | True → **False** | True → **True** |

Fill cost is ~0.15 s on a 31k-face mesh.

## Node wiring (`AtlasExportReliefMesh`)

Four optional widgets appended to `atlas_camera/comfy/nodes.py`'s
`AtlasExportReliefMesh.INPUT_TYPES` (all default to off / disabled, so every
saved workflow keeps working unchanged):

| Widget | Type | Default | Meaning |
|---|---|---|---|
| `fill_interior_holes` | BOOLEAN | `False` | Master switch. Off → no change. |
| `max_hole_edges` | INT (3–4096) | `64` | Edge-count threshold; a loop fills only if `edges < this`. (The core function also treats `<= 0` as disabled, but the widget's `min` is 3 so that path is unreachable from the node — `fill_interior_holes` is the off switch.) |
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

## Tests (`tests/test_mesh_repair.py`, 19 tests)

Fixture `_grid_mesh(z=-10.0, z_outer=None)`: a 4×4 flat grid, 2 tris/quad, with
the center quad `(1,1)` removed → a 4-edge **interior hole loop**; the grid
perimeter → a 12-edge **outer frame loop**. `z_outer` places perimeter verts at
a different depth (mirrors a real relief mesh whose frame spans near-to-far
while an interior tear sits at one depth). Vertices are 1:1 with uvs
(`uv == xy`). `_identity_view()` = `np.eye(4)`.

`_grid(n, drop)` carves an arbitrary tear footprint (L-shaped, pinched) rather
than only one convex quad. `_noisy_relief_mesh()` builds a **real**
`build_relief_mesh` output — genuinely non-planar, non-convex tears. That level
matters: the flat fixtures pass every orientation heuristic by luck, and the
back-facing-fill bug was invisible until a non-planar mesh was tested.

| Test | Pins |
|---|---|
| `test_boundary_loops_found` | 16 boundary edges → loops `[4, 12]` |
| `test_fill_interior_keeps_outer_frame` | threshold-only fills `[4]`, +2 faces, 12-edge frame stays open |
| `test_threshold_blocks_hole` | `max_hole_edges=3` → `[]` (4 ≥ 3) |
| `test_largest_loop_guard_even_above_threshold` | `max_hole_edges=4096` → still `[4]` (frame skipped as largest) |
| `test_no_vertices_fills_nothing` | no `vertices` → `[]` (triangulation undecidable from connectivity) |
| `test_depth_window_includes_hole_excludes_frame` | z=10 / z_outer=30, window `[5,15]` → `[4]`, frame stays open |
| `test_depth_window_can_admit_frame_when_set_to_it` | window `[25,35]` → `[12]` (no largest guard in window mode) |
| `test_depth_window_excludes_both_fills_nothing` | window `[50,60]` → `[]` |
| `test_depth_window_zero_falls_back_to_threshold_only` | `(0, 0)` → `[4]` (largest guard re-engages) |
| `test_apply_to_mesh_in_place` | fills 1 loop `[4]`, +2 faces, uvs identity-preserved |
| `test_apply_noop_when_disabled` | `max_hole_edges=0` → `(0, [])` + mesh unchanged; fill then 2nd pass = idempotent `(0, [])` |
| `test_watertight_mesh_is_noop` | tetrahedron → `[]` |
| `test_trimesh_validates_fill` | hole-only fill → χ=1 disk; `[5,15]` window fill of both loops → χ=2 sphere |
| `test_nonconvex_hole_fills_inside_the_hole_only` | L-tear: no mixed winding; fill area == hole area exactly |
| `test_fill_adds_no_degenerate_faces` | L-tear: no zero-area fill faces |
| `test_fill_winding_matches_surrounding_mesh` | flat L-tear: fill normals == mesh normal |
| `test_pinch_vertex_does_not_drop_a_hole` | two tears sharing a vertex → `[4, 4, 16]`, both fill |
| `test_fill_winding_consistent_on_a_real_relief_mesh` | **real non-planar mesh**: no directed edge occurs twice |
| `test_fill_never_degrades_a_real_relief_mesh` | **real mesh**: no non-manifold edge, no degenerate face, no invented vertex |

`trimesh` (4.12.2, MIT) is used **only** for the optional round-trip validation
test (`pytest.importorskip`), never in the runtime path.

## Bugs found during review, and how

The original implementation was verified only against one convex, planar, 4-edge
hole, and was green on it. Three real defects hid behind that fixture; each was
found by measurement rather than by reading:

1. **Fan triangulation spilled outside non-convex holes** (30% of real fills).
   Found by a controlled A/B through the real node —
   `is_winding_consistent: True → False`, `degenerate: 0 → 3` — proving the base
   mesh was clean and the fill dirtied it. Fixed by ear clipping.
2. **Pinch vertices silently dropped loops** (4% of boundary verts; 15% on one
   mesh). The undirected walk wanders between the two tears and bails. Fixed by
   the pivot walk + `_split_at_repeats`.
3. **Back-facing fills on non-planar holes.** The first fix aligned winding by
   comparing the loop's best-fit-plane normal to an adjacent face normal — green
   on every flat unit test, still `winding_consistent: False` end-to-end. Fixed
   by the exact per-edge rule.

**The transferable lesson:** flat, convex, planar fixtures are the easy case, and
mesh code passes them by luck. Two of the three bugs were invisible to unit tests
and only showed up in an end-to-end A/B against a real `build_relief_mesh`
output — so the suite now carries `_noisy_relief_mesh` regression tests at that
level.

## Relationship to the experimental branch

This is the **main-branch** (non-experimental) answer to "fill mesh holes
without LaRI / World-Tracing". It is purely topological (no learned model, no
extra deps, no Docker) — it caps existing tear holes in the existing mesh.
It does **not** *predict* hidden geometry behind occluders the way LaRI/WT do;
for predicted-geometry reveals you still need the experimental
`AtlasPredictHiddenGeometry`. The two are complementary: this repairs what's
already there; that invents what isn't.

## Suite status

The full suite (`python -m pytest -q`) is **609 passed, 3 skipped, 0 failed** in
the repo's dev environment (604 pre-existing + this feature's tests).

An earlier revision of this document reported "5 failed, 572 passed, 26 skipped"
and attributed the failures to `ExecutionBlocker` object-comparison tests. That
does **not** reproduce here and was an artifact of running under a different
interpreter (a portable ComfyUI install where `comfy_execution.graph_utils`
happens to be importable, so the guard those tests rely on doesn't trigger). It
was never a property of this branch — don't carry it forward as a standing
caveat.

## Files changed / created

**New files:**
- `atlas_camera/core/mesh_repair.py` — pure-numpy core module (boundary-edge
  discovery, pivot loop walk + pinch splitting, mesh-aligned ear-clip fill with
  the two scope gates, in-place node-facing wrapper).
- `tests/test_mesh_repair.py` — 19 tests pinning the selective-fill behavior and
  the "never degrade the mesh" contract (incl. real-relief-mesh regressions).
- `docs/dev/archive/atlas_mesh_repair_solution.md` — this document.

**Edited files:**
- `atlas_camera/comfy/nodes.py` — `AtlasExportReliefMesh`:
  - `INPUT_TYPES` optional block gained 4 widgets
    (`fill_interior_holes`, `max_hole_edges`, `fill_depth_near_m`,
    `fill_depth_far_m`).
  - `export()` signature gained the 4 matching kwargs (appended last).
  - Fill applied to the resolved `mesh` between mesh resolution and the
    texture/OBJ/GLB writers, gated behind `if fill_interior_holes`.

**Unchanged (verified by re-read AND by an end-to-end export, no edit needed):**
- `atlas_camera/exporters/relief_mesh_exporter.py` — both OBJ and GLB writers
  work unchanged because the fill reuses existing vertex indices and adds no
  vertices/UVs. Verified on a real export: every OBJ face carries `v idx == vt
  idx`, all indices in range, and both OBJ and GLB load in trimesh.

**Verified, not just asserted:**
- **Export-only.** A real `AtlasExportReliefMesh.export()` run with the fill on
  leaves the solve's `proxy_geometry` byte-identical (`mesh_from_primitive`
  rebuilds fresh arrays via `np.asarray(list)`, and the fill rebinds
  `mesh.faces`), so the live viewport projection mesh keeps its deliberate tears.
- **Never degrades the mesh.** Controlled A/B (fill OFF vs ON) across 5 real
  meshes: `is_winding_consistent` stays `True`, zero degenerate faces added,
  zero non-manifold edges added.