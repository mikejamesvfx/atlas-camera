# Hand-authored polygon planes (`AtlasAddPlanePolygon`)

Date: 2026-08-04
Status: approved design, not yet implemented

## Problem

`examples/atlas_derive_planar_geometry_workflow.json` derives planes from depth
via `AtlasDeriveProjectionGeometry` (walls) and the roofs/facades branch. In
practice the projected result smears and the artist has no way to correct it.

Three distinct causes, established by reading the code against live screenshots
of the ghosttown plate:

1. **The projection is not UV-driven.** `makeProjectionMaterial`
   (`atlas_camera/comfy/web/atlas_blockout.js:962`) projects by world position
   through `uAtlasViewMatrix` plus `fx/fy/cx/cy`. Geometry UVs are ignored for
   proxy planes. So a plane that looks wrong is *placed* wrong; its UVs are
   irrelevant.
2. **Derived walls are axis-extent rectangles, not silhouette-clipped.**
   `_cluster_walls_by_azimuth` (`atlas_camera/core/proxy_geometry.py:612`)
   turns each azimuth cluster into a rectangle spanning the cluster's extents.
   The rectangle covers image area the real wall never occupied, so it receives
   paint from unrelated parts of the plate — the visible smear.
3. **`max_objects > 0` emits one scene-swallowing box.** `_derive_objects`
   (`proxy_geometry.py:648`) clusters every point that is neither ground nor
   wall. On a plate like ghosttown that is effectively one cluster, whose
   oriented bounding box encloses the scene.

Causes 2 and 3 are filed separately. This spec addresses the root need: let the
artist define the surface directly.

## Solution

A new ComfyUI node, `AtlasAddPlanePolygon`, with an image-click widget. The
artist clicks a polygon on the plate; the node fits a 3D plane to that region,
emits an exactly-clipped polygon mesh into the solve's proxy geometry, and the
existing viewport projection paints it. Because the mesh footprint is exactly
what was clicked, there is no rectangle overshoot and no smear.

Polygons are N-gons, not restricted to quads: four points for a wall, six for
an L-shaped facade, more to trace a roofline.

## Non-goals (v1)

- Fixing `max_objects` (cause 3) — separate change.
- Silhouette-clipping the derived walls (cause 2) — separate change, and it
  alters the output of every shipped planar workflow.
- Vertex snapping to vanishing-point directions.
- Magnetic edge snapping to image gradients.
- Re-projecting existing polygons after a solve's scale changes.
- A shipping example workflow. The repo was trimmed to three quickstarts in
  0.8.1; a new example drags in the path/UUID pin tests for little gain.

## Architecture

Dependency direction is preserved: the core module knows nothing about
ComfyUI, and the node is the only thing that touches both sides.

### `atlas_camera/core/polygon_planes.py` (new, host-agnostic)

```python
fit_polygon_plane(
    points_px, *, depth, view_matrix, fx, fy, cx, cy, mode
) -> PolygonPlaneFit
```

`PolygonPlaneFit` carries `vertices` (world-space, flat), `faces` (triangle
indices), `uvs` (planar 0..1), `normal`, `distance_m`, `method`, `confidence`,
`stats`.

Two private fitters:

- **`_fit_from_depth_ransac`** — rasterizes the polygon to a boolean mask, back-
  projects only the pixels inside it, and runs
  `plane_extraction.extract_planes_ransac` on that scoped point set. Confidence
  derives from inlier fraction and residual.
- **`_fit_from_rectangle_homography`** — depth-free. Treats the quad as a real-
  world rectangle and recovers orientation from the vanishing points of the two
  edge pairs plus the known intrinsics. Distance comes from the median depth
  sample, or from ground contact when the bottom edge sits on the ground plane.
  Quads only; N-gons refuse this path.

Tiering lives in `fit_polygon_plane`: RANSAC first, falling back when the
inlier fraction is below `min_inlier_fraction` or region depth is invalid.
`method` is always reported — the fallback is never silent.

Triangulation is **ear clipping**, not a naive fan. A fan silently produces
inverted triangles on a concave outline, and concave outlines (rooflines,
L-shaped facades) are a primary use case. Self-intersecting outlines are
rejected rather than triangulated into garbage.

### `AtlasAddPlanePolygon` (new node, `atlas_camera/comfy/nodes_geometry.py`)

Appends its polygons to `solve.projection_scene.proxy_geometry` and passes the
solve through. It does **not** clobber. The existing clobber doctrine governs
*derive* nodes, which regenerate a whole geometry set from depth; a hand-
authored addition is semantically an add, as `AtlasAddPatchView` is. This lets
several of these chain for several surfaces with no branching and no
`AtlasMergeGeometry` per plane.

Each polygon emits one primitive in the shape `relief_mesh_primitive` already
uses (`proxy_geometry.py:1289`): `primitive_type="mesh"`, identity transform,
world-space `vertices` / `faces` / `uvs` in `metadata`, plus
`role=PROXY_ROLE`, `source="hand_polygon"`, `method`, `confidence`,
`point_count`. The frontend renders this today via the `e.type === "mesh"`
branch in `buildDerivedProxies` (`atlas_blockout.js:1180`), so **no viewport
code changes are required** to display or project the result.

#### Inputs

| Name | Type | Notes |
|---|---|---|
| `solve` | `ATLAS_SOLVE` | required; polygons append to its projection scene |
| `image` | `IMAGE` | the plate to click on; supplies width/height |
| `depth` | `ATLAS_DEPTH_MAP` | optional. Absent means `depth_ransac` is unavailable; the node runs rectangle-only and says so in the report |

#### Widgets

Order is fixed at ship and append-only afterwards.

- **`polygons`** — STRING, multiline. Written by the canvas widget, also hand-
  editable and diffable:
  ```json
  {"version": 1, "polygons": [
    {"id": "p1", "label": "saloon facade",
     "points": [[0.12,0.34],[0.31,0.30],[0.31,0.78],[0.12,0.81]],
     "fit_mode": "inherit", "enabled": true}]}
  ```
  Points are normalized 0..1 against the plate so a resolution change does not
  invalidate clicks.
- **`fit_mode`** — combo `auto | depth_ransac | rectangle`. `auto` is the
  tiering above. Values are append-only forever.
- **`name_prefix`** — STRING, default `hand_plane`. Primitives become
  `hand_plane_01`, `hand_plane_02`, …
- **`min_inlier_fraction`** — FLOAT, default 0.35. RANSAC below this triggers
  the fallback.

#### Outputs

`solve` (`ATLAS_SOLVE`), `report` (STRING).

#### Gate doctrine

`polygons` is a persisted widget that gates which geometry exists, so the node
implements `fingerprint_inputs` over the polygons string and `fit_mode`.
Without it ComfyUI serves a cached solve after a quad is edited.

Nothing is skipped silently. Every polygon appears in the report as one of:

```
ok(depth_ransac, inliers 0.71)
ok(rectangle_homography, fallback: inliers 0.12 < 0.35)
skipped(self_intersecting)
```

### `atlas_camera/comfy/web/atlas_add_plane.js` (new frontend)

A DOM canvas widget showing the plate with a polygon overlay. Built with
`addDOMWidget` and **chained** lifecycle callbacks (`onResize` / `onRemoved` /
`onConfigure`) — assignment orphans DOM on workflow switch. Sizing is CSS-only
(`height:100%` chain, `min-width:0`, `object-fit:contain`); no JS resize hooks.
The overlay redraws on pointer events, not on a rAF loop.

Interactions (v1):

- Click — add a point to the active polygon.
- Double-click / Enter — close the polygon, begin a new one.
- Esc — discard the in-progress polygon.
- Drag a vertex — move it. Dragging is the entire edit story; no midpoint
  insertion, no curves.
- Click a polygon body — make it active. Delete — remove the active polygon.
- A compact list beside the canvas: label, point count, enabled checkbox,
  last-run status badge.

The node returns a `ui` dict keyed by polygon `id` with
`{method, confidence, note}`. The widget colours each polygon's outline by
outcome — RANSAC fit, rectangle fallback, or failed — so a fallback is visible
on the canvas rather than buried in the report string. Fills stay neutral so
the plate remains readable.

## Failure handling

Nothing here may kill the graph.

- numpy missing → guarded `ImportError` carrying the install hint (core's
  zero-required-dependency rule).
- `depth` resolution differs from the plate → `ValueError` naming both shapes.
  This is a wiring mistake and should fail loudly.
- Polygon with fewer than 3 points, zero area, or self-intersecting →
  `skipped(reason)`, no primitive, solve still passes through.
- Both fitters fail → `skipped(no_fit)`. The node never emits a guessed plane.
- Report assembly is wrapped so a formatting bug cannot take out an otherwise
  good solve.

## Testing

`tests/test_polygon_planes.py` — core math, no ComfyUI import:

- Synthetic depth of a known tilted plane → recovered normal within ~1°,
  distance within ~1%.
- Zeroed depth plus a synthetic rectangle → the rectangle path recovers the
  same orientation.
- Concave L-shaped outline → ear clipping yields consistently wound triangles
  with no inversions.
- Self-intersecting bowtie → rejected.
- Points normalized 0..1 scale correctly across two plate resolutions.

`tests/test_add_plane_node.py`:

- Appends without clobbering pre-existing `PROXY_ROLE` geometry. This is the
  compositional promise of the node and gets an explicit test.
- Fingerprint changes when the polygons blob changes, and does not change when
  unrelated inputs move.
- Missing `depth` routes to the rectangle path and the report says so.
- A failed polygon leaves the solve intact.

## Contracts touched

- `tests/test_comfy_node_registry.py` — pinned surface goes 92 → 93. The
  registered key and display name freeze the moment this ships.
- `tests/test_facade_surface.py` — add the name if it is re-exported through
  the `nodes.py` façade.
- `docs/NODE_CATALOG.md` — new table row.
- `docs/DESIGN_RULES.md` — record that hand-authored polygon planes append
  while derive nodes clobber, so the distinction is written down rather than
  folklore.
