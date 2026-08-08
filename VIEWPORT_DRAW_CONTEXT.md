# Atlas Viewport — draw / box / sphere, with snap + edit (working context)

Scope: **only** the in-viewport blockout drawing tools on the **Atlas Viewport**
node (`AtlasBlockoutViewport`). Hand this to Claude Code before tweaking those
tools so it doesn't have to re-read ~6k lines of `atlas_blockout.js`. Nothing
here covers the rest of the viewport (projection, band boxes, debug overlays).

## One-paragraph model

The viewport is a single Three.js widget on the Atlas Viewport node. On top of
the orbit view it has a **draw overlay** (`drawGroup`, name `atlas_draw_overlay`,
flagged `atlasHelper` so render/export passes skip it). You draw blockout shapes
onto the scene; they collect in one array, `drawnPolygons`; and **✅ Apply** bakes
them into the solve and re-queues the graph so the `solve` output carries them
downstream (retopo / export). Nothing is baked until Apply — drawing only builds
overlay previews.

All three primitives share ONE data shape and ONE edit path:

```
{ id, label, enabled, kind, points_world }     // kind: "polygon" | "box" | "sphere"
```

- polygon → `points_world` = outline vertices (N)
- box → `points_world` = 8 corners (4 footprint, 4 top)
- sphere → `points_world` = `[centre, surfacePoint]` (radius = |surface − centre|)

Because box and sphere reuse the same `points_world` array a polygon uses,
**Edit's handles, snapping and deletion work on all three with no separate code
path.** That reuse is deliberate — don't fork it lightly.

## Files

| Piece | File | Notes |
|---|---|---|
| Viewport widget + all draw tools | `atlas_camera/comfy/web/atlas_blockout.js` | the tools live ~L2520–3700 |
| Widget ⇄ node bridge | `atlas_camera/comfy/viewport_payload.py` | serialises the drawn shapes to/from the node |
| The node | `atlas_camera/comfy/nodes_viewport.py` | `AtlasBlockoutViewport` (menu: "Atlas Viewport") |
| Mesh builders | `atlas_camera/core/polygon_planes.py` | `box_mesh_from_corners` (~L248), `sphere_mesh` (~L302), polygon-plane meshing |
| Primitive → triangles | `atlas_camera/core/primitive_mesh.py` | `tessellate_primitive` + unit builders, for measure/export |

Everything about the DRAW / SNAP / EDIT *feel* is in `atlas_blockout.js`. The
`.py` files only matter when a baked shape's geometry comes out wrong.

## Shared state (`atlas_blockout.js`, ~L2521)

```
drawnPolygons          // committed shapes, awaiting Apply
drawPoints/drawRays/drawHits   // outline being drawn + click rays + geometry hits
drawPlane              // { normal, offset } — polygons are drawn ON this plane
editOn                 // ✎ Edit mode
editDrag               // { poly, index } while a handle is dragged
editSnap               // Snap toggle, default TRUE — shared by draw AND edit
drawDirty              // shapes changed since last Apply
drawTilt / drawPush    // nudge the draw plane (radians / metres along normal)
drawGroup              // the atlas_draw_overlay Three.Group (excluded from render)
drawTargets()          // meshes flagged atlasDerived / atlasPatch — what draws hit + snap to
```

## The tools (left tool rail, ~L3300+)

The tools live on a **DCC-style vertical rail** (`drawRail`) pinned to the
TOP-LEFT of the canvas: 44 px buttons with monochrome line-art SVG icons
(`RAIL_ICONS`, stroke:currentColor so state colours recolour them), active
tool = blue fill / snap = green (`syncRailActive`, which runs AFTER each
button's legacy onclick tints and wins). A **status chip** (`railStatus`,
`updateRailStatus`) sits right of the rail showing the live tool + snap state
("Box · Snap on"); every rail onclick is WRAPPED (not replaced — the
mutual-exclusion cross-calls still work) to refresh it. `metaHud` moved to
left:66px/top:46px to clear the rail. Icon-only (full wording stays in
each button's tooltip): **🪄 Wand**, **✏️ Draw**, **⬜ Quad**, **➬ Extrude**,
**▣ Box**, **● Sphere** │ **✎ Edit**,
**🧲 Snap**, **🗑 Delete** │ **✅ Apply**. A **chevron toggle** (`railToggleBtn`,
`syncRailCollapsed`, 2026-08-07) sits at the TOP of the rail and folds the tools
away; the buttons live in an inner `railTools` container so the toggle itself
stays on screen when they are hidden (a control that can hide its own only way
back is a trap). Tools are VISIBLE by default. Collapsing is presentation-only —
it never changes the active tool, touches `drawnPolygons` or sets `drawDirty`,
so folding mid-draw cannot lose work, and `railStatus` stays visible because
with the rail hidden it is the only thing still reporting that Draw or Snap is
live. Session-only, never written to `client_data`: a saved workflow that opened
with its tools already hidden would read as a broken viewport.
🗑 removes the `editSel` shape (or,
with no selection, the most recent shape) and sets `drawDirty`; Apply persists
even a now-EMPTY list when dirty, so deleting the last shape can actually
unbake it. The rail is a child of `canvasWrap` and is NEVER
reparented by `mountControls` — it stays on the viewport even when the main
toolbar moves to an AtlasViewportControls node. The tilt/push `drawAdjust` row
is a contextual flyout anchored right of the rail, shown only while Draw is
active. Draw / Box / Sphere / Edit are mutually exclusive — entering one
cancels the others. **Enter ALWAYS exits the tool** (back to orbit — the
artist's next move is always orbiting to the next hole), in every tool and
every state: it commits the in-progress shape when commit-able (outline ≥3
points, box with height, sphere with radius) and discards a half-built one.
This matters because box/sphere auto-commit on their final CLICK — the artist
then hits Enter with nothing in progress and must still get orbit back. Esc
stays in the tool and just discards the in-progress shape.

**✏️ Draw (polygon)** — click to drop outline points. The first hits on geometry
(`drawTargets()`) establish `drawPlane`; later points project onto that plane
(rays kept in `drawRays` so they re-project if the plane is tilted/pushed). Enter
or Apply closes the loop (`closeDrawnOutline`, needs ≥3 points). Fills a
*see-through hole* — i.e. a plane.

**🪄 Wand** — one-click hole fill (`meshBoundaryLoops`, `onWandClick`): click
INSIDE any enclosed tear. Boundary loops are extracted per drawTargets() mesh
(edges owned by exactly one triangle; vertices deduped by rounded position so
buffer seams don't break loops) and cached on `userData._atlasWandLoops`
keyed by geometry uuid (re-extracted after every execution's rebuild).
**Pinched walks are split at extraction** (`splitLoopAtRepeats`, 2026-08-08):
where two tears meet at a shared vertex the boundary walk returns a figure-8
that reuses that vertex — the backend refuses it as self-intersecting — so
closed walks are split at every repeated vertex id into simple sub-loops
(exact integer ids, before world mapping; sub-loops under 3 verts dropped;
open chains stay whole for the bay fallback). Each lobe fills independently;
`alreadyFilled` dedup compares the first TWO rim vertices because sibling
lobes share their starting pinch vertex. The
click picks the INNERMOST containing loop by projected NDC area
(`WAND_MAX_NDC_AREA 0.8` rejects the mesh's outer border; `WAND_MAX_RIM 600`
rejects monster rims → use Quad/Draw). The fill commits the loop's EXACT rim
vertices as an ordinary drawn polygon (`established_from.rule: "wand_fill"`)
— born welded at every vertex. Already-filled rims are skipped (no duplicate
stacking). **Boundary-bay fallback** (`wandBayFromPath`, rule
`"wand_bay_fill"`): a hole that OPENS onto the mesh's outer border has no
closed interior loop, so when the first pass finds nothing the wand looks for
two rim vertices near the click that nearly touch in WORLD space — mouth ≤
`WAND_GAP_EDGE_FACTOR 8` × the rim's own median edge length (`medianRimEdge`;
tears jag at relief-grid resolution, so that ≈ 8 grid cells — stable across
zoom and scene scale, unlike the earlier NDC tolerance) — bridges them, and
fills the enclosed arc/run. The pair scan is limited to rim verts within
`WAND_BAY_LOCAL_R 0.7` NDC of the click so the O(n²) stays tiny. CRITICAL extraction rules (both found live as "no bay
ever fills"): the boundary walk is NOT length-capped (`WAND_MAX_RIM` caps
what a FILL may use, not extraction — capping the walk silently discarded
the outer border), and walks that dead-end at junction vertices (tears
pinching the border, degree > 2) are KEPT as open chains — bay candidates
come from closed loops AND those chains. Degenerate rims are safe: `_apply_drawn_polygons` wraps each
polygon in try/except and reports "skipped(...)" instead of failing the node.

**⬜ Quad** — Maya-style live quad draw, the tear-filler: 4 clicks (any order —
re-ordered into the non-crossing loop by min perimeter, `orderQuad`) commit a
quad on the 4th; each following quad costs 2 clicks, seeded from the nearest
edge of the previous one (`nearestQuadEdge`), so strips grow in any direction.
Clicks edge/vertex-snap to the tear rim like ✏️ Draw; off-mesh (mid-tear)
clicks land on the plane fit from the points so far. Esc/Enter ends the strip,
Backspace pops point → last quad. **Not a new kind**: every committed quad is
an ordinary kind-less 4-point polygon (`{points_world, plane}`,
`established_from.rule: "quad_draw"`) — meshing, Edit, gizmo, 🗑 and Apply need
zero new code. Adjacent quads COPY the shared edge's values (JSON shapes can't
share references); they coincide at commit, and edge-snap re-meets them if
edited later. Intended split: ⬜ Quad for enclosed medium/large tears, ✏️ Draw
n-gons for boundary edges, ▣ Box / ● Sphere for large-scale mass.

**➬ Extrude** — pull a new quad out of ANY existing drawn edge (quad, n-gon,
box wire; spheres have no edges): grab near an edge (`nearestDrawnEdge`,
screen-space ~14 px pick), drag — the two endpoints are copied and translate
on a CAMERA-FACING plane through the grab point (always well-conditioned; no
grazing-plane blowups), release commits. Output is the same kind-less 4-point
polygon as ⬜ Quad (`established_from.rule: "edge_extrude"`). Backspace pops
the last shape; Enter/Esc exits. Related guard in ⬜ Quad's off-mesh landing:
a landing further from the existing points than 4× their spread (the
grazing-ray blowup after an orbit, found live) is re-landed on a camera-facing
plane through the last point.

**Box (~L3370)** — three-stage blockout SOLID for mass the camera never saw round
the back of: `boxStage` 1 pick base corner → 2 drag footprint → 3 drag height.
Ground contact is **Y-only**: the first click rests the base on the ground (or
geometry via ctrl-click) but X/Z stay where the cursor ray landed, and the
footprint/height follow the cursor freely (the old 1 m grid quantise was
reverted live — it fought the artist when hugging a torn edge).
`boxCornersNow()` builds the 8 corners; `refreshBoxPreview` draws the wire
preview. Enter or a third click finishes, Esc cancels.

**Sphere (~L3600)** — two-stage: click the touch-down point (ctrl-click snaps it
to geometry via `snapHitToEdge`, else it rests on the ground), then drag the
radius freely. `sphereControlPoints()` returns `[centre, surface]`,
stored as `kind:"sphere"`. Preview is three great-circle rings.

## Snap (`editSnap`, default ON)

One flavour: **edge/vertex snap** — drawn/dragged points snap to the nearest
mesh edge/vertex under the cursor (`snapHitToEdge`). This is what makes a patch
rim actually meet a torn hole's geometry. (A second flavour, ground-GRID
quantise of box footprints/heights and sphere radii, existed and was reverted
live — the 1 m jumps fought the artist; ground contact is now Y-only on the
first click.)

**Shift bypasses snap** while drawing (free placement). In **Edit**, Shift instead
constrains a drag to one axis (or press X/Y/Z). The snap helper early-returns
the raw point when `editSnap` is false — so a "snap won't turn off" bug means a
path that isn't checking the flag.

## Edit (`✎ Edit`, `editOn`)

Drag a handle to slide a point WITHIN its own plane; ctrl-click a handle to delete
it (guarded so a polygon can't drop below 3 points, ~L2911). Orbit still works —
only the grab itself suppresses it. Box faces move as quads (`EDIT_BOX_QUADS`),
translating freely. **Vertex weld** (`editWeld`, `findWeldTarget`): dragging a
single vertex near ANOTHER shape's vertex outranks mesh-edge snap — both draw
RED and the drag locks onto the target's exact coordinates; release welds
(coincident values — shapes can't share references through JSON). This is how
the copied shared edge between two ⬜ quads is re-closed after editing.
Same-shape vertices are excluded (a weld inside one outline would duplicate a
point its own triangulation chokes on); gated on the 🧲 Snap toggle like every
other snap. Weld fires in THREE places: live during a free drag; ON RELEASE of
gizmo / Shift / X-Y-Z axis drags (they bypass the live preview to keep the
axis pure but still weld when they land on a corner); and at CREATION — a
⬜ Quad or ✏️ Draw click on an existing drawn corner takes its exact
coordinates (born welded; in Draw it also counts as a plane-establishing hit).
The backend never re-fits `points_world` (`polygon_from_world_points`:
"nothing is re-fitted here"), so coincident coordinates survive Apply intact. When a polygon's points move, its plane is refit from
the moved hits (`atlasEstablishPlaneFromHits`, ~L3027) so the projection basis
stays correct. Edits set `drawDirty`; **Apply** rebuilds.

**Translate gizmo (`editSel` / `editGizmo`).** A grab becomes a persistent
selection (`editSel = {poly, indices}`); on release a Maya-style gizmo — three
coloured, screen-constant axis arrows — appears at the selection centroid
(`ensureEditGizmo`/`updateEditGizmo`, driven per-frame from `animate()` beside
`updatePivotGizmo`). Dragging an arrow tip (`pickGizmoAxis`, same NDC tolerance
as `findEditPointNear`) starts an `editDrag` with `axis` PRE-SET — all movement
still goes through the existing `atlasClosestPointOnAxis` branch. Axis lock
(gizmo, Shift, or X/Y/Z) brightens that arrow and dims the others
(`setGizmoAxisEmphasis`); idle hover over a tip highlights it. Empty click
deselects (without eating the event, so orbit works); deleting the selected
shape, toggling Edit off, or entering another tool hides the gizmo. The gizmo
lives directly in `scene` (NOT `drawGroup`, which is cleared every
`refreshDrawOverlay`) and is `atlasHelper`-tagged on every child so it never
leaks into renders/exports.

## Apply flow

`✅ Apply` (`drawApplyBtn.onclick`): closes any open outline → clears `drawDirty`
→ `persistDrawnPolygonsToClientData()` hands `drawnPolygons` to the node payload →
`app.queuePrompt(0, 1)` re-runs the graph. Backend: `viewport_payload.py` reads
the shapes and `polygon_planes.py` builds the meshes into the solve.

## Gotchas when tweaking

- **`points_world` is world-space and shared across all three kinds.** If you add
  a kind or change the array shape, update all of: the `kind ==` branches in the
  overlay renderer (~L2606), the edit-handle logic, `viewport_payload.py`, and
  `polygon_planes.py`. One shared array is what keeps Edit/Snap kind-agnostic.
- **Nothing persists without Apply.** Drawing only builds overlay previews; the
  payload is written only in `persistDrawnPolygonsToClientData`, and geometry
  rebuilds only on the re-queue.
- **Snap is one flag, two behaviours.** Edge-snap (to `drawTargets()`) vs
  ground-grid snap are different code — a "snap feels wrong" bug is usually in the
  wrong one of the two.
- **The overlay group is `atlasHelper` and skipped by render/export.** Add any new
  preview mesh to `drawGroup` and flag `userData.atlasHelper = true`, or it leaks
  into the projection.
- Line numbers here are approximate — grep the function name, the file drifts.
