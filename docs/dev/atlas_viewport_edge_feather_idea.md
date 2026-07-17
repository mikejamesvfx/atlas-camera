# Viewport edge feather — GLSL design sketch (noted 2026-07-17, not scheduled)

User request: soften the relief mesh's hard triangle/tear edges in the
VIEWPORT ONLY, so silhouettes read as feathered mattes instead of jagged
geometry. Display-only — never touches exports, mattes, or measurement
(same doctrine as `preview_expand`).

Three composable GLSL mechanisms, cheapest first — all live in
`PROJECTION_FRAGMENT_SHADER` (`atlas_blockout.js`), all synced by the
existing per-frame uniform push (`syncProjectionLightUniforms` pattern,
since projection materials are rebuilt every execution):

1. **Soft matte edge via dithered discard (cheapest, biggest win).** The
   shader already samples `uMatte` and hard-discards at 0.5. Replace with
   `a = smoothstep(0.5 - f, 0.5 + f, matte)` and discard by an ordered
   Bayer threshold on `gl_FragCoord` (screen-door transparency). Dithering
   keeps discard semantics — no blending state, no depth-sort problems
   across the multi-layer stack, works with `depthWrite:true` unchanged.
   One uniform (`uMatteFeather`), one 4x4 Bayer constant.

2. **Boundary-distance vertex attribute (true geometric feather).**
   `build_relief_mesh` already knows exactly which cells border a tear/
   silhouette/band clip — bake per-vertex distance-to-boundary IN CELLS
   (BFS over the decimated grid, ~free at build time), ship as a vertex
   attribute, and in GLSL: `alpha = smoothstep(0.0, uFeatherCells,
   vBoundaryDist)` feeding the same dithered discard. This feathers the
   GEOMETRY edge itself, not just the matte — the jagged triangle rim
   fades out over N cells. Backend cost: one extra float attribute in the
   serialized payload (~4 bytes/vert).

3. **Stretch fade (bonus — targets the ugliest triangles).** Where a
   triangle is viewed at extreme stretch (grazing skirt, disocclusion
   rubber-sheet), texels smear into streaks. In-shader:
   `stretch = length(fwidth(vUv) * uImageSize)` and fade alpha above a
   threshold — the streaky triangles dissolve instead of smearing.
   Zero attributes needed; `fwidth` is core WebGL2.

Recommended order: (1) alone likely covers most of the perceived
harshness; (2) if bare tears (no matte) still read jagged; (3) as a
toggle for grazing-angle smear. Ship behind ONE toolbar toggle
("🪶 Feather" + a pixels/cells slider), default OFF so every existing
look is untouched, session-only like ☀/📊/💡. The deterministic
export passes (`renderAllPasses`) and ⏺ Bake must stash the toggle OFF
during capture — same guard pattern as the 🎯 pivot gizmo.

Known tension to design around: dithered discard sparkles under motion at
low feather widths — mitigate with a blue-noise texture instead of Bayer
if it bothers, or clamp the minimum feather to ~2px.
