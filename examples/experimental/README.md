# Experimental / demonstration workflows

**Not the shipping catalog.** `examples/*.json` is pinned to exactly three
workflows by `tests/test_example_workflows.py` (quickstart, staged master, OCIO
quickstart) so an accidental deletion *or* an unreviewed addition fails loudly.
That pin globs `examples/*.json` and is **not** recursive, so this directory is
the home for scene-specific demonstration builds that shouldn't widen the
supported surface.

These are calibrated against **one specific plate** and one machine's installed
node packs. Treat them as worked examples to read and cannibalise, not as
defaults.

---

## `atlas_castle_dmp_marketing.json`

The full 2.5D matte-painting build on the sea-cliff castle plate — sky, ocean,
and castle+rocks as separate projected layers, plus an exportable castle mesh.

**Verified 2026-07-15** against a real ComfyUI: 84 nodes / 65 links / 10 groups
load with zero node errors, all 37 `GetNode` rails resolve, and `/prompt`
server-side validation returns 200 (the 84 UI nodes flatten to 24 executing
nodes — Set/Get are frontend-only virtual nodes). It has **not** been run to
completion end-to-end.

### The shape

```
LoadImage → 🧭 AtlasAssessImage ──rails──► plate / sam_sky / sam_fg
                                              │
      ┌───────────────────────────────────────┴──────────────────┐
      │  solve → ✅ AtlasSolveGate → rail: solve                  │
      └──────────────────────────┬───────────────────────────────┘
                                 │
     depth rails:  MoGe (near/castle)  ·  DA3 (far/sky+water)
                                 │
   ┌─────────────── DMP SCENE (projection_sources) ──────────────┐
   │ ☁ sky dome (DA3 depth, LaMa-inpainted plate, outpainted)    │
   │ 🌊 water   (band_geometry=ground, band_side=background)      │
   │ 🏰 castle  (relief, band_side=foreground, SAM3-scoped)       │  → 🖼 master viewport
   └──────────────────────────────────────────────────────────────┘
   ┌─────────────── EXPORT MESH (proxy_geometry) ────────────────┐
   │ AtlasDeriveReliefMesh (exclude = NOT castle)                 │
   │ 🔧 AtlasExportReliefMesh (fill_interior_holes)               │  → 🔍 fill-preview viewport
   └──────────────────────────────────────────────────────────────┘
```

### Why two branches

`AtlasExportReliefMesh` reads `projection_scene.proxy_geometry`, which
`AtlasDeriveReliefMesh` writes. Clean-plate layers append to
`projection_sources` instead — so **a castle built as a layer is not exportable
by that node at all**. The DMP scene and the OBJ are two products off the same
gated solve, not one chain.

### Load-bearing settings

| Setting | Why |
|---|---|
| `camera_height_m = 40`, `height_mode = assume` | Single-image scale is ambiguous; the 1.6 m default is ~25× small on a clifftop and **every** metric follows it (band cutoffs, the hole-fill band box, DCC cameras). 40 is a sighting-in guess — dial it against the ℹ Info HUD. |
| MoGe near / DA3 far | MoGe's far field runs away (>1000 m) and it culls sky — disqualifying for a sky dome. Depth model is a **per-band** choice, not a global default. |
| Priorities sky −10 < water −5 < castle 0 | **Farthest-highest.** At a watertight seam the layer *behind* must win the depth near-tie; nearest-highest renders each band's edge smear in front of the layer behind it (striped seams). |
| `edge_extend_px`: castle 0, others 32/48 | The smear lives on the layers **behind**; the frontmost band keeps a clean cut matte. |
| LaMa `seed = 0, fixed` | ComfyUI auto-adds `randomize` to any widget named `seed`, silently re-rolling the plate every queue. |
| `fill_depth_near_m/far_m = 0` (band box off) | Both must be > 0; it needs a trustworthy metric scale; and window mode bypasses the largest-loop guard — the one way to accidentally cap the outer frame. |
| `max_hole_edges = 128` @ `relief_grid = 1024` | Counts **grid** edges, so it scales with the grid (128 @ 1024 ≈ 64 @ 512). Measured on this plate: 128 → 97 holes/+1316 faces; 1024 → 101/+2669 — the 4 extra are 128–617-edge loops, i.e. big flat caps over real silhouettes. |

### Requirements

ComfyUI-KJNodes (Set/Get rails), ComfyUI-RMBG (`SAM3Segment`),
comfyui-inpaint-nodes + `big-lama.pt`, the `[moge]` and `[neural-da3]` extras,
a vision VLM on lmstudio `:1234`, and `atlas_seacliff_castle.png` in
`ComfyUI/input/`.

No VLM? The SAM3 prompts fall back to the literals typed into the nodes
(`sky` / `castle and rocks` / `ocean sea water`) — everything downstream still
works. **Note:** this ComfyUI's `LoadImage` lists no subdirectories, so the
plate must sit at the root of `input/`.
