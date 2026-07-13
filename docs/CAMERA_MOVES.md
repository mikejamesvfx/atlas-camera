# Camera Moves & Marketing Renders

**From a single still photo to an animated camera move in Nuke — with occluded
areas filled by predicted "X-ray" geometry instead of tearing to black.**

This is the pipeline behind the marketing shots: take one image, recover its
camera, build a projected 2.5D scene (visible surfaces **plus** a predicted
hidden-geometry layer), export a Nuke scene, and keyframe a dolly. As the camera
slides, foreground elements part and reveal geometry the original photo never
saw.

> **Prerequisite — experimental mode.** The X-ray node is gated. Launch ComfyUI
> with `ATLAS_EXPERIMENTAL=1` (add `set "ATLAS_EXPERIMENTAL=1"` to your
> `run_nvidia_gpu.bat`). LaRI (the X-ray backend) is CUDA-only and user-cloned —
> see [INSTALL.md](INSTALL.md). Non-CUDA users can still do the *plain*
> projection + camera move; they just skip the X-ray fill.

## The pipeline

```
LoadImage → AtlasLearnedSolveFromImage → AtlasDepthMap
    ├─ AtlasDeriveReliefMesh ............ base visible geometry
    ├─ AtlasDepthLayerMask → AtlasPredictHiddenGeometry (LaRI)
    │        → hidden_mask → GrowMask → InvertMask   (X-ray region)
    │        → patched depth + paint_matte
    ├─ INPAINT (ExpandMask → LaMa) ...... clean plate behind occluders
    ├─ AtlasCleanPlateLayer [FG] ........ original photo, visible surfaces
    ├─ AtlasCleanPlateLayer [X-RAY] ..... predicted geometry + inpainted plate
    └─ AtlasExportNukeLayers ............ one .nk with both layers + RenderCam
```

The two `ProjectionSource` layers — `fg_occluders` (the real photo) and
`bg_xray` (predicted hidden geometry, painted with a LaMa clean plate) — export
as separate projection cameras through one `ScanlineRender`.

## Per-scene settings

Pick the depth model and sky handling by scene type:

| Scene | `depth_model` | `sky_heuristic` | Notes |
|---|---|---|---|
| **Outdoor** (architecture, landscape) | `V2-Metric-Outdoor` | **on** | the default; sky correctly excluded |
| **Interior** (rooms, hangars) | `V2-Metric-Indoor` or MoGe-2 | **off** | it auto-disarms on interiors, but off is explicit |

X-ray pays off most where a **foreground occludes structure** — dense cityscapes,
temple/ruin fields, interiors with consoles. Open landscapes (snow, water, plain
aerial) get little hidden geometry (LaRI is an architecture model) — the base +
foreground projection still gives a fully dolly-able scene, the `bg_xray` layer
is just small. Measured coverage: temple city ~50%, interior portal ~24%, open
terrain near 0 (graceful).

## The camera move, in Nuke

1. **File → Open** the exported `nuke_layers.nk`. `RenderCam1` auto-wires to
   `ScanlineRender1` on load (a `Root.onScriptLoad` callback does it).
2. Select **`RenderCam1`** and keyframe **`translate` x** across the timeline.
   Wide 2.39:1 plates suit a **dolly-left/right** (e.g. −6 → +6) — the sideways
   parallax is where the X-ray reveal reads best. The channels are unlocked
   (`rot_order XYZ` + translate/rotate, not `useMatrix`), so they keyframe.
3. Drop a **Write** on `ScanlineRender1` and render.

As the camera moves off the recovered viewpoint, foreground silhouettes slide
and the `bg_xray` layer shows through — predicted surface where there would
otherwise be a black hole.

**If a silhouette looks steppy** on a big move, raise `relief_grid` (384 → 512+)
and re-export that scene; the projected texture is already full-res, so this only
sharpens the geometry.

**If a foreground subject's relief runs away backward** (monocular depth
"bananas" tall/soft structures), drop an `AtlasBoundedBand` between the solve and
the layers: feed it the subject's mask and it measures the subject's own depth
extent `W`, emitting one cutoff at `near + 2·W`. Wire its `band_split` into both
the foreground clean-plate layer (`band_side=foreground` — relief clipped at the
cutoff) and the background card (`band_side=background` — the card falls back
behind the cutoff for stronger dolly parallax). One measured boundary, both
layers, no hand-tuned distances.

## Performance & memory

The full band + inpaint pipeline at 4–8K is **RAM-heavy**: each full-resolution
RGBA-float plate is ~0.5 GB, and several bands + inpaint passes are held at once.
On a memory-constrained machine (or with other big apps open — Maya, a browser
with many tabs) a `layers=4` + `inpaint` run at 7–9K can hit
`Unable to allocate … MiB`. If that happens:

- close other large applications (free RAM matters more than VRAM here),
- lower `mesh_resolution` (512 → 384/256) and/or `layers`,
- or downscale very large plates before solving.

The single-image X-ray → Nuke marketing workflow above is lighter than the full
4-band clean-plate master and generally fits comfortably.

## Batch tip

To render many scenes, save one workflow per image (swap `LoadImage` + the
export `output_dir`) so each is reloadable and tunable, rather than firing them
API-direct. See the shipped `atlas_input_quickstart_workflow.json` for the
plain (no-X-ray) camera-move path, and [DCC_EXPORTS.md](DCC_EXPORTS.md) for the
Nuke/Maya/USD export details.
