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

## Two ways to author the move

The Nuke route below is the **hand-keyed** one: you open the exported script and
key the camera yourself. It is still the right answer when a comp artist owns the
move. But Atlas has authored camera moves **server-side** since the camera-path
system landed, and that lane is what every automated consumer reads:

| | Hand-keyed in Nuke | Authored as an `ATLAS_CAMERA_PATH` |
|---|---|---|
| Author with | `RenderCam1` translate/rotate keys | the 🧊 Viewport's 🎥 Camera Path mode, or `AtlasCameraMovePreset` 🎬 headless (12 presets: orbit, dolly, crane, push-in, vertigo, …) |
| Interpolation | Nuke's curves | Catmull-Rom with per-segment easing (`core/camera_path.py`, mirrored in JS for live scrubbing) |
| Lens over time | whatever you key | a vertical `fov_deg` channel — `vertigo` keys it, so a counter-zoom is one preset |
| Reaches | the comp | `AtlasMoveBudget` 📐 (how far the plate supports moving), `AtlasDisocclusionGuide` 🟣, `AtlasGhostPixelMap` 👻, `AtlasConditioningBundle` 🎛, `AtlasExportCameraPathUSD` 🎥 |
| Exports | the render | an **animated USD camera** (the only animated-camera export Atlas writes; Nuke/Maya/Blender receive a still camera plus a keyable rig) |

A path is worth authoring in Atlas whenever something other than a human needs to
read the move — a move budget, a disocclusion guide, or a generative video model.

### Handing the move to a video model

`AtlasConditioningBundle` 🎛 renders every frame of a path once and emits what a
video model can actually be conditioned on: metric depth, world normals, world
position, **analytically derived optical flow**, per-frame K, and the ghost class
map that says which pixels the photograph never saw. This is the geometric
alternative to the semantic camera control every hosted video platform ships —
instead of "slow dolly left, 35 mm", the model receives the exact displacement of
every pixel and the exact set of pixels it is allowed to invent.

Flow is derived, not estimated: with depth and two calibrated cameras it is
closed-form, so Atlas already knows the answer a flow network would guess at.

`AtlasAdherenceScore` 📐 then measures whether the model obeyed. Note what it
refuses to do: it will not report a headline number for a clip that repeats its
first frame (which scores *well* on a naive measure, since a static plate agrees
with the reprojection wherever the move is small), and it requires a
prompt-only control arm as a wired input. See its catalog row for why both
refusals are load-bearing.

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

**If the metric scale looks wrong** (distances tiny for an elevated vista), the
solve had no ground plane to fit and fell back to an assumed ~1.6 m eye height —
often ~10× too small looking out over a city. Drop an `AtlasScaleOverride`
between the solve and the geometry and dial `scale` (10.0 = "1:10") or set an
absolute `camera_height_m`. Every distance — the projected geometry, the 📏 Band
Box cutoffs, the exported Nuke/Maya cameras — scales together; the view itself is
unchanged. Toggle **📏 Band Box** in the viewport to read each layer's clip
distance while you dial it in. (A known-size reference via `AtlasReferenceScaleSolve`
is a true lock if you have one; the override is the honest by-eye fix when you don't.)

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
