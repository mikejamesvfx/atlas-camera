# Atlas Camera — the technology, and why it is a different category from other ComfyUI 3D systems

**Thesis.** Almost every "3D" node pack for ComfyUI is a *mesh generator*: feed an
image, get a textured mesh (or a gaussian splat) out. Atlas Camera does not
generate a mesh. It **recovers a real, metric camera** from a single still, and
turns the photo itself into a **camera-projected matte painting** — the exact
workflow a film/VFX pipeline uses for set extensions and DMP. The output is a
camera and a projection setup you can open in Nuke, Maya, Blender, or USD, not an
asset you have to wrangle. This document explains the technology and, section by
section, why that makes it a fundamentally different tool.

---

## TL;DR

- **It solves a camera, not a mesh.** The primary result is a pinhole camera —
  intrinsics, extrinsics, horizon, confidence — in a documented right-handed,
  Y-up world. That is the thing a matchmove/projection pipeline actually needs.
- **Deterministic first, learned only where it must be.** The geometric solve is
  closed-form and reproducible; the optional neural prior (GeoCalib) is a
  *calibration* step feeding deterministic math, not a diffusion black box.
- **Matte-painting projection, not baked texture.** Texels are assigned by
  ray-casting each surface point back through the recovered camera into the
  source photo. Geometry at slightly-wrong depth still receives exactly the
  pixels its silhouette subtends — the classic Nuke/Maya camera-projection model.
- **CPU-first, zero-dependency core.** The core is pure Python/NumPy. `torch` is
  imported lazily *inside* the neural nodes, so **all 54 nodes register with no
  GPU and no torch present** — only the AI features need `[neural]`.
- **DCC-native handoff verified in the real applications.** The Nuke `.nk` and
  Maya `.ma` exporters were validated by actually rendering in Nuke and opening
  in Maya (via `mayapy`) — not by reading docs. Coordinate conversions happen
  only at adapter boundaries.
- **A realtime, navigable, fullscreen viewport — not a preview.** Compose simple,
  elegant camera moves — dolly, orbit, pan — over the projected environment, go
  true fullscreen, and render to a project format up to **8K** — live, inside the
  ComfyUI graph. No other 3D pack has an interactive viewport at all.
- **Color-managed, float-safe throughput up to 8K.** Plates are tracked by
  reference in their real colorspace (ACEScg) and bit depth (EXR 16f/32f); the
  projection path stays floating-point end to end and hands off cleanly to OCIO /
  Nuke / Maya / Resolve.
- **A full 2.5D DMP toolkit, not a single mesh.** Depth-banded clean-plate
  layers, sky-dome separation, disocclusion depth-inpainting, multi-angle patch
  projection, hidden-geometry (X-ray) prediction, edge mattes and beveled skirts.

---

## 1. What "ComfyUI 3D" means today, and where Atlas sits

The existing ecosystem clusters into three families, all excellent at what they
do — and all doing a *different job* from Atlas:

1. **Generative image-to-3D.** TripoSR, InstantMesh, Hunyuan3D, TRELLIS, CRM,
   Stable-Fast-3D, LGM, Zero123-style multi-view diffusion. These synthesize a
   textured mesh or splat from an image using large model weights on a GPU. The
   geometry is *invented* to look plausible; it lands in a normalized, unit-cube
   frame with no metric scale and no relationship to a real camera.
2. **Monocular depth → displacement.** MiDaS / Depth Anything drive a heightfield
   or displaced plane. Useful for parallax, but the "camera" is implicit and the
   mesh is a naive depth surface with no silhouette handling.
3. **Gaussian splatting / NeRF-adjacent.** Radiance-field representations, usually
   from many views, optimized on a GPU.

**Atlas is none of these.** Its job is *camera recovery and camera projection*
from a single photograph — the inverse-problem, geometry-first side of the field.
It sits closer to a photogrammetry/matchmove tool (PTGui, 3DEqualizer, Nuke's
CameraTracker) than to an image-to-3D generator, but built to run inside ComfyUI
and hand off to a DCC.

---

## 2. The output is a real camera — with conventions a pipeline can trust

The recovered result is an `AtlasSolve`/`LatentScene` carrying an `AtlasCamera`:
full 4×4 view matrix, focal length in mm *and* pixels, sensor size, principal
point, horizon line, and a **confidence value with a source-method tag** so you
know *how* it was solved and how much to trust it.

Crucially, the conventions are fixed and documented, not incidental:

- World is **right-handed, Y-up**.
- The recovered camera **faces world −Z** (canonicalized so Maya/Nuke default
  cameras line up without a manual 180° fix — a bug found by a real Maya lineup).
- Image space is **origin top-left, x-right, y-down**.
- All geometry/camera math uses the full 4×4 view matrix end-to-end; the 3×3
  rotation is never used to rebuild world math (it has a transpose ambiguity).

Generative-mesh systems discard the camera entirely — you get geometry in an
arbitrary frame and have to guess a camera to film it. Atlas gives you the camera
*first*, because in a projection/matchmove pipeline the camera **is** the deliverable.

---

## 3. Deterministic geometry first; the learned prior is calibration, not generation

Two solve engines, both auditable:

- **`vanishing_points`** — a closed-form geometric solve from detected vanishing
  points and the horizon. Seeded and reproducible: same photo, same numbers,
  every run. Fast and dependency-light (NumPy + OpenCV).
- **`learned`** — the GeoCalib single-image prior predicts focal length and
  gravity for AI-generated or perspective-ambiguous images where classical VP
  detection is fragile. But it feeds the **same deterministic downstream math** —
  it is a calibration estimate, not a generative reconstruction.

This matters for production: the result is explainable (here are the vanishing
points, here is the horizon, here is the confidence), reproducible, and free of
the seed-roulette nondeterminism inherent to diffusion-based 3D. Scale is
**measured, not assumed**, via a tiered cascade (known-size reference object →
depth ground-plane fit → flagged default), and no low-confidence estimate is ever
silently promoted.

---

## 4. The core architectural difference: camera-projection texel assignment

This is the single most important distinction, and it is worth stating precisely.

A generative-mesh node **bakes** a texture into UVs once; the mesh then carries
that texture wherever it goes. Atlas instead treats geometry as a **projection
surface**. In the viewport's projection shader (and in the exported Nuke/Maya
projection rigs), every surface fragment's world position is cast **back through
the recovered camera** to find which source-photo pixel it subtends; fragments
that fall behind the camera or outside the frame are discarded.

The consequence is the defining property of matte-painting projection:

> Geometry at slightly-wrong depth still receives exactly the pixels its
> silhouette subtends. From the recovered viewpoint the photo reassembles
> perfectly; scale error shows up only as *parallax* when you orbit — never as
> smeared or mis-sampled texture.

That is why you can block out crude proxy geometry (a few planes, a relief mesh)
and still get a clean projected image: the projection is exact by construction,
independent of how approximate the geometry is. A baked-texture mesh cannot do
this — get the geometry wrong and the texture is wrong with it.

Atlas builds real projection **topology** for the DCCs to match: a Nuke
`Project3D2 → Card/ReadGeo2 → ScanlineRender` graph, a Maya `projection` network
driven by `cameraShape.message → projection.linkedCamera`. These were reverse-
engineered by rendering them, not by trusting documentation (see §6).

---

## 5. CPU-first, zero-dependency core — no GPU required for the basics

The dependency architecture is a deliberate feature, and it is verifiable:

- `atlas_camera.core` imports **no torch at all**.
- In the ComfyUI node module, `import torch` / `import numpy` live **inside** the
  functions that use them — never at module load — so **all 54 nodes register in
  ComfyUI with zero heavy dependencies present**. Only the neural nodes raise an
  informative install hint if `[neural]` is missing; the pack never fails to load.

This yields three honest capability tiers:

| Tier | Install | What runs | GPU |
|---|---|---|---|
| **Zero-dependency** | core (pure Python) | schema, solve load/save, Maya/Blender/Nuke/USD export | No |
| **Pure NumPy** | `[vision]` (numpy + opencv) | vanishing-point camera solve, ground/horizon/depth-from-camera masks, reference-scale, projection math | **No** |
| **Neural** | `[neural]` (torch + GeoCalib) | learned single-image solve, monocular depth, depth-driven geometry, VLM assess, patches, hidden geometry | CPU-capable, GPU-recommended |

Most 3D packs hard-require a CUDA `torch` and multi-gigabyte weights just to
*load*. Atlas lets an artist solve a camera and export a projection setup to their
DCC on a laptop with no GPU — then scale up to the neural features when they have
one. The `tools/smoke_check.py` script proves this per machine: on a torch-free
box the neural tier reports `skip` while core, node-registration, and the NumPy
vision solve all pass.

---

## 6. DCC-native handoff, verified inside the real applications

Atlas is built to *leave* ComfyUI cleanly. Exporters target Nuke (Python script
+ native `.nk`), Maya (`.ma` + review scene), Blender, USD (camera + animated
camera path), a relief-mesh OBJ/GLB, and an OCIO-aware, float-safe plate-tracking
"Output Desk" for color-managed handoff. Coordinate-system conversions happen
**only at adapter boundaries** (Blender Z-up, USD stage axis, OpenCV in the vision
layer) and are never silent.

These were not written from documentation and hoped to work — they were validated
in the shipping applications:

- The **Nuke** graph was built live in Nuke and *rendered* end-to-end, which
  surfaced real bugs a doc read would never catch: `Card3D` has no `xsize`/`ysize`
  (the plain `Card` is the right node), `ScanlineRender`'s inputs are bg/obj/cam
  not obj/cam, `Project3D2` is a 2-input node, and Windows backslash paths are
  eaten by TCL escaping (forward slashes fix it).
- The **Maya** `.ma` was opened in **Maya 2027 via `mayapy`** (37 automated
  checks), which caught two real bugs now fixed in both Maya exporters: the
  `projection` node has no focal/aperture attrs (the frustum comes from the linked
  camera), and Maya's OBJ importer lands values as centimeters regardless of scene
  unit.

The Nuke and Maya layered exporters share one collection pass, so the two DCCs
can never drift out of sync.

**Color-managed, float-safe, up to 8K.** An "Output Desk" tracks the real plate
by *reference*, not by the ComfyUI preview — its registered colorspace (ACEScg by
default) and bit depth (EXR 16f/32f) ride the solve into the Nuke/Maya/USD exports,
with an explicit proxy guard so an 8-bit browser preview is never mistaken for
final data. Atlas deliberately does **not** reinvent color science — final
fidelity belongs to OCIO Write, Nuke, Maya, and Resolve — it keeps the pipeline
*honest*: the projection path stays floating-point end to end (the only 8-bit step
is an optional AI patch), and the render format is a project-level ShotCam up to
**8192 px**. That is a genuinely different posture from a generative-mesh node,
which bakes an 8-bit texture and has no concept of a working colorspace at all.

---

## 7. A 2.5D matte-painting toolkit, not a single mesh

Because the projection model is exact, Atlas can go far beyond one surface. It
implements the full digital-matte-painting (DMP) 2.5D reconstruction doctrine as
composable nodes:

- **Depth-banded clean-plate layers** — split a solved photo into depth bands,
  inpaint what each foreground occluder hides, and project each clean plate onto
  its own depth-clipped geometry, so orbiting off the recovered view no longer
  reveals black holes.
- **Sky-dome separation** — a real segmentation drives a constant-distance sky
  card with deterministic edge-extend and frame-outpaint (Nuke-style, not an
  inpaint) for parallax and pan slack.
- **Disocclusion fill that inpaints the *depth*, not just the color** — a band
  clip leaves a hole in the background geometry exactly where the occluder stood;
  Atlas diffuses depth across that footprint so the inpainted pixels land on real
  geometry.
- **Multi-angle patch projection** — register an AI novel view by *constructing*
  a patch camera (orbiting the recovered camera in its own world frame) and
  projecting the new angle onto the existing geometry, filling grazing/occluded
  areas the primary camera never saw.
- **Hidden-geometry ("X-ray") prediction** — layered ray-intersection models
  (LaRI / World-Tracing) predict the surfaces *behind* occluders so a dolly-in
  reveals predicted geometry rather than a guess.
- **Per-pixel edge mattes, beveled occlusion skirts, facing-ratio masks** — the
  edge doctrine that keeps silhouettes clean without infinite tessellation.

Each is a small, single-job, composable node — combined explicitly through a
Nuke-Merge-style geometry-merge node, never auto-guessed.

---

## 8. Architecture: a composable "latent scene," honest about what it knows

- **`LatentScene`** is a growable container with slots for `depth`, `geometry`,
  `lighting`, and `semantics`. Unsupported components are *described* in review
  packages, not silently omitted — the schema is honest about its own frontier.
- **Geometry derivation is artist-selected, never auto-detected.** Relief mesh vs.
  fitted primitives vs. RANSAC planes vs. Manhattan room — the artist picks the
  strategy that fits the shot, because auto-detection produces confidently-wrong
  results on the cases it misjudges. Convenience presets exist, but they are
  explicit bundles, not hidden magic.
- **Confidence and scale are first-class.** Every solve records which evidence
  tier set its metric scale; VLM/LLM suggestions are recorded as candidates and
  only applied on explicit confirmation.
- **Zero required runtime dependencies** in the core; every optional import is
  guarded with an actionable `pip install` hint.

---

## 9. A realtime, navigable viewport — not a preview thumbnail

This is the part that has to be *seen* to land, and it is unlike anything else in
the ComfyUI 3D space. Atlas ships a self-contained Three.js viewport node
(vendored bundle, no CDN) that inherits the recovered camera and projects the
photo onto the derived geometry **in real time**, and then lets you *compose a
camera move through it the way you would on a stage*:

- **Simple, elegant camera moves.** Compose the moves a film camera makes —
  dolly, orbit, pan, push-in — with intuitive, real-time viewport controls (track
  in/out, left/right, up/down, Shift for 4×) and a roll-preserving orbit, so you
  block a shot the way a layout artist works a stage, not the way you nudge a
  thumbnail.
- **True fullscreen.** One click takes the canvas (and every HUD/diagram overlay)
  to fullscreen at the render resolution — a working review surface, not a
  postage stamp in a node.
- **Scales to the format, up to 8K.** The render/output resolution is driven by a
  project-level ShotCam (sensor × lens × long-edge, to **8192 px**), independent
  of whichever photo was solved — so the viewport conforms to a real
  anamorphic/UHD delivery format, not the source image's incidental size.
- **Authoring and passes in the same surface.** Keyframed camera moves (author,
  play at 60 fps, bake frames for a Video Combine node, export a time-sampled USD
  camera), render passes (shaded / depth / normal / mask), a Safe-Zone probe that
  *measures* the exact orbit envelope with full coverage, and live overlays
  (vanishing-point fan, horizon, camera HUD).

Every other family in §1 hands you a static asset and expects you to bring your
own DCC to look at it. Atlas gives you a color-correct, camera-projected,
navigable environment **inside the graph**, in real time — the projection shader
carries a hand-written linear→sRGB encode and optional per-light relighting, so
what you move the camera through matches what you export.

---

## 10. Side-by-side

| | Generative image-to-3D | Depth displacement | **Atlas Camera** |
|---|---|---|---|
| Primary output | Textured mesh / splat | Height-displaced plane | **Recovered metric camera + projection setup** |
| Camera | Discarded / implicit | Implicit | **Explicit, metric, documented conventions** |
| Determinism | Diffusion, seed-dependent | Deterministic depth | **Closed-form solve; learned prior is calibration only** |
| Texture model | Baked into UVs once | Baked | **Live camera projection — exact from the recovered view** |
| Metric scale | None (unit cube) | None | **Measured, tiered, confidence-tagged** |
| GPU to load | Required (CUDA torch + weights) | Usually required | **None — 54 nodes register on CPU; GPU only for AI features** |
| DCC handoff | `.glb`/`.obj` to wrangle | Mesh | **Native Nuke/Maya/USD projection rigs, verified in-app** |
| Reveals on camera move | N/A (full mesh) | Stretches | **Clean-plate layers + patches + X-ray predicted geometry** |
| Interactive viewport | None (export & open elsewhere) | None | **Realtime, fullscreen, cinematic camera moves, in-graph** |
| Color / resolution | 8-bit baked texture, no colorspace | 8-bit | **Float-safe, OCIO-aware plate tracking, render to 8K** |

---

## 11. Where it fits in a real pipeline

```
still photo (or AI render)
      │
      ▼
Atlas solve  ──►  recovered camera + horizon + confidence
      │
      ├─►  derive projection geometry (relief mesh / primitives / planes)
      │
      ├─►  layer it (sky dome, depth bands, clean plates, patches, X-ray)
      │
      ├─►  inspect in the in-graph viewport (project, orbit, author a camera move)
      │
      └─►  export ──►  Nuke .nk / Maya .ma / Blender / USD / relief OBJ+GLB
                       (camera + projection network, color-managed plate)
```

The deliverable is a **set-extension / DMP projection setup that opens in the
compositor or 3D app ready to render** — camera solved, plate projected, geometry
in place — which is precisely what a matte-painting or environment shot needs and
precisely what a mesh generator does not provide.

---

## 12. Honest non-goals and limits

- **It is not a mesh generator.** If you want an invented, fully-3D asset from an
  image, use a generative image-to-3D pack — that is a different job.
- **Single-photo reconstruction only sees a cone.** Derived geometry covers what
  the recovered camera saw; large orbits reveal un-photographed space unless you
  add clean-plate layers, patches, or X-ray geometry (which is exactly why those
  tools exist).
- **The learned solve and all depth-driven features need `[neural]` (torch).**
  They are CPU-capable but GPU-recommended; the *basics* (VP solve, masks, DCC
  export) are the part that needs no GPU.
- **Geometry strategy is a deliberate choice, not auto-magic.** Picking the wrong
  derivation for a shot produces a wrong result; the presets help, but the artist
  is in the loop by design.

---

*Atlas Camera is MIT-licensed, has a zero-dependency core, and installs into
ComfyUI as a clone-and-go custom node. See `README.md` and `INSTALL.md` to get
started, and `docs/USER_GUIDE.md` / `docs/ECOSYSTEM_GUIDE.md` for the artist- and
system-level guides.*
