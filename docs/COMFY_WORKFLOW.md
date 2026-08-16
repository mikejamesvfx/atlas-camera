# Atlas Camera in ComfyUI — mental model and troubleshooting

**What this page is:** the two-track mental model for how the node pack is
shaped, how to pick a source image the solver can actually use, and the four
symptoms that look like bugs and are not.

**What it is not:** a node reference or an install guide. Those live in
[docs/NODE_CATALOG.md](NODE_CATALOG.md) (every node's inputs, outputs and
behaviour; the symlink setup; the `/atlas/camera_data` endpoint; the
`comfy/` module layout) and [INSTALL.md](../INSTALL.md). This page used to
carry cut-down copies of both, and they drifted — the copy went stale while
the originals moved on. It now links instead.

For the artist-facing walkthrough see [docs/USER_GUIDE.md](USER_GUIDE.md); for
the ecosystem map and the node catalog in wiring order see
[docs/ECOSYSTEM_GUIDE.md](ECOSYSTEM_GUIDE.md).

---

## The two tracks

The `atlas_camera.comfy` package splits across two tracks, and almost every
confusing symptom below comes from mistaking one for the other.

**Track 1 — Python-only nodes.** Solve, decompose, depth, masks, geometry
derivation, per-DCC exports. They run entirely in ComfyUI's Python process,
have no browser dependency, and behave like any other node: one execution, one
set of outputs.

**Track 2 — `AtlasBlockoutViewport`.** A Three.js viewport embedded in the node
panel. The recovered camera is applied to the Three.js camera so the scene
pre-aligns with the source photo; the artist places geometry, clicks Render
Proxy Passes, and four IMAGE outputs (shaded / depth / normal / mask) flow back
into the graph.

Track 2 is a **browser round trip**, which is why it takes two executions to
produce anything. That is the single most important thing on this page.

---

## Choosing a source image the solver can use

The classical vanishing-point solve (`AtlasSolveFromImage`) needs strong
perspective cues:

- **real photographs** rather than AI renders — generated images are often
  locally plausible and globally inconsistent, which breaks the RANSAC fit
- **exterior, eye height, visible ground plane**
- **at least one clear vanishing direction** — a road, a building face, a
  tiled floor
- **horizon in the upper third** of the frame

Hard cases, where you should expect `cam_y ≈ 0` or a degenerate result: AI
imagery, interiors with heavy occlusion, fisheye or strongly tilted lenses,
industrial scenes with no readable ground.

For those, prefer `AtlasLearnedSolveFromImage` — the GeoCalib prior reads focal
length and gravity from image content and reports meaningful confidence, which
is why it is the recommended single-image default. `AtlasConstrainedSolve` with
explicit line and scale constraints is the artist-guided fallback.

---

## Symptoms that are not bugs

### 1 · The blockout outputs are black on the first run

**Expected.** Track 2 is a two-pass cycle:

1. `AtlasBlockoutViewport.render()` runs with empty `client_data`, returns
   blank tensors, and caches the recovered camera.
2. The browser extension picks the camera up on `node.onExecuted`, applies it
   to the Three.js camera, and loads the source photo as the background.
3. You place geometry and click **Render Proxy Passes**, which fills
   `client_data` and re-queues the prompt automatically.
4. `render()` runs again with populated `client_data`, decodes the base64
   passes, and returns real tensors.

There is no way to shorten this: pass 1 is what tells the browser which camera
to use. Black outputs before step 3 mean the cycle is working.

### 2 · `AtlasGroundDepthMap` is solid black

**Cause:** `cam_y ≤ 0` — the solve put the camera on or below the Y=0 ground
plane. The depth compute requires it to be above
(`atlas_camera/core/depth_geometry.py`, `valid = (np.abs(ry) > 1e-5) & (cam_y > 0)`),
so every pixel fails the validity test.

**Confirm:** read `cam_y` off `AtlasDecomposeCamera`. For a ground-level
photograph it should be > 0.1.

**Fix:** give the solve a scale reference. `AtlasReferenceScaleSolve` measures
height from a known-size object and is the strongest tier;
`AtlasScaleOverride` sets it by hand; `AtlasConstrainedSolve` accepts
`{"scale_constraints": [{"type": "camera_height_m", "value": 1.6}]}`.

**Status:** by design. An auto-solve with no scale reference cannot determine
camera height — see the scale tiers in
[docs/USER_GUIDE.md](USER_GUIDE.md) Part 1.

### 3 · `AtlasVPVisualization` returns the image unchanged

**Cause:** `solve.vanishing_points` is empty, or every VP projects outside the
frame. Common on AI imagery and heavily occluded scenes.

It is also **empty by construction on the learned solve path** — GeoCalib does
not compute vanishing points at all, so there is nothing to draw. A blank
overlay after `AtlasLearnedSolveFromImage` is correct, not a failure.

**Status:** expected behaviour for low-cue images and for the learned path.

### 4 · `RuntimeError: Added route will never be executed, method HEAD is
already registered`

**Cause:** `atlas_camera/comfy/__init__.py` is executed twice at startup — once
as the ComfyUI custom node and once as the Python package — and both runs try
to register the same aiohttp route.

**Status:** fixed, and the guard that fixes it is load-bearing. See the
double-import section of [docs/NODE_CATALOG.md](NODE_CATALOG.md) before
touching route registration. If the error reappears, check whether another
custom node registered a route with the same path pattern.

---

## Where the viewport gets three.js

The frontend extension imports a **vendored local bundle**,
`atlas_camera/comfy/web/lib/atlas-three.bundle.js` — three.js core plus
`OBJLoader`/`FBXLoader` in one self-contained ESM file, committed to the repo.
There is no CDN fallback and no npm step at runtime, deliberately: ComfyUI's
frontend exposes no three.js import surface to build against.

To upgrade: bump `three` in `ui/package.json`, then
`cd ui && npm install && npm run build:comfy-three`, and commit the rebuilt
bundle.

If the viewport shows its toolbar but no canvas, the bundle is the first thing
to check — a partial checkout leaves the file missing. Open DevTools and look
for `[AtlasBlockout]` errors.

---

## Proxy data is proxy data

Viewport passes and baked path frames are **browser preview data**, 8-bit and
display-referred. When the real source exists as an EXR or another
high-bit-depth plate, wire `AtlasRegisterPlate` → `AtlasAttachSourcePlate` so
the Nuke/Maya/review exporters use the file-backed plate instead of the
preview. The browser preview is display-inferred only; final OCIO and LUT
fidelity belongs to a colour-managed tool downstream.
