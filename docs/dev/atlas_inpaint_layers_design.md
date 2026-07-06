# Atlas Inpaint Layers — 2.5D Clean-Plate Parallax

**Status:** design + implementation brief for Claude Code
**Author:** Mike (design), drafted 2026-07-06
**Depends on external node packs:** `Acly/comfyui-inpaint-nodes` (GPL-3.0), `scraed/LanPaint` (optional generative tier)

---

## 1. Goal

Give Atlas the classic VFX 2.5D matte-painting move: separate a single still into
depth layers, **inpaint the occluded regions behind each foreground layer to make
clean plates**, then project each clean plate onto its own layer geometry. The
result is a genuine parallax camera move (dolly / orbit) from one photo — with no
video model — because when the camera pushes in, the background layer reveals
*inpainted* pixels instead of the black holes we get today.

This is the missing piece behind the existing "geometry goes black when you orbit
off the recovered viewpoint" limitation documented in `CLAUDE.md` (Orbit coverage).
The multi-angle patch system (`AtlasAddPatchView`) already solves this for *novel
AI views at other angles*; this solves it for the *same camera* by clean-plating
the layers behind foreground occluders. It needs **no angle calibration**, which
makes it strictly simpler and more reliable than the patch-view path.

## 2. Key architectural decision — reuse `ProjectionSource`, don't invent

The `ProjectionSource` dataclass (`atlas_camera/core/schema.py`) is *already* the
exact vehicle we need:

```python
@dataclass(slots=True)
class ProjectionSource:
    camera: LatentCamera          # for a clean-plate layer: the PRIMARY camera, un-orbited
    name: str = "patch"
    image_b64: str | None = None  # the inpainted clean plate for this layer
    proxy_geometry: list[AtlasProxyPrimitive] = ...  # this layer's own relief mesh (depth-band-clipped)
    priority: float = 0.0         # depth order: nearer band = higher priority
    metadata: dict = ...
```

And `LatentScene.projection_sources` is already serialized to the viewport in
`_extract_blockout_camera()` → the `projection_sources` payload, consumed by
`buildPatchSources()` in `atlas_blockout.js`, which already:

- builds each source's geometry with **its own projection material bound to that
  source's camera + image**, and
- orders overlapping sources by `priority` (`priorityToRenderOrder` /
  `priorityToOffsetUnits`).

**Crucial property this buys us:** each `ProjectionSource` textures *only its own
proxy_geometry*. So a background layer (its own back-band mesh + inpainted back
plate) and a foreground layer (its own front-band mesh + original pixels) never
fight over texels — each paints itself. From Camera View they reassemble exactly;
on orbit they separate in parallax. This is why the design is small: **we're
producing more `ProjectionSource`s, and the render path already exists.**

### The one real difference from patch views (requires a small JS branch)

Patch sources are *novel views from other angles*, so `buildPatchSources` discards
grazing fragments via a facing-ratio threshold (`uFacingThreshold ≈ 0.2`).
Clean-plate layers are **same-camera** plates — they must paint head-on *and*
grazing, exactly like the primary (which passes threshold `-1` = never discard).
So clean-plate sources need to be flagged and rendered with `facingThreshold = -1`,
relying on depth + priority (not facing angle) to order the layers. This is the
only non-additive change on the frontend. See §6.

## 3. External nodes we call (confirmed signatures)

We do **not** re-implement masking or inpainting. We orchestrate these in the graph
and consume their output. Confirmed node IDs / sockets:

### Acly `comfyui-inpaint-nodes` (modern `io.ComfyNode` API, category `inpaint`)

| node_id | inputs | outputs | role |
|---|---|---|---|
| `INPAINT_LoadInpaintModel` | `model_name` (combo from `models/inpaint/`) | `INPAINT_MODEL` | load LaMa/MAT |
| `INPAINT_InpaintWithModel` | `inpaint_model` (INPAINT_MODEL), `image` (IMAGE), `mask` (MASK), `seed` (INT), `optional_upscale_model?` | `IMAGE` (inpainted) | **the clean-plate generator (LaMa/MAT)** |
| `INPAINT_ExpandMask` | `mask` (MASK), `grow` (INT, def 16), `blur` (INT, def 7), `blur_type` | `MASK` | dilate the occluder silhouette |
| `INPAINT_MaskedFill` | `image`, `mask`, `fill` (`neutral`/`telea`/`navier-stokes`), `falloff` | `IMAGE` | cheap pre-fill / small gaps |

- LaMa internally resizes work to 256², MAT to 512² — **any input resolution is
  fine**; the node composites the fill back at native res via the mask.
- Model file: `big-lama.pt` → `ComfyUI/models/inpaint/`
  (`https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt`).
  MAT fp16 safetensors also supported.
- License is **GPL-3.0** → keep it a *separately installed node pack wired in the
  graph*. Do **not** import its code into `atlas_camera` or vendor it. Graph-level
  composition is not linking; our license is unaffected.

### `scraed/LanPaint` (optional generative tier, category `sampling`)

Drop-in KSampler replacement: **"LanPaint KSampler"** and **"LanPaint KSampler
(Advanced)"**. Same sockets as `KSampler` (model, positive, negative, latent_image,
seed, steps, cfg, sampler_name, scheduler) **plus a "steps of thinking" input**.
Works with any diffusion model (Flux, SDXL, SD3.5, Qwen, SD1.5) via a masked latent
(`Set Latent Noise Mask` / `InpaintModelConditioning`). Use this tier when LaMa
smears on a large/complex disocclusion. *(Confirm exact registered node id/class in
`LanPaint`'s node file during implementation — the README documents it by display
name only.)*

Both are **runtime graph dependencies**, not Python imports of Atlas. Add a short
"Optional inpaint integration" section to `INSTALL.md`.

## 4. New Atlas nodes

Two new nodes (plus one optional convenience). They follow the existing composable
one-job pattern (`AtlasDeriveWalls` / `AtlasDeriveReliefMesh`), take a shared
`ATLAS_DEPTH_MAP`, and use the same helpers already in `comfy/nodes.py`
(`_solve_camera_params`, `_depth_map_for_solve`, `_horizon_y_from_solve`,
`_image_tensor_to_pil`, `_save_image_tensor_to_tmp`).

### 4a. `AtlasDepthLayerMask` — one depth band → (layer_mask, inpaint_mask)

Composable: instantiate once per background layer. Emits the two masks the inpaint
graph needs.

```
INPUT_TYPES:
  required:
    solve  : ATLAS_SOLVE
    depth  : ATLAS_DEPTH_MAP
  optional:
    near_m       : FLOAT  (0 = auto: use percentile)         default 0.0
    far_m        : FLOAT  (0 = auto: +inf)                    default 0.0
    near_pct     : FLOAT  0..1 (used when near_m==0)          default 0.0
    far_pct      : FLOAT  0..1 (used when far_m==0)           default 0.5
    feather_px   : INT                                        default 4
RETURN_TYPES = ("MASK", "MASK")
RETURN_NAMES = ("layer_mask", "occlusion_mask")
```

- `layer_mask` = pixels whose metric depth ∈ [near, far] (this band's own pixels).
- `occlusion_mask` = pixels **nearer** than `near` (i.e. everything that occludes
  this band) — this is what must be inpainted in this band's plate. Feed it into
  `INPAINT_ExpandMask` (grow ~16–32) → `INPAINT_InpaintWithModel`.
- Metric depth via the same path `AtlasDeriveReliefMesh` uses: `_depth_map_for_solve`
  + the ground `scale` from `relief_mesh.estimate_ground_scale` (so bands are in real
  metres, consistent with geometry). Reuse `_solve_camera_params` /
  `_horizon_y_from_solve`.

### 4b. `AtlasCleanPlateLayer` — inpainted plate + band → append a `ProjectionSource`

The heart of it. Takes the clean plate produced by the external inpaint nodes and
turns one depth band into a layer.

```
INPUT_TYPES:
  required:
    solve        : ATLAS_SOLVE
    depth        : ATLAS_DEPTH_MAP
    plate_image  : IMAGE          # inpainted clean plate for THIS layer (from InpaintWithModel)
  optional:
    near_m/far_m/near_pct/far_pct : (same as 4a — MUST match the mask node's band)
    name         : STRING  default "layer"
    priority     : FLOAT   default 0.0   # set nearer bands higher
    relief_grid  : INT     default 128
    depth_edge_rel : FLOAT default 0.5
RETURN_TYPES = ("ATLAS_SOLVE",)
```

Behaviour (mirrors `AtlasAddPatchView`, minus the orbit):

1. `camera` = the **primary** camera unchanged (`solve.camera`) — same intrinsics,
   same extrinsics. No `orbit_camera`. This is the whole simplification.
2. Build this band's relief mesh from `depth`, **clipped to [near, far]** so
   out-of-band pixels become holes (see §5 — small `build_relief_mesh` addition).
   Reuse `estimate_ground_scale` + `build_relief_mesh` exactly as
   `AtlasDeriveReliefMesh` does.
3. Encode `plate_image` → JPEG data-URI (`image_b64`), same as `AtlasAddPatchView`.
4. Append `ProjectionSource(camera=primary, image_b64=plate, proxy_geometry=[mesh],
   priority=priority, metadata={"projection_mode": "clean_plate", "near_m":…,
   "far_m":…, "source": "inpaint_layer"})`.
5. Return a deep-copied solve with the source appended (`AtlasSolve.from_dict(
   solve.to_dict())` then `.projection_sources.append(...)`).

Chain one per layer (front→back or back→front, order doesn't matter; `priority`
decides overlap). The frontmost layer typically uses the **original photo**
(no inpaint needed) so nothing is lost; only the layers *behind* an occluder need a
clean plate.

### 4c. (optional) `AtlasDepthLayerSplit` — convenience, 3 common bands at once

Emits `fg_mask, mid_mask, bg_mask` + `mid_occ, bg_occ` in one node for the common
3-layer case, so the graph isn't three `AtlasDepthLayerMask`s. Pure convenience over
4a; skip for the first pass.

## 5. Core change — depth-band clip in `build_relief_mesh`

`atlas_camera/core/relief_mesh.py :: build_relief_mesh(...)` currently meshes the
whole depth map (tearing at discontinuities and sky). Add two optional params:

```python
def build_relief_mesh(depth, ..., band_min_m=None, band_max_m=None, ...):
    # after scaling depth to metres, treat pixels with scaled_depth < band_min_m
    # or > band_max_m as holes (same mechanism as the existing sky/edge tear:
    # exclude the quad, don't clamp), so a layer's mesh contains only its band.
```

This reuses the existing "tear into a hole" path — out-of-band pixels are just
another reason to drop a quad, identical to how `detect_sky_mask` already excludes
sky. Add a unit test in `tests/test_relief_mesh.py`: a synthetic 2-plane depth map,
assert a band clip yields only the near plane's vertices.

## 6. Frontend change — `atlas_blockout.js :: buildPatchSources`

One branch. When a serialized source has `metadata.projection_mode === "clean_plate"`
(surface it in the `_extract_blockout_camera` projection_sources dict — add
`"projection_mode": (src.metadata or {}).get("projection_mode")`), bind its
projection material with **`facingThreshold = -1`** (never discard on grazing angle,
same as the primary) instead of the patch default (~0.2). Everything else —
per-source camera/image material, `priority` → `renderOrder`/`polygonOffset` — is
unchanged. Verify live: from Camera View the layered plates must reassemble into the
original frame pixel-for-pixel; on orbit the background plate stays painted where the
foreground used to occlude.

No change needed to `serialize_proxy_geometry`, `computeGeometryPivot`, the 🎬
Backdrop toggle, or Camera Path — clean-plate sources are just more entries in the
existing `projection_sources` list.

## 7. Proof-of-concept workflow (2 layers)

Minimum viable graph to validate before formalizing. Node names are the real IDs.

```
Load Image
   │
   ├─► AtlasLearnedSolveFromImage ──► solve ─────────────────────────────┐
   │                                                                      │
   └─► AtlasDepthMap ──► depth ──┬───────────────────────────────────────┤
                                 │                                        │
   (BACKGROUND LAYER)            │                                        │
   solve,depth ─► AtlasDepthLayerMask(far_pct=1.0, near_pct=0.35)         │
                     │ occlusion_mask                                     │
                     └─► INPAINT_ExpandMask(grow=24)                      │
                              │                                           │
   Load Image (original) ─────┴─► INPAINT_InpaintWithModel ── plate_bg    │
        + INPAINT_LoadInpaintModel(big-lama.pt)     │                     │
                                                     ▼                     │
   solve,depth,plate_bg ─► AtlasCleanPlateLayer(name="bg", priority=0,    │
                                     far_pct=1.0, near_pct=0.35) ─► solve1─┤
                                                                          │
   (FOREGROUND LAYER — original pixels, no inpaint)                       │
   solve1,depth, <original image> ─► AtlasCleanPlateLayer(name="fg",      │
                                     priority=10, near_pct=0.0,           │
                                     far_pct=0.35) ─► solve2 ─────────────┘
                                                          │
                                          AtlasBlockoutViewport(solve2, source_image)
                                             → hit 📽 Project, then orbit / dolly
```

Success criterion: with 📽 Project on, dolly in — the gap the foreground uncovers
shows inpainted background, not black. Compare against the same graph without the bg
clean-plate layer (today's behaviour = black reveal).

For the hard-disocclusion variant, swap `INPAINT_InpaintWithModel` for a **LanPaint
KSampler** subgraph (VAE encode → Set Latent Noise Mask(expanded occlusion) →
LanPaint KSampler → VAE decode) and feed its decoded image in as `plate_bg`.

## 8. Implementation order for Claude Code

1. **Core:** add `band_min_m`/`band_max_m` to `build_relief_mesh` + unit test.
   (Smallest, fully testable in isolation, no Comfy/torch UI.)
2. **Node 4a `AtlasDepthLayerMask`:** metric-depth banding → two MASK outputs.
   Test with a synthetic solve + depth (see `tests/test_derive_geometry_nodes.py`
   for the fixture pattern). Assert band membership + occlusion = "nearer than near".
3. **Node 4b `AtlasCleanPlateLayer`:** append a `ProjectionSource` with
   `projection_mode="clean_plate"`. Test: given a solve + depth + dummy plate, output
   solve has one extra `projection_source`, camera equals primary (identity orbit),
   `proxy_geometry` non-empty, round-trips through `AtlasSolve.from_dict(to_dict())`.
4. **Serialization:** add `projection_mode` to the `projection_sources` dict in
   `_extract_blockout_camera`. (One line; assert present in a serialization test.)
5. **Frontend:** the `facingThreshold = -1` branch in `buildPatchSources`. Manual
   live verification (Camera View reassembly + orbit reveal) — this is the
   load-bearing check, do it on a real photo, not a synthetic scene.
6. **Register** all nodes in `NODE_CLASS_MAPPINGS` / display-name map in
   `comfy/nodes.py` (and `web` extension if any widget UX is added — none required).
7. **Docs:** node-catalog rows in `CLAUDE.md`, an entry in `docs/ECOSYSTEM_GUIDE.md`,
   an `examples/atlas_camera_inpaint_layers_workflow.json`, and the optional-deps note
   in `INSTALL.md`.
8. **Verification task (do not skip):** load the POC graph on 3–5 varied real photos
   (a person in front of a wall, a car on a street, a foreground tree). Confirm no
   NaN/Inf in the solve JSON, exactly one `ProjectionSource` per layer, and — the
   real test — that the dolly-in reveal is inpainted, not black. A subagent review of
   the frontend blending branch is warranted since it touches the shared projection
   render path.

## 9. Risks / calibration (be honest in the node tooltips)

- **Inpaint quality is the ceiling.** LaMa continues texture (walls, ground, foliage,
  sky) excellently; it smears on complex disocclusions (a face fully hidden behind a
  person). Route those layers through LanPaint/SDXL. Say so in the `AtlasCleanPlateLayer`
  docstring.
- **Band boundaries are only as good as monocular depth.** Auto-percentile bands are
  crude at edges; expose `near_m/far_m` for manual metric control, and allow feeding an
  external SAM mask (from `comfyui_segment_anything` / Impact-Pack) as the layer mask
  instead of a pure depth band — a natural follow-on (`AtlasCleanPlateLayer` could take
  an optional `layer_mask` IMAGE/MASK override).
- **Parallax budget.** This shines for *moderate* pushes; very large moves expose the
  flatness of each billboard-ish layer and the limits of the fill. That's expected —
  it's 2.5D, not full 3D reconstruction. Frame it that way to users.
- **Same known limitation as everything here:** frames pushed far outside the recovered
  camera's cone still hit the documented Orbit-coverage black regions; layers reduce the
  in-frame holes, they don't extend the cone.

## 10. What this deliberately does NOT do (scope guards)

- No new inpainting/segmentation code in `atlas_camera` — orchestrate the external
  packs. Keeps the core dependency-light and the GPL boundary clean.
- No change to `AtlasAddPatchView`, `AtlasMergeGeometry`, or `scene_type` presets —
  purely additive nodes, backward compatible.
- No exporter work yet (USD/Maya/Nuke per-layer plate export) — structurally similar
  follow-on, out of scope for the first pass, same as the ShotCam exporter deferral.
```
