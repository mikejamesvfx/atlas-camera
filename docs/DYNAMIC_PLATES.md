# Atlas Dynamic Plates v0.1

**Atlas owns the shot. The video model makes the water move.**

A `DynamicPlate` represents one time-varying region of a solved still (ocean,
clouds, smoke, ...) as: a crop-adjusted camera derived from the solve + an
artist/auto matte + known receiver geometry + an optional generated frame
sequence. The temporal sequence is a texture projected through the FIXED crop
camera onto the receiver; the artist's render camera moves independently. The
camera move is **never** baked into the generated video.

## Workflow

```text
source still ─ Atlas solve ─ matte (artist or SAM3 assist)
    → matte bbox → overscan ROI → crop camera (fx'=fx, cx'=cx-roi.x, ...)
    → receiver plane (ROI rays ∩ world plane)
    → artifact package (dynamic/WATER_0001/...)
    → temporal generator (LTX via ComfyUI, optional)
    → Blender scene: projection camera (fixed) + render camera (free)
```

## CLI

```bash
python -m atlas_camera.dynamic create \
    --image castle.png --matte ocean_mask.png --type water \
    --solve atlas_solve.json --out shots/castle \
    --generator ltx --template atlas_ltx25_frames_template.json \
    --frames 97 --fps 24 --seed 42 --blender
python -m atlas_camera.dynamic validate --package shots/castle/dynamic/WATER_0001
```

- No `--solve` → `atlas.recover` runs (`--method vanishing_points|learned`).
- `--auto-matte "ocean, sea water"` uses the SAM3 assist when the `[sam3]`
  stack is present; the artist `--matte` path never depends on it.
- Generator missing/unreachable → the package still builds and reports
  `generator status = not_available`. Exit code stays 0.

## Viewport — seeing the plate in ComfyUI

The CLI above is the *producer*; **`Atlas Load Dynamic Plate` 🌊**
(`AtlasLoadDynamicPlate`, `nodes_dynamic.py`) is the *consumer*. Give it the
solve and the package directory the CLI wrote and it appends the receiver plane
plus the temporal projection to the solve as a `ProjectionSource`: the plate's
FIXED crop camera stays the projector while the viewport camera moves freely.
Generated frames stream to the frontend per-frame and play on the render ticker;
with no generated frames yet it projects the still crop.

```text
python -m atlas_camera.dynamic create ...   →  dynamic/WATER_0001/
                                            →  AtlasLoadDynamicPlate → Atlas Viewport 🧊
```

The node is in the **standard** tier as of 2026-08-14 — no `ATLAS_EXPERIMENTAL`
flag is needed. Full input/output row in
[docs/NODE_CATALOG.md](NODE_CATALOG.md).

## Package layout

```text
dynamic/WATER_0001/
├── manifest.json            # DynamicPlate.to_dict() — the contract
├── source/{crop,matte,context}.png
├── camera/{source_camera,crop_camera}.json
├── geometry/receiver.obj
├── generated/frame_0000.png ...   # AUTHORITATIVE output
└── preview/                       # derivative only
```

## Shipped v0.1 decisions and limits

- **Input modes: image-to-video AND video-to-video.** `--mode v2v` renders
  the crop along a world-space dolly FIRST (`core/dynamic_plate_render.py`:
  per-frame plane homography `H_crop @ inv(H_render)`, verified against the
  ray-chain ground truth), writes `rendered/frame_*.png`, and the adapter
  encodes them to MP4 (PyAV, optional) for LoadVideo-style templates — the
  input video already carries the geometrically correct camera motion, so
  the generator only adds surface motion (`camera_preservation =
  "atlas_rendered_v2v"`). Caveat: the plane homography is exact only for
  pixels ON the receiver; non-plane content inside the crop (foreground
  rocks/architecture) warps approximately — the matte confines generation to
  the water, and the DCC composite uses static projection for everything
  else. Disoccluded frame edges fall back to the still crop.
- **Receiver: horizontal plane** at configurable height. Fine for distant
  ocean + moderate moves; no spectral ocean / FLIP / displacement — temporal
  appearance only.
- **Camera preservation is unverified for i2v**: results carry
  `metadata.camera_preservation = "unverified_i2v"`; the prompt preset locks
  the camera, but a generator can still drift. Atlas registration is
  unaffected — the plate projects through the solved crop camera regardless.
- **Occlusion** comes from the reconstructed static scene via normal depth
  testing in the DCC (the receiver sits behind foreground geometry); the 2D
  matte only controls where content is generated.
- **Color**: generated frames are treated as sRGB; `color_metadata` records
  every hop. Nothing is labeled scene-linear unless it actually is.
- Registered node packs, ComfyUI availability, and template node types are
  probed at runtime (`LTXComfyGenerator.available()`); Atlas imports never
  require torch/ComfyUI (`tests/test_dynamic_generators.py` pins this).

## LTX template contract

The adapter is template-driven (`--template` / `ATLAS_LTX_TEMPLATE`): any
ComfyUI workflow JSON (UI or API format) that ends in **SaveImage** frames.
Overridden inputs: `LoadImage.image` (the crop), `{PROMPT}` marker anywhere
(else first `CLIPTextEncode`), `seed/noise_seed`, `length/frames/num_frames`,
`fps/frame_rate`, `width/height` (default: the crop raster — pick overscan so
the ROI lands on the model's multiple-of-32 grid; LTX also wants
`frames % 8 == 1`). Site knobs: `config.extra["overrides"]`
(`{"<nodeId>.<input>": value}`).

## Registration guarantee (release gate)

`tests/test_dynamic_plate_receiver.py` verifies numerically that
full-image pixel → crop pixel → crop-camera ray → receiver intersection
equals the original camera's ray intersection (1e-6), including crop+resize.

## v0.3 additions

**Occlusion inpaint (Track A)** — `python -m atlas_camera.dynamic occlusion-fill
--solve scene_solve.json --image plate.png --orbit "12,0,1" --frames 49
--generator ltx --template atlas_ltx25_inpaint_v2v.json --out pkg`
renders the solved scene along the orbit with LTX's chroma-green inpaint
sentinel (same rasterizer as `AtlasDisocclusionGuide`), writes `guide/` +
`mask/` sequences, and runs the LTX-2.5 inpaint IC-LoRA
(`ltx23\ltx-2.3-22b-ic-lora-in-outpainting-0.9`) via the `{GUIDE_VIDEO}` /
`{MASK_VIDEO}` upload markers. `patch_exact.txt` re-enters the peak-
disocclusion frame through `AtlasAddPatchView(exact_view_override=...)`.
`--exr` wraps filled frames as 32f EXR (display-referred container — NOT
scene-linear; LTX-2.5 has no HDR path, the HDR IC-LoRA is 2.3-only).

**Dynamic objects (Track B)** — `create --type actor --card "px,py,dist,width"`
places a camera-facing billboard card at `dist` metres along the anchor
pixel's ray (`build_receiver_card`); `generate --matte-mode chroma` keys the
`ACTOR_PROMPT_DEFAULT` backdrop into `generated/matte_*.png`. The viewport
node streams mattes alongside frames (`?matte=1` on the frame route,
`uMatte` swapped in the same tick); `mask_b64` carries matte 0 for the
headless renderer and exports.
