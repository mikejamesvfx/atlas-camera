# Third-Party Notices & License Boundaries

Atlas Camera itself is **MIT** (© 2026 Miike James Burns — see LICENSE), and the
core package (`atlas_camera.core`) has **zero required runtime dependencies**.
Every model, vision, USD, UI and research capability is an *optional* extra or an
*external* ComfyUI node pack, guarded by a `try/except` with an install hint.

**The boundary principle:** nothing below is vendored into this repository or
linked into the package. Python extras are ordinary optional dependencies;
ComfyUI node packs are combined only at the *graph* level (a workflow wiring
nodes together is composition, not linking); research models are **user-cloned**
upstream repos that Atlas points at via a path or env var. So a GPL or
non-commercial third-party piece does not change Atlas's own MIT terms — but if
*you* ship or sell work made with one, its terms apply to you. Check upstream for
authoritative license text; the notes below are a map, not legal advice.

## Optional Python extras (`pip install atlas-camera[...]`)

| Extra | Brings | License (upstream) | Notes |
|---|---|---|---|
| `[vision]` | numpy, opencv-python | BSD / Apache-2.0 | geometric solve |
| `[image]` | Pillow | HPND (permissive) | image I/O |
| `[ui]` | FastAPI, uvicorn, Pillow | MIT / BSD | optional workbench backend |
| `[usd]` | usd-core | Apache-2.0 (modified, Pixar) | USD export |
| `[oiio]` | OpenImageIO | Apache-2.0 | float plate I/O, built-in ACES config |
| `[raw]` | rawpy, exifread, Pillow, opencv-python | MIT / BSD / HPND / Apache-2.0 | camera RAW decode + EXIF intrinsics |
| `[raw-lens]` | lensfunpy | LGPL-3.0 (lensfun database) | optional lens undistort |
| `[neural]` | torch, GeoCalib, Depth-Anything-V2 (via transformers) | BSD-3 / Apache-2.0 / Apache-2.0 | **default** learned solve + depth; SegFormer (`AtlasSemanticMask`) rides transformers |
| `[sam3]` | transformers (SAM3 model classes) | Apache-2.0 (transformers); `facebook/sam3` weights **Meta SAM-License-1.0**, gated on Hugging Face | preferred sky/scope segmenter in `AtlasInput`'s cascade, no `triton`; commercial use permitted, military/ITAR use carved out — one-time `hf auth login` after requesting access, see INSTALL.md |
| `[moge]` | MoGe-2 (`Ruicheng/MoGe`) | **MIT** | interior-specialist depth |
| `[neural-da3]` | Depth Anything 3 | see upstream (GitHub-only) | selectable depth; **never the default**. `DA3NESTED-GIANT` weights are **CC BY-NC-ND (non-commercial)** |
| `[record3d]` | Record3D `.r3d` import | see upstream | iPhone/iPad LiDAR capture (gated behind `ATLAS_IOS`) |
| `[mcp]` | mcp SDK | MIT | optional stdio MCP server |

Commercial-friendly by default: the shipping depth default (`V2-Metric-Outdoor`)
and the whole `[neural]` tier are permissive (Apache / BSD / MIT).

## ComfyUI node packs (external, graph-level — user-installed)

| Pack | Provides | License | Commercial note |
|---|---|---|---|
| [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | `SAM3Segment` (still used directly by `AtlasSegmentedSDXLInpaint` for per-instance separation; no longer part of `AtlasInput`'s own sky/scope cascade, which now prefers native `AtlasSAM3Mask`) | see upstream | needs `triton` (CUDA-only) — see INSTALL.md |
| [comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes) | LaMa / MAT clean plates | **GPL-3.0** | graph-level use only; never linked into Atlas |
| [LanPaint](https://github.com/scraed/LanPaint) | generative inpaint tier | see upstream | optional hard-disocclusion tier |
| ComfyUI-OCIO | Nuke-style OCIO nodes | see upstream | optional — Atlas owns its own float path via `[oiio]` |
| KJNodes, rgthree-comfy, ComfyUI-Custom-Scripts, VideoCombinePlus | rails / UI / video | see upstream | staged-master + dolly demos |
| Qwen-Image-Edit-2511 + Multiple-Angles LoRA | multi-angle patch generation | see model card | check the model + LoRA terms |
| big-lama.pt weights | LaMa inpaint weights | Apache-2.0 | ✅ |
| three.js r185 | Blockout viewport | MIT | **vendored** (`atlas_camera/comfy/web/lib/atlas-three.bundle.js`, built from `ui/`'s pinned dependency) |

## Research / non-commercial tier (user-cloned, NOT vendored)

These are **not installed by Atlas** — you clone the upstream repo and point a
path or env var at it. They are gated behind `ATLAS_EXPERIMENTAL=1`.

| Model | Role | License | ⚠ Commercial |
|---|---|---|---|
| [LaRI](https://github.com/ruili3/lari) (`ruili3/LaRI` weights) | X-ray hidden geometry | **NO license upstream (all rights reserved)** | research/eval only until upstream licenses it |
| World Tracing (`haoz19/...` weights) | X-ray hidden geometry | checkpoint **CC BY-NC-ND 4.0** (HF-gated) | **non-commercial** |
| [NVIDIA Fixer](https://github.com/nv-tlabs/Fixer) | render repair | repo Apache-2.0; weights **NVIDIA Open Model License** | commercial OK |
| `triton-windows` | enables `SAM3Segment` on Windows/NVIDIA | MIT | — |

### The two research-only backends, stated plainly

The **hidden-geometry track was removed before beta 0.8.**
`AtlasPredictHiddenGeometry` is no longer registered in any tier, so neither
backend below is reachable from ComfyUI. Their helper modules remain in source
(`inference/lari_hidden_geometry.py`, `inference/wt_hidden_geometry.py`) and
neither ever vendored upstream code or weights, so nothing restrictive was or is
redistributed. Recorded here because the constraints apply again the moment the
node is re-registered:

- **LaRI** ships with no license file, which legally defaults to
  all-rights-reserved — stricter than any non-commercial license. Atlas never
  vendors or redistributes its code or weights; the node requires the user's own
  clone and warns in its report output. If the track matures, the right move is
  asking the authors for a license.
- **World Tracing**'s scene checkpoint is gated on HuggingFace and licensed
  CC BY-NC-ND 4.0: non-commercial, no derivatives of the weights. The same
  user-clone pattern applies.

## The GPL boundary

Masking and inpainting are never implemented inside `atlas_camera`. GPL-licensed
ComfyUI packs (comfyui-inpaint-nodes) participate only as **separate nodes wired
into a graph** — graph-level composition, not linking — so Atlas's MIT license is
unaffected. This boundary is deliberate and documented in INSTALL.md's "Optional
Inpaint Integration"; keep it. Any future inpaint capability belongs in the
graph, not in this package.

## Weights are not code

Model weights downloaded at runtime (HuggingFace, pack model folders) are
governed by their own model cards and licenses regardless of the wrapper code's
license. When in doubt about a deployment, check the **weights'** terms — the
tables above list them where known, but model cards change and the card is
authoritative.

## Bottom line for shippers

- **Atlas + the default pipeline** (MIT + Apache/BSD/MIT extras) — clean for
  commercial work.
- **Avoid for commercial output**: World Tracing (CC BY-NC-ND), LaRI (no
  license), `DA3NESTED-GIANT` (CC BY-NC-ND), and Depth Anything V2's **large**
  weights (CC BY-NC 4.0 — small/base are Apache 2.0). These are
  experimental/eval tiers.
- **GPL (inpaint)** is graph-level composition, not linking — it does not
  relicense Atlas, but the LaMa/MAT node's own terms govern its use.
