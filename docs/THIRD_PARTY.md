# Third-Party Dependencies & License Boundaries

Atlas Camera itself is **MIT-licensed** (© 2026 Miike James Burns) and the core
package (`atlas_camera.core`) has **zero required runtime dependencies**. Every
model, vision, USD, UI, and research capability is an *optional* extra or an
*external* ComfyUI node pack, guarded by a `try/except` with an install hint.

**The boundary principle:** nothing below is vendored into this repo or linked
into the package. Python extras are ordinary optional dependencies; ComfyUI
node packs are combined only at the *graph* level (a workflow wiring nodes
together is composition, not linking); research models are **user-cloned**
upstream repos that Atlas points at via a path/env var. So a GPL or
non-commercial third-party piece does not change Atlas's own MIT terms — but if
*you* ship or sell work made with one, its terms apply to you. Check upstream
for authoritative license text; the notes below are a map, not legal advice.

## Optional Python extras (`pip install atlas-camera[...]`)

| Extra | Brings | License (upstream) | Notes |
|---|---|---|---|
| `[vision]` | numpy, opencv-python | BSD / Apache-2.0 | geometric solve |
| `[image]` | Pillow | HPND (permissive) | image I/O |
| `[ui]` | FastAPI, uvicorn, Pillow | MIT / BSD | optional workbench backend |
| `[usd]` | usd-core | Apache-2.0 (modified, Pixar) | USD export |
| `[neural]` | torch, GeoCalib, Depth-Anything-V2 (via transformers) | BSD-3 / Apache-2.0 / Apache-2.0 | **default** learned solve + depth; SegFormer (`AtlasSemanticMask`) rides transformers |
| `[sam3]` | transformers (SAM3 model classes) | Apache-2.0 (transformers); `facebook/sam3` weights **Meta SAM-License-1.0**, gated on Hugging Face | preferred sky/scope segmenter in `AtlasInput`'s cascade, no `triton`; commercial use permitted, military/ITAR use carved out — one-time `hf auth login` after requesting access, see INSTALL.md |
| `[moge]` | MoGe-2 (`Ruicheng/MoGe`) | **MIT** | interior-specialist depth |
| `[neural-da3]` | Depth Anything 3 | see upstream (GitHub-only) | selectable depth; default only on `experimental-da3-default` branch. `DA3NESTED-GIANT` weights are **CC BY-NC-ND (non-commercial)** |

Commercial-friendly by default: the shipping depth default (V2-Metric) and the
whole `[neural]` tier are permissive (Apache/BSD/MIT).

## ComfyUI node packs (external, graph-level — user-installed)

| Pack | Provides | License | Commercial note |
|---|---|---|---|
| [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | `SAM3Segment` (still used directly by `AtlasSegmentedSDXLInpaint` for per-instance separation; no longer part of `AtlasInput`'s own sky/scope cascade, which now prefers native `AtlasSAM3Mask`) | see upstream | needs `triton` (CUDA-only) — see INSTALL.md |
| [comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes) | LaMa / MAT clean plates | **GPL-3.0** | graph-level use only; never linked into Atlas |
| [LanPaint](https://github.com/scraed/LanPaint) | generative inpaint tier | see upstream | optional hard-disocclusion tier |
| ComfyUI-OCIO | `OCIORead` (ACEScg EXR) | see upstream | OCIO color-managed handoff |
| KJNodes, rgthree-comfy, ComfyUI-Custom-Scripts, VideoCombinePlus | rails / UI / video | see upstream | staged-master + dolly demos |

## Research / non-commercial tier (user-cloned, NOT vendored)

These are **not installed by Atlas** — you clone the upstream repo and point a
path/env var at it. They are gated behind `ATLAS_EXPERIMENTAL=1`.

| Model | Role | License | ⚠ Commercial |
|---|---|---|---|
| [LaRI](https://github.com/ruili3/lari) (`ruili3/LaRI` weights) | X-ray hidden geometry | **NO license upstream (all rights reserved)** | research/eval only until upstream licenses it |
| World Tracing (`haoz19/...` weights) | X-ray hidden geometry | checkpoint **CC BY-NC-ND 4.0** (HF-gated) | **non-commercial** |
| [NVIDIA Fixer](https://github.com/nv-tlabs/Fixer) | render repair | repo Apache-2.0; weights **NVIDIA Open Model License** | **commercial OK** |
| Qwen-Image-Edit-2511 + Multiple-Angles LoRA | multi-angle patches | see upstream | check the model + LoRA terms |
| `triton-windows` | enables SAM3 on Windows/NVIDIA | MIT | — |

## Bottom line for shippers

- **Atlas + the default pipeline** (MIT + Apache/BSD/MIT extras) — clean for
  commercial work.
- **Avoid for commercial output**: World Tracing (CC BY-NC-ND), LaRI (no
  license), `DA3NESTED-GIANT` (CC BY-NC-ND). These are experimental/eval tiers.
- **GPL (inpaint)** is graph-level composition, not linking — it doesn't
  relicense Atlas, but the LaMa/MAT node's own terms govern its use.
