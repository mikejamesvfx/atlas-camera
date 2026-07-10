# Installing Atlas Camera

Atlas Camera starts with a low-dependency Python core.

## Development Install

```powershell
cd C:\Users\miike\Documents\AtlasCamera
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

## Optional Image Metadata Support

Install Pillow when you want `solve_still_image()` to infer image size directly
from image files:

```powershell
pip install -e ".[image]"
```

Without Pillow, pass `image_size=(width, height)`.

## Optional Vision Solver Support

Install NumPy and OpenCV when you want automatic line detection, vanishing-point
solving, and debug overlays:

```powershell
pip install -e ".[vision]"
```

The development extra includes these dependencies for the test suite:

```powershell
pip install -e ".[dev]"
```

## Optional Local UI

The UI backend is optional and keeps FastAPI out of the core runtime install:

```powershell
pip install -e ".[ui,vision]"
python -m atlas_camera.ui
```

This starts the local FastAPI service for projects, image files, constraints,
solves, local model guidance, and review-package export. The React workbench is
the artist-facing surface for 2D guides and the 3D lineup viewport.

If port `8787` is already occupied, either stop the existing backend or choose a
different port:

```powershell
python -m atlas_camera.ui --port 8788
```

On Windows, inspect the process using the default UI port with:

```powershell
Get-NetTCPConnection -LocalPort 8787 | Select-Object LocalAddress,LocalPort,State,OwningProcess
```

For frontend development, run the Vite workbench separately:

```powershell
cd ui
npm install
npm run dev
```

The frontend uses React, Vite, lucide icons, and Three.js. The 3D viewport is a
local browser feature only; it does not add Three.js or WebGL dependencies to
the Python core package.

Run an end-to-end solve package:

```powershell
python tools\solve_image.py --image path\to\concept.png --output-dir review_packages
```

Run an artist-guided constraints package:

```powershell
python tools\solve_constraints.py --image path\to\concept.png --constraints path\to\constraints.json --output-dir review_packages
```

## Optional USD Support

USD import/export is lazy. Importing Atlas Camera does not require USD.

```powershell
pip install -e ".[usd]"
```

If `usd-core` is not installed, requesting USD export or import raises a clear
runtime error.

## Optional Depth Anything 3 Backend

Depth Anything 3 (DA3) is a second depth backend selected per node via the
`depth_model` combo (`depth-anything/DA3METRIC-LARGE`, `DA3MONO-LARGE`,
`DA3NESTED-GIANT-LARGE-1.1`). **Since 2026-07-09, `DA3METRIC-LARGE` is the
default for newly added nodes** (measured A/B: ~3× fewer relief-mesh tears,
much stronger metric ground fits — see `docs/dev/da3_backend_test_plan.md`).
Existing saved workflows keep their stored V2 values, and every V2 model
remains in the combo. Without the `[neural-da3]` extra installed, a node left
on the DA3 default fails with an informative install hint — switch it to a V2
model or install the extra below. `DA3METRIC-LARGE` converts canonical depth to
metres using the *solved* focal length when the node has one (`focal_source:
"solve"` in the depth summary); on image-only nodes it falls back to an
assumed normal-lens focal (`"assumed"` — the metric model is a depth-only
head and predicts no camera; downstream ground-pinning re-normalizes the
scale anyway). Note `DA3NESTED-GIANT-LARGE-1.1` is licensed CC BY-NC 4.0
(non-commercial).

Into a fresh/dedicated venv the extra works directly:

```powershell
pip install -e ".[neural-da3]"
```

**Into an existing ComfyUI venv, do NOT install with dependencies.** The
`depth-anything-3` package hard-requires `xformers` and `numpy<2` and pins
`moviepy==1.0.3` — a full dependency install can downgrade or clobber ComfyUI's
torch/numpy. Install the package alone and let it use what ComfyUI already has:

```powershell
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
# Then verify the import and that torch/numpy were untouched:
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -c "import torch, numpy; print(torch.__version__, numpy.__version__); from depth_anything_3.api import DepthAnything3; print('DA3 OK')"
```

If the import fails on a missing small dependency (e.g. `einops`, `omegaconf`,
`safetensors`), install just that package — never the full dependency set.
Model weights download from Hugging Face on first use. A GPU (cuda) is
recommended: DA3's inference autocasts to bf16/fp16 by device type.

### Second backend: World Tracing (diffusion, also research-only)

The node's `model` combo can select `world-tracing-scene` — WT-DiT's r69l
scene model (840×840, 6 layers, ~17s at 20 diffusion steps; generative, so pin
the `seed` widget for reproducibility). Setup mirrors LaRI:

```powershell
git clone https://github.com/haoz19/world-tracing.git C:\path\to\world-tracing
```

then set the node's `wt_path` widget (or `ATLAS_WT_PATH`). Two extra
requirements beyond LaRI's: the small `jaxtyping` package
(`pip install --no-deps jaxtyping` into the ComfyUI venv), and the checkpoint
is **HF-gated** — request access on the `haoz19` model pages, then
authenticate the machine once (`huggingface-cli login` / `hf auth login` with
a read token) so the first-use download works. License: **CC BY-NC-ND 4.0**
(non-commercial research, no redistributed derivatives).

For the V2-vs-DA3 accuracy comparison protocol, see
`docs/dev/da3_backend_test_plan.md` and `tools/compare_depth_backends.py`.

## Experimental: Hidden-Geometry Prediction (research-only)
> **Gated node:** this node registers only when `ATLAS_EXPERIMENTAL=1` is set before launching ComfyUI (the `experimental` branch enables it by default; `main` hides it so the node menu stays universal).


`AtlasPredictHiddenGeometry` 🔬 predicts the surfaces hidden behind foreground
occluders (LaRI layered ray intersections) and outputs an "X-ray" copy of an
`ATLAS_DEPTH_MAP` with occluders replaced by predicted hidden depth — wire it
into background band layers so disocclusion reveals get predicted geometry
instead of diffusion-smoothed guesses. Best on indoor/architectural scenes;
see `docs/dev/hidden_geometry_training_free_research.md` for measured limits.

**The upstream LaRI repository has NO license (all rights reserved) — research
use only, and atlas_camera bundles none of it.** You must clone it yourself:

```powershell
git clone https://github.com/ruili3/lari.git C:\path\to\lari
```

then set the node's `lari_path` widget (or the `ATLAS_LARI_PATH` env var) to
that folder. Inference needs only the `[neural]` extra + CUDA — **no
PyTorch3D** (that's only in LaRI's dataset tooling) and none of LaRI's pinned
requirements. Weights (~1.3GB) download from HuggingFace (`ruili3/LaRI`) on
first use. Without a clone the node fails with these instructions.

## Experimental: Fixer Render Repair (Docker)
> **Gated node:** this node registers only when `ATLAS_EXPERIMENTAL=1` is set before launching ComfyUI (the `experimental` branch enables it by default; `main` hides it so the node menu stays universal).


`AtlasRenderFix` 🔬 repairs projected-render artifacts (torn silhouettes,
stretched texels, hard tear-holes) in an IMAGE batch with NVIDIA **Fixer**
(the Difix3D+ successor, single-step diffusion) — typically wired between
`AtlasBlockoutViewport`'s baked `path_frames` and a Video Combine node.
Licensing is friendlier than the hidden-geometry track: the Fixer repo is
Apache-2.0 and the `nvidia/Fixer` weights ship under the NVIDIA Open Model
License (commercial use permitted).

**Inference runs in Docker** — Fixer's `cosmos-predict2`/`transformer_engine`
stack has no native Windows build, so this node shells out to a container
instead of importing torch in-process. Three setup steps, each once:

```powershell
# 1. Clone Fixer and download its weights (~5.2GB, ungated)
git clone https://github.com/nv-tlabs/Fixer.git C:\path\to\Fixer
cd C:\path\to\Fixer
hf download nvidia/Fixer --local-dir models

# 2. Build the inference image (public NGC PyTorch base, ~35GB; the official
#    cosmos container is auth-locked on nvcr.io, this recipe reproduces it)
docker build -t fixer-spike-env -f docker/fixer/Dockerfile docker/fixer/

# 3. Point the node at the clone
#    (fixer_path widget, or the ATLAS_FIXER_PATH env var)
```

Docker Desktop with GPU support (WSL2 backend) must be running when the node
executes. Budget ~1 minute model load/warmup per queue plus ~0.5 s/frame;
frames near Fixer's native 576×1024 round-trip with the least softening.
Known limits (spike-measured): mild overall softening, and large frame-edge
reveals are not outpainted. Without docker/image/weights the node fails with
these instructions.

## Optional Inpaint Integration

The inpaint-layers feature (`AtlasDepthLayerMask` + `AtlasCleanPlateLayer`,
2.5D clean-plate parallax) orchestrates external ComfyUI node packs rather
than reimplementing masking/inpainting inside `atlas_camera` — this keeps the
core package dependency-light and avoids pulling a GPL-3.0 license into this
codebase. Both packs below are **runtime graph dependencies only** — installed
as ComfyUI custom nodes, never imported by Atlas's Python.

**Required for the clean-plate tier — `Acly/comfyui-inpaint-nodes` (GPL-3.0):**

```powershell
cd <COMFYUI_ROOT>\custom_nodes
git clone https://github.com/Acly/comfyui-inpaint-nodes.git
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install -r comfyui-inpaint-nodes\requirements.txt
```

Download the LaMa model (or a MAT fp16 safetensors) into ComfyUI's inpaint
model directory:

```powershell
New-Item -ItemType Directory -Force "<COMFYUI_ROOT>\models\inpaint"
# Download big-lama.pt into that directory from:
# https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
```

Restart ComfyUI. Wire `INPAINT_LoadInpaintModel` (loads `big-lama.pt`) →
`INPAINT_ExpandMask` (dilate `AtlasDepthLayerMask`'s `occlusion_mask`, grow
~16-32) → `INPAINT_InpaintWithModel` (image + expanded mask → clean plate) →
`AtlasCleanPlateLayer`'s `plate_image` input.

**Optional generative tier for hard disocclusions — `scraed/LanPaint`:**

```powershell
cd <COMFYUI_ROOT>\custom_nodes
git clone https://github.com/scraed/LanPaint.git
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install -r LanPaint\requirements.txt
```

LaMa/MAT continue texture (walls, ground, foliage, sky) excellently but smear
on complex disocclusions — e.g. a face fully hidden behind a person. LanPaint
is a drop-in KSampler replacement ("LanPaint KSampler" / "LanPaint KSampler
(Advanced)") that works with any diffusion model (Flux, SDXL, SD3.5, Qwen,
SD1.5) via a masked latent (`Set Latent Noise Mask` / `InpaintModelConditioning`)
— route the harder layers through a VAE encode → masked latent → LanPaint
KSampler → VAE decode subgraph instead, and feed its output into
`AtlasCleanPlateLayer` as `plate_image`.


## Optional Master-Workflow Integrations (2026-07-08)

The hero workflow `examples/atlas_camera_master_dmp_workflow.json` uses three
optional external pieces — each fails soft or has a documented placeholder:

- **Sky segmentation** — [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)
  provides the `SAM3Segment` node (prompt it with `sky`). Its MASK output
  feeds `AtlasSkyDomeLayer.sky_mask` AND every layer node's `exclude_mask`
  (a real segmentation replaces Atlas's internal sky heuristic).
- **VLM pre-flight** — `AtlasAssessImage` talks to a local vision-language
  server: Ollama (`ollama run gemma3:4b`, default `http://127.0.0.1:11434`),
  LM Studio (default `http://127.0.0.1:1234/v1`), or llama.cpp
  (`http://127.0.0.1:8080/v1`). No server running → the node reports how to
  start one and ▶ Continue Workflow still works. The report displays on the
  node itself; the optional Show Text wiring uses
  [pythongosssss custom-scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts).
- **Multi-angle patch generation** — the embedded Qwen Image Edit 2511
  subgraph needs `qwen_image_edit_2511_fp8_e4m3fn.safetensors`,
  `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors`, the
  Lightning 4-step LoRA, and the `qwen-image-edit-2511-multiple-angles-lora`
  (see `examples/atlas_qwen_image_edit_2511_multiangle_camera.json`). The
  branch stays paused (ExecutionBlocker) until 📐 Extract Angle runs, so the
  rest of the workflow works without these models installed.

## ComfyUI Adapter

The `atlas_camera.comfy` package is scaffolded, but this first pass does not yet
install Atlas Camera as a complete ComfyUI custom node package. The nodes wrap
core functions and avoid making ComfyUI a core dependency.

## DCC Adapters

Maya, Blender, and Nuke integrations are script writers. Run the generated
scripts inside each DCC application. Maya is the most concrete first-pass
handoff; Blender and Nuke are placeholders for future production exporters.
