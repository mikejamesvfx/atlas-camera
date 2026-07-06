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

## ComfyUI Adapter

The `atlas_camera.comfy` package is scaffolded, but this first pass does not yet
install Atlas Camera as a complete ComfyUI custom node package. The nodes wrap
core functions and avoid making ComfyUI a core dependency.

## DCC Adapters

Maya, Blender, and Nuke integrations are script writers. Run the generated
scripts inside each DCC application. Maya is the most concrete first-pass
handoff; Blender and Nuke are placeholders for future production exporters.
