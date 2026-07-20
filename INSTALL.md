# Installing Atlas Camera

Atlas Camera starts with a low-dependency Python core.

## Development Install

```powershell
cd <REPO_ROOT>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

## ComfyUI Node Pack Install

**Clone-and-go (simplest — no pip install):** clone the repository straight
into ComfyUI's `custom_nodes` and restart. The repo-root `__init__.py` puts
the checkout on `sys.path` and registers all Atlas nodes; the example
workflows, proxy meshes, and frontend all work from the checkout.

```powershell
cd <COMFYUI_ROOT>\custom_nodes
git clone https://github.com/mikejamesvfx/atlas-camera.git
```

No requirements step is needed for the core nodes — ComfyUI already ships
numpy/Pillow/torch. The `[neural]` features (learned solve, depth, derive
nodes) additionally need GeoCalib in ComfyUI's Python. Pick the recipe that
matches your ComfyUI install:

**Standard `venv` install** - pip resolves GeoCalib's dependency tree normally:

```powershell
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install "git+https://github.com/cvg/GeoCalib.git" transformers
```

**Portable ComfyUI (`python_embeded`) - `--no-deps`, protect the CUDA stack.**
The embedded build ships torch/torchvision/numpy compiled against a specific
ABI, and the standard command above lets pip re-resolve GeoCalib's (unpinned)
dependency tree, which can pull a numpy or torch wheel that clobbers that ABI
and breaks torch or other custom nodes (the same hazard as the DA3 `--no-deps`
note below). GeoCalib itself is pure Python (`py3-none-any`), so install it with
no deps and add its runtime deps separately:

```powershell
# GeoCalib (pure Python) - no deps, so pip can't touch your torch/numpy:
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install --no-deps "git+https://github.com/cvg/GeoCalib.git"
# GeoCalib eager-imports cv2 + kornia at load; transformers powers depth.
# Install these normally - their only shared dep, numpy, is already present and
# satisfied (left untouched), and `kornia` pulls its required `kornia-rs`:
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install opencv-python kornia transformers
```

> `--no-deps` on GeoCalib **alone is not enough** - it imports `cv2` and `kornia`
> at module load and fails with `ModuleNotFoundError: No module named 'cv2'`
> without the second line. GeoCalib also declares `matplotlib`, but that is used
> only by its visualization helpers, never the camera solve, so it is omitted
> here. Restart ComfyUI after installing so the embedded interpreter picks up
> the new packages.

**Development install (editable + symlink):** keeps the checkout wherever you
work on it; Python changes are live without reinstalling. See CLAUDE.md's
Commands section — editable-install `atlas_camera` into ComfyUI's venv and
symlink `custom_nodes\AtlasCamera` at `atlas_camera\comfy`. Don't combine
both routes in one ComfyUI install (the nodes would register twice).

## Optional Image Metadata Support

Install Pillow when you want `solve_still_image()` to infer image size directly
from image files:

```powershell
pip install -e ".[image]"
```

Without Pillow, pass `image_size=(width, height)`.

## Optional Camera RAW Input (NEF / CR2 / CR3 / RAF / ARW)

Install the `[raw]` extra for the `AtlasLoadRAW` 📷 node — native camera-RAW
decode (rawpy/libraw, incl. Canon CR3 and Fuji X-Trans), EXIF focal-length +
camera-model→sensor-size metadata feeding the solve's intrinsics, and a
scene-linear EXR sidecar for the Output Desk / OCIO path:

```powershell
pip install -e ".[raw]"

# Optional: lensfun geometry correction for the node's undistort toggle.
# A separate extra because lensfunpy wheels can lag new Python/Windows
# releases — RAW decode and metadata work fine without it.
pip install -e ".[raw-lens]"
```

Notes:

- **EXR sidecar codec:** writing the linear `.exr` uses OpenCV's OpenEXR
  codec, which requires **opencv-python 4.x** (the 5.x wheels dropped the
  codec — `[raw]` pins `<5`) and the environment variable
  `OPENCV_IO_ENABLE_OPENEXR=1` set **before Python starts** (put it in your
  ComfyUI launch `.bat`, same requirement as the ComfyUI-OCIO workflow). If
  the write fails, the node degrades gracefully: the plate_ref is marked
  proxy and the report says exactly what to fix.
- **Colorspace:** the sidecar is scene-linear with **sRGB/Rec.709 primaries**
  (tagged `Linear Rec.709 (sRGB)`), *not* ACEScg — convert downstream with
  OCIO nodes.
- **CR3 metadata is best-effort** (pure-Python readers; no ExifTool
  dependency): when EXIF can't be read, the report says so and you can type
  focal/sensor into the solve node manually.
- **Fuji RAF:** decode works, but lensfun profile coverage for X-mount is
  thin (Fuji applies corrections in-body) — the undistort step will usually
  report `no_profile_lens` and pass pixels through unchanged.

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
`DA3NESTED-GIANT-LARGE-1.1`). It was briefly the default (2026-07-09) but on
**2026-07-13 the `main` default reverted to `V2-Metric-Outdoor`** (a 4-scene
A/B found V2 best-or-tied on exteriors, and V2 needs no extra install — see
`docs/dev/archive/da3_backend_test_plan.md`). **DA3 is now a selectable choice, and the
default only on the `experimental-da3-default` branch.** Every V2 model remains
in the combo. Without the `[neural-da3]` extra installed, selecting a DA3 model
fails with an informative install hint — switch it to a V2 model or install the
extra below. `DA3METRIC-LARGE` converts canonical depth to
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

**Into an existing ComfyUI install, do NOT install with dependencies.** The
`depth-anything-3` package declares `xformers`, `numpy<2`, `moviepy==1.0.3`, and a
full gaussian-splat/COLMAP export stack (`gsplat`, `open3d`, `pycolmap`, `trimesh`,
...) - a normal install can downgrade or clobber ComfyUI's torch/numpy, and several
of those export deps have no wheels for recent torch/Python on Windows. None are
used by depth inference, so install the package alone: Atlas auto-stubs the export
stack at import time (`depth_estimator._install_da3_export_stubs`), leaving you only
DA3 itself plus a few small pure-Python deps.

Standard `venv`:

```powershell
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
& "<COMFYUI_ROOT>\venv\Scripts\python.exe" -m pip install omegaconf einops addict "moviepy==1.0.3"
```

Portable ComfyUI (`python_embeded`) - same idea, but DA3's hatchling build backend
and a too-tight `requires-python` upper bound (`<=3.13`, which pip reads as excluding
3.13.1+) need two extra flags:

```powershell
# DA3 builds with hatchling; install it, then build without isolation and ignore the
# over-strict requires-python pin (3.13.x is fine):
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install hatchling
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install --no-deps --no-build-isolation --ignore-requires-python "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install omegaconf einops addict "moviepy==1.0.3"
# Verify torch/numpy were untouched:
& "<COMFYUI_ROOT>\python_embeded\python.exe" -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

Notes: `moviepy` MUST be `<2` (DA3 imports the removed `moviepy.editor`); DA3's
`numpy<2` pin is conservative - inference is verified working on numpy 2.x. `xformers`
is left absent on purpose so DINOv2 falls back to standard attention (a MagicMock stub
would flip its availability flag and corrupt the forward pass). Do NOT install the
heavy export deps (`gsplat`/`open3d`/`pycolmap`/...) - Atlas stubs them. Model weights
download from Hugging Face on first use. A GPU (cuda) is recommended: DA3's inference
autocasts to bf16/fp16 by device type.

## Optional MoGe-2 Depth Backend

MoGe-2 (Microsoft, **MIT-licensed**) is a light-dependency alternative to DA3 in
the `depth_model` combo (`Ruicheng/moge-2-vitl-normal`, `Ruicheng/moge-2-vitb-normal`).
It predicts metric depth PLUS per-pixel normals, and Atlas feeds it the solved
focal as `fov_x` so its geometry lands in the recovered camera's frame. Unlike
DA3 it needs no export-only stubbing (no gsplat/open3d/pycolmap stack) and no
non-commercial license caveat.

Into a fresh/dedicated venv:

```powershell
pip install -e ".[moge]"
```

Into an existing ComfyUI venv install `--no-deps` (moge's own list pulls
gradio/matplotlib/pipeline); torch/torchvision/opencv/scipy are already present:

```powershell
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install --no-deps "git+https://github.com/microsoft/MoGe.git"
& "<COMFYUI_ROOT>\python_embeded\python.exe" -m pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
# Verify:
& "<COMFYUI_ROOT>\python_embeded\python.exe" -c "from moge.model.v2 import MoGeModel; print('MoGe OK')"
```

Weights download from Hugging Face on first use. If the import fails on a missing
small dependency (e.g. `scipy`), install just that package. A GPU is recommended;
`infer()` autocasts to fp16 by default.

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
`docs/dev/archive/da3_backend_test_plan.md` and `tools/compare_depth_backends.py`.

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

### Geometry beneath a removed subject

The cleanplate image and the geometry supporting it are separate decisions.
For a removed car, castle, person, or other foreground object whose contact
surface must remain continuous during an orbit, run a second `AtlasDepthMap`
on the **approved full-frame cleanplate** and feed that depth to a full-range
background `AtlasCleanPlateLayer` (`band_side=manual`, `near_pct=0`,
`far_pct=0`, `fill_occluded=false`). Keep the original image depth only for a
SAM/artist-matted foreground layer. This is the pattern used by the canonical
OCIO/DCC workflows in `examples/showcase/`.

Do not use a far `AtlasBoundedBand` plus `fill_occluded` as the support surface
for a large removal. That mode interpolates depth from the surrounding band's
boundary; it is useful for narrow disocclusion slivers, but on a road or
headland it can place the filled footprint at the far cutoff and produce the
visible vertical cliff/floating-object failure. `AtlasBoundedBand` remains the
right tool for limiting a foreground relief that otherwise extrudes too far.

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

The hero workflow `examples/atlas_camera_staged_master_workflow.json` uses three
optional external pieces — each fails soft or has a documented placeholder:

- **Sky / scope segmentation** — `AtlasSAM3Mask` (this package's own node,
  `[sam3]` extra) is the preferred segmenter in `AtlasInput`'s cascade: real
  SAM3 loaded straight from `transformers>=5.5.4`, no `triton` dependency, so
  it works on CUDA, CPU, **and Mac (MPS)** alike.

  ```powershell
  pip install -e ".[sam3]"
  ```

  `facebook/sam3` is **gated** on Hugging Face (Meta's SAM-License-1.0 —
  commercial use permitted, military/ITAR use carved out). One-time setup:

  1. Request access at https://huggingface.co/facebook/sam3 (click "Agree
     and access repository").
  2. Create a token at https://huggingface.co/settings/tokens (Read scope).
  3. Run `hf auth login` (or set `HF_TOKEN`) and paste the token.

  If `transformers<5.5.4` (or `[sam3]` isn't installed), `AtlasInput`
  automatically falls back to **`AtlasSemanticMask`** (SegFormer/ADE20K,
  `[neural]`, no triton either — a learned CPU/MPS sky/scope mask), and the
  numpy sky heuristic is the zero-dependency floor.

  The third-party `SAM3Segment` node
  ([ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)) still works if
  manually wired (e.g. in `examples/atlas_camera_staged_master_workflow.json`)
  but is no longer preferred by `AtlasInput`'s own cascade. It hard-requires
  `triton` (CUDA-only — on Windows + NVIDIA, `python_embeded\python.exe -m pip
  install triton-windows`; on Mac/CPU/AMD it cannot load at all). Grounded-SAM2
  (GroundingDINO + a SAM2 pack) remains an optional premium Mac tier for
  SAM-grade edges outside Atlas's own cascade, at the cost of two extra models.
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
  (the reference Qwen subgraph example was removed from the shipped set
  2026-07-12 — recover it with
  `git show 10e600b:examples/atlas_qwen_image_edit_2511_multiangle_camera.json`). The
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
