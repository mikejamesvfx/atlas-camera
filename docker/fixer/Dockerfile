# Inference environment for the AtlasRenderFix node (NVIDIA Fixer render repair).
#
# Fixer's official environment is the NGC cosmos-predict2 container, which is
# auth-locked on nvcr.io (anonymous pull -> 403). This recipe rebuilds an
# equivalent environment on the PUBLIC NGC PyTorch image (torch 2.8 +
# transformer_engine preinstalled, Blackwell/sm_120-ready) — spike-verified
# end-to-end on real Atlas frames 2026-07-10.
#
# Build (once):
#   docker build -t fixer-spike-env -f docker/fixer/Dockerfile docker/fixer/
#
# The transformer-engine 2.x API shim (sitecustomize.py) is deliberately NOT
# baked in — the AtlasRenderFix node bind-mounts it from the installed
# atlas_camera package (inference/fixer_shim/) on every run, so the shim has
# one source of truth.
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Three container quirks, all found live during the spike:
#  - its dill 0.4 lacks .extend -> breaks `import torch` (datapipes)
#  - its tabulate package doesn't export tabulate() -> breaks fvcore
#  - its torchao has broken metadata (Version: None, no RECORD): pip cannot
#    uninstall it and transformers' is_torchao_available crashes parsing the
#    None version. Hard-delete — and do it in the FINAL layer too (see below).
RUN pip install --no-cache-dir "dill<0.4" && \
    pip install --no-cache-dir --force-reinstall "tabulate>=0.9"

# The cosmos-predict2 wheel bundles the `imaginaire` package Fixer imports;
# --no-deps keeps its cu126 pins (torch/triton/flash-attn/TE 1.13) from
# clobbering the container's own newer stack.
RUN pip install --no-cache-dir --no-deps "cosmos-predict2==1.0.9"

RUN pip install --no-cache-dir attrs hydra-core omegaconf loguru einops termcolor \
    "diffusers==0.33.1" "transformers==4.51.3" "peft==0.14.0" safetensors sentencepiece \
    "imageio[ffmpeg]" mediapy webdataset ftfy fvcore iopath pycocotools \
    better-profanity nltk qwen-vl-utils "megatron-core==0.10.0" \
    lpips natsort

# torchao purge must be the LAST layer: an earlier-layer rm was observed to
# resurface in the final image. The assert makes a regression fail the build.
RUN rm -rf /usr/local/lib/python3.12/dist-packages/torchao* && \
    python -c "import importlib.util; assert importlib.util.find_spec('torchao') is None, 'torchao still importable'" && \
    echo "torchao purged"

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
