"""World Tracing layered-geometry inference (EXPERIMENTAL, research-only).

Second backend for the hidden-geometry track (see ``lari_hidden_geometry.py``
for the first): WT-DiT predicts, per pixel, an ordered stack of camera-space
3D points — layer 0 the visible surface, later layers front-to-back occluded
intersections (arXiv 2606.13652). Scene config ``r69l`` (840x840, 6 layers,
1.5B params, ~17s at 20 diffusion steps). Unlike LaRI this is GENERATIVE
(diffusion): results vary by seed, pinned via the ``seed`` argument.

LICENSING / ACCESS: the upstream code (github.com/haoz19/world-tracing) is
CC BY-NC-ND 4.0 — non-commercial research use, no redistributed derivatives —
so atlas_camera vendors none of it: the user clones the repository themselves
and points ``wt_path`` (or the ATLAS_WT_PATH env var) at the clone. The
checkpoints are additionally GATED on HuggingFace (per-account approval by the
authors — request access on the haoz19 model pages, then authenticate the
venv via ``huggingface-cli login`` / HF_TOKEN).

The scene model's output is RELATIVE scale (median-log normalized), not
metric — irrelevant to consumers, because ``core/hidden_geometry``'s layer-0
median-scale registration absorbs any global scale by design.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from atlas_camera.inference._common import bounded_cache_set, resolve_device
from atlas_camera.inference.lari_hidden_geometry import LayeredDepthResult

WT_SCENE_CONFIG = "r69l"
WT_RESOLUTION = 840  # scene model's working long edge

_WT_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_WT_MODEL_CACHE_MAX = 1  # 1.5B-param DiT; keep at most one resident


def _require_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "World Tracing inference requires torch. Install with:\n"
            "    pip install -e .[neural]"
        ) from exc


def _resolve_wt_root(wt_path: str | None) -> Path:
    p = (wt_path or "").strip() or os.environ.get("ATLAS_WT_PATH", "").strip()
    root = Path(p) if p else None
    if root is None or not root.is_dir():
        raise RuntimeError(
            "World Tracing repository not found. This EXPERIMENTAL feature needs\n"
            "the world-tracing code, which atlas_camera cannot bundle (upstream\n"
            "license CC BY-NC-ND 4.0 — non-commercial research use only). Clone it\n"
            "and point at the clone:\n"
            "    git clone https://github.com/haoz19/world-tracing.git\n"
            "then set the node's wt_path widget (or the ATLAS_WT_PATH env var).\n"
            "The checkpoints are HF-GATED: request access on the haoz19 model\n"
            "pages, then run `huggingface-cli login` (or set HF_TOKEN) in the\n"
            "ComfyUI venv so the download can authenticate."
        )
    return root


def _require_wt(wt_path: str | None) -> Any:
    """Import the world-tracing package from a user-provided clone."""
    root = _resolve_wt_root(wt_path)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    try:
        import wt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Found a world-tracing checkout at {root} but importing its `wt` "
            f"package failed ({exc}). Make sure the clone is complete, torch is "
            "installed, and its small dependencies (einops, safetensors, "
            "beartype, jaxtyping, structlog) are present in this venv."
        ) from exc
    return wt


def _get_wt_model(wt_path: str | None, device: str):
    """Load (model, cfg) via wt's own checkpoint resolver — a bare config name
    resolves to the gated HF checkpoint and downloads on first use."""
    root = str(_resolve_wt_root(wt_path))
    cached = _WT_MODEL_CACHE.get((root, device))
    if cached is not None:
        return cached
    torch = _require_torch()
    _require_wt(wt_path)
    from wt.checkpoint import build_model_and_load_ckpt  # type: ignore

    model, cfg = build_model_and_load_ckpt(
        WT_SCENE_CONFIG, WT_SCENE_CONFIG, torch.device(device)
    )
    bounded_cache_set(_WT_MODEL_CACHE, (root, device), (model, cfg),
                      _WT_MODEL_CACHE_MAX)
    return model, cfg


def predict_layered_depth_wt(
    image_path: str | Path,
    *,
    wt_path: str | None = None,
    device: str | None = None,
    steps: int = 20,
    seed: int = 0,
) -> LayeredDepthResult:
    """Run World Tracing's scene model on one image -> layered depth stack.

    Returns the same :class:`LayeredDepthResult` contract as the LaRI backend:
    ``layers`` (H, W, L) float32 forward depth, model-native (relative) units,
    at the model's working resolution (840x840 square for r69l — the consumer's
    bilinear upsample back to the source frame absorbs the aspect change).
    Invalid layers (WT's own per-layer ray-stop mask) are zeroed, which
    ``core.hidden_geometry``'s ``z > valid_min`` test skips naturally.

    Mirrors ``examples/infer_scene.py`` in the WT repo: full-frame scene mode
    (no center-crop, no auto-alpha, raw RGB to the encoder), bf16 autocast on
    cuda, ``use_gt_mask=True`` with the preprocessing mask.
    """
    torch = _require_torch()
    import numpy as np

    device = resolve_device(device, torch)
    _require_wt(wt_path)
    model, cfg = _get_wt_model(wt_path, device)
    from wt.data import load_rgba_image, preprocess_rgba_for_model  # type: ignore
    from wt.inference import (  # type: ignore
        _bypass_activation_checkpointing,
        inference_diffusion,
    )

    rgba = load_rgba_image(Path(image_path), auto_alpha=False)
    h, w = rgba.shape[:2]
    rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
        rgba,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        alpha_erode_px=0,
        center_crop=False,   # scene mode: full-frame input
        bg_color=None,       # raw RGB to the encoder
    )
    rgb_t = rgb_t.to(device)
    mask_t = mask_t.to(device)
    intr_t = intr_t.to(device)

    torch.manual_seed(seed)
    if device == "cuda" or str(device).startswith("cuda"):
        torch.cuda.manual_seed(seed)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )
    infer_kwargs = dict(cfg["inference_kwargs"])
    infer_kwargs["num_steps"] = int(steps)

    with torch.no_grad(), autocast_ctx, _bypass_activation_checkpointing(model):
        xyz_pred, mask_pred, _ = inference_diffusion(
            model,
            rgb_t,
            gt_mask=mask_t,
            use_gt_mask=True,
            intrinsics=intr_t,
            invalid_fill_mode="noise",
            **infer_kwargs,
        )

    xyz = xyz_pred[0].float().cpu().numpy()          # (L, H', W', 3)
    valid = mask_pred[0].cpu().numpy().astype(bool)  # (L, H', W')
    z = np.transpose(xyz[..., 2], (1, 2, 0))         # (H', W', L)
    valid = np.transpose(valid, (1, 2, 0))
    if valid.any() and float((z[valid] > 0).mean()) < 0.5:
        z = -z  # sign check, same guard as the LaRI path
    z = np.where(valid, z, 0.0)

    return LayeredDepthResult(
        layers=np.ascontiguousarray(z, dtype=np.float32),
        image_width=w,
        image_height=h,
        metadata={
            "device": device,
            "backend": "world-tracing",
            "config": WT_SCENE_CONFIG,
            "n_layers": int(z.shape[-1]),
            "working_width": int(z.shape[1]),
            "working_height": int(z.shape[0]),
            "steps": int(steps),
            "seed": int(seed),
            "relative_scale": True,
            "license": "CC BY-NC-ND 4.0 (non-commercial research)",
            "research_only": True,
        },
    )
