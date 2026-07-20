"""Native SAM3 concept segmentation via `transformers` (no triton).

The third-party `SAM3Segment` node (comfyui-rmbg) hard-requires `triton`,
which does not exist on Mac (MPS), CPU-only, or AMD boxes — those users can
never load it, even though nothing about SAM3 itself requires triton. This
module loads SAM3 straight from `transformers` (`Sam3Model`/`Sam3Processor`,
the single-image concept-conditioned detector — NOT the video tracker,
which Atlas never needs since it masks stills, not clips), inspired by
lettidude/LiveActionAOV's `passes/matte/sam3.py`.

Heavy dependencies (torch + transformers>=5.5.4) are imported lazily so the
core package stays dependency-free — same contract as depth_estimator.py /
semantic_segmenter.py. Install with:  pip install -e .[sam3]

`facebook/sam3` is GATED on Hugging Face (Meta's SAM-License-1.0 — commercial
use permitted, military/ITAR use carved out). One-time setup: request access
at https://huggingface.co/facebook/sam3, then `hf auth login` (or set
HF_TOKEN). See INSTALL.md.
"""

from __future__ import annotations

from typing import Any

from atlas_camera.inference._common import bounded_cache_set, resolve_device

DEFAULT_SAM3_MODEL = "facebook/sam3"

_MIN_TRANSFORMERS_VERSION = (5, 5, 4)

_SAM3_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_SAM3_MODEL_CACHE_MAX = 2


class Sam3GatedRepoError(RuntimeError):
    """Raised when a SAM3 repo (e.g. facebook/sam3) is gated on Hugging Face
    and the caller hasn't requested access / authenticated yet. A distinct
    type so callers (e.g. AtlasSAM3Mask) can catch this specific,
    recoverable-by-the-user case without also swallowing a genuine
    version/import RuntimeError from _require_sam3()."""


def _meets_min_version(version_str: str,
                       minimum: tuple = _MIN_TRANSFORMERS_VERSION) -> bool:
    """Best-effort numeric-prefix version compare ('5.5.4.dev0' -> (5,5,4)),
    pure and dependency-free so it's directly unit-testable without a real
    transformers install."""
    parts = []
    for chunk in version_str.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts) >= minimum


def native_sam3_available() -> bool:
    """Cheap, network-free capability probe: torch AND transformers
    importable, with transformers >= _MIN_TRANSFORMERS_VERSION (SAM3's model
    classes only exist from transformers ~5.5). Never imports torch,
    downloads weights, or touches the network — torch's presence is checked
    via `importlib.util.find_spec` (importable without importing). Used by
    node_helpers._native_sam3_available(), which AtlasInput's build-time
    cascade decision calls."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import transformers
    except ImportError:
        return False
    return _meets_min_version(transformers.__version__)


def _wrap_if_gated_repo(repo: str, exc: BaseException):
    """Detect Hugging Face's gated-repo 401/OSError shape and translate it
    into an actionable Sam3GatedRepoError. Returns None (caller re-raises
    the original exception) for any other exception shape."""
    text = str(exc)
    if "gated repo" not in text and "401" not in text:
        return None
    return Sam3GatedRepoError(
        f"'{repo}' is a gated Hugging Face repo (Meta's SAM-License-1.0). "
        "One-time setup:\n"
        f"  1. Request access at https://huggingface.co/{repo} "
        "(click \"Agree and access repository\")\n"
        "  2. Create a token at https://huggingface.co/settings/tokens (Read scope)\n"
        "  3. Run `hf auth login` (or set HF_TOKEN) and paste the token\n"
        "See INSTALL.md for details."
    )


def _require_sam3():
    """Raise an actionable RuntimeError unless native SAM3's dependencies
    (torch + transformers>=5.5.4) are satisfied; otherwise return
    (torch, Sam3Model, Sam3Processor). Only imports the heavier SAM3 model
    classes once native_sam3_available() has already confirmed the version
    floor."""
    if not native_sam3_available():
        raise RuntimeError(
            "Native SAM3 requires transformers>=5.5.4 and torch. Install with:\n"
            "    pip install -e .[sam3]"
        )
    import torch
    from transformers import Sam3Model, Sam3Processor
    return torch, Sam3Model, Sam3Processor


def _get_sam3(model_id: str, device: str):
    cached = _SAM3_MODEL_CACHE.get((model_id, device))
    if cached is not None:
        return cached
    torch, Sam3Model, Sam3Processor = _require_sam3()
    try:
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id)
    except Exception as exc:
        wrapped = _wrap_if_gated_repo(model_id, exc)
        if wrapped is not None:
            raise wrapped from exc
        raise
    model = model.to(device).eval()
    bounded_cache_set(_SAM3_MODEL_CACHE, (model_id, device), (processor, model),
                      _SAM3_MODEL_CACHE_MAX, release_cuda=True)
    return processor, model


def _run_sam3_detector(image, token: str, model_id: str, device: str,
                       confidence_threshold: float):
    """One SAM3 forward pass for a single concept -> a unioned instance
    mask. Mirrors the exact processor/model call shape verified against a
    real SAM3 integration (lettidude/LiveActionAOV, passes/matte/sam3.py):
    `processor(images=..., text=..., return_tensors="pt")` -> `model(**inputs)`
    -> `processor.post_process_instance_segmentation(...)` returning a list
    of {"masks", "scores"} dicts, one per input image."""
    import numpy as np
    torch, _, _ = _require_sam3()
    processor, model = _get_sam3(model_id, device)

    inputs = processor(images=image, text=token, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=confidence_threshold, mask_threshold=0.5,
        target_sizes=[(image.height, image.width)])
    mask = np.zeros((image.height, image.width), dtype=bool)
    if not results:
        return mask, False
    instance_masks = results[0].get("masks")
    if instance_masks is None:
        return mask, False
    n = int(instance_masks.shape[0]) if hasattr(instance_masks, "shape") else 0
    for i in range(n):
        m = instance_masks[i]
        m_np = (m.float().cpu().numpy() if hasattr(m, "float")
               else np.asarray(m, dtype="float32"))
        mask |= (m_np > 0.5)
    return mask, n > 0


def _detect_one_concept(image, token: str, model_id: str, device: str,
                        confidence_threshold: float):
    """One concept's mask, with a one-shot MPS -> CPU retry: SAM3's ops are
    new to transformers and untested on MPS (LiveActionAOV's own SAM3
    integration never tries MPS at all), so a RuntimeError from an
    unsupported op reloads the model on cpu and retries once instead of
    crashing the whole mask build."""
    try:
        return _run_sam3_detector(image, token, model_id, device,
                                  confidence_threshold)
    except RuntimeError:
        if device != "mps":
            raise
        _SAM3_MODEL_CACHE.pop((model_id, "mps"), None)
        return _run_sam3_detector(image, token, model_id, "cpu",
                                  confidence_threshold)


def sam3_concept_mask(image, concepts: str,
                      model_id: str = DEFAULT_SAM3_MODEL,
                      device: str | None = None,
                      confidence_threshold: float = 0.5):
    """Segment `image` (PIL) and return a bool mask covering `concepts`.

    Returns ``(mask, matched, coverage)``: an (H, W) bool numpy array at the
    image's own resolution, the list of concept tokens that had >=1
    detection above `confidence_threshold`, and the mask's frame-coverage
    fraction — same return shape as semantic_segmenter.semantic_class_mask.
    Comma-separated `concepts` runs one SAM3 forward pass per token (its
    classification head is single-concept-per-forward) and unions every
    detected instance across all tokens.
    """
    import numpy as np

    torch, _, _ = _require_sam3()
    device = resolve_device(device, torch)
    tokens = [t.strip() for t in (concepts or "").split(",") if t.strip()]
    if not tokens:
        return np.zeros((image.height, image.width), dtype=bool), [], 0.0

    mask = np.zeros((image.height, image.width), dtype=bool)
    matched: list[str] = []
    for token in tokens:
        token_mask, hit = _detect_one_concept(
            image, token, model_id, device, confidence_threshold)
        if hit:
            matched.append(token)
        mask |= token_mask
    return mask, matched, float(mask.mean())
