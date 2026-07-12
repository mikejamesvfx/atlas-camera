"""Semantic segmentation (SegFormer / ADE20K) for named-class masks.

A lightweight, promptless alternative to SAM3 text segmentation: SegFormer
predicts one of the 150 fixed ADE20K scene classes per pixel ("sky", "floor",
"building", "tree", "person", ...). Deterministic and small (b0 is ~15MB),
so it works as a native sky-mask source and as a geometry-prior fallback for
`AtlasScopeMask` when a free-text SAM prompt no-matches.

Heavy dependencies (torch + transformers) are imported lazily so the core
package stays dependency-free — same contract as depth_estimator.py. Install
with:  pip install -e .[neural]
"""

from __future__ import annotations

from typing import Any

from atlas_camera.inference._common import bounded_cache_set, resolve_device

DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# Combo choices for the ComfyUI node — append-only (values serialize).
SEGFORMER_MODELS = (
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
)

_SEG_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_SEG_MODEL_CACHE_MAX = 2


def _require_segformer() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import (AutoImageProcessor,
                                  SegformerForSemanticSegmentation)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Semantic segmentation requires torch and transformers. Install with:\n"
            "    pip install -e .[neural]"
        ) from exc
    return torch, AutoImageProcessor, SegformerForSemanticSegmentation


def _get_segformer(model_id: str, device: str):
    cached = _SEG_MODEL_CACHE.get((model_id, device))
    if cached is not None:
        return cached
    torch, AutoImageProcessor, SegformerForSemanticSegmentation = _require_segformer()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model = model.to(device).eval()
    bounded_cache_set(_SEG_MODEL_CACHE, (model_id, device), (processor, model),
                      _SEG_MODEL_CACHE_MAX, release_cuda=True)
    return processor, model


def match_class_ids(id2label: dict, query: str) -> tuple[set, list[str]]:
    """Resolve a comma-separated class query against ADE20K labels.

    ADE labels are short names, occasionally with synonym lists
    ("building;edifice") or trailing spaces in the shipped configs. Each
    query token is matched EXACT-FIRST against the split/stripped label
    parts; substring is only a per-token fallback when nothing matched
    exactly — so "window" still finds "windowpane", but "sky" resolves to
    only "sky", never bleeding into "skyscraper" (found live). Pure function
    (no torch) so it's unit-testable without the extra.
    """
    items = [(int(raw_id), str(raw_label).strip(),
              [p.strip().lower() for p in str(raw_label).split(";")])
             for raw_id, raw_label in id2label.items()]
    tokens = [t.strip().lower() for t in (query or "").split(",") if t.strip()]
    ids: set = set()
    matched: list[str] = []
    for tok in tokens:
        hits = [(i, label) for i, label, parts in items if tok in parts]
        if not hits:
            hits = [(i, label) for i, label, parts in items
                    if any(tok in p for p in parts)]
        for i, label in hits:
            ids.add(i)
            if label not in matched:
                matched.append(label)
    return ids, matched


def semantic_class_mask(image, classes: str,
                        model_id: str = DEFAULT_SEGFORMER_MODEL,
                        device: str | None = None):
    """Segment `image` (PIL) and return a bool mask covering `classes`.

    Returns ``(mask, matched, coverage)``: an (H, W) bool numpy array at the
    image's own resolution (logits upsampled bilinearly before argmax — the
    canonical SegFormer post-process), the list of matched ADE label names,
    and the mask's frame-coverage fraction.
    """
    import numpy as np

    torch, _, _ = _require_segformer()
    device = resolve_device(device, torch)
    processor, model = _get_segformer(model_id, device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits  # (1, n_classes, h/4, w/4)
        logits = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width),
            mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy()

    id2label = getattr(model.config, "id2label", {}) or {}
    ids, matched = match_class_ids(id2label, classes)
    if not ids:
        return np.zeros(labels.shape, dtype=bool), [], 0.0
    mask = np.isin(labels, sorted(ids))
    return mask, matched, float(mask.mean())


def available_labels(model_id: str = DEFAULT_SEGFORMER_MODEL,
                     device: str | None = None) -> list[str]:
    """The model's ADE20K label names (for no-match reports)."""
    torch, _, _ = _require_segformer()
    device = resolve_device(device, torch)
    _, model = _get_segformer(model_id, device)
    return [str(v).strip() for v in (getattr(model.config, "id2label", {}) or {}).values()]
