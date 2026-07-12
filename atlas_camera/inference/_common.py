"""Shared helpers for the inference layer's lazily-imported torch models.

A leaf module (imports nothing from sibling inference modules) so
depth_estimator.py and learned_prior.py can both depend on it without a
circular import through inference/__init__.py.
"""

from __future__ import annotations

from typing import Any


def resolve_device(device: str | None, torch: Any) -> str:
    """cuda -> mps -> cpu autodetect. Was duplicated verbatim in
    depth_estimator.py and learned_prior.py — commit 02f3100 ("Fix MPS device
    detection...") had to patch this exact logic in both files simultaneously
    for a real shipped bug (silent CPU-only inference on Apple Silicon),
    concrete proof the duplication already caused a defect rather than being
    a theoretical risk.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bounded_cache_set(cache: dict, key: Any, value: Any, max_size: int,
                      *, release_cuda: bool = False) -> None:
    """Insert into a module-level cache dict, evicting the oldest entry (dict
    preserves insertion order) once it would exceed `max_size`. Same pattern
    already used for `_ATLAS_BLOCKOUT_CACHE` in comfy/nodes.py — applied here
    to the model caches, which previously grew unbounded (each entry holds a
    full loaded torch model, unlike the lightweight dict payloads that
    pattern was applied to elsewhere).

    ``release_cuda``: pass True for caches whose entries hold GPU models —
    after evicting one, the CUDA caching allocator still holds its blocks
    until `empty_cache()`, which matters on a contended card (measured
    31.9/32.6 GB during a real staged run). No-op without CUDA.
    """
    evicted = False
    if len(cache) >= max_size:
        oldest = next(iter(cache))
        del cache[oldest]
        evicted = True
    cache[key] = value
    if evicted and release_cuda:
        try:
            import gc

            import torch
            gc.collect()  # drop the evicted model's tensors first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
