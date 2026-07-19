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
    """Cheap, network-free capability probe: transformers importable AND
    >= _MIN_TRANSFORMERS_VERSION (SAM3's model classes only exist from
    transformers ~5.5). Never imports torch, downloads weights, or touches
    the network. Used by node_helpers._native_sam3_available(), which
    AtlasInput's build-time cascade decision calls."""
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
