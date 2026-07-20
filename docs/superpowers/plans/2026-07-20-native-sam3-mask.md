# Native SAM3 via `transformers` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the triton-locked third-party `SAM3Segment` (comfyui-rmbg) in Atlas Camera's sky/scope segmentation cascade with a native, `transformers`-only SAM3 node, so non-CUDA users (Mac/MPS, CPU, AMD) get real SAM3 quality instead of always falling back to the weaker SegFormer mask.

**Architecture:** a new leaf inference module (`atlas_camera/inference/sam3_segmenter.py`, mirroring the existing `semantic_segmenter.py`) wraps `transformers.Sam3Model`/`Sam3Processor` with lazy imports, a version-gated capability probe, gated-HF-repo error translation, and model caching. A new node (`AtlasSAM3Mask`) exposes it with the same interface shape as `AtlasSemanticMask`. `AtlasInput`'s `segment()` cascade is rewired to prefer it, with `AtlasSemanticMask` remaining the learned fallback and the third-party `SAM3Segment` removed entirely from Atlas's own cascade.

**Tech Stack:** Python, `transformers>=5.5.4,<6` (new `[sam3]` extra), `torch`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-20-native-sam3-mask-design.md`

---

## Task 1: `[sam3]` extra in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the new extra**

Open `pyproject.toml` and find the `moge` extra block (currently ends right before `image = [...]`). Insert a new `sam3` extra immediately after it:

```toml
# Native SAM3 (AtlasSAM3Mask) — transformers-only, no triton/comfyui-rmbg dependency, so it
# works on Mac (MPS) / CPU / AMD where the third-party SAM3Segment node (which hard-requires
# triton) cannot load at all. SAM3's model classes only exist from transformers ~5.5, hence
# the separate, narrower pin from [neural]'s own >=4.40 floor. facebook/sam3 is GATED on
# Hugging Face (Meta's SAM-License-1.0) — one-time `hf auth login` (or HF_TOKEN env) required
# after requesting access at https://huggingface.co/facebook/sam3.
sam3 = [
    "numpy>=1.24",
    "torch>=2.0",
    "transformers>=5.5.4,<6",
]
```

- [ ] **Step 2: Verify the TOML is well-formed**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
build: add [sam3] extra for native SAM3 segmentation

transformers>=5.5.4,<6 is required for the SAM3 model classes, separate
from [neural]'s looser >=4.40 floor used by Depth Anything V2/SegFormer.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 2: version probe + gated-repo error type (`sam3_segmenter.py`, part 1)

**Files:**
- Create: `atlas_camera/inference/sam3_segmenter.py`
- Test: `tests/test_sam3_segmenter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sam3_segmenter.py`:

```python
"""Native SAM3 (AtlasSAM3Mask) — version probe, gated-repo wrapper, and
concept-union logic.

The real SAM3 inference path needs downloaded weights + a gated HF repo, so
it is exercised live (not here) — same split as test_semantic_segmenter.py.
Everything pure/mockable is pinned below: the transformers version compare,
the capability probe, the gated-repo error translation, and the
comma-separated-concept union logic (via mocking the actual per-concept
detector call).
"""

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import atlas_camera.inference.sam3_segmenter as sam3_mod
from atlas_camera.inference.sam3_segmenter import (
    Sam3GatedRepoError,
    _meets_min_version,
    _wrap_if_gated_repo,
    native_sam3_available,
)


def test_meets_min_version_exact_and_above():
    assert _meets_min_version("5.5.4")
    assert _meets_min_version("5.5.5")
    assert _meets_min_version("5.6.0")
    assert _meets_min_version("6.0.0")


def test_meets_min_version_below():
    assert not _meets_min_version("5.5.3")
    assert not _meets_min_version("5.4.9")
    assert not _meets_min_version("4.40.0")


def test_meets_min_version_dev_suffix():
    # '5.5.4.dev0' -> numeric prefix (5, 5, 4) still compares correctly
    assert _meets_min_version("5.5.4.dev0")
    assert not _meets_min_version("5.5.3.dev999")


def test_native_sam3_available_true_with_new_transformers(monkeypatch):
    fake = types.SimpleNamespace(__version__="5.5.4")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert native_sam3_available() is True


def test_native_sam3_available_false_with_old_transformers(monkeypatch):
    fake = types.SimpleNamespace(__version__="4.40.0")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert native_sam3_available() is False


def test_native_sam3_available_false_when_transformers_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert native_sam3_available() is False


def test_require_sam3_raises_actionable_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(sam3_mod, "native_sam3_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"\[sam3\]"):
        sam3_mod._require_sam3()


def test_wrap_if_gated_repo_detects_gated_shape():
    exc = OSError("You are trying to access a gated repo. 401 Client Error.")
    wrapped = _wrap_if_gated_repo("facebook/sam3", exc)
    assert isinstance(wrapped, Sam3GatedRepoError)
    assert "hf auth login" in str(wrapped)
    assert "facebook/sam3" in str(wrapped)


def test_wrap_if_gated_repo_passes_through_unrelated_errors():
    exc = ValueError("some unrelated failure")
    assert _wrap_if_gated_repo("facebook/sam3", exc) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sam3_segmenter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atlas_camera.inference.sam3_segmenter'`

- [ ] **Step 3: Create the module with the version probe and gated-repo wrapper**

Create `atlas_camera/inference/sam3_segmenter.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sam3_segmenter.py -v`
Expected: 8 tests PASS (the concept-mask/node-helper tests referenced later in this plan aren't written yet — this step only collects the 8 tests written in Step 1)

- [ ] **Step 5: Commit**

```bash
git add atlas_camera/inference/sam3_segmenter.py tests/test_sam3_segmenter.py
git commit -m "$(cat <<'EOF'
feat: add native SAM3 version probe and gated-repo error wrapper

First slice of atlas_camera/inference/sam3_segmenter.py: a network-free
transformers>=5.5.4 capability probe (native_sam3_available), and a
Sam3GatedRepoError translator for Hugging Face's gated-repo 401/OSError
shape, ported from lettidude/LiveActionAOV's passes/matte/sam3.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 3: model cache + concept-mask union logic (`sam3_segmenter.py`, part 2)

**Files:**
- Modify: `atlas_camera/inference/sam3_segmenter.py`
- Test: `tests/test_sam3_segmenter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sam3_segmenter.py`:

```python
def test_sam3_concept_mask_unions_across_comma_separated_concepts(monkeypatch):
    monkeypatch.setattr(sam3_mod, "_require_sam3", lambda: (torch, None, None))

    h, w = 4, 4
    def fake_detect(image, token, model_id, device, confidence_threshold):
        m = np.zeros((h, w), dtype=bool)
        if token == "sky":
            m[0, :] = True
        elif token == "person":
            m[:, 0] = True
        return m, token in ("sky", "person")

    monkeypatch.setattr(sam3_mod, "_detect_one_concept", fake_detect)
    img = types.SimpleNamespace(height=h, width=w)

    mask, matched, coverage = sam3_mod.sam3_concept_mask(img, "sky, person, ghost")

    assert matched == ["sky", "person"]           # "ghost" never detected
    assert mask[0, :].all() and mask[:, 0].all()  # union of both hits
    assert coverage == pytest.approx(float(mask.mean()))


def test_sam3_concept_mask_empty_concepts_returns_empty_mask(monkeypatch):
    monkeypatch.setattr(sam3_mod, "_require_sam3", lambda: (torch, None, None))
    img = types.SimpleNamespace(height=4, width=4)
    mask, matched, coverage = sam3_mod.sam3_concept_mask(img, "")
    assert not mask.any() and matched == [] and coverage == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sam3_segmenter.py -k concept_mask -v`
Expected: FAIL with `AttributeError: module 'atlas_camera.inference.sam3_segmenter' has no attribute 'sam3_concept_mask'`

- [ ] **Step 3: Append the model cache, detector call, and concept-mask function**

Append to `atlas_camera/inference/sam3_segmenter.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sam3_segmenter.py -v`
Expected: 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atlas_camera/inference/sam3_segmenter.py tests/test_sam3_segmenter.py
git commit -m "$(cat <<'EOF'
feat: add native SAM3 model cache and concept-union detector

sam3_concept_mask() splits comma-separated concepts, runs one SAM3
forward pass per token (its classification head is single-concept-per-
forward), and unions every detected instance across all tokens — same
return shape as semantic_segmenter.semantic_class_mask. Includes a
one-shot MPS -> CPU retry since SAM3's ops are new to transformers and
untested on MPS.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 4: `_native_sam3_available()` capability probe (`node_helpers.py`)

**Files:**
- Modify: `atlas_camera/comfy/node_helpers.py:1342` (right after `_comfy_registry`), `atlas_camera/comfy/node_helpers.py:1517` (`__all__` list)
- Test: `tests/test_sam3_segmenter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sam3_segmenter.py`:

```python
def test_node_helpers_native_sam3_available_delegates(monkeypatch):
    from atlas_camera.comfy import node_helpers
    monkeypatch.setattr(sam3_mod, "native_sam3_available", lambda: True)
    assert node_helpers._native_sam3_available() is True


def test_node_helpers_native_sam3_available_fails_soft(monkeypatch):
    from atlas_camera.comfy import node_helpers
    def _boom():
        raise RuntimeError("simulated broken inference module")
    monkeypatch.setattr(sam3_mod, "native_sam3_available", _boom)
    assert node_helpers._native_sam3_available() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sam3_segmenter.py -k native_sam3_available_delegates -v`
Expected: FAIL with `AttributeError: module 'atlas_camera.comfy.node_helpers' has no attribute '_native_sam3_available'`

- [ ] **Step 3: Add `_native_sam3_available()` to `node_helpers.py`**

Read `atlas_camera/comfy/node_helpers.py` around line 1342 to confirm the current `_comfy_registry` definition is still at that location, then add immediately after it:

```python
def _native_sam3_available() -> bool:
    """Cheap, network-free capability probe for native SAM3 (AtlasSAM3Mask),
    used by AtlasInput's build-time cascade decision. Native SAM3 is
    ALWAYS registered (it's Atlas's own node class), so registry presence
    (unlike third-party packs) can't distinguish "the [sam3] extra +
    transformers>=5.5.4 actually works" from "the class merely exists" —
    this delegates to the real inference-layer check instead. Any failure
    (module missing, unexpected error) is treated as unavailable, the same
    fail-soft contract as _comfy_registry()."""
    try:
        from atlas_camera.inference.sam3_segmenter import native_sam3_available
        return native_sam3_available()
    except Exception:
        return False
```

Then find the `__all__` list near the end of the file and add `'_native_sam3_available',` right after `'_comfy_registry',`:

```python
    '_comfy_registry',
    '_native_sam3_available',
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sam3_segmenter.py -v`
Expected: 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atlas_camera/comfy/node_helpers.py tests/test_sam3_segmenter.py
git commit -m "$(cat <<'EOF'
feat: add _native_sam3_available() capability probe to node_helpers

Native SAM3 is Atlas's own node class, so it's always in the ComfyUI
registry regardless of whether transformers>=5.5.4 is actually
installed -- registry-presence probing (the pattern used for
third-party packs) can't tell those two cases apart. This delegates to
sam3_segmenter.native_sam3_available() instead, fail-soft like
_comfy_registry().

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 5: `AtlasSAM3Mask` node (`nodes_inpaint.py`)

**Files:**
- Modify: `atlas_camera/comfy/nodes_inpaint.py` (insert right after the `AtlasSemanticMask` class, before `class AtlasInpaintCrop:`)
- Test: `tests/test_sam3_mask_node.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sam3_mask_node.py`:

```python
"""AtlasSAM3Mask 🪄 — native SAM3 concept mask node.

Mirrors tests/test_moge_normals_node.py's pattern: the real SAM3 inference
call (sam3_concept_mask) is mocked here, pinning the node's report
formatting and its gated-repo-vs-raised-error boundary. Live inference
needs real weights + transformers>=5.5.4 + HF auth, so it isn't exercised
here (same "pure logic tested, inference exercised live" split as
test_semantic_segmenter.py / test_sam3_segmenter.py).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_inpaint import AtlasSAM3Mask
from atlas_camera.inference.sam3_segmenter import Sam3GatedRepoError


def _image(h=8, w=8):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def test_segment_reports_matched_concepts_and_coverage(monkeypatch):
    mask_np = np.zeros((8, 8), dtype=bool)
    mask_np[0, :] = True
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask",
        lambda *a, **k: (mask_np, ["sky"], float(mask_np.mean())))

    mask, report = AtlasSAM3Mask().segment(_image(), concepts="sky")

    assert mask.shape == (1, 8, 8)
    assert bool(mask[0, 0, :].all())
    assert "sky" in report and "facebook/sam3" in report


def test_segment_reports_no_match(monkeypatch):
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask",
        lambda *a, **k: (np.zeros((8, 8), dtype=bool), [], 0.0))
    mask, report = AtlasSAM3Mask().segment(_image(), concepts="ghost")
    assert "NO MATCH" in report
    assert not bool(mask.any())


def test_segment_catches_gated_repo_error_into_report(monkeypatch):
    def _boom(*a, **k):
        raise Sam3GatedRepoError("request access at https://huggingface.co/facebook/sam3")
    monkeypatch.setattr(
        "atlas_camera.inference.sam3_segmenter.sam3_concept_mask", _boom)

    mask, report = AtlasSAM3Mask().segment(_image(), concepts="sky")

    assert not bool(mask.any())                 # empty mask, never raises
    assert "huggingface.co/facebook/sam3" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sam3_mask_node.py -v`
Expected: FAIL with `ImportError: cannot import name 'AtlasSAM3Mask' from 'atlas_camera.comfy.nodes_inpaint'`

- [ ] **Step 3: Add the `AtlasSAM3Mask` class**

Read `atlas_camera/comfy/nodes_inpaint.py` to find the exact end of the `AtlasSemanticMask` class (its `segment` method returns `(mask, report)` right before `class AtlasInpaintCrop:`), then insert this new class between them:

```python
class AtlasSAM3Mask:
    """🪄 Native SAM3 concept mask via transformers — no triton/comfyui-rmbg
    dependency.

    The third-party `SAM3Segment` node (comfyui-rmbg) hard-requires triton,
    which does not exist on Mac (MPS) / CPU / AMD — those users could never
    load real SAM3 and always fell back to `AtlasSemanticMask` (SegFormer).
    This node loads SAM3 straight from `transformers>=5.5.4`
    (`Sam3Model`/`Sam3Processor`), so it works everywhere `[sam3]` installs:
    CUDA, CPU, and MPS (best-effort — see the device note below).

    `AtlasInput`'s sky/scope cascade now prefers this node over
    `AtlasSemanticMask`, which remains the learned fallback tier when
    `transformers<5.5.4` (or `[sam3]` isn't installed).

    `facebook/sam3` is GATED on Hugging Face (Meta's SAM-License-1.0).
    One-time setup: request access at https://huggingface.co/facebook/sam3,
    then `hf auth login` (or set HF_TOKEN). A gated-repo failure is caught
    and returned as the `report` string rather than raised — it's a one-time
    auth step, not a broken install. See INSTALL.md.
    """
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "segment"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "concepts": ("STRING", {"default": "sky",
                    "tooltip": "Comma-separated open-vocabulary concepts (e.g. 'sky', "
                               "'person, vehicle'). The mask is the UNION of all detected "
                               "instances across every concept."}),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                    "step": 0.01}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    def segment(self, image, concepts="sky", confidence_threshold=0.5, device="auto", **_extra):
        from atlas_camera.inference.sam3_segmenter import (
            DEFAULT_SAM3_MODEL, Sam3GatedRepoError, sam3_concept_mask)
        torch = _require_torch()

        pil = _image_tensor_to_pil(image)
        dev = None if device == "auto" else device
        try:
            mask_np, matched, coverage = sam3_concept_mask(
                pil, concepts, model_id=DEFAULT_SAM3_MODEL, device=dev,
                confidence_threshold=confidence_threshold)
        except Sam3GatedRepoError as exc:
            empty = torch.zeros((1, pil.height, pil.width), dtype=torch.float32)
            return (empty, str(exc))
        mask = torch.from_numpy(mask_np.astype("float32")).unsqueeze(0)
        if matched:
            report = (f"matched {sorted(set(matched))} -> {coverage:.1%} of frame "
                      f"({DEFAULT_SAM3_MODEL})")
        else:
            report = f"NO MATCH for '{concepts}' — mask is empty ({DEFAULT_SAM3_MODEL})."
        return (mask, report)
```

`_require_torch` and `_image_tensor_to_pil` are already imported at the top of `nodes_inpaint.py` (used by `AtlasSemanticMask` right above) — no new imports needed in that file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sam3_mask_node.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atlas_camera/comfy/nodes_inpaint.py tests/test_sam3_mask_node.py
git commit -m "$(cat <<'EOF'
feat: add AtlasSAM3Mask node

Native SAM3 concept mask via transformers, same interface shape as
AtlasSemanticMask (comma-separated concepts -> union mask + report) so
the two are interchangeable in the segmentation cascade. A gated-HF-repo
failure is caught and returned as the report string; version/import
errors raise normally.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 6: register `AtlasSAM3Mask` in the node registry and façade

**Files:**
- Modify: `atlas_camera/comfy/node_registry.py:72-83` (import block), `:176` (`NODE_CLASS_MAPPINGS`), `:257` (`NODE_DISPLAY_NAME_MAPPINGS`)
- Modify: `atlas_camera/comfy/nodes.py:147-158` (façade import block)
- Modify: `tests/test_comfy_node_registry.py:21-44` (`NORMAL_KEYS`, count)

- [ ] **Step 1: Add the import and registry entries in `node_registry.py`**

In the `from atlas_camera.comfy.nodes_inpaint import (...)` block (currently lines 72-83), add `AtlasSAM3Mask` right after `AtlasSemanticMask`:

```python
from atlas_camera.comfy.nodes_inpaint import (
    AtlasScopeMask,
    AtlasSemanticMask,
    AtlasSAM3Mask,
    AtlasInpaintCrop,
    AtlasInpaintStitch,
    AtlasSDXLInpaint,
    AtlasInstanceMask,
    AtlasSegmentedSDXLInpaint,
    AtlasCleanPlateLayer,
    AtlasCleanPlateStack,
    AtlasSkyDomeLayer,
)
```

In `NODE_CLASS_MAPPINGS` (currently around line 177), add right after the `AtlasSemanticMask` entry:

```python
    "AtlasSemanticMask":          AtlasSemanticMask,
    "AtlasSAM3Mask":              AtlasSAM3Mask,
```

In `NODE_DISPLAY_NAME_MAPPINGS` (currently around line 257), add right after the `AtlasSemanticMask` entry:

```python
    "AtlasSemanticMask":          "Atlas Semantic Mask 🧩",
    "AtlasSAM3Mask":              "Atlas SAM3 Mask 🪄",
```

- [ ] **Step 2: Add the façade re-export in `nodes.py`**

In `atlas_camera/comfy/nodes.py`'s `from atlas_camera.comfy.nodes_inpaint import (...)` block (currently lines 147-158), add `AtlasSAM3Mask` right after `AtlasSemanticMask`:

```python
from atlas_camera.comfy.nodes_inpaint import (
    AtlasScopeMask,
    AtlasSemanticMask,
    AtlasSAM3Mask,
    AtlasInpaintCrop,
    AtlasInpaintStitch,
    AtlasSDXLInpaint,
    AtlasInstanceMask,
    AtlasSegmentedSDXLInpaint,
    AtlasCleanPlateLayer,
    AtlasCleanPlateStack,
    AtlasSkyDomeLayer,
)
```

- [ ] **Step 3: Update `tests/test_comfy_node_registry.py`**

Add `"AtlasSAM3Mask"` to the `NORMAL_KEYS` set (currently lines 21-44) — insert alphabetically (ASCII order — the set's set literal ordering doesn't affect correctness, just readability), right after `"AtlasRollTrim"` and before `"AtlasSDXLInpaint"`:

```python
    "AtlasOcclusionMask", "AtlasPitchTrim", "AtlasReferenceScaleSolve",
    "AtlasRegisterPlate", "AtlasRollTrim", "AtlasSAM3Mask", "AtlasSDXLInpaint",
    "AtlasScaleOverride",
```

Update the doc comment above `NORMAL_KEYS` (currently "67 standard + 4 experimental = 71") to "68 standard + 4 experimental = 72", and update the count assertion:

```python
def test_normal_registry_keys_exact():
    assert set(nodes.NODE_CLASS_MAPPINGS) == NORMAL_KEYS
    assert len(nodes.NODE_CLASS_MAPPINGS) == 68
```

- [ ] **Step 4: Run the registry tests**

Run: `python -m pytest tests/test_comfy_node_registry.py -v`
Expected: all tests PASS (in particular `test_normal_registry_keys_exact`, `test_display_name_mapping_covers_registry`, `test_mapping_values_are_the_registered_classes`, `test_facade_reexports_public_helpers`)

If `test_representative_public_class_imports` fails, open it and check whether it needs `AtlasSAM3Mask` added to its import list — if so, add it there too, matching the existing style.

- [ ] **Step 5: Commit**

```bash
git add atlas_camera/comfy/node_registry.py atlas_camera/comfy/nodes.py tests/test_comfy_node_registry.py
git commit -m "$(cat <<'EOF'
feat: register AtlasSAM3Mask in the node registry and facade

Appends the new node to NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS
(node registry keys are a saved-workflow contract -- existing entries
untouched) and the nodes.py compatibility facade. Bumps the pinned
registry surface from 67 to 68 standard nodes.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 7: rewire `AtlasInput`'s `segment()` cascade (`nodes_viewport.py`)

**Files:**
- Modify: `atlas_camera/comfy/nodes_viewport.py:13-30` (import block), `:546`, `:606`, `:613`, `:679-717`, `:745-747`, `:825-828`

- [ ] **Step 1: Add the new import**

In `atlas_camera/comfy/nodes_viewport.py`'s `from atlas_camera.comfy.node_helpers import (...)` block, add `_native_sam3_available` right after `_comfy_registry` (keeping the existing alphabetical-ish ordering of that block):

```python
    _blockout_cache_set,
    _clone_solve_with_metadata,
    _comfy_registry,
    _native_sam3_available,
    _decode_b64_to_tensor,
```

- [ ] **Step 2: Rewrite the cascade in `build()`**

Read `atlas_camera/comfy/nodes_viewport.py` around lines 673-720 to confirm current line numbers, then replace this block (currently lines 679-717):

```python
        registry = _comfy_registry()
        have_sam = "SAM3Segment" in registry
        # AtlasSemanticMask is our own node (SegFormer/ADE20K, pure transformers,
        # NO triton/CUDA requirement) — the non-CUDA fallback for text-prompt
        # segmentation. SAM3 needs triton, which does not exist on Mac(MPS)/CPU/
        # AMD, so those users can never load it; SegFormer keeps sky+scope on a
        # LEARNED mask there instead of collapsing to the bare heuristic.
        have_semantic = "AtlasSemanticMask" in registry
        have_inpaint = ("INPAINT_InpaintWithModel" in registry
                        and "INPAINT_LoadInpaintModel" in registry
                        and "INPAINT_ExpandMask" in registry)
        notes: list = []
        g = _graph_builder()

        def sam3(image_ref, prompt_value):
            return g.node("SAM3Segment", image=image_ref, prompt=prompt_value,
                          output_mode="Merged", confidence_threshold=0.5,
                          max_segments=0, segment_pick=0, mask_blur=0,
                          mask_offset=0, device="Auto", invert_output=False,
                          unload_model=False, background="Alpha",
                          background_color="#222222")

        def segment(image_ref, prompt_value):
            """Text-prompt segmentation with an automatic non-CUDA fallback.
            SAM3 (open-vocab, needs triton/CUDA) is preferred; on a box without
            it we fall back to AtlasSemanticMask (SegFormer, CPU/MPS) so sky and
            scope still get a learned mask with no rewiring. Returns a MASK ref,
            or None when neither segmenter is installed (caller then drops to the
            heuristic). SAM3 mask is out(1); AtlasSemanticMask mask is out(0)."""
            if have_sam:
                return sam3(image_ref, prompt_value).out(1)
            if have_semantic:
                return g.node("AtlasSemanticMask", image=image_ref,
                              classes=prompt_value).out(0)
            return None

        if not have_sam and have_semantic:
            notes.append("SAM3 absent -> AtlasSemanticMask (SegFormer, CPU/MPS) "
                         "fallback for sky/scope")
```

with:

```python
        registry = _comfy_registry()
        # Native SAM3 (AtlasSAM3Mask, transformers>=5.5.4, no triton) fully
        # supersedes the third-party SAM3Segment (comfyui-rmbg) in Atlas's own
        # cascade — it works on CUDA/CPU/MPS alike, so there's no case where
        # preferring the triton-locked node is better. AtlasSemanticMask
        # (SegFormer/ADE20K, [neural], no triton) remains the learned fallback
        # for transformers<5.5.4 / [sam3] not installed.
        have_native_sam3 = _native_sam3_available()
        have_semantic = "AtlasSemanticMask" in registry
        have_inpaint = ("INPAINT_InpaintWithModel" in registry
                        and "INPAINT_LoadInpaintModel" in registry
                        and "INPAINT_ExpandMask" in registry)
        notes: list = []
        g = _graph_builder()

        def segment(image_ref, prompt_value):
            """Text-prompt segmentation with an automatic fallback cascade.
            Native SAM3 (AtlasSAM3Mask, transformers>=5.5.4, no triton) is
            preferred; on a box without it (or [sam3] not installed) we fall
            back to AtlasSemanticMask (SegFormer, CPU/MPS) so sky and scope
            still get a learned mask with no rewiring. Returns a MASK ref, or
            None when neither segmenter is available (caller then drops to
            the heuristic). Both AtlasSAM3Mask's and AtlasSemanticMask's mask
            are out(0)."""
            if have_native_sam3:
                return g.node("AtlasSAM3Mask", image=image_ref,
                              concepts=prompt_value).out(0)
            if have_semantic:
                return g.node("AtlasSemanticMask", image=image_ref,
                              classes=prompt_value).out(0)
            return None

        if not have_native_sam3 and have_semantic:
            notes.append("native SAM3 absent -> AtlasSemanticMask (SegFormer, CPU/MPS) "
                         "fallback for sky/scope")
```

- [ ] **Step 3: Update the "SKIPPED" notes**

Replace (currently line 747):
```python
                notes.append("sky SKIPPED — no segmenter (SAM3 / AtlasSemanticMask absent)")
```
with:
```python
                notes.append("sky SKIPPED — no segmenter (native SAM3 / AtlasSemanticMask absent)")
```

Replace (currently line 828):
```python
                            notes.append(f"{name} scope SKIPPED — no segmenter (SAM3 / AtlasSemanticMask absent)")
```
with:
```python
                            notes.append(f"{name} scope SKIPPED — no segmenter (native SAM3 / AtlasSemanticMask absent)")
```

- [ ] **Step 4: Update the docstrings/tooltips referencing `SAM3Segment`**

Replace (currently line 546, in the `AtlasInput` class docstring):
```python
    SAM3Segment / LaMa by registry name) so every inner step keeps its own
```
with:
```python
    LaMa by registry name; native SAM3 via AtlasSAM3Mask, our own node) so
    every inner step keeps its own
```

Replace (currently line 606, the `sky` widget tooltip):
```python
                "sky": ("BOOLEAN", {"default": False,
                    "tooltip": "SAM-segment the sky onto its own flat card, and feed the mask "
                               "into every mesh's exclude_mask + band_ref_mask. Needs "
                               "ComfyUI-RMBG (SAM3Segment) — skipped + noted if absent."}),
```
with:
```python
                "sky": ("BOOLEAN", {"default": False,
                    "tooltip": "SAM-segment the sky onto its own flat card, and feed the mask "
                               "into every mesh's exclude_mask + band_ref_mask. Uses native "
                               "SAM3 (transformers>=5.5.4, [sam3] extra) or falls back to "
                               "AtlasSemanticMask — skipped + noted if neither is available."}),
```

Replace (currently line 613, the `scope_prompts` widget tooltip's trailing sentence):
```python
                               "band-only automatically. The VLM's prompts win when use_vlm. "
                               "Needs ComfyUI-RMBG."}),
```
with:
```python
                               "band-only automatically. The VLM's prompts win when use_vlm. "
                               "Uses native SAM3 ([sam3] extra) or AtlasSemanticMask."}),
```

- [ ] **Step 5: Run the atlas_input tests to check for expected failures**

Run: `python -m pytest tests/test_atlas_input.py -v`
Expected: FAIL — several tests still reference `SAM3Segment`/`have_sam`/registry-based SAM gating, which Task 8 fixes. Confirm the failures are in the SAM-related tests specifically (`test_sky_and_scope_fall_back_to_semantic_mask_without_sam`, `test_sky_and_scope_wire_when_sam_present`, `test_vlm_wires_plan_and_forces_four_bands`), not unrelated ones.

- [ ] **Step 6: Commit**

```bash
git add atlas_camera/comfy/nodes_viewport.py
git commit -m "$(cat <<'EOF'
feat: rewire AtlasInput's segment() cascade to prefer native SAM3

Native SAM3 (AtlasSAM3Mask) now fully supersedes the third-party
SAM3Segment (comfyui-rmbg) in Atlas's own sky/scope cascade -- it works
on CUDA/CPU/MPS alike, so there's no case where the triton-locked node
is preferable. New cascade: native SAM3 -> AtlasSemanticMask ->
heuristic. tests/test_atlas_input.py is updated in the next commit.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 8: rework `tests/test_atlas_input.py` for the new cascade

**Files:**
- Modify: `tests/test_atlas_input.py`

- [ ] **Step 1: Update the module docstring and `_expand` helper**

Replace the module docstring (currently lines 1-7):

```python
"""Tests for AtlasInput 🎬 — the all-in-one expansion-wrapper entry node.

The expansion assembly is pure graph construction, so it's tested here via
the _MiniGraphBuilder shim (no ComfyUI needed): outside ComfyUI the registry
is {} which exercises exactly the graceful-degrade paths, and the SAM/inpaint
paths are exercised by monkeypatching the registry probe.
"""
```

with:

```python
"""Tests for AtlasInput 🎬 — the all-in-one expansion-wrapper entry node.

The expansion assembly is pure graph construction, so it's tested here via
the _MiniGraphBuilder shim (no ComfyUI needed): outside ComfyUI the registry
is {} which exercises exactly the graceful-degrade paths. The inpaint path is
exercised by monkeypatching the registry probe (_comfy_registry); the native
SAM3 path is exercised by monkeypatching the separate capability probe
(_native_sam3_available), since AtlasSAM3Mask is Atlas's own node and is
therefore always present in the registry regardless of whether its actual
[sam3] dependency is satisfied.
"""
```

Replace the comment above the `nodes_mod` import (currently lines 13-15):

```python
# AtlasInput lives in nodes_viewport after the nodes.py modularization; its
# node-expansion helpers (_comfy_registry) resolve in that module's namespace,
# so the registry-probe monkeypatch must target it there.
```

with:

```python
# AtlasInput lives in nodes_viewport after the nodes.py modularization; its
# node-expansion helpers (_comfy_registry, _native_sam3_available) resolve in
# that module's namespace, so both probe monkeypatches must target it there.
```

Replace `FULL_REGISTRY` (currently line 22) — drop the now-unused `SAM3Segment` key:

```python
FULL_REGISTRY = {"INPAINT_InpaintWithModel": object,
                 "INPAINT_LoadInpaintModel": object, "INPAINT_ExpandMask": object}
```

Replace the `_expand` helper (currently lines 29-34):

```python
def _expand(monkeypatch, registry=None, **kw):
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: registry or {})
    out = AtlasInput().build(IMG, **kw)
    assert set(out) == {"result", "expand"}
    _assert_atlas_inputs_valid(out["expand"])
    return out["expand"], out["result"]
```

with:

```python
def _expand(monkeypatch, registry=None, native_sam3=False, **kw):
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: registry or {})
    monkeypatch.setattr(nodes_mod, "_native_sam3_available", lambda: native_sam3)
    out = AtlasInput().build(IMG, **kw)
    assert set(out) == {"result", "expand"}
    _assert_atlas_inputs_valid(out["expand"])
    return out["expand"], out["result"]
```

- [ ] **Step 2: Update `test_sky_and_scope_skip_gracefully_without_any_segmenter`**

Replace (currently lines 126-137):

```python
def test_sky_and_scope_skip_gracefully_without_any_segmenter(monkeypatch):
    # Empty registry: neither SAM3 nor AtlasSemanticMask -> drop to heuristic.
    graph, result = _expand(monkeypatch, sky=True, layers=2,
                            scope_prompts="rocks\nperson")
    report = result[4]
    assert "sky SKIPPED" in report and "no segmenter" in report
    assert "scope SKIPPED" in report
    assert not any(n["class_type"] == "SAM3Segment" for n in graph.values())
    assert not any(n["class_type"] == "AtlasSemanticMask" for n in graph.values())
    # sky_mask output degrades to the SolidMask zero
    solid_id = next(i for i, n in graph.items() if n["class_type"] == "SolidMask")
    assert result[3] == [solid_id, 0]
```

with:

```python
def test_sky_and_scope_skip_gracefully_without_any_segmenter(monkeypatch):
    # Empty registry + native SAM3 unavailable -> drop to heuristic.
    graph, result = _expand(monkeypatch, sky=True, layers=2,
                            scope_prompts="rocks\nperson")
    report = result[4]
    assert "sky SKIPPED" in report and "no segmenter" in report
    assert "scope SKIPPED" in report
    assert not any(n["class_type"] == "AtlasSAM3Mask" for n in graph.values())
    assert not any(n["class_type"] == "AtlasSemanticMask" for n in graph.values())
    # sky_mask output degrades to the SolidMask zero
    solid_id = next(i for i, n in graph.items() if n["class_type"] == "SolidMask")
    assert result[3] == [solid_id, 0]
```

- [ ] **Step 3: Rename and update `test_sky_and_scope_fall_back_to_semantic_mask_without_sam`**

Replace (currently lines 140-160):

```python
def test_sky_and_scope_fall_back_to_semantic_mask_without_sam(monkeypatch):
    # Non-CUDA box: SAM3 (triton) can't load, but our SegFormer node can — sky
    # and scope must route to AtlasSemanticMask, not collapse to the heuristic.
    graph, result = _expand(monkeypatch, registry={"AtlasSemanticMask": object},
                            sky=True, layers=2, scope_prompts="rocks")
    report = result[4]
    assert "AtlasSemanticMask" in report and "fallback" in report
    assert not any(n["class_type"] == "SAM3Segment" for n in graph.values())
    sems = [n for n in graph.values() if n["class_type"] == "AtlasSemanticMask"]
    assert len(sems) == 2                        # sky + one scope line
    # The sky dome is actually built (not skipped), fed by the SegFormer mask.
    assert any(n["class_type"] == "AtlasSkyDomeLayer" for n in graph.values())
    assert any(n["inputs"].get("classes") == "sky" for n in sems)
    # Scope wires AtlasScopeMask off the SegFormer mask — which is out(0), not
    # SAM3's out(1).
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1
    rocks_id = next(i for i, n in graph.items()
                    if n["class_type"] == "AtlasSemanticMask"
                    and n["inputs"].get("classes") == "rocks")
    assert scopes[0]["inputs"]["segment_mask"] == [rocks_id, 0]
```

with:

```python
def test_sky_and_scope_fall_back_to_semantic_mask_without_native_sam3(monkeypatch):
    # Non-CUDA box, or [sam3] not installed: native SAM3 unavailable, but our
    # SegFormer node can still run — sky and scope must route to
    # AtlasSemanticMask, not collapse to the heuristic.
    graph, result = _expand(monkeypatch, registry={"AtlasSemanticMask": object},
                            sky=True, layers=2, scope_prompts="rocks")
    report = result[4]
    assert "AtlasSemanticMask" in report and "fallback" in report
    assert not any(n["class_type"] == "AtlasSAM3Mask" for n in graph.values())
    sems = [n for n in graph.values() if n["class_type"] == "AtlasSemanticMask"]
    assert len(sems) == 2                        # sky + one scope line
    # The sky dome is actually built (not skipped), fed by the SegFormer mask.
    assert any(n["class_type"] == "AtlasSkyDomeLayer" for n in graph.values())
    assert any(n["inputs"].get("classes") == "sky" for n in sems)
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1
    rocks_id = next(i for i, n in graph.items()
                    if n["class_type"] == "AtlasSemanticMask"
                    and n["inputs"].get("classes") == "rocks")
    assert scopes[0]["inputs"]["segment_mask"] == [rocks_id, 0]
```

- [ ] **Step 4: Rename and update `test_sky_and_scope_wire_when_sam_present`**

Replace (currently lines 163-180):

```python
def test_sky_and_scope_wire_when_sam_present(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, sky=True,
                            layers=2, scope_prompts="rocks")
    sams = [n for n in graph.values() if n["class_type"] == "SAM3Segment"]
    assert len(sams) == 2                        # sky + one scope line
    sky_layer = next(n for n in graph.values()
                     if n["class_type"] == "AtlasSkyDomeLayer")
    # Generous sky smear (96/128) so ridge-silhouette reveals never go black.
    assert sky_layer["inputs"]["edge_extend_px"] == 96
    assert sky_layer["inputs"]["frame_outpaint_px"] == 128
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1 and scopes[0]["inputs"]["prompt"] == "rocks"
    # sky mask feeds band_ref_mask on every band layer (the drift rule)
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    sky_sam_id = next(i for i, n in graph.items()
                      if n["class_type"] == "SAM3Segment"
                      and n["inputs"]["prompt"] == "sky")
    assert all(b["inputs"].get("band_ref_mask") == [sky_sam_id, 1] for b in bands)
```

with:

```python
def test_sky_and_scope_wire_when_native_sam3_available(monkeypatch):
    graph, result = _expand(monkeypatch, native_sam3=True, sky=True,
                            layers=2, scope_prompts="rocks")
    sams = [n for n in graph.values() if n["class_type"] == "AtlasSAM3Mask"]
    assert len(sams) == 2                        # sky + one scope line
    sky_layer = next(n for n in graph.values()
                     if n["class_type"] == "AtlasSkyDomeLayer")
    # Generous sky smear (96/128) so ridge-silhouette reveals never go black.
    assert sky_layer["inputs"]["edge_extend_px"] == 96
    assert sky_layer["inputs"]["frame_outpaint_px"] == 128
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1 and scopes[0]["inputs"]["prompt"] == "rocks"
    # sky mask feeds band_ref_mask on every band layer (the drift rule)
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    sky_sam_id = next(i for i, n in graph.items()
                      if n["class_type"] == "AtlasSAM3Mask"
                      and n["inputs"]["concepts"] == "sky")
    assert all(b["inputs"].get("band_ref_mask") == [sky_sam_id, 0] for b in bands)
```

Note the two deliberate changes: no `registry=FULL_REGISTRY` (native SAM3 availability is no longer registry-gated), and `band_ref_mask == [sky_sam_id, 0]` (`AtlasSAM3Mask`'s mask is its own `out(0)`, unlike third-party `SAM3Segment`'s `Merged` mode which put it at `out(1)`).

- [ ] **Step 5: Update `test_vlm_wires_plan_and_forces_four_bands`**

Replace (currently lines 211-233):

```python
def test_vlm_wires_plan_and_forces_four_bands(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, use_vlm=True,
                            layers=2, sky=True)
    assess_id = next(i for i, n in graph.items()
                     if n["class_type"] == "AtlasAssessImage")
    assess = graph[assess_id]
    assert assess["inputs"]["auto_continue"] is True
    assert assess["inputs"]["offload_model"] is True
    assert result[1] == [assess_id, 0]           # image flows THROUGH the assess node
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    assert len(bands) == 4                       # forced (VLM plan = 4 band slots)
    assert "layers 2 → 4" in result[4]
    # band + geometry overrides come from the assess node's outputs 12..15 / 8..11
    band_refs = sorted(b["inputs"]["band_override"][1] for b in bands)
    geom_refs = sorted(b["inputs"]["geometry_override"][1] for b in bands)
    assert band_refs == [12, 13, 14, 15]
    assert geom_refs == [8, 9, 10, 11]
    assert all(b["inputs"]["band_override"][0] == assess_id for b in bands)
    # sky SAM prompt comes from the plan too (output 3)
    sky_sam = next(n for n in graph.values() if n["class_type"] == "SAM3Segment"
                   and isinstance(n["inputs"]["prompt"], list)
                   and n["inputs"]["prompt"][1] == 3)
    assert sky_sam["inputs"]["prompt"][0] == assess_id
```

with:

```python
def test_vlm_wires_plan_and_forces_four_bands(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, native_sam3=True,
                            use_vlm=True, layers=2, sky=True)
    assess_id = next(i for i, n in graph.items()
                     if n["class_type"] == "AtlasAssessImage")
    assess = graph[assess_id]
    assert assess["inputs"]["auto_continue"] is True
    assert assess["inputs"]["offload_model"] is True
    assert result[1] == [assess_id, 0]           # image flows THROUGH the assess node
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    assert len(bands) == 4                       # forced (VLM plan = 4 band slots)
    assert "layers 2 → 4" in result[4]
    # band + geometry overrides come from the assess node's outputs 12..15 / 8..11
    band_refs = sorted(b["inputs"]["band_override"][1] for b in bands)
    geom_refs = sorted(b["inputs"]["geometry_override"][1] for b in bands)
    assert band_refs == [12, 13, 14, 15]
    assert geom_refs == [8, 9, 10, 11]
    assert all(b["inputs"]["band_override"][0] == assess_id for b in bands)
    # sky SAM prompt comes from the plan too (output 3)
    sky_sam = next(n for n in graph.values() if n["class_type"] == "AtlasSAM3Mask"
                   and isinstance(n["inputs"]["concepts"], list)
                   and n["inputs"]["concepts"][1] == 3)
    assert sky_sam["inputs"]["concepts"][0] == assess_id
```

- [ ] **Step 6: Run the full test_atlas_input.py suite**

Run: `python -m pytest tests/test_atlas_input.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_atlas_input.py
git commit -m "$(cat <<'EOF'
test: rework test_atlas_input.py for the native SAM3 cascade

Replaces the registry-mocked SAM3Segment assertions with the new
_native_sam3_available probe + AtlasSAM3Mask class_type/concepts/out(0)
shape, covering all three cascade tiers: native available / native
absent + SegFormer available / neither available.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 9: documentation — `INSTALL.md` and `CLAUDE.md`

**Files:**
- Modify: `INSTALL.md:436-448`
- Modify: `CLAUDE.md` (node catalog table row after `AtlasSemanticMask`, and the `AtlasInput` design-rule bullet)

- [ ] **Step 1: Update `INSTALL.md`'s sky/scope segmentation bullet**

Read `INSTALL.md` around line 431-448 to confirm current line numbers, then replace the "Sky / scope segmentation" bullet (currently lines 436-448):

```markdown
- **Sky / scope segmentation** — [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)
  provides the `SAM3Segment` node (prompt it with `sky`, `buildings`, etc.).
  Its MASK output feeds `AtlasSkyDomeLayer.sky_mask` AND every layer node's
  `exclude_mask` (a real segmentation replaces Atlas's internal sky heuristic).
  **SAM3 requires `triton`** (CUDA-only). On **Windows + NVIDIA**, install it
  into the ComfyUI env — `python_embeded\python.exe -m pip install triton-windows`
  — or `SAM3Segment` fails to load ("No module named 'triton'", Manager reports
  the pack "missing"). **On Mac (MPS) / CPU / AMD there is no triton**, so SAM3
  can't run there at all: `AtlasInput` automatically falls back to
  **`AtlasSemanticMask`** (SegFormer/ADE20K, `[neural]`, no triton — a learned
  CPU/MPS sky/scope mask), and the numpy sky heuristic is the zero-dependency
  floor. Grounded-SAM2 (GroundingDINO + a SAM2 pack) is an optional premium
  Mac tier for SAM-grade edges, at the cost of two extra models.
```

with:

```markdown
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
```

- [ ] **Step 2: Add the `AtlasSAM3Mask` catalog row to `CLAUDE.md`**

Find the `AtlasSemanticMask` table row in `CLAUDE.md` (search for `` | `AtlasSemanticMask` | ``) and insert a new row immediately after it:

```markdown
| `AtlasSAM3Mask` | image (IMAGE), concepts (STRING, default "sky"), ±confidence_threshold, ±device | mask (MASK), report (STRING) | 🪄 Native SAM3 concept mask via `transformers>=5.5.4` (`[sam3]` extra) — no `triton`/comfyui-rmbg dependency, so it works on CUDA, CPU, and Mac (MPS) alike, unlike the third-party `SAM3Segment` node it supersedes in `AtlasInput`'s own cascade. Same interface shape as `AtlasSemanticMask` (comma-separated concepts → union mask + report), which remains the learned fallback tier when `transformers<5.5.4`. `facebook/sam3` is gated on Hugging Face (Meta's SAM-License-1.0) — one-time `hf auth login` after requesting access; a gated-repo failure is caught and returned as the report string rather than raised (a one-time auth step, not a broken install), while version/import errors still raise normally. Inspired by lettidude/LiveActionAOV's `passes/matte/sam3.py`. |
```

- [ ] **Step 3: Update the `AtlasInput` design-rule bullet**

Find the paragraph beginning `- **`AtlasInput` — the all-in-one entry node is a NODE-EXPANSION wrapper` (search for `Automatic non-CUDA segmentation fallback`). Replace this sentence:

```
**Automatic non-CUDA segmentation fallback (`segment()` helper, 2026-07-13):** `SAM3Segment` (comfyui-rmbg) hard-requires `triton`, which does not exist on Mac(MPS)/CPU/AMD — those users can never load it, and the sky/scope steps used to silently drop to the bare numpy heuristic. `build()` now routes ALL text-prompt segmentation through one `segment(image_ref, prompt)` cascade: SAM3 when in the registry (open-vocab, GPU) → else `AtlasSemanticMask` (SegFormer/ADE20K, `[neural]`, NO triton — a learned CPU/MPS mask, our own node, ~always present) → else None (heuristic floor). Non-CUDA users get a learned sky/scope mask with no rewiring; the `report` states which path fired. **CUDA users are unaffected** (SAM3 still preferred). Watch the output slots: SAM3 mask is `out(1)`, `AtlasSemanticMask` mask is `out(0)`. Windows+NVIDIA users who want SAM3 install `triton-windows` (INSTALL.md). Covered by `test_atlas_input.py` (no-segmenter skip + SegFormer-fallback wiring).
```

with:

```
**Segmentation cascade (`segment()` helper, 2026-07-13, superseded 2026-07-20):** the third-party `SAM3Segment` (comfyui-rmbg) hard-requires `triton`, which does not exist on Mac(MPS)/CPU/AMD — those users could never load it, and the sky/scope steps used to silently drop to the bare numpy heuristic. `build()` now routes ALL text-prompt segmentation through one `segment(image_ref, prompt)` cascade: **native `AtlasSAM3Mask`** (`transformers>=5.5.4`, `[sam3]` extra, no triton — works on CUDA/CPU/MPS alike, feature-detected via `node_helpers._native_sam3_available()` rather than the registry, since it's Atlas's own node and is therefore always registered regardless of whether its actual dependency is satisfied) → else `AtlasSemanticMask` (SegFormer/ADE20K, `[neural]`, also no triton — a learned CPU/MPS mask) → else `None` (heuristic floor). The third-party `SAM3Segment` is no longer part of this cascade at all — see the `AtlasSAM3Mask` catalog entry. Both `AtlasSAM3Mask`'s and `AtlasSemanticMask`'s mask are `out(0)`. Covered by `test_atlas_input.py` (no-segmenter skip, SegFormer-fallback wiring, and native-SAM3-available wiring).
```

- [ ] **Step 4: Verify the doc edits don't break anything mechanical**

Run: `python -m pytest tests/test_example_workflows.py tests/test_frontend_mirrors.py -v`
Expected: PASS (these tests don't read CLAUDE.md/INSTALL.md content directly, but confirm nothing else was accidentally touched)

- [ ] **Step 5: Commit**

```bash
git add INSTALL.md CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: document native SAM3 (AtlasSAM3Mask) and the updated cascade

INSTALL.md's sky/scope segmentation section now leads with the [sam3]
extra + one-time HF gated-repo auth steps, with SAM3Segment demoted to
"still works if manually wired, no longer preferred." CLAUDE.md gets a
new AtlasSAM3Mask catalog row and an updated AtlasInput cascade bullet.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015v8ftd43keoowMAtNuda4R
EOF
)"
```

---

## Task 10: full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests PASS, no unexpected skips beyond the usual torch-optional skips

- [ ] **Step 2: Run the specific new/changed test files once more in isolation for a clean signal**

Run: `python -m pytest tests/test_sam3_segmenter.py tests/test_sam3_mask_node.py tests/test_atlas_input.py tests/test_comfy_node_registry.py -v`
Expected: all PASS

- [ ] **Step 3: Confirm the working tree is clean**

Run: `git status --short`
Expected: empty (everything committed across Tasks 1-9)

- [ ] **Step 4: Sanity-check the new node imports cleanly outside a real transformers install**

Run: `python -c "from atlas_camera.comfy.nodes import AtlasSAM3Mask, NODE_CLASS_MAPPINGS; print('AtlasSAM3Mask' in NODE_CLASS_MAPPINGS); print(AtlasSAM3Mask.INPUT_TYPES())"`
Expected: prints `True` followed by the INPUT_TYPES dict — the class itself must import and be introspectable with zero heavy dependencies actually installed (the lazy-import contract).

No commit for this task — it's verification only. If any step fails, fix the underlying issue in the task where it was introduced and re-run this task's steps.
