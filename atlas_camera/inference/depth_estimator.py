"""Monocular depth estimation (Depth Anything V2).

Provides an independent, learned depth map for the ``depth`` LatentComponent and,
combined with the learned camera orientation, lets Atlas *measure* camera height
by fitting the ground plane — instead of assuming a default eye height.

Heavy dependencies (torch + transformers) are imported lazily so the core package
stays dependency-free. Install with:  pip install -e .[neural]

Model variants (Hugging Face):
  - relative:  ``depth-anything/Depth-Anything-V2-Small-hf`` (fast, up-to-scale)
  - metric indoor:  ``depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf``
  - metric outdoor: ``depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf``

Only *metric* models yield depth in meters (needed for absolute camera height).
Relative depth still recovers the ground plane and camera height up to an unknown
global scale.

Depth Anything 3 (opt-in second backend, ``pip install -e .[neural-da3]``):
  - metric:   ``depth-anything/DA3METRIC-LARGE`` (canonical depth -> meters via the
    focal length; pass ``focal_px`` from the Atlas solve to close the loop, else an
    assumed normal-lens focal is used — the model itself predicts no intrinsics)
  - relative: ``depth-anything/DA3MONO-LARGE`` (up-to-scale, larger = farther)
  - metric:   ``depth-anything/DA3NESTED-GIANT-LARGE-1.1`` (already meters;
    CC BY-NC 4.0 — non-commercial license)
DA3 model ids dispatch to the ``depth_anything_3`` package (GitHub-only) instead of
transformers; everything else (DepthResult contract, caching) is shared.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.inference._common import bounded_cache_set, resolve_device


# Model ids that emit metric (meters) depth rather than up-to-scale relative depth.
_METRIC_HINT = "metric"

DEFAULT_RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_METRIC_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DEFAULT_METRIC_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

# Depth Anything 3 model ids (opt-in backend — see module docstring).
DA3_METRIC_MODEL = "depth-anything/DA3METRIC-LARGE"
DA3_MONO_MODEL = "depth-anything/DA3MONO-LARGE"
DA3_NESTED_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# DA3METRIC emits canonical depth normalised by this constant: metres = focal_px * out / 300.
_DA3_CANONICAL_FOCAL_NORM = 300.0

# Relative (disparity) models: normalised disparity is floored here before the
# reciprocal depth conversion — a 25:1 depth-ratio cap that keeps the sky /
# horizon tail from blowing the dynamic range. Everything at or below the
# floor lands on ONE far plane; the fraction that did is recorded in
# DepthResult.metadata["floored_fraction"].
_DISPARITY_FLOOR = 0.04


def _is_da3_model(model_id: str) -> bool:
    """True for Depth Anything 3 ids (``depth-anything/DA3...``); no V2 id matches."""
    return "/da3" in model_id.lower()


def _require_depth_backend() -> tuple[Any, Any, Any]:
    """Import torch + transformers depth-estimation classes lazily."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Monocular depth estimation requires torch and transformers. Install with:\n"
            "    pip install -e .[neural]"
        ) from exc
    return torch, AutoImageProcessor, AutoModelForDepthEstimation


def _require_torch() -> Any:
    """Import torch alone lazily (the DA3 path needs no transformers)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Monocular depth estimation requires torch. Install with:\n"
            "    pip install -e .[neural]"
        ) from exc
    return torch


# DA3's ``api`` module eagerly imports its gaussian-splat/COLMAP export, trajectory
# eval, and plotting stack (gsplat, open3d, e3nn, pycolmap, trimesh, plyfile,
# pillow_heif, evo, matplotlib) — none of which the depth forward pass touches, and
# several of which have no wheels for recent torch/Python on Windows. When the
# package is installed with ``--no-deps`` (the documented ComfyUI route, see
# INSTALL.md) those imports would abort the whole load. We fabricate the missing
# export-only modules on demand so inference works without the heavy 3D stack.
#
# ``xformers`` is deliberately NOT stubbed: DINOv2 guards it with
# ``try: from xformers.ops import ... ; XFORMERS_AVAILABLE = True / except
# ImportError: False`` and then uses it when available — a stub would flip that flag
# true and route attention through a fake module, corrupting inference. Left absent,
# the guard correctly falls back to standard attention.
_DA3_EXPORT_ONLY_ROOTS = (
    "gsplat", "open3d", "e3nn", "pycolmap", "trimesh",
    "plyfile", "pillow_heif", "evo", "matplotlib",
)
_DA3_STUBS_INSTALLED = False


class _DA3StubAny:
    """Permissive placeholder: any attribute access or call yields another one.

    Enough for ``import x`` / ``from x import Y`` / class-body attribute reads to
    succeed at import time. Never actually invoked during a depth forward pass.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> "_DA3StubAny":
        return _DA3StubAny()

    def __getattr__(self, name: str) -> "_DA3StubAny":
        # Dunders must read as genuinely absent so ``inspect``/``torch.library``
        # stack-walking over sys.modules (e.g. probing ``__file__``) sees None
        # instead of a stub object and does not crash.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DA3StubAny()


def _install_da3_export_stubs() -> None:
    """Register a meta-path finder for DA3's export-only deps, once, if absent.

    Only roots that are genuinely not installed are stubbed, so a user's real
    matplotlib (etc.) is never shadowed.
    """
    global _DA3_STUBS_INSTALLED
    if _DA3_STUBS_INSTALLED:
        return

    import sys
    import types
    import importlib.abc
    import importlib.machinery
    import importlib.util

    absent = set()
    for root in _DA3_EXPORT_ONLY_ROOTS:
        try:
            if importlib.util.find_spec(root) is None:
                absent.add(root)
        except (ImportError, ValueError):
            absent.add(root)

    if absent:

        class _StubLoader(importlib.abc.Loader):
            def create_module(self, spec):  # type: ignore[override]
                module = types.ModuleType(spec.name)
                module.__path__ = []  # treat as a package so submodules resolve

                def _stub_getattr(name):
                    # Absent dunders (e.g. __file__) must raise so introspection
                    # over sys.modules treats them as unset, not as a stub object.
                    if name.startswith("__") and name.endswith("__"):
                        raise AttributeError(name)
                    return _DA3StubAny()

                module.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
                return module

            def exec_module(self, module):  # type: ignore[override]
                pass

        class _StubFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):  # type: ignore[override]
                if name.split(".")[0] in absent:
                    return importlib.machinery.ModuleSpec(
                        name, _StubLoader(), is_package=True
                    )
                return None

        sys.meta_path.insert(0, _StubFinder())

    _DA3_STUBS_INSTALLED = True


def _require_da3() -> Any:
    """Import the Depth Anything 3 API lazily with an informative error."""
    _install_da3_export_stubs()
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Depth Anything 3 models require the depth_anything_3 package. Install with:\n"
            "    pip install -e .[neural-da3]\n"
            "(GitHub-only: pip install "
            "'git+https://github.com/ByteDance-Seed/Depth-Anything-3.git')"
        ) from exc
    return DepthAnything3


@dataclass(slots=True)
class DepthResult:
    """A recovered depth map plus provenance.

    ``depth`` is a HxW float32 numpy array of forward distance. For metric models
    the unit is metres; for relative models it is an arbitrary (up-to-scale) unit
    where larger = farther. ``is_metric`` distinguishes the two.
    """

    depth: Any  # numpy.ndarray HxW float32
    is_metric: bool
    model_id: str
    image_width: int
    image_height: int
    near: float = 0.0
    far: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary (no heavy array) for the depth LatentComponent."""
        return {
            "model_id": self.model_id,
            "is_metric": self.is_metric,
            "unit": "meters" if self.is_metric else "relative",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "near": float(self.near),
            "far": float(self.far),
            **self.metadata,
        }


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_MODEL_CACHE_MAX = 4  # each entry holds a full loaded depth model; bound VRAM growth

# Cross-call depth-RESULT cache (distinct from _MODEL_CACHE above, which only
# caches loaded weights). Several ComfyUI nodes independently call
# estimate_depth() on the same photo with no way to share a result across
# nodes (only AtlasDepthMap-based composable nodes share via the ATLAS_DEPTH_MAP
# type) — e.g. the project's own simplest example workflow runs full
# depth-model inference twice on the identical image. Keyed by image content
# hash (not path — nodes routinely save the same tensor to a fresh temp file
# per call, so path-based caching would never hit) + model + device.
_DEPTH_RESULT_CACHE: dict[tuple[str, str, str, float | None], "DepthResult"] = {}
_DEPTH_RESULT_CACHE_MAX = 8

# DA3 models are cached separately: _MODEL_CACHE values are (processor, model)
# tuples from transformers, while a DA3 entry is the bare DepthAnything3 module.
# Max 2 — the nested giant alone is 1.4B params.
_DA3_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_DA3_MODEL_CACHE_MAX = 2


def _record_and_clamp_negative(depth: Any, metadata: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Record the negative-pixel fraction, then clamp metric depth at >= 0.

    The DA3 backend occasionally emits negative raw depth (documented
    watch-item). Downstream validity masks already exclude negatives, but the
    3x3 median filter and edge tests still ingest them as NEIGHBORS — clamping
    at the source protects every consumer, and recording first keeps the
    diagnostic (`AtlasDebugReport` reads metadata['negative_fraction'] as the
    pre-clamp truth).
    """
    import numpy as np

    neg = float((depth < 0).mean()) if depth.size else 0.0
    metadata["negative_fraction"] = round(neg, 6)
    if neg > 0:
        depth = np.maximum(depth, 0.0)
    return depth, metadata


def _disparity_to_depth(disparity: Any, metadata: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Convert a relative model's disparity map to normalised depth ([0,1],
    larger = farther).

    Disparity is proportional to 1/depth, so the conversion must be
    RECIPROCAL — the pre-audit linear `1 - d` flip was rank-preserving but
    systematically warped spacing (near range compressed, far stretched).
    Normalised disparity is floored at `_DISPARITY_FLOOR` (a 25:1 depth-ratio
    cap) so the sky/horizon tail doesn't blow the dynamic range; everything
    at/below the floor collapses to ONE far plane, and the fraction that did
    is recorded in metadata["floored_fraction"].

    Pure numpy, extracted from the V2 inference path per code review so the
    spacing behavior is pinnable without model weights.
    """
    import numpy as np

    d = disparity - disparity.min()
    d = d / (d.max() or 1.0)
    inv = 1.0 / np.maximum(d, _DISPARITY_FLOOR)
    inv -= inv.min()
    depth = (inv / (inv.max() or 1.0)).astype(np.float32)
    metadata["disparity_floor"] = _DISPARITY_FLOOR
    metadata["floored_fraction"] = round(float((d <= _DISPARITY_FLOOR).mean()), 6)
    return depth, metadata


def _get_model(model_id: str, device: str):
    cached = _MODEL_CACHE.get((model_id, device))
    if cached is not None:
        return cached
    torch, AutoImageProcessor, AutoModelForDepthEstimation = _require_depth_backend()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    bounded_cache_set(_MODEL_CACHE, (model_id, device), (processor, model), _MODEL_CACHE_MAX,
                      release_cuda=True)
    return processor, model


def _get_da3_model(model_id: str, device: str):
    cached = _DA3_MODEL_CACHE.get((model_id, device))
    if cached is not None:
        return cached
    DepthAnything3 = _require_da3()
    model = DepthAnything3.from_pretrained(model_id).to(device=device).eval()
    bounded_cache_set(_DA3_MODEL_CACHE, (model_id, device), model, _DA3_MODEL_CACHE_MAX,
                      release_cuda=True)
    return model


def _da3_metric_from_canonical(
    net_depth: Any,
    *,
    focal_px: float | None,
    source_width: int,
    processed_width: int,
    predicted_focal: float | None,
) -> tuple[Any, str, float]:
    """Convert DA3METRIC canonical depth to metres.

    ``metres = focal_at_processed_res * canonical / 300``. ``focal_px`` is the
    solve's focal in SOURCE-image pixels, so it is rescaled by
    ``processed_width / source_width`` first (DA3 resizes aspect-preserving, so
    the width ratio applies to fy too). When no solve focal is supplied, the
    model's own predicted intrinsics (already at processed resolution) are used.
    Returns ``(depth_m, focal_source, focal_px_processed)``.
    """
    if focal_px is not None and focal_px > 0:
        f = float(focal_px) * (processed_width / max(source_width, 1))
        source = "solve"
    elif predicted_focal is not None and predicted_focal > 0:
        f = float(predicted_focal)
        source = "predicted"
    else:  # pragma: no cover - DA3METRIC always predicts intrinsics
        raise ValueError(
            "DA3 metric conversion needs a focal length: pass focal_px or use a "
            "model that predicts intrinsics."
        )
    return net_depth * (f / _DA3_CANONICAL_FOCAL_NORM), source, f


def _estimate_depth_da3(
    image_path: str | Path,
    *,
    model_id: str,
    device: str,
    focal_px: float | None,
) -> DepthResult:
    """DA3 inference path: canonical/metric/relative branch per model family."""
    torch = _require_torch()
    import numpy as np
    from PIL import Image

    model = _get_da3_model(model_id, device)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Defensive no-grad: the V2 path guards this explicitly; if the DA3
    # package's inference ever runs ungated, autograd would silently
    # accumulate activation memory on a card that's already contended.
    with torch.inference_mode():
        prediction = model.inference([np.asarray(image)])
    net = np.asarray(prediction.depth[0], dtype=np.float32)
    proc_h, proc_w = net.shape

    metadata: dict[str, Any] = {
        "device": device,
        "backend": "da3",
        "processed_width": int(proc_w),
        "processed_height": int(proc_h),
    }
    conf = getattr(prediction, "conf", None)
    if conf is not None:
        metadata["conf_mean"] = float(np.mean(conf[0]))

    lower = model_id.lower()
    if "da3metric" in lower:
        intrinsics = getattr(prediction, "intrinsics", None)
        predicted_focal = None
        if intrinsics is not None:
            k = np.asarray(intrinsics[0], dtype=np.float64)
            predicted_focal = 0.5 * (float(k[0, 0]) + float(k[1, 1]))
        # DA3METRIC-LARGE is a depth-only head — confirmed live: it returns
        # intrinsics=None (only the main/nested series predicts cameras). With
        # no solve focal either, fall back to an assumed normal-lens focal
        # (f = processed width ~= 53 deg hFOV); downstream ground-pinning
        # (estimate_ground_scale) re-normalizes the metric scale anyway.
        assumed = predicted_focal is None or predicted_focal <= 0
        if assumed and (focal_px is None or focal_px <= 0):
            predicted_focal = float(proc_w)
        depth, focal_source, f_used = _da3_metric_from_canonical(
            net,
            focal_px=focal_px,
            source_width=width,
            processed_width=proc_w,
            predicted_focal=predicted_focal,
        )
        is_metric = True
        if focal_source == "predicted" and assumed:
            focal_source = "assumed"
        metadata["focal_source"] = focal_source
        metadata["focal_px_processed"] = float(f_used)
    elif "da3nested" in lower:
        depth = net  # already metres
        is_metric = True
    else:
        # DA3MONO predicts relative DEPTH (larger = farther) — unlike V2's
        # disparity output, no inversion is needed; just normalise to [0, 1].
        d = net - net.min()
        depth = d / (d.max() or 1.0)
        is_metric = False

    if depth.shape != (height, width):
        # BILINEAR, deliberately not bicubic: depth is not a photograph.
        # Bicubic RINGS at discontinuities — at a silhouette edge it
        # overshoots below the local minimum, which on metric maps produces
        # negative-depth halos exactly where meshes tear (observed live as
        # depth.near = -11.4m on a ridge shot). Bilinear cannot overshoot.
        t = torch.from_numpy(np.ascontiguousarray(depth))[None, None]
        depth = (
            torch.nn.functional.interpolate(
                t, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0]
            .numpy()
            .astype(np.float32)
        )

    if is_metric:
        depth, metadata = _record_and_clamp_negative(depth, metadata)

    return DepthResult(
        depth=depth,
        is_metric=is_metric,
        model_id=model_id,
        image_width=width,
        image_height=height,
        near=float(depth.min()),
        far=float(depth.max()),
        metadata=metadata,
    )


def estimate_depth(
    image_path: str | Path,
    *,
    model_id: str = DEFAULT_METRIC_OUTDOOR,
    device: str | None = None,
    focal_px: float | None = None,
) -> DepthResult:
    """Predict a depth map for a single image (Depth Anything V2 or 3).

    Returns forward distance (metres for metric models). The map is resized back
    to the source image resolution. ``focal_px`` (the solve's focal length in
    source-image pixels) is consumed only by DA3METRIC — it converts canonical
    depth to metres using the *solved* focal instead of the model's own predicted
    intrinsics; every other model ignores it.
    """
    torch = _require_torch()

    device = resolve_device(device, torch)

    content_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    # Only the model family that consumes focal_px fragments the cache on it.
    focal_key = (
        round(float(focal_px), 3)
        if (focal_px and "da3metric" in model_id.lower())
        else None
    )
    cache_key = (content_hash, model_id, device, focal_key)
    cached_result = _DEPTH_RESULT_CACHE.get(cache_key)
    if cached_result is not None:
        return cached_result

    if _is_da3_model(model_id):
        result = _estimate_depth_da3(
            image_path, model_id=model_id, device=device, focal_px=focal_px
        )
    else:
        result = _estimate_depth_v2(image_path, model_id=model_id, device=device)
    bounded_cache_set(_DEPTH_RESULT_CACHE, cache_key, result, _DEPTH_RESULT_CACHE_MAX)
    return result


def _estimate_depth_v2(
    image_path: str | Path,
    *,
    model_id: str,
    device: str,
) -> DepthResult:
    """Depth Anything V2 inference path (transformers), unchanged behavior."""
    torch, _, _ = _require_depth_backend()
    from PIL import Image

    processor, model = _get_model(model_id, device)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted = outputs.predicted_depth  # (1, h', w')
    if predicted.dim() == 3:
        predicted = predicted.unsqueeze(1)
    predicted = torch.nn.functional.interpolate(
        # Bilinear, not bicubic — see the DA3 path's comment: bicubic rings
        # at depth discontinuities and can overshoot into negative halos.
        predicted, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]

    is_metric = _METRIC_HINT in model_id.lower()
    depth = predicted.detach().float().cpu().numpy()
    metadata: dict[str, Any] = {"device": device}

    if not is_metric:
        # Relative models emit DISPARITY (larger = closer) — reciprocal
        # conversion + floor cap; see _disparity_to_depth for the full story.
        depth, metadata = _disparity_to_depth(depth, metadata)
    else:
        depth, metadata = _record_and_clamp_negative(depth, metadata)

    near = float(depth.min())
    far = float(depth.max())
    return DepthResult(
        depth=depth,
        is_metric=is_metric,
        model_id=model_id,
        image_width=width,
        image_height=height,
        near=near,
        far=far,
        metadata=metadata,
    )
