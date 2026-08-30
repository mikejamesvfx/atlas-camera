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

Depth Pro (``apple/DepthPro-hf``, transformers >= 4.48): metric depth AND a
predicted focal length / horizontal FOV from the model's own FOV head. The focal
estimate rides in ``metadata["predicted_focal_px"]`` (source-image pixel scale)
purely as PROVENANCE — the verdict on whether it agrees with the solve's
intrinsics belongs to ``core.scene_health`` (the single red-flag evaluator),
never here. Depth Pro ignores ``focal_px`` input (it predicts its own), so it
does not fragment the result cache on focal. Weights ship under the Apple ML
research license (non-commercial) — see the combo annotation in node_helpers.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.core.image_tiling import (       # noqa: F401  (re-exported)
    _feather_weights,
    assemble_tiles,
    fit_affine_to_reference,
    tile_boxes,
)
from atlas_camera.inference._common import bounded_cache_set, resolve_device


# Model ids that emit metric (meters) depth rather than up-to-scale relative depth.
_METRIC_HINT = "metric"

DEFAULT_RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_METRIC_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DEFAULT_METRIC_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

#: MoGe's own default token budget (0-9). Ours too — a lower value trades
#: detail for speed and MUST stay opt-in, or every existing graph silently
#: changes quality.
MOGE_RESOLUTION_LEVEL_DEFAULT = 9
# Resolution level for the optional fov-free SECOND pass that only harvests
# MoGe's predicted intrinsics (report_free_focal). The focal head is stable
# well below full token budget; 6 keeps the extra pass cheap.
MOGE_FREE_FOCAL_RESOLUTION_LEVEL = 6

# Depth Anything 3 model ids (opt-in backend — see module docstring).
DA3_METRIC_MODEL = "depth-anything/DA3METRIC-LARGE"
DA3_MONO_MODEL = "depth-anything/DA3MONO-LARGE"
DA3_NESTED_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# DA3METRIC emits canonical depth normalised by this constant: metres = focal_px * out / 300.
_DA3_CANONICAL_FOCAL_NORM = 300.0

# Apple Depth Pro (transformers backend): metric depth + predicted focal/FOV.
DEPTH_PRO_MODEL = "apple/DepthPro-hf"

# Lotus-2 (arXiv 2512.01030): a FLUX.1-dev backbone LoRA-finetuned for dense
# geometry — a core predictor doing single-step regression plus an optional
# multi-step "detail sharpener" constrained to the predictor's manifold. Paper
# reports avg. depth rank 3.6 against DA-V2's 7.3, MoGe-2's 10.4, Marigold's 9.2.
#
# RELATIVE, not metric, and it runs at 1024 (auto: >1024 -> 1024, <512 -> 512,
# else native), so it sits between DA-V2's 518 and Depth Pro's native 1536.
#
# TWO LICENCES, and they differ: Lotus-2's own code/weights are Apache-2.0, but it
# loads `black-forest-labs/FLUX.1-dev` as its base, which is NON-COMMERCIAL. That
# is why this is opt-in behind a local clone rather than an auto-download like the
# MIT-licensed MoGe entries — selecting it must be a deliberate act.
LOTUS2_MODEL = "jingheya/Lotus-2"

# Lotus-2 is a repo, not a pip package: `from pipeline import Lotus2Pipeline` is a
# repo-LOCAL import, so the clone root goes on sys.path. Same shape as the Fixer
# integration's ATLAS_FIXER_PATH.
LOTUS2_PATH_ENV = "ATLAS_LOTUS2_PATH"
LOTUS2_FLUX_BASE = "black-forest-labs/FLUX.1-dev"
LOTUS2_DEFAULT_STEPS = 10

# Free-VRAM threshold below which the pipeline is CPU-offloaded instead of moved
# wholesale to the GPU. The bf16 FLUX transformer is ~24 GB; 26 GB leaves room for
# the VAE, text encoders and activations. Measured live: a 32 GB RTX 5090 with an
# ordinary ComfyUI session running had 15.0 GB free, so the un-offloaded path
# cannot be the default for a node that lives inside ComfyUI.
LOTUS2_VRAM_BYTES = 26 * 1024**3

# Relative (disparity) models: normalised disparity is floored here before the
# reciprocal depth conversion — a 25:1 depth-ratio cap that keeps the sky /
# horizon tail from blowing the dynamic range. Everything at or below the
# floor lands on ONE far plane; the fraction that did is recorded in
# DepthResult.metadata["floored_fraction"].
_DISPARITY_FLOOR = 0.04


def _is_da3_model(model_id: str) -> bool:
    """True for Depth Anything 3 ids (``depth-anything/DA3...``); no V2 id matches."""
    return "/da3" in model_id.lower()


def _is_moge_model(model_id: str) -> bool:
    """True for MoGe ids (e.g. ``Ruicheng/moge-2-vitl-normal``). MIT-licensed,
    light-dependency alternative to the DA3 backend (canonical depth + normals)."""
    return "moge" in model_id.lower()


def _is_lotus2_model(model_id: str) -> bool:
    """True for Lotus-2 only.

    Deliberately matches the ``lotus-2`` token rather than just ``lotus``: the
    earlier SD-based family (``jingheya/lotus-depth-g-v2-1-disparity`` and
    friends) is a DIFFERENT architecture with a different loader, and a loose
    ``"lotus" in id`` test would silently route those here.
    """
    return "lotus-2" in model_id.lower()


def _is_depth_pro_model(model_id: str) -> bool:
    """True for Apple Depth Pro ids (``apple/DepthPro*``)."""
    return model_id.lower().startswith("apple/depthpro")


is_da3_model = _is_da3_model
is_moge_model = _is_moge_model
is_depth_pro_model = _is_depth_pro_model



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
    # depth_anything_3.utils.export.gs does `import moviepy.editor` at module
    # scope for gaussian-splat VIDEO export, and depth_anything_3.api imports
    # that module transitively — so a depth-only install dies on an import it
    # will never call. Same class as the others here: export-only, absent by
    # design, stubbed rather than installed. Found 2026-08-30 on a --no-deps
    # install, where the alternative was dragging moviepy (and a numpy
    # downgrade) into a working ComfyUI to satisfy an unused code path.
    "moviepy",
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
    # Per-pixel predicted surface normals (HxWx3 float32) when the model provides
    # them (MoGe *-normal variants) — in the MODEL's camera frame, so a consumer
    # must align them to the recovered world frame (see normals.align_predicted_
    # normals_to_world). None otherwise. Deliberately NOT in summary()/metadata
    # (which must stay JSON-safe); it's a heavy array like `depth`.
    normal: Any = None
    # The model's native metric POINTMAP (HxWx3 float32) when it predicts one
    # (MoGe: `out["points"]`), in the MODEL's camera frame — OpenCV axes, +X
    # right, +Y down, +Z forward, so `points[..., 2] == depth`. NaN where the
    # validity mask is False. MoGe derives depth AND intrinsics FROM this map,
    # so it is the more primitive output and costs nothing to keep. Consumers
    # (patch-camera registration, scale registration against an external
    # metric predictor) convert with `core.depth_geometry.opencv_points_to_
    # atlas_cam` — Atlas never re-derives its own camera XYZ from it. Heavy
    # array, so like `normal` it is excluded from summary()/metadata. Additive
    # field: the depth contract is unchanged.
    points: Any = None

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


_MOGE_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MOGE_MODEL_CACHE_MAX = 1


def _resolve_lotus2_root(lotus2_path: str = "") -> Path:
    """Locate the Lotus-2 clone (argument wins over ``ATLAS_LOTUS2_PATH``).

    Same shape as ``resolve_fixer_root``. The clone must contain ``pipeline.py``
    and ``infer.py``, because those are repo-local modules the loader imports by
    name — Lotus-2 is not distributed as a package.
    """
    raw = (lotus2_path or "").strip() or os.environ.get(LOTUS2_PATH_ENV, "").strip()
    root = Path(raw).expanduser() if raw else None
    if root is None or not root.is_dir():
        raise RuntimeError(
            "Lotus-2 depth needs a local clone (its code is Apache-2.0):\n"
            "    git clone https://github.com/EnVision-Research/Lotus-2.git\n"
            f"then set {LOTUS2_PATH_ENV} to it (or pass checkpoint_path).\n"
            "NOTE its base model is "
            f"{LOTUS2_FLUX_BASE}, which is GATED on HuggingFace and licensed "
            "NON-COMMERCIALLY — accept the licence and `hf auth login` first. "
            "Weights (~Apache-2.0 LoRAs) come from jingheya/Lotus-2."
        )
    missing = [n for n in ("pipeline.py", "infer.py") if not (root / n).exists()]
    if missing:
        raise RuntimeError(
            f"Lotus-2 clone at {root} is incomplete — missing {', '.join(missing)}. "
            "Expected the repository root of EnVision-Research/Lotus-2."
        )
    return root


def _require_lotus2(root: Path) -> tuple[Any, Any, Any, Any]:
    """Import Lotus-2's repo-local modules plus the diffusers pieces it needs."""
    root_str = str(root)
    if root_str not in sys.path:
        # Appended, not prepended: prepending a third-party checkout ahead of the
        # stdlib is how a repo with a `types.py` or `utils.py` shadows something
        # important and produces a baffling failure elsewhere in the process.
        sys.path.append(root_str)
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    except ImportError as exc:  # pragma: no cover - needs the extra
        raise RuntimeError(
            "Lotus-2 needs diffusers. Install with:  pip install diffusers"
        ) from exc
    try:
        from infer import load_lora_and_lcm_weights, process_single_image
        from pipeline import Lotus2Pipeline
    except ImportError as exc:  # pragma: no cover - needs the clone
        raise RuntimeError(
            f"Lotus-2 clone at {root} did not import ({exc}). Its requirements.txt "
            "must be installed into this interpreter."
        ) from exc
    return (Lotus2Pipeline, load_lora_and_lcm_weights, process_single_image,
            (FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel))


def _get_lotus2_pipeline(root: Path, device: str, task: str = "depth"):
    """Build (and cache) the Lotus-2 pipeline. Mirrors the upstream app.py exactly.

    Cached because construction loads the full FLUX transformer — many GB and
    tens of seconds. Keyed on (root, device, task) since the LoRA rank differs
    between depth (128) and normal (256).
    """
    cache_key = (str(root), device, task)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    torch = _require_torch()
    Lotus2Pipeline, load_weights, process_single_image, (Sched, Transformer) = \
        _require_lotus2(root)

    weight_dtype = torch.bfloat16

    # VRAM BUDGET. The FLUX transformer alone is ~24 GB in bf16, and Atlas runs
    # INSIDE ComfyUI, which is always already holding models — measured live on a
    # 32 GB RTX 5090 with a normal ComfyUI session: 17.2 GB used, 15.0 GB free, so
    # a plain `.to(device)` OOMs during shard loading. Upstream's app.py can do
    # `.to(device)` because it owns a dedicated Space; a ComfyUI node cannot.
    # Below the threshold we hand the pipeline to diffusers' CPU offload, which
    # streams modules on demand — far slower, but it runs instead of dying.
    free_bytes = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            free_bytes = int(torch.cuda.mem_get_info()[0])
        except Exception:  # noqa: BLE001
            free_bytes = 0
    offload = bool(device.startswith("cuda")) and free_bytes < LOTUS2_VRAM_BYTES

    scheduler = Sched.from_pretrained(
        LOTUS2_FLUX_BASE, subfolder="scheduler", num_train_timesteps=10)
    transformer = Transformer.from_pretrained(
        LOTUS2_FLUX_BASE, subfolder="transformer", revision=None, variant=None,
        torch_dtype=weight_dtype)
    transformer.requires_grad_(False)
    if not offload:
        transformer.to(device=device, dtype=weight_dtype)
    # load_lora_and_lcm_weights reads transformer.device/.dtype, so under offload
    # the LoRAs are merged on CPU before the pipeline takes over placement.
    transformer, lcm = load_weights(transformer, None, None, None, task)
    pipeline = Lotus2Pipeline.from_pretrained(
        LOTUS2_FLUX_BASE, scheduler=scheduler, transformer=transformer,
        revision=None, variant=None, torch_dtype=weight_dtype)
    pipeline.local_continuity_module = lcm
    if offload:
        if hasattr(lcm, "to"):
            # The continuity module is assigned as a bare attribute, so
            # enable_model_cpu_offload() does not see it and will not place it.
            lcm.to(device=device, dtype=weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline._atlas_offloaded = offload
    bounded_cache_set(_MODEL_CACHE, cache_key, (pipeline, process_single_image),
                      _MODEL_CACHE_MAX)
    return pipeline, process_single_image


def _estimate_depth_lotus2(
    image_path: str | Path,
    *,
    model_id: str,
    device: str,
    num_inference_steps: int = LOTUS2_DEFAULT_STEPS,
    checkpoint_path: str = "",
) -> DepthResult:
    """Lotus-2 relative depth, converted to Atlas's forward-distance convention.

    Lotus-2 emits an affine-invariant map where LARGER = NEARER — inferred from
    the family's ``*-disparity`` v1 names and upstream's ``reverse_color=True``,
    then CONFIRMED live (rho +0.98 vs Depth Pro, +0.94 vs DA-V2, where those two
    agree at +0.95). Atlas's DepthResult contract is the opposite —
    forward distance, larger = farther — so the output goes through the SAME
    `_disparity_to_depth` reciprocal used by the relative V2 path rather than a
    bespoke flip. Reusing it matters: a linear `1 - d` inversion is
    rank-preserving and looks fine while systematically warping near/far spacing,
    which is precisely the bug that conversion was extracted to prevent.

    Upstream takes the THIRD return of process_single_image (`output_npy`), not
    the second — the second is a colourised visualisation, and feeding that in as
    depth would look plausible and be meaningless.
    """
    import numpy as np
    from PIL import Image

    torch = _require_torch()
    root = _resolve_lotus2_root(checkpoint_path)
    pipeline, process_single_image = _get_lotus2_pipeline(root, device, "depth")

    with Image.open(image_path) as im:
        width, height = im.size

    _, _, output_npy = process_single_image(
        str(image_path), pipeline, task_name="depth", device=device,
        num_inference_steps=int(num_inference_steps), process_res=None)

    disparity = np.asarray(output_npy, dtype=np.float32)
    if disparity.ndim == 3:                       # (H, W, C) -> mean, as upstream
        disparity = disparity.mean(axis=-1)

    metadata: dict[str, Any] = {
        "backend": "lotus-2",
        "flux_base": LOTUS2_FLUX_BASE,
        "num_inference_steps": int(num_inference_steps),
        "clone": str(root),
        # VERIFIED live 2026-07-30 on ghosttown.jpg, not assumed: rank
        # correlation against two independently-trusted backends came out
        # +0.983 (DepthPro) and +0.935 (DA-V2), where those two agree with each
        # other at +0.949. Lotus-2 matches DepthPro MORE closely than the pair
        # match each other, so the disparity reading is right. Kept in metadata
        # because it is still the first thing to check if a future checkpoint
        # changes convention.
        "polarity": "disparity (larger = nearer), verified rho +0.98 vs DepthPro",
        "cpu_offloaded": bool(getattr(pipeline, "_atlas_offloaded", False)),
    }
    depth, metadata = _disparity_to_depth(disparity, metadata)

    if depth.shape != (height, width):
        # BILINEAR for the same reason as the V2 path: bicubic rings at
        # discontinuities and overshoots below the local minimum at silhouettes.
        t = torch.from_numpy(np.ascontiguousarray(depth))[None, None]
        depth = (
            torch.nn.functional.interpolate(
                t, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0].numpy().astype(np.float32)
        )

    return DepthResult(
        depth=depth,
        is_metric=False,
        model_id=model_id,
        image_width=width,
        image_height=height,
        near=float(depth.min()),
        far=float(depth.max()),
        metadata=metadata,
    )


def _require_moge() -> Any:
    """Import the MoGe-2 model class lazily with an informative error.

    MoGe is MIT-licensed and light on dependencies (no gsplat/open3d/pycolmap
    stack), so unlike DA3 it needs no export-only stubbing — just the package
    plus its ``utils3d`` companion.
    """
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "MoGe depth models require the 'moge' package (MIT licensed). Install with:\n"
            "    pip install --no-deps git+https://github.com/microsoft/MoGe.git\n"
            "    pip install --no-deps "
            "'git+https://github.com/EasternJournalist/utils3d.git"
            "@3fab839f0be9931dac7c8488eb0e1600c236e183'"
        ) from exc
    return MoGeModel


_XFORMERS_DISPATCH_PROBED: dict[str, bool] = {}


def _disable_xformers_if_undispatchable(device: str) -> bool:
    """Fall back to standard attention when xFormers has no kernel for this GPU.

    DINOv2 guards xFormers on IMPORTABILITY (``try: from xformers.ops import
    ...``), but importability is not dispatchability: on a GPU newer than the
    installed wheel's kernels — Blackwell/sm_120 was the case that surfaced this
    — the import succeeds and ``memory_efficient_attention`` then raises
    ``NotImplementedError: No operator found`` mid-forward, which reads as a
    broken model rather than a missing kernel.

    So probe the real call once per device and, if it cannot dispatch, set
    ``XFORMERS_DISABLED`` (which DINOv2 honours) before MoGe is imported. When
    MoGe is ALREADY imported the env var has been read, so flip the module flags
    too. Returns True when xFormers was turned off.
    """

    cached = _XFORMERS_DISPATCH_PROBED.get(device)
    if cached is not None:
        return cached

    disabled = False
    if str(device).startswith("cuda"):
        try:
            import torch
            from xformers.ops import memory_efficient_attention
        except Exception:  # noqa: BLE001 - absent xformers already falls back
            _XFORMERS_DISPATCH_PROBED[device] = False
            return False
        try:
            probe = torch.zeros((1, 4, 1, 8), dtype=torch.float16, device=device)
            memory_efficient_attention(probe, probe, probe)
        except NotImplementedError:
            os.environ["XFORMERS_DISABLED"] = "1"
            disabled = True
        except Exception:  # noqa: BLE001 - any other failure is not ours to judge
            disabled = False

    if disabled:
        for name in (
            "moge.model.dinov2.layers.attention",
            "moge.model.dinov2.layers.block",
            "moge.model.dinov2.layers.swiglu_ffn",
        ):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "XFORMERS_AVAILABLE", False):
                module.XFORMERS_AVAILABLE = False

    _XFORMERS_DISPATCH_PROBED[device] = disabled
    return disabled


def _get_moge_model(model_id: str, device: str, checkpoint_path: str = ""):
    """Load (and cache) a MoGe model, optionally from a LOCAL checkpoint.

    ``MoGeModel.from_pretrained`` already branches on ``Path(x).exists()``, so a
    local ``model.pt`` is passed straight through and no download happens. That
    is the whole mechanism — air-gapped installs and shared model directories
    point at a file instead of a HuggingFace id.

    NOT compatible with ComfyUI core's ``models/geometry_estimation/*.safetensors``:
    from_pretrained does ``torch.load(..., weights_only=True)`` and reads
    ``checkpoint['model_config']``, neither of which a safetensors file provides.
    """
    key = (model_id, device, checkpoint_path or "")
    cached = _MOGE_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    # Must run BEFORE the import: DINOv2 reads XFORMERS_DISABLED at module load.
    _disable_xformers_if_undispatchable(device)
    MoGeModel = _require_moge()
    source = model_id
    if checkpoint_path:
        p = Path(checkpoint_path)
        if not p.is_file():
            raise RuntimeError(
                f"MoGe checkpoint_path does not exist: {checkpoint_path}\n"
                "Expected a MoGe `model.pt` (the format from_pretrained reads). "
                "ComfyUI core's geometry_estimation/*.safetensors will NOT work — "
                "different container, and it carries no model_config.")
        source = str(p)
    model = MoGeModel.from_pretrained(source).to(device).eval()
    bounded_cache_set(_MOGE_MODEL_CACHE, key, model, _MOGE_MODEL_CACHE_MAX,
                      release_cuda=True)
    return model


def _estimate_depth_moge(
    image_path: str | Path,
    *,
    model_id: str,
    device: str,
    focal_px: float | None,
    resolution_level: int = MOGE_RESOLUTION_LEVEL_DEFAULT,
    max_side: int = 0,
    checkpoint_path: str = "",
    tile_side: int = 0,
    tile_overlap: float = 0.25,
    report_free_focal: bool = False,
) -> DepthResult:
    """MoGe-2 inference path: metric forward-Z depth from the point map.

    MoGe predicts its own camera, but ``infer`` accepts a known horizontal FOV;
    when the solve supplies a focal we feed it (``fov_x`` in degrees) so the
    geometry lands in the RECOVERED camera's frame rather than MoGe's own guess.
    ``depth`` is already forward-Z (== ``points[..., 2]``, verified live); the
    validity ``mask`` becomes NaN holes, and downstream ground-pinning
    re-normalizes absolute scale regardless of the model's metric estimate.
    ``normal`` (from ``*-normal`` variants) rides in metadata for future use by
    the relief mesh's normal-bend tear test.

    Two cost knobs, both INERT at their defaults so existing graphs are
    bit-identical:

    ``resolution_level`` (0-9) is MoGe's own token-budget dial; 9 is its default
    and ours. Lower trades detail for speed.

    ``max_side`` caps the longer edge BEFORE the GPU tensor is built. MoGe
    resamples internally anyway, so this is a memory/time lever, not a quality
    one — but the tensor itself is not free: a 7680x4512 plate is ~415 MB of
    float32 on the device before MoGe touches it. 0 disables. Outputs are always
    returned at SOURCE resolution regardless, so nothing downstream can tell.

    Intrinsics provenance. MoGe returns ``out["intrinsics"]`` (a NORMALIZED 3x3:
    fx in image widths, cx in image widths). When we fed ``fov_x`` the matrix is
    just an echo of the solve, so it is recorded as ``intrinsics_source=
    "echo_of_solve"`` and ``core.scene_health`` must NOT read it as agreement.
    ``report_free_focal`` runs a SECOND, fov-free pass (depth discarded, at a
    reduced resolution level) and records ``predicted_focal_px_free`` — the only
    value that is an independent estimate when a solve focal was supplied.
    Measured 2026-08-15 on sh001 (metric-solved, 6207 px): the free pass predicts
    5278 px, a 15% disagreement that was invisible before this field existed.
    """
    import math
    import numpy as np
    from PIL import Image

    torch = _require_torch()
    model = _get_moge_model(model_id, device, checkpoint_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Downscale for inference only. Every output is resized back to (height,
    # width) below, so `width`/`height` stay the SOURCE dims throughout and the
    # DepthResult contract is unchanged.
    infer_image = image
    cap = int(max_side or 0)
    if cap > 0 and max(width, height) > cap:
        scale = cap / float(max(width, height))
        infer_image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.LANCZOS)

    arr = np.asarray(infer_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)  # (3,H,W) in [0,1]

    fov_x = None
    focal_source = "predicted"
    if focal_px is not None and float(focal_px) > 0:
        # fov_x is an ANGLE, so it is computed from the SOURCE width/focal and is
        # invariant to the inference downscale — deliberately not infer_image.
        fov_x = math.degrees(2.0 * math.atan(width / (2.0 * float(focal_px))))
        focal_source = "solve"

    level = int(resolution_level)
    with torch.inference_mode():
        out = model.infer(tensor, fov_x=fov_x, resolution_level=level)

    def _to_source(t, mode):
        """Resize a MoGe output back to the source frame. No-op at max_side=0."""
        if tuple(t.shape[-2:]) == (height, width):
            return t
        squeeze = t.dim() == 2
        x = t[None, None] if squeeze else t.permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(
            x.float(), size=(height, width), mode=mode,
            **({} if mode == "nearest" else {"align_corners": False}))
        return x[0, 0] if squeeze else x[0].permute(1, 2, 0)

    depth_t = _to_source(out["depth"].float(), "bilinear")
    depth = depth_t.cpu().numpy()  # forward-Z metres
    metadata: dict[str, Any] = {
        "device": device, "backend": "moge", "focal_source": focal_source,
    }
    if fov_x is not None:
        metadata["fov_x_deg"] = float(fov_x)
    metadata["resolution_level"] = level
    if infer_image is not image:
        # Say what actually ran, not what was asked for — same principle as the
        # exclude-mask coverage line: a silent downscale is a silent quality
        # change, and the debug report is where an artist would look for it.
        metadata["inference_downscaled_to"] = list(infer_image.size)   # (w, h)
        metadata["max_side"] = cap
    if checkpoint_path:
        metadata["checkpoint_path"] = str(checkpoint_path)
    mask = None
    if "mask" in out:
        # NEAREST, and applied AFTER the depth resize. Bilinear on a boolean
        # would invent fractional validity, and masking before the resize would
        # smear NaN holes outward across every interpolated neighbour.
        mask = _to_source(out["mask"].detach().float(), "nearest").cpu().numpy() > 0.5
        depth = np.where(mask, depth, np.nan)
        metadata["valid_fraction"] = float(mask.mean())

    # ---- native pointmap + predicted intrinsics ---------------------------
    # Kept as provenance/registration inputs only. Depth stays the contract.
    predicted_points = None
    if "points" in out:
        pts_t = out["points"].detach().float()
        if pts_t.dim() == 4:                          # (B,H,W,3)
            pts_t = pts_t[0]
        if pts_t.dim() == 3 and pts_t.shape[0] == 3 and pts_t.shape[-1] != 3:
            pts_t = pts_t.permute(1, 2, 0)            # (3,H,W) -> (H,W,3)
        pts_t = _to_source(pts_t, "bilinear")
        predicted_points = np.asarray(pts_t.cpu().numpy(), dtype=np.float32)
        if mask is not None:
            predicted_points = np.where(mask[..., None], predicted_points, np.nan)
        metadata["has_pointmap"] = True
    intr_t = out.get("intrinsics") if hasattr(out, "get") else None
    if intr_t is not None:
        k = np.asarray(intr_t.detach().float().cpu().numpy(), dtype=np.float64)
        if k.ndim == 3:
            k = k[0]
        if k.shape == (3, 3):
            # Normalized intrinsics: fx, cx in image WIDTHS; fy, cy in HEIGHTS.
            metadata["predicted_focal_px"] = round(float(k[0, 0]) * width, 2)
            metadata["predicted_fy_px"] = round(float(k[1, 1]) * height, 2)
            metadata["predicted_cx_px"] = round(float(k[0, 2]) * width, 2)
            metadata["predicted_cy_px"] = round(float(k[1, 2]) * height, 2)
            metadata["intrinsics_source"] = (
                "echo_of_solve" if fov_x is not None else "moge_fov_head")
    if report_free_focal and fov_x is not None:
        # An INDEPENDENT focal: re-run with no fov hint. Depth is discarded; a
        # lower resolution level keeps the cost well under a full pass.
        free_level = max(0, min(level, MOGE_FREE_FOCAL_RESOLUTION_LEVEL))
        with torch.inference_mode():
            free_out = model.infer(tensor, fov_x=None, resolution_level=free_level)
        free_k = free_out.get("intrinsics") if hasattr(free_out, "get") else None
        if free_k is not None:
            fk = np.asarray(free_k.detach().float().cpu().numpy(), dtype=np.float64)
            if fk.ndim == 3:
                fk = fk[0]
            if fk.shape == (3, 3):
                metadata["predicted_focal_px_free"] = round(float(fk[0, 0]) * width, 2)
                metadata["free_focal_resolution_level"] = free_level
        del free_out
    elif report_free_focal and "predicted_focal_px" in metadata:
        # No solve focal was fed, so the first pass WAS the free pass.
        metadata["predicted_focal_px_free"] = metadata["predicted_focal_px"]
    predicted_normal = None
    if "normal" in out:
        # Predicted per-pixel surface normals in the MODEL's camera frame — kept
        # for the relight (aligned to world downstream) and the mesh's normal-bend
        # tear test (both cleaner than gradient-of-depth normals).
        normal_t = out["normal"].detach().float()
        if normal_t.dim() == 4:                       # (B,H,W,3) or (B,3,H,W)
            normal_t = normal_t[0]
        if normal_t.dim() == 3 and normal_t.shape[0] == 3:   # (3,H,W) -> (H,W,3)
            normal_t = normal_t.permute(1, 2, 0)
        # Interpolating unit vectors shortens them; renormalise so downstream
        # Procrustes alignment still sees unit normals.
        normal_t = _to_source(normal_t, "bilinear")
        normal_t = normal_t / normal_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        predicted_normal = np.asarray(normal_t.cpu().numpy(), dtype=np.float32)
        metadata["has_predicted_normals"] = True

    # ---- optional native-resolution tiling -------------------------------
    # The pass above always runs, tiled or not: it is the GLOBAL REFERENCE every
    # tile is anchored to. Without it the tiles have no shared frame and their
    # individual scales cannot be reconciled.
    side = int(tile_side or 0)
    if side > 0 and max(width, height) > side:
        boxes = tile_boxes(width, height, side, float(tile_overlap))
        collected = []
        for (x0, y0, x1, y1) in boxes:
            crop = image.crop((x0, y0, x1, y1))
            cw, ch = crop.size
            c_arr = np.asarray(crop, dtype=np.float32) / 255.0
            c_t = torch.from_numpy(c_arr).permute(2, 0, 1).to(device)

            # A tile is a CROP, so its horizontal FOV is narrower than the full
            # frame's — same focal, fewer pixels. Passing the frame's fov_x here
            # would tell MoGe the tile spans a much wider angle than it does and
            # skew the geometry of every tile independently.
            c_fov = None
            if focal_px is not None and float(focal_px) > 0:
                c_fov = math.degrees(2.0 * math.atan(cw / (2.0 * float(focal_px))))

            with torch.inference_mode():
                c_out = model.infer(c_t, fov_x=c_fov, resolution_level=level)
            c_depth = c_out["depth"].float()
            if tuple(c_depth.shape[-2:]) != (ch, cw):
                c_depth = torch.nn.functional.interpolate(
                    c_depth[None, None], size=(ch, cw), mode="bilinear",
                    align_corners=False)[0, 0]
            c_np = c_depth.cpu().numpy().astype(np.float64)
            if "mask" in c_out:
                c_mask = c_out["mask"].detach().float()
                if tuple(c_mask.shape[-2:]) != (ch, cw):
                    c_mask = torch.nn.functional.interpolate(
                        c_mask[None, None], size=(ch, cw), mode="nearest")[0, 0]
                c_np = np.where(c_mask.cpu().numpy() > 0.5, c_np, np.nan)

            a, b = fit_affine_to_reference(c_np, depth[y0:y1, x0:x1], np)
            collected.append(((x0, y0, x1, y1), c_np * a + b))
            del c_out, c_depth, c_t

        ramp = max(1, int(round(side * float(tile_overlap) * 0.5)))
        tiled = assemble_tiles(collected, width, height, ramp, np)
        # Keep the global pass wherever tiling produced nothing (a fully invalid
        # tile), so tiling can only add detail, never punch new holes.
        depth = np.where(np.isfinite(tiled), tiled, depth).astype(np.float32)
        metadata["tiled"] = {
            "tile_side": side,
            "overlap": float(tile_overlap),
            "tiles": len(boxes),
            "feather_px": ramp,
            "anchored_to": "global_pass",
        }
        if predicted_points is not None:
            # The tiled depth is an affine-fitted composite; the global-pass
            # pointmap no longer agrees with it pixel-for-pixel, and per-tile
            # pointmaps live in per-tile camera frames. Drop rather than lie.
            predicted_points = None
            metadata["has_pointmap"] = False
            metadata["points_dropped_reason"] = "tiled"

    depth, metadata = _record_and_clamp_negative(depth, metadata)
    valid = np.isfinite(depth)
    near = float(np.nanmin(depth[valid])) if valid.any() else 0.0
    far = float(np.nanmax(depth[valid])) if valid.any() else 0.0
    return DepthResult(
        depth=depth,
        is_metric=True,
        model_id=model_id,
        image_width=width,
        image_height=height,
        near=near,
        far=far,
        metadata=metadata,
        normal=predicted_normal,
        points=predicted_points,
    )


def estimate_depth(
    image_path: str | Path,
    *,
    model_id: str = DEFAULT_METRIC_OUTDOOR,
    device: str | None = None,
    focal_px: float | None = None,
    resolution_level: int = MOGE_RESOLUTION_LEVEL_DEFAULT,
    max_side: int = 0,
    tile_side: int = 0,
    tile_overlap: float = 0.25,
    checkpoint_path: str = "",
    report_free_focal: bool = False,
) -> DepthResult:
    """Predict a depth map for a single image (Depth Anything V2 / V3, or MoGe-2).

    Returns forward distance (metres for metric models). The map is resized back
    to the source image resolution. ``focal_px`` (the solve's focal length in
    source-image pixels) is consumed by DA3METRIC (converts canonical depth to
    metres using the *solved* focal) and by MoGe (fed as ``fov_x`` so its
    geometry lands in the recovered camera's frame); every other model ignores
    it. Backend is chosen by ``model_id``: ``depth-anything/DA3*`` -> DA3,
    anything with ``moge`` -> MoGe-2, ``apple/DepthPro*`` -> Depth Pro (metric
    depth + its own predicted focal in metadata), else transformers Depth
    Anything V2.
    """
    torch = _require_torch()

    device = resolve_device(device, torch)

    content_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    # Only the model families that consume focal_px fragment the cache on it
    # (DA3METRIC's canonical->metric conversion, and MoGe's fov_x injection).
    focal_key = (
        round(float(focal_px), 3)
        if (focal_px and (_is_da3_model(model_id) or _is_moge_model(model_id)))
        else None
    )

    # MoGe-only knobs join the key. Without this a re-run at a different
    # resolution_level or max_side would hit the cache and return the FIRST
    # call's map — the knob would appear to do nothing, which is the worst
    # possible failure for a quality/speed dial.
    moge_key = (
        (int(resolution_level), int(max_side or 0), str(checkpoint_path or ""),
         int(tile_side or 0), round(float(tile_overlap), 4),
         bool(report_free_focal))
        if _is_moge_model(model_id) else None
    )
    # Lotus-2 resolves its clone from checkpoint_path, so that must join the key
    # or pointing at a second clone would silently return the first one's map —
    # the same class of bug the moge_key comment above records.
    lotus_key = (
        str(checkpoint_path or "") if _is_lotus2_model(model_id) else None
    )
    cache_key = (content_hash, model_id, device, focal_key, moge_key, lotus_key)
    cached_result = _DEPTH_RESULT_CACHE.get(cache_key)
    if cached_result is not None:
        return cached_result

    if _is_da3_model(model_id):
        result = _estimate_depth_da3(
            image_path, model_id=model_id, device=device, focal_px=focal_px
        )
    elif _is_moge_model(model_id):
        result = _estimate_depth_moge(
            image_path, model_id=model_id, device=device, focal_px=focal_px,
            resolution_level=resolution_level, max_side=max_side,
            tile_side=tile_side, tile_overlap=tile_overlap,
            checkpoint_path=checkpoint_path,
            report_free_focal=bool(report_free_focal),
        )
    elif _is_depth_pro_model(model_id):
        result = _estimate_depth_depth_pro(image_path, model_id=model_id, device=device)
    elif _is_lotus2_model(model_id):
        result = _estimate_depth_lotus2(
            image_path, model_id=model_id, device=device,
            checkpoint_path=checkpoint_path,
        )
    else:
        result = _estimate_depth_v2(image_path, model_id=model_id, device=device)
    bounded_cache_set(_DEPTH_RESULT_CACHE, cache_key, result, _DEPTH_RESULT_CACHE_MAX)
    return result


def _estimate_depth_depth_pro(
    image_path: str | Path,
    *,
    model_id: str,
    device: str,
) -> DepthResult:
    """Apple Depth Pro inference path (transformers >= 4.48).

    The model's FOV head predicts its own focal length, which the processor's
    ``post_process_depth_estimation`` uses to convert canonical inverse depth
    into metric metres at the target resolution. That predicted focal (rescaled
    to SOURCE-image pixels) and the horizontal FOV are recorded as
    ``metadata["predicted_focal_px"]`` / ``metadata["predicted_fov_h_deg"]`` —
    an independent intrinsics estimate that ``core.scene_health`` cross-checks
    against the solve's fx. This function only REPORTS the estimate; it never
    judges agreement (verdicts live in scene_health alone).
    """
    import math

    import numpy as np
    from PIL import Image

    torch, _, _ = _require_depth_backend()
    processor, model = _get_model(model_id, device)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    metadata: dict[str, Any] = {"device": device, "backend": "depth_pro"}
    post_fn = getattr(processor, "post_process_depth_estimation", None)
    if post_fn is not None:
        post = post_fn(outputs, target_sizes=[(height, width)])[0]
        depth = np.asarray(
            post["predicted_depth"].detach().float().cpu().numpy(), dtype=np.float32
        )
        focal = post.get("focal_length")
        fov = post.get("field_of_view")
        if focal is not None:
            # post_process returns the focal at target (== source) resolution.
            metadata["predicted_focal_px"] = float(
                focal.item() if hasattr(focal, "item") else focal
            )
        if fov is not None:
            metadata["predicted_fov_h_deg"] = float(
                fov.item() if hasattr(fov, "item") else fov
            )
        elif focal is not None and metadata.get("predicted_focal_px", 0) > 0:
            metadata["predicted_fov_h_deg"] = math.degrees(
                2.0 * math.atan(width / (2.0 * metadata["predicted_focal_px"]))
            )
    else:  # pragma: no cover - transformers without the DepthPro post-processor
        predicted = outputs.predicted_depth
        if predicted.dim() == 3:
            predicted = predicted.unsqueeze(1)
        predicted = torch.nn.functional.interpolate(
            # Bilinear, not bicubic — see the DA3 path's comment on ringing.
            predicted, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
        depth = predicted.detach().float().cpu().numpy().astype(np.float32)

    depth, metadata = _record_and_clamp_negative(depth, metadata)
    return DepthResult(
        depth=depth,
        is_metric=True,
        model_id=model_id,
        image_width=width,
        image_height=height,
        near=float(depth.min()),
        far=float(depth.max()),
        metadata=metadata,
    )


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
