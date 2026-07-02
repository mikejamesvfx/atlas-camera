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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Model ids that emit metric (meters) depth rather than up-to-scale relative depth.
_METRIC_HINT = "metric"

DEFAULT_RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_METRIC_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DEFAULT_METRIC_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


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


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_model(model_id: str, device: str):
    cached = _MODEL_CACHE.get((model_id, device))
    if cached is not None:
        return cached
    torch, AutoImageProcessor, AutoModelForDepthEstimation = _require_depth_backend()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    _MODEL_CACHE[(model_id, device)] = (processor, model)
    return processor, model


def estimate_depth(
    image_path: str | Path,
    *,
    model_id: str = DEFAULT_METRIC_OUTDOOR,
    device: str | None = None,
) -> DepthResult:
    """Predict a depth map for a single image with Depth Anything V2.

    Returns forward distance (metres for metric models). The map is resized back
    to the source image resolution.
    """
    torch, _, _ = _require_depth_backend()
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        predicted, size=(height, width), mode="bicubic", align_corners=False
    )[0, 0]

    is_metric = _METRIC_HINT in model_id.lower()
    depth = predicted.detach().float().cpu().numpy()

    if not is_metric:
        # Relative models emit disparity-like values (larger = closer). Convert to
        # a distance-like map (larger = farther), normalised to [0, 1] up to scale.
        import numpy as np

        d = depth - depth.min()
        d = d / (d.max() or 1.0)
        depth = 1.0 - d  # now larger = farther, still unitless

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
        metadata={"device": device},
    )
