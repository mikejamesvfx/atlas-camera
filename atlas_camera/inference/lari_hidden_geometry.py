"""LaRI layered-ray-intersection inference (EXPERIMENTAL, research-only).

Predicts, per pixel, the ordered stack of surfaces the camera ray intersects
(layer 0 = visible, later layers = occluded) using LaRI's scene checkpoint —
the "geometry hidden behind camera shadow rays" signal consumed by
``core/hidden_geometry.py``. See docs/dev/hidden_geometry_training_free_research.md
for the spike that validated (and bounded) this.

LICENSING / INSTALL: the LaRI repository (github.com/ruili3/lari) currently
ships NO license file — all rights reserved by default. atlas_camera therefore
never vendors or redistributes any of it; the user must clone the repository
themselves and point ``lari_path`` (or the ATLAS_LARI_PATH env var) at the
clone. Weights download from HuggingFace (ruili3/LaRI) on first use. Treat all
outputs as research-only.

Empirical conventions (2026-07-09 spike, this machine):
 - inference needs NO PyTorch3D (that dependency is only in LaRI's dataset
   curation/metrics code) and none of LaRI's pinned requirements beyond torch;
 - the SCENE checkpoint's raw ``pts3d`` z is already positive-forward (the
   public demo's negation applies to their object pipeline) — handled by a
   sign check rather than trusting either convention;
 - the model processes at 512px (long edge, black-padded square).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.inference._common import bounded_cache_set, resolve_device

LARI_SCENE_CHECKPOINT = "lari_scene_pointmap.pth"
LARI_HF_REPO = "ruili3/LaRI"
LARI_RESOLUTION = 512

_LARI_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_LARI_MODEL_CACHE_MAX = 1  # DINOv2-vitl14 backbone; keep at most one resident


def _require_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LaRI hidden-geometry inference requires torch. Install with:\n"
            "    pip install -e .[neural]"
        ) from exc


def _resolve_lari_root(lari_path: str | None) -> Path:
    p = (lari_path or "").strip() or os.environ.get("ATLAS_LARI_PATH", "").strip()
    root = Path(p) if p else None
    if root is None or not (root / "src" / "lari" / "model").is_dir():
        raise RuntimeError(
            "LaRI repository not found. This EXPERIMENTAL feature needs the LaRI\n"
            "code, which atlas_camera cannot bundle (the upstream repo has no\n"
            "license — research use only). Clone it yourself and point at it:\n"
            "    git clone https://github.com/ruili3/lari.git <somewhere>\\lari\n"
            "then set the node's lari_path widget (or the ATLAS_LARI_PATH env\n"
            "var) to that folder. Inference needs no PyTorch3D and no extra\n"
            "installs beyond the [neural] extra."
        )
    return root


def _require_lari(lari_path: str | None) -> Any:
    """Import LaRIModel from a user-provided clone of the LaRI repository."""
    root = _resolve_lari_root(lari_path)
    root_str = str(root)
    if root_str not in sys.path:
        # LaRI's code imports itself as `src.lari...`, so its ROOT goes on the
        # path. `src` is a regrettably generic package name — known, accepted
        # risk for an experimental feature; the path is appended (not
        # prepended) to minimize shadowing.
        sys.path.append(root_str)
    try:
        from src.lari.model import LaRIModel  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Found a LaRI checkout at {root} but importing its model failed "
            f"({exc}). Make sure the clone is complete and torch is installed."
        ) from exc
    return LaRIModel


def _get_lari_model(lari_path: str | None, device: str):
    root = str(_resolve_lari_root(lari_path))
    cached = _LARI_MODEL_CACHE.get((root, device))
    if cached is not None:
        return cached
    torch = _require_torch()
    LaRIModel = _require_lari(lari_path)
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(LARI_HF_REPO, LARI_SCENE_CHECKPOINT)
    model = LaRIModel(
        use_pretrained=None, pretrained_path="", num_output_layer=5,
        head_type="point",
    ).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    bounded_cache_set(_LARI_MODEL_CACHE, (root, device), model, _LARI_MODEL_CACHE_MAX)
    return model


@dataclass(slots=True)
class LayeredDepthResult:
    """Layered per-ray depths plus provenance (research-only hypothesis data).

    ``layers`` is (H, W, L) float32 forward depth in the MODEL'S OWN normalized
    units, front-to-back, at the model's working resolution (H, W follow the
    source aspect; long edge = LARI_RESOLUTION). Registration into pipeline
    units is the consumer's job (core.hidden_geometry).
    """

    layers: Any  # numpy (H, W, L) float32
    image_width: int   # source image size the crop corresponds to
    image_height: int
    metadata: dict[str, Any] = field(default_factory=dict)


def predict_layered_depth(
    image_path: str | Path,
    *,
    lari_path: str | None = None,
    device: str | None = None,
) -> LayeredDepthResult:
    """Run LaRI's scene model on one image -> layered depth stack."""
    torch = _require_torch()
    import numpy as np
    from PIL import Image

    device = resolve_device(device, torch)
    model = _get_lari_model(lari_path, device)

    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size
    scale = LARI_RESOLUTION / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = pil.resize((nw, nh), Image.BILINEAR)
    pad_top = (LARI_RESOLUTION - nh) // 2
    pad_left = (LARI_RESOLUTION - nw) // 2
    padded = Image.new("RGB", (LARI_RESOLUTION, LARI_RESOLUTION), (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))

    arr = np.asarray(padded, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    inp = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device)

    with torch.no_grad():
        pred = model(inp)
    pts3d = pred["pts3d"].squeeze(0).float().cpu().numpy()  # (H, W, L, 3)
    z = pts3d[..., 2]
    # Sign check instead of trusting a convention (scene ckpt: raw z positive).
    if float((z > 0).mean()) < 0.5:
        z = -z
    z = z[pad_top:pad_top + nh, pad_left:pad_left + nw, :]  # crop padding away

    return LayeredDepthResult(
        layers=np.ascontiguousarray(z, dtype=np.float32),
        image_width=w,
        image_height=h,
        metadata={
            "device": device,
            "checkpoint": LARI_SCENE_CHECKPOINT,
            "n_layers": int(z.shape[-1]),
            "working_width": int(nw),
            "working_height": int(nh),
            "research_only": True,
        },
    )
