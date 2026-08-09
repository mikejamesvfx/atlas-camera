"""Thin ComfyUI boundary for deterministic photographed multi-view solves.

The solver owns every registration decision.  This module only validates the
photographed RAW evidence ComfyUI hands it, adapts IMAGE tensors at the host
boundary, and converts the deterministic result back to ComfyUI values.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from typing import Any

from atlas_camera.core.multiview_solver import solve_multiview
from atlas_camera.core.multiview_types import MultiViewFrame, MultiViewSettings


_RAW_PIXEL_FIELDS = {"linear_rgb", "display_srgb"}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "AtlasMultiViewSolve requires NumPy. Install with: pip install -e .[vision]"
        ) from exc
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "AtlasMultiViewSolve requires PyTorch, which is included with ComfyUI."
        ) from exc
    return torch


def _array_signature(value: Any) -> dict[str, Any]:
    """Return a deterministic content signature without importing NumPy.

    ComfyUI IMAGE values are torch tensors, but keeping this helper duck-typed
    lets cache calculation remain a small adapter concern and keeps imports
    optional until the node is actually used.
    """
    tensor = value
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    shape = tuple(int(part) for part in getattr(tensor, "shape", ()))
    dtype = str(getattr(tensor, "dtype", type(tensor).__name__))
    if hasattr(tensor, "tobytes"):
        try:
            pixels = tensor.tobytes(order="C")
        except TypeError:
            pixels = tensor.tobytes()
    else:
        try:
            pixels = memoryview(tensor).tobytes()
        except TypeError as exc:
            raise TypeError("IMAGE input must provide contiguous pixel bytes") from exc
    return {
        "shape": shape,
        "dtype": dtype,
        "sha256": hashlib.sha256(pixels).hexdigest(),
    }


def _cache_value(value: Any, *, exclude_raw_pixels: bool = False) -> Any:
    """Make link values JSON-stable without importing optional packages.

    RawImportResult carries two image arrays in addition to the metadata.  Its
    display pixels are represented separately as a RAW/IMAGE binding signature;
    `linear_rgb` is deliberately omitted because solve_multiview never reads it.
    This avoids repeatedly hashing a 24 MP linear master that cannot affect a
    solve, while retaining every trusted RAW metadata field.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cache_value(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _cache_value(getattr(value, item.name))
            for item in fields(value)
            if not (exclude_raw_pixels and item.name in _RAW_PIXEL_FIELDS)
        }
    if hasattr(value, "shape") and (hasattr(value, "tobytes") or hasattr(value, "numpy")):
        return {"array": _array_signature(value)}
    if hasattr(value, "to_dict"):
        return _cache_value(value.to_dict())
    return str(value)


def _raw_display_signature(raw_meta: Any) -> Any:
    """Fingerprint the display pixels that must bind to the IMAGE socket.

    This separate signature is required even though genuine inputs duplicate
    IMAGE content: a changed RawImportResult must invalidate Comfy's cache so
    runtime validation can reject an attempted sidecar substitution.
    """
    display_srgb = getattr(raw_meta, "display_srgb", None)
    return _array_signature(display_srgb) if display_srgb is not None else None


def _cache_fingerprint(
    image_1: Any,
    image_2: Any,
    image_3: Any,
    raw_meta_1: Any,
    raw_meta_2: Any,
    raw_meta_3: Any,
    plate_ref_1: Any,
    plate_ref_2: Any,
    plate_ref_3: Any,
    capture_mode: str,
    camera_height_m: float,
    match_quality: str,
    seed: int,
) -> str:
    """Hash every content-bearing link and persisted widget in socket order."""
    payload = {
        "images": [_array_signature(image_1), _array_signature(image_2),
                   _array_signature(image_3) if image_3 is not None else None],
        "raw_metadata": [
            _cache_value(raw_meta_1, exclude_raw_pixels=True),
            _cache_value(raw_meta_2, exclude_raw_pixels=True),
            _cache_value(raw_meta_3, exclude_raw_pixels=True),
        ],
        "raw_display_bindings": [
            _raw_display_signature(raw_meta_1),
            _raw_display_signature(raw_meta_2),
            _raw_display_signature(raw_meta_3),
        ],
        "plate_references": [
            _cache_value(plate_ref_1), _cache_value(plate_ref_2),
            _cache_value(plate_ref_3),
        ],
        "widgets": {
            "capture_mode": capture_mode,
            "camera_height_m": camera_height_m,
            "match_quality": match_quality,
            "seed": seed,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(b"atlas-multiview-comfy-v1\0" + encoded).hexdigest()


def _image_to_hwc_float32(image: Any, name: str, np: Any) -> Any:
    """Adapt one Comfy IMAGE batch element to the solver's HWC ndarray."""
    tensor = image
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    pixels = np.asarray(tensor)
    if pixels.ndim != 4:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} must be a BHWC IMAGE tensor; got {pixels.ndim} dimensions."
        )
    if int(pixels.shape[0]) != 1:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} must contain exactly one photograph "
            f"(batch size 1); got {pixels.shape[0]}."
        )
    if int(pixels.shape[-1]) != 3:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} must have exactly 3 channels in BHWC order; "
            f"got {pixels.shape[-1]}."
        )
    if not np.issubdtype(pixels.dtype, np.floating):
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} must contain floating-point values; "
            f"got {pixels.dtype}."
        )
    return np.ascontiguousarray(pixels[0], dtype=np.float32)


def _raw_display_to_hwc_float32(raw_meta: Any, name: str, np: Any) -> Any:
    """Canonicalize the AtlasLoadRAW display image used to bind IMAGE evidence."""
    display_srgb = getattr(raw_meta, "display_srgb", None)
    if display_srgb is None:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} requires RawImportResult.display_srgb "
            "to bind the IMAGE socket to photographed RAW evidence."
        )
    display = np.asarray(display_srgb)
    if display.ndim != 3 or int(display.shape[-1]) != 3:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} has malformed RawImportResult.display_srgb; "
            "expected HWC with 3 channels."
        )
    if not np.issubdtype(display.dtype, np.floating):
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} has non-floating RawImportResult.display_srgb."
        )
    return np.ascontiguousarray(display, dtype=np.float32)


def _require_photographed_frame(
    name: str,
    image: Any,
    raw_meta: Any,
    plate_ref: Any,
    np: Any,
) -> MultiViewFrame:
    """Accept only matching provenance from one AtlasLoadRAW photograph."""
    from atlas_camera.core.schema import AtlasPlateRef
    from atlas_camera.raw.pipeline import RawImportResult

    if not isinstance(raw_meta, RawImportResult) or not isinstance(plate_ref, AtlasPlateRef):
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} requires a complete photographed RAW frame; "
            "wire image, raw_meta, and plate_ref from the same AtlasLoadRAW node."
        )
    plate_metadata = plate_ref.metadata or {}
    if (
        plate_ref.is_proxy
        or plate_ref.role != "source"
        or plate_metadata.get("registered_from") != "AtlasLoadRAW"
    ):
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} rejects a generated or proxy projection source; "
            "registration frames must use a non-proxy AtlasLoadRAW plate_ref."
        )
    if not plate_ref.image_path or not plate_ref.preview_b64:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} requires a photographed preview and durable "
            "plate reference from AtlasLoadRAW."
        )
    if not raw_meta.source_path or plate_metadata.get("raw_source") != raw_meta.source_path:
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} requires a complete photographed RAW frame; "
            "plate_ref raw_source must match raw_meta.source_path."
        )

    pixels = _image_to_hwc_float32(image, name, np)
    if (int(raw_meta.width), int(raw_meta.height)) != (pixels.shape[1], pixels.shape[0]):
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} image dimensions {pixels.shape[1]}x{pixels.shape[0]} "
            f"do not match trusted RAW metadata {raw_meta.width}x{raw_meta.height}."
        )
    if not np.array_equal(pixels, _raw_display_to_hwc_float32(raw_meta, name, np)):
        raise RuntimeError(
            f"AtlasMultiViewSolve: {name} pixels do not match trusted RAW display_srgb; "
            "wire the IMAGE output from the same AtlasLoadRAW node."
        )
    return MultiViewFrame(
        image=pixels,
        raw_meta=raw_meta,
        plate_ref=plate_ref,
        label=name.replace("image_", "photo_"),
    )


def _write_failure_debug(details: dict[str, Any], overlays: tuple[Any, ...], np: Any) -> str:
    """Persist failure diagnostics where an artist can reach them.

    The adapter must raise on failure (ComfyUI cannot return links and raise in
    one execution), which would otherwise strand the overlays and structured
    diagnostics of exactly the runs that need inspecting.  Mirrors the
    AtlasDebugReport doctrine: a stable path under ComfyUI's CWD, and a debug
    write failure must never mask the real registration error.
    """
    try:
        debug_dir = os.path.abspath("atlas_debug")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, "multiview_failure.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(details, handle, sort_keys=True, indent=2)
        try:
            from PIL import Image
            for index, overlay in enumerate(overlays, start=1):
                pixels = np.clip(
                    np.ascontiguousarray(overlay, dtype=np.float32) * 255.0,
                    0.0, 255.0,
                ).astype(np.uint8)
                Image.fromarray(pixels).save(
                    os.path.join(debug_dir, f"multiview_failure_pair_{index}.png")
                )
        except Exception:  # noqa: BLE001 - overlays are best-effort extras.
            pass
        return path
    except Exception:  # noqa: BLE001 - never mask the registration error.
        return ""


def _overlay_batch(overlays: tuple[Any, ...], np: Any, torch: Any) -> Any:
    if overlays:
        pixels = np.stack([
            np.ascontiguousarray(overlay, dtype=np.float32) for overlay in overlays
        ])
    else:
        pixels = np.empty((0, 0, 0, 3), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32))


class AtlasMultiViewSolve:
    """Recover a deterministic camera rig from two or three photographed RAW frames."""

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("solve", "report", "registration_json", "match_overlays")
    FUNCTION = "solve"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
            "optional": {
                # Link/socket order is a saved-workflow contract. Widgets append last.
                "image_3": ("IMAGE", {"forceInput": True}),
                "raw_meta_1": ("ATLAS_RAW_META", {"forceInput": True}),
                "raw_meta_2": ("ATLAS_RAW_META", {"forceInput": True}),
                "raw_meta_3": ("ATLAS_RAW_META", {"forceInput": True}),
                "plate_ref_1": ("ATLAS_PLATE_REF", {"forceInput": True}),
                "plate_ref_2": ("ATLAS_PLATE_REF", {"forceInput": True}),
                "plate_ref_3": ("ATLAS_PLATE_REF", {"forceInput": True}),
                "capture_mode": (["auto", "translated", "rotation_only"], {"default": "auto"}),
                "camera_height_m": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "match_quality": (["balanced", "conservative", "permissive"], {"default": "balanced"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        image_1: Any,
        image_2: Any,
        image_3: Any = None,
        raw_meta_1: Any = None,
        raw_meta_2: Any = None,
        raw_meta_3: Any = None,
        plate_ref_1: Any = None,
        plate_ref_2: Any = None,
        plate_ref_3: Any = None,
        capture_mode: str = "auto",
        camera_height_m: float = 0.0,
        match_quality: str = "balanced",
        seed: int = 0,
    ) -> str:
        return _cache_fingerprint(
            image_1, image_2, image_3,
            raw_meta_1, raw_meta_2, raw_meta_3,
            plate_ref_1, plate_ref_2, plate_ref_3,
            capture_mode, camera_height_m, match_quality, seed,
        )

    def solve(
        self,
        image_1: Any,
        image_2: Any,
        image_3: Any = None,
        raw_meta_1: Any = None,
        raw_meta_2: Any = None,
        raw_meta_3: Any = None,
        plate_ref_1: Any = None,
        plate_ref_2: Any = None,
        plate_ref_3: Any = None,
        capture_mode: str = "auto",
        camera_height_m: float = 0.0,
        match_quality: str = "balanced",
        seed: int = 0,
    ):
        np = _require_numpy()
        frames = [
            _require_photographed_frame("image_1", image_1, raw_meta_1, plate_ref_1, np),
            _require_photographed_frame("image_2", image_2, raw_meta_2, plate_ref_2, np),
        ]
        if image_3 is not None:
            frames.append(_require_photographed_frame(
                "image_3", image_3, raw_meta_3, plate_ref_3, np,
            ))
        elif raw_meta_3 is not None or plate_ref_3 is not None:
            raise RuntimeError(
                "AtlasMultiViewSolve: image_3 must be connected when raw_meta_3 or plate_ref_3 is supplied."
            )

        outcome = solve_multiview(
            frames,
            MultiViewSettings(
                capture_mode=capture_mode,
                camera_height_m=float(camera_height_m),
                match_quality=match_quality,
                seed=int(seed),
            ),
        )
        details = outcome.diagnostics.to_dict()
        if outcome.solve is None:
            code = outcome.diagnostics.outcome_code
            summary = outcome.diagnostics.summary
            debug_path = _write_failure_debug(details, outcome.overlays, np)
            debug_hint = (
                f"\nfailure diagnostics and overlays written to: {debug_path}"
                if debug_path else ""
            )
            raise RuntimeError(
                f"AtlasMultiViewSolve [{code}]: {summary}\n"
                f"registration diagnostics: {json.dumps(details, sort_keys=True)}"
                f"{debug_hint}"
            )

        registration_json = json.dumps(details, sort_keys=True)
        report = f"{outcome.diagnostics.outcome_code}: {outcome.diagnostics.summary}"
        return (
            outcome.solve,
            report,
            registration_json,
            _overlay_batch(outcome.overlays, np, _require_torch()),
        )


__all__ = ["AtlasMultiViewSolve"]
