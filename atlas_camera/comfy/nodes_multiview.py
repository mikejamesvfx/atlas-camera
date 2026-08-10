"""Thin ComfyUI boundary for deterministic photographed multi-view solves.

The solver owns every registration decision.  This module only validates the
photographed RAW evidence ComfyUI hands it, adapts IMAGE tensors at the host
boundary, and converts the deterministic result back to ComfyUI values.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace as dataclass_replace
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
    learned_anchor_fallback: bool = False,
    baseline_m: float = 0.0,
    learned_scale_fallback: bool = False,
    pair_topology: str = "auto",
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
            "learned_anchor_fallback": bool(learned_anchor_fallback),
            "baseline_m": float(baseline_m),
            "learned_scale_fallback": bool(learned_scale_fallback),
            "pair_topology": str(pair_topology),
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


def _overlay_unit_float(overlay: Any, np: Any) -> Any:
    """Adapt a core overlay to 0..1 float32 regardless of its dtype.

    render_match_overlay returns uint8 0..255 canvases; treating those as
    already-unit floats white-clips every preview (found live on the first
    real X-H2 failure overlay).
    """
    pixels = np.ascontiguousarray(overlay)
    if pixels.dtype == np.uint8:
        return pixels.astype(np.float32) / 255.0
    return np.clip(pixels.astype(np.float32), 0.0, 1.0)


def _learned_anchor_up_hint(anchor_image: Any, np: Any) -> tuple[tuple[float, float, float], str]:
    """Run GeoCalib on the anchor plate and return (up_cam, source label).

    Torch stays at this adapter boundary: core receives only the resulting
    up vector via MultiViewSettings.anchor_up_hint.
    """
    try:
        from atlas_camera.inference.learned_prior import estimate_camera_prior
    except ImportError as exc:
        raise RuntimeError(
            "AtlasMultiViewSolve: learned_anchor_fallback needs the [neural] "
            "extra — pip install -e .[neural] and "
            "pip install \"git+https://github.com/cvg/GeoCalib.git\""
        ) from exc
    import tempfile
    from PIL import Image

    pixels = np.clip(np.ascontiguousarray(anchor_image, dtype=np.float32), 0.0, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anchor.png")
        Image.fromarray((pixels * 255.0 + 0.5).astype(np.uint8)).save(path)
        prior = estimate_camera_prior(path)
    up = tuple(float(v) for v in prior.up_cam)
    return (up[0], up[1], up[2]), f"learned prior ({prior.source_model})"


def _learned_metric_depth(anchor_image: Any, np: Any) -> Any:
    """Run the outdoor metric depth model on the anchor plate.

    Depth-model doctrine: exteriors use V2-Metric-Outdoor (the estimator's
    default).  Returns the HxW float32 metres map at plate resolution; torch
    stays at this adapter boundary — core only samples the array.
    """
    try:
        from atlas_camera.inference.depth_estimator import estimate_depth
    except ImportError as exc:
        raise RuntimeError(
            "AtlasMultiViewSolve: learned_scale_fallback needs the [neural] "
            "extra — pip install -e .[neural]"
        ) from exc
    import tempfile
    from PIL import Image

    pixels = np.clip(np.ascontiguousarray(anchor_image, dtype=np.float32), 0.0, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anchor.png")
        Image.fromarray((pixels * 255.0 + 0.5).astype(np.uint8)).save(path)
        result = estimate_depth(path)
    if not result.is_metric:
        raise RuntimeError(
            "AtlasMultiViewSolve: the configured depth model returned relative "
            "depth; the scale fallback needs a metric model."
        )
    return np.ascontiguousarray(result.depth, dtype=np.float32)


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
                    _overlay_unit_float(overlay, np) * 255.0,
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
        pixels = np.stack([_overlay_unit_float(overlay, np) for overlay in overlays])
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
                "match_quality": (["balanced", "conservative", "permissive", "salvage"], {"default": "balanced"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "learned_anchor_fallback": ("BOOLEAN", {"default": False}),
                "baseline_m": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "learned_scale_fallback": ("BOOLEAN", {"default": False}),
                "pair_topology": (["auto", "anchor_star"], {"default": "auto"}),
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
        learned_anchor_fallback: bool = False,
        baseline_m: float = 0.0,
        learned_scale_fallback: bool = False,
        pair_topology: str = "auto",
    ) -> str:
        return _cache_fingerprint(
            image_1, image_2, image_3,
            raw_meta_1, raw_meta_2, raw_meta_3,
            plate_ref_1, plate_ref_2, plate_ref_3,
            capture_mode, camera_height_m, match_quality, seed,
            learned_anchor_fallback, baseline_m, learned_scale_fallback,
            pair_topology,
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
        learned_anchor_fallback: bool = False,
        baseline_m: float = 0.0,
        learned_scale_fallback: bool = False,
        pair_topology: str = "auto",
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

        anchor_up_hint = None
        anchor_up_hint_source = ""
        if learned_anchor_fallback:
            anchor_up_hint, anchor_up_hint_source = _learned_anchor_up_hint(
                frames[0].image, np,
            )
        if learned_scale_fallback:
            frames[0] = dataclass_replace(
                frames[0], metric_depth=_learned_metric_depth(frames[0].image, np),
            )

        outcome = solve_multiview(
            frames,
            MultiViewSettings(
                capture_mode=capture_mode,
                camera_height_m=float(camera_height_m),
                match_quality=match_quality,
                seed=int(seed),
                anchor_up_hint=anchor_up_hint,
                anchor_up_hint_source=anchor_up_hint_source,
                baseline_m=float(baseline_m),
                pair_topology=pair_topology,
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


#: Every capture container the burst loader accepts — RAW bodies plus
#: camera-processed JPEG (the trusted-EXIF tier added 2026-08-10).
_BURST_EXTENSIONS = (
    ".raf", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2",
    ".dng", ".orf", ".rw2", ".pef", ".srw", ".rwl", ".3fr", ".fff", ".iiq",
    ".x3f", ".jpg", ".jpeg",
)


class AtlasMultiViewSolveBurst:
    """Recover one deterministic camera rig from a burst folder (2-16 frames).

    Feeds every selected file through the same RAW/JPEG import and the same
    registration engine as AtlasMultiViewSolve.  Beyond three frames the
    solver switches to an anchor-star pair topology — every frame must share
    overlap with the FIRST file in the folder (sorted by name), which is the
    natural shape of a walking burst.  frame_stride thins dense bursts;
    baseline_m, when set, is the measured distance between the first two
    SELECTED frames.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("solve", "report", "registration_json", "match_overlays",
                    "anchor_image")
    FUNCTION = "solve"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "burst_dir": ("STRING", {"default": "CameraRaw/burst"}),
            },
            "optional": {
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 16}),
                "max_frames": ("INT", {"default": 8, "min": 2, "max": 16}),
                "half_size": ("BOOLEAN", {"default": True}),
                "capture_mode": (["auto", "translated", "rotation_only"], {"default": "auto"}),
                "camera_height_m": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "match_quality": (["balanced", "conservative", "permissive", "salvage"], {"default": "balanced"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "learned_anchor_fallback": ("BOOLEAN", {"default": False}),
                "baseline_m": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "learned_scale_fallback": ("BOOLEAN", {"default": False}),
                "write_plates": ("BOOLEAN", {"default": True}),
                "plates_dir": ("STRING", {"default": "atlas_exports/burst_plates"}),
                "pair_topology": (["auto", "anchor_star"], {"default": "auto"}),
            },
        }

    @staticmethod
    def _resolve_input_dir(burst_dir: str) -> Any:
        from pathlib import Path
        raw = str(burst_dir or "").strip()
        path = Path(raw).expanduser()
        if not raw or path.is_absolute() or path.is_dir():
            return path
        try:
            import folder_paths  # ComfyUI runtime module; optional in tests.
            return Path(folder_paths.get_input_directory()) / path
        except (ImportError, AttributeError, TypeError):
            return path

    @classmethod
    def _selected_files(cls, burst_dir: str, frame_stride: int, max_frames: int):
        directory = cls._resolve_input_dir(burst_dir)
        if not directory.is_dir():
            raise RuntimeError(
                f"AtlasMultiViewSolveBurst: burst_dir is not a folder: {directory}"
            )
        files = sorted(
            item for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in _BURST_EXTENSIONS
        )
        selected = files[:: max(1, int(frame_stride))][: max(2, int(max_frames))]
        if len(selected) < 2:
            raise RuntimeError(
                "AtlasMultiViewSolveBurst: fewer than two capture files in "
                f"{directory} after stride {frame_stride} "
                f"({len(files)} candidates)."
            )
        return selected

    @classmethod
    def IS_CHANGED(
        cls,
        burst_dir: str = "CameraRaw/burst",
        frame_stride: int = 1,
        max_frames: int = 8,
        half_size: bool = True,
        capture_mode: str = "auto",
        camera_height_m: float = 0.0,
        match_quality: str = "balanced",
        seed: int = 0,
        learned_anchor_fallback: bool = False,
        baseline_m: float = 0.0,
        learned_scale_fallback: bool = False,
        write_plates: bool = True,
        plates_dir: str = "atlas_exports/burst_plates",
        pair_topology: str = "auto",
    ) -> str:
        try:
            files = cls._selected_files(burst_dir, frame_stride, max_frames)
            listing = [
                (str(item), int(item.stat().st_mtime_ns), int(item.stat().st_size))
                for item in files
            ]
        except Exception as exc:  # noqa: BLE001 — an unreadable dir must re-run.
            listing = [("error", str(exc))]
        payload = {
            "files": listing,
            "widgets": {
                "burst_dir": str(burst_dir),
                "frame_stride": int(frame_stride),
                "max_frames": int(max_frames),
                "half_size": bool(half_size),
                "capture_mode": capture_mode,
                "camera_height_m": float(camera_height_m),
                "match_quality": match_quality,
                "seed": int(seed),
                "learned_anchor_fallback": bool(learned_anchor_fallback),
                "baseline_m": float(baseline_m),
                "learned_scale_fallback": bool(learned_scale_fallback),
                "write_plates": bool(write_plates),
                "plates_dir": str(plates_dir),
                "pair_topology": str(pair_topology),
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _plate_ref_for_frame(result: Any, source_path: str, write_plates: bool,
                             plates_dir: str, np: Any) -> Any:
        """Build the per-frame plate reference every ProjectionSource projects.

        Reuses AtlasLoadRAW's EXR sidecar writer so every burst frame gets the
        same scene-linear plate the three-photo path gets from its loaders —
        this is what turns the extra burst cameras from geometry-only into
        texture-projecting layers.  A failed EXR write degrades to a preview
        proxy, exactly like the loader.
        """
        import base64
        import io

        from PIL import Image

        from atlas_camera.comfy.nodes_solve import AtlasLoadRAW
        from atlas_camera.core.schema import AtlasPlateRef

        exr_path = None
        exr_warning = ""
        if write_plates:
            exr_path, exr_warning = AtlasLoadRAW._write_exr_sidecar(
                result.linear_rgb, source_path, plates_dir,
                headroom=result.headroom,
            )
        pixels = (
            np.clip(np.asarray(result.display_srgb, dtype=np.float32), 0.0, 1.0)
            * 255.0 + 0.5
        ).astype(np.uint8)
        pil = Image.fromarray(pixels, mode="RGB")
        pil.thumbnail((1280, 1280))
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=85)
        preview = (
            "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
        return AtlasPlateRef(
            image_path=(str(exr_path) if exr_path else None),
            preview_b64=preview,
            colorspace="Linear Rec.709 (sRGB)",
            bit_depth="16f" if exr_path else "8-bit/proxy",
            role="source",
            is_proxy=exr_path is None,
            metadata={
                "registered_from": "AtlasMultiViewSolveBurst",
                "raw_source": str(source_path),
                "camera_model": result.camera_model,
                "undistort_status": result.undistort_status,
                **({"plate_warning": exr_warning} if exr_warning else {}),
            },
        )

    def solve(
        self,
        burst_dir: str = "CameraRaw/burst",
        frame_stride: int = 1,
        max_frames: int = 8,
        half_size: bool = True,
        capture_mode: str = "auto",
        camera_height_m: float = 0.0,
        match_quality: str = "balanced",
        seed: int = 0,
        learned_anchor_fallback: bool = False,
        baseline_m: float = 0.0,
        learned_scale_fallback: bool = False,
        write_plates: bool = True,
        plates_dir: str = "atlas_exports/burst_plates",
        pair_topology: str = "auto",
    ):
        np = _require_numpy()
        from atlas_camera.raw.pipeline import import_raw

        files = self._selected_files(burst_dir, frame_stride, max_frames)
        frames = []
        for item in files:
            result = import_raw(str(item), half_size=bool(half_size))
            frames.append(MultiViewFrame(
                image=np.ascontiguousarray(result.display_srgb, dtype=np.float32),
                raw_meta=result,
                label=item.name,
                plate_ref=self._plate_ref_for_frame(
                    result, str(item), bool(write_plates), str(plates_dir), np,
                ),
            ))

        anchor_up_hint = None
        anchor_up_hint_source = ""
        if learned_anchor_fallback:
            anchor_up_hint, anchor_up_hint_source = _learned_anchor_up_hint(
                frames[0].image, np,
            )
        if learned_scale_fallback:
            frames[0] = dataclass_replace(
                frames[0], metric_depth=_learned_metric_depth(frames[0].image, np),
            )

        outcome = solve_multiview(
            frames,
            MultiViewSettings(
                capture_mode=capture_mode,
                camera_height_m=float(camera_height_m),
                match_quality=match_quality,
                seed=int(seed),
                anchor_up_hint=anchor_up_hint,
                anchor_up_hint_source=anchor_up_hint_source,
                baseline_m=float(baseline_m),
                pair_topology=pair_topology,
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
                f"AtlasMultiViewSolveBurst [{code}]: {summary}\n"
                f"registration diagnostics: {json.dumps(details, sort_keys=True)}"
                f"{debug_hint}"
            )

        torch = _require_torch()
        anchor_batch = torch.from_numpy(np.ascontiguousarray(
            frames[0].image[None, ...], dtype=np.float32,
        ))
        report = (
            f"{outcome.diagnostics.outcome_code}: {outcome.diagnostics.summary} "
            f"({len(frames)} frames from {files[0].parent.name}/)"
        )
        return (
            outcome.solve,
            report,
            json.dumps(details, sort_keys=True),
            _overlay_batch(outcome.overlays, np, torch),
            anchor_batch,
        )


def _photographed_patch_sources(solve: Any) -> list[Any]:
    """The registered flanking photographs on a solve, anchor EXCLUDED.

    The anchor is deliberately not a candidate. Every hole here IS a hole in the
    anchor's own plate, so nominating the anchor would answer "patch it from the
    photograph that has the hole" — and on a real depth tear the occluder that
    made the hole is still standing in front of the fill planes anyway.
    """
    sources = list(getattr(solve, "projection_sources", None) or [])
    return [
        source for source in sources
        if (source.metadata or {}).get("evidence_type") == "photographed"
        and getattr(source, "camera", None) is not None
    ]


def _patch_plate_pixels(source: Any, np: Any) -> tuple[Any, str]:
    """(H, W, 3) float32 pixels for one photographed source, plus provenance.

    Prefers the durable float plate over the browser preview: `preview_b64` is a
    JPEG thumbnailed to 1280 px long side, so a crop taken from it is both lossy
    and at a different scale from the camera that was ranked. Returns
    ``(None, reason)`` when neither is readable — a black patch that claims to be
    photographic evidence is worse than an honest refusal.
    """
    plate_ref = getattr(source, "plate_ref", None)
    image_path = getattr(plate_ref, "image_path", None) if plate_ref else None
    if image_path and os.path.exists(str(image_path)):
        try:
            from atlas_camera.plate.oiio_io import read_plate

            read = read_plate(str(image_path))
            pixels = np.ascontiguousarray(read.pixels, dtype=np.float32)
            if pixels.ndim == 3 and pixels.shape[2] >= 3:
                return pixels[..., :3], f"float plate {os.path.basename(str(image_path))}"
        except Exception as exc:  # OIIO missing, or an unreadable/moved plate
            plate_error = f"{type(exc).__name__}: {exc}"
        else:
            plate_error = "plate had no RGB channels"
    else:
        plate_error = "no durable plate on disk"

    b64 = (getattr(plate_ref, "preview_b64", None) if plate_ref else None) \
        or getattr(source, "image_b64", None)
    if b64:
        try:
            import base64
            import io

            from PIL import Image

            raw = base64.b64decode(str(b64).split(",", 1)[-1])
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            pixels = np.asarray(pil, dtype=np.float32) / 255.0
            return pixels, f"preview proxy {pil.width}x{pil.height} ({plate_error})"
        except Exception as exc:
            return None, f"{plate_error}; preview undecodable ({type(exc).__name__})"
    return None, plate_error


class AtlasSolveBurstPatchCrops:
    """📷✂️ Which registered photograph sees into this hole — and where in it.

    Answers the middle-anchor question: the anchor frame carries the shot, but the
    frames either side of it in the burst stood somewhere else and photographed the
    surfaces the anchor could not see. This node rasterizes the hole's fill planes
    against every registered flanking camera, ranks them by revealed pixels, and
    returns the exact crop out of the winning photograph's own plate.

    The crop is REAL PHOTOGRAPHIC EVIDENCE, not a generated patch — which is why it
    comes from that frame's `plate_ref` and its ROI is reported in that frame's
    pixel space, never the anchor's.
    """

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("cropped_patch", "patch_frame_index", "crop_roi", "report")
    FUNCTION = "solve_crops"
    CATEGORY = "Atlas/multiview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "hole_mask": ("MASK",),
            },
            "optional": {
                "margin_px": ("INT", {"default": 16, "min": 0, "max": 256}),
                "resolution": ("INT", {"default": 384, "min": 128, "max": 1536}),
                "min_visible_pixels": ("INT", {"default": 8, "min": 1, "max": 10000}),
            },
        }

    @staticmethod
    def _empty(torch: Any, roi: str, message: str):
        return (
            torch.zeros((1, 64, 64, 3), dtype=torch.float32),
            -1,
            roi,
            f"AtlasSolveBurstPatchCrops: {message}",
        )

    def solve_crops(
        self,
        solve,
        hole_mask,
        margin_px=16,
        resolution=384,
        min_visible_pixels=8,
    ):
        from atlas_camera.comfy.nodes import _relief_mesh_from_solve
        from atlas_camera.core.view_solver import rank_burst_frames

        np = _require_numpy()
        torch = _require_torch()

        intr = solve.camera.intrinsics
        width = int(intr.image_width or 0)
        height = int(intr.image_height or 0)
        full_roi = f"0,0,{width},{height}"

        mesh = _relief_mesh_from_solve(solve)
        if mesh is None:
            return self._empty(torch, full_roi, (
                "no relief mesh on this solve — run AtlasDeriveReliefMesh upstream first."
            ))

        sources = _photographed_patch_sources(solve)
        if not sources:
            return self._empty(torch, full_roi, (
                "this solve carries no registered flanking photographs. Patch crops need a "
                "multi-view rig (AtlasMultiViewSolveBurst / AtlasMultiViewSolve) — a "
                "single-image solve has only the camera that owns the hole."
            ))

        mask_np = self._hole_mask_array(hole_mask, width, height, np, torch)

        scores = rank_burst_frames(
            mesh,
            mask_np,
            source_camera=solve.camera,
            burst_cameras=[source.camera for source in sources],
            resolution=int(resolution),
            margin_px=int(margin_px),
            min_visible_pixels=int(min_visible_pixels),
        )

        def _label(score: Any) -> str:
            source = sources[score.frame_index]
            frame_index = (source.metadata or {}).get("frame_index", score.frame_index)
            return f"{source.name} (frame {frame_index})"

        if not scores or scores[0].visible_px == 0:
            return self._empty(torch, full_roi, (
                f"none of the {len(sources)} registered photograph(s) reveal these hole "
                "planes. The occluded surface was never photographed from a second angle — "
                "send it to a clean plate or a generated patch instead."
            ))

        best = scores[0]
        best_source = sources[best.frame_index]
        best_frame = int((best_source.metadata or {}).get("frame_index", best.frame_index))
        u_min, v_min, u_max, v_max = best.crop_roi

        pixels, provenance = _patch_plate_pixels(best_source, np)
        if pixels is None:
            return self._empty(torch, f"{u_min},{v_min},{u_max},{v_max}", (
                f"{_label(best)} reveals {best.visible_px} px of hole geometry but its plate "
                f"could not be read ({provenance}). Re-run the solve with write_plates on, or "
                "keep the burst frames on disk."
            ))

        # The ROI is in the winning camera's own full-resolution pixel space. A preview
        # proxy is a scaled copy of that space, so the box has to be scaled with it —
        # cropping proxy pixels with full-res coordinates is how a patch silently
        # becomes the wrong part of the right photograph.
        cam_intr = best_source.camera.intrinsics
        cam_w = int(cam_intr.image_width or pixels.shape[1])
        cam_h = int(cam_intr.image_height or pixels.shape[0])
        px_scale_x = pixels.shape[1] / max(cam_w, 1)
        px_scale_y = pixels.shape[0] / max(cam_h, 1)
        c_u0 = max(0, min(pixels.shape[1] - 1, int(round(u_min * px_scale_x))))
        c_v0 = max(0, min(pixels.shape[0] - 1, int(round(v_min * px_scale_y))))
        c_u1 = max(c_u0 + 1, min(pixels.shape[1], int(round(u_max * px_scale_x))))
        c_v1 = max(c_v0 + 1, min(pixels.shape[0], int(round(v_max * px_scale_y))))

        crop = np.ascontiguousarray(
            pixels[c_v0:c_v1, c_u0:c_u1, :][None, ...], dtype=np.float32,
        )

        ranked = ", ".join(
            f"{_label(score)} {score.visible_px}px" for score in scores[:5]
        )
        report = "\n".join([
            f"AtlasSolveBurstPatchCrops: best = {_label(best)} — {best.visible_px} px of "
            f"hole geometry across {best.islands_seen} island(s)",
            f"crop {u_max - u_min}x{v_max - v_min} px at ({u_min}, {v_min}) in that frame; "
            f"pixels from {provenance}",
            f"ranked ({len(sources)} photographed source(s)): {ranked}",
        ])

        return (
            torch.from_numpy(crop),
            best_frame,
            f"{u_min},{v_min},{u_max},{v_max}",
            report,
        )

    @staticmethod
    def _hole_mask_array(hole_mask: Any, width: int, height: int, np: Any, torch: Any):
        """A ComfyUI MASK as a (height, width) bool array in the anchor's frame."""
        if isinstance(hole_mask, torch.Tensor):
            array = hole_mask.detach().cpu().numpy()
        else:
            array = np.asarray(hole_mask)
        array = np.asarray(array, dtype=np.float32)
        while array.ndim > 2:
            array = array[0]
        if array.ndim != 2:
            raise RuntimeError(
                f"AtlasSolveBurstPatchCrops: hole_mask must be 2D per batch item, got "
                f"shape {tuple(np.shape(hole_mask))}."
            )
        selected = array > 0.5
        if selected.shape != (height, width):
            from PIL import Image

            resized = Image.fromarray(selected.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST,
            )
            selected = np.asarray(resized) > 127
        return selected


__all__ = ["AtlasMultiViewSolve", "AtlasMultiViewSolveBurst", "AtlasSolveBurstPatchCrops"]
