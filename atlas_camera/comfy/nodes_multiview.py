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
    ) -> str:
        return _cache_fingerprint(
            image_1, image_2, image_3,
            raw_meta_1, raw_meta_2, raw_meta_3,
            plate_ref_1, plate_ref_2, plate_ref_3,
            capture_mode, camera_height_m, match_quality, seed,
            learned_anchor_fallback, baseline_m, learned_scale_fallback,
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


__all__ = ["AtlasMultiViewSolve", "AtlasMultiViewSolveBurst"]
