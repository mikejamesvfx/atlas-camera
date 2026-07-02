"""ComfyUI node library for Atlas Camera."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.solver import solve_from_constraints, solve_still_image
from atlas_camera.exporters.blender_exporter import write_blender_scene_script
from atlas_camera.exporters.nuke_exporter import write_nuke_projection_script
from atlas_camera.exporters.review_package import build_review_package
from atlas_camera.importers.usd_camera_loader import USDCameraLoader

# Module-level cache: node_id → camera_data dict, populated by AtlasBlockoutViewport.render()
# Capped at 64 entries to prevent unbounded growth in long ComfyUI sessions.
_ATLAS_BLOCKOUT_CACHE: dict[str, dict[str, Any]] = {}
_ATLAS_BLOCKOUT_CACHE_MAX = 64


def _blockout_cache_set(node_id: str, data: dict[str, Any]) -> None:
    if len(_ATLAS_BLOCKOUT_CACHE) >= _ATLAS_BLOCKOUT_CACHE_MAX:
        # Evict the oldest entry (dict preserves insertion order in Python 3.7+)
        oldest = next(iter(_ATLAS_BLOCKOUT_CACHE))
        del _ATLAS_BLOCKOUT_CACHE[oldest]
    _ATLAS_BLOCKOUT_CACHE[node_id] = data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise RuntimeError(
            "This node requires numpy. Install with: pip install -e .[vision]"
        ) from exc


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "This node requires PyTorch, which should be present in any ComfyUI environment."
        ) from exc


def _require_pil():
    try:
        from PIL import Image
        return Image
    except ImportError as exc:
        raise RuntimeError(
            "This node requires Pillow. Install with: pip install Pillow"
        ) from exc


def _image_tensor_to_pil(image_tensor):
    """Convert ComfyUI IMAGE tensor (1×H×W×3 float32) to a PIL Image (RGB)."""
    PILImage = _require_pil()
    arr = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return PILImage.fromarray(arr, mode="RGB")


def _pil_to_image_tensor(pil_img):
    """Convert PIL Image to ComfyUI IMAGE tensor (1×H×W×3 float32)."""
    np = _require_numpy()
    torch = _require_torch()
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # 1×H×W×3


def _save_image_tensor_to_tmp(image_tensor) -> str:
    """Write a ComfyUI IMAGE tensor to a temp PNG and return the path."""
    pil = _image_tensor_to_pil(image_tensor)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil.save(f.name, format="PNG")
        return f.name


def _decode_b64_to_tensor(b64str: str, width: int, height: int):
    """Decode a base64-encoded PNG/JPEG string to a ComfyUI IMAGE tensor."""
    torch = _require_torch()
    PILImage = _require_pil()
    np = _require_numpy()
    if not b64str:
        return torch.zeros(1, height, width, 3, dtype=torch.float32)
    try:
        raw = base64.b64decode(b64str.split(",", 1)[-1])
        img = PILImage.open(io.BytesIO(raw)).convert("RGB").resize((width, height))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return torch.zeros(1, height, width, 3, dtype=torch.float32)


def _extract_blockout_camera(solve, source_image, target_width: int, target_height: int,
                              preview_expand: float = 1.0) -> dict[str, Any]:
    """Serialize the recovered camera into a dict the browser extension can consume."""
    cam = solve.camera
    intr = cam.intrinsics
    extr = cam.extrinsics
    fx = intr.fx_px or 0.0
    fy = intr.fy_px or fx
    cx = intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0
    cy = intr.cy_px if intr.cy_px is not None else intr.image_height / 2.0
    # view_matrix is the Atlas camera_view_matrix (4×4, row-major)
    vm = [list(row) for row in extr.camera_view_matrix]

    # Encode source image as JPEG base64 so the browser can use it as background
    source_b64 = ""
    try:
        PILImage = _require_pil()
        pil = _image_tensor_to_pil(source_image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        source_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass

    # Derived projection proxies (ground/walls/boxes/cylinders/backdrop) for the
    # viewport to build; empty list when nothing was derived. preview_expand>1
    # dilates them outward from the camera for wider orbit coverage — display
    # only, never mutates the primitives stored on the solve.
    from atlas_camera.core.proxy_geometry import serialize_proxy_geometry
    proxy_geometry = serialize_proxy_geometry(
        solve.projection_scene,
        preview_expand=preview_expand,
        preview_pivot=extr.camera_position,
    )

    # Vanishing points + horizon (2D image-space diagnostics — meaningful only
    # against the flat source photo, not the 3D scene) for the viewport's
    # layered VP/horizon/ground overlay.
    vanishing_points = [
        {
            "position_px": list(vp.position_px),
            "direction_label": vp.direction_label,
            "confidence": float(vp.confidence),
        }
        for vp in (solve.vanishing_points or [])
    ]
    horizon_line = None
    if solve.horizon_line is not None:
        horizon_line = {
            "endpoints_px": [list(p) for p in solve.horizon_line.endpoints_px]
                            if solve.horizon_line.endpoints_px else None,
            "line_coefficients": list(solve.horizon_line.line_coefficients),
            "confidence": float(solve.horizon_line.confidence),
        }

    # Solved latent-camera metadata (lens, distance, provenance) for the HUD.
    fov_h_deg = None
    if fx > 0 and intr.image_width:
        fov_h_deg = math.degrees(2.0 * math.atan(intr.image_width / (2.0 * fx)))
    scene_depth_m = None
    for prim in solve.projection_scene.proxy_geometry:
        if prim.name == "projection_backdrop":
            scene_depth_m = (prim.metadata or {}).get("distance_m")
            break
    camera_meta = {
        "confidence": float(getattr(solve, "confidence", 0.0) or 0.0),
        "source_method": getattr(solve, "source_method", None),
        "scale_source": (solve.debug_metadata or {}).get("scale_source"),
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "fov_h_deg": fov_h_deg,
        "camera_height_m": float(extr.camera_position[1]) if extr.camera_position else None,
        "scene_depth_m": scene_depth_m,
    }

    return {
        "view_matrix": vm,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_position": list(extr.camera_position),
        "image_width": intr.image_width,
        "image_height": intr.image_height,
        "target_width": target_width,
        "target_height": target_height,
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "source_image_b64": source_b64,
        "proxy_geometry": proxy_geometry,
        "vanishing_points": vanishing_points,
        "horizon_line": horizon_line,
        "camera_meta": camera_meta,
    }


def _ground_depth_compute(solve, width: int, height: int, near: float, far: float):
    """
    Per-pixel ray-plane intersection against Y=0 ground plane.
    Returns (depth_rgb, valid_mask) both as H×W numpy float32 arrays.
    Port of DEPTH_FRAGMENT_SHADER in ui/src/ProjectionMaterial.ts.
    """
    np = _require_numpy()

    cam = solve.camera
    intr = cam.intrinsics
    extr = cam.extrinsics

    fx = intr.fx_px or 0.0
    fy = intr.fy_px or fx
    if fx <= 0 or fy <= 0:
        return None, None

    cx = intr.cx_px if intr.cx_px is not None else width / 2.0
    cy = intr.cy_px if intr.cy_px is not None else height / 2.0

    vm = np.array(extr.camera_view_matrix, dtype=np.float64)  # 4×4
    cam_to_world = np.linalg.inv(vm)
    cam_y = float(extr.camera_position[1])

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))

    # Camera-space rays (cam looks along -Z, image Y is downward)
    ray_x = (uu - cx) / fx
    ray_y = -(vv - cy) / fy
    ray_z = -np.ones((height, width), dtype=np.float64)
    rays_cam = np.stack([ray_x, ray_y, ray_z], axis=-1)  # H×W×3
    norms = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_cam = rays_cam / np.maximum(norms, 1e-12)

    # Rotate to world space (direction only — upper-left 3×3 of camToWorld)
    R = cam_to_world[:3, :3]
    rays_world = rays_cam @ R.T  # H×W×3

    ry = rays_world[..., 1]  # H×W

    # Ground intersect: cameraPos.y + t * ry = 0  →  t = -cam_y / ry
    valid = (np.abs(ry) > 1e-5) & (cam_y > 0)
    t = np.where(valid, -cam_y / ry, 0.0)
    valid = valid & (t > 0.001)

    # Normalize to [0, 1] in [near, far]
    t_norm = np.clip((t - near) / max(far - near, 1e-6), 0.0, 1.0)
    t_norm[~valid] = 0.0

    # 4-stop warm→cool heatmap (identical stops to DEPTH_FRAGMENT_SHADER)
    c0 = np.array([0.90, 0.12, 0.04], dtype=np.float32)  # near: red
    c1 = np.array([0.96, 0.72, 0.08], dtype=np.float32)  # yellow
    c2 = np.array([0.20, 0.84, 0.60], dtype=np.float32)  # teal
    c3 = np.array([0.08, 0.22, 0.86], dtype=np.float32)  # far: blue

    t3 = t_norm[..., np.newaxis].astype(np.float32)
    rgb = np.where(t3 < 0.333,
                   c0 + (c1 - c0) * (t3 * 3.0),
                   np.where(t3 < 0.667,
                            c1 + (c2 - c1) * ((t3 - 0.333) * 3.0),
                            c2 + (c3 - c2) * ((t3 - 0.667) * 3.0)))
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb[~valid] = 0.0

    return rgb.astype(np.float32), valid.astype(np.float32)


# ---------------------------------------------------------------------------
# Existing nodes (unchanged)
# ---------------------------------------------------------------------------

class AtlasLoadImageSolveCamera:
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
                "image_width": ("INT", {"default": 0, "min": 0,
                                        "tooltip": "0 = auto (read from the image file)"}),
                "image_height": ("INT", {"default": 0, "min": 0,
                                         "tooltip": "0 = auto (read from the image file)"}),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 35.0, "min": 0.0}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
            },
        }

    def solve(self, image_path, image_width, image_height,
              focal_length_mm=None, sensor_width_mm=36.0):
        hints = {}
        if focal_length_mm:
            hints["focal_length_mm"] = focal_length_mm
            hints["sensor_width_mm"] = sensor_width_mm
        # 0×0 → let the solver read the image's true dimensions from the file.
        image_size = (image_width, image_height) if (image_width and image_height) else None
        return (solve_still_image(image_path,
                                  image_size=image_size,
                                  intrinsics_hint=hints,
                                  detect_vanishing_points=True),)


class AtlasExportReviewPackage:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            }
        }

    def export(self, solve, output_dir):
        result = build_review_package(solve, output_dir)
        return (str(result.package_dir),)


class AtlasExportSolveJSON:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_path": ("STRING", {"default": "atlas_solve.json"}),
            }
        }

    def export(self, solve, output_path):
        return (str(save_solve_json(solve, output_path)),)


class AtlasExportMayaReviewScene:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            }
        }

    def export(self, solve, output_dir):
        result = build_review_package(solve, output_dir, include_usd=False)
        return (str(result.files["maya_open_scene"]),)


class AtlasUSDCameraLoader:
    RETURN_TYPES = ("ATLAS_CAMERA",)
    FUNCTION = "load"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "usd_path": ("STRING", {"default": ""}),
                "image_width": ("INT", {"default": 1920, "min": 1}),
                "image_height": ("INT", {"default": 1080, "min": 1}),
            }
        }

    def load(self, usd_path, image_width, image_height):
        return (USDCameraLoader().load(usd_path, image_size=(image_width, image_height)),)


# ---------------------------------------------------------------------------
# Track 1 — New Python-only nodes
# ---------------------------------------------------------------------------

class AtlasSolveFromImage:
    """Solve camera from a ComfyUI IMAGE tensor (no file path needed)."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 0.0, "min": 0.0,
                                              "tooltip": "0 = auto-detect or EXIF"}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
                "detect_vanishing_points": ("BOOLEAN", {"default": True,
                    "tooltip": "Run line/VP detection. Off = metadata-only solve "
                               "(no fx, cam_y=0 -> black depth/blockout)."}),
            },
        }

    def solve(self, image, focal_length_mm=0.0, sensor_width_mm=36.0,
              detect_vanishing_points=True):
        tmp = _save_image_tensor_to_tmp(image)
        try:
            hints: dict[str, Any] = {}
            if focal_length_mm and focal_length_mm > 0:
                hints["focal_length_mm"] = focal_length_mm
                hints["sensor_width_mm"] = sensor_width_mm
            return (solve_still_image(tmp, intrinsics_hint=hints or None,
                                      detect_vanishing_points=detect_vanishing_points),)
        finally:
            os.unlink(tmp)


class AtlasConstrainedSolve:
    """Guided solve using line constraints JSON (from Atlas UI or hand-crafted)."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "constraints_json": ("STRING", {"default": "{}", "multiline": True,
                                                "tooltip": "Atlas constraints dict with line_groups and scale_constraints"}),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
            },
        }

    def solve(self, image, constraints_json, focal_length_mm=0.0, sensor_width_mm=36.0):
        constraints = json.loads(constraints_json)
        tmp = _save_image_tensor_to_tmp(image)
        try:
            hint: dict[str, Any] | None = None
            if focal_length_mm and focal_length_mm > 0:
                hint = {"focal_length_mm": focal_length_mm, "sensor_width_mm": sensor_width_mm}
            return (solve_from_constraints(tmp, constraints, intrinsics_hint=hint),)
        finally:
            os.unlink(tmp)


class AtlasLearnedSolveFromImage:
    """Solve a camera from a ComfyUI IMAGE using the learned GeoCalib prior.

    Robust alternative to vanishing-point detection for AI-generated images:
    predicts focal length and gravity (up-vector) directly from image content, so
    it does not depend on clean straight edges converging to consistent VPs.
    Requires the [neural] extra (torch + geocalib) in ComfyUI's venv.
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "height_mode": (["measure_from_depth", "assume"], {"default": "measure_from_depth",
                    "tooltip": "measure_from_depth = fit the ground plane with Depth Anything V2 "
                               "(no assumed eye height); assume = use camera_height_m."}),
                "camera_height_m": ("FLOAT", {"default": 1.6, "min": 0.01, "max": 1000.0,
                    "tooltip": "Fallback / assumed camera height when not measured or low-confidence."}),
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "tooltip": "Metric depth model for height measurement (Outdoor=exteriors, Indoor=interiors)."}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
                "weights": (["pinhole", "simple_radial"], {"default": "pinhole",
                    "tooltip": "pinhole = no lens distortion (best for clean AI renders)."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    def solve(self, image, height_mode="measure_from_depth", camera_height_m=1.6,
              depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
              sensor_width_mm=36.0, weights="pinhole", device="auto"):
        from atlas_camera.core.solver import solve_still_image_learned
        tmp = _save_image_tensor_to_tmp(image)
        try:
            h, w = int(image.shape[1]), int(image.shape[2])
            camera_height = "auto" if height_mode == "measure_from_depth" else camera_height_m
            return (solve_still_image_learned(
                tmp,
                image_size=(w, h),
                camera_height=camera_height,
                sensor_width_mm=sensor_width_mm,
                weights=weights,
                depth_model=depth_model,
                device=None if device == "auto" else device,
            ),)
        finally:
            os.unlink(tmp)


class AtlasDepthAnything:
    """Monocular depth (Depth Anything V2) as a standalone IMAGE + the raw solve depth slot.

    Outputs a normalized grayscale depth image for preview/compositing. Requires the
    [neural] extra (torch + transformers) in ComfyUI's venv.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_image",)
    FUNCTION = "estimate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Small-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    def estimate(self, image, depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto"):
        from atlas_camera.inference.depth_estimator import estimate_depth
        np = _require_numpy()
        torch = _require_torch()
        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device)
            d = result.depth.astype(np.float32)
            # Normalize for viewing: near=bright, far=dark.
            lo, hi = float(d.min()), float(d.max())
            norm = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
            gray = 1.0 - norm
            rgb = np.stack([gray, gray, gray], axis=-1)
            return (torch.from_numpy(rgb).unsqueeze(0),)
        finally:
            os.unlink(tmp)


def _reference_id_choices() -> list[str]:
    try:
        from atlas_camera.reference_data import load_scale_references
        return [r.id for r in load_scale_references()]
    except Exception:
        return ["person_175cm", "door_210cm", "sedan_car"]


class AtlasReferenceScaleSolve:
    """Fix a solve's metric scale from a known-size reference object.

    The most reliable way to set absolute camera height: mark the pixel box of a
    known object (person, door, car, …) and Atlas solves the metric camera height
    by single-view geometry using the solve's orientation + focal — no assumed
    eye height. Composable after any solve node (e.g. the learned GeoCalib solve).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "FLOAT")
    RETURN_NAMES = ("solve", "camera_height_m")
    FUNCTION = "apply"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "reference_id": (_reference_id_choices(), ),
                "bbox_x0": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0}),
                "bbox_y0": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0,
                                      "tooltip": "Top edge (smaller y) of the object box."}),
                "bbox_x1": ("FLOAT", {"default": 100.0, "min": 0.0, "step": 1.0}),
                "bbox_y1": ("FLOAT", {"default": 400.0, "min": 0.0, "step": 1.0,
                                      "tooltip": "Bottom edge (larger y) — the object's base on the ground."}),
            },
            "optional": {
                "height_override_m": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "tooltip": "0 = use the reference's registry height; else override in metres."}),
            },
        }

    def apply(self, solve, reference_id, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
              height_override_m=0.0):
        from atlas_camera.core.solver import apply_reference_scale
        ref: dict[str, Any] = {
            "reference_id": reference_id,
            "bbox_px": [bbox_x0, bbox_y0, bbox_x1, bbox_y1],
        }
        if height_override_m and height_override_m > 0:
            ref["height_m"] = height_override_m
        apply_reference_scale(solve, [ref])
        return (solve, float(solve.camera.extrinsics.camera_position[1]))


class AtlasVLMScaleCues:
    """Detect scale-reference objects with a local vision-language model.

    Runs a local VLM (LM Studio / llama.cpp / Ollama) to find known-size objects
    (people, doors, cars, …) and emits ``scale_references`` JSON for
    AtlasApplyScaleReferences. Requires a running local VLM server and the model to
    return pixel bounding boxes. Advisory only — nothing is applied without the
    artist confirming in AtlasApplyScaleReferences.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scale_references", "summary")
    FUNCTION = "analyze"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "provider": (["ollama", "lmstudio", "llamacpp"], {"default": "ollama"}),
                "model": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "", "tooltip": "Blank = provider default URL"}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def analyze(self, image, provider="ollama", model="", base_url="", min_confidence=0.0):
        from atlas_camera.inference.multimodal_helper import (
            create_multimodal_provider,
            scale_references_from_observation,
        )
        from atlas_camera.reference_data import load_scale_references

        tmp = _save_image_tensor_to_tmp(image)
        try:
            candidate_ids = [r.id for r in load_scale_references()]
            prov = create_multimodal_provider(provider, model=model, base_url=base_url or None)
            obs = prov.analyze_image(tmp, candidate_reference_ids=candidate_ids)
            refs = scale_references_from_observation(obs, min_confidence=min_confidence)
            lines = [obs.summary or "VLM analysis complete."]
            for r in refs:
                target = r.get("reference_id") or f"{r.get('height_m')} m"
                lines.append(f"• {r.get('label')} → {target}  bbox={r['bbox_px']}  conf={r['confidence']:.2f}")
            if not refs:
                lines.append("(no usable scale references detected)")
            return (json.dumps(refs), "\n".join(str(s) for s in lines if s))
        except Exception as exc:  # provider offline / model missing — fail soft
            return ("[]", f"VLM scale cues unavailable: {exc}")
        finally:
            os.unlink(tmp)


class AtlasApplyScaleReferences:
    """Apply VLM/JSON scale references to a solve — only when the artist confirms.

    Takes ``scale_references`` JSON (from AtlasVLMScaleCues or hand-written) and,
    when ``confirm`` is on, rescales the solve's metric camera height via single-view
    geometry. With ``confirm`` off the references are recorded as candidates only
    (LLM cues are never auto-promoted; the toggle is the one-click confirmation).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "FLOAT", "STRING")
    RETURN_NAMES = ("solve", "camera_height_m", "report")
    FUNCTION = "apply"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "scale_references": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "JSON list of scale references (from AtlasVLMScaleCues)."}),
            },
            "optional": {
                "confirm": ("BOOLEAN", {"default": False,
                    "tooltip": "Confirm to actually rescale the camera. Off = record candidates only."}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def apply(self, solve, scale_references, confirm=False, min_confidence=0.0):
        from atlas_camera.core.solver import apply_reference_scale
        try:
            refs = json.loads(scale_references) if scale_references.strip() else []
        except json.JSONDecodeError:
            refs = []
        if not isinstance(refs, list):
            refs = []
        if min_confidence > 0:
            refs = [r for r in refs if float(r.get("confidence", 1.0)) >= min_confidence]

        apply_reference_scale(solve, refs, adopt=bool(confirm))
        rs = solve.debug_metadata.get("reference_scale", {})
        report = json.dumps({
            "confirmed": bool(confirm),
            "adopted": rs.get("adopted"),
            "scale_source": solve.debug_metadata.get("scale_source"),
            "camera_height_m": rs.get("camera_height_m"),
            "confidence": rs.get("confidence"),
            "references_in": len(refs),
        }, indent=2)
        return (solve, float(solve.camera.extrinsics.camera_position[1]), report)


class AtlasDeriveProjectionGeometry:
    """Derive camera-projection proxy geometry (ground/walls/boxes/cylinders/backdrop)
    from a Depth Anything V2 depth map + the solve's recovered camera.

    The blockout viewport builds these primitives and can project the source image
    onto them from the recovered camera — the classic VFX matte-painting setup.
    Requires the [neural] extra (re-runs metric depth internally; the IMAGE from
    AtlasDepthAnything is normalized and unusable for metric geometry).

    ``primitive_method`` selects how "primitives" mode derives geometry
    (only relevant when ``geometry_mode`` includes "primitives"):
    - ``azimuth_walls`` (default) — vertical walls only, general-purpose.
    - ``ransac_planes`` — any-orientation planes (sloped roofs, stepped/angled
      facades) via sequential RANSAC seeded by a 2D normal-orientation
      histogram. Best for exterior/architectural shots.
    - ``room_cuboid`` — Manhattan-aligned floor + up to 4 walls + optional
      ceiling. Best for orthogonal interiors; silently produces skewed walls
      on non-orthogonal rooms (pick a different method for those shots).
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 8}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 6,
                                        "tooltip": "Max foreground boxes/cylinders."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "geometry_mode": (["relief_mesh", "primitives", "both"], {"default": "relief_mesh",
                    "tooltip": "What the viewport receives. relief_mesh = contoured depth mesh "
                               "(recommended); primitives = flat blockout planes/boxes; both "
                               "overlaps the two on the same surfaces (enclosure + z-shimmer)."}),
                "relief_grid": ("INT", {"default": 96, "min": 16, "max": 256,
                    "tooltip": "Viewport relief-mesh density (long-edge grid columns)."}),
                "primitive_method": (["azimuth_walls", "ransac_planes", "room_cuboid"],
                    {"default": "azimuth_walls",
                     "tooltip": "azimuth_walls (default) = vertical walls only. "
                                "ransac_planes = any-orientation planes (roofs, stepped "
                                "facades) — exteriors. room_cuboid = Manhattan floor+walls"
                                "+ceiling — orthogonal interiors. Only affects "
                                "geometry_mode=primitives/both; max_walls is reused as the "
                                "plane budget for ransac_planes and ignored by room_cuboid."}),
            },
        }

    def derive(self, solve, image,
               depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
               max_walls=4, max_objects=3, device="auto",
               geometry_mode="relief_mesh", relief_grid=96,
               primitive_method="azimuth_walls"):
        from atlas_camera.core.plane_extraction import PlaneRansacConfig, extract_planes_ransac
        from atlas_camera.core.proxy_geometry import (
            PROXY_ROLE,
            ProxyDerivationConfig,
            derive_projection_proxies,
            relief_mesh_primitive,
        )
        from atlas_camera.core.relief_mesh import build_relief_mesh
        from atlas_camera.core.room_layout import RoomCuboidConfig, extract_room_cuboid
        from atlas_camera.core.schema import AtlasSolve
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.inference.depth_estimator import estimate_depth

        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        width = int(intr.image_width or image.shape[2])
        height = int(intr.image_height or image.shape[1])
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        if fx <= 0:
            # No focal — cannot back-project; return the solve untouched.
            return (solve,)
        cx = intr.cx_px if intr.cx_px is not None else width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else height / 2.0

        depth_map = result.depth
        if depth_map.shape != (height, width):
            depth_map = _resize_depth(depth_map, width, height)

        horizon_y = None
        if solve.horizon_line and solve.horizon_line.endpoints_px:
            p1, p2 = solve.horizon_line.endpoints_px
            horizon_y = 0.5 * (float(p1[1]) + float(p2[1]))

        if primitive_method == "ransac_planes":
            prims, stats = extract_planes_ransac(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                max_planes=max(int(max_walls), 1) * 2,
                horizon_y=horizon_y,
                config=PlaneRansacConfig(),
            )
        elif primitive_method == "room_cuboid":
            prims, stats = extract_room_cuboid(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=horizon_y,
                config=RoomCuboidConfig(),
            )
        else:
            cfg = ProxyDerivationConfig(max_objects=int(max_objects))
            prims, stats = derive_projection_proxies(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                max_walls=int(max_walls),
                horizon_y=horizon_y,
                config=cfg,
            )
        stats["primitive_method"] = primitive_method

        keep: list = []
        if geometry_mode in ("both", "primitives"):
            keep.extend(prims)
        else:
            # relief_mesh-only mode still keeps the backdrop: every extractor
            # always emits it as a far "catch-all" anchor (a plane sized to the
            # full recovered frustum, well beyond the mesh's own coverage), and
            # the relief mesh alone has real gaps — torn at every depth
            # discontinuity so foreground silhouettes don't rubber-sheet. Orbit
            # even slightly and those tears open onto empty void without this;
            # dropping the backdrop here was a bug, not intended behavior.
            keep.extend(p for p in prims if p.name == "projection_backdrop")
        if geometry_mode in ("both", "relief_mesh"):
            # Reuse the derivation's ground scale so the mesh matches the
            # primitives' world (ground on Y=0).
            mesh = build_relief_mesh(
                depth_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                grid_long_edge=int(relief_grid),
                scale=float(stats.get("ground_scale", 1.0)),
            )
            keep.append(relief_mesh_primitive(mesh))
            stats["relief_mesh"] = {
                "n_vertices": mesh.stats["n_vertices"],
                "n_faces": mesh.stats["n_faces"],
            }

        # Deep-copy: never mutate the upstream node's cached ATLAS_SOLVE.
        out = AtlasSolve.from_dict(solve.to_dict())
        out.projection_scene.proxy_geometry = [
            p for p in out.projection_scene.proxy_geometry
            if (p.metadata or {}).get("role") != PROXY_ROLE
        ]
        out.projection_scene.proxy_geometry.extend(keep)
        out.projection_scene.debug_metadata["proxy_derivation"] = {
            **stats, "depth_model": depth_model, "geometry_mode": geometry_mode,
        }
        return (out,)


class AtlasExportReliefMesh:
    """Export a depth relief mesh (OBJ + MTL + texture) for Maya / Nuke / ZBrush.

    Triangulates the metric depth map into a world-space mesh, torn at depth
    silhouettes, with the recovered-camera projection baked into per-vertex UVs —
    the mesh imports already textured with the source photo, ready to retopo /
    reproject. Formats: OBJ (+MTL+PNG; Maya/Nuke/ZBrush) and/or GLB (single
    binary, texture embedded; Blender/Substance/web). Ground lands on Y=0 (scale
    reconciled to the solve's camera height). Requires the [neural] extra.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("obj_path", "glb_path")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            },
            "optional": {
                "grid_long_edge": ("INT", {"default": 128, "min": 16, "max": 512,
                    "tooltip": "Mesh density: grid columns along the longest image edge."}),
                "depth_edge_rel": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Relative depth jump that tears the mesh (silhouette holes)."}),
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "format": (["both", "obj", "glb"], {"default": "both"}),
            },
        }

    def export(self, solve, image, output_dir="atlas_exports", grid_long_edge=128,
               depth_edge_rel=0.5,
               depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
               device="auto", format="both"):
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.exporters.relief_mesh_exporter import (
            export_relief_mesh,
            export_relief_mesh_glb,
        )
        from atlas_camera.inference.depth_estimator import estimate_depth

        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        width = int(intr.image_width or image.shape[2])
        height = int(intr.image_height or image.shape[1])
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        if fx <= 0:
            raise ValueError(
                "Relief mesh export needs a solved focal length — run a solve node "
                "(e.g. Atlas Learned Solve) before this node."
            )
        cx = intr.cx_px if intr.cx_px is not None else width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else height / 2.0

        depth_map = result.depth
        if depth_map.shape != (height, width):
            depth_map = _resize_depth(depth_map, width, height)

        scale, scale_info = estimate_ground_scale(
            depth_map, view_matrix=extr.camera_view_matrix,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        mesh = build_relief_mesh(
            depth_map, view_matrix=extr.camera_view_matrix,
            fx=fx, fy=fy, cx=cx, cy=cy,
            grid_long_edge=int(grid_long_edge),
            depth_edge_rel=float(depth_edge_rel),
            scale=scale,
        )
        texture = _image_tensor_to_pil(image)
        obj_path = glb_path = ""
        if format in ("both", "obj"):
            obj_path = export_relief_mesh(mesh, output_dir, texture=texture)["obj"]
        if format in ("both", "glb"):
            glb_path = export_relief_mesh_glb(mesh, output_dir, texture=texture)["glb"]
        return (obj_path, glb_path)


class AtlasLoadSolveJSON:
    """Load a previously saved AtlasSolve from a JSON file."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "load"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_path": ("STRING", {"default": "atlas_solve.json"}),
            }
        }

    def load(self, json_path):
        return (load_solve_json(json_path),)


class AtlasDecomposeSolve:
    """Break an ATLAS_SOLVE into its typed component outputs."""
    RETURN_TYPES = ("ATLAS_CAMERA", "FLOAT", "STRING", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("camera", "confidence", "source_method", "image_width", "image_height", "solve_json", "horizon_angle_deg")
    FUNCTION = "decompose"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"solve": ("ATLAS_SOLVE",)}}

    def decompose(self, solve):
        horizon_deg = float(
            (solve.debug_metadata or {})
            .get("camera_estimation", {})
            .get("horizon_angle", 0.0)
        )
        return (
            solve.camera,
            float(solve.confidence),
            str(solve.source_method),
            int(solve.camera.intrinsics.image_width),
            int(solve.camera.intrinsics.image_height),
            solve.to_json(),
            horizon_deg,
        )


class AtlasDecomposeCamera:
    """Extract intrinsic and extrinsic floats from an ATLAS_CAMERA for downstream routing."""
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT",
                    "FLOAT", "FLOAT", "FLOAT",
                    "FLOAT", "FLOAT")
    RETURN_NAMES = ("fx", "fy", "cx", "cy",
                    "cam_x", "cam_y", "cam_z",
                    "focal_mm", "fov_h_deg")
    FUNCTION = "decompose"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"camera": ("ATLAS_CAMERA",)}}

    def decompose(self, camera):
        intr = camera.intrinsics
        extr = camera.extrinsics
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        cx = intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else intr.image_height / 2.0
        pos = extr.camera_position
        focal_mm = intr.focal_length_mm or 0.0
        fov_h = 0.0
        if fx > 0 and intr.image_width > 0:
            fov_h = math.degrees(2 * math.atan(intr.image_width / (2.0 * fx)))
        return (
            float(fx), float(fy), float(cx), float(cy),
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(focal_mm), float(fov_h),
        )


def _solve_image_size(solve, width: int = 0, height: int = 0) -> tuple[int, int]:
    """Resolve output dimensions, auto-adopting the source image's size/aspect.

    ``width``/``height`` of 0 (the default) means "use the source image dimensions
    carried on the solve". A positive value overrides that axis.
    """
    intr = getattr(solve, "camera", None) and solve.camera.intrinsics
    iw = int((intr.image_width if intr else 0) or getattr(solve, "image_width", 0) or 0)
    ih = int((intr.image_height if intr else 0) or getattr(solve, "image_height", 0) or 0)
    w = int(width) if width and int(width) > 0 else (iw or 1024)
    h = int(height) if height and int(height) > 0 else (ih or 1024)
    return w, h


def _fit_long_edge(width: int, height: int, long_edge: int, multiple: int = 8) -> tuple[int, int]:
    """Scale (width, height) so its longest side is ``long_edge``, rounded to ``multiple``."""
    width = max(1, int(width))
    height = max(1, int(height))
    scale = long_edge / float(max(width, height))
    def _round(v: float) -> int:
        return max(multiple, int(round(v / multiple)) * multiple)
    return _round(width * scale), _round(height * scale)


class AtlasGroundDepthMap:
    """
    Generate a ground-plane depth heatmap as an IMAGE tensor.
    Ports the GLSL DEPTH_FRAGMENT_SHADER (ProjectionMaterial.ts) to numpy:
    per-pixel ray cast → Y=0 intersection → warm-to-cool colormap.
    """
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_image", "ground_mask")
    FUNCTION = "generate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image_width": ("INT", {"default": 0, "min": 0, "max": 8192,
                                        "tooltip": "0 = auto (adopt source image width)"}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 8192,
                                         "tooltip": "0 = auto (adopt source image height)"}),
                "near_m": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 500.0, "step": 0.1}),
                "far_m": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
            }
        }

    def generate(self, solve, image_width, image_height, near_m, far_m):
        torch = _require_torch()
        image_width, image_height = _solve_image_size(solve, image_width, image_height)
        rgb, mask = _ground_depth_compute(solve, image_width, image_height, near_m, far_m)
        if rgb is None:
            blank_img = torch.zeros(1, image_height, image_width, 3, dtype=torch.float32)
            blank_mask = torch.zeros(1, image_height, image_width, dtype=torch.float32)
            return (blank_img, blank_mask)
        image_tensor = torch.from_numpy(rgb).unsqueeze(0)   # 1×H×W×3
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)   # 1×H×W
        return (image_tensor, mask_tensor)


class AtlasGroundMask:
    """Binary MASK: 1 = ground visible (ray hits Y=0 plane), 0 = sky/above horizon."""
    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image_width": ("INT", {"default": 0, "min": 0, "max": 8192,
                                        "tooltip": "0 = auto (adopt source image width)"}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 8192,
                                         "tooltip": "0 = auto (adopt source image height)"}),
            }
        }

    def generate(self, solve, image_width, image_height):
        torch = _require_torch()
        image_width, image_height = _solve_image_size(solve, image_width, image_height)
        _, mask = _ground_depth_compute(solve, image_width, image_height, 1.0, 50.0)
        if mask is None:
            return (torch.zeros(1, image_height, image_width, dtype=torch.float32),)
        return (torch.from_numpy(mask).unsqueeze(0),)


class AtlasHorizonMask:
    """
    Sky mask: 1 = above horizon (sky), 0 = below horizon (ground).
    Uses the horizon line coefficients from the solved horizon_line (ax+by+c=0).
    """
    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image_width": ("INT", {"default": 0, "min": 0, "max": 8192,
                                        "tooltip": "0 = auto (adopt source image width)"}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 8192,
                                         "tooltip": "0 = auto (adopt source image height)"}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 200,
                                       "tooltip": "Gaussian feather in pixels around horizon edge"}),
            }
        }

    def generate(self, solve, image_width, image_height, feather_px):
        np = _require_numpy()
        torch = _require_torch()

        image_width, image_height = _solve_image_size(solve, image_width, image_height)
        horizon = solve.horizon_line
        if horizon is None:
            # No horizon solved — return full-image sky mask (all ones)
            return (torch.ones(1, image_height, image_width, dtype=torch.float32),)

        a, b, c = horizon.line_coefficients  # ax + by + c = 0

        uu, vv = np.meshgrid(np.arange(image_width, dtype=np.float32),
                             np.arange(image_height, dtype=np.float32))
        signed = a * uu + b * vv + c  # positive = above horizon (sky)

        if feather_px > 0 and abs(b) > 1e-6:
            # Soft transition: sigmoid-based feather
            horizon_normal_len = math.sqrt(a * a + b * b)
            dist = signed / horizon_normal_len  # signed pixel distance from line
            sigma = max(feather_px / 3.0, 0.1)
            feathered = 1.0 / (1.0 + np.exp(-dist / sigma))
            mask = feathered.astype(np.float32)
        else:
            mask = (signed >= 0).astype(np.float32)

        return (torch.from_numpy(mask).unsqueeze(0),)


class AtlasVPVisualization:
    """Draw vanishing-point convergence lines and horizon onto an image using PIL."""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "show_horizon": ("BOOLEAN", {"default": True}),
                "show_vp_lines": ("BOOLEAN", {"default": True}),
                "line_opacity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def visualize(self, image, solve, show_horizon=True, show_vp_lines=True, line_opacity=0.7):
        PILImage = _require_pil()
        from PIL import ImageDraw

        pil = _image_tensor_to_pil(image).copy()
        W, H = pil.size
        overlay = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        vp_colors = {"left": (255, 120, 50, 200), "right": (50, 160, 255, 200),
                     "vertical": (80, 220, 100, 200)}

        if show_vp_lines:
            for vp in solve.vanishing_points:
                color = vp_colors.get(str(vp.direction_label), (200, 200, 200, 180))
                vx, vy = vp.position_px
                # Draw convergence lines from each supporting segment to VP
                for seg in vp.supporting_lines[:12]:
                    mid_x = (seg[0][0] + seg[1][0]) / 2
                    mid_y = (seg[0][1] + seg[1][1]) / 2
                    draw.line([(mid_x, mid_y), (vx, vy)], fill=color, width=1)
                # VP circle
                r = 6
                draw.ellipse([(vx - r, vy - r), (vx + r, vy + r)],
                             outline=color, width=2)

        if show_horizon and solve.horizon_line and solve.horizon_line.endpoints_px:
            p1, p2 = solve.horizon_line.endpoints_px
            draw.line([tuple(p1), tuple(p2)], fill=(255, 220, 0, 200), width=2)

        alpha = int(line_opacity * 255)
        r, g, b, a = overlay.split()
        a = a.point(lambda v: int(v * alpha / 255))
        overlay = PILImage.merge("RGBA", (r, g, b, a))
        pil_rgba = pil.convert("RGBA")
        pil_rgba.paste(overlay, mask=overlay.split()[3])
        return (_pil_to_image_tensor(pil_rgba.convert("RGB")),)


class AtlasExportUSD:
    """Export the solved camera as a USD camera asset (.usda)."""
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("usd_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            }
        }

    def export(self, solve, output_dir):
        from atlas_camera.exporters.usd_exporter import USDExporter
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "camera.usda"
        USDExporter().export_camera(solve, dest)
        return (str(dest),)


class AtlasExportBlender:
    """Export a Blender Python scene-build script for the recovered camera."""
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("script_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            }
        }

    def export(self, solve, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "build_scene.py"
        write_blender_scene_script(solve, dest)
        return (str(dest),)


class AtlasExportNuke:
    """Export a Nuke Python projection script for the recovered camera."""
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("script_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            }
        }

    def export(self, solve, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "nuke_projection.py"
        write_nuke_projection_script(solve, dest)
        return (str(dest),)


# ---------------------------------------------------------------------------
# Track 2 — AtlasBlockout viewport node
# ---------------------------------------------------------------------------

class AtlasBlockoutViewport:
    """
    Browser-based 3D blockout viewport initialized with the recovered camera.
    Pattern mirrors Yedp Blockout: browser renders → base64 JSON → Python decodes
    → four IMAGE outputs (shaded, depth, normal, mask).

    Workflow:
    1. Connect an ATLAS_SOLVE and a source IMAGE, then queue the prompt.
    2. The Three.js viewport in the ComfyUI node opens, camera pre-aligned to the photo.
    3. Place primitive geometry (box, plane, person card, etc.).
    4. Click "Render Passes" in the viewport — fills client_data and re-queues.
    5. Four IMAGE outputs are now available for ControlNet or compositing.
    """
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("shaded", "depth", "normal", "mask")
    FUNCTION = "render"
    CATEGORY = "Atlas Camera/Blockout"
    OUTPUT_NODE = True  # kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE",),
                "resolution": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 8,
                    "tooltip": "Long-edge render resolution; the short side auto-follows the "
                               "source image aspect (viewport inherits the image's aspect)."}),
                "client_data": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "preview_expand": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Dilate derived geometry outward from the camera for wider orbit "
                               "coverage before it disappears into unreconstructed space. "
                               "1.0 = off (accurate geometry). Display only — never affects "
                               "DCC exports or measured geometry."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def render(self, solve, source_image, resolution, client_data, preview_expand=1.4, unique_id=None):
        torch = _require_torch()

        # Auto-adopt the source image aspect: derive W×H from the incoming image,
        # scaled so the long edge is `resolution`.
        src_h, src_w = int(source_image.shape[1]), int(source_image.shape[2])
        width, height = _fit_long_edge(src_w, src_h, int(resolution))

        # Store camera data for the browser extension to fetch
        node_id = str(unique_id) if unique_id is not None else "0"
        _blockout_cache_set(node_id, _extract_blockout_camera(
            solve, source_image, width, height, preview_expand=float(preview_expand)))

        # IMPORTANT: return a "ui" payload. ComfyUI only emits the "executed"
        # websocket message (which triggers node.onExecuted / the frontend's
        # api "executed" event) for nodes whose result includes UI output —
        # without this the browser extension never learns the solve is ready
        # and never fetches the camera data / background / proxies.
        ui_payload = {"atlas_ready": [node_id]}

        if not client_data.strip():
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return {"ui": ui_payload, "result": (blank, blank, blank, blank)}

        try:
            data = json.loads(client_data)
        except json.JSONDecodeError:
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return {"ui": ui_payload, "result": (blank, blank, blank, blank)}

        shaded = _decode_b64_to_tensor(data.get("shaded", ""), width, height)
        depth  = _decode_b64_to_tensor(data.get("depth",  ""), width, height)
        normal = _decode_b64_to_tensor(data.get("normal", ""), width, height)
        mask   = _decode_b64_to_tensor(data.get("mask",   ""), width, height)
        return {"ui": ui_payload, "result": (shaded, depth, normal, mask)}

    @classmethod
    def IS_CHANGED(cls, client_data="", **_):
        import hashlib
        return hashlib.md5(client_data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Node registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # Existing
    "AtlasLoadImageSolveCamera":  AtlasLoadImageSolveCamera,
    "AtlasExportReviewPackage":   AtlasExportReviewPackage,
    "AtlasExportSolveJSON":       AtlasExportSolveJSON,
    "AtlasExportMayaReviewScene": AtlasExportMayaReviewScene,
    "AtlasUSDCameraLoader":       AtlasUSDCameraLoader,
    # Track 1 — solve
    "AtlasSolveFromImage":        AtlasSolveFromImage,
    "AtlasLearnedSolveFromImage": AtlasLearnedSolveFromImage,
    "AtlasReferenceScaleSolve":   AtlasReferenceScaleSolve,
    "AtlasVLMScaleCues":          AtlasVLMScaleCues,
    "AtlasApplyScaleReferences":  AtlasApplyScaleReferences,
    "AtlasDeriveProjectionGeometry": AtlasDeriveProjectionGeometry,
    "AtlasConstrainedSolve":      AtlasConstrainedSolve,
    "AtlasLoadSolveJSON":         AtlasLoadSolveJSON,
    # Track 1 — decompose
    "AtlasDecomposeSolve":        AtlasDecomposeSolve,
    "AtlasDecomposeCamera":       AtlasDecomposeCamera,
    # Track 1 — image generation
    "AtlasDepthAnything":         AtlasDepthAnything,
    "AtlasGroundDepthMap":        AtlasGroundDepthMap,
    "AtlasGroundMask":            AtlasGroundMask,
    "AtlasHorizonMask":           AtlasHorizonMask,
    "AtlasVPVisualization":       AtlasVPVisualization,
    # Track 1 — export
    "AtlasExportReliefMesh":      AtlasExportReliefMesh,
    "AtlasExportUSD":             AtlasExportUSD,
    "AtlasExportBlender":         AtlasExportBlender,
    "AtlasExportNuke":            AtlasExportNuke,
    # Track 2 — blockout viewport
    "AtlasBlockoutViewport":      AtlasBlockoutViewport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing
    "AtlasLoadImageSolveCamera":  "Atlas Load Image / Solve Camera",
    "AtlasExportReviewPackage":   "Atlas Export Review Package",
    "AtlasExportSolveJSON":       "Atlas Export Solve JSON",
    "AtlasExportMayaReviewScene": "Atlas Export Maya Review Scene",
    "AtlasUSDCameraLoader":       "Atlas USD Camera Loader",
    # Track 1 — solve
    "AtlasSolveFromImage":        "Atlas Solve Camera from Image",
    "AtlasLearnedSolveFromImage": "Atlas Learned Solve (GeoCalib) 🧠",
    "AtlasReferenceScaleSolve":   "Atlas Reference-Object Scale 📏",
    "AtlasVLMScaleCues":          "Atlas VLM Scale Cues 👁",
    "AtlasApplyScaleReferences":  "Atlas Apply Scale References ✅",
    "AtlasDeriveProjectionGeometry": "Atlas Derive Projection Geometry 📽",
    "AtlasConstrainedSolve":      "Atlas Constrained Solve",
    "AtlasLoadSolveJSON":         "Atlas Load Solve JSON",
    # Track 1 — decompose
    "AtlasDecomposeSolve":        "Atlas Decompose Solve",
    "AtlasDecomposeCamera":       "Atlas Decompose Camera",
    # Track 1 — image generation
    "AtlasDepthAnything":         "Atlas Depth Anything V2 🧠",
    "AtlasGroundDepthMap":        "Atlas Ground Depth Map",
    "AtlasGroundMask":            "Atlas Ground Mask",
    "AtlasHorizonMask":           "Atlas Horizon / Sky Mask",
    "AtlasVPVisualization":       "Atlas VP Visualization",
    # Track 1 — export
    "AtlasExportReliefMesh":      "Atlas Export Relief Mesh (OBJ) 🗻",
    "AtlasExportUSD":             "Atlas Export USD",
    "AtlasExportBlender":         "Atlas Export Blender Scene",
    "AtlasExportNuke":            "Atlas Export Nuke Script",
    # Track 2 — blockout viewport
    "AtlasBlockoutViewport":      "Atlas Blockout Viewport 🧊",
}
