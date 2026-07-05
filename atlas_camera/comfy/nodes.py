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
                              preview_expand: float = 1.0, shot_intrinsics=None) -> dict[str, Any]:
    """Serialize the recovered camera into a dict the browser extension can consume.

    `shot_intrinsics` (optional, from AtlasShotCam via intrinsics_from_shot_cam)
    conforms the RENDER/VIEWING camera to a project-level shot format. It must
    stay entirely separate from `fx`/`fy`/`cx`/`cy` below: those are also read
    by the frontend's makeProjectionMaterial() for the PRIMARY source's own
    texture-sampling (applyCamera(data) and setProxies(data) — which builds
    the primary's projection material — both consume this SAME dict), so
    overwriting them would corrupt how the actual photo gets projected onto
    geometry. Only `render_fy`/`render_image_height` (read solely by
    applyRecoveredCamera for the live orbit camera's FOV) carry the
    shot-conformed values; `target_width`/`target_height` are always already
    independent of `image_width`/`image_height` (routinely resized via
    resolution/_fit_long_edge regardless of shot_cam), so they're set
    directly from the shot format by the caller with no separate key needed.
    """
    cam = solve.camera
    intr = cam.intrinsics
    extr = cam.extrinsics
    fx = intr.fx_px or 0.0
    fy = intr.fy_px or fx
    cx = intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0
    cy = intr.cy_px if intr.cy_px is not None else intr.image_height / 2.0
    if shot_intrinsics is not None:
        render_fy = shot_intrinsics.fy_px or shot_intrinsics.fx_px or fy
        render_image_height = shot_intrinsics.image_height
    else:
        render_fy = fy
        render_image_height = intr.image_height
    # view_matrix is the Atlas camera_view_matrix (4×4, row-major)
    vm = [list(row) for row in extr.camera_view_matrix]

    # Encode source image as JPEG base64 so the browser can use it as background
    source_b64 = ""
    try:
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

    # Multi-angle patch sources (AtlasAddPatchView): each is its own camera +
    # novel-view image + geometry, layered over the primary to fill areas the
    # primary camera couldn't see. Serialized like the primary so the viewport
    # can bind a projection material per source. Empty for single-camera solves.
    from atlas_camera.core.schema import AtlasProjectionScene
    projection_sources = []
    for src in (getattr(solve, "projection_sources", None) or []):
        s_intr = src.camera.intrinsics
        s_extr = src.camera.extrinsics
        s_fx = s_intr.fx_px or 0.0
        s_fy = s_intr.fy_px or s_fx
        s_cx = s_intr.cx_px if s_intr.cx_px is not None else (s_intr.image_width or 1) / 2.0
        s_cy = s_intr.cy_px if s_intr.cy_px is not None else (s_intr.image_height or 1) / 2.0
        projection_sources.append({
            "name": src.name,
            "view_matrix": [list(row) for row in s_extr.camera_view_matrix],
            "camera_position": list(s_extr.camera_position),
            "fx": s_fx, "fy": s_fy, "cx": s_cx, "cy": s_cy,
            "image_width": s_intr.image_width,
            "image_height": s_intr.image_height,
            "image_b64": src.image_b64 or "",
            "priority": float(src.priority),
            "azimuth_deg": float(src.azimuth_deg),
            "elevation_deg": float(src.elevation_deg),
            "proxy_geometry": serialize_proxy_geometry(
                AtlasProjectionScene(proxy_geometry=list(src.proxy_geometry)),
            ),
        })

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
        "render_fy": render_fy,
        "render_image_height": render_image_height,
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "source_image_b64": source_b64,
        "proxy_geometry": proxy_geometry,
        "projection_sources": projection_sources,
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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            },
            "optional": {
                "relief_mesh_obj_path": ("STRING", {"default": "",
                    "tooltip": "Optional obj_path output from AtlasExportReliefMesh. When set, the "
                               "relief mesh is imported into the Maya scene instead of being omitted — "
                               "wire AtlasExportReliefMesh's obj_path here to see real derived geometry "
                               "(not just the camera) when opening the scene."}),
            },
        }

    def export(self, solve, output_dir, relief_mesh_obj_path=""):
        result = build_review_package(
            solve, output_dir, include_usd=False,
            relief_mesh_obj_path=relief_mesh_obj_path or None,
        )
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
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
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
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
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
      Height comes from a percentile clip of the 3D points that individually
      pass a near-vertical-normal filter — a sloped roof, spire, or tower
      never qualifies, so on complex facades the wall only ever reflects the
      plain section below it (confirmed on real church/tower photos).
    - ``ransac_planes`` — any-orientation planes (sloped roofs, stepped/angled
      facades) via sequential RANSAC seeded by a 2D normal-orientation
      histogram. Best for exterior/architectural shots.
    - ``room_cuboid`` — Manhattan-aligned floor + up to 4 walls + optional
      ceiling. Best for orthogonal interiors; silently produces skewed walls
      on non-orthogonal rooms (pick a different method for those shots).
    - ``vertical_extrusion`` — same wall orientation/distance detection as
      ``azimuth_walls``, but height comes from the image-space silhouette
      instead: the topmost non-sky pixel per column (see
      ``depth_geometry.detect_sky_mask``), back-projected at that pixel's own
      depth regardless of its local surface normal. A flat vertical
      "billboard" extruded to the real silhouette top, per Hoiem/Efros/
      Hebert's "Automatic Photo Pop-up" (SIGGRAPH 2005) — reaches sloped
      roofs, spires, and towers that ``azimuth_walls`` truncates. Best for
      complex exterior architecture where a single flat wall height is the
      wrong shape but full RANSAC plane-fitting is overkill.

    ``scene_type`` (default "manual") is a one-choice convenience preset over
    the three widgets above, for artists who'd rather pick a shot type than
    reason about geometry_mode/primitive_method/depth_model separately:
    "organic" -> relief_mesh, "indoor" -> primitives+room_cuboid+Indoor depth
    model, "outdoor" -> primitives+ransac_planes+Outdoor depth model. Purely
    a preset — it sets the same three parameters this node already exposes,
    never a new solving code path. "manual" leaves them untouched.
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
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "geometry_mode": (["relief_mesh", "primitives", "both"], {"default": "relief_mesh",
                    "tooltip": "What the viewport receives. relief_mesh = contoured depth mesh "
                               "(recommended); primitives = flat blockout planes/boxes; both "
                               "overlaps the two on the same surfaces (enclosure + z-shimmer)."}),
                "relief_grid": ("INT", {"default": 128, "min": 16, "max": 1024,
                    "tooltip": "Viewport relief-mesh density (long-edge grid columns). Higher = "
                               "fewer/smaller torn holes on noisy AI-image depth (each quad spans "
                               "less real-world area, so it's less likely to straddle a spurious "
                               "depth jump) at the cost of a larger mesh payload sent to the "
                               "browser and a slower/heavier viewport. Overridden by "
                               "relief_quality unless that's set to 'custom'."}),
                "relief_quality": (["custom", "low", "medium", "high", "ultra"], {"default": "custom",
                    "tooltip": "Quick-pick override for relief_grid: low=64, medium=256, high=512, "
                               "ultra=1024. 'custom' (default) leaves relief_grid exactly as set "
                               "above — fully backward compatible. Same convenience-preset "
                               "pattern as scene_type: this only sets relief_grid, no new solving "
                               "path. 'ultra' produces a much larger mesh — expect a slower "
                               "viewport and bigger solve JSON exports."}),
                "depth_edge_rel": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Relative depth jump that tears the mesh into a silhouette hole. "
                               "Lower = tears more readily (cleaner silhouettes, more holes on "
                               "noisy depth); higher = tears less (fewer holes, more risk of "
                               "rubber-sheeting a real silhouette onto the background). Same "
                               "parameter and default as AtlasExportReliefMesh."}),
                "primitive_method": (["azimuth_walls", "ransac_planes", "room_cuboid",
                                       "vertical_extrusion"],
                    {"default": "azimuth_walls",
                     "tooltip": "azimuth_walls (default) = vertical walls only, height clipped "
                                "to the plain wall (truncates sloped roofs/spires/towers). "
                                "ransac_planes = any-orientation planes (roofs, stepped "
                                "facades) — exteriors. room_cuboid = Manhattan floor+walls"
                                "+ceiling — orthogonal interiors. vertical_extrusion = same wall "
                                "orientation as azimuth_walls but height extruded to the real "
                                "image-space silhouette top (reaches towers/spires/sloped roofs "
                                "azimuth_walls truncates). Only affects "
                                "geometry_mode=primitives/both; max_walls is reused as the "
                                "plane budget for ransac_planes and ignored by room_cuboid. "
                                "Ignored when scene_type != manual."}),
                "scene_type": ([
                    "manual", "organic", "mountains", "forests", "aerial",
                    "indoor", "outdoor", "simple_walls", "towers_spires",
                ], {"default": "manual",
                    "tooltip": "The one choice that matters — picks a complete, self-consistent "
                               "combination of geometry_mode/primitive_method/relief_quality/"
                               "depth_edge_rel/max_objects/depth_model for a named shot type, so "
                               "you never have to know which of those five widgets actually does "
                               "anything for your scene (e.g. primitive_method is silently ignored "
                               "whenever geometry_mode=relief_mesh — this picks a combination where "
                               "that can't happen). When this is anything but 'manual', the widgets "
                               "below it grey out and show the values this preset is using.\n"
                               "  organic = smooth relief mesh, general-purpose natural/cluttered "
                               "scenes.\n"
                               "  mountains = relief mesh at high density (terrain/ridgelines need "
                               "more grid resolution than the default to read as continuous rather "
                               "than faceted).\n"
                               "  forests = relief mesh at high density with a relaxed tear "
                               "threshold — dense canopy depth is genuinely noisy at a small scale, "
                               "so the default threshold shreds it into holes; this trades a little "
                               "silhouette accuracy for a filled-in canopy instead of swiss cheese.\n"
                               "  aerial = relief mesh AND primitives together (geometry_mode=both) "
                               "with more foreground objects allowed — buildings read as boxes "
                               "sitting on/above the relief-mesh ground and treeline, the drone/"
                               "top-down shot case.\n"
                               "  indoor = primitives + room_cuboid + the Indoor depth model "
                               "(orthogonal interiors).\n"
                               "  outdoor = primitives + ransac_planes + the Outdoor depth model "
                               "(sloped roofs, stepped facades).\n"
                               "  simple_walls = primitives + azimuth_walls (fast flat-wall "
                               "blockout, general exteriors).\n"
                               "  towers_spires = primitives + vertical_extrusion (reaches tall/"
                               "sloped silhouettes azimuth_walls truncates).\n"
                               "  manual (default) leaves every widget below exactly as set — fully "
                               "backward compatible with workflows saved before this widget existed. "
                               "If AtlasLearnedSolveFromImage's height_mode=measure_from_depth, set "
                               "its own depth_model to match by hand — this preset only reaches "
                               "this node's depth estimation, not the upstream solve node's."}),
            },
        }

    _SCENE_TYPE_PRESETS = {
        "organic": {"geometry_mode": "relief_mesh"},
        "mountains": {"geometry_mode": "relief_mesh", "relief_quality": "high"},
        "forests": {"geometry_mode": "relief_mesh", "relief_quality": "high", "depth_edge_rel": 1.0},
        "aerial": {"geometry_mode": "both", "primitive_method": "azimuth_walls",
                   "relief_quality": "medium", "max_objects": 6},
        "indoor": {"geometry_mode": "primitives", "primitive_method": "room_cuboid",
                   "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"},
        "outdoor": {"geometry_mode": "primitives", "primitive_method": "ransac_planes",
                    "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"},
        "simple_walls": {"geometry_mode": "primitives", "primitive_method": "azimuth_walls"},
        "towers_spires": {"geometry_mode": "primitives", "primitive_method": "vertical_extrusion"},
    }
    _RELIEF_QUALITY_PRESETS = {"low": 64, "medium": 256, "high": 512, "ultra": 1024}

    def derive(self, solve, image,
               depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
               max_walls=4, max_objects=3, device="auto",
               geometry_mode="relief_mesh", relief_grid=128, relief_quality="custom",
               depth_edge_rel=0.5,
               primitive_method="azimuth_walls", scene_type="manual"):
        preset = self._SCENE_TYPE_PRESETS.get(scene_type)
        if preset:
            geometry_mode = preset.get("geometry_mode", geometry_mode)
            primitive_method = preset.get("primitive_method", primitive_method)
            depth_model = preset.get("depth_model", depth_model)
            relief_quality = preset.get("relief_quality", relief_quality)
            depth_edge_rel = preset.get("depth_edge_rel", depth_edge_rel)
            max_objects = preset.get("max_objects", max_objects)
        if relief_quality in self._RELIEF_QUALITY_PRESETS:
            relief_grid = self._RELIEF_QUALITY_PRESETS[relief_quality]
        from atlas_camera.core.plane_extraction import PlaneRansacConfig, extract_planes_ransac
        from atlas_camera.core.proxy_geometry import (
            PROXY_ROLE,
            ProxyDerivationConfig,
            derive_projection_proxies,
            derive_vertical_extrusion_proxies,
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
        elif primitive_method == "vertical_extrusion":
            cfg = ProxyDerivationConfig(max_objects=int(max_objects))
            prims, stats = derive_vertical_extrusion_proxies(
                depth_map,
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                max_walls=int(max_walls),
                horizon_y=horizon_y,
                config=cfg,
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
                depth_edge_rel=float(depth_edge_rel),
                scale=float(stats.get("ground_scale", 1.0)),
                horizon_y=horizon_y,
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
            "scene_type": scene_type, "depth_edge_rel": float(depth_edge_rel),
            "relief_grid": int(relief_grid), "relief_quality": relief_quality,
            "max_objects": int(max_objects),
        }
        return (out,)


# ---------------------------------------------------------------------------
# Track 5 — composable geometry derivation (shared depth + single-purpose
# derive nodes + an explicit merge), an alternative to AtlasDeriveProjectionGeometry's
# scene_type presets for scenes that mix strategies (e.g. foreground buildings
# over background terrain) — see the "Composable geometry derivation" key
# design rule in CLAUDE.md for the full rationale. AtlasDeriveProjectionGeometry
# itself is untouched; these are additive.
# ---------------------------------------------------------------------------

def _solve_camera_params(solve, depth_result):
    """fx/fy/cx/cy/width/height for a solve, falling back to the depth
    estimate's own resolution — same fallback logic AtlasDeriveProjectionGeometry
    uses (there falling back to the source IMAGE tensor's shape instead, since
    that node takes an image directly; these nodes take an ATLAS_DEPTH_MAP,
    which already carries its own width/height from DepthResult).
    Returns None when there's no usable focal length (caller should return the
    solve unchanged, matching AtlasDeriveProjectionGeometry's own behavior).
    """
    intr = solve.camera.intrinsics
    width = int(intr.image_width or depth_result.image_width)
    height = int(intr.image_height or depth_result.image_height)
    fx = intr.fx_px or 0.0
    fy = intr.fy_px or fx
    if fx <= 0:
        return None
    cx = intr.cx_px if intr.cx_px is not None else width / 2.0
    cy = intr.cy_px if intr.cy_px is not None else height / 2.0
    return width, height, fx, fy, cx, cy


def _horizon_y_from_solve(solve):
    """Image row of the solved horizon, or None — same extraction
    AtlasDeriveProjectionGeometry already does from solve.horizon_line."""
    if solve.horizon_line and solve.horizon_line.endpoints_px:
        p1, p2 = solve.horizon_line.endpoints_px
        return 0.5 * (float(p1[1]) + float(p2[1]))
    return None


def _depth_map_for_solve(depth_result, width, height):
    """The depth estimate's raw array, resized to match the solve's
    intrinsics resolution if they disagree (same as AtlasDeriveProjectionGeometry)."""
    from atlas_camera.core.solver import _resize_depth
    depth_map = depth_result.depth
    if depth_map.shape != (height, width):
        depth_map = _resize_depth(depth_map, width, height)
    return depth_map


def _replace_proxy_role_geometry(solve, new_prims, stats, extra_metadata):
    """Deep-copy `solve`, strip any prior PROXY_ROLE-tagged geometry, and
    replace it with `new_prims` — the exact pattern AtlasDeriveProjectionGeometry
    and AtlasAddPatchView already use before mutating a solve's geometry lists.
    This is why derive nodes never chain (each call clobbers the previous
    derivation's output) — AtlasMergeGeometry is the explicit, visible place
    two branches' geometry actually combines."""
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import AtlasSolve
    out = AtlasSolve.from_dict(solve.to_dict())
    out.projection_scene.proxy_geometry = [
        p for p in out.projection_scene.proxy_geometry
        if (p.metadata or {}).get("role") != PROXY_ROLE
    ]
    out.projection_scene.proxy_geometry.extend(new_prims)
    out.projection_scene.debug_metadata["proxy_derivation"] = {**stats, **extra_metadata}
    return out


class AtlasDepthMap:
    """Shared metric depth estimate — wire this into one or more of
    AtlasDeriveReliefMesh / AtlasDeriveWalls / AtlasDeriveTowersSpires /
    AtlasDeriveRoofsFacades / AtlasDeriveInteriorRoom so a photo's depth is
    estimated ONCE and shared, instead of each derivation node re-running
    Depth-Anything independently. This matters for correctness, not just
    speed: every extraction strategy fits its own ground plane from whatever
    depth map it's given, so two branches fed slightly different depth
    estimates could disagree on metric scale and merge inconsistently.
    Requires the [neural] extra.

    Distinct from AtlasDepthAnything: that node's IMAGE output is a lossy,
    per-image min-max-normalized preview — the real near/far distances and
    is_metric flag are computed then discarded, so it cannot be used for
    metric geometry. This node keeps the full DepthResult (raw array +
    provenance) intact for the geometry nodes to consume.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "estimate"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    def estimate(self, image, depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto"):
        from atlas_camera.inference.depth_estimator import estimate_depth
        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)
        return (result,)


class AtlasDeriveReliefMesh:
    """Continuous depth-following relief mesh — one job, so there's no
    geometry_mode/primitive_method combination that silently ignores this
    node's own widgets. Takes an already-estimated ATLAS_DEPTH_MAP
    (AtlasDepthMap) instead of an image, so it can share one depth pass with
    sibling derivation nodes wired from the same photo (see AtlasMergeGeometry
    to combine their outputs). Fits its own ground scale/backdrop directly
    (relief_mesh.estimate_ground_scale + depth_geometry.build_backdrop_primitive)
    rather than borrowing them from a primitive-fitting pass — a relief mesh
    alone never needed the wall/object derivation AtlasDeriveProjectionGeometry's
    relief_mesh mode runs internally just to get those two numbers.
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "relief_grid": ("INT", {"default": 128, "min": 16, "max": 1024,
                    "tooltip": "Mesh density (long-edge grid columns). Higher = fewer/"
                               "smaller torn holes on noisy AI-image depth, at the cost "
                               "of a larger mesh payload and a heavier viewport."}),
                "relief_quality": (["custom", "low", "medium", "high", "ultra"], {"default": "custom",
                    "tooltip": "Quick-pick override for relief_grid: low=64, medium=256, "
                               "high=512, ultra=1024. 'custom' leaves relief_grid as set above."}),
                "depth_edge_rel": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Relative depth jump that tears the mesh into a silhouette "
                               "hole. Lower = tears more readily; higher = tears less but "
                               "risks rubber-sheeting a real silhouette onto the background."}),
            },
        }

    _RELIEF_QUALITY_PRESETS = {"low": 64, "medium": 256, "high": 512, "ultra": 1024}

    def derive(self, solve, depth, relief_grid=128, relief_quality="custom", depth_edge_rel=0.5):
        if relief_quality in self._RELIEF_QUALITY_PRESETS:
            relief_grid = self._RELIEF_QUALITY_PRESETS[relief_quality]
        from atlas_camera.core.depth_geometry import back_project_normals, build_backdrop_primitive
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale

        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics

        scale, ground_info = estimate_ground_scale(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=horizon_y)
        bp = back_project_normals(depth_map, view_matrix=extr.camera_view_matrix,
                                   fx=fx, fy=fy, cx=cx, cy=cy)
        scaled_depth = depth_map * scale
        backdrop = build_backdrop_primitive(
            bp=bp, scaled_depth=scaled_depth, valid_depth=bp.valid_depth,
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, scale=scale)
        mesh = build_relief_mesh(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            grid_long_edge=int(relief_grid), depth_edge_rel=float(depth_edge_rel),
            scale=scale, horizon_y=horizon_y)
        prims = [backdrop, relief_mesh_primitive(mesh)]
        stats = {
            "ground_scale": scale, "ground_fit": ground_info,
            "relief_mesh": {"n_vertices": mesh.stats["n_vertices"], "n_faces": mesh.stats["n_faces"]},
        }
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "relief_grid": int(relief_grid), "relief_quality": relief_quality,
            "depth_edge_rel": float(depth_edge_rel), "derive_node": "AtlasDeriveReliefMesh",
        })
        return (out,)


class AtlasDeriveWalls:
    """Vertical wall planes + foreground boxes/cylinders (azimuth_walls) — one
    job, general-purpose exterior blockout. Height is clipped to whatever 3D
    points individually pass a near-vertical-normal filter, so it truncates
    sloped roofs/spires/towers — use AtlasDeriveTowersSpires for those.
    Set max_objects=0 for walls/ground/backdrop only (no foreground boxes)."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 8}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 6,
                    "tooltip": "Max foreground boxes/cylinders (e.g. buildings, in an "
                               "aerial/top-down shot). 0 = walls/ground/backdrop only."}),
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=3):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_projection_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        cfg = ProxyDerivationConfig(max_objects=int(max_objects))
        prims, stats = derive_projection_proxies(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_walls=int(max_walls), horizon_y=horizon_y, config=cfg)
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "azimuth_walls", "derive_node": "AtlasDeriveWalls",
        })
        return (out,)


class AtlasDeriveTowersSpires:
    """Vertical wall planes extruded to the real image-space silhouette top
    (vertical_extrusion) — one job, reaches towers/spires/sloped roofs that
    AtlasDeriveWalls' azimuth_walls truncates. Per Hoiem/Efros/Hebert's
    "Automatic Photo Pop-up" (SIGGRAPH 2005) billboard-cutout technique."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 8}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 6,
                                        "tooltip": "Max foreground boxes/cylinders."}),
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=3):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_vertical_extrusion_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        cfg = ProxyDerivationConfig(max_objects=int(max_objects))
        prims, stats = derive_vertical_extrusion_proxies(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_walls=int(max_walls), horizon_y=horizon_y, config=cfg)
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "vertical_extrusion", "derive_node": "AtlasDeriveTowersSpires",
        })
        return (out,)


class AtlasDeriveRoofsFacades:
    """Any-orientation planes via sequential RANSAC (ransac_planes) — one
    job, sloped roofs and stepped/angled facades. Best for exterior
    architecture where a single flat wall height is the wrong shape."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "max_planes": ("INT", {"default": 8, "min": 1, "max": 16,
                    "tooltip": "Plane budget (roofs, facades, ramps)."}),
            },
        }

    def derive(self, solve, depth, max_planes=8):
        from atlas_camera.core.plane_extraction import PlaneRansacConfig, extract_planes_ransac
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        prims, stats = extract_planes_ransac(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_planes=int(max_planes), horizon_y=horizon_y, config=PlaneRansacConfig())
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "ransac_planes", "derive_node": "AtlasDeriveRoofsFacades",
        })
        return (out,)


class AtlasDeriveInteriorRoom:
    """Manhattan-aligned floor + up to 4 walls + optional ceiling
    (room_cuboid) — one job, orthogonal interiors. Produces confidently
    wrong/skewed results on non-orthogonal rooms — pick a different node
    for those shots."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "derive"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
        }

    def derive(self, solve, depth):
        from atlas_camera.core.room_layout import RoomCuboidConfig, extract_room_cuboid
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        prims, stats = extract_room_cuboid(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=horizon_y, config=RoomCuboidConfig())
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "room_cuboid", "derive_node": "AtlasDeriveInteriorRoom",
        })
        return (out,)


class AtlasMergeGeometry:
    """Explicit combinator for two independently-derived solves' geometry —
    the Nuke-Merge-node equivalent for AtlasDeriveWalls/AtlasDeriveReliefMesh/
    etc. Chain multiple instances for 3+-way combination
    (Merge(fg, bg) -> Merge(that, sky)).

    solve_a's camera/intrinsics become the merged solve's camera — wire both
    branches from the SAME upstream solve so they share a camera; this node
    does not check for or correct a mismatch between solve_a and solve_b.

    Derive nodes never chain on their own (each one strips any prior
    PROXY_ROLE-tagged geometry before adding its own, specifically so a
    re-run never silently accumulates stale geometry) — this node is the one
    explicit, visible place two branches' geometry actually combines.

    Only merges solve_b's PROXY_ROLE-tagged geometry — i.e. only what
    solve_b's own derive node actually added — never solve_b's full
    proxy_geometry list. This was found empirically (live end-to-end run,
    not reasoned in the original design): both branches used to inherit a
    "ground_plane" pass-through entry from their shared upstream solve
    (projection_scene.create_default_projection_scene()'s placeholder,
    tagged role="ground", not PROXY_ROLE) that neither derive node touched
    — naively concatenating solve_b's entire list duplicated that inherited
    entry on top of solve_a's own copy of the exact same thing, even though
    solve_a already provides it via `out`. That specific placeholder has
    since been removed for being confusingly named and having no consumer,
    but this filter stays as the correct general contract: a merge should
    only ever combine what each side's own derive node actually produced.

    Also deduplicates the always-emitted "projection_backdrop" plane: every
    derivation strategy emits exactly one, so merging two PROXY_ROLE lists
    that each have one would still produce two overlapping backdrop planes.
    Keeps solve_a's.

    Optional `shot_cam` (ATLAS_SHOT_CAM, from AtlasDefineShotCam): when
    connected, attached onto the merged solve as `out.shot_cam` — a pure
    attachment, never a mutation of `out.camera`. Geometry is world-space and
    doesn't care about sensor/lens format; only the FINAL render/export
    camera does, and this just lets that format ride along with the merged
    result so it reaches AtlasBlockoutViewport/exporters without having to
    be re-wired in separately. solve_a's own camera intrinsics/extrinsics —
    what any of its projection sources actually use to sample their own
    photos — are completely untouched either way.
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "merge"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve_a": ("ATLAS_SOLVE",),
                "solve_b": ("ATLAS_SOLVE",),
            },
            "optional": {
                "shot_cam": ("ATLAS_SHOT_CAM", {
                    "tooltip": "Optional project/shot camera format (AtlasDefineShotCam) — "
                               "attached to the merged solve for AtlasBlockoutViewport/exporters "
                               "to conform to. Never affects this merge's own geometry/camera."}),
            },
        }

    def merge(self, solve_a, solve_b, shot_cam=None):
        from atlas_camera.core.proxy_geometry import PROXY_ROLE
        from atlas_camera.core.schema import AtlasSolve
        out = AtlasSolve.from_dict(solve_a.to_dict())
        seen_backdrop = any(p.name == "projection_backdrop" for p in out.projection_scene.proxy_geometry)
        merged_from_b = 0
        for p in solve_b.projection_scene.proxy_geometry:
            if (p.metadata or {}).get("role") != PROXY_ROLE:
                continue  # pass-through geometry solve_b inherited, not something its derive node added
            if p.name == "projection_backdrop":
                if seen_backdrop:
                    continue
                seen_backdrop = True
            out.projection_scene.proxy_geometry.append(p)
            merged_from_b += 1
        out.projection_scene.debug_metadata["proxy_derivation_merge"] = {
            "solve_a_prims": len(solve_a.projection_scene.proxy_geometry),
            "solve_b_prims_merged": merged_from_b,
            "merged_prims_total": len(out.projection_scene.proxy_geometry),
        }
        if shot_cam is not None:
            out.shot_cam = shot_cam
        return (out,)


class AtlasDefineShotCam:
    """Project-level render/output camera format — sensor width/height (mm)
    + lens (focal length mm) + target resolution, analogous to a Nuke/Resolve
    project format setting. Wire its output into AtlasMergeGeometry (to
    attach it onto a merged solve so it flows downstream automatically) or
    directly into AtlasBlockoutViewport (an explicit direct wire always wins
    over an inherited one) to conform the FINAL render/export to this format,
    regardless of what aspect ratio any individual source photo happened to
    have. Intrinsics-only — no position; camera placement still comes from
    whichever solve's own recovered pose is already in play. Never affects
    how any photo gets projected onto geometry — see AtlasShotCam's own
    docstring in schema.py for why that's safe.
    """
    RETURN_TYPES = ("ATLAS_SHOT_CAM",)
    RETURN_NAMES = ("shot_cam",)
    FUNCTION = "define"
    CATEGORY = "Atlas Camera/Project"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 1000.0,
                    "tooltip": "Shot format sensor width in mm (with sensor_height_mm, defines "
                               "the output aspect ratio — e.g. 36x24 for 3:2, 36x20.25 for 16:9)."}),
                "sensor_height_mm": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 1000.0}),
                "focal_length_mm": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 2000.0,
                    "tooltip": "Shot format lens — the FINAL render/export camera's focal length, "
                               "independent of any individual source photo's own solved lens."}),
                "resolution": ("INT", {"default": 1920, "min": 128, "max": 8192, "step": 8,
                    "tooltip": "Long-edge output resolution; the short edge follows the sensor "
                               "aspect above (same long-edge convention as AtlasBlockoutViewport's "
                               "own resolution widget)."}),
            },
        }

    def define(self, sensor_width_mm=36.0, sensor_height_mm=24.0, focal_length_mm=35.0, resolution=1920):
        from atlas_camera.core.schema import AtlasShotCam
        return (AtlasShotCam(
            sensor_width_mm=float(sensor_width_mm),
            sensor_height_mm=float(sensor_height_mm),
            focal_length_mm=float(focal_length_mm),
            resolution_long_edge_px=int(resolution),
        ),)


# Exact named views from ComfyUI-qwenmultiangle / the Multiple-Angles LoRA, shared
# by every node that places a patch/target camera relative to a source photo
# (`AtlasAddPatchView`, `AtlasOcclusionMask`) so the same choice is picked
# everywhere and the two nodes' camera placement can never drift apart. Azimuth
# is absolute about the subject's front; distance scales the orbit radius
# (close-up pulls in).
_AZIMUTH_VIEWS = {
    "front view": 0.0, "front-right quarter view": 45.0, "right side view": 90.0,
    "back-right quarter view": 135.0, "back view": 180.0, "back-left quarter view": 225.0,
    "left side view": 270.0, "front-left quarter view": 315.0,
}
_ELEVATION_VIEWS = {
    "low-angle shot": -30.0, "eye-level shot": 0.0, "elevated shot": 30.0, "high-angle shot": 60.0,
}
_DISTANCE_VIEWS = {"close-up": 0.6, "medium shot": 1.0, "wide shot": 1.8}


def _named_view_orbit_delta(
    patch_azimuth_view, patch_elevation_view, patch_distance,
    source_azimuth_view, source_elevation_view, flip_azimuth,
):
    """Resolve absolute (subject-relative) LoRA named views into the actual
    orbit delta to apply to the recovered/source camera: ``patch - source``.

    Returns ``(d_azimuth_deg, d_elevation_deg, distance_scale)``.
    """
    d_azimuth = _AZIMUTH_VIEWS[patch_azimuth_view] - _AZIMUTH_VIEWS[source_azimuth_view]
    d_azimuth = ((d_azimuth + 180.0) % 360.0) - 180.0   # shortest way round
    if flip_azimuth:
        d_azimuth = -d_azimuth
    d_elevation = _ELEVATION_VIEWS[patch_elevation_view] - _ELEVATION_VIEWS[source_elevation_view]
    distance_scale = _DISTANCE_VIEWS[patch_distance]  # source assumed "medium shot"
    return float(d_azimuth), float(d_elevation), float(distance_scale)


class AtlasAddPatchView:
    """Add an AI novel-view "patch" to fill areas the primary camera can't see.

    Camera projection from a single photo can only texture what the recovered
    camera saw — orbit slightly and occluded/grazing areas go black. This node
    takes a novel view of the same scene generated at a defined angle (the
    Qwen-Image-Edit-2511 Multiple-Angles LoRA — e.g. via the ComfyUI-qwenmultiangle
    "Qwen Multiangle Camera" node), constructs a "patch camera" by orbiting the
    recovered camera around the scene pivot to that view (so it shares the
    primary's world frame — `camera_math.orbit_camera`), derives the patch view's
    own relief geometry in that frame (Depth Anything), and appends it to the
    solve as a ``ProjectionSource``. Chain one per angle; the viewport layers them
    over the primary, filling the occluded areas. Needs the [neural] extra.

    IMPORTANT — the LoRA's angles are ABSOLUTE (subject-relative), not relative to
    your source view: "right side view" = 90° around the *subject's* front, etc.
    So to place the patch camera correctly you must tell this node BOTH the view
    your SOURCE photo represents (``source_*``) and the view the PATCH was
    generated at (``patch_*``, matching what you set in the Qwen Multiangle Camera
    node); the actual orbit = patch − source. If the source is a straight-on
    front shot, leave ``source_azimuth_view`` = "front view" and the patch's named
    view maps directly. ``flip_azimuth`` swaps left/right if the recovered
    camera's handedness comes out mirrored (a one-click calibration fix).
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "add_patch"
    CATEGORY = "Atlas Camera"

    # Aliases onto the shared module-level dicts (see above) — kept as class
    # attributes since tests/test_add_patch_view.py references
    # AtlasAddPatchView._AZIMUTH_VIEWS/_ELEVATION_VIEWS directly.
    _AZIMUTH_VIEWS = _AZIMUTH_VIEWS
    _ELEVATION_VIEWS = _ELEVATION_VIEWS
    _DISTANCE_VIEWS = _DISTANCE_VIEWS

    @classmethod
    def INPUT_TYPES(cls):
        azimuths = list(cls._AZIMUTH_VIEWS)
        elevations = list(cls._ELEVATION_VIEWS)
        distances = list(cls._DISTANCE_VIEWS)
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "patch_image": ("IMAGE",),
            },
            "optional": {
                "patch_azimuth_view": (azimuths, {"default": "front-right quarter view",
                    "tooltip": "The LoRA azimuth the patch was generated at — MUST match the "
                               "Qwen Multiangle Camera node. Absolute about the subject's front."}),
                "patch_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "The LoRA elevation the patch was generated at (match the LoRA node)."}),
                "patch_distance": (distances, {"default": "medium shot",
                    "tooltip": "The LoRA distance the patch was generated at (match the LoRA node)."}),
                "source_azimuth_view": (azimuths, {"default": "front view",
                    "tooltip": "Which view your SOURCE photo already is, in the LoRA's absolute "
                               "frame. Orbit applied = patch − source. Leave 'front view' for a "
                               "straight-on source."}),
                "source_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "Elevation of the SOURCE photo in the LoRA's frame."}),
                "flip_azimuth": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip left/right if the patch lands on the wrong side "
                               "(recovered-camera handedness) — a calibration convenience."}),
                "name": ("STRING", {"default": "patch"}),
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "relief_grid": ("INT", {"default": 96, "min": 16, "max": 256,
                    "tooltip": "Patch relief-mesh density (long-edge grid columns)."}),
                "priority": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blend priority among patches (higher wins). The primary photo "
                               "is always highest; patches only fill where it can't see."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    def add_patch(self, solve, patch_image,
                  patch_azimuth_view="front-right quarter view",
                  patch_elevation_view="eye-level shot",
                  patch_distance="medium shot",
                  source_azimuth_view="front view",
                  source_elevation_view="eye-level shot",
                  flip_azimuth=False, name="patch",
                  depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                  relief_grid=96, priority=1.0, device="auto"):
        from atlas_camera.core.camera_math import (
            ground_lookat_pivot,
            horizon_row_from_extrinsics,
            orbit_camera,
        )
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.core.schema import (
            AtlasIntrinsics,
            AtlasSolve,
            LatentCamera,
            ProjectionSource,
        )
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.inference.depth_estimator import estimate_depth

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        p_w = int(intr.image_width or 0)
        p_h = int(intr.image_height or 0)
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        if fx <= 0 or p_w <= 0:
            # No focal / dims on the primary — can't place a patch; pass through.
            return (solve,)
        cx = intr.cx_px if intr.cx_px is not None else p_w / 2.0
        cy = intr.cy_px if intr.cy_px is not None else p_h / 2.0

        # Absolute LoRA views -> the ACTUAL orbit delta (patch - source), since
        # the LoRA angle is subject-relative, not relative to the source view.
        d_azimuth, d_elevation, distance_scale = _named_view_orbit_delta(
            patch_azimuth_view, patch_elevation_view, patch_distance,
            source_azimuth_view, source_elevation_view, flip_azimuth,
        )

        # Patch camera: orbit the recovered camera around the scene pivot (the
        # point it looks at) so the patch shares the primary's world frame.
        pivot = ground_lookat_pivot(extr)
        patch_extr = orbit_camera(
            extr, pivot,
            d_azimuth_deg=float(d_azimuth),
            d_elevation_deg=float(d_elevation),
            distance_scale=float(distance_scale),
        )

        # Patch image dimensions + intrinsics (same angular FOV as the primary,
        # scaled to the patch resolution; principal point centered).
        patch_h = int(patch_image.shape[1])
        patch_w = int(patch_image.shape[2])
        pfx = fx * (patch_w / p_w)
        pfy = fy * (patch_h / p_h)
        pcx = patch_w / 2.0
        pcy = patch_h / 2.0
        patch_intr = AtlasIntrinsics(
            image_width=patch_w,
            image_height=patch_h,
            focal_length_mm=intr.focal_length_mm,
            sensor_width_mm=intr.sensor_width_mm,
            fx_px=pfx, fy_px=pfy, cx_px=pcx, cy_px=pcy,
            lens_model=intr.lens_model,
        )

        # The patch camera is constructed (orbited), not solved, so it carries
        # no solve.horizon_line of its own. Derive its real horizon row exactly
        # (see horizon_row_from_extrinsics) so sky-exclusion during meshing
        # uses this camera's actual tilt instead of the generic height*0.45
        # fallback in estimate_ground_scale / build_relief_mesh.
        patch_horizon_y = horizon_row_from_extrinsics(patch_extr, fy=pfy, cy=pcy)

        # Depth -> relief geometry in the patch camera's frame.
        tmp = _save_image_tensor_to_tmp(patch_image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)
        depth_map = result.depth
        if depth_map.shape != (patch_h, patch_w):
            depth_map = _resize_depth(depth_map, patch_w, patch_h)

        scale, _scale_info = estimate_ground_scale(
            depth_map, view_matrix=patch_extr.camera_view_matrix,
            fx=pfx, fy=pfy, cx=pcx, cy=pcy,
            horizon_y=patch_horizon_y,
        )
        mesh = build_relief_mesh(
            depth_map, view_matrix=patch_extr.camera_view_matrix,
            fx=pfx, fy=pfy, cx=pcx, cy=pcy,
            horizon_y=patch_horizon_y,
            grid_long_edge=int(relief_grid),
            scale=scale,
        )
        patch_geom = [relief_mesh_primitive(mesh, name=f"{name}_relief_mesh")]

        # Encode the novel view as a JPEG data-URI (viewport texture).
        image_b64 = ""
        try:
            pil = _image_tensor_to_pil(patch_image)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

        source = ProjectionSource(
            camera=LatentCamera(intrinsics=patch_intr, extrinsics=patch_extr, name=name),
            name=name,
            image_b64=image_b64,
            proxy_geometry=patch_geom,
            azimuth_deg=float(d_azimuth),      # actual orbit delta applied
            elevation_deg=float(d_elevation),
            distance_scale=float(distance_scale),
            priority=float(priority),
            metadata={
                "source": "multi_angle_lora_patch",
                "patch_azimuth_view": patch_azimuth_view,
                "patch_elevation_view": patch_elevation_view,
                "patch_distance": patch_distance,
                "source_azimuth_view": source_azimuth_view,
                "flip_azimuth": bool(flip_azimuth),
                "pivot": [float(v) for v in pivot],
                "n_vertices": mesh.stats.get("n_vertices"),
                "n_faces": mesh.stats.get("n_faces"),
                "depth_model": depth_model,
            },
        )

        out = AtlasSolve.from_dict(solve.to_dict())
        out.projection_sources.append(source)
        return (out,)


class AtlasOcclusionMask:
    """Mask where a target/patch novel view has geometry the PRIMARY camera
    could not validly project onto (behind-camera, outside-frame, or too
    grazing) — white = primary is missing there, so a patch/composite should
    fill it; black = primary already has valid, sufficiently head-on coverage.

    Places the target/patch camera identically to ``AtlasAddPatchView``
    (same named-view widgets, same ``camera_math.orbit_camera`` construction —
    see ``_named_view_orbit_delta``), so the mask lines up with whatever patch
    geometry that node will later derive from the same image. Intended
    pipeline: ``Solve -> AtlasOcclusionMask -> ImageCompositeMasked (primary
    projected image + this target image) -> AtlasAddPatchView``. This is a
    Phase 1 ("simple") mask — frustum/frame/facing-angle only, not true
    depth-shadow occlusion (an object hiding another object from the primary's
    view but still projecting inside its frame/angle limits is not detected
    yet). Pure backend/numpy — no browser round-trip, runs headlessly.
    """
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("occlusion_mask", "coverage_mask")
    FUNCTION = "generate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        azimuths = list(_AZIMUTH_VIEWS)
        elevations = list(_ELEVATION_VIEWS)
        distances = list(_DISTANCE_VIEWS)
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "target_image": ("IMAGE",),
            },
            "optional": {
                "patch_azimuth_view": (azimuths, {"default": "front-right quarter view",
                    "tooltip": "The LoRA azimuth target_image was generated at — should match "
                               "whatever you'll later pass to AtlasAddPatchView for this image."}),
                "patch_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "The LoRA elevation target_image was generated at."}),
                "patch_distance": (distances, {"default": "medium shot",
                    "tooltip": "The LoRA distance target_image was generated at."}),
                "source_azimuth_view": (azimuths, {"default": "front view",
                    "tooltip": "Which view your SOURCE photo already is, in the LoRA's absolute "
                               "frame. Must match the value you'll use in AtlasAddPatchView."}),
                "source_elevation_view": (elevations, {"default": "eye-level shot",
                    "tooltip": "Elevation of the SOURCE photo in the LoRA's frame."}),
                "flip_azimuth": ("BOOLEAN", {"default": False,
                    "tooltip": "Must match the AtlasAddPatchView setting for this patch."}),
                "depth_model": ([
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                ], {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "angle_threshold": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Facing-angle gate in degrees for the PRIMARY camera's coverage. "
                               "90 (default) = only frustum/behind-camera/out-of-frame failures are "
                               "masked. Lower values also mask surfaces too grazing to the primary."}),
                "dilate_px": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "Expand the white (missing) mask region by this many pixels."}),
                "soft_edge_px": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "Blur the dilated mask edge by this many pixels, for compositing."}),
                "power": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Gamma remap after blur; > 1 makes the patch contribution more solid "
                               "near the feathered edge."}),
            },
        }

    def generate(self, solve, target_image,
                 patch_azimuth_view="front-right quarter view",
                 patch_elevation_view="eye-level shot",
                 patch_distance="medium shot",
                 source_azimuth_view="front view",
                 source_elevation_view="eye-level shot",
                 flip_azimuth=False,
                 depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto",
                 angle_threshold=90.0, dilate_px=0, soft_edge_px=0, power=1.0):
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera
        from atlas_camera.core.depth_geometry import back_project_normals, primary_camera_validity_mask
        from atlas_camera.inference.depth_estimator import estimate_depth

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        p_w = int(intr.image_width or 0)
        p_h = int(intr.image_height or 0)
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        target_h = int(target_image.shape[1])
        target_w = int(target_image.shape[2])
        if fx <= 0 or p_w <= 0:
            # No focal/dims on the primary — nothing to test against; treat
            # as fully missing so downstream compositing still gets a signal.
            mask = torch.ones(1, target_h, target_w, dtype=torch.float32)
            return (mask, 1.0 - mask)
        cx = intr.cx_px if intr.cx_px is not None else p_w / 2.0
        cy = intr.cy_px if intr.cy_px is not None else p_h / 2.0

        d_azimuth, d_elevation, distance_scale = _named_view_orbit_delta(
            patch_azimuth_view, patch_elevation_view, patch_distance,
            source_azimuth_view, source_elevation_view, flip_azimuth,
        )
        pivot = ground_lookat_pivot(extr)
        target_extr = orbit_camera(
            extr, pivot,
            d_azimuth_deg=d_azimuth, d_elevation_deg=d_elevation,
            distance_scale=distance_scale,
        )

        tfx = fx * (target_w / p_w)
        tfy = fy * (target_h / p_h)
        tcx = target_w / 2.0
        tcy = target_h / 2.0

        tmp = _save_image_tensor_to_tmp(target_image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)
        depth_map = result.depth
        if depth_map.shape != (target_h, target_w):
            from atlas_camera.core.solver import _resize_depth
            depth_map = _resize_depth(depth_map, target_w, target_h)

        bp = back_project_normals(
            depth_map, view_matrix=target_extr.camera_view_matrix,
            fx=tfx, fy=tfy, cx=tcx, cy=tcy,
        )
        invalid = primary_camera_validity_mask(
            bp.pts_world, bp.valid_depth, bp.normals, bp.valid_normal,
            primary_view_matrix=extr.camera_view_matrix,
            primary_fx=fx, primary_fy=fy, primary_cx=cx, primary_cy=cy,
            primary_width=p_w, primary_height=p_h,
            angle_threshold_deg=float(angle_threshold),
        )
        mask = invalid.astype(np.float32)

        # 4-connected binary dilation, one pixel per iteration. np.roll wraps
        # at the image border rather than clamping — negligible in practice
        # since dilate_px is capped at 200 and target images are typically
        # much larger, but a very small image + large dilate_px could wrap.
        for _ in range(int(dilate_px)):
            grown = mask.copy()
            grown = np.maximum(grown, np.roll(mask, 1, axis=0))
            grown = np.maximum(grown, np.roll(mask, -1, axis=0))
            grown = np.maximum(grown, np.roll(mask, 1, axis=1))
            grown = np.maximum(grown, np.roll(mask, -1, axis=1))
            mask = grown

        soft_edge_px = int(soft_edge_px)
        if soft_edge_px > 0:
            # Separable 2D box blur via cumulative sums (horizontal pass then
            # vertical) — numpy-only, no scipy (matches core/ convention).
            radius = soft_edge_px
            for axis in (1, 0):
                padded = np.pad(mask, [(radius, radius) if a == axis else (0, 0)
                                       for a in (0, 1)], mode="edge")
                csum = np.cumsum(padded, axis=axis)
                csum = np.insert(csum, 0, 0, axis=axis)
                n = 2 * radius + 1
                lo = np.take(csum, range(0, csum.shape[axis] - n), axis=axis)
                hi = np.take(csum, range(n, csum.shape[axis]), axis=axis)
                mask = (hi - lo) / n

        mask = np.clip(mask, 0.0, 1.0) ** float(power)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        coverage_t = torch.from_numpy((1.0 - mask).astype(np.float32)).unsqueeze(0)
        return (mask_t, coverage_t)


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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

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
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
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

        horizon_y = None
        if solve.horizon_line and solve.horizon_line.endpoints_px:
            p1, p2 = solve.horizon_line.endpoints_px
            horizon_y = 0.5 * (float(p1[1]) + float(p2[1]))

        scale, scale_info = estimate_ground_scale(
            depth_map, view_matrix=extr.camera_view_matrix,
            fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=horizon_y,
        )
        mesh = build_relief_mesh(
            depth_map, view_matrix=extr.camera_view_matrix,
            fx=fx, fy=fy, cx=cx, cy=cy,
            grid_long_edge=int(grid_long_edge),
            depth_edge_rel=float(depth_edge_rel),
            scale=scale,
            horizon_y=horizon_y,
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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

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
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

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
# Track 3 — camera path animation (see AtlasBlockoutViewport's Camera Path mode)
# ---------------------------------------------------------------------------

class AtlasExportCameraPathUSD:
    """Export a keyframed camera path as a time-sampled USD camera (.usda).

    Separate from AtlasExportUSD because it takes a different required input
    (ATLAS_CAMERA_PATH, produced by AtlasBlockoutViewport's Camera Path mode)
    rather than a single static solve pose.
    """
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("usd_path",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "camera_path": ("ATLAS_CAMERA_PATH",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            }
        }

    def export(self, solve, camera_path, output_dir):
        from atlas_camera.exporters.usd_exporter import USDExporter
        if camera_path is None or not camera_path.keyframes:
            raise ValueError(
                "No camera path yet — open AtlasBlockoutViewport, use 🎥 Camera Path "
                "to add at least one keyframe, then click ⏺ Bake Path before queuing "
                "this export node."
            )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "camera_path.usda"
        USDExporter().export_camera_animation(camera_path, solve.camera.intrinsics, dest)
        return (str(dest),)


# ---------------------------------------------------------------------------
# Track 2 — AtlasBlockout viewport node
# ---------------------------------------------------------------------------

class AtlasViewportControls:
    """Detached toolbar/panel for AtlasBlockoutViewport — connect its output to
    that node's `controls` input to move every button and panel (primitives,
    📽 Project / 📊 Diagram / ℹ Info, 🎥 Camera Path + presets + FBX import,
    Render Passes) off the viewport node, leaving it perspective-only and
    freely resizable.

    Carries no real data: the single output exists only so the two nodes'
    frontend JS extensions can find each other via the graph link (see
    atlas_blockout.js — the viewport reparents its toolbar DOM into this
    node's container when connected). Python does nothing but return an
    empty placeholder string.
    """
    RETURN_TYPES = ("ATLAS_VIEWPORT_LINK",)
    RETURN_NAMES = ("controls",)
    FUNCTION = "noop"
    CATEGORY = "Atlas Camera/Blockout"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def noop(self):
        return ("",)


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
    6. Optional: use 🎥 Camera Path mode to author a keyframed camera move (fly
       nav, unclamped — leaving the recovered cone is expected here), then
       click "⏺ Bake Path" to fill client_data with a rendered frame sequence.
       `path_frames` (an IMAGE batch) feeds a core Video Combine node directly;
       `camera_path` (the raw keyframes) feeds AtlasExportCameraPathUSD for a
       DCC-facing animated camera. Frames sampled outside the recovered
       camera's cone will show the same documented black/undefined regions as
       orbiting past it under 📽 Project — expected, not a bug.
    7. Optional: connect an AtlasViewportControls node to `controls` to move
       every button/panel (primitives, Project/Diagram/Info, Camera Path +
       presets + FBX import, Render Passes) OUT of this node — this node then
       shows the perspective render only, and can be freely resized by
       dragging its corner. `controls` carries no real data (a link exists
       purely so the two nodes' frontend JS can find each other); Python
       ignores it. With nothing connected, all controls still appear locally
       here, unchanged — fully backward-compatible with saved workflows that
       predate AtlasViewportControls.
    """
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "ATLAS_CAMERA_PATH")
    RETURN_NAMES = ("shaded", "depth", "normal", "mask", "path_frames", "camera_path")
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
                               "source image aspect (viewport inherits the image's aspect). "
                               "Also settable by dragging the node's own resize handle."}),
                "client_data": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "preview_expand": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Dilate derived geometry outward from the camera for wider orbit "
                               "coverage before it disappears into unreconstructed space. "
                               "1.0 = off (accurate geometry). Display only — never affects "
                               "DCC exports or measured geometry. CAUTION: values above 1.0 "
                               "dilate geometry beyond what the camera actually photographed, "
                               "so with 📽 Project active the dilated fringe has no real photo "
                               "data and renders as empty/black the moment you orbit off the "
                               "exact recovered viewpoint — leave at 1.0 if you plan to use "
                               "Project, raise it only for inspecting undressed blockout shapes."}),
                "controls": ("ATLAS_VIEWPORT_LINK", {
                    "tooltip": "Connect an AtlasViewportControls node here to move all buttons/"
                               "panels off this node (perspective-only, freely resizable). Carries "
                               "no real data — Python ignores it; the link only lets the two "
                               "nodes' frontend JS find each other."}),
                "shot_cam": ("ATLAS_SHOT_CAM", {
                    "tooltip": "Optional project/shot camera format (AtlasDefineShotCam) — conforms "
                               "the render resolution/aspect and viewing-camera FOV to this format "
                               "instead of auto-following source_image's own aspect. A direct wire "
                               "here wins over one attached to `solve` by AtlasMergeGeometry. Never "
                               "affects how the source photo is projected onto geometry."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def render(self, solve, source_image, resolution, client_data, preview_expand=1.0, controls=None,
               shot_cam=None, unique_id=None):
        torch = _require_torch()

        # A direct shot_cam wire wins over one inherited from the solve (e.g.
        # attached earlier by AtlasMergeGeometry) — explicit beats inherited.
        resolved_shot_cam = shot_cam if shot_cam is not None else getattr(solve, "shot_cam", None)
        shot_intrinsics = None
        if resolved_shot_cam is not None:
            from atlas_camera.core.intrinsics import intrinsics_from_shot_cam
            shot_intrinsics = intrinsics_from_shot_cam(resolved_shot_cam)
            width, height = shot_intrinsics.image_width, shot_intrinsics.image_height
        else:
            # Auto-adopt the source image aspect: derive W×H from the incoming
            # image, scaled so the long edge is `resolution`.
            src_h, src_w = int(source_image.shape[1]), int(source_image.shape[2])
            width, height = _fit_long_edge(src_w, src_h, int(resolution))

        # Store camera data for the browser extension to fetch
        node_id = str(unique_id) if unique_id is not None else "0"
        _blockout_cache_set(node_id, _extract_blockout_camera(
            solve, source_image, width, height, preview_expand=float(preview_expand),
            shot_intrinsics=shot_intrinsics))

        # IMPORTANT: return a "ui" payload. ComfyUI only emits the "executed"
        # websocket message (which triggers node.onExecuted / the frontend's
        # api "executed" event) for nodes whose result includes UI output —
        # without this the browser extension never learns the solve is ready
        # and never fetches the camera data / background / proxies.
        ui_payload = {"atlas_ready": [node_id]}

        if not client_data.strip():
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return {"ui": ui_payload, "result": (blank, blank, blank, blank, blank, None)}

        try:
            data = json.loads(client_data)
        except json.JSONDecodeError:
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return {"ui": ui_payload, "result": (blank, blank, blank, blank, blank, None)}

        shaded = _decode_b64_to_tensor(data.get("shaded", ""), width, height)
        depth  = _decode_b64_to_tensor(data.get("depth",  ""), width, height)
        normal = _decode_b64_to_tensor(data.get("normal", ""), width, height)
        mask   = _decode_b64_to_tensor(data.get("mask",   ""), width, height)

        path_frames_b64 = data.get("path_frames") or []
        if path_frames_b64:
            path_frames = torch.cat(
                [_decode_b64_to_tensor(b64, width, height) for b64 in path_frames_b64], dim=0
            )
        else:
            path_frames = torch.zeros(1, height, width, 3, dtype=torch.float32)

        camera_path_data = data.get("camera_path")
        camera_path = None
        if camera_path_data:
            from atlas_camera.core.schema import AtlasCameraPath
            camera_path = AtlasCameraPath.from_dict(camera_path_data)

        return {"ui": ui_payload, "result": (shaded, depth, normal, mask, path_frames, camera_path)}

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
    "AtlasAddPatchView":          AtlasAddPatchView,
    "AtlasOcclusionMask":         AtlasOcclusionMask,
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
    "AtlasViewportControls":      AtlasViewportControls,
    "AtlasBlockoutViewport":      AtlasBlockoutViewport,
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   AtlasExportCameraPathUSD,
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              AtlasDepthMap,
    "AtlasDeriveReliefMesh":      AtlasDeriveReliefMesh,
    "AtlasDeriveWalls":           AtlasDeriveWalls,
    "AtlasDeriveTowersSpires":    AtlasDeriveTowersSpires,
    "AtlasDeriveRoofsFacades":    AtlasDeriveRoofsFacades,
    "AtlasDeriveInteriorRoom":    AtlasDeriveInteriorRoom,
    "AtlasMergeGeometry":         AtlasMergeGeometry,
    # Track 6 — shot format
    "AtlasDefineShotCam":         AtlasDefineShotCam,
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
    "AtlasAddPatchView":          "Atlas Add Patch View (multi-angle) 🩹",
    "AtlasOcclusionMask":         "Atlas Occlusion Mask 🕳",
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
    "AtlasViewportControls":      "Atlas Viewport Controls 🎛",
    "AtlasBlockoutViewport":      "Atlas Viewport 🧊",
    # Track 3 — camera path animation
    "AtlasExportCameraPathUSD":   "Atlas Export Camera Path (USD) 🎥",
    # Track 5 — composable geometry derivation
    "AtlasDepthMap":              "Atlas Depth Map 🌊",
    "AtlasDeriveReliefMesh":      "Atlas Derive Relief Mesh 🏔",
    "AtlasDeriveWalls":           "Atlas Derive Walls 🧱",
    "AtlasDeriveTowersSpires":    "Atlas Derive Towers & Spires 🗼",
    "AtlasDeriveRoofsFacades":    "Atlas Derive Roofs & Facades 🏛",
    "AtlasDeriveInteriorRoom":    "Atlas Derive Interior Room 🛋",
    "AtlasMergeGeometry":         "Atlas Merge Geometry 🔀",
    # Track 6 — shot format
    "AtlasDefineShotCam":         "Atlas Define Shot Cam 🎬",
}
