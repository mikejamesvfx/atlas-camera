"""ComfyUI node library for Atlas Camera."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.solver import solve_from_constraints, solve_still_image
from atlas_camera.exporters.blender_exporter import write_blender_scene_script
from atlas_camera.exporters.nuke_exporter import write_nuke_native_script, write_nuke_projection_script
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


def _extend_edge_colors(rgb, valid, px):
    """Deterministic edge-extend (the classic Nuke premult->dilate trick):
    push ``valid`` pixels' colors outward into the invalid region by ``px``
    pixels via iterative neighbor-mean propagation. Returns (rgb, mask) with
    the extension applied and the validity dilated to match.

    Runs the propagation at quarter resolution (sky is low-frequency; a
    smeared gradient is exactly the desired look) and composites the
    extension back only where the original was invalid - original pixels are
    never touched. Pure numpy, no scipy/cv2. This is deliberately NOT an
    inpaint: for narrow disocclusion slivers of smooth sky it is
    indistinguishable from one at a fraction of the cost; large structured
    reveals (clouds behind a building) still want the LaMa/inpaint chain on
    plate_image instead.
    """
    np = _require_numpy()
    rgb = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    H, W = valid.shape
    ds = 4
    small = rgb[::ds, ::ds].copy()
    v = valid[::ds, ::ds].copy()
    steps = max(1, int(round(px / ds)))
    for _ in range(steps):
        acc = np.zeros_like(small)
        cnt = np.zeros(v.shape, dtype=np.float32)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.zeros_like(small)
            shv = np.zeros_like(v)
            if dr == 1:
                sh[1:], shv[1:] = small[:-1], v[:-1]
            elif dr == -1:
                sh[:-1], shv[:-1] = small[1:], v[1:]
            elif dc == 1:
                sh[:, 1:], shv[:, 1:] = small[:, :-1], v[:, :-1]
            else:
                sh[:, :-1], shv[:, :-1] = small[:, 1:], v[:, 1:]
            acc += np.where(shv[..., None], sh, 0.0)
            cnt += shv
        newly = ~v & (cnt > 0)
        if not newly.any():
            break
        small[newly] = acc[newly] / cnt[newly, None]
        v |= newly
    # Upsample the extension and composite only into originally-invalid pixels.
    up = np.repeat(np.repeat(small, ds, axis=0), ds, axis=1)[:H, :W]
    upv = np.repeat(np.repeat(v, ds, axis=0), ds, axis=1)[:H, :W]
    out = rgb.copy()
    fill = ~valid & upv
    out[fill] = up[fill]
    return out, valid | fill


def _mask_to_b64_png(mask_arr) -> str:
    """(H,W) bool/float numpy array -> grayscale PNG data URI, for
    ProjectionSource.mask_b64 (the per-pixel edge matte the projection shader
    samples). PNG (lossless), not JPEG — a matte's 0.5 threshold must not
    pick up ringing artifacts at the exact edge being cut. Fails soft to ""
    like the image_b64 encoders."""
    try:
        np = _require_numpy()
        PILImage = _require_pil()
        arr = (np.asarray(mask_arr, dtype=np.float32).clip(0.0, 1.0) * 255).astype("uint8")
        pil = PILImage.fromarray(arr, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _image_tensor_to_preview_b64(image_tensor, *, quality: int = 85) -> str:
    try:
        pil = _image_tensor_to_pil(image_tensor)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _plate_ref_to_dict(plate_ref) -> dict[str, Any] | None:
    if plate_ref is None:
        return None
    if hasattr(plate_ref, "to_dict"):
        return plate_ref.to_dict()
    if isinstance(plate_ref, dict):
        return dict(plate_ref)
    return None


def _output_profile_to_dict(output_profile) -> dict[str, Any] | None:
    if output_profile is None:
        return None
    if hasattr(output_profile, "to_dict"):
        return output_profile.to_dict()
    if isinstance(output_profile, dict):
        return dict(output_profile)
    return None


def _clone_solve_with_metadata(solve, *, source_plate=None, output_profile=None):
    from atlas_camera.core.schema import AtlasOutputProfile, AtlasPlateRef

    # copy.deepcopy, not AtlasSolve.from_dict(solve.to_dict()): the JSON
    # round-trip walks every nested array (relief-mesh vertices/faces/uvs can
    # be hundreds of thousands of floats) through _json_ready and back twice —
    # once serializing, once reconstructing — purely to get an independent
    # copy for in-process mutation. deepcopy is the C-optimized recursive
    # copy for exactly this case; to_dict()/from_dict() remain the right tool
    # at actual serialization boundaries (file export, cache keys).
    out = copy.deepcopy(solve)
    if source_plate is not None:
        out.source_plate = source_plate if isinstance(source_plate, AtlasPlateRef) else AtlasPlateRef.from_dict(source_plate)
        if out.source_plate and out.source_plate.image_path and not out.source_plate.is_proxy:
            out.image_path = out.source_plate.image_path
    if output_profile is not None:
        out.output_profile = (
            output_profile if isinstance(output_profile, AtlasOutputProfile)
            else AtlasOutputProfile.from_dict(output_profile)
        )
    return out


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


def _image_fingerprint(image) -> str:
    """Short identity hash of an IMAGE tensor (16x-subsampled digest) — the
    approval token for AtlasAssessImage's ▶ Continue: proceed only applies to
    the image it was clicked FOR, so swapping the input photo re-arms the
    assessment gate instead of sailing through a stale approval (the same
    staleness class as 📐 Extract Angle's solve fingerprint)."""
    import hashlib

    arr = image[0, ::16, ::16].cpu().numpy()
    h = hashlib.md5()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _solve_fingerprint(solve, source_image) -> str:
    """Short identity hash of (recovered camera, source image) — stamped into
    📐 Extract Angle's client_data.patch_angle by the frontend and validated
    by AtlasBlockoutViewport.render(): an extraction from a DIFFERENT solve/
    image than the current one is treated as not-extracted, so swapping the
    input photo re-arms the patch-branch pause instead of silently running
    the previous image's stale angle. Image identity uses a 16x-subsampled
    tensor digest (full 4K hashing per execution is needless cost; a swapped
    photo always changes the subsample)."""
    import hashlib

    h = hashlib.md5()
    extr = solve.camera.extrinsics
    intr = solve.camera.intrinsics
    h.update(repr(extr.camera_view_matrix).encode())
    h.update(repr((intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px,
                   intr.image_width, intr.image_height)).encode())
    arr = source_image[0, ::16, ::16].cpu().numpy()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _execution_blocker():
    """ComfyUI's ExecutionBlocker sentinel (silent variant), or None outside a
    ComfyUI runtime. Returning it on an OUTPUT makes every downstream node
    that consumes it skip silently — the native "pause this branch" mechanism
    (used by AtlasBlockoutViewport's patch_* outputs until 📐 Extract Angle
    runs). Import is guarded because atlas_camera.comfy imports cleanly with
    no ComfyUI installed (tests, plain python) — callers fall back to plain
    default values when this returns None.
    """
    try:
        from comfy_execution.graph import ExecutionBlocker
        return ExecutionBlocker(None)
    except ImportError:
        try:
            from comfy.graph import ExecutionBlocker  # pre-2024 layout
            return ExecutionBlocker(None)
        except ImportError:
            return None


def _extract_blockout_camera(solve, source_image, target_width: int, target_height: int,
                              preview_expand: float = 1.0, shot_intrinsics=None,
                              output_profile=None, solve_fingerprint: str = "") -> dict[str, Any]:
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

    try:
        from atlas_camera.core.camera_math import ground_lookat_pivot
        orbit_pivot = [float(v) for v in ground_lookat_pivot(extr)]
    except Exception:
        orbit_pivot = [0.0, 0.0, 0.0]

    # Encode source image as JPEG base64 so the browser can use it as background
    source_b64 = _image_tensor_to_preview_b64(source_image, quality=85)
    source_plate = getattr(solve, "source_plate", None)
    if not source_b64 and source_plate is not None:
        source_b64 = source_plate.preview_b64 or ""

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
            "mask_b64": getattr(src, "mask_b64", None) or "",
            "plate_ref": _plate_ref_to_dict(getattr(src, "plate_ref", None)),
            "priority": float(src.priority),
            "azimuth_deg": float(src.azimuth_deg),
            "elevation_deg": float(src.elevation_deg),
            "projection_mode": (src.metadata or {}).get("projection_mode"),
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
        "shot_cam": shot_intrinsics is not None,
        "render_fy": render_fy,
        "render_image_height": render_image_height,
        # ground_lookat_pivot: the EXACT pivot orbit_camera uses to construct
        # patch cameras backend-side. The 📐 Extract Angle button computes its
        # orbit delta about this (not the viewport's own geometry-centroid
        # orbit pivot, which differs) so extracted angles round-trip exactly
        # through AtlasAddPatchView/AtlasOcclusionMask's orbit_camera call.
        "orbit_pivot": orbit_pivot,
        # Identity of this solve+image — 📐 Extract Angle echoes it back so a
        # stale extraction (different photo) can never drive the patch branch.
        "solve_fingerprint": solve_fingerprint,
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "source_image_b64": source_b64,
        "source_plate": _plate_ref_to_dict(source_plate),
        "output_profile": _output_profile_to_dict(
            output_profile if output_profile is not None else getattr(solve, "output_profile", None)
        ),
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
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata to embed in the review package."}),
            },
        }

    def export(self, solve, output_dir, relief_mesh_obj_path="", output_profile=None):
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
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


class AtlasRegisterPlate:
    """Register a projection plate for float-safe final handoff.

    The IMAGE passes through unchanged. The ATLAS_PLATE_REF carries a durable
    file path/colorspace when supplied; if no path is supplied it is explicitly
    marked proxy-only so exporters do not mistake a browser/JPEG preview for
    final EXR data.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_PLATE_REF")
    RETURN_NAMES = ("image", "plate_ref")
    FUNCTION = "register"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "plate_path": ("STRING", {"default": "",
                    "tooltip": "Original/final plate path, ideally EXR 16f/32f. Leave blank to mark this as proxy-only."}),
                "colorspace": ("STRING", {"default": "ACEScg",
                    "tooltip": "Source plate colorspace for Nuke/Maya/OCIO handoff."}),
                "bit_depth": ("STRING", {"default": "auto",
                    "tooltip": "Descriptive bit depth, e.g. 16f, 32f, 10-bit, 8-bit, or auto."}),
                "role": (["source", "patch", "clean_plate", "matte", "proxy"], {"default": "source"}),
                "lut_path": ("STRING", {"default": ""}),
            },
        }

    def register(self, image, plate_path="", colorspace="ACEScg", bit_depth="auto", role="source", lut_path=""):
        from atlas_camera.core.schema import AtlasPlateRef

        path = str(plate_path or "").strip() or None
        suffix = Path(path).suffix.lower() if path else ""
        inferred_depth = bit_depth if bit_depth and bit_depth != "auto" else (
            "16f/32f" if suffix == ".exr" else "8-bit/proxy"
        )
        plate_ref = AtlasPlateRef(
            image_path=path,
            preview_b64=_image_tensor_to_preview_b64(image, quality=85),
            colorspace=colorspace or "ACEScg",
            bit_depth=inferred_depth,
            role=role or "source",
            is_proxy=not bool(path),
            lut_path=(str(lut_path).strip() or None),
            metadata={
                "path_exists": bool(path and Path(path).is_file()),
                "registered_from": "AtlasRegisterPlate",
            },
        )
        return (image, plate_ref)


class AtlasAttachSourcePlate:
    """Attach a registered source plate to an Atlas solve."""

    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "attach"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "plate_ref": ("ATLAS_PLATE_REF",),
            },
        }

    def attach(self, solve, plate_ref):
        return (_clone_solve_with_metadata(solve, source_plate=plate_ref),)


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


class AtlasAssessImage:
    """VLM pre-flight for the whole DMP pipeline — wire it directly after
    LoadImage, BEFORE anything else consumes the photo.

    A local vision-language model (Ollama / LM Studio / llama.cpp — the same
    provider layer as `AtlasVLMScaleCues`) analyzes the photo against an
    expert instruction prompt encoding Atlas Camera's full settings knowledge
    (`inference.assessor.ATLAS_ASSESSMENT_SYSTEM_PROMPT`): scene type /
    depth-model choice, sky separation, depth-band layer design, disocclusion
    fill, edge mattes, relief tuning, scale-reference opportunities, and an
    honest camera-move viability rubric (score + max orbit degrees + what
    breaks first). The `report` output is human-readable (wire to a
    Show Text node); `settings_json` is the machine-readable
    recommended_settings block.

    EXECUTION PAUSE: while `proceed` is False (default) the `image` output
    returns ExecutionBlocker — everything downstream of the photo is silently
    skipped, so the first Queue costs only the assessment. Read the report,
    apply the recommended settings to the graph, then click ▶ Continue
    Workflow (sets `proceed` and re-queues; the assessment itself is cached
    per image+provider so continuing doesn't re-run the VLM). Same native
    pause mechanism as the viewport's 📐 Extract Angle gating.

    Advisory only, per the LLM-confirm principle: the VLM never changes a
    setting itself — it recommends, the artist decides. Fails soft to a
    "provider unreachable" report; `proceed` still works without an
    assessment.
    """
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "report", "settings_json")
    FUNCTION = "assess"
    CATEGORY = "Atlas Camera"
    # OUTPUT_NODE so the assessment ALWAYS runs and shows its report on the
    # node itself (ui.text, rendered by atlas_assess.js) — without this, a
    # graph where nothing consumed `report` gave zero visible output (found
    # live: "the VLM did nothing").
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "provider": (["ollama", "lmstudio", "llamacpp"], {"default": "ollama",
                    "tooltip": "Local VLM server. Blank model/base_url use each provider's own "
                               "defaults (same conventions as AtlasVLMScaleCues)."}),
                "model": ("STRING", {"default": "",
                    "tooltip": "Vision model id; blank = provider default (ollama: gemma3:4b)."}),
                "base_url": ("STRING", {"default": ""}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Optional artist notes appended to the assessment request — e.g. "
                               "'the camera move is a slow dolly-in on the tower'. The VLM tailors "
                               "band/patch advice to the intended move."}),
                "proceed": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF = the image output is paused (downstream skipped) so you can "
                               "read the report and apply settings first. Turn ON (or click "
                               "▶ Continue Workflow) and re-queue to run the full pipeline. "
                               "A ▶ Continue click approves THIS image only (see approved_for); "
                               "a manual toggle here is an unconditional override."}),
                "approved_for": ("STRING", {"default": "",
                    "tooltip": "Managed by ▶ Continue Workflow: the fingerprint of the image the "
                               "current proceed=True was approved for. When the input image "
                               "changes, the gate re-arms automatically instead of running a "
                               "stale approval. Leave empty when toggling proceed by hand "
                               "(empty = unconditional)."}),
            },
        }

    def assess(self, image, provider="ollama", model="", base_url="",
               extra_instructions="", proceed=False, approved_for="", **_extra):
        # **_extra: API-format exports can serialize the ▶ Continue Workflow
        # BUTTON widget as a bogus input key — tolerate unknown kwargs.
        import hashlib

        from atlas_camera.inference.assessor import assess_image

        # Cache per image+provider settings so flipping `proceed` (which
        # re-executes this node) doesn't re-run a 30-120s VLM call.
        key_src = image.cpu().numpy().tobytes()
        key = hashlib.md5(key_src).hexdigest() + f"|{provider}|{model}|{base_url}|{extra_instructions}"
        cached = _ATLAS_ASSESS_CACHE.get(key)
        if cached is None:
            tmp = _save_image_tensor_to_tmp(image)
            try:
                cached = assess_image(
                    tmp, provider=provider, model=model,
                    base_url=base_url.strip() or None,
                    extra_instructions=extra_instructions)
            finally:
                os.unlink(tmp)
            # Never cache FAILED assessments: the user typically starts the
            # provider after seeing the failure report — the retry must
            # actually retry.
            if cached.ok:
                if len(_ATLAS_ASSESS_CACHE) >= 8:
                    _ATLAS_ASSESS_CACHE.pop(next(iter(_ATLAS_ASSESS_CACHE)))
                _ATLAS_ASSESS_CACHE[key] = cached

        settings_json = json.dumps(
            (cached.payload or {}).get("recommended_settings", {}), indent=1) if cached.ok else "{}"

        # ▶ Continue approvals are per-image: a non-empty approved_for that
        # doesn't match the CURRENT image re-arms the gate (found live — the
        # proceed widget persists, so a new image sailed through the previous
        # image's approval). An empty approved_for with proceed=True is the
        # manual unconditional override.
        img_fp = _image_fingerprint(image)
        report = cached.report
        effective_proceed = bool(proceed) and (not approved_for or approved_for == img_fp)
        if proceed and approved_for and approved_for != img_fp:
            report = ("*** GATE RE-ARMED: the input image changed since ▶ Continue was "
                      "clicked — review the fresh assessment below, then ▶ Continue "
                      "again for this image. ***\n\n" + report)

        if effective_proceed:
            img_out = image
        else:
            blocker = _execution_blocker()
            img_out = blocker if blocker is not None else image
        # ui.text renders the report directly on the node (atlas_assess.js);
        # ui.fingerprint is what the ▶ button stamps into approved_for.
        return {"ui": {"text": [report], "fingerprint": [img_fp]},
                "result": (img_out, report, settings_json)}


_ATLAS_ASSESS_CACHE: dict = {}


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

    ``hole_mask`` mirrors the relief mesh's own discarded hole/tear data
    (`ReliefMesh.hole_mask`) whenever ``geometry_mode`` builds one ("both"/
    "relief_mesh") - full source-image resolution, white where no triangle
    covers that pixel. A zero mask when ``geometry_mode="primitives"``, since
    no relief mesh is built to have holes in that mode.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "MASK")
    RETURN_NAMES = ("solve", "hole_mask")
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
                "relief_grid": ("INT", {"default": 128, "min": 16, "max": 4096,
                    "tooltip": "Viewport relief-mesh density (long-edge grid columns). Higher = "
                               "fewer/smaller torn holes on noisy AI-image depth (each quad spans "
                               "less real-world area, so it's less likely to straddle a spurious "
                               "depth jump) at the cost of a larger mesh payload sent to the "
                               "browser and a slower/heavier viewport. Overridden by "
                               "relief_quality unless that's set to 'custom'."}),
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
                # Appended at the end (not inserted earlier in this dict) so that
                # ComfyUI's positional widgets_values array stays backward
                # compatible: a workflow saved before these two existed just gets
                # its own defaults filled in for these trailing slots, instead of
                # every later value shifting into the wrong widget.
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
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional external exclusion (e.g. a real sky segmentation from "
                               "SAM/RMBG) ORed on top of the internal sky heuristic before "
                               "triangulation - never replaces it, only excludes more. Only "
                               "affects the relief_mesh branch (geometry_mode both/relief_mesh); "
                               "the primitives/wall-fitting branch is unaffected. Any resolution - "
                               "resized to match depth."}),
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
               primitive_method="azimuth_walls", scene_type="manual", exclude_mask=None):
        torch = _require_torch()
        np = _require_numpy()
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
            zero = torch.zeros(1, int(image.shape[1]), int(image.shape[2]), dtype=torch.float32)
            return (solve, zero)
        cx = intr.cx_px if intr.cx_px is not None else width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else height / 2.0
        resolved_exclude = _resolve_exclude_mask(exclude_mask, height, width)

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

        hole_mask_arr = np.zeros((height, width), dtype=np.float32)
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
                exclude_mask=resolved_exclude,
                apply_sky_heuristic=resolved_exclude is None,
            )
            keep.append(relief_mesh_primitive(mesh))
            stats["relief_mesh"] = {
                "n_vertices": mesh.stats["n_vertices"],
                "n_faces": mesh.stats["n_faces"],
            }
            hole_mask_arr = mesh.hole_mask.astype(np.float32)

        # Deep-copy (not to_dict()/from_dict() — see _clone_solve_with_metadata's
        # comment): never mutate the upstream node's cached ATLAS_SOLVE.
        out = copy.deepcopy(solve)
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
        hole_t = torch.from_numpy(hole_mask_arr).unsqueeze(0)
        return (out, hole_t)


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
    out = copy.deepcopy(solve)
    out.projection_scene.proxy_geometry = [
        p for p in out.projection_scene.proxy_geometry
        if (p.metadata or {}).get("role") != PROXY_ROLE
    ]
    out.projection_scene.proxy_geometry.extend(new_prims)
    out.projection_scene.debug_metadata["proxy_derivation"] = {**stats, **extra_metadata}
    return out


class _MetricDepthSetup(NamedTuple):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    extr: Any
    depth_map: Any
    scale: float
    horizon_y: float
    metric: Any
    valid: Any
    exclude_mask: Any  # resolved (H,W) bool numpy array, or None if not supplied


# Segmentation masks (SAM and similar) systematically FADE at frame borders
# (measured live: SAM3 sky coverage 5.9% at row 0, 29% at row 5, 100% by row
# 50 on a clear-sky plate) — so any border-touching mechanism (frame
# outpaint's edge replication, the sky card's own ring) sees a boundary row
# that lies about its content. Floods within this margin of each border.
_BORDER_FLOOD_PX = 64


def _flood_mask_to_frame_borders(mask, margin_px=_BORDER_FLOOD_PX):
    """Flood a boolean mask to the frame borders wherever it touches within
    ``margin_px``: a column with sky at row 40 is sky at rows 0-39 too (the
    segmenter faded, physics didn't). Content genuinely cut by the frame (a
    spire reaching the top edge) has no mask in the margin and is untouched.
    Applied per border, perpendicular fill only."""
    np = _require_numpy()
    m = np.asarray(mask, dtype=bool).copy()
    k = int(margin_px)
    if k <= 0 or not m.any():
        return m
    # top: propagate True upward within the margin slice
    m[:k] |= np.flip(np.logical_or.accumulate(np.flip(m[:k], axis=0), axis=0), axis=0)
    # bottom
    m[-k:] |= np.logical_or.accumulate(m[-k:], axis=0)
    # left
    m[:, :k] |= np.flip(np.logical_or.accumulate(np.flip(m[:, :k], axis=1), axis=1), axis=1)
    # right
    m[:, -k:] |= np.logical_or.accumulate(m[:, -k:], axis=1)
    return m


def _resolve_exclude_mask(mask_tensor, height, width):
    """Convert an optional ComfyUI MASK tensor (any resolution, values 0..1)
    into a (height, width) bool numpy array - True = exclude this pixel from
    the mesh, same semantics as depth_geometry.detect_sky_mask's own output
    (e.g. a real sky segmentation from SAM/RMBG run upstream, since the
    internal detect_sky_mask heuristic is often wrong on complex real photos).
    Resized via the same nearest-neighbour path _resize_depth already uses
    for depth (no cv2 dependency). Returns None when no mask was supplied -
    callers OR this into their own validity mask, never replace it, so an
    absent mask is always a no-op.
    """
    if mask_tensor is None:
        return None
    np = _require_numpy()
    from atlas_camera.core.solver import _resize_depth
    arr = mask_tensor[0].detach().cpu().numpy().astype(np.float64)
    if arr.shape != (height, width):
        arr = _resize_depth(arr, width, height)
    return arr > 0.5


def _metric_depth_and_validity(solve, depth, exclude_mask=None) -> "_MetricDepthSetup | None":
    """Shared metric-depth + validity-mask setup for the inpaint-layers nodes.

    Was previously inlined identically in both ``AtlasDepthLayerMask.generate``
    and ``AtlasCleanPlateLayer.add_layer`` (~15 lines each) — extracted here so
    the two nodes can never disagree about what "metric depth" or "a valid,
    non-sky pixel" means for a given solve, the same reasoning that motivated
    the separate ``_resolve_depth_band`` extraction just below. Returns
    ``None`` when the solve has no usable focal length (caller should pass
    the input through/return zero masks, matching the existing per-node
    no-focal-length conventions).

    ``exclude_mask`` (optional ComfyUI MASK tensor) ORs an external exclusion
    on top of the internal sky heuristic — see ``_resolve_exclude_mask``. The
    resolved (H,W) bool array is also returned on the setup so callers can
    pass the identical mask into their own ``build_relief_mesh`` call without
    resolving it twice.
    """
    np = _require_numpy()
    from atlas_camera.core.depth_geometry import detect_sky_mask
    from atlas_camera.core.relief_mesh import estimate_ground_scale

    params = _solve_camera_params(solve, depth)
    if params is None:
        return None
    width, height, fx, fy, cx, cy = params
    depth_map = _depth_map_for_solve(depth, width, height)
    horizon_y = _horizon_y_from_solve(solve)
    if horizon_y is None:
        horizon_y = height * 0.45  # same fallback build_relief_mesh uses internally
    extr = solve.camera.extrinsics

    scale, _ground_info = estimate_ground_scale(
        depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
        horizon_y=horizon_y)
    metric = depth_map.astype(np.float64) * scale
    resolved_exclude = _resolve_exclude_mask(exclude_mask, height, width)
    valid = np.isfinite(depth_map) & (depth_map > 1e-4)
    if resolved_exclude is not None:
        # An explicit segmentation REPLACES the internal sky heuristic. The
        # heuristic flags above-horizon far/rough pixels as sky, which eats
        # real tall geometry (buttes, towers, spires) that a real SAM mask
        # correctly leaves alone — found live: ~50% of monument valley's
        # butte silhouettes were heuristic-excluded despite a perfect SAM sky
        # mask being wired in. Need both signals? OR the sky into your
        # exclusion mask externally.
        valid &= ~resolved_exclude
    else:
        valid &= ~detect_sky_mask(depth_map, horizon_y=horizon_y)
    return _MetricDepthSetup(width, height, fx, fy, cx, cy, extr, depth_map, scale, horizon_y, metric, valid,
                              resolved_exclude)


def _resolve_depth_band(metric, valid, near_m, far_m, near_pct, far_pct):
    """Resolve a metric depth band from explicit metres (``near_m``/``far_m``,
    0 = unset) or, as a fallback, percentiles (``near_pct``/``far_pct``, 0..1)
    over the valid (non-sky) metric depth distribution.

    Shared by ``AtlasDepthLayerMask`` and ``AtlasCleanPlateLayer`` so the two
    nodes' bands can never drift apart — the inpaint-layers design requires the
    mask node's band and the clean-plate node's mesh clip to match exactly.
    ``far_pct<=0`` is a deliberate explicit "no upper bound" (+inf) rather than
    a degenerate 0th-percentile far edge, since ``near_pct``/``far_pct`` share
    the same 0..1 range but mean different things at 0 (near defaults to the
    very nearest pixels; far defaults to "no cap" via ``far_pct=0.5``, and an
    artist setting ``far_pct=0`` clearly means "no upper band edge", not
    "collapse the band to nothing").
    """
    np = _require_numpy()
    values = metric[valid] if valid.any() else None
    if near_m and near_m > 0:
        near = float(near_m)
    elif values is not None and near_pct > 0:
        near = float(np.percentile(values, near_pct * 100.0))
    else:
        near = 0.0
    if far_m and far_m > 0:
        far = float(far_m)
    elif values is not None and far_pct > 0:
        far = float(np.percentile(values, far_pct * 100.0))
    else:
        far = float("inf")
    return near, far


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

    ``hole_mask`` mirrors `build_relief_mesh`'s own discarded hole/tear data
    (see `ReliefMesh.hole_mask`) - full source-image resolution, white where
    no triangle covers that pixel (sky/invalid/silhouette tear). This is the
    literal "where will Project show black" signal, not a heuristic.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "MASK")
    RETURN_NAMES = ("solve", "hole_mask")
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
                "relief_grid": ("INT", {"default": 128, "min": 16, "max": 4096,
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
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional external exclusion (e.g. a real sky segmentation from "
                               "SAM/RMBG) ORed on top of the internal sky heuristic before "
                               "triangulation - never replaces it, only excludes more. Any "
                               "resolution - resized to match depth."}),
            },
        }

    _RELIEF_QUALITY_PRESETS = {"low": 64, "medium": 256, "high": 512, "ultra": 1024}

    def derive(self, solve, depth, relief_grid=128, relief_quality="custom", depth_edge_rel=0.5,
               exclude_mask=None):
        torch = _require_torch()
        np = _require_numpy()
        if relief_quality in self._RELIEF_QUALITY_PRESETS:
            relief_grid = self._RELIEF_QUALITY_PRESETS[relief_quality]
        from atlas_camera.core.depth_geometry import back_project_normals, build_backdrop_primitive
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale

        params = _solve_camera_params(solve, depth)
        if params is None:
            h, w = int(depth.image_height), int(depth.image_width)
            return (solve, torch.zeros(1, h, w, dtype=torch.float32))
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        resolved_exclude = _resolve_exclude_mask(exclude_mask, height, width)

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
            scale=scale, horizon_y=horizon_y, exclude_mask=resolved_exclude,
            apply_sky_heuristic=resolved_exclude is None)
        prims = [backdrop, relief_mesh_primitive(mesh)]
        stats = {
            "ground_scale": scale, "ground_fit": ground_info,
            "relief_mesh": {"n_vertices": mesh.stats["n_vertices"], "n_faces": mesh.stats["n_faces"]},
        }
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "relief_grid": int(relief_grid), "relief_quality": relief_quality,
            "depth_edge_rel": float(depth_edge_rel), "derive_node": "AtlasDeriveReliefMesh",
        })
        hole_t = torch.from_numpy(mesh.hole_mask.astype(np.float32)).unsqueeze(0)
        return (out, hole_t)


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
        out = copy.deepcopy(solve_a)
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


def _parse_view_prompt(text):
    """Parse a Multiple-Angles LoRA prompt — "<sks> [azimuth] [elevation]
    [distance]", the exact string 📐 Extract Angle's `patch_prompt` output
    emits — back into the three named views. Returns (azimuth, elevation,
    distance) or None when the text doesn't match the vocabulary.

    Exists because ComfyUI's backend REJECTS a STRING link into a combo-list
    input ("received_type(STRING) mismatch input_type([...])" at prompt
    validation), so the viewport's per-view STRING outputs can't wire into
    the named-view dropdowns directly — instead one `patch_view_override`
    STRING socket takes the whole prompt and this parses it. The names
    contain spaces, so parsing is greedy prefix-matching against the known
    vocabularies (longest first), which is unambiguous because the LoRA's
    view names are a fixed, non-overlapping set.
    """
    rest = (text or "").strip()
    if rest.startswith("<sks>"):
        rest = rest[len("<sks>"):].strip()
    parsed = []
    for table in (_AZIMUTH_VIEWS, _ELEVATION_VIEWS, _DISTANCE_VIEWS):
        match = next((name for name in sorted(table, key=len, reverse=True)
                      if rest.startswith(name)), None)
        if match is None:
            return None
        parsed.append(match)
        rest = rest[len(match):].strip()
    if rest:
        return None
    return tuple(parsed)


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
                "relief_grid": ("INT", {"default": 96, "min": 16, "max": 4096,
                    "tooltip": "Patch relief-mesh density (long-edge grid columns)."}),
                "priority": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blend priority among patches (higher wins). The primary photo "
                               "is always highest; patches only fill where it can't see."}),
                "plate_ref": ("ATLAS_PLATE_REF", {
                    "tooltip": "Optional registered final plate reference. Browser still uses image_b64 preview; exporters use this for EXR/float-safe handoff."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "patch_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_prompt output here "
                               "(the '<sks> [azimuth] [elevation] [distance]' string from 📐 "
                               "Extract Angle) — when connected it OVERRIDES the three patch_* "
                               "dropdowns above, so the extracted angle drives both the Qwen "
                               "generation and this node identically with one wire. (A single "
                               "STRING socket because ComfyUI's backend rejects STRING links "
                               "into combo dropdowns.) Errors loudly if the string doesn't "
                               "parse, rather than silently patching at the wrong angle."}),
                "mask_unseen_only": ("BOOLEAN", {"default": True,
                    "tooltip": "Embed an UNSEEN-AREAS matte on the patch (ProjectionSource."
                               "mask_b64): the patch only paints where the PRIMARY camera's "
                               "projection is invalid at the patch view (behind-camera, out-of-"
                               "frame, and — when primary_depth is wired — hidden behind nearer "
                               "geometry, the true MPTK depth-shadow test). Everywhere the "
                               "primary CAN see keeps the primary's real pixels; the AI patch "
                               "fills only genuine gaps. Also rides into the Nuke/Maya exports "
                               "as the patch plate's alpha."}),
                "unseen_dilate_px": ("INT", {"default": 16, "min": 0, "max": 200,
                    "tooltip": "Dilate the unseen matte so the patch slightly overlaps the "
                               "primary's coverage edge (hides hairline seams at the boundary)."}),
                "primary_depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "STRONGLY RECOMMENDED: the shared AtlasDepthMap of the SOURCE "
                               "photo. Enables (1) overlap-based scale REGISTRATION — the patch "
                               "mesh's metric scale is solved by matching its depth against the "
                               "primary's in the mutually-visible region, so the patch actually "
                               "sits in the primary's world instead of trusting an independent "
                               "(and fragile, on AI-generated views) ground fit; and (2) the true "
                               "depth-shadow term in the unseen matte."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Segmentation of the PATCH image's sky (run SAM3Segment on the "
                               "generated novel view, prompt 'sky'). In reuse_scene mode it keeps "
                               "the patch from painting sky onto scene geometry; in own_depth "
                               "mode it REPLACES the internal sky heuristic during meshing "
                               "(hallucinated near-depth sky otherwise triangulates into "
                               "geometry bulging toward the camera)."}),
                "geometry_source": (["reuse_scene", "own_depth"], {"default": "reuse_scene",
                    "tooltip": "reuse_scene (recommended): the patch derives NO geometry of its "
                               "own — it becomes a pure texture projector onto copies of the "
                               "geometry already in the solve (sky dome, band meshes, derived "
                               "proxies), exactly how a DMP artist projects new paint from a "
                               "second camera onto the SAME geo in Nuke. The scale/registration "
                               "problem dissolves: that geometry is in the primary's world by "
                               "construction, and Qwen scene mismatch shows only as texture "
                               "misalignment, never floating geometry. No depth model runs. "
                               "own_depth: the previous behavior (Depth Anything on the patch + "
                               "overlap registration) — for patches revealing genuinely NEW "
                               "terrain no existing geometry covers. Auto-falls back to "
                               "own_depth when the solve carries no geometry to reuse."}),
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
                  relief_grid=96, priority=1.0, plate_ref=None, device="auto",
                  patch_view_override="", mask_unseen_only=True, unseen_dilate_px=16,
                  primary_depth=None, exclude_mask=None, geometry_source="reuse_scene"):
        if patch_view_override and patch_view_override.strip():
            parsed = _parse_view_prompt(patch_view_override)
            if parsed is None:
                raise ValueError(
                    f"patch_view_override {patch_view_override!r} does not parse as "
                    "'<sks> [azimuth] [elevation] [distance]' — wire AtlasBlockoutViewport's "
                    "patch_prompt output here, or disconnect to use the dropdowns."
                )
            patch_azimuth_view, patch_elevation_view, patch_distance = parsed
        from atlas_camera.core.camera_math import (
            ground_lookat_pivot,
            horizon_row_from_extrinsics,
            orbit_camera,
        )
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.core.schema import (
            AtlasIntrinsics,
            AtlasPlateRef,
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

        np = _require_numpy()
        from atlas_camera.core.depth_geometry import (
            back_project_normals,
            primary_camera_validity_mask,
        )
        from atlas_camera.core.proxy_geometry import PROXY_ROLE

        resolved_exclude = _resolve_exclude_mask(exclude_mask, patch_h, patch_w)

        # --- reuse_scene: the patch is a TEXTURE PROJECTOR onto the geometry
        # the scene already has — the DMP move (project new paint from a
        # second camera onto the SAME geo). Deriving geometry from monocular
        # depth on a HALLUCINATED image can never reliably land in the
        # primary's metric world (scale+shift error plus genuine scene
        # mismatch — per-pixel registration confirmed insufficient in Nuke),
        # so we stop trying: reused geometry is in the primary's world by
        # construction, and any Qwen mismatch shows as texture misalignment,
        # never floating geometry.
        reused_geom = []
        fallback_reason = None
        if geometry_source == "reuse_scene":
            for prim in solve.projection_scene.proxy_geometry:
                if (prim.metadata or {}).get("role") == PROXY_ROLE:
                    reused_geom.append(copy.deepcopy(prim))
            for prev in solve.projection_sources:
                for prim in prev.proxy_geometry:
                    reused_geom.append(copy.deepcopy(prim))
            for i, prim in enumerate(reused_geom):
                prim.name = f"{name}_reuse{i}_{prim.name}"
            if not reused_geom:
                geometry_source = "own_depth"
                fallback_reason = "no scene geometry to reuse"

        depth_map = None
        if geometry_source == "own_depth":
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

        # --- Patch scale: REGISTER against the primary's metric world when the
        # shared primary depth is available; ground-fit is only the fallback.
        # An independent estimate_ground_scale on an AI-generated novel view is
        # fragile — when it misfits, the whole patch mesh lands at the wrong
        # world scale ("the patch doesn't sit with the main geometry", found
        # live). Registration exploits the OVERLAP both cameras see: scaling
        # about the patch camera makes each point's depth in the PRIMARY
        # camera affine in the scale s — z(s) = z_cam + s·(z_p − z_cam) — so
        # each overlap pixel yields a closed-form s = (m − z_cam)/(z_p − z_cam)
        # against the primary's stored metric depth m, and the median over
        # thousands of pixels is a robust one-parameter alignment.
        primary_metric_map = None
        if primary_depth is not None:
            p_map = _depth_map_for_solve(primary_depth, p_w, p_h)
            p_scale, _ = estimate_ground_scale(
                p_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=_horizon_y_from_solve(solve))
            primary_metric_map = np.asarray(p_map, dtype=np.float64) * float(p_scale)

        if geometry_source == "reuse_scene":
            patch_geom = reused_geom
            mesh = None
            scale = 1.0
            scale_source = "reuse_scene"
            # Unseen matte by FORWARD SPLAT of the primary's real metric
            # points into the patch view — coverage means "the primary has
            # trusted data that lands on this patch pixel"; no hallucinated
            # patch depth is involved at all.
            mask_b64 = None
            if mask_unseen_only and primary_metric_map is not None:
                stride = max(1, int(np.ceil(max(p_w, p_h) / 1536.0)))
                sub = primary_metric_map[::stride, ::stride]
                bp_p = back_project_normals(
                    sub, view_matrix=extr.camera_view_matrix,
                    fx=fx / stride, fy=fy / stride,
                    cx=cx / stride, cy=cy / stride)
                pts = bp_p.pts_world[bp_p.valid_depth]
                qvm = np.asarray(patch_extr.camera_view_matrix, dtype=np.float64)
                Rq, tq = qvm[:3, :3], qvm[:3, 3]
                cam_q = pts @ Rq.T + tq
                zq = -cam_q[:, 2]
                front = zq > 1e-6
                with np.errstate(all="ignore"):
                    uq = pcx + pfx * cam_q[:, 0] / np.where(front, zq, np.nan)
                    vq = pcy - pfy * cam_q[:, 1] / np.where(front, zq, np.nan)
                hit = front & np.isfinite(uq) & np.isfinite(vq) & \
                    (uq >= 0) & (uq < patch_w) & (vq >= 0) & (vq < patch_h)
                coverage = np.zeros((patch_h, patch_w), dtype=bool)
                coverage[vq[hit].astype(np.int64), uq[hit].astype(np.int64)] = True
                # Close splat sparsity (patch pixels between projected
                # samples) so 'seen' isn't undercounted — an undercounted
                # coverage would let the AI patch overwrite real pixels.
                close_px = max(2, int(round(2.0 * patch_w * stride / p_w)))
                for _ in range(close_px):
                    up = np.zeros_like(coverage); up[:-1, :] = coverage[1:, :]
                    dn = np.zeros_like(coverage); dn[1:, :] = coverage[:-1, :]
                    lf = np.zeros_like(coverage); lf[:, :-1] = coverage[:, 1:]
                    rt = np.zeros_like(coverage); rt[:, 1:] = coverage[:, :-1]
                    coverage = coverage | up | dn | lf | rt
                unseen = ~coverage
                if resolved_exclude is not None:
                    unseen &= ~resolved_exclude  # never paint sky onto geometry
                for _ in range(int(unseen_dilate_px)):
                    up = np.zeros_like(unseen); up[:-1, :] = unseen[1:, :]
                    dn = np.zeros_like(unseen); dn[1:, :] = unseen[:-1, :]
                    lf = np.zeros_like(unseen); lf[:, :-1] = unseen[:, 1:]
                    rt = np.zeros_like(unseen); rt[:, 1:] = unseen[:, :-1]
                    unseen = unseen | up | dn | lf | rt
                mask_b64 = _mask_to_b64_png(unseen) or None
            return self._finish_patch(
                solve, patch_image, patch_intr, patch_extr, patch_geom, mesh,
                mask_b64, plate_ref, name, priority,
                d_azimuth, d_elevation, distance_scale,
                patch_azimuth_view, patch_elevation_view, patch_distance,
                source_azimuth_view, flip_azimuth, pivot, depth_model,
                scale_source, scale, fallback_reason)

        scale = None
        scale_source = "ground_fit"
        if primary_metric_map is not None:
            bp_raw = back_project_normals(
                depth_map, view_matrix=patch_extr.camera_view_matrix,
                fx=pfx, fy=pfy, cx=pcx, cy=pcy)
            pvm = np.asarray(extr.camera_view_matrix, dtype=np.float64)
            R, t = pvm[:3, :3], pvm[:3, 3]
            cam_pts = bp_raw.pts_world @ R.T + t
            z_p = -cam_pts[..., 2]
            patch_cam = np.asarray(
                [float(v) for v in patch_extr.camera_position], dtype=np.float64)
            z_cam = float(-(R @ patch_cam + t)[2])
            with np.errstate(all="ignore"):
                px = cx + fx * cam_pts[..., 0] / np.where(z_p > 1e-6, z_p, np.nan)
                py = cy - fy * cam_pts[..., 1] / np.where(z_p > 1e-6, z_p, np.nan)
            in_frame = np.isfinite(px) & np.isfinite(py) & \
                (px >= 0) & (px < p_w) & (py >= 0) & (py < p_h)
            sx = np.clip(np.where(in_frame, px, 0.0), 0, primary_metric_map.shape[1] - 1).astype(np.int64)
            sy = np.clip(np.where(in_frame, py, 0.0), 0, primary_metric_map.shape[0] - 1).astype(np.int64)
            m = primary_metric_map[sy, sx]
            denom = z_p - z_cam
            ok = bp_raw.valid_depth & in_frame & (z_p > 1e-6) & \
                np.isfinite(m) & (m > 1e-4) & (np.abs(denom) > 1e-3)
            if resolved_exclude is not None:
                ok &= ~resolved_exclude  # sky pixels are noise for registration
            with np.errstate(all="ignore"):
                s_samples = (m - z_cam) / denom
            ok &= np.isfinite(s_samples) & (s_samples > 1e-3) & (s_samples < 1e3)
            if int(ok.sum()) >= 500:
                scale = float(np.median(s_samples[ok]))
                scale_source = "primary_registration"

        if scale is None:
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
            exclude_mask=resolved_exclude,
            apply_sky_heuristic=resolved_exclude is None,
        )
        patch_geom = [relief_mesh_primitive(mesh, name=f"{name}_relief_mesh")]

        # Unseen-areas matte: the patch should only paint where the PRIMARY
        # camera's projection is invalid at this patch view — everywhere the
        # primary CAN see keeps its real photographed pixels, and the AI
        # novel view fills only genuine gaps. Same math as AtlasOcclusionMask
        # (frustum/frame + optional depth-shadow), embedded directly as this
        # source's per-pixel edge matte instead of a separate composite step.
        # Uses the REGISTERED scale so the depth-shadow comparison happens in
        # the same metric world the mesh lives in.
        mask_b64 = None
        if mask_unseen_only:
            bp = back_project_normals(
                depth_map * float(scale), view_matrix=patch_extr.camera_view_matrix,
                fx=pfx, fy=pfy, cx=pcx, cy=pcy)
            unseen = primary_camera_validity_mask(
                bp.pts_world, bp.valid_depth, bp.normals, bp.valid_normal,
                primary_view_matrix=extr.camera_view_matrix,
                primary_fx=fx, primary_fy=fy, primary_cx=cx, primary_cy=cy,
                primary_width=p_w, primary_height=p_h,
                angle_threshold_deg=90.0,
                primary_depth_map=primary_metric_map)
            for _ in range(int(unseen_dilate_px)):
                up = np.zeros_like(unseen); up[:-1, :] = unseen[1:, :]
                dn = np.zeros_like(unseen); dn[1:, :] = unseen[:-1, :]
                lf = np.zeros_like(unseen); lf[:, :-1] = unseen[:, 1:]
                rt = np.zeros_like(unseen); rt[:, 1:] = unseen[:, :-1]
                unseen = unseen | up | dn | lf | rt
            mask_b64 = _mask_to_b64_png(unseen) or None

        return self._finish_patch(
            solve, patch_image, patch_intr, patch_extr, patch_geom, mesh,
            mask_b64, plate_ref, name, priority,
            d_azimuth, d_elevation, distance_scale,
            patch_azimuth_view, patch_elevation_view, patch_distance,
            source_azimuth_view, flip_azimuth, pivot, depth_model,
            scale_source, scale, fallback_reason)

    def _finish_patch(self, solve, patch_image, patch_intr, patch_extr,
                      patch_geom, mesh, mask_b64, plate_ref, name, priority,
                      d_azimuth, d_elevation, distance_scale,
                      patch_azimuth_view, patch_elevation_view, patch_distance,
                      source_azimuth_view, flip_azimuth, pivot, depth_model,
                      scale_source, scale, fallback_reason):
        from atlas_camera.core.schema import AtlasPlateRef, LatentCamera, ProjectionSource

        # Encode the novel view as a JPEG data-URI (viewport texture).
        image_b64 = ""
        try:
            pil = _image_tensor_to_pil(patch_image)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

        metadata = {
            "source": "multi_angle_lora_patch",
            "patch_azimuth_view": patch_azimuth_view,
            "patch_elevation_view": patch_elevation_view,
            "patch_distance": patch_distance,
            "source_azimuth_view": source_azimuth_view,
            "flip_azimuth": bool(flip_azimuth),
            "pivot": [float(v) for v in pivot],
            "n_vertices": mesh.stats.get("n_vertices") if mesh is not None else None,
            "n_faces": mesh.stats.get("n_faces") if mesh is not None else None,
            "depth_model": depth_model,
            "scale_source": scale_source,
            "scale": float(scale),
            "n_reused_primitives": len(patch_geom) if scale_source == "reuse_scene" else 0,
        }
        if fallback_reason:
            metadata["geometry_fallback"] = fallback_reason

        source = ProjectionSource(
            camera=LatentCamera(intrinsics=patch_intr, extrinsics=patch_extr, name=name),
            name=name,
            image_b64=image_b64,
            mask_b64=mask_b64,
            plate_ref=plate_ref if isinstance(plate_ref, AtlasPlateRef) else AtlasPlateRef.from_dict(plate_ref),
            proxy_geometry=patch_geom,
            azimuth_deg=float(d_azimuth),      # actual orbit delta applied
            elevation_deg=float(d_elevation),
            distance_scale=float(distance_scale),
            priority=float(priority),
            metadata=metadata,
        )

        out = copy.deepcopy(solve)
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
    projected image + this target image) -> AtlasAddPatchView``.

    ``occlusion_mode="simple"`` (default) is the Phase-1 mask — frustum/
    frame/facing-angle only. ``occlusion_mode="depth_shadow"`` additionally
    detects true MPTK-style self-occlusion — a surface hidden behind NEARER
    geometry from the primary's view despite projecting inside its frame/
    angle limits — by treating the primary camera as a light and its own
    depth map as the shadow map (`primary_camera_validity_mask`'s
    ``primary_depth_map``; no rasterizer/render pass, still pure numpy and
    headless). Requires ``primary_depth`` connected (an `AtlasDepthMap` run
    on the PRIMARY/source photo — the same shared depth the derive nodes
    use); falls back to simple when it isn't. Both the primary shadow map
    and the target back-projection are ground-pinned to metric via
    `estimate_ground_scale` in this mode, so the depth comparison happens in
    one consistent world scale (simple mode's math is left byte-identical to
    before). ``depth_bias`` is the relative tolerance against depth-precision
    false positives — a point counts as shadowed only when it is more than
    ``depth_bias`` (fraction) farther than the stored primary depth at its
    pixel.
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
                "occlusion_mode": (["simple", "depth_shadow"], {"default": "simple",
                    "tooltip": "simple = Phase-1 frustum/frame/facing tests only (unchanged). "
                               "depth_shadow = additionally detect surfaces hidden behind NEARER "
                               "geometry from the primary's view (true MPTK camera-as-light "
                               "shadow test, using primary_depth as the shadow map). Falls back "
                               "to simple when primary_depth isn't connected."}),
                "primary_depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "AtlasDepthMap run on the PRIMARY/source photo — the shadow map "
                               "for depth_shadow mode. Wire the same shared AtlasDepthMap the "
                               "derive nodes already use."}),
                "depth_bias": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "depth_shadow only: relative depth tolerance before a point counts "
                               "as shadowed — guards against monocular depth-precision false "
                               "positives. 0.05 = must be 5% farther than the stored depth."}),
                "patch_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_prompt output here — "
                               "overrides the three patch_* dropdowns with 📐 Extract Angle's "
                               "snapped views, keeping this mask aligned with the same "
                               "AtlasAddPatchView wiring. Errors loudly if unparseable."}),
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
                 angle_threshold=90.0, dilate_px=0, soft_edge_px=0, power=1.0,
                 occlusion_mode="simple", primary_depth=None, depth_bias=0.05,
                 patch_view_override=""):
        if patch_view_override and patch_view_override.strip():
            parsed = _parse_view_prompt(patch_view_override)
            if parsed is None:
                raise ValueError(
                    f"patch_view_override {patch_view_override!r} does not parse as "
                    "'<sks> [azimuth] [elevation] [distance]' — wire AtlasBlockoutViewport's "
                    "patch_prompt output here, or disconnect to use the dropdowns."
                )
            patch_azimuth_view, patch_elevation_view, patch_distance = parsed
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.camera_math import ground_lookat_pivot, horizon_row_from_extrinsics, orbit_camera
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

        # depth_shadow mode: ground-pin BOTH sides to one metric world so the
        # shadow comparison (in the primary's camera space) is meaningful —
        # the same estimate_ground_scale reconciliation AtlasAddPatchView
        # applies to its patch geometry. simple mode's math stays
        # byte-identical to the original Phase-1 behavior.
        primary_metric_map = None
        if occlusion_mode == "depth_shadow" and primary_depth is not None:
            from atlas_camera.core.relief_mesh import estimate_ground_scale

            t_horizon = horizon_row_from_extrinsics(target_extr, fy=tfy, cy=tcy)
            t_scale, _ = estimate_ground_scale(
                depth_map, view_matrix=target_extr.camera_view_matrix,
                fx=tfx, fy=tfy, cx=tcx, cy=tcy, horizon_y=t_horizon)
            depth_map = depth_map * float(t_scale)

            p_map = _depth_map_for_solve(primary_depth, p_w, p_h)
            p_scale, _ = estimate_ground_scale(
                p_map, view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=_horizon_y_from_solve(solve))
            primary_metric_map = np.asarray(p_map, dtype=np.float64) * float(p_scale)

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
            primary_depth_map=primary_metric_map,
            depth_bias_rel=float(depth_bias),
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
    reproject. OBJ/MTL references a file-backed source plate when the solve has
    one; otherwise it writes a PNG preview texture. GLB remains a preview/proxy
    payload with embedded PNG texture. Ground lands on Y=0 (scale reconciled to
    the solve's camera height). Requires the [neural] extra.
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
                "grid_long_edge": ("INT", {"default": 128, "min": 16, "max": 4096,
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
        plate = getattr(solve, "source_plate", None)
        texture_path = None
        if plate is not None and getattr(plate, "image_path", None) and not getattr(plate, "is_proxy", True):
            texture_path = plate.image_path
        obj_path = glb_path = ""
        if format in ("both", "obj"):
            obj_path = export_relief_mesh(
                mesh,
                output_dir,
                texture=texture,
                texture_path=texture_path,
            )["obj"]
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
            },
            "optional": {
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata to include in the exported solve context."}),
            },
        }

    def export(self, solve, output_dir, output_profile=None):
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "build_scene.py"
        write_blender_scene_script(solve, dest)
        return (str(dest),)


class AtlasExportNuke:
    """Export a Nuke Python projection script, plus a native .nk scene, for
    the recovered camera.

    Both files describe the identical camera-projection graph (Read ->
    Project3D2 -> Card or ReadGeo2 -> ScanlineRender, Camera2 feeding both
    the projection and the render camera); the .py needs a Script Editor
    (`exec(open(...).read()); build_projection()`), the .nk opens directly
    via File > Open or drag-and-drop. Both were verified by actually
    building and rendering this graph in Nuke (16.1v3) rather than only
    reading documentation — see nuke_exporter.py's module docstring and
    CLAUDE.md's "Nuke camera-projection topology" note for what that caught
    (Card3D has no xsize/ysize, ScanlineRender has no format knob, and the
    real obj/cam input indices are 1/2, not 0/1) and for the relief-mesh
    case specifically (ReadGeo2 imports OBJ/FBX natively, but does NOT
    auto-apply the OBJ/MTL's own texture — it still needs the live
    Project3D2 projection wired into its own image input, same as Card).
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script_path", "nk_path")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            },
            "optional": {
                "relief_mesh_obj_path": ("STRING", {"default": "",
                    "tooltip": "Optional obj_path output from AtlasExportReliefMesh. When set, the real "
                               "derived relief mesh is imported (ReadGeo2) and live-projected onto instead "
                               "of the default flat 40x40m ground card — wire AtlasExportReliefMesh's "
                               "obj_path here to see real derived geometry in Nuke."}),
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata for Read/colorspace annotations."}),
            },
        }

    def export(self, solve, output_dir, relief_mesh_obj_path="", output_profile=None):
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        py_dest = out / "nuke_projection.py"
        nk_dest = out / "nuke_projection.nk"
        # AtlasExportReliefMesh's obj_path is relative to ComfyUI's own
        # working directory (same convention as this node's own output_dir),
        # not to wherever an artist eventually launches Nuke from - resolve
        # to absolute so the generated script/scene stays portable.
        mesh_path = str(Path(relief_mesh_obj_path).resolve()) if relief_mesh_obj_path else None
        write_nuke_projection_script(solve, py_dest, relief_mesh_obj_path=mesh_path)
        write_nuke_native_script(solve, nk_dest, relief_mesh_obj_path=mesh_path)
        return (str(py_dest), str(nk_dest))


class AtlasExportNukeLayers:
    """Export EVERY projection layer on a solve (sky dome, clean-plate bands,
    multi-angle patches — each `ProjectionSource`) as ONE native .nk scene:
    per-layer Read (plate) + Camera2 (that layer's own camera) + Project3D2 +
    ReadGeo2 (that layer's mesh, written as OBJ+MTL alongside), all merged
    through a single Scene node into one ScanlineRender rendered from the
    PRIMARY solved camera.

    This is the DCC handoff for the viewport's layered 📽 Project — the same
    stacked-projections model, except layer overlap is resolved by Nuke's
    real z-buffer instead of priority/facing masks (true depth wins; for
    spatially-exclusive layers — bands, sky at radius_m — that matches the
    viewport's result). Plate images come from each source's registered
    non-proxy `plate_ref` when present (float/EXR-safe), else the browser
    preview is decoded to a PNG next to the .nk. Complements — never
    replaces — `AtlasExportNuke`, which stays the single-projection
    (primary camera + one mesh/card) exporter.

    Sources without mesh geometry or a plate are skipped (summarized in the
    second output). Errors loudly when NO exportable layer exists — chain at
    least one AtlasSkyDomeLayer / AtlasCleanPlateLayer / AtlasAddPatchView
    first.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("nk_path", "summary")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports/nuke_layers"}),
            },
            "optional": {
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata for annotations."}),
            },
        }

    def export(self, solve, output_dir, output_profile=None):
        from atlas_camera.exporters.nuke_exporter import write_nuke_layers_script
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        result = write_nuke_layers_script(solve, output_dir)
        summary = f"{len(result['layers'])} layer(s): {', '.join(result['layers'])}"
        if result["skipped"]:
            summary += f" | skipped: {'; '.join(result['skipped'])}"
        return (result["nk_path"], summary)


class AtlasExportMayaLayers:
    """Export EVERY projection layer on a solve (sky dome, clean-plate bands,
    multi-angle patches — each `ProjectionSource`) as ONE Maya ASCII scene:
    per-layer projector cameras as native .ma nodes, plus an embedded on-open
    scriptNode that imports each layer's OBJ and builds the proven
    camera-projection shading network (place3dTexture parented to that
    layer's camera -> projection.pm, projType 8 — the same verified setup as
    AtlasExportMayaReviewScene's single projection).

    The Maya twin of `AtlasExportNukeLayers`: identical shared layer
    collection and on-disk assets (plates with edge mattes embedded in
    ALPHA + standalone matte PNGs + OBJ meshes), so a layer that exports to
    Nuke always exports to Maya the same way. Edge mattes drive
    lambert.transparency via the plate's alpha (the mesh's baked UVs match
    the plate frame by construction). Drag/File > Open the .ma; if Maya's
    script security blocks the on-open scriptNode, the OBJs sit next to the
    .ma for manual import.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ma_path", "summary")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera/Export"
    OUTPUT_NODE = True  # terminal write-to-disk node; kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "atlas_exports/maya_layers"}),
            },
            "optional": {
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata for annotations."}),
            },
        }

    def export(self, solve, output_dir, output_profile=None):
        from atlas_camera.exporters.maya_exporter import write_maya_layers_scene
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)
        result = write_maya_layers_scene(solve, output_dir)
        summary = f"{len(result['layers'])} layer(s): {', '.join(result['layers'])}"
        if result["skipped"]:
            summary += f" | skipped: {'; '.join(result['skipped'])}"
        return (result["ma_path"], summary)


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
                "to add at least one keyframe, then click ⏺ Bake Proxy Path before queuing "
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
    """Atlas Output Desk companion for AtlasBlockoutViewport.

    Output 0 remains the original controls link for compatibility. Output 1
    carries an OCIO-style output profile that exporters and the viewport can
    use as metadata. Browser preview is display-inferred/proxy; final color
    fidelity belongs to OCIO Write, Nuke, Maya, Resolve, etc.
    """
    RETURN_TYPES = ("ATLAS_VIEWPORT_LINK", "ATLAS_OUTPUT_PROFILE")
    RETURN_NAMES = ("controls", "output_profile")
    FUNCTION = "profile"
    CATEGORY = "Atlas Camera/Blockout"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "config_label": ("STRING", {"default": "ACES 2.0 / Studio"}),
                "config_path": ("STRING", {"default": ""}),
                "working_colorspace": ("STRING", {"default": "ACEScg"}),
                "output_colorspace": ("STRING", {"default": "ACES - ACEScg"}),
                "display": ("STRING", {"default": "sRGB - Display"}),
                "view": ("STRING", {"default": "ACES 2.0 SDR-video"}),
                "look": ("STRING", {"default": "None"}),
                "lut_path": ("STRING", {"default": ""}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "display_trim": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
        }

    def profile(self, config_label="ACES 2.0 / Studio", config_path="",
                working_colorspace="ACEScg", output_colorspace="ACES - ACEScg",
                display="sRGB - Display", view="ACES 2.0 SDR-video", look="None",
                lut_path="", exposure=0.0, gamma=1.0, display_trim=1.0):
        from atlas_camera.core.schema import AtlasOutputProfile

        return ("", AtlasOutputProfile(
            config_label=config_label or "ACES 2.0 / Studio",
            config_path=(config_path.strip() or None),
            working_colorspace=working_colorspace or "ACEScg",
            output_colorspace=output_colorspace or "ACES - ACEScg",
            display=display or "sRGB - Display",
            view=view or "ACES 2.0 SDR-video",
            look=look or "None",
            lut_path=(lut_path.strip() or None),
            exposure=float(exposure),
            gamma=float(gamma),
            display_trim=float(display_trim),
            preview_only=True,
            metadata={"source": "AtlasViewportControls"},
        ))


class AtlasBlockoutViewport:
    """
    Browser-based 3D blockout viewport initialized with the recovered camera.
    Pattern mirrors Yedp Blockout: browser renders → base64 JSON → Python decodes
    → four proxy/LDR IMAGE outputs (shaded, depth, normal, mask).

    Workflow:
    1. Connect an ATLAS_SOLVE and a source IMAGE, then queue the prompt.
    2. The Three.js viewport in the ComfyUI node opens, camera pre-aligned to the photo.
    3. Place primitive geometry (box, plane, person card, etc.).
    4. Click "Render Proxy Passes" in the viewport — fills client_data and re-queues.
    5. Four proxy/LDR IMAGE outputs are now available for ControlNet or compositing.
    6. Optional: use 🎥 Camera Path mode to author a keyframed camera move (fly
       nav, unclamped — leaving the recovered cone is expected here), then
       click "⏺ Bake Proxy Path" to fill client_data with a rendered frame sequence.
       `path_frames` (a proxy/LDR IMAGE batch) feeds a core Video Combine node directly;
       `camera_path` (the raw keyframes) feeds AtlasExportCameraPathUSD for a
       DCC-facing animated camera. Frames sampled outside the recovered
       camera's cone will show the same documented black/undefined regions as
       orbiting past it under 📽 Project — expected, not a bug.
    7. Optional: connect an AtlasViewportControls node to `controls` to move
       every button/panel (primitives, Project/Diagram/Info, Camera Path +
       presets + FBX import, Render Proxy Passes) OUT of this node — this node then
       shows the perspective render only, and can be freely resized by
       dragging its corner. `controls` carries no real data (a link exists
       purely so the two nodes' frontend JS can find each other); Python
       ignores it. With nothing connected, all controls still appear locally
       here, unchanged — fully backward-compatible with saved workflows that
       predate AtlasViewportControls.
    8. Optional: 📐 Extract Angle — orbit/fly to the view you want a patch
       generated at (e.g. the last frame of an intended camera move, MPTK
       style), click 📐, and the measured orbit delta from the RECOVERED
       camera is snapped to the Qwen Multiple-Angles LoRA's nearest named
       views (assuming the source photo is "front view"/"eye-level shot")
       and re-queued into four STRING outputs: `patch_azimuth_view`/
       `patch_elevation_view`/`patch_distance` (wire into AtlasAddPatchView/
       AtlasOcclusionMask's widgets converted-to-inputs) and `patch_prompt`
       (the ready-to-use "<sks> ..." LoRA prompt for generating the novel
       view). The delta is computed about `camera_math.ground_lookat_pivot`
       — the SAME pivot orbit_camera uses backend-side — so the snapped
       views round-trip exactly through those nodes' own camera
       construction. UNTIL an angle is extracted these outputs return
       ExecutionBlocker — downstream nodes wired to them (the Qwen
       generation, AtlasAddPatchView, exports) are silently PAUSED rather
       than running a wasted zero-orbit patch; clicking 📐 re-queues and the
       branch resumes automatically. Outside a ComfyUI runtime they fall
       back to the zero-orbit named-view defaults.
    """
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "ATLAS_CAMERA_PATH",
                    "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shaded", "depth", "normal", "mask", "path_frames", "camera_path",
                    "patch_azimuth_view", "patch_elevation_view", "patch_distance", "patch_prompt")
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
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata from AtlasViewportControls. "
                               "Browser preview remains display-inferred/proxy; final fidelity belongs "
                               "to OCIO Write, Nuke, Maya, or Resolve."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def render(self, solve, source_image, resolution, client_data, preview_expand=1.0, controls=None,
               shot_cam=None, output_profile=None, unique_id=None):
        torch = _require_torch()
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)

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
        solve_fingerprint = _solve_fingerprint(solve, source_image)
        _blockout_cache_set(node_id, _extract_blockout_camera(
            solve, source_image, width, height, preview_expand=float(preview_expand),
            shot_intrinsics=shot_intrinsics, output_profile=output_profile,
            solve_fingerprint=solve_fingerprint))

        # IMPORTANT: return a "ui" payload. ComfyUI only emits the "executed"
        # websocket message (which triggers node.onExecuted / the frontend's
        # api "executed" event) for nodes whose result includes UI output —
        # without this the browser extension never learns the solve is ready
        # and never fetches the camera data / background / proxies.
        ui_payload = {"atlas_ready": [node_id]}
        # Filled in below once client_data is parsed — lets the frontend show
        # a "patch branch paused — 📐 Extract Angle" hint instead of the
        # paused branch silently looking like a failed run.
        _patch_state = {"paused": True}

        # 📐 Extract Angle results (written into client_data by the viewport's
        # Extract Angle button). UNTIL an angle has been extracted, the four
        # patch_* outputs return ComfyUI's ExecutionBlocker sentinel — any
        # downstream node wired to them (the Qwen generation, AtlasAddPatchView,
        # the Nuke layers export) is silently SKIPPED rather than running a
        # wasted zero-orbit no-op patch. Extract Angle already re-queues on
        # click, so the paused branch resumes by itself the moment the artist
        # picks an angle — an automatic pause/resume, no manual resume node.
        # Outside a ComfyUI runtime (unit tests, plain-python imports) the
        # blocker class doesn't exist, so the zero-orbit named-view defaults
        # are returned instead — those also remain the fallback semantics if
        # a future ComfyUI ever drops ExecutionBlocker.
        _pa_defaults = ("front view", "eye-level shot", "medium shot",
                        "<sks> front view eye-level shot medium shot")

        def _patch_angle_strings(parsed):
            pa = (parsed or {}).get("patch_angle") or {}
            # A stale extraction — made from a DIFFERENT solve/image than the
            # one now wired in (or from before fingerprints existed) — must
            # re-arm the pause, not silently patch at the old image's angle.
            if pa and pa.get("fingerprint") != solve_fingerprint:
                pa = {}
            _patch_state["paused"] = not pa
            if not pa:
                blocker = _execution_blocker()
                if blocker is not None:
                    return (blocker,) * 4
                return _pa_defaults
            return (
                str(pa.get("azimuth_view") or _pa_defaults[0]),
                str(pa.get("elevation_view") or _pa_defaults[1]),
                str(pa.get("distance_view") or _pa_defaults[2]),
                str(pa.get("prompt") or _pa_defaults[3]),
            )

        if not client_data.strip():
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            pa_strings = _patch_angle_strings(None)
            ui_payload["atlas_patch_paused"] = [_patch_state["paused"]]
            return {"ui": ui_payload,
                    "result": (blank, blank, blank, blank, blank, None) + pa_strings}

        try:
            data = json.loads(client_data)
        except json.JSONDecodeError:
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            pa_strings = _patch_angle_strings(None)
            ui_payload["atlas_patch_paused"] = [_patch_state["paused"]]
            return {"ui": ui_payload,
                    "result": (blank, blank, blank, blank, blank, None) + pa_strings}

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

        pa_strings = _patch_angle_strings(data)
        ui_payload["atlas_patch_paused"] = [_patch_state["paused"]]
        return {"ui": ui_payload,
                "result": (shaded, depth, normal, mask, path_frames, camera_path)
                + pa_strings}

    @classmethod
    def IS_CHANGED(cls, client_data="", **_):
        import hashlib
        return hashlib.md5(client_data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Track 7 — inpaint layers (2.5D clean-plate parallax)
#
# Depth-band-clip a single solved photo into independent layers, inpaint the
# region each layer's foreground occluder hides ("clean plate"), and project
# each plate onto its own depth-banded relief mesh as an additional
# ProjectionSource. On a dolly/orbit move, the background layer reveals
# inpainted pixels instead of the black holes documented in CLAUDE.md's
# "Orbit coverage" rule — for the SAME camera, no angle calibration needed
# (contrast AtlasAddPatchView, which fills gaps via novel AI views at OTHER
# angles). Deliberately reuses ProjectionSource rather than inventing new
# schema (see docs/dev/atlas_inpaint_layers_design.md §2) — the viewport's
# per-source projection material already does everything needed; these nodes
# are orchestration only. Masking/inpainting itself is NOT implemented here —
# it's delegated to external ComfyUI node packs wired into the graph
# (Acly/comfyui-inpaint-nodes, GPL-3.0; scraed/LanPaint, optional generative
# tier for hard disocclusions) — see INSTALL.md's "Optional Inpaint
# Integration" section. Graph-level composition keeps the GPL boundary clean:
# no inpainting/segmentation code lives in atlas_camera.
# ---------------------------------------------------------------------------

class AtlasDepthLayerMask:
    """One depth band -> (layer_mask, occlusion_mask). Composable: instantiate
    once per background layer you plan to clean-plate.

    ``layer_mask`` is 1 where a pixel's *metric* depth falls in
    ``[near, far]`` — this band's own pixels. ``occlusion_mask`` is 1 where a
    pixel is NEARER than ``near`` — i.e. everything that occludes this band —
    feed it into `INPAINT_ExpandMask` (grow ~16-32) then
    `INPAINT_InpaintWithModel` to build this layer's clean plate.

    ``near_m``/``far_m`` (0 = unset) give explicit metric bounds; when unset,
    ``near_pct``/``far_pct`` (0..1) fall back to percentiles over the valid
    (non-sky) metric depth distribution. Metric depth uses the same
    ground-scale path `AtlasDeriveReliefMesh` uses
    (`relief_mesh.estimate_ground_scale`), so bands are consistent with the
    geometry `AtlasCleanPlateLayer` builds from the identical band settings —
    the two nodes share `_resolve_depth_band` internally so their bands can't
    drift apart; always pass matching near/far/pct values to both.

    ``hole_mask`` (opt-in via ``compute_hole_mask``) is a THIRD, independent
    signal: this band's mesh's own discarded hole/tear data
    (`ReliefMesh.hole_mask`) - white wherever this layer's relief mesh will
    show black under Project (sky/invalid depth/silhouette tear), regardless
    of whether that pixel is nearer or farther than the band. `occlusion_mask`
    only answers "is something nearer in the way"; it's blind to a tear
    *inside* the band itself (e.g. a noisy-depth patch or a silhouette edge
    on the subject). Computing it here - rather than only reading it off
    `AtlasCleanPlateLayer` afterward - is what lets it drive the inpaint step
    instead of just reporting on it after the fact; it necessarily duplicates
    `AtlasCleanPlateLayer`'s own later mesh build for the same band (that
    node's mesh can only be built once `plate_image` already exists), which
    is why it's off by default. Not auto-combined into `occlusion_mask` -
    union them explicitly with a mask-max node before `INPAINT_ExpandMask`
    if you want both signals to drive inpainting, same pattern as
    `AtlasOcclusionMask`'s separate `occlusion_mask`/`coverage_mask`.
    Requires `relief_grid`/`depth_edge_rel` matching whatever
    `AtlasCleanPlateLayer` will use downstream for the two to agree.
    """
    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("layer_mask", "occlusion_mask", "hole_mask")
    FUNCTION = "generate"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "near_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Band near edge in metres. 0 = auto (use near_pct percentile)."}),
                "far_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Band far edge in metres. 0 = auto (use far_pct percentile)."}),
                "near_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used when near_m==0: percentile of valid metric depth. LOWER = "
                               "smaller/closer near threshold = TIGHTER occlusion (isolates just "
                               "the true near-camera foreground). Higher near_pct occludes MORE of "
                               "the frame, not less — try 0.05-0.15 for a typical foreground object; "
                               "a photo with one large nearby structure filling the frame may need "
                               "an even lower value."}),
                "far_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used when far_m==0: percentile of valid metric depth. "
                               "0 means no upper bound (+inf), not a degenerate empty band."}),
                "feather_px": ("INT", {"default": 4, "min": 0, "max": 64,
                    "tooltip": "Dilate occlusion_mask's edge by this many pixels — a small "
                               "safety margin on top of whatever grow INPAINT_ExpandMask "
                               "applies downstream."}),
                "compute_hole_mask": ("BOOLEAN", {"default": False,
                    "tooltip": "Build this band's own relief mesh (same as AtlasCleanPlateLayer "
                               "will do later) to derive hole_mask - the mesh's real tear/sky "
                               "hole data, not a depth-band heuristic. Off by default: this is "
                               "a real (duplicate) mesh build, not free like the other two masks."}),
                "relief_grid": ("INT", {"default": 384, "min": 16, "max": 4096,
                    "tooltip": "Only used when compute_hole_mask=True. MUST match the "
                               "AtlasCleanPlateLayer call downstream for hole_mask to reflect "
                               "the actual final mesh (default 384 = the band-layer calibration)."}),
                "depth_edge_rel": ("FLOAT", {"default": 1.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Only used when compute_hole_mask=True. MUST match the "
                               "AtlasCleanPlateLayer call downstream for hole_mask to reflect "
                               "the actual final mesh (default 1.5 = the band-layer calibration)."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional external exclusion (e.g. a real sky segmentation from "
                               "SAM/RMBG) ORed on top of the internal sky heuristic - never "
                               "replaces it, only excludes more. Affects layer_mask/occlusion_mask "
                               "(excluded pixels can't belong to any band) AND hole_mask when "
                               "compute_hole_mask=True. Any resolution - resized to match depth."}),
                "fill_occluded": ("BOOLEAN", {"default": False,
                    "tooltip": "Only used when compute_hole_mask=True. MUST match the "
                               "AtlasCleanPlateLayer setting downstream for hole_mask to reflect "
                               "the actual final mesh - when the layer will diffusion-fill the "
                               "occluder footprint, that footprint is no longer a hole here "
                               "either."}),
            },
        }

    def generate(self, solve, depth, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5, feather_px=4,
                 compute_hole_mask=False, relief_grid=384, depth_edge_rel=1.5, exclude_mask=None,
                 fill_occluded=False):
        np = _require_numpy()
        torch = _require_torch()

        setup = _metric_depth_and_validity(solve, depth, exclude_mask=exclude_mask)
        if setup is None:
            h, w = int(depth.image_height), int(depth.image_width)
            zero = torch.zeros(1, h, w, dtype=torch.float32)
            return (zero, zero.clone(), zero.clone())
        metric, valid = setup.metric, setup.valid

        near, far = _resolve_depth_band(metric, valid, near_m, far_m, near_pct, far_pct)

        layer_mask = valid & (metric >= near) & (metric <= far)
        occlusion_mask = valid & (metric < near)

        hole_mask_arr = np.zeros_like(metric, dtype=np.float32)
        if compute_hole_mask:
            from atlas_camera.core.relief_mesh import build_relief_mesh
            fill = (valid & (metric < near)) if fill_occluded else None
            mesh = build_relief_mesh(
                setup.depth_map, view_matrix=setup.extr.camera_view_matrix,
                fx=setup.fx, fy=setup.fy, cx=setup.cx, cy=setup.cy,
                grid_long_edge=int(relief_grid), depth_edge_rel=float(depth_edge_rel),
                scale=setup.scale, horizon_y=setup.horizon_y,
                band_min_m=near, band_max_m=(None if far == float("inf") else far),
                exclude_mask=setup.exclude_mask, fill_mask=fill,
                apply_sky_heuristic=setup.exclude_mask is None)
            # No edge overhang here, deliberately: the layer's mesh only
            # overhangs when embed_matte is on (this node can't know that),
            # and a PESSIMISTIC hole_mask (a couple of boundary cells extra)
            # over-inpaints safely, while an optimistic one under-inpaints.
            hole_mask_arr = mesh.hole_mask.astype(np.float32)

        if feather_px > 0 and occlusion_mask.any():
            grown = occlusion_mask.copy()
            for _ in range(int(feather_px)):
                # Explicit zero-padded shifts, NOT np.roll — np.roll wraps
                # around image borders, which would bleed occlusion from one
                # edge (e.g. a foreground object touching the bottom of the
                # frame, the common case) onto the opposite edge.
                up = np.zeros_like(grown); up[:-1, :] = grown[1:, :]
                down = np.zeros_like(grown); down[1:, :] = grown[:-1, :]
                left = np.zeros_like(grown); left[:, :-1] = grown[:, 1:]
                right = np.zeros_like(grown); right[:, 1:] = grown[:, :-1]
                grown = grown | up | down | left | right
            occlusion_mask = grown

        layer_t = torch.from_numpy(layer_mask.astype(np.float32)).unsqueeze(0)
        occ_t = torch.from_numpy(occlusion_mask.astype(np.float32)).unsqueeze(0)
        hole_t = torch.from_numpy(hole_mask_arr).unsqueeze(0)
        return (layer_t, occ_t, hole_t)


class AtlasCleanPlateLayer:
    """Inpainted clean plate + depth band -> append a ProjectionSource.

    Behaves like `AtlasAddPatchView` minus the orbit: the camera is the
    PRIMARY camera UNCHANGED (same intrinsics/extrinsics — no
    `camera_math.orbit_camera` call anywhere here), since a clean-plate layer
    is a same-camera plate, not a novel angle. This is the whole
    simplification vs. patch views — no angle calibration needed.

    Builds this band's own relief mesh from `depth`, clipped to
    `[near, far]` metres (`relief_mesh.build_relief_mesh`'s `band_min_m`/
    `band_max_m`) so out-of-band pixels become holes — each layer's mesh only
    ever contains its own band, so overlapping layers never fight over the
    same texels; from Camera View they reassemble exactly, and on orbit/dolly
    they separate in parallax. `near_m`/`far_m`/`near_pct`/`far_pct` MUST
    match the `AtlasDepthLayerMask` call that produced `plate_image`'s
    inpaint mask — both nodes share `_resolve_depth_band` so passing the same
    values keeps them in lockstep.

    Chain one per layer (front-to-back or back-to-front; `priority` decides
    overlap, higher wins). The frontmost layer typically needs no inpainting
    at all (wire in the original photo) since nothing occludes it.

    Caveat (be honest about the ceiling): inpaint quality is only as good as
    the external inpaint model. LaMa/MAT (`Acly/comfyui-inpaint-nodes`)
    continue texture (walls, ground, foliage, sky) excellently but smear on
    complex disocclusions (e.g. a face fully hidden behind a person) — route
    those layers through a LanPaint/SDXL generative pass instead. Band
    boundaries are also only as good as monocular depth; expose `near_m`/
    `far_m` for manual metric control on troublesome scenes.

    ``hole_mask`` surfaces this layer's own mesh's discarded hole/tear data
    (`ReliefMesh.hole_mask`) - a post-hoc QA signal for whether `plate_image`
    actually covers everywhere this layer will show black under Project.
    Computed from the same `build_relief_mesh` call this node already makes,
    so it's free - but it necessarily runs AFTER inpainting already produced
    `plate_image`, so it can't drive the inpaint step itself. For that, see
    `AtlasDepthLayerMask`'s own optional `compute_hole_mask`.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "MASK", "MASK")
    RETURN_NAMES = ("solve", "hole_mask", "extend_mask")
    FUNCTION = "add_layer"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
                "plate_image": ("IMAGE",),
            },
            "optional": {
                "near_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "MUST match the AtlasDepthLayerMask band that produced plate_image."}),
                "far_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "MUST match the AtlasDepthLayerMask band that produced plate_image."}),
                "near_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Must resolve to the same band as the AtlasDepthLayerMask that "
                               "produced plate_image (both call the shared _resolve_depth_band "
                               "helper, so identical near_m/far_m/near_pct/far_pct here and there "
                               "always agree). LOWER near_pct = tighter occlusion, not looser — "
                               "see AtlasDepthLayerMask's near_pct tooltip for the worked example."}),
                "far_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Must resolve to the same band as the AtlasDepthLayerMask that "
                               "produced plate_image. 0 means no upper bound (+inf)."}),
                "name": ("STRING", {"default": "layer"}),
                "priority": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blend priority among layers (higher wins) — nearer bands should "
                               "get a higher priority. Ordering is by depth + priority, never "
                               "facing angle (clean-plate sources paint head-on AND grazing, "
                               "unlike multi-angle patches)."}),
                "plate_ref": ("ATLAS_PLATE_REF", {
                    "tooltip": "Optional registered final clean-plate reference. Browser still uses image_b64 preview; exporters use this for EXR/float-safe handoff."}),
                "relief_grid": ("INT", {"default": 384, "min": 16, "max": 4096,
                    "tooltip": "Band-clipped meshes tear at band boundaries ON TOP OF normal "
                               "silhouette tearing, so per-layer meshes want more density than "
                               "the generic 128 default - 384/1.5 is the empirically-calibrated "
                               "band-layer setting (hangar + monument valley)."}),
                "depth_edge_rel": ("FLOAT", {"default": 1.5, "min": 0.05, "max": 5.0, "step": 0.05,
                    "tooltip": "Looser than the generic 0.5: safe WITHIN a band because the band "
                               "clip already bounds the mesh's depth range."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Optional external exclusion (e.g. a real sky segmentation from "
                               "SAM/RMBG). When connected it REPLACES the internal sky heuristic "
                               "(which otherwise eats tall geometry above the horizon) - should match "
                               "whatever was passed to the AtlasDepthLayerMask call that produced "
                               "plate_image, for hole_mask/band resolution to stay in lockstep."}),
                "fill_occluded": ("BOOLEAN", {"default": False,
                    "tooltip": "Diffusion-fill this band's mesh across the foreground occluder's "
                               "footprint (the band clip otherwise leaves a hole exactly there) so "
                               "the INPAINTED plate content lands on real geometry instead of a "
                               "hole - the disocclusion 'shadow ray' mesh. Synthesized depth is a "
                               "smooth interpolation of the surrounding background, reported in "
                               "metadata as n_filled_cells. Excluded (sky) regions are never "
                               "filled."}),
                "embed_matte": ("BOOLEAN", {"default": False,
                    "tooltip": "Embed a full-resolution per-pixel edge matte on this layer "
                               "(ProjectionSource.mask_b64) - the projection shader then cuts the "
                               "TRUE band silhouette per-pixel instead of the mesh's blocky "
                               "grid-resolution tear edge, and the Nuke layers export writes it "
                               "into the plate's alpha. Auto-computed from this band (in-band "
                               "pixels, plus the filled occluder footprint when fill_occluded is "
                               "on, minus exclude_mask); wire layer_matte to override with a "
                               "hand/SAM matte."}),
                "layer_matte": ("MASK", {
                    "tooltip": "Optional explicit edge matte (overrides the auto-computed band "
                               "matte when embed_matte is on) - e.g. a SAM segmentation of this "
                               "layer's subject for a crisper edge than depth banding gives."}),
                "edge_extend_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4,
                    "tooltip": "Deterministic edge-extend for THIS layer, same trick as the sky "
                               "dome's: smear the plate's colors past the matte edge by this many "
                               "pixels (quarter-res neighbor propagation — NOT an inpaint), dilate "
                               "the embedded matte to expose the extension on disocclusion, and "
                               "grow the mesh's boundary skirt to receive it. The invented region "
                               "is reported on the extend_mask output AND exported to Nuke/Maya as "
                               "{layer}_extend_matte.png so it can be processed downstream "
                               "(regrain, blur, replace). Smeared pixels are plausible only for "
                               "narrow slivers — large reveals still want a real inpainted plate. "
                               "Turns on embed_matte implicitly (an extension needs a matte edge)."}),
                "skirt_bevel": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Bevel the mesh's boundary skirt AWAY from the camera, as a slope "
                               "in local cell units: 1.0 recedes one cell per extension ring (a "
                               "45° skirt), 0 = today's flat skirt. Physically motivated — an "
                               "occluded surface continues away from the camera behind its "
                               "silhouette, so a receding bevel is the least-wrong geometry at a "
                               "tear edge. Try 1.0–2.0 with edge_extend_px: the smeared colors "
                               "land on the receding skirt and the extend matte marks them for "
                               "regrain."}),
                "frame_outpaint_px": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8,
                    "tooltip": "Outpaint THIS layer past the FRAME edges by this many pixels — "
                               "the same per-source widened-camera trick the sky dome uses: the "
                               "plate canvas is padded edge-replicated, this source gets its OWN "
                               "camera with shifted cx/cy and grown W/H (the primary solve and "
                               "every other layer are untouched), and the band mesh extends past "
                               "the original frustum to carry it. Closes the frame-edge reveal "
                               "that 🧭 Safe Zone measurements show is the binding constraint on "
                               "wide scenes (ground layers used to end exactly at the photo "
                               "boundary). The ring is INVENTED pixels: it lands in extend_mask / "
                               "{layer}_extend_matte.png for downstream regrain, and turns on "
                               "embed_matte implicitly. 0 = off."}),
                "exclude_choke_cells": ("INT", {"default": 2, "min": 0, "max": 16,
                    "tooltip": "Choke-and-reskirt against the exclude_mask edge: segmentation "
                               "and depth edges never align exactly, leaving a ribbon of cells "
                               "the mask calls rock but whose depth IS sky — they back-project "
                               "high above the real silhouette as a jagged floating band. This "
                               "erodes the layer N grid cells away from the exclusion, then the "
                               "boundary skirt regrows the ring with clean neighbor depth: same "
                               "coverage, geometry hugging the true surface. Raise for sloppier "
                               "segmentation masks; 0 disables."}),
            },
        }

    def add_layer(self, solve, depth, plate_image, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5,
                  name="layer", priority=0.0, plate_ref=None, relief_grid=384, depth_edge_rel=1.5,
                  exclude_mask=None, fill_occluded=False, embed_matte=False, layer_matte=None,
                  edge_extend_px=0, skirt_bevel=0.0, frame_outpaint_px=0,
                  exclude_choke_cells=2):
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_relief_mesh
        from atlas_camera.core.schema import (
            AtlasIntrinsics,
            AtlasPlateRef,
            LatentCamera,
            ProjectionSource,
        )

        torch = _require_torch()
        np = _require_numpy()

        setup = _metric_depth_and_validity(solve, depth, exclude_mask=exclude_mask)
        if setup is None:
            h, w = int(depth.image_height), int(depth.image_width)
            blank = torch.zeros(1, h, w, dtype=torch.float32)
            return (solve, blank, blank)
        # An extension needs a matte edge to extend past.
        if edge_extend_px and int(edge_extend_px) > 0:
            embed_matte = True
        fx, fy, cx, cy = setup.fx, setup.fy, setup.cx, setup.cy
        extr, depth_map = setup.extr, setup.depth_map
        scale, horizon_y = setup.scale, setup.horizon_y
        near, far = _resolve_depth_band(setup.metric, setup.valid, near_m, far_m, near_pct, far_pct)

        # Frame outpaint (the sky dome's proven widened-camera trick, applied
        # per band layer): pad EVERYTHING edge-replicated into one padded
        # pixel space — depth (so the mesh extends past the original frustum),
        # the band arrays, the plate, and this source's OWN intrinsics
        # (cx/cy + P, W/H + 2P; pose and every other layer untouched). Closes
        # the frame-edge reveal 🧭 Safe Zone measures as the binding
        # constraint on wide scenes. The ring is invented → matted + declared.
        pad = max(0, int(frame_outpaint_px))
        if pad:
            embed_matte = True
        fill = (setup.valid & (setup.metric < near)) if fill_occluded else None
        depth_m, metric_m, valid_m = depth_map, setup.metric, setup.valid
        exclude_m, fill_m = setup.exclude_mask, fill
        if exclude_m is not None:
            # Border-flood the segmentation (see _flood_mask_to_frame_borders)
            # and re-derive validity from the healed mask: the faded border
            # rows carry sky depth that otherwise builds a floating ring at
            # the top of frame (found live — 86% of the bg layer's
            # above-skyline vertices projected into the top outpaint ring).
            exclude_m = _flood_mask_to_frame_borders(exclude_m)
            valid_m = valid_m & ~exclude_m
        cx_m, cy_m, horizon_m = cx, cy, horizon_y
        Hp, Wp = setup.height, setup.width
        if pad:
            depth_m = np.pad(depth_map, pad, mode="edge")
            metric_m = np.pad(setup.metric, pad, mode="edge")
            valid_m = np.pad(setup.valid, pad, mode="edge")
            if exclude_m is not None:
                exclude_m = np.pad(exclude_m, pad, mode="edge")
            if fill_m is not None:
                fill_m = np.pad(fill_m, pad, mode="edge")
            cx_m, cy_m = cx + pad, cy + pad
            if horizon_m is not None:
                horizon_m = float(horizon_m) + pad
            Hp, Wp = Hp + 2 * pad, Wp + 2 * pad

        choke = int(exclude_choke_cells) if exclude_m is not None else 0
        overhang_cells = 0
        if embed_matte:
            overhang_cells = 2
            if edge_extend_px and int(edge_extend_px) > 0:
                cell_px = max(1, int(round(max(Hp, Wp) / max(int(relief_grid), 2))))
                overhang_cells = 2 + int(np.ceil(int(edge_extend_px) / cell_px))
            # The skirt must regrow the choked ring fully before extending.
            overhang_cells += choke
        mesh = build_relief_mesh(
            depth_m, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx_m, cy=cy_m,
            grid_long_edge=int(relief_grid), depth_edge_rel=float(depth_edge_rel),
            scale=scale, horizon_y=horizon_m,
            band_min_m=near, band_max_m=(None if far == float("inf") else far),
            exclude_mask=exclude_m, fill_mask=fill_m,
            apply_sky_heuristic=exclude_m is None,
            overhang_bevel_rel=float(skirt_bevel),
            exclude_choke_cells=choke,
            edge_overhang_cells=overhang_cells)
        patch_geom = [relief_mesh_primitive(mesh, name=f"{name}_relief_mesh")]

        # This source's OWN camera: same pose, widened intrinsics when
        # outpainted (per-ProjectionSource cameras make this free — exactly
        # the sky dome's pattern).
        src_camera = solve.camera
        if pad:
            src_camera = LatentCamera(
                intrinsics=AtlasIntrinsics(
                    image_width=Wp, image_height=Hp,
                    focal_length_mm=solve.camera.intrinsics.focal_length_mm,
                    sensor_width_mm=solve.camera.intrinsics.sensor_width_mm,
                    fx_px=fx, fy_px=fy, cx_px=cx_m, cy_px=cy_m),
                extrinsics=extr)

        # Per-layer edge extend (same deterministic trick as the sky dome's):
        # computed on the auto/explicit matte below, so the plate encode is
        # deferred until the matte exists.
        extended_plate = None
        extend_region = None

        image_b64 = ""
        try:
            if pad:
                plate_np0 = np.asarray(
                    _image_tensor_to_pil(plate_image).convert("RGB"), dtype=np.float32)
                if plate_np0.shape[:2] != (setup.height, setup.width):
                    PILImage = _require_pil()
                    plate_np0 = np.asarray(
                        PILImage.fromarray(plate_np0.astype("uint8")).resize(
                            (setup.width, setup.height)), dtype=np.float32)
                plate_padded = np.pad(plate_np0, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                PILImage = _require_pil()
                pil = PILImage.fromarray(plate_padded.clip(0, 255).astype("uint8"), mode="RGB")
            else:
                plate_padded = None
                pil = _image_tensor_to_pil(plate_image)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            plate_padded = None

        source = ProjectionSource(
            camera=src_camera,  # primary POSE unchanged; intrinsics widened when outpainted
            name=name,
            image_b64=image_b64,
            plate_ref=plate_ref if isinstance(plate_ref, AtlasPlateRef) else AtlasPlateRef.from_dict(plate_ref),
            proxy_geometry=patch_geom,
            priority=float(priority),
            metadata={
                "projection_mode": "clean_plate",
                "source": "inpaint_layer",
                "near_m": None if near <= 0 else float(near),
                "far_m": None if far == float("inf") else float(far),
                "ground_scale": scale,
                "n_vertices": mesh.stats.get("n_vertices"),
                "n_faces": mesh.stats.get("n_faces"),
                "n_filled_cells": mesh.stats.get("n_filled_cells", 0),
                "skirt_bevel": float(skirt_bevel),
            },
        )

        # Optional per-pixel edge matte: geometry tears at grid-quad
        # resolution; the matte cuts the true band silhouette in the shader.
        # Everything below works in the (possibly padded) plate pixel space.
        if embed_matte:
            if layer_matte is not None:
                matte = _resolve_exclude_mask(layer_matte, setup.height, setup.width)
                if pad:
                    matte = np.pad(matte, pad, mode="edge")
            else:
                matte = valid_m & (metric_m <= far)
                if not fill_occluded:
                    # Without disocclusion fill the occluder footprint has no
                    # geometry, so the matte matches the band exactly; with it,
                    # the filled footprint must stay INSIDE the matte (the
                    # inpainted plate content lives there).
                    matte = matte & (metric_m >= near)
            # Real (photographed) pixels: the interior only — the outpaint
            # ring is invented even where the matte covers it.
            if pad:
                original_matte = np.zeros_like(matte)
                original_matte[pad:-pad, pad:-pad] = matte[pad:-pad, pad:-pad]
                source.metadata["frame_outpaint_px"] = pad
            else:
                original_matte = matte
            if edge_extend_px and int(edge_extend_px) > 0:
                if plate_padded is not None:
                    plate_np = plate_padded
                else:
                    plate_np = np.asarray(_image_tensor_to_pil(plate_image).convert("RGB"),
                                          dtype=np.float32)
                    if plate_np.shape[:2] != matte.shape:
                        PILImage = _require_pil()
                        plate_np = np.asarray(
                            PILImage.fromarray(plate_np.astype("uint8")).resize(
                                (matte.shape[1], matte.shape[0])), dtype=np.float32)
                extended_plate, matte = _extend_edge_colors(
                    plate_np, matte, int(edge_extend_px))
                source.metadata["edge_extend_px"] = int(edge_extend_px)
                # Re-encode the plate WITH the extension baked in.
                try:
                    PILImage = _require_pil()
                    pil = PILImage.fromarray(
                        extended_plate.clip(0, 255).astype("uint8"), mode="RGB")
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=88)
                    source.image_b64 = ("data:image/jpeg;base64,"
                                        + base64.b64encode(buf.getvalue()).decode("ascii"))
                except Exception:
                    pass
            # The excluded region (sky) is a hard boundary for the matte too:
            # dilation/smear exposure must not paint this layer over the sky
            # layer's territory (same rule as the mesh skirt).
            if exclude_m is not None:
                matte = matte & ~exclude_m
            # Invented pixels = smears + the outpaint ring (whatever the final
            # matte exposes beyond real photographed content).
            extend_region = matte & ~original_matte
            if extend_region.any():
                source.extend_mask_b64 = _mask_to_b64_png(extend_region) or None
            else:
                extend_region = None
            source.mask_b64 = _mask_to_b64_png(matte) or None

        out = copy.deepcopy(solve)
        out.projection_sources.append(source)
        # hole_mask output stays in the ORIGINAL plate frame (crop the pad) so
        # downstream previews line up with the source photo; extend_mask stays
        # in the padded PLATE frame (it describes the exported plate's pixels)
        # — both matching the sky dome's conventions.
        hole = mesh.hole_mask[pad:-pad, pad:-pad] if pad else mesh.hole_mask
        hole_t = torch.from_numpy(hole.astype(np.float32)).unsqueeze(0)
        if extend_region is not None:
            ext_t = torch.from_numpy(extend_region.astype(np.float32)).unsqueeze(0)
        else:
            ext_t = torch.zeros(1, Hp, Wp, dtype=torch.float32)
        return (out, hole_t, ext_t)


class AtlasSkyDomeLayer:
    """Same-camera sky clean-plate, projected onto a simple constant-depth
    card instead of a depth-following relief mesh — the standard DMP move
    (Nuke and similar): separate sky from real geometry so it clean-plates
    and projects without fighting noisy monocular sky depth, or tearing at a
    boundary that's really just "where the segmentation mask ends," not a
    genuine depth discontinuity.

    Unlike `AtlasCleanPlateLayer` (which clips a REAL relief mesh to a
    metric depth band), this node ignores actual depth VALUES entirely for
    the card's own shape — `sky_mask` (from a real segmenter, e.g.
    ComfyUI-RMBG's SAM3 Segmentation prompted with "sky") is the sole
    authority on which pixels belong to it. `depth`/`solve` are still
    required, purely for camera intrinsics/extrinsics via the same shared
    `_metric_depth_and_validity` setup `AtlasDepthLayerMask`/
    `AtlasCleanPlateLayer` use — the real depth array itself is never read.

    Geometrically this is a flat card at a constant forward-Z depth
    (`radius_m`), the same convention `build_relief_mesh` uses everywhere
    else (and the same convention every extractor's own `projection_backdrop`
    plane already uses) — not a literal sphere/hemisphere. For any normal
    camera FOV this is visually equivalent to a dome; a true sphere would
    need different (unreused) triangulation math for real benefit only at
    extreme wide-angle/360 coverage. See `relief_mesh.build_sky_dome_mesh`.

    `plate_image` should be a CLEAN sky plate: invert `sky_mask` (ComfyUI's
    `InvertMask`, or SAM3's own `invert_output`) to get the mask of
    everything occluding the sky, feed that through `INPAINT_ExpandMask` ->
    `INPAINT_InpaintWithModel` on the original photo, and wire the result
    here — the same external-inpaint chain the other inpaint-layers nodes
    use (see INSTALL.md's "Optional Inpaint Integration").

    Camera is the PRIMARY camera UNCHANGED, same as `AtlasCleanPlateLayer` —
    no orbit, since this is a same-camera plate. Chain alongside
    `AtlasCleanPlateLayer`/`AtlasDeriveReliefMesh` layers; default `priority`
    is low (-10) since `sky_mask` makes this layer spatially exclusive from
    ground/foreground layers in practice — priority only matters if masks
    overlap. `hole_mask` mirrors the other inpaint-layers nodes: white where
    `sky_mask`'s own boundary didn't survive onto the grid (QA signal, not
    something to feed back into inpainting).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "MASK", "MASK")
    RETURN_NAMES = ("solve", "hole_mask", "extend_mask")
    FUNCTION = "add_layer"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
                "sky_mask": ("MASK", {
                    "tooltip": "Real segmentation marking sky pixels (e.g. ComfyUI-RMBG's SAM3 "
                               "Segmentation prompted with 'sky'). Sole authority on this layer's "
                               "shape — real depth values are never read for the card's geometry."}),
                "plate_image": ("IMAGE", {
                    "tooltip": "A CLEAN sky plate — invert sky_mask, run it through an external "
                               "inpaint chain (INPAINT_ExpandMask -> INPAINT_InpaintWithModel) on "
                               "the original photo, wire the result here."}),
            },
            "optional": {
                "radius_m": ("FLOAT", {"default": 300.0, "min": 1.0, "max": 100000.0, "step": 1.0,
                    "tooltip": "Card distance in metres (forward-Z, same convention as every "
                               "other relief mesh here) — should comfortably exceed the scene's "
                               "own derived backdrop distance (AtlasDeriveReliefMesh's backdrop, "
                               "or the Info HUD's scene depth) so it never intersects real "
                               "geometry. Distance doesn't affect appearance (texel assignment is "
                               "by ray, not depth) — only how far you can dolly before it reveals "
                               "itself as flat, which at typical FOV is a very long way."}),
                "relief_grid": ("INT", {"default": 96, "min": 16, "max": 4096,
                    "tooltip": "Card mesh density (long-edge grid columns). A flat, constant-depth "
                               "card needs far less density than real geometry — default is lower "
                               "than AtlasDeriveReliefMesh's."}),
                "name": ("STRING", {"default": "sky"}),
                "priority": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blend priority among layers (higher wins). Low by default since "
                               "sky_mask makes this layer spatially exclusive from ground/"
                               "foreground layers in practice."}),
                "plate_ref": ("ATLAS_PLATE_REF", {
                    "tooltip": "Optional registered final clean-plate reference. Browser still uses image_b64 preview; exporters use this for EXR/float-safe handoff."}),
                "edge_extend_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 4,
                    "tooltip": "Deterministic edge-extend (the classic Nuke premult->dilate trick, "
                               "NOT an inpaint): smears the sky's edge colors this many pixels past "
                               "the silhouette into the plate, dilates the matte to match, and "
                               "overhangs the dome mesh accordingly - so orbiting reveals plausible "
                               "gradient sky behind foreground silhouettes instead of black slivers. "
                               "Enough for narrow disocclusions of smooth sky; large structured "
                               "reveals (clouds behind a building) still want a real LaMa/inpaint "
                               "chain on plate_image. 0 = off."}),
                "frame_outpaint_px": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8,
                    "tooltip": "Outpaint the sky past the FRAME edges by this many pixels (edge-"
                               "replicated then smeared, same deterministic trick as "
                               "edge_extend_px) so a small orbit/pan doesn't slam into the plate "
                               "boundary. The sky source gets its own enlarged canvas + widened "
                               "intrinsics (cx/cy shifted, W/H grown), and the dome mesh extends "
                               "past the original frustum to carry it. Purely this layer's "
                               "camera - the primary solve and every other layer are untouched. "
                               "0 = off."}),
            },
        }

    def add_layer(self, solve, depth, sky_mask, plate_image, radius_m=300.0, relief_grid=96,
                  name="sky", priority=-10.0, plate_ref=None, edge_extend_px=48,
                  frame_outpaint_px=64):
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive
        from atlas_camera.core.relief_mesh import build_sky_dome_mesh
        from atlas_camera.core.schema import AtlasIntrinsics, AtlasPlateRef, LatentCamera, ProjectionSource

        torch = _require_torch()
        np = _require_numpy()

        setup = _metric_depth_and_validity(solve, depth)
        if setup is None:
            h, w = int(depth.image_height), int(depth.image_width)
            blank = torch.zeros(1, h, w, dtype=torch.float32)
            return (solve, blank, blank)

        mask_arr = _resolve_exclude_mask(sky_mask, setup.height, setup.width)
        if mask_arr is not None:
            # Heal the segmenter's border fade (see _flood_mask_to_frame_borders):
            # without it the card's outpaint ring inherits a mostly-false top
            # row and doesn't cover above the skyline.
            mask_arr = _flood_mask_to_frame_borders(mask_arr)
        if mask_arr is None or not mask_arr.any():
            blank = torch.zeros(1, setup.height, setup.width, dtype=torch.float32)
            return (solve, blank, blank)

        # Everything below works at PLATE resolution in ONE padded pixel space:
        # frame outpaint (pad the canvas, shift cx/cy - the sky source gets
        # its own wider-FOV camera so a small orbit never hits the plate
        # boundary), then the silhouette edge-extend, then the dome mesh -
        # all sharing the same coordinates, so plate/matte/mesh stay aligned.
        plate_np = (plate_image[0].cpu().numpy() * 255.0)
        Hp, Wp = plate_np.shape[:2]
        if (Hp, Wp) != mask_arr.shape:
            from atlas_camera.core.solver import _resize_depth
            m = _resize_depth(mask_arr.astype(np.float64), Wp, Hp) > 0.5
        else:
            m = mask_arr
        sx, sy = Wp / float(setup.width), Hp / float(setup.height)
        fx_p, fy_p = setup.fx * sx, setup.fy * sy
        cx_p, cy_p = setup.cx * sx, setup.cy * sy

        pad = max(0, int(frame_outpaint_px))
        if pad:
            plate_np = np.pad(plate_np, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            m = np.pad(m, pad, mode="edge")
            cx_p += pad
            cy_p += pad

        matte_arr = m
        plate_arr = plate_np if pad else None  # padded canvas always re-encodes
        step = max(1, int(round(max(m.shape) / max(int(relief_grid), 2))))
        overhang_cells = 2
        # Invented pixels: the frame-outpaint ring (edge-replicated pad) is
        # synthetic wherever the matte exposes it, and the silhouette extend
        # below adds more. Both land in extend_mask for downstream regrain.
        original_matte = np.zeros_like(m)
        if pad:
            original_matte[pad:-pad, pad:-pad] = m[pad:-pad, pad:-pad]
        else:
            original_matte[:] = m
        if edge_extend_px and int(edge_extend_px) > 0:
            plate_arr, matte_arr = _extend_edge_colors(plate_np, m, int(edge_extend_px))
            overhang_cells = 2 + int(np.ceil(int(edge_extend_px) / step))
        extend_region = matte_arr & ~original_matte

        mesh = build_sky_dome_mesh(
            m, view_matrix=setup.extr.camera_view_matrix,
            fx=fx_p, fy=fy_p, cx=cx_p, cy=cy_p,
            radius_m=float(radius_m), grid_long_edge=int(relief_grid),
            edge_overhang_cells=overhang_cells)
        patch_geom = [relief_mesh_primitive(mesh, name=f"{name}_dome_mesh")]

        # This source's OWN camera: same pose as the primary (no orbit), but
        # with the padded/rescaled intrinsics so the outpainted canvas is
        # real texture space for the projection shader and the Nuke export
        # (each ProjectionSource carries its own camera by design).
        src_camera = solve.camera
        if pad or (Hp, Wp) != (setup.height, setup.width):
            src_camera = LatentCamera(
                intrinsics=AtlasIntrinsics(
                    image_width=Wp + 2 * pad, image_height=Hp + 2 * pad,
                    sensor_width_mm=solve.camera.intrinsics.sensor_width_mm,
                    fx_px=fx_p, fy_px=fy_p, cx_px=cx_p, cy_px=cy_p),
                extrinsics=setup.extr)

        image_b64 = ""
        try:
            if plate_arr is not None:
                PILImage = _require_pil()
                pil = PILImage.fromarray(plate_arr.clip(0, 255).astype("uint8"), mode="RGB")
            else:
                pil = _image_tensor_to_pil(plate_image)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

        source = ProjectionSource(
            camera=src_camera,  # primary POSE unchanged; intrinsics widened when outpainted
            name=name,
            image_b64=image_b64,
            plate_ref=plate_ref if isinstance(plate_ref, AtlasPlateRef) else AtlasPlateRef.from_dict(plate_ref),
            proxy_geometry=patch_geom,
            priority=float(priority),
            # The SAM/segmentation mask IS the perfect full-resolution edge
            # matte for this layer — embed it so the projection shader cuts
            # the true sky silhouette per-pixel instead of the card mesh's
            # grid-resolution staircase edge. With edge_extend_px the matte
            # is the DILATED mask, exposing the smeared extension on
            # disocclusion.
            mask_b64=_mask_to_b64_png(matte_arr) or None,
            extend_mask_b64=_mask_to_b64_png(extend_region) or None,
            metadata={
                "projection_mode": "clean_plate",
                "source": "sky_dome",
                "radius_m": float(radius_m),
                "edge_extend_px": int(edge_extend_px),
                "frame_outpaint_px": pad,
                "n_vertices": mesh.stats.get("n_vertices"),
                "n_faces": mesh.stats.get("n_faces"),
            },
        )

        out = copy.deepcopy(solve)
        out.projection_sources.append(source)
        # hole_mask output stays in the ORIGINAL plate frame (crop the pad) so
        # downstream previews/composites line up with the source photo.
        hole = mesh.hole_mask[pad:pad + Hp, pad:pad + Wp] if pad else mesh.hole_mask
        hole_t = torch.from_numpy(hole.astype(np.float32)).unsqueeze(0)
        # extend_mask output stays in the padded PLATE frame (it describes the
        # exported plate's pixels, unlike hole_mask which previews against the
        # source photo).
        ext_t = torch.from_numpy(extend_region.astype(np.float32)).unsqueeze(0)
        return (out, hole_t, ext_t)


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
    "AtlasRegisterPlate":         AtlasRegisterPlate,
    "AtlasAttachSourcePlate":     AtlasAttachSourcePlate,
    # Track 1 — solve
    "AtlasSolveFromImage":        AtlasSolveFromImage,
    "AtlasLearnedSolveFromImage": AtlasLearnedSolveFromImage,
    "AtlasReferenceScaleSolve":   AtlasReferenceScaleSolve,
    "AtlasVLMScaleCues":          AtlasVLMScaleCues,
    "AtlasAssessImage":           AtlasAssessImage,
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
    "AtlasExportNukeLayers":      AtlasExportNukeLayers,
    "AtlasExportMayaLayers":      AtlasExportMayaLayers,
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
    # Track 7 — inpaint layers
    "AtlasDepthLayerMask":        AtlasDepthLayerMask,
    "AtlasCleanPlateLayer":       AtlasCleanPlateLayer,
    "AtlasSkyDomeLayer":          AtlasSkyDomeLayer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing
    "AtlasLoadImageSolveCamera":  "Atlas Load Image / Solve Camera",
    "AtlasExportReviewPackage":   "Atlas Export Review Package",
    "AtlasExportSolveJSON":       "Atlas Export Solve JSON",
    "AtlasExportMayaReviewScene": "Atlas Export Maya Review Scene",
    "AtlasUSDCameraLoader":       "Atlas USD Camera Loader",
    "AtlasRegisterPlate":         "Atlas Register Plate (Float-Safe) 🎞",
    "AtlasAttachSourcePlate":     "Atlas Attach Source Plate 🎞",
    # Track 1 — solve
    "AtlasSolveFromImage":        "Atlas Solve Camera from Image",
    "AtlasLearnedSolveFromImage": "Atlas Learned Solve (GeoCalib) 🧠",
    "AtlasReferenceScaleSolve":   "Atlas Reference-Object Scale 📏",
    "AtlasAssessImage":           "Atlas Assess Image 🧭",
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
    "AtlasExportNukeLayers":      "Atlas Export Nuke Layers 🎞",
    "AtlasExportMayaLayers":      "Atlas Export Maya Layers 🧊",
    # Track 2 — blockout viewport
    "AtlasViewportControls":      "Atlas Output Desk 🎛",
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
    # Track 7 — inpaint layers
    "AtlasDepthLayerMask":        "Atlas Depth Layer Mask 🎭",
    "AtlasCleanPlateLayer":       "Atlas Clean Plate Layer 🖼",
    "AtlasSkyDomeLayer":          "Atlas Sky Dome Layer ☁",
}
