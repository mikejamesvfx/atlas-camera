"""ComfyUI node library for Atlas Camera."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.solver import solve_from_constraints, solve_still_image
from atlas_camera.exporters.blender_exporter import write_blender_scene_script
from atlas_camera.exporters.nuke_exporter import write_nuke_native_script, write_nuke_projection_script
from atlas_camera.exporters.review_package import build_review_package
from atlas_camera.importers.usd_camera_loader import USDCameraLoader

# Shared depth_model combo choices. APPEND-ONLY: ComfyUI serializes combo VALUES,
# so adding entries is safe; removing/renaming breaks saved workflows.
# DA3 models need the [neural-da3] extra. DA3METRIC converts canonical depth to
# metres using the solve's focal when the node has one (else an assumed
# normal-lens focal — it predicts no intrinsics itself; ground-pinning
# re-normalizes downstream). DA3NESTED is CC BY-NC 4.0 — non-commercial license.
# MoGe-2 (Ruicheng/moge-*) is the MIT-licensed, light-dependency alternative:
# metric depth + predicted normals, fed the solve's focal as fov_x. Needs the
# [moge] extra (`pip install git+https://github.com/microsoft/MoGe.git`).
_DEPTH_MODEL_CHOICES = [
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depth-anything/DA3METRIC-LARGE",
    "depth-anything/DA3MONO-LARGE",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "Ruicheng/moge-2-vitl-normal",
    "Ruicheng/moge-2-vitb-normal",
    "Ruicheng/moge-2-vits-normal",
]

# MoGe `*-normal` checkpoints, largest→smallest — the models that predict surface
# normals (used by AtlasMogeNormals, and the normal-capable subset of the depth
# choices above). ViT-S (35M) is the CPU/MPS-viable one for non-CUDA users; ViT-B
# (104M) a lighter GPU option; ViT-L (331M) the best quality. MIT-licensed,
# auto-downloaded from HuggingFace. APPEND-ONLY (values serialize into workflows).
_MOGE_NORMAL_MODEL_CHOICES = [
    "Ruicheng/moge-2-vitl-normal",
    "Ruicheng/moge-2-vitb-normal",
    "Ruicheng/moge-2-vits-normal",
]

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

def _solve_focal_px_for_image(solve, image):
    """Solved focal expressed in the wired IMAGE tensor's pixel scale, or None.

    Backs the image-only depth nodes' optional ``solve`` input: DA3METRIC's
    canonical→metric conversion prefers the solved focal (focal_source="solve")
    over its assumed normal-lens fallback. The solve's focal comes from
    GeoCalib/VP — independent of whichever depth model the solve node used —
    so it is valid for any depth backend. V2 models ignore focal_px entirely.
    """
    if solve is None:
        return None
    try:
        intr = solve.camera.intrinsics
    except AttributeError:
        return None
    fx = intr.fx_px or 0.0
    if fx <= 0:
        return None
    width = int(intr.image_width or image.shape[2])
    return float(fx) * (int(image.shape[2]) / max(width, 1))


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


def _resolve_raw_hints(focal_widget_mm, sensor_widget_mm, raw_meta):
    """Resolve (focal_hint, sensor_w, sensor_h) from widget values + an
    optionally wired ATLAS_RAW_META (AtlasLoadRAW's RawImportResult).

    Precedence: an explicit widget value (>0 focal / non-default sensor)
    always beats the wired metadata, so an artist override never fights
    the EXIF. sensor_height flows only from raw_meta (no widget exists).
    """
    focal_hint = float(focal_widget_mm) if focal_widget_mm and focal_widget_mm > 0 else None
    sensor_w = float(sensor_widget_mm)
    sensor_h = None
    if raw_meta is not None:
        if focal_hint is None and getattr(raw_meta, "focal_length_mm", None):
            focal_hint = float(raw_meta.focal_length_mm)
        if sensor_w == 36.0 and getattr(raw_meta, "sensor_width_mm", None):
            sensor_w = float(raw_meta.sensor_width_mm)
            if getattr(raw_meta, "sensor_height_mm", None):
                sensor_h = float(raw_meta.sensor_height_mm)
    return focal_hint, sensor_w, sensor_h


def _scale_summary_suffix(solve) -> str:
    """Export-summary warning when the solve's metric scale isn't verified.

    Single source of truth is core.scene_health — never re-derive from
    scale_source ad hoc. Empty string when the scale is trustworthy, so
    healthy summaries are unchanged.
    """
    from atlas_camera.core.scene_health import scale_health
    sh = scale_health(solve)
    if sh.safe_to_export:
        return ""
    return f" | ⚠ scale {sh.status.upper()} — not verified"


def _stamp_raw_provenance(solve, raw_meta):
    """Record where a RAW import's hints came from on the solve (in place)."""
    if raw_meta is None:
        return
    solve.debug_metadata["raw_import"] = {
        "source_path": getattr(raw_meta, "source_path", None),
        "camera_make": getattr(raw_meta, "camera_make", None),
        "camera_model": getattr(raw_meta, "camera_model", None),
        "lens_model": getattr(raw_meta, "lens_model", None),
        "focal_length_mm": getattr(raw_meta, "focal_length_mm", None),
        "sensor_width_mm": getattr(raw_meta, "sensor_width_mm", None),
        "sensor_height_mm": getattr(raw_meta, "sensor_height_mm", None),
        "sensor_source": getattr(raw_meta, "sensor_source", None),
        "undistort_status": getattr(raw_meta, "undistort_status", None),
    }


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


def _b64_png_to_mask(b64: str):
    """Inverse of _mask_to_b64_png: PNG data URI -> (H,W) bool numpy array.
    Used to thread AtlasPredictHiddenGeometry's hidden_mask (stored JSON-safe
    in the patched depth's metadata) into a band layer's ProjectionSource.
    Fails soft to None."""
    try:
        np = _require_numpy()
        PILImage = _require_pil()
        raw = base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)
        arr = np.asarray(PILImage.open(io.BytesIO(raw)).convert("L"))
        return arr > 127
    except Exception:
        return None


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
            # Predicted world-normal relight map (MoGe *-normal), aligned to the
            # recovered frame — the projection shader samples it for the lights.
            "normal_map_b64": getattr(src, "normal_map_b64", None) or "",
            "plate_ref": _plate_ref_to_dict(getattr(src, "plate_ref", None)),
            "priority": float(src.priority),
            "azimuth_deg": float(src.azimuth_deg),
            "elevation_deg": float(src.elevation_deg),
            "projection_mode": (src.metadata or {}).get("projection_mode"),
            # Band metrics (metres) — a finite far_m is the AtlasBoundedBand
            # cutoff on a foreground clean-plate layer; drives the 📏 Band Box
            # overlay (near_m None = 0 = the near plane).
            "near_m": (src.metadata or {}).get("near_m"),
            "far_m": (src.metadata or {}).get("far_m"),
            "band_geometry": (src.metadata or {}).get("band_geometry"),
            # 🩻 hidden-geometry provenance (AtlasPredictHiddenGeometry via a
            # band layer) — drives the viewport's debug tint overlay.
            "hidden_mask_b64": (src.metadata or {}).get("hidden_mask_b64") or "",
            "hidden_backend": (src.metadata or {}).get("hidden_backend") or "",
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
    from atlas_camera.core.scene_health import scale_health
    sh = scale_health(solve)
    camera_meta = {
        "confidence": float(getattr(solve, "confidence", 0.0) or 0.0),
        "source_method": getattr(solve, "source_method", None),
        "scale_source": (solve.debug_metadata or {}).get("scale_source"),
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "fov_h_deg": fov_h_deg,
        "camera_height_m": float(extr.camera_position[1]) if extr.camera_position else None,
        "scene_depth_m": scene_depth_m,
        "scale_health": {"status": sh.status,
                         "safe_to_export": sh.safe_to_export,
                         "detail": sh.detail},
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
    """DEPRECATED — file-path-based solve kept only so saved workflows load.

    Prefer AtlasSolveFromImage (geometric VP solve) or AtlasLearnedSolveFromImage
    (GeoCalib prior) — both take an IMAGE tensor and sit in a normal image chain.
    """

    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"
    DEPRECATED = True

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
        import logging
        logging.warning(
            "AtlasLoadImageSolveCamera is deprecated — use AtlasSolveFromImage "
            "or AtlasLearnedSolveFromImage (IMAGE-tensor inputs) instead.")
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


class AtlasLoadRAW:
    """📷 Camera RAW loader (NEF / CR2 / CR3 / RAF / ARW) — [raw] extra.

    One node replaces the ACR round-trip: rawpy demosaic -> IMAGE tensor for
    solve/preview, EXIF focal + camera-model->sensor lookup -> `raw_meta`
    (wire into a solve node's raw_meta input so the solve stops guessing
    intrinsics), optional lensfun undistort ([raw-lens]), and a scene-linear
    EXR sidecar + ATLAS_PLATE_REF so RAW slots into the OCIO Output Desk path
    exactly where OCIORead does. The EXR and the tensor share one demosaic
    and one undistort grid — geometrically identical by construction.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_PLATE_REF", "ATLAS_RAW_META", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "plate_ref", "raw_meta", "focal_length_mm",
                    "sensor_width_mm", "report")
    FUNCTION = "load"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "",
                    "tooltip": "Path to a camera RAW file (.nef .cr2 .cr3 .raf .arw .dng)."}),
            },
            "optional": {
                # Widget order below is FROZEN (positional serialization) —
                # new widgets append at the end only.
                "undistort": ("BOOLEAN", {"default": True,
                    "tooltip": "Lensfun geometry correction from the EXIF lens model "
                               "([raw-lens] extra). Skipped with a report line when no "
                               "profile matches (common for Fuji X — in-body corrections)."}),
                "half_size": ("BOOLEAN", {"default": False,
                    "tooltip": "Half-resolution demosaic for fast iteration on 36-100MP files."}),
                "white_balance": (["camera", "auto"], {"default": "camera"}),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0,
                                          "step": 0.1}),
                "write_exr": ("BOOLEAN", {"default": True,
                    "tooltip": "Write a scene-linear EXR sidecar and reference it in "
                               "plate_ref (needs opencv 4.x + OPENCV_IO_ENABLE_OPENEXR=1 "
                               "set before ComfyUI starts — same constraint as the OCIO "
                               "path). On failure the plate_ref degrades to proxy."}),
                "output_dir": ("STRING", {"default": "atlas_exports/raw_plates"}),
                "colorspace": ("STRING", {"default": "Linear Rec.709 (sRGB)",
                    "tooltip": "Colorspace TAG for the sidecar. rawpy's linear output has "
                               "sRGB/Rec.709 primaries — NOT ACEScg; convert downstream "
                               "via OCIO. Retag only if your config names it differently."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, file_path, **kwargs):
        try:
            stat = os.stat(str(file_path))
            return f"{file_path}:{stat.st_mtime_ns}:{stat.st_size}:{sorted(kwargs.items())}"
        except OSError:
            return f"{file_path}:missing:{sorted(kwargs.items())}"

    def load(self, file_path, undistort=True, half_size=False, white_balance="camera",
             exposure_ev=0.0, write_exr=True, output_dir="atlas_exports/raw_plates",
             colorspace="Linear Rec.709 (sRGB)"):
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.schema import AtlasPlateRef
        try:
            from atlas_camera.raw import import_raw
        except ImportError as exc:
            raise RuntimeError(
                "AtlasLoadRAW requires the [raw] extra. "
                "Install with: pip install -e .[raw]") from exc

        path = str(file_path or "").strip()
        if not path or not Path(path).is_file():
            raise RuntimeError(f"AtlasLoadRAW: RAW file not found: {path!r}")

        result = import_raw(path, undistort=bool(undistort),
                            half_size=bool(half_size),
                            white_balance=white_balance,
                            exposure_ev=float(exposure_ev))

        image = torch.from_numpy(
            np.ascontiguousarray(result.display_srgb)).unsqueeze(0)

        exr_path, exr_warning = (None, None)
        if write_exr:
            exr_path, exr_warning = self._write_exr_sidecar(
                result.linear_rgb, path, output_dir)

        report_lines = result.summary_lines()
        if exr_path:
            report_lines.append(f"linear EXR: {exr_path} ({colorspace})")
        elif exr_warning:
            report_lines.append(exr_warning)

        plate_ref = AtlasPlateRef(
            image_path=exr_path,
            preview_b64=_image_tensor_to_preview_b64(image, quality=85),
            colorspace=colorspace or "Linear Rec.709 (sRGB)",
            bit_depth="16f" if exr_path else "8-bit/proxy",
            role="source",
            is_proxy=exr_path is None,
            metadata={
                "registered_from": "AtlasLoadRAW",
                "raw_source": path,
                "camera_model": result.camera_model,
                "undistort_status": result.undistort_status,
            },
        )
        return (image, plate_ref, result,
                float(result.focal_length_mm or 0.0),
                float(result.sensor_width_mm or 36.0),
                "\n".join(report_lines))

    @staticmethod
    def _write_exr_sidecar(linear_rgb, raw_path, output_dir):
        """Write the scene-linear half-float EXR. Returns (path, warning)."""
        try:
            import cv2
        except ImportError:
            return None, ("EXR sidecar skipped: opencv-python missing "
                          "(pip install -e .[raw]).")
        out_dir = Path(str(output_dir or "atlas_exports/raw_plates"))
        out_dir.mkdir(parents=True, exist_ok=True)
        exr_path = out_dir / (Path(raw_path).stem + "_linear.exr")
        bgr = linear_rgb[..., ::-1].astype("float32")
        try:
            ok = cv2.imwrite(str(exr_path),
                             bgr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        except Exception:  # noqa: BLE001 — codec-disabled builds raise
            ok = False
        if not ok or not exr_path.is_file():
            return None, ("EXR sidecar FAILED: opencv needs the OpenEXR codec — "
                          "use opencv-python 4.x and set OPENCV_IO_ENABLE_OPENEXR=1 "
                          "before ComfyUI starts (same requirement as the OCIO path). "
                          "plate_ref downgraded to proxy.")
        return str(exr_path), None


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
                    "tooltip": "0 = auto-detect, or EXIF via a wired raw_meta"}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
                "detect_vanishing_points": ("BOOLEAN", {"default": True,
                    "tooltip": "Run line/VP detection. Off = metadata-only solve "
                               "(no fx, cam_y=0 -> black depth/blockout)."}),
                # Link input (not a widget — saved-workflow-safe): AtlasLoadRAW's
                # metadata; supplies EXIF focal + measured sensor unless the
                # widgets above are explicitly set.
                "raw_meta": ("ATLAS_RAW_META",),
            },
        }

    def solve(self, image, focal_length_mm=0.0, sensor_width_mm=36.0,
              detect_vanishing_points=True, raw_meta=None):
        tmp = _save_image_tensor_to_tmp(image)
        try:
            focal_hint, sensor_w, sensor_h = _resolve_raw_hints(
                focal_length_mm, sensor_width_mm, raw_meta)
            hints: dict[str, Any] = {}
            if focal_hint:
                hints["focal_length_mm"] = focal_hint
                hints["sensor_width_mm"] = sensor_w
                if sensor_h:
                    hints["sensor_height_mm"] = sensor_h
            solve = solve_still_image(tmp, intrinsics_hint=hints or None,
                                      detect_vanishing_points=detect_vanishing_points)
            _stamp_raw_provenance(solve, raw_meta)
            return (solve,)
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
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "tooltip": "Metric depth backend (fed the solved focal). V2-Metric-Outdoor "
                               "(DEFAULT) / V2-Metric-Indoor: Apache, transformers-only (NO extra "
                               "install), best all-round; Outdoor wins on sky/exterior scenes. "
                               "MoGe-2 (Ruicheng/moge-*): MIT, cleanest on ENCLOSED/INTERIOR shots "
                               "but masks sky (poor outdoors) — needs [moge]. DA3* (EXPERIMENTAL): "
                               "strong metric, heavy deps, DA3NESTED is non-commercial CC BY-NC — "
                               "needs [neural-da3]. (4-scene A/B 2026-07-13.)"}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
                "weights": (["pinhole", "simple_radial"], {"default": "pinhole",
                    "tooltip": "pinhole = no lens distortion (best for clean AI renders)."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                # APPENDED 2026-07-18 (positional widget rule: new widgets go last).
                "focal_length_mm": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "tooltip": "0 = GeoCalib predicts the focal. >0 (or a wired AtlasLoadRAW "
                               "raw_meta) = trusted focal (e.g. EXIF) wins; GeoCalib still "
                               "supplies gravity/roll."}),
                # Link input (not a widget — saved-workflow-safe).
                "raw_meta": ("ATLAS_RAW_META",),
            },
        }

    def solve(self, image, height_mode="measure_from_depth", camera_height_m=1.6,
              depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
              sensor_width_mm=36.0, weights="pinhole", device="auto",
              focal_length_mm=0.0, raw_meta=None):
        from atlas_camera.core.solver import solve_still_image_learned
        focal_hint, sensor_w, sensor_h = _resolve_raw_hints(
            focal_length_mm, sensor_width_mm, raw_meta)
        tmp = _save_image_tensor_to_tmp(image)
        try:
            h, w = int(image.shape[1]), int(image.shape[2])
            camera_height = "auto" if height_mode == "measure_from_depth" else camera_height_m
            solve = solve_still_image_learned(
                tmp,
                image_size=(w, h),
                camera_height=camera_height,
                sensor_width_mm=sensor_w,
                sensor_height_mm=sensor_h,
                focal_length_mm_hint=focal_hint,
                weights=weights,
                depth_model=depth_model,
                device=None if device == "auto" else device,
            )
            _stamp_raw_provenance(solve, raw_meta)
            return (solve,)
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
                "depth_model": (
                    list(_DEPTH_MODEL_CHOICES) + ["depth-anything/Depth-Anything-V2-Small-hf"],
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "solve": ("ATLAS_SOLVE", {"tooltip": "Optional — supplies the SOLVED focal "
                          "(GeoCalib/VP) for DA3METRIC's canonical→metric conversion "
                          "(focal_source='solve' instead of the assumed normal-lens fallback). "
                          "Ignored by V2 models."}),
            },
        }

    def estimate(self, image, depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto", solve=None):
        from atlas_camera.inference.depth_estimator import estimate_depth
        np = _require_numpy()
        torch = _require_torch()
        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device,
                                    focal_px=_solve_focal_px_for_image(solve, image))
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


class AtlasScaleOverride:
    """📐 Manual metric-scale dial for a solve — the artist's scale override.

    Single-image camera recovery has an inherent SCALE ambiguity: with no ground
    plane to fit and no known-size reference, the solve falls back to an assumed
    ~1.6 m eye height (`scale_source=assumed_default`), which is often far off for
    elevated vistas (a cityscape overlook can read ~10× too small). Metric scale
    is PROPORTIONAL to camera height (`scale = −cam_y/g`, with g fixed by the
    depth), so this node rescales the solve by a single factor — multiplying the
    camera position and both extrinsics matrices' translation columns — and EVERY
    downstream metric follows: geometry distances, the 📏 Band Box cutoffs, and
    the DCC-export camera positions. The projection is purely angular, so the
    view/texture mapping is pixel-identical — only the metric numbers move.

    `scale` is a plain multiplier (10.0 = ten times as far/big — the "1:10" case).
    `camera_height_m` (0 = off) instead SETS an absolute camera height when you
    know the real vantage, and the node computes the factor for you. Composable
    companion node (works after ANY solve); stamps `scale_source="manual_override"`.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "override"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 100000.0, "step": 0.1,
                    "tooltip": "Metric scale multiplier — 10.0 = the whole scene is 10× as far/big "
                               "(the single-image '1:10' case). Metric scale ∝ camera height, so this "
                               "uniformly rescales every downstream distance (geometry, 📏 cutoffs, "
                               "DCC-export cameras); the projected view is unchanged. Ignored when "
                               "camera_height_m > 0."}),
                "camera_height_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000000.0, "step": 0.1,
                    "tooltip": "Absolute override: SET the camera height in metres when you know the "
                               "real vantage (the node computes the factor). 0 = use the scale "
                               "multiplier instead."}),
            },
        }

    def override(self, solve, scale=1.0, camera_height_m=0.0):
        import copy
        out = copy.deepcopy(solve)
        extr = out.camera.extrinsics
        vm = [list(r) for r in extr.camera_view_matrix]   # world->cam
        # Camera WORLD position p = -R_wc^T @ t_wc (robust: some solves leave the
        # camera_position field at 0 but always populate the view matrix). The
        # translation column scales by the same factor, so p_new = p * factor.
        t = [vm[0][3], vm[1][3], vm[2][3]]
        p = [-(vm[0][k] * t[0] + vm[1][k] * t[1] + vm[2][k] * t[2]) for k in range(3)]
        cur_h = p[1]
        if float(camera_height_m) > 0.0 and abs(cur_h) > 1e-6:
            factor = float(camera_height_m) / abs(cur_h)
        else:
            factor = float(scale)
        if not (factor > 0.0):
            factor = 1.0

        for r in range(3):
            vm[r][3] = vm[r][3] * factor
        extr.camera_view_matrix = tuple(tuple(r) for r in vm)
        extr.camera_position = tuple(c * factor for c in p)
        wm = [list(r) for r in extr.camera_world_matrix]
        for r in range(3):
            wm[r][3] = wm[r][3] * factor
        extr.camera_world_matrix = tuple(tuple(r) for r in wm)
        meta = dict(getattr(out, "debug_metadata", None) or {})
        meta["scale_override"] = factor
        meta["scale_source"] = "manual_override"
        out.debug_metadata = meta

        new_h = extr.camera_position[1]
        report = (
            f"AtlasScaleOverride: ×{factor:.4g}  |  camera height {cur_h:.2f} m → {new_h:.2f} m\n"
            "  Rescales ALL downstream metric — geometry distances, 📏 Band Box cutoffs, and DCC "
            "export cameras — uniformly; the projection/view is unchanged (angular). Insert between "
            "the solve and the geometry/viewport nodes.")
        return (out, report)


class AtlasRollTrim:
    """🎚 Manual roll trim for a solve — level a leaning solve by eye.

    GeoCalib's gravity estimate can drift a few degrees on AI-generated plates
    with no true horizon (measured live: −5.6° solved vs ~−2.6° implied by the
    architecture's verticals on a sci-fi interior), and the classical VP
    cross-check often finds nothing on greebled/non-rectilinear scenes. This is
    the roll counterpart of `AtlasScaleOverride`'s scale dial: rotate the
    recovered camera about its own VIEW AXIS by `roll_deg` and let everything
    downstream follow. The camera position and view direction are INVARIANT —
    only the camera's up/right spin — so framing is preserved and the fix is
    purely orientational.

    Wire it between the solve and the depth/derive nodes (like the scale
    dial): geometry back-projects through the view matrix, so a trim applied
    AFTER derivation leaves already-built geometry in the old frame (the
    report warns if the incoming solve carries proxy geometry). Pure Python,
    zero deps; composable after any solve; stamps
    `debug_metadata["roll_trim_deg"]` (accumulates across chained trims).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "trim"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "roll_deg": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 0.05,
                    "tooltip": "Extra roll (degrees) about the recovered camera's view axis. "
                               "0 = no-op. Positive rotates the projected scene counter-clockwise "
                               "on screen (the horizon's right end rises); negative clockwise. "
                               "Dial until verticals/horizon read level. Position and view "
                               "direction never move."}),
            },
        }

    def trim(self, solve, roll_deg=0.0):
        import copy
        import math
        out = copy.deepcopy(solve)
        d = float(roll_deg)
        if abs(d) < 1e-9:
            return (out, "AtlasRollTrim: 0.00° — no-op (dial roll_deg to level the solve)")
        extr = out.camera.extrinsics
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))

        # V' = Rz(d) @ V — an extra roll in the CAMERA frame, left-multiplied
        # onto the world→cam view matrix. Rz preserves the camera z axis, so
        # the view direction is untouched; the rigid inverse below shows the
        # position is too (Rz's translation is zero).
        vm = [list(r) for r in extr.camera_view_matrix]
        rz = ((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        vm2 = [[sum(rz[r][k] * vm[k][col] for k in range(4)) for col in range(4)] for r in range(4)]
        extr.camera_view_matrix = tuple(tuple(row) for row in vm2)

        # Rigid inverse → world matrix; R_cw (columns = camera axes in world)
        # is the transpose of the view rotation block.
        r_wc = [[vm2[r][k] for k in range(3)] for r in range(3)]
        t_wc = [vm2[r][3] for r in range(3)]
        r_cw = [[r_wc[k][r] for k in range(3)] for r in range(3)]
        pos = [-sum(r_cw[r][k] * t_wc[k] for k in range(3)) for r in range(3)]
        extr.camera_world_matrix = tuple(
            tuple([*r_cw[r], pos[r]]) for r in range(3)
        ) + ((0.0, 0.0, 0.0, 1.0),)
        extr.camera_rotation_matrix = tuple(tuple(row) for row in r_cw)
        extr.camera_position = tuple(pos)

        # Recompute the stored horizon LINE for the rolled camera (no longer a
        # single image row): the vanishing line of world-horizontal planes is
        # the set of pixels whose backprojected rays have zero world-Y
        # direction — linear in (u, v). Ray(u,v) ∝ ((u-cx)/fx, -(v-cy)/fy, -1)
        # in the camera frame; world-Y component = R_cw row 1 · ray = 0.
        horizon_note = ""
        intr = out.camera.intrinsics
        if out.horizon_line is not None and intr.fx_px and intr.image_width:
            fx = float(intr.fx_px)
            fy = float(intr.fy_px or intr.fx_px)
            cx = float(intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0)
            cy = float(intr.cy_px if intr.cy_px is not None else (intr.image_height or 0) / 2.0)
            w = float(intr.image_width)
            a = r_cw[1][0] / fx
            b = -r_cw[1][1] / fy
            cc = -r_cw[1][0] * cx / fx + r_cw[1][1] * cy / fy - r_cw[1][2]
            if abs(b) > 1e-12:
                y_at = lambda u: (-cc - a * u) / b  # noqa: E731
                y0, y1 = y_at(0.0), y_at(w)
                out.horizon_line.endpoints_px = ((0.0, y0), (w, y1))
                out.horizon_line.line_coefficients = (a, b, cc)
                tilt = math.degrees(math.atan2(y1 - y0, w))
                horizon_note = f"  |  horizon tilt now {tilt:+.2f}°"
                meta_ce = dict((out.debug_metadata or {}).get("camera_estimation") or {})
                meta_ce["horizon_angle"] = tilt
                meta = dict(out.debug_metadata or {})
                meta["camera_estimation"] = meta_ce
                out.debug_metadata = meta

        meta = dict(out.debug_metadata or {})
        meta["roll_trim_deg"] = float(meta.get("roll_trim_deg", 0.0)) + d
        out.debug_metadata = meta

        geom_warn = ""
        scene = getattr(out, "projection_scene", None)
        if scene is not None and getattr(scene, "proxy_geometry", None):
            geom_warn = ("\n  ⚠ this solve already carries derived geometry, built in the UN-trimmed "
                         "frame — wire AtlasRollTrim BEFORE the depth/derive nodes instead.")
        report = (
            f"AtlasRollTrim: {d:+.2f}° about the view axis{horizon_note}\n"
            "  Camera position and view direction unchanged — only up/right rotate; every "
            "downstream derive/export follows. Composable after any solve." + geom_warn)
        return (out, report)


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

    A vision-language model (Ollama / LM Studio / llama.cpp locally, or the
    `openai` provider — any OpenAI-compatible cloud endpoint + api_key, for
    users without local models; the same provider layer as
    `AtlasVLMScaleCues`) analyzes the photo against an
    expert instruction prompt encoding Atlas Camera's full settings knowledge
    (`inference.assessor.ATLAS_ASSESSMENT_SYSTEM_PROMPT`): scene type /
    depth-model choice, sky separation, depth-band layer design, disocclusion
    fill, edge mattes, relief tuning, scale-reference opportunities, and an
    honest camera-move viability rubric (score + max orbit degrees + what
    breaks first). The `report` output is human-readable (wire to a
    Show Text node); `settings_json` is the machine-readable
    recommended_settings block.

    STAGED 5-LAYER PLAN: the assessment also divides the photo into the
    staged master workflow's five fixed layers (sky + far/bg/mid/fg depth
    bands) and emits one SAM3 prompt STRING output per layer
    (`sam_prompt_*`) — wire them into the sky SAM3Segment and the four SAM
    SCOPE rows' prompt inputs so each row's segmentation prompt comes from
    the assessment instead of hand-typing. Not every image has every layer:
    an absent layer (no sky, empty mid band, ...) yields "" and the report
    says to leave that stage bypassed; only sky falls back to the literal
    "sky" (a no-match prompt there returns an empty mask, which IS the
    correct sky mask for a skyless photo).

    EXECUTION PAUSE — opt-in since 2026-07-11 (`auto_continue`, default ON):
    by default the node is ADVISORY: the assessment runs, its staged
    prompts/geometry flow downstream, and the same queue continues — the ✅
    solve gate (and the 📐 patch gate) are the workflow's checkpoints. With
    `auto_continue` OFF the original hard gate returns: while `proceed` is
    False the `image` output returns ExecutionBlocker — everything
    downstream of the photo is silently skipped, so the first Queue costs
    only the assessment; ▶ Continue Workflow approves THIS image (the
    assessment is cached per image+provider, so continuing never re-runs the
    VLM). Same native pause mechanism as 📐 Extract Angle gating.

    Advisory only, per the LLM-confirm principle: the VLM never changes a
    setting itself — it recommends, the artist decides. Fails soft to a
    "provider unreachable" report; `proceed` still works without an
    assessment.
    """
    RETURN_TYPES = ("IMAGE", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "report", "settings_json",
                    "sam_prompt_sky", "sam_prompt_far", "sam_prompt_bg",
                    "sam_prompt_mid", "sam_prompt_fg",
                    "geom_far", "geom_bg", "geom_mid", "geom_fg",
                    "band_far", "band_bg", "band_mid", "band_fg")
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
                "provider": (["ollama", "lmstudio", "llamacpp", "openai"], {"default": "ollama",
                    "tooltip": "VLM backend. ollama/lmstudio/llamacpp are local servers; "
                               "'openai' is any OpenAI-compatible CLOUD endpoint (api.openai.com "
                               "by default, OpenRouter etc. via base_url) for users without local "
                               "models — needs api_key. Blank model/base_url use each provider's "
                               "own defaults (same conventions as AtlasVLMScaleCues)."}),
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
                # APPENDED last (widgets_values is positional — never insert).
                "api_key": ("STRING", {"default": "",
                    "tooltip": "API key for the 'openai' cloud provider (ignored by local "
                               "providers). SAVED INTO THE WORKFLOW FILE — prefer leaving this "
                               "blank and setting the OPENAI_API_KEY environment variable so "
                               "shared workflows never carry your key."}),
                "offload_model": ("BOOLEAN", {"default": False,
                    "tooltip": "Free the VLM's VRAM after a SUCCESSFUL assessment so the heavy "
                               "pipeline (depth/SAM/LaMa) doesn't fight it for memory — the "
                               "assessment is cached per image, so ▶ Continue never reloads the "
                               "model. Per provider: ollama = keep_alive:0 (clean unload); "
                               "lmstudio = request ttl for JIT loads + the 'lms' CLI when on "
                               "PATH for GUI-loaded models; llamacpp = NOT possible (the server "
                               "owns its model — restart it to free VRAM); openai = nothing "
                               "local. A failed assessment keeps the model warm for the retry. "
                               "The report shows what actually happened."}),
                "auto_continue": ("BOOLEAN", {"default": True,
                    "tooltip": "ON (default): advisory mode — the assessment runs, its SAM "
                               "prompts/geometry flow downstream, and the SAME queue continues "
                               "without a ▶ Continue click; the ✅ solve gate (and the later 📐 "
                               "patch gate) become the workflow's checkpoints. Turn OFF to "
                               "restore the hard per-image gate: the image output blocks until "
                               "▶ Continue Workflow approves THIS image."}),
            },
        }

    def assess(self, image, provider="ollama", model="", base_url="",
               extra_instructions="", proceed=False, approved_for="",
               api_key="", offload_model=False, auto_continue=True, **_extra):
        # **_extra: API-format exports can serialize the ▶ Continue Workflow
        # BUTTON widget as a bogus input key — tolerate unknown kwargs.
        import hashlib

        from atlas_camera.inference.assessor import (
            assess_image,
            staged_layer_bands,
            staged_layer_geometry,
            staged_layer_prompts,
        )

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
                    api_key=api_key.strip() or None,
                    extra_instructions=extra_instructions,
                    offload_model=bool(offload_model))
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
        # auto_continue (default ON): advisory mode — never block; the solve
        # gate downstream is the first checkpoint. OFF restores the hard
        # per-image ▶ Continue gate with its stale-approval re-arming.
        effective_proceed = bool(auto_continue) or (
            bool(proceed) and (not approved_for or approved_for == img_fp))
        if not auto_continue and proceed and approved_for and approved_for != img_fp:
            report = ("*** GATE RE-ARMED: the input image changed since ▶ Continue was "
                      "clicked — review the fresh assessment below, then ▶ Continue "
                      "again for this image. ***\n\n" + report)

        if effective_proceed:
            img_out = image
        else:
            blocker = _execution_blocker()
            img_out = blocker if blocker is not None else image
        # Staged 5-layer SAM prompts + per-band geometry recommendations —
        # plain strings, NOT gated: everything they feed (SAM3 nodes /
        # AtlasCleanPlateLayer) also consumes the gated image via the plate
        # rail, so the image blocker already pauses it. geom_* wires into
        # AtlasCleanPlateLayer.geometry_override ("" = no recommendation,
        # the layer node's own band_geometry combo applies).
        sam = staged_layer_prompts(cached.payload if cached.ok else {})
        geom = staged_layer_geometry(cached.payload if cached.ok else {})
        # Watertight band boundaries (jointly derived — adjacent bands share
        # edges by construction); "" when no assessment = nodes keep widgets.
        band = staged_layer_bands(cached.payload if cached.ok else {})

        # ui.text renders the report directly on the node (atlas_assess.js);
        # ui.fingerprint is what the ▶ button stamps into approved_for.
        # ui.sam_prompts / ui.sam_geometry let the frontend mirror the
        # resolved values into LINKED widgets — a widget converted to a
        # linked input keeps displaying its stale typed text otherwise
        # (found live: values flowed at execution but were invisible).
        return {"ui": {"text": [report], "fingerprint": [img_fp],
                       "sam_prompts": [sam["sky"], sam["far"], sam["bg"],
                                       sam["mid"], sam["fg"]],
                       "sam_geometry": [geom["far"], geom["bg"],
                                        geom["mid"], geom["fg"]],
                       "sam_bands": [band["far"], band["bg"],
                                     band["mid"], band["fg"]]},
                "result": (img_out, report, settings_json,
                           sam["sky"], sam["far"], sam["bg"], sam["mid"], sam["fg"],
                           geom["far"], geom["bg"], geom["mid"], geom["fg"],
                           band["far"], band["bg"], band["mid"], band["fg"])}


_ATLAS_ASSESS_CACHE: dict = {}


class AtlasSolveGate:
    """✅ Solve-confirm checkpoint — pause the heavy graph until the artist
    approves the camera solve.

    The third gate in the established family (AtlasAssessImage gates the
    whole graph on VLM pre-flight; 📐 pauses the patch branch): wire
    `solve → viewport` UNGATED for a cheap preview (a low-grid relief costs
    seconds) and `solve → this gate → the heavy stack` (grid-1024 band
    layers, sky dome, Fixer, exports — the minutes). The first Queue costs a
    solve and a thumbnail-grade preview; check the camera in ℹ/📊 (or the
    report rendered on this node), click ✅ Approve Solve, and the re-queue
    runs the expensive graph exactly once, on a solve you signed off.

    Approval is fingerprint-scoped to (solve camera + source image): a new
    photo OR a re-solve with different settings re-arms the gate instead of
    sailing through a stale approval (the persisted-gating rule every gate
    here follows). Empty `approved_for` with proceed=True stays the manual
    unconditional override. Outside a ComfyUI runtime the gate degrades to
    pass-through (no ExecutionBlocker available).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "gate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE", {"tooltip":
                    "The photo this solve came from — part of the approval "
                    "identity, so swapping the image re-arms the gate."}),
            },
            "optional": {
                "proceed": ("BOOLEAN", {"default": False, "tooltip":
                    "While off, the solve output returns ExecutionBlocker and "
                    "everything downstream of the gate is silently paused. "
                    "The ✅ Approve Solve button sets this and re-queues."}),
                "approved_for": ("STRING", {"default": "", "tooltip":
                    "Fingerprint of the solve+image the current approval was "
                    "given for (stamped by ✅). Mismatch re-arms the gate. "
                    "Leave empty when toggling proceed by hand to approve "
                    "unconditionally."}),
            },
        }

    def gate(self, solve, source_image, proceed=False, approved_for="", **_extra):
        # **_extra: API-format exports can serialize the button widget as a
        # bogus input key — tolerate unknown kwargs (AssessImage precedent).
        import math as _math

        np = _require_numpy()

        fp = _solve_fingerprint(solve, source_image)
        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        try:
            vm = np.array(extr.camera_view_matrix, dtype=np.float64)
            fwd = np.linalg.inv(vm)[:3, :3] @ np.array([0.0, 0.0, -1.0])
            pitch = _math.degrees(_math.asin(max(-1.0, min(1.0, float(fwd[1])))))
        except Exception:
            pitch = float("nan")
        fov = (2 * _math.degrees(_math.atan((intr.image_width or 0) /
               (2 * intr.fx_px))) if intr.fx_px else float("nan"))
        cam_h = (extr.camera_position or (0, float("nan"), 0))[1]
        meta = solve.debug_metadata or {}
        from atlas_camera.core.scene_health import scale_health
        sh = scale_health(solve)
        effective = bool(proceed) and (not approved_for or approved_for == fp)
        lines = [
            "✅ SOLVE APPROVED — heavy graph running." if effective else
            "⏸ SOLVE GATE — downstream paused. Review, then ✅ Approve Solve.",
            (f"focal: {intr.focal_length_mm:.1f}mm ({fov:.1f}° hFOV) on "
             f"{intr.sensor_width_mm}mm") if intr.focal_length_mm else "focal: n/a",
            f"camera height: {cam_h:.2f}m  (scale: {sh.status} / "
            f"{meta.get('scale_source', 'n/a')})",
            f"pitch: {pitch:+.1f}°",
            (f"confidence: {solve.confidence:.2f}  ({solve.source_method})"
             if getattr(solve, "confidence", None) is not None else ""),
        ]
        if not sh.safe_to_export:
            lines.insert(1, f"⚠ SCALE NOT VERIFIED — {sh.detail}")
        if proceed and approved_for and approved_for != fp:
            lines.insert(0, "*** GATE RE-ARMED: the solve or image changed since "
                            "approval — review and ✅ Approve again. ***")
        report = "\n".join(l for l in lines if l)

        if effective:
            out = solve
        else:
            blocker = _execution_blocker()
            out = blocker if blocker is not None else solve
        return {"ui": {"text": [report], "fingerprint": [fp]},
                "result": (out, report)}


class AtlasVLMScaleCues:
    """Detect scale-reference objects with a local vision-language model.

    Runs a local VLM (LM Studio / llama.cpp / Ollama) to find known-size objects
    (people, doors, cars, …) and emits ``scale_references`` JSON for
    AtlasApplyScaleReferences. Requires a running local VLM server — or, for
    users without local models, the ``openai`` provider: any OpenAI-compatible
    cloud endpoint via ``base_url`` + ``api_key``. The model must return pixel
    bounding boxes. Advisory only — nothing is applied without the artist
    confirming in AtlasApplyScaleReferences.
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
                "provider": (["ollama", "lmstudio", "llamacpp", "openai"], {"default": "ollama",
                    "tooltip": "ollama/lmstudio/llamacpp are local; 'openai' is any "
                               "OpenAI-compatible cloud endpoint (needs api_key)."}),
                "model": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "", "tooltip": "Blank = provider default URL"}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                # APPENDED last (widgets_values is positional — never insert).
                "api_key": ("STRING", {"default": "",
                    "tooltip": "API key for the 'openai' cloud provider (ignored by local "
                               "providers). SAVED INTO THE WORKFLOW FILE — prefer the "
                               "OPENAI_API_KEY environment variable for shared workflows."}),
            },
        }

    def analyze(self, image, provider="ollama", model="", base_url="", min_confidence=0.0,
                api_key=""):
        from atlas_camera.inference.multimodal_helper import (
            create_multimodal_provider,
            scale_references_from_observation,
        )
        from atlas_camera.reference_data import load_scale_references

        tmp = _save_image_tensor_to_tmp(image)
        try:
            candidate_ids = [r.id for r in load_scale_references()]
            prov = create_multimodal_provider(provider, model=model, base_url=base_url or None,
                                              api_key=api_key.strip() or None)
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
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 32,
                                        "tooltip": "Max foreground boxes/cylinders. Street-level scenes: try 0 — the 2D occupancy clustering merges cars/fences/trees into oversized near-camera boxes that dominate any orbit."}),
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
                               "SAM/RMBG) which REPLACES the internal sky heuristic before "
                               "triangulation - so it must cover EVERYTHING you want gone. Only "
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
        # Presets use the zero-extra-install V2 metric models (Apache, transformers
        # only) so a fresh install never errors on a missing DA3/MoGe extra. A 4-scene
        # A/B (2026-07-13) reverted the 2026-07-09 DA3 default: V2-Metric-Outdoor was
        # best-or-tied on every outdoor/sky scene; MoGe masks sky (poor outdoors, great
        # indoors); DA3 is the experimental branch's default. Artists opt into DA3/MoGe
        # per shot via the depth_model widget.
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

        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        width = int(intr.image_width or image.shape[2])
        height = int(intr.image_height or image.shape[1])
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx

        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device,
                                    # fx is in solve-image pixels; the tmp file is the
                                    # wired tensor's resolution (usually identical).
                                    focal_px=(fx * (image.shape[2] / width)) if fx > 0 else None)
        finally:
            os.unlink(tmp)

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


_GROUND_SCALE_CACHE: dict = {}


def _ground_scale_cached(depth_map, view_matrix, fx, fy, cx, cy, horizon_y):
    """Memoized estimate_ground_scale for the shared-depth node family.

    The staged master runs _metric_depth_and_validity in EVERY band node
    (mask + layer, x4 bands, + sky) — 8+ identical full-resolution ground
    fits per queue on the same DepthResult. The fit is deterministic in
    (depth content, camera, horizon), so memoize on the FULL array's
    float32 hash + shape + rounded camera params (id()-based keys are
    unsafe — CPython reuses ids after GC; a strided sample hash was the
    first version, dropped per code review: two maps identical at the
    samples but differing elsewhere would silently return the wrong
    scale). The full hash is ~tens of ms at 4K vs the seconds-long fit
    a hit saves.
    """
    import hashlib as _hashlib

    np = _require_numpy()
    sig = _hashlib.md5(
        np.ascontiguousarray(depth_map, dtype=np.float32).tobytes()).hexdigest()
    vm = tuple(round(float(x), 6) for row in view_matrix for x in row)
    key = (sig, depth_map.shape, vm, round(float(fx), 3), round(float(fy), 3),
           round(float(cx), 3), round(float(cy), 3),
           None if horizon_y is None else round(float(horizon_y), 2))
    hit = _GROUND_SCALE_CACHE.get(key)
    if hit is not None:
        return hit[0], dict(hit[1])   # copy the info dict — a caller mutating
                                      # it must never poison the cache
    from atlas_camera.core.relief_mesh import estimate_ground_scale
    out = estimate_ground_scale(depth_map, view_matrix=view_matrix,
                                fx=fx, fy=fy, cx=cx, cy=cy, horizon_y=horizon_y)
    if len(_GROUND_SCALE_CACHE) >= 16:
        _GROUND_SCALE_CACHE.pop(next(iter(_GROUND_SCALE_CACHE)))
    _GROUND_SCALE_CACHE[key] = out
    return out[0], dict(out[1])


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

    scale, _ground_info = _ground_scale_cached(
        depth_map, extr.camera_view_matrix, fx, fy, cx, cy, horizon_y)
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
    0 = unset) or, as a fallback, POSITIONS ALONG THE SCENE'S LOG-DEPTH RANGE
    (``near_pct``/``far_pct``, 0..1; 0.5 = the geometric mean of the robust
    depth range — see ``log_depth_position`` below for why this replaced
    pixel-count percentiles).

    Shared by ``AtlasDepthLayerMask`` and ``AtlasCleanPlateLayer`` so the two
    nodes' bands can never drift apart — the inpaint-layers design requires the
    mask node's band and the clean-plate node's mesh clip to match exactly.
    ``far_pct<=0`` is a deliberate explicit "no upper bound" (+inf) rather than
    a degenerate zero-position far edge, since ``near_pct``/``far_pct`` share
    the same 0..1 range but mean different things at 0 (near defaults to the
    very nearest pixels; far defaults to "no cap" via ``far_pct=0.5``, and an
    artist setting ``far_pct=0`` clearly means "no upper band edge", not
    "collapse the band to nothing").
    """
    np = _require_numpy()
    values = metric[valid] if valid.any() else None

    def log_depth_position(t):
        # LOG-DEPTH interpolation, not a pixel-count percentile: metric depth
        # is hugely skewed (near ground dominates the pixel count, the whole
        # far scene compresses into the top percentiles), so a linear
        # percentile slider wasted 0-0.9 on the foreground (user-measured:
        # useful bg splits landed at 0.9-0.95). Position t along the scene's
        # log depth range is perceptually linear: 0.5 = the geometric mean of
        # the (robust, 1st-99th percentile) depth range. t>=0.995 = no cap.
        import math
        d_lo = float(np.percentile(values, 1.0))
        d_hi = float(np.percentile(values, 99.0))
        if not (d_hi > d_lo > 0):
            return float(np.percentile(values, t * 100.0))  # degenerate scene
        return math.exp(math.log(d_lo) + t * (math.log(d_hi) - math.log(d_lo)))

    if near_m and near_m > 0:
        near = float(near_m)
    elif values is not None and near_pct > 0:
        near = log_depth_position(min(float(near_pct), 1.0))
    else:
        near = 0.0
    if far_m and far_m > 0:
        far = float(far_m)
    elif values is not None and 0 < far_pct < 0.995:
        far = log_depth_position(float(far_pct))
    else:
        far = float("inf")
    return near, far


def _parse_band_override(text):
    """Parse the assess node's band_far/bg/mid/fg output format
    ('near_pct=<f> far_pct=<f>') into a (near_pct, far_pct) tuple. "" / None
    -> None (no override). Errors loudly on garbage, per the
    patch_view_override / geometry_override precedent. The strings come from
    `assessor.staged_layer_bands`, whose joint boundary derivation guarantees
    adjacent bands share edges exactly — never hand-assemble these per band."""
    t = (text or "").strip()
    if not t:
        return None
    m = re.fullmatch(r"near_pct=([0-9.]+)\s+far_pct=([0-9.]+)", t)
    if not m:
        raise ValueError(
            f"Unparseable band override {text!r} — expected 'near_pct=<f> far_pct=<f>' "
            "(the AtlasAssessImage band_* output format).")
    near, far = float(m.group(1)), float(m.group(2))
    if not (0.0 <= near <= far <= 1.0):
        raise ValueError(f"Band override out of range (need 0 <= near <= far <= 1): {text!r}")
    return near, far


def _band_resolution_validity(setup, band_ref_mask):
    """Validity used ONLY for percentile band-edge resolution.

    Per-layer scoped excludes (sky ∪ NOT(segment), the 🎯 scope rows) change
    each layer's depth POPULATION, so the same near/far percentages resolve
    to different metres per layer — the debug report flagged real metric
    GAPs between adjacent bands (mid far 8.26m vs bg near 9.46m on the same
    run). Wiring the same `band_ref_mask` (the plain sky mask) into every
    band node makes all layers resolve band edges over one shared population
    again, restoring the "bands can never drift" contract. Default (None) =
    setup.valid, i.e. the legacy behavior, so existing calibrated workflows
    are untouched.
    """
    if band_ref_mask is None:
        return setup.valid
    np = _require_numpy()
    v = np.isfinite(setup.metric) & (setup.metric > 1e-6)
    ref = _resolve_exclude_mask(band_ref_mask, setup.height, setup.width)
    if ref is not None:
        v = v & ~ref.astype(bool)
    return v


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
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "solve": ("ATLAS_SOLVE", {"tooltip": "Optional — supplies the SOLVED focal "
                          "(GeoCalib/VP) for DA3METRIC's canonical→metric conversion "
                          "(focal_source='solve' instead of the assumed normal-lens fallback). "
                          "Ignored by V2 models."}),
            },
        }

    def estimate(self, image, depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto", solve=None):
        from atlas_camera.inference.depth_estimator import estimate_depth
        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device,
                                    focal_px=_solve_focal_px_for_image(solve, image))
        finally:
            os.unlink(tmp)
        return (result,)


def _resize_normal_field(normals, target_hw):
    """Resize an (H,W,3) unit-normal field to ``target_hw`` (h, w) and
    renormalize. Bilinear via PIL mode 'F' per channel; nearest-neighbour numpy
    fallback if PIL is unavailable. A no-op when already the right shape."""
    import numpy as np
    th, tw = int(target_hw[0]), int(target_hw[1])
    n = np.asarray(normals, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] < 3:
        raise ValueError(f"expected an (H,W,3) normal field, got {n.shape}")
    if n.shape[:2] == (th, tw):
        out = n[..., :3]
    else:
        try:
            PILImage = _require_pil()
            chans = []
            for c in range(3):
                im = PILImage.fromarray(np.ascontiguousarray(n[..., c]), mode="F")
                chans.append(np.asarray(im.resize((tw, th), PILImage.BILINEAR), dtype=np.float32))
            out = np.stack(chans, axis=-1)
        except Exception:
            ys = np.linspace(0, n.shape[0] - 1, th).astype(int)
            xs = np.linspace(0, n.shape[1] - 1, tw).astype(int)
            out = n[np.ix_(ys, xs)][..., :3].astype(np.float32)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norm, 1e-12)).astype(np.float32)


class AtlasMogeNormals:
    """🧭 Predicted surface normals from MoGe, DECOUPLED from the depth source.

    Wire BETWEEN AtlasDepthMap (any model) and AtlasCleanPlateLayer. Runs a MoGe
    ``*-normal`` model PURELY for its per-pixel normals, discards MoGe's own
    depth, and attaches those normals (resized to the input depth's resolution)
    onto a COPY of the input ATLAS_DEPTH_MAP. The clean-plate layer then embeds
    them as its world-normal relight map exactly as if MoGe had been the depth
    model — so you keep V2/DA3 depth (whose far-field behaves on exteriors, where
    MoGe's runs away) AND get MoGe's cleaner predicted normals for the lights.

    Reuses AtlasCleanPlateLayer's existing ``depth.normal`` channel — no new
    widget on that node (its capability freeze). The attach on the layer still
    requires ``frame_outpaint_px == 0`` there (an outpainted plate's normal map
    would be out of uv-registration with the widened plate). Pass-through (depth
    unchanged) if the chosen model returns no normals. Requires the [moge] extra.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "STRING")
    RETURN_NAMES = ("depth", "report")
    FUNCTION = "attach"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("ATLAS_DEPTH_MAP",),
                "image": ("IMAGE",),
            },
            "optional": {
                "normal_model": (list(_MOGE_NORMAL_MODEL_CHOICES),
                    {"default": "Ruicheng/moge-2-vitl-normal",
                     "tooltip": "MoGe *-normal checkpoint. vitl=best quality, vitb=lighter GPU, "
                     "vits=35M CPU/MPS-viable (non-CUDA). Auto-downloads from HuggingFace."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "solve": ("ATLAS_SOLVE", {"tooltip": "Optional — feeds the SOLVED focal to MoGe "
                          "(fov_x) for better geometry; the normals are aligned to the recovered "
                          "world frame downstream regardless, so this is a minor quality knob."}),
            },
        }

    def attach(self, depth, image, normal_model="Ruicheng/moge-2-vitl-normal",
               device="auto", solve=None):
        import copy
        base = getattr(depth, "depth", None)
        if base is None:
            return (depth, "AtlasMogeNormals: input depth carries no array — passed through unchanged.")
        from atlas_camera.inference.depth_estimator import estimate_depth
        tmp = _save_image_tensor_to_tmp(image)
        try:
            moge = estimate_depth(tmp, model_id=normal_model,
                                  device=None if device == "auto" else device,
                                  focal_px=_solve_focal_px_for_image(solve, image))
        finally:
            os.unlink(tmp)
        raw = getattr(moge, "normal", None)
        if raw is None:
            return (depth, f"AtlasMogeNormals: '{normal_model}' returned no normals — is it a "
                           "'*-normal' variant? Depth passed through unchanged (no relight normals).")
        import numpy as np
        target_hw = np.asarray(base).shape[:2]
        rn = _resize_normal_field(raw, target_hw)
        out = copy.copy(depth)            # new instance sharing arrays; override only .normal
        out.normal = rn
        report = ("AtlasMogeNormals: attached {model} normals resized to {hw} onto the depth map "
                  "(depth itself unchanged). Feed into AtlasCleanPlateLayer with frame_outpaint_px=0 "
                  "to embed them as the world-normal relight map.").format(
                      model=normal_model, hw=tuple(int(v) for v in target_hw))
        return (out, report)


class AtlasPredictHiddenGeometry:
    """🔬 EXPERIMENTAL, RESEARCH-ONLY — "X-ray" depth map via LaRI layered ray
    intersections.

    Predicts the surfaces HIDDEN behind foreground occluders (per pixel, the
    first ray intersection that clears the visible surface) and returns a
    patched copy of the input ATLAS_DEPTH_MAP with occluder pixels replaced by
    that predicted hidden depth — a depth map of "the world with the occluders
    removed". Wire the ORIGINAL depth into foreground band layers and this
    node's output into BACKGROUND band layers so disocclusion reveals get
    predicted geometry instead of diffusion-smoothed guesses.

    Hidden depth is a HYPOTHESIS, never a measurement: the report output
    carries registration quality + coverage, and `hidden_mask` marks every
    substituted pixel for provenance. Works best on indoor/architectural
    scenes (the model's training domain — see
    docs/dev/hidden_geometry_training_free_research.md); outdoor terrain can
    collapse to near-zero coverage, in which case the depth passes through
    almost unchanged.

    Requires a user-cloned LaRI repository (github.com/ruili3/lari — NO
    upstream license, research use only; atlas_camera bundles none of it).
    Point `lari_path` (or the ATLAS_LARI_PATH env var) at the clone.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("depth", "hidden_mask", "report", "paint_matte")
    FUNCTION = "predict"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("ATLAS_DEPTH_MAP",),
                "image": ("IMAGE",),
            },
            "optional": {
                "lari_path": ("STRING", {"default": "", "tooltip":
                    "Path to your clone of github.com/ruili3/lari (research-only, "
                    "unlicensed upstream). Blank = the ATLAS_LARI_PATH env var."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "clear_rel": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0,
                    "step": 0.01, "tooltip":
                    "A hidden layer must be at least this fraction of the visible "
                    "depth BEHIND it to count as a separate surface (occluder back "
                    "faces are closer than this and get skipped)."}),
                "min_clear_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0,
                    "step": 0.1, "tooltip":
                    "Absolute clearance floor in the depth map's units. 0 = auto "
                    "(2% of the median visible depth) — the scene-adaptive margin "
                    "shallow scenes need."}),
                "restrict_mask": ("MASK", {"tooltip":
                    "Optional — only substitute hidden depth inside this mask "
                    "(e.g. a foreground band's layer_mask). Without it, every "
                    "confidently-detected occluder is replaced."}),
                "model": (["lari-scene", "world-tracing-scene"],
                    {"default": "lari-scene", "tooltip":
                    "Layered-ray-intersection backend. lari-scene = LaRI (fast "
                    "regression, ~0.2s, unlicensed upstream). world-tracing-scene "
                    "= WT-DiT r69l (diffusion, ~17s/20 steps, CC BY-NC-ND 4.0, "
                    "HF-gated checkpoint). Both are research-only."}),
                "wt_path": ("STRING", {"default": "", "tooltip":
                    "Path to your clone of github.com/haoz19/world-tracing "
                    "(only used by the world-tracing-scene backend). Blank = the "
                    "ATLAS_WT_PATH env var."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip":
                    "Diffusion sampling steps (world-tracing backend only)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                    "tooltip": "Diffusion seed (world-tracing backend only — "
                    "WT is generative; pin this for reproducible hidden geometry)."}),
                "smooth_px": ("INT", {"default": 31, "min": 0, "max": 201,
                    "tooltip": "Gaussian-smooth the substituted hidden depth "
                    "(sigma ≈ 0.75×this, px). Layer-switch seams and fill-block "
                    "steps shred the downstream relief mesh via its world-edge "
                    "check (immune to depth_edge_rel — measured; and a MEDIAN "
                    "filter preserves exactly those steps, also measured). "
                    "0 = off."}),
                "fill_gaps": ("BOOLEAN", {"default": True,
                    "tooltip": "Diffuse the predictions across the WHOLE "
                    "restrict_mask region (needs restrict_mask wired): treats "
                    "scattered per-pixel predictions as samples of ONE coherent "
                    "hidden surface, so the X-ray layer meshes continuously "
                    "instead of shredding on fragmented masks (foliage). "
                    "Filled depth is clamped to stay BEHIND the visible surface."}),
            },
        }

    def predict(self, depth, image, lari_path="", device="auto",
                clear_rel=0.15, min_clear_m=0.0, restrict_mask=None,
                model="lari-scene", wt_path="", steps=20, seed=0,
                smooth_px=31, fill_gaps=True):
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.hidden_geometry import select_hidden_surface
        from atlas_camera.inference.depth_estimator import DepthResult

        tmp = _save_image_tensor_to_tmp(image)
        try:
            if model == "world-tracing-scene":
                from atlas_camera.inference.wt_hidden_geometry import (
                    predict_layered_depth_wt,
                )
                layered = predict_layered_depth_wt(
                    tmp, wt_path=wt_path,
                    device=None if device == "auto" else device,
                    steps=steps, seed=seed)
            else:
                from atlas_camera.inference.lari_hidden_geometry import (
                    predict_layered_depth,
                )
                layered = predict_layered_depth(
                    tmp, lari_path=lari_path,
                    device=None if device == "auto" else device)
        finally:
            os.unlink(tmp)

        raw = np.asarray(depth.depth, dtype=np.float64)
        H, W = raw.shape
        lt = torch.from_numpy(layered.layers).permute(2, 0, 1)[None]  # (1,L,h,w)
        layers_up = torch.nn.functional.interpolate(
            lt, size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).numpy().astype(np.float64)

        hidden, hidden_valid, stats = select_hidden_surface(
            layers_up, raw, clear_rel=clear_rel,
            min_clear=(min_clear_m if min_clear_m > 0 else None))

        region = None
        if restrict_mask is not None:
            m = restrict_mask
            if m.dim() == 3:
                m = m[0]
            m = torch.nn.functional.interpolate(
                m[None, None].float(), size=(H, W), mode="nearest"
            )[0, 0].numpy() > 0.5
            hidden_valid = hidden_valid & m
            stats["restricted_coverage"] = float(hidden_valid.mean())
            region = m & (raw > 1e-6)

        # Coherence pass (see the smooth_px/fill_gaps tooltips): fragmented
        # per-pixel predictions shred the downstream relief mesh via its
        # world-edge check, so (a) diffuse the predictions into ONE surface
        # across the restrict region, (b) median-smooth the layer-switch
        # seams, (c) clamp the result to stay BEHIND the visible surface.
        if fill_gaps and region is not None and hidden_valid.any():
            from atlas_camera.core.hidden_geometry import fill_hidden_gaps
            n_pred = int(hidden_valid.sum())
            hidden, hidden_valid = fill_hidden_gaps(hidden, hidden_valid, region)
            stats["filled_fraction"] = float(
                (int(hidden_valid.sum()) - n_pred) / max(int(hidden_valid.sum()), 1))
        if smooth_px and int(smooth_px) > 1 and hidden_valid.any():
            try:
                # GAUSSIAN, not median (calibrated 2026-07-09): median is
                # edge-preserving, so it kept the fill's block steps intact and
                # the mesh kept shredding (jungle hole-in-paint 0.455 median vs
                # 0.260 gaussian). The diffusion fill already handles outliers.
                from scipy.ndimage import gaussian_filter
                field = np.where(hidden_valid, hidden, raw)
                hidden = gaussian_filter(field, sigma=0.75 * float(smooth_px))
                stats["smooth_px"] = int(smooth_px)
            except ImportError:
                stats["warning_smooth"] = "scipy unavailable — smoothing skipped"
        # Geometry vs paint are SEPARATE concerns (jungle calibration lesson):
        # the substituted surface must stay CONTINUOUS to mesh (no clamping —
        # clamping filled depth out to a farther visible surface at see-through
        # gaps re-creates the metre-scale seams the fill just removed), while
        # PAINTING is only correct where the hidden surface is genuinely behind
        # a nearer occluder. paint_matte = those pixels; wire it into the
        # X-ray band's layer_matte so see-through gaps discard in the shader
        # (revealing the base mesh's real far content) without fragmenting
        # the geometry.
        paint = hidden_valid & (hidden > raw * 1.02)
        stats["paint_fraction"] = float(paint.mean())

        patched = raw.copy()
        patched[hidden_valid] = hidden[hidden_valid]

        scalar_stats = {k: v for k, v in stats.items()
                        if isinstance(v, (int, float, str))}
        backend = "world-tracing" if model == "world-tracing-scene" else "lari"
        # Provenance for the viewport's 🩻 debug overlay: WHICH pixels were
        # substituted and by WHICH backend, threaded (JSON-safe PNG data URI —
        # DepthResult.metadata must stay summary()-serializable) through
        # AtlasCleanPlateLayer into the ProjectionSource payload.
        provenance = {"hidden_backend": backend}
        if paint.any():
            # The 🩻 tint marks PAINTED hidden surface (paint matte), not the
            # full continuity-filled region — see the geometry-vs-paint note.
            hb64 = _mask_to_b64_png(paint)
            if hb64:
                provenance["hidden_mask_b64"] = hb64
        out = DepthResult(
            depth=patched.astype(np.float32),
            is_metric=depth.is_metric,
            model_id=f"{depth.model_id}+{backend}_hidden",
            image_width=depth.image_width,
            image_height=depth.image_height,
            near=float(patched.min()),
            far=float(patched.max()),
            metadata={**depth.metadata, "research_only": True, **provenance,
                      **{f"hidden_{k}": v for k, v in scalar_stats.items()}},
        )
        mask_t = torch.from_numpy(hidden_valid.astype(np.float32))[None]
        paint_t = torch.from_numpy(paint.astype(np.float32))[None]

        rel_mad = stats.get("registration_rel_mad", float("inf"))
        quality = ("good" if rel_mad < 0.2 else
                   "shaky" if rel_mad < 0.5 else "poor")
        backend_line = (
            "World Tracing r69l — CC BY-NC-ND 4.0, non-commercial; "
            f"diffusion steps {steps}, seed {seed}"
            if model == "world-tracing-scene"
            else "LaRI — upstream repo has NO license; do not use commercially"
        )
        report = (
            f"🔬 RESEARCH-ONLY hidden-geometry prediction ({backend_line}).\n"
            f"registration: scale {stats.get('scale', 0):.3f}, rel MAD "
            f"{rel_mad:.3f} ({quality})\n"
            f"substituted pixels: {int(hidden_valid.sum())} "
            f"({100.0 * float(hidden_valid.mean()):.1f}% of frame)\n"
            f"median hidden-vs-visible separation: "
            f"{stats.get('median_separation') if stats.get('median_separation') is not None else 'n/a'}\n"
            f"layer histogram (index of first clearing layer): "
            f"{stats.get('layer_used_histogram')}\n"
            + ("warning: " + stats["warning"] + "\n" if "warning" in stats else "")
            + ("warning: no restrict_mask wired — substitution covers "
               f"{100.0 * float(hidden_valid.mean()):.0f}% of the frame, "
               "including VISIBLE background surfaces (LaRI predicts "
               "through-wall structure there). For band workflows wire the "
               "foreground band's layer_mask into restrict_mask so only real "
               "occluders are replaced.\n"
               if restrict_mask is None and float(hidden_valid.mean()) > 0.25
               else "")
            + "Hidden depth is a hypothesis — best on indoor/architectural "
              "scenes; verify by orbiting the projected result."
        )
        return (out, mask_t, report, paint_t)


class AtlasRenderFix:
    """🔬 EXPERIMENTAL — repair projected-render artifacts with NVIDIA Fixer.

    Runs the pretrained Fixer model (single-step diffusion, the Difix3D+
    successor, trained to fix rendered-novel-view artifacts) over an IMAGE
    batch — typically `AtlasBlockoutViewport`'s baked `path_frames` before a
    Video Combine node. Spike-verified on this repo's own baked orbits
    (2026-07-10): fills ~1/3 of hard black tear pixels on a bare relief
    mesh, softens stretched-texel smears on the full DMP rig, adds no
    temporal flicker, ~0.3–0.45 s/frame on an RTX 5090 (plus ~1 min model
    load/warmup per queue). Costs/limits: mild overall softening (single-step
    regeneration at an internal 576×1024), and it does NOT outpaint large
    frame-edge reveals — band-layer frame outpainting stays the answer there.

    Unlike the LaRI/WT experimental nodes this runs in a DOCKER CONTAINER
    (the cosmos/transformer_engine stack has no native Windows build):
    build the image once from docker/fixer/Dockerfile, clone
    github.com/nv-tlabs/Fixer (Apache-2.0) with its weights (NVIDIA Open
    Model License — commercial use permitted), and point `fixer_path` (or
    ATLAS_FIXER_PATH) at the clone. See INSTALL.md 'Experimental: Fixer
    Render Repair'. Fails loud with actionable errors when docker/image/
    weights are missing.
    """
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "fix"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip":
                    "Frames to repair — e.g. AtlasBlockoutViewport's baked "
                    "path_frames. Fixer works internally at 576×1024; frames "
                    "near that resolution round-trip with the least "
                    "softening."}),
            },
            "optional": {
                "fixer_path": ("STRING", {"default": "", "tooltip":
                    "Path to your clone of github.com/nv-tlabs/Fixer with "
                    "weights downloaded into models/ (hf download nvidia/Fixer "
                    "--local-dir models). Blank = the ATLAS_FIXER_PATH env "
                    "var."}),
                "docker_image": ("STRING", {"default": "fixer-spike-env",
                    "tooltip": "Inference container image — build once with: "
                    "docker build -t fixer-spike-env -f docker/fixer/Dockerfile "
                    "docker/fixer/"}),
                "timestep": ("INT", {"default": 250, "min": 1, "max": 999,
                    "tooltip": "Fixer's single denoising timestep (upstream "
                    "default 250; the older difix checkpoint used 199)."}),
                "timeout_s": ("INT", {"default": 900, "min": 60, "max": 7200,
                    "tooltip": "Kill the container after this many seconds. "
                    "Budget ~1 min load/warmup + ~0.5 s/frame."}),
            },
        }

    def fix(self, images, fixer_path="", docker_image="fixer-spike-env",
            timestep=250, timeout_s=900):
        import shutil
        import time
        np = _require_numpy()
        torch = _require_torch()
        PILImage = _require_pil()
        from atlas_camera.inference.fixer_render_fix import (
            resolve_fixer_root, run_fixer_on_dir,
        )

        root = resolve_fixer_root(fixer_path)
        exchange = Path(tempfile.mkdtemp(prefix="atlas_fixer_"))
        in_dir = exchange / "in"
        out_dir = exchange / "out"
        in_dir.mkdir()
        try:
            frames = images.cpu().numpy()  # (B,H,W,3) float 0-1
            for i in range(frames.shape[0]):
                arr = (frames[i] * 255.0).clip(0, 255).astype("uint8")
                PILImage.fromarray(arr, mode="RGB").save(
                    in_dir / f"frame_{i:05d}.png")
            t0 = time.time()
            log_tail = run_fixer_on_dir(
                in_dir, out_dir, root, docker_image=docker_image,
                timestep=timestep, timeout_s=timeout_s)
            elapsed = time.time() - t0
            outs = sorted(out_dir.glob("*.png"))
            fixed = []
            for i, f in enumerate(outs):
                arr = np.array(PILImage.open(f).convert("RGB"),
                               dtype=np.float32) / 255.0
                # Fixer returns input resolution, but guard against drift so a
                # mismatched frame can't crash the stack() below.
                if arr.shape[:2] != frames.shape[1:3]:
                    pil = PILImage.fromarray(
                        (arr * 255).astype("uint8")).resize(
                        (frames.shape[2], frames.shape[1]), PILImage.LANCZOS)
                    arr = np.array(pil, dtype=np.float32) / 255.0
                fixed.append(arr)
            out_t = torch.from_numpy(np.stack(fixed, axis=0))
            report = (
                "🔬 EXPERIMENTAL Fixer render repair (weights: NVIDIA Open "
                "Model License; single-step diffusion in Docker).\n"
                f"{len(fixed)} frame(s) at "
                f"{frames.shape[2]}x{frames.shape[1]} in {elapsed:.1f}s "
                f"({elapsed / max(len(fixed), 1):.2f}s/frame incl. "
                f"load+warmup), timestep {timestep}.\n"
                "Known costs: mild softening; large frame-edge reveals are "
                "not outpainted (use band-layer frame outpainting for "
                "those).\n--- container log tail ---\n" + log_tail
            )
            return (out_t, report)
        finally:
            shutil.rmtree(exchange, ignore_errors=True)


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
                               "SAM/RMBG) which REPLACES the internal sky heuristic before "
                               "triangulation - so it must cover EVERYTHING you want gone. Any "
                               "resolution - resized to match depth."}),
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "World-space edge tear threshold: a quad tears when its world edge "
                               "exceeds this x the expected local sample spacing. SEPARATE from "
                               "depth_edge_rel, and often the DOMINANT tear cause on deep / "
                               "narrow-FOV / interior scenes, where grazing walls and receding "
                               "floors span large world distances between adjacent samples and "
                               "trip the default 12x even where the surface is continuous. Raise "
                               "(20-40) to close spurious 'comb' tears; too high (>80) rubber-"
                               "sheets real foreground silhouettes onto the background."}),
                "sky_heuristic": ("BOOLEAN", {"default": True,
                    "tooltip": "Exclude above-horizon far/rough regions as sky before "
                               "triangulation. Correct for OUTDOOR plates; turn OFF for INTERIORS "
                               "(it otherwise eats the ceiling / vault / far wall as 'sky', "
                               "punching large holes). Automatically off when exclude_mask is "
                               "wired (an explicit mask always governs)."}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "0 = off. When set, a THIRD tear test: a triangle tears when its "
                               "corner surface-normals bend by more than this angle. Unlike "
                               "max_edge_factor (which trips on ANY grazing/receding surface), "
                               "this fires only where the surface ORIENTATION changes sharply - a "
                               "real crease or occlusion silhouette - so it tears genuine edges "
                               "while leaving a smoothly-receding wall/floor intact. Pair it with "
                               "a HIGHER max_edge_factor: raise mef to stop comb-tearing continuous "
                               "grazing surfaces, then set ~40-70 here to keep real silhouettes "
                               "torn. Lower = tears more readily."}),
            },
        }

    _RELIEF_QUALITY_PRESETS = {"low": 64, "medium": 256, "high": 512, "ultra": 1024}

    def derive(self, solve, depth, relief_grid=128, relief_quality="custom", depth_edge_rel=0.5,
               exclude_mask=None, max_edge_factor=12.0, sky_heuristic=True, normal_edge_deg=0.0):
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
            max_edge_factor=float(max_edge_factor),
            normal_edge_deg=(float(normal_edge_deg) if float(normal_edge_deg) > 0 else None),
            apply_sky_heuristic=(resolved_exclude is None) and bool(sky_heuristic))
        prims = [backdrop, relief_mesh_primitive(mesh)]
        stats = {
            "ground_scale": scale, "ground_fit": ground_info,
            "relief_mesh": {"n_vertices": mesh.stats["n_vertices"], "n_faces": mesh.stats["n_faces"]},
        }
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "relief_grid": int(relief_grid), "relief_quality": relief_quality,
            "depth_edge_rel": float(depth_edge_rel), "max_edge_factor": float(max_edge_factor),
            "sky_heuristic": bool(sky_heuristic), "normal_edge_deg": float(normal_edge_deg),
            "derive_node": "AtlasDeriveReliefMesh",
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
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 32,
                    "tooltip": "Max foreground boxes/cylinders (e.g. buildings, in an "
                               "aerial/top-down shot). 0 = walls/ground/backdrop only."}),
                "distance_modes": ("INT", {"default": 1, "min": 1, "max": 16,
                    "tooltip": "Walls per azimuth DIRECTION. 1 = classic: one plane at "
                               "the median distance of everything facing that way. A "
                               "street-grid skyline has ~2 facing directions but many "
                               "depths — raise this (with max_walls) so each direction "
                               "splits into one wall per depth mode (building row) "
                               "instead of collapsing the skyline into one slab."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Remove these pixels from wall/object fitting (e.g. a SAM "
                               "segment of everything EXCEPT one building — invert per "
                               "branch to scope each derive to one structure, then chain "
                               "AtlasMergeGeometry). Ground fit/scale/backdrop stay "
                               "full-frame so masked branches share one metric world."}),
                "ground_anchor": ("BOOLEAN", {"default": False,
                    "tooltip": "Wall DISTANCE from ray-through-base-pixel x the analytic "
                               "Y=0 ground plane — pure geometry, immune to monocular "
                               "depth's low-frequency 'banana' warp on tall structures. "
                               "Assumes the building's ground contact is VISIBLE: for "
                               "best accuracy inpaint cars/fences off the ground line "
                               "before solving (most street/architectural photos show "
                               "enough contact as-is; occluded bases are detected and "
                               "fall back to the classic depth-median distance)."}),
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=3, distance_modes=1,
               exclude_mask=None, ground_anchor=False):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_projection_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        cfg = ProxyDerivationConfig(max_objects=int(max_objects),
                                    wall_distance_modes=int(distance_modes),
                                    ground_anchor=bool(ground_anchor))
        prims, stats = derive_projection_proxies(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_walls=int(max_walls), horizon_y=horizon_y, config=cfg,
            exclude_mask=_resolve_exclude_mask(exclude_mask, height, width))
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "azimuth_walls", "derive_node": "AtlasDeriveWalls",
            "distance_modes": int(distance_modes),
            "ground_anchor": bool(ground_anchor),
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
                "max_walls": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_objects": ("INT", {"default": 3, "min": 0, "max": 32,
                                        "tooltip": "Max foreground boxes/cylinders. Street-level scenes: try 0 — the 2D occupancy clustering merges cars/fences/trees into oversized near-camera boxes that dominate any orbit."}),
                "distance_modes": ("INT", {"default": 1, "min": 1, "max": 16,
                    "tooltip": "Walls per azimuth DIRECTION. 1 = classic: one plane at "
                               "the median distance of everything facing that way. A "
                               "street-grid skyline has ~2 facing directions but many "
                               "depths — raise this (with max_walls) so each direction "
                               "splits into one wall per depth mode (building row) "
                               "instead of collapsing the skyline into one slab."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Remove these pixels from wall/object fitting (e.g. a SAM "
                               "segment of everything EXCEPT one building — invert per "
                               "branch to scope each derive to one structure, then chain "
                               "AtlasMergeGeometry). Ground fit/scale/backdrop stay "
                               "full-frame so masked branches share one metric world."}),
                "ground_anchor": ("BOOLEAN", {"default": False,
                    "tooltip": "Wall DISTANCE from ray-through-base-pixel x the analytic "
                               "Y=0 ground plane — pure geometry, immune to monocular "
                               "depth's low-frequency 'banana' warp on tall structures. "
                               "Assumes the building's ground contact is VISIBLE: for "
                               "best accuracy inpaint cars/fences off the ground line "
                               "before solving (most street/architectural photos show "
                               "enough contact as-is; occluded bases are detected and "
                               "fall back to the classic depth-median distance)."}),
                "roofline_split": ("BOOLEAN", {"default": False,
                    "tooltip": "Split each wall cluster at silhouette-height steps: a "
                               "row of buildings becomes one plane per roofline (each "
                               "cut to its own top, and with ground_anchor each gets "
                               "its own footprint distance) instead of one rectangle "
                               "spanning sky above the shorter buildings."}),
            },
        }

    def derive(self, solve, depth, max_walls=4, max_objects=3, distance_modes=1,
               exclude_mask=None, ground_anchor=False, roofline_split=False):
        from atlas_camera.core.proxy_geometry import ProxyDerivationConfig, derive_vertical_extrusion_proxies
        params = _solve_camera_params(solve, depth)
        if params is None:
            return (solve,)
        width, height, fx, fy, cx, cy = params
        depth_map = _depth_map_for_solve(depth, width, height)
        horizon_y = _horizon_y_from_solve(solve)
        extr = solve.camera.extrinsics
        cfg = ProxyDerivationConfig(max_objects=int(max_objects),
                                    wall_distance_modes=int(distance_modes),
                                    ground_anchor=bool(ground_anchor),
                                    roofline_split=bool(roofline_split))
        prims, stats = derive_vertical_extrusion_proxies(
            depth_map, view_matrix=extr.camera_view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
            max_walls=int(max_walls), horizon_y=horizon_y, config=cfg,
            exclude_mask=_resolve_exclude_mask(exclude_mask, height, width))
        out = _replace_proxy_role_geometry(solve, prims, stats, {
            "primitive_method": "vertical_extrusion", "derive_node": "AtlasDeriveTowersSpires",
            "distance_modes": int(distance_modes),
            "ground_anchor": bool(ground_anchor),
            "roofline_split": bool(roofline_split),
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


def _parse_exact_view(text):
    """Parse an EXACT orbit delta — "azimuth_deg=<f> elevation_deg=<f>
    distance_scale=<f>", the string 📐 Extract Angle's `patch_exact` output
    emits (raw measured floats, BEFORE named-view snapping). Returns
    (d_azimuth_deg, d_elevation_deg, distance_scale) or None when the text
    doesn't carry all three keys.

    The render-conditioned patch loop needs this precision: a frame baked at
    the artist's real orbit must be projected back from the IDENTICAL pose —
    snapping to the LoRA's 45° azimuth grid would misregister the projection.
    Key=value format (any order, comma or space separated) so the string is
    self-documenting in Show Text nodes and export logs.
    """
    import re

    vals = dict(re.findall(
        r"(azimuth_deg|elevation_deg|distance_scale)\s*=\s*(-?\d+(?:\.\d+)?)",
        text or ""))
    if set(vals) != {"azimuth_deg", "elevation_deg", "distance_scale"}:
        return None
    return (float(vals["azimuth_deg"]), float(vals["elevation_deg"]),
            float(vals["distance_scale"]))


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


class AtlasExtractAnglePatch:
    """Write a Photoshop-friendly patch package from an extracted viewport angle.

    This is the MVP bridge for the ``Extract Angle`` control. The incoming
    ``plate_image`` is normally the viewport's shaded/projection render and
    ``matte`` is the artist-selected region to repair. The node crops both to
    one padded rectangle, writes image/matte/depth/normal passes plus a JSON
    sidecar containing the exact orbit string and source solve, and returns a
    typed package for :class:`AtlasImportAnglePatch`.

    It deliberately does not invent a new camera: ``patch_exact`` is preserved
    byte-for-byte so the downstream ``AtlasAddPatchView.exact_view_override``
    can reconstruct the same pose after Photoshop round-tripping.
    """
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "ATLAS_PATCH")
    RETURN_NAMES = ("patch_image", "patch_matte", "manifest_path", "patch_package")
    FUNCTION = "extract"
    CATEGORY = "Atlas Camera/Patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "plate_image": ("IMAGE",),
                "matte": ("MASK",),
                "patch_exact": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "atlas_exports/angle_patches"}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "normal": ("IMAGE",),
                "name": ("STRING", {"default": "angle_patch"}),
                "padding_px": ("INT", {"default": 128, "min": 0, "max": 2048}),
                "colorspace": (["ACEScg", "sRGB - Display"], {"default": "ACEScg"}),
            },
        }

    def extract(self, solve, plate_image, matte, patch_exact, output_dir,
                depth=None, normal=None, name="angle_patch", padding_px=128,
                colorspace="ACEScg"):
        np = _require_numpy()
        torch = _require_torch()
        PILImage = _require_pil()
        if not patch_exact or not patch_exact.strip():
            raise ValueError("patch_exact is empty; click Extract Angle before exporting a patch.")
        if plate_image.ndim != 4 or plate_image.shape[0] < 1:
            raise ValueError("plate_image must be a non-empty ComfyUI IMAGE batch.")
        rgb = plate_image[0].detach().cpu().numpy().clip(0.0, 1.0)
        mask_arr = matte[0].detach().cpu().numpy().clip(0.0, 1.0)
        if mask_arr.shape != rgb.shape[:2]:
            raise ValueError("matte dimensions must match plate_image dimensions.")
        ys, xs = np.where(mask_arr > 1.0 / 255.0)
        if len(xs) == 0:
            raise ValueError("matte contains no non-zero pixels; select the Photoshop repair region first.")
        pad = max(0, int(padding_px))
        y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "angle_patch")).strip("._") or "angle_patch"
        root = Path(output_dir).expanduser().resolve() / safe_name
        root.mkdir(parents=True, exist_ok=True)

        def save_rgb(arr, path):
            PILImage.fromarray((arr * 255.0).clip(0, 255).astype("uint8"), mode="RGB").save(path, format="PNG")

        patch_rgb = rgb[y0:y1, x0:x1]
        patch_mask = mask_arr[y0:y1, x0:x1]
        image_path = root / "patch.png"
        matte_path = root / "patch_matte.png"
        save_rgb(patch_rgb, image_path)
        PILImage.fromarray((patch_mask * 255.0).clip(0, 255).astype("uint8"), mode="L").save(matte_path, format="PNG")
        # The FULL frame is required for reprojection: AtlasAddPatchView's
        # ProjectionSource samples uv across the whole patch-camera frustum, so
        # the import node must paste the edited crop back into this frame — a
        # bare crop fed downstream would stretch across the frustum and
        # misregister. The crop exists purely as the Photoshop convenience.
        full_path = root / "plate_full.png"
        save_rgb(rgb, full_path)

        pass_paths = {"image": str(image_path), "matte": str(matte_path),
                      "plate_full": str(full_path)}
        for label, tensor in (("depth", depth), ("normal", normal)):
            if tensor is not None:
                arr = tensor[0].detach().cpu().numpy().clip(0.0, 1.0)
                pass_path = root / f"patch_{label}.png"
                save_rgb(arr[y0:y1, x0:x1], pass_path)
                pass_paths[label] = str(pass_path)

        # Camera block only — never the full solve: a layered solve's to_dict()
        # carries megabytes of base64 plates and would balloon the sidecar.
        try:
            camera_dict = solve.camera.to_dict()
        except Exception:
            camera_dict = {}
        from atlas_camera import __version__ as _atlas_version
        manifest = {
            "schema": 1,
            "kind": "atlas_angle_patch",
            "atlas_version": _atlas_version,
            "patch_exact": patch_exact.strip(),
            "source_camera": camera_dict,
            "crop_bbox_xyxy": [x0, y0, x1, y1],
            "padding_px": pad,
            "image_wh": [int(x1 - x0), int(y1 - y0)],
            "full_wh": [int(rgb.shape[1]), int(rgb.shape[0])],
            "colorspace_intent": colorspace,
            "colorspace_written": "sRGB 8-bit PNG (proxy/LDR viewport plate; EXR is the planned float path)",
            "premultiplied": False,
            "photoshop_roundtrip": {
                "edit_image": "patch.png",
                "preserve_matte": "patch_matte.png",
                "write_back_as": "patch_edited.png",
            },
            "passes": pass_paths,
        }
        manifest_path = root / "atlas_angle_patch.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        package = {"manifest": str(manifest_path), "passes": pass_paths, "patch_exact": patch_exact.strip(), "crop_bbox_xyxy": manifest["crop_bbox_xyxy"]}
        return (_pil_to_image_tensor(PILImage.fromarray((patch_rgb * 255).astype("uint8"), mode="RGB")),
                torch.from_numpy(patch_mask.astype("float32")).unsqueeze(0), str(manifest_path), package)


class AtlasImportAnglePatch:
    """Load an edited angle patch, paste it back into the FULL frame, and
    expose the exact pose for reprojection.

    The extraction crop is a Photoshop convenience only — reprojection needs
    the full frame, because ``AtlasAddPatchView``'s ProjectionSource samples
    uv across the whole patch-camera frustum (a bare crop would stretch
    across the frustum and misregister). This node loads ``plate_full.png``,
    pastes the edited crop at the manifest's ``crop_bbox_xyxy``, and returns
    FULL-FRAME image and matte tensors.

    Wire ``patch_image`` into ``AtlasAddPatchView.patch_image`` and
    ``patch_exact`` into its ``exact_view_override`` input. This keeps the
    Photoshop edit in the same camera frame that produced the extraction.
    """
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "ATLAS_PATCH")
    RETURN_NAMES = ("patch_image", "patch_matte", "patch_exact", "patch_package")
    FUNCTION = "import_patch"
    CATEGORY = "Atlas Camera/Patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"patch_package": ("ATLAS_PATCH",)},
            "optional": {
                "edited_image": ("IMAGE", {"tooltip": "Optional Photoshop-edited CROP (same size as patch.png); otherwise patch.png is loaded."}),
                "edited_matte": ("MASK", {"tooltip": "Optional edited CROP matte; otherwise patch_matte.png is loaded."}),
            },
        }

    def import_patch(self, patch_package, edited_image=None, edited_matte=None):
        np = _require_numpy()
        torch = _require_torch()
        PILImage = _require_pil()
        if not isinstance(patch_package, dict) or not patch_package.get("manifest"):
            raise ValueError("patch_package is not an Atlas angle-patch package.")
        manifest_path = Path(str(patch_package["manifest"])).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Atlas angle-patch manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "atlas_angle_patch":
            raise ValueError("manifest kind is not atlas_angle_patch")
        passes = manifest.get("passes", {})
        bbox = manifest.get("crop_bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise ValueError("angle patch manifest has no crop_bbox_xyxy — re-extract with a current Atlas.")
        x0, y0, x1, y1 = (int(v) for v in bbox)

        full_path = Path(passes.get("plate_full", ""))
        if not full_path.is_file():
            raise FileNotFoundError(
                "plate_full.png missing from the patch package — reprojection "
                "needs the full frame to paste the edited crop into. "
                "Re-extract with a current Atlas.")
        full = np.asarray(PILImage.open(full_path).convert("RGB"), dtype=np.float32) / 255.0

        if edited_image is None:
            image_path = Path(passes.get("image", ""))
            if not image_path.is_file():
                raise FileNotFoundError("No edited_image was supplied and patch.png is missing.")
            crop = np.asarray(PILImage.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        else:
            crop = edited_image[0].detach().cpu().numpy()[..., :3].clip(0.0, 1.0)
        want_hw = (y1 - y0, x1 - x0)
        if crop.shape[:2] != want_hw:
            raise ValueError(
                f"edited patch is {crop.shape[1]}x{crop.shape[0]} but the extraction "
                f"crop was {want_hw[1]}x{want_hw[0]} — Photoshop must not resize the "
                "canvas (crop/uncrop changes registration).")
        full[y0:y1, x0:x1] = crop
        image_tensor = torch.from_numpy(full.astype("float32")).unsqueeze(0)

        full_mask = np.zeros(full.shape[:2], dtype=np.float32)
        if edited_matte is None:
            matte_path = Path(passes.get("matte", ""))
            if not matte_path.is_file():
                raise FileNotFoundError("No edited_matte was supplied and patch_matte.png is missing.")
            crop_mask = np.asarray(PILImage.open(matte_path).convert("L"), dtype=np.float32) / 255.0
        else:
            crop_mask = edited_matte[0].detach().cpu().numpy().clip(0.0, 1.0)
        if crop_mask.shape != want_hw:
            raise ValueError(
                f"edited matte is {crop_mask.shape[1]}x{crop_mask.shape[0]} but the "
                f"extraction crop was {want_hw[1]}x{want_hw[0]}.")
        full_mask[y0:y1, x0:x1] = crop_mask
        matte_tensor = torch.from_numpy(full_mask).unsqueeze(0)

        exact = str(manifest.get("patch_exact", "")).strip()
        if not exact:
            raise ValueError("angle patch manifest has no patch_exact camera pose.")
        package = dict(patch_package)
        package["manifest_data"] = manifest
        package["imported"] = True
        return image_tensor, matte_tensor, exact, package


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
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
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
                "exact_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_exact output here "
                               "('azimuth_deg=.. elevation_deg=.. distance_scale=..' — 📐's RAW "
                               "measured orbit, before named-view snapping). When connected it "
                               "WINS over patch_view_override AND the dropdowns, and "
                               "flip_azimuth is ignored (the raw delta is already in "
                               "orbit_camera's own convention). This is the render-conditioned "
                               "patch loop's channel: a frame baked at the artist's real orbit "
                               "(then repaired by AtlasRenderFix) must project back from the "
                               "IDENTICAL pose — the 45° named-view grid would misregister it. "
                               "Errors loudly if unparseable."}),
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
                  primary_depth=None, exclude_mask=None, geometry_source="reuse_scene",
                  exact_view_override=""):
        exact_delta = None
        if exact_view_override and exact_view_override.strip():
            exact_delta = _parse_exact_view(exact_view_override)
            if exact_delta is None:
                raise ValueError(
                    f"exact_view_override {exact_view_override!r} does not parse as "
                    "'azimuth_deg=<f> elevation_deg=<f> distance_scale=<f>' — wire "
                    "AtlasBlockoutViewport's patch_exact output here, or disconnect it."
                )
        elif patch_view_override and patch_view_override.strip():
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
        # An exact override (📐's raw measured floats) is already that delta in
        # orbit_camera's own convention — no view arithmetic, no flip.
        if exact_delta is not None:
            d_azimuth, d_elevation, distance_scale = exact_delta
        else:
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
                                        device=None if device == "auto" else device,
                                        focal_px=pfx)  # patch-image pixels
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
                    up = np.zeros_like(coverage)
                    up[:-1, :] = coverage[1:, :]
                    dn = np.zeros_like(coverage)
                    dn[1:, :] = coverage[:-1, :]
                    lf = np.zeros_like(coverage)
                    lf[:, :-1] = coverage[:, 1:]
                    rt = np.zeros_like(coverage)
                    rt[:, 1:] = coverage[:, :-1]
                    coverage = coverage | up | dn | lf | rt
                unseen = ~coverage
                if resolved_exclude is not None:
                    unseen &= ~resolved_exclude  # never paint sky onto geometry
                for _ in range(int(unseen_dilate_px)):
                    up = np.zeros_like(unseen)
                    up[:-1, :] = unseen[1:, :]
                    dn = np.zeros_like(unseen)
                    dn[1:, :] = unseen[:-1, :]
                    lf = np.zeros_like(unseen)
                    lf[:, :-1] = unseen[:, 1:]
                    rt = np.zeros_like(unseen)
                    rt[:, 1:] = unseen[:, :-1]
                    unseen = unseen | up | dn | lf | rt
                mask_b64 = _mask_to_b64_png(unseen) or None
            return self._finish_patch(
                solve, patch_image, patch_intr, patch_extr, patch_geom, mesh,
                mask_b64, plate_ref, name, priority,
                d_azimuth, d_elevation, distance_scale,
                patch_azimuth_view, patch_elevation_view, patch_distance,
                source_azimuth_view, flip_azimuth, pivot, depth_model,
                scale_source, scale, fallback_reason, exact_view_override,
                exact_delta)

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
                up = np.zeros_like(unseen)
                up[:-1, :] = unseen[1:, :]
                dn = np.zeros_like(unseen)
                dn[1:, :] = unseen[:-1, :]
                lf = np.zeros_like(unseen)
                lf[:, :-1] = unseen[:, 1:]
                rt = np.zeros_like(unseen)
                rt[:, 1:] = unseen[:, :-1]
                unseen = unseen | up | dn | lf | rt
            mask_b64 = _mask_to_b64_png(unseen) or None

        return self._finish_patch(
            solve, patch_image, patch_intr, patch_extr, patch_geom, mesh,
            mask_b64, plate_ref, name, priority,
            d_azimuth, d_elevation, distance_scale,
            patch_azimuth_view, patch_elevation_view, patch_distance,
            source_azimuth_view, flip_azimuth, pivot, depth_model,
            scale_source, scale, fallback_reason, exact_view_override,
            exact_delta)

    def _finish_patch(self, solve, patch_image, patch_intr, patch_extr,
                      patch_geom, mesh, mask_b64, plate_ref, name, priority,
                      d_azimuth, d_elevation, distance_scale,
                      patch_azimuth_view, patch_elevation_view, patch_distance,
                      source_azimuth_view, flip_azimuth, pivot, depth_model,
                      scale_source, scale, fallback_reason,
                      exact_view_override="", exact_delta=None):
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
            "source": ("exact_render_patch" if exact_delta is not None
                       else "multi_angle_lora_patch"),
            "patch_azimuth_view": patch_azimuth_view,
            "patch_elevation_view": patch_elevation_view,
            "patch_distance": patch_distance,
            "source_azimuth_view": source_azimuth_view,
            "exact_view_override": (exact_view_override.strip()
                                    if exact_delta is not None else None),
            "flip_azimuth": bool(flip_azimuth) if exact_delta is None else None,
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
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
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
                "exact_view_override": ("STRING", {"forceInput": True,
                    "tooltip": "Optional: wire AtlasBlockoutViewport's patch_exact output here "
                               "(📐's RAW orbit floats) — wins over patch_view_override and the "
                               "dropdowns, flip_azimuth ignored, placing this mask's target "
                               "camera IDENTICALLY to an AtlasAddPatchView driven by the same "
                               "string (the shared never-drift contract). Errors loudly if "
                               "unparseable."}),
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
                 patch_view_override="", exact_view_override=""):
        exact_delta = None
        if exact_view_override and exact_view_override.strip():
            exact_delta = _parse_exact_view(exact_view_override)
            if exact_delta is None:
                raise ValueError(
                    f"exact_view_override {exact_view_override!r} does not parse as "
                    "'azimuth_deg=<f> elevation_deg=<f> distance_scale=<f>' — wire "
                    "AtlasBlockoutViewport's patch_exact output here, or disconnect it."
                )
        elif patch_view_override and patch_view_override.strip():
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

        if exact_delta is not None:
            d_azimuth, d_elevation, distance_scale = exact_delta
        else:
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
                                    device=None if device == "auto" else device,
                                    focal_px=tfx)  # target-image pixels
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


def _format_hole_fill_report(enabled, n_filled, filled, faces_added, loops_left,
                             max_hole_edges, near_m, far_m):
    """Human-readable summary of an interior hole fill, for the export node.

    The fill is export-only, so this and ``preview_solve`` are the ONLY ways an
    artist learns what it did without a DCC round-trip — state the scope that
    was actually applied, not just the counts, since a disappointing result is
    usually a too-tight scope rather than a failed fill.
    """
    if not enabled:
        return "🔧 interior hole fill: off"
    lines = ["🔧 interior hole fill: ON"]
    if n_filled:
        lo, hi = min(filled), max(filled)
        span = f"{lo} edges" if lo == hi else f"{lo}–{hi} edges"
        lines.append(f"  filled {n_filled} hole{'s' if n_filled != 1 else ''} "
                     f"({span}, +{faces_added} faces)")
    else:
        lines.append("  filled 0 holes — nothing matched the scope below")
    # The outer frame is always one of these and must stay open by design.
    lines.append(f"  still open: {loops_left} boundary loop"
                 f"{'s' if loops_left != 1 else ''} (the outer frame is one)")
    scope = [f"max_hole_edges={int(max_hole_edges)}"]
    if near_m > 0.0 and far_m > 0.0:
        scope.append(f"band box {near_m:g}–{far_m:g} m")
    else:
        scope.append("band box off (set BOTH bounds > 0)")
    lines.append("  scope: " + ", ".join(scope))
    return "\n".join(lines)


def _solve_with_relief_mesh(solve, mesh):
    """A deep copy of ``solve`` whose relief-mesh primitive is ``mesh``.

    Lets the export node hand the viewport the geometry it ACTUALLY wrote,
    without touching the input solve (whose live projection mesh keeps its
    deliberate tears).
    """
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    out = copy.deepcopy(solve)
    scene = getattr(out, "projection_scene", None)
    if scene is None:
        return out
    prim = relief_mesh_primitive(mesh)
    prims = list(getattr(scene, "proxy_geometry", None) or [])
    for i, p in enumerate(prims):
        if p.primitive_type == "mesh" and (p.metadata or {}).get("source") == "depth_relief_mesh":
            prims[i] = prim
            break
    else:
        prims.append(prim)
    scene.proxy_geometry = prims
    return out


def _relief_mesh_from_solve(solve):
    """The relief mesh already derived onto a solve (AtlasDeriveReliefMesh /
    AtlasInput), reconstructed for export so its edge tuning carries over
    exactly. Looks for the ``depth_relief_mesh``-sourced primitive in the
    projection scene; returns a ReliefMesh, or None when the solve carries no
    relief mesh (bare solve, or a bands-only / primitives-only solve)."""
    from atlas_camera.exporters._layers import mesh_from_primitive
    scene = getattr(solve, "projection_scene", None)
    prims = (getattr(scene, "proxy_geometry", None) or []) if scene is not None else []
    for p in prims:
        meta = p.metadata or {}
        if p.primitive_type == "mesh" and meta.get("source") == "depth_relief_mesh":
            return mesh_from_primitive(p)
    return None


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
    RETURN_TYPES = ("STRING", "STRING", "ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("obj_path", "glb_path", "preview_solve", "report")
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
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "format": (["both", "obj", "glb"], {"default": "both"}),
                "use_solve_mesh": ("BOOLEAN", {"default": True,
                    "tooltip": "Export the relief mesh ALREADY on the solve (from "
                               "AtlasDeriveReliefMesh / AtlasInput) so ALL its edge tuning — "
                               "max_edge_factor, normal_edge_deg, the band near-clip, sky_heuristic "
                               "— carries into the OBJ/GLB exactly, with no widget to re-set. Turn "
                               "OFF to re-derive from depth at this node's own grid/thresholds "
                               "below (e.g. to export a HIGHER-resolution mesh than the viewport). "
                               "Auto-falls-back to re-derive when the solve carries no relief mesh."}),
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Re-derive only (use_solve_mesh off): world-space edge tear "
                               "threshold. Raise to 40-80 on deep/interior scenes to stop combs."}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Re-derive only: 0 = off; tears on surface-normal bend (real creases) "
                               "while leaving smooth grazing surfaces intact."}),
                "fill_interior_holes": ("BOOLEAN", {"default": False,
                    "tooltip": "EXPORT-ONLY (the live viewport projection mesh is never touched). "
                               "Fan-fill small interior tear holes in the OBJ/GLB so it retopologizes "
                               "and booleans cleanly in a DCC. Fills ONLY interior enclosed boundary "
                               "loops — never the outer silhouette/frame boundary — by re-using each "
                               "hole's existing boundary vertices, so projection-baked UVs stay valid. "
                               "Off by default: a torn silhouette is the DMP-correct look."}),
                "max_hole_edges": ("INT", {"default": 64, "min": 3, "max": 4096,
                    "tooltip": "A boundary loop is filled only if its edge count is below this. "
                               "The outer frame is ~the grid perimeter (e.g. ~512 at grid 128), "
                               "interior tears are ~4-30, so 64 separates them by construction."}),
                "fill_depth_near_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Band-box spatial scope: only fill loops whose EVERY boundary "
                               "vertex's forward depth (recovered-camera view space, same axis "
                               "as AtlasBoundedBand's cutoff) lies within [near, far] metres. "
                               "Transcribe off a bounded band's near and cutoff_m. 0 = off "
                               "(edge-count-only mode; the single largest loop is always left open)."}),
                "fill_depth_far_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                    "tooltip": "Band-box far bound (the cutoff). 0 = off (see fill_depth_near_m)."}),
                "retopo_method": (["off", "quad", "decimate", "smooth"],
                    {"default": "off",
                     "tooltip": "EXPORT-ONLY retopology pass on the OBJ/GLB (the live viewport "
                                "projection mesh is never touched). off = no change (default, so "
                                "every saved workflow keeps working). quad = pyinstantmeshes "
                                "orientation-field quad remesh (cleanest DCC handoff; needs the "
                                "pyinstantmeshes package). decimate = quadric decimation via "
                                "fast-simplification (fewer faces, same topology class). smooth = "
                                "trimesh Taubin relax (topology-preserving, UVs kept). quad/decimate "
                                "change the vertex count so projection-baked UVs are REGENERATED "
                                "from the recovered camera (pure numpy) and the retopologized mesh "
                                "stays textured. Runs AFTER any interior hole-fill."}),
                "retopo_target_vertex_count": ("INT", {"default": 2000, "min": 4, "max": 2000000,
                    "tooltip": "Target vertex count for quad / target face count for decimate "
                               "(decimate targets ~2x this in faces). Ignored by smooth."}),
                "retopo_smooth_iterations": ("INT", {"default": 0, "min": 0, "max": 100,
                    "tooltip": "quad: Instant Meshes post-smooth iterations. smooth: Taubin "
                               "relax iterations (the actual smoothing strength). decimate: "
                               "ignored."}),
                "retopo_crease_angle": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "quad only: crease angle (deg) below which adjacent faces are "
                               "treated as one smooth surface in the orientation field."}),
                "retopo_pure_quad": ("BOOLEAN", {"default": False,
                    "tooltip": "quad only: force a pure-quad output (no triangles). False allows "
                               "quad-dominant (triangles where the field can't place a quad)."}),
            },
        }

    def export(self, solve, image, output_dir="atlas_exports", grid_long_edge=128,
               depth_edge_rel=0.5,
               depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
               device="auto", format="both", use_solve_mesh=True,
               max_edge_factor=12.0, normal_edge_deg=0.0,
               fill_interior_holes=False, max_hole_edges=64,
               fill_depth_near_m=0.0, fill_depth_far_m=0.0,
               retopo_method="off", retopo_target_vertex_count=2000,
               retopo_smooth_iterations=0, retopo_crease_angle=30.0,
               retopo_pure_quad=False):
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.core.solver import _resize_depth
        from atlas_camera.exporters.relief_mesh_exporter import (
            export_relief_mesh,
            export_relief_mesh_glb,
        )
        from atlas_camera.inference.depth_estimator import estimate_depth

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

        # Prefer the relief mesh ALREADY derived onto the solve — it carries all
        # the edge tuning (max_edge_factor / normal_edge_deg / band near-clip /
        # sky_heuristic) exactly, so the OBJ matches the viewport with no widget
        # to re-set. Re-derive only when asked (higher-res export) or when the
        # solve has no relief mesh (e.g. a bare solve node).
        mesh = _relief_mesh_from_solve(solve) if use_solve_mesh else None
        if mesh is None:
            tmp = _save_image_tensor_to_tmp(image)
            try:
                result = estimate_depth(tmp, model_id=depth_model,
                                        device=None if device == "auto" else device,
                                        # fx is in solve-image pixels; the tmp file is the
                                        # wired tensor's resolution (usually identical).
                                        focal_px=fx * (image.shape[2] / width))
            finally:
                os.unlink(tmp)

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
                max_edge_factor=float(max_edge_factor),
                normal_edge_deg=(float(normal_edge_deg) if float(normal_edge_deg) > 0 else None),
            )
        # EXPORT-ONLY interior hole fill. Never touches the live projection
        # mesh (which keeps its deliberate silhouette tears for DMP); caps the
        # exported OBJ/GLB so it retopologizes/booleans cleanly in a DCC.
        n_filled, filled, faces_added, loops_left = 0, [], 0, 0
        if fill_interior_holes:
            from atlas_camera.core.mesh_repair import (
                apply_interior_hole_fill,
                boundary_edges,
                walk_loops,
            )
            n_before = len(mesh.faces)
            n_filled, filled = apply_interior_hole_fill(
                mesh,
                max_hole_edges=int(max_hole_edges),
                view_matrix=extr.camera_view_matrix,
                depth_near_m=float(fill_depth_near_m),
                depth_far_m=float(fill_depth_far_m),
            )
            faces_added = len(mesh.faces) - n_before
            # What's STILL open is the actionable half: a disappointing fill is
            # usually a too-tight scope, and the count says so at a glance.
            be = boundary_edges(mesh.faces)
            loops_left = len(walk_loops(be, faces=mesh.faces)) if len(be) else 0
        # EXPORT-ONLY retopology (quad / decimate / smooth) — same doctrine as
        # the hole-fill above: never touches the live viewport projection mesh
        # or solve.proxy_geometry. Runs AFTER the hole-fill so it retopologizes
        # the capped mesh. quad/decimate change the vertex count, so the 1:1
        # vertex-UV mapping is regenerated from the recovered camera (pure
        # numpy); smooth preserves topology and keeps the existing UVs.
        retopo_note = ""
        if retopo_method and retopo_method != "off":
            from atlas_camera.core.mesh_retopo import apply_retopo
            rrep = apply_retopo(
                mesh,
                method=str(retopo_method),
                target_vertex_count=int(retopo_target_vertex_count),
                view_matrix=extr.camera_view_matrix,
                fx=fx, fy=fy, cx=cx, cy=cy,
                image_width=width, image_height=height,
                pure_quad=bool(retopo_pure_quad),
                crease_angle=float(retopo_crease_angle),
                smooth_iterations=int(retopo_smooth_iterations),
            )
            if rrep.get("changed"):
                retopo_note = (
                    f"\n\U0001f53b retopo [{rrep.get('method', retopo_method)}]: "
                    f"{rrep.get('in_verts', '?')} → {rrep.get('out_verts', '?')} verts, "
                    f"{rrep.get('in_faces', '?')} → {rrep.get('out_faces', '?')} faces "
                    f"— {rrep.get('note', '')}"
                )
            else:
                retopo_note = (
                    f"\n\U0001f53b retopo [{retopo_method}]: no change "
                    f"— {rrep.get('note', '')}"
                )
        report = _format_hole_fill_report(
            fill_interior_holes, n_filled, filled, faces_added, loops_left,
            max_hole_edges, float(fill_depth_near_m), float(fill_depth_far_m)) \
            + retopo_note + _scale_summary_suffix(solve)
        # The viewport gets the geometry that was ACTUALLY written, off the same
        # widgets — so what an artist tunes here is what lands in Maya/Nuke.
        preview_solve = _solve_with_relief_mesh(solve, mesh)
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
        return {"ui": {"text": [report]},
                "result": (obj_path, glb_path, preview_solve, report)}


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
        try:
            result = write_nuke_layers_script(solve, output_dir)
        except ValueError as exc:
            # The LAYER export needs ProjectionSources (sky / clean-plate bands /
            # patches). A layers=0 single relief mesh has none — don't crash the
            # queue; return a clear pointer. Use AtlasInput layers>=1 for the full
            # DCC handoff, or AtlasExportUSD (camera) for the single-relief case.
            return ("", f"Nuke layer export skipped — {exc}")
        summary = f"{len(result['layers'])} layer(s): {', '.join(result['layers'])}"
        if result["skipped"]:
            summary += f" | skipped: {'; '.join(result['skipped'])}"
        summary += _scale_summary_suffix(solve)
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
        try:
            result = write_maya_layers_scene(solve, output_dir)
        except ValueError as exc:
            # See AtlasExportNukeLayers: the LAYER export needs ProjectionSources;
            # a layers=0 single relief mesh has none. Graceful skip, not a crash.
            return ("", f"Maya layer export skipped — {exc}")
        summary = f"{len(result['layers'])} layer(s): {', '.join(result['layers'])}"
        if result["skipped"]:
            summary += f" | skipped: {'; '.join(result['skipped'])}"
        summary += _scale_summary_suffix(solve)
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
                # look/lut_path/exposure/gamma widgets removed 2026-07-10
                # (user: redundant — exposure duplicated the viewport's own ☀
                # control, gamma was a crude CSS filter, look/lut were inert
                # metadata) to make room on the Output Desk. profile() still
                # accepts them as kwargs and the AtlasOutputProfile schema
                # keeps the fields defaulted, so old prompts/API workflows and
                # every downstream consumer are unaffected. NOTE: this is the
                # one sanctioned widget REMOVAL — display_trim moved from
                # widgets_values index 10 to 6, and every shipped example
                # carrying this node was re-saved in the same commit.
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
                    "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shaded", "depth", "normal", "mask", "path_frames", "camera_path",
                    "patch_azimuth_view", "patch_elevation_view", "patch_distance", "patch_prompt",
                    "patch_exact")
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
                        "<sks> front view eye-level shot medium shot",
                        "azimuth_deg=0.0 elevation_deg=0.0 distance_scale=1.0")

        def _patch_exact_string(pa):
            # 📐 stores the RAW measured orbit floats alongside the snapped
            # named views (client_data.patch_angle.raw) — the exact-angle
            # channel the render-conditioned patch loop needs (bake a frame
            # at the artist's real orbit, fix it, project it back from the
            # IDENTICAL pose via AtlasAddPatchView's exact_view_override; a
            # 45°-grid named view would misregister the projection). Records
            # without `raw` fall back to the snapped named views' own delta,
            # so the string is always a pose AddPatchView can reproduce.
            raw = pa.get("raw") or {}
            try:
                d_az = float(raw["d_azimuth_deg"])
                d_el = float(raw["d_elevation_deg"])
                dist = float(raw["distance_scale"])
            except (KeyError, TypeError, ValueError):
                try:
                    d_az, d_el, dist = _named_view_orbit_delta(
                        str(pa.get("azimuth_view") or _pa_defaults[0]),
                        str(pa.get("elevation_view") or _pa_defaults[1]),
                        str(pa.get("distance_view") or _pa_defaults[2]),
                        "front view", "eye-level shot", False)
                except KeyError:
                    return _pa_defaults[4]
            return (f"azimuth_deg={d_az:.4f} elevation_deg={d_el:.4f} "
                    f"distance_scale={dist:.4f}")

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
                    return (blocker,) * 5
                return _pa_defaults
            return (
                str(pa.get("azimuth_view") or _pa_defaults[0]),
                str(pa.get("elevation_view") or _pa_defaults[1]),
                str(pa.get("distance_view") or _pa_defaults[2]),
                str(pa.get("prompt") or _pa_defaults[3]),
                _patch_exact_string(pa),
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
# schema (see docs/dev/archive/atlas_inpaint_layers_design.md §2) — the viewport's
# per-source projection material already does everything needed; these nodes
# are orchestration only. Masking/inpainting itself is NOT implemented here —
# it's delegated to external ComfyUI node packs wired into the graph
# (Acly/comfyui-inpaint-nodes, GPL-3.0; scraed/LanPaint, optional generative
# tier for hard disocclusions) — see INSTALL.md's "Optional Inpaint
# Integration" section. Graph-level composition keeps the GPL boundary clean:
# no inpainting/segmentation code lives in atlas_camera.
# ---------------------------------------------------------------------------

class AtlasDepthBandSplit:
    """One authoritative fg/bg depth boundary, shared by every band node.

    The split is a POSITION ALONG THE SCENE'S LOG-DEPTH RANGE (the same
    exponential / inverse-log mapping `_resolve_depth_band` uses: 0.5 = the
    geometric mean of the scene's depth range), so the SAME split value
    adapts per solve — 0.55 means "just past mid-scene" on any image,
    resolving to different metres per scene. `split_m` (metres) overrides
    when nonzero.

    Wire the output into `AtlasCleanPlateLayer`/`AtlasDepthLayerMask`'s
    `band_split` input and set each node's `band_side` (foreground /
    background): fg becomes [0, split), bg becomes [split, +inf) — one wire,
    the two layers' bands can never drift apart (previously the boundary
    lived in TWO widgets, bg.near_pct and fg.far_pct, kept in lockstep by
    hand). Config-carrier node: no computation, same in-process pattern as
    `AtlasDefineShotCam`.
    """
    RETURN_TYPES = ("ATLAS_BAND_SPLIT",)
    RETURN_NAMES = ("band_split",)
    FUNCTION = "define"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "split": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "The fg/bg boundary as a position along the scene's LOG-depth "
                               "range (0.5 = geometric mean of the depth range = perceptually "
                               "mid-scene). Scene-relative: the same value adapts to each "
                               "solve's own depth distribution."}),
                "split_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Absolute boundary in metres — overrides `split` when nonzero "
                               "(for when you've measured the scene and want a hard number)."}),
            },
        }

    def define(self, split=0.55, split_m=0.0):
        return ({"split": float(split), "split_m": float(split_m)},)


def _apply_band_split(band_split, band_side, metric, valid,
                      near_m, far_m, near_pct, far_pct):
    """Resolve the effective band, honoring a connected `band_split`.

    With a split connected and a side chosen, the node's own near/far widgets
    are ignored: foreground = [0, split), background = [split, +inf). Both
    sides resolve the boundary through the same `_resolve_depth_band` log
    mapping, so fg and bg partition EXACTLY (shared helper = no drift).
    """
    if band_split is None or band_side == "manual":
        return _resolve_depth_band(metric, valid, near_m, far_m, near_pct, far_pct)
    s_pct = float(band_split.get("split", 0.55))
    s_m = float(band_split.get("split_m", 0.0))
    boundary, _ = _resolve_depth_band(metric, valid, s_m, 0.0, s_pct, 0.0)
    if band_side == "foreground":
        return 0.0, boundary
    return boundary, float("inf")


# A partition split can never be a true no-op for BOTH sides (one side always
# gets the empty tail). When AtlasBoundedBand can't measure, it emits this large
# sentinel so the FOREGROUND (the layer we care about) is effectively unclipped
# ([0, 1e6m] passes every real scene depth); the background tail goes empty and
# the report says why. Never emit split_m=0 — that collapses the foreground band
# to [0, 0] and empties the relief.
_BOUNDED_BAND_NOOP_M = 1.0e6


class AtlasBoundedBand:
    """📏 Measure the FOREGROUND's own metric depth extent and emit ONE
    `ATLAS_BAND_SPLIT` that clips a relief layer at a guessed distance while
    the background card falls back behind it.

    The classic single-photo failure: monocular depth "bananas" a foreground
    subject (buildings, a statue, a foreground ridge) so its relief mesh
    extrudes far past where the object actually ends, with no bound on how far
    back it runs. This node measures the subject's front-to-back depth extent
    `W = P(far_pct) − P(near_pct)` over its mask and returns a cutoff at
    `near + extrude_multiplier · W` (default 2×).

    Wire the ONE `band_split` output into BOTH clean-plate layers'
    `band_split` input, with `band_side` set:
      • foreground layer (`band_side=foreground`) → `[0, cutoff]`: the relief
        is clipped at the guessed distance — no runaway extrusion.
      • background layer (`band_side=background`) → `[cutoff, +inf]`: the card
        sits at the median depth of everything beyond the cutoff — pushed back
        for dolly parallax.
    The split is an ABSOLUTE distance (`split_m`), so both layers resolve the
    identical boundary regardless of their own pixel populations — no band
    drift, no `band_ref_mask` needed (unlike percentile splits).

    Composition-only: reuses `AtlasCleanPlateLayer`'s existing `band_split`
    input, so it respects that node's capability freeze. `foreground_mask` is
    the subject segmentation (e.g. the same SAM3 mask that scopes the
    foreground layer). Needs the `[neural]` extra (metric depth). Fails soft to
    an unclipped sentinel + an explanatory report when it can't measure.
    """
    RETURN_TYPES = ("ATLAS_BAND_SPLIT", "FLOAT", "STRING")
    RETURN_NAMES = ("band_split", "cutoff_m", "report")
    FUNCTION = "measure"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
                "foreground_mask": ("MASK",),
            },
            "optional": {
                "extrude_multiplier": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.25,
                    "tooltip": "cutoff = near + this × (foreground depth extent W). 2.0 = the "
                               "relief may extrude back at most twice its own front-to-back width "
                               "before being clipped. 0 = clip at the near edge."}),
                "near_pct": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Percentile of the foreground pixels' metric depth taken as the "
                               "subject's NEAR edge (robust to a few stray near pixels)."}),
                "far_pct": ("FLOAT", {"default": 95.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Percentile taken as the subject's FAR edge. W = P(far_pct) − P(near_pct)."}),
            },
        }

    def measure(self, solve, depth, foreground_mask,
                extrude_multiplier=2.0, near_pct=5.0, far_pct=95.0):
        np = _require_numpy()
        noop = ({"split": 0.0, "split_m": _BOUNDED_BAND_NOOP_M}, float(_BOUNDED_BAND_NOOP_M))
        setup = _metric_depth_and_validity(solve, depth)
        if setup is None:
            return noop + (
                "AtlasBoundedBand: no metric depth (needs [neural] + a solved focal length) — "
                "emitting an unclipped sentinel so the foreground relief is unaffected.",)
        valid = setup.valid & np.isfinite(setup.metric)
        fg = _resolve_exclude_mask(foreground_mask, setup.height, setup.width)
        if fg is not None:
            valid = valid & fg.astype(bool)
        n = int(valid.sum())
        if n < 16:
            return noop + (
                f"AtlasBoundedBand: foreground mask covers only {n} valid-depth pixels (need ≥16) — "
                "emitting an unclipped sentinel (check the mask / solve).",)
        lo, hi = sorted((float(near_pct), float(far_pct)))
        vals = setup.metric[valid]
        near = float(np.percentile(vals, lo))
        far = float(np.percentile(vals, hi))
        width = max(far - near, 0.0)
        cutoff = near + float(extrude_multiplier) * width
        if width <= 1e-6 or not (cutoff > 0.0):
            return noop + (
                f"AtlasBoundedBand: degenerate extent (near={near:.2f}m far={far:.2f}m W={width:.3f}m) — "
                "the mask has no depth spread; emitting an unclipped sentinel.",)
        report = (
            f"AtlasBoundedBand: foreground {n} px | near(P{lo:.0f})={near:.2f}m "
            f"far(P{hi:.0f})={far:.2f}m | W={width:.2f}m ×{extrude_multiplier:.2f} "
            f"→ cutoff={cutoff:.2f}m\n"
            f"  band_split → foreground layer (band_side=foreground): relief clipped to [0, {cutoff:.2f}m]\n"
            f"  band_split → background layer (band_side=background): card median beyond {cutoff:.2f}m")
        return ({"split": 0.0, "split_m": float(cutoff)}, float(cutoff), report)


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
                    "tooltip": "Band near edge in metres. 0 = auto (use near_pct)."}),
                "far_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Band far edge in metres. 0 = auto (use far_pct)."}),
                "near_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Used when near_m==0: POSITION ALONG THE SCENE'S LOG-DEPTH "
                               "RANGE, not a pixel percentile (depth is skewed — pixel percentiles "
                               "wasted 0-0.9 on the foreground; 0.5 here = the geometric mean of "
                               "the scene's depth range, perceptually mid-scene). LOWER = closer "
                               "near threshold = tighter occlusion. Try 0.2-0.4 for a typical "
                               "foreground object."}),
                "far_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Used when far_m==0: position along the scene's LOG-depth "
                               "range (see near_pct). 0 means no upper bound (+inf); values at or "
                               "above ~1.0 also mean no cap."}),
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
                               "SAM/RMBG) which REPLACES the internal sky heuristic - so it "
                               "must cover EVERYTHING you want gone. Affects layer_mask/occlusion_mask "
                               "(excluded pixels can't belong to any band) AND hole_mask when "
                               "compute_hole_mask=True. Any resolution - resized to match depth."}),
                "fill_occluded": ("BOOLEAN", {"default": False,
                    "tooltip": "Only used when compute_hole_mask=True. MUST match the "
                               "AtlasCleanPlateLayer setting downstream for hole_mask to reflect "
                               "the actual final mesh - when the layer will diffusion-fill the "
                               "occluder footprint, that footprint is no longer a hole here "
                               "either."}),
                "band_side": (["manual", "foreground", "background"], {"default": "manual",
                    "tooltip": "With band_split connected: foreground = [0, split), background "
                               "= [split, +inf) — the node's own near/far widgets are ignored. "
                               "manual = use this node's own near/far settings."}),
                "band_split": ("ATLAS_BAND_SPLIT", {
                    "tooltip": "Wire ONE AtlasDepthBandSplit into every band node (with "
                               "band_side set) so the fg/bg boundary lives in exactly one "
                               "widget and the layers can never drift apart."}),
                "band_ref_mask": ("MASK", {
                    "tooltip": "Exclusion used ONLY for resolving near/far percentages to "
                               "metres. When exclude_mask carries per-layer scoping (🎯 scope "
                               "rows), each layer's depth population differs and the shared "
                               "band edges DRIFT apart (metric gaps between adjacent bands — "
                               "debug-report finding). Wire the plain SKY mask here on every "
                               "band node so all layers resolve identical edges. Unwired = "
                               "legacy behavior (band edges from exclude_mask's population)."}),
                # APPENDED last (widgets_values is positional — never insert).
                "band_override": ("STRING", {"default": "",
                    "tooltip": "Optional band override STRING ('near_pct=<f> far_pct=<f>') — "
                               "wins over this node's near/far widgets when non-empty. Wire "
                               "AtlasAssessImage's band_far/bg/mid/fg output here so the VLM's "
                               "subject-aware band boundaries flow in (jointly derived, so "
                               "adjacent bands always share edges exactly). Loses to a "
                               "connected band_split. Errors loudly on garbage."}),
            },
        }

    def generate(self, solve, depth, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5, feather_px=4,
                 compute_hole_mask=False, relief_grid=384, depth_edge_rel=1.5, exclude_mask=None,
                 fill_occluded=False, band_side="manual", band_split=None, band_ref_mask=None,
                 band_override=""):
        np = _require_numpy()
        torch = _require_torch()

        setup = _metric_depth_and_validity(solve, depth, exclude_mask=exclude_mask)
        if setup is None:
            h, w = int(depth.image_height), int(depth.image_width)
            zero = torch.zeros(1, h, w, dtype=torch.float32)
            return (zero, zero.clone(), zero.clone())
        metric, valid = setup.metric, setup.valid

        override = _parse_band_override(band_override)
        if override is not None:
            near_m = far_m = 0.0
            near_pct, far_pct = override
        near, far = _apply_band_split(band_split, band_side, metric,
                                      _band_resolution_validity(setup, band_ref_mask),
                                      near_m, far_m, near_pct, far_pct)

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
                up = np.zeros_like(grown)
                up[:-1, :] = grown[1:, :]
                down = np.zeros_like(grown)
                down[1:, :] = grown[:-1, :]
                left = np.zeros_like(grown)
                left[:, :-1] = grown[:, 1:]
                right = np.zeros_like(grown)
                right[:, 1:] = grown[:, :-1]
                grown = grown | up | down | left | right
            occlusion_mask = grown

        layer_t = torch.from_numpy(layer_mask.astype(np.float32)).unsqueeze(0)
        occ_t = torch.from_numpy(occlusion_mask.astype(np.float32)).unsqueeze(0)
        hole_t = torch.from_numpy(hole_mask_arr).unsqueeze(0)
        return (layer_t, occ_t, hole_t)


class AtlasDebugReport:
    """🔍 One-stop machine-readable diagnostic of the layered master scene.

    Wire the FINAL solve (whatever feeds the master viewport) plus, optionally,
    the shared depth, the 🎯 scope-status strings, and the 🧭 VLM report. Every
    execution introspects the whole stack — camera summary, every
    ProjectionSource's geometry type / vertex count / band range / matte
    coverage — runs red-flag analysis (zero-vertex layers, band gaps/overlaps,
    near-empty mattes, no-match scope fallbacks), renders a human-readable
    report ON the node, and writes the full structured JSON to a STABLE path
    (`file_path`, default `atlas_debug/master_debug.json` under ComfyUI's CWD).

    The JSON exists specifically so an AI assistant (or any tool) can read one
    file and see the same facts that otherwise take a live payload/history
    autopsy to reconstruct — built after exactly such a session: two layers
    silently shipped zero-vertex meshes and only a viewport-payload dump
    revealed it. Zero heavy computation: everything is read off data the solve
    already carries (matte coverage decodes the embedded PNGs, the one
    non-trivial step).
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "json_path")
    FUNCTION = "report"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "depth": ("ATLAS_DEPTH_MAP",),
                "file_path": ("STRING", {"default": "atlas_debug/master_debug.json",
                    "tooltip": "Stable JSON path (relative to ComfyUI's working directory) — "
                               "keep it constant so external tooling can always find the "
                               "latest diagnostic."}),
                "status_1": ("STRING", {"forceInput": True}),
                "status_2": ("STRING", {"forceInput": True}),
                "status_3": ("STRING", {"forceInput": True}),
                "status_4": ("STRING", {"forceInput": True}),
                "vlm_report": ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def _matte_coverage(mask_b64):
        if not mask_b64:
            return None
        try:
            np = _require_numpy()
            PILImage = _require_pil()
            raw = base64.b64decode(mask_b64.split(",", 1)[-1])
            arr = np.asarray(PILImage.open(io.BytesIO(raw)).convert("L"))
            return round(float((arr > 127).mean()), 4)
        except Exception:
            return None

    def report(self, solve, depth=None, file_path="atlas_debug/master_debug.json",
               status_1="", status_2="", status_3="", status_4="", vlm_report="",
               **_extra):
        import datetime

        flags: list = []
        cam = solve.camera
        intr, extr = cam.intrinsics, cam.extrinsics
        # Camera height from the full 4x4 (the view-matrix convention rule) —
        # extrinsics.camera_position can legitimately be an unset default.
        cam_y = None
        if extr is not None and extr.camera_view_matrix is not None:
            try:
                np = _require_numpy()
                cam_y = round(float(np.linalg.inv(np.asarray(
                    extr.camera_view_matrix, dtype=float))[1, 3]), 4)
            except Exception:
                cam_y = None
        camera = {
            "image_wh": [intr.image_width, intr.image_height],
            "focal_mm": intr.focal_length_mm, "sensor_mm": intr.sensor_width_mm,
            "fx_px": intr.fx_px, "camera_height_m": cam_y,
            "confidence": getattr(solve, "confidence", None),
            "source_method": getattr(solve, "source_method", None),
            "scale_source": (solve.debug_metadata or {}).get("scale_source"),
        }
        if cam_y is not None and cam_y <= 0:
            flags.append("camera height <= 0 — ground-based features (ground depth, "
                         "band_geometry=ground) will fail")

        sources = []
        for src in getattr(solve, "projection_sources", None) or []:
            meta = src.metadata or {}
            n_verts = sum(int((g.metadata or {}).get("n_vertices") or 0)
                          for g in (src.proxy_geometry or []))
            n_faces = sum(int((g.metadata or {}).get("n_faces") or 0)
                          for g in (src.proxy_geometry or []))
            cov = self._matte_coverage(getattr(src, "mask_b64", None))
            entry = {
                "name": src.name, "priority": src.priority,
                "projection_mode": meta.get("projection_mode"),
                "band_geometry": meta.get("band_geometry"),
                "near_m": meta.get("near_m"), "far_m": meta.get("far_m"),
                "n_vertices": n_verts, "n_faces": n_faces,
                "n_filled_cells": meta.get("n_filled_cells"),
                "source_camera_wh": [src.camera.intrinsics.image_width,
                                     src.camera.intrinsics.image_height]
                                    if src.camera else None,
                "matte_coverage": cov,
                "has_extend_mask": bool(getattr(src, "extend_mask_b64", None)),
            }
            sources.append(entry)
            if n_verts == 0:
                flags.append(f"{src.name}: ZERO vertices — this layer contributes no "
                             "geometry (empty band, exclude-everything scope, or a "
                             "failed flat-mode region)")
            elif cov is not None and cov < 0.005:
                flags.append(f"{src.name}: matte covers only {cov:.2%} of the frame — "
                             "layer will paint almost nothing")

        # Band continuity (clean-plate band layers only, sorted by near edge).
        bands = sorted((s for s in sources
                        if s["projection_mode"] == "clean_plate" and s["near_m"] is not None
                        or (s["near_m"] is None and s["far_m"] is not None)),
                       key=lambda s: s["near_m"] or 0.0)
        for a, b in zip(bands, bands[1:]):
            fa, nb = a.get("far_m"), b.get("near_m")
            if fa is not None and nb is not None and abs(fa - nb) > max(0.05, 0.02 * fa):
                kind = "GAP" if nb > fa else "OVERLAP"
                flags.append(f"band {kind} between {a['name']} (far {fa:.2f}m) and "
                             f"{b['name']} (near {nb:.2f}m)")

        statuses = {f"status_{i}": s for i, s in
                    enumerate((status_1, status_2, status_3, status_4), 1) if s}
        for k, s in statuses.items():
            if "FALLBACK" in s:
                flags.append(f"scope {k}: {s}")

        # DA3 watch-item made measurable: the DA3 backend occasionally emits
        # NEGATIVE raw depth (documented; ground-pinning renormalizes, so it
        # has been harmless so far) — surface the actual fraction so a shot
        # where a band misbehaves points here first (observed live: an alpine
        # ridge shot reported depth.near = -11.4m).
        depth_info = None
        if depth is not None:
            depth_info = {"model_id": depth.model_id, "is_metric": depth.is_metric,
                          "near": depth.near, "far": depth.far,
                          "wh": [depth.image_width, depth.image_height]}
            try:
                np = _require_numpy()
                # Prefer the estimator's recorded PRE-CLAMP fraction (the
                # source now clamps metric depth at >= 0, so recomputing from
                # the array would always read 0 and hide the watch-item).
                recorded = (depth.metadata or {}).get("negative_fraction")
                if recorded is not None:
                    neg = float(recorded)
                else:
                    arr = np.asarray(depth.depth)
                    neg = float((arr < 0).mean())
                depth_info["negative_fraction"] = round(neg, 4)
                if neg > 0.01:
                    flags.append(
                        f"depth: {neg:.1%} of raw depth is NEGATIVE (DA3 watch-item) — "
                        "ground-pinning renormalizes it, but suspect this first if a "
                        "band's geometry misbehaves on this shot")
            except Exception:
                pass

        try:
            from atlas_camera import __version__ as _atlas_version
        except Exception:
            _atlas_version = "unknown"
        data = {
            # Versioned for external consumers (this JSON is parsed by
            # tooling/AI assistants, not just eyeballed): bump "schema" on
            # any breaking key change.
            "schema": 1,
            "atlas_version": _atlas_version,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "camera": camera,
            "depth": depth_info,
            "shot_cam": (solve.shot_cam.to_dict()
                         if getattr(solve, "shot_cam", None) and
                         hasattr(solve.shot_cam, "to_dict") else None),
            "projection_sources": sources,
            "primary_proxy_geometry": [
                {"name": g.name, "type": g.primitive_type,
                 "n_vertices": (g.metadata or {}).get("n_vertices")}
                for g in (getattr(solve.projection_scene, "proxy_geometry", None) or [])],
            "scope_status": statuses,
            "vlm_report": vlm_report or None,
            "flags": flags,
        }
        path = os.path.abspath(file_path or "atlas_debug/master_debug.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
        except OSError as exc:
            flags.append(f"could not write {path}: {exc}")
            path = ""

        lines = ["ATLAS MASTER DEBUG", "=" * 18, ""]
        lines.append(f"camera  {camera['image_wh'][0]}x{camera['image_wh'][1]}  "
                     f"{camera['focal_mm'] or '?'}mm  height {cam_y if cam_y is not None else '?'}m  "
                     f"conf {camera['confidence']}")
        lines.append("")
        lines.append("LAYERS (name / geometry / band / verts / matte)")
        for s in sources:
            band = (f"{s['near_m'] or 0:.1f}-" +
                    (f"{s['far_m']:.1f}m" if s["far_m"] is not None else "inf"))
            cov = f"{s['matte_coverage']:.1%}" if s["matte_coverage"] is not None else "-"
            lines.append(f"  {s['name']:10s} {s['band_geometry'] or '-':7s} {band:12s} "
                         f"{s['n_vertices']:>8d} verts  matte {cov}")
        if statuses:
            lines += ["", "SCOPE STATUS"] + [f"  {v}" for v in statuses.values()]
        lines += ["", f"FLAGS ({len(flags)})"] + ([f"  ! {f}" for f in flags] or ["  (none — stack looks healthy)"])
        if path:
            lines += ["", f"full JSON: {path}"]
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report, path)}


# Hand-mirrors atlas_blockout.js's LAYER_DEBUG_PALETTE / LAYER_DEBUG_PRIMARY
# (the 🎨 Layers legend) — keep both in sync by hand, the accepted-duplication
# pattern. Index = position in projection_sources; -1 = the primary teal.
_LAYER_DEBUG_PRIMARY_HEX = "2fd6c3"
_LAYER_DEBUG_PALETTE_HEX = ("ff6a3d", "3d8bff", "ffd23d", "c95aff", "6aff5a", "ff5aa8")


class AtlasLayerPreview:
    """🎨 Cut-out layer preview: the plate's pixels where this layer's matte
    is, and the layer's 🎨 Layers debug color (opaque) everywhere else — one
    image that shows WHAT the layer will project AND which layer it is, in
    the exact color the master viewport's 🎨 Layers legend uses for it.

    Replaces the mask-preview + plate-preview pairs in the staged master's
    per-stage debug strip (user feedback: "the preview should only show
    already cut-out layers, then we don't need the mask preview").
    `layer_index` is the layer's position in `projection_sources` (the 🎨
    legend order — staged master: sky=0, far=1, bg=2, mid=3, fg=4; -1 = the
    primary teal); `color_hex` overrides the palette when set.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The layer's plate (post-inpaint)."}),
                "mask": ("MASK", {"tooltip": "The layer's matte (band layer_mask / sky mask)."}),
            },
            "optional": {
                "layer_index": ("INT", {"default": 0, "min": -1, "max": 15,
                    "tooltip": "Position in projection_sources = the 🎨 Layers legend order "
                               "(staged master: sky=0, far=1, bg=2, mid=3, fg=4). -1 = primary "
                               "teal. Sets the opaque surround color."}),
                "color_hex": ("STRING", {"default": "",
                    "tooltip": "Optional explicit hex color (e.g. ff6a3d) — overrides "
                               "layer_index's palette pick."}),
            },
        }

    def preview(self, image, mask, layer_index=0, color_hex=""):
        torch = _require_torch()
        import torch.nn.functional as F

        hexs = (color_hex or "").strip().lstrip("#")
        if not hexs:
            hexs = (_LAYER_DEBUG_PRIMARY_HEX if int(layer_index) < 0 else
                    _LAYER_DEBUG_PALETTE_HEX[int(layer_index) % len(_LAYER_DEBUG_PALETTE_HEX)])
        try:
            rgb = tuple(int(hexs[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except (ValueError, IndexError):
            rgb = (1.0, 0.0, 1.0)  # loud magenta for a malformed hex
        h, w = int(image.shape[1]), int(image.shape[2])
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        if tuple(m.shape[1:]) != (h, w):
            m = F.interpolate(m.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        mm = (m[:1] > 0.5).float().unsqueeze(-1)
        color = torch.tensor(rgb, dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        out = image[:, :, :, :3] * mm + color * (1.0 - mm)
        return (out,)


def _comfy_registry():
    """ComfyUI's global node registry, or {} outside ComfyUI — used by
    AtlasInput's expansion to feature-detect third-party packs (SAM3 /
    inpaint) without ever importing their code."""
    try:
        import nodes as comfy_nodes  # ComfyUI's own top-level module
        return comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return {}


class _MiniGraphBuilder:
    """Test-shim mirror of comfy_execution.graph_utils.GraphBuilder (same
    node()/out()/finalize() surface) so AtlasInput's expansion assembly is
    unit-testable outside ComfyUI. Real runs always use the real one — it
    namespaces inner node ids for the executor's caching."""

    class _Node:
        def __init__(self, nid, class_type, inputs):
            self.id, self.class_type, self.inputs = nid, class_type, inputs

        def out(self, index):
            return [self.id, index]

    def __init__(self):
        self.nodes = {}
        self._i = 0

    def node(self, class_type, **inputs):
        self._i += 1
        n = self._Node(str(self._i), class_type, inputs)
        self.nodes[n.id] = n
        return n

    def finalize(self):
        return {nid: {"class_type": n.class_type, "inputs": n.inputs}
                for nid, n in self.nodes.items()}


def _graph_builder():
    try:
        from comfy_execution.graph_utils import GraphBuilder
        return GraphBuilder()
    except Exception:
        return _MiniGraphBuilder()


# The documented proven band splits per layer count (log-depth positions).
_ATLAS_INPUT_BOUNDARIES = {2: (0.55,), 3: (0.2, 0.65), 4: (0.3, 0.6, 0.8)}
_ATLAS_INPUT_BAND_NAMES = ("band_far", "band_bg", "band_mid", "band_fg")


class AtlasInput:
    """🎬 The all-in-one entry point — one node between LoadImage and the
    viewport that wraps the staged master's logic via NODE EXPANSION: at
    execution it emits the real mini-graph (our nodes by class, third-party
    SAM3Segment / LaMa by registry name) so every inner step keeps its own
    cache, and missing packs degrade gracefully (skipped + named in the
    `report` output) instead of erroring.

    Out of the box (instant relief): layers=0, VLM/SAM/inpaint off — the
    first queue costs solve + depth + ONE high-resolution relief mesh, and
    the `solve`/`image` outputs wire straight into AtlasBlockoutViewport.
    Turn knobs up from there:
    - `layers` 2–4 splits into depth-band clean-plate layers on the proven
      splits, watertight by construction (the band_override channel).
    - `use_vlm` puts AtlasAssessImage in front (advisory mode, VRAM
      offloaded after) and wires its prompts / per-band geometry / band
      boundaries into the inner nodes exactly like the staged master. With
      layers>0 this forces the 4-band plan (the VLM speaks 5 fixed slots).
    - `sky` + `sky_prompt` adds the SAM sky card; the mask also feeds every
      mesh's exclude_mask AND band_ref_mask (the band-drift rule).
    - `scope_prompts` (one line per band, far→near) adds self-disarming 🎯
      scope rows; `inpaint` adds the ✂crop→LaMa(+upscale)→✂stitch clean-
      plate chain per occluded band.

    When you outgrow it, the staged master IS this graph with stages,
    gates, rails, and per-layer debug previews — see
    examples/atlas_camera_staged_master_workflow.json.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "IMAGE", "ATLAS_DEPTH_MAP", "MASK", "STRING")
    RETURN_NAMES = ("solve", "image", "depth", "sky_mask", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "layers": ("INT", {"default": 0, "min": 0, "max": 4,
                    "tooltip": "0 = one full-range mesh (instant relief). 2/3/4 = depth-band "
                               "clean-plate layers on the proven splits (2→0.55; 3→0.2/0.65; "
                               "4→0.3/0.6/0.8), watertight by construction. 1 = one full-range "
                               "clean-plate layer (useful with mesh=card/ground)."}),
                "mesh": (["relief", "card", "ground"], {"default": "relief",
                    "tooltip": "layers=0: relief = depth-following mesh; card/ground = ONE flat "
                               "plane (band-median card / analytic ground). layers>0: the "
                               "DEFAULT band geometry — the VLM's per-band call wins when "
                               "use_vlm is on."}),
                "mesh_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Relief grid (long-edge cells). Internal tear threshold pairs "
                               "automatically: 0.5 for the single full-range mesh, 1.5 for "
                               "band-clipped layers (the calibrated pairings)."}),
                "use_vlm": ("BOOLEAN", {"default": False,
                    "tooltip": "Run the 🧭 VLM assessment first (advisory — never blocks; VRAM "
                               "offloaded after) and wire its SAM prompts, per-band geometry, "
                               "and band boundaries into the inner nodes. With layers>0 this "
                               "forces the 4-band plan (the VLM's plan has 5 fixed slots)."}),
                "vlm_provider": (["ollama", "lmstudio", "llamacpp", "openai"],
                    {"default": "lmstudio"}),
                "vlm_model": ("STRING", {"default": "",
                    "tooltip": "Blank = the provider's default model."}),
                "sky": ("BOOLEAN", {"default": False,
                    "tooltip": "SAM-segment the sky onto its own flat card, and feed the mask "
                               "into every mesh's exclude_mask + band_ref_mask. Needs "
                               "ComfyUI-RMBG (SAM3Segment) — skipped + noted if absent."}),
                "sky_prompt": ("STRING", {"default": "sky",
                    "tooltip": "Manual sky segmentation prompt; the VLM's wins when use_vlm."}),
                "scope_prompts": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Manual per-band SAM scoping, ONE PROMPT PER LINE far→near "
                               "(line 1 = farthest band). Blank line = that band stays "
                               "band-only. Self-disarming: a no-match segment falls back to "
                               "band-only automatically. The VLM's prompts win when use_vlm. "
                               "Needs ComfyUI-RMBG."}),
                "inpaint": ("BOOLEAN", {"default": False,
                    "tooltip": "Build each occluded band's clean plate: occlusion mask → "
                               "expand → ✂crop → LaMa → ✂stitch (the 256²-bottleneck fix). "
                               "Needs comfyui-inpaint-nodes + big-lama.pt — skipped + noted if "
                               "absent. Off = bands project the original photo (honest holes "
                               "on reveal)."}),
                "upscale_model": ("STRING", {"default": "",
                    "tooltip": "Optional upscale model FILENAME (models/upscale_models) fed to "
                               "the inner LaMa nodes — e.g. 4xRealWebPhoto_v4_dat2.safetensors. "
                               "Measured 6.5× fill detail vs legacy. Blank = off."}),
                "edge_extend_px": ("INT", {"default": 24, "min": 0, "max": 256, "step": 4,
                    "tooltip": "Behind-band edge-extend (layers>0): how far plate colours smear "
                               "PAST each silhouette to hide grid-step tears — the frontmost band "
                               "always keeps a clean 0 cut. Was baked at 64 (tuned for smooth "
                               "ridgelines); lower it for high-frequency content like foliage, "
                               "which 64 shreds into halos. 0 = clean cut on every band."}),
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "World-space edge tear threshold (layers=0 relief AND layers>0 "
                               "bands), SEPARATE from depth_edge_rel and often the DOMINANT tear "
                               "cause on deep / narrow-FOV / interior scenes — grazing walls and "
                               "receding floors trip the default 12x even where continuous, "
                               "shredding the mesh into 'combs'. Raise to 40-80 to close them; "
                               ">80 rubber-sheets real foreground silhouettes onto the background."}),
                "sky_heuristic": ("BOOLEAN", {"default": True,
                    "tooltip": "layers=0 relief mesh: exclude above-horizon far/rough regions as "
                               "sky before triangulation. Correct OUTDOORS; turn OFF for INTERIORS "
                               "(it eats the ceiling / vault / far wall as 'sky', punching large "
                               "holes). Ignored when sky (the SAM card) is on — that mask governs. "
                               "(layers>0 bands: sky handled per-band via exclude/scope instead.)"}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "0 = off. A THIRD tear test on surface-normal BEND (layers=0 relief "
                               "AND layers>0 bands): tears real creases / occlusion silhouettes "
                               "while leaving smoothly-receding walls and floors intact (unlike "
                               "max_edge_factor, which trips on any grazing surface). Pair with a "
                               "HIGHER max_edge_factor: raise mef to stop comb-tearing continuous "
                               "grazing surfaces, then set ~40-70 here to keep genuine edges torn."}),
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                     "tooltip": "Monocular depth backend (fed the solved focal). "
                                "V2-Metric-Outdoor (DEFAULT) = Apache, transformers-only (NO extra "
                                "install), best all-round on OUTDOOR/sky scenes; V2-Metric-Indoor "
                                "is its interior twin. MoGe-2 (Ruicheng/moge-*) = MIT, cleanest on "
                                "ENCLOSED/INTERIOR shots but masks sky (poor outdoors) — needs "
                                "[moge]. DA3* (EXPERIMENTAL) = strong metric, heavy deps, DA3NESTED "
                                "is non-commercial CC BY-NC — needs [neural-da3]. Pick per shot: "
                                "outdoor->V2-Outdoor, interior->MoGe or V2-Indoor. (A/B 2026-07-13.)"}),
                "vlm_scope": ("BOOLEAN", {"default": True,
                    "tooltip": "When use_vlm: also SCOPE each band by the VLM's SAM prompt "
                               "(band ∩ segment). A PARTIAL segment match legitimately keeps "
                               "the scope and can cut real band geometry — found live on the "
                               "ghost-town plate, where the mid band's 4.6% segment left the "
                               "rest of the band exposing the behind-band's fill smear. OFF = "
                               "VLM still drives bands/geometry, layers stay band-only "
                               "(robust full coverage — best for camera moves)."}),
            },
        }

    # --- assembly ---------------------------------------------------------
    def build(self, image, layers=0, mesh="relief", mesh_resolution=512,
              use_vlm=False, vlm_scope=True, vlm_provider="lmstudio", vlm_model="",
              sky=False, sky_prompt="sky", scope_prompts="", inpaint=False,
              upscale_model="", edge_extend_px=24, max_edge_factor=12.0,
              sky_heuristic=True, normal_edge_deg=0.0,
              depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", **_extra):
        registry = _comfy_registry()
        have_sam = "SAM3Segment" in registry
        # AtlasSemanticMask is our own node (SegFormer/ADE20K, pure transformers,
        # NO triton/CUDA requirement) — the non-CUDA fallback for text-prompt
        # segmentation. SAM3 needs triton, which does not exist on Mac(MPS)/CPU/
        # AMD, so those users can never load it; SegFormer keeps sky+scope on a
        # LEARNED mask there instead of collapsing to the bare heuristic.
        have_semantic = "AtlasSemanticMask" in registry
        have_inpaint = ("INPAINT_InpaintWithModel" in registry
                        and "INPAINT_LoadInpaintModel" in registry
                        and "INPAINT_ExpandMask" in registry)
        notes: list = []
        g = _graph_builder()

        def sam3(image_ref, prompt_value):
            return g.node("SAM3Segment", image=image_ref, prompt=prompt_value,
                          output_mode="Merged", confidence_threshold=0.5,
                          max_segments=0, segment_pick=0, mask_blur=0,
                          mask_offset=0, device="Auto", invert_output=False,
                          unload_model=False, background="Alpha",
                          background_color="#222222")

        def segment(image_ref, prompt_value):
            """Text-prompt segmentation with an automatic non-CUDA fallback.
            SAM3 (open-vocab, needs triton/CUDA) is preferred; on a box without
            it we fall back to AtlasSemanticMask (SegFormer, CPU/MPS) so sky and
            scope still get a learned mask with no rewiring. Returns a MASK ref,
            or None when neither segmenter is installed (caller then drops to the
            heuristic). SAM3 mask is out(1); AtlasSemanticMask mask is out(0)."""
            if have_sam:
                return sam3(image_ref, prompt_value).out(1)
            if have_semantic:
                return g.node("AtlasSemanticMask", image=image_ref,
                              classes=prompt_value).out(0)
            return None

        if not have_sam and have_semantic:
            notes.append("SAM3 absent -> AtlasSemanticMask (SegFormer, CPU/MPS) "
                         "fallback for sky/scope")

        # 0. optional VLM assessment (advisory: auto_continue never blocks).
        image_ref = image
        vlm = None
        if use_vlm:
            vlm = g.node("AtlasAssessImage", image=image,
                         provider=vlm_provider, model=vlm_model,
                         auto_continue=True, offload_model=True)
            image_ref = vlm.out(0)
            notes.append("VLM assessment ON — prompts/geometry/bands from the plan")
            if layers > 0 and layers != 4:
                notes.append(f"layers {layers} → 4 (the VLM plan has 4 band slots)")
                layers = 4

        # 1. solve + shared depth (always). depth_model is fed the solve so
        # DA3METRIC / MoGe get the recovered focal (metric scale / fov_x).
        solve = g.node("AtlasLearnedSolveFromImage", image=image_ref)
        depth = g.node("AtlasDepthMap", image=image_ref, solve=solve.out(0),
                       depth_model=depth_model)

        # 2. sky mask (SolidMask zero when off/unavailable — every consumer
        # nearest-resizes masks, so the 64px placeholder is fine).
        zero_mask = g.node("SolidMask", value=0.0, width=64, height=64)
        sky_mask_ref = zero_mask.out(0)
        sky_on = bool(sky)
        if sky_on:
            sky_prompt_ref = vlm.out(3) if vlm is not None else sky_prompt
            sky_mask = segment(image_ref, sky_prompt_ref)
            if sky_mask is None:
                notes.append("sky SKIPPED — no segmenter (SAM3 / AtlasSemanticMask absent)")
                sky_on = False
            else:
                sky_mask_ref = sky_mask
                notes.append("sky card ON")

        # 3. geometry.
        solve_chain = solve.out(0)
        if sky_on:
            # Generous smear (the ultra workflow's 96/128 calibration): the
            # sky card must reach well below every ridge silhouette so orbit
            # reveals show smeared sky, never black.
            sky_layer = g.node("AtlasSkyDomeLayer", solve=solve_chain,
                               depth=depth.out(0), sky_mask=sky_mask_ref,
                               plate_image=image_ref,
                               edge_extend_px=96, frame_outpaint_px=128)
            solve_chain = sky_layer.out(0)

        exclude_kw = {"exclude_mask": sky_mask_ref} if sky_on else {}
        band_ref_kw = {"band_ref_mask": sky_mask_ref} if sky_on else {}

        if layers == 0:
            if mesh == "relief":
                relief = g.node("AtlasDeriveReliefMesh", solve=solve_chain,
                                depth=depth.out(0),
                                relief_grid=int(mesh_resolution),
                                depth_edge_rel=0.5,
                                max_edge_factor=float(max_edge_factor),
                                sky_heuristic=bool(sky_heuristic),
                                normal_edge_deg=float(normal_edge_deg), **exclude_kw)
                solve_chain = relief.out(0)
                notes.append(f"single relief mesh, grid {int(mesh_resolution)}")
            else:
                flat = g.node("AtlasCleanPlateLayer", solve=solve_chain,
                              depth=depth.out(0), plate_image=image_ref,
                              near_pct=0.0, far_pct=0.0,  # full range (+inf)
                              name="full_range", priority=0.0,
                              relief_grid=int(mesh_resolution), depth_edge_rel=1.5,
                              embed_matte=sky_on, band_geometry=mesh,
                              frame_outpaint_px=64,  # orbit slack past the frame edge
                              **exclude_kw, **band_ref_kw)
                solve_chain = flat.out(0)
                notes.append(f"single full-range {mesh} plane")
        else:
            bounds = _ATLAS_INPUT_BOUNDARIES.get(int(layers), ())
            edges = [0.0, *bounds, 1.0]
            n_bands = len(edges) - 1
            # far -> near, staged priorities 0/5/10/15
            scope_lines = [s.strip() for s in (scope_prompts or "").splitlines()]
            lama_loader = None
            upscaler = None
            if inpaint and not have_inpaint:
                notes.append("inpaint SKIPPED — comfyui-inpaint-nodes not installed")
            inpaint_on = bool(inpaint) and have_inpaint
            if inpaint_on:
                lama_loader = g.node("INPAINT_LoadInpaintModel", model_name="big-lama.pt")
                if (upscale_model or "").strip():
                    upscaler = g.node("UpscaleModelLoader",
                                      model_name=upscale_model.strip())
                notes.append("per-band LaMa inpaint ON"
                             + (" + upscale model" if upscaler else ""))

            for i in range(n_bands):  # i=0 farthest
                near, far = edges[n_bands - 1 - i], edges[n_bands - i]
                override = f"near_pct={near:.3f} far_pct={far:.3f}"
                if vlm is not None:
                    override = vlm.out(12 + i)  # band_far..band_fg
                name = (_ATLAS_INPUT_BAND_NAMES[i] if n_bands == 4
                        else f"band_{n_bands - i}")

                # scope: manual line (or VLM prompt) -> SAM -> AtlasScopeMask
                exclude_ref = sky_mask_ref if sky_on else zero_mask.out(0)
                prompt_val = scope_lines[i] if i < len(scope_lines) else ""
                if vlm is not None:
                    prompt_val = None  # replaced by the VLM output below
                wants_scope = (vlm is not None and vlm_scope) or bool(prompt_val)
                if wants_scope:
                    p_ref = vlm.out(4 + i) if vlm is not None else prompt_val
                    seg_ref = segment(image_ref, p_ref)
                    if seg_ref is None:
                        if prompt_val:
                            notes.append(f"{name} scope SKIPPED — no segmenter (SAM3 / AtlasSemanticMask absent)")
                    else:
                        scope = g.node("AtlasScopeMask",
                                       sky_mask=(sky_mask_ref if sky_on else zero_mask.out(0)),
                                       prompt=p_ref, segment_mask=seg_ref)
                        exclude_ref = scope.out(0)

                # plate: original photo, or the inpainted clean plate
                plate_ref = image_ref
                if inpaint_on and i < n_bands - 1:  # frontmost band never needs it
                    band_mask = g.node("AtlasDepthLayerMask", solve=solve.out(0),
                                       depth=depth.out(0), band_override=override,
                                       exclude_mask=exclude_ref, **band_ref_kw)
                    grown = g.node("INPAINT_ExpandMask", mask=band_mask.out(1),
                                   grow=32, blur=8, blur_type="gaussian")
                    crop = g.node("AtlasInpaintCrop", image=image_ref,
                                  mask=grown.out(0), context_pad_px=128)
                    lama_kw = {"optional_upscale_model": upscaler.out(0)} if upscaler else {}
                    lama = g.node("INPAINT_InpaintWithModel",
                                  inpaint_model=lama_loader.out(0),
                                  image=crop.out(0), mask=crop.out(1), seed=0,
                                  **lama_kw)
                    stitch = g.node("AtlasInpaintStitch", original_image=image_ref,
                                    inpainted_crop=lama.out(0),
                                    crop_region=crop.out(2))
                    plate_ref = stitch.out(0)

                layer_kw = {}
                if vlm is not None:
                    layer_kw["geometry_override"] = vlm.out(8 + i)  # geom_far..fg
                # DMP seam doctrine (artist-corrected 2026-07-12): the
                # extension/outpaint belongs on the layer BEHIND — the front
                # layer keeps a clean cut matte. So the frontmost band gets
                # NO extend/outpaint/skirt, every band behind gets the
                # generous smear (64/1.5/64) that covers the seam on camera
                # move. Priorities are FARTHEST-HIGHEST for the same reason:
                # at a watertight seam the two bands' surfaces are depth-
                # adjacent, and the near-tie priority bias decides which
                # paints — with nearest-highest (the first attempt) each
                # band's smear rendered IN FRONT of the layer behind it
                # (striped columns at every seam, artist-reported); with
                # farthest-highest the behind layer's real pixels win the
                # seam ribbon and its extension shows only in true reveals,
                # while genuinely nearer real geometry still wins by plain
                # depth test.
                is_front = (i == n_bands - 1)
                layer = g.node("AtlasCleanPlateLayer", solve=solve_chain,
                               depth=depth.out(0), plate_image=plate_ref,
                               band_override=override, name=name,
                               priority=float(5 * (n_bands - 1 - i)),
                               relief_grid=int(mesh_resolution),
                               depth_edge_rel=1.5,
                               max_edge_factor=float(max_edge_factor),
                               normal_edge_deg=float(normal_edge_deg),
                               fill_occluded=(inpaint_on and i < n_bands - 1),
                               embed_matte=True,
                               edge_extend_px=0 if is_front else int(edge_extend_px),
                               skirt_bevel=0.0 if is_front else 1.5,
                               frame_outpaint_px=0 if is_front else 64,
                               band_geometry=mesh,
                               exclude_mask=exclude_ref, **band_ref_kw, **layer_kw)
                solve_chain = layer.out(0)
            notes.append(f"{n_bands} band layer(s), grid {int(mesh_resolution)}")

        report = "ATLAS INPUT — expanded graph\n" + "\n".join(f"  · {n}" for n in notes)
        return {
            "result": (solve_chain, image_ref, depth.out(0), sky_mask_ref, report),
            "expand": g.finalize(),
        }


def _seg_coverage(mask_tensor) -> float:
    """Fraction of the frame a MASK covers (first batch item, >0.5).

    Shared by AtlasScopeMask's check_lazy_status and build so a borderline
    segment can't pass one coverage test and fail the other (code-review
    minor: the two used to compute it differently — whole raw tensor vs
    first-item-after-resize). Resolution-independent, so no resize needed.
    """
    t = mask_tensor if mask_tensor.dim() == 3 else mask_tensor.unsqueeze(0)
    return float((t[0] > 0.5).float().mean())


class AtlasScopeMask:
    """🎯 Per-band scope exclude builder — `sky ∪ NOT(grow(segment))`, with
    SELF-DISARMING fallbacks so a scope row can stay permanently active.

    Replaces the staged master's hand-built GrowMask → InvertMask →
    MaskComposite scope rows. The v4 design relied on the ARTIST bypassing a
    row when its layer is absent; with `AtlasAssessImage` auto-feeding the
    prompts that became a live trap (found on a real run): an ACTIVE row
    whose prompt is "" (VLM says the layer is absent), or whose prompt the
    segmenter simply can't match ("desert floor and boulder" scored 0.0%
    coverage on SAM3), inverted an EMPTY segment into an exclude-everything
    mask and silently emptied the whole layer to zero mesh.

    Fallback rule: no prompt, no segment wired, or segment coverage below
    `min_coverage_pct` → the output is the plain sky mask (= band-only
    behavior, exactly what a bypassed row used to forward). The `status`
    output says which path fired. `segment_mask` is LAZY: with an empty
    prompt the segmenter branch is never even executed.

    FAILURE MODES COVERED vs NOT: the fallbacks handle empty/no-match
    RESULTS. A SAM3Segment ERROR (model not installed, VRAM OOM) still
    aborts the whole queue — by design, a crashed segmenter is a config
    problem to surface, not to paper over.

    REQUIRED COMPANION when the output feeds percentile band nodes: wire
    the plain SKY mask into those nodes' `band_ref_mask` too. A scoped
    exclude changes each layer's depth POPULATION, so identical near/far
    percentages resolve to different metres per layer — adjacent bands
    drift apart into real metric gaps (debug-report finding, 2026-07-11).
    `band_ref_mask` pins band-edge resolution to one shared population.
    """
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("exclude_mask", "status")
    FUNCTION = "build"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sky_mask": ("MASK", {"tooltip": "The always-on base exclusion (SAM sky mask). "
                                                 "Every fallback path returns exactly this."}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "",
                    "tooltip": "The scope prompt — wire the same sam_* rail that feeds this "
                               "row's SAM3Segment. Empty = this layer is unscoped/absent: the "
                               "node returns the sky mask alone and (via lazy evaluation) the "
                               "segmenter never runs."}),
                "segment_mask": ("MASK", {"lazy": True,
                    "tooltip": "The SAM3 segment for this band's content. Only evaluated when "
                               "prompt is non-empty."}),
                "grow_px": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Dilate the segment before inverting (keeps silhouettes from "
                               "clipping — the old GrowMask 16 default)."}),
                "min_coverage_pct": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 50.0,
                    "step": 0.1,
                    "tooltip": "If the segment covers less than this % of the frame, treat the "
                               "prompt as a NO-MATCH and fall back to band-only instead of "
                               "excluding the whole layer."}),
                "fallback_mask": ("MASK", {"lazy": True,
                    "tooltip": "Geometry-prior fallback, tried BEFORE band-only when the SAM "
                               "prompt no-matches — wire an AtlasSemanticMask (fixed ADE20K "
                               "vocabulary, can't miss the way free-text prompts can). Lazy: "
                               "only evaluated on an actual no-match."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    @staticmethod
    def _wired(dynprompt, unique_id, name):
        """True when `name` is an actual graph link on this node. A lazy kwarg
        is None BOTH when unevaluated and when unconnected, and asking the
        executor for an unconnected input raises NodeInputError ("no input to
        that node at all") — so wiring must be read from the prompt graph."""
        try:
            return isinstance(dynprompt.get_node(unique_id)["inputs"].get(name), list)
        except Exception:
            return False

    def check_lazy_status(self, sky_mask, prompt="", segment_mask=None,
                          grow_px=16, min_coverage_pct=0.2, fallback_mask=None,
                          dynprompt=None, unique_id=None, **_extra):
        if not (prompt or "").strip():
            return []
        if segment_mask is None:
            if self._wired(dynprompt, unique_id, "segment_mask"):
                return ["segment_mask"]
            return []  # unwired: build() falls back to band-only
        # Segment arrived — pull the fallback only when it will actually be
        # used (coverage no-match). Same computation as build()'s, so the two
        # can never disagree on a borderline segment.
        if (_seg_coverage(segment_mask) < float(min_coverage_pct) / 100.0
                and fallback_mask is None
                and self._wired(dynprompt, unique_id, "fallback_mask")):
            return ["fallback_mask"]
        return []

    def build(self, sky_mask, prompt="", segment_mask=None, grow_px=16,
              min_coverage_pct=0.2, fallback_mask=None, **_extra):
        torch = _require_torch()
        import torch.nn.functional as F

        sky = sky_mask if sky_mask.dim() == 3 else sky_mask.unsqueeze(0)
        prompt = (prompt or "").strip()
        if not prompt:
            return (sky, "band-only (no scope prompt — layer absent or unscoped)")
        if segment_mask is None:
            return (sky, f"band-only (prompt '{prompt}' but no segment wired)")

        def _scope_with(seg_in, label):
            """Try scoping with one segment. Returns (excl, status, cov);
            excl/status are None when the segment's coverage no-matches."""
            seg = seg_in if seg_in.dim() == 3 else seg_in.unsqueeze(0)
            if tuple(seg.shape[1:]) != tuple(sky.shape[1:]):
                seg = F.interpolate(seg.unsqueeze(1).float(), size=tuple(sky.shape[1:]),
                                    mode="nearest").squeeze(1)
            cov = _seg_coverage(seg)
            if cov < float(min_coverage_pct) / 100.0:
                return None, None, cov
            grown = seg
            if grow_px and int(grow_px) > 0:
                k = int(grow_px) * 2 + 1
                grown = F.max_pool2d(seg.unsqueeze(1).float(), k, stride=1,
                                     padding=k // 2).squeeze(1)
            excl = torch.clamp(sky.float() + (1.0 - (grown > 0.5).float()), 0.0, 1.0)
            status = (f"scoped to '{prompt}' via {label} ({cov:.1%} segment, "
                      f"grown {int(grow_px)}px)")
            return excl, status, cov

        excl, status, coverage = _scope_with(segment_mask, "SAM segment")
        if excl is not None:
            return (excl, status)
        if fallback_mask is not None:
            fb_excl, fb_status, _fb_cov = _scope_with(fallback_mask, "semantic FALLBACK")
            if fb_excl is not None:
                return (fb_excl, f"{fb_status} — SAM prompt no-matched at {coverage:.2%}")
        return (sky, f"band-only FALLBACK — segment for '{prompt}' covered "
                     f"{coverage:.2%} of the frame (no-match); scoping skipped "
                     "so the layer keeps its full band")


class AtlasSemanticMask:
    """🧩 Named-class semantic mask via SegFormer/ADE20K.

    A promptless, deterministic alternative to SAM3 text segmentation:
    SegFormer assigns every pixel one of ADE20K's 150 fixed scene classes
    ("sky", "floor", "building", "tree", "person", ...). Two intended roles:
    a native sky-mask source when ComfyUI-RMBG isn't installed, and a
    geometry-prior fallback for `AtlasScopeMask.fallback_mask` when a
    free-text SAM prompt no-matches (a fixed vocabulary can't miss the way
    "desert floor and boulder" did). b0 is tiny (~15MB) and CPU-viable.
    Needs `[neural]` (transformers)."""
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "segment"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        from atlas_camera.inference.semantic_segmenter import SEGFORMER_MODELS
        return {
            "required": {
                "image": ("IMAGE",),
                "classes": ("STRING", {"default": "sky",
                    "tooltip": "Comma-separated ADE20K class names (sky, floor, building, "
                               "tree, person, road, water, mountain, ceiling, wall, ...). "
                               "The mask is the UNION of all matched classes."}),
            },
            "optional": {
                "model": (list(SEGFORMER_MODELS), {"default": SEGFORMER_MODELS[0],
                    "tooltip": "b0 = fastest/smallest, b4 = most accurate."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    def segment(self, image, classes="sky", model=None, device="auto", **_extra):
        from atlas_camera.inference.semantic_segmenter import (
            DEFAULT_SEGFORMER_MODEL, available_labels, semantic_class_mask)
        torch = _require_torch()

        pil = _image_tensor_to_pil(image)
        dev = None if device == "auto" else device
        model_id = model or DEFAULT_SEGFORMER_MODEL
        mask_np, matched, coverage = semantic_class_mask(
            pil, classes, model_id=model_id, device=dev)
        mask = torch.from_numpy(mask_np.astype("float32")).unsqueeze(0)
        if matched:
            report = (f"matched {sorted(set(matched))} -> {coverage:.1%} of frame "
                      f"({model_id.rsplit('/', 1)[-1]})")
        else:
            labels = ", ".join(sorted(available_labels(model_id, dev))[:40])
            report = (f"NO MATCH for '{classes}' — mask is empty. "
                      f"ADE20K classes include: {labels}, ...")
        return (mask, report)


class AtlasInpaintCrop:
    """✂ Crop a padded box around the inpaint mask BEFORE the inpaint model.

    The quality lever for LaMa-class inpainters, found by reading the
    installed comfyui-inpaint-nodes source: INPAINT_InpaintWithModel squashes
    the ENTIRE image to a 256×256 square for LaMa (512 for MAT), inpaints
    there, and bilinear-upscales back — on a 4K plate that is a 16× linear
    downscale, which IS the documented "LaMa smears fine structure" ceiling.
    Cropping first spends that fixed internal budget on the hole's
    neighborhood instead of the whole frame.

    `context_pad_px` is the quality/context tradeoff slider: tight = more
    effective resolution in the fill, but less surrounding texture for the
    model to sample; wide = more context, softer fill. Orchestration only
    (a tensor crop) — the inpainting itself stays in the external node
    packs, per the repo's GPL scope boundary.

    Pair with `AtlasInpaintStitch` (wire `crop_region` across). Multiple
    disjoint holes are covered by ONE union bounding box — if holes span the
    whole frame the crop degrades gracefully toward the full image, i.e.
    today's behavior, never worse.
    """
    RETURN_TYPES = ("IMAGE", "MASK", "ATLAS_CROP_REGION")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_region")
    FUNCTION = "crop"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "The inpaint mask (e.g. INPAINT_ExpandMask's "
                                             "output). The crop is its bounding box plus "
                                             "context_pad_px on every side."}),
            },
            "optional": {
                "context_pad_px": ("INT", {"default": 128, "min": 16, "max": 2048, "step": 8,
                    "tooltip": "THE quality slider: padding around the mask's bounding box. "
                               "Tight (32-64) = the inpainter's fixed internal resolution is "
                               "spent almost entirely on the hole → maximum detail, but little "
                               "surrounding texture to sample. Wide (256+) = more context, "
                               "softer fill. 128 is a good 4K-plate default."}),
            },
        }

    def crop(self, image, mask, context_pad_px=128):
        torch = _require_torch()
        import torch.nn.functional as F

        h, w = int(image.shape[1]), int(image.shape[2])
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        if tuple(m.shape[1:]) != (h, w):
            m = F.interpolate(m.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        hot = m[0] > 0.5
        ys, xs = torch.nonzero(hot, as_tuple=True)
        if len(ys) == 0:
            # Empty mask: nothing to inpaint — pass through, full-frame region.
            region = {"x0": 0, "y0": 0, "x1": w, "y1": h, "width": w, "height": h}
            return (image, m, region)
        pad = max(0, int(context_pad_px))
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(h, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(w, int(xs.max()) + 1 + pad)
        region = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": w, "height": h}
        return (image[:, y0:y1, x0:x1, :], m[:, y0:y1, x0:x1], region)


class AtlasInpaintStitch:
    """✂ Paste an inpainted crop back into the original frame.

    The other half of `AtlasInpaintCrop` — wire its `crop_region` output
    here. If the inpainted crop comes back at a different size (an upscale
    model on the inpaint node, a generative inpainter snapping to
    multiples-of-8), it is resized to the region first.

    By default the whole rectangle is pasted — exact for LaMa/MAT, whose
    node already returns original pixels outside the mask. For generative
    inpainters that re-render the entire crop, wire the SAME mask into
    `mask` (and optionally feather it) so only masked pixels land back.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "inpainted_crop": ("IMAGE",),
                "crop_region": ("ATLAS_CROP_REGION",),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional: restrict the paste to these pixels "
                                             "(full-frame mask, same one the crop used). Needed "
                                             "only for inpainters that re-render the whole crop; "
                                             "LaMa/MAT return original pixels outside the mask, "
                                             "so the default whole-rect paste is already exact."}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Soften the mask edge by this many pixels when a mask is wired "
                               "(box blur) — hides seams from generative inpainters. 0 = hard."}),
            },
        }

    def stitch(self, original_image, inpainted_crop, crop_region, mask=None, feather_px=0):
        torch = _require_torch()
        import torch.nn.functional as F

        x0, y0, x1, y1 = (int(crop_region[k]) for k in ("x0", "y0", "x1", "y1"))
        rh, rw = y1 - y0, x1 - x0
        crop = inpainted_crop
        if tuple(crop.shape[1:3]) != (rh, rw):
            crop = F.interpolate(crop.permute(0, 3, 1, 2), size=(rh, rw),
                                 mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        out = original_image.clone()
        if mask is None:
            out[:, y0:y1, x0:x1, :] = crop.to(out.dtype)
            return (out,)
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        h, w = int(original_image.shape[1]), int(original_image.shape[2])
        if tuple(m.shape[1:]) != (h, w):
            m = F.interpolate(m.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        m = m[:, y0:y1, x0:x1].unsqueeze(1).float()
        if feather_px and int(feather_px) > 0:
            k = int(feather_px) * 2 + 1
            m = F.avg_pool2d(F.pad(m, (k // 2,) * 4, mode="replicate"), k, stride=1)
        m = m.squeeze(1).unsqueeze(-1).clamp(0, 1)
        region_orig = out[:, y0:y1, x0:x1, :]
        out[:, y0:y1, x0:x1, :] = region_orig * (1.0 - m) + crop.to(out.dtype) * m
        return (out,)


_BAND_GEOMETRY_CHOICES = ("relief", "card", "ground")


def _resolve_band_geometry(band_geometry: str, geometry_override: str) -> str:
    """The override STRING (usually wired from AtlasAssessImage's geom_*
    outputs — ComfyUI rejects STRING→combo links, same constraint as
    patch_view_override) WINS over the combo when non-empty. Errors loudly
    on garbage, per the patch-view-override precedent."""
    value = (geometry_override or "").strip().lower() or (band_geometry or "relief")
    if value not in _BAND_GEOMETRY_CHOICES:
        raise ValueError(
            f"Unknown band geometry '{value}' — expected one of "
            f"{', '.join(_BAND_GEOMETRY_CHOICES)}.")
    return value


def _analytic_ground_forward_depth(extr, fx, fy, cx, cy, height, width):
    """Per-pixel forward depth of the ray∩(Y=0 ground plane) intersection,
    NaN where the ray never hits ground (at/above horizon) or the camera is
    at/below ground. Matches build_relief_mesh's back-projection EXACTLY:
    with the unnormalized camera ray ((u-cx)/fx, -(v-cy)/fy, -1), the ray
    parameter IS forward depth, so feeding this array back through the mesh
    builder lands every vertex on Y=0 by construction (same ray-plane math
    as _ground_depth_compute / the viewport's DEPTH_FRAGMENT_SHADER)."""
    np = _require_numpy()
    vm = np.array(extr.camera_view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(vm)
    R = c2w[:3, :3]
    cam_y = float(c2w[1, 3])
    out = np.full((height, width), np.nan)
    if cam_y <= 1e-6:
        return out
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    kx = (uu - cx) / fx
    ky = -(vv - cy) / fy
    # World-Y component of the unnormalized ray direction.
    ry = R[1, 0] * kx + R[1, 1] * ky - R[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -cam_y / ry
    ok = np.isfinite(s) & (s > 1e-3)
    out[ok] = s[ok]
    return out


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
                    "display": "slider",
                    "tooltip": "Must resolve to the same band as the AtlasDepthLayerMask that "
                               "produced plate_image (both call the shared _resolve_depth_band "
                               "helper, so identical near_m/far_m/near_pct/far_pct here and there "
                               "always agree). LOWER near_pct = tighter occlusion, not looser — "
                               "see AtlasDepthLayerMask's near_pct tooltip for the worked example."}),
                "far_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Must resolve to the same band as the AtlasDepthLayerMask that "
                               "produced plate_image. 0 means no upper bound (+inf)."}),
                "name": ("STRING", {"default": "layer"}),
                "priority": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blend priority among layers (higher wins). FARTHEST bands take the "
                               "HIGHER priority (far 15 / bg 10 / mid 5 / fg 0): at a watertight "
                               "seam the surfaces are depth-adjacent and this near-tie bias picks "
                               "the winner, so nearest-highest renders a band's edge smear IN "
                               "FRONT of the layer behind it (striped seams). Min is 0; a sky "
                               "dome goes negative via AtlasSkyDomeLayer. "
                               "Ordering is by depth + priority, never "
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
                "band_side": (["manual", "foreground", "background"], {"default": "manual",
                    "tooltip": "With band_split connected: foreground = [0, split), background "
                               "= [split, +inf) — the node's own near/far widgets are ignored. "
                               "manual = use this node's own near/far settings."}),
                "band_split": ("ATLAS_BAND_SPLIT", {
                    "tooltip": "Wire ONE AtlasDepthBandSplit into every band node (with "
                               "band_side set) so the fg/bg boundary lives in exactly one "
                               "widget and the layers can never drift apart."}),
                # APPENDED last (widgets_values is positional — never insert).
                "band_geometry": (list(_BAND_GEOMETRY_CHOICES), {"default": "relief",
                    "tooltip": "How this band's projection surface is built. relief (default) = "
                               "the depth-following mesh, for anything with real 3D shape inside "
                               "the band. card = ONE flat fronto-parallel plane at the band's "
                               "median depth (classic DMP card) — for distant/flat-facing layers "
                               "with negligible internal parallax (far mountains at the horizon, "
                               "a hangar's back wall, a skyline backdrop); never tears, zero "
                               "depth noise. ground = the exact analytic Y=0 ground plane — for "
                               "flat horizontal surfaces the camera stands over (desert floor, "
                               "water, road); zero depth-noise bumps. Both flat modes keep band "
                               "membership from the REAL depth (which pixels belong) and only "
                               "flatten WHERE they sit; matte/edge-extend/outpaint all still "
                               "apply."}),
                "geometry_override": ("STRING", {"default": "",
                    "tooltip": "Optional geometry-type override STRING — wins over band_geometry "
                               "when non-empty ('relief'/'card'/'ground'). Exists because ComfyUI "
                               "rejects STRING→combo links: wire AtlasAssessImage's geom_far/bg/"
                               "mid/fg output here so the VLM's per-layer geometry recommendation "
                               "flows in (same pattern as patch_view_override). Unknown values "
                               "error loudly."}),
                "band_ref_mask": ("MASK", {
                    "tooltip": "Exclusion used ONLY for resolving near/far percentages to "
                               "metres. When exclude_mask carries per-layer scoping (🎯 scope "
                               "rows), each layer's depth population differs and the shared "
                               "band edges DRIFT apart (metric gaps between adjacent bands — "
                               "debug-report finding). Wire the plain SKY mask here on every "
                               "band node so all layers resolve identical edges. Unwired = "
                               "legacy behavior (band edges from exclude_mask's population)."}),
                # APPENDED last (widgets_values is positional — never insert).
                "band_override": ("STRING", {"default": "",
                    "tooltip": "Optional band override STRING ('near_pct=<f> far_pct=<f>') — "
                               "wins over this node's near/far widgets when non-empty. Wire "
                               "AtlasAssessImage's band_far/bg/mid/fg output here so the VLM's "
                               "subject-aware band boundaries flow in (jointly derived, so "
                               "adjacent bands always share edges exactly). MUST be the same "
                               "string the paired AtlasDepthLayerMask received. Loses to a "
                               "connected band_split. Errors loudly on garbage."}),
                # Tearing knobs, mirroring AtlasDeriveReliefMesh (freeze exception:
                # these are core mesh-tearing params, siblings of depth_edge_rel /
                # relief_grid, not a new capability — band mode was the only relief
                # path that couldn't reach them).
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "World-space edge tear threshold (SEPARATE from depth_edge_rel). "
                               "Dominant tear cause on deep / narrow-FOV / interior bands: raise "
                               "to 40-80 to stop comb-tearing continuous grazing surfaces. >80 "
                               "rubber-sheets real silhouettes."}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "0 = off. Tears where surface NORMALS bend past this angle — real "
                               "creases / occlusion silhouettes — while leaving smoothly-receding "
                               "walls intact. Pair with a higher max_edge_factor: raise mef to kill "
                               "spurious combs, then ~40-70 here to keep genuine edges torn."}),
            },
        }

    def add_layer(self, solve, depth, plate_image, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5,
                  name="layer", priority=0.0, plate_ref=None, relief_grid=384, depth_edge_rel=1.5,
                  exclude_mask=None, fill_occluded=False, embed_matte=False, layer_matte=None,
                  edge_extend_px=0, skirt_bevel=0.0, frame_outpaint_px=0,
                  exclude_choke_cells=2, band_side="manual", band_split=None,
                  band_geometry="relief", geometry_override="", band_ref_mask=None,
                  band_override="", max_edge_factor=12.0, normal_edge_deg=0.0):
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
        override = _parse_band_override(band_override)
        if override is not None:
            near_m = far_m = 0.0
            near_pct, far_pct = override
        near, far = _apply_band_split(band_split, band_side, setup.metric,
                                      _band_resolution_validity(setup, band_ref_mask),
                                      near_m, far_m, near_pct, far_pct)

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

        # Per-layer geometry type: the flat modes substitute the depth FIELD
        # fed to build_relief_mesh — band membership still comes from the
        # REAL depth (which pixels belong to this layer); geometry only
        # changes WHERE those pixels sit. Out-of-region pixels become NaN,
        # which is invalid-but-regrowable exactly like band clipping (matte
        # skirts still grow); real exclusions stay the hard skirt forbid.
        geometry = _resolve_band_geometry(band_geometry, geometry_override)
        band_min_for_mesh = near
        band_max_for_mesh = None if far == float("inf") else far
        fill_for_mesh = fill_m
        heuristic = exclude_m is None
        if geometry != "relief":
            band_region = valid_m & (metric_m >= near)
            if far != float("inf"):
                band_region &= metric_m <= far
            if fill_m is not None:
                # Flat depth covers the occluder footprint for free — include
                # it in the region instead of diffusion-filling it.
                band_region = band_region | (
                    fill_m if exclude_m is None else (fill_m & ~exclude_m))
            if geometry == "card":
                # One fronto-parallel plane at the band's median depth — the
                # classic DMP card; matches the projection_backdrop / sky
                # dome constant-forward-Z convention.
                const_raw = float(np.median(depth_m[band_region])) if band_region.any() else 1.0
                geo_depth = np.full(depth_m.shape, const_raw, dtype=np.float64)
            else:  # ground
                # The exact analytic Y=0 plane along each pixel ray — raw
                # units are metric/scale so build_relief_mesh's internal
                # rescale-about-camera lands vertices on Y=0 on the nose.
                geo_metric = _analytic_ground_forward_depth(extr, fx, fy, cx_m, cy_m, Hp, Wp)
                if not np.isfinite(geo_metric).any():
                    raise ValueError(
                        "band_geometry='ground' needs a camera above the ground plane "
                        "(solved camera height <= 0, or no ray ever hits Y=0).")
                band_region &= np.isfinite(geo_metric)
                # Non-ground pixels in the band (a wall base, an occluder's
                # side) have analytic ground depths FAR beyond the band —
                # near-horizontal rays run out toward the horizon. Cap at the
                # band's far edge (or 4x the band's real 99th-pct depth when
                # the band is open-ended) so only plausible ground-plane
                # membership survives; the rest become holes/skirt.
                if far != float("inf"):
                    ground_cap = float(far)
                elif band_region.any():
                    ground_cap = 4.0 * float(np.percentile(metric_m[band_region], 99.0))
                else:
                    ground_cap = float("inf")
                with np.errstate(invalid="ignore"):
                    band_region &= ~(geo_metric > ground_cap)
                geo_depth = geo_metric / max(float(scale), 1e-9)
            depth_m = np.where(band_region, geo_depth, np.nan)
            band_min_for_mesh = None   # region already encodes membership;
            band_max_for_mesh = None   # analytic ground may exceed the band
            fill_for_mesh = None
            heuristic = False          # constant/far flat depth IS "sky" to
            #                            the heuristic — must never run here

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
            band_min_m=band_min_for_mesh, band_max_m=band_max_for_mesh,
            exclude_mask=exclude_m, fill_mask=fill_for_mesh,
            apply_sky_heuristic=heuristic,
            # Flat modes feed an ANALYTIC field: the far-percentile clamp
            # would float legit on-plane ground off the plane, and smoothing
            # only corrupts a field with no noise to remove.
            far_clip_percentile=(0.0 if geometry != "relief" else 97.0),
            smooth_iterations=(0 if geometry != "relief" else 2),
            max_edge_factor=float(max_edge_factor),
            normal_edge_deg=(float(normal_edge_deg) if float(normal_edge_deg) > 0 else None),
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

        # Predicted-normal relight map (MoGe *-normal): align the model-frame
        # per-pixel normals to the recovered WORLD frame and embed them so the
        # viewport lights read the true surface orientation at image resolution.
        # Skipped when the source is frame-outpainted (pad > 0) — the normal map
        # would then be out of uv-registration with the widened plate; the
        # geometry normal + luminance bump still apply there.
        normal_map_b64 = None
        raw_normal = getattr(depth, "normal", None)
        if raw_normal is not None and pad == 0:
            try:
                from atlas_camera.core.normals import (
                    align_predicted_normals_to_world,
                    encode_normal_map_b64,
                )
                rn = np.asarray(raw_normal, dtype=np.float64)
                if rn.ndim == 3 and rn.shape[:2] == setup.depth_map.shape:
                    world_n, n_valid = align_predicted_normals_to_world(
                        rn, setup.depth_map, view_matrix=extr.camera_view_matrix,
                        fx=fx, fy=fy, cx=cx, cy=cy)
                    normal_map_b64 = encode_normal_map_b64(world_n, n_valid)
            except Exception:
                normal_map_b64 = None

        source = ProjectionSource(
            camera=src_camera,  # primary POSE unchanged; intrinsics widened when outpainted
            name=name,
            image_b64=image_b64,
            plate_ref=plate_ref if isinstance(plate_ref, AtlasPlateRef) else AtlasPlateRef.from_dict(plate_ref),
            proxy_geometry=patch_geom,
            priority=float(priority),
            normal_map_b64=normal_map_b64,
            metadata={
                "projection_mode": "clean_plate",
                "source": "inpaint_layer",
                "band_geometry": geometry,
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

        # 🩻 Hidden-geometry provenance pass-through: when the wired depth was
        # patched by AtlasPredictHiddenGeometry, its metadata carries the
        # substitution mask + backend — ride them into this ProjectionSource so
        # the viewport's debug overlay can tint the invented surface region.
        # Resized/padded to this source's (possibly frame-outpainted) plate/uv
        # space, matching the embedded matte's conventions.
        dmeta = getattr(depth, "metadata", None) or {}
        if dmeta.get("hidden_mask_b64"):
            hm = _b64_png_to_mask(dmeta["hidden_mask_b64"])
            if hm is not None:
                from atlas_camera.core.solver import _resize_depth
                if hm.shape != (setup.height, setup.width):
                    hm = _resize_depth(
                        hm.astype(np.float32), setup.width, setup.height) > 0.5
                if pad:
                    hm = np.pad(hm, pad, mode="edge")
                enc = _mask_to_b64_png(hm)
                if enc:
                    source.metadata["hidden_mask_b64"] = enc
                    source.metadata["hidden_backend"] = (
                        dmeta.get("hidden_backend") or "lari")

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


class AtlasCleanPlateStack:
    """🧽 Up to FOUR artist-painted cleanplates + alphas → layered scene.

    The multi-slot cleanplate injection port: the artist separates the plate
    in Photoshop (e.g. sky / mountains / buildings / dirt road), saves each
    stratum as a full-frame plate plus an alpha matte, and wires each pair
    into a slot. Slot 1 is the FARTHEST stratum, slot 4 the nearest —
    priorities are assigned farthest-highest (15/10/5/0, the seam doctrine),
    and every used slot except the NEAREST gets `edge_extend_px` smear while
    the nearest keeps a clean cut (the DMP seam rule, baked in).

    Pure composition over :class:`AtlasCleanPlateLayer` (its capability
    freeze is respected — this node adds no math): per slot the matte is
    grown by `grow_px`, its inverse becomes the geometry `exclude_mask`
    (mask-membership, the X-ray layer pattern) and the raw matte becomes the
    paint `layer_matte`. Slots missing a plate OR a matte — or with an empty
    matte — are skipped and named in the report, never an error. With no
    complete slot the input solve passes through untouched.

    Tip: save each separation as a PNG with alpha and wire ONE LoadImage per
    slot — IMAGE → plate_N and MASK → matte_N. ComfyUI's LoadImage MASK
    output marks TRANSPARENT pixels, so flip `mattes_are_transparency` ON
    for that wiring (or pre-invert with InvertMask).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "stack"
    CATEGORY = "Atlas Camera"

    _PRIORITIES = (15.0, 10.0, 5.0, 0.0)   # slot 1..4, farthest-highest

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        defaults_name = ("far_sky", "background", "midground", "foreground")
        defaults_geo = ("card", "relief", "relief", "relief")
        for i in range(1, 5):
            opt[f"plate_{i}"] = ("IMAGE",)
            opt[f"matte_{i}"] = ("MASK",)
        for i in range(1, 5):
            opt[f"name_{i}"] = ("STRING", {"default": defaults_name[i - 1]})
            opt[f"geometry_{i}"] = (["relief", "card", "ground"],
                                    {"default": defaults_geo[i - 1]})
        opt["grow_px"] = ("INT", {"default": 12, "min": 0, "max": 256,
                                  "tooltip": "matte safety grow before the geometry cut"})
        opt["edge_extend_px"] = ("INT", {"default": 24, "min": 0, "max": 256,
                                         "tooltip": "smear on the BEHIND slots; the nearest used slot always stays a clean cut"})
        opt["relief_grid"] = ("INT", {"default": 384, "min": 16, "max": 4096})
        opt["depth_edge_rel"] = ("FLOAT", {"default": 1.5, "min": 0.05, "max": 8.0, "step": 0.05})
        opt["mattes_are_transparency"] = ("BOOLEAN", {"default": False,
                                          "tooltip": "ON when mattes come straight from LoadImage's MASK output (which marks TRANSPARENT pixels) — inverts them"})
        return {"required": {"solve": ("ATLAS_SOLVE",), "depth": ("ATLAS_DEPTH_MAP",)},
                "optional": opt}

    def stack(self, solve, depth, grow_px=12, edge_extend_px=24, relief_grid=384,
              depth_edge_rel=1.5, mattes_are_transparency=False, **slots):
        torch = _require_torch()
        import torch.nn.functional as F

        def grown(matte):
            if grow_px <= 0:
                return matte
            k = 2 * int(grow_px) + 1
            return F.max_pool2d(matte.unsqueeze(1), kernel_size=k, stride=1,
                                padding=int(grow_px)).squeeze(1)

        used = []
        report = []
        for i in range(1, 5):
            plate = slots.get(f"plate_{i}")
            matte = slots.get(f"matte_{i}")
            if plate is None and matte is None:
                continue
            if plate is None or matte is None:
                report.append(f"slot {i}: SKIPPED — needs BOTH plate_{i} and matte_{i}")
                continue
            if mattes_are_transparency:
                matte = 1.0 - matte
            if float(matte.max()) <= 0.0:
                report.append(f"slot {i}: SKIPPED — matte is empty")
                continue
            used.append((i, plate, matte))

        if not used:
            report.append("no complete plate+matte slots — solve passes through untouched")
            return (copy.deepcopy(solve), "\n".join(report))

        nearest_i = used[-1][0]
        cur = solve
        layer_node = AtlasCleanPlateLayer()
        for i, plate, matte in used:
            g = grown(matte)
            exclude = 1.0 - g
            smear = 0 if i == nearest_i else int(edge_extend_px)
            name = slots.get(f"name_{i}") or f"cleanplate_{i}"
            geometry = slots.get(f"geometry_{i}") or "relief"
            cur = layer_node.add_layer(
                cur, depth, plate,
                near_pct=0.0, far_pct=1.0,
                name=name, priority=self._PRIORITIES[i - 1],
                relief_grid=int(relief_grid), depth_edge_rel=float(depth_edge_rel),
                exclude_mask=exclude, fill_occluded=False,
                embed_matte=True, layer_matte=matte,
                edge_extend_px=smear, band_geometry=geometry,
            )[0]
            report.append(f"slot {i}: '{name}' added — geometry={geometry} "
                          f"priority={self._PRIORITIES[i - 1]:g} edge_extend={smear}"
                          + ("  (nearest: clean cut)" if i == nearest_i else ""))
        return (cur, "\n".join(report))


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

    Geometrically this is a flat card at a constant forward-Z depth —
    `radius_m` alone (legacy), or `distance_m` when set, with `radius_m`
    then acting as the card's minimum half-extent (SIZE, grown via honest
    outpaint) — the same convention `build_relief_mesh` uses everywhere
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
                    "tooltip": "With distance_m at 0 (default): the card's DISTANCE in metres "
                               "(forward-Z, legacy behavior) — should comfortably exceed the "
                               "scene's own derived backdrop distance so it never intersects real "
                               "geometry. With distance_m set: the card's minimum half-extent — "
                               "its SIZE, radius in the dome sense — the card is enlarged (never "
                               "shrunk below frustum coverage) via extra outpaint so it reaches "
                               "this world size at that distance. Distance doesn't affect "
                               "appearance from the solve camera (texel assignment is by ray); it "
                               "controls parallax — how far you can dolly/orbit before the card "
                               "reveals itself."}),
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
                # APPENDED last (widgets_values is positional — never insert).
                "distance_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 1.0,
                    "tooltip": "Card distance from the camera in metres. 0 (default) = legacy "
                               "behavior: radius_m IS the distance and size follows from the "
                               "frustum. When set, this places the card and radius_m becomes its "
                               "minimum half-extent (SIZE): if the frustum footprint at this "
                               "distance is smaller than radius_m, the card grows via extra "
                               "outpaint (edge-replicated pixels, declared in extend_mask; total "
                               "padding memory-capped at half the plate's long edge per side). "
                               "Distance = parallax; size = orbit/pan slack."}),
            },
        }

    def add_layer(self, solve, depth, sky_mask, plate_image, radius_m=300.0, relief_grid=96,
                  name="sky", priority=-10.0, plate_ref=None, edge_extend_px=48,
                  frame_outpaint_px=64, distance_m=0.0):
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

        # Distance vs size (user feature request 2026-07-11): with distance_m
        # set, the card sits THERE and radius_m becomes its minimum
        # half-extent (SIZE). Extra size is honest outpaint — the frustum
        # footprint at that distance is padded out with edge-replicated
        # pixels (declared invented via extend_mask below) until the card's
        # world half-extent reaches radius_m. Never shrinks below frustum
        # coverage (that would punch holes around the sky's frame edges).
        # distance_m=0 keeps the legacy single-knob behavior bit-identical.
        card_distance = float(distance_m) if float(distance_m) > 0.0 else float(radius_m)
        pad = max(0, int(frame_outpaint_px))
        size_pad = 0
        if float(distance_m) > 0.0:
            need_x = float(radius_m) * fx_p / card_distance - (Wp / 2.0 + pad)
            need_y = float(radius_m) * fy_p / card_distance - (Hp / 2.0 + pad)
            size_pad = int(np.ceil(max(0.0, need_x, need_y)))
            if size_pad:
                # Memory guard: total extra padding capped at half the plate
                # long edge per side (canvas at most ~2x linear).
                size_pad = min(size_pad, max(Hp, Wp) // 2)
                pad += size_pad
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
            radius_m=card_distance, grid_long_edge=int(relief_grid),
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
                "distance_m": card_distance,     # where the card actually sits
                "size_pad_px": size_pad,         # extra outpaint added for SIZE
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
    "AtlasLoadRAW":               AtlasLoadRAW,
    # Track 1 — solve
    "AtlasSolveFromImage":        AtlasSolveFromImage,
    "AtlasLearnedSolveFromImage": AtlasLearnedSolveFromImage,
    "AtlasScaleOverride":         AtlasScaleOverride,
    "AtlasRollTrim":              AtlasRollTrim,
    "AtlasReferenceScaleSolve":   AtlasReferenceScaleSolve,
    "AtlasVLMScaleCues":          AtlasVLMScaleCues,
    "AtlasAssessImage":           AtlasAssessImage,
    "AtlasSolveGate":             AtlasSolveGate,
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
    "AtlasMogeNormals":           AtlasMogeNormals,
    # Experimental (research-only)
    "AtlasDeriveReliefMesh":      AtlasDeriveReliefMesh,
    "AtlasDeriveWalls":           AtlasDeriveWalls,
    "AtlasDeriveTowersSpires":    AtlasDeriveTowersSpires,
    "AtlasDeriveRoofsFacades":    AtlasDeriveRoofsFacades,
    "AtlasDeriveInteriorRoom":    AtlasDeriveInteriorRoom,
    "AtlasMergeGeometry":         AtlasMergeGeometry,
    # Track 6 — shot format
    "AtlasDefineShotCam":         AtlasDefineShotCam,
    # Track 7 — inpaint layers
    "AtlasDepthBandSplit":        AtlasDepthBandSplit,
    "AtlasBoundedBand":           AtlasBoundedBand,
    "AtlasDepthLayerMask":        AtlasDepthLayerMask,
    "AtlasCleanPlateLayer":       AtlasCleanPlateLayer,
    "AtlasCleanPlateStack":       AtlasCleanPlateStack,
    "AtlasSkyDomeLayer":          AtlasSkyDomeLayer,
    "AtlasInpaintCrop":           AtlasInpaintCrop,
    "AtlasInpaintStitch":         AtlasInpaintStitch,
    "AtlasScopeMask":             AtlasScopeMask,
    "AtlasSemanticMask":          AtlasSemanticMask,
    "AtlasDebugReport":           AtlasDebugReport,
    "AtlasLayerPreview":          AtlasLayerPreview,
    "AtlasInput":                 AtlasInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing
    "AtlasLoadImageSolveCamera":  "Atlas Load Image / Solve Camera (Deprecated)",
    "AtlasExportReviewPackage":   "Atlas Export Review Package",
    "AtlasExportSolveJSON":       "Atlas Export Solve JSON",
    "AtlasExportMayaReviewScene": "Atlas Export Maya Review Scene",
    "AtlasUSDCameraLoader":       "Atlas USD Camera Loader",
    "AtlasRegisterPlate":         "Atlas Register Plate (Float-Safe) 🎞",
    "AtlasAttachSourcePlate":     "Atlas Attach Source Plate 🎞",
    "AtlasLoadRAW":               "Atlas Load RAW (NEF/CR2/CR3/RAF/ARW) 📷",
    # Track 1 — solve
    "AtlasSolveFromImage":        "Atlas Solve Camera from Image",
    "AtlasLearnedSolveFromImage": "Atlas Learned Solve (GeoCalib) 🧠",
    "AtlasScaleOverride":         "Atlas Scale Override 📐",
    "AtlasRollTrim":              "Atlas Roll Trim 🎚",
    "AtlasReferenceScaleSolve":   "Atlas Reference-Object Scale 📏",
    "AtlasAssessImage":           "Atlas Assess Image 🧭",
    "AtlasSolveGate":             "Atlas Solve Gate ✅",
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
    "AtlasMogeNormals":           "Atlas MoGe Normals 🧭",
    "AtlasDeriveReliefMesh":      "Atlas Derive Relief Mesh 🏔",
    "AtlasDeriveWalls":           "Atlas Derive Walls 🧱",
    "AtlasDeriveTowersSpires":    "Atlas Derive Towers & Spires 🗼",
    "AtlasDeriveRoofsFacades":    "Atlas Derive Roofs & Facades 🏛",
    "AtlasDeriveInteriorRoom":    "Atlas Derive Interior Room 🛋",
    "AtlasMergeGeometry":         "Atlas Merge Geometry 🔀",
    # Track 6 — shot format
    "AtlasDefineShotCam":         "Atlas Define Shot Cam 🎬",
    # Track 7 — inpaint layers
    "AtlasDepthBandSplit":        "Atlas Depth Band Split 🎚",
    "AtlasBoundedBand":           "Atlas Bounded Band 📏",
    "AtlasDepthLayerMask":        "Atlas Depth Layer Mask 🎭",
    "AtlasCleanPlateLayer":       "Atlas Clean Plate Layer 🖼",
    "AtlasCleanPlateStack":       "Atlas Clean Plate Stack 🧽 (up to 4 plates + alphas)",
    "AtlasSkyDomeLayer":          "Atlas Sky Dome Layer ☁",
    "AtlasInpaintCrop":           "Atlas Inpaint Crop ✂",
    "AtlasInpaintStitch":         "Atlas Inpaint Stitch ✂",
    "AtlasScopeMask":             "Atlas Scope Mask 🎯",
    "AtlasSemanticMask":          "Atlas Semantic Mask 🧩",
    "AtlasDebugReport":           "Atlas Debug Report 🔍",
    "AtlasLayerPreview":          "Atlas Layer Preview 🎨",
    "AtlasInput":                 "Atlas Input 🎬",
}

# ---------------------------------------------------------------------------
# Experimental tier (🔬) — heavier external requirements than the core node
# set (user-cloned upstream repos, Docker, CUDA-class GPUs). Registered only
# when the ATLAS_EXPERIMENTAL env var is truthy, so the standard install's
# node menu stays universal and nothing here can confuse a stock ComfyUI.
# The `experimental` branch ships ATLAS_EXPERIMENTAL_DEFAULT = "1" (that one
# line is the entire branch delta); on any branch, setting
# ATLAS_EXPERIMENTAL=1 (or 0) before launching ComfyUI overrides the default.
ATLAS_EXPERIMENTAL_DEFAULT = "0"

EXPERIMENTAL_NODE_CLASS_MAPPINGS = {
    "AtlasPredictHiddenGeometry": AtlasPredictHiddenGeometry,
    "AtlasRenderFix": AtlasRenderFix,
    "AtlasExtractAnglePatch": AtlasExtractAnglePatch,
    "AtlasImportAnglePatch": AtlasImportAnglePatch,
}

EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasPredictHiddenGeometry": "Atlas Predict Hidden Geometry 🔬 (research)",
    "AtlasRenderFix": "Atlas Render Fix 🔬 (experimental)",
    "AtlasExtractAnglePatch": "Atlas Extract Angle Patch 🔬 → Photoshop",
    "AtlasImportAnglePatch": "Atlas Import Angle Patch 🔬 ← Photoshop",
}


def _experimental_enabled() -> bool:
    v = os.environ.get("ATLAS_EXPERIMENTAL", ATLAS_EXPERIMENTAL_DEFAULT)
    return v.strip().lower() not in ("", "0", "false", "off", "no")


if _experimental_enabled():
    NODE_CLASS_MAPPINGS.update(EXPERIMENTAL_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS)

