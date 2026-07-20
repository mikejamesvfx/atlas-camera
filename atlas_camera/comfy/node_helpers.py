"""Shared helpers, constants, and internal adapter classes for the
Atlas ComfyUI node library.

Extracted verbatim from ``nodes.py`` during its modularization. This is a
leaf module: it depends only on ``atlas_camera.core`` / exporters /
importers, never on any registered node class, so it cannot introduce an
import cycle.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
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


_IDENTITY_COMMENT_PREFIX = {".nk": "# ", ".py": "# ", ".ma": "// "}


def _write_export_manifest(solve, output_dir, kind_paths, exporter: str) -> None:
    """Write/merge atlas_project.json beside an export + embed the identity
    hash as a leading comment in text artifacts that tolerate one (.nk/.py/.ma).

    A manifest failure must NEVER fail the export — everything degrades to a
    log line. Called with [(kind, path), ...]; empty paths are skipped.
    """
    import logging
    try:
        from atlas_camera.exporters.manifest import (
            ManifestArtifact,
            manifest_identity_hash,
            write_project_manifest,
        )
        pairs = [(k, str(p)) for k, p in kind_paths if p]
        if not pairs:
            return
        write_project_manifest(
            solve, output_dir,
            artifacts=[ManifestArtifact(k, p, exporter) for k, p in pairs])
        ident = manifest_identity_hash(solve)
        for _, p in pairs:
            prefix = _IDENTITY_COMMENT_PREFIX.get(Path(p).suffix.lower())
            if not prefix or not Path(p).is_file():
                continue
            try:
                text = Path(p).read_text(encoding="utf-8")
                marker = f"{prefix}atlas_project_identity: "
                if text.startswith(marker):
                    text = text.split("\n", 1)[-1]
                Path(p).write_text(f"{marker}{ident}\n{text}", encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    except Exception as exc:  # noqa: BLE001
        logging.warning("atlas_project.json manifest skipped: %s", exc)


def _health_summary_suffix(solve) -> str:
    """Export-summary marker when a scene-health stamp records warn/fail.

    Reads only the AtlasSceneHealthGate stamp (debug_metadata["scene_health"])
    — an acknowledged warning must survive into every artifact's summary.
    """
    stamp = (getattr(solve, "debug_metadata", None) or {}).get("scene_health")
    if not isinstance(stamp, dict) or stamp.get("level") in (None, "pass"):
        return ""
    n = len(stamp.get("flags") or [])
    ack = "acknowledged" if stamp.get("acknowledged") else "UNACKNOWLEDGED"
    return f" | 🩺 health: {str(stamp['level']).upper()} ({n} flag(s) {ack})"


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


def _reference_id_choices() -> list[str]:
    try:
        from atlas_camera.reference_data import load_scale_references
        return [r.id for r in load_scale_references()]
    except Exception:
        return ["person_175cm", "door_210cm", "sedan_car"]


def _extrinsics_from_view(extr, vm2):
    """Write a new 4x4 world→cam view matrix onto ``extr`` and rebuild the
    rigid family (world matrix, 3x3 rotation, position) from it. Returns the
    cam→world rotation rows (r_cw) for horizon recomputation."""
    extr.camera_view_matrix = tuple(tuple(row) for row in vm2)
    r_wc = [[vm2[r][k] for k in range(3)] for r in range(3)]
    t_wc = [vm2[r][3] for r in range(3)]
    r_cw = [[r_wc[k][r] for k in range(3)] for r in range(3)]
    pos = [-sum(r_cw[r][k] * t_wc[k] for k in range(3)) for r in range(3)]
    extr.camera_world_matrix = tuple(
        tuple([*r_cw[r], pos[r]]) for r in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    extr.camera_rotation_matrix = tuple(tuple(row) for row in r_cw)
    extr.camera_position = tuple(pos)
    return r_cw


def _recompute_horizon_line(out, r_cw):
    """Refresh the stored horizon line for a re-oriented camera (the RollTrim
    vanishing-line math: world-Y ray component zero, linear in (u, v))."""
    intr = out.camera.intrinsics
    if out.horizon_line is None or not intr.fx_px or not intr.image_width:
        return
    fx = float(intr.fx_px)
    fy = float(intr.fy_px or intr.fx_px)
    cx = float(intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0)
    cy = float(intr.cy_px if intr.cy_px is not None else (intr.image_height or 0) / 2.0)
    w = float(intr.image_width)
    a = r_cw[1][0] / fx
    b = -r_cw[1][1] / fy
    cc = -r_cw[1][0] * cx / fx + r_cw[1][1] * cy / fy - r_cw[1][2]
    if abs(b) > 1e-12:
        y0 = (-cc - a * 0.0) / b
        y1 = (-cc - a * w) / b
        out.horizon_line.endpoints_px = ((0.0, y0), (w, y1))
        out.horizon_line.line_coefficients = (a, b, cc)


_ATLAS_ASSESS_CACHE: dict = {}


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


# Hand-mirrors atlas_blockout.js's LAYER_DEBUG_PALETTE / LAYER_DEBUG_PRIMARY
# (the 🎨 Layers legend) — keep both in sync by hand, the accepted-duplication
# pattern. Index = position in projection_sources; -1 = the primary teal.
_LAYER_DEBUG_PRIMARY_HEX = "2fd6c3"
_LAYER_DEBUG_PALETTE_HEX = ("ff6a3d", "3d8bff", "ffd23d", "c95aff", "6aff5a", "ff5aa8")


def _comfy_registry():
    """ComfyUI's global node registry, or {} outside ComfyUI — used by
    AtlasInput's expansion to feature-detect third-party packs (SAM3 /
    inpaint) without ever importing their code."""
    try:
        import nodes as comfy_nodes  # ComfyUI's own top-level module
        return comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return {}


def _native_sam3_available() -> bool:
    """Cheap, network-free capability probe for native SAM3 (AtlasSAM3Mask),
    used by AtlasInput's build-time cascade decision. Native SAM3 is
    ALWAYS registered (it's Atlas's own node class), so registry presence
    (unlike third-party packs) can't distinguish "the [sam3] extra +
    transformers>=5.5.4 actually works" from "the class merely exists" —
    this delegates to the real inference-layer check instead. Any failure
    (module missing, unexpected error) is treated as unavailable, the same
    fail-soft contract as _comfy_registry()."""
    try:
        from atlas_camera.inference.sam3_segmenter import native_sam3_available
        return native_sam3_available()
    except Exception:
        return False


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


def _seg_coverage(mask_tensor) -> float:
    """Fraction of the frame a MASK covers (first batch item, >0.5).

    Shared by AtlasScopeMask's check_lazy_status and build so a borderline
    segment can't pass one coverage test and fail the other (code-review
    minor: the two used to compute it differently — whole raw tensor vs
    first-item-after-resize). Resolution-independent, so no resize needed.
    """
    t = mask_tensor if mask_tensor.dim() == 3 else mask_tensor.unsqueeze(0)
    return float((t[0] > 0.5).float().mean())


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


__all__ = [
    '_DEPTH_MODEL_CHOICES',
    '_MOGE_NORMAL_MODEL_CHOICES',
    '_ATLAS_BLOCKOUT_CACHE',
    '_ATLAS_BLOCKOUT_CACHE_MAX',
    '_blockout_cache_set',
    '_solve_focal_px_for_image',
    '_require_numpy',
    '_require_torch',
    '_require_pil',
    '_image_tensor_to_pil',
    '_pil_to_image_tensor',
    '_save_image_tensor_to_tmp',
    '_resolve_raw_hints',
    '_scale_summary_suffix',
    '_IDENTITY_COMMENT_PREFIX',
    '_write_export_manifest',
    '_health_summary_suffix',
    '_stamp_raw_provenance',
    '_extend_edge_colors',
    '_b64_png_to_mask',
    '_mask_to_b64_png',
    '_image_tensor_to_preview_b64',
    '_clone_solve_with_metadata',
    '_decode_b64_to_tensor',
    '_image_fingerprint',
    '_solve_fingerprint',
    '_execution_blocker',
    '_ground_depth_compute',
    '_reference_id_choices',
    '_extrinsics_from_view',
    '_recompute_horizon_line',
    '_ATLAS_ASSESS_CACHE',
    '_solve_camera_params',
    '_horizon_y_from_solve',
    '_depth_map_for_solve',
    '_replace_proxy_role_geometry',
    '_MetricDepthSetup',
    '_BORDER_FLOOD_PX',
    '_flood_mask_to_frame_borders',
    '_resolve_exclude_mask',
    '_GROUND_SCALE_CACHE',
    '_ground_scale_cached',
    '_metric_depth_and_validity',
    '_resolve_depth_band',
    '_parse_band_override',
    '_band_resolution_validity',
    '_resize_normal_field',
    '_AZIMUTH_VIEWS',
    '_ELEVATION_VIEWS',
    '_DISTANCE_VIEWS',
    '_parse_view_prompt',
    '_parse_exact_view',
    '_named_view_orbit_delta',
    '_format_hole_fill_report',
    '_solve_with_relief_mesh',
    '_relief_mesh_from_solve',
    '_solve_image_size',
    '_apply_band_split',
    '_BOUNDED_BAND_NOOP_M',
    '_LAYER_DEBUG_PRIMARY_HEX',
    '_LAYER_DEBUG_PALETTE_HEX',
    '_comfy_registry',
    '_native_sam3_available',
    '_MiniGraphBuilder',
    '_graph_builder',
    '_ATLAS_INPUT_BOUNDARIES',
    '_ATLAS_INPUT_BAND_NAMES',
    '_seg_coverage',
    '_BAND_GEOMETRY_CHOICES',
    '_resolve_band_geometry',
    '_analytic_ground_forward_depth',
]
