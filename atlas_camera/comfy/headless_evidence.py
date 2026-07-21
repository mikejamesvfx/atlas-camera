"""Backend evidence rendering for terminal/headless Atlas assessments.

The interactive viewport is WebGL-owned: its ``shaded`` output is deliberately
blank until the browser writes a render into ``client_data``.  Agentic queues
cannot click that button, so this module reconstructs the recovered-camera
projection from the *actual* projection plates, per-pixel mattes, and relief
mesh UV coverage carried by the solve.

This is intentionally a canonical-pose evidence render, not a replacement for
the viewport or a DCC renderer.  It proves camera-frame layer coverage,
inpaint seams, projection boundaries, and colour continuity at the recovered
camera.  A separate deterministic orbit-coverage census reports small-move
holes without pretending that the canonical image proves orbit occlusion.

No OpenCV is used.  Pillow rasterizes topology masks and NumPy performs the
straight-RGB composites.  No colour transform is applied; source alpha/mattes
remain data and are only used as coverage for this display-proxy evidence.
"""
from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HeadlessEvidence:
    image: Any
    coverage_mask: Any
    metadata: dict[str, Any]


def _pil_modules():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - ComfyUI installs Pillow
        raise ImportError("Pillow is required for Atlas headless evidence") from exc
    return Image, ImageDraw


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - ComfyUI installs NumPy
        raise ImportError("NumPy is required for Atlas headless evidence") from exc
    return np


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - ComfyUI installs torch
        raise ImportError("PyTorch is required for Atlas headless evidence") from exc
    return torch


def _fit_long_edge(width: int, height: int, max_edge: int) -> tuple[int, int]:
    scale = min(1.0, float(max_edge) / float(max(width, height, 1)))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _tensor_rgba(image: Any, width: int, height: int):
    np = _numpy()
    Image, _ = _pil_modules()
    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("source image must be HxWxRGB(A)")
    rgba = np.ones((arr.shape[0], arr.shape[1], 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr[..., :3], 0.0, 1.0)
    if arr.shape[2] >= 4:
        rgba[..., 3] = np.clip(arr[..., 3], 0.0, 1.0)
    if (arr.shape[1], arr.shape[0]) != (width, height):
        u8 = np.rint(rgba * 255.0).astype(np.uint8)
        rgba = (np.asarray(Image.fromarray(u8, mode="RGBA").resize(
            (width, height), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0)
    return rgba


def _decode_rgba(data_uri: str, width: int, height: int):
    np = _numpy()
    Image, _ = _pil_modules()
    if not data_uri:
        return None
    try:
        raw = base64.b64decode(data_uri.split(",", 1)[-1])
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0
    except (ValueError, OSError, binascii.Error):
        return None


def _decode_mask(data_uri: str | None, width: int, height: int):
    np = _numpy()
    Image, _ = _pil_modules()
    if not data_uri:
        return np.ones((height, width), dtype=np.float32)
    try:
        raw = base64.b64decode(data_uri.split(",", 1)[-1])
        image = Image.open(io.BytesIO(raw)).convert("L")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0
    except (ValueError, OSError, binascii.Error):
        return np.zeros((height, width), dtype=np.float32)


def _mesh_coverage(primitives: list[Any], width: int, height: int,
                   *, supersample: int = 2):
    """Rasterize the exact relief topology in baked source UV space."""
    np = _numpy()
    Image, ImageDraw = _pil_modules()
    scale = max(1, int(supersample))
    mask = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(mask)
    mesh_count = 0
    face_count = 0
    for primitive in primitives or []:
        if getattr(primitive, "primitive_type", None) != "mesh":
            continue
        meta = getattr(primitive, "metadata", None) or {}
        uvs = meta.get("uvs") or []
        faces = meta.get("faces") or []
        if len(uvs) < 6 or len(faces) < 3:
            continue
        uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        valid = ((tri >= 0).all(axis=1) & (tri < len(uv)).all(axis=1))
        tri = tri[valid]
        if not len(tri):
            continue
        points = np.empty_like(uv)
        points[:, 0] = uv[:, 0] * max(width * scale - 1, 1)
        points[:, 1] = (1.0 - uv[:, 1]) * max(height * scale - 1, 1)
        for indices in tri:
            draw.polygon([tuple(points[int(index)]) for index in indices], fill=255)
        mesh_count += 1
        face_count += int(len(tri))
    if mesh_count == 0:
        return None, {"mesh_count": 0, "face_count": 0}
    if scale > 1:
        mask = mask.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(mask, dtype=np.float32) / 255.0, {
        "mesh_count": mesh_count,
        "face_count": face_count,
    }


def reconstruct_camera_view(solve: Any, source_image: Any,
                            *, max_edge: int = 1280) -> HeadlessEvidence | None:
    """Reconstruct the recovered-camera output from solve-carried evidence."""
    if source_image is None:
        return None
    np = _numpy()
    torch = _torch()
    src_arr = source_image.detach().cpu().float().numpy()
    if src_arr.ndim == 4:
        src_arr = src_arr[0]
    if src_arr.ndim != 3 or src_arr.shape[2] < 3:
        return None
    width, height = _fit_long_edge(
        int(src_arr.shape[1]), int(src_arr.shape[0]), int(max_edge))
    canvas_rgb = np.zeros((height, width, 3), dtype=np.float32)
    canvas_alpha = np.zeros((height, width), dtype=np.float32)
    total_faces = 0
    rendered_sources: list[str] = []
    skipped_sources: list[str] = []

    def composite(rgba, coverage, label):
        nonlocal canvas_rgb, canvas_alpha, total_faces
        if rgba is None or coverage is None:
            skipped_sources.append(label)
            return
        alpha = np.clip(coverage * rgba[..., 3], 0.0, 1.0)
        # Straight-RGB over.  Alpha is not colour-transformed or filtered with
        # RGB; multiplication exists only for this display evidence composite.
        canvas_rgb = (rgba[..., :3] * alpha[..., None]
                      + canvas_rgb * (1.0 - alpha[..., None]))
        canvas_alpha = alpha + canvas_alpha * (1.0 - alpha)
        rendered_sources.append(label)

    scene = getattr(solve, "projection_scene", None)
    primary_primitives = list(getattr(scene, "proxy_geometry", None) or [])
    primary_coverage, primary_meta = _mesh_coverage(
        primary_primitives, width, height)
    total_faces += int(primary_meta["face_count"])
    if primary_coverage is not None:
        composite(_tensor_rgba(source_image, width, height),
                  primary_coverage, "primary")

    projection_sources = sorted(
        list(getattr(solve, "projection_sources", None) or []),
        key=lambda source: float(getattr(source, "priority", 0.0)))
    for index, source in enumerate(projection_sources):
        label = str(getattr(source, "name", "") or f"layer_{index + 1}")
        rgba = _decode_rgba(getattr(source, "image_b64", None) or "",
                            width, height)
        coverage, mesh_meta = _mesh_coverage(
            list(getattr(source, "proxy_geometry", None) or []), width, height)
        total_faces += int(mesh_meta["face_count"])
        if coverage is not None:
            coverage *= _decode_mask(
                getattr(source, "mask_b64", None), width, height)
        composite(rgba, coverage, label)

    coverage_fraction = float((canvas_alpha > (1.0 / 255.0)).mean())
    if not rendered_sources or coverage_fraction <= 0.0001:
        return None
    image = torch.from_numpy(np.clip(canvas_rgb, 0.0, 1.0)).unsqueeze(0).float()
    mask = torch.from_numpy(np.clip(canvas_alpha, 0.0, 1.0)).unsqueeze(0).float()
    return HeadlessEvidence(image=image, coverage_mask=mask, metadata={
        "method": "canonical_projection_reconstruction_v1",
        "pose": "recovered_camera",
        "resolution": [width, height],
        "coverage_fraction": round(coverage_fraction, 6),
        "mesh_face_count": total_faces,
        "rendered_sources": rendered_sources,
        "skipped_sources": skipped_sources,
        "ocio_alpha_contract": (
            "display-proxy reconstruction only; no RGB colour transform; source alpha "
            "and mattes remain data and are used only as straight coverage"),
        "limitations": [
            "canonical recovered-camera pose only",
            "does not emulate WebGL lighting, tone mapping, facing-ratio discard, or the "
            "interactive occlusion toggle",
            "small-orbit tearing is evaluated by deterministic orbit coverage, not inferred "
            "from this canonical image",
        ],
    })


def compare_source_structure(output_image: Any, source_image: Any,
                             *, eval_edge: int = 256) -> dict[str, Any]:
    """Measure canonical structural/content drift without OpenCV or OCIO.

    Correlations are intentionally exposure-tolerant; MAE and changed-pixel
    fraction catch wholesale content replacement. This is a release-QA signal,
    not a photometric identity test: a deliberately large clean-plate rewrite
    can trigger it and should then be acknowledged by the run intent.
    """

    np = _numpy()
    Image, _ = _pil_modules()

    def rgb(image):
        arr = image.detach().cpu().float().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("comparison images must be HxWxRGB(A)")
        height, width = arr.shape[:2]
        out_width, out_height = _fit_long_edge(width, height, eval_edge)
        u8 = np.rint(np.clip(arr[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.asarray(Image.fromarray(u8, mode="RGB").resize(
            (out_width, out_height), Image.Resampling.LANCZOS),
            dtype=np.float32) / 255.0

    output = rgb(output_image)
    source = rgb(source_image)
    if output.shape != source.shape:
        source_u8 = np.rint(source * 255.0).astype(np.uint8)
        source = np.asarray(Image.fromarray(source_u8, mode="RGB").resize(
            (output.shape[1], output.shape[0]), Image.Resampling.LANCZOS),
            dtype=np.float32) / 255.0
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    output_luma = output @ weights
    source_luma = source @ weights

    def correlation(a, b):
        aa = a.reshape(-1).astype(np.float64)
        bb = b.reshape(-1).astype(np.float64)
        aa -= aa.mean()
        bb -= bb.mean()
        denominator = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
        if denominator < 1e-12:
            return 1.0 if np.allclose(a, b, atol=1.0 / 255.0) else 0.0
        return float(np.dot(aa, bb) / denominator)

    def edges(luma):
        dx = np.diff(luma, axis=1, append=luma[:, -1:])
        dy = np.diff(luma, axis=0, append=luma[-1:, :])
        return np.hypot(dx, dy)

    luma_correlation = correlation(output_luma, source_luma)
    edge_correlation = correlation(edges(output_luma), edges(source_luma))
    difference = np.abs(output - source)
    mean_absolute_error = float(difference.mean())
    changed_fraction = float((difference.mean(axis=2) > 0.15).mean())
    severe = ((luma_correlation < 0.70 and edge_correlation < 0.40)
              or (mean_absolute_error > 0.12 and changed_fraction > 0.25))
    warning = ((luma_correlation < 0.80 and edge_correlation < 0.60)
               or (mean_absolute_error > 0.08 and changed_fraction > 0.15))
    return {
        "method": "source_structure_drift_v1_pillow_numpy",
        "eval_resolution": [int(output.shape[1]), int(output.shape[0])],
        "status": "severe" if severe else "warn" if warning else "within_tolerance",
        "luma_correlation": round(luma_correlation, 6),
        "edge_correlation": round(edge_correlation, 6),
        "rgb_mean_absolute_error": round(mean_absolute_error, 6),
        "changed_fraction_gt_0_15": round(changed_fraction, 6),
        "limitations": (
            "exposure-tolerant structural release-QA heuristic; deliberately broad "
            "artist-approved clean-plate replacement may require an explicit override"),
    }


def _mesh_arrays(solve: Any):
    np = _numpy()
    out = []

    def add(primitives, prefix):
        for primitive in primitives or []:
            if getattr(primitive, "primitive_type", None) != "mesh":
                continue
            meta = getattr(primitive, "metadata", None) or {}
            vertices = meta.get("vertices") or []
            faces = meta.get("faces") or []
            if len(vertices) < 9 or len(faces) < 3:
                continue
            out.append((
                f"{prefix}{getattr(primitive, 'name', 'mesh')}",
                np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
                np.asarray(faces, dtype=np.int64).reshape(-1, 3),
                meta,
            ))

    scene = getattr(solve, "projection_scene", None)
    add(getattr(scene, "proxy_geometry", None) or [], "")
    for source in getattr(solve, "projection_sources", None) or []:
        add(getattr(source, "proxy_geometry", None) or [],
            f"{getattr(source, 'name', 'layer')}/")
    return out


def _project(vertices, view, fx, fy, cx, cy):
    np = _numpy()
    hom = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    camera = hom @ np.asarray(view, dtype=np.float64).T
    forward = -camera[:, 2]
    safe = np.where(np.abs(forward) < 1e-9, 1e-9, forward)
    pixels = np.stack([
        cx + fx * camera[:, 0] / safe,
        cy - fy * camera[:, 1] / safe,
    ], axis=1)
    return pixels, forward


def orbit_coverage_summary(solve: Any, *, res: int = 256,
                           azimuth_steps: tuple[float, ...] = (3.0, 6.0),
                           stretch_ratio: float = 12.0) -> dict[str, Any] | None:
    """No-OpenCV geometry coverage census at canonical and small orbit poses."""
    np = _numpy()
    Image, ImageDraw = _pil_modules()
    from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera

    meshes = _mesh_arrays(solve)
    if not meshes:
        return None
    intrinsics = solve.camera.intrinsics
    extrinsics = solve.camera.extrinsics
    image_width = int(intrinsics.image_width or 1)
    image_height = int(intrinsics.image_height or 1)
    scale = float(res) / float(max(image_width, image_height, 1))
    width = max(8, int(round(image_width * scale)))
    height = max(8, int(round(image_height * scale)))
    fx = float(intrinsics.fx_px or 0.0) * scale
    fy = float(intrinsics.fy_px or intrinsics.fx_px or 0.0) * scale
    cx = float(intrinsics.cx_px if intrinsics.cx_px is not None
               else image_width / 2.0) * scale
    cy = float(intrinsics.cy_px if intrinsics.cy_px is not None
               else image_height / 2.0) * scale
    if fx <= 0 or fy <= 0:
        return None

    prepared = []
    for name, vertices, faces, meta in meshes:
        valid = ((faces >= 0).all(axis=1)
                 & (faces < len(vertices)).all(axis=1))
        faces = faces[valid]
        if not len(faces):
            continue
        a, b, c = (vertices[faces[:, index]] for index in range(3))
        lengths = np.stack([
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ], axis=1)
        ratios = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-9)
        prepared.append((name, vertices, faces, ratios > stretch_ratio, meta))
    if not prepared:
        return None

    pivot = ground_lookat_pivot(extrinsics)
    poses = [(0.0, 0.0)]
    for step in azimuth_steps:
        poses.extend([(float(step), 0.0), (-float(step), 0.0)])
    rows = []
    for delta_azimuth, delta_elevation in poses:
        pose = (extrinsics if delta_azimuth == delta_elevation == 0.0 else
                orbit_camera(
                    extrinsics, pivot, d_azimuth_deg=delta_azimuth,
                    d_elevation_deg=delta_elevation))
        cover_image = Image.new("L", (width, height), 0)
        stretch_image = Image.new("L", (width, height), 0)
        cover_draw = ImageDraw.Draw(cover_image)
        stretch_draw = ImageDraw.Draw(stretch_image)
        for _, vertices, faces, stretched, _ in prepared:
            pixels, forward = _project(
                vertices, pose.camera_view_matrix, fx, fy, cx, cy)
            visible = forward[faces].min(axis=1) > 1e-6
            for face, is_stretched in zip(faces[visible], stretched[visible]):
                polygon = [tuple(pixels[int(index)]) for index in face]
                cover_draw.polygon(polygon, fill=1)
                if bool(is_stretched):
                    stretch_draw.polygon(polygon, fill=1)
        cover = np.asarray(cover_image, dtype=np.uint8)
        stretched = np.asarray(stretch_image, dtype=np.uint8)
        frame = float(width * height)
        rows.append({
            "d_azimuth_deg": delta_azimuth,
            "d_elevation_deg": delta_elevation,
            "hole_pct": round(100.0 * (1.0 - float(cover.sum()) / frame), 3),
            "stretch_pct": round(
                100.0 * float((stretched & cover).sum()) / frame, 3),
        })
    return {
        "method": "geometry_coverage_raster_v1_pillow",
        "eval_resolution": [width, height],
        "stretch_ratio_threshold": float(stretch_ratio),
        "pivot": [round(float(value), 4) for value in pivot],
        "poses": rows,
        "limitations": (
            "geometry coverage lower bound; per-pixel mattes, facing discard, lighting, "
            "and z-order are not applied; browser/DCC render remains the final oracle"),
    }


__all__ = ["HeadlessEvidence", "reconstruct_camera_view",
           "compare_source_structure", "orbit_coverage_summary"]
