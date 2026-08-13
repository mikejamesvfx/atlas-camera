"""Disocclusion inpaint inputs: guide + mask sequences along a camera move.

A camera move reveals pixels no photograph covered. This module renders the
solved static scene from each view of the move (the same pure-numpy z-buffered
rasterizer the AtlasDisocclusionGuide node uses — `core.projection_render`),
paints the uncovered pixels with LTX's chroma-green inpaint sentinel, and
emits the pair the LTX-2.5 inpaint IC-LoRA workflow wants verbatim:

    guide/frame_*.png   original video (scene render, green in the holes)
    mask/frame_*.png    B&W mask video  (white = inpaint, black = keep)

Layering: dynamic/ may import core, never comfy — the texture plumbing the
node does is reimplemented here on the same core calls.

Needs numpy + Pillow (``pip install -e .[vision,image]``).
"""
from __future__ import annotations

import base64
import io

from pathlib import Path
from typing import Any

from atlas_camera.core.camera_spec import CameraSpec
from atlas_camera.core.projection_render import gather_scene_meshes, render_scene

# LTX inpaint pipeline sentinel (LTXVInpaintPreprocess _BG_COLOR_RGB).
LTX_INPAINT_GREEN = (102 / 255.0, 1.0, 0.0)


def _require_deps():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "occlusion-fill requires numpy + Pillow. Install with:\n"
            "    pip install -e .[vision,image]") from exc
    return np, Image


def _decode_data_uri(np, Image, data_uri: str):
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    with Image.open(io.BytesIO(base64.b64decode(payload))) as im:
        return np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0


def build_scene_textures(solve, source_image) -> dict:
    """Texture dict for `render_scene`: the primary plate + every projection
    source's plate (its own matte multiplied into alpha, matching the node)."""
    np, Image = _require_deps()
    textures: dict[str, Any] = {
        "primary": np.asarray(source_image).astype(np.float32) / 255.0
        if np.asarray(source_image).dtype != np.float32
        else np.asarray(source_image)}
    for src in getattr(solve, "projection_sources", None) or []:
        if not src.image_b64:
            continue
        rgba = _decode_data_uri(np, Image, src.image_b64)
        if getattr(src, "mask_b64", None):
            matte = _decode_data_uri(np, Image, src.mask_b64)[..., 0]
            rgba[..., 3] *= matte
        textures[src.name] = rgba
    return textures


def _dilate(np, mask, pad_px: int):
    """Grow ``mask`` by ``pad_px`` 4-connected steps (diffusion wants context
    past the exact tear — same doctrine as the node)."""
    if int(pad_px) <= 0:
        return mask
    dil = mask.copy()
    for _ in range(int(pad_px)):
        grown = dil.copy()
        grown[1:, :] |= dil[:-1, :]
        grown[:-1, :] |= dil[1:, :]
        grown[:, 1:] |= dil[:, :-1]
        grown[:, :-1] |= dil[:, 1:]
        dil = grown
    return dil


def render_disocclusion_sequence(solve, source_image, views, *,
                                 resolution: int = 1024,
                                 hole_dilate_px: int = 8):
    """Per view: (guide HxWx3 uint8, mask HxW uint8) + coverage stats.

    guide = scene render with LTX green in every uncovered pixel; mask white
    where the generator should invent. Holes are dilated so the diffusion
    gets context past the exact tear (same doctrine as the node).
    """
    np, _ = _require_deps()
    meshes = gather_scene_meshes(solve, with_uvs=True)
    if not meshes:
        raise ValueError(
            "solve has no projectable meshes — run the Atlas scene build "
            "(relief mesh / layers) before occlusion-fill")
    textures = build_scene_textures(solve, source_image)
    intr = solve.camera.intrinsics
    scale = resolution / max(intr.image_width, intr.image_height)
    width = max(8, int(round(intr.image_width * scale)))
    height = max(8, int(round(intr.image_height * scale)))
    spec = CameraSpec.from_intrinsics(intr)
    fx, fy = spec.fx * scale, spec.fy * scale
    cx, cy = spec.cx * scale, spec.cy * scale

    green = np.asarray(LTX_INPAINT_GREEN, dtype=np.float32)
    out = []
    for view in views:
        rgb, alpha, _stats = render_scene(meshes, textures, view,
                                          fx, fy, cx, cy, width, height)
        hole = _dilate(np, alpha <= 0.0, hole_dilate_px)
        guide = rgb.astype(np.float32)
        guide[hole] = green
        out.append(((guide * 255).clip(0, 255).astype(np.uint8),
                    (hole.astype(np.uint8) * 255),
                    float(hole.mean())))
    return out


def render_crop_sequence(solve, source_image, views, roi, *,
                         hole_dilate_px: int = 8, meshes=None, textures=None):
    """`render_disocclusion_sequence` for ONE ROI, at NATIVE resolution.

    A camera-matrix crop is a shifted principal point with a smaller raster,
    which is exactly what `crop_intrinsics` produces and exactly what
    `render_scene` already consumes — so this is the same rasterizer, not a
    new path, and the long-edge normalisation of the full-frame renderer is
    simply not used. That is where native resolution comes from: the ROI's own
    pixels are rendered 1:1 with the plate.

    ``meshes``/``textures`` may be passed in so a multi-ROI run gathers the
    scene once. Returns the same ``(guide, mask, hole_frac)`` triples.
    """
    np, _ = _require_deps()
    from atlas_camera.core.camera_crop import crop_intrinsics

    if meshes is None:
        meshes = gather_scene_meshes(solve, with_uvs=True)
    if not meshes:
        raise ValueError(
            "solve has no projectable meshes — run the Atlas scene build "
            "(relief mesh / layers) before occlusion-fill")
    if textures is None:
        textures = build_scene_textures(solve, source_image)

    intr = crop_intrinsics(solve.camera.intrinsics, roi)
    spec = CameraSpec.from_intrinsics(intr)
    green = np.asarray(LTX_INPAINT_GREEN, dtype=np.float32)
    out = []
    for view in views:
        rgb, alpha, stats = render_scene(
            meshes, textures, view, spec.fx, spec.fy, spec.cx, spec.cy,
            roi.width, roi.height)
        hole = _dilate(np, alpha <= 0.0, hole_dilate_px)
        guide = rgb.astype(np.float32)
        guide[hole] = green
        out.append(((guide * 255).clip(0, 255).astype(np.uint8),
                    (hole.astype(np.uint8) * 255),
                    float(hole.mean()), stats.get("depth")))
    return out


def crop_context_depth(frames) -> float:
    """Median scene depth (metres) of the CONTEXT around each frame's holes.

    Resolution demand falls with distance: a facade 60 m away already projects
    a fraction of the plate pixels per metre that one at 8 m does, so a distant
    crop can be generated below 1:1 and resampled back with no real loss of
    recoverable detail. The depth measured is the rendered surface AROUND the
    hole (the hole itself has no depth — that is what makes it a hole), which
    is the right proxy because a disocclusion is filled with the continuation
    of its surroundings.

    Returns 0.0 when nothing was rasterized.
    """
    np, _ = _require_deps()
    samples = []
    for frame in frames:
        depth = frame[3] if len(frame) > 3 else None
        if depth is None:
            continue
        d = np.asarray(depth, dtype=np.float64)
        finite = d[np.isfinite(d)]
        if finite.size:
            samples.append(float(np.median(finite)))
    return float(np.median(samples)) if samples else 0.0


def write_sequences(frames, out_dir: Path) -> tuple[list[Path], list[Path]]:
    """Write guide/ + mask/ PNG sequences; returns (guide_paths, mask_paths)."""
    _, Image = _require_deps()
    guide_dir = Path(out_dir) / "guide"
    mask_dir = Path(out_dir) / "mask"
    guide_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    guide_paths, mask_paths = [], []
    for index, frame in enumerate(frames):
        guide, mask = frame[0], frame[1]
        gp = guide_dir / f"frame_{index:04d}.png"
        mp = mask_dir / f"frame_{index:04d}.png"
        Image.fromarray(guide).save(gp)
        Image.fromarray(mask, mode="L").save(mp)
        guide_paths.append(gp)
        mask_paths.append(mp)
    return guide_paths, mask_paths


def write_exr_sequence(frame_paths, out_dir: Path) -> list[Path]:
    """Optional 32f EXR wrap of PNG frames. Display-referred content in a
    float container — honest metadata, no scene-linear claim (values are the
    sRGB-encoded pixels divided by 255)."""
    np, Image = _require_deps()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "EXR output requires opencv-python ([vision]) with OpenEXR "
            "enabled (set OPENCV_IO_ENABLE_OPENEXR=1)") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for path in frame_paths:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        target = out / (Path(path).stem + ".exr")
        cv2.imwrite(str(target), arr[..., ::-1])  # BGR for OpenCV
        written.append(target)
    return written
