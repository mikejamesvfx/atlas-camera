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


#: How far "at infinity" is, in metres. With a move baseline of order 10 m the
#: residual parallax at 1e6 m is ~1e-5 rad — far under a survey pixel — so the
#: warp is the pure-rotation (infinite) homography for every practical purpose,
#: while still going through the same convention-correct project/back-project
#: helpers as the rest of the codebase instead of a hand-rolled K R K^-1.
_INFINITY_M = 1.0e6


def plate_hole_survey(solve, source_image, *, resolution=1024):
    """The SOLVED pose's own hole mask + the intrinsics it was rendered with.

    The reference every move-revealed test is made against. Rendered once and
    passed around, because both the ROI SELECTION (full-frame survey raster)
    and the ROI's own crop (native raster, shifted principal point) need it and
    they must agree.
    """
    np, _ = _require_deps()
    view = solve.camera.extrinsics.camera_view_matrix
    frames = render_disocclusion_sequence(
        solve, source_image, [view], resolution=int(resolution),
        hole_dilate_px=0)
    mask = frames[0][1] > 127
    intr = solve.camera.intrinsics
    scale = float(resolution) / max(intr.image_width, intr.image_height)
    spec = CameraSpec.from_intrinsics(intr)
    return {"mask": mask, "view": view,
            "fx": spec.fx * scale, "fy": spec.fy * scale,
            "cx": spec.cx * scale, "cy": spec.cy * scale,
            "height": mask.shape[0], "width": mask.shape[1]}


def not_disocclusion_mask(plate, *, view, fx, fy, cx, cy, width, height):
    """Which pixels of a camera are NOT move-revealed disocclusion.

    Two classes, both decided by looking BACK at the plate camera along each
    survey pixel's ray taken to infinity:

    - **already a hole in the plate.** Sky, and anything else the scene mesh
      never covered from the solved pose, was not occluded by anything — there
      is nothing to dis-occlude. The comparison has to be made through the
      cameras, not pixel-for-pixel: two views that differ by a rotation do not
      see the sky at the same pixels, and the straight subtraction this
      replaced left a crescent of skyline ranking as a genuine hole (found
      live 2026-08-15 on an arc). Sky is at infinity, which is exactly the
      case a rotation-only mapping is EXACT for; translation — the whole
      source of real disocclusion — cancels at ``_INFINITY_M``.
    - **outside the plate's frame entirely.** A move can swing the camera past
      the edge of the photograph. Those pixels have no plate counterpart at
      all, so no subtraction can classify them — and they are not disocclusion
      either: nothing was occluding them, the camera simply never looked
      there. That is OUTPAINTING, a different job with different nodes (band /
      outpaint layers), and it must not consume the fill budget. This class is
      what actually dominated the synthetic fixture, and it grows with the arc
      angle.

    Real disocclusion survives both: it is inside the plate frame, and it is
    not a hole there (something opaque was standing in front of it).

    ``plate`` is a ``plate_hole_survey`` dict; the target camera is given
    explicitly (view + intrinsics + raster) so the SAME test serves the
    full-frame survey that RANKS ROIs and the ROI's own native crop, whose
    principal point is shifted. Those two must agree: ranking a cluster with
    sky excluded and then handing the generator a crop mask that includes the
    sky inside the ROI's rect puts the sky back — measured live 2026-08-15
    (the arc-left patch came back with a sky-textured sheet above the
    roofline).
    """
    r = reproject_at_infinity(plate, view=view, fx=fx, fy=fy, cx=cx, cy=cy,
                              width=width, height=height)
    np, _ = _require_deps()
    plate_hole = np.asarray(plate["mask"], dtype=bool)
    drop = ~r["inside"]
    ui = np.where(r["inside"], r["u"], 0).astype(np.int64)
    vi = np.where(r["inside"], r["v"], 0).astype(np.int64)
    return drop | (r["inside"] & plate_hole[vi, ui])


def reproject_at_infinity(plate, *, view, fx, fy, cx, cy, width, height):
    """Where each pixel of a camera lands in the PLATE, assuming infinity.

    Returns ``{"inside", "u", "v"}`` — boolean plus float plate-raster
    coordinates (the raster ``plate`` was surveyed at). Split out of
    ``not_disocclusion_mask`` because the coordinates are useful in their own
    right: sky is at infinity, so sampling the plate through this mapping IS
    the correct sky for the moved camera, which is how a dropped (not
    disocclusion) pixel gets real content instead of a sentinel.
    """
    np, _ = _require_deps()
    from atlas_camera.core.depth_geometry import back_project_normals
    from atlas_camera.core.patch_registration import _project

    p_h, p_w = np.asarray(plate["mask"]).shape[:2]
    height, width = int(height), int(width)
    depth = np.full((height, width), _INFINITY_M, dtype=np.float64)
    bp = back_project_normals(depth, view_matrix=view,
                              fx=fx, fy=fy, cx=cx, cy=cy)
    u, v, fwd = _project(bp.pts_world.reshape(-1, 3),
                         {"view_matrix": plate["view"], "fx": plate["fx"],
                          "fy": plate["fy"], "cx": plate["cx"],
                          "cy": plate["cy"]})
    inside = (np.isfinite(u) & np.isfinite(v) & (fwd > 1e-6)
              & (u >= 0) & (u < p_w) & (v >= 0) & (v < p_h))
    return {"inside": inside.reshape(height, width),
            "u": np.nan_to_num(u).reshape(height, width),
            "v": np.nan_to_num(v).reshape(height, width),
            "plate_width": p_w, "plate_height": p_h}


def survey_hole_rois(solve, source_image, views, *, survey_resolution=1024,
                     pad_frac=0.10, min_area_px=64, snap=64,
                     exclude_mask=None, move_revealed_only=False):
    """THE hole-clustering recipe, shared so CLI and nodes cannot drift.

    Survey the views' disocclusion holes at a cheap raster, cluster them
    UNDILATED (survey-res dilation bridges separate tears — measured on
    DSC_2289: 4 px took top-4 ROI coverage from 23% to 124% of the plate),
    lift the ROIs to plate resolution, clamp, snap, and sort by area
    descending. Padding and the snap only ever GROW a crop, so the lift is
    conservative.

    Two sky failsafes, both subtractive (they can only ever REMOVE candidate
    pixels, never invent them):

    - ``exclude_mask``: an explicit "never target this" override — a region
      already carried by something other than geometry (a SkyDome, a matte).
      Resized nearest to the survey raster and subtracted from EVERY survey
      mask UNWARPED, because it is an artist declaration rather than a
      geometric test and the caller decides which frame it is authored in.
      Same doctrine as AtlasDisocclusionGuide's input of the same name.
    - ``move_revealed_only``: keep only what the MOVE revealed, decided per
      view by ``_not_disocclusion_in_view`` — through the cameras, not
      pixel-for-pixel. It removes both the plate's own holes (sky and any
      never-derived geometry: nothing was occluding them) and everything that
      falls outside the plate's frame altogether (the camera never looked
      there — that is outpainting, not disocclusion). The G5 field run
      measured what happens without any of this: the auto-ROI ranked a sky
      cluster first and aimed a generator at it. The straight pixel
      subtraction that first shipped for it under-delivered — see that
      helper's docstring for the two failure classes it left behind.

    Returns ``(rois, roi_set, survey_masks, peak_hole_frac)`` where ``rois``
    is the area-sorted plate-resolution list and ``roi_set`` carries the
    survey-resolution components + dropped entries for reporting. Policy —
    artist-wins, oversize decline/tiled, the max_rois budget — deliberately
    stays with the caller (`dynamic/cli.py`); AtlasCropROI's auto mode simply
    indexes this list by rank.
    """
    np, _ = _require_deps()
    from atlas_camera.core.camera_crop import RegionROI, hole_rois

    intr = solve.camera.intrinsics
    plate_w, plate_h = int(intr.image_width), int(intr.image_height)
    survey = render_disocclusion_sequence(
        solve, source_image, views, resolution=int(survey_resolution),
        hole_dilate_px=0)
    survey_masks = [mask for _guide, mask, _cov in survey]
    survey_h, survey_w = survey_masks[0].shape[:2]
    peak = max(cov for _g, _m, cov in survey)

    # The artist's exclude_mask is subtracted UNWARPED, identically from every
    # survey mask (see the docstring): it is an explicit "never target this"
    # override, not a geometric test.
    flat_drop = np.zeros((survey_h, survey_w), dtype=bool)
    if exclude_mask is not None:
        m = np.asarray(exclude_mask)
        if m.shape != (survey_h, survey_w):
            # Nearest-neighbour index remap — a mask must not be blurred into
            # fractional values that then need a second threshold.
            yi = (np.arange(survey_h) * (m.shape[0] / survey_h)).astype(int)
            xi = (np.arange(survey_w) * (m.shape[1] / survey_w)).astype(int)
            m = m[yi.clip(0, m.shape[0] - 1)][:, xi.clip(0, m.shape[1] - 1)]
        flat_drop |= m > (127 if m.dtype.kind in "ui" else 0.5)

    # The move-revealed test is per view and goes through the cameras.
    plate = (plate_hole_survey(solve, source_image,
                               resolution=int(survey_resolution))
             if move_revealed_only else None)

    if plate is not None or flat_drop.any():
        intr = solve.camera.intrinsics
        scale = float(survey_resolution) / max(intr.image_width,
                                               intr.image_height)
        spec = CameraSpec.from_intrinsics(intr)
        kept = []
        for view, mask in zip(views, survey_masks):
            drop = flat_drop
            if plate is not None:
                drop = drop | not_disocclusion_mask(
                    plate, view=view, fx=spec.fx * scale, fy=spec.fy * scale,
                    cx=spec.cx * scale, cy=spec.cy * scale,
                    width=survey_w, height=survey_h)
            kept.append(np.where(drop, 0, mask))
        survey_masks = kept

    roi_set = hole_rois(survey_masks, pad_frac=float(pad_frac),
                        min_area_px=int(min_area_px), snap=1, max_rois=0)
    sx = plate_w / float(survey_w)
    sy = plate_h / float(survey_h)
    rois = []
    for roi in roi_set.rois:
        scaled = RegionROI(x=int(roi.x * sx), y=int(roi.y * sy),
                           width=max(1, int(round(roi.width * sx))),
                           height=max(1, int(round(roi.height * sy))))
        rois.append(scaled.clamped(plate_w, plate_h).snapped(
            int(snap), image_width=plate_w, image_height=plate_h))
    rois.sort(key=lambda r: r.area_px, reverse=True)
    return rois, roi_set, survey_masks, float(peak)
