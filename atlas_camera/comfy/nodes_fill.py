"""In-graph two-pass fill nodes: the gate and the composite corrections.

The two-pass engine (WAN structure -> gate -> SDXL texture) shipped first as a
CLI (`hole-crop-fill --engine two-pass`), where the inter-pass gate and the
membrane/colour composite live in Python between two hidden template graphs.
These nodes put both into the WORKFLOW so the whole recipe distributes as one
graph with no CLI:

  AtlasInterpassGate      scores a structure fill BEFORE it is re-textured and
                          performs its own fallback: on failure the GUIDE
                          passes through, so a downstream texture pass
                          re-touches nothing and the composite is a no-op.
                          A texture model must never launder a broken
                          structure into confident fiction.
  AtlasMembraneComposite  plate-referenced colour pair + the offset membrane +
                          the masked composite, in one node. The membrane is
                          what took the rim gradient from 2.2x the plate's own
                          statistics to ~1.0 (measured 2026-08-14) after every
                          generation-side seam fix failed.

Tensor conventions: IMAGE (B,H,W,C) float 0-1, MASK (B,H,W) or (H,W); both
nodes operate on frame 0 (the single-frame fill doctrine) and say so in the
report when handed more.
"""
from __future__ import annotations

from atlas_camera.comfy.node_helpers import _require_numpy, _require_torch


def _img_to_uint8(np, torch, image):
    """(B,H,W,C) 0-1 tensor -> (H,W,3) uint8 of frame 0, plus frame count."""
    arr = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    frames = int(arr.shape[0]) if arr.ndim == 4 else 1
    if arr.ndim == 4:
        arr = arr[0]
    rgb = np.clip(arr[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return rgb, frames


def _mask_to_bool(np, torch, mask, height, width):
    arr = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    hole = arr > 0.5
    if hole.shape != (height, width):
        from PIL import Image as PILImage
        hole = np.asarray(
            PILImage.fromarray((hole * 255).astype(np.uint8)).resize(
                (width, height), PILImage.NEAREST)) > 127
    return hole


def _uint8_to_img(np, torch, rgb):
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _fit(np, rgb, height, width):
    """Lanczos-resize a fill to the reference raster when they differ (the
    WAN branch generates at 720p-class and composites at plate res)."""
    if rgb.shape[:2] == (height, width):
        return rgb
    from PIL import Image as PILImage
    return np.asarray(PILImage.fromarray(rgb).resize((width, height),
                                                     PILImage.LANCZOS))


class AtlasPathFrameIndex:
    """🔢 Frame indices for a camera path — computed, never hand-typed.

    The in-graph two-pass fill selects "the last N frames of the move" and
    "the move's final frame" with ImageFromBatch nodes. Typing those indices
    is a hand-sync against the path's frame count, and it failed on first
    contact: a 30-frame arc rendered with the indices still set for a 5-frame
    default, so the "repair frame" was frame 4 of the move's beginning.

    Uses the SAME sampler as AtlasDisocclusionGuide (`sample_camera_path`),
    so the count always agrees with the guide batch this node indexes into.
    """

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame_count", "last_index", "window_start", "report")
    FUNCTION = "index"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_path": ("ATLAS_CAMERA_PATH", {
                    "tooltip": "The move (the viewport's camera_path "
                               "output)."}),
            },
            "optional": {
                "window": ("INT", {
                    "default": 5, "min": 1, "max": 97,
                    "tooltip": "Trailing window length (the WAN pass wants "
                               "4k+1: 5, 9, ...). window_start = "
                               "frame_count - window, clamped to 0."}),
            },
        }

    def index(self, camera_path, window=5):
        from atlas_camera.core.camera_path import sample_camera_path

        sampled = sample_camera_path(camera_path) if camera_path is not None \
            else []
        count = len(sampled)
        if count == 0:
            return (1, 0, 0,
                    "AtlasPathFrameIndex: no path (or an empty one) — the "
                    "guide renders 1 frame at the solved pose; indices 0.")
        last = count - 1
        start = max(0, count - int(window))
        report = (f"AtlasPathFrameIndex: {count} frames — window "
                  f"[{start}..{last}] (len {min(int(window), count)}), "
                  f"final frame {last}.")
        if int(window) % 4 != 1:
            report += (f" NOTE: window {int(window)} is not 4k+1 — WAN VACE "
                       f"wants 5, 9, 13, ...")
        return (count, last, start, report)


class AtlasInterpassGate:
    """🚦 Score a structure fill; pass the GUIDE through on failure.

    Wraps `dynamic.two_pass.interpass_gate` (G2 vs the edge-extend smear,
    global phase-correlation shift, sentinel bleed — each check encodes a
    failure the two-pass pipeline actually produced). The fallback is built
    into the output: `fill` is the candidate when the gate passes and the
    untouched guide when it fails, so the downstream texture pass and
    composite degrade to a no-op instead of polishing garbage.
    """

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("fill", "ok", "report")
    FUNCTION = "gate"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill": ("IMAGE", {"tooltip": "Pass-1 structure fill."}),
                "guide": ("IMAGE", {"tooltip": "The plate render the fill "
                                    "must join (holes may carry the "
                                    "sentinel)."}),
                "mask": ("MASK", {"tooltip": "The generation hole."}),
            },
            "optional": {
                "max_shift_px": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 32.0, "step": 0.5,
                    "tooltip": "Maximum global misregistration (phase "
                               "correlation over the unmasked area)."}),
            },
        }

    def gate(self, fill, guide, mask, max_shift_px=2.0):
        import atlas_camera.dynamic.two_pass as tp

        np = _require_numpy()
        torch = _require_torch()
        guide_rgb, gframes = _img_to_uint8(np, torch, guide)
        fill_rgb, fframes = _img_to_uint8(np, torch, fill)
        height, width = guide_rgb.shape[:2]
        fill_rgb = _fit(np, fill_rgb, height, width)
        hole = _mask_to_bool(np, torch, mask, height, width)
        if not bool(hole.any()):
            return (_uint8_to_img(np, torch, guide_rgb), False,
                    "AtlasInterpassGate: empty hole — nothing was generated, "
                    "guide passed through.")

        prior = tp.MAX_INTERPASS_SHIFT_PX
        tp.MAX_INTERPASS_SHIFT_PX = float(max_shift_px)
        try:
            verdict = tp.interpass_gate(fill_rgb, guide_rgb, hole)
        finally:
            tp.MAX_INTERPASS_SHIFT_PX = prior
        lines = [f"AtlasInterpassGate: {'PASS' if verdict.ok else 'FAIL'} — "
                 f"g2 {verdict.g2:.4f}, shift {verdict.shift_px:.2f}px, "
                 f"sentinel bleed {verdict.sentinel_bleed_frac:.3%}"]
        lines += [f"  {r}" for r in verdict.reasons]
        if gframes > 1 or fframes > 1:
            lines.append("  note: batched input — frame 0 gated (single-frame "
                         "fill doctrine)")
        out = fill_rgb if verdict.ok else guide_rgb
        if not verdict.ok:
            lines.append("  guide passed through — the texture pass will "
                         "re-touch nothing")
        return (_uint8_to_img(np, torch, out), bool(verdict.ok),
                "\n".join(lines))


class AtlasMembraneComposite:
    """🩹 Colour pair + offset membrane + masked composite, in one node.

    The full correction stack the CLI engine applies around every generated
    fill: plate-referenced gain/offset colour match, chroma-only cast
    neutralisation from a ring of REAL pixels, the harmonic membrane that
    erases the rim seam (2.2x -> ~1.0 of the plate's own gradient statistics,
    measured), then the composite of hole pixels into the reference.
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "composite"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill": ("IMAGE", {"tooltip": "Generated fill (any raster — "
                                   "resized to the reference if needed)."}),
                "reference": ("IMAGE", {"tooltip": "The plate render to "
                                        "composite into. Holes may carry the "
                                        "sentinel; the corrections only ever "
                                        "read real pixels."}),
                "mask": ("MASK", {"tooltip": "Hole: where fill pixels land."}),
            },
            "optional": {
                "cast_band_px": ("INT", {
                    "default": 48, "min": 4, "max": 256,
                    "tooltip": "Ring width outside the hole used to measure "
                               "the generator's colour cast against real "
                               "pixels."}),
                "colour_correct": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply the plate-referenced colour pair before "
                               "the membrane."}),
            },
        }

    def composite(self, fill, reference, mask, cast_band_px=48,
                  colour_correct=True):
        from atlas_camera.core.camera_crop import (
            match_reference_colour,
            membrane_blend,
            neutralize_fill_cast,
        )

        np = _require_numpy()
        torch = _require_torch()
        ref_rgb, _rframes = _img_to_uint8(np, torch, reference)
        fill_rgb, _fframes = _img_to_uint8(np, torch, fill)
        height, width = ref_rgb.shape[:2]
        fill_rgb = _fit(np, fill_rgb, height, width)
        hole = _mask_to_bool(np, torch, mask, height, width)
        if not bool(hole.any()):
            return (_uint8_to_img(np, torch, ref_rgb),
                    "AtlasMembraneComposite: empty hole — reference returned "
                    "unchanged.")

        if colour_correct:
            fill_rgb = neutralize_fill_cast(
                match_reference_colour(fill_rgb, ref_rgb, hole), hole,
                reference=ref_rgb, band_px=int(cast_band_px))
        fill_rgb = membrane_blend(fill_rgb, ref_rgb, hole)
        out = ref_rgb.copy()
        out[hole] = fill_rgb[hole]
        report = (f"AtlasMembraneComposite: {int(hole.sum()):,} px composited "
                  f"({float(hole.mean()):.1%} of frame), colour pair "
                  f"{'on' if colour_correct else 'OFF'}, membrane applied.")
        return (_uint8_to_img(np, torch, out), report)


class AtlasCropROI:
    """✂️ One artist ▦ Fill ROI as a generation-ready crop, with its camera.

    The CLI's crop economy, in-graph: a camera-matrix crop is a shifted
    principal point with a smaller raster (`crop_intrinsics`), so the ROI
    renders 1:1 with the plate through the same rasterizer — no whole-frame
    generation, no wasted pixels. The variable-ROI-count problem that kept
    this in a CLI disappears under the artist budget: three slots, three
    static branches, and an unused slot degrades to a no-op end to end (the
    gate passes its guide through, AtlasCompositeCrop returns its input).

    Outputs are GENERATION-READY: guide + mask at an aspect-preserving,
    /16-snapped raster no longer than ``max_gen_long_edge``, plus that raster
    as INTs to wire straight into WanVaceToVideo's width/height (whose bare
    int widgets would otherwise hard-code one raster for every crop). The
    ATLAS_CROP handle carries the native rect for the exact paste-back.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "ATLAS_CROP", "STRING")
    RETURN_NAMES = ("guide", "mask", "gen_width", "gen_height", "crop",
                    "report")
    FUNCTION = "crop"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE", {}),
                "source_image": ("IMAGE", {}),
            },
            "optional": {
                "camera_path": ("ATLAS_CAMERA_PATH", {
                    "tooltip": "The move; the crop renders its FINAL frame "
                               "(whose holes are a superset of every earlier "
                               "frame's). No path = the solved pose."}),
                "roi_slot": ("INT", {
                    "default": 1, "min": 1, "max": 3,
                    "tooltip": "Which artist ▦ Fill ROI (budget 3). An "
                               "unused slot emits an empty crop that every "
                               "downstream node treats as a no-op."}),
                "pad_frac": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01}),
                "snap": ("INT", {
                    "default": 64, "min": 8, "max": 128,
                    "tooltip": "ROI grid snap (/64 covers every adapter)."}),
                "hole_dilate_px": ("INT", {
                    "default": 4, "min": 0, "max": 64}),
                "max_gen_long_edge": ("INT", {
                    "default": 1280, "min": 256, "max": 4096,
                    "tooltip": "Generation raster cap; the crop is scaled "
                               "down (aspect kept, /16) when its native long "
                               "edge exceeds this."}),
                # APPENDED 2026-08-14 (positional widget contract): auto mode.
                "roi_source": (["artist", "auto_largest"], {
                    "default": "artist",
                    "tooltip": "artist: the viewport's drawn Fill ROIs (the "
                               "default; artist selection wins). auto_largest: "
                               "survey the end frame's disocclusion holes and "
                               "take the roi_slot-th LARGEST cluster — the "
                               "holes the viewport wand cannot fix. Same "
                               "clustering recipe as the CLI "
                               "(survey_hole_rois), so the two cannot drift."}),
                "min_area_px": ("INT", {
                    "default": 1024, "min": 1, "max": 1 << 20,
                    "tooltip": "auto_largest only: clusters smaller than this "
                               "(survey-resolution px) are ignored — the wand "
                               "or a band layer handles those."}),
            },
        }

    def crop(self, solve, source_image, camera_path=None, roi_slot=1,
             pad_frac=0.10, snap=64, hole_dilate_px=4, max_gen_long_edge=1280,
             roi_source="artist", min_area_px=1024):
        from PIL import Image as PILImage

        from atlas_camera.core.camera_crop import rois_from_world_regions
        from atlas_camera.core.camera_spec import CameraSpec
        from atlas_camera.dynamic.occlusion_fill import render_crop_sequence

        np = _require_numpy()
        torch = _require_torch()

        def empty(reason):
            g = torch.zeros(1, 64, 64, 3)
            m = torch.zeros(1, 64, 64)
            return (g, m, 64, 64, {"empty": True},
                    f"AtlasCropROI slot {roi_slot}: {reason}")

        intr = solve.camera.intrinsics
        spec = CameraSpec.from_intrinsics(intr)
        width = int(intr.image_width)
        height = int(intr.image_height)
        view = None
        if camera_path is not None:
            from atlas_camera.core.camera_path import sample_camera_path
            sampled = sample_camera_path(camera_path)
            if sampled:
                view = sampled[-1].camera_view_matrix
        if view is None:
            view = solve.camera.extrinsics.camera_view_matrix

        src = source_image.detach().cpu().numpy()[0]
        src = np.clip(src * 255.0, 0, 255).astype(np.uint8)
        if src.shape[:2] != (height, width):
            src = np.asarray(PILImage.fromarray(src).resize(
                (width, height), PILImage.LANCZOS))

        if roi_source == "auto_largest":
            from atlas_camera.dynamic.occlusion_fill import survey_hole_rois
            auto_rois, roi_set, _masks, peak = survey_hole_rois(
                solve, src, [view], survey_resolution=1024,
                pad_frac=float(pad_frac), min_area_px=int(min_area_px),
                snap=int(snap))
            if roi_slot > len(auto_rois):
                return empty(f"auto_largest found only {len(auto_rois)} "
                             f"cluster(s) >= {min_area_px}px — no-op branch")
            roi = auto_rois[roi_slot - 1]
            source_note = (f"auto rank {roi_slot}/{len(auto_rois)} "
                           f"(peak hole {peak:.1%})")
        else:
            scene = getattr(solve, "projection_scene", None)
            meta = getattr(scene, "debug_metadata", None) or {}
            regions = list((meta.get("fill_rois") or {}).get("regions") or [])
            if roi_slot > len(regions):
                return empty(f"no artist region (only {len(regions)} drawn) "
                             f"— no-op branch")
            picked = rois_from_world_regions(
                [regions[roi_slot - 1]], view, fx=spec.fx, fy=spec.fy,
                cx=spec.cx, cy=spec.cy, image_width=width,
                image_height=height, pad_frac=float(pad_frac),
                snap=int(snap))
            if not picked.rois:
                reason = (picked.dropped[0]["reason"] if picked.dropped
                          else "region did not project into this view")
                return empty(reason)
            roi = picked.rois[0]
            source_note = f"artist slot {roi_slot}"
        guide, mask, cov, _depth = render_crop_sequence(
            solve, src, [view], roi, hole_dilate_px=int(hole_dilate_px))[0]

        scale = min(1.0, float(max_gen_long_edge)
                    / max(roi.width, roi.height))
        gen_w = max(16, int(round(roi.width * scale / 16)) * 16)
        gen_h = max(16, int(round(roi.height * scale / 16)) * 16)
        if (gen_w, gen_h) != (roi.width, roi.height):
            guide = np.asarray(PILImage.fromarray(guide).resize(
                (gen_w, gen_h), PILImage.LANCZOS))
            mask = np.asarray(PILImage.fromarray(mask, mode="L").resize(
                (gen_w, gen_h), PILImage.NEAREST))

        handle = {"empty": False, "x": roi.x, "y": roi.y,
                  "width": roi.width, "height": roi.height,
                  "gen_w": gen_w, "gen_h": gen_h}
        report = (f"AtlasCropROI ({source_note}): {roi.width}x{roi.height} "
                  f"at ({roi.x},{roi.y}), hole {cov:.1%}, generation raster "
                  f"{gen_w}x{gen_h}"
                  + (" (1:1 native)" if scale >= 1.0
                     else f" (capped from native, 1:{roi.width / gen_w:.2f})"))
        g = torch.from_numpy(guide.astype(np.float32) / 255.0).unsqueeze(0)
        m = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return (g, m, gen_w, gen_h, handle, report)


class AtlasCompositeCrop:
    """📌 Paste a corrected crop fill back into the full frame, exactly.

    The inverse of AtlasCropROI: the fill (already gated, colour-corrected
    and membrane-blended at the crop raster, so every non-hole pixel equals
    the frame's own pixels) is resized to the crop's NATIVE rect and pasted
    at its coordinates. Outside the crop the frame is untouched; an empty
    crop handle returns the frame unchanged, which is what lets unused
    artist slots ride the same graph as no-ops. Chain one per slot:
    frame -> paste(slot 1) -> paste(slot 2) -> paste(slot 3).
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "paste"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The full frame (or the "
                                    "previous slot's paste)."}),
                "fill": ("IMAGE", {"tooltip": "The corrected crop fill."}),
                "crop": ("ATLAS_CROP", {}),
            },
        }

    def paste(self, image, fill, crop):
        from PIL import Image as PILImage

        np = _require_numpy()
        torch = _require_torch()
        if not isinstance(crop, dict) or crop.get("empty", True):
            return (image, "AtlasCompositeCrop: empty crop — frame returned "
                           "unchanged (unused slot).")
        frame = image.detach().cpu().numpy()[0]
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8).copy()
        height, width = frame.shape[:2]
        x, y = int(crop["x"]), int(crop["y"])
        w, h = int(crop["width"]), int(crop["height"])
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return (image,
                    f"AtlasCompositeCrop: crop rect ({x},{y},{w},{h}) does "
                    f"not fit the {width}x{height} frame — frame returned "
                    f"unchanged.")
        f = fill.detach().cpu().numpy()[0]
        f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        if f.shape[:2] != (h, w):
            f = np.asarray(PILImage.fromarray(f).resize(
                (w, h), PILImage.LANCZOS))
        frame[y:y + h, x:x + w] = f
        out = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
        return (out, f"AtlasCompositeCrop: pasted {w}x{h} at ({x},{y}).")


class AtlasCameraMovePreset:
    """🎬 The viewport's one-click moves as a NODE — path + exact end pose.

    Mirrors the viewport's move buttons (`applyMovePreset` in
    atlas_blockout.js; constants pinned by tests/test_frontend_mirrors.py) so
    the standard shot recipe — arc left, arc right — can run WITHOUT anyone
    opening the viewport. That is what makes the auto-fill graph automatic:
    preset path in, holes surveyed at its end frame, WAN fills the big ones,
    the artist opens the viewport afterwards to tweak.

    ONE DELIBERATE DIVERGENCE (documented in `build_preset_camera_path`): the
    pivot is `ground_lookat_pivot`, not the viewport's mesh-centre pivot —
    because that is the pivot `AtlasAddPatchView` orbits when it consumes an
    `exact_view_override`. The `exact_view` output is therefore EXACT by
    construction: wire it straight into patch re-entry and the filled end
    frame projects back from the identical pose, no 'Bake Repair Frame'
    click required. Pan moves swivel in place and have no orbit-delta
    representation — they emit the zero delta and say so in the report.
    """

    RETURN_TYPES = ("ATLAS_CAMERA_PATH", "STRING", "STRING")
    RETURN_NAMES = ("camera_path", "exact_view", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        from atlas_camera.core.camera_path import PRESET_MOVES
        return {
            "required": {
                "solve": ("ATLAS_SOLVE", {}),
                "move": (list(PRESET_MOVES), {
                    "default": "arc_left",
                    "tooltip": "Same moves as the viewport buttons. Values "
                               "are append-only (they serialize into saved "
                               "workflows)."}),
            },
            "optional": {
                "angle_deg": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 90.0, "step": 0.5,
                    "tooltip": "Orbit/arc/pan angle. 15 matches the viewport "
                               "buttons."}),
                "frames": ("INT", {"default": 100, "min": 2, "max": 1000}),
                "easing": (["ease_in_out", "ease_in", "ease_out", "linear"], {
                    "default": "ease_in_out"}),
            },
        }

    def build(self, solve, move, angle_deg=15.0, frames=100,
              easing="ease_in_out"):
        import math

        from atlas_camera.core.camera_path import build_preset_camera_path

        fov_deg = None
        intr = solve.camera.intrinsics
        if getattr(intr, "fy_px", None) and getattr(intr, "image_height", None):
            fov_deg = math.degrees(2.0 * math.atan(
                float(intr.image_height) / (2.0 * float(intr.fy_px))))
        path, delta = build_preset_camera_path(
            solve.camera.extrinsics, move, angle_deg=float(angle_deg),
            frame_count=int(frames), easing=easing, fov_deg=fov_deg)
        exact = (f"azimuth_deg={delta[0]:.4f} elevation_deg={delta[1]:.4f} "
                 f"distance_scale={delta[2]:.4f}")
        report = (f"AtlasCameraMovePreset: {move} — {len(path.keyframes)} "
                  f"keyframes, {path.frame_count} frames @ {path.fps:g} fps, "
                  f"angle {angle_deg:g} deg.\nexact_view '{exact}' reproduces "
                  f"the END pose via AtlasAddPatchView (ground-ray pivot — "
                  f"deliberately not the viewport's mesh-centre pivot).")
        if move.startswith("pan_"):
            report += ("\nNOTE: pan swivels in place — no orbit delta can "
                       "express it, so exact_view is the ZERO delta and "
                       "patch re-entry lands at the recovered pose.")
        return (path, exact, report)
