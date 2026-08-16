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
                "exclude_mask": ("MASK", {
                    "tooltip": "auto_largest only: region already carried by "
                               "something other than geometry (SkyDome, "
                               "matte). Subtracted before clustering so a "
                               "generator is never aimed at it. Auto mode "
                               "ALSO always drops what the move did not "
                               "reveal — holes whose ray at infinity lands on "
                               "a hole in the plate (sky: nothing was ever "
                               "occluding it) or outside the plate frame "
                               "entirely (never looked at — that is "
                               "outpainting). This input is for regions those "
                               "geometric tests cannot know about."}),
            },
        }

    def crop(self, solve, source_image, camera_path=None, roi_slot=1,
             pad_frac=0.10, snap=64, hole_dilate_px=4, max_gen_long_edge=1280,
             roi_source="artist", min_area_px=1024, exclude_mask=None):
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
            exclude = None
            if exclude_mask is not None:
                exclude = _mask_to_bool(np, torch, exclude_mask, height,
                                        width)
            # move_revealed_only is the SKY FAILSAFE and always on for auto
            # selection: sky is a hole from the solved pose too, and anything
            # outside the plate's frame was never looked at — neither is
            # disocclusion, so neither ever ranks.
            auto_rois, roi_set, _masks, peak = survey_hole_rois(
                solve, src, [view], survey_resolution=1024,
                pad_frac=float(pad_frac), min_area_px=int(min_area_px),
                snap=int(snap), exclude_mask=exclude,
                move_revealed_only=True)
            excl_note = (", exclude_mask applied" if exclude is not None else "")
            if roi_slot > len(auto_rois):
                return empty(f"auto_largest found only {len(auto_rois)} "
                             f"move-revealed cluster(s) >= {min_area_px}px"
                             f"{excl_note} — no-op branch")
            roi = auto_rois[roi_slot - 1]
            source_note = (f"auto rank {roi_slot}/{len(auto_rois)}, "
                           f"move-revealed only" + excl_note
                           + f" (peak hole {peak:.1%})")
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

        # SELECTION and the CROP MASK must apply the same test. Ranking a
        # cluster with sky excluded and then emitting the rect's full hole mask
        # puts the sky straight back: the ROI is a RECTANGLE, so a legitimate
        # cluster's padded rect can contain sky the ranking already rejected.
        # Measured live 2026-08-15 — the arc-left patch came back with a
        # sky-textured sheet above the roofline, because that sky was inside
        # the winning rect, got generated, got pasted, and therefore rode the
        # pasted_mask into AtlasAddPatchView as geometry to build.
        # Auto mode only: an artist who drew a region gets what they drew.
        sky_note = ""
        if roi_source == "auto_largest":
            from atlas_camera.core.camera_crop import crop_intrinsics
            from atlas_camera.dynamic.occlusion_fill import (
                not_disocclusion_mask,
                plate_hole_survey,
                reproject_at_infinity,
            )
            c_intr = crop_intrinsics(intr, roi)
            c_spec = CameraSpec.from_intrinsics(c_intr)
            plate = plate_hole_survey(solve, src, resolution=1024)
            cam = dict(view=view, fx=c_spec.fx, fy=c_spec.fy, cx=c_spec.cx,
                       cy=c_spec.cy, width=roi.width, height=roi.height)
            drop = not_disocclusion_mask(plate, **cam)
            before = int((mask > 127).sum())
            mask = np.where(drop, 0, mask)
            removed = before - int((mask > 127).sum())
            if removed:
                # Those pixels keep the sentinel otherwise — the membrane
                # composite only touches the hole, so a dropped pixel would
                # ride the saved end frame as chroma green. Sky IS at infinity,
                # so the plate sampled through the same mapping is its real
                # content for this camera; only genuinely off-plate pixels
                # (nothing was ever photographed there) keep the sentinel.
                r = reproject_at_infinity(plate, **cam)
                # r's coordinates are in the SURVEY raster; `src` is the plate
                # at full resolution (width/height here are the plate's).
                sx = width / float(r["plate_width"])
                sy = height / float(r["plate_height"])
                px = np.clip((r["u"] * sx).astype(np.int64), 0, width - 1)
                py = np.clip((r["v"] * sy).astype(np.int64), 0, height - 1)
                paint = drop & r["inside"]
                guide = guide.copy()
                guide[paint] = src[py[paint], px[paint]]
                sky_note = (f", {removed:,} px ({removed / max(before, 1):.0%} "
                            f"of the hole) dropped as not-disocclusion "
                            f"(sky/off-plate); {int(paint.sum()):,} of them "
                            f"resampled from the plate at infinity")
            if not (mask > 127).any():
                return empty(f"every hole pixel in the rank-{roi_slot} rect is "
                             f"sky or off-plate — no-op branch")

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
                     else f" (capped from native, 1:{roi.width / gen_w:.2f})")
                  + sky_note)
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

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "report", "pasted_mask")
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
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "The crop's HOLE mask (AtlasCropROI's mask "
                               "output). When wired, pasted_mask accumulates "
                               "exactly the filled hole pixels at plate "
                               "coordinates — the matte patch re-entry needs "
                               "so a repaired frame contributes only its "
                               "fills. Without it the whole crop rect "
                               "accumulates."}),
                "prior_mask": ("MASK", {
                    "tooltip": "The previous slot's pasted_mask; unioned in, "
                               "so a chain of pastes emits one matte."}),
            },
        }

    def paste(self, image, fill, crop, mask=None, prior_mask=None):
        from PIL import Image as PILImage

        np = _require_numpy()
        torch = _require_torch()

        def out_mask(height, width, hole_rect=None):
            acc = np.zeros((height, width), dtype=np.float32)
            if prior_mask is not None:
                prior = _mask_to_bool(np, torch, prior_mask, height, width)
                acc[prior] = 1.0
            if hole_rect is not None:
                y, x, hole = hole_rect
                acc[y:y + hole.shape[0], x:x + hole.shape[1]][hole] = 1.0
            return torch.from_numpy(acc).unsqueeze(0)

        if not isinstance(crop, dict) or crop.get("empty", True):
            h, w = int(image.shape[1]), int(image.shape[2])
            return (image, "AtlasCompositeCrop: empty crop — frame returned "
                           "unchanged (unused slot).", out_mask(h, w))
        frame = image.detach().cpu().numpy()[0]
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8).copy()
        height, width = frame.shape[:2]
        x, y = int(crop["x"]), int(crop["y"])
        w, h = int(crop["width"]), int(crop["height"])
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return (image,
                    f"AtlasCompositeCrop: crop rect ({x},{y},{w},{h}) does "
                    f"not fit the {width}x{height} frame — frame returned "
                    f"unchanged.", out_mask(height, width))
        f = fill.detach().cpu().numpy()[0]
        f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        if f.shape[:2] != (h, w):
            f = np.asarray(PILImage.fromarray(f).resize(
                (w, h), PILImage.LANCZOS))
        frame[y:y + h, x:x + w] = f
        if mask is not None:
            hole = _mask_to_bool(np, torch, mask, h, w)
        else:
            hole = np.ones((h, w), dtype=bool)
        out = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
        return (out, f"AtlasCompositeCrop: pasted {w}x{h} at ({x},{y}), "
                     f"pasted_mask carries "
                     f"{'hole pixels only' if mask is not None else 'the whole rect'}.",
                out_mask(height, width, (y, x, hole)))


class AtlasCropSourcePhoto:
    """📷✂️ The pristine PHOTO crop of a Fill ROI, at the generation raster —
    the input a subject-centric novel-view model wants.

    The Qwen-Image-Edit-2511 Multiple-Angles LoRA (96 absolute poses, trained
    on Gaussian-splat renders of ONE clear subject) works best, fastest and at
    real detail when it is handed the region of interest, not a whole 36 MP
    plate: diffusion resolution is capped and a full frame spends the pixel
    budget on everything but the hole. `AtlasCropROI` already picks the ROI
    (artist or auto hole cluster) and defines its crop rect + generation
    raster; its `guide` output is a RENDER through the geometry (holes marked)
    for inpainting. This node crops the untouched source photo through the
    SAME handle so the two agree pixel-for-pixel:

        AtlasCropROI ──crop──> AtlasCropSourcePhoto ──source_crop──> Qwen
        Multiple-Angles ──novel view──> AtlasAddPatchView
        (camera_source=register_to_primary, primary_image=full plate)

    The registration step MEASURES the crop's novel-view camera against the
    full primary (SIFT + MoGe pointmap + RANSAC similarity), so the crop needs
    no bookkeeping about where it came from — the pixels say. An empty crop
    handle (unused slot) returns a 64x64 black no-op like the rest of the
    crop family.
    """

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("source_crop", "gen_width", "gen_height", "report")
    FUNCTION = "crop_photo"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "The full plate (same image "
                                                       "AtlasCropROI was given)."}),
                "crop": ("ATLAS_CROP", {"tooltip": "AtlasCropROI's crop handle."}),
            },
            "optional": {
                "pad_frac": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "EXTRA context around the ROI rect (fraction of the rect "
                               "size, clamped to the plate) — a subject-centric model "
                               "wants the subject with some surroundings. 0 = the exact "
                               "ROI rect. The output raster keeps the crop's aspect."}),
                "square": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Expand the rect to a square (centred, clamped to the "
                               "plate) — many novel-view LoRAs were trained square."}),
            },
        }

    def crop_photo(self, source_image, crop, pad_frac=0.0, square=False):
        from PIL import Image as PILImage
        np = _require_numpy()
        torch = _require_torch()

        if not isinstance(crop, dict) or crop.get("empty", True):
            return (torch.zeros(1, 64, 64, 3), 64, 64,
                    "AtlasCropSourcePhoto: empty crop — no-op branch")
        src = source_image.detach().cpu().numpy()[0]
        src = np.clip(src * 255.0, 0, 255).astype(np.uint8)
        H, W = src.shape[:2]
        x, y = int(crop["x"]), int(crop["y"])
        w, h = int(crop["width"]), int(crop["height"])
        gen_w, gen_h = int(crop.get("gen_w", w)), int(crop.get("gen_h", h))
        # Optional context + square, clamped to the plate.
        px, py = int(round(w * float(pad_frac))), int(round(h * float(pad_frac)))
        x0, y0, x1, y1 = x - px, y - py, x + w + px, y + h + py
        if square:
            side = max(x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            x0, x1 = int(round(cx - side / 2.0)), int(round(cx + side / 2.0))
            y0, y1 = int(round(cy - side / 2.0)), int(round(cy + side / 2.0))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return (torch.zeros(1, 64, 64, 3), 64, 64,
                    f"AtlasCropSourcePhoto: rect ({x0},{y0})-({x1},{y1}) is degenerate "
                    "against the plate — no-op branch")
        patch = src[y0:y1, x0:x1]
        # Keep the crop family's generation raster when the rect is unchanged;
        # otherwise scale by the same long-edge ratio and snap to /16.
        if (x0, y0, x1 - x0, y1 - y0) == (x, y, w, h):
            out_w, out_h = gen_w, gen_h
        else:
            ratio = gen_w / float(w) if w else 1.0
            out_w = max(16, int(round((x1 - x0) * ratio / 16)) * 16)
            out_h = max(16, int(round((y1 - y0) * ratio / 16)) * 16)
        if (out_w, out_h) != (x1 - x0, y1 - y0):
            patch = np.asarray(PILImage.fromarray(patch).resize(
                (out_w, out_h), PILImage.LANCZOS))
        t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        return (t, out_w, out_h,
                f"AtlasCropSourcePhoto: photo crop ({x0},{y0}) {x1 - x0}x{y1 - y0} -> "
                f"{out_w}x{out_h}"
                + (f", pad {pad_frac:.2f}" if pad_frac else "")
                + (", square" if square else "")
                + " — feed to the novel-view model; register the result with "
                  "AtlasAddPatchView camera_source=register_to_primary.")


class AtlasCameraMovePreset:
    """🎬 The viewport's one-click moves as a NODE — path + exact end pose.

    Mirrors the viewport's move buttons (`applyMovePreset` in
    atlas_blockout.js; constants pinned by tests/test_frontend_mirrors.py) so
    the standard shot recipe — arc left, arc right — can run WITHOUT anyone
    opening the viewport. That is what makes the auto-fill graph automatic:
    preset path in, holes surveyed at its end frame, WAN fills the big ones,
    the artist opens the viewport afterwards to tweak.

    The pivot is the viewport's: `scene_median_depth_pivot` — the central view
    ray at the scene's median vertex depth. It shipped (2026-08-14) orbiting
    `ground_lookat_pivot` instead, because that was the only pivot
    `AtlasAddPatchView` could reproduce from an `exact_view_override`. On a
    near-level camera the ground ray lands far past the subject (~43 m on the
    ghost-town street against the viewport's ~9.8 m), and orbit travel is
    `2·R·sin(angle/2)` — so the arcs swung the eye ~4x further than the ⤴/⤵
    buttons do and opened four times the disocclusion. The pivot now travels
    WITH the delta (`pivot=x,y,z` in the exact-view string), so exactness costs
    no radius: wire `exact_view` straight into patch re-entry and the filled
    end frame projects back from the identical pose, no 'Bake Repair Frame'
    click required. Pan moves swivel in place and have no orbit-delta
    representation — they emit the zero delta and say so in the report.

    `angle_deg` defaults to 12, not the viewport buttons' 15: the automated
    recipe fills what it opens, and 12° at the scene pivot is the artist-set
    budget for one arc (2026-08-15). Raise it for more parallax, at the cost of
    bigger holes for the generator.
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
                    "default": 12.0, "min": 1.0, "max": 90.0, "step": 0.5,
                    "tooltip": "Orbit/arc/pan angle about the scene pivot. 12 "
                               "is the automated recipe's fill budget; the "
                               "viewport buttons use 15."}),
                "frames": ("INT", {"default": 100, "min": 2, "max": 1000}),
                "easing": (["ease_in_out", "ease_in", "ease_out", "linear"], {
                    "default": "ease_in_out"}),
            },
        }

    def build(self, solve, move, angle_deg=12.0, frames=100,
              easing="ease_in_out"):
        import math

        from atlas_camera.core.camera_path import (
            build_preset_camera_path,
            scene_median_depth_pivot,
        )
        from atlas_camera.comfy.view_prompts import _format_exact_view

        fov_deg = None
        intr = solve.camera.intrinsics
        if getattr(intr, "fy_px", None) and getattr(intr, "image_height", None):
            fov_deg = math.degrees(2.0 * math.atan(
                float(intr.image_height) / (2.0 * float(intr.fy_px))))
        pivot = scene_median_depth_pivot(solve)
        eye = solve.camera.extrinsics.camera_position
        radius = math.dist(tuple(float(v) for v in eye), pivot)
        path, delta = build_preset_camera_path(
            solve.camera.extrinsics, move, angle_deg=float(angle_deg),
            frame_count=int(frames), easing=easing, fov_deg=fov_deg,
            pivot=pivot)
        exact = _format_exact_view(delta, pivot)
        travel = 2.0 * radius * math.sin(math.radians(float(angle_deg)) / 2.0)
        report = (f"AtlasCameraMovePreset: {move} — {len(path.keyframes)} "
                  f"keyframes, {path.frame_count} frames @ {path.fps:g} fps, "
                  f"angle {angle_deg:g} deg.\n"
                  f"Scene pivot {pivot[0]:.2f},{pivot[1]:.2f},{pivot[2]:.2f} "
                  f"(radius {radius:.2f} m) — the viewport's own median-depth "
                  f"pivot; eye travels {travel:.2f} m.\nexact_view '{exact}' "
                  f"reproduces the END pose via AtlasAddPatchView (the "
                  f"pivot= term is what makes that exact off the ground ray).")
        if move.startswith("pan_"):
            report += ("\nNOTE: pan swivels in place — no orbit delta can "
                       "express it, so exact_view is the ZERO delta and "
                       "patch re-entry lands at the recovered pose.")
        return (path, exact, report)
