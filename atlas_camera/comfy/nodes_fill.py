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
