"""Atlas ComfyUI nodes — inpaint group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
from __future__ import annotations

import base64
import copy
import io

from atlas_camera.comfy.node_helpers import (
    _BAND_GEOMETRY_CHOICES,
    _analytic_ground_forward_depth,
    _apply_band_split,
    _b64_png_to_mask,
    _band_resolution_validity,
    _comfy_registry,
    _extend_edge_colors,
    _flood_mask_to_frame_borders,
    _graph_builder,
    _image_tensor_to_pil,
    _mask_to_b64_png,
    _metric_depth_and_validity,
    _native_sam3_available,
    _parse_band_override,
    _require_numpy,
    _require_pil,
    _require_torch,
    _resolve_band_geometry,
    _resolve_exclude_mask,
    _seg_coverage,
)




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


class AtlasSAM3Mask:
    """🪄 Native SAM3 concept mask via transformers — no triton/comfyui-rmbg
    dependency.

    The third-party `SAM3Segment` node (comfyui-rmbg) hard-requires triton,
    which does not exist on Mac (MPS) / CPU / AMD — those users could never
    load real SAM3 and always fell back to `AtlasSemanticMask` (SegFormer).
    This node loads SAM3 straight from `transformers>=5.5.4`
    (`Sam3Model`/`Sam3Processor`), so it works everywhere `[sam3]` installs:
    CUDA, CPU, and MPS alike (MPS support is best-effort — the underlying
    sam3_concept_mask() falls back to CPU automatically if an op isn't yet
    supported on MPS).

    `AtlasInput`'s sky/scope cascade now prefers this node over
    `AtlasSemanticMask`, which remains the learned fallback tier when
    `transformers<5.5.4` (or `[sam3]` isn't installed).

    `facebook/sam3` is GATED on Hugging Face (Meta's SAM-License-1.0).
    One-time setup: request access at https://huggingface.co/facebook/sam3,
    then `hf auth login` (or set HF_TOKEN). A gated-repo failure is caught
    and returned as the `report` string rather than raised — it's a one-time
    auth step, not a broken install. See INSTALL.md.
    """
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "segment"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "concepts": ("STRING", {"default": "sky",
                    "tooltip": "Comma-separated open-vocabulary concepts (e.g. 'sky', "
                               "'person, vehicle'). The mask is the UNION of all detected "
                               "instances across every concept."}),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                    "step": 0.01}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                # APPENDED 2026-07-21 (positional widgets_values rule): the
                # per-instance view. SAM3 has always returned a stack from
                # post_process_instance_segmentation; `merged` just unions it.
                # Exposing `separate` is what lets AtlasInstanceMask /
                # AtlasSegmentedSDXLInpaint stop calling the third-party
                # SAM3Segment, which hard-requires triton and therefore cannot
                # load on Mac/CPU/AMD at all.
                "output_mode": (["merged", "separate"], {"default": "merged",
                    "tooltip": "merged = one union mask (the default, and what every "
                               "sky/scope consumer wants). separate = an (N,H,W) stack, "
                               "one instance per slice, ordered LARGEST FIRST — feed "
                               "AtlasInstanceMask to pick one."}),
                "max_instances": ("INT", {"default": 0, "min": 0, "max": 128,
                    "tooltip": "separate mode only: keep at most N instances (largest "
                               "first). 0 = unlimited. Ignored when merged."}),
            },
        }

    def segment(self, image, concepts="sky", confidence_threshold=0.5, device="auto",
                output_mode="merged", max_instances=0, **_extra):
        from atlas_camera.inference.sam3_segmenter import (
            DEFAULT_SAM3_MODEL, Sam3GatedRepoError, sam3_concept_mask,
            sam3_instance_masks)
        torch = _require_torch()

        pil = _image_tensor_to_pil(image)
        dev = None if device == "auto" else device
        empty = torch.zeros((1, pil.height, pil.width), dtype=torch.float32)

        if str(output_mode) == "separate":
            try:
                instances, matched = sam3_instance_masks(
                    pil, concepts, model_id=DEFAULT_SAM3_MODEL, device=dev,
                    confidence_threshold=confidence_threshold,
                    max_instances=int(max_instances))
            except Sam3GatedRepoError as exc:
                return (empty, str(exc))
            if not instances:
                return (empty, f"NO MATCH for '{concepts}' — no instances "
                               f"({DEFAULT_SAM3_MODEL}).")
            import numpy as np
            stack = torch.from_numpy(
                np.stack(instances).astype("float32"))
            sizes = ", ".join(f"{float(m.mean()):.1%}" for m in instances[:8])
            return (stack, f"matched {sorted(set(matched))} -> {len(instances)} "
                           f"instance(s), largest first [{sizes}] "
                           f"({DEFAULT_SAM3_MODEL})")

        try:
            mask_np, matched, coverage = sam3_concept_mask(
                pil, concepts, model_id=DEFAULT_SAM3_MODEL, device=dev,
                confidence_threshold=confidence_threshold)
        except Sam3GatedRepoError as exc:
            return (empty, str(exc))
        mask = torch.from_numpy(mask_np.astype("float32")).unsqueeze(0)
        if matched:
            report = (f"matched {sorted(set(matched))} -> {coverage:.1%} of frame "
                      f"({DEFAULT_SAM3_MODEL})")
        else:
            report = f"NO MATCH for '{concepts}' — mask is empty ({DEFAULT_SAM3_MODEL})."
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


class AtlasSDXLInpaint:
    """Native ComfyUI SDXL inpaint adapter.

    This deliberately expands to ComfyUI's stock checkpoint/conditioning/
    latent inpaint nodes instead of importing a model implementation. It can
    therefore use SDXL checkpoints already installed in ``models/checkpoints``
    and stays compatible with ComfyUI's memory/offload policies. Feed it a
    cropped image and mask (usually from AtlasInpaintCrop), then stitch its
    IMAGE output with AtlasInpaintStitch.
    """
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "expand_sdxl"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "checkpoint": ("STRING", {"default": "SDXL/sd_xl_base_1.0.safetensors",
                "tooltip": "Checkpoint filename in models/checkpoints. Use an SDXL-compatible "
                           "inpaint/base checkpoint."}),
            "positive_prompt": ("STRING", {"default": "high detail, coherent architecture",
                "multiline": True}),
            "negative_prompt": ("STRING", {"default": "blurry, warped, duplicate, text",
                "multiline": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
            "denoise": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            "grow_mask_by": ("INT", {"default": 8, "min": 0, "max": 64}),
            # APPENDED 2026-07-18 (positional rule): perf clamp. Oversized
            # crops force the sampler + VAE through OOM-retry ladders and
            # tiled encoding (measured live: 782s for 4 crops on a 3690px
            # plate). 0 = off (back-compat).
            "max_side": ("INT", {"default": 0, "min": 0, "max": 4096,
                "tooltip": "Downscale the crop to this long edge before SDXL (1024 = "
                           "SDXL native) and lanczos-upscale the result back. Kills the "
                           "OOM/tiled-VAE slow path on big crops; inpainted content is "
                           "generative, so the quality cost is minor. 0 = off."}),
        }, "optional": {
            # APPENDED 2026-07-19 (positional rule). SDXL's architectural
            # prior strongly prefers eye-level front elevations when a large
            # mask removes most of an oblique facade. Keep this optional so
            # old API/UI workflows that do not serialize it use the Python
            # default and remain executable.
            "preserve_perspective": ("BOOLEAN", {"default": True,
                "tooltip": "Append camera-geometry guidance that tells SDXL to continue "
                           "the source viewpoint, facade angle, foreshortening, and "
                           "converging lines; negatively conditions straight-on and "
                           "orthographic facades. Disable for intentionally novel views."}),
        }}

    def expand_sdxl(self, image, mask, checkpoint, positive_prompt,
                    negative_prompt, seed=0, steps=30, cfg=5.5,
                    denoise=0.85, grow_mask_by=8, max_side=0,
                    preserve_perspective=True):
        registry = _comfy_registry()
        required = ("CheckpointLoaderSimple", "CLIPTextEncode",
                    "InpaintModelConditioning", "KSampler", "VAEDecode")
        missing = [name for name in required if name not in registry]
        if missing:
            raise RuntimeError("Native SDXL inpaint requires ComfyUI nodes missing from "
                               "the registry: " + ", ".join(missing))

        # Perf clamp: python-side downscale (real tensors are available at
        # expansion time), upscale back to the EXACT original size via an
        # in-graph ImageScale so AtlasInpaintStitch pastes 1:1.
        orig_h, orig_w = int(image.shape[1]), int(image.shape[2])
        scaled = False
        if max_side and max(orig_h, orig_w) > int(max_side):
            import torch.nn.functional as F
            factor = float(max_side) / float(max(orig_h, orig_w))
            nh = max(8, int(round(orig_h * factor / 8.0)) * 8)
            nw = max(8, int(round(orig_w * factor / 8.0)) * 8)
            image = F.interpolate(image.movedim(-1, 1), size=(nh, nw),
                                  mode="bilinear", antialias=True).movedim(1, -1)
            m = mask if mask.dim() == 3 else mask.unsqueeze(0)
            mask = F.interpolate(m.unsqueeze(1).float(), size=(nh, nw),
                                 mode="bilinear").squeeze(1)
            scaled = True

        g = _graph_builder()
        ckpt = g.node("CheckpointLoaderSimple", ckpt_name=str(checkpoint))
        positive_text = str(positive_prompt).strip()
        negative_text = str(negative_prompt).strip()
        if preserve_perspective:
            positive_text += (
                ", same subject seen from the exact source camera viewpoint, preserve "
                "the source camera perspective and facade angle, preserve strong "
                "foreshortening, continue vertical and horizontal lines with the same "
                "vanishing directions as the surrounding unmasked image")
            negative_text += (
                ", front elevation, straight-on facade, eye-level view, orthographic "
                "view, centered symmetrical building, flat perspective")
        positive = g.node("CLIPTextEncode", text=positive_text, clip=ckpt.out(1))
        negative = g.node("CLIPTextEncode", text=negative_text, clip=ckpt.out(1))
        conditioning = g.node("InpaintModelConditioning",
                               positive=positive.out(0), negative=negative.out(0),
                               pixels=image, vae=ckpt.out(2), mask=mask,
                               noise_mask=True)
        sampled = g.node("KSampler", model=ckpt.out(0), seed=int(seed),
                         steps=int(steps), cfg=float(cfg), sampler_name="dpmpp_2m",
                         scheduler="karras", positive=conditioning.out(0),
                         negative=conditioning.out(1), latent_image=conditioning.out(2),
                         denoise=float(denoise))
        decoded = g.node("VAEDecode", samples=sampled.out(0), vae=ckpt.out(2))
        out_ref = decoded.out(0)
        size_note = ""
        if scaled:
            up = g.node("ImageScale", image=decoded.out(0), upscale_method="lanczos",
                        width=orig_w, height=orig_h, crop="disabled")
            out_ref = up.out(0)
            size_note = (f", sampled at {int(image.shape[2])}x{int(image.shape[1])} "
                         f"(max_side {int(max_side)}) → {orig_w}x{orig_h}")
        report = (f"SDXL inpaint via InpaintModelConditioning — checkpoint={checkpoint}, "
                  f"steps={int(steps)}, cfg={float(cfg):g}, denoise={float(denoise):g}, "
                  f"perspective={'preserve' if preserve_perspective else 'prompt-only'}"
                  + size_note)
        return {"result": (out_ref, report), "expand": g.finalize()}


class AtlasInstanceMask:
    """Select one SAM3 ``Separate`` instance and optionally scope it."""
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "select"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "instance_index": ("INT", {"default": 0, "min": 0, "max": 127}),
        }, "optional": {
            "restrict_mask": ("MASK",),
            "min_coverage": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
        }}

    def select(self, mask, instance_index=0, restrict_mask=None, min_coverage=0.001):
        torch = _require_torch()
        import torch.nn.functional as F
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        idx = int(instance_index)
        if idx < 0 or idx >= int(m.shape[0]):
            out = torch.zeros((1, m.shape[1], m.shape[2]), dtype=m.dtype, device=m.device)
            return out, f"instance {idx}: empty (SAM3 returned {int(m.shape[0])} instance(s))"
        out = m[idx:idx + 1].float().clamp(0, 1)
        if restrict_mask is not None:
            r = restrict_mask if restrict_mask.dim() == 3 else restrict_mask.unsqueeze(0)
            if tuple(r.shape[-2:]) != tuple(out.shape[-2:]):
                r = F.interpolate(r.unsqueeze(1).float(), size=out.shape[-2:], mode="nearest").squeeze(1)
            out = out * r[:1].to(out.device)
        coverage = float((out > 0.5).float().mean())
        if coverage < float(min_coverage):
            out.zero_()
            return out, f"instance {idx}: rejected ({coverage:.3%} coverage)"
        return out, f"instance {idx}: {coverage:.3%} coverage"


class AtlasSegmentedSDXLInpaint:
    """SAM3-separated building masks -> per-instance SDXL crop/stitch stack."""
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "expand_stack"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "restrict_mask": ("MASK",),
            "prompt": ("STRING", {"default": "building facade, photorealistic continuation", "multiline": True}),
            "checkpoint": ("STRING", {"default": "SDXL/sd_xl_base_1.0.safetensors"}),
            "max_instances": ("INT", {"default": 4, "min": 1, "max": 8}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {
            # APPENDED 2026-07-18: perf default ON — building crops on a big
            # plate blow past SDXL's native scale and hit the OOM/tiled-VAE
            # slow path (782s measured for 4 crops). See AtlasSDXLInpaint.
            "crop_max_side": ("INT", {"default": 1024, "min": 0, "max": 4096,
                "tooltip": "Per-crop long-edge clamp fed to the inner SDXL inpaint "
                           "(1024 = SDXL native; result upscaled back). 0 = off."}),
        }}

    def expand_stack(self, image, restrict_mask, prompt, checkpoint, max_instances=4,
                     steps=30, cfg=4.0, denoise=0.65, seed=0, crop_max_side=1024):
        registry = _comfy_registry()
        # Native SAM3 (transformers, no triton) is PREFERRED — the same
        # cascade AtlasInput's segment() uses. The third-party SAM3Segment
        # stays as the fallback for installs that predate [sam3], but it
        # cannot load on Mac/CPU/AMD at all, which is why this node was
        # arm64-blocked before AtlasSAM3Mask grew output_mode="separate".
        use_native = _native_sam3_available()
        for name in (("AtlasSAM3Mask",) if use_native else ("SAM3Segment",)) + (
                "INPAINT_ExpandMask", "AtlasInpaintCrop",
                "AtlasSDXLInpaint", "AtlasInpaintStitch"):
            if name not in registry:
                raise RuntimeError(f"Segmented SDXL inpaint requires node '{name}'")
        g = _graph_builder()
        if use_native:
            sam = g.node("AtlasSAM3Mask", image=image, concepts="building",
                         confidence_threshold=0.5, device="auto",
                         output_mode="separate", max_instances=int(max_instances))
            instances = sam.out(0)          # AtlasSAM3Mask: mask is slot 0
        else:
            sam = g.node("SAM3Segment", image=image, prompt="building",
                         output_mode="Separate", confidence_threshold=0.5,
                         max_segments=int(max_instances), segment_pick=0,
                         mask_blur=0, mask_offset=0, device="Auto",
                         invert_output=False, unload_model=False,
                         background="Alpha", background_color="#222222")
            instances = sam.out(1)          # SAM3Segment: IMAGE(0), MASK(1)
        plate = image
        for i in range(int(max_instances)):
            selected = g.node("AtlasInstanceMask", mask=instances, restrict_mask=restrict_mask,
                              instance_index=i, min_coverage=0.001)
            grown = g.node("INPAINT_ExpandMask", mask=selected.out(0), grow=32,
                           blur=16, blur_type="gaussian")
            crop = g.node("AtlasInpaintCrop", image=plate, mask=grown.out(0), context_pad_px=128)
            fill = g.node("AtlasSDXLInpaint", image=crop.out(0), mask=crop.out(1),
                          checkpoint=checkpoint, positive_prompt=prompt,
                          negative_prompt="fantasy, sci-fi, warped, duplicate, text, seams",
                          seed=int(seed) + i, steps=int(steps), cfg=float(cfg),
                          denoise=float(denoise), grow_mask_by=8,
                          max_side=int(crop_max_side))
            plate = g.node("AtlasInpaintStitch", original_image=plate,
                           inpainted_crop=fill.out(0), crop_region=crop.out(2),
                           mask=grown.out(0), feather_px=24).out(0)
        report = (f"SAM3 Separate building stack via "
                  f"{'AtlasSAM3Mask (native)' if use_native else 'SAM3Segment (triton)'} — "
                  f"{int(max_instances)} instance slot(s), SDXL denoise {float(denoise):g}")
        return {"result": (plate, report), "expand": g.finalize()}


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
                "quad_coherence": ("BOOLEAN", {"default": True,
                    "tooltip": "Reject both triangles when either half of a grid quad fails; avoids "
                               "surviving diagonal UV wedges at band boundaries."}),
            },
        }

    def add_layer(self, solve, depth, plate_image, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5,
                  name="layer", priority=0.0, plate_ref=None, relief_grid=384, depth_edge_rel=1.5,
                  exclude_mask=None, fill_occluded=False, embed_matte=False, layer_matte=None,
                  edge_extend_px=0, skirt_bevel=0.0, frame_outpaint_px=0,
                  exclude_choke_cells=2, band_side="manual", band_split=None,
                  band_geometry="relief", geometry_override="", band_ref_mask=None,
                  band_override="", max_edge_factor=12.0, normal_edge_deg=0.0,
                  quad_coherence=True):
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
            quad_coherence=bool(quad_coherence),
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
                "quad_coherence": bool(quad_coherence),
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
