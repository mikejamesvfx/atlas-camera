"""Atlas ComfyUI nodes — depth group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
from __future__ import annotations

import copy
import math
import os

from atlas_camera.core.mask_ops import dilate
from atlas_camera.comfy.node_helpers import (
    _BOUNDED_BAND_NOOP_M,
    _DEPTH_MODEL_CHOICES,
    _MOGE_NORMAL_MODEL_CHOICES,
    _apply_band_split,
    _band_resolution_validity,
    _depth_map_for_solve,
    _ground_depth_compute,
    _image_tensor_to_pil,
    _metric_depth_and_validity,
    _parse_band_override,
    _pil_to_image_tensor,
    _require_numpy,
    _require_pil,
    _require_torch,
    _resize_normal_field,
    _resolve_exclude_mask,
    _save_image_tensor_to_tmp,
    _solve_focal_px_for_image,
    _solve_image_size,
)




class AtlasDepthAnything:
    """Monocular depth (Depth Anything V2) as a standalone IMAGE + the raw solve depth slot.

    Outputs a normalized grayscale depth image for preview/compositing. Requires the
    [neural] extra (torch + transformers) in ComfyUI's venv.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_image",)
    FUNCTION = "estimate"
    CATEGORY = "Atlas/advanced"

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
    CATEGORY = "Atlas"

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
                # APPENDED 2026-07-28 (positional widgets_values rule): MoGe-only
                # cost dials, both INERT at their defaults so every saved graph
                # behaves exactly as before. Ignored by non-MoGe backends.
                "moge_resolution_level": ("INT", {"default": 9, "min": 0, "max": 9,
                    "tooltip": "MoGe ONLY. Its own token-budget dial; 9 is MoGe's default and "
                               "full detail. Lower = faster and coarser. No effect on DA/V2/DepthPro."}),
                "moge_max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 64,
                    "tooltip": "MoGe ONLY. Cap the longer edge BEFORE inference (0 = off). MoGe "
                               "resamples internally anyway, so this buys VRAM and time, not "
                               "quality — an 8K plate is ~415 MB of float32 on the device before "
                               "MoGe touches it. Depth/normals still come back at SOURCE size."}),
                "moge_checkpoint_path": ("STRING", {"default": "",
                    "tooltip": "MoGe ONLY. Local MoGe `model.pt` to load instead of downloading "
                               "from HuggingFace (air-gapped / shared model dirs). NOT ComfyUI "
                               "core's geometry_estimation/*.safetensors — different container, "
                               "and it carries no model_config."}),
                "moge_tile_side": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 128,
                    "tooltip": "MoGe ONLY. Run inference on overlapping TILES of this size at "
                               "SOURCE resolution (0 = off). The opposite lever to moge_max_side: "
                               "that one downscales to save VRAM, this one refuses to downscale so "
                               "a 36MP plate keeps its fine structure — the model spends its whole "
                               "token budget on each tile instead of on the shrunken whole. Costs "
                               "one inference pass PER TILE plus one global pass, so a 4x4 tiling "
                               "is ~17x the time. Every tile is affine-fitted onto that global "
                               "pass first: monocular depth is scale-ambiguous per input, so raw "
                               "tiles disagree and pasting them steps at every seam. Try 1024."}),
                "moge_tile_overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "MoGe ONLY. Tile overlap as a fraction of tile size. More overlap "
                               "= wider blend and more tiles (slower). Only used when "
                               "moge_tile_side > 0."}),
                # APPENDED 2026-08-16 (positional widgets_values rule). INERT by
                # default. Lights up the scene-health focal cross-check on MoGe.
                "moge_report_free_focal": ("BOOLEAN", {"default": False,
                    "tooltip": "MoGe ONLY. When a solve is wired, MoGe is FED the solve's focal, "
                               "so its own intrinsics just echo it. This runs a second, fov-free "
                               "pass (depth discarded, cheaper resolution level) and records "
                               "MoGe's INDEPENDENT focal so the 🩺 debug report can flag a "
                               "focal_mismatch (band 0.75-1.33). ~+30% time. sh001 example: solve "
                               "6207 px vs MoGe 5278 px — invisible without this."}),
            },
        }

    def estimate(self, image, depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto", solve=None, moge_resolution_level=9, moge_max_side=0,
                 moge_checkpoint_path="", moge_tile_side=0, moge_tile_overlap=0.25,
                 moge_report_free_focal=False):
        from atlas_camera.inference.depth_estimator import estimate_depth
        tmp = _save_image_tensor_to_tmp(image)
        try:
            result = estimate_depth(tmp, model_id=depth_model,
                                    device=None if device == "auto" else device,
                                    focal_px=_solve_focal_px_for_image(solve, image),
                                    resolution_level=int(moge_resolution_level),
                                    max_side=int(moge_max_side),
                                    checkpoint_path=str(moge_checkpoint_path or ""),
                                    tile_side=int(moge_tile_side),
                                    tile_overlap=float(moge_tile_overlap),
                                    report_free_focal=bool(moge_report_free_focal))
        finally:
            os.unlink(tmp)
        return (result,)


class AtlasDepthOutlierMask:
    """Build an explicit mask for local monocular-depth outliers."""
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("outlier_mask", "report")
    FUNCTION = "detect"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"depth": ("ATLAS_DEPTH_MAP",)}, "optional": {
            "solve": ("ATLAS_SOLVE",),
            "relative_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 3.0, "step": 0.05}),
            "mad_threshold": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 50.0, "step": 0.5}),
            "dilate_px": ("INT", {"default": 2, "min": 0, "max": 32}),
        }}

    def detect(self, depth, solve=None, relative_threshold=0.35,
               mad_threshold=6.0, dilate_px=2):
        torch = _require_torch()
        np = _require_numpy()
        h = int(getattr(depth, "image_height", depth.depth.shape[0]))
        w = int(getattr(depth, "image_width", depth.depth.shape[1]))
        d = _depth_map_for_solve(depth, w, h).astype(np.float32)
        valid = np.isfinite(d) & (d > 1e-4)
        pad = np.pad(d, 1, mode="edge")
        samples = np.stack([pad[dy:dy + h, dx:dx + w]
                            for dy in range(3) for dx in range(3)], axis=0)
        med = np.nanmedian(np.where(samples > 1e-4, samples, np.nan), axis=0)
        abs_dev = np.abs(samples - med[None])
        mad = np.nanmedian(np.where(samples > 1e-4, abs_dev, np.nan), axis=0)
        rel_bad = np.abs(d - med) / np.maximum(med, 1e-4) > float(relative_threshold)
        robust_bad = np.abs(d - med) > float(mad_threshold) * np.maximum(mad, 1e-3)
        bad = valid & np.isfinite(med) & rel_bad & robust_bad
        # Small dilation keeps the bad cell from becoming a one-cell stretched
        # bridge when the relief grid samples just beside it.
        for _ in range(max(0, int(dilate_px))):
            b = bad.copy()
            b[1:] |= bad[:-1]; b[:-1] |= bad[1:]
            b[:, 1:] |= bad[:, :-1]; b[:, :-1] |= bad[:, 1:]
            bad = b
        mask = torch.from_numpy(bad.astype(np.float32)).unsqueeze(0)
        return mask, f"depth outlier mask: {int(bad.sum())} px ({float(bad.mean()):.2%})"


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
    CATEGORY = "Atlas/advanced"

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
                # APPENDED 2026-07-28 (positional widgets_values rule). Inert at
                # their defaults. This node is MoGe by definition, so unlike on
                # AtlasDepthMap they always apply.
                "resolution_level": ("INT", {"default": 9, "min": 0, "max": 9,
                    "tooltip": "MoGe's token-budget dial; 9 is its default and full detail. "
                               "Lower = faster, coarser normals."}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 64,
                    "tooltip": "Cap the longer edge before inference (0 = off). Normals come back "
                               "at SOURCE size either way — this only buys VRAM and time. Cheap "
                               "here: normals are lower-frequency than depth, so a downscaled "
                               "normal pass costs less quality than a downscaled depth pass."}),
                "checkpoint_path": ("STRING", {"default": "",
                    "tooltip": "Local MoGe `model.pt` instead of a HuggingFace download. NOT "
                               "ComfyUI core's *.safetensors — different container."}),
            },
        }

    def attach(self, depth, image, normal_model="Ruicheng/moge-2-vitl-normal",
               device="auto", solve=None, resolution_level=9, max_side=0,
               checkpoint_path=""):
        import copy
        base = getattr(depth, "depth", None)
        if base is None:
            return (depth, "AtlasMogeNormals: input depth carries no array — passed through unchanged.")
        from atlas_camera.inference.depth_estimator import estimate_depth
        tmp = _save_image_tensor_to_tmp(image)
        try:
            moge = estimate_depth(tmp, model_id=normal_model,
                                  device=None if device == "auto" else device,
                                  focal_px=_solve_focal_px_for_image(solve, image),
                                  resolution_level=int(resolution_level),
                                  max_side=int(max_side),
                                  checkpoint_path=str(checkpoint_path or ""))
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


class AtlasDepthDetailEnhance:
    """🔬 Emboss the normal map's high-frequency shape onto the shared depth.

    Monocular depth is metrically sound but low-frequency — brick courses,
    window reveals, rock striations flatten out. The predicted normals (MoGe
    ``*-normal`` depth, or attached via AtlasMogeNormals) carry exactly that
    detail. This node integrates them into a height field (Frankot-Chellappa),
    strips everything below ``detail_cutoff_px`` (so the metric base can never
    tilt or re-scale), and blends only the surviving fine detail onto a COPY of
    the ATLAS_DEPTH_MAP — log-domain, median-renormalized, scale-preserving by
    construction. Passes through unchanged (with the reason in the report) when
    the input carries no normals. Wire an ``exclude_mask`` (e.g. sky) to keep
    the emboss off regions whose normals are meaningless.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "STRING")
    RETURN_NAMES = ("depth", "report")
    FUNCTION = "enhance"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"depth": ("ATLAS_DEPTH_MAP",)},
            "optional": {
                "strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_cutoff_px": ("INT", {"default": 64, "min": 4, "max": 1024,
                                             "tooltip": "Wavelengths longer than this (px) are "
                                             "discarded from the integrated surface — only finer "
                                             "detail is embossed, which is what keeps the metric "
                                             "scale untouchable."}),
                "exclude_mask": ("MASK", {"tooltip": "1 = leave this pixel's depth untouched "
                                          "(sky, glass — anywhere normals are meaningless)."}),
            },
        }

    def enhance(self, depth, strength=0.35, detail_cutoff_px=64, exclude_mask=None):
        np = _require_numpy()
        base = getattr(depth, "depth", None)
        if base is None:
            return (depth, "AtlasDepthDetailEnhance: input depth carries no array — passed through.")
        raw_normal = getattr(depth, "normal", None)
        if raw_normal is None:
            return (depth, "AtlasDepthDetailEnhance: no normals on this depth map — use a MoGe "
                           "'*-normal' depth model or wire AtlasMogeNormals upstream. "
                           "Depth passed through unchanged.")
        if float(strength) <= 0.0:
            return (depth, "AtlasDepthDetailEnhance: strength 0 — passed through unchanged.")

        from atlas_camera.core.depth_detail import (
            blend_depth_detail,
            highpass_detail,
            integrate_normals_frankot_chellappa,
        )
        d = np.asarray(base, dtype=np.float64)
        normal = _resize_normal_field(raw_normal, d.shape[:2])
        excl = _resolve_exclude_mask(exclude_mask, d.shape[0], d.shape[1])
        height_field = integrate_normals_frankot_chellappa(normal)
        detail = highpass_detail(height_field, float(detail_cutoff_px))
        blended = blend_depth_detail(d, detail, float(strength), exclude_mask=excl)

        out = copy.copy(depth)            # never mutate the SHARED depth object
        out.depth = blended
        valid = np.isfinite(blended) & (blended > 0)
        if valid.any():
            out.near = float(blended[valid].min())
            out.far = float(blended[valid].max())
        out.metadata = {**(depth.metadata or {}),
                        "detail_enhanced": True,
                        "detail_strength": float(strength),
                        "detail_cutoff_px": int(detail_cutoff_px)}
        med = float(np.nanmedian(d[np.isfinite(d) & (d > 0)])) if valid.any() else 0.0
        return (out, "AtlasDepthDetailEnhance: embossed normal-integrated detail "
                     f"(strength {float(strength):.2f}, cutoff {int(detail_cutoff_px)} px, "
                     f"median depth preserved at {med:.2f}). Metric scale unchanged.")


class AtlasDepthCombine:
    """➕ Combine two shared depth maps into one.

    Modes:
      * ``high_freq_detail`` — graft the SOURCE's fine structure onto the
        base's metric far-field (the "MoGe detail on V2 exterior" combo);
        log-domain and median-renormalized, so base scale is preserved.
      * ``min`` / ``max`` — per-pixel envelope (NaNs lose).
      * ``masked`` — lerp base -> source under ``blend_mask`` * ``strength``.

    Always returns a COPY carrying the base's is_metric flag and provenance;
    mixing metric with relative inputs is reported as a warning, not hidden.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "STRING")
    RETURN_NAMES = ("depth", "report")
    FUNCTION = "combine"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_base": ("ATLAS_DEPTH_MAP",),
                "depth_detail_src": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "mode": (["high_freq_detail", "min", "max", "masked"],
                         {"default": "high_freq_detail"}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_cutoff_px": ("INT", {"default": 64, "min": 4, "max": 1024,
                                             "tooltip": "high_freq_detail mode only."}),
                "blend_mask": ("MASK", {"tooltip": "masked mode only: 1 = take the source."}),
            },
        }

    def combine(self, depth_base, depth_detail_src, mode="high_freq_detail",
                strength=0.5, detail_cutoff_px=64, blend_mask=None):
        np = _require_numpy()
        base = np.asarray(depth_base.depth, dtype=np.float64)
        h, w = base.shape[:2]
        src = np.asarray(_depth_map_for_solve(depth_detail_src, w, h), dtype=np.float64)

        warnings = []
        if bool(depth_base.is_metric) != bool(depth_detail_src.is_metric):
            warnings.append(
                f"WARNING: mixing metric={depth_base.is_metric} base with "
                f"metric={depth_detail_src.is_metric} source — min/max/masked "
                "values are not unit-compatible; result keeps the base's flag.")

        if mode == "high_freq_detail":
            from atlas_camera.core.depth_detail import combine_depth_high_freq
            result = combine_depth_high_freq(base, src, float(strength),
                                             cutoff_px=float(detail_cutoff_px))
        elif mode == "min":
            result = np.fmin(base, src).astype(np.float32)
        elif mode == "max":
            result = np.fmax(base, src).astype(np.float32)
        elif mode == "masked":
            if blend_mask is None:
                m = np.zeros((h, w), dtype=np.float64)
                warnings.append("WARNING: masked mode with no blend_mask — base returned unchanged.")
            else:
                from atlas_camera.core.solver import _resize_depth
                m = blend_mask[0].detach().cpu().numpy().astype(np.float64)
                if m.shape != (h, w):
                    m = _resize_depth(m, w, h)
            m = np.clip(m, 0.0, 1.0) * float(strength)
            src_ok = np.isfinite(src)
            m = np.where(src_ok, m, 0.0)
            result = (base * (1.0 - m) + np.where(src_ok, src, 0.0) * m).astype(np.float32)
        else:  # pragma: no cover - combo constrains the values
            raise ValueError(f"unknown combine mode: {mode}")

        out = copy.copy(depth_base)       # never mutate the SHARED depth object
        out.depth = result
        valid = np.isfinite(result) & (result > 0)
        if valid.any():
            out.near = float(result[valid].min())
            out.far = float(result[valid].max())
        out.metadata = {**(depth_base.metadata or {}),
                        "combined_mode": mode,
                        "combined_src_model": depth_detail_src.model_id}
        report = (f"AtlasDepthCombine: {mode} (strength {float(strength):.2f}) — base "
                  f"{depth_base.model_id} + source {depth_detail_src.model_id}.")
        if warnings:
            report += "\n" + "\n".join(warnings)
        return (out, report)


class AtlasGroundDepthMap:
    """
    Generate a ground-plane depth heatmap as an IMAGE tensor.
    Ports the GLSL DEPTH_FRAGMENT_SHADER (ProjectionMaterial.ts) to numpy:
    per-pixel ray cast → Y=0 intersection → warm-to-cool colormap.
    """
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_image", "ground_mask")
    FUNCTION = "generate"
    CATEGORY = "Atlas/advanced"

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
    CATEGORY = "Atlas/advanced"

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
    CATEGORY = "Atlas/advanced"

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

        # Canonicalize the sign so positive really is ABOVE the horizon (2026-07-27).
        # ax+by+c=0 describes the same line for (a,b,c) and (-a,-b,-c), and the
        # two producers disagree: the learned path emits (0, 1, -horizon_y)
        # (solver.py), for which `signed` grows DOWNWARD and this node returned
        # the ground as sky; the VP path builds its line from two vanishing
        # points (vanishing_points.line_from_points), whose sign flips with the
        # ORDER of those points, so its polarity was not even deterministic.
        # Image v runs downward, so forcing b <= 0 makes `signed` decrease with
        # v — positive above, which is what the docstring and the node's own
        # "Sky Mask" name promise. Nothing consumed this node (it was one of the
        # audit's zero-evidence nodes), so no saved graph depended on the old
        # inverted output.
        if b > 0.0:
            a, b, c = -a, -b, -c

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
    CATEGORY = "Atlas/advanced"

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
    CATEGORY = "Atlas/advanced"

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
    CATEGORY = "Atlas/advanced"

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
    # `report` APPENDED 2026-08-17. The no-focal guard returned three all-ZERO
    # masks and nothing else: hole_mask claimed no holes, occlusion_mask
    # claimed nothing occluded, and layer_mask made the band vanish from the
    # stack. Appended last, so existing slots keep their index.
    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("layer_mask", "occlusion_mask", "hole_mask", "report")
    FUNCTION = "generate"
    CATEGORY = "Atlas/advanced"

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
                "quad_coherence": ("BOOLEAN", {"default": True,
                    "tooltip": "Only used when compute_hole_mask=True. Match AtlasCleanPlateLayer "
                               "to keep hole QA identical to the final mesh."}),
            },
        }

    def generate(self, solve, depth, near_m=0.0, far_m=0.0, near_pct=0.0, far_pct=0.5, feather_px=4,
                 compute_hole_mask=False, relief_grid=384, depth_edge_rel=1.5, exclude_mask=None,
                 fill_occluded=False, band_side="manual", band_split=None, band_ref_mask=None,
                 band_override="", quad_coherence=True):
        np = _require_numpy()
        torch = _require_torch()

        setup = _metric_depth_and_validity(solve, depth, exclude_mask=exclude_mask)
        if setup is None:
            # layer/occlusion ZERO is honest — this band selects nothing and
            # nothing is occluded. hole_mask is ONES: no geometry was measured,
            # so nothing is covered. Zeros there would claim a flawless layer,
            # which is what AtlasPlanarHolePatch and the inpaint router read.
            h, w = int(depth.image_height), int(depth.image_width)
            zero = torch.zeros(1, h, w, dtype=torch.float32)
            return (zero, zero.clone(), torch.ones(1, h, w, dtype=torch.float32),
                    "SKIPPED — no metric depth (needs a solved focal length): no "
                    "band was measured, so layer_mask and occlusion_mask are "
                    "empty and hole_mask is all-ONES. Wire a solve carrying fx "
                    "(AtlasSolveFromImage / AtlasLearnedSolveFromImage).")
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
                apply_sky_heuristic=setup.exclude_mask is None,
                quad_coherence=bool(quad_coherence))
            # No edge overhang here, deliberately: the layer's mesh only
            # overhangs when embed_matte is on (this node can't know that),
            # and a PESSIMISTIC hole_mask (a couple of boundary cells extra)
            # over-inpaints safely, while an optimistic one under-inpaints.
            hole_mask_arr = mesh.hole_mask.astype(np.float32)

        if feather_px > 0 and occlusion_mask.any():
            # Clamped borders, not np.roll: wrapping would bleed occlusion from
            # one edge (a foreground object touching the bottom of the frame,
            # the common case) onto the opposite one.
            occlusion_mask = dilate(occlusion_mask, int(feather_px))

        layer_t = torch.from_numpy(layer_mask.astype(np.float32)).unsqueeze(0)
        occ_t = torch.from_numpy(occlusion_mask.astype(np.float32)).unsqueeze(0)
        hole_t = torch.from_numpy(hole_mask_arr).unsqueeze(0)
        return (layer_t, occ_t, hole_t,
                f"AtlasDepthLayerMask: layer {float(layer_t.mean()) * 100:.1f}% "
                f"of frame, occlusion {float(occ_t.mean()) * 100:.1f}%, "
                f"holes {float(hole_t.mean()) * 100:.1f}%")


class AtlasOutpaintDepth:
    """🪟 Extend a depth map to match an OUTPAINTED plate — geometry for the ring.

    `AtlasCleanPlateLayer.frame_outpaint_px` can already widen a layer past the
    frame edges, and its own tooltip calls the frame-edge reveal "the binding
    constraint on wide scenes". But the ring it adds is edge-replicated smear,
    and depth cannot follow it — `AtlasMogeNormals` refuses to run at all when
    `frame_outpaint_px != 0` because the normal map falls out of registration.
    The result is colour with no surface underneath: push the camera into that
    ring and there is nothing to project onto.

    This node closes that. Feed it the ORIGINAL depth and an ALREADY-WIDENED
    plate — outpainted by `AtlasSDXLInpaint`, by any generative node, or by
    hand — and it re-runs depth on the widened image and stitches the two.

    THE PROMPT LIVES UPSTREAM, deliberately. Whatever text conditioned the RGB
    outpaint is what shapes the new geometry, because the depth model reads the
    invented pixels. That keeps one generative step in the graph instead of two
    that could disagree about what is out there.

    WHY IT IS NOT A PASTE. A monocular model run on the widened image returns a
    DIFFERENT scale than the same model on the original — different framing,
    different content, different implied camera. Pasting the ring on directly
    puts a step at the frame boundary that reads as a wall of geometry. The
    widened depth is affine-fitted onto the original across the region they
    share first; the report prints the recovered scale, shift and residual so a
    bad fit is visible rather than silently baked in.

    The interior always keeps the ORIGINAL depth. It was estimated from real
    pixels; the widened pass saw invented ones and has no claim on it.

    `ring_mask` marks the invented region — the same contract as
    `extend_mask` / `{layer}_extend_matte.png`, so downstream regrain and
    matte work can treat it as suspect.
    """
    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "MASK", "STRING", "ATLAS_SOLVE")
    RETURN_NAMES = ("depth", "ring_mask", "report", "widened_solve")
    FUNCTION = "outpaint"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("ATLAS_DEPTH_MAP", {"tooltip":
                    "Depth for the ORIGINAL, un-widened plate."}),
                "widened_image": ("IMAGE", {"tooltip":
                    "The plate AFTER outpainting — larger than the original. The "
                    "padding is derived from the size difference."}),
            },
            "optional": {
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                     "tooltip": "Use the SAME model that produced the input depth. A "
                                "different one changes both scale and character, and the "
                                "affine fit can only absorb the scale."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "solve": ("ATLAS_SOLVE", {"tooltip":
                    "The solve for the ORIGINAL plate. Strongly recommended — without it "
                    "there is no `widened_solve` output, and every geometry node reads "
                    "width/cx/cy from the SOLVE rather than the depth map, so widened "
                    "depth on the original camera misregisters the new ring while looking "
                    "entirely plausible."}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4,
                    "tooltip": "Blend the new depth into the original across this many "
                               "pixels INSIDE the frame edge. 0 keeps every measured pixel "
                               "exactly and takes new depth only outside the frame. Raise it "
                               "only if a residual step is visible at the boundary — it "
                               "trades real measurement for a smoother join."}),
                "pad_override": ("STRING", {"default": "",
                    "tooltip": "Explicit 'left,top,right,bottom' padding in pixels. Leave "
                               "empty to split the size difference evenly, which is what a "
                               "symmetric outpaint produces."}),
            },
        }

    def outpaint(self, depth, widened_image,
                 depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                 device="auto", solve=None, feather_px=0, pad_override=""):
        import copy as _copy

        from atlas_camera.core.depth_outpaint import outpaint_depth
        from atlas_camera.inference.depth_estimator import estimate_depth

        np = _require_numpy()
        torch = _require_torch()

        base = np.asarray(depth.depth, dtype=np.float64)
        h, w = base.shape
        wide_h, wide_w = int(widened_image.shape[1]), int(widened_image.shape[2])

        if (wide_w, wide_h) == (w, h):
            raise ValueError(
                "AtlasOutpaintDepth: widened_image is the same size as the depth map "
                f"({w}x{h}) — there is no ring to fill. Wire the OUTPAINTED plate here, "
                "not the original.")
        if wide_w < w or wide_h < h:
            raise ValueError(
                f"AtlasOutpaintDepth: widened_image ({wide_w}x{wide_h}) is smaller than "
                f"the depth map ({w}x{h}). This node extends a plate, it does not crop one.")

        if str(pad_override).strip():
            try:
                pad = tuple(int(v) for v in str(pad_override).split(","))
                if len(pad) != 4:
                    raise ValueError
            except Exception:
                raise ValueError(
                    f"pad_override must be 'left,top,right,bottom', got {pad_override!r}")
        else:
            # Split the difference evenly; the remainder goes right/bottom so the
            # totals always reconstruct the widened size exactly.
            dx, dy = wide_w - w, wide_h - h
            pad = (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2)

        tmp = _save_image_tensor_to_tmp(widened_image)
        try:
            # The focal is a property of the LENS, not the crop, so it carries
            # over unchanged — the widened plate simply spans a wider angle at
            # the same focal length, which is what makes the ring meaningful.
            # fx UNSCALED, deliberately. `_solve_focal_px_for_image` rescales by
            # the image-width ratio, which is right for a RESIZED plate and wrong
            # for a widened one: outpainting adds pixels at the same angular
            # scale, so focal-in-pixels is unchanged. (It also crashes on a None
            # image, which is how this was caught.)
            focal = None
            if solve is not None:
                fx = getattr(solve.camera.intrinsics, "fx_px", None)
                focal = float(fx) if fx else None
            wide = estimate_depth(tmp, model_id=depth_model,
                                  device=None if device == "auto" else device,
                                  focal_px=focal)
        finally:
            os.unlink(tmp)

        res = outpaint_depth(base, np.asarray(wide.depth, dtype=np.float64),
                             pad=pad, feather_px=int(feather_px))

        out = _copy.copy(depth)
        out.depth = res.depth
        out.image_width = int(res.depth.shape[1])
        out.image_height = int(res.depth.shape[0])
        valid = np.isfinite(res.depth) & (res.depth > 0)
        out.near = float(res.depth[valid].min()) if valid.any() else 0.0
        out.far = float(res.depth[valid].max()) if valid.any() else 0.0
        meta = dict(getattr(depth, "metadata", {}) or {})
        meta["outpainted"] = dict(res.metadata, scale=res.scale, shift=res.shift,
                                  anchor_samples=res.anchor_samples,
                                  anchor_residual=res.anchor_residual)
        out.metadata = meta
        # The normal field belongs to the ORIGINAL frame and is now the wrong
        # size. Dropping it is deliberate: a silently mis-registered normal map
        # is exactly the failure AtlasMogeNormals refuses frame_outpaint_px over.
        dropped_normals = getattr(depth, "normal", None) is not None
        if dropped_normals:
            out.normal = None

        ring = torch.from_numpy(res.ring_mask.astype(np.float32)).unsqueeze(0)

        lines = [
            f"AtlasOutpaintDepth: {w}x{h} -> {out.image_width}x{out.image_height} "
            f"(pad l{pad[0]} t{pad[1]} r{pad[2]} b{pad[3]})",
            f"  ring is {res.metadata['ring_fraction'] * 100:.1f}% of the new frame "
            "— INVENTED geometry, from invented pixels",
        ]
        if res.metadata["anchored"]:
            lines.append(
                f"  anchored to the original: scale {res.scale:.4f}, shift {res.shift:+.4f} "
                f"on {res.anchor_samples} samples, residual {res.anchor_residual:.4f} m")
            if res.anchor_residual_rel > 0.02:
                lines.append(
                    f"  WARNING residual is {res.anchor_residual_rel * 100:.1f}% of scene "
                    "depth — the widened pass disagrees with the original about the part "
                    "they SHARE, so the ring is unlikely to be trustworthy.")
                lines.append(
                    "    Usual cause: the input depth was POST-PROCESSED after its model "
                    "ran (AtlasDepthDetailEnhance, AtlasDepthCombine) while this node "
                    "re-ran a raw pass. An affine fit cannot absorb that — branch the "
                    "outpaint off the RAW depth instead. A different depth_model does it "
                    "too.")
        else:
            lines.append(
                "  NOT anchored (too little valid overlap) — the ring keeps the widened "
                "pass's own scale and will probably step at the frame edge.")
        if dropped_normals:
            lines.append(
                "  predicted normals DROPPED: they belong to the original frame and "
                "would be mis-registered against the widened plate.")
        if int(feather_px) > 0:
            lines.append(
                f"  feathered {int(feather_px)} px inside the frame — that band is now a "
                "mixture, not pure measurement.")

        # The widened camera — the missing half of outpainting. Every geometry
        # node reads width/cx/cy from the SOLVE, not from the depth map, so a
        # widened depth on the original camera misregisters the new ring while
        # looking entirely plausible.
        widened_solve = None
        if solve is not None:
            from atlas_camera.core.depth_outpaint import widen_intrinsics
            widened_solve = _copy.deepcopy(solve)
            widened_solve.camera.intrinsics = widen_intrinsics(
                solve.camera.intrinsics, pad)
            widened_solve.image_width = out.image_width
            widened_solve.image_height = out.image_height
            wi, oi = widened_solve.camera.intrinsics, solve.camera.intrinsics
            lines.append(
                f"  widened camera: {wi.image_width}x{wi.image_height}, "
                f"cx {float(oi.cx_px or 0):.1f} -> {wi.cx_px:.1f}, "
                f"cy {float(oi.cy_px or 0):.1f} -> {wi.cy_px:.1f} "
                "(focal unchanged — the lens did not change, the frame did)")
            lines.append(
                "  USE `widened_solve` downstream, NOT the original: geometry nodes "
                "read width/cx/cy from the solve rather than the depth map.")
        else:
            lines.append(
                "  NO solve supplied, so no widened camera was produced. The depth is "
                "widened but nothing downstream knows it — wire the solve in.")

        return (out, ring, "\n".join(lines), widened_solve)


# ---------------------------------------------------------------------------
# Depth calibration — fit a model's characteristic error against MEASURED
# depth, store it per (model_id, scene_type), and apply it later to a plate
# where no measurement exists. Steps (2) and (4) of docs/ROADMAP.md.
#
# Nothing here applies automatically. `AtlasDepthMap` is untouched, so every
# existing graph behaves exactly as before; a calibration only ever reaches the
# depth chain because an artist wired `AtlasApplyDepthCalibration` into it.
# That is deliberate — `ATLAS_DEPTH_MAP` is consumed by nine node modules, and
# a stored coefficient silently rescaling shared depth would move geometry,
# bands, exports and the viewport at once, from a file the graph never names.


class AtlasFitDepthCalibration:
    """📐 Fit a depth model's error against MEASURED depth, and optionally store it.

    Wire `measured` from a real measurement — `AtlasLoadRecord3D` 📱 is the
    intended source, since an ARKit/LiDAR capture carries per-pixel metric
    truth — and `predicted` from `AtlasDepthMap` running the model you want to
    correct, on the SAME frame at the same resolution.

    The point is NOT to make the LiDAR capture better; it already carries
    measured metric depth and there is nothing to correct. The point is
    TRANSFER: learn how a model misreads a kind of scene where truth happens
    to be available, then apply that on an ordinary photograph where it is not.

    `save` is OFF by default. Fit, read the report, and only then decide —
    propose, never silently apply.
    """

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("correction_json", "report")
    FUNCTION = "fit"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        from atlas_camera.core.depth_calibration import MODELS
        from atlas_camera.core.depth_calibration_store import SCENE_TYPES
        return {
            "required": {
                "measured": ("ATLAS_DEPTH_MAP", {"tooltip":
                    "MEASURED depth — the truth side. AtlasLoadRecord3D's depth "
                    "output (ARKit/LiDAR, metres). Must be the same frame and "
                    "resolution as `predicted`."}),
                "predicted": ("ATLAS_DEPTH_MAP", {"tooltip":
                    "The model estimate to correct — AtlasDepthMap running the "
                    "model you want calibrated. Its model_id becomes half the "
                    "store key."}),
            },
            "optional": {
                "scene_type": (list(SCENE_TYPES), {"default": "outdoor",
                    "tooltip": "The other half of the store key. A correction is "
                               "only valid for the kind of scene it was fitted on; "
                               "lookup is an EXACT match and never falls back."}),
                "model": (["auto", *MODELS], {"default": "auto",
                    "tooltip": "'auto' fits all three and picks on HELD-OUT error "
                               "with a 5% margin, simplest-first among ties. Pick "
                               "one explicitly only if you know the sensor."}),
                "mask": ("MASK", {"tooltip":
                    "Restrict the fit to these pixels. Use the capture's "
                    "confidence_mask to drop low-confidence LiDAR returns."}),
                "store_path": ("STRING", {"default": "atlas_depth_calibration.json",
                    "tooltip": "Where the calibration store lives. Relative paths "
                               "resolve against ComfyUI's working directory."}),
                "save": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF by default. Fit and read the report first; turn "
                               "this on only once it looks right. Overwrites any "
                               "existing entry for this (model, scene_type)."}),
                "note": ("STRING", {"default": "",
                    "tooltip": "Free text stored beside the coefficients — which "
                               "capture, which lens, which day. You will want it."}),
            },
        }

    def fit(self, measured, predicted, scene_type="outdoor", model="auto",
            mask=None, store_path="atlas_depth_calibration.json", save=False,
            note=""):
        import json as _json

        from atlas_camera.core.depth_calibration import (
            choose_correction, fit_depth_correction)
        from atlas_camera.core.depth_calibration_store import CalibrationStore
        np = _require_numpy()

        lines = ["\U0001F4D0 Depth calibration fit"]
        model_id = getattr(predicted, "model_id", "") or "unknown"
        lines.append(f"  estimate: {model_id}")
        lines.append(f"  measured: {getattr(measured, 'model_id', '') or 'unknown'}"
                     f" (is_metric={getattr(measured, 'is_metric', None)})")

        if not getattr(measured, "is_metric", False):
            lines.append(
                "  ! the measured side is NOT flagged metric. A correction onto a "
                "relative depth map learns that map's arbitrary scale, not the "
                "world's — this is almost certainly the wrong input.")

        m_arr = np.asarray(measured.depth, dtype=np.float64)
        p_arr = np.asarray(predicted.depth, dtype=np.float64)
        if m_arr.shape != p_arr.shape:
            return ("", "\n".join(lines) + (
                f"\n  REFUSED: measured {m_arr.shape} and predicted {p_arr.shape} "
                "are different resolutions. Resample one onto the other — a fit "
                "across mismatched rasters correlates unrelated pixels."))

        mask_arr = None
        if mask is not None:
            mask_arr = np.asarray(
                mask.squeeze().cpu().numpy() if hasattr(mask, "cpu") else mask)
            if mask_arr.shape != p_arr.shape:
                return ("", "\n".join(lines) + (
                    f"\n  REFUSED: mask {mask_arr.shape} does not match the depth "
                    f"{p_arr.shape}."))

        try:
            if model == "auto":
                corr = choose_correction(p_arr, m_arr, mask=mask_arr)
            else:
                corr = fit_depth_correction(p_arr, m_arr, mask=mask_arr, model=model)
        except ValueError as exc:
            return ("", "\n".join(lines) + f"\n  REFUSED: {exc}")

        lo, hi = corr.predicted_range
        lines.append(f"  model:    {corr.model} (a={corr.a:.6f} b={corr.b:.6f})")
        lines.append(f"  samples:  {corr.n_samples}, predictions spanning "
                     f"{lo:.2f}-{hi:.2f} m ({corr.dynamic_range:.1f}x)")
        lines.append(f"  error:    {corr.mae_before:.4f} -> {corr.mae_after:.4f} m "
                     f"median ({corr.improvement:.0%} removed)")
        if "candidates" in corr.metadata:
            lines.append(f"  candidates: {corr.metadata['candidates']}")
        for key in ("selection", "selection_tie", "selection_warning",
                    "narrow_fit", "warning"):
            if key in corr.metadata:
                lines.append(f"  {key}: {corr.metadata[key]}")

        if corr.improvement <= 0:
            lines.append(
                "  ! this correction does not reduce the error. Storing it would "
                "make the depth chain slower and no better.")

        if save:
            store = CalibrationStore.load(store_path)
            existing = store.lookup(model_id, scene_type)
            store.put(model_id, scene_type, corr, note=note)
            written = store.save(store_path)
            lines.append(
                f"  SAVED to {written} under ({model_id}, {scene_type})"
                + ("  [replaced an existing entry]" if existing else ""))
        else:
            lines.append(
                "  not saved (save=False). Turn `save` on to store this under "
                f"({model_id}, {scene_type}).")

        return (_json.dumps(corr.to_dict(), indent=1), "\n".join(lines))


class AtlasApplyDepthCalibration:
    """📐 Apply a stored calibration to a depth map. Reports when it does not.

    Looks up (`depth.model_id`, `scene_type`) in the store. An EXACT match or
    nothing — a near-miss fallback is how a coefficient fitted on a 1.2 m
    interior wall ends up rescaling a 200 m exterior.

    A miss is a passthrough and SAYS SO. A silent branch-skip that returns the
    input unchanged is indistinguishable from a calibration that had no effect,
    and the artist needs to know which of those happened.
    """

    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "STRING")
    RETURN_NAMES = ("depth", "report")
    FUNCTION = "apply"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        from atlas_camera.core.depth_calibration_store import SCENE_TYPES
        return {
            "required": {"depth": ("ATLAS_DEPTH_MAP",)},
            "optional": {
                "scene_type": (list(SCENE_TYPES), {"default": "outdoor",
                    "tooltip": "Must match the scene_type the correction was fitted "
                               "and stored under. Lookup is exact."}),
                "store_path": ("STRING", {"default": "atlas_depth_calibration.json",
                    "tooltip": "The calibration store written by "
                               "AtlasFitDepthCalibration."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Off = passthrough, stated in the report. Use it to "
                               "A/B a calibration without rewiring."}),
                "on_extrapolation": (["report", "nan"], {"default": "report",
                    "tooltip": "'report' applies everywhere and says how far outside "
                               "the fitted range it went. 'nan' voids out-of-range "
                               "samples instead — never a clamp, which would return "
                               "confident depth the fit has no evidence for."}),
            },
        }

    def apply(self, depth, scene_type="outdoor",
              store_path="atlas_depth_calibration.json", enabled=True,
              on_extrapolation="report"):
        from atlas_camera.core.depth_calibration import apply_depth_correction
        from atlas_camera.core.depth_calibration_store import CalibrationStore

        model_id = getattr(depth, "model_id", "") or "unknown"
        lines = ["\U0001F4D0 Depth calibration"]

        if not enabled:
            return (depth, "\n".join(lines) + "\n  disabled — depth passed through "
                    "unchanged, no calibration was looked up.")

        try:
            store = CalibrationStore.load(store_path)
        except (OSError, ValueError) as exc:
            return (depth, "\n".join(lines) + (
                f"\n  store at {store_path} could not be read ({exc}). Depth passed "
                "through UNCALIBRATED."))

        corr = store.lookup(model_id, scene_type)
        if corr is None:
            return (depth, "\n".join(lines) + "\n" + (
                f"  no calibration for ({model_id}, {scene_type}).\n"
                f"  store: {store_path} — {store.describe()}\n"
                "  Depth passed through UNCALIBRATED. Fit one with "
                "AtlasFitDepthCalibration against a measured capture."))

        corrected, report = apply_depth_correction(
            depth.depth, corr, on_extrapolation=on_extrapolation)

        out = copy.copy(depth)
        out.depth = corrected
        out.metadata = dict(getattr(depth, "metadata", {}) or {})
        out.metadata["depth_calibration"] = {
            "model_id": model_id, "scene_type": scene_type,
            "correction": corr.to_dict(), "application": report.to_dict(),
        }

        lo, hi = corr.predicted_range
        lines.append(f"  applied ({model_id}, {scene_type}): {corr.model} "
                     f"a={corr.a:.6f} b={corr.b:.6f}")
        lines.append(f"  fitted on {corr.n_samples} samples spanning "
                     f"{lo:.2f}-{hi:.2f} m, improvement {corr.improvement:.0%}")
        lines.append(f"  {report}")
        note = store.note_for(model_id, scene_type)
        if note:
            lines.append(f"  note: {note}")
        return (out, "\n".join(lines))
