"""Atlas ComfyUI nodes — depth group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
from __future__ import annotations

import copy
import math
import os

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


class AtlasDepthOutlierMask:
    """Build an explicit mask for local monocular-depth outliers."""
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("outlier_mask", "report")
    FUNCTION = "detect"
    CATEGORY = "Atlas Camera/Derive Geometry"

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
                apply_sky_heuristic=setup.exclude_mask is None,
                quad_coherence=bool(quad_coherence))
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
