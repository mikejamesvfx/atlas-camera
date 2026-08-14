"""Two-pass occlusion fill: structure and texture are SEPARATE jobs.

Measured 2026-08-14 (docs/dev/occlusion_arms_2026-08-14): WAN VACE invents the
best masked GEOMETRY of every generator tested (unmasked delta 0.010-0.014)
but carries a crosshatch texture that more sampling steps make WORSE; LTX and
SDXL produce clean texture over softer structure. A single-frame fill with a
matte needs a video model for NEITHER half, so:

    pass 1  WAN VACE fills the hole            (structure; texture ignored)
    gate    the structure is SCORED before it is ever made to look convincing
    pass 2  SDXL latent img2img over the hole  (texture; structure locked at
                                                low denoise)

The inter-pass gate is the load-bearing part. Pass 2's whole job is making
pixels plausible — fed a broken pass-1 it launders garbage into confident
fiction, which is strictly worse than an honest hole. A gated failure falls
back to the one-pass LTX path rather than proceeding.

Layering: dynamic/ may import core, never comfy. GPU work goes through the
template-driven generator adapter; everything here is orchestration, scoring
and guard-rails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.dynamic.fill_metrics import (
    G2_MIN_EDGE_EXTEND_DIFF,
    score_fill,
)
from atlas_camera.dynamic.occlusion_fill import _require_deps

#: Maximum tolerated global misregistration of the pass-1 fill against the
#: guide, in pixels at the generation raster (phase correlation over the
#: unmasked area). The tiled-sampler and VACE arms both measured 0.0; the
#: E6 near-miss showed a visual "shift" read needs the measurement, so the
#: gate carries it rather than trusting eyes.
MAX_INTERPASS_SHIFT_PX = 2.0

#: Fraction of hole pixels still reading as the LTX chroma-green sentinel.
#: WAN has NO sentinel convention — at CFG > 1 it treats green guide content
#: as content to preserve (measured: sentinel visibly bled through at cfg 6).
MAX_SENTINEL_BLEED_FRAC = 0.005

_SENTINEL_RGB = (102, 255, 0)


@dataclass(slots=True)
class InterpassVerdict:
    """The gate's decision plus every number it was based on."""

    ok: bool
    reasons: list = field(default_factory=list)
    g2: float = 0.0
    shift_px: float = 0.0
    sentinel_bleed_frac: float = 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons),
                "g2": self.g2, "shift_px": self.shift_px,
                "sentinel_bleed_frac": self.sentinel_bleed_frac}


def interpass_gate(fill, guide, hole) -> InterpassVerdict:
    """Score a pass-1 (structure) fill BEFORE it is re-textured.

    Three checks, each of which caught a real failure this pipeline produced:

    - G2 vs the edge-extend smear: a fill indistinguishable from the
      deterministic null is not worth a second model's polish.
    - Global shift by phase correlation over the UNMASKED area: a displaced
      fill composites misregistered no matter how good it looks.
    - Sentinel bleed: chroma-green surviving in the hole means the generator
      treated the mask contract as content (WAN at CFG > 1 did exactly this).
    """
    np, _ = _require_deps()
    mask = np.asarray(hole)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > (127 if mask.dtype == np.uint8 else 0.5)

    scores = score_fill(fill, guide, mask)
    verdict = InterpassVerdict(ok=True, g2=scores["mean_abs_vs_edge_extend"])
    if not scores["g2_pass"]:
        verdict.ok = False
        verdict.reasons.append(
            f"G2 {verdict.g2:.4f} <= {G2_MIN_EDGE_EXTEND_DIFF:.4f} — "
            f"indistinguishable from an edge-extend smear")

    sentinel = np.asarray(_SENTINEL_RGB, dtype=np.float64)
    f = np.asarray(fill, dtype=np.float64)
    if f.max() <= 1.0:
        f = f * 255.0
    bleed = np.abs(f[mask][:, :3] - sentinel).max(axis=1) < 40
    verdict.sentinel_bleed_frac = float(bleed.mean()) if bleed.size else 0.0
    if verdict.sentinel_bleed_frac > MAX_SENTINEL_BLEED_FRAC:
        verdict.ok = False
        verdict.reasons.append(
            f"sentinel bleed {verdict.sentinel_bleed_frac:.2%} > "
            f"{MAX_SENTINEL_BLEED_FRAC:.2%} — the generator preserved the "
            f"mask sentinel as content")

    try:
        import cv2

        keep = ~mask
        gl = np.asarray(_grey(np, guide), dtype=np.float32)
        fl = np.asarray(_grey(np, fill), dtype=np.float32)
        # neutralize the hole so only shared content votes
        fl2 = fl.copy()
        fl2[mask] = gl[mask]
        (dx, dy), _resp = cv2.phaseCorrelate(gl, fl2)
        verdict.shift_px = float(max(abs(dx), abs(dy)))
        if verdict.shift_px > MAX_INTERPASS_SHIFT_PX:
            verdict.ok = False
            verdict.reasons.append(
                f"global shift {verdict.shift_px:.1f}px > "
                f"{MAX_INTERPASS_SHIFT_PX}px — fill is misregistered")
    except ImportError:
        # cv2 is part of [vision]; without it the shift check is skipped
        # LOUDLY rather than silently passed.
        verdict.reasons.append("shift check skipped: opencv unavailable")
    return verdict


def _grey(np, img):
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    if a.max() <= 1.0:
        a = a * 255.0
    return a


# ---------------------------------------------------------------------------
# Template guard-rails — each one is a trap that cost a live run.

def load_template_guarded(path) -> dict:
    """Load a generator template and refuse the known template-layer traps.

    The adapter's marker substitution unicode-escapes template text, so a
    model path like ``SDXL\\albedobase...`` reaches the server with ``\\a`` as
    BEL (0x07) unless the backslash is doubled in the file. A control
    character in any string input is that bug — name it instead of shipping
    it to the server as a mangled path.
    """
    raw = Path(path).read_text(encoding="utf-8")
    template = json.loads(raw)
    for nid, node in template.items():
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, str):
                bad = [c for c in value if ord(c) < 0x20 and c not in "\n\t"]
                if bad:
                    raise ValueError(
                        f"template {Path(path).name} node {nid}.{key} contains "
                        f"control character {bad[0]!r} — a model path with a "
                        f"backslash before an escapable letter must be "
                        f"DOUBLE-backslashed (the adapter unicode-escapes "
                        f"template text)")
    return template


def check_wan_template(template: dict) -> list:
    """WAN-specific MUSTs. Returns problem strings (empty = fine).

    - CFG must be 1 with the CausVid speed LoRA: at CFG 6 the sentinel bled
      through into the fill (WAN has no sentinel convention).
    - length must satisfy 4k+1 (VACE latent packing).
    - WanVaceToVideo carries its OWN raster; the caller must pass those dims
      as the generation config or the adapter stomps them with the ROI
      raster (the --gen-size trap).
    """
    problems = []
    for nid, node in template.items():
        ct = node.get("class_type")
        if ct == "KSampler":
            cfg = node.get("inputs", {}).get("cfg", 1.0)
            if isinstance(cfg, (int, float)) and cfg > 1.5:
                problems.append(
                    f"node {nid}: KSampler cfg={cfg} — WAN preserves the "
                    f"sentinel as content above cfg 1 (measured at cfg 6)")
        if ct == "WanVaceToVideo":
            length = node.get("inputs", {}).get("length", 0)
            if isinstance(length, int) and length % 4 != 1:
                problems.append(
                    f"node {nid}: length={length} violates 4k+1")
    return problems


def wan_generation_raster(template: dict) -> tuple:
    """The raster the WAN graph declares — pass THIS as the generation size.

    ``WanVaceToVideo`` exposes bare ``width``/``height`` ints, and the
    adapter's config-resize pushes the fixture raster into every such widget;
    reading the template's own values and echoing them back is the guard.
    """
    for node in template.values():
        if node.get("class_type") == "WanVaceToVideo":
            inputs = node.get("inputs", {})
            w, h = inputs.get("width"), inputs.get("height")
            if isinstance(w, int) and isinstance(h, int):
                return w, h
    raise ValueError("template has no WanVaceToVideo with literal width/height")


# ---------------------------------------------------------------------------
# Orchestration

def run_two_pass_fill(generator_factory, roi_dir, guide, mask, *,
                      wan_template, sdxl_template, prompt: str,
                      seed: int = 7, host: str = "127.0.0.1:8188") -> dict:
    """WAN structure -> gate -> SDXL texture. Returns a result dict.

    ``generator_factory()`` returns a fresh template-driven generator (the
    LTX ComfyUI adapter drives any graph via upload markers). Both passes use
    the SAME seed and the SAME content prompt — the policies the measured
    winning run used. On a gate failure the caller falls back to the one-pass
    engine; this function never re-textures a failed structure.
    """
    np, Image = _require_deps()
    roi_dir = Path(roi_dir)
    hole = np.asarray(mask)
    if hole.ndim == 3:
        hole = hole[..., 0]
    hole = hole > (127 if hole.dtype == np.uint8 else 0.5)
    if not bool(hole.any()):
        # A region with no disocclusion has nothing to invent — running
        # either pass would only re-render real pixels (the artist mis-click
        # case; found live when the gate crashed on an empty mask).
        return {"status": "empty_hole"}
    h, w = np.asarray(guide).shape[:2]

    wan_tmpl = load_template_guarded(wan_template)
    problems = check_wan_template(wan_tmpl)
    if problems:
        return {"status": "invalid_wan_template", "problems": problems}
    gen_w, gen_h = wan_generation_raster(wan_tmpl)
    sdxl_tmpl = load_template_guarded(sdxl_template)

    # ---- pass 1: WAN structure
    result1 = _run_template(generator_factory, roi_dir / "pass1", wan_template,
                            guide, mask, prompt, seed, host,
                            gen_w=gen_w, gen_h=gen_h)
    if result1.get("status") != "ok":
        return {"status": "pass1_failed", **result1}
    fill1 = _load_fill(np, Image, result1["frame"], w, h)

    from atlas_camera.core.camera_crop import (
        match_reference_colour,
        membrane_blend,
        neutralize_fill_cast,
    )
    fill1 = neutralize_fill_cast(
        match_reference_colour(fill1, guide, hole), hole,
        reference=guide, band_px=48)
    fill1 = membrane_blend(fill1, guide, hole)

    verdict = interpass_gate(fill1, guide, hole)
    if not verdict.ok:
        return {"status": "gate_failed", "gate": verdict.to_dict()}

    # ---- pass 2: SDXL texture over the WAN-structured composite
    composite = np.asarray(guide).copy()
    composite[hole] = np.asarray(fill1)[hole]
    # SDXL runs at a bounded raster; its template resizes internally
    result2 = _run_template(generator_factory, roi_dir / "pass2",
                            sdxl_template, composite, mask, prompt, seed, host,
                            gen_w=None, gen_h=None)
    if result2.get("status") != "ok":
        return {"status": "pass2_failed", "gate": verdict.to_dict(), **result2}
    fill2 = _load_fill(np, Image, result2["frame"], w, h)
    fill2 = neutralize_fill_cast(
        match_reference_colour(fill2, composite, hole), hole,
        reference=composite, band_px=48)
    fill2 = membrane_blend(fill2, guide, hole)
    return {"status": "ok", "gate": verdict.to_dict(), "fill": fill2,
            "pass1_frame": result1["frame"], "pass2_frame": result2["frame"]}


def _run_template(generator_factory, work_dir, template_path, image, mask,
                  prompt, seed, host, *, gen_w, gen_h) -> dict:
    np, Image = _require_deps()
    from atlas_camera.dynamic.generators import TemporalGenerationConfig

    work_dir = Path(work_dir)
    for sub in ("guide", "mask", "source"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    img = np.asarray(image)
    Image.fromarray(img).save(work_dir / "guide" / "frame_0000.png")
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = (m * 255).astype(np.uint8) if m.max() <= 1.0 else m.astype(np.uint8)
    Image.fromarray(m, mode="L").save(work_dir / "mask" / "frame_0000.png")
    Image.fromarray(img).save(work_dir / "source" / "crop.png")

    generator = generator_factory()
    generator.host = host
    generator.template_path = str(template_path)
    # stale-mp4 trap: the runner skips encoding when the file exists
    for p in (work_dir / "guide.mp4", work_dir / "mask.mp4"):
        if p.exists():
            p.unlink()
    # 1-frame clips encode at 1 fps or the silent AAC track rounds to zero
    # samples and LTX-family AV graphs die in VAEEncodeAudio
    err = generator._encode_rendered_mp4(work_dir / "guide",
                                         work_dir / "guide.mp4", 1.0)
    err = err or generator._encode_rendered_mp4(work_dir / "mask",
                                                work_dir / "mask.mp4", 1.0)
    if err:
        return {"status": "encode_failed", "error": str(err)}

    class _PseudoPlate:
        source_roi = None
        crop_camera = None

    h, w = img.shape[:2]
    config = TemporalGenerationConfig(
        prompt=prompt, seed=seed, fps=1.0, frame_count=1,
        width=(gen_w or w), height=(gen_h or h),
        extra={"upload_markers": {"{GUIDE_VIDEO}": work_dir / "guide.mp4",
                                  "{MASK_VIDEO}": work_dir / "mask.mp4"}})
    result = generator.generate(_PseudoPlate(), work_dir, config)
    if result.status != "ok":
        return {"status": str(result.status),
                "warnings": [str(x) for x in result.warnings]}
    frames = sorted((work_dir / "generated").glob("frame_*.png"))
    if not frames:
        return {"status": "no_frames"}
    return {"status": "ok", "frame": str(frames[-1])}


def _load_fill(np, Image, path, width, height):
    """Generated frame -> RGB array at the ROI raster (Lanczos if resampled)."""
    with Image.open(path) as im:
        fill = np.asarray(im.convert("RGB"))
    if fill.shape[:2] != (height, width):
        fill = np.asarray(Image.fromarray(fill).resize((width, height),
                                                       Image.LANCZOS))
    return fill
