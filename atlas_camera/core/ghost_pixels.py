"""Per-pixel VALID / GHOST / INVALID classification of a rendered target view.

WHY THIS MODULE EXISTS. Atlas already computed all three classes and already
knew they were different things — `AtlasDisocclusionGuide`'s docstring says so
outright ("'UNCOVERED' IS NOT THE SAME AS 'DISOCCLUDED', and the report
separates them"), and `dynamic.occlusion_fill.not_disocclusion_mask` is the
test that separates them. What was missing is a per-pixel MAP and a structured
count. The split was reported as sentences in a STRING, which is how the
occlusion-fill doctrine ended up recording a raw "peak hole" that climbed
64.8% -> 86.1% ON AN IMPROVING RUN: the number counted sky, and prose does not
subtract.

THE THREE CLASSES ARE NOT INTERCHANGEABLE. Collapsing them is the whole bug
this module exists to prevent:

- **VALID** — the rasteriser covered this pixel. We know its appearance.
- **GHOST** — geometry says a surface is here and the plate never observed it,
  because something opaque stood in front. This is the only class a generator
  should be pointed at.
- **INVALID** — the plate has no geometry here either (sky, or a partial
  relief mesh). Nothing was occluding it, so there is nothing to dis-occlude;
  inventing pixels here papers over an upstream gap.
- **OUT_OF_BOUNDS** — the target camera swung past the edge of the photograph.
  That is OUTPAINTING, a different job with different nodes, and it must not
  consume a fill budget.

This module is the ALGEBRA only: it takes boolean fields that a caller has
already measured and combines them. It deliberately does not render, does not
reproject and does not import `dynamic` — `core` may not (the layering runs
one way). `comfy.nodes_viewport.AtlasGhostPixelMap` is the composer that
supplies these fields from `projection_render.render_scene`,
`occlusion_fill.plate_hole_survey` and `occlusion_fill.reproject_at_infinity`.

Host-agnostic: numpy only, no torch, no ComfyUI.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

from atlas_camera.core.hole_field import components


class GhostClass(IntEnum):
    """Per-pixel class. Values are a stable contract — append, never renumber.

    `VISIBILITY_CONFLICT` is defined and never emitted: the mesh rasteriser
    resolves depth disagreement internally with a true per-pixel z-test, so
    there is no conflict left to report. The slot is reserved rather than
    faked, so a future multi-source fusion can fill it without an API break.
    """

    INVALID = 0
    VALID = 1
    GHOST = 2
    OUT_OF_BOUNDS = 3
    LOW_CONFIDENCE = 4
    VISIBILITY_CONFLICT = 5


#: Overlay colours, RGB floats in [0, 1]. Magenta for GHOST matches the
#: sentinel `AtlasDisocclusionGuide` paints uncovered pixels with by default,
#: so the two nodes read the same way side by side.
CLASS_COLOURS: dict[int, tuple[float, float, float]] = {
    GhostClass.INVALID: (1.0, 0.0, 0.0),            # red
    GhostClass.VALID: (0.0, 1.0, 0.0),              # green
    GhostClass.GHOST: (1.0, 0.0, 1.0),              # magenta
    GhostClass.OUT_OF_BOUNDS: (0.0, 0.35, 1.0),     # blue
    GhostClass.LOW_CONFIDENCE: (1.0, 1.0, 0.0),     # yellow
    GhostClass.VISIBILITY_CONFLICT: (1.0, 0.5, 0.0),  # orange
}

#: Default overlay strength. Below 1.0 so the reprojected RGB stays readable
#: underneath — the overlay is for judging geometry, not for replacing it.
DEFAULT_OVERLAY_ALPHA = 0.55


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.ghost_pixels requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def classify_target_coverage(*, covered: Any, inside_plate_frame: Any,
                             plate_hole_at_target: Any,
                             exclude: Any = None) -> Any:
    """Combine measured boolean fields into one `GhostClass` map.

    Every argument is a HxW boolean field measured by the caller:

    - `covered` — the rasteriser produced content (alpha > 0). VALID.
    - `inside_plate_frame` — this target pixel's ray, taken to infinity, lands
      inside the plate's raster. False means the camera looked somewhere the
      photograph never covered.
    - `plate_hole_at_target` — the plate was ALSO a hole where this pixel's
      ray lands. Sampled through the cameras at infinity rather than
      pixel-for-pixel, because two views differing by a rotation do not see
      the sky at the same pixels.
    - `exclude` — optional artist declaration ("never target this"). It wins
      over every geometric class, because it is a statement about intent, not
      about geometry; it must never be reported as a factual disocclusion.

    Precedence is deliberate and total: exclude > covered > out-of-bounds >
    plate-hole > ghost. GHOST is what SURVIVES every other explanation, which
    is the conservative direction the classifier is required to fail in.
    """
    np = _require_numpy()
    covered = np.asarray(covered, dtype=bool)
    inside = np.asarray(inside_plate_frame, dtype=bool)
    plate_hole = np.asarray(plate_hole_at_target, dtype=bool)

    for name, arr in (("inside_plate_frame", inside),
                      ("plate_hole_at_target", plate_hole)):
        if arr.shape != covered.shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match covered "
                f"{covered.shape}; every field must be the target raster")

    out = np.full(covered.shape, int(GhostClass.GHOST), dtype=np.uint8)
    out[~inside] = int(GhostClass.OUT_OF_BOUNDS)
    out[inside & plate_hole] = int(GhostClass.INVALID)
    out[covered] = int(GhostClass.VALID)

    if exclude is not None:
        ex = np.asarray(exclude, dtype=bool)
        if ex.shape != covered.shape:
            raise ValueError(
                f"exclude shape {ex.shape} does not match covered "
                f"{covered.shape}")
        out[ex] = int(GhostClass.LOW_CONFIDENCE)
    return out


def class_masks(class_map: Any) -> dict[str, Any]:
    """Split a class map into the named boolean masks the node returns."""
    np = _require_numpy()
    cm = np.asarray(class_map)
    return {
        "valid": cm == int(GhostClass.VALID),
        "ghost": cm == int(GhostClass.GHOST),
        "invalid": cm == int(GhostClass.INVALID),
        "out_of_bounds": cm == int(GhostClass.OUT_OF_BOUNDS),
        "low_confidence": cm == int(GhostClass.LOW_CONFIDENCE),
    }


def ghost_stats(class_map: Any) -> dict[str, Any]:
    """Counts, fractions and ghost-region shape from a class map.

    FRACTIONS EXCLUDE THE EXCLUDED, from numerator AND denominator. This
    mirrors `move_budget.disocclusion_fraction`'s `ignore_mask` rule, which
    exists because without it sky reads as a permanent hole and every budget
    collapses to zero. `total_px` is the raster; `scored_px` is what the
    fractions are actually over, and both are reported so a reader can tell.
    """
    np = _require_numpy()
    cm = np.asarray(class_map)
    masks = class_masks(cm)

    total = int(cm.size)
    excluded = int(masks["low_confidence"].sum())
    scored = total - excluded

    counts = {f"{name}_pixel_count": int(m.sum()) for name, m in masks.items()}
    stats: dict[str, Any] = {
        "total_px": total,
        "scored_px": scored,
        **counts,
    }
    for name in ("valid", "ghost", "invalid", "out_of_bounds"):
        stats[f"{name}_fraction"] = (
            float(masks[name].sum()) / scored if scored else 0.0)
    stats["excluded_fraction"] = float(excluded) / total if total else 0.0

    islands = components(masks["ghost"], connectivity=4)
    stats["ghost_region_count"] = len(islands)
    stats["largest_ghost_region_px"] = (
        max((len(c) for c in islands), default=0))
    return stats


def build_debug_overlay(rgb: Any, class_map: Any, *,
                        alpha: float = DEFAULT_OVERLAY_ALPHA) -> Any:
    """Tint the reprojected RGB by class, keeping the image readable beneath.

    Returns HxWx3 float32 in [0, 1]. VALID is tinted too, so a fully-working
    frame is unmistakably green rather than merely un-marked — an all-black
    overlay and a perfect render must not look alike.
    """
    np = _require_numpy()
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"rgb must be HxWx3, got shape {img.shape}")
    img = img[..., :3]
    cm = np.asarray(class_map)
    if cm.shape != img.shape[:2]:
        raise ValueError(
            f"class_map shape {cm.shape} does not match image "
            f"{img.shape[:2]}")
    if img.max() > 1.0 + 1e-6:      # tolerate a uint8-valued float array
        img = img / 255.0

    a = float(np.clip(alpha, 0.0, 1.0))
    out = img.astype(np.float32).copy()
    for value, colour in CLASS_COLOURS.items():
        sel = cm == int(value)
        if not sel.any():
            continue
        tint = np.asarray(colour, dtype=np.float32)
        out[sel] = (1.0 - a) * out[sel] + a * tint
    return np.clip(out, 0.0, 1.0)
