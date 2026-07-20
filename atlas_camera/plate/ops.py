"""Pixel operations on plates — the numpy side of the plate package.

The final move of `docs/dev/node_helpers_layering_plan.md`, deferred until
`atlas_camera/plate/` existed (it arrived with the OpenImageIO work, PR #24).

Both of these are pure numpy and host-agnostic, so living in the ComfyUI
adapter was the layering violation this refactor set out to fix. They are not
generic "image ops" though — they are things you do to a PLATE before it is
projected, which is why they sit beside `plate/oiio_io.py` rather than in a
vaguely-named `core/image_ops.py`.

`_extend_edge_colors` is the classic Nuke premult -> dilate edge-extend: for a
narrow disocclusion sliver of smooth gradient (sky, most often) it is
indistinguishable from an inpaint at a fraction of the cost, and it is
deterministic, which an inpaint is not. Propagation runs at quarter
resolution because the content it smears is low-frequency by definition.

`_flood_mask_to_frame_borders` exists because of a bug found live: a faded
segmentation border row carried sky depth into a frame-outpaint ring and built
a floating ring at the top of frame (86% of one layer's above-skyline vertices
landed in the ring before this). Flooding an exclusion mask out to the frame
edge before padding is what stops it.

Kept importable from `comfy/node_helpers.py` by re-export, so nothing that
uses them had to change.
"""

from __future__ import annotations


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of the package
        raise ImportError("plate pixel ops need numpy — pip install -e .[vision]") from exc
    return np


# Segmentation masks (SAM and similar) systematically FADE at frame borders
# (measured live: SAM3 sky coverage 5.9% at row 0, 29% at row 5, 100% by row
# 50 on a clear-sky plate) — so any border-touching mechanism (frame
# outpaint's edge replication, the sky card's own ring) sees a boundary row
# that lies about its content. Floods within this margin of each border.
_BORDER_FLOOD_PX = 64
def _flood_mask_to_frame_borders(mask, margin_px=_BORDER_FLOOD_PX):
    """Flood a boolean mask to the frame borders wherever it touches within
    ``margin_px``: a column with sky at row 40 is sky at rows 0-39 too (the
    segmenter faded, physics didn't). Content genuinely cut by the frame (a
    spire reaching the top edge) has no mask in the margin and is untouched.
    Applied per border, perpendicular fill only."""
    np = _require_numpy()
    m = np.asarray(mask, dtype=bool).copy()
    k = int(margin_px)
    if k <= 0 or not m.any():
        return m
    # top: propagate True upward within the margin slice
    m[:k] |= np.flip(np.logical_or.accumulate(np.flip(m[:k], axis=0), axis=0), axis=0)
    # bottom
    m[-k:] |= np.logical_or.accumulate(m[-k:], axis=0)
    # left
    m[:, :k] |= np.flip(np.logical_or.accumulate(np.flip(m[:, :k], axis=1), axis=1), axis=1)
    # right
    m[:, -k:] |= np.logical_or.accumulate(m[:, -k:], axis=1)
    return m
def _extend_edge_colors(rgb, valid, px):
    """Deterministic edge-extend (the classic Nuke premult->dilate trick):
    push ``valid`` pixels' colors outward into the invalid region by ``px``
    pixels via iterative neighbor-mean propagation. Returns (rgb, mask) with
    the extension applied and the validity dilated to match.

    Runs the propagation at quarter resolution (sky is low-frequency; a
    smeared gradient is exactly the desired look) and composites the
    extension back only where the original was invalid - original pixels are
    never touched. Pure numpy, no scipy/cv2. This is deliberately NOT an
    inpaint: for narrow disocclusion slivers of smooth sky it is
    indistinguishable from one at a fraction of the cost; large structured
    reveals (clouds behind a building) still want the LaMa/inpaint chain on
    plate_image instead.
    """
    np = _require_numpy()
    rgb = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    H, W = valid.shape
    ds = 4
    small = rgb[::ds, ::ds].copy()
    v = valid[::ds, ::ds].copy()
    steps = max(1, int(round(px / ds)))
    for _ in range(steps):
        acc = np.zeros_like(small)
        cnt = np.zeros(v.shape, dtype=np.float32)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.zeros_like(small)
            shv = np.zeros_like(v)
            if dr == 1:
                sh[1:], shv[1:] = small[:-1], v[:-1]
            elif dr == -1:
                sh[:-1], shv[:-1] = small[1:], v[1:]
            elif dc == 1:
                sh[:, 1:], shv[:, 1:] = small[:, :-1], v[:, :-1]
            else:
                sh[:, :-1], shv[:, :-1] = small[:, 1:], v[:, 1:]
            acc += np.where(shv[..., None], sh, 0.0)
            cnt += shv
        newly = ~v & (cnt > 0)
        if not newly.any():
            break
        small[newly] = acc[newly] / cnt[newly, None]
        v |= newly
    # Upsample the extension and composite only into originally-invalid pixels.
    up = np.repeat(np.repeat(small, ds, axis=0), ds, axis=1)[:H, :W]
    upv = np.repeat(np.repeat(v, ds, axis=0), ds, axis=1)[:H, :W]
    out = rgb.copy()
    fill = ~valid & upv
    out[fill] = up[fill]
    return out, valid | fill

__all__ = [
    "_BORDER_FLOOD_PX",
    "_flood_mask_to_frame_borders",
    "_extend_edge_colors",
]
