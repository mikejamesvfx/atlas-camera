"""Nuke-style grade for plates — the numpy side (float-safe by construction).

Matching an AI patch / clean plate to the source plate is the most common
pre-projection fix, and round-tripping to Nuke for a two-knob grade is
overkill. This is the classic lift/gamma/gain in scene-linear:

    graded = (in * (gain - lift) + lift) ** (1 / gamma)

with saturation as a Rec.709-luma lerp and ``mix`` as the final dry/wet.

Float-safety rules (the same doctrine as ``plate/deband.py``):
  * never clamps highlights — >1.0 scene-linear values ride through the
    linear stage untouched and through gamma as plain ``pow``;
  * negative inputs (rare but legal in linear) skip the ``pow`` (which would
    be NaN) and take the linear stage only;
  * ``mix=0`` and the identity knobs (0/1/1/1) are an EXACT no-op.

Lives in ``plate/`` beside deband: a thing you do to a PLATE before it is
projected. IMAGE-only at the node layer — cannot touch an ATLAS_PLATE_REF,
so a graded tensor can never masquerade as the registered final plate.
"""

from __future__ import annotations


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of the package
        raise ImportError("plate grade needs numpy — pip install -e .[vision]") from exc
    return np


# Rec.709 luma weights — the same constants headless_evidence uses for its
# structural comparison; duplicated here so plate/ stays free of comfy imports.
_LUMA = (0.2126, 0.7152, 0.0722)


def grade_plate(
    image,
    *,
    lift: float = 0.0,
    gamma: float = 1.0,
    gain: float = 1.0,
    saturation: float = 1.0,
    mix: float = 1.0,
):
    """Grade a HxWxC (or HxW) float plate. Returns float32, same shape."""
    np = _require_numpy()
    img = np.asarray(image, dtype=np.float64)
    m = float(np.clip(mix, 0.0, 1.0))
    identity = (lift == 0.0 and gamma == 1.0 and gain == 1.0 and saturation == 1.0)
    if m == 0.0 or identity or img.size == 0:
        return img.astype(np.float32)

    chan = img if img.ndim == 3 else img[..., None]
    out = chan[..., :3].copy() if chan.shape[-1] >= 3 else chan.copy()

    lf, gm, gn = float(lift), float(gamma), float(gain)
    linear = out * (gn - lf) + lf
    if gm != 1.0 and gm > 1e-6:
        inv_g = 1.0 / gm
        pos = linear > 0
        linear = np.where(pos, np.power(np.where(pos, linear, 1.0), inv_g), linear)
    out = linear

    if saturation != 1.0 and out.shape[-1] >= 3:
        luma = (out[..., 0] * _LUMA[0] + out[..., 1] * _LUMA[1]
                + out[..., 2] * _LUMA[2])[..., None]
        out = luma + float(saturation) * (out - luma)

    if chan.shape[-1] > out.shape[-1]:
        # Alpha (and any extra channels) pass through UNGRADED — grading a
        # matte would corrupt coverage.
        out = np.concatenate([out, chan[..., out.shape[-1]:]], axis=-1)
    if m < 1.0:
        out = chan + m * (out - chan)

    if img.ndim == 2:
        out = out[..., 0]
    return out.astype(np.float32)
