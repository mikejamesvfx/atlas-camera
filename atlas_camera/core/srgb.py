"""The sRGB transfer curve, once (IEC 61966-2-1).

Pure numpy, host-agnostic: ``core``, ``raw``, ``plate`` and the exporters all
import from here. Three variants, because their callers need different things:

- :func:`srgb_to_linear` / :func:`linear_to_srgb` -- float64 COPIES, input
  clipped to 0..1 first. The precise path (vertex colours, small arrays).
- :func:`srgb_to_linear_f32` -- float32, built in place with one full-size
  temporary. The plate path: on an 8K plate the float64 version was the
  stitch's memory peak. Agrees with the float64 curve to float32 precision,
  not bit for bit.

Copies that deliberately stay elsewhere (they are NOT this curve):
``raw.decode.srgb_encode`` computes in the INPUT dtype and returns float32;
``comfy.nodes_matrixzone._tonemap_preview`` applies a Reinhard tonemap first
and does not clip.
"""

from __future__ import annotations

from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError("sRGB conversion needs numpy — pip install -e .[vision]") from exc
    return np


def srgb_to_linear(c: Any) -> Any:
    """sRGB display (clipped to 0..1) -> linear, float64 copy."""
    np = _require_numpy()
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: Any) -> Any:
    """Linear (clipped to 0..1) -> sRGB display, float64 copy."""
    np = _require_numpy()
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_to_linear_f32(srgb: Any) -> Any:
    """sRGB display (clipped to 0..1) -> linear, float32, built in place.

    Same curve as :func:`srgb_to_linear`, which works in float64 with ~6
    full-size temporaries: on an 8K plate that one call was the stitch's
    memory peak, and its float64 result was held for the whole post-pass.
    Agrees with it to float32 precision.
    """
    np = _require_numpy()
    c = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    low = c <= 0.04045
    lin = c / np.float32(12.92)
    c += np.float32(0.055)
    c /= np.float32(1.055)
    np.power(c, np.float32(2.4), out=c)
    np.copyto(c, lin, where=low)
    return c
