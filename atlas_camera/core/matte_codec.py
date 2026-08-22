"""Strict grayscale-matte <-> PNG data-URI codec.

Wire format is exactly the one the viewport and the DCC layer writers already
consume: ``data:image/png;base64,<...>`` holding a single-channel 8-bit PNG,
white = keep. It is the form carried by ``AtlasProxyPrimitive.metadata
["silhouette_matte_b64"]`` and ``ProjectionSource.mask_b64``.

Why this exists next to ``comfy.node_helpers._mask_to_b64_png``: that helper
fails soft to ``""`` on any error, which is right for a node that would rather
render an un-matted layer than abort a graph. Evidence-side callers need the
opposite — a matte that cannot be encoded or decoded is a defect, not a
degraded render — so these raise. Two contracts, one wire format. Nothing
outside ``comfy/`` may import that helper anyway.
"""
from __future__ import annotations

import base64
import io
from typing import Any

_DATA_URI_PREFIX = "data:image/png;base64,"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError("matte encoding needs numpy — pip install -e .[vision]") from exc
    return np


def _require_pil() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError("matte encoding needs Pillow — pip install -e .[image]") from exc
    return Image


def encode_matte_png(matte: Any) -> str:
    """``(H, W)`` bool/float matte -> ``data:image/png;base64,...`` data URI.

    Values are clipped to ``[0, 1]`` and quantised to 8 bits. PNG (lossless),
    never JPEG: a matte's 0.5 threshold must not pick up ringing at the exact
    edge being cut. Raises on anything that is not a finite 2D array.
    """

    np = _require_numpy()
    PILImage = _require_pil()
    array = np.asarray(matte)
    if array.ndim != 2:
        raise ValueError("matte must be a 2D (H, W) array")
    if array.size == 0:
        raise ValueError("matte must not be empty")
    if array.dtype.kind not in "biuf":
        raise ValueError("matte must be numeric or boolean")
    values = array.astype(np.float32, copy=False)
    if not bool(np.isfinite(values).all()):
        raise ValueError("matte must be finite")
    # Truncate, do not round: this must be byte-identical to
    # comfy.node_helpers._mask_to_b64_png, whose output is already serialized
    # into saved solves. Rounding sends 0.5 to 128 instead of 127, which flips
    # that decoder's `> 127` threshold on exactly the mid-grey value.
    quantised = (values.clip(0.0, 1.0) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    PILImage.fromarray(quantised, mode="L").save(buffer, format="PNG", optimize=True)
    return _DATA_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_matte_png(data_uri: str) -> Any:
    """Inverse of :func:`encode_matte_png` -> ``(H, W)`` float32 in ``[0, 1]``.

    Accepts a bare base64 payload as well as the full data URI, because that is
    what the viewport round-trips. Raises on empty or undecodable input.
    """

    np = _require_numpy()
    PILImage = _require_pil()
    if not isinstance(data_uri, str) or not data_uri:
        raise ValueError("matte data URI must be a non-empty string")
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("matte payload is not valid base64") from exc
    if not raw:
        raise ValueError("matte payload decoded to zero bytes")
    try:
        image = PILImage.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError("matte payload is not a readable PNG") from exc
    if image.format != "PNG":
        raise ValueError("matte payload must be a PNG")
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0
