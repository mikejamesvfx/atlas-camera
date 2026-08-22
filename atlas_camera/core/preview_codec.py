"""Scene-linear ACEScg -> display-referred preview, as a PNG data URI.

WHY THIS MODULE EXISTS. A roundtripped card carried ``mask_b64`` and
``image_b64: null``. The viewport textures a projection source from
``image_b64`` (``comfy.headless_evidence._decode_rgba``), so a card with an
alpha and no pixels is invisible — which is what the DSC_2552 verify graph
showed: three correctly-placed, correctly-matted cards that drew nothing.

The pixels live in an ACEScg EXR, and turning those into something displayable
is a COLOUR TRANSFORM, not a cast. AP1 primaries are not sRGB primaries, so
writing scene-linear values into an 8-bit PNG gets two things wrong at once:
the wrong primaries (a saturated ACEScg red is outside the sRGB gamut) and the
missing transfer function (linear 0.18 is display 0.46, so the whole frame
comes back roughly half as bright as it should be).

WHAT THIS IS NOT. It is a PREVIEW. There is no tonemap — highlights clip — and
the result is 8-bit. Float work reads the EXR through ``plate.oiio_io``; the
``plate_ref`` on every source points at it. This exists so a viewport can draw
the card, and it must never be mistaken for the radiance.

Why not OCIO: ``plate/oiio_io.py`` does the real colour management and needs
OpenImageIO. The ACEScg->sRGB matrix is a published constant, so a preview can
be exact without dragging that dependency into ``core``.
"""

from __future__ import annotations

import base64
import io
from typing import Any

_DATA_URI_PREFIX = "data:image/png;base64,"

#: ACEScg (AP1) -> sRGB/Rec.709 linear. Rows sum to 1, which is what keeps a
#: neutral neutral; a matrix that loses that property tints every grey.
ACESCG_TO_SRGB = (
    (1.70505, -0.62179, -0.08326),
    (-0.13026, 1.14080, -0.01055),
    (-0.02400, -0.12897, 1.15297),
)

#: Full-frame card previews are 7380x4928. Both viewport decoders resample to
#: whatever raster they are drawing into, so shipping the full resolution only
#: bloats the solve JSON — which is a contract other tools parse.
DEFAULT_MAX_LONG_EDGE = 2048

SUPPORTED_COLORSPACES = ("acescg", "srgb_display")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError(
            "preview encoding needs numpy — pip install -e .[vision]") from exc
    return np


def _require_pil() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded like the rest of core
        raise ImportError(
            "preview encoding needs Pillow — pip install -e .[image]") from exc
    return Image


def _srgb_encode(np: Any, linear: Any) -> Any:
    """The sRGB EOTF. Linear 0.18 -> ~0.46, which is the whole point."""
    clipped = np.clip(linear, 0.0, 1.0)
    return np.where(clipped <= 0.0031308,
                    clipped * 12.92,
                    1.055 * np.power(clipped, 1.0 / 2.4) - 0.055)


def acescg_to_srgb_display(rgb: Any) -> Any:
    """``(..., 3)`` scene-linear ACEScg -> display-referred sRGB in ``[0, 1]``.

    Out-of-gamut colours clip rather than being gamut-mapped: a preview that
    invents a rolloff nobody asked for is harder to reason about than one that
    clips and says so.
    """
    np = _require_numpy()
    array = np.asarray(rgb, dtype=np.float64)
    if array.shape[-1] < 3:
        raise ValueError(f"expected a trailing RGB axis, got shape {array.shape}")
    values = array[..., :3]
    if not bool(np.isfinite(values).all()):
        raise ValueError("preview pixels must be finite")
    matrix = np.asarray(ACESCG_TO_SRGB, dtype=np.float64)
    linear = values @ matrix.T
    return _srgb_encode(np, linear)


def encode_preview_png(image: Any, *, colorspace: str = "acescg",
                       max_long_edge: int = DEFAULT_MAX_LONG_EDGE) -> str:
    """``(H, W, 3|4)`` image -> ``data:image/png;base64,...`` display preview.

    An alpha channel is DROPPED, not composited: the alpha travels separately
    as ``mask_b64``, and baking it in here would matte the card twice and
    darken its own edge.

    ``colorspace`` is explicit because applying the transform to data that is
    already display-referred is a silent double-brightening with no symptom
    other than a washed-out card.
    """
    np = _require_numpy()
    PILImage = _require_pil()

    if colorspace not in SUPPORTED_COLORSPACES:
        raise ValueError(
            f"unknown colorspace {colorspace!r}; expected one of "
            f"{SUPPORTED_COLORSPACES}")

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected an HxWx3 or HxWx4 image, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("preview image must not be empty")

    rgb = np.asarray(array[..., :3], dtype=np.float64)
    if not bool(np.isfinite(rgb).all()):
        raise ValueError("preview pixels must be finite")

    display = (acescg_to_srgb_display(rgb) if colorspace == "acescg"
               else np.clip(rgb, 0.0, 1.0))

    # Truncate rather than round, matching core.matte_codec: one quantisation
    # rule across the two codecs that write into the same solve JSON.
    quantised = np.clip(display * 255.0, 0.0, 255.0).astype(np.uint8)
    picture = PILImage.fromarray(quantised, mode="RGB")

    cap = int(max_long_edge)
    long_edge = max(picture.size)
    if cap > 0 and long_edge > cap:
        scale = cap / float(long_edge)
        picture = picture.resize(
            (max(1, round(picture.size[0] * scale)),
             max(1, round(picture.size[1] * scale))),
            PILImage.Resampling.LANCZOS)

    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    return _DATA_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")
