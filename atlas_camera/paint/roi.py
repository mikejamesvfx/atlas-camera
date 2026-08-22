"""Cut the ROI crop that makes an external generative edit survivable.

This is the leg that runs BEFORE the paint package, and for at least one vendor
it is not optional. Affinity's ``generativeEditImage`` is image-to-image
REGENERATION, not inpainting: handed a whole frame it returns a whole new
frame. Measured twice on 2026-08-21, same call, same app version — the boiler
plate kept its composition but scored containment 0.3740, and the street plate
was replaced by a different building entirely. So the boiler result was luck,
not behaviour.

The only reliable confinement is GEOMETRIC: hand the model a crop, and
everything outside it is untouched by construction rather than by hope. The
crop must be tight enough that regeneration cannot invent a new scene, and
loose enough that the model still sees the context it needs to continue paving,
kerbs and shadow.

Whether a given vendor needs this at all is a MEASUREMENT, not an assumption —
see ``atlas_camera.paint.vendors``. Photoshop's ``syntheticFill`` exposes an
explicit ``inpaint`` mode, which may make the crop unnecessary; until that is
scored at full resolution, the vendor table says ``None`` and the tools refuse
to guess.

The manifest written here is the bridge CONTRACT: it carries the crop rectangle
so the edited crop can be composited back at exactly the right offset, and the
OCIO config identity so a score is never silently compared across configs.
"""
from __future__ import annotations

from pathlib import Path

MANIFEST_KEYS = (
    "plate", "plate_width", "plate_height", "roi", "object_bbox",
    "margin_px", "drop_px", "roi_fraction_of_frame", "input_colorspace",
    "roi_exr", "ocio",
)


def export_roi(*, plate_path, mask_path, out_path, manifest_path,
               out_mask_path=None, margin_px: int = 240, drop_px: int = 0,
               bit_depth: str = "float") -> dict:
    """Crop ``plate_path`` around the mask's bbox and write the crop + manifest.

    ``bit_depth`` defaults to ``float`` (zip, lossless): the crop is an
    intermediate that gets gated, and the dwab DCT codec that ``half`` selects
    moves every pixel past the scorer's change threshold.
    """
    import numpy as np
    from PIL import Image

    from atlas_camera.paint.masks import drop as drop_mask
    from atlas_camera.paint.ocio import config_identity
    from atlas_camera.plate.oiio_io import read_plate, write_exr

    plate_path = Path(plate_path)
    out_path = Path(out_path)
    manifest_path = Path(manifest_path)

    plate = read_plate(str(plate_path), raw_data=True)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape != (plate.height, plate.width):
        raise ValueError(
            f"mask raster {mask.shape[::-1]} does not match the plate "
            f"{plate.width}x{plate.height}")
    binary = mask > 0.5
    if not binary.any():
        raise ValueError("mask is empty: nothing to crop around")

    # Grow DOWNWARD before measuring the bbox, so a ground-standing object's
    # legs, footings and contact shadow fall inside the crop. Gravity-directed
    # growth costs no sideways bloat, which an equivalent dilation would.
    grown = drop_mask(np, binary, int(drop_px)) if drop_px else binary
    ys, xs = np.where(grown)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    m = int(margin_px)
    cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
    cx1, cy1 = min(plate.width, x1 + m), min(plate.height, y1 + m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(str(out_path), plate.pixels[cy0:cy1, cx0:cx1], bit_depth=bit_depth,
              source_colorspace=plate.input_colorspace or None,
              extra_attribs={"atlas:roi_of": str(plate_path),
                             "atlas:roi_x": cx0, "atlas:roi_y": cy0})

    manifest = {
        "plate": str(plate_path),
        "plate_width": int(plate.width),
        "plate_height": int(plate.height),
        "roi": {"x": cx0, "y": cy0,
                "width": int(cx1 - cx0), "height": int(cy1 - cy0)},
        "object_bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        "margin_px": m,
        "drop_px": int(drop_px),
        "roi_fraction_of_frame":
            float((cx1 - cx0) * (cy1 - cy0)) / float(plate.width * plate.height),
        "input_colorspace": plate.input_colorspace,
        "roi_exr": str(out_path),
        # A colourspace name without a config is not a contract.
        "ocio": config_identity(),
    }
    if out_mask_path:
        out_mask_path = Path(out_mask_path)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            (binary[cy0:cy1, cx0:cx1] * 255).astype("uint8")).save(out_mask_path)
        manifest["roi_mask"] = str(out_mask_path)

    write_manifest(manifest_path, manifest)
    return manifest


def write_manifest(path, manifest: dict) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def read_manifest(path) -> dict:
    """Load a manifest and check it carries the contract's required keys."""
    import json

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(
            f"{path}: not a paint-bridge ROI manifest — missing {missing}. "
            f"Regenerate it with tools/paint_roi_export.py.")
    return manifest
