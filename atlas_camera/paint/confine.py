"""Confine an external generative edit to the region Atlas authorised.

The division of labour the bridges settled on: **the paint package selects and
paints; Atlas decodes RAW, owns colorimetry, confines, judges and stitches.**
The authorised mask stays Atlas-side because the judge must be independent of
the editor.

This is the confine leg. It composites ``edited`` over ``original`` through a
dropped, dilated, feathered version of the object mask and writes BOTH:

* the confined plate — the edit accepted only inside its brief, and
* the *authorised mask* that composite actually used.

Writing the mask matters. The scorer measures containment against the mask it
is handed, and **a feather is spill unless the authorised mask includes it**:
that gate rejected a "clean" feathered darken at 0.9329 because the feather
itself painted outside a binary mask. So the mask emitted here is the FULL
support of the blend ramp, which is the honest description of where pixels were
allowed to move.

Three measured hazards this routes around (2026-08-21):

**Paint packages mislabel their EXR exports.** A plate handed to Affinity
tagged ``lin_rec709_scene`` came back tagged ``ACEScg`` with the pixel values
UNTOUCHED — 1.6% of the frame bit-identical, 17.2% inside 1e-4, and a
least-squares 3x3 fit made the residual *worse* (0.0683 vs 0.0557), proving the
difference was content rather than colorimetry. ``read_plate`` believes a file
that self-describes, and a declared ``oiio:ColorSpace`` overrides even an
explicitly-passed ``input_colorspace`` — so naming the colourspace downstream
is INERT as a defence. Everything here reads ``raw_data=True`` and the output
is RE-TAGGED with the original's colourspace. That re-tag is what actually
protects the downstream graph.

**Write the confined plate LOSSLESS.** ``write_exr(bit_depth='half')`` selects
the dwab DCT codec, which moves essentially every pixel further than the
scorer's 1e-4 threshold: the first confined boiler plate scored 9,918,844
changed pixels against an 8,305,439-pixel edit — the codec outweighed the edit.
Hence ``float`` (zip) by default; ``half`` only for a delivery copy nothing
will be gated on.

**Gates are necessary, not sufficient.** A confined boiler plate passed both
containment and seam while being visibly wrong: a hazy blend has a LOW rim
gradient, so a smooth wrong answer scores well. Look at the picture.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def confine(*, original_path, edited_path, mask_path, out_path,
            out_mask_path=None, roi_manifest=None, drop_px: int = 0,
            dilate_px: int = 45, feather_px: int = 12,
            bit_depth: str = "float") -> dict:
    """Composite ``edited`` into ``original`` through the authorised ramp."""
    import numpy as np
    from PIL import Image

    from atlas_camera.paint.masks import dilate, drop, feather
    from atlas_camera.paint.ocio import config_identity
    from atlas_camera.paint.roi import read_manifest
    from atlas_camera.plate.oiio_io import read_plate, write_exr

    out_path = Path(out_path)
    original = read_plate(str(original_path), raw_data=True)
    edited_read = read_plate(str(edited_path), raw_data=True)

    roi = None
    if roi_manifest is not None:
        manifest = read_manifest(roi_manifest)
        roi = manifest["roi"]
        if (manifest["plate_width"], manifest["plate_height"]) != (
                original.width, original.height):
            raise ValueError(
                f"manifest describes a "
                f"{manifest['plate_width']}x{manifest['plate_height']} plate "
                f"but the original is {original.width}x{original.height}")
        if (edited_read.height, edited_read.width) != (roi["height"], roi["width"]):
            raise ValueError(
                f"raster mismatch: the edit is "
                f"{edited_read.width}x{edited_read.height} but the manifest's "
                f"ROI is {roi['width']}x{roi['height']} — export the crop at "
                f"its own size, without resampling")
        # Outside the ROI the edit simply does not exist, so seed the full frame
        # with the master: every pixel there stays bit-identical.
        edited_pixels = original.pixels.copy()
        edited_pixels[roi["y"]:roi["y"] + roi["height"],
                      roi["x"]:roi["x"] + roi["width"]] = edited_read.pixels
    else:
        if (original.height, original.width) != (edited_read.height,
                                                 edited_read.width):
            raise ValueError(
                f"raster mismatch: original {original.width}x{original.height} "
                f"vs edited {edited_read.width}x{edited_read.height} — the edit "
                f"must come back at the plate raster (or pass a roi_manifest if "
                f"it is a crop)")
        edited_pixels = edited_read.pixels

    edited = SimpleNamespace(pixels=edited_pixels)

    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape != (original.height, original.width):
        raise ValueError(
            f"mask raster {mask.shape[::-1]} does not match the plate "
            f"{original.width}x{original.height}")
    binary = mask > 0.5

    dropped = drop(np, binary, int(drop_px))
    grown = dilate(np, dropped, int(dilate_px))
    ramp = feather(np, grown.astype(np.float32), int(feather_px))
    # The authorised region is every pixel the ramp lets move AT ALL.
    authorised = ramp > 0.0

    a = ramp[..., None]
    composite = original.pixels * (1.0 - a) + edited.pixels * a

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(str(out_path), composite, bit_depth=bit_depth,
              # RE-TAG with what the plate actually is. This, not any downstream
              # widget, is what keeps a mislabelled vendor export from poisoning
              # the graph.
              source_colorspace=original.input_colorspace or None,
              extra_attribs={
                  "atlas:confined_from": str(edited_path),
                  "atlas:authorised_dilate_px": int(dilate_px),
                  "atlas:authorised_feather_px": int(feather_px),
                  "atlas:authorised_drop_px": int(drop_px),
              })

    changed_all = np.abs(edited.pixels - original.pixels).max(axis=-1) > 1e-4
    stats = {
        "object_px": int(binary.sum()),
        "dropped_px": int(dropped.sum()),
        "drop_px": int(drop_px),
        "dilated_px": int(grown.sum()),
        "authorised_px": int(authorised.sum()),
        "edit_changed_px": int(changed_all.sum()),
        "edit_changed_outside_authorised_px": int((changed_all & ~authorised).sum()),
        "kept_fraction_of_edit": (float((changed_all & authorised).sum())
                                  / float(max(1, changed_all.sum()))),
        "dilate_px": int(dilate_px),
        "feather_px": int(feather_px),
        "bit_depth": bit_depth,
        "out": str(out_path),
        "source_colorspace": original.input_colorspace,
        "ocio": config_identity(),
    }
    if roi:
        stats["roi"] = roi
        stats["roi_manifest"] = str(roi_manifest)

    if out_mask_path:
        out_mask_path = Path(out_mask_path)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((authorised * 255).astype("uint8")).save(out_mask_path)
        stats["out_mask"] = str(out_mask_path)

    return stats
