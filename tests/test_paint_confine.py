"""The confine + ROI legs: the part of a paint bridge Atlas is responsible for.

The doctrine these tests defend: the paint package selects and paints; Atlas
confines, judges and stitches, and the authorised mask stays Atlas-side because
the judge must be independent of the editor.

Every test here encodes something that was learned by a measurement going
wrong, so a regression reads as the specific mistake it repeats rather than as
a generic assertion failure.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("OpenImageIO")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from atlas_camera.paint.confine import confine  # noqa: E402
from atlas_camera.paint.roi import (MANIFEST_KEYS, export_roi,  # noqa: E402
                                    read_manifest)
from atlas_camera.plate.oiio_io import (read_plate,  # noqa: E402
                                        resolve_colorspace, write_exr)


W, H = 96, 64


def _plate(tmp_path, name="plate.exr", colorspace="Linear Rec.709 (sRGB)"):
    """A smooth non-constant plate, so a codec has something to damage."""
    yy, xx = np.mgrid[0:H, 0:W]
    px = np.stack([xx / W, yy / H, np.full((H, W), 0.25)], axis=-1).astype("float32")
    path = tmp_path / name
    write_exr(str(path), px, bit_depth="float", source_colorspace=colorspace)
    return path, px


def _mask(tmp_path, name="mask.png", box=(20, 30, 26, 20)):
    x, y, w, h = box
    arr = np.zeros((H, W), "uint8")
    arr[y:y + h, x:x + w] = 255
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return path


def _edited(tmp_path, base_px, value=0.9, box=(20, 30, 26, 20), name="edited.exr"):
    x, y, w, h = box
    px = base_px.copy()
    px[y:y + h, x:x + w] = value
    path = tmp_path / name
    write_exr(str(path), px, bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)")
    return path, px


# --- containment as a construction ------------------------------------------

def test_outside_the_authorised_mask_is_bit_exact(tmp_path):
    """Containment must be a CONSTRUCTION, not a hope.

    If the composite differed from the original outside the ramp even in the
    last float bit, the scorer's changed-pixel alpha would leak past the
    authorised mask and containment would fall below 1.0 for reasons that have
    nothing to do with what the paint package did.
    """
    plate_path, plate_px = _plate(tmp_path)
    mask_path = _mask(tmp_path)
    edited_path, _ = _edited(tmp_path, plate_px)

    out = tmp_path / "confined.exr"
    out_mask = tmp_path / "authorised.png"
    confine(original_path=plate_path, edited_path=edited_path,
            mask_path=mask_path, out_path=out, out_mask_path=out_mask,
            dilate_px=3, feather_px=2)

    composite = read_plate(str(out), raw_data=True).pixels
    authorised = np.asarray(Image.open(out_mask).convert("L")) > 127
    outside = ~authorised
    assert outside.any()
    assert np.array_equal(composite[outside], plate_px[outside])


def test_authorised_mask_includes_the_whole_feather(tmp_path):
    """A feather is SPILL unless the authorised mask includes it.

    The containment gate once rejected a 'clean' feathered edit at 0.9329
    because the feather itself painted outside a binary mask. The mask this
    writes must therefore be the ramp's full support, not the object mask.
    """
    plate_path, plate_px = _plate(tmp_path)
    mask_path = _mask(tmp_path)
    edited_path, _ = _edited(tmp_path, plate_px)

    out = tmp_path / "confined.exr"
    out_mask = tmp_path / "authorised.png"
    confine(original_path=plate_path, edited_path=edited_path,
            mask_path=mask_path, out_path=out, out_mask_path=out_mask,
            dilate_px=4, feather_px=3)

    composite = read_plate(str(out), raw_data=True).pixels
    authorised = np.asarray(Image.open(out_mask).convert("L")) > 127
    moved = np.abs(composite - plate_px).max(axis=-1) > 0
    # Every pixel that actually moved must be inside the authorised region.
    assert not (moved & ~authorised).any()
    # And the authorised region must be strictly larger than the object mask,
    # which is the whole point of recording the ramp support.
    obj = np.asarray(Image.open(mask_path).convert("L")) > 127
    assert authorised.sum() > obj.sum()


def test_drop_extends_the_authorised_region_downward(tmp_path):
    plate_path, plate_px = _plate(tmp_path)
    mask_path = _mask(tmp_path)
    edited_path, _ = _edited(tmp_path, plate_px)

    stats = {}
    for drop in (0, 8):
        out_mask = tmp_path / f"auth_{drop}.png"
        stats[drop] = confine(
            original_path=plate_path, edited_path=edited_path,
            mask_path=mask_path, out_path=tmp_path / f"c_{drop}.exr",
            out_mask_path=out_mask, drop_px=drop, dilate_px=2, feather_px=1)
    assert stats[8]["authorised_px"] > stats[0]["authorised_px"]
    assert stats[8]["drop_px"] == 8


# --- colour: the re-tag is the real defence ---------------------------------

def test_confined_plate_is_retagged_with_the_originals_colourspace(tmp_path):
    """The declared tag wins on read, so the RE-TAG is what protects the graph.

    A vendor that mislabels its export (Affinity returns ACEScg over untouched
    Rec.709-linear values) would otherwise poison everything downstream.
    Naming AtlasLoadPlate.input_colorspace does NOT help -- see the companion
    test below.
    """
    plate_path, plate_px = _plate(tmp_path, colorspace="Linear Rec.709 (sRGB)")
    mask_path = _mask(tmp_path)

    # The vendor hands back a MISLABELLED file: same values, wrong tag.
    # Compare RESOLVED names throughout: write_exr tags with the active
    # config's own name for a space, so `ACEScg` lands on disk as `lin_ap1_scene`
    # under the built-in config. Asserting literal strings here would be
    # testing the config, not the re-tag.
    _, edited_px = _edited(tmp_path, plate_px, name="_tmp.exr")
    mislabelled = tmp_path / "vendor_output.exr"
    write_exr(str(mislabelled), edited_px, bit_depth="float",
              source_colorspace="ACEScg")
    vendor_tag = read_plate(str(mislabelled), raw_data=True).input_colorspace
    assert resolve_colorspace(vendor_tag) == resolve_colorspace("ACEScg")

    out = tmp_path / "confined.exr"
    stats = confine(original_path=plate_path, edited_path=mislabelled,
                    mask_path=mask_path, out_path=out,
                    dilate_px=2, feather_px=1)

    original_tag = read_plate(str(plate_path), raw_data=True).input_colorspace
    confined_tag = read_plate(str(out), raw_data=True).input_colorspace
    # The confined plate carries the ORIGINAL's space, not the vendor's claim.
    assert resolve_colorspace(confined_tag) == resolve_colorspace(original_tag)
    assert resolve_colorspace(confined_tag) != resolve_colorspace("ACEScg")
    assert stats["source_colorspace"] == original_tag


def test_a_declared_tag_overrides_an_explicit_input_colorspace(tmp_path):
    """Pins atlas_camera/plate/oiio_io.py: the file's own oiio:ColorSpace wins
    UNCONDITIONALLY, even over an explicitly passed input_colorspace.

    This is why the AtlasLoadPlate.input_colorspace widget is inert as a
    defence against a mislabelled vendor export, and why confine re-tags.
    """
    px = np.full((H, W, 3), 0.18, dtype="float32")
    path = tmp_path / "declared.exr"
    write_exr(str(path), px, bit_depth="float", source_colorspace="ACEScg")

    # Ask for something else entirely; the declared tag must still be used.
    got = read_plate(str(path), input_colorspace="Linear Rec.709 (sRGB)",
                     output_colorspace=None, raw_data=True)
    assert resolve_colorspace(got.input_colorspace) == resolve_colorspace("ACEScg")
    assert (resolve_colorspace(got.input_colorspace)
            != resolve_colorspace("Linear Rec.709 (sRGB)"))


def test_float_is_lossless_and_half_is_not(tmp_path):
    """'Gate-bound plates must be float/zip' is executable, not folklore.

    write_exr(bit_depth='half') selects the dwab DCT codec, which moves
    essentially every pixel further than the scorer's 1e-4 change threshold --
    once scoring 9,918,844 changed pixels against an 8,305,439-pixel edit.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    px = np.stack([xx / W, yy / H, (xx + yy) / (W + H)], axis=-1).astype("float32")

    lossless = tmp_path / "f.exr"
    write_exr(str(lossless), px, bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)")
    back = read_plate(str(lossless), raw_data=True).pixels
    assert np.array_equal(back, px), "float/zip must round-trip bit-exact"

    lossy = tmp_path / "h.exr"
    write_exr(str(lossy), px, bit_depth="half",
              source_colorspace="Linear Rec.709 (sRGB)")
    half_back = read_plate(str(lossy), raw_data=True).pixels
    assert np.abs(half_back - px).max() > 1e-6, (
        "half/dwab is expected to be LOSSY; if this ever passes bit-exact the "
        "codec default changed and the float-only rule needs revisiting")


# --- the ROI contract -------------------------------------------------------

def test_roi_manifest_carries_the_full_contract(tmp_path):
    plate_path, _ = _plate(tmp_path)
    mask_path = _mask(tmp_path)
    manifest_path = tmp_path / "roi.json"
    manifest = export_roi(plate_path=plate_path, mask_path=mask_path,
                          out_path=tmp_path / "roi.exr",
                          manifest_path=manifest_path,
                          out_mask_path=tmp_path / "roi_mask.png",
                          margin_px=6)

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in MANIFEST_KEYS:
        assert key in on_disk, f"manifest lost the contract key {key!r}"
    roi = manifest["roi"]
    assert manifest["roi_fraction_of_frame"] == pytest.approx(
        roi["width"] * roi["height"] / float(W * H))
    # The manifest must record which OCIO config produced it: a colourspace
    # name without a config is not a contract.
    assert "config_sha256" in on_disk["ocio"]
    assert read_manifest(manifest_path)["roi"] == roi


def test_read_manifest_rejects_a_foreign_json(tmp_path):
    bogus = tmp_path / "nope.json"
    bogus.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a paint-bridge ROI manifest"):
        read_manifest(bogus)


def test_roi_paste_back_lands_exactly_inside_the_roi(tmp_path):
    """An off-by-one in the paste offset is the likeliest bridge bug, and it
    would be invisible in a score: the edit would simply be one pixel wrong."""
    plate_path, plate_px = _plate(tmp_path)
    mask_path = _mask(tmp_path)

    manifest_path = tmp_path / "roi.json"
    roi_exr = tmp_path / "roi.exr"
    manifest = export_roi(plate_path=plate_path, mask_path=mask_path,
                          out_path=roi_exr, manifest_path=manifest_path,
                          margin_px=5)
    roi = manifest["roi"]

    # "Paint" the whole crop a unique constant no plate pixel has.
    sentinel = 0.7654321
    crop = read_plate(str(roi_exr), raw_data=True).pixels.copy()
    crop[:] = sentinel
    edited_crop = tmp_path / "roi_edited.exr"
    write_exr(str(edited_crop), crop, bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)")

    out = tmp_path / "confined.exr"
    # feather/dilate 0 so the ramp is exactly the mask: this test is about the
    # OFFSET, and a ramp would blur the boundary being checked.
    confine(original_path=plate_path, edited_path=edited_crop,
            mask_path=mask_path, out_path=out, roi_manifest=manifest_path,
            dilate_px=0, feather_px=0)

    composite = read_plate(str(out), raw_data=True).pixels
    hit = np.isclose(composite[..., 0], sentinel, atol=1e-5)
    assert hit.any(), "the pasted crop never appeared in the composite"

    ys, xs = np.where(hit)
    obj = np.asarray(Image.open(mask_path).convert("L")) > 127
    oy, ox = np.where(obj)
    # With no ramp, the moved region is exactly the object mask -- and it must
    # sit inside the ROI rectangle the manifest recorded.
    assert xs.min() == ox.min() and xs.max() == ox.max()
    assert ys.min() == oy.min() and ys.max() == oy.max()
    assert roi["x"] <= xs.min() and xs.max() < roi["x"] + roi["width"]
    assert roi["y"] <= ys.min() and ys.max() < roi["y"] + roi["height"]


def test_roi_paste_back_rejects_a_resized_crop(tmp_path):
    """The paint package must not resize the crop; if it did, every pixel would
    land in the wrong place and the score would still look plausible."""
    plate_path, _ = _plate(tmp_path)
    mask_path = _mask(tmp_path)
    manifest_path = tmp_path / "roi.json"
    export_roi(plate_path=plate_path, mask_path=mask_path,
               out_path=tmp_path / "roi.exr", manifest_path=manifest_path,
               margin_px=5)

    wrong = tmp_path / "resized.exr"
    write_exr(str(wrong), np.zeros((9, 11, 3), "float32"), bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)")
    with pytest.raises(ValueError, match="without resampling"):
        confine(original_path=plate_path, edited_path=wrong,
                mask_path=mask_path, out_path=tmp_path / "c.exr",
                roi_manifest=manifest_path)


def test_confine_rejects_a_mismatched_full_frame(tmp_path):
    plate_path, _ = _plate(tmp_path)
    mask_path = _mask(tmp_path)
    wrong = tmp_path / "wrong.exr"
    write_exr(str(wrong), np.zeros((H // 2, W, 3), "float32"), bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)")
    with pytest.raises(ValueError, match="raster mismatch"):
        confine(original_path=plate_path, edited_path=wrong,
                mask_path=mask_path, out_path=tmp_path / "c.exr")
