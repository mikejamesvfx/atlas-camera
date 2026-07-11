"""Tests for AtlasInpaintCrop / AtlasInpaintStitch — the crop-before-LaMa
quality lever (2026-07-11). Pure tensor orchestration: the crop spends the
inpaint node's fixed internal resolution (LaMa: the WHOLE image squashed to
256x256) on the hole's neighborhood instead of the full frame. No inpainting
math lives here — only torch needed.
"""

import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasInpaintCrop,
    AtlasInpaintStitch,
)

H, W = 200, 320


def _image():
    return torch.rand(1, H, W, 3, dtype=torch.float32)


def _mask_blob(y0=60, y1=90, x0=100, x1=140):
    m = torch.zeros(1, H, W, dtype=torch.float32)
    m[:, y0:y1, x0:x1] = 1.0
    return m


def test_nodes_registered():
    assert NODE_CLASS_MAPPINGS["AtlasInpaintCrop"] is AtlasInpaintCrop
    assert NODE_CLASS_MAPPINGS["AtlasInpaintStitch"] is AtlasInpaintStitch
    assert AtlasInpaintCrop.RETURN_TYPES == ("IMAGE", "MASK", "ATLAS_CROP_REGION")


def test_crop_bounds_are_mask_bbox_plus_padding_clamped():
    img, mask = _image(), _mask_blob()
    cimg, cmask, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=32)
    assert (region["x0"], region["y0"], region["x1"], region["y1"]) == (68, 28, 172, 122)
    assert cimg.shape == (1, 94, 104, 3)
    assert cmask.shape == (1, 94, 104)
    assert torch.equal(cimg, img[:, 28:122, 68:172, :])
    # A huge pad clamps to the frame instead of erroring.
    _, _, r2 = AtlasInpaintCrop().crop(img, mask, context_pad_px=2048)
    assert (r2["x0"], r2["y0"], r2["x1"], r2["y1"]) == (0, 0, W, H)


def test_crop_empty_mask_passes_through_full_frame():
    img = _image()
    cimg, cmask, region = AtlasInpaintCrop().crop(img, torch.zeros(1, H, W), context_pad_px=64)
    assert cimg.shape == img.shape
    assert (region["x1"], region["y1"]) == (W, H)


def test_stitch_roundtrip_is_identity_outside_and_replaces_inside():
    img, mask = _image(), _mask_blob()
    cimg, cmask, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=16)
    fake_inpaint = torch.ones_like(cimg) * 0.5  # pretend the model repainted the crop
    (out,) = AtlasInpaintStitch().stitch(img, fake_inpaint, region)
    y0, y1, x0, x1 = region["y0"], region["y1"], region["x0"], region["x1"]
    assert torch.allclose(out[:, y0:y1, x0:x1, :], fake_inpaint)      # rect replaced
    outside = out.clone()
    outside[:, y0:y1, x0:x1, :] = img[:, y0:y1, x0:x1, :]
    assert torch.equal(outside, img)                                  # rest untouched


def test_stitch_resizes_mismatched_crop():
    # Generative inpainters snap to multiples of 8 / upscalers return 4x —
    # the stitch must resize back to the region before pasting.
    img, mask = _image(), _mask_blob()
    _, _, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=16)
    rh, rw = region["y1"] - region["y0"], region["x1"] - region["x0"]
    upscaled = torch.ones(1, rh * 2, rw * 2, 3) * 0.25
    (out,) = AtlasInpaintStitch().stitch(img, upscaled, region)
    assert out.shape == img.shape
    assert torch.allclose(out[:, region["y0"]:region["y1"], region["x0"]:region["x1"], :],
                          torch.full((1, rh, rw, 3), 0.25), atol=1e-5)


def test_stitch_masked_paste_keeps_unmasked_crop_pixels():
    img, mask = _image(), _mask_blob()
    cimg, cmask, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=16)
    fake = torch.zeros_like(cimg)
    (out,) = AtlasInpaintStitch().stitch(img, fake, region, mask=mask, feather_px=0)
    # Inside the mask: repainted; inside the rect but OUTSIDE the mask: original.
    assert torch.allclose(out[:, 60:90, 100:140, :], torch.zeros(1, 30, 40, 3))
    assert torch.equal(out[:, 45:59, 100:140, :], img[:, 45:59, 100:140, :])
