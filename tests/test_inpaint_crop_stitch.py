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


def test_crop_bounds_contain_mask_bbox_plus_padding_and_align_to_64():
    """The crop must CONTAIN bbox+pad (context only ever grows) and its
    dimensions snap to multiples of 64 — the 2026-07-25 alignment rule (odd
    latent grids trip cuDNN SDPA; SD models want 64-multiples regardless)."""
    img, mask = _image(), _mask_blob()
    cimg, cmask, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=32)
    # bbox+pad was (68, 28)-(172, 122); the aligned crop must contain it.
    assert region["x0"] <= 68 and region["y0"] <= 28
    assert region["x1"] >= 172 and region["y1"] >= 122
    ch, cw = cimg.shape[1], cimg.shape[2]
    assert ch % 64 == 0 and cw % 64 == 0
    assert cmask.shape == (1, ch, cw)
    assert torch.equal(
        cimg, img[:, region["y0"]:region["y1"], region["x0"]:region["x1"], :])
    # A huge pad clamps to the frame's largest aligned box instead of erroring.
    _, _, r2 = AtlasInpaintCrop().crop(img, mask, context_pad_px=2048)
    assert (r2["x1"] - r2["x0"]) % 64 == 0 and (r2["y1"] - r2["y0"]) % 64 == 0
    assert r2["x0"] >= 0 and r2["y0"] >= 0 and r2["x1"] <= W and r2["y1"] <= H


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


def test_crop_dimensions_snap_to_multiples_of_64():
    """Unaligned crops reach the VAE as odd latent grids and trip torch/cu130's
    cuDNN SDPA ("query is not correctly aligned (strideM)") — found live on a
    538x1446 crop. SD-family models want 64-multiples anyway. Growth must be
    outward (context can only increase) and stay inside the frame."""
    import torch

    from atlas_camera.comfy.nodes import AtlasInpaintCrop

    img = torch.zeros((1, 1088, 1920, 3))
    mask = torch.zeros((1, 1088, 1920))
    mask[0, 300:838, 200:1646] = 1.0          # 538 x 1446 hot region

    cimg, cmask, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=0)
    ch, cw = cimg.shape[1], cimg.shape[2]
    assert ch % 64 == 0 and cw % 64 == 0, (ch, cw)
    assert ch >= 538 and cw >= 1446           # outward growth only
    assert cmask.shape[1:] == (ch, cw)        # mask stays aligned to the crop
    assert region["y1"] - region["y0"] == ch
    assert region["x1"] - region["x0"] == cw
    assert 0 <= region["x0"] and region["x1"] <= 1920
    assert 0 <= region["y0"] and region["y1"] <= 1088
    # The hot pixels themselves must all still be inside the crop.
    assert cmask.sum() == mask.sum()


def test_crop_alignment_shifts_inward_at_frame_borders():
    import torch

    from atlas_camera.comfy.nodes import AtlasInpaintCrop

    img = torch.zeros((1, 512, 512, 3))
    mask = torch.zeros((1, 512, 512))
    mask[0, 0:30, 480:512] = 1.0              # corner blob

    cimg, _, region = AtlasInpaintCrop().crop(img, mask, context_pad_px=16)
    assert cimg.shape[1] % 64 == 0 and cimg.shape[2] % 64 == 0
    assert region["x1"] <= 512 and region["y0"] >= 0


def test_crop_of_a_tiny_image_falls_back_to_even_latent_grid():
    import torch

    from atlas_camera.comfy.nodes import AtlasInpaintCrop

    img = torch.zeros((1, 40, 40, 3))
    mask = torch.zeros((1, 40, 40))
    mask[0, 10:30, 10:30] = 1.0

    cimg, _, _ = AtlasInpaintCrop().crop(img, mask, context_pad_px=8)
    assert cimg.shape[1] % 8 == 0 and cimg.shape[2] % 8 == 0
