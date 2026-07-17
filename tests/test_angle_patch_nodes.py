"""AtlasExtractAnglePatch / AtlasImportAnglePatch — the Photoshop round trip.

The registration contract under test: the extraction CROP is a Photoshop
convenience only; import must paste the edited crop back into the FULL frame
(AtlasAddPatchView's ProjectionSource samples uv across the whole
patch-camera frustum, so a bare crop would stretch and misregister), and the
exact orbit string must survive byte-for-byte.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import AtlasExtractAnglePatch, AtlasImportAnglePatch


def _extract(tmp_path, make_atlas_solve):
    solve = make_atlas_solve(image_width=8, image_height=6)
    plate = torch.zeros(1, 6, 8, 3, dtype=torch.float32)
    plate[:, 1:5, 2:7, :] = 0.75
    matte = torch.zeros(1, 6, 8, dtype=torch.float32)
    matte[:, 2:4, 3:6] = 1.0
    return plate, AtlasExtractAnglePatch().extract(
        solve, plate, matte, "azimuth_deg=12.5 elevation_deg=-2.0 distance_scale=1.1",
        str(tmp_path), padding_px=1, colorspace="ACEScg", name="photoshop_fix",
    )


def test_extract_writes_crop_full_frame_and_manifest(tmp_path, make_atlas_solve):
    _, (image, mask, manifest_path, package) = _extract(tmp_path, make_atlas_solve)
    root = tmp_path / "photoshop_fix"
    manifest = json.loads((root / "atlas_angle_patch.json").read_text())
    assert manifest["kind"] == "atlas_angle_patch"
    assert manifest["crop_bbox_xyxy"] == [2, 1, 7, 5]
    assert manifest["full_wh"] == [8, 6]
    assert (root / "plate_full.png").is_file()
    # camera block only — a layered solve's full to_dict would balloon the sidecar
    assert "source_solve" not in manifest
    assert manifest.get("source_camera")
    assert manifest["atlas_version"] != "0.5.0"
    assert image.shape == (1, 4, 5, 3)
    assert mask.shape == (1, 4, 5)
    assert package["patch_exact"].startswith("azimuth_deg=12.5")


def test_import_pastes_edit_back_into_full_frame(tmp_path, make_atlas_solve):
    plate, (crop_image, crop_mask, _mp, package) = _extract(tmp_path, make_atlas_solve)
    edited = crop_image.clone()
    edited[:, 1, 1, :] = torch.tensor([1.0, 0.0, 0.0])  # the "Photoshop edit"

    full_img, full_mask, exact, pkg = AtlasImportAnglePatch().import_patch(
        package, edited_image=edited, edited_matte=crop_mask)

    # FULL frame back, not the crop
    assert full_img.shape == (1, 6, 8, 3)
    assert full_mask.shape == (1, 6, 8)
    # the edit landed at bbox-offset position (crop origin x0=2, y0=1)
    assert torch.allclose(full_img[0, 2, 3], torch.tensor([1.0, 0.0, 0.0]), atol=2 / 255)
    # every pixel OUTSIDE the crop bbox is the original plate — registration holds
    x0, y0, x1, y1 = pkg["crop_bbox_xyxy"]
    outside = torch.ones(6, 8, dtype=torch.bool)
    outside[y0:y1, x0:x1] = False
    assert torch.allclose(full_img[0][outside], plate[0][outside], atol=2 / 255)
    # matte pasted at offset, zero elsewhere
    assert full_mask[0, 2, 3] == 1.0
    assert full_mask[0][outside].max() == 0.0
    assert exact == "azimuth_deg=12.5 elevation_deg=-2.0 distance_scale=1.1"
    assert pkg["imported"] is True


def test_import_rejects_resized_edit(tmp_path, make_atlas_solve):
    _, (_i, _m, _mp, package) = _extract(tmp_path, make_atlas_solve)
    wrong = torch.zeros(1, 3, 3, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="must not resize"):
        AtlasImportAnglePatch().import_patch(package, edited_image=wrong)


def test_extract_rejects_empty_matte(tmp_path, make_atlas_solve):
    solve = make_atlas_solve(image_width=4, image_height=4)
    with pytest.raises(ValueError, match="no non-zero"):
        AtlasExtractAnglePatch().extract(
            solve, torch.zeros(1, 4, 4, 3), torch.zeros(1, 4, 4),
            "azimuth_deg=0 elevation_deg=0 distance_scale=1", str(tmp_path),
        )
