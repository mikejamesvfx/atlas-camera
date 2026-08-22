"""Task 2 evidence-first float plate and auxiliary artifact contracts."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

pytest.importorskip(
    "atlas_world",
    reason="the evidence-plate layer ships in the private atlas-world "
           "distribution, which atlas-camera does not depend on",
)


def _mod():
    return importlib.import_module("atlas_world.plate_artifacts")


def _master(h=4, w=6):
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[..., 0] = 0.25
    image[..., 1] = 0.5
    image[..., 2] = 2.0
    return image


def _metadata():
    return {
        "coordinate_system": "metric_right_handed_y_up",
        "image_origin": "top_left",
        "depth_convention": "positive_camera_forward",
        "color_space": "ACEScg",
        "ocio_config": "ocio://default",
        "ocio_config_hash": "a" * 64,
        "ocio_transform": "scene_linear_to_ACEScg",
    }


def _full_aux_channels(h=2, w=2):
    zero = np.zeros((h, w), dtype=np.float32)
    one = np.ones((h, w), dtype=np.float32)
    return {
        "depth.Z": one.copy(),
        "P_world.red": zero.copy(), "P_world.green": zero.copy(), "P_world.blue": zero.copy(),
        "N_world.red": zero.copy(), "N_world.green": one.copy(), "N_world.blue": zero.copy(),
        "validity": one.copy(),
        "object_id": np.zeros((h, w), dtype=np.float32), "semantic_id": np.zeros((h, w), dtype=np.float32),
        "card_id": np.zeros((h, w), dtype=np.float32), "workspace_id": np.zeros((h, w), dtype=np.float32),
        "generated_support": zero.copy(), "disocclusion": zero.copy(), "approval": one.copy(),
        "distortion.u": zero.copy(), "distortion.v": zero.copy(),
        "undistortion.u": zero.copy(), "undistortion.v": zero.copy(),
    }


def test_import_is_dependency_free(monkeypatch):
    """Importing the contract module must not import either optional package."""
    for name in ("numpy", "OpenImageIO"):
        monkeypatch.setitem(sys.modules, name, None)
    sys.modules.pop("atlas_world.plate_artifacts", None)
    module = importlib.import_module("atlas_world.plate_artifacts")
    assert hasattr(module, "MasterPlate")


def test_capability_failure_is_structured(monkeypatch):
    module = _mod()
    monkeypatch.setitem(sys.modules, "OpenImageIO", None)
    with pytest.raises(module.PlateCapabilityError) as error:
        module.require_plate_oiio()
    assert error.value.capability == "plate_artifacts"
    assert "OpenImageIO" in str(error.value)


def test_master_roundtrip_is_half_acescg_and_zip(tmp_path):
    module = _mod()
    path = tmp_path / "master.exr"
    result = module.write_master_plate(path, _master(), metadata=_metadata())
    assert result == path
    plate = module.read_master_plate(path)
    assert plate.pixels.shape == (4, 6, 3)
    assert plate.pixels.dtype == np.float32
    assert plate.pixels[0, 0, 2] == pytest.approx(2.0, abs=0.002)
    assert plate.metadata["color_space"] == "ACEScg"
    header = module.inspect_exr(path)
    assert header["compression"] == "zip"
    assert header["channels"]["R"] in {"half", "float16"}
    assert header["data_window"] == (0, 0, 6, 4)


def test_aux_roundtrip_has_required_channels_and_stable_id_types(tmp_path):
    module = _mod()
    h, w = 4, 6
    channels = {
        "depth.Z": np.full((h, w), 4.0, dtype=np.float32),
        "P_world.red": np.zeros((h, w), dtype=np.float32),
        "P_world.green": np.ones((h, w), dtype=np.float32),
        "P_world.blue": np.full((h, w), -2.0, dtype=np.float32),
        "N_world.red": np.zeros((h, w), dtype=np.float32),
        "N_world.green": np.ones((h, w), dtype=np.float32),
        "N_world.blue": np.zeros((h, w), dtype=np.float32),
        "validity": np.ones((h, w), dtype=np.float32),
            "object_id": np.full((h, w), 7, dtype=np.float32),
            "semantic_id": np.full((h, w), 3, dtype=np.float32),
            "card_id": np.full((h, w), 11, dtype=np.float32),
            "workspace_id": np.full((h, w), 13, dtype=np.float32),
        "generated_support": np.zeros((h, w), dtype=np.float32),
        "disocclusion": np.zeros((h, w), dtype=np.float32),
        "approval": np.ones((h, w), dtype=np.float32),
        "distortion.u": np.zeros((h, w), dtype=np.float32),
        "distortion.v": np.zeros((h, w), dtype=np.float32),
        "undistortion.u": np.zeros((h, w), dtype=np.float32),
        "undistortion.v": np.zeros((h, w), dtype=np.float32),
    }
    path = tmp_path / "aux.exr"
    module.write_auxiliary_exr(path, channels, metadata=_metadata())
    got = module.read_auxiliary_exr(path)
    assert got["object_id"].dtype == np.float32
    assert set(channels).issubset(got)
    header = module.inspect_exr(path)
    assert header["compression"] == "zip"
    assert header["channels"]["depth.Z"] in {"float", "float32"}


def test_aux_rejects_invalid_depth_normals_ids_and_missing_validity():
    module = _mod()
    valid = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="validity"):
        module.validate_auxiliary_channels({"depth.Z": valid})
    with pytest.raises(ValueError, match="depth"):
        bad = _full_aux_channels(); bad["depth.Z"] = -valid
        module.validate_auxiliary_channels(bad)
    with pytest.raises(ValueError, match="normal"):
        bad = _full_aux_channels(); bad["N_world.red"] = valid; bad["N_world.green"] = valid; bad["N_world.blue"] = valid
        module.validate_auxiliary_channels(bad)
    with pytest.raises(ValueError, match="finite"):
        bad = _full_aux_channels(); bad["depth.Z"] = np.array([[np.nan, 1], [1, 1]], dtype=np.float32)
        module.validate_auxiliary_channels(bad)
    with pytest.raises(ValueError, match="finite integers"):
        bad = _full_aux_channels(); bad["object_id"] = np.array([[1.5, 1], [1, 1]], dtype=np.float32)
        module.validate_auxiliary_channels(bad)


def test_aux_requires_complete_schema_and_binary_mattes():
    module = _mod()
    with pytest.raises(ValueError, match="required"):
        module.validate_auxiliary_channels({"validity": np.ones((2, 2), dtype=np.float32)})
    with pytest.raises(ValueError, match="binary"):
        bad = _full_aux_channels(); bad["validity"][:] = 2
        module.validate_auxiliary_channels(bad)


def test_aux_ids_require_uint32_and_preserve_high_values(tmp_path):
    module = _mod()
    channels = _full_aux_channels(1, 2)
    channels["object_id"] = np.array([[2**24 - 1, 2**24]], dtype=np.float32)
    path = tmp_path / "high_ids.exr"
    module.write_auxiliary_exr(path, channels, metadata=_metadata())
    got = module.read_auxiliary_exr(path)
    assert got["object_id"].dtype == np.float32
    assert np.array_equal(got["object_id"], channels["object_id"])
    raw = module.require_plate_oiio().ImageInput.open(str(path))
    spec = raw.spec(); index = list(spec.channelnames).index("object_id")
    assert str(spec.get_channelformats()[index]) in {"float", "float32"}
    raw.close()
    channels["object_id"] = np.array([[2**24 + 1, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        module.validate_auxiliary_channels(channels)


def test_generated_patch_uses_full_canvas_windows_and_support(tmp_path):
    module = _mod()
    patch = np.full((2, 3, 3), 9.0, dtype=np.float32)
    path = tmp_path / "patch.exr"
    module.write_generated_patch(
        path, patch, canvas_size=(6, 4), data_window=(2, 1, 3, 2),
        approved_support=np.array([[1, 1, 1], [1, 1, 1]], dtype=np.uint8),
        validity=np.ones((2, 3), dtype=np.float32), source_sha256="b" * 64,
        metadata=_metadata(),
    )
    info = module.inspect_exr(path)
    assert info["data_window"] == (2, 1, 3, 2)
    assert info["display_window"] == (0, 0, 6, 4)
    got = module.read_generated_patch(path)
    assert got.data_window == (2, 1, 3, 2)
    assert got.source_sha256 == "b" * 64
    assert np.array_equal(got.validity, np.ones((2, 3), dtype=np.float32))
    assert np.array_equal(got.approved_support, np.ones((2, 3), dtype=np.float32))


def test_observed_card_is_sparse_half_rgba_with_source_binding(tmp_path):
    module = _mod()
    rgba = np.zeros((2, 3, 4), dtype=np.float32)
    rgba[..., :3] = _master(2, 3)
    rgba[..., 3] = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.float32)
    path = tmp_path / "observed_card.exr"
    module.write_observed_card_exr(
        path,
        rgba,
        canvas_size=(6, 4),
        data_window=(2, 1, 3, 2),
        source_sha256="e" * 64,
        metadata=_metadata(),
    )
    info = module.inspect_exr(path)
    assert info["data_window"] == (2, 1, 3, 2)
    assert info["display_window"] == (0, 0, 6, 4)
    assert all(info["channels"][name] in {"half", "float16"} for name in "RGBA")
    card = module.read_observed_card_exr(path)
    assert card.source_sha256 == "e" * 64
    assert card.metadata["atlas:artifact"] == "observed_card"
    assert card.metadata["atlas:provenance"] == "OBSERVED"
    assert np.array_equal(card.pixels[..., 3] > 0.5, rgba[..., 3] > 0.5)
    bad_metadata = _metadata()
    bad_metadata.pop("color_space")
    with pytest.raises(ValueError, match="ACEScg"):
        module.write_observed_card_exr(
            tmp_path / "untagged.exr",
            rgba,
            canvas_size=(6, 4),
            data_window=(2, 1, 3, 2),
            source_sha256="e" * 64,
            metadata=bad_metadata,
        )


def test_redistortion_exr_is_lossless_float_data(tmp_path):
    module = _mod()
    u = np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    v = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    path = tmp_path / "redistortion.exr"
    module.write_redistortion_exr(
        path,
        u,
        v,
        source_sha256="1" * 64,
        metadata=_metadata(),
    )
    info = module.inspect_exr(path)
    assert info["compression"] == "zip"
    assert info["channels"] == {"distortion.u": "float", "distortion.v": "float"}
    assert info["metadata"]["atlas:origin"] == "bottom_left_nuke"
    assert info["metadata"]["atlas:source_sha256"] == "1" * 64


def test_auxiliary_invalid_data_requires_zero_sentinel():
    module = _mod()
    validity = np.array([[1, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="zero sentinel"):
        bad = _full_aux_channels(1, 2); bad["validity"] = validity; bad["depth.Z"] = np.array([[2, 1]], dtype=np.float32)
        module.validate_auxiliary_channels(bad)


def test_patch_reconstruction_proves_outside_support_unchanged():
    module = _mod()
    source = _master(4, 6)
    patch = np.full((2, 2, 3), 9.0, dtype=np.float32)
    support = np.zeros((4, 6), dtype=np.uint8)
    support[1:3, 2:4] = 1
    out = module.reconstruct_full_frame(
        source, [(patch, (2, 1, 2, 2))], approved_support=support
    )
    assert np.array_equal(out[0], source[0])
    assert np.all(out[1:3, 2:4] == 9.0)
    with pytest.raises(ValueError, match="approved support"):
        module.reconstruct_full_frame(
            source, [(patch, (1, 1, 2, 2))], approved_support=support
        )


def test_generated_patch_reconstruction_only_writes_irregular_support():
    module = _mod()
    source = _master(4, 6)
    patch_pixels = np.full((2, 3, 3), 9.0, dtype=np.float32)
    local_support = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    full_support = np.zeros((4, 6), dtype=np.float32)
    full_support[1:3, 2:5] = local_support
    patch = module.GeneratedPatch(
        patch_pixels,
        (2, 1, 3, 2),
        (0, 0, 6, 4),
        "f" * 64,
        _metadata(),
        np.ones((2, 3), dtype=np.float32),
        local_support,
    )
    out = module.reconstruct_full_frame(source, [patch], approved_support=full_support)
    expected = source.copy()
    region = expected[1:3, 2:5]
    region[local_support > 0] = 9.0
    assert np.array_equal(out, expected)


def test_patch_rejects_stale_source_and_invalid_window(tmp_path):
    module = _mod()
    with pytest.raises(ValueError, match="window"):
        module.write_generated_patch(
            tmp_path / "bad.exr", np.ones((2, 2, 3), dtype=np.float32),
            canvas_size=(4, 4), data_window=(3, 3, 2, 2),
            approved_support=np.ones((2, 2), dtype=np.uint8),
            validity=np.ones((2, 2), dtype=np.float32), source_sha256="c" * 64,
            metadata=_metadata(),
        )
    with pytest.raises(ValueError, match="stale|hash"):
        module.validate_source_hash(b"source", "d" * 64)


def test_card_assets_require_exact_object_to_world_and_object_normals(tmp_path):
    module = _mod()
    normals = _full_aux_channels()
    normals.update({name: np.zeros((2, 2), dtype=np.float32) for name in (
        "N_object.red", "N_object.green", "N_object.blue")}
    )
    normals["N_object.green"][:] = 1
    matrix = np.eye(4, dtype=np.float64)
    path = tmp_path / "card.exr"
    module.write_card_auxiliary_exr(
        path, normals, object_to_world=matrix, metadata=_metadata()
    )
    got = module.read_card_auxiliary_exr(path)
    assert got.object_to_world == tuple(tuple(float(v) for v in row) for row in matrix)
    sparse = tmp_path / "card_sparse.exr"
    module.write_card_auxiliary_exr(
        sparse,
        normals,
        object_to_world=matrix,
        metadata=_metadata(),
        canvas_size=(8, 6),
        data_window=(3, 2, 2, 2),
    )
    sparse_info = module.inspect_exr(sparse)
    assert sparse_info["data_window"] == (3, 2, 2, 2)
    assert sparse_info["display_window"] == (0, 0, 8, 6)
    with pytest.raises(ValueError, match="object_to_world"):
        module.write_card_auxiliary_exr(
            tmp_path / "bad.exr", normals,
            object_to_world=np.zeros((4, 4)), metadata=_metadata()
        )
    with pytest.raises(ValueError, match="required"):
        module.write_card_auxiliary_exr(
            tmp_path / "incomplete.exr", {"N_object.red": normals["N_object.red"], "N_object.green": normals["N_object.green"], "N_object.blue": normals["N_object.blue"]},
            object_to_world=matrix, metadata=_metadata()
        )


def test_readers_reject_wrong_exr_header_contracts(tmp_path):
    module = _mod()
    oiio = module.require_plate_oiio()
    # Deliberately craft an auxiliary with half channels and non-ZIP compression.
    bad_aux = tmp_path / "bad_aux.exr"
    spec = oiio.ImageSpec(2, 2, 1, "half")
    spec.channelnames = ["validity"]
    spec.attribute("compression", "none")
    out = oiio.ImageOutput.create(str(bad_aux)); assert out.open(str(bad_aux), spec)
    out.write_image(np.ones((2, 2, 1), dtype=np.float32)); out.close()
    with pytest.raises(ValueError, match="compression|float32"):
        module.read_auxiliary_exr(bad_aux)

    bad_master = tmp_path / "bad_master.exr"
    spec = oiio.ImageSpec(2, 2, 3, "float")
    spec.channelnames = ["R", "G", "B"]
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(str(bad_master)); assert out.open(str(bad_master), spec)
    out.write_image(np.ones((2, 2, 3), dtype=np.float32)); out.close()
    with pytest.raises(ValueError, match="half|metadata|format"):
        module.read_master_plate(bad_master)


def test_patch_reader_rejects_reordered_channels_and_invalid_windows(tmp_path):
    module = _mod(); oiio = module.require_plate_oiio()
    metadata = _metadata()
    bad = tmp_path / "bad_patch.exr"
    spec = oiio.ImageSpec(2, 2, 5, "float")
    spec.channelnames = ["R", "G", "B", "generated_support", "validity"]
    spec.channelformats = [oiio.TypeDesc("half")] * 3 + [oiio.TypeDesc("float")] * 2
    spec.x, spec.y = -1, 0; spec.full_x, spec.full_y = 0, 0; spec.full_width, spec.full_height = 2, 2
    spec.attribute("compression", "zip")
    spec.attribute("atlas:artifact", "generated_patch")
    spec.attribute("atlas:source_sha256", "b" * 64)
    for key, value in metadata.items(): spec.attribute(key, value)
    out = oiio.ImageOutput.create(str(bad)); assert out.open(str(bad), spec)
    out.write_image(np.ones((2, 2, 5), dtype=np.float32)); out.close()
    with pytest.raises(ValueError, match="channels|window"):
        module.read_generated_patch(bad)
