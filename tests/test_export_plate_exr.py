"""AtlasExportPlateEXR 📤 — file-to-file OCIO conversion to a tagged ACEScg EXR.

The node is pure orchestration over plate.read_plate/write_exr; these pin the
contract: converted file tagged oiio:ColorSpace=ACEScg, pixels actually
transformed, plate_ref updated (non-proxy, 16f), and the proxy/missing-file
soft-fail that must never kill an export chain.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.schema import AtlasPlateRef


def _oiio():
    return pytest.importorskip("OpenImageIO")


def _linear_fixture(tmp_path):
    """A tiny scene-linear Rec.709 EXR written through the real writer."""
    from atlas_camera.plate import write_exr

    px = np.zeros((8, 8, 3), dtype=np.float32)
    px[..., 0] = 0.5   # saturated-ish red patch: primaries conversion must move it
    px[2:6, 2:6] = (0.1, 0.4, 0.9)
    path = tmp_path / "src_linear.exr"
    write_exr(str(path), px, bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)")
    return path, px


def test_export_plate_exr_converts_and_tags(tmp_path):
    _oiio()
    from atlas_camera.comfy.nodes_export import AtlasExportPlateEXR

    src, px = _linear_fixture(tmp_path)
    ref = AtlasPlateRef(image_path=str(src), colorspace="Linear Rec.709 (sRGB)",
                        bit_depth="32f", role="source", is_proxy=False)
    exr_path, new_ref, report = AtlasExportPlateEXR().export(
        ref, output_colorspace="ACEScg", output_dir=str(tmp_path / "out"))

    assert exr_path.endswith("_acescg.exr")
    import OpenImageIO as oiio
    buf = oiio.ImageBuf(exr_path)
    # OIIO canonicalizes the tag to the config's internal name for ACEScg
    # (ACES 2.0 built-in config: 'lin_ap1_scene'); either spelling identifies
    # linear AP1 and resolves back through the same config in a DCC.
    tag = buf.spec().get_string_attribute("oiio:ColorSpace")
    assert tag in ("ACEScg", "lin_ap1_scene", "lin_ap1"), tag
    out_px = buf.get_pixels(oiio.FLOAT)[..., :3]
    # Rec.709 -> ACEScg is a real primaries change: pixels must move but stay finite.
    assert np.isfinite(out_px).all()
    assert not np.allclose(out_px, px, atol=1e-4), "conversion must change pixel values"
    # plate_ref contract for the DCC handoff
    assert new_ref.image_path == exr_path
    assert new_ref.colorspace == "ACEScg"
    assert new_ref.bit_depth == "16f"
    assert new_ref.is_proxy is False
    assert new_ref.metadata["converted_from"] == str(src)
    assert "ACEScg" in report


def test_export_plate_exr_proxy_soft_fail(tmp_path):
    from atlas_camera.comfy.nodes_export import AtlasExportPlateEXR

    ref = AtlasPlateRef(image_path=None, is_proxy=True)
    exr_path, out_ref, report = AtlasExportPlateEXR().export(
        ref, output_dir=str(tmp_path))
    assert exr_path == ""
    assert out_ref is ref, "input ref passes through untouched"
    assert "SKIPPED" in report
