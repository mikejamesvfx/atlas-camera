"""Colour-management regression tests for atlas_camera.plate.oiio_io.

Guarded by ``importorskip`` because the colour path rides the optional [oiio]
extra — the same contract as the rest of the OIIO-dependent code, so a bare dev
venv skips these rather than erroring. Run them where OpenImageIO is installed
(any ComfyUI venv, or ``pip install -e .[oiio]``).

These lock two things the beta relied on implicitly:
  * that Atlas's canonical space names resolve into whatever OCIO config is live
    (built-in ACES *or* a user's $OCIO studio config), via role/alias resolution;
  * that the Linear Rec.709 <-> ACEScg conversion is a real, invertible primaries
    change and not a silent no-op or a lossy detour.
"""

from __future__ import annotations

import pytest

pytest.importorskip("OpenImageIO")

from atlas_camera.plate.oiio_io import (  # noqa: E402  (after importorskip by design)
    list_colorspaces,
    read_plate,
    resolve_colorspace,
    write_exr,
)


def test_canonical_names_resolve_into_active_config():
    """Every space Atlas asks for by name must resolve to one the config knows."""
    spaces = set(list_colorspaces())
    assert spaces, "active OCIO config exposed no colourspaces"
    for name in ("ACEScg", "ACES2065-1", "Linear Rec.709 (sRGB)"):
        resolved = resolve_colorspace(name)
        assert resolved in spaces, f"{name!r} resolved to {resolved!r}, not in config"


def test_scene_linear_role_matches_acescg_name():
    """Role-based resolution is the point of the fix: ``scene_linear`` must land on
    the same space Atlas requests as ``ACEScg`` in any ACES config. A config that
    defines no such role is a legitimate (if unusual) setup, so skip rather than
    fail there."""
    try:
        via_role = resolve_colorspace("scene_linear")
    except RuntimeError:
        pytest.skip("active OCIO config defines no scene_linear role")
    assert via_role == resolve_colorspace("ACEScg")


def test_rec709_to_acescg_roundtrips(tmp_path):
    """Linear Rec.709 -> ACEScg -> Linear Rec.709 must return the original within
    float tolerance, and the forward step must actually MOVE a saturated colour
    (a neutral grey is near-invariant and would prove nothing)."""
    import numpy as np

    src = np.array([[[0.8, 0.2, 0.05]]], dtype="float32")  # 1x1x3 HxWx3, saturated

    acescg_path = str(tmp_path / "patch_acescg.exr")
    write_exr(acescg_path, src, bit_depth="float",
              source_colorspace="Linear Rec.709 (sRGB)", output_colorspace="ACEScg")

    acescg = read_plate(acescg_path, raw_data=True)

    tag = acescg.metadata.get("oiio:ColorSpace")
    if tag:  # the writer tags the resolved output name; it must mean ACEScg
        assert resolve_colorspace(tag) == resolve_colorspace("ACEScg")

    assert not np.allclose(acescg.pixels[0, 0], src[0, 0], atol=1e-4), (
        "Rec.709 -> ACEScg produced no change on a saturated colour — "
        "the conversion silently no-op'd")

    back_path = str(tmp_path / "patch_back.exr")
    write_exr(back_path, acescg.pixels, bit_depth="float",
              source_colorspace="ACEScg", output_colorspace="Linear Rec.709 (sRGB)")
    back = read_plate(back_path, raw_data=True)

    assert np.allclose(back.pixels[0, 0], src[0, 0], atol=1e-3), (
        f"round-trip drift too large: {back.pixels[0, 0]} vs {src[0, 0]}")


def test_unknown_colorspace_raises_clearly():
    """A name that maps to no space, role, or alias must fail loudly at the call
    site rather than silently mis-tagging a plate."""
    with pytest.raises(RuntimeError):
        resolve_colorspace("Definitely Not A Real Colourspace 12345")
