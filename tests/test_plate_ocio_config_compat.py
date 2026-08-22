"""Colourspace resolution must survive a SWITCH of OCIO config.

Atlas runs on OIIO's built-in ACES config when ``$OCIO`` is unset, and on the
user's studio config when it is set. The two disagree about names, and the
disagreement is not cosmetic — found live on 2026-08-21 while planning the
Photoshop bridge, pointing ``$OCIO`` at
``fn-nuke_cg-config-v1.0.0_aces-v1.3_ocio-v2.1`` broke two names Atlas depends
on, either of which would have taken the whole plate path down:

* ``lin_rec709_scene`` — the tag written on every confined clean plate the
  paint bridges produce. The built-in config carries it as a canonical name;
  fn-nuke_cg carries the same space as ``Linear Rec.709 (sRGB)`` with
  ``lin_rec709_scene`` among its ALIASES. ``list_colorspaces()`` enumerates
  canonical names only, so the alias read as "missing" from a config that
  resolves it perfectly.
* ``sRGB - Display`` — which is ``COMFY_WORKING_COLORSPACE``, the default
  ``output_colorspace`` of every ``read_plate``. fn-nuke_cg declares it as a
  DISPLAY colourspace, and OIIO's ``getNumColorSpaces()`` does not count those,
  so it too read as "missing" — while ``colorconvert`` to it worked correctly
  all along (0.18 ACEScg -> 0.46135, the value recorded in ``plate/__init__``).

Both were resolver-lookup bugs, not capability gaps. These tests pin the fix
against BOTH configs so a future change cannot quietly reintroduce either.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("OpenImageIO")

from atlas_camera.plate import oiio_io  # noqa: E402


# The studio config this was found against. Absent on a fresh clone or CI, so
# every test that needs it skips rather than fails — but the built-in leg still
# runs everywhere, which is what keeps this file honest by default.
STUDIO_CONFIG = Path(r"C:\OCIO\fn-nuke_cg-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio")

# Names Atlas itself emits or depends on, which must resolve under ANY config.
# Each is here because something in the codebase writes or reads it:
#   lin_rec709_scene       - the tag on a confined clean plate
#   Linear Rec.709 (sRGB)  - what AtlasLoadRAW tags its EXR sidecar
#   sRGB - Display         - COMFY_WORKING_COLORSPACE
#   ACEScg / ACES2065-1    - the scene-linear and interchange spaces
LOAD_BEARING_NAMES = [
    "lin_rec709_scene",
    "Linear Rec.709 (sRGB)",
    "sRGB - Display",
    "ACEScg",
    "ACES2065-1",
]


def _reset_config_cache():
    """ColorConfigCache holds ONE config per process, so a test that changes
    $OCIO must drop it or it keeps resolving against the previous config."""
    oiio_io.ColorConfigCache._cfg = None


@pytest.fixture
def studio_config(monkeypatch):
    if not STUDIO_CONFIG.is_file():
        pytest.skip(f"studio OCIO config not on this machine: {STUDIO_CONFIG}")
    monkeypatch.setenv("OCIO", str(STUDIO_CONFIG))
    _reset_config_cache()
    yield STUDIO_CONFIG
    _reset_config_cache()


@pytest.fixture
def builtin_config(monkeypatch):
    monkeypatch.delenv("OCIO", raising=False)
    _reset_config_cache()
    yield
    _reset_config_cache()


@pytest.mark.parametrize("name", LOAD_BEARING_NAMES)
def test_load_bearing_names_resolve_under_builtin_config(builtin_config, name):
    assert oiio_io.resolve_colorspace(name)


@pytest.mark.parametrize("name", LOAD_BEARING_NAMES)
def test_load_bearing_names_resolve_under_studio_config(studio_config, name):
    assert oiio_io.resolve_colorspace(name)


def test_alias_only_name_resolves_to_the_configs_canonical_name(studio_config):
    """`lin_rec709_scene` is an ALIAS in fn-nuke_cg, absent from the enumerated
    names. Resolving it must yield the canonical name, not raise."""
    assert oiio_io.resolve_colorspace("lin_rec709_scene") == "Linear Rec.709 (sRGB)"
    assert "lin_rec709_scene" not in set(oiio_io.list_colorspaces())


def test_display_colorspace_resolves_though_it_is_not_an_enumerated_colorspace(
        studio_config):
    """`sRGB - Display` is a DISPLAY in fn-nuke_cg. It must still resolve,
    because it is the default output colourspace of every plate read."""
    assert oiio_io.resolve_colorspace("sRGB - Display") == "sRGB - Display"
    assert "sRGB - Display" not in set(oiio_io.list_colorspaces())


def test_display_colorspace_actually_converts_under_the_studio_config(studio_config):
    """The resolver may not lie: a name it accepts must really convert.

    0.18 scene-linear through `sRGB - Display` is 0.4614 — the same sanity
    value recorded in atlas_camera/plate/__init__.py, so this pins the NUMBER
    and not merely the absence of an exception.
    """
    np = pytest.importorskip("numpy")
    import OpenImageIO as oiio
    from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec

    buf = ImageBuf(ImageSpec(4, 4, 3, "float"))
    buf.set_pixels(oiio.ROI(), np.full((4, 4, 3), 0.18, dtype="float32"))
    out = ImageBufAlgo.colorconvert(
        buf,
        oiio_io.resolve_colorspace("ACEScg"),
        oiio_io.resolve_colorspace("sRGB - Display"))
    assert not out.has_error, out.geterror()
    assert out.get_pixels(oiio.FLOAT)[0, 0, 0] == pytest.approx(0.4614, abs=1e-3)


def test_unknown_name_still_fails_loudly(studio_config):
    """The alias and display fallbacks must not turn a typo into a silent pass;
    a mis-set colourspace has to fail at the call site."""
    with pytest.raises(RuntimeError, match="not in the active OCIO config"):
        oiio_io.resolve_colorspace("totally_bogus_space")


def test_config_identity_reports_the_builtin_honestly(builtin_config):
    """The built-in config has no file, so an empty path and sha256 are the
    honest answer — a manifest must not claim a hash it does not have."""
    identity = oiio_io.config_identity()
    assert identity["available"] is True
    assert identity["config_path"] == ""
    assert identity["config_sha256"] == ""
    assert identity["n_colorspaces"] > 0


def test_config_identity_hashes_a_real_config_file(studio_config):
    """A colourspace NAME is not a contract: the paint bridges record this
    identity so two scores are only ever compared under the same config."""
    identity = oiio_io.config_identity()
    assert identity["config_path"] == str(STUDIO_CONFIG)
    assert len(identity["config_sha256"]) == 64
    assert identity["n_displays"] > 0
    # configname is a METHOD on the binding; reading it as an attribute yields a
    # bound-method repr, which once landed in a manifest as the config's name.
    assert "bound method" not in identity["config_name"]


def test_switching_config_changes_the_recorded_identity(monkeypatch):
    """Guards the cache-reset contract: without dropping ColorConfigCache, a
    process keeps resolving against the config it first saw."""
    if not STUDIO_CONFIG.is_file():
        pytest.skip("studio OCIO config not on this machine")

    monkeypatch.delenv("OCIO", raising=False)
    _reset_config_cache()
    builtin = oiio_io.config_identity()

    monkeypatch.setenv("OCIO", str(STUDIO_CONFIG))
    _reset_config_cache()
    studio = oiio_io.config_identity()

    _reset_config_cache()
    assert builtin["config_sha256"] != studio["config_sha256"]
    assert builtin["n_colorspaces"] != studio["n_colorspaces"]


def test_a_written_plate_self_describes_under_a_studio_config(studio_config,
                                                              tmp_path):
    """Every EXR Atlas writes must carry its colourspace under ANY config.

    OIIO only persists the standard ``oiio:ColorSpace`` attribute when the
    active config can supply a ``colorInteropID`` for the space. The built-in
    config can; fn-nuke_cg v1.0.0 cannot, and OIIO then silently dropped the
    tag — so under a studio config every plate Atlas wrote was UNTAGGED.

    The tail is what makes it serious rather than cosmetic: an untagged plate
    read on 'auto' falls back to guessing from the extension, and .exr guesses
    ACEScg. A RAW sidecar is Linear Rec.709, so it would come back read as
    ACEScg and the colour would be quietly wrong — exactly the failure
    docs/USER_GUIDE.md warns about. Atlas therefore writes its own tag too.
    """
    np = pytest.importorskip("numpy")

    for space in ("ACEScg", "Linear Rec.709 (sRGB)"):
        path = tmp_path / f"{space.replace(' ', '_').replace('.', '')}.exr"
        oiio_io.write_exr(str(path), np.full((4, 4, 3), 0.5, "float32"),
                          source_colorspace=space)
        got = oiio_io.read_plate(str(path), raw_data=True)
        assert got.input_colorspace, f"{space}: plate written with NO colourspace"
        assert (oiio_io.resolve_colorspace(got.input_colorspace)
                == oiio_io.resolve_colorspace(space))

        # And the tag must still beat a conflicting explicit hint, which is the
        # property the paint bridges' re-tag defence depends on.
        hinted = oiio_io.read_plate(str(path), input_colorspace="sRGB - Display",
                                    raw_data=True)
        assert hinted.input_colorspace == got.input_colorspace


def test_the_atlas_tag_is_written_even_when_oiio_drops_the_standard_one(
        studio_config, tmp_path):
    """Pins the mechanism, not just the outcome: if OIIO later starts writing
    oiio:ColorSpace under every config, this test still passes; if Atlas stops
    writing its own tag, it fails."""
    np = pytest.importorskip("numpy")
    import OpenImageIO as oiio

    path = tmp_path / "tagged.exr"
    oiio_io.write_exr(str(path), np.full((4, 4, 3), 0.5, "float32"),
                      source_colorspace="ACEScg")
    src = oiio.ImageInput.open(str(path))
    attribs = {a.name: a.value for a in src.spec().extra_attribs}
    src.close()
    assert oiio_io.ATLAS_COLORSPACE_ATTR in attribs
