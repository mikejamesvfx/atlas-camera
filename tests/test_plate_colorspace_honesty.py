"""A source colourspace is either KNOWN or ABSENT — never invented.

Found live 2026-08-07 on ``DSC_2190.NEF``: a delivered .nk tagged its Read
``ACEScg`` while the EXR on disk was ``lin_rec709_scene``. Rec.709 primaries
were being interpreted as AP1, silently.

The cause was ``primary_plate_colorspace`` falling back to
``output_profile.working_colorspace`` (default ``ACEScg``) when the plate had
no colourspace of its own. That conflates two genuinely different things: the
COMP WORKING SPACE the artist wants to end up in, and the SOURCE FILE's own
space. They coincide often enough for the bug to hide, and when they diverge
the result is a wrong tag rather than a missing one.

An absent tag is strictly better than a wrong one. Absent makes the DCC fall
back to its own default and the artist SEES an unspecified colourspace; wrong
looks authoritative and gets believed.
"""

import ast

import pytest

from atlas_camera.core.schema import AtlasOutputProfile, AtlasPlateRef
from atlas_camera.exporters._plate import primary_plate_colorspace
from atlas_camera.exporters.nuke_exporter import (
    write_nuke_native_script,
    write_nuke_projection_script,
)


def _with_profile(solve, working="ACEScg"):
    """Attach an output profile whose working space is the trap value.

    Constructed, not skipped-around: the whole bug lives in the branch that
    only runs when an output_profile EXISTS, so a fixture without one would
    make every assertion below pass vacuously.
    """
    solve.output_profile = AtlasOutputProfile(working_colorspace=working)
    return solve


def test_a_plates_own_colorspace_is_returned(make_atlas_solve):
    solve = make_atlas_solve()
    solve.source_plate = AtlasPlateRef(image_path="/x/plate.exr",
                                       colorspace="Linear Rec.709 (sRGB)")
    assert primary_plate_colorspace(solve) == "Linear Rec.709 (sRGB)"


def test_no_plate_colorspace_returns_none_not_the_working_space(make_atlas_solve):
    """The regression itself. With no plate colourspace the answer is None —
    NOT the comp working space, which is what tagged a Rec.709 file ACEScg."""
    solve = _with_profile(make_atlas_solve(), working="ACEScg")
    solve.source_plate = AtlasPlateRef(image_path="/x/plate.exr", colorspace=None)
    assert primary_plate_colorspace(solve) is None


def test_no_plate_at_all_returns_none(make_atlas_solve):
    solve = _with_profile(make_atlas_solve(), working="ACEScg")
    solve.source_plate = None
    assert primary_plate_colorspace(solve) is None


def test_the_working_space_is_never_substituted(make_atlas_solve):
    """Explicit: an output profile carrying ACEScg must not leak into the
    source tag by any route."""
    solve = make_atlas_solve()
    solve.source_plate = None
    _with_profile(solve, working="ACEScg")
    assert primary_plate_colorspace(solve) != "ACEScg"


def test_native_nk_omits_the_colorspace_line_when_unknown(tmp_path, make_atlas_solve):
    """A .nk with no ` colorspace ` line lets Nuke apply its own default and
    shows the artist an untouched knob. Writing one we guessed does not."""
    solve = _with_profile(make_atlas_solve(), working="ACEScg")
    solve.source_plate = None
    path = write_nuke_native_script(solve, tmp_path / "scene.nk")
    script = path.read_text(encoding="utf-8")
    assert " colorspace " not in script
    assert "unspecified" in script


def test_python_nk_writer_still_parses_with_an_unknown_colorspace(
        tmp_path, make_atlas_solve):
    solve = _with_profile(make_atlas_solve(), working="ACEScg")
    solve.source_plate = None
    script = write_nuke_projection_script(
        solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    ast.parse(script)          # the None must serialise, not crash the writer
    assert "source_colorspace = None" in script


def test_manifest_records_null_rather_than_the_working_space(make_atlas_solve):
    from atlas_camera.exporters.manifest import _plate_info
    solve = make_atlas_solve()
    solve.source_plate = None
    _with_profile(solve, working="ACEScg")
    assert _plate_info(solve)["colorspace"] is None


# --- resolving the name against the target config ----------------------------
#
# Second half of the same live failure: the .nk wrote `source_colorspace`
# VERBATIM into ` colorspace <name>`. When that literal is absent from the
# user's Nuke OCIO config, Nuke silently falls back to its scene_linear
# default — which is how the delivered script showed `Read1 (scene_linear)`
# even where a name had been written. Resolving through OCIO roles and known
# cross-config aliases first is strictly better than a raw literal.


def test_resolved_or_verbatim_passes_none_through():
    from atlas_camera.exporters._plate import _resolved_or_verbatim
    assert _resolved_or_verbatim(None) is None
    assert _resolved_or_verbatim("") is None


def test_resolved_or_verbatim_returns_the_resolved_name(monkeypatch):
    import atlas_camera.plate as plate_mod
    monkeypatch.setattr(plate_mod, "resolve_colorspace",
                        lambda n: "Utility - Linear - Rec.709", raising=False)
    from atlas_camera.exporters._plate import _resolved_or_verbatim
    assert _resolved_or_verbatim("Linear Rec.709 (sRGB)") == \
        "Utility - Linear - Rec.709"


def test_resolved_or_verbatim_falls_back_to_the_literal_when_unresolvable(monkeypatch):
    """Soft guard. No OIIO, no config, or a name this config cannot place must
    degrade to the artist's own string — never to an exception, and never to a
    substituted space. An export is not allowed to fail over a colour NAME."""
    import atlas_camera.plate as plate_mod

    def boom(_name):
        raise RuntimeError("no OCIO config here")

    monkeypatch.setattr(plate_mod, "resolve_colorspace", boom, raising=False)
    from atlas_camera.exporters._plate import _resolved_or_verbatim
    assert _resolved_or_verbatim("Weird Studio Space") == "Weird Studio Space"


def test_native_nk_writes_the_resolved_name(tmp_path, make_atlas_solve, monkeypatch):
    import atlas_camera.plate as plate_mod
    monkeypatch.setattr(plate_mod, "resolve_colorspace",
                        lambda n: "ACES - ACEScg", raising=False)
    solve = make_atlas_solve()
    solve.source_plate = AtlasPlateRef(image_path="/x/plate.exr",
                                       colorspace="ACEScg")
    script = write_nuke_native_script(
        solve, tmp_path / "scene.nk").read_text(encoding="utf-8")
    assert " colorspace ACES - ACEScg\n" in script


# --- the file's own tag vs the declared one ----------------------------------
#
# plate.write_exr already stamps oiio:ColorSpace on every EXR it writes, and
# until now literally no consumer read it back. When the declared plate_ref
# colourspace and the file's own attribute disagree, ONE of them is wrong and
# the export is the last place anyone could notice.


def test_plate_file_colorspace_is_none_for_an_unreadable_file(tmp_path):
    from atlas_camera.exporters._plate import plate_file_colorspace
    assert plate_file_colorspace(str(tmp_path / "nope.exr")) is None
    assert plate_file_colorspace(None) is None


def test_plate_file_colorspace_never_raises(tmp_path):
    """Reading provenance must not be able to fail an export."""
    from atlas_camera.exporters._plate import plate_file_colorspace
    junk = tmp_path / "junk.exr"
    junk.write_bytes(b"not an exr at all")
    assert plate_file_colorspace(str(junk)) is None


def test_the_sticky_note_flags_a_disagreement(tmp_path, make_atlas_solve,
                                              monkeypatch):
    import atlas_camera.exporters.nuke_exporter as nuke_mod
    monkeypatch.setattr(nuke_mod, "plate_file_colorspace",
                        lambda _p: "lin_rec709_scene")
    solve = make_atlas_solve()
    solve.source_plate = AtlasPlateRef(image_path="/x/plate.exr",
                                       colorspace="ACEScg")
    script = write_nuke_native_script(
        solve, tmp_path / "scene.nk").read_text(encoding="utf-8")
    assert "lin_rec709_scene" in script
    assert "MISMATCH" in script


def test_the_sticky_note_is_quiet_when_they_agree(tmp_path, make_atlas_solve,
                                                  monkeypatch):
    import atlas_camera.exporters.nuke_exporter as nuke_mod
    monkeypatch.setattr(nuke_mod, "plate_file_colorspace",
                        lambda _p: "ACEScg")
    solve = make_atlas_solve()
    solve.source_plate = AtlasPlateRef(image_path="/x/plate.exr",
                                       colorspace="ACEScg")
    script = write_nuke_native_script(
        solve, tmp_path / "scene.nk").read_text(encoding="utf-8")
    assert "MISMATCH" not in script
