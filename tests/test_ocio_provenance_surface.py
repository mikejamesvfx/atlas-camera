"""Atlas must always be able to say WHICH OCIO config it is converting with.

Atlas ships no OCIO config of its own. There is exactly one place a config is
constructed — ``plate/oiio_io.py``, ``oiio.ColorConfig()`` with no path — so
OIIO's own resolution order governs: ``$OCIO`` when set, otherwise OIIO's
built-in ACES config. (It is emphatically NOT Blender's bundled config;
nothing in Atlas reads that, and a grep suggesting otherwise is matching the
substring "aces" inside "faces".)

That makes a colourspace NAME meaningless on its own: `ACEScg` under two
different configs is two different plates. So every report and payload names
the config in force, and the one UI field that LOOKS like it selects a config
but does not is required to say so and to flag a disagreement.
"""
from __future__ import annotations

import pytest


def test_active_identity_names_a_config_or_reports_unavailable():
    from atlas_camera.comfy.nodes_viewport import active_ocio_identity

    identity = active_ocio_identity()
    # {} is legitimate: the core package has no required dependencies, so a
    # machine without [oiio] genuinely has no config. What is NOT allowed is
    # raising, because this runs inside a debug report.
    assert isinstance(identity, dict)
    if identity.get("available"):
        assert identity["config_name"], "an available config must be named"
        assert identity["n_colorspaces"] > 0


def test_report_line_names_the_live_config():
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy.nodes_viewport import _active_ocio_lines

    line, flag = _active_ocio_lines(None)
    assert line.startswith("ocio")
    assert "spaces" in line
    # With no declared config_path there is nothing to disagree with.
    assert flag == ""


def test_a_declared_config_path_that_is_not_in_force_is_FLAGGED():
    """`AtlasViewportControls.config_path` is DCC-handoff metadata and is never
    wired to the resolver. A user who types a path there and sees no
    contradiction would reasonably believe it took effect — so when it
    disagrees with the config actually in use, the report must say so."""
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy.nodes_viewport import _active_ocio_lines

    class _Profile:
        config_path = r"C:\somewhere\else\studio-config.ocio"

    line, flag = _active_ocio_lines(_Profile())
    assert line
    assert flag, "a config_path that is not in force must be flagged"
    assert "does NOT select" in flag
    assert "$OCIO" in flag, "the flag must say how to actually change it"


def test_no_flag_when_the_declared_path_matches_what_is_in_force(monkeypatch,
                                                                tmp_path):
    """The flag must fire on DISAGREEMENT, not merely on the field being set —
    otherwise it is noise and gets ignored."""
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy import nodes_viewport
    from atlas_camera.plate import oiio_io

    config = tmp_path / "cfg.ocio"
    config.write_text("ocio_profile_version: 2\n", encoding="utf-8")

    monkeypatch.setattr(nodes_viewport, "active_ocio_identity",
                        lambda: {"available": True, "config_name": str(config),
                                 "config_path": str(config), "config_sha256": "",
                                 "n_colorspaces": 3, "n_displays": 1})

    class _Profile:
        config_path = str(config)

    _line, flag = nodes_viewport._active_ocio_lines(_Profile())
    assert flag == ""
    assert oiio_io is not None


def test_config_path_widget_declares_itself_metadata_only():
    """The tooltip is the only thing standing between a user and the belief
    that this field selects their config. It must keep saying so."""
    from atlas_camera.comfy.nodes_viewport import AtlasViewportControls

    spec = AtlasViewportControls.INPUT_TYPES()["optional"]["config_path"]
    tooltip = (spec[1] or {}).get("tooltip", "")
    assert tooltip, "config_path must carry a tooltip"
    assert "does not select" in tooltip.lower()
    assert "$OCIO" in tooltip


def test_viewport_payload_records_the_config_in_camera_meta():
    """The browser and the MCP census both read camera_meta, so recording it
    there is what makes the config visible without a round trip."""
    pytest.importorskip("OpenImageIO")
    from atlas_camera.comfy.viewport_payload import _active_ocio_identity

    identity = _active_ocio_identity()
    assert isinstance(identity, dict)
    if identity.get("available"):
        assert "config_sha256" in identity
        assert "config_path" in identity


def test_identity_never_raises_when_openimageio_is_missing(monkeypatch):
    """A colour-provenance line must not be able to fail a debug report."""
    import builtins

    from atlas_camera.comfy import nodes_viewport

    real_import = builtins.__import__

    def _no_oiio(name, *args, **kwargs):
        if "oiio_io" in name or name == "OpenImageIO":
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_oiio)
    assert nodes_viewport.active_ocio_identity() == {}
    line, flag = nodes_viewport._active_ocio_lines(None)
    assert "unavailable" in line
    assert flag == ""
