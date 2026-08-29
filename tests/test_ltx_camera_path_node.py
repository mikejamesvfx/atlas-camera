"""AtlasLoadCameraPath reads what Atlas Director actually writes.

The fixtures are real output from `app/director/export/ltx.ts`, not JSON typed
out here, so a change to the writer's shape fails on this side instead of
surfacing as a wrong render in ComfyUI. The three formats mean three different
things about how far the picture can be trusted, and the node's job is to keep
them distinguishable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_camera.comfy.nodes_ltx import AtlasLoadCameraPath

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE = FIXTURES / "atlas_ltx_single.json"
CHAIN = FIXTURES / "atlas_ltx_chain.json"
COMPRESSED = FIXTURES / "atlas_ltx_compressed.json"


def load(path, **kwargs):
    return AtlasLoadCameraPath().load(str(path), **kwargs)


def test_a_single_path_yields_the_widget_string():
    keyframes, report, count = load(SINGLE)
    knots = json.loads(keyframes)
    assert count == 1
    assert len(knots) > 1
    # The node parses this string; anything Atlas adds must stay outside it.
    assert set(knots[0]) == {"f", "az", "el", "dist", "px", "py", "pz"}
    assert knots[0]["f"] == 1


def test_the_report_leads_with_the_settings_that_fail_silently():
    """pivot_override off and keep_source_aim on both render a DIFFERENT move.

    Neither raises anything anywhere, which is exactly why they are the first
    thing the report says rather than a footnote under the residuals.
    """
    _, report, _ = load(SINGLE)
    head = report.split("Carried exactly")[0]
    assert "use_keyframes   = True" in head
    assert "pivot_override  = True" in head
    assert "keep_source_aim = False" in head


def test_the_report_states_the_loss_rather_than_only_the_fidelity():
    _, report, _ = load(SINGLE)
    assert "Not carried" in report
    assert "roll" in report
    assert "orbit" in report.lower()


def test_a_chain_reports_its_length_and_serves_a_segment():
    keyframes, report, count = load(CHAIN, segment=1)
    assert count > 1
    assert json.loads(keyframes)[0]["f"] == 1
    assert "SEGMENT 1 of" in report


def test_every_segment_after_the_first_says_its_source_is_generated():
    """The hand-off is the part a user gets wrong.

    Segment 2 is conditioned on segment 1's final GENERATED frame, so its
    geometry must be re-derived from an invented image. Silence here reads as
    'feed it the plate again', which quietly breaks the chain.
    """
    _, first, count = load(CHAIN, segment=1)
    assert "PREVIOUS segment" not in first
    for index in range(2, count + 1):
        _, report, _ = load(CHAIN, segment=index)
        assert "PREVIOUS segment's final generated frame" in report


def test_a_segment_past_the_end_is_refused_with_the_count():
    _, _, count = load(CHAIN, segment=1)
    with pytest.raises(ValueError, match=f"chain of {count} segment"):
        load(CHAIN, segment=count + 1)


def test_a_compressed_path_announces_the_scale_it_lost():
    """A third-depth push must never read as the take.

    The scale and both travels are in the report because the keyframes alone
    look exactly like an honest short push.
    """
    _, report, _ = load(COMPRESSED)
    assert "COMPRESSED" in report
    assert "previz only" in report
    assert "parallax is not" in report
    document = json.loads(COMPRESSED.read_text(encoding="utf-8"))
    assert f"{document['compression']['scale']:.3f}" in report


def test_an_unknown_format_is_refused_rather_than_guessed():
    """The reason the format id is stamped at all.

    A compressed path read as an ordinary one renders a scaled-down move with
    nothing anywhere saying so, so an id this node does not know is an error.
    """
    stray = FIXTURES / "atlas_ltx_unknown.json"
    stray.write_text(json.dumps({"format": "atlas.ltx.something_later"}), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="does not know"):
            load(stray)
    finally:
        stray.unlink()


def test_a_missing_file_says_how_to_make_one():
    with pytest.raises(ValueError, match="Export a take with the 'ltx' target"):
        load(FIXTURES / "no_such_path.json")


def test_a_quoted_windows_path_is_accepted():
    # Copying a path out of Explorer brings the quotes with it.
    keyframes, _, _ = load(f'"{SINGLE}"')
    assert json.loads(keyframes)


def test_the_node_declares_what_it_returns():
    assert AtlasLoadCameraPath.RETURN_NAMES == ("keyframes", "report", "segment_count")
    assert AtlasLoadCameraPath.FUNCTION == "load"
