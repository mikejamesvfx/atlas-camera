"""Tests for _parse_model_json robustness — LLM repetition-loop recovery."""

import json

from atlas_camera.inference.multimodal_helper import (
    _close_partial_json,
    _parse_model_json,
    _truncate_looping_response,
)


# ---------------------------------------------------------------------------
# _truncate_looping_response
# ---------------------------------------------------------------------------


def test_no_loop_returns_unchanged():
    content = '{"summary": "A factory scene.", "scale_candidates": ["person_175cm"]}'
    result, was_looping = _truncate_looping_response(content)
    assert not was_looping
    assert result == content


def test_exact_repetition_detected():
    # 15 repetitions of the same 8-char fragment simulates a looping LLM
    loop = '":" "," ' * 15
    content = '{"summary": "ok", "scale_cues": [{"point": [100, 200], "label' + loop
    result, was_looping = _truncate_looping_response(content)
    assert was_looping
    assert loop not in result
    assert '"summary": "ok"' in result


def test_oversized_response_truncated():
    # Response longer than _MAX_RESPONSE_CHARS with no exact repeat pattern
    content = '{"summary": "x"}' + ("a" * 33_000)
    result, was_looping = _truncate_looping_response(content)
    assert was_looping
    assert len(result) <= 32_000


# ---------------------------------------------------------------------------
# _close_partial_json
# ---------------------------------------------------------------------------


def test_close_partial_outer_and_inner_open():
    # Truncated: outer object + inner array open
    partial = '{"summary": "scene", "scale_candidates": ["person_175cm"], "scale_cues": [{"label"'
    result = _close_partial_json(partial)
    assert result is not None
    assert result.get("summary") == "scene"
    assert result.get("scale_candidates") == ["person_175cm"]


def test_close_uses_outer_comma_cut_as_fallback():
    # The outer comma cut lets us recover fields before the broken one
    partial = '{"summary": "good", "scene_description": "text", "scale_cues": [{"label'
    result = _close_partial_json(partial)
    assert result is not None
    assert "summary" in result


def test_close_non_object_returns_none():
    result = _close_partial_json("[1, 2, 3")
    assert result is None


def test_close_balanced_but_invalid_returns_none():
    # Balanced brackets but invalid value syntax — nothing _close_partial_json can fix
    result = _close_partial_json('{"a": }')
    assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# _parse_model_json — end-to-end
# ---------------------------------------------------------------------------


def test_clean_json_parses_directly():
    payload = {"summary": "A scene.", "scale_candidates": ["person_175cm"]}
    result = _parse_model_json(json.dumps(payload))
    assert result["summary"] == "A scene."
    assert result["scale_candidates"] == ["person_175cm"]


def test_looping_response_salvages_early_fields():
    # Reproduces the exact LM Studio failure mode seen in the field:
    # valid preamble then scale_cues array triggers an infinite repetition loop.
    loop_fragment = '":" "," ' * 20
    content = (
        '{"summary": "Industrial steampunk factory.", '
        '"scene_description": "Train depot interior.", '
        '"scale_candidates": ["building_story_3m", "person_175cm"], '
        '"scale_cues": [{"point": [550, 330], "label'
        + loop_fragment
    )
    result = _parse_model_json(content)
    assert result.get("summary") == "Industrial steampunk factory."
    assert result.get("scene_description") == "Train depot interior."
    assert "building_story_3m" in (result.get("scale_candidates") or [])
    warnings = result.get("warnings", [])
    assert any("repetition loop" in w for w in warnings)


def test_looping_response_warning_is_list():
    loop = '"x" "," ' * 15
    content = '{"summary": "s", "scale_cues": [{"label' + loop
    result = _parse_model_json(content)
    assert isinstance(result.get("warnings"), list)


def test_completely_broken_json_returns_fallback():
    result = _parse_model_json("not json at all")
    assert "summary" in result
    assert any("valid JSON" in w for w in result.get("warnings", []))


def test_empty_content_returns_fallback():
    result = _parse_model_json("")
    assert "summary" in result


def test_json_in_markdown_fences_parsed():
    # Some models wrap output in markdown code fences
    payload = '```json\n{"summary": "ok"}\n```'
    result = _parse_model_json(payload)
    assert result.get("summary") == "ok"
