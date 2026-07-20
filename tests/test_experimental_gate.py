"""Tests for the ATLAS_EXPERIMENTAL registration gate.

Main ships ATLAS_EXPERIMENTAL_DEFAULT="0" (experimental nodes hidden — a
stock ComfyUI's node menu stays universal); the `experimental` branch flips
that single constant to "1". Either way the env var overrides at launch.
The gate merges at import time, so these tests exercise the helper + dicts
rather than re-importing the module under different environments.
"""

import atlas_camera.comfy.nodes as nodes


def test_experimental_dicts_cover_exactly_the_experimental_nodes():
    assert set(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == {
        "AtlasPredictHiddenGeometry", "AtlasRenderFix",
        "AtlasExtractAnglePatch", "AtlasImportAnglePatch",
        "AtlasWorkflowGenerator", "AtlasMegaPipeline"}
    assert set(nodes.EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS) == set(
        nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS)
    for name in nodes.EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS.values():
        assert "🔬" in name  # the tier is visibly labeled in the menu


def test_gate_helper_truthiness(monkeypatch):
    for off in ("", "0", "false", "OFF", "no"):
        monkeypatch.setenv("ATLAS_EXPERIMENTAL", off)
        assert nodes._experimental_enabled() is False, off
    for on in ("1", "true", "yes", "on", "anything"):
        monkeypatch.setenv("ATLAS_EXPERIMENTAL", on)
        assert nodes._experimental_enabled() is True, on
    monkeypatch.delenv("ATLAS_EXPERIMENTAL", raising=False)
    expected = nodes.ATLAS_EXPERIMENTAL_DEFAULT not in ("", "0")
    assert nodes._experimental_enabled() is expected


def test_registration_state_matches_the_gate():
    # Whatever environment this suite runs under, the merged mappings must
    # agree with the gate's verdict — no half-registered states.
    registered = "AtlasRenderFix" in nodes.NODE_CLASS_MAPPINGS
    assert registered == nodes._experimental_enabled()
    assert ("AtlasPredictHiddenGeometry" in nodes.NODE_CLASS_MAPPINGS) == registered
    if registered:
        for k in nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS:
            assert k in nodes.NODE_DISPLAY_NAME_MAPPINGS


def test_core_nodes_unaffected_by_gate():
    for core in ("AtlasBlockoutViewport", "AtlasAddPatchView",
                 "AtlasCleanPlateLayer", "AtlasLearnedSolveFromImage"):
        assert core in nodes.NODE_CLASS_MAPPINGS
