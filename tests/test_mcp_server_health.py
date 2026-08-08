"""atlas_health capability manifest + licence metadata (2026-08-08 hygiene
pass, from the deep-research report's month-one recommendations).

An agent integrating a registry-installed Atlas must be able to ask "what can
this install actually do, and under which licences?" from ONE call, without
interpreting docs — including the explicit camera_move_bake gap (browser-only,
absent from MCP by design). Offline: ComfyUI HTTP is monkeypatched.

Needs the mcp SDK (same import the server itself needs); skipped without it.
"""
import json
import re

import pytest

pytest.importorskip("mcp")

from atlas_camera.mcp import server as S


@pytest.fixture()
def health(monkeypatch):
    monkeypatch.setattr(S.C, "http_json", lambda *a, **k: {
        "system": {"comfyui_version": "0.27.0"}, "devices": []})
    monkeypatch.setattr(S.C, "fetch_object_info", lambda host: {
        "AtlasInput": {}, "AtlasBlockoutViewport": {}})
    return json.loads(S.atlas_health())


def test_health_carries_a_capability_manifest(health):
    caps = health.get("capabilities")
    assert caps, "atlas_health has no capabilities block"
    # The one browser-bound gap must be machine-readable, not a docs footnote.
    assert caps["camera_move_bake"] == "browser_only"
    assert set(caps["dcc_exporters"]) >= {"nuke", "maya", "blender", "usd"}
    assert "atlas_health" in caps["mcp_tools"]


def test_manifest_tool_list_matches_the_decorated_tools(health):
    """The static list must track the @mcp.tool() surface — a new tool that
    forgets the manifest, or a stale entry, fails here."""
    src = open(S.__file__, encoding="utf-8").read()
    decorated = set(re.findall(r"@mcp\.tool\(\)\s*\ndef (\w+)", src))
    assert set(health["capabilities"]["mcp_tools"]) == decorated


def test_health_separates_code_licence_from_model_licences(health):
    lic = health.get("licences")
    assert lic, "atlas_health has no licences block"
    assert lic["atlas_camera"] == "MIT"
    models = lic["models"]
    # The two facts INSTALL.md warns about must be machine-readable: gated
    # SAM3 and the non-commercial DA3 giant weights.
    assert models["sam3"]["gated"] is True
    assert "NC" in models["depth_anything_3"]["weights"]
    assert models["moge_2"]["licence"] == "MIT"
