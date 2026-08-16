"""A skipped manifest must reach the artist, not just ComfyUI's console.

atlas_project.json is the P0 trust artifact — scale_health, the confidence
vector, export provenance. The rule that a manifest failure must never fail an
export is correct and stays; what was missing is that the failure only went to
`logging.warning`, so an export that shipped without one looked exactly like a
complete delivery. A headless or agent-driven run never reads that console.
"""
from __future__ import annotations

import pytest

from atlas_camera.comfy import node_reports
from atlas_camera.comfy.nodes_export import _with_manifest_note


def test_success_returns_an_empty_note(tmp_path, monkeypatch):
    from test_blender_measured_bridge import _solve

    dest = tmp_path / "scene.nk"
    dest.write_text("Root { }\n", encoding="utf-8")
    note = node_reports._write_export_manifest(
        _solve(), tmp_path, [("nuke_scene", str(dest))], "TestExporter")
    assert note == ""


def test_a_failed_manifest_returns_a_note_and_does_not_raise(monkeypatch, tmp_path):
    from test_blender_measured_bridge import _solve

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "atlas_camera.exporters.manifest.write_project_manifest", boom)
    note = node_reports._write_export_manifest(
        _solve(), tmp_path, [("solve_json", str(tmp_path / "s.json"))], "TestExporter")
    assert "SKIPPED" in note and "disk full" in note


def test_the_wrapper_leaves_the_result_tuple_untouched():
    """No output slot was added, so saved graphs keep their wires."""
    assert _with_manifest_note(("a", "b"), "") == ("a", "b")

    wrapped = _with_manifest_note(("a", "b"), "manifest SKIPPED: nope")
    assert wrapped["result"] == ("a", "b")
    assert wrapped["ui"]["text"] == ["manifest SKIPPED: nope"]
