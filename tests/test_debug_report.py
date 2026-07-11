"""Tests for AtlasDebugReport — the machine-readable master-workflow
diagnostic (2026-07-11). Built after a live session where two layers silently
shipped zero-vertex meshes and only a viewport-payload autopsy revealed it;
the JSON this node writes must surface that class of failure in one read.
"""

import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, "tests")
from test_inpaint_layers_nodes import _depth_result, _occluder_depth, _plate_image, _solve

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasCleanPlateLayer,
    AtlasDebugReport,
)


def _layered_solve():
    solve, depth, plate = _solve(), _depth_result(_occluder_depth()), _plate_image()
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        solve, depth, plate, near_m=8.0, far_m=12.0, name="band_bg",
        priority=5, embed_matte=True)
    out, _h, _e = AtlasCleanPlateLayer().add_layer(
        out, depth, plate, near_m=0.0, far_m=5.0, name="band_fg",
        priority=15, band_geometry="ground", embed_matte=True)
    return out, depth


def test_debug_report_json_and_flags(tmp_path):
    assert NODE_CLASS_MAPPINGS["AtlasDebugReport"] is AtlasDebugReport
    solve, depth = _layered_solve()
    path = str(tmp_path / "dbg" / "master_debug.json")
    out = AtlasDebugReport().report(
        solve, depth=depth, file_path=path,
        status_1="scoped to 'rocks' (8.5% segment, grown 16px)",
        status_2="band-only FALLBACK — segment for 'floor' covered 0.00% "
                 "of the frame (no-match); scoping skipped",
        vlm_report="ATLAS IMAGE ASSESSMENT ...")
    report, json_path = out["result"]
    assert out["ui"]["text"] == [report]
    assert json_path == path or json_path.endswith("master_debug.json")

    data = json.loads(open(json_path, encoding="utf-8").read())
    names = [s["name"] for s in data["projection_sources"]]
    assert names == ["band_bg", "band_fg"]
    bg, fg = data["projection_sources"]
    assert bg["n_vertices"] > 0 and fg["n_vertices"] > 0
    assert fg["band_geometry"] == "ground"
    assert bg["matte_coverage"] is not None
    assert data["depth"]["model_id"] == "fake"
    assert data["camera"]["camera_height_m"] == pytest.approx(1.6)
    # The no-match scope status must surface as a flag.
    assert any("FALLBACK" in f for f in data["flags"])
    # Band gap between fg far (5m) and bg near (8m) must be flagged.
    assert any("GAP" in f for f in data["flags"])
    assert "FLAGS" in report and "band_fg" in report


def test_debug_report_flags_zero_vertex_layer(tmp_path):
    """The exact live failure this node exists for."""
    from atlas_camera.core.schema import ProjectionSource

    solve, depth = _layered_solve()
    solve.projection_sources.append(ProjectionSource(
        camera=solve.camera, name="band_mid", priority=10.0,
        proxy_geometry=[], metadata={"projection_mode": "clean_plate"}))
    out = AtlasDebugReport().report(solve, file_path=str(tmp_path / "d.json"))
    data = json.loads(open(out["result"][1], encoding="utf-8").read())
    assert any("band_mid" in f and "ZERO vertices" in f for f in data["flags"])
    assert "ZERO vertices" in out["result"][0]


def test_debug_report_healthy_stack_says_so(tmp_path):
    solve, depth = _layered_solve()
    # close the band gap so no flags fire
    for src, (near, far) in zip(solve.projection_sources, ((5.0, 12.0), (0.0, 5.0))):
        src.metadata["near_m"], src.metadata["far_m"] = near, far
    out = AtlasDebugReport().report(solve, file_path=str(tmp_path / "d.json"))
    data = json.loads(open(out["result"][1], encoding="utf-8").read())
    assert data["flags"] == []
    assert "stack looks healthy" in out["result"][0]
