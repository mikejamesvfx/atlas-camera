"""Tests for AtlasSolveGate — the solve-confirm checkpoint.

Third member of the gate family (AtlasAssessImage's ▶ Continue, 📐's patch
pause): the solve output blocks until an artist approval whose fingerprint
matches the CURRENT solve+image, so neither a swapped photo nor a re-solve
with different settings can sail through a stale approval.
"""

import sys
import types

import pytest

from atlas_camera.comfy.nodes import AtlasSolveGate, _solve_fingerprint
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import (
    AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera,
)


def _solve(fx=700.0):
    eye, target = (0.0, 1.6, 0.0), (0.0, 0.5, -10.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(
        camera_position=eye, camera_rotation_matrix=rot3,
        camera_world_matrix=world, camera_view_matrix=view,
    )
    intr = AtlasIntrinsics(
        image_width=800, image_height=600, focal_length_mm=35.0,
        sensor_width_mm=36.0, fx_px=fx, fy_px=fx, cx_px=400.0, cy_px=300.0,
    )
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


class FakeBlocker:
    def __init__(self, message):
        self.message = message


@pytest.fixture()
def comfy_runtime(monkeypatch):
    mod = types.ModuleType("comfy_execution.graph")
    mod.ExecutionBlocker = FakeBlocker
    pkg = types.ModuleType("comfy_execution")
    pkg.graph = mod
    monkeypatch.setitem(sys.modules, "comfy_execution", pkg)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", mod)


def test_gate_blocks_until_proceed(comfy_runtime):
    torch = pytest.importorskip("torch")
    out = AtlasSolveGate().gate(_solve(), torch.rand(1, 600, 800, 3))
    assert isinstance(out["result"][0], FakeBlocker)
    assert "PAUSED" in out["result"][1].upper() or "paused" in out["result"][1]


def test_gate_passes_with_matching_fingerprint(comfy_runtime):
    torch = pytest.importorskip("torch")
    solve, image = _solve(), torch.rand(1, 600, 800, 3)
    fp = _solve_fingerprint(solve, image)
    out = AtlasSolveGate().gate(solve, image, proceed=True, approved_for=fp)
    assert out["result"][0] is solve
    assert "APPROVED" in out["result"][1]
    assert out["ui"]["fingerprint"] == [fp]


def test_stale_approval_rearms(comfy_runtime):
    torch = pytest.importorskip("torch")
    solve, image = _solve(), torch.rand(1, 600, 800, 3)
    fp_old = _solve_fingerprint(_solve(fx=999.0), image)  # different solve
    out = AtlasSolveGate().gate(solve, image, proceed=True, approved_for=fp_old)
    assert isinstance(out["result"][0], FakeBlocker)
    assert "RE-ARMED" in out["result"][1]


def test_manual_override_with_empty_approved_for(comfy_runtime):
    torch = pytest.importorskip("torch")
    solve, image = _solve(), torch.rand(1, 600, 800, 3)
    out = AtlasSolveGate().gate(solve, image, proceed=True, approved_for="")
    assert out["result"][0] is solve


def test_degrades_to_passthrough_outside_comfy():
    torch = pytest.importorskip("torch")
    solve, image = _solve(), torch.rand(1, 600, 800, 3)
    out = AtlasSolveGate().gate(solve, image)  # no blocker importable
    assert out["result"][0] is solve


def test_report_carries_solve_summary(comfy_runtime):
    torch = pytest.importorskip("torch")
    out = AtlasSolveGate().gate(_solve(), torch.rand(1, 600, 800, 3))
    report = out["result"][1]
    assert "35.0mm" in report          # focal
    assert "1.60m" in report           # camera height
    assert "pitch:" in report


def test_gate_report_warns_on_unverified_scale(comfy_runtime):
    torch = pytest.importorskip("torch")
    s = _solve()
    s.debug_metadata["scale_source"] = "assumed_default"
    out = AtlasSolveGate().gate(s, torch.rand(1, 600, 800, 3))
    assert "SCALE NOT VERIFIED" in out["result"][1]


def test_gate_report_no_warning_on_manual_scale(comfy_runtime):
    torch = pytest.importorskip("torch")
    s = _solve()
    s.debug_metadata["scale_source"] = "manual_override"
    out = AtlasSolveGate().gate(s, torch.rand(1, 600, 800, 3))
    assert "SCALE NOT VERIFIED" not in out["result"][1]
    assert "scale: manual" in out["result"][1]
