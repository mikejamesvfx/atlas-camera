"""AtlasSceneHealthGate 🩺 — Gate 4: scene-health checkpoint before export.

Mirrors test_solve_gate.py's structure. Doctrine (gate_state_table.md): ships
closed for warn/fail, content-fingerprint approval identity, every skip has a
visible explanation, and the health stamp is indelible either way.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import AtlasSceneHealthGate, _solve_fingerprint
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)


def _cam(w=800, h=600):
    eye, target = (0.0, 1.6, 0.0), (0.0, 0.5, -10.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(camera_position=eye, camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=w, image_height=h, focal_length_mm=35.0,
                           sensor_width_mm=36.0, fx_px=700.0, fy_px=700.0,
                           cx_px=w / 2, cy_px=h / 2)
    return LatentCamera(intrinsics=intr, extrinsics=extr)


def _healthy_solve():
    s = AtlasSolve(camera=_cam())
    s.debug_metadata["scale_source"] = "manual_override"
    return s


def _warn_solve():
    s = _healthy_solve()
    s.projection_sources = [ProjectionSource(
        camera=_cam(320, 240), name="empty_layer",
        proxy_geometry=[AtlasProxyPrimitive(
            name="m", primitive_type="mesh", metadata={"n_vertices": 0})],
        metadata={"projection_mode": "clean_plate", "near_m": 0.0, "far_m": 5.0})]
    return s


def _img():
    return torch.rand(1, 600, 800, 3)


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


def test_pass_level_flows_without_click(comfy_runtime):
    s = _healthy_solve()
    out = AtlasSceneHealthGate().gate(s, _img())
    assert out["result"][0] is s
    assert "PASS" in out["result"][1]


def test_pass_through_off_still_gates(comfy_runtime):
    out = AtlasSceneHealthGate().gate(_healthy_solve(), _img(),
                                      pass_through_on_pass=False)
    assert isinstance(out["result"][0], FakeBlocker)


def test_warn_level_ships_closed(comfy_runtime):
    out = AtlasSceneHealthGate().gate(_warn_solve(), _img())
    assert isinstance(out["result"][0], FakeBlocker)
    assert "paused" in out["result"][1]
    assert "ZERO vertices" in out["result"][1]


def test_acknowledge_with_matching_fingerprint_flows(comfy_runtime):
    s, img = _warn_solve(), _img()
    fp = _solve_fingerprint(s, img)
    out = AtlasSceneHealthGate().gate(s, img, proceed=True, approved_for=fp)
    assert out["result"][0] is s
    assert "acknowledged" in out["result"][1]


def test_stale_fingerprint_rearms(comfy_runtime):
    out = AtlasSceneHealthGate().gate(_warn_solve(), _img(), proceed=True,
                                      approved_for="deadbeefdeadbeef")
    assert isinstance(out["result"][0], FakeBlocker)
    assert "RE-ARMED" in out["result"][1]


def test_manual_unconditional_override(comfy_runtime):
    s = _warn_solve()
    out = AtlasSceneHealthGate().gate(s, _img(), proceed=True, approved_for="")
    assert out["result"][0] is s


def test_stamp_is_indelible_both_states(comfy_runtime):
    # Blocked: stamp present, unacknowledged.
    s, img = _warn_solve(), _img()
    AtlasSceneHealthGate().gate(s, img)
    stamp = s.debug_metadata["scene_health"]
    assert stamp["level"] == "fail"           # zero-vertex layer = fail
    assert stamp["acknowledged"] is False
    assert stamp["evaluated_at"]
    # Acknowledged: stamp flips.
    fp = _solve_fingerprint(s, img)
    AtlasSceneHealthGate().gate(s, img, proceed=True, approved_for=fp)
    assert s.debug_metadata["scene_health"]["acknowledged"] is True


def test_pass_stamp_is_not_marked_acknowledged(comfy_runtime):
    s = _healthy_solve()
    AtlasSceneHealthGate().gate(s, _img())
    assert s.debug_metadata["scene_health"]["level"] == "pass"
    assert s.debug_metadata["scene_health"]["acknowledged"] is False


def test_pass_through_outside_comfy():
    # No comfy runtime -> blocker unavailable -> degrades to pass-through.
    s = _warn_solve()
    out = AtlasSceneHealthGate().gate(s, _img())
    assert out["result"][0] is s


def test_health_summary_suffix_reads_stamp(comfy_runtime):
    from atlas_camera.comfy.nodes import _health_summary_suffix
    s, img = _warn_solve(), _img()
    assert _health_summary_suffix(s) == ""     # no stamp yet
    AtlasSceneHealthGate().gate(s, img)
    suffix = _health_summary_suffix(s)
    assert "FAIL" in suffix and "UNACKNOWLEDGED" in suffix
    fp = _solve_fingerprint(s, img)
    AtlasSceneHealthGate().gate(s, img, proceed=True, approved_for=fp)
    assert "acknowledged" in _health_summary_suffix(s)


# --- Gate-module extraction (2026-08-01) -----------------------------------
# The five gate steps now live in atlas_camera.comfy.gate; these pin what the
# node still OWNS — its own re-arm wording, and the doctrine that a clean
# pass-through must not swallow the explanation of a stale acknowledgement.

def test_rearm_sentence_is_this_nodes_own_wording(comfy_runtime):
    s, img = _warn_solve(), _img()
    out = AtlasSceneHealthGate().gate(s, img, proceed=True, approved_for="stale")
    assert out["result"][1].startswith(
        "*** GATE RE-ARMED: the solve or image changed since the "
        "acknowledgement — review and ✅ again. ***\n")
    assert isinstance(out["result"][0], FakeBlocker)


def test_clean_passthrough_still_reports_a_stale_acknowledgement(comfy_runtime):
    # bypass (pass_through_on_pass) widens what FLOWS, never what is SEEN:
    # the user may override a warning but must never lose it.
    s = _healthy_solve()
    out = AtlasSceneHealthGate().gate(s, _img(), proceed=True, approved_for="stale")
    assert out["result"][0] is s                      # a clean scene still flows
    assert "GATE RE-ARMED" in out["result"][1]


def test_envelope_shape_is_the_frontend_contract(comfy_runtime):
    # atlas_scene_health_gate.js reads ui.text and ui.fingerprint by name.
    s, img = _warn_solve(), _img()
    out = AtlasSceneHealthGate().gate(s, img)
    assert set(out) == {"ui", "result"}
    assert out["ui"] == {"text": [out["result"][1]],
                         "fingerprint": [_solve_fingerprint(s, img)]}
    assert isinstance(out["result"], tuple) and len(out["result"]) == 2
