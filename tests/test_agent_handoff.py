"""AtlasAgentHandoff — brief / blocking wait / resume / auto-import contract.

No agent, no Blender: the resume is written by the test (as an agent would via
MCP or HTTP), and export_meshes.py is faked the way test_blender_measured_bridge
fakes run_recipe.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.comfy import agent_handoff as AH  # noqa: E402

from test_blender_measured_bridge import _box, _solve, _write_out  # noqa: E402


class TestBriefResume:
    def test_brief_written_with_token_and_history(self, tmp_path):
        b = AH.write_brief("7", {"task": "model the church", "timeout_s": 60}, root=tmp_path, now=1e9)
        d = tmp_path / AH.AGENT_DIRNAME / "7"
        assert (d / AH.BRIEF_NAME).is_file()
        assert b["token"] and b["deadline"] == pytest.approx(1e9 + 60)
        assert any(h.name.endswith("_brief.json") for h in (d / AH.HISTORY_DIRNAME).iterdir())
        assert AH.read_brief("7", root=tmp_path)["task"] == "model the church"
        assert "atlas_agent_resume" in b["resume"]["how"][0]

    def test_resume_requires_matching_token(self, tmp_path):
        b = AH.write_brief("7", {"task": "x", "timeout_s": 5}, root=tmp_path)
        with pytest.raises(ValueError, match="token"):
            AH.write_resume("7", {"token": "wrong", "status": "done", "reply": "hi"}, root=tmp_path)
        with pytest.raises(ValueError, match="status"):
            AH.write_resume("7", {"token": b["token"], "status": "later"}, root=tmp_path)
        rec = AH.write_resume("7", {"token": b["token"], "status": "done", "reply": "built 2 boxes",
                                    "blend_file": "C:/x/scene.blend"}, root=tmp_path)
        assert rec["blend_file"] == "C:/x/scene.blend"
        assert AH.read_resume("7", root=tmp_path)["reply"] == "built 2 boxes"

    def test_new_brief_clears_stale_resume(self, tmp_path):
        b1 = AH.write_brief("9", {"task": "a", "timeout_s": 5}, root=tmp_path)
        AH.write_resume("9", {"token": b1["token"], "status": "done", "reply": "old"}, root=tmp_path)
        AH.write_brief("9", {"task": "b", "timeout_s": 5}, root=tmp_path)
        assert AH.read_resume("9", root=tmp_path) is None

    def test_wait_returns_on_resume_and_ignores_wrong_token(self, tmp_path):
        b = AH.write_brief("3", {"task": "x", "timeout_s": 5}, root=tmp_path)
        clock = {"t": 0.0}
        def sleep(s): clock["t"] += s
        def now(): return clock["t"]
        # a stale (wrong-token) resume sits there first; then the right one lands.
        d = AH.agent_dir("3", root=tmp_path)
        (d / AH.RESUME_NAME).write_text(json.dumps({"token": "stale", "status": "done", "reply": "no"}),
                                        encoding="utf-8")
        calls = {"n": 0}
        def sleep2(s):
            sleep(s); calls["n"] += 1
            if calls["n"] == 2:
                (d / AH.RESUME_NAME).write_text(json.dumps(
                    {"token": b["token"], "status": "done", "reply": "yes"}), encoding="utf-8")
        rec = AH.wait_for_resume("3", b["token"], timeout_s=100, poll_s=1, root=tmp_path,
                                 sleep=sleep2, clock=now)
        assert rec["status"] == "done" and rec["reply"] == "yes"
        hist = list((d / AH.HISTORY_DIRNAME).iterdir())
        assert any("stale_" in h.name for h in hist)
        assert not (d / AH.RESUME_NAME).exists()          # consumed

    def test_wait_times_out(self, tmp_path):
        clock = {"t": 0.0}
        rec = AH.wait_for_resume("4", "tok", timeout_s=3, poll_s=1, root=tmp_path,
                                 sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                                 clock=lambda: clock["t"])
        assert rec["status"] == "timeout" and rec["waited_s"] >= 3


class TestNode:
    def _node(self, monkeypatch, tmp_path):
        monkeypatch.setattr(AH, "output_root", lambda default="output": tmp_path)
        from atlas_camera.comfy.nodes_agent import AtlasAgentHandoff
        return AtlasAgentHandoff()

    def _resume_later(self, node_id, tmp_path, payload, delay=0.3):
        def go():
            time.sleep(delay)
            b = AH.read_brief(node_id, root=tmp_path)
            AH.write_resume(node_id, {**payload, "token": b["token"]}, root=tmp_path)
        threading.Thread(target=go, daemon=True).start()

    def test_done_without_import_passes_reply(self, monkeypatch, tmp_path):
        node = self._node(monkeypatch, tmp_path)
        self._resume_later("n1", tmp_path, {"status": "done", "reply": "looked, nothing to add"})
        out, reply, brief_path, report = node.handoff(
            _solve(), "check the scene", timeout_s=10, poll_s=0.1, auto_import=False, unique_id="n1")
        assert reply == "looked, nothing to add"
        assert "resume: status=done" in report
        assert json.loads(open(brief_path, encoding="utf-8").read())["task"] == "check the scene"
        assert out.projection_scene.debug_metadata["agent_handoff"]["status"] == "done"

    def test_timeout_continue_and_fail(self, monkeypatch, tmp_path):
        node = self._node(monkeypatch, tmp_path)
        out, reply, _, report = node.handoff(_solve(), "x", timeout_s=1, poll_s=0.1,
                                             unique_id="n2", on_timeout="continue")
        assert "TIMEOUT" in report and reply == ""
        with pytest.raises(RuntimeError, match="no resume"):
            node.handoff(_solve(), "x", timeout_s=1, poll_s=0.1, unique_id="n3", on_timeout="fail")

    def test_done_with_blend_runs_export_and_appends(self, monkeypatch, tmp_path):
        import atlas_camera.blender as B
        node = self._node(monkeypatch, tmp_path)
        exdir = tmp_path / "ex"; exdir.mkdir()
        s = _solve()
        from atlas_camera.blender.measured import solve_seed_fingerprint
        (exdir / "seed.json").write_text(json.dumps(
            {"params": {"solve_fingerprint": solve_seed_fingerprint(s), "ground_y_m": 0.0,
                        "measured": {"camera_height_m": 1.6}}}), encoding="utf-8")
        blend = exdir / "scene.blend"; blend.write_bytes(b"BLENDER")
        v, f = _box()
        calls = []
        def fake_run(recipe, ex, *, blender_path="", timeout_s=300, blend_file=""):
            calls.append((recipe, str(blend_file)))
            _write_out(ex, [("agent_church", v, f, {"kind": "agent"})])
            return {"meshes_out": 1, "selection_rule": "atlas_out"}
        monkeypatch.setattr(B, "run_recipe", fake_run)
        self._resume_later("n4", tmp_path, {"status": "done", "reply": "added a church volume",
                                            "blend_file": str(blend)})
        out, reply, _, report = node.handoff(
            s, "add the church", exchange_dir=str(exdir), timeout_s=10, poll_s=0.1, unique_id="n4")
        assert calls and calls[0][0] == "export_meshes.py" and calls[0][1] == str(blend)
        names = [p.name for p in out.projection_scene.proxy_geometry]
        assert "agent_church" in names
        prim = out.projection_scene.proxy_geometry[-1]
        assert prim.metadata["source"] == "blender_import"
        assert prim.metadata["agent_reply"] == "added a church volume"
        assert out.projection_scene.debug_metadata["agent_handoff"]["meshes_added"] == 1
        # brief carried the scene + measured numbers + return contract
        brief = AH.read_brief("n4", root=tmp_path)
        assert brief["scene_blend"] == str(blend) and brief["measured"]["camera_height_m"] == 1.6
        assert brief["collections"]["model_under"] == "atlas_out"

    def test_skip_passes_through(self, monkeypatch, tmp_path):
        node = self._node(monkeypatch, tmp_path)
        self._resume_later("n5", tmp_path, {"status": "skip", "reply": "nothing needed"})
        s = _solve()
        out, reply, _, report = node.handoff(s, "x", timeout_s=10, poll_s=0.1, unique_id="n5")
        assert "agent said skip" in report
        assert len(out.projection_scene.proxy_geometry) == len(s.projection_scene.proxy_geometry)


def test_routes_and_mcp_wired():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    init = (root / "atlas_camera/comfy/__init__.py").read_text(encoding="utf-8")
    assert '"/atlas/agent/brief/{node_id}"' in init and '"/atlas/agent/resume/{node_id}"' in init
    from atlas_camera.mcp import server as S
    assert hasattr(S, "atlas_agent_brief") and hasattr(S, "atlas_agent_resume")


def test_mode_switch_skips_or_imports_without_pausing(monkeypatch, tmp_path):
    monkeypatch.setattr(AH, "output_root", lambda default="output": tmp_path)
    from atlas_camera.comfy.nodes_agent import AtlasAgentHandoff
    s = _solve()
    t0 = time.time()
    out, reply, brief_path, report = AtlasAgentHandoff().handoff(
        s, "x", timeout_s=600, unique_id="m1", mode="skip")
    assert time.time() - t0 < 2 and "mode=skip" in report and brief_path == ""
    # import_only: no wait, brief still written, imports what exists (nothing here)
    out, reply, brief_path, report = AtlasAgentHandoff().handoff(
        s, "x", timeout_s=600, unique_id="m2", mode="import_only")
    assert time.time() - t0 < 4 and "import_only" in report and AH.read_brief("m2", root=tmp_path)
    # env var wins over the widget
    monkeypatch.setenv("ATLAS_AGENT_MODE", "skip")
    out, reply, brief_path, report = AtlasAgentHandoff().handoff(
        s, "x", timeout_s=600, unique_id="m3", mode="wait")
    assert "ATLAS_AGENT_MODE" in report
