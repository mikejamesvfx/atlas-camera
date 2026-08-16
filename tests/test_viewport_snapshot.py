"""Automatic end-of-run viewport snapshots — the Python half (no ComfyUI).

The frontend renders and POSTs; `comfy/viewport_snapshot.py` writes the stable
latest PNGs, a timestamped history, a JSON sidecar, and stamps the record onto
the cached camera payload so `atlas_inspect_viewport` can hand out the paths.
"""
from __future__ import annotations

import base64
import json
import struct
import zlib

import pytest

from atlas_camera.comfy import viewport_snapshot as vs


def _png(w=4, h=3, rgb=(200, 30, 30)) -> str:
    """A tiny valid PNG as base64 (no PIL needed)."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def test_saves_latest_history_and_sidecar(tmp_path):
    rec = vs.save_viewport_snapshot(
        {"node_id": 12, "projected_b64": _png(), "geometry_b64": _png(rgb=(90, 90, 90)),
         "width": 1280, "height": 853, "solve_fingerprint": "abcd1234", "reason": "executed"},
        output_dir=tmp_path, now=1_700_000_000.0)
    root = tmp_path / vs.SNAPSHOT_DIRNAME
    assert (root / "viewport_12_projected.png").is_file()
    assert (root / "viewport_12_geometry.png").is_file()
    assert (root / "viewport_12.json").is_file()
    assert rec["files"]["projected"].endswith("viewport_12_projected.png")
    assert rec["long_edge"] == 1280 and rec["width"] == 1280
    hist = list((root / vs.HISTORY_DIRNAME).glob("*_12_*.png"))
    assert len(hist) == 2
    side = json.loads((root / "viewport_12.json").read_text(encoding="utf-8"))
    assert side["solve_fingerprint"] == "abcd1234"
    assert vs.read_snapshot_record(12, output_dir=tmp_path)["stamp"] == rec["stamp"]


def test_latest_is_overwritten_and_history_grows(tmp_path):
    vs.save_viewport_snapshot({"node_id": "7", "projected_b64": _png()},
                              output_dir=tmp_path, now=1_700_000_000.0)
    vs.save_viewport_snapshot({"node_id": "7", "projected_b64": _png(rgb=(1, 2, 3))},
                              output_dir=tmp_path, now=1_700_000_060.0)
    root = tmp_path / vs.SNAPSHOT_DIRNAME
    assert len(list(root.glob("viewport_7_*.png"))) == 1
    assert len(list((root / vs.HISTORY_DIRNAME).glob("*_7_projected.png"))) == 2


def test_history_is_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "MAX_HISTORY_PER_NODE", 2)
    for i in range(5):
        vs.save_viewport_snapshot({"node_id": "9", "projected_b64": _png(), "geometry_b64": _png()},
                                  output_dir=tmp_path, now=1_700_000_000.0 + i)
    hist = list((tmp_path / vs.SNAPSHOT_DIRNAME / vs.HISTORY_DIRNAME).glob("*_9_*.png"))
    assert len(hist) == 4          # 2 runs x 2 kinds kept


def test_geometry_only_is_fine_and_garbage_is_refused(tmp_path):
    rec = vs.save_viewport_snapshot({"node_id": 3, "geometry_b64": _png()}, output_dir=tmp_path)
    assert set(rec["files"]) == {"geometry"}
    with pytest.raises(ValueError, match="no snapshot images"):
        vs.save_viewport_snapshot({"node_id": 3}, output_dir=tmp_path)
    with pytest.raises(ValueError, match="not a PNG"):
        vs.save_viewport_snapshot({"node_id": 3, "projected_b64": base64.b64encode(b"nope").decode()},
                                  output_dir=tmp_path)


def test_node_id_is_sanitised_for_the_filesystem(tmp_path):
    rec = vs.save_viewport_snapshot({"node_id": "../evil id", "geometry_b64": _png()},
                                    output_dir=tmp_path)
    assert "/" not in rec["files"]["geometry"].split(vs.SNAPSHOT_DIRNAME)[1].lstrip("\\/")
    assert (tmp_path / vs.SNAPSHOT_DIRNAME / "viewport_.._evil_id_geometry.png").is_file()


def test_record_is_attached_to_the_camera_data_cache(tmp_path):
    cache = {"12": {"camera_meta": {}}, "13": {"camera_meta": {}}}
    rec = vs.save_viewport_snapshot({"node_id": 12, "geometry_b64": _png()}, output_dir=tmp_path)
    vs.attach_snapshot_to_cache(cache, rec)
    assert cache["12"]["viewport_snapshot"]["files"]["geometry"].endswith("viewport_12_geometry.png")
    assert "viewport_snapshot" not in cache["13"]
    vs.attach_snapshot_to_cache(cache, {**rec, "node_id": "99"})   # not cached: no-op


def test_frontend_wires_the_snapshot_route():
    """The JS must POST to the same route the server registers, hide helpers,
    render both 📽 states from the recovered camera at 1280, and only fire on
    real executions."""
    from pathlib import Path
    js = Path(__file__).resolve().parents[1] / "atlas_camera/comfy/web/atlas_blockout.js"
    src = js.read_text(encoding="utf-8")
    init = (Path(__file__).resolve().parents[1] / "atlas_camera/comfy/__init__.py").read_text(encoding="utf-8")
    assert '"/atlas/viewport_snapshot"' in src and '"/atlas/viewport_snapshot"' in init
    assert "ATLAS_SNAPSHOT_LONG_EDGE = 1280" in src
    assert "applyRecoveredCamera(snapCam, recoveredData)" in src
    assert "applyProjection(true);\n            projected = atlasRenderSceneToBase64" in src
    assert "applyProjection(false);\n          geometry = atlasRenderSceneToBase64" in src
    assert "refreshFromSolve({ snapshot: true })" in src
    assert 'captureSnapshots?.("executed")' in src
