"""Cross-language mirror-sync tests (spec-panel long-term tier, 2026-07-11).

Three constants/algorithms are deliberately hand-duplicated between Python
and the frontend JS (the repo's documented accepted-duplication pattern):
the 🎨 layer debug palette, the scene_type presets, and the camera-path
Catmull-Rom + easing math. "Keep in sync by hand" only works if something
fails when a hand slips — before this file, only the Python side of the
palette was pinned; a JS edit would ship silently skewed.

The JS sources are checked by TEXT extraction (regex), and the Catmull-Rom
math is executed for real via `node -e` and compared numerically against
camera_path.py — skipped cleanly when node isn't installed.
"""

import json
import os
import re
import shutil
import subprocess

import pytest

WEB = os.path.join(os.path.dirname(__file__), "..", "atlas_camera", "comfy", "web")


def _read(name):
    return open(os.path.join(WEB, name), encoding="utf-8").read()


# --- 🎨 layer debug palette (atlas_blockout.js <-> nodes.py) -----------------

def test_layer_debug_palette_mirrors_js():
    from atlas_camera.comfy.nodes import (
        _LAYER_DEBUG_PALETTE_HEX,
        _LAYER_DEBUG_PRIMARY_HEX,
    )
    src = _read("atlas_blockout.js")
    primary = re.search(r"LAYER_DEBUG_PRIMARY\s*=\s*0x([0-9a-fA-F]{6})", src)
    assert primary, "LAYER_DEBUG_PRIMARY not found in atlas_blockout.js"
    assert primary.group(1).lower() == _LAYER_DEBUG_PRIMARY_HEX

    block = re.search(r"LAYER_DEBUG_PALETTE\s*=\s*\[(.*?)\];", src, re.DOTALL)
    assert block, "LAYER_DEBUG_PALETTE not found in atlas_blockout.js"
    js_hexes = tuple(h.lower() for h in re.findall(r"0x([0-9a-fA-F]{6})", block.group(1)))
    assert js_hexes == _LAYER_DEBUG_PALETTE_HEX


# --- scene_type presets (atlas_derive_geometry.js <-> nodes.py) --------------

def test_scene_type_presets_mirror_js():
    from atlas_camera.comfy.nodes import AtlasDeriveProjectionGeometry

    py_presets = AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS
    src = _read("atlas_derive_geometry.js")
    block = re.search(r"SCENE_TYPE_PRESETS\s*=\s*\{(.*?)\n\};", src, re.DOTALL)
    assert block, "SCENE_TYPE_PRESETS not found in atlas_derive_geometry.js"
    js_block = block.group(1)

    # Every Python preset must exist in the JS mirror with the same override
    # KEYS (the JS uses them to decide widget visibility, values to hide).
    js_names = set(re.findall(r"^\s*(\w+)\s*:\s*\{", js_block, re.MULTILINE))
    assert js_names == set(py_presets), (
        f"preset name drift: JS-only {js_names - set(py_presets)}, "
        f"Python-only {set(py_presets) - js_names}")
    for name, overrides in py_presets.items():
        entry = re.search(rf"^\s*{name}\s*:\s*\{{(.*?)\}}", js_block,
                          re.MULTILINE | re.DOTALL)
        assert entry, name
        for key in overrides:
            assert key in entry.group(1), f"{name}: key '{key}' missing in JS mirror"


# --- Catmull-Rom + easing (atlas_blockout.js <-> camera_path.py) -------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_catmull_rom_and_easing_numerically_match_js():
    from atlas_camera.core.camera_path import _apply_easing, _catmull_rom

    src = _read("atlas_blockout.js")
    cr = re.search(r"(function catmullRom3JS\(.*?\n  \})", src, re.DOTALL)
    ez = re.search(r"(function applyEasingJS\(.*?\n  \})", src, re.DOTALL)
    assert cr and ez, "camera-path JS mirrors not found"

    pts = [{"x": 0.0, "y": 1.0, "z": -2.0}, {"x": 1.5, "y": 0.5, "z": -4.0},
           {"x": 3.0, "y": 2.0, "z": -3.0}, {"x": 5.0, "y": 1.0, "z": -8.0}]
    ts = [0.0, 0.2, 0.5, 0.77, 1.0]
    easings = ["linear", "ease_in", "ease_out", "ease_in_out"]
    script = (cr.group(1) + "\n" + ez.group(1) + "\n" +
              f"const pts = {json.dumps(pts)}; const ts = {json.dumps(ts)};\n" +
              f"const es = {json.dumps(easings)};\n" +
              "const out = {cr: ts.map(t => catmullRom3JS(pts[0], pts[1], pts[2], pts[3], t)),"
              " ez: es.map(e => ts.map(t => applyEasingJS(t, e)))};\n"
              "console.log(JSON.stringify(out));")
    result = subprocess.run(["node", "-e", script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    js = json.loads(result.stdout)

    as_tuple = lambda p: (p["x"], p["y"], p["z"])  # noqa: E731
    for t, js_p in zip(ts, js["cr"]):
        py_p = _catmull_rom(as_tuple(pts[0]), as_tuple(pts[1]),
                            as_tuple(pts[2]), as_tuple(pts[3]), t)
        assert abs(py_p[0] - js_p["x"]) < 1e-9
        assert abs(py_p[1] - js_p["y"]) < 1e-9
        assert abs(py_p[2] - js_p["z"]) < 1e-9
    for easing, row in zip(easings, js["ez"]):
        for t, js_v in zip(ts, row):
            assert abs(_apply_easing(t, easing) - js_v) < 1e-12, (easing, t)
