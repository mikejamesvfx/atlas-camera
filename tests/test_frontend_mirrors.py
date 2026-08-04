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


# --- Gravity compass direction contract ------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_gravity_compass_direction_round_trips_and_stays_local():
    src = _read("atlas_gravity_compass.js")
    forward = re.search(
        r"(function gravityDirectionFromAngles\(.*?\n\})", src, re.DOTALL)
    inverse = re.search(
        r"(function anglesFromGravityDirection\(.*?\n\})", src, re.DOTALL)
    assert forward and inverse, "gravity compass conversion functions missing"
    assert 'import("./lib/atlas-three.bundle.js")' in src
    assert "https://" not in src and "http://" not in src
    assert '["AtlasGravityCompass", "AtlasSolveGate"]' in src
    assert "addDOMWidget" in src and "priorRemoved?.apply" in src
    assert 'message?.source_image' in src
    assert '_atlasGravityCompassPlate.src = source' in src
    assert 'new THREE.Vector3(0, .88, 0), 2.15' in src
    assert 'new THREE.RingGeometry(.58, .7, 64)' in src
    assert 'mode: headingHit ? "heading" : "gravity"' in src
    assert 'node._atlasCompassPendingApproval = true' in src
    assert 'find((w) => w.name === "heading_override")' in src
    assert "widget.hidden = true" in src
    assert "widget.options.hidden = true" in src

    gate_src = _read("atlas_solve_gate.js")
    assert "this._atlasCompassPendingApproval" in gate_src
    assert "approvedFor.value = fp" in gate_src

    # The expensive downstream branch queues once when an interaction ends,
    # never continuously while the artist drags the compass.
    assert "function queueDecision()" in src
    assert "if (changed) queueDecision();" in src
    pointer_move = src.split('canvas.addEventListener("pointermove"', 1)[1].split(
        'const endDrag', 1)[0]
    assert "queuePrompt" not in pointer_move

    samples = [[0, 0], [30, 0], [-18.5, 42], [75, -130], [-88, 179]]
    script = ("const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));\n" +
              forward.group(1) + "\n" + inverse.group(1) + "\n" +
              f"const samples={json.dumps(samples)};\n" +
              "console.log(JSON.stringify(samples.map(([p,r])=>" +
              "anglesFromGravityDirection(gravityDirectionFromAngles(p,r)))));" )
    result = subprocess.run(["node", "-e", script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for (pitch, roll), got in zip(samples, json.loads(result.stdout)):
        assert got["pitch"] == pytest.approx(pitch, abs=1e-9)
        assert got["roll"] == pytest.approx(roll, abs=1e-9)


# --- Camera-path compact repair bake contract -------------------------------

def test_camera_path_repair_bake_is_indexed_and_invalidates_stale_frames():
    src = _read("atlas_blockout.js")

    assert 'repairBakeBtn.textContent = "📷 Bake Repair Frame"' in src
    assert 'bakeBtn.textContent = "⏺ Bake Full Path"' in src
    assert "baked_frame_indices: bakedFrameIndices" in src
    assert "frame_indices: bakedFrameIndices" in src
    assert "delete existing.path_frames" in src
    assert "delete existing.atlas_proxy_path" in src


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


def test_scene_type_hidden_widgets_support_both_comfy_renderers():
    src = _read("atlas_derive_geometry.js")

    # LiteGraph and the Vue renderer use separate visibility fields.  The
    # legacy type/computeSize trick alone collapses rows but leaves values
    # painted over their neighbours on current ComfyUI builds.
    assert "widget.hidden = true" in src
    assert "widget.options.hidden = true" in src
    assert "delete widget.hidden" in src
    assert "delete widget.options.hidden" in src


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


# --- 🌀 fov channel (atlas_blockout.js <-> camera_path.py) -------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_fov_channel_numerically_matches_js():
    """sampleFovChannel (JS) vs sample_camera_path_fov_deg (Python) — the
    keyframed lens ramp behind 🌀 Vertigo. Executes the extracted JS for real
    across a path that exercises fill-forward (a middle keyframe without fov),
    mixed easings, and out-of-range clamping."""
    from atlas_camera.core.camera_path import sample_camera_path_fov_deg
    from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath

    src = _read("atlas_blockout.js")
    ez = re.search(r"(function applyEasingJS\(.*?\n  \})", src, re.DOTALL)
    cr1 = re.search(r"(function catmullRom1JS\(.*?\n  \})", src, re.DOTALL)
    fc = re.search(r"(function sampleFovChannel\(.*?\n  \})", src, re.DOTALL)
    assert ez and cr1 and fc, "fov-channel JS mirrors not found"

    kfs = [
        {"frame_index": 2, "position": [0, 1.6, 8], "target": [0, 1.6, 0],
         "fov_deg": 40.0, "easing": "ease_in_out"},
        {"frame_index": 6, "position": [0, 1.6, 7], "target": [0, 1.6, 0],
         "fov_deg": None, "easing": "ease_in"},
        {"frame_index": 12, "position": [0, 1.6, 6.4], "target": [0, 1.6, 0],
         "fov_deg": 48.930242541023, "easing": "linear"},
    ]
    frame_count = 15
    script = (ez.group(1) + "\n" + cr1.group(1) + "\n" + fc.group(1) + "\n" +
              f"const kfs = {json.dumps(kfs)};\n" +
              f"const out = Array.from({{length: {frame_count}}}, (_, f) => sampleFovChannel(kfs, f));\n" +
              "console.log(JSON.stringify(out));")
    result = subprocess.run(["node", "-e", script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    js_fovs = json.loads(result.stdout)

    path = AtlasCameraPath(
        keyframes=[
            AtlasCameraKeyframe(
                frame_index=k["frame_index"],
                position=tuple(k["position"]),
                target=tuple(k["target"]),
                fov_deg=k["fov_deg"],
                easing=k["easing"],
            )
            for k in kfs
        ],
        fps=24.0,
        frame_count=frame_count,
    )
    py_fovs = sample_camera_path_fov_deg(path)
    assert py_fovs is not None and len(py_fovs) == len(js_fovs) == frame_count
    for frame, (py_v, js_v) in enumerate(zip(py_fovs, js_fovs)):
        assert abs(py_v - js_v) < 1e-9, frame

    # The static-lens contract: no keyframed fov -> both sides yield "no channel".
    no_fov_kfs = [dict(k, fov_deg=None) for k in kfs]
    script2 = (ez.group(1) + "\n" + cr1.group(1) + "\n" + fc.group(1) + "\n" +
               f"console.log(JSON.stringify(sampleFovChannel({json.dumps(no_fov_kfs)}, 5)));")
    result2 = subprocess.run(["node", "-e", script2], capture_output=True,
                             text=True, timeout=30)
    assert result2.returncode == 0, result2.stderr
    assert json.loads(result2.stdout) is None
    no_fov_path = AtlasCameraPath(
        keyframes=[
            AtlasCameraKeyframe(
                frame_index=k["frame_index"], position=tuple(k["position"]),
                target=tuple(k["target"]), easing=k["easing"])
            for k in kfs
        ],
        fps=24.0, frame_count=frame_count)
    assert sample_camera_path_fov_deg(no_fov_path) is None


# --- viewport drawn-plane rules --------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_drawn_plane_rules_numerically_match_js():
    """atlas_blockout.js's plane rules must agree with core/polygon_planes.py.

    These decide where a drawn N-gon lands in 3D. A hand-slip on either side
    would put the JS preview and the applied geometry in different places —
    the artist would draw one plane and get another.
    """
    from atlas_camera.core.polygon_planes import (
        establish_plane_from_hits, intersect_ray_with_plane)

    src = _read("atlas_blockout.js")
    establish = re.search(
        r"(  function atlasEstablishPlaneFromHits\(.*?\n  \})", src, re.DOTALL)
    intersect = re.search(
        r"(  function atlasIntersectRayWithPlane\(.*?\n  \})", src, re.DOTALL)
    assert establish and intersect, "drawn-plane mirror functions missing from the JS"

    cases = [
        [[1.0, 0.5, -4.0], [3.0, 2.5, -4.0]],                    # two hits -> vertical
        [[0.0, 0.0, -6.0], [2.0, 0.0, -6.0], [0.0, 3.0, -8.0]],  # three -> Newell
        [[0.0, 2.0, -5.0], [4.0, 2.0, -5.0],
         [4.0, 4.0, -9.0], [0.0, 4.0, -9.0]],                    # coplanar sloped roof
        [[0.0, 0.0, -5.0], [1.0, 1.0, -5.0], [2.0, 2.0, -5.0]],  # collinear -> vertical
        [[1.0, 0.0, -5.0], [1.0, 4.0, -5.0]],                    # degenerate -> null
    ]
    ray = {"origin": [0.0, 1.6, 0.0], "direction": [0.2, 0.1, -1.0]}

    script = (establish.group(1) + "\n" + intersect.group(1) + "\n"
              + "const cases = " + json.dumps(cases) + ";\n"
              + "const ray = " + json.dumps(ray) + ";\n"
              + "const out = cases.map((c) => {\n"
              + "  const p = atlasEstablishPlaneFromHits(c);\n"
              + "  if (!p) return null;\n"
              + "  return {normal: p.normal, offset: p.offset,\n"
              + "          hit: atlasIntersectRayWithPlane(ray.origin, ray.direction, p)};\n"
              + "});\n"
              + "console.log(JSON.stringify(out));")
    js = json.loads(subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True).stdout)

    for hits, got in zip(cases, js):
        expected = establish_plane_from_hits(hits)
        if expected is None:
            assert got is None, f"JS established a plane Python refuses: {hits}"
            continue
        assert got is not None, f"JS refused a plane Python establishes: {hits}"
        normal, offset = expected
        for a, b in zip(normal, got["normal"]):
            assert a == pytest.approx(b, abs=1e-9)
        assert offset == pytest.approx(got["offset"], abs=1e-9)

        py_hit = intersect_ray_with_plane(
            ray["origin"], ray["direction"], (normal, offset))
        if py_hit is None:
            assert got["hit"] is None
        else:
            for a, b in zip(py_hit, got["hit"]):
                assert float(a) == pytest.approx(b, abs=1e-9)
